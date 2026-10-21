"""Execute released programs chronologically; scoring feedback is delayed until mature."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from .artifacts import TaskArtifacts, matured_windows
from .backbone import TimestralBackend
from .policy import EarlyStopping, combine


@dataclass(frozen=True)
class Config:
    checkpoint_path: str = "mistralai/Timestral-209M"
    device: str = "cuda:0"
    run_root: str = "timersi15-runs"
    artifacts_dir: str | None = None
    regenerate_decisions: bool = False
    agent_model: str = "claude-opus-5"
    effort: str = "high"
    decision_timeout: int = 900
    max_turns: int = 32
    vertex_project: str | None = None


def configure_isolation():
    """Mount shared runtime code only, not the installed artifact directory."""
    from timersi_u1 import isolation
    original = isolation.docker_command
    if getattr(original, "_timersi15", False):
        return

    def restricted(*args, **kwargs):
        command = original(*args, **kwargs)
        runtime = Path(isolation.__file__).resolve().parent
        workspace = Path(args[0] if args else kwargs["workspace"]).resolve()
        for i, value in enumerate(command):
            if value.startswith("type=bind,src=") and ",dst=/opt/timersi/src,readonly" in value:
                command[i] = f"type=bind,src={runtime},dst=/opt/timersi/src/timersi_u1,readonly"
        bootstrap = Path(__file__).with_name("candidate_bootstrap.py").resolve()
        command += ["--mount", f"type=bind,src={bootstrap},dst=/opt/timersi-bootstrap/sitecustomize.py,readonly",
                    "--env", "PYTHONPATH=/opt/timersi-bootstrap:/opt/timersi/src",
                    "--env", "TIMERSI15_CANDIDATE=1", "--env", f"TIMERSI_E185_ROOT={workspace}",
                    "--env", f"TIMERSI_E185_CONTROL={workspace}/compute", "--env", "PYTHONHASHSEED=0"]
        return command
    restricted._timersi15 = True
    isolation.docker_command = restricted
    os.environ.setdefault("TIMERSI_ISOLATION_IMAGE", "timersi-isolation:1.5")


def program_bank(records, bounded):
    from timersi_u1.core import ProgramBank, RAW, blend

    class ReleasedBank(ProgramBank):
        def predict(self, name, view, tools, memory, cache):
            if name in cache:
                return cache[name]
            spec = self.records[name]
            if spec["kind"] == "blend":
                result = blend([self.predict(n, view, tools, memory, cache) for n in spec["members"]], spec["weights"])
            elif spec["kind"] == "route":
                from timersi_u1.selection import route_forecasts
                try:
                    result = route_forecasts(view, {n: self.predict(n, view, tools, memory, cache)
                        for n in spec["members"]}, spec["routes"], spec["fallback"])
                except ValueError as error:
                    if not name.startswith("released_incumbent_") or not any(
                            text in str(error) for text in ("Nonfinite forecast", "Quantile crossing")):
                        raise
                    result = self.predict("raw", view, tools, memory, cache)
            elif spec["kind"] == "scenario":
                from timersi_u1.scenario_selection import apply_scenario
                result = apply_scenario(view, lambda n: self.predict(n, view, tools, memory, cache), spec["policy"])
            elif name == "raw" and spec["code"] == RAW and not spec.get("files"):
                result = tools.timesfm(view)
            else:
                from timersi_u1.isolation import predict
                result = predict(spec, view, tools, memory)
            if bounded and name != "raw" and not name.startswith("released_incumbent_"):
                from timersi_u1.numerical_boundary import protect
                result = protect(result, self.predict("raw", view, tools, memory, cache), view, name, cache, spec)
            cache[name] = result.validate()
            return result

    return ReleasedBank(records)


def execute(task, config, output, *, backend=None):
    from timersi_u1.core import Forecast, write
    from timersi_u1.data import EvaluationContext, make_view
    from timersi_u1.selection import route_forecasts
    from timersi_u1.tools import Tools
    from timersi_u1.compute import install
    from timersi_u1.lasso186 import install as install_lasso

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TIMERSI_E185_ROOT"] = str(output)
    os.environ["TIMERSI_E185_CONTROL"] = str(output / "compute")
    os.environ.setdefault("TIMERSI_U1_BATCH_SIZE", "4")
    install()
    install_lasso()
    configure_isolation()
    artifacts = TaskArtifacts(task.task_name, config.artifacts_dir)
    recipe = artifacts.recipe
    if recipe.get("release_revision") != "211-C-EE1":
        raise ValueError("Artifact and implementation revisions differ")
    if task.to_dict() != recipe["task"]:
        raise ValueError("Task definition differs from the released program inputs")
    if not 1 <= task.num_windows <= len(recipe["windows"]):
        raise ValueError("Unexpected forecast window count")
    task.load_full_dataset(num_proc=1)
    bank = program_bank(recipe["bank"], recipe["bounded"])
    tools = Tools(config.checkpoint_path, config.device, output / "native",
                  backend=backend or TimestralBackend(config.checkpoint_path, config.device))
    gate = EarlyStopping(len(task.target_columns), len(task.known_dynamic_columns), len(task.past_dynamic_columns))
    history, bounds, observed = [], [], set()
    feedback, previous_live_plan = [], None
    started = time.monotonic()
    count = 0
    try:
        for k, plan in enumerate(recipe["windows"]):
            window = task.get_window(k, num_proc=1)
            view = make_view(window, task)
            mature = matured_windows(bounds, view)
            # Only now may the judge score these earlier, fully observed targets.
            for j in mature:
                if j in observed:
                    continue
                with np.load(output / f"window-{j:03d}.npz", allow_pickle=False) as z:
                    raw = Forecast(z["raw_point"], z["raw_quantiles"])
                    enhanced = Forecast(z["enhanced_point"], z["enhanced_quantiles"])
                judge = EvaluationContext(task.get_window(j, num_proc=1), task)
                history[j] = dict(raw_SQL=judge.prepare(raw).score()["SQL"],
                                  loop_SQL=None if history[j]["stopped"] else judge.prepare(enhanced).score()["SQL"],
                                  stopped=history[j]["stopped"])
                if config.regenerate_decisions:
                    from .context import mature_feedback
                    feedback[j] = mature_feedback(task, j, output, feedback[j])
                observed.add(j)
            stopped = gate.decide(k, mature, history)
            cache = {}
            raw = bank.predict("raw", view, tools, [], cache)
            evidence = dict(decision=dict(default=dict(weights={"raw": 1.}), groups={}))
            if stopped:
                enhanced = final = raw
            else:
                if config.regenerate_decisions:
                    from .context import regenerate
                    live_plan = plan if plan["incumbent"] in bank.records else previous_live_plan
                    if live_plan is None:
                        enhanced = raw
                    else:
                        enhanced, evidence = regenerate(config, task, window, view, k, bank,
                            artifacts, live_plan, tools, cache, output, feedback, bounds)
                        previous_live_plan = live_plan
                else:
                    memory = artifacts.visible_memory(plan, view)
                    if any(str(item["id"]) not in plan["routes"] or any(str(target) not in
                           plan["routes"][str(item["id"])] for target in view.targets) for item in view.items):
                        raise ValueError("Released routes do not cover the current inputs")
                    names = list(dict.fromkeys(n for item in view.items for target in view.targets
                        for n in plan["routes"][str(item["id"])][str(target)]["members"]))
                    forecasts = {n: bank.predict(n, view, tools, memory, cache) for n in names}
                    enhanced = route_forecasts(view, forecasts, plan["routes"], plan["fallback"])
                paired = [[history[j]["raw_SQL"], history[j]["loop_SQL"]] for j in mature]
                final = combine(raw, enhanced, paired)
            np.savez_compressed(output / f"window-{k:03d}.npz", point=final.point, quantiles=final.quantiles,
                raw_point=raw.point, raw_quantiles=raw.quantiles,
                enhanced_point=enhanced.point, enhanced_quantiles=enhanced.quantiles)
            history.append(dict(raw_SQL=None, loop_SQL=None, stopped=stopped))
            if config.regenerate_decisions:
                feedback.append(dict(window=k, decision=None if stopped else evidence["decision"]))
            bounds.append(dict(ids=[str(i["id"]) for i in view.items],
                               ends=[str(i["timestamps"][-1]) for i in view.items]))
            count += len(view.items) * len(view.targets)
        elapsed = time.monotonic() - started
        # The submitted CSV is an accounted equivalent-execution estimate, not
        # this frozen-program replay's stopwatch. Keep the two records distinct.
        timing = dict(training_time_s=recipe["upstream_training_time_s"],
            inference_time_s=recipe["upstream_inference_time_s"],
            measured_execution_s=elapsed, num_forecasts=count, windows=len(bounds))
        timing.update(timing_basis="recorded_cost_estimate_with_paired_execution_measurements",
                      regenerated_selection=config.regenerate_decisions,
                      full_fresh_search_timed=False)
        if config.regenerate_decisions:
            timing.update(inference_time_s=elapsed, timing_basis="recorded_adaptation_plus_measured_new_selection")
        write(output / "RUNTIME.json", timing)
        return timing
    finally:
        artifacts.close()


def fit_predict(task, config=None):
    from timersi_u1.core import Forecast, write
    from timersi_u1.data import predictions
    config = config or Config()
    root = Path(config.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="task-", dir=root))
    request = output / "request.json"
    write(request, dict(task=task.to_dict(), config=asdict(config)))
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", LOKY_MAX_CPU_COUNT="8")
    started = time.monotonic()
    subprocess.run([sys.executable, "-m", "timersi15.runtime", str(request)], env=env, check=True)
    timing = json.loads((output / "RUNTIME.json").read_text())
    forecasts = []
    for k in range(timing["windows"]):
        with np.load(output / f"window-{k:03d}.npz", allow_pickle=False) as z:
            forecasts.append(predictions(Forecast(z["point"], z["quantiles"]).validate(), task))
    timing["measured_replay_wall_s"] = time.monotonic() - started
    if config.regenerate_decisions:
        timing["inference_time_s"] += max(0., timing["measured_replay_wall_s"] - timing["measured_execution_s"])
    return forecasts, timing


if __name__ == "__main__":
    import fev
    request = Path(sys.argv[1])
    value = json.loads(request.read_text())
    execute(fev.Task(**value["task"]), Config(**value["config"]), request.parent)
