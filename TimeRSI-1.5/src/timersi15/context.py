"""Rebuild selection inputs from visible data and the released program pool."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import numpy as np


def score(task, window, forecast):
    from fev.metrics import SQL
    from timersi_u1.data import predictions
    arrays, ids = window._prepare_arrays(predictions(forecast, task), task.quantile_levels)
    cells = SQL().compute(**arrays, seasonality=task.seasonality,
                          quantile_levels=task.quantile_levels, reduce_axes=(1,))
    return cells, list(map(str, ids))


def mature_feedback(task, index, output, previous):
    """Called by the judge only after this older forecast is fully observed."""
    from timersi_u1.core import Forecast
    window = task.get_window(index, num_proc=1)
    with np.load(output/f'window-{index:03d}.npz', allow_pickle=False) as z:
        final = Forecast(z['point'], z['quantiles'])
        cells, ids = score(task, window, final)
    path = output/'P/official'/f'window-{index:03d}'/'candidate-forecasts.npz'
    candidates = {}
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            for name in [key[:-7] for key in z.files if key.endswith('__point')]:
                candidates[name] = score(task, window, Forecast(z[name+'__point'], z[name+'__quantiles']))[0]
    result = dict(previous, ids=ids, targets=list(task.target_columns),
                  candidate_cells=candidates, new_cells=cells)
    from timersi_u1.core import write
    path = output/'P/official'/f'window-{index:03d}'/'MATURE-FEEDBACK.json'
    write(path, result)
    import json
    return json.loads(path.read_text())


def regenerate(config, task, window, view, index, bank, artifacts, plan, tools, cache,
               output, feedback, bounds):
    from timersi_u1.core import Forecast, write
    from timersi_u1.data import historical_windows, make_view, predictions
    from timersi_u1.window_planner import packet_for
    from timersi_u1.trajectory_evidence import enrich
    from .history import publish_history
    from .probe import evaluate
    from .search import decide
    memory = artifacts.visible_memory(plan, view, historical_subset=True)
    current = {}; current_errors = {}; past_errors = {}
    for name, program in plan['candidates'].items():
        try: current[name] = bank.predict(program, view, tools, memory, cache)
        except Exception as error: current_errors[name] = repr(error)
    if not {'raw','incumbent167'} <= current.keys():
        raise ValueError('Raw and complete incumbent forecasts are required')
    for name, program in plan['candidates'].items():
        if name != program:
            bank.records[name] = deepcopy(bank.records[program])
    packet, groups = packet_for(view, bank, current, 'incumbent167', {}, [])
    packet = enrich(view, current, packet)
    template = output/'prepared'/f'window-{index:03d}'
    template.mkdir(parents=True)
    np.savez_compressed(template/'actual-candidates.npz', item_ids=np.asarray([str(i['id']) for i in view.items]),
        **{name+'__point': f.point for name, f in current.items()},
        **{name+'__quantiles': f.quantiles for name, f in current.items()})
    publish_history(view, template/'visible-history')
    past = template/'composition-past'; past.mkdir()
    episodes = []; available = []; evidence = {}
    warm = historical_windows(task, 64, 1, anchor=window)
    if 1 in warm:
        past_window = warm[1]; past_view = make_view(past_window, task)
        observed = {str(i['id']): np.datetime64(i['timestamps'][i['y'].shape[1]-1]) for i in view.items}
        assert all(np.datetime64(i['timestamps'][-1]) <= observed[str(i['id'])] for i in past_view.items)
        past_memory = artifacts.visible_memory(plan, past_view, historical_subset=True)
        past_cache = {}
        forecasts = {}
        for name in current:
            try: forecasts[name] = bank.predict(plan['candidates'][name], past_view, tools, past_memory, past_cache)
            except Exception as error: past_errors[name] = repr(error)
        if 'raw' in forecasts:
            scoring, ids = past_window._prepare_arrays(predictions(forecasts['raw'], task), task.quantile_levels)
            available = list(forecasts)
            np.savez_compressed(past/'recent.npz', item_ids=np.asarray(ids),
                **{k:v for k,v in scoring.items() if k in ('y_true','y_past','y_past_lengths')},
                **{name+'__point': f.point for name,f in forecasts.items()},
                **{name+'__quantiles': f.quantiles for name,f in forecasts.items()})
            episodes = [dict(arrays='recent.npz', context=dict(kind='recent',offset=1,
                ids=list(map(str,ids)), forecast_ends=[str(i['timestamps'][-1]) for i in past_view.items]))]
    write(past/'MANIFEST.json', dict(candidates=available, targets=list(view.targets),
        groups=[{k:g[k] for k in ('key','target','members')} for g in groups],
        seasonality=task.seasonality, quantile_levels=task.quantile_levels, episodes=episodes))
    for name in available:
        result = evaluate(past, dict(default=dict(weights={name:1.}),groups={}))
        evidence[name] = result['episodes']
        packet['candidates'][name]['historical_S_SQL'] = result['episodes'][0]['metrics']['SQL']
    seen = set()
    def publish(name):
        if name in seen: return
        seen.add(name); spec = bank.records[name]
        if spec['kind'] == 'code':
            folder = template/'candidate-code'/name; folder.mkdir(parents=True,exist_ok=True)
            (folder/'candidate.py').write_text(spec['code'])
            for relative, text in spec.get('files',{}).items():
                target=folder/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(text)
        for child in spec.get('members',[]): publish(child)
    for name in current: publish(name)
    shutil.copy2(Path(__file__).with_name('probe_client.py'), template/'composition_probe.py')
    packet.update(python_executable=sys.executable, native_request_budget=0,
        current_candidate_arrays='actual-candidates.npz',
        complete_visible_history=dict(index='visible-history/INDEX.json',arrays='visible-history/inputs.npz'),
        composition_probe=dict(directory='composition-past',script='composition_probe.py'),
        latest_past_selection_evidence=evidence,available_past_candidates=available,
        current_candidate_failures=current_errors,past_candidate_failures=past_errors,current_labels_published=False)
    cfg = dict(projects=[config.vertex_project], agent_model=config.agent_model, effort=config.effort,
               decision_timeout=config.decision_timeout,max_turns=config.max_turns)
    job = dict(output=str(output),index=0)
    ends = [dict(zip(row['ids'],row['ends'])) for row in bounds]
    result, decision, receipt, error = decide(cfg,job,'P',index,view,current,packet,template,feedback,ends)
    destination = output/'P/official'/f'window-{index:03d}'
    shutil.copy2(template/'actual-candidates.npz',destination/'candidate-forecasts.npz')
    return result, dict(decision=decision,receipt=receipt,operational_fallback=error)
