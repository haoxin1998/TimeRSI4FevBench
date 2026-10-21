# TimeRSI-1.5

This package executes the released agent-evolved programs with Timestral-209M. The shared search implementation remains in the [root package](../src/timersi_u1/search.py).

This release builds on TimeRSI-1.0's agent-evolved forecasting programs:

- **Timestral-209M backbone:** execute the existing programs with frozen Timestral instead of TimesFM 3.0.
- **Early stopping for efficiency:** use completed historical feedback to stop unnecessary enhancement and continue with direct backbone forecasts.
- **Adaptive quantile aggregation:** task-local AdaNormalHedge combines raw and enhanced quantiles using fully matured SQL feedback. This aggregation leaves the point-forecast path unchanged.
- **Task-local composition:** current inputs and historical backtests guide group/target-specific revisions while retaining useful complete pipelines. Historical evaluation reuses immutable inputs and invariant computations.

Foundation-model weights are not fine-tuned. Forecast decisions use only observations, declared known-future covariates and feedback available before each cutoff.

## FEV-Bench results

100 tasks, 1,308 windows, 235,039 forecasts. Skill scores use the official Seasonal Naive reference; higher is better.

| Version | SQL Skill | MASE Skill | WAPE Skill | WQL Skill |
|---|---:|---:|---:|---:|
| TimeRSI-1.0 | 51.243% | 40.579% | 44.788% | 55.759% |
| TimeRSI-1.5 | 51.822% | 41.440% | 45.400% | 56.059% |

[Per-task results](results/timersi-1_5.csv) include metrics, train/inference times and forecast counts.

## Reproduction

Use Python 3.12+, Docker and a CUDA-capable GPU. From this repository:

```bash
git lfs pull --include='TimeRSI-1.5/src/timersi15/artifacts/*-state-*.zip'
python -m pip install ./TimeRSI-1.5
docker build -t timersi-isolation:1.5 - < TimeRSI-1.5/Dockerfile
```

The [FEV adapter](src/timersi15/runtime.py) runs the supplied programs and historical state chronologically. Early stopping and quantile weights are recomputed using fully matured feedback. Candidate containers receive only current inputs and cutoff-visible historical state, not evaluator targets or the artifact directory. No LLM call is required to execute the released programs.

From the FEV checkout containing the `timersi15` wrapper:

```bash
python models/evaluate.py -m timersi15 -n TimeRSI-1.5 -b benchmarks/fev_bench/tasks.yaml
```

To regenerate deployment decisions over the released program pool, pass `-k '{"regenerate_decisions":true}'`; this requires Claude Code and Vertex AI configuration. The [selection implementation](src/timersi15/search.py), [causal input preparation](src/timersi15/context.py) and [historical evaluator](src/timersi15/probe.py) are included. The shared code-research implementation remains in the root package.

## Runtime

| Median seconds / 100 forecasts | Value |
|---|---:|
| Adaptation: research, historical backtests and candidate preparation | 2121.199 |
| Inference: final selection/composition after candidates are ready | 523.951 |
| End-to-end | 2674.092 |

FEV computes `median_task(100 * (training_time_s + inference_time_s) / num_forecasts)`. Counts cover series-target forecasts across windows, not horizon points. Phase medians need not sum to the end-to-end median.

Reported times are recorded-cost estimates adjusted using paired equivalent-execution measurements; required search attempts remain included. Candidate preparation is included in adaptation, not omitted. A new invocation separately records its actual replay wall time in `RUNTIME.json`; frozen-program replay alone is not the full search procedure.

Hardware: 8 NVIDIA A100-SXM4-80GB GPUs, 96 logical Intel Xeon CPUs and approximately 1.31 TiB RAM. Backbone services were preloaded.
