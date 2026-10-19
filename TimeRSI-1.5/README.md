# TimeRSI-1.5(Timestral)

This directory contains the version's results and method summary. The shared TimeRSI-1.0 search implementation remains in the [root package](../src/timersi_u1/search.py).

This release builds on TimeRSI-1.0's agent-evolved forecasting programs with three changes:

- **Timestral-209M backbone:** execute the existing programs with frozen Timestral instead of TimesFM 3.0.
- **Early stopping for efficiency:** use completed historical feedback to stop unnecessary enhancement and continue with direct backbone forecasts.
- **Adaptive quantile aggregation:** task-local AdaNormalHedge combines raw and enhanced quantiles using fully matured SQL feedback. This aggregation leaves the point-forecast path unchanged.

Foundation-model weights are not fine-tuned. Forecast decisions use only observations, declared known-future covariates and feedback available before each cutoff.

## FEV-Bench results

100 tasks, 1,308 windows, 235,039 forecasts. Skill scores use the official Seasonal Naive reference; higher is better.

| Version | SQL Skill | MASE Skill | WAPE Skill | WQL Skill |
|---|---:|---:|---:|---:|
| TimeRSI-1.0 | 51.243% | 40.579% | 44.788% | 55.759% |
| TimeRSI-1.5(Timestral) | 51.532% | 40.852% | 44.830% | 55.771% |

[Per-task results](results/timersi-1_5-timestral.csv) include metrics, train/inference times and forecast counts.

## Runtime

| Median seconds / 100 forecasts | Value |
|---|---:|
| Train: research/adaptation and candidate backtests | 1174.393 |
| Inference: forecasting/composition, Timestral and quantile processing | 894.853 |
| End-to-end | 2224.991 |

FEV computes `median_task(100 * (training_time_s + inference_time_s) / num_forecasts)`. Counts cover series-target forecasts across windows, not horizon points. Phase medians need not sum to the end-to-end median.

Runtime is reconstructed from recorded costs with verified unused post-stop work deducted, not a fresh end-to-end measurement. Pre-stop work, Raw forecasting, active failed attempts and unseparated costs remain included; nested calls are not deducted twice. Measured Timestral execution and quantile processing are retained. Archive loading and external score verification are excluded.

Hardware: 8 NVIDIA A100-SXM4-80GB GPUs, 96 logical Intel Xeon CPUs and approximately 1.31 TiB RAM; Fable 5/high for inherited research. Backbone services were preloaded; service idle time and measured interruption gaps are excluded.
