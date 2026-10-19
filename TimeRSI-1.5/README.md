# TimeRSI-1.5(Timestral)

This release builds on TimeRSI-1.0's agent-evolved forecasting programs with three changes:

- **Timestral-209M backbone:** execute the existing programs with frozen Timestral instead of TimesFM 3.0.
- **Permanent historical-feedback fallback:** stop enhancement and use direct backbone forecasts when completed historical windows trigger the fallback rules.
- **Adaptive quantile aggregation:** task-local AdaNormalHedge combines raw and enhanced quantiles using fully matured SQL feedback. This aggregation leaves the point-forecast path unchanged.

Foundation-model weights are not fine-tuned. Forecast decisions use only observations, declared known-future covariates and feedback available before each cutoff.

## FEV-Bench results

100 tasks, 1,308 windows, 235,039 forecasts. Skill scores use the official Seasonal Naive reference; higher is better.

| Version | SQL Skill | MASE Skill | WAPE Skill | WQL Skill |
|---|---:|---:|---:|---:|
| TimeRSI-1.0 | 51.243% | 40.579% | 44.788% | 55.759% |
| TimeRSI-1.5(Timestral) | 51.532% | 40.852% | 44.830% | 55.771% |

[Per-task results](results/timersi-1_5-timestral.csv) include train/inference times and forecast counts. The accounted medians are **1,174.393 s train / 894.853 s inference / 2,224.991 s end-to-end per 100 forecasts**. These are stop-aware recorded-cost reconstructions, not a new end-to-end timing run. See [runtime accounting](RUNTIME.md).
