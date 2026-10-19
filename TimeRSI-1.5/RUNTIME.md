# Runtime accounting

The [results CSV](results/timersi-1_5-timestral.csv) reports `training_time_s`, `inference_time_s` and `num_forecasts` for every task.

- **Train:** retained research/adaptation and candidate backtests; no foundation-model weight training.
- **Inference:** retained forecasting/composition, measured Timestral execution and quantile postprocessing, including local specialist fitting used for deployment decisions.

| Median seconds / 100 forecasts | Value |
|---|---:|
| Train | 1174.393 |
| Inference | 894.853 |
| End-to-end | 2224.991 |

FEV computes `median_task(100 * (training_time_s + inference_time_s) / num_forecasts)`. Each phase has its own median, so phase medians need not sum to the end-to-end median. Forecast counts cover series-target forecasts across all windows, not horizon points or GPUs.

This is reconstruction from recorded costs, not a fresh end-to-end timing run. Early stopping avoids enhancement on 31 tasks / 303 windows. Only unused post-stop research/deployment spans and separate composition-agent sessions are deducted; nested calls are not deducted twice. Pre-stop, Raw and unseparated costs remain charged. Measured Timestral execution is retained, not reduced again by window percentages. Quantile postprocessing is measured on actual forecast arrays, excluding archive loading and external score verification.

Recorded hardware: 8 NVIDIA A100-SXM4-80GB GPUs, 96 logical Intel Xeon CPUs and approximately 1.31 TiB RAM; Fable 5/high for inherited research. Backbone services were preloaded; service idle time and measured interruption gaps are excluded.
