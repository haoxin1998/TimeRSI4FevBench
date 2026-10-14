# Runtime

FEV aggregates `median_task(100 * (training_time_s + inference_time_s) / num_forecasts)`; the count includes series-target forecasts across all evaluation windows, not horizon points.

The recorded evaluation includes full code research, candidate evaluation and chronological composition. `training_time_s` accounts for pipeline research/adaptation; `inference_time_s` accounts for other active execution, including deployment decisions and local specialist fitting. TimesFM weights remain frozen.

| Recorded median, seconds / 100 forecasts | Value |
|---|---:|
| Research/adaptation | 1,348.761 |
| Online execution | 1,007.643 |
| End-to-end | 2,511.713 |

Hardware: 8 NVIDIA A100-SXM4-80GB GPUs, a 96-logical-CPU Intel Xeon host with approximately 1.31 TiB RAM, and remote Fable 5/high.

Timing scope: TimesFM services were preloaded. Service idle time and measured interruption gaps are excluded; search calls, ordinary waits, candidate computation and active failed attempts are included. The [CSV](../results/timersi.csv) contains per-task counts and times. Separate phase medians do not sum to the end-to-end median.

The public runner also measures container and broker overhead. It allocates its candidate-generation pass to adaptation and final composition to inference; compare end-to-end totals because phase boundaries differ from the recorded accounting.
