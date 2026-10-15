# TimeRSI for FEV-Bench


TimeRSI uses Claude Code / Fable 5 to develop task-specific forecasting programs around frozen TimesFM 3.0. Agents investigate historical data, implement and evaluate candidates, then compare and combine their predictions for each forecast origin. TimesFM weights are not trained or fine-tuned; generated programs may fit lightweight specialists on available history.

Paper: https://arxiv.org/abs/2610.19725

## FEV-Bench results

100 tasks, 1,308 forecast windows. Skill scores use the official Seasonal Naive reference; higher is better.

| Method | SQL Skill | MASE Skill | WAPE Skill | WQL Skill |
|---|---:|---:|---:|---:|
| Raw TimesFM 3, paired reference | 48.984% | 37.786% | 41.645% | 53.269% |
| TimeRSI | **51.243%** | **40.579%** | **44.788%** | **55.759%** |

Recorded end-to-end runtime: **2,511.713 seconds per 100 forecasts**, using FEV's task-normalized median and including code research, candidate evaluation and chronological composition. See [runtime accounting](docs/RUNTIME.md) and [per-task results](results/timersi.csv).

## Code and inspection

1. **The program:** We provide the [search implementation](src/timersi_u1/search.py), [FEV task adapter](src/timersi_u1/fev_adapter.py) and [installation and evaluation instructions](docs/REPRODUCE.md).
2. **Execution environment:** The public runner uses Docker to isolate code-agent sessions and generated candidates. [Implementation](https://github.com/haoxin1998/TimeRSI4FevBench/blob/main/src/timersi_u1/isolation.py).
3. **Mini20 trajectories:** We release the complete recorded primary agent-session transcripts for all 20 official mini tasks: **821 sessions and 17,992 tool calls**, including prompts, candidate code and feedback. Our review of the recorded calls found no direct retrieval of current or future evaluation targets or scores. [Download](https://github.com/haoxin1998/TimeRSI4FevBench/releases/tag/v1.1.0) · [Artifact index](evidence/mini20-index.json).

See the [data-access protocol](docs/DATA-ACCESS.md) for cutoff handling and evidence scope.

Updated at 2026-10-14 02:02 PDT.
