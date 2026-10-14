# Data access

At each forecast origin, agents receive observations up to the per-series cutoff, declared known-future covariates and historical backtest evidence. Past-window feedback becomes available only after its forecast end is observable. Current and future target values remain with the evaluator.

1. **The program:** The [search implementation](../src/timersi_u1/search.py) and [FEV task adapter](../src/timersi_u1/fev_adapter.py) expose the research and forecast execution paths. The [full100 cutoff manifest](../evidence/full100-cutoff-manifest.json) records 1,308 origins and 6,098,374 passing feedback/evidence timestamp comparisons.
2. **Execution environment:** The public runner uses Docker to isolate code-agent sessions and generated candidates. [Implementation](https://github.com/haoxin1998/TimeRSI4FevBench/blob/main/src/timersi_u1/isolation.py). Containers have workspace-scoped mounts and no external network access; restricted host brokers provide LLM messages and numeric model predictions. The [Dockerfile](../docker/Dockerfile) is included.
3. **Mini20 inspection:** The [released primary trajectories](https://github.com/haoxin1998/TimeRSI4FevBench/releases/tag/v1.1.0) cover all 20 official mini tasks, 821 sessions and 17,992 tool calls. Automated scans and review of flagged calls and responses found no direct retrieval of current or future evaluation targets or scores in the recorded calls.

The inspection finding concerns the recorded benchmark trajectories; the isolation implementation describes the public runner. The [artifact index](../evidence/mini20-index.json) lists archive checksums, included files, redactions and omitted caches.
