# Running TimeRSI

Requirements: Linux, Docker, a CUDA GPU for TimesFM 3, Python 3.12, Claude Code and access to Fable 5 through Vertex AI. All credentials remain on the evaluator host, outside agent/candidate containers.

```bash
git clone https://github.com/haoxin1998/TimeRSI4FevBench.git
cd TimeRSI4FevBench
pip install -e .
pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git@45e0a3bc7fc4acef17b7ba7910488be2159bae5f'
docker build -t timersi-isolation:1.1 -f docker/Dockerfile docker
export TIMERSI_VERTEX_PROJECT=YOUR_PROJECT
gcloud auth application-default login
```

Install Claude Code 2.1.245 and make `claude` available on PATH, or set `CLAUDE_CODE_BINARY` to its native executable. Configure/download `google/timesfm-3.0-pytorch`; the FEV wrapper accepts `checkpoint_path` for a local checkpoint.

Copy `fev_submission/models/timersi` to `models/timersi` in a checkout of `autogluon/fev`. Install the package locally for development, or use the pinned release requirement. Run from the FEV checkout:

```bash
python models/evaluate.py -m timersi -n TimeRSI \
  -b benchmarks/fev_bench/tasks.yaml \
  -k '{"checkpoint_path":"/path/to/timesfm-3.0-pytorch"}'
```

Use `benchmarks/fev_bench/tasks_mini.yaml` for the official mini20. Each task starts independently and saves research, composition, predictions and timing beneath `timersi-runs/`. Use a service manager or detached session for long runs.

Defaults: Fable 5/high; up to 12 initial probes; adaptive follow-up budgets of 2–4 probes; three research folds and one selection fold when available; 64 research entities; the complete official evaluation population.

All tasks use the same policy. Research produces code and initial deployment decisions; chronological composition uses those same-origin candidates with its own matured feedback. Neither pass receives current/future targets. Fresh searches are stochastic; recorded results and trajectories are linked from the [README](../README.md).
