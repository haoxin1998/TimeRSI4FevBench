"""Timestral through the shared native forecasting interface."""
from types import SimpleNamespace

import numpy as np


class TimestralBackend:
    def __init__(self, checkpoint="mistralai/Timestral-209M", device="cuda:0", model=None, max_batch=4):
        self.checkpoint, self.device = checkpoint, device
        self.model, self.max_batch = model, max_batch

    def predict_batch(self, *, contexts, horizon, past_only_covariates=None,
                      past_future_covariates=None, return_quantiles=True,
                      use_symmetric_averaging=True, make_positive=True, sort_quantiles=True):
        if not return_quantiles or not sort_quantiles:
            raise ValueError("Ordered quantile forecasts are required")
        if self.model is None:
            from timestral import TimestralForecaster
            self.model = TimestralForecaster.from_pretrained(self.checkpoint, device=self.device,
                revision="9874d252ab0aae16f7f7c999f98f98d6ad7d8365")
        past = past_only_covariates or [None] * len(contexts)
        known = past_future_covariates or [None] * len(contexts)
        if len(contexts) != len(past) or len(contexts) != len(known):
            raise ValueError("Context and covariate counts differ")
        rows = [tuple(None if x is None else np.asarray(x, np.float32) for x in row)
                for row in zip(contexts, past, known)]
        index = 0
        while index < len(rows):
            first = rows[index]
            shape = tuple(None if x is None else x.shape for x in first)
            cells = sum(x.size for x in first if x is not None)
            size = max(1, min(self.max_batch, 131072 // max(cells, 1)))
            end = index + 1
            while end < min(index + size, len(rows)) and tuple(
                    None if x is None else x.shape for x in rows[end]) == shape:
                end += 1
            batch = rows[index:end]
            inputs = [None if first[j] is None else np.stack([r[j] for r in batch]) for j in range(3)]
            result = self.model.forecast(inputs[0], prediction_length=horizon,
                past_covariates=inputs[1], known_covariates=inputs[2],
                symmetric_averaging=use_symmetric_averaging, clip_nonnegative=make_positive)
            levels = [result.quantile_levels.index(i / 10) for i in range(1, 10)]
            quantiles = result.quantiles[..., levels].detach().cpu().numpy()
            expected = (len(batch), first[0].shape[0], horizon, 9)
            if quantiles.shape != expected or not np.isfinite(quantiles).all():
                raise ValueError("Invalid Timestral output")
            for q in quantiles:
                yield SimpleNamespace(forecast=q[..., 4], quantiles=q)
            index = end
