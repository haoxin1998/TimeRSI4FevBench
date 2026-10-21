"""Early stopping and task-local quantile aggregation from matured SQL feedback."""
import numpy as np
from scipy.special import logsumexp


class EarlyStopping:
    def __init__(self, targets, known, past):
        self.enabled = [targets == 1, known + past == 0]
        self.checked = [False, False]
        self.stopped = False

    def decide(self, window, mature, history):
        if self.stopped:
            return True
        for i, (k, threshold) in enumerate(((3, -.05), (8, .02))):
            if not self.enabled[i] or self.checked[i] or len(mature) < k:
                continue
            indexes = list(mature[:k])
            if any(j >= window or history[j].get("loop_SQL") is None for j in indexes):
                raise ValueError("Early stopping requires completed earlier enhancement forecasts")
            raw = np.maximum([history[j]["raw_SQL"] for j in indexes], 1e-12)
            enhanced = np.maximum([history[j]["loop_SQL"] for j in indexes], 1e-12)
            gain = 1 - enhanced / raw
            mean = 1 - float(enhanced.mean() / raw.mean())
            win = float((gain > 0).mean())
            statistic = float(gain.mean() - gain.std() / np.sqrt(k)) if i == 0 else mean
            self.checked[i] = True
            if statistic <= threshold and (i == 0 or win <= .5):
                self.stopped = True
                return True
        return False


def enhanced_weight(matured_sql_pairs=()):
    pairs = np.asarray(matured_sql_pairs, dtype=float).reshape(-1, 2)
    regret, variation = np.zeros(2), np.zeros(2)

    def weights():
        upper = np.maximum(regret + 1, 0) ** 2 / (3 * (variation + 1))
        lower = np.maximum(regret - 1, 0) ** 2 / (3 * (variation + 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            logw = upper + np.log(-np.expm1(lower - upper)) - np.log(2)
            result = np.exp(logw - logsumexp(logw))
        return np.where(np.isfinite(result), result, .5)

    for loss in pairs:
        if not np.isfinite(loss).all() or (loss < 0).any():
            continue
        total = loss.sum()
        scaled = loss / total if total > 0 else np.full(2, .5)
        residual = (weights() * scaled).sum() - scaled
        regret += residual
        variation += abs(residual)
    return float(weights()[1])


def combine(raw, enhanced, matured_sql_pairs=()):
    from timersi_u1.core import Forecast
    raw.validate()
    if raw.point.shape != enhanced.point.shape or raw.quantiles.shape != enhanced.quantiles.shape:
        raise ValueError("Raw and enhanced forecast shapes differ")
    if not np.isfinite(enhanced.point).all() or not np.isfinite(enhanced.quantiles).all():
        return Forecast(raw.point.copy(), raw.quantiles.copy()).validate()
    weight = enhanced_weight(matured_sql_pairs)
    q = enhanced.quantiles.copy() if np.array_equal(raw.quantiles, enhanced.quantiles) else (
        (1 - weight) * raw.quantiles + weight * enhanced.quantiles)
    return Forecast(enhanced.point.copy(), q).validate()
