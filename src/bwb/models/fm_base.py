"""Common interface for zero-shot time-series foundation models.

Every adapter implements `predict_batch(contexts, horizon) -> (mean [B, H], quantiles [B, 9, H])` for the quantile levels
0.1 ... 0.9. Contexts are 1-D float arrays ending at the forecast origin, possibly containing NaN (real gaps).
Adapters declare how they treat gaps: models that accept missing values natively receive NaN; the others receive a
linearly interpolated context (edges filled with the nearest value) - recorded per model in `nan_policy`.
Zero-shot: there is no fitting on BanglaWeatherBench data.
"""

from __future__ import annotations

import numpy as np

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def interpolate_gaps(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN, nearest-value fill at the edges; all-NaN -> zeros."""
    x = np.asarray(x, dtype=np.float32).copy()
    bad = np.isnan(x)
    if not bad.any():
        return x
    if bad.all():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def round_up_to(value: int, multiple: int) -> int:
    """Smallest multiple of `multiple` that is >= value (patch-based models need a context length they can divide)."""
    return -(-int(value) // int(multiple)) * int(multiple)


def trim_leading_nan(x: np.ndarray) -> np.ndarray:
    """Drop NaN before the first observation (history not yet started); keep interior gaps."""
    x = np.asarray(x, dtype=np.float32)
    first = np.flatnonzero(~np.isnan(x))
    return x[first[0]:] if len(first) else x[-1:]


class FoundationModel:
    """Base adapter. Subclasses set `name`, `max_context`, `nan_policy` ('native' or 'interpolate') and implement `_predict`."""

    name = "base"
    max_context = 2048
    nan_policy = "interpolate"
    probabilistic = True

    def __init__(self, device: str = "cuda", batch_size: int = 256):
        self.device, self.batch_size = device, batch_size

    def prepare(self, context: np.ndarray, context_length: int) -> np.ndarray:
        x = trim_leading_nan(np.asarray(context, dtype=np.float32)[-min(context_length, self.max_context):])
        return x if self.nan_policy == "native" else interpolate_gaps(x)

    def _predict(self, contexts: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict_batch(self, contexts: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
        means, qs = [], []
        for i in range(0, len(contexts), self.batch_size):
            m, q = self._predict(contexts[i:i + self.batch_size], horizon)
            means.append(np.asarray(m, dtype=np.float32))
            qs.append(np.asarray(q, dtype=np.float32))
        mean, q = np.concatenate(means), np.concatenate(qs)
        if q.shape[1:] != (len(QUANTILES), horizon) or mean.shape[1:] != (horizon,):
            raise ValueError(f"{self.name}: bad output shapes mean {mean.shape}, quantiles {q.shape}")
        return mean, np.sort(q, axis=1)


def count_parameters(obj, max_depth: int = 4) -> int:
    """Unique parameters of every torch module reachable from an adapter's attributes.

    Adapters hold models in different places (a Chronos pipeline's `.model`, a GluonTS predictor's `.prediction_net`,
    lazily built TTM/Moirai models in dicts), so the attributes are walked instead of assuming one layout. Tied weights
    are counted once. Returns 0 without torch.
    """
    import types

    try:
        from torch import nn
    except ImportError:
        return 0
    params, visited = {}, set()

    def walk(o, depth):
        if depth > max_depth or id(o) in visited or isinstance(o, (type, types.ModuleType, str, bytes)):
            return
        visited.add(id(o))
        if isinstance(o, nn.Module):
            for p in o.parameters():
                params[id(p)] = p.numel()
            return
        if isinstance(o, dict):
            children = list(o.values())
        elif isinstance(o, (list, tuple, set)):
            children = list(o)
        elif hasattr(o, "__dict__"):
            children = list(vars(o).values())
        else:
            return
        for c in children:
            walk(c, depth + 1)

    walk(obj, 0)
    return int(sum(params.values()))
