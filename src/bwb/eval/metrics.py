"""Point and probabilistic forecast metrics."""

from __future__ import annotations

import numpy as np


def seasonal_naive_scale(train: np.ndarray, season: int) -> float:
    """In-sample mean absolute error of the seasonal-naive forecast y[t] = y[t - season]."""
    train = np.asarray(train, dtype=float)
    if len(train) <= season:
        raise ValueError(f"need more than {season} training points, got {len(train)}")
    return float(np.nanmean(np.abs(train[season:] - train[:-season])))


def mase(y_true, y_pred, train, season: int) -> float:
    """Mean absolute scaled error, scaled by the in-sample seasonal-naive MAE."""
    scale = seasonal_naive_scale(train, season)
    if scale == 0:
        return float("nan")
    err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return float(np.nanmean(err) / scale)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.nanmean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def pinball_loss(y_true, q_pred, q: float) -> float:
    diff = np.asarray(y_true, float) - np.asarray(q_pred, float)
    return float(np.nanmean(np.maximum(q * diff, (q - 1) * diff)))


def crps_from_quantiles(y_true, quantile_preds: np.ndarray, quantiles: np.ndarray) -> float:
    """Approximate CRPS as twice the mean pinball loss over the quantile levels.

    quantile_preds has shape (n_quantiles, horizon).
    """
    quantile_preds = np.asarray(quantile_preds, float)
    losses = [pinball_loss(y_true, quantile_preds[i], q) for i, q in enumerate(quantiles)]
    return float(2.0 * np.mean(losses))


def interval_coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, float)
    return float(np.nanmean((y >= np.asarray(lower, float)) & (y <= np.asarray(upper, float))))
