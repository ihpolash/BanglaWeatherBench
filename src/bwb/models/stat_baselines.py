"""Statistical baselines (AutoETS, AutoTheta, AutoARIMA) on climatological anomalies.

Daily weather has a dominant annual cycle that ETS/ARIMA cannot fit with season_length=365 on a few years of
context. We model the anomaly from each series' training-period day-of-year climatology, then add the
climatology back. Context gaps are linearly interpolated (edges: nearest value); a context with no observations
falls back to zero anomaly. All windows of a (track, variable) go through one parallel StatsForecast call,
one series per window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

from bwb.eval.gap_matching import apply_mask
from bwb.models.baselines import QUANTILES, Climatology, doy_noleap

MODELS = {"AutoETS": AutoETS, "AutoTheta": AutoTheta, "AutoARIMA": AutoARIMA}
LEVELS = [20, 40, 60, 80]
QCOLS = [f"q{q:g}" for q in QUANTILES]


def _level_columns(name: str) -> list[str]:
    # q0.1..q0.4 = lower bounds of the 80/60/40/20% intervals; q0.5 = point; q0.6..q0.9 = upper bounds of 20/40/60/80%.
    return [f"{name}-lo-80", f"{name}-lo-60", f"{name}-lo-40", f"{name}-lo-20", name,
            f"{name}-hi-20", f"{name}-hi-40", f"{name}-hi-60", f"{name}-hi-80"]


def anomaly_contexts(series: dict, windows: pd.DataFrame, train_end: str, context_length: int, horizon: int,
                     window_masks: dict | None = None):
    """Long frame (unique_id=window_id, ds, y) of gap-filled anomaly contexts, and {window_id: target climatology}.

    `window_masks` maps window_id -> a donor missingness mask (True = missing), applied right-aligned at the origin
    before gap filling. That is the gap-matched control: the same masks the foundation models received.
    """
    frames, clim_by_window = [], {}
    for (var, sid), w in windows.groupby(["var", "series_id"], sort=False):
        s = series[(var, sid)]
        clim = Climatology().fit(s.loc[:train_end])
        anom = s - clim.mean_[doy_noleap(s.index)]
        for wid, origin in zip(w.window_id, w.origin):
            ctx = anom.loc[:origin].iloc[-context_length:]
            if window_masks is not None and wid in window_masks:
                ctx = pd.Series(apply_mask(ctx.to_numpy(dtype=np.float32), window_masks[wid]), index=ctx.index)
            y = ctx.interpolate(limit_direction="both").fillna(0.0).to_numpy()
            frames.append(pd.DataFrame({"unique_id": wid, "ds": np.arange(len(y)), "y": y}))
            clim_by_window[wid] = clim.predict(None, origin, horizon)[0]
    return pd.concat(frames, ignore_index=True), clim_by_window


def run_stat_models(series: dict, windows: pd.DataFrame, train_end: str, horizon: int, model_names=("AutoETS", "AutoTheta"),
                    context_length: int = 730, non_negative: bool = False, n_jobs: int = -1,
                    window_masks: dict | None = None) -> dict[str, pd.DataFrame]:
    """Return {model_name: predictions frame (window_id, lead, mean, q0.1..q0.9)}."""
    long, clim = anomaly_contexts(series, windows, train_end, context_length, horizon, window_masks)
    models = [MODELS[n](season_length=1, alias=n) for n in model_names]
    fc = StatsForecast(models=models, freq=1, n_jobs=n_jobs).forecast(df=long, h=horizon, level=LEVELS)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    fc = fc.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    fc["lead"] = fc.groupby("unique_id").cumcount() + 1
    clim_arr = np.concatenate([clim[w] for w in fc["unique_id"].drop_duplicates()])
    out = {}
    for n in model_names:
        q = np.sort(fc[_level_columns(n)].to_numpy() + clim_arr[:, None], axis=1)
        mean = fc[n].to_numpy() + clim_arr
        if non_negative:
            q, mean = np.maximum(q, 0), np.maximum(mean, 0)
        pred = pd.DataFrame(q, columns=QCOLS)
        pred.insert(0, "mean", mean)
        pred.insert(0, "lead", fc["lead"].to_numpy())
        pred.insert(0, "window_id", fc["unique_id"].to_numpy())
        out[n] = pred
    return out
