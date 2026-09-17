"""Run forecasters over a window index, cache predictions, and score them.

Prediction cache: data/predictions_cache/<model>/<track>__<var>.parquet with columns
window_id, lead, mean, q0.1 ... q0.9. Scoring reads only the cache, so models never need to coexist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bwb.eval.windows import slice_window
from bwb.models.baselines import QUANTILES

QCOLS = [f"q{q:g}" for q in QUANTILES]
CACHE = Path(__file__).resolve().parents[3] / "data" / "predictions_cache"


def cache_path(model: str, track: str, var: str, cache_dir: Path | None = None) -> Path:
    return (cache_dir or CACHE) / model / f"{track}__{var}.parquet"


def run_model(make_model: Callable[[str], object], series: dict, windows: pd.DataFrame, train_end: str,
              horizon: int, context_length: int | None = None) -> pd.DataFrame:
    """Fit one model per series on data <= train_end, then forecast every window of that series."""
    out = []
    for (var, sid), w in windows.groupby(["var", "series_id"], sort=False):
        s = series[(var, sid)]
        model = make_model(var).fit(s.loc[:train_end])
        for wid, origin in zip(w.window_id, w.origin):
            ctx, _ = slice_window(s, origin, horizon, context_length)
            mean, q = model.predict(ctx, origin, horizon)
            block = pd.DataFrame(np.asarray(q, float).T, columns=QCOLS)
            block.insert(0, "mean", np.asarray(mean, float))
            block.insert(0, "lead", np.arange(1, horizon + 1))
            block.insert(0, "window_id", wid)
            out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["window_id", "lead", "mean", *QCOLS])


def train_scales(series: dict, train_end: str, season: int = 365) -> pd.Series:
    """Per-series MASE scale: in-sample MAE of the seasonal-naive forecast on the training split."""
    rows = {}
    for key, s in series.items():
        tr = s.loc[:train_end].asfreq("D")
        d = (tr - tr.shift(season)).abs().dropna()
        rows[key] = float(d.mean()) if len(d) else np.nan
    return pd.Series(rows)


def score(preds: pd.DataFrame, windows: pd.DataFrame, series: dict, scales: pd.Series, horizon: int) -> pd.DataFrame:
    """Per window x lead: absolute error, squared error, quantile-CRPS, 80% interval hit, scaled errors.

    Point-forecast rule (identical for every model): absolute error / MASE use the predictive median (q0.5), the
    Bayes-optimal point under absolute loss; squared error / RMSE use the mean. Mixing them is not neutral on
    zero-inflated rainfall: on BMD rainfall, climatology scored on its mean gives MASE 0.82 at lead 30 but 0.58 on its
    median, which is why an earlier comparison made quantile-loss neural nets look far better than they are.
    """
    w = windows.set_index("window_id")
    obs = np.empty(len(preds))
    scale = np.empty(len(preds))
    for wid, idx in preds.groupby("window_id").indices.items():
        row = w.loc[wid]
        s = series[(row["var"], row["series_id"])]
        tgt = s.loc[row["origin"] + pd.Timedelta(days=1): row["origin"] + pd.Timedelta(days=horizon)].to_numpy()
        leads = preds["lead"].to_numpy()[idx]
        obs[idx] = tgt[leads - 1]
        scale[idx] = scales[(row["var"], row["series_id"])]
    y = obs
    qv = preds[QCOLS].to_numpy()
    diff = y[:, None] - qv
    pinball = np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff)
    res = preds[["window_id", "lead"]].copy()
    res["y"] = y
    res["abs_err"] = np.abs(y - preds["q0.5"].to_numpy())
    res["sq_err"] = (y - preds["mean"].to_numpy()) ** 2
    res["crps"] = 2 * pinball.mean(axis=1)
    res["hit80"] = (y >= qv[:, 0]) & (y <= qv[:, -1])
    res["scale"] = scale
    res["ase"] = res["abs_err"] / scale
    res["scaled_crps"] = res["crps"] / scale
    return res


def aggregate(scores: pd.DataFrame, windows: pd.DataFrame, leads=(1, 3, 7, 14, 30), by=("track", "var")) -> pd.DataFrame:
    """Leaderboard rows: MASE, scaled CRPS, RMSE and 80% coverage at selected leads (mean over windows)."""
    m = scores.merge(windows[["window_id", *by]], on="window_id")
    m = m[m.lead.isin(leads)]
    g = m.groupby([*by, "lead"])
    return pd.DataFrame({
        "MASE": g.ase.mean(), "sCRPS": g.scaled_crps.mean(), "RMSE": np.sqrt(g.sq_err.mean()),
        "coverage80": g.hit80.mean(), "n_windows": g.window_id.nunique(),
    }).reset_index()
