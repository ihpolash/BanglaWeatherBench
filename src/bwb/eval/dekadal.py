"""CHIRPS dekadal (10-day) rainfall task: step-based windows, reference baselines, and scoring.

Dekad dates fall on day 1, 11 and 21 of each month, so calendar-day arithmetic does not apply; everything here works
on step positions. Horizons are in dekads (1, 2, 3, 6). Seasonal lag = 36 dekads (one year).
Metric definitions match the daily harness: MASE scaled by in-sample seasonal-naive MAE on the training split,
quantile CRPS, 80% interval coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
QCOLS = [f"q{q:g}" for q in QUANTILES]
SEASON = 36


def dekad_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """0..35."""
    return (dates.month.to_numpy() - 1) * 3 + np.minimum((dates.day.to_numpy() - 1) // 10, 2)


def build_windows(series: dict, split_start: str, split_end: str, horizon: int = 6, stride: int = 1,
                  min_history: int = 36) -> pd.DataFrame:
    rows = []
    for uid, s in series.items():
        idx = s.index
        vals = s.to_numpy()
        for i in range(min_history - 1, len(s) - horizon, stride):
            tgt_dates = idx[i + 1:i + 1 + horizon]
            if tgt_dates[0] < pd.Timestamp(split_start) or tgt_dates[-1] > pd.Timestamp(split_end):
                continue
            if np.isnan(vals[i + 1:i + 1 + horizon]).any():
                continue
            rows.append((uid, idx[i], i))
    w = pd.DataFrame(rows, columns=["series_id", "origin", "origin_pos"])
    w.insert(0, "window_id", np.arange(len(w)))
    return w


class DekadClimatology:
    name = "climatology"

    def fit(self, train: pd.Series):
        tr = train.dropna()
        d = dekad_of_year(tr.index)
        self.mean_ = np.array([tr.to_numpy()[d == k].mean() for k in range(36)])
        self.q_ = np.stack([np.quantile(tr.to_numpy()[d == k], QUANTILES) for k in range(36)], axis=1)
        return self

    def predict(self, s: pd.Series, pos: int, horizon: int):
        d = dekad_of_year(s.index[pos + 1:pos + 1 + horizon])
        return self.mean_[d], self.q_[:, d]


class DekadSeasonalNaive:
    name = "seasonal_naive"

    def fit(self, train: pd.Series):
        self.clim_ = DekadClimatology().fit(train)
        r = (train - train.shift(SEASON)).dropna().to_numpy()
        self.rq_ = np.quantile(r, QUANTILES) - np.median(r)
        return self

    def predict(self, s: pd.Series, pos: int, horizon: int):
        lag_pos = np.arange(pos + 1, pos + 1 + horizon) - SEASON
        lagged = s.to_numpy()[lag_pos]
        mean = np.where(np.isnan(lagged), self.clim_.predict(s, pos, horizon)[0], lagged)
        mean = np.maximum(mean, 0)
        return mean, np.sort(np.maximum(mean[None, :] + self.rq_[:, None], 0), axis=0)


class DekadPersistence:
    name = "naive"

    def fit(self, train: pd.Series):
        self.dq_ = np.stack([np.quantile((train.shift(-h) - train).dropna(), QUANTILES) for h in range(1, 7)], axis=1)
        self.dq_ -= self.dq_[4]
        return self

    def predict(self, s: pd.Series, pos: int, horizon: int):
        ctx = s.iloc[:pos + 1].dropna()
        mean = np.full(horizon, float(ctx.iloc[-1]))
        return mean, np.sort(np.maximum(mean[None, :] + self.dq_[:, :horizon], 0), axis=0)


BASELINES = {"naive": DekadPersistence, "seasonal_naive": DekadSeasonalNaive, "climatology": DekadClimatology}


def run(model_cls, series: dict, windows: pd.DataFrame, train_end: str, horizon: int) -> pd.DataFrame:
    out = []
    for uid, w in windows.groupby("series_id", sort=False):
        s = series[uid]
        m = model_cls().fit(s.loc[:train_end])
        for wid, pos in zip(w.window_id, w.origin_pos):
            view = s.copy()
            view.iloc[pos + 1:] = np.nan  # the model cannot see anything after the origin
            mean, q = m.predict(view if model_cls is not DekadClimatology else s, pos, horizon)
            b = pd.DataFrame(np.asarray(q).T, columns=QCOLS)
            b.insert(0, "mean", mean)
            b.insert(0, "lead", np.arange(1, horizon + 1))
            b.insert(0, "window_id", wid)
            out.append(b)
    return pd.concat(out, ignore_index=True)


def score(preds: pd.DataFrame, windows: pd.DataFrame, series: dict, train_end: str) -> pd.DataFrame:
    w = windows.set_index("window_id")
    scales = {uid: float((s.loc[:train_end] - s.loc[:train_end].shift(SEASON)).abs().mean()) for uid, s in series.items()}
    y = np.empty(len(preds)); sc = np.empty(len(preds))
    for wid, idx in preds.groupby("window_id").indices.items():
        uid, pos = w.loc[wid, "series_id"], int(w.loc[wid, "origin_pos"])
        y[idx] = series[uid].to_numpy()[pos + preds["lead"].to_numpy()[idx]]
        sc[idx] = scales[uid]
    qv = preds[QCOLS].to_numpy()
    diff = y[:, None] - qv
    res = preds[["window_id", "lead"]].copy()
    res["y"] = y
    res["abs_err"] = np.abs(y - preds["q0.5"].to_numpy())  # median point forecast, as in the daily harness
    res["crps"] = 2 * np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean(axis=1)
    res["hit80"] = (y >= qv[:, 0]) & (y <= qv[:, -1])
    res["ase"] = res.abs_err / sc
    res["scaled_crps"] = res.crps / sc
    return res


def check_predictions(preds: pd.DataFrame, windows: pd.DataFrame, horizon: int = 6) -> list[str]:
    """Integrity checks before a dekadal forecast file enters the leaderboard (rules of scripts/verify_predictions.py)."""
    bad = []
    if len(preds) != len(windows) * horizon:
        bad.append(f"rows {len(preds)} != {len(windows) * horizon}")
    if set(preds.window_id.unique()) != set(windows.window_id):
        bad.append("window ids differ from index")
    leads = preds.sort_values(["window_id", "lead"]).lead.to_numpy()
    if len(leads) % horizon or not (leads.reshape(-1, horizon) == np.arange(1, horizon + 1)).all():
        bad.append(f"leads not 1..{horizon} per window")
    if preds[["mean", *QCOLS]].isna().any().any():
        bad.append("missing values")
    if (np.diff(preds[QCOLS].to_numpy(), axis=1) < -1e-6).any():
        bad.append("quantiles not monotone")
    if (preds[["mean", *QCOLS]].to_numpy() < 0).any():
        bad.append("negative rainfall")
    return bad
