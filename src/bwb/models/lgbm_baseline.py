"""Global LightGBM baseline: one model per (track, variable) across all series, direct multi-horizon.

Target: anomaly y(origin + h) - climatology(origin + h). Features use only data up to the origin:
recent anomalies (lags 0-6), anomaly means over 7/30/90 days, context completeness over 30 days, lead h,
target day-of-year (sin/cos), target climatology mean and spread, and the series id (categorical).
Training samples come from origins in the training split; the validation split gives early stopping and
per-lead residual quantiles for the predictive intervals.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from bwb.models.baselines import QUANTILES, Climatology, doy_noleap

QCOLS = [f"q{q:g}" for q in QUANTILES]
FEATURES = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "m7", "m30", "m90", "obs30", "lead", "doy_sin", "doy_cos",
            "clim_mean", "clim_spread", "sid"]


def _features(anom: np.ndarray, clim_mean: np.ndarray, clim_spread: np.ndarray, dates: pd.DatetimeIndex,
              pos: np.ndarray, horizon: int, sid_code: int) -> pd.DataFrame:
    """Feature rows for origins at integer positions `pos` (one row per origin x lead). Uses anom[:pos+1] only."""
    n = len(pos)
    lags = np.stack([anom[pos - k] for k in range(7)], axis=1)
    csum = np.concatenate([[0.0], np.nancumsum(anom)])
    cobs = np.concatenate([[0], np.cumsum(~np.isnan(anom))])

    def window_mean(L):
        lo = np.maximum(pos + 1 - L, 0)
        cnt = cobs[pos + 1] - cobs[lo]
        return np.where(cnt > 0, (csum[pos + 1] - csum[lo]) / np.maximum(cnt, 1), np.nan)

    obs30 = (cobs[pos + 1] - cobs[np.maximum(pos + 1 - 30, 0)]) / 30.0
    base = np.column_stack([lags, window_mean(7), window_mean(30), window_mean(90), obs30])
    rows = np.repeat(base, horizon, axis=0)
    lead = np.tile(np.arange(1, horizon + 1), n)
    tgt_pos = np.repeat(pos, horizon) + lead
    tdates = dates[np.minimum(tgt_pos, len(dates) - 1)] if len(dates) > tgt_pos.max() else pd.DatetimeIndex(
        np.repeat(dates[pos], horizon) + pd.to_timedelta(lead, unit="D"))
    doy = doy_noleap(pd.DatetimeIndex(tdates))
    df = pd.DataFrame(rows, columns=FEATURES[:11])
    df["lead"] = lead
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["clim_mean"] = clim_mean[doy]
    df["clim_spread"] = clim_spread[doy]
    df["sid"] = sid_code
    return df, doy


class GlobalLGBM:
    name = "lightgbm"

    def __init__(self, horizon=30, train_stride=10, non_negative=False, seed=0, num_boost_round=600):
        self.horizon, self.train_stride, self.non_negative = horizon, train_stride, non_negative
        self.seed, self.num_boost_round = seed, num_boost_round

    def _prep(self, series, train_end):
        self.clim_, self.codes_ = {}, {}
        for i, (key, s) in enumerate(sorted(series.items())):
            c = Climatology().fit(s.loc[:train_end])
            self.clim_[key] = c
            self.codes_[key] = i

    def _samples(self, series, start, end):
        X, y = [], []
        for key, s in series.items():
            s = s.asfreq("D")
            c = self.clim_[key]
            anom = (s - c.mean_[doy_noleap(s.index)]).to_numpy()
            spread = c.q_[-1] - c.q_[0]
            idx = np.flatnonzero((s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end) - pd.Timedelta(days=self.horizon)))
            idx = idx[(idx >= 90) & (idx % self.train_stride == 0)]
            if not len(idx):
                continue
            f, _ = _features(anom, c.mean_, spread, s.index, idx, self.horizon, self.codes_[key])
            tgt = anom[np.repeat(idx, self.horizon) + f["lead"].to_numpy()]
            keep = ~np.isnan(tgt)
            X.append(f[keep]); y.append(tgt[keep])
        return pd.concat(X, ignore_index=True), np.concatenate(y)

    def fit(self, series: dict, train_end: str, val_start: str, val_end: str):
        self._prep(series, train_end)
        Xtr, ytr = self._samples(series, "1900-01-01", train_end)
        Xva, yva = self._samples(series, val_start, val_end)
        params = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, seed=self.seed, verbose=-1, num_threads=0)
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=["sid"], free_raw_data=True)
        dva = lgb.Dataset(Xva, yva, categorical_feature=["sid"], reference=dtr)
        self.booster_ = lgb.train(params, dtr, self.num_boost_round, valid_sets=[dva],
                                  callbacks=[lgb.early_stopping(50, verbose=False)])
        resid = yva - self.booster_.predict(Xva, num_iteration=self.booster_.best_iteration)
        rq = pd.DataFrame({"lead": Xva["lead"].to_numpy(), "r": resid}).groupby("lead").r
        self.resid_q_ = np.stack([rq.quantile(q).reindex(range(1, self.horizon + 1)).to_numpy() for q in QUANTILES])
        self.resid_q_ -= self.resid_q_[4]  # centre on the median residual
        return self

    def predict_windows(self, series: dict, windows: pd.DataFrame) -> pd.DataFrame:
        out = []
        for (var, sid), w in windows.groupby(["var", "series_id"], sort=False):
            key = (var, sid)
            s = series[key].asfreq("D")
            c = self.clim_[key]
            origin_pos = s.index.get_indexer(w.origin)
            anom = (s - c.mean_[doy_noleap(s.index)]).to_numpy()
            # Blank everything after each origin is handled inside _features (it reads anom[pos-k] only);
            # still, hide post-origin values defensively per window batch by truncating at the latest origin.
            anom = anom.copy()
            anom[origin_pos.max() + 1:] = np.nan
            f, doy = _features(anom, c.mean_, c.q_[-1] - c.q_[0], s.index, origin_pos, self.horizon, self.codes_[key])
            pred_anom = self.booster_.predict(f[FEATURES], num_iteration=self.booster_.best_iteration)
            clim_t = c.mean_[doy]
            mean = clim_t + pred_anom
            q = mean[:, None] + self.resid_q_[:, f["lead"].to_numpy() - 1].T
            if self.non_negative:
                mean, q = np.maximum(mean, 0), np.maximum(q, 0)
            block = pd.DataFrame(np.sort(q, axis=1), columns=QCOLS)
            block.insert(0, "mean", mean)
            block.insert(0, "lead", f["lead"].to_numpy())
            block.insert(0, "window_id", np.repeat(w.window_id.to_numpy(), self.horizon))
            out.append(block)
        return pd.concat(out, ignore_index=True)
