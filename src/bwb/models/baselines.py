"""Naive reference forecasters. Every model is fitted on the training split only and predicts from a context.

Interface: `fit(train: pd.Series) -> self`; `predict(context: pd.Series, origin, horizon) -> (mean[H], quantiles[Q, H])`.
Target calendar dates are known at forecast time (origin + lead), so calendar lookups are not leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
DAY = pd.Timedelta(days=1)


def doy_noleap(dates: pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 365-day calendar (29 Feb shares day 59 with 28 Feb)."""
    doy = dates.dayofyear.to_numpy().copy()
    leap_after_feb = dates.is_leap_year & (dates.month > 2)
    doy[leap_after_feb] -= 1
    return np.minimum(doy, 365)


def _target_dates(origin, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(origin) + DAY, periods=horizon, freq="D")


class Climatology:
    """Training-period day-of-year distribution, pooled over a +-`half_window`-day calendar window."""

    name = "climatology"

    def __init__(self, half_window: int = 15, non_negative: bool = False):
        self.half_window, self.non_negative = half_window, non_negative

    def fit(self, train: pd.Series):
        tr = train.dropna()
        if tr.empty:
            raise ValueError("no training data")
        d = doy_noleap(tr.index)
        vals = tr.to_numpy()
        self.mean_ = np.full(366, np.nan)
        self.q_ = np.full((len(QUANTILES), 366), np.nan)
        for day in range(1, 366):
            dist = np.abs(d - day)
            dist = np.minimum(dist, 365 - dist)
            sel = vals[dist <= self.half_window]
            if len(sel):
                self.mean_[day] = sel.mean()
                self.q_[:, day] = np.quantile(sel, QUANTILES)
        return self

    def predict(self, context, origin, horizon):
        d = doy_noleap(_target_dates(origin, horizon))
        return self.mean_[d].copy(), self.q_[:, d].copy()


class SeasonalNaive:
    """y(t) = y(t - 365 days); gaps fall back to climatology. Quantiles add training lag-365 residual quantiles."""

    name = "seasonal_naive"

    def __init__(self, non_negative: bool = False):
        self.non_negative = non_negative

    def fit(self, train: pd.Series):
        self.clim_ = Climatology(non_negative=self.non_negative).fit(train)
        s = train.asfreq("D")
        resid = (s - s.shift(365)).dropna().to_numpy()
        self.resid_q_ = np.quantile(resid, QUANTILES) - np.median(resid) if len(resid) else np.zeros(len(QUANTILES))
        return self

    def predict(self, context, origin, horizon):
        dates = _target_dates(origin, horizon)
        lagged = context.reindex(dates - pd.Timedelta(days=365)).to_numpy()
        clim_mean, _ = self.clim_.predict(context, origin, horizon)
        mean = np.where(np.isnan(lagged), clim_mean, lagged)
        q = mean[None, :] + self.resid_q_[:, None]
        if self.non_negative:
            mean, q = np.maximum(mean, 0), np.maximum(q, 0)
        return mean, np.sort(q, axis=0)


class Persistence:
    """y(origin + h) = last observed value; quantiles add training h-day change quantiles."""

    name = "naive"

    def __init__(self, max_horizon: int = 30, non_negative: bool = False):
        self.max_horizon, self.non_negative = max_horizon, non_negative

    def fit(self, train: pd.Series):
        s = train.asfreq("D")
        self.dq_ = np.zeros((len(QUANTILES), self.max_horizon))
        for h in range(1, self.max_horizon + 1):
            d = (s.shift(-h) - s).dropna().to_numpy()
            if len(d):
                self.dq_[:, h - 1] = np.quantile(d, QUANTILES) - np.median(d)
        self.fallback_ = float(train.dropna().mean())
        return self

    def predict(self, context, origin, horizon):
        last = context.dropna()
        level = float(last.iloc[-1]) if len(last) else self.fallback_
        mean = np.full(horizon, level)
        q = mean[None, :] + self.dq_[:, :horizon]
        if self.non_negative:
            mean, q = np.maximum(mean, 0), np.maximum(q, 0)
        return mean, np.sort(q, axis=0)


BASELINES = {"naive": Persistence, "seasonal_naive": SeasonalNaive, "climatology": Climatology}
