"""Evaluation-stratum labels: monsoon phases, active/break spells, and extremes.

These labels stratify *scoring* only. They are never model inputs. Anything estimated from data
(daily rainfall climatology, percentile thresholds) is estimated on the training period only.

Definitions
- Seasons follow the BMD convention: winter (Dec-Feb), pre-monsoon (Mar-May), monsoon (Jun-Sep),
  post-monsoon (Oct-Nov).
- Monsoon phases refine the season: Bangladesh monsoon onset is mid-June and withdrawal mid-October
  (Ferdoushi, Quadir & Hassan 2023, Heliyon 9:e20347).
- Active/break spells (same paper): July-August, per station; daily rainfall anomaly relative to the
  daily climatology, standardised by that year's July-August standard deviation; active when the
  standardised anomaly is >= +0.5 and break when <= -0.5, for at least 3 consecutive days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
                   6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post_monsoon", 11: "post_monsoon"}


def season(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.array([SEASON_OF_MONTH[m] for m in dates.month])


def monsoon_phase(dates: pd.DatetimeIndex) -> np.ndarray:
    """Calendar monsoon phase: dry, pre_monsoon, onset (1 Jun-15 Jun), peak (16 Jun-15 Oct), withdrawal (16 Oct-15 Nov)."""
    md = dates.month * 100 + dates.day
    return np.select(
        [(md >= 301) & (md <= 531), (md >= 601) & (md <= 615), (md >= 616) & (md <= 1015), (md >= 1016) & (md <= 1115)],
        ["pre_monsoon", "onset", "peak", "withdrawal"],
        default="dry",
    )


def daily_climatology(s: pd.Series, train_end: str, window: int = 31) -> pd.Series:
    """Day-of-year mean over the training period, smoothed with a circular rolling mean. Index 1..366."""
    tr = s.loc[:train_end].dropna()
    doy = tr.groupby(tr.index.dayofyear).mean().reindex(range(1, 367))
    doy = doy.interpolate(limit_direction="both")
    padded = pd.concat([doy.iloc[-window:], doy, doy.iloc[:window]])
    return padded.rolling(window, center=True, min_periods=1).mean().iloc[window:-window]


def _runs_at_least(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros(len(mask), dtype=bool)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def active_break_spells(rain: pd.Series, train_end: str, threshold: float = 0.5, min_days: int = 3) -> pd.Series:
    """Label July-August days 'active', 'break' or 'normal'; all other days are '' (not applicable)."""
    rain = rain.asfreq("D")
    clim = daily_climatology(rain, train_end)
    anom = rain - clim.reindex(rain.index.dayofyear).to_numpy()
    label = pd.Series("", index=rain.index, dtype=object)
    for _, idx in rain.index[rain.index.month.isin([7, 8])].to_series().groupby(lambda d: d.year):
        a = anom.loc[idx.index]
        sd = a.std()
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (a / sd).to_numpy()
        valid = ~np.isnan(z)
        active = _runs_at_least((z >= threshold) & valid, min_days)
        brk = _runs_at_least((z <= -threshold) & valid, min_days)
        lab = np.where(active, "active", np.where(brk, "break", "normal"))
        lab[~valid] = ""
        label.loc[idx.index] = lab
    return label


def extreme_thresholds(df: pd.DataFrame, train_end: str, rain_col: str = "Rainfall", temp_col: str = "Temperature") -> dict:
    """Training-period thresholds: wet-day (>=1 mm) rainfall 95th/99th percentile; per-calendar-month temperature 95th/5th."""
    tr = df.loc[:train_end]
    wet = tr[rain_col][tr[rain_col] >= 1.0]
    t = tr[temp_col].dropna()
    return {
        "rain_p95": float(wet.quantile(0.95)),
        "rain_p99": float(wet.quantile(0.99)),
        "temp_p95_by_month": t.groupby(t.index.month).quantile(0.95).to_dict(),
        "temp_p05_by_month": t.groupby(t.index.month).quantile(0.05).to_dict(),
    }


def extreme_flags(df: pd.DataFrame, thresholds: dict, rain_col: str = "Rainfall", temp_col: str = "Temperature") -> pd.DataFrame:
    month = df.index.month
    hot = df[temp_col] >= pd.Series(month, index=df.index).map(thresholds["temp_p95_by_month"])
    cold = df[temp_col] <= pd.Series(month, index=df.index).map(thresholds["temp_p05_by_month"])
    return pd.DataFrame({
        "heavy_rain": df[rain_col] >= thresholds["rain_p95"],
        "very_heavy_rain": df[rain_col] >= thresholds["rain_p99"],
        "hot_day": hot,
        "cold_day": cold,
    }, index=df.index)
