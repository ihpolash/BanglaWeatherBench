"""Quality-control helpers: flag values that look imputed rather than observed.

The BMD station dataset (Zubair et al. 2024) was published after forward/backward
filling, mean imputation and linear interpolation. Imputed stretches are smoother
than real weather and would inflate forecast skill if scored, so we flag them and
mask them out of the metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """For each position, the length of the run of True values it belongs to (0 if False)."""
    out = np.zeros(len(mask), dtype=int)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def flag_constant_runs(s: pd.Series, min_len: int = 5, ignore_value: float | None = None) -> pd.Series:
    """Flag runs of >= min_len identical consecutive values.

    `ignore_value` exempts a legitimately repeated value (0.0 for rainfall: dry spells are real).
    """
    v = s.to_numpy(dtype=float)
    same_as_prev = np.zeros(len(v), dtype=bool)
    same_as_prev[1:] = np.isclose(v[1:], v[:-1], equal_nan=False)
    # A run of k identical values has k-1 "same as previous" links; mark both ends.
    in_run = same_as_prev.copy()
    in_run[:-1] |= same_as_prev[1:]
    if ignore_value is not None:
        in_run &= ~np.isclose(v, ignore_value)
    lengths = _run_lengths(in_run)
    return pd.Series(lengths >= min_len, index=s.index, name="flag_constant")


def flag_linear_runs(s: pd.Series, min_len: int = 5, atol: float = 1e-6) -> pd.Series:
    """Flag runs of >= min_len points lying on a straight line with non-zero slope (linear interpolation)."""
    v = s.to_numpy(dtype=float)
    d1 = np.diff(v)
    d2 = np.diff(d1)
    # d2 == 0 at position k means v[k], v[k+1], v[k+2] are collinear.
    collinear = np.zeros(len(v), dtype=bool)
    ok = np.isclose(d2, 0.0, atol=atol) & ~np.isclose(d1[1:], 0.0, atol=atol)
    for k in np.flatnonzero(ok):
        collinear[k : k + 3] = True
    lengths = _run_lengths(collinear)
    return pd.Series(lengths >= min_len, index=s.index, name="flag_linear")


def flag_monthly_mean_fill(s: pd.Series, decimals: int = 1) -> pd.Series:
    """Flag values equal to the station's calendar-month mean (mean imputation).

    Requires a DatetimeIndex. Only flags when the matching value occurs on >= 3 days
    of that month-of-year, so a single coincidental match is not flagged.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("flag_monthly_mean_fill needs a DatetimeIndex")
    month = s.index.month
    means = s.groupby(month).transform("mean").round(decimals)
    match = s.round(decimals) == means
    counts = match.groupby(month).transform("sum")
    return (match & (counts >= 3)).rename("flag_mean_fill")


def flag_offgrid_precision(s: pd.Series, decimals: int) -> pd.Series:
    """Flag values finer than the reporting precision (e.g. non-integer humidity): a sign of interpolation or mean fill."""
    v = s.to_numpy(dtype=float)
    off = ~np.isclose(v, np.round(v, decimals), atol=1e-9) & ~np.isnan(v)
    return pd.Series(off, index=s.index, name="flag_offgrid")


def imputation_mask(s: pd.Series, variable: str, min_len: int = 5, decimals: int | None = None) -> pd.DataFrame:
    """Combine the constant-run and linear-run detectors (plus off-grid precision when `decimals` is given).

    Rainfall exempts zero runs.

    `flag_monthly_mean_fill` is deliberately excluded: on BMD 2016-2023 it flags the same share of
    days against a shifted placebo target (e.g. temperature 2.22% real vs 2.40% placebo), so its
    hits are coincidence, not imputation.
    """
    ignore = 0.0 if variable.lower().startswith("rain") else None
    flags = pd.concat(
        [
            flag_constant_runs(s, min_len=min_len, ignore_value=ignore),
            flag_linear_runs(s, min_len=min_len),
        ],
        axis=1,
    )
    if decimals is not None:
        flags["flag_offgrid"] = flag_offgrid_precision(s, decimals)
    flags["flag_any"] = flags.any(axis=1)
    return flags
