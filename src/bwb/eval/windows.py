"""Rolling-origin window index and leakage-safe slicing.

A window is (series, origin). The model sees the series up to and including `origin`; it is scored on
the `horizon` days after it. Scoring rule (configs/splits.yaml): the target must be fully observed and lie
inside the evaluated split, and the series needs `min_history_days` of history before the origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)


def make_origins(split_start: str, split_end: str, horizon: int, stride: int) -> pd.DatetimeIndex:
    """Origins such that every target day lies inside [split_start, split_end]."""
    first = pd.Timestamp(split_start) - DAY
    last = pd.Timestamp(split_end) - horizon * DAY
    return pd.date_range(first, last, freq=f"{stride}D")


def build_windows(series: dict, track: str, split_start: str, split_end: str, horizon: int = 30,
                  stride: int = 7, min_history_days: int = 365) -> pd.DataFrame:
    origins = make_origins(split_start, split_end, horizon, stride)
    rows = []
    for (var, sid), s in series.items():
        s = s.asfreq("D")
        if s.empty:
            continue
        obs = s.notna().to_numpy()
        csum = np.concatenate([[0], np.cumsum(obs)])
        first_obs = s.first_valid_index()
        pos = s.index.get_indexer(origins)
        for o, i in zip(origins, pos):
            if i < 0 or i + horizon >= len(s) or first_obs is None or (o - first_obs).days < min_history_days:
                continue
            if csum[i + 1 + horizon] - csum[i + 1] != horizon:  # target not fully observed
                continue
            lo = max(0, i - 364)
            ctx_frac = (csum[i + 1] - csum[lo]) / 365.0
            rows.append((track, var, sid, o, round(ctx_frac, 4)))
    w = pd.DataFrame(rows, columns=["track", "var", "series_id", "origin", "ctx_obs_frac_365"])
    w.insert(0, "window_id", np.arange(len(w)))
    validate_windows(w, split_start, split_end, horizon)
    return w


def validate_windows(w: pd.DataFrame, split_start: str, split_end: str, horizon: int) -> None:
    """Raise if any window's target could touch data outside the evaluated split."""
    if w.empty:
        return
    tgt_start = w["origin"] + DAY
    tgt_end = w["origin"] + horizon * DAY
    bad = (tgt_start < pd.Timestamp(split_start)) | (tgt_end > pd.Timestamp(split_end))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} windows have targets outside [{split_start}, {split_end}]")
    if w.duplicated(["track", "var", "series_id", "origin"]).any():
        raise ValueError("duplicate windows")


def slice_window(s: pd.Series, origin: pd.Timestamp, horizon: int, context_length: int | None = None) -> tuple[pd.Series, pd.Series]:
    """Return (context copy ending exactly at origin, target of `horizon` days). Context never includes target days."""
    ctx = s.loc[:origin]
    if context_length is not None:
        ctx = ctx.iloc[-context_length:]
    tgt = s.loc[origin + DAY: origin + horizon * DAY]
    if len(ctx) and ctx.index[-1] != origin:
        raise ValueError(f"context does not end at origin {origin}")
    if len(tgt) != horizon or (len(ctx) and tgt.index[0] <= ctx.index[-1]):
        raise ValueError("target overlaps context or has wrong length")
    return ctx.copy(), tgt.copy()
