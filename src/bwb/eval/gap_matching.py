"""Gap-matched ablation: impose Bangladesh-like context missingness on temperate contexts.

GHCN-Bangladesh contexts are ~82-86% observed over the last 365 days; temperate contexts ~99-100%. A tropical-vs-
temperate skill gap could therefore come from missing context rather than climate or pretraining. This module
draws real missingness patterns (the NaN mask of the last `length` context days) from GHCN-Bangladesh windows and
applies them to temperate contexts, deterministically per window. Targets are never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def donor_masks(series: dict, windows: pd.DataFrame, var: str, length: int) -> np.ndarray:
    """Boolean masks (True = missing) of the last `length` context days for every donor window of `var`."""
    masks = []
    for sid, origin in zip(windows.loc[windows["var"] == var, "series_id"], windows.loc[windows["var"] == var, "origin"]):
        s = series[(var, sid)]
        ctx = s.loc[:origin].to_numpy(dtype=np.float32)[-length:]
        m = np.isnan(ctx)
        if len(m) < length:  # history shorter than the mask: treat the pre-history part as missing
            m = np.concatenate([np.ones(length - len(m), dtype=bool), m])
        masks.append(m)
    if not masks:
        raise ValueError(f"no donor windows for {var}")
    return np.stack(masks)


def apply_mask(context: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set the last len(mask) context days to NaN where the donor was missing (right-aligned at the origin)."""
    out = np.asarray(context, dtype=np.float32).copy()
    k = min(len(out), len(mask))
    tail = out[len(out) - k:]
    tail[mask[len(mask) - k:]] = np.nan
    out[len(out) - k:] = tail
    return out


def assign_donors(recipient_window_ids: np.ndarray, n_donors: int, seed: int = 0) -> np.ndarray:
    """Deterministic donor index per recipient window (same assignment for every model)."""
    rng = np.random.default_rng(seed)
    order = rng.integers(0, n_donors, size=int(np.max(recipient_window_ids)) + 1)
    return order[np.asarray(recipient_window_ids)]
