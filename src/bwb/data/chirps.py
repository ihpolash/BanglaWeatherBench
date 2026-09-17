"""Loader for the WFP/HDX CHIRPS subnational dekadal rainfall file (bgd-rainfall-subnat-full.csv).

Source quirk: two administrative units are each split into two polygons that share one PCODE
(BD10 Barisal division: 55 + 345 pixels; BD1009 Bhola district: 55 + 59 pixels). The two series
are different areas, not duplicates, so dropping one would be wrong. `rfh`/`rfh_avg` are
pixel means, so we merge them with pixel-count weights and recompute the anomaly `rfq`.
The merged BD10 pixel count (400) equals the sum over its district children, which confirms this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VALUE_COLS = ["rfh", "rfh_avg", "r1h", "r1h_avg", "r3h", "r3h_avg"]


def _weighted_merge(g: pd.DataFrame) -> pd.Series:
    w = g["n_pixels"].to_numpy(dtype=float)
    out = {"n_pixels": w.sum(), "n_polygons": len(g)}
    for col in VALUE_COLS:
        v = g[col].to_numpy(dtype=float)
        m = ~np.isnan(v)
        out[col] = float(np.average(v[m], weights=w[m])) if m.any() else np.nan
    out["version"] = "prelim" if (g["version"] == "prelim").any() else "final"
    return pd.Series(out)


def load_chirps_dekadal(path: str | Path, adm_level: int = 2) -> pd.DataFrame:
    """Return one row per (PCODE, date) at the requested admin level, with split polygons merged."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["adm_level"] == adm_level]
    merged = (
        df.groupby(["PCODE", "date"], sort=True)[["n_pixels", "version", *VALUE_COLS]]
        .apply(_weighted_merge)
        .reset_index()
    )
    for base in ("rf", "r1", "r3"):
        avg = merged[f"{base}h_avg"]
        merged[f"{base}q"] = np.where(avg > 0, 100.0 * merged[f"{base}h"] / avg, np.nan)
    merged["n_polygons"] = merged["n_polygons"].astype(int)
    return merged
