"""Uniform access to every daily forecasting series: {(variable, series_id): daily pd.Series with NaN gaps}.

Tracks
  bmd                         BMD stations; suspected-imputed values masked to NaN
  ghcn_bangladesh             GHCN-Daily Bangladesh stations (GHCN date convention)
  ghcn_temperate              GHCN-Daily temperate reference stations
  nasa_power_bmd              NASA POWER at the 35 BMD station locations
  nasa_power_ghcn_bangladesh  NASA POWER at the 10 GHCN Bangladesh locations
  nasa_power_ghcn_temperate   NASA POWER at the temperate reference locations
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

PROC = Path(__file__).resolve().parents[3] / "data" / "processed"

TRACK_VARIABLES = {
    "bmd": ["Rainfall", "Temperature", "Humidity", "Sunshine"],
    "ghcn_bangladesh": ["Rainfall", "Tavg"],
    "ghcn_temperate": ["Rainfall", "Tavg"],
    "nasa_power_bmd": ["PRECTOTCORR", "T2M", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "nasa_power_ghcn_bangladesh": ["PRECTOTCORR", "T2M"],
    "nasa_power_ghcn_temperate": ["PRECTOTCORR", "T2M"],
}

# Variables that are physically non-negative (forecasts are clipped at 0).
NON_NEGATIVE = {"Rainfall", "PRECTOTCORR", "Sunshine", "ALLSKY_SFC_SW_DWN"}


@lru_cache(maxsize=None)
def _frame(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / name)


def load_track(track: str, proc_dir: Path | None = None) -> dict[tuple[str, str], pd.Series]:
    if proc_dir is not None:
        global PROC
        PROC = Path(proc_dir)
        _frame.cache_clear()
    out: dict[tuple[str, str], pd.Series] = {}
    variables = TRACK_VARIABLES[track]
    if track == "bmd":
        df = _frame("bmd_daily.parquet")
        for station, g in df.groupby("station"):
            g = g.set_index("date").sort_index()
            for v in variables:
                out[(v, station)] = g[v].mask(g[f"{v}_flag"]).asfreq("D").rename(v)
    elif track.startswith("ghcn_"):
        df = _frame("ghcn_daily.parquet")
        df = df[df.region == track.removeprefix("ghcn_")]
        for sid, g in df.groupby("station_id"):
            g = g.set_index("date").sort_index()
            for v in variables:
                out[(v, sid)] = g[v].asfreq("D").rename(v)
    elif track.startswith("nasa_power_"):
        df = _frame("nasa_power_points.parquet")
        sub = track.removeprefix("nasa_power_")
        if sub == "bmd":
            df = df[df.track == "bmd"]
        else:
            df = df[(df.track == "ghcn") & (df.region == sub.removeprefix("ghcn_"))]
        for sid, g in df.groupby("station_id"):
            g = g.set_index("date").sort_index()
            for v in variables:
                out[(v, sid)] = g[v].asfreq("D").rename(v)
    else:
        raise KeyError(f"unknown track {track!r}")
    return out
