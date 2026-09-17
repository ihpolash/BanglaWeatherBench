"""Loader for GHCN-Daily by-station CSV files (NOAA NCEI).

Row format (no header): ID, YYYYMMDD, ELEMENT, VALUE, MFLAG, QFLAG, SFLAG, OBS-TIME.
PRCP is in tenths of mm; TMAX/TMIN/TAVG in tenths of degrees C. A non-empty QFLAG means the
value failed an NCEI quality check, so it is dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["id", "date", "element", "value", "mflag", "qflag", "sflag", "obs_time"]
ELEMENTS = {"PRCP": "Rainfall", "TMAX": "Tmax", "TMIN": "Tmin", "TAVG": "Tavg"}

# GHCN Bangladesh station -> co-located BMD station name (Mendeley file naming).
BG_TO_BMD = {
    "BGM00041859": "Rangpur",
    "BGM00041883": "Bogra",
    "BGM00041891": "Sylhet",
    "BGM00041907": "Ishurdi",
    "BGM00041923": "Dhaka",
    "BGM00041936": "Jessore",
    "BGM00041943": "Feni",
    "BGM00041950": "Barisal",
    "BGM00041978": "Chittagong",
    "BGM00041992": "Coxsbazar",
}


# GHCN (GSOD-derived) labels a 24-h rainfall total by its start day; BMD labels it by its end day.
# Shifting GHCN PRCP +1 day raises daily r vs BMD from 0.31-0.63 to 0.68-0.90 at all 10 co-located
# stations (2000-2023). Temperature needs no shift (Tavg r = 0.95-0.99 at lag 0).
RAIN_SHIFT_TO_BMD_DAYS = 1


def align_to_bmd_convention(df: pd.DataFrame) -> pd.DataFrame:
    """Return a daily frame with Rainfall relabelled to BMD's end-of-period date convention."""
    rain = df["Rainfall"].copy()
    rain.index = rain.index + pd.Timedelta(days=RAIN_SHIFT_TO_BMD_DAYS)
    out = df.drop(columns="Rainfall").join(rain.rename("Rainfall"), how="outer")
    return out.reindex(columns=df.columns).asfreq("D")


def load_ghcn_station(path: str | Path, drop_failed_qc: bool = True) -> pd.DataFrame:
    """Wide daily frame indexed by date with columns Rainfall (mm), Tmax, Tmin, Tavg (deg C).

    Days with no report are absent from the index; call `.asfreq("D")` to expose gaps as NaN.
    """
    raw = pd.read_csv(path, header=None, names=COLUMNS, dtype={"qflag": "string", "mflag": "string"})
    raw = raw[raw["element"].isin(list(ELEMENTS))]
    if drop_failed_qc:
        raw = raw[raw["qflag"].isna() | (raw["qflag"].str.strip() == "")]
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    raw["value"] = raw["value"] / 10.0
    wide = raw.pivot_table(index="date", columns="element", values="value", aggfunc="first")
    return wide.rename(columns=ELEMENTS).reindex(columns=list(ELEMENTS.values())).sort_index()
