"""Loader for the BMD 35-station daily dataset (Zubair et al. 2024, Mendeley DOI 10.17632/tbrhznpwg9.1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

VARIABLES = ["Rainfall", "Temperature", "Humidity", "Sunshine"]

# Mendeley file/station names -> names used in the BMD normals PDFs.
PDF_NAME = {
    "Ambaganctg": "Ambagan(Ctg)",
    "Coxsbazar": "Cox's Bazar",
    "Mcourt": "Maijdi Court",
    "Sydpur": "Sayedpur",
}


def load_bmd_daily(path: str | Path) -> pd.DataFrame:
    """Long-format frame indexed by (station, date), one column per variable."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day), errors="coerce")
    bad = df["date"].isna().sum()
    if bad:
        raise ValueError(f"{bad} rows have invalid calendar dates")
    df = df.rename(columns={"Station": "station"})
    return df.set_index(["station", "date"]).sort_index()[VARIABLES]
