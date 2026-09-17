"""Loader for the NASA POWER daily district file (All_Districts_Combined.csv), used as the reanalysis track."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FILL_VALUE = -999.0


def doy_to_date(year, doy) -> pd.DatetimeIndex:
    """YEAR + day-of-year -> dates. Handles leap years (DOY 366 only exists in leap years).

    Validates explicitly: pandas' "%Y%j" parser does not reject DOY 366 in a non-leap year.
    """
    year = np.asarray(year, dtype=int)
    doy = np.asarray(doy, dtype=int)
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    bad = (doy < 1) | (doy > np.where(leap, 366, 365))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} invalid day-of-year values, e.g. year={year[bad][0]} doy={doy[bad][0]}")
    return pd.to_datetime(year.astype("int64") * 1000 + doy, format="%Y%j")


def load_nasa_power(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = doy_to_date(df["YEAR"], df["DOY"])
    df = df.replace(FILL_VALUE, np.nan).rename(columns={"Region": "district"})
    df = df.drop(columns=["YEAR", "DOY"]).set_index(["district", "date"]).sort_index()
    if not df.index.is_unique:
        raise ValueError("duplicate (district, date) rows")
    return df
