"""Readers for NOAA PSL monthly climate-index text files (ONI, DMI).

Format: a header line "<first_year> <last_year>", then one row per year with 12 monthly values,
then footer notes. Missing values use sentinels such as -99.9 / -9999.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_psl_monthly(path: str | Path, name: str) -> pd.Series:
    lines = Path(path).read_text().splitlines()
    first, last = (int(x) for x in lines[0].split()[:2])
    values = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        if not first <= year <= last:
            continue
        for month, v in enumerate(parts[1:], start=1):
            values[pd.Timestamp(year=year, month=month, day=1)] = float(v)
    s = pd.Series(values, name=name).sort_index()
    return s.mask(s <= -99)
