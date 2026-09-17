from pathlib import Path

import pandas as pd
import pytest

from bwb.data.nasa_power import doy_to_date, load_nasa_power

ROOT = Path(__file__).resolve().parents[1]


def test_doy_to_date_leap_years():
    d = doy_to_date([2020, 2020, 2021, 2024], [60, 366, 60, 1])
    assert list(d.strftime("%Y-%m-%d")) == ["2020-02-29", "2020-12-31", "2021-03-01", "2024-01-01"]


def test_doy_366_invalid_in_non_leap_year():
    with pytest.raises(ValueError):
        doy_to_date([2021], [366])


@pytest.mark.skipif(not (ROOT / "data/All_Districts_Combined.csv").exists(), reason="NASA POWER file absent")
def test_nasa_power_balanced_monotonic_panel():
    df = load_nasa_power(ROOT / "data/All_Districts_Combined.csv")
    dates = df.index.get_level_values("date")
    assert df.index.get_level_values("district").nunique() == 38
    assert dates.min() == pd.Timestamp("2012-01-01") and dates.max() == pd.Timestamp("2024-05-01")
    for _, g in df.groupby(level="district"):
        d = g.index.get_level_values("date")
        assert d.is_monotonic_increasing and (d.to_series().diff().dropna() == pd.Timedelta("1D")).all()


@pytest.mark.skipif(not (ROOT / "data/external/bmd_mendeley/combined/BD_weather.csv").exists(), reason="BMD file absent")
def test_bmd_counts_match_publication():
    from bwb.data.bmd import load_bmd_daily

    df = load_bmd_daily(ROOT / "data/external/bmd_mendeley/combined/BD_weather.csv")
    assert len(df) == 543_839
    assert df.index.is_unique
    starts = df.reset_index().groupby("station").date.min().dt.year
    assert set(starts[starts == 1961].index) == {"Bogra", "Chittagong", "Coxsbazar", "Dhaka", "Sylhet"}
