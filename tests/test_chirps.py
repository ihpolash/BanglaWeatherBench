from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bwb.data.chirps import load_chirps_dekadal

RAW = Path(__file__).resolve().parents[1] / "data" / "bgd-rainfall-subnat-full.csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="CHIRPS raw file not present")


@pytest.fixture(scope="module")
def adm2():
    return load_chirps_dekadal(RAW, adm_level=2)


def test_adm2_is_a_complete_unique_panel(adm2):
    assert not adm2.duplicated(["PCODE", "date"]).any()
    assert adm2["PCODE"].nunique() == 64
    assert adm2["date"].nunique() == 1644
    assert len(adm2) == 64 * 1644


def test_dekad_dates_only(adm2):
    assert set(adm2["date"].dt.day.unique()) <= {1, 11, 21}


def test_split_polygons_are_pixel_weighted():
    raw = pd.read_csv(RAW, parse_dates=["date"])
    one = raw[(raw.PCODE == "BD1009") & (raw.date == raw.date.min())]
    expected = np.average(one["rfh"], weights=one["n_pixels"])
    got = load_chirps_dekadal(RAW, 2).query("PCODE == 'BD1009'").iloc[0]
    assert got["n_polygons"] == 2
    assert got["n_pixels"] == one["n_pixels"].sum()
    assert np.isclose(got["rfh"], expected)


def test_adm1_barisal_pixels_match_district_children():
    adm1 = load_chirps_dekadal(RAW, adm_level=1)
    adm2 = load_chirps_dekadal(RAW, adm_level=2)
    d = adm1["date"].min()
    bd10 = adm1[(adm1.PCODE == "BD10") & (adm1.date == d)]["n_pixels"].item()
    children = adm2[(adm2.PCODE.str[:4] == "BD10") & (adm2.date == d)]["n_pixels"].sum()
    assert bd10 == children == 400
