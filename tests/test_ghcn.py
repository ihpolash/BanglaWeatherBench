from pathlib import Path

import pandas as pd
import pytest

from bwb.data.ghcn import BG_TO_BMD, load_ghcn_station

ROOT = Path(__file__).resolve().parents[1]
DHAKA = ROOT / "data/external/ghcnd/by_station/BGM00041923.csv.gz"


def test_parses_units_and_drops_failed_qc(tmp_path):
    f = tmp_path / "X.csv"
    f.write_text(
        "X,20200101,PRCP,125,,,S,\n"
        "X,20200101,TAVG,251,H,,S,\n"
        "X,20200102,PRCP,99999,,G,S,\n"  # failed QC (gap check) -> dropped
        "X,20200102,TMAX,-5,,,S,\n"
    )
    df = load_ghcn_station(f)
    assert df.loc["2020-01-01", "Rainfall"] == 12.5
    assert df.loc["2020-01-01", "Tavg"] == 25.1
    assert pd.isna(df.loc["2020-01-02", "Rainfall"])
    assert df.loc["2020-01-02", "Tmax"] == -0.5
    assert list(df.columns) == ["Rainfall", "Tmax", "Tmin", "Tavg"]


def test_station_mapping_covers_ten_distinct_bmd_stations():
    assert len(BG_TO_BMD) == 10 and len(set(BG_TO_BMD.values())) == 10


@pytest.mark.skipif(not DHAKA.exists(), reason="GHCN Dhaka file absent")
def test_dhaka_values_physically_plausible():
    df = load_ghcn_station(DHAKA)
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert df["Tavg"].dropna().between(5, 40).all()
    assert (df["Rainfall"].dropna() >= 0).all()


def test_align_to_bmd_shifts_only_rainfall():
    from bwb.data.ghcn import align_to_bmd_convention

    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    df = pd.DataFrame({"Rainfall": [10.0, 0.0], "Tmax": [30.0, 31.0], "Tmin": [20.0, 21.0], "Tavg": [25.0, 26.0]}, index=idx)
    out = align_to_bmd_convention(df)
    assert out.loc["2020-01-02", "Rainfall"] == 10.0  # start-day total moved to end day
    assert out.loc["2020-01-03", "Rainfall"] == 0.0
    assert pd.isna(out.loc["2020-01-01", "Rainfall"])
    assert out.loc["2020-01-01", "Tavg"] == 25.0  # temperatures unchanged
    assert out.index.freqstr == "D"
