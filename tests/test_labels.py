from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bwb.data.indices import read_psl_monthly
from bwb.data.labels import active_break_spells, extreme_flags, extreme_thresholds, monsoon_phase, season

ROOT = Path(__file__).resolve().parents[1]


def test_seasons_follow_bmd_convention():
    d = pd.to_datetime(["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15", "2020-12-01"])
    assert season(d).tolist() == ["winter", "pre_monsoon", "monsoon", "post_monsoon", "winter"]
    d2 = pd.to_datetime(["2020-06-10", "2020-06-20", "2020-10-20", "2020-11-20", "2020-03-01"])
    assert monsoon_phase(d2).tolist() == ["onset", "peak", "withdrawal", "dry", "pre_monsoon"]


def _synthetic_rain(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    base = 2 + 12 * np.clip(np.sin((idx.dayofyear - 120) / 365 * 2 * np.pi), 0, None)
    return pd.Series(rng.gamma(2, base / 2), index=idx)


def test_planted_active_and_break_spells_detected():
    rain = _synthetic_rain()
    # Plant in separate years: the standardisation uses each year's own Jul-Aug sd, so an extreme
    # active spell inflates that year's sd and would mask a break spell in the same season.
    rain.loc["2018-07-10":"2018-07-14"] = 60.0  # 5-day active spell
    rain.loc["2019-08-05":"2019-08-09"] = 0.0   # 5-day break spell
    lab = active_break_spells(rain, train_end="2010-12-31")
    assert (lab.loc["2018-07-10":"2018-07-14"] == "active").all()
    assert (lab.loc["2019-08-05":"2019-08-09"] == "break").all()
    assert (lab.loc["2018-01-01":"2018-06-30"] == "").all()


def test_thresholds_use_train_period_only():
    rain = _synthetic_rain()
    df = pd.DataFrame({"Rainfall": rain, "Temperature": 25 + np.sin(rain.index.dayofyear / 58)})
    th = extreme_thresholds(df, train_end="2010-12-31")
    df2 = df.copy()
    df2.loc["2016":, ["Rainfall", "Temperature"]] *= 10  # tamper with the test period
    th2 = extreme_thresholds(df2, train_end="2010-12-31")
    assert th == th2
    flags = extreme_flags(df, th)
    assert flags["very_heavy_rain"].mean() < flags["heavy_rain"].mean()


@pytest.mark.skipif(not (ROOT / "data/external/indices/oni.data").exists(), reason="indices absent")
def test_psl_index_readers():
    oni = read_psl_monthly(ROOT / "data/external/indices/oni.data", "ONI")
    dmi = read_psl_monthly(ROOT / "data/external/indices/dmi.had.long.data", "DMI")
    assert oni.index.min() == pd.Timestamp("1950-01-01") and oni.dropna().between(-3.5, 3.5).all()
    assert dmi.index.min() == pd.Timestamp("1870-01-01") and dmi.dropna().abs().max() < 3
    assert oni.loc["1997-12-01"] > 2  # 1997/98 El Nino
