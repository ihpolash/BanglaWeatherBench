"""Leakage guards: split ordering, and consistency of processed outputs with the split config."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/splits.yaml"))
PROC = ROOT / "data/processed"


def _ts(x):
    return pd.Timestamp(x)


@pytest.mark.parametrize("freq", ["daily", "dekadal"])
def test_splits_are_ordered_and_disjoint(freq):
    s = CFG[freq]
    assert _ts(s["train"]["end"]) < _ts(s["val"]["start"]) <= _ts(s["val"]["end"]) < _ts(s["test"]["start"]) <= _ts(s["test"]["end"])
    assert _ts(s["test"]["end"]) < _ts(s["test_extension"]["start"])


def test_daily_splits_are_contiguous():
    s = CFG["daily"]
    assert _ts(s["val"]["start"]) - _ts(s["train"]["end"]) == pd.Timedelta("1D")
    assert _ts(s["test"]["start"]) - _ts(s["val"]["end"]) == pd.Timedelta("1D")


def test_dekadal_boundaries_fall_on_dekads():
    for part in ("train", "val", "test", "test_extension"):
        for k in ("start", "end"):
            v = CFG["dekadal"][part].get(k)
            if v:
                assert _ts(v).day in (1, 11, 21), (part, k, v)


def test_scoring_requires_complete_targets():
    ro = CFG["rolling_origin"]
    assert ro["require_complete_target"] is True
    assert ro["min_history_days"] >= 365
    assert max(ro["daily"]["horizons_days"]) <= 30


@pytest.mark.skipif(not (PROC / "bmd_labels.parquet").exists(), reason="processed labels absent")
def test_labels_align_with_observations_and_spells_only_in_jul_aug():
    obs = pd.read_parquet(PROC / "bmd_daily.parquet", columns=["station", "date"])
    lab = pd.read_parquet(PROC / "bmd_labels.parquet")
    assert len(lab) == len(obs) and not lab.duplicated(["station", "date"]).any()
    spell_months = set(lab.loc[lab.spell != "", "date"].dt.month.unique())
    assert spell_months <= {7, 8}


@pytest.mark.skipif(not (PROC / "national_spells.parquet").exists(), reason="national spells absent")
def test_spell_climatology_period_precedes_test():
    ns = pd.read_parquet(PROC / "national_spells.parquet")
    # The spell reference climatology needs >= 25 reporting stations inside the training period.
    first_valid = ns.national_rain_mm.first_valid_index()
    assert first_valid < _ts(CFG["daily"]["train"]["end"])
