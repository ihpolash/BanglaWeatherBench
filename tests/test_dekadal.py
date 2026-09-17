import numpy as np
import pandas as pd

from bwb.eval.dekadal import BASELINES, DekadClimatology, build_windows, dekad_of_year, run, score

TRAIN_END, TEST, H = "2010-12-21", ("2016-01-01", "2017-12-21"), 6


def _dekads(start="2000-01-01", end="2018-12-21"):
    months = pd.date_range(start, end, freq="MS")
    return pd.DatetimeIndex(sorted(d + pd.Timedelta(days=k) for d in months for k in (0, 10, 20)))


def _series(seed=0):
    idx = _dekads()
    rng = np.random.default_rng(seed)
    base = 20 + 120 * np.clip(np.sin((dekad_of_year(idx) - 9) / 36 * 2 * np.pi), 0, None)
    return pd.Series(rng.gamma(2, base / 2), index=idx)


def test_dekad_of_year_maps_36_slots():
    idx = _dekads("2020-01-01", "2020-12-21")
    assert dekad_of_year(idx).tolist() == list(range(36))


def test_windows_targets_inside_test_and_complete():
    s = _series()
    s.iloc[600] = np.nan
    w = build_windows({"A": s}, *TEST, horizon=H)
    for pos in w.origin_pos:
        tgt = s.iloc[pos + 1:pos + 1 + H]
        assert tgt.index[0] >= pd.Timestamp(TEST[0]) and tgt.index[-1] <= pd.Timestamp(TEST[1])
        assert tgt.notna().all()
    assert len(w) > 50


def test_baselines_cannot_see_after_origin_and_scores_sane():
    s = _series()
    series = {"A": s}
    w = build_windows(series, *TEST, horizon=H)
    tampered = {"A": s.copy()}
    tampered["A"].loc[TEST[0]:] *= 5
    for name, cls in BASELINES.items():
        p = run(cls, series, w, TRAIN_END, H)
        first = w[w.window_id == 0]
        a = run(cls, series, first, TRAIN_END, H)
        b = run(cls, {"A": pd.concat([s.iloc[:int(first.origin_pos.iloc[0]) + 1], tampered["A"].iloc[int(first.origin_pos.iloc[0]) + 1:]])}, first, TRAIN_END, H)
        assert np.allclose(a["mean"], b["mean"]), name
        sc = score(p, w, series, TRAIN_END)
        assert len(sc) == len(w) * H and sc.ase.notna().all() and sc.hit80.between(0, 1).all()
    clim = score(run(DekadClimatology, series, w, TRAIN_END, H), w, series, TRAIN_END)
    assert 0.5 < clim.hit80.mean() < 1.0


def test_check_predictions_flags_each_problem():
    from bwb.eval.dekadal import QCOLS, check_predictions

    w = pd.DataFrame({"window_id": [0, 1], "series_id": ["A", "A"], "origin_pos": [40, 41]})
    good = pd.DataFrame({"window_id": np.repeat([0, 1], H), "lead": np.tile(np.arange(1, H + 1), 2), "mean": 5.0,
                         **{c: float(i) for i, c in enumerate(QCOLS)}})
    assert check_predictions(good, w, H) == []
    assert any("rows" in b for b in check_predictions(good.iloc[:-1], w, H))
    unsorted = good.copy()
    unsorted["q0.9"] = -1.0
    bad = check_predictions(unsorted, w, H)
    assert "quantiles not monotone" in bad and "negative rainfall" in bad
