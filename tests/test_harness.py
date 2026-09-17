"""Harness correctness, including deliberate-leakage canaries that must fail loudly."""
import numpy as np
import pandas as pd
import pytest

from bwb.eval.harness import aggregate, run_model, score, train_scales
from bwb.eval.windows import build_windows, slice_window, validate_windows
from bwb.models.baselines import Climatology, Persistence, SeasonalNaive, doy_noleap

TRAIN_END, TEST = "2010-12-31", ("2016-01-01", "2017-12-31")
H = 30


def _series(seed=0, gaps=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", "2018-12-31", freq="D")
    s = pd.Series(20 + 8 * np.sin(2 * np.pi * doy_noleap(idx) / 365) + rng.normal(0, 1, len(idx)), index=idx)
    if gaps:
        s.iloc[rng.choice(len(s), 400, replace=False)] = np.nan
    return s


def test_windows_targets_complete_inside_split_and_contexts_end_at_origin():
    s = _series()
    w = build_windows({("T", "A"): s}, "synthetic", *TEST, horizon=H, stride=7)
    # Independent brute-force count: origins whose 30-day target is fully observed and inside the test split.
    origins = pd.date_range(pd.Timestamp(TEST[0]) - pd.Timedelta(days=1), pd.Timestamp(TEST[1]) - pd.Timedelta(days=H), freq="7D")
    expected = sum(s.loc[o + pd.Timedelta(days=1): o + pd.Timedelta(days=H)].notna().sum() == H for o in origins)
    assert len(w) == expected > 0
    for o in w.origin:
        ctx, tgt = slice_window(s, o, H)
        assert ctx.index[-1] == o and tgt.index[0] == o + pd.Timedelta(days=1)
        assert tgt.notna().all()
        assert tgt.index[0] >= pd.Timestamp(TEST[0]) and tgt.index[-1] <= pd.Timestamp(TEST[1])


def test_leakage_canary_windows_reaching_outside_split_raise():
    leaky = pd.DataFrame({"window_id": [0, 1], "track": "x", "var": "T", "series_id": "A",
                          "origin": pd.to_datetime(["2015-12-20", "2017-12-25"])})  # target starts in val / ends after test
    with pytest.raises(ValueError, match="outside"):
        validate_windows(leaky, *TEST, horizon=H)


class _Spy:
    """Records what the harness exposes; a model that sees future data would be caught here."""
    seen_train_max, seen_ctx_max = [], []

    def fit(self, train):
        _Spy.seen_train_max.append(train.index.max())
        return self

    def predict(self, context, origin, horizon):
        _Spy.seen_ctx_max.append((context.index.max(), origin))
        return np.zeros(horizon), np.zeros((9, horizon))


def test_leakage_canary_harness_never_exposes_future_or_test_data_to_models():
    s = _series()
    w = build_windows({("T", "A"): s}, "synthetic", *TEST, horizon=H, stride=30)
    _Spy.seen_train_max.clear(); _Spy.seen_ctx_max.clear()
    run_model(lambda v: _Spy(), {("T", "A"): s}, w, TRAIN_END, H)
    assert max(_Spy.seen_train_max) <= pd.Timestamp(TRAIN_END)
    assert all(cmax == origin for cmax, origin in _Spy.seen_ctx_max)


def test_leakage_canary_oracle_is_detectably_perfect():
    """If a model could read the target, it would score exactly zero - the harness must make that visible."""
    s = _series(gaps=False)
    series = {("T", "A"): s}
    w = build_windows(series, "synthetic", *TEST, horizon=H, stride=30)

    class Oracle:
        def fit(self, train):
            return self

        def predict(self, context, origin, horizon):
            y = s.loc[origin + pd.Timedelta(days=1): origin + pd.Timedelta(days=horizon)].to_numpy()  # cheats via closure
            return y, np.tile(y, (9, 1))

    sc = score(run_model(lambda v: Oracle(), series, w, TRAIN_END, H), w, series, train_scales(series, TRAIN_END), H)
    assert sc.ase.max() == 0.0 and sc.crps.max() == 0.0


def test_climatology_matches_training_doy_mean_and_ignores_test_period():
    s = _series(gaps=False)
    tampered = s.copy()
    tampered.loc["2011":] += 100.0
    a = Climatology().fit(s.loc[:TRAIN_END]).predict(None, pd.Timestamp("2016-06-30"), H)[0]
    b = Climatology().fit(tampered.loc[:TRAIN_END]).predict(None, pd.Timestamp("2016-06-30"), H)[0]
    assert np.allclose(a, b)
    assert abs(a.mean() - s.loc[:TRAIN_END][s.loc[:TRAIN_END].index.month == 7].mean()) < 0.6


def test_seasonal_naive_uses_lag365_and_falls_back_to_climatology():
    s = _series(gaps=False)
    origin = pd.Timestamp("2016-03-31")
    ctx = s.loc[:origin].copy()
    m = SeasonalNaive().fit(s.loc[:TRAIN_END])
    mean, _ = m.predict(ctx, origin, H)
    lag_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=H) - pd.Timedelta(days=365)
    assert np.allclose(mean, s.reindex(lag_dates).to_numpy())
    ctx.loc[lag_dates[0]] = np.nan
    mean2, _ = m.predict(ctx, origin, H)
    assert np.isclose(mean2[0], m.clim_.predict(None, origin, H)[0][0])


def test_persistence_and_nonnegative_clipping():
    s = _series(gaps=False).clip(lower=0) - 25  # mostly negative values
    m = Persistence(non_negative=True).fit(s.loc[:TRAIN_END])
    mean, q = m.predict(s.loc[:"2016-01-31"], pd.Timestamp("2016-01-31"), H)
    assert (mean >= 0).all() and (q >= 0).all()


def test_scores_and_aggregate_shapes():
    s = _series()
    series = {("T", "A"): s}
    w = build_windows(series, "synthetic", *TEST, horizon=H, stride=14)
    preds = run_model(lambda v: Climatology(), series, w, TRAIN_END, H)
    sc = score(preds, w, series, train_scales(series, TRAIN_END), H)
    assert len(sc) == len(w) * H and sc.y.notna().all()
    lb = aggregate(sc, w)
    assert set(lb.lead) == {1, 3, 7, 14, 30}
    assert (lb.coverage80.between(0, 1)).all() and (lb.MASE > 0).all()


def test_point_forecast_rule_median_for_abs_error_mean_for_squared_error():
    """MASE must use the median for every model; RMSE the mean. Guards against mixing point definitions."""
    s = _series(gaps=False)
    series = {("T", "A"): s}
    w = build_windows(series, "synthetic", *TEST, horizon=H, stride=90)
    preds = run_model(lambda v: Climatology(), series, w, TRAIN_END, H)
    y = score(preds, w, series, train_scales(series, TRAIN_END), H).y.to_numpy()
    shifted = preds.copy()
    shifted["mean"] = shifted["mean"] + 5.0  # moving only the mean must not change absolute error
    a = score(preds, w, series, train_scales(series, TRAIN_END), H)
    b = score(shifted, w, series, train_scales(series, TRAIN_END), H)
    assert np.allclose(a.abs_err, b.abs_err)
    assert np.allclose(a.abs_err, np.abs(y - preds["q0.5"].to_numpy()))
    assert not np.allclose(a.sq_err, b.sq_err)
