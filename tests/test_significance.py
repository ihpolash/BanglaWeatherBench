"""Significance machinery: the tests check behaviour that would otherwise silently overstate confidence."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from bwb.eval.significance import (align, block_indices, bootstrap_ci, bootstrap_ci_multi, diebold_mariano,
                                   friedman_nemenyi, holm, per_origin_loss, per_station_loss, wilcoxon_paired)


def _panel(seed=0, n_origins=200, n_stations=5):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2016-01-01", periods=n_origins, freq="7D")
    rows = [{"window_id": i * n_stations + j, "origin": o, "series_id": f"S{j}"}
            for i, o in enumerate(origins) for j in range(n_stations)]
    w = pd.DataFrame(rows)
    scores = pd.DataFrame({"window_id": w.window_id, "lead": 7, "scaled_crps": rng.gamma(2, 0.5, len(w))})
    return scores, w


def test_per_origin_and_per_station_aggregation():
    scores, w = _panel()
    o = per_origin_loss(scores, w, lead=7)
    s = per_station_loss(scores, w, lead=7)
    assert len(o) == 200 and o.index.is_monotonic_increasing
    assert len(s) == 5
    assert np.isclose(o.mean(), scores.scaled_crps.mean())
    assert per_origin_loss(scores, w, lead=30).empty  # lead not present


def test_diebold_mariano_identical_forecasts_are_not_significant():
    scores, w = _panel()
    loss = per_origin_loss(scores, w, lead=7)
    r = diebold_mariano(loss, loss, horizon=7)
    assert r.mean_diff == 0 and r.p_value == 1.0


def test_diebold_mariano_detects_a_genuinely_better_model_and_signs_correctly():
    rng = np.random.default_rng(1)
    b = rng.gamma(2, 0.5, 300)
    a = b - 0.25 + rng.normal(0, 0.05, 300)  # A is better by a clear margin
    r = diebold_mariano(a, b, horizon=7)
    assert r.mean_diff < 0 and r.statistic < 0 and r.p_value < 0.01


def test_diebold_mariano_hac_widens_with_autocorrelated_differentials():
    """Overlapping windows make losses autocorrelated; ignoring that inflates significance."""
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 400)
    d = np.convolve(e, np.ones(20) / 20, mode="same") + 0.05  # strongly autocorrelated, small mean
    naive = abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    assert abs(diebold_mariano(d, np.zeros_like(d), horizon=30).statistic) < naive


def test_wilcoxon_paired_across_stations():
    rng = np.random.default_rng(3)
    b = rng.gamma(2, 0.5, 20)
    assert wilcoxon_paired(b - 0.3, b).p_value < 0.01
    assert wilcoxon_paired(b, b).p_value == 1.0


def test_block_indices_preserve_length_and_contiguity():
    rng = np.random.default_rng(4)
    idx = block_indices(50, block=8, rng=rng)
    assert len(idx) == 50 and idx.max() < 50
    starts = idx[::8]
    assert all(np.array_equal(idx[i:i + 8], np.arange(s, s + 8)) for i, s in zip(range(0, 48, 8), starts[:6]))
    assert len(block_indices(5, block=99, rng=rng)) == 5  # block longer than the series


def test_bootstrap_ci_covers_the_truth_and_widens_with_block_length():
    x = np.random.default_rng(5).normal(1.0, 1.0, 400)
    point, lo, hi = bootstrap_ci(x, block=1, n_boot=500, seed=0)
    assert lo < 1.0 < hi and np.isclose(point, x.mean())
    wide = bootstrap_ci(x, block=40, n_boot=500, seed=0)
    assert (wide[2] - wide[1]) > (hi - lo)


def test_bootstrap_ci_multi_supports_difference_in_differences():
    rng = np.random.default_rng(6)
    samples = {"bd": rng.normal(0.5, 0.1, 200), "temperate": rng.normal(0.2, 0.1, 200)}
    stat = lambda s: s["temperate"].mean() - s["bd"].mean()
    point, lo, hi = bootstrap_ci_multi(samples, stat, block=4, n_boot=500, seed=1)
    assert np.isclose(point, -0.3, atol=0.05) and lo < point < hi and hi < 0  # a real, significant gap


def test_friedman_nemenyi_ranks_and_critical_difference():
    tasks = [f"t{i}" for i in range(16)]
    rng = np.random.default_rng(7)
    losses = pd.DataFrame({"good": rng.normal(0.5, 0.05, 16), "mid": rng.normal(0.7, 0.05, 16),
                           "bad": rng.normal(0.9, 0.05, 16)}, index=tasks)
    out = friedman_nemenyi(losses)
    assert list(out["avg_ranks"].index) == ["good", "mid", "bad"]
    assert out["p_value"] < 0.01 and out["n_tasks"] == 16 and out["n_models"] == 3
    expected = stats.studentized_range.ppf(0.95, 3, np.inf) / np.sqrt(2) * np.sqrt(3 * 4 / (6 * 16))
    assert np.isclose(out["cd"], expected)
    assert (out["avg_ranks"]["bad"] - out["avg_ranks"]["good"]) > out["cd"]  # separated beyond the CD


def test_holm_adjustment_is_monotone_and_conservative():
    adj = holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert adj["a"] == pytest.approx(0.003) and adj["b"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.04)
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_align_uses_the_common_index_only():
    a = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
    b = pd.Series([1.0, 5.0], index=["y", "w"])
    va, vb = align(a, b)
    assert va.tolist() == [2.0] and vb.tolist() == [1.0]
