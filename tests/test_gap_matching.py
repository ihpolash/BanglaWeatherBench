import numpy as np
import pandas as pd

from bwb.eval.gap_matching import apply_mask, assign_donors, donor_masks


def _series(frac_missing, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", "2017-12-31", freq="D")
    s = pd.Series(rng.normal(20, 3, len(idx)), index=idx)
    s[rng.random(len(idx)) < frac_missing] = np.nan
    return s


def test_donor_masks_reflect_donor_missingness_and_length():
    series = {("Tavg", "BD1"): _series(0.2, 0), ("Tavg", "BD2"): _series(0.2, 1)}
    w = pd.DataFrame({"var": "Tavg", "series_id": ["BD1", "BD2"], "origin": pd.to_datetime(["2016-06-30", "2017-03-31"])})
    m = donor_masks(series, w, "Tavg", length=365)
    assert m.shape == (2, 365) and m.dtype == bool
    assert 0.12 < m.mean() < 0.28


def test_short_history_counts_as_missing():
    s = _series(0.0, 2).loc["2016-01-01":]
    w = pd.DataFrame({"var": "Tavg", "series_id": ["X"], "origin": pd.to_datetime(["2016-03-31"])})
    m = donor_masks({("Tavg", "X"): s}, w, "Tavg", length=365)
    assert m[0, : 365 - 91].all() and not m[0, -91:].any()


def test_apply_mask_right_aligned_and_does_not_mutate_input():
    ctx = np.arange(10, dtype=np.float32)
    mask = np.array([True, False, True])
    out = apply_mask(ctx, mask)
    assert np.isnan(out[7]) and out[8] == 8 and np.isnan(out[9])
    assert not np.isnan(ctx).any() and np.isnan(out).sum() == 2
    short = apply_mask(np.array([1.0, 2.0], dtype=np.float32), np.array([True, True, False]))
    assert np.isnan(short[0]) and short[1] == 2.0


def test_donor_assignment_is_deterministic_and_in_range():
    ids = np.array([5, 0, 17, 5])
    a, b = assign_donors(ids, n_donors=4, seed=0), assign_donors(ids, n_donors=4, seed=0)
    assert (a == b).all() and a[0] == a[3] and ((0 <= a) & (a < 4)).all()


def test_masks_of_different_lengths_are_tails_of_each_other():
    """The control must see the same missingness the foundation models saw: the 365- and 730-day masks used by the
    trained baselines are the tails of the 1,024-day masks used by the FMs, for the same donor window."""
    import numpy as np
    import pandas as pd

    from bwb.eval.gap_matching import donor_masks

    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", "2018-12-31", freq="D")
    s = pd.Series(rng.normal(size=len(idx)), index=idx)
    s.iloc[rng.choice(len(s), 900, replace=False)] = np.nan
    w = pd.DataFrame({"window_id": [0, 1], "var": "Tavg", "series_id": "A",
                      "origin": pd.to_datetime(["2016-06-30", "2017-06-30"])})
    series = {("Tavg", "A"): s}
    long, short = donor_masks(series, w, "Tavg", length=1024), donor_masks(series, w, "Tavg", length=365)
    assert short.shape == (2, 365) and long.shape == (2, 1024)
    assert np.array_equal(short, long[:, -365:])
