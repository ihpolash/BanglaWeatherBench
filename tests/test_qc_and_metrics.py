import numpy as np
import pandas as pd

from bwb.data.qc import flag_constant_runs, flag_linear_runs, imputation_mask
from bwb.eval.metrics import crps_from_quantiles, interval_coverage, mase, seasonal_naive_scale


def _noisy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(25 + rng.normal(0, 2, n), index=pd.date_range("2016-01-01", periods=n, freq="D"))


def test_constant_run_detected():
    s = _noisy()
    s.iloc[20:28] = 30.0  # planted 8-day constant fill
    flags = flag_constant_runs(s, min_len=5)
    assert flags.iloc[20:28].all()
    assert flags.sum() == 8


def test_zero_rain_runs_not_flagged():
    s = pd.Series([0.0] * 10 + [3.2, 5.1] + [0.0] * 10)
    assert not flag_constant_runs(s, min_len=5, ignore_value=0.0).any()


def test_linear_run_detected_but_noise_not():
    s = _noisy()
    s.iloc[30:37] = np.linspace(20, 26, 7)  # planted linear interpolation
    flags = flag_linear_runs(s, min_len=5)
    assert flags.iloc[30:37].all()
    assert not _noisy(seed=1).pipe(flag_linear_runs).any()


def test_imputation_mask_combines_flags():
    s = _noisy()
    s.iloc[5:12] = 27.0
    mask = imputation_mask(s, "temperature")
    assert mask["flag_any"].iloc[5:12].all()


def test_mase_is_one_for_seasonal_naive_on_its_reference():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 200)
    season = 7
    # Seasonal-naive forecasts of the training series itself reproduce the scale exactly.
    assert np.isclose(mase(y[season:], y[:-season], y, season), 1.0)
    assert seasonal_naive_scale(y, season) > 0


def test_crps_and_coverage_sanity():
    y = np.array([1.0, 2.0, 3.0])
    qs = np.array([0.1, 0.5, 0.9])
    perfect = np.tile(y, (3, 1))
    assert crps_from_quantiles(y, perfect, qs) == 0.0
    assert interval_coverage(y, y - 1, y + 1) == 1.0


def test_offgrid_precision_flags_interpolated_humidity():
    from bwb.data.qc import flag_offgrid_precision

    s = pd.Series([78.0, 79.0, 78.5, np.nan, 80.0])
    assert flag_offgrid_precision(s, decimals=0).tolist() == [False, False, True, False, False]
    mask = imputation_mask(s, "humidity", decimals=0)
    assert "flag_offgrid" in mask and bool(mask["flag_any"].iloc[2])
