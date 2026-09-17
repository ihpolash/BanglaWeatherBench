import numpy as np
import pandas as pd

from bwb.eval.harness import score, train_scales
from bwb.eval.windows import build_windows
from bwb.models.baselines import doy_noleap
from bwb.models.lgbm_baseline import GlobalLGBM
from bwb.models.stat_baselines import run_stat_models

TRAIN_END, VAL, TEST, H = "2008-12-31", ("2009-01-01", "2010-12-31"), ("2012-01-01", "2012-12-31"), 30


def _ar_series(seed, n_years=13, phi=0.9):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=365 * n_years, freq="D")
    a = np.zeros(len(idx))
    for t in range(1, len(idx)):
        a[t] = phi * a[t - 1] + rng.normal(0, 1)
    return pd.Series(25 + 5 * np.sin(2 * np.pi * doy_noleap(idx) / 365) + a, index=idx)


def test_stat_models_shapes_sorted_quantiles_and_clipping():
    series = {("T", "A"): _ar_series(0), ("T", "B"): _ar_series(1) - 30}
    w = build_windows(series, "syn", *TEST, horizon=H, stride=60)
    out = run_stat_models(series, w, TRAIN_END, H, ("AutoETS", "AutoTheta"), context_length=365, non_negative=True, n_jobs=1)
    for name, p in out.items():
        assert len(p) == len(w) * H
        q = p[[c for c in p.columns if c.startswith("q")]].to_numpy()
        assert (np.diff(q, axis=1) >= -1e-9).all()
        assert (q >= 0).all() and (p["mean"] >= 0).all()


def test_stat_models_beat_climatology_at_short_lead_on_persistent_anomalies():
    series = {("T", "A"): _ar_series(2, phi=0.97)}
    w = build_windows(series, "syn", *TEST, horizon=H, stride=7)
    ets = run_stat_models(series, w, TRAIN_END, H, ("AutoETS",), context_length=365, n_jobs=1)["AutoETS"]
    sc = score(ets, w, series, train_scales(series, TRAIN_END), H)
    from bwb.eval.harness import run_model
    from bwb.models.baselines import Climatology
    clim = score(run_model(lambda v: Climatology(), series, w, TRAIN_END, H), w, series, train_scales(series, TRAIN_END), H)
    assert sc[sc.lead == 1].ase.mean() < 0.7 * clim[clim.lead == 1].ase.mean()


def test_lgbm_learns_persistence_and_ignores_future_values():
    series = {("T", k): _ar_series(i, phi=0.95) for i, k in enumerate("ABC")}
    w = build_windows(series, "syn", *TEST, horizon=H, stride=14)
    m = GlobalLGBM(horizon=H, train_stride=5, num_boost_round=200).fit(series, TRAIN_END, *VAL)
    p1 = m.predict_windows(series, w)
    tampered = {k: s.copy() for k, s in series.items()}
    for s in tampered.values():
        s.loc[TEST[0]:] += 50.0  # change every value after the first origin's context
    first = w.origin.min()
    w1 = w[w.origin == first]
    a = m.predict_windows(series, w1)
    b = m.predict_windows({k: pd.concat([series[k].loc[:first], tampered[k].loc[first + pd.Timedelta(days=1):]]) for k in series}, w1)
    assert np.allclose(a["mean"], b["mean"])
    sc = score(p1, w, series, train_scales(series, TRAIN_END), H)
    assert sc[sc.lead == 1].ase.mean() < sc[sc.lead == 30].ase.mean()
