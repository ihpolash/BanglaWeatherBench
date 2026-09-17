"""Week-5 significance tests for forecast comparisons on this benchmark's rolling-origin design.

The scores are a panel: every (track, variable, lead) has ~414 forecast origins x 10-35 stations, and neighbouring
origins overlap (stride 7 days, horizon up to 30), so plain per-row tests would badly overstate the sample size.
Everything here therefore treats the **origin** as the unit of resampling and the station dimension as within-origin
correlation to be averaged over:

- `per_origin_loss`   mean loss across stations for each origin -> one time series per model.
- `per_station_loss`  mean loss across origins for each station -> paired samples for a station-level Wilcoxon.
- `diebold_mariano`   DM test on the paired loss differential with a Newey-West HAC variance (lag = horizon - 1,
                      the standard choice for h-step forecasts) and the Harvey-Leybourne-Newbold small-sample
                      correction, referred to a t distribution with n - 1 degrees of freedom.
- `bootstrap_ci`      moving-block bootstrap over origins (blocks preserve the overlap-induced autocorrelation).
- `friedman_nemenyi`  Friedman test over tasks x models plus the Nemenyi critical difference, computed from the
                      studentized range (scikit-posthocs/autorank are not dependencies of this project).

Losses are always "lower is better" (CRPS, scaled CRPS, absolute error), so a negative mean difference means model A
beats model B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def per_origin_loss(scores: pd.DataFrame, windows: pd.DataFrame, lead: int, column: str = "scaled_crps") -> pd.Series:
    """Mean loss over stations at one lead, indexed by forecast origin (sorted)."""
    s = scores[scores.lead == lead][["window_id", column]].merge(windows[["window_id", "origin"]], on="window_id")
    return s.groupby("origin")[column].mean().sort_index()


def per_station_loss(scores: pd.DataFrame, windows: pd.DataFrame, lead: int, column: str = "scaled_crps") -> pd.Series:
    """Mean loss over origins at one lead, indexed by station."""
    s = scores[scores.lead == lead][["window_id", column]].merge(windows[["window_id", "series_id"]], on="window_id")
    return s.groupby("series_id")[column].mean().sort_index()


def align(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Values of two loss series on their common index (origins or stations)."""
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(dtype=float), b.loc[common].to_numpy(dtype=float)


@dataclass
class TestResult:
    n: int
    mean_diff: float
    statistic: float
    p_value: float

    def __repr__(self) -> str:  # compact for logs
        return f"n={self.n} diff={self.mean_diff:+.4f} stat={self.statistic:+.3f} p={self.p_value:.2e}"


def _hac_variance(d: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of the loss differential (Bartlett weights)."""
    n = len(d)
    dc = d - d.mean()
    gamma0 = float(dc @ dc) / n
    total = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        gamma_k = float(dc[k:] @ dc[:-k]) / n
        total += 2.0 * (1.0 - k / (lags + 1)) * gamma_k
    return total if total > 0 else gamma0


def diebold_mariano(loss_a: pd.Series | np.ndarray, loss_b: pd.Series | np.ndarray, horizon: int) -> TestResult:
    """Two-sided DM test with HLN correction. Negative statistic => model A has the lower loss."""
    if isinstance(loss_a, pd.Series) and isinstance(loss_b, pd.Series):
        a, b = align(loss_a, loss_b)
    else:
        a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    d = a - b
    n = len(d)
    if n < 3:
        return TestResult(n, float(d.mean()) if n else np.nan, np.nan, np.nan)
    var = _hac_variance(d, max(horizon - 1, 0))
    if var <= 0 or np.isclose(d.std(), 0):
        return TestResult(n, float(d.mean()), 0.0, 1.0)
    stat = d.mean() / np.sqrt(var / n)
    h = max(horizon, 1)
    correction = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))  # Harvey, Leybourne & Newbold (1997)
    stat *= correction
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return TestResult(n, float(d.mean()), float(stat), float(p))


def wilcoxon_paired(loss_a: pd.Series | np.ndarray, loss_b: pd.Series | np.ndarray) -> TestResult:
    """Wilcoxon signed-rank test on paired per-station (or per-origin) mean losses."""
    if isinstance(loss_a, pd.Series) and isinstance(loss_b, pd.Series):
        a, b = align(loss_a, loss_b)
    else:
        a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    d = a - b
    if len(d) < 3 or np.allclose(d, 0):
        return TestResult(len(d), float(d.mean()) if len(d) else np.nan, np.nan, 1.0)
    res = stats.wilcoxon(a, b)
    return TestResult(len(d), float(d.mean()), float(res.statistic), float(res.pvalue))


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap resample of positions 0..n-1."""
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def bootstrap_ci(values, stat=np.mean, block: int = 8, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Moving-block bootstrap percentile CI for a statistic of one per-origin series."""
    x = np.asarray(values.to_numpy() if isinstance(values, pd.Series) else values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([stat(x[block_indices(len(x), block, rng)]) for _ in range(n_boot)])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def bootstrap_ci_multi(samples: dict[str, np.ndarray], stat, block: int = 8, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for a statistic combining several independent per-origin series.

    Used for contrasts that mix tracks (e.g. the equity DiD, which combines Bangladesh and temperate origins, or
    observations vs reanalysis): each series is resampled with its own blocks, then `stat(dict)` is evaluated.
    """
    arrays = {k: np.asarray(v.to_numpy() if isinstance(v, pd.Series) else v, dtype=float) for k, v in samples.items()}
    rng = np.random.default_rng(seed)
    draws = np.array([stat({k: v[block_indices(len(v), block, rng)] for k, v in arrays.items()}) for _ in range(n_boot)])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(stat(arrays)), float(lo), float(hi)


def friedman_nemenyi(losses: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman test across tasks (rows) x models (columns) plus the Nemenyi critical difference.

    `losses` holds one loss value per task and model (lower is better). Returns average ranks (1 = best), the
    Friedman chi-square and p-value, and the critical difference: two models differ at `alpha` if their average
    ranks differ by more than CD.
    """
    x = losses.dropna(axis=0, how="any")
    n_tasks, k = x.shape
    ranks = x.rank(axis=1)  # 1 = lowest loss = best
    avg = ranks.mean(axis=0)
    chi2, p = stats.friedmanchisquare(*[x[c].to_numpy() for c in x.columns])
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * n_tasks)))
    return {"n_tasks": int(n_tasks), "n_models": int(k), "chi2": float(chi2), "p_value": float(p),
            "avg_ranks": avg.sort_values(), "cd": cd, "alpha": alpha}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of comparisons (keeps the keys)."""
    items = sorted(p_values.items(), key=lambda kv: (np.inf if np.isnan(kv[1]) else kv[1]))
    m, out, running = len(items), {}, 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p) if not np.isnan(p) else np.nan
        running = max(running, adj) if not np.isnan(adj) else running  # enforce monotonicity
        out[key] = running if not np.isnan(adj) else np.nan
    return out
