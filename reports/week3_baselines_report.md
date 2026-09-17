# Week 3 — Evaluation harness and baselines

Date: 2026-09-15. Status: **complete**. 9 baselines × 16 daily tasks, plus the CHIRPS dekadal task. Numbers are from `reports/leaderboard_daily.csv`, `reports/leaderboard_bmd_phase.csv` and `reports/leaderboard_dekadal.csv`. These are preliminary: significance testing is Week 5, and foundation models are Week 4.

## What was built
| Component | File | Notes |
|---|---|---|
| Unified series store | `src/bwb/data/store.py` | 6 daily tracks → `{(variable, series_id): daily series}`; flagged BMD values masked |
| Rolling-origin windows | `src/bwb/eval/windows.py` | 30-day horizon, 7-day stride, test 2016–2023; `validate_windows` rejects targets outside the split |
| Harness + metrics | `src/bwb/eval/harness.py` | Forecast cache per model/track/variable; MASE (median point, training-split seasonal-naive scale, m = 365), quantile CRPS, RMSE (mean), 80% coverage |
| Naive baselines | `src/bwb/models/baselines.py` | Persistence, seasonal-naive (lag 365, climatology fallback), climatology (training day-of-year ±15 days) |
| Statistical baselines | `src/bwb/models/stat_baselines.py` | AutoETS, AutoTheta on climatological anomalies (730-day context). AutoARIMA implemented, not run on the full set (cost) |
| LightGBM | `src/bwb/models/lgbm_baseline.py` | Global per track/variable; direct multi-horizon; validation early stopping and residual quantiles |
| Deep learning | `scripts/run_neural_baselines.py` | NHITS, DLinear, PatchTST; multi-quantile loss; 365-day input; trained on the training split; run on Kaggle GPUs |
| CHIRPS dekadal task | `src/bwb/eval/dekadal.py` | Step-based windows (6-dekad horizon), dekad-of-year climatology, seasonal lag 36 |
| Leaderboards | `scripts/build_leaderboard.py` | Common-window rule; per-window scores saved for Week-5 significance tests; skill vs same-track climatology |

## Leakage protection (tested; 45 tests pass)
- Windows whose target reaches outside the test split raise an error.
- A spy model confirms the harness shows models no data after the training cutoff when fitting, and none after the origin when forecasting.
- An oracle that reads the target scores exactly 0, so any leakage would show up as implausibly perfect scores.
- Climatology, LightGBM and the CHIRPS baselines give identical forecasts when test-period values are tampered with.
- A point-forecast rule test prevents mixing median and mean point forecasts (see below).

## Design decisions
1. **Score complete targets, not complete contexts.** A ≥90%-complete-context rule leaves **0** GHCN-Bangladesh rainfall windows. The rule is instead:
   - the 30-day target must be fully observed and unflagged;
   - the series needs ≥ 365 days of history;
   - models receive contexts with their real gaps.

   Each window stores `ctx_obs_frac_365` (median: BMD 0.96–1.00; GHCN-Bangladesh 0.82–0.86; temperate 0.99–1.00).
2. **Point-forecast rule.** Absolute error and MASE use each model's predictive **median**; RMSE uses the mean; **CRPS is the headline metric.**

   An earlier comparison scored means for climatology and the statistical models, but medians for the quantile-loss networks. On zero-inflated rainfall that is not neutral. Climatology's MASE at lead 30 is 0.82 on its mean but 0.58 on its median, so NHITS falsely *appeared* 25% better than climatology. CRPS, which does not depend on the point forecast, showed climatology ahead.
3. **Cross-track comparisons use skill relative to same-track climatology** (CRPSS). Raw MASE is not comparable across tracks: NASA POWER and GHCN-Bangladesh rainfall have MASE > 1 even for climatology, because their seasonal-naive scales behave differently.

**Window counts:** 170,571 daily windows. Minimum per series is ≥117 everywhere except GHCN-Bangladesh rainfall (minimum 91, slightly below the ≥100 target). NASA POWER tracks have the full 414 origins per series. All 9 models were scored on identical windows in every task.

## Results

### Which model is best (lowest scaled CRPS per task, 16 daily tasks)
| Lead | PatchTST | NHITS | Climatology | LightGBM |
|---|---|---|---|---|
| 1 day | 9 | 7 | 0 | 0 |
| 7 days | 10 | 3 | 3 | 0 |
| 30 days | 6 | 2 | 4 | 4 |

At 30 days, **climatology is the best model for every station rainfall task** (BMD, GHCN-Bangladesh, GHCN-temperate) and for NASA POWER temperate rainfall.

### Skill vs climatology (CRPSS; 0 = climatology, higher is better): best model per task
| Task | Day 1 | Day 7 | Day 30 |
|---|---|---|---|
| BMD Rainfall | +0.017 (NHITS) | −0.022 | −0.030 |
| BMD Temperature | +0.342 (PatchTST) | +0.065 (PatchTST) | +0.018 (PatchTST) |
| BMD Humidity | +0.283 (PatchTST) | +0.054 (PatchTST) | +0.047 (PatchTST) |
| BMD Sunshine | +0.162 (NHITS) | +0.030 (PatchTST) | +0.027 (LightGBM) |
| GHCN-Bangladesh Rainfall | +0.060 (NHITS) | +0.009 (NHITS) | −0.022 |
| GHCN-Bangladesh Tavg | +0.287 (PatchTST) | +0.072 (PatchTST) | +0.031 (LightGBM) |
| GHCN-temperate Rainfall | +0.016 (NHITS) | −0.026 | −0.037 |
| GHCN-temperate Tavg | +0.418 (NHITS) | +0.061 (PatchTST) | +0.057 (PatchTST) |

Negative values mean no model beat climatology (the value shown is the least-bad model).

Model families:
- **AutoETS and AutoTheta** match persistence at lead 1 for smooth variables (BMD temperature ≈ +0.31) but degrade badly at long leads: on non-rainfall tasks, CRPSS at 30 days runs from −0.27 (sunshine) to −0.78 (temperate Tavg), as anomaly extrapolation drifts.
- **Persistence and seasonal-naive** are never competitive beyond lead 1.

### Preliminary contrast 1 — observations vs reanalysis (same 35 BMD stations; best CRPSS)
| Variable | Day 1: BMD | Day 1: NASA POWER | Day 7: BMD | Day 7: NASA POWER | Day 30: BMD | Day 30: NASA POWER |
|---|---|---|---|---|---|---|
| Rainfall | +0.017 | **+0.270** | −0.022 | **+0.070** | −0.030 | **+0.053** |
| Temperature | +0.342 | **+0.513** | +0.065 | **+0.104** | +0.018 | +0.021 |
| Humidity | +0.283 | **+0.612** | +0.054 | **+0.378** | +0.047 | **+0.298** |
| Sunshine / radiation | +0.162 | +0.176 | +0.030 | +0.041 | +0.027 | +0.021 |

**Reanalysis looks consistently more predictable than the observations it represents**, strongly for rainfall and humidity. A model developed and scored on NASA POWER would overstate real forecast skill at Bangladesh stations. This supports contribution #5, pending Week-5 significance tests.

### Preliminary contrast 2 — tropical vs temperate (GHCN vs GHCN; best CRPSS vs same-track climatology)
| Variable | Day 1: Bangladesh | Day 1: temperate | Day 7: Bangladesh | Day 7: temperate | Day 30: Bangladesh | Day 30: temperate |
|---|---|---|---|---|---|---|
| Rainfall | +0.060 | +0.016 | +0.009 | −0.026 | −0.022 | −0.037 |
| Tavg | +0.287 | +0.418 | +0.072 | +0.061 | +0.031 | +0.057 |

**These trained baselines show no clear tropical disadvantage.** Bangladesh rainfall skill is marginally *higher*; temperature skill is lower at day 1 and similar afterwards. The GHCN-Bangladesh track is small (10 stations, ~1.2–1.5k windows), so differences of a few hundredths are within noise until tested.

**Why this matters for the paper:** these baselines are trained *per track*, so they measure intrinsic predictability, not pretraining bias. They are the **control** for the equity test. A tropical skill gap in *zero-shot foundation models* that is absent in these per-track baselines would point to geographic imbalance in pretraining rather than a harder climate. That is the comparison Week 4 enables.

### Monsoon phase (BMD; CRPSS vs climatology by target-day phase)
| Phase | Rainfall day 1: NHITS | Rainfall day 1: PatchTST | Temperature day 1: PatchTST | Temperature day 7: PatchTST | Temperature day 7: LightGBM |
|---|---|---|---|---|---|
| Dry | −0.117 | −0.606 | +0.369 | +0.050 | +0.050 |
| Pre-monsoon | −0.060 | −0.162 | +0.315 | +0.037 | +0.040 |
| **Onset** | +0.030 | −0.002 | **+0.208** | **−0.026** | **−0.024** |
| Peak | +0.043 | +0.000 | +0.356 | +0.125 | +0.046 |
| Withdrawal | +0.005 | −0.061 | +0.362 | +0.010 | +0.042 |

- **Monsoon onset is where temperature skill disappears.** It is the lowest phase at day 1 for every model except seasonal-naive, which is negative in every phase: onset +0.09 to +0.21, against +0.25 to +0.37 in the dry season. It is negative for every model at day 7. This matches the regime-transition failure seen in other domains (arXiv 2606.18367) and gives the phase stratification its first concrete finding.
- The small positive rainfall skill is concentrated in **monsoon onset and peak** (NHITS +0.030 and +0.043), is negligible in withdrawal (+0.005), and is absent in the dry and pre-monsoon phases.
- In the dry season, every model loses to climatology. AutoETS and AutoTheta collapse there (CRPSS −5.6 and −6.9 at day 1) by forecasting drizzle on dry days.
- Onset has the fewest samples (~590 window-days at lead 1); significance comes in Week 5.

### Calibration (80% interval coverage at day 7; nominal 0.80)
- **PatchTST** is the best calibrated neural model: 0.77–0.87.
- **Climatology** is near nominal on temperature (0.77–0.81) but over-covers station rainfall (0.86–0.92).
- **AutoETS and AutoTheta** over-cover almost everywhere (0.84–0.93).
- **DLinear** intervals are far too narrow: 0.28–0.57 on every task.
- **NHITS** under-covers temperature: 0.48–0.74.
- **LightGBM** badly under-covers temperate rainfall: 0.37–0.39.

Point accuracy therefore cannot be read as trustworthy uncertainty; calibration is reported alongside skill.

### CHIRPS dekadal rainfall (18,112 windows, 64 districts; MASE, median point)
| Lead (dekads) | Climatology | Persistence | Seasonal-naive |
|---|---|---|---|
| 1 | **0.704** | 0.949 | 0.980 |
| 2 | **0.706** | 1.072 | 0.982 |
| 3 | **0.706** | 1.171 | 0.982 |
| 6 | **0.707** | 1.523 | 0.982 |

Climatology dominates at every lead, with 80% coverage of 0.75–0.76.

## Compute
- **Statistical and LightGBM:** ran locally with StatsForecast on 4 workers, taking about 3 h for all 16 tasks. LightGBM itself takes seconds per task. Its early stopping gives a rough predictability signal: 9 trees on BMD rainfall against 60 on NASA POWER rainfall, consistent with contrast 1.
- **Deep learning:** ran on **Kaggle GPUs**, using a private dataset (`ipolas/bwb-bundle`) and a private notebook (`ipolas/bwb-neural-baselines`):
  - 2× Tesla T4, PyTorch 2.10 + CUDA 12.8, neuralforecast 3.2.2;
  - tracks split one process per GPU;
  - **18 min for all 16 tasks**, against an estimated 4–5 h on the local M1.
- **Output check:** all 48 forecast files passed (row counts, window IDs, complete leads, no missing values, monotone quantiles, non-negativity).
- **Hardware cross-check:** BMD rainfall and temperature, run both locally (MPS) and on Kaggle (CUDA), agree closely. DLinear is identical to three decimals; NHITS and PatchTST differ by ≤ 0.02. All reported deep-learning results come from Kaggle; the local MPS outputs are kept only for reference, in `data/interim/local_mps_neural/`.
- **Operational lesson:** the 8 GB M1 cannot run heavy jobs concurrently. It hit 14.6–16.9 GB of swap twice, and no finished results were lost. Heavy jobs run sequentially, and GPU work goes to Kaggle.

## Mid-project novelty re-check
Completed 2026-09-15, before Week 4. Full detail in `NOVELTY_LOG.md`, section "R1 mid-project re-check".

**Method.** All Week-1 query families (A1–A8) were re-run for 2026, together with searches for the three claims that emerged this week:
- reanalysis vs observations;
- interval calibration;
- zero-shot rainfall forecasting.

That came to 17 searches. **11 new items** were verified from primary sources.

**Verdict: GO.** The core gap is still open. Explicit searches found no time-series foundation model evaluated on Bangladesh or South Asian station weather, no released Bangladesh forecasting benchmark, and no monsoon-phase-stratified TSFM evaluation.

**Two claims are narrowed and must be worded carefully:**
- **Climatic-regime stratification.** arXiv 2606.06348 (Jun 2026) calls itself "the first regime-stratified, sub-regional benchmark of an operational MLWP model over Brazil". It uses GraphCast, the IFS analysis, and upper-air variables only, with no precipitation. Our claim is therefore scoped to TSFMs × station observations including precipitation × monsoon phases.
- **Leakage-free zero-shot evaluation.** TIME (arXiv 2602.12147, ICML 2026) already provides this for 12 TSFMs, though without climate strata. We cite it rather than present leakage-freeness as unique.

**Other updates:**
- MAUSAM is now formally published in *JAMES* 2026 (doi 10.1029/2025MS005568). It is the precedent for AI weather-model errors being 15–45% larger against stations than against reanalysis. Our Week-3 reanalysis-vs-observation result is the TSFM/baseline analogue.
- New neighbours to cite: GraphCast on the Indian summer monsoon (arXiv 2607.11905); probabilistic AI monsoon-onset forecasts (arXiv 2603.07893); Chronos-2's zero-shot rainfall–runoff result (discharge target, NSE 0.68 vs LSTM 0.90).

## Carried into Week 4 (foundation models, zero-shot)
- Run Chronos-2, TimesFM 2.5, Toto-2.0, TTM r2, Chronos-Bolt, Moirai-2, TiRex and Sundial through the same harness on Kaggle, reusing the private bundle. Pass contexts with gaps (NaN) where a model supports it, otherwise interpolate.
- Headline comparisons:
  - foundation model vs per-track baseline skill, by track (the equity test);
  - observation vs reanalysis;
  - monsoon phase, especially onset.
- Also:
  - check whether the foundation models' intervals are better calibrated than DLinear's and NHITS';
  - gap-matched temperate ablation (impose Bangladesh-like context gaps);
  - AutoARIMA on a subset, if time allows.
