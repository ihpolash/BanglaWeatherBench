# Week 5 — Significance, figures and the leakage audit

Date: 2026-09-16. Status: **complete.** Significance tests, result figures, the pretraining-overlap audit, the like-for-like gap-matched control and the 2,048-day context test are all done.

Week 4 gave point estimates. Week 5 asks which of them survive a test that respects how the benchmark is built.

## Protocol — what counts as a sample
The scores are a panel: every (track, variable, lead) has ~414 forecast origins × 10–35 stations, and neighbouring origins overlap (7-day stride, horizons to 30 days). Treating each window as independent would badly overstate confidence, so (`src/bwb/eval/significance.py`, 11 unit tests):

- **The forecast origin is the unit of resampling.** Station scores are averaged within an origin, so cross-station correlation is absorbed rather than counted as extra samples.
- **Diebold–Mariano** on the paired loss differential, with a Newey–West HAC variance at lag = horizon − 1 and the Harvey–Leybourne–Newbold small-sample correction, against a t distribution. A test on the same data without HAC gives a visibly larger statistic; that inflation is exactly what the correction removes.
- **Moving-block bootstrap** over origins (block = 8 origins ≈ 8 weeks, 2,000 resamples). Paired series — a model, its climatology reference, and the control baseline — are resampled with the *same* blocks; separate tracks are resampled independently. The difference-in-differences interval therefore carries the control's uncertainty too, rather than treating the control as a fixed number.
- **Wilcoxon signed-rank** across stations as a secondary check.
- **Friedman + Nemenyi** across the 16 daily tasks for the ranking, with the critical difference computed from the studentized range.
- **Holm–Bonferroni** across the 48 task-and-lead comparisons, because 48 tests invite false positives.

Outputs: `reports/significance_{pairwise,skill_ci,equity,obs_vs_reanalysis,spells,ranks}.csv` and `reports/significance.log`, from `scripts/run_significance.py`.

## S1. Foundation models vs the best baseline: most differences do not survive correction
Best foundation model vs best baseline per task and lead, Holm-adjusted DM over all 48 comparisons:

| Outcome | Count of 48 |
|---|---|
| Foundation model significantly better | **14** |
| Baseline significantly better | **5** |
| Not distinguishable | **29** |

- **All 5 baseline wins are climatology on rainfall at leads 7 and 30** (BMD, GHCN-temperate, NASA POWER temperate). On the 35-station BMD record and on CHIRPS dekadal rainfall, nothing beats climatology beyond a few days. The 10-station GHCN-Bangladesh track is the exception worth stating: the best foundation model holds a positive point estimate at every lead there (Toto-2.0 +0.095 at day 1, Chronos-2 +0.019 at day 30), but none of those margins survives correction (Holm p = 0.88, 1.00, 1.00), so they are not separable from climatology either.
- **The 14 foundation-model wins concentrate at lead 1 and on reanalysis tracks**, where series are smooth: NASA POWER radiation, precipitation, humidity and temperature at lead 1, plus temperate Tavg and rainfall at lead 1.
- **On station observations at leads 7 and 30 the margins are real but small and rarely significant.** Chronos-2 beats PatchTST on BMD humidity at lead 30 by 0.0033 CRPS, which the test cannot separate from noise.
- **Wilcoxon over stations is far more permissive** (many p < 0.01 where DM is not significant), because it pairs station means and ignores temporal overlap. Both are reported; DM with HAC is the one to quote.
- **Skill intervals against climatology:** 272 of 432 model-task-lead CRPSS intervals exclude zero.

## S2. The equity test, with a like-for-like control
Difference-in-differences against the best trained baseline per track; positive = the model loses more skill in the tropics than the control does. Three variants, each with 95% moving-block intervals:

- **as observed** — neither side masked;
- **model gap-matched** — the model's temperate contexts carry Bangladesh-like missingness, the control's do not;
- **both gap-matched** — the control was re-run on the same masked contexts (all 48 rows now have this).

| Variable | Lead | Control | Significant of 8 | Median DiD | Model-masked | Both-masked |
|---|---|---|---|---|---|---|
| Tavg | 1 | AutoARIMA | 1 | +0.000 | 3 sig, −0.012 | **0 sig, −0.001** |
| Tavg | 7 | PatchTST | 3 | +0.036 | 1 sig, −0.007 | **1 sig, −0.007** |
| **Tavg** | **30** | **AutoARIMA** | **7** | **+0.112** | 3 sig, +0.022 | **4 sig, +0.043** |
| Rainfall | 1 | NHITS | 2 | −0.028 | 4 sig, −0.038 | **4 sig, −0.033** |
| Rainfall | 7 | NHITS | 2 | +0.022 | 0 sig, +0.015 | **1 sig, +0.022** |
| Rainfall | 30 | NHITS | 1 | −0.011 | 2 sig, −0.017 | **0 sig, −0.010** |

Per model at the one lead where the effect is real (Tavg, 30 days), both sides masked:

| Model | DiD as observed | Both gap-matched (95% CI) | Still significant |
|---|---|---|---|
| TTM r2 | +0.197 | **+0.187** (0.089 to 0.300) | yes |
| Chronos-Bolt | +0.139 | **+0.138** (0.089 to 0.189) | yes |
| Chronos-2 | +0.052 | **+0.060** (0.022 to 0.097) | yes |
| TiRex | +0.080 | **+0.052** (0.005 to 0.097) | yes |
| Sundial | +0.049 | +0.035 (−0.022 to 0.089) | no |
| TimesFM 2.5 | +0.084 | +0.034 (−0.022 to 0.103) | no |
| Moirai-2 | +0.148 | −0.005 (−0.090 to 0.089) | no |
| Toto-2.0 | +0.140 | −0.008 (−0.101 to 0.088) | no |

**What this settles.**
- **The tropical temperature penalty at long leads is real for half the roster.** With the control masked exactly as the model is, 4 of 8 models keep a significant penalty at lead 30: TTM r2, Chronos-Bolt, Chronos-2 and TiRex.
- **Context missingness explains most of it, but not all.** The median penalty falls from +0.112 to +0.043 once both sides face Bangladesh-like gaps — roughly 60%, not the ~80% the model-only masking suggested.
- **Masking only the model understated the effect**, because the control also loses skill on gappy contexts. That is why the honest comparison is the one reported here; it raises the median penalty from +0.022 to +0.043 and moves TiRex back into significance (3 significant models become 4).
- **Rainfall shows no systematic tropical penalty.** At lead 1 four models are significant with a *negative* sign (Chronos-2 −0.028, Toto-2.0 −0.038, TiRex −0.052, Sundial −0.087): foundation models do relatively *better* in Bangladesh than at temperate stations. At lead 30 nothing is significant. The one exception to state rather than bury is Toto-2.0 at lead 7, +0.028 (CI 0.005 to 0.053) — a single model at a single lead, against four significant results in the opposite direction.
- **At lead 1 the penalty disappears entirely** (0 of 8 significant), so this is a long-horizon effect, not a general tropical handicap.

**Wording this supports:** *for temperature at long horizons, half the zero-shot TSFMs lose significantly more skill on Bangladesh stations than trained baselines do, and roughly 60% of that difference is attributable to sparser station context rather than to the tropics themselves.* It does not support a blanket "foundation models fail in the Global South" claim. For rainfall the honest statement is that there is no systematic penalty — the significant results at lead 1 run the other way, with one isolated exception (Toto-2.0 at lead 7).

**The short-vs-gappy confound is now resolved** — see S8: with the control masked the same way, doubling the context does not reduce the penalty.

## S3. Reanalysis vs observations: the most robust result in the benchmark
78 of 108 model-variable-lead comparisons are significant, and the pattern is consistent across model families:

- **Humidity:** significant at every lead for all 9 models (mean +0.29 to +0.35).
- **Rainfall:** significant for all 9 at leads 1 and 30, 7 of 9 at lead 7.
- **Temperature:** all 9 at lead 1, 6 of 9 at lead 7, only 2 of 9 at lead 30.
- **Sunshine vs radiation:** weakest (5, 2, 2 of 9) — and the two quantities are not physically identical, so this row is reported but not leaned on.

A model developed and validated on NASA POWER would therefore look substantially better than the same model scored at the stations it claims to represent. This is the TSFM counterpart of MAUSAM's critique for AI weather-prediction models (JAMES 2026, doi 10.1029/2025MS005568).

## S4. Monsoon regimes: the reversal is real
61 of 72 spell contrasts are significant, with intervals blocked by spell *event* rather than by station-day, so the effective sample is the number of spells, not the ~400 station-days they contain.

- **Break spells:** every model is significantly better than climatology, and the foundation models lead — Chronos-Bolt +0.440 (CI 0.333 to 0.510) on temperature at lead 30.
- **Active spells:** every model is significantly *worse* than climatology at leads 7 and 30, and the foundation models lose most — Chronos-Bolt −0.989 (CI −1.254 to −0.749), against PatchTST's −0.191.
- Humidity repeats the pattern; rainfall shows it weakly.
- **Reading:** skill is regime-dependent in a way an aggregate leaderboard hides completely. A model that looks strong on monsoon-season averages is at its worst exactly when the monsoon is active.

## S5. Ranking: the top group is one statistical cluster
Friedman over the 16 daily tasks × 18 models, with the Nemenyi critical difference (CD = 6.58 at every lead):

| Lead | χ² | p | Top of the ranking |
|---|---|---|---|
| 1 | 220.1 | 2.6 × 10⁻³⁷ | TiRex 1.62, Toto-2.0 3.75, Chronos-2 4.19 |
| 7 | 215.5 | 2.1 × 10⁻³⁶ | Chronos-2 2.12, TiRex 3.44, TimesFM 4.44 |
| 30 | 209.8 | 3.0 × 10⁻³⁵ | Chronos-2 2.00, TiRex 3.62, TimesFM 3.75 |

The models differ overall (p ≈ 10⁻³⁵), but the six leading foundation models plus PatchTST sit inside one critical difference of the best at leads 7 and 30: **the ordering within the top group is not statistically meaningful**, even though the group as a whole clearly separates from the weaker baselines.

## S6. Pretraining-overlap audit: no target leakage, and the temperate side is clean
Checked against primary sources (technical reports and dataset tables, not model cards):

| Model | Weather/climate data in its pretraining corpus | Station observations? |
|---|---|---|
| Chronos-2, Chronos-Bolt | **USHCN** (US daily stations), **WeatherBench** (ERA5-derived) | Yes — US only |
| TiRex | Chronos corpus + GIFT-Eval Pretrain (adds Oikolab, Monash Weather, Jena Weather) | Yes — US only, via Chronos |
| Moirai-2 | LOTSA: ERA5, CMIP6, **SubseasonalClimateUSA** (precip/temp) | US only |
| Sundial | TimeBench, **>60% ERA5** | No |
| TimesFM 2.5 | Google Trends, Wiki pageviews, synthetic, M4/electricity/traffic, Jena weather | No |
| Toto-2.0 | ~75% Datadog observability telemetry + synthetic | No |
| TTM r2 | Monash subset (incl. Australian Weather) | No |

- **No corpus names GHCN-Daily, BMD, or any Bangladesh source.** The tropical targets are leakage-free by construction, which is the argument GIFT-Eval concedes it cannot make for itself.
- **The temperate reference track is also clean:** all 32 stations are European (France 5, Serbia 5, Czechia 5, Bulgaria 4, Romania 3, Ireland 2, and one each in the Netherlands, Germany, Greece, Slovakia, Italy, Gibraltar, Denmark, Hungary) — **zero US stations**. The only station corpora in play (USHCN, SubseasonalClimateUSA) are US-only, so they cannot contain our temperate targets either.
- **Residual exposure is symmetric:** ERA5/CMIP6 gridded fields cover Europe and Bangladesh alike, so they cannot bias a tropical-vs-temperate contrast in one direction.
- **One nuance worth stating:** Chronos-2 saw US daily station data and yet shows the *smallest* tropical penalty, so a simple "memorised temperate stations" explanation does not fit the evidence.

## S7. Sundial's under-dispersion is the model, not our adapter
Week 4 flagged that Sundial's 80% intervals cover 32% of outcomes while its point forecasts are competitive. Three explanations were separable on one GPU (notebook `ipolas/bwb-sundial-check`, 512 BMD temperature windows, context 1,024):

| Samples | Raw output shape | Mean 80% width (°C) | 80% coverage | Trajectory spread (°C) | RMSE of sample mean (°C) |
|---|---|---|---|---|---|
| 20 | (512, 20, 30) | 1.44 | 0.350 | 0.61 | 1.52 |
| 100 | (512, 100, 30) | 1.56 | 0.386 | 0.63 | 1.51 |
| 500 | (512, 500, 30) | 1.59 | 0.391 | 0.64 | 1.51 |

- **Not too few samples:** widths gain 1.5% going from 100 to 500 trajectories, and coverage saturates near 0.39.
- **Not a reshape bug:** the raw output is `(batch, samples, horizon)`, which is what the adapter assumes; trajectories are genuinely diverse across the batch (spread across windows 3.16 °C, close to the target spread of 3.66 °C).
- **Genuine overconfidence:** trajectories differ by 0.64 °C while the sample mean is wrong by 1.51 °C, so the predictive distribution is ~2.4× too narrow for the model's own accuracy.

The calibration table in Week 4 stands as written, and Sundial's coverage can be reported as a property of the model.

## S8. Short context or gappy context? Separating the two at 2,048 days
Week-4 R7 found the tropical Tavg gap at lead 30 shrinks sharply when models get 2,048 days of context instead of 1,024, which suggested the penalty was mostly about how much history a Global-South station has. But at 1,024 days a Bangladesh context is *both* shorter in real observations and gappier, so the two explanations were confounded. Chronos-2, TiRex and TimesFM 2.5 were therefore re-run on temperate contexts at 2,048 days **with Bangladesh-like missingness imposed** (notebook `ipolas/bwb-fm-gm2048`), and scored against a control masked the same way.

Tavg, lead 30 — DiD against the best trained baseline:

| Model | 1,024 as observed | 2,048 as observed | 1,024 both masked | 2,048 both masked (95% CI) |
|---|---|---|---|---|
| Chronos-2 | +0.052 | +0.043 | +0.060 | **+0.059** (0.025 to 0.092), significant |
| TiRex | +0.080 | +0.058 | +0.052 | **+0.052** (0.012 to 0.092), significant |
| TimesFM 2.5 | +0.084 | +0.012 | +0.034 | +0.022 (−0.012 to 0.054), not significant |
| *median* | +0.080 | +0.043 | **+0.052** | **+0.052** |

- **The apparent context effect was mostly the missingness effect.** Without masking, the median penalty halves from +0.080 to +0.043 when context doubles — the R7 result. With the control masked like-for-like, the median is **+0.052 at both context lengths**: doubling the history changes nothing.
- **Chronos-2 and TiRex keep a real, context-independent penalty** at lead 30: about +0.05 CRPSS, significant at 1,024 and at 2,048 days alike.
- **TimesFM 2.5's penalty was genuinely context-related** and is gone by 2,048 days.
- **Rainfall stays null** at 2,048 days: nothing significant once the control is masked, at either lead.
- **Trained baselines keep their own fixed context** (730 days statistical, 365 days neural), so this ablation varies the foundation models' history only. That is the right comparison for the question asked — does *a foundation model's* context length explain its tropical deficit — but it means the control is not itself a 2,048-day model.

**Correction this forces to Week 4's R7:** the reading there — "the long-lead temperature deficit is substantially a context-length effect" — holds only for the unmasked comparison. Once the control is treated identically, the deficit for Chronos-2 and TiRex is not explained by context length at all.

## Figures (F6–F12, `scripts/make_result_figures.py`)
| Figure | Content |
|---|---|
| F6 | Critical-difference ranking per lead, with the not-separable band |
| F7 | Skill vs lead: station temperature, station rainfall, CHIRPS dekadal |
| F8 | Equity DiD with 95% intervals, as observed vs gap-matched |
| F9 | Context length: skill vs history, and the shrinking tropical gap |
| F10 | Monsoon phase heatmap and active/break spell intervals |
| F11 | Compute vs skill Pareto (marker area = parameters) |
| F12 | Reanalysis minus observation skill with intervals |

Each figure writes its table view to `reports/figures/data/`.

## Open
1. **Week-6 write-up and release** — the results are stable enough to write against: paper, Hugging Face dataset, Zenodo DOI, tagged GitHub release, arXiv submission (cs.AI primary, cs.LG + physics.ao-ph cross-lists), and the final novelty re-check three days before submitting.

Resolved this week, all with verified runs rather than assumptions:
- the **like-for-like gap-matched control** — AutoETS/AutoTheta/AutoARIMA locally (85 min) and NHITS/DLinear/PatchTST on Kaggle (`ipolas/bwb-neural-gapmatched`), all 6 files verified before install, folded into S2;
- the **Sundial dispersion question** (S7) — the under-dispersion is the model, not the adapter;
- the **short-vs-gappy context confound** (S8) — separated at 2,048 days, and it corrects Week-4 R7's reading.

## Framing decision (the plan's Week-5 checkpoint)
The evidence supports a **mixed, mechanism-led result**, not a triumphal or a purely negative one:

1. Zero-shot foundation models are the best family overall, but on station observations at leads beyond a day most margins do not survive correction (S1), and the top six are one statistical cluster (S5).
2. **Nobody beats climatology on tropical rainfall**, on the BMD station record beyond a few days and on CHIRPS dekadal rainfall at any horizon. (On the small GHCN-Bangladesh track the point estimates are positive but never significant.) This is the firmest negative result and a boundary on the "forecast anywhere" claim — state it scoped, because the unscoped version is falsifiable from our own leaderboard.
3. **The equity finding is narrow and mechanistic:** a long-lead temperature penalty for half the roster, about 60% of it explained by sparser context, and for Chronos-2 and TiRex a residual that more history does not fix (S2, S8).
4. **Benchmarks built on reanalysis overstate skill** at the stations they claim to represent (S3), and **skill is regime-dependent** in a way aggregate leaderboards hide (S4).
5. The benchmark is **leakage-free on both sides** of the equity contrast (S6) and reproducible in ~4 GPU-hours on free hardware (Week 4 R6).

Headline for the paper: *foundation models inherit a tropical blind spot that is mostly a data-availability problem, not a climate one — except at long horizons, where part of it is the models.*
