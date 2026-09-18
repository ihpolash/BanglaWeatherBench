# BanglaWeatherBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22825164.svg)](https://doi.org/10.5281/zenodo.22825164)

A benchmark of time-series foundation models on tropical-monsoon station observations from Bangladesh. Scores are stratified by monsoon phase, with two comparison tracks: a matched temperate reference set of weather stations, and NASA POWER reanalysis at the same locations.

Status: **Week 5 complete** (Week 6 = paper write-up and release):
- data pipeline, evaluation strata and EDA;
- a leakage-tested evaluation harness;
- 10 baselines (naive, statistical incl. AutoARIMA, LightGBM, NHITS/DLinear/PatchTST on Kaggle GPU) on 16 daily tasks, plus the CHIRPS dekadal task;
- 8 zero-shot foundation models (Chronos-2, Chronos-Bolt, TimesFM 2.5, Toto-2.0, TiRex, Moirai-2, TTM r2, Sundial) on the same 16 daily tasks, plus a gap-matched temperate ablation;
- a context-length ablation (96–4,096 days), CHIRPS dekadal zero-shot runs at 4 context lengths, and compute cost (parameters, peak GPU memory, throughput) for every model.

Week 5 adds significance testing (Diebold–Mariano with HAC, moving-block bootstrap over forecast origins, Friedman–Nemenyi, Holm correction), result figures F6–F12, a pretraining-overlap audit, a like-for-like gap-matched control, and a 2,048-day context test.

Headline results, after correction:
- Zero-shot foundation models are the strongest family, but on station observations most margins beyond day 1 are not statistically separable, and the top six sit within one critical difference.
- **No model beats climatology on tropical rainfall** — on the 35-station BMD record beyond a few days, and on 10-day CHIRPS rainfall at any horizon (on the 10-station GHCN-Bangladesh track point estimates are positive but never statistically separable).
- **Reanalysis looks markedly more predictable than the stations it represents** (78 of 108 comparisons significant), so reanalysis-only benchmarks overstate skill.
- A long-lead tropical **temperature** penalty holds for half the roster; ~60% is explained by sparser station context, and for Chronos-2 and TiRex a residual survives even at 2,048 days of history. Rainfall shows no systematic penalty — at day 1 the significant results run the other way.
- Skill is strongly **regime-dependent**: strong in monsoon break spells, significantly worse than climatology in active spells.
- The benchmark is **leakage-free on both sides** of the tropical-vs-temperate contrast, and reproducible in ~4 GPU-hours on free hardware.

Results: `reports/week3_baselines_report.md`, `reports/week4_fm_report.md`, `reports/week5_significance_report.md`. Week 6 is the paper write-up and release.

## Quickstart
```bash
uv sync                      # Python 3.12 environment
bash scripts/run_all.sh      # downloads -> QC -> processed tracks -> labels -> figures -> tests
uv run pytest -q             # tests only
```
Foundation models run in isolated environments under `envs/` (e.g. `envs/chronos2`); they write forecasts to `data/predictions_cache/`.

## Data tracks
| Track | Source | Coverage | Role |
|---|---|---|---|
| BMD stations | Zubair et al. 2024, Mendeley 10.17632/tbrhznpwg9.1 | 35 stations, daily, 1961–2023 | Primary observational target |
| GHCN-Daily Bangladesh | NOAA NCEI | 10 stations co-located with BMD | Second observation track; cross-check |
| GHCN-Daily temperate | NOAA NCEI | 32 Global-North mid-latitude stations, same WMO network | Tropical-vs-temperate contrast |
| NASA POWER points | NASA LaRC | all 77 station locations, 1981–2025 | Reanalysis-vs-observation contrast |
| CHIRPS dekadal | WFP / HDX | 64 districts, 1981–2026 | Independent satellite rainfall |

See `DATASHEET.md` for composition, preprocessing and limitations, `LICENSING.md` for terms, and `NOVELTY_LOG.md` for the literature position.

## Layout
```
src/bwb/data/    loaders (bmd, ghcn, chirps, nasa_power, indices), QC detectors, evaluation-stratum labels
src/bwb/eval/    metrics (MASE, CRPS, pinball, coverage)
scripts/         pipeline steps (run_all.sh lists the order)
configs/         splits.yaml (temporal splits, rolling-origin protocol, tracks)
data/processed/  parquet tracks + labels (rebuilt by the pipeline, not committed)
reports/         go/no-go memo, audits, data report, figures (+ CSV table views)
tests/           loaders, QC, metrics, labels, split-leakage guards
```

## Leakage rules
- Climatologies, percentile thresholds and model fitting use the training period only (≤ 2010).
- Forecast context may reach into earlier splits; scored targets must lie inside the evaluated split.
- Only windows with complete context and target are scored, identically in every track.
- Stratum labels (phase, spell, extremes, cyclones) are for scoring only and are never model inputs.
