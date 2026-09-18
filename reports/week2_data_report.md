# Week 2, Data pipeline and EDA report

Date: 2026-09-15. Rebuild everything with `bash scripts/run_all.sh`. Composition and limitations are in `DATASHEET.md`.

## Deliverables
| Planned (Week 2) | Status | Where |
|---|---|---|
| Ingest all sources | Done: BMD, GHCN (Bangladesh + temperate), NASA POWER (77 points, with precipitation), CHIRPS, IBTrACS, ONI/DMI, boundaries | `data/processed/*.parquet` |
| District  and  station  and  grid crosswalk | Done: 83 locations mapped to admin1/admin2 PCODE (all inside a polygon; BMD covers 27/64 districts) | `data/processed/crosswalk_spatial.csv` |
| QC + imputation masks | Done: flags per variable. Detectors calibrated against placebo tests | `bmd_daily.parquet`, `reports/bmd_imputation_audit.csv` |
| Leakage-safe splits | Done: train ≤2010 / val 2011–15 / test 2016–23 / extension 2024–25; complete-window scoring | `configs/splits.yaml`, `tests/test_splits.py` |
| Monsoon strata + extremes | Done: phases, active/break spells (validated vs literature), extremes, cyclone days | `bmd_labels.parquet`, `national_spells.parquet` |
| EDA figures F1–F5 | Done, visually checked; table view per figure | `reports/figures/`, `reports/figures/data/` |
| Data card | Done (draft) | `DATASHEET.md` |
| Unit tests | 30+ tests passing | `tests/` |

## Findings that matter for the paper
1. **Active/break spells reproduce the literature.** With Ferdoushi et al.'s (2023) definition applied to the all-station mean, 1988–2017 gives 70 active phases / 283 days and 108 break phases / 530 days, vs the published 65 / 271 and 104 / 517. Applying the rule per station instead gave a 4.7 break/active ratio, which confirms spells are a national-scale property.
2. **Reanalysis rainfall is a different target from observed rainfall.** NASA POWER vs BMD at the 35 stations, 2016–2023:
   - Rainfall r median 0.53 (range 0.36–0.64); totals +47%.
   - Wet-day agreement 76%; heavy-rain days (≥50 mm) captured 62%.
   - Temperature r 0.95; humidity r 0.70.

   This motivates the reanalysis-vs-observation contribution.
3. **Two independent rainfall sources are offset by one day relative to BMD.** BMD labels a 24-h total by its end day, while GHCN (best lag +1 at all 10 stations) and NASA POWER (+1 at 31/35) label it by its start day. Cross-source comparisons must align dates. Forecasting each track on its own calendar is unaffected.
4. **Rainfall is zero-inflated and heavy-tailed.** 67.5% of station-days are dry (Tamim et al. 2026 report 68% on the same data). The wet-day 95th and 99th percentiles are 77 mm and 144 mm; the maximum is 590 mm. Probabilistic metrics (CRPS) and MASE must handle the zeros; sMAPE is unsuitable.
5. **Seasonality dominates.** Median monthly rainfall peaks at 454 mm in July against 7 mm in January. Temperature is flat at ~28 °C from April to September. Seasonal-naive and climatology baselines will be strong, expect a hard bar for zero-shot foundation models.
6. **ENSO/IOD links to national monsoon rainfall are weak.** All lead correlations are |r| ≤ 0.16 for 1961–2010, below the p < 0.05 threshold of 0.29. Teleconnection covariates should enter as an ablation, not a core input.
7. **The test period has a small but usable extreme subset.** Heavy-rain days 1.41%; very-heavy 0.30%; hot days 11.98%; cyclone station-days 432 across 14 storms (e.g. Amphan, Bulbul, Mocha).
8. **Data completeness diverges between tracks.** Over 2016–2023, the GHCN Bangladesh stations are ~65% complete (2021 is nearly empty) and the temperate stations ~99% complete (F3). Complete-window scoring keeps this from biasing skill; the gap itself is reported as data-divide evidence.

## Decisions taken this week
- **The imputation mask includes an off-grid-precision detector;** the month-mean-fill detector was dropped (placebo showed coincidence only).
- **CHIRPS split polygons are merged with pixel weights,** not dropped.
- **The temperate reference excludes Algeria and Iran** so it stays Global-North; 32 stations in 14 countries.
- **The Bangladesh-vs-temperate comparison uses GHCN on both sides,** without date shifting, so any skill gap cannot be a data-source artifact.
- **NASA POWER is fetched at exact station locations.** The legacy district file sits up to 172 km from stations (Teknaf) and is kept only as covariates.

## Open items carried into Week 3
- Rolling-origin harness and window index cache: enumerate complete windows per track/variable/horizon and confirm ≥100 windows per task.
- Baselines: seasonal-naive, climatology, ETS/Theta/ARIMA, LightGBM, and small deep models.
- A deliberate-leakage test that must fail: a harness sanity check.
- Mid-project novelty re-check (end of Week 3).
- Decide whether Kaggle hosts a private processed-data bundle for GPU runs (needs your confirmation before any upload).
