# Datasheet: BanglaWeatherBench v0 (processed tracks)

Status: Week 2 draft (2026-09-15), following Gebru et al., *Datasheets for Datasets*. Numbers refer to the files in `data/processed/`, rebuilt by `scripts/run_all.sh`.

## Motivation
- **Purpose:** benchmark time-series foundation models on tropical-monsoon station observations, stratified by monsoon regime. The comparison tracks — a GHCN temperate reference and NASA POWER reanalysis — let the benchmark measure (a) tropical-vs-temperate skill gaps and (b) observation-vs-reanalysis skill gaps.
- **Gap filled:** no existing Bangladesh weather dataset defines forecasting tasks, splits or strata (see `NOVELTY_LOG.md`).

## Composition
| File | Unit | Size | Variables |
|---|---|---|---|
| `bmd_daily.parquet` | station-day | 543,839 rows; 35 BMD stations; 1961–2023 (per-station starts) | Rainfall (mm), Temperature (daily mean °C), Humidity (%), Sunshine (h); `<var>_flag` = suspected imputation |
| `bmd_labels.parquet` | station-day | aligned 1:1 with `bmd_daily` | season, monsoon_phase, spell (active/break/normal, Jul–Aug), heavy_rain, very_heavy_rain, hot_day, cold_day, cyclone_day |
| `national_spells.parquet` | day | 1961–2023 | all-station mean rainfall, reporting-station count, spell |
| `ghcn_daily.parquet` | station-day | 1,051,444 rows; 10 Bangladesh + 32 temperate stations | Rainfall, Tavg, Tmax, Tmin. Dates kept in GHCN's convention, not shifted to BMD's |
| `nasa_power_points.parquet` | point-day | 1,265,572 rows; 77 points (35 BMD + 10 GHCN-BD + 32 temperate); 1981–2025 | T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, ALLSKY_SFC_SW_DWN (6.7% missing, pre-1984), WS2M, T2MDEW, PS |
| `chirps_dekadal_adm2.parquet` | district-dekad | 105,216 rows; 64 districts × 1,644 dekads; 1981–2026 | rfh, rfh_avg, r1h, r3h, anomaly ratios |
| `nasa_power_districts_daily.parquet` | district-day | 171,190 rows; 38 districts; 2012–2024 | legacy reanalysis covariates, no precipitation |
| `bmd_station_coords.csv`, `crosswalk_spatial.csv` | station | 35 / 83 locations | WMO id, lat/lon, admin1/admin2 PCODE, nearest legacy NASA POWER district |
| `cyclone_station_days.parquet` | station-day | 2,183 station-days, 71 storms (1961–) | storm id, name, minimum distance, maximum wind |
| `bmd_extreme_thresholds.csv` | station | 35 | training-period wet-day 95th/99th percentile rainfall |

## Collection and provenance
- **BMD:** Zubair et al. (2024), Mendeley DOI 10.17632/tbrhznpwg9.1. Upstream providers had already filled gaps (forward/backward fill, mean imputation, interpolation). All 36 files were SHA-256-verified against the Mendeley metadata.
- **Coordinates:** from NOAA ISD history (WMO block 41), since the BMD dataset ships none. The GHCN station list covers the two stations missing from ISD (Jessore, Sylhet).
- **Other sources:**
  - GHCN-Daily (NOAA NCEI);
  - NASA POWER daily point API, local-solar-time days;
  - CHIRPS via WFP/HDX;
  - IBTrACS v04r01;
  - HDX COD-AB boundaries;
  - NOAA PSL ONI/DMI indices.

## Preprocessing
- **Suspected imputation** is *flagged, not removed*. Detectors:
  - constant runs of 5 or more days (zero rainfall exempt);
  - linear runs of 5 or more days;
  - values finer than the reporting precision (off-grid).

  A month-mean-fill detector was dropped after a placebo test showed it flags only coincidence. Flag rates over the full record: Rainfall 0.28%, Temperature 1.71%, Humidity 4.56%, Sunshine 3.89%.
- **CHIRPS:** two administrative units are split into two polygons that share one PCODE (BD10, BD1009). They are merged with pixel-count weights, and the anomaly ratios are recomputed.
- **GHCN QC:** values with a non-empty QFLAG are dropped (0.01–0.03%).
- **Calendar alignment, GHCN vs BMD:** GHCN rainfall is labelled by the day the 24-h total starts; BMD labels it by the day it ends. `align_to_bmd_convention()` shifts GHCN by +1 day, raising daily r from 0.31–0.63 to 0.68–0.90. The same offset appears for NASA POWER rainfall (best lag +1 at 31 of 35 stations).
- **Leakage:** climatologies and extreme thresholds are estimated on the training period only, using unflagged values.

## Splits and evaluation strata
- **Splits** (`configs/splits.yaml`): train ≤ 2010; validation 2011–2015; test 2016–2023; extension 2024–2025-08 (GHCN, NASA POWER).
- **Scored windows:** only rolling-origin windows with complete context and target are scored, identically in every track.
- **Monsoon phases:** dry, pre-monsoon (Mar–May), onset (1–15 Jun), peak (16 Jun–15 Oct), withdrawal (16 Oct–15 Nov). Onset and withdrawal dates follow Ferdoushi et al. (2023).
- **Active/break spells:** Ferdoushi et al. (2023), applied to the all-station mean.
  - Validation, 1988–2017: 70 active phases / 283 days and 108 break phases / 530 days, vs the paper's 65 / 271 and 104 / 517.
  - Test period, Jul–Aug: 16.7% active, 28.8% break.
- **Extremes (test period):** heavy rain (≥ training wet-day 95th percentile) 1.41% of station-days; very heavy rain (≥99th) 0.30%; hot day (≥ monthly 95th percentile) 11.98%; cyclone day (storm ≥34 kt within 300 km) 0.42%.
- **Cyclones in the test period:** 432 station-days across 14 storms, including Roanu, Mora, Fani, Bulbul, Amphan, Sitrang, Hamoon, Midhili and Mocha.
- **Label use:** labels stratify *scoring* only and must never be used as model inputs. Spell and phase labels use the observed rainfall of the target period.

## Known limitations
- **BMD gap filling is upstream and not fully recoverable.** The flags are an upper bound on detectable imputation, not a complete record of it.
- **GHCN Bangladesh stations are only ~65% complete in 2016–2023,** with near-empty years such as 2021. The temperate stations are ~99% complete.
- **The temperate reference is Europe-heavy:** no North American or East Asian stations in the same WMO network met the daily-mean temperature requirement.
- **BMD "Chittagong" and GHCN "Shah Amanat Intl" disagree more than the other co-located pairs** (Tavg MAE 0.86 °C), which suggests different sites.
- **NASA POWER rainfall differs strongly from BMD** (median r 0.53, totals +47%, heavy-rain capture 62%). This is a studied property of the reanalysis track, not an error in the data.
- **The cyclone subset is small** (14 test storms) and suits case studies, not a leaderboard.
- **ENSO/IOD correlations with national monsoon rainfall are weak** in 1961–2010 (|r| < 0.16; p < 0.05 needs |r| ≥ 0.29).

## Distribution and licensing
- **Derived data:** CC-BY-4.0, with source attributions listed in `LICENSING.md`.
- **Not redistributed:** the raw BMD normals PDFs (terms unstated). Only derived values are shipped.
