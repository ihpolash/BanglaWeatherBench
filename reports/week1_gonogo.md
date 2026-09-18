# Week 1, Go/No-Go memo

Date: 2026-09-15. Scope: novelty lock + data feasibility for BanglaWeatherBench.

## Decision: GO

Both gates pass.

### Gate 1, Novelty: GO
Full evidence in `NOVELTY_LOG.md` and the Week 1 plan. In short:
- **Open:** a released Bangladesh weather forecasting benchmark; time-series foundation models (TSFMs) on any South Asian weather data; monsoon-phase-stratified TSFM evaluation; the reanalysis-vs-observation skill gap for TSFMs.
- **Narrowed:** geographic disparity. SAFE (arXiv 2510.26099) already measured it for gridded AI weather models on ERA5. Our version is TSFMs × station observations × precipitation × monsoon regime, against a temperate reference track.
- **Narrowed:** monsoon benchmarking. Masiwal et al. (arXiv 2602.03767) benchmark monsoon onset for AI weather models over India.

### Gate 2, Is the BMD observational spine usable? GO
Source: Zubair et al. 2024, Mendeley 10.17632/tbrhznpwg9.1. All 36 files downloaded, and every SHA-256 matches the Mendeley metadata (`scripts/download_bmd_mendeley.py`).

| Check | Result |
|---|---|
| Rows / stations | 543,839 / 35, matches the publication |
| Invalid dates, duplicate (station, date) | 0 / 0 |
| Calendar gaps, nulls | 0 / 0. Every station runs daily to 2023-12-31, so gaps were filled upstream |
| Start years | 1961 ×5 (Bogra, Chittagong, Cox's Bazar, Dhaka, Sylhet); 1985 ×8; latest start 2008 (Ambagan) |
| Every station has 2016–2023 data | Yes (2,922 days each) |

**Imputation audit** (`scripts/audit_bmd_imputation.py`  to  `reports/bmd_imputation_audit.csv`). Share of 2016–2023 days flagged by constant-run (≥5 days) or linear-run (≥5 days) detectors, median / max across the 35 stations:

| Variable | Median | Max |
|---|---|---|
| Rainfall | 0.00% | 0.72% |
| Temperature | 1.10% | 3.94% |
| Sunshine | 1.06% | 14.10% (Kutubdia) |
| Humidity | 4.24% | 7.91% |

- Only 1 of 140 station-variable series exceeds 10% flagged (Kutubdia sunshine).
- **Off-grid precision** (e.g. non-integer humidity), the clearest sign of interpolation, is 0.00% median and ≤0.10% max in the test period. It is present only in earlier years (sunshine up to 11.8% over the full record). Imputation is concentrated in historical data, not the test window.
- **Placebo calibration:**
  - A month-mean-fill detector flagged real data at the same rate as a shifted placebo target (temperature 2.22% vs 2.40%), so it was removed as noise.
  - Constant runs exceed a within-month shuffle baseline by 0.2–1.5 percentage points, depending on the variable. Treat the flags as an upper bound on real imputation. Humidity is recorded as integers, so it repeats naturally.
- **Action:** mask flagged days out of the metrics; publish the mask with the dataset. Consider dropping Kutubdia sunshine from Tier 2.

### Supporting Week 1 checks
| Plan item | Result |
|---|---|
| uv Python 3.12 env on M1 | OK (pandas 3.0.5, numpy 2.5.3, statsmodels, pdfplumber) |
| Station join: Mendeley  and  BMD normals PDFs | 35/35 humidity normals; 34/35 rainfall + daily Tmin normals. Only Ambaganctg is missing from the 34-station PDFs |
| PDF extraction | `pdfplumber.extract_table()` handles the ruled tables directly, and the feared baseline-jitter problem does not occur. Extracted values match rendered crops (`reports/verify_*.png`) |
| CHIRPS duplicate PCODEs | They are **split polygons, not duplicates**. BD10 = 55 + 345 px = 400 px, which equals the sum over its districts. Merged with pixel weights in `bwb.data.chirps`; the panel is 64 × 1,644, with unique keys |
| Chronos-2 smoke test | Runs on **M1 CPU**: load 45 s; 30-day forecast 1.2 s; Dhaka Jan-2016 MAE 1.29 °C. Writes the prediction cache, which reads back from the main env. The Kaggle GPU run itself still needs the user's account |
| Temperate reference track | GHCN-Daily: 4,947 stations at 35–60°N with PRCP+TMAX+TMIN covering 1985–2023 (US 2,904, DE 285, UK 72, FR 70, JP 81) |

## New findings that change the design
1. **Local CPU inference is viable** for the smaller TSFMs. Chronos-2 takes ~1.2 s per series-forecast on the M1, so part of the leaderboard can run locally and Kaggle becomes a speed-up, not a requirement. Benchmark throughput with batching in Week 3.
2. **GHCN-Daily has 10 Bangladesh stations** (Rangpur, Bogra, Sylhet/Osmany, Ishurdi, Dhaka/Tejgaon, Jessore, Feni, Barisal, Chittagong/Shah Amanat, Cox's Bazar) with PRCP/TMAX/TMIN from ~1982 to 2024–2026. This gives us:
   - a **same-source** tropical-vs-temperate contrast, removing the objection that the difference is a data-source artifact;
   - an independent cross-check of BMD values at co-located stations;
   - test data beyond BMD's 2023 cutoff.
3. **Correction to the Week 1 plan:** FoundTS is an earlier title of TSFM-Bench under the **same arXiv ID (2410.11802)**, so `FoundTS.pdf` being identical to `TFB-TSFM-Bench.pdf` is expected. Only TFB (arXiv 2403.20150) was genuinely missing; it is now in `Papers/`.

## Still open, needs the user
- **arXiv endorser** for cs.AI: identify and contact one.
- **Kaggle/Colab account** for the GPU run of the larger models (TimesFM 2.5, Toto-2.0 313M, Sundial).
- Decide whether to add GHCN Bangladesh stations as a second observational track (recommended).

## Papers added to `Papers/` this week
arXiv 2509.01879 (MAUSAM), 2406.14399 (WEATHER-5K), 2606.18367, 2603.05710, 2510.26099 (SAFE), 2602.03767 (Masiwal), 2403.20150 (TFB), 2605.01126 (EWB), 2510.13654.
