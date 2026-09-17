#!/usr/bin/env bash
# One-command rebuild of the BanglaWeatherBench data pipeline (Weeks 1-2), from raw downloads to EDA figures.
# Requires: uv. Network access for downloads. Re-runs skip completed downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="uv run python -W ignore"

echo "== 1. downloads"
$PY scripts/download_bmd_mendeley.py
bash scripts/download_external.sh

echo "== 2. quality audit + reference-station selection"
$PY scripts/audit_bmd_imputation.py
$PY scripts/select_temperate_reference.py
$PY scripts/finalize_temperate_reference.py

echo "== 3. spatial metadata"
$PY scripts/build_station_coords.py
$PY scripts/build_crosswalk.py
$PY scripts/build_cyclone_days.py

echo "== 4. reanalysis at station locations (slow: ~77 API calls)"
$PY scripts/fetch_nasa_power_points.py

echo "== 5. processed tracks, labels"
$PY scripts/build_processed.py
$PY scripts/build_nasa_power_points.py
$PY scripts/build_labels.py

echo "== 6. EDA tables + figures"
$PY scripts/eda_stats.py
$PY scripts/make_eda_figures.py

echo "== 7. baselines (Week 3): window index + naive, statistical, LightGBM, deep learning, CHIRPS dekadal"
$PY scripts/run_baselines.py                      # builds data/processed/windows_daily.parquet
$PY scripts/run_stat_lgbm.py                      # AutoETS, AutoTheta, LightGBM (~2 h on 8 cores)
# Deep-learning baselines: run on Kaggle GPU (private dataset + notebook; ~1-2 h) rather than locally (~4-5 h on M1):
#   bash scripts/kaggle_bundle.sh version && uv run kaggle kernels push -p kaggle/neural_kernel --accelerator NvidiaTeslaT4
#   (wait for completion) uv run kaggle kernels output ipolas/bwb-neural-baselines -p data/interim/kaggle_neural_output
#   then copy data/interim/kaggle_neural_output/predictions_cache/* into data/predictions_cache/
# Local fallback (slow; must not overlap other heavy jobs on 8 GB RAM):
#   PYTHONPATH=src uv run --project envs/neural python -W ignore scripts/run_neural_baselines.py --accelerator mps
$PY scripts/run_dekadal_baselines.py
# Integrity gate: every cached forecast must pass before any leaderboard is built.
$PY scripts/verify_predictions.py --models naive,seasonal_naive,climatology,AutoETS,AutoTheta,lightgbm,NHITS,DLinear,PatchTST
# Week 4 zero-shot foundation models run on Kaggle (kaggle/fm_kernel, MODE="full"); download their caches, then:
#   $PY scripts/verify_predictions.py --models chronos2,chronos_bolt,timesfm25,toto2,tirex,moirai2,ttm_r2,sundial
$PY scripts/build_leaderboard.py

echo "== 8. tests"
uv run pytest -q
