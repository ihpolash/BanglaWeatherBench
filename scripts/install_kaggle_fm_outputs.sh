#!/usr/bin/env bash
# Download the Week-4 foundation-model run from Kaggle, verify every forecast file, and only then install it into the
# shared prediction cache, rebuild the leaderboard and run the Week-4 contrasts.
# Usage: bash scripts/install_kaggle_fm_outputs.sh [kernel-slug]
#
# Download design (a single `kaggle kernels output` call hung for over an hour at 0% CPU on the full output):
# - one download per model folder via --file-pattern;
# - each attempt wall-clock limited by scripts/_run_with_timeout.py, which kills the whole process group;
# - the kaggle binary is called directly (no `uv run` layer whose grandchild could outlive the time limit);
# - up to 5 attempts per folder, files kept across attempts; a folder is done at 16 files (main) or 2 (gap-matched).
set -euo pipefail
cd "$(dirname "$0")/.."
KERNEL="${1:-ipolas/bwb-fm-zero-shot}"
OUT=data/interim/kaggle_fm_full
MODELS="chronos2 chronos_bolt timesfm25 toto2 tirex moirai2 ttm_r2 sundial"
PY="uv run python -W ignore"
KAGGLE=.venv/bin/kaggle
ATTEMPT_TIMEOUT=1200
mkdir -p "$OUT"

count_parquet() { find "$OUT/predictions_cache/$1" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' '; }

download_pattern() {  # download_pattern <label> <regex>
  .venv/bin/python scripts/_run_with_timeout.py "$ATTEMPT_TIMEOUT" "$KAGGLE" kernels output "$KERNEL" -p "$OUT" -q \
    --file-pattern "$2" > "$OUT/download_$1.log" 2>&1
}

download_folder() {  # download_folder <folder> <expected_files>
  local folder=$1 expected=$2 attempt n
  for attempt in 1 2 3 4 5; do
    n=$(count_parquet "$folder")
    if [ "$n" -ge "$expected" ]; then echo "  $folder: $n/$expected files"; return 0; fi
    echo "  $folder: attempt $attempt ($n/$expected present)"
    download_pattern "$folder" "predictions_cache/${folder}/" || echo "    attempt $attempt ended: $(tail -1 "$OUT/download_$folder.log")"
  done
  n=$(count_parquet "$folder")
  if [ "$n" -ge "$expected" ]; then echo "  $folder: $n/$expected files"; return 0; fi
  echo "  $folder: INCOMPLETE after 5 attempts ($n/$expected)"; return 1
}

echo "== run records and results"
for attempt in 1 2 3; do
  if download_pattern records '^fm_'; then break; fi
  sleep 20
done
cat "$OUT"/fm_full_results.json 2>/dev/null || echo "WARNING: no fm_full_results.json"

echo "== forecasts (per model folder)"
incomplete=()
for m in $MODELS; do
  download_folder "$m" 16 || incomplete+=("$m")
  download_folder "${m}_gapmatched" 2 || incomplete+=("${m}_gapmatched")
done

echo "== verification"
# ${arr[@]+"${arr[@]}"} expands safely when arr is empty (macOS bash 3.2 treats a plain "${arr[@]}" as unbound under set -u).
main_ok=(); gap_ok=(); failed=()
for m in $MODELS; do
  if [ -d "$OUT/predictions_cache/$m" ] && $PY scripts/verify_predictions.py --models "$m" --root "$OUT/predictions_cache"; then
    main_ok+=("$m")
  else
    failed+=("$m")
  fi
  g="${m}_gapmatched"
  if [ -d "$OUT/predictions_cache/$g" ] && $PY scripts/verify_predictions.py --models "$g" --root "$OUT/predictions_cache" --tracks ghcn_temperate; then
    gap_ok+=("$g")
  else
    failed+=("$g")
  fi
done
echo "incomplete downloads: ${incomplete[*]+${incomplete[*]}}"
echo "verified main: ${main_ok[*]+${main_ok[*]}}"
echo "verified gap-matched: ${gap_ok[*]+${gap_ok[*]}}"
echo "failed verification: ${failed[*]+${failed[*]}}"

echo "== install (verified models only)"
for m in ${main_ok[@]+"${main_ok[@]}"} ${gap_ok[@]+"${gap_ok[@]}"}; do
  mkdir -p "data/predictions_cache/$m"
  cp "$OUT/predictions_cache/$m"/*.parquet "data/predictions_cache/$m/"
done
cp "$OUT"/fm_run_*.jsonl reports/ 2>/dev/null || true

echo "== leaderboard + contrasts"
if ! $PY scripts/build_leaderboard.py > reports/build_leaderboard.log 2>&1; then
  echo "leaderboard build FAILED - see reports/build_leaderboard.log; contrasts not run"; exit 1
fi
echo "leaderboard rebuilt"
$PY scripts/analyze_week4.py | tee reports/week4_contrasts.log
