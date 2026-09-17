#!/usr/bin/env bash
# Download the remaining Week-4 Kaggle runs, verify them, install only what passes, and rebuild the affected tables.
#   ipolas/bwb-fm-week4b        run records, CHIRPS dekadal forecasts (8 models x 4 contexts), context-ablation scores
#   ipolas/bwb-autoarima-a / -b AutoARIMA daily forecasts (8 + 8 tasks)
# Usage: caffeinate -dimsu bash scripts/install_kaggle_week4b.sh   (the Mac must not sleep during downloads)
# Download design as in install_kaggle_fm_outputs.sh: per-pattern downloads, wall-clock-limited attempts, file counts.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/interim/kaggle_week4b
PY="uv run python -W ignore"
KAGGLE=.venv/bin/kaggle
ATTEMPT_TIMEOUT=1200
mkdir -p "$OUT/fm" "$OUT/arima_a" "$OUT/arima_b"

count() { find "$1" -name "$2" 2>/dev/null | wc -l | tr -d ' '; }

fetch() {  # fetch <kernel> <dest> <regex> <label> <count-dir> <name-glob> <expected>
  local a n
  for a in 1 2 3 4 5; do
    n=$(count "$5" "$6")
    if [ "$n" -ge "$7" ]; then echo "  $4: $n/$7 files"; return 0; fi
    echo "  $4: attempt $a ($n/$7 present)"
    .venv/bin/python scripts/_run_with_timeout.py "$ATTEMPT_TIMEOUT" "$KAGGLE" kernels output "$1" -p "$2" -q \
      --file-pattern "$3" > "$2/download_$4.log" 2>&1 || echo "    attempt $a ended: $(tail -1 "$2/download_$4.log")"
  done
  n=$(count "$5" "$6")
  if [ "$n" -ge "$7" ]; then echo "  $4: $n/$7 files"; return 0; fi
  echo "  $4: INCOMPLETE ($n/$7)"; return 1
}

echo "== download"
fetch ipolas/bwb-fm-week4b "$OUT/fm" '^(records/|week4b_results)' records "$OUT/fm/records" '*.jsonl' 29 || true
cat "$OUT/fm/week4b_results.json" 2>/dev/null || echo "WARNING: no week4b_results.json"
fetch ipolas/bwb-fm-week4b "$OUT/fm" '^dekadal_cache/' dekadal "$OUT/fm/dekadal_cache" 'chirps__rfh.parquet' 32 || true
fetch ipolas/bwb-fm-week4b "$OUT/fm" '^ablation/board_' ablation_boards "$OUT/fm/ablation" 'board_*.csv' 13 || true
fetch ipolas/bwb-fm-week4b "$OUT/fm" '^ablation/scores/' ablation_scores "$OUT/fm/ablation/scores" '*.parquet' 208 || true
for part in a b; do
  fetch "ipolas/bwb-autoarima-$part" "$OUT/arima_$part" '^(predictions_cache/|arima_)' "arima_$part" "$OUT/arima_$part/predictions_cache" '*.parquet' 8 || true
  cat "$OUT/arima_$part/arima_${part}_results.json" 2>/dev/null || echo "WARNING: no arima_${part}_results.json"
done

echo "== verify + install: dekadal forecasts"
ok=$($PY - "$OUT/fm/dekadal_cache" <<'PY'
import sys
from pathlib import Path
import pandas as pd
from bwb.eval.dekadal import check_predictions
w = pd.read_parquet("data/processed/windows_dekadal.parquet")
for f in sorted(Path(sys.argv[1]).glob("*/chirps__rfh.parquet")):
    bad = check_predictions(pd.read_parquet(f), w, 6)
    print(f.parent.name if not bad else f"FAILED:{f.parent.name}:{bad}", file=sys.stdout if not bad else sys.stderr)
PY
)
for name in $ok; do mkdir -p "data/predictions_cache/$name"; cp "$OUT/fm/dekadal_cache/$name/chirps__rfh.parquet" "data/predictions_cache/$name/"; done
echo "installed dekadal: $(echo $ok | wc -w | tr -d ' ') runs"

echo "== install: context-ablation scores (verified on Kaggle by score_fm_runs.py) and run records"
mkdir -p reports/ablation && cp -R "$OUT/fm/ablation/." reports/ablation/ 2>/dev/null || true
cp "$OUT"/fm/records/*.jsonl reports/ 2>/dev/null || true

echo "== verify + install: AutoARIMA"
mkdir -p "$OUT/arima/AutoARIMA"
cp "$OUT"/arima_a/predictions_cache/AutoARIMA/*.parquet "$OUT"/arima_b/predictions_cache/AutoARIMA/*.parquet "$OUT/arima/AutoARIMA/" 2>/dev/null || true
arima_ok=0
if $PY scripts/verify_predictions.py --models AutoARIMA --root "$OUT/arima"; then
  mkdir -p data/predictions_cache/AutoARIMA && cp "$OUT"/arima/AutoARIMA/*.parquet data/predictions_cache/AutoARIMA/ && arima_ok=1
  echo "AutoARIMA installed"
else
  echo "AutoARIMA NOT installed (verification failed or incomplete)"
fi

echo "== rebuild"
$PY scripts/run_dekadal_baselines.py | tee reports/leaderboard_dekadal.log || echo "dekadal leaderboard reported problems"
if [ "$arima_ok" = 1 ]; then
  $PY scripts/build_leaderboard.py > reports/build_leaderboard.log 2>&1 || { echo "leaderboard build FAILED"; exit 1; }
  $PY scripts/analyze_week4.py > reports/week4_contrasts.log
  $PY scripts/week4_summary.py > reports/week4_summary.log 2>&1
  echo "daily leaderboard, contrasts and summary rebuilt with AutoARIMA"
fi
$PY scripts/analyze_week4b.py | tee reports/week4b_analysis.log
