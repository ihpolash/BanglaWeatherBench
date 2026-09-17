#!/usr/bin/env bash
# Build the private Kaggle dataset bundle for Kaggle runs: code (neural baselines, foundation models daily + dekadal, run scoring,
# AutoARIMA) + processed parquet.
# Usage: bash scripts/kaggle_bundle.sh [create|version]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-create}"
OWNER=$(uv run python -c "import json,pathlib,os; print(os.environ.get('KAGGLE_USERNAME','ipolas'))")
STAGE=data/interim/kaggle_dataset
rm -rf "$STAGE" && mkdir -p "$STAGE"

uv run python - <<'EOF'
import zipfile, pathlib
files = [*pathlib.Path("src/bwb").rglob("*.py"), *(pathlib.Path("scripts") / f for f in ("run_neural_baselines.py", "run_fm_zero_shot.py", "run_fm_dekadal.py", "score_fm_runs.py",
                                                "verify_predictions.py", "run_stat_lgbm.py")), pathlib.Path("configs/splits.yaml"),
         *(pathlib.Path("data/processed") / f for f in ("windows_daily.parquet", "bmd_daily.parquet", "ghcn_daily.parquet", "nasa_power_points.parquet",
                                                       "chirps_dekadal_adm2.parquet", "windows_dekadal.parquet"))]
with zipfile.ZipFile("data/interim/kaggle_dataset/bwb_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.as_posix())
print(f"bundle: {len(files)} files, {pathlib.Path('data/interim/kaggle_dataset/bwb_bundle.zip').stat().st_size/1e6:.1f} MB")
EOF

cat > "$STAGE/dataset-metadata.json" <<EOF
{
  "title": "bwb-bundle",
  "id": "${OWNER}/bwb-bundle",
  "subtitle": "Private: BanglaWeatherBench code + processed tracks for GPU baseline runs",
  "licenses": [{"name": "CC-BY-4.0"}]
}
EOF

if [ "$MODE" = "create" ]; then
  uv run kaggle datasets create -p "$STAGE" -q   # private by default (no --public flag)
else
  uv run kaggle datasets version -p "$STAGE" -m "update bundle" -q
fi
