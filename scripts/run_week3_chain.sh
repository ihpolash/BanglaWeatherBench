#!/usr/bin/env bash
# Run heavy Week-3 baseline jobs sequentially (8 GB RAM cannot hold them concurrently).
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_baselines.py" >/dev/null; do sleep 20; done
echo "chain: naive baselines finished $(date)" >> reports/run_stat_lgbm.log
uv run python -W ignore scripts/run_stat_lgbm.py --n-jobs 4 --chunk 1500 >> reports/run_stat_lgbm.log 2>&1
echo "chain: stat+lgbm exited with $? $(date) windows done" >> reports/run_stat_lgbm.log
PYTHONPATH=src uv run --project envs/neural python -W ignore scripts/run_neural_baselines.py --max-steps 1000 --accelerator mps >> reports/run_neural.log 2>&1
echo "chain: neural exited with $? $(date) windows done" >> reports/run_neural.log
