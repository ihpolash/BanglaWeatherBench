"""Integrity checks for cached forecasts before they enter a leaderboard.

Usage: uv run python scripts/verify_predictions.py --models chronos2,timesfm25 [--root data/predictions_cache]
Checks per (model, track, variable): expected row count (windows x 30), window ids identical to the index, leads 1..30
for every window, no missing values, monotone quantiles, non-negativity for non-negative variables. Exit 1 on failure.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bwb.data.store import NON_NEGATIVE, TRACK_VARIABLES
from bwb.eval.harness import QCOLS

H = 30


def check(p: pd.DataFrame, w: pd.DataFrame, var: str) -> list[str]:
    bad = []
    if len(p) != len(w) * H:
        bad.append(f"rows {len(p)} != {len(w) * H}")
    if set(p.window_id.unique()) != set(w.window_id):
        bad.append("window ids differ from index")
    leads = p.sort_values(["window_id", "lead"]).lead.to_numpy()
    if len(leads) % H or not (leads.reshape(-1, H) == np.arange(1, H + 1)).all():
        bad.append("leads not 1..30 per window")
    if p[["mean", *QCOLS]].isna().any().any():
        bad.append("missing values")
    if (np.diff(p[QCOLS].to_numpy(), axis=1) < -1e-6).any():
        bad.append("quantiles not monotone")
    if var in NON_NEGATIVE and (p[["mean", *QCOLS]].to_numpy() < 0).any():
        bad.append("negative values for non-negative variable")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--root", default="data/predictions_cache")
    ap.add_argument("--tracks", default=None, help="comma list; default = every track (gap-matched runs exist only for ghcn_temperate)")
    args = ap.parse_args()
    windows = pd.read_parquet("data/processed/windows_daily.parquet")
    failures, checked, missing = [], 0, []
    for model in args.models.split(","):
        for track, variables in TRACK_VARIABLES.items():
            if args.tracks and track not in args.tracks.split(","):
                continue
            for var in variables:
                f = Path(args.root) / model / f"{track}__{var}.parquet"
                if not f.exists():
                    missing.append(f"{model}/{track}__{var}")
                    continue
                w = windows[(windows.track == track) & (windows["var"] == var)]
                bad = check(pd.read_parquet(f), w, var)
                checked += 1
                if bad:
                    failures.append((model, track, var, bad))
    print(f"files checked: {checked}; missing: {len(missing)}; with problems: {len(failures)}")
    for m in missing:
        print("  MISSING:", m)
    for fl in failures:
        print("  PROBLEM:", fl)
    sys.exit(1 if failures or missing else 0)


if __name__ == "__main__":
    main()
