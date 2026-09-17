"""Verify and score daily foundation-model runs where they were produced, so only scores need downloading.

Used on Kaggle for the context-length ablation: a full daily run is ~250 MB of forecasts, its scores a few MB.
Usage: PYTHONPATH=src python scripts/score_fm_runs.py --models chronos2_ctx512[,...] [--root data/predictions_cache] \
    [--out reports/ablation] [--tracks bmd,...]
Every forecast file must pass the checks of scripts/verify_predictions.py; each task is then scored with the Week-3
harness (same windows, MASE scales and metrics as build_leaderboard.py). Writes
  <out>/scores/<model>/<track>__<var>.parquet   per window x lead, leads 1/3/7/14/30 (for Week-5 tests)
  <out>/board_<model>.csv                        track, var, lead -> MASE, sCRPS, RMSE, coverage80, n_windows
Exit 1 if any file is missing or fails a check. Skill vs climatology is added locally from the main leaderboard.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from bwb.data.store import TRACK_VARIABLES, load_track
from bwb.eval.harness import CACHE, aggregate, score, train_scales
from verify_predictions import check

CFG = yaml.safe_load(open("configs/splits.yaml"))
TRAIN_END = CFG["daily"]["train"]["end"]
LEADS = tuple(CFG["rolling_origin"]["daily"]["horizons_days"])
H = max(LEADS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--root", default=str(CACHE))
    ap.add_argument("--out", default="reports/ablation")
    ap.add_argument("--tracks", default=",".join(TRACK_VARIABLES))
    args = ap.parse_args()
    models, out = args.models.split(","), Path(args.out)
    windows = pd.read_parquet("data/processed/windows_daily.parquet")
    boards, problems = {m: [] for m in models}, []
    for track in args.tracks.split(","):
        series = scales = None
        for var in TRACK_VARIABLES[track]:
            w = windows[(windows.track == track) & (windows["var"] == var)]
            for m in models:
                f = Path(args.root) / m / f"{track}__{var}.parquet"
                if not f.exists():
                    problems.append(f"{m}/{track}__{var}: missing")
                    continue
                p = pd.read_parquet(f)
                bad = check(p, w, var)
                if bad:
                    problems.append(f"{m}/{track}__{var}: {bad}")
                    continue
                if series is None:
                    series = load_track(track)
                    scales = train_scales(series, TRAIN_END)
                sc = score(p, w, {k: v for k, v in series.items() if k[0] == var}, scales, H)
                sc = sc[sc.lead.isin(LEADS)]
                dest = out / "scores" / m / f"{track}__{var}.parquet"
                dest.parent.mkdir(parents=True, exist_ok=True)
                sc.to_parquet(dest, index=False)
                boards[m].append(aggregate(sc, w, leads=LEADS).assign(model=m))
                print(f"scored {m:22s} {track:28s} {var:18s} windows={len(w)}", flush=True)
    for m, rows in boards.items():
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(out / f"board_{m}.csv", index=False)
    for p in problems:
        print("PROBLEM:", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
