"""Build the daily window index for every track and run the naive reference baselines.

Usage: uv run python scripts/run_baselines.py [--models naive,seasonal_naive,climatology] [--tracks bmd,...]
Outputs:
  data/processed/windows_daily.parquet                  window index (all tracks)
  data/predictions_cache/<model>/<track>__<var>.parquet  forecasts (resumable: existing files are skipped)
  reports/leaderboard_baselines.csv                     MASE / sCRPS / RMSE / coverage by track, var, model, lead
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from bwb.data.store import NON_NEGATIVE, TRACK_VARIABLES, load_track
from bwb.eval.harness import SPLIT, WINDOWS, aggregate, cache_path, run_model, score, train_scales
from bwb.eval.windows import build_windows
from bwb.models.baselines import BASELINES

CFG = yaml.safe_load(open("configs/splits.yaml"))
TRAIN_END = CFG["daily"]["train"]["end"]
SPLIT_DATES = (CFG["daily"][SPLIT]["start"], CFG["daily"][SPLIT]["end"])
ID_OFFSET = 0 if SPLIT == "test" else 1_000_000  # validation window ids never collide with test ids
RO = CFG["rolling_origin"]
H = max(RO["daily"]["horizons_days"])
STRIDE = RO["daily"]["stride_days"]
WIN = WINDOWS


def get_windows(tracks):
    if WIN.exists():
        w = pd.read_parquet(WIN)
        if set(tracks) <= set(w.track.unique()):
            return w
    parts = []
    for t in TRACK_VARIABLES:
        parts.append(build_windows(load_track(t), t, *SPLIT_DATES, horizon=H, stride=STRIDE, min_history_days=RO["min_history_days"]))
    w = pd.concat(parts, ignore_index=True)
    w["window_id"] = range(ID_OFFSET, ID_OFFSET + len(w))
    w.to_parquet(WIN, index=False)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="naive,seasonal_naive,climatology")
    ap.add_argument("--tracks", default=",".join(TRACK_VARIABLES))
    args = ap.parse_args()
    tracks, models = args.tracks.split(","), args.models.split(",")

    windows = get_windows(tracks)
    print("windows per track/var:\n", windows.groupby(["track", "var"]).size().to_string())
    rows = []
    for track in tracks:
        series = load_track(track)
        scales = train_scales(series, TRAIN_END)
        for var in TRACK_VARIABLES[track]:
            w = windows[(windows.track == track) & (windows["var"] == var)]
            sub = {k: v for k, v in series.items() if k[0] == var}
            for m in models:
                path = cache_path(m, track, var)
                if path.exists():
                    preds = pd.read_parquet(path)
                else:
                    t0 = time.time()
                    nonneg = var in NON_NEGATIVE
                    preds = run_model(lambda v, m=m: BASELINES[m](non_negative=nonneg), sub, w, TRAIN_END, H)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    preds.to_parquet(path, index=False)
                    print(f"{m:15s} {track:28s} {var:18s} {len(w):6d} windows  {time.time() - t0:6.1f}s", flush=True)
                sc = score(preds, w, sub, scales, H)
                lb = aggregate(sc, w).assign(model=m)
                rows.append(lb)
    board = pd.concat(rows, ignore_index=True)
    Path("reports").mkdir(exist_ok=True)
    board.to_csv(f"reports/leaderboard_baselines{'' if SPLIT == 'test' else '_val'}.csv", index=False)
    piv = board[board.lead.isin([1, 7, 30])].pivot_table(index=["track", "var", "lead"], columns="model", values="MASE").round(3)
    print("\nMASE (lower is better):\n", piv.to_string())


if __name__ == "__main__":
    main()
