"""Run statistical (AutoETS, AutoTheta[, AutoARIMA]) and LightGBM baselines on the daily window index.

Usage: uv run python scripts/run_stat_lgbm.py [--models AutoETS,AutoTheta,lightgbm] [--tracks bmd,...]
Resumable: cached prediction files are skipped. Requires data/processed/windows_daily.parquet (run_baselines.py).

`--gap-matched` runs the Week-5 control for the equity test: temperate contexts carrying GHCN-Bangladesh
missingness, with the same donor masks and deterministic assignment the foundation models received (cache tag
`_gapmatched`, ghcn_temperate only). LightGBM is excluded - it builds features from the series rather than from an
explicit context, and it is never the best trained baseline on the GHCN tracks, so it is never the control.
"""
import argparse
import time

import pandas as pd
import yaml

from bwb.data.store import NON_NEGATIVE, TRACK_VARIABLES, load_track
from bwb.eval.gap_matching import assign_donors, donor_masks
from bwb.eval.harness import cache_path
from bwb.models.lgbm_baseline import GlobalLGBM
from bwb.models.stat_baselines import run_stat_models

CFG = yaml.safe_load(open("configs/splits.yaml"))
TRAIN_END = CFG["daily"]["train"]["end"]
VAL = (CFG["daily"]["val"]["start"], CFG["daily"]["val"]["end"])
H = max(CFG["rolling_origin"]["daily"]["horizons_days"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="AutoETS,AutoTheta,lightgbm")
    ap.add_argument("--tracks", default=",".join(TRACK_VARIABLES))
    ap.add_argument("--context", type=int, default=730)
    ap.add_argument("--chunk", type=int, default=1500, help="windows per StatsForecast call (bounds memory)")
    ap.add_argument("--n-jobs", type=int, default=4, help="StatsForecast worker processes (8 GB RAM: keep <= 4)")
    ap.add_argument("--gap-matched", action="store_true",
                    help="equity-test control: impose GHCN-Bangladesh context missingness on ghcn_temperate contexts")
    args = ap.parse_args()
    models = args.models.split(",")
    stat_models = [m for m in models if m != "lightgbm"]
    windows = pd.read_parquet("data/processed/windows_daily.parquet")
    tracks = args.tracks.split(",")
    suffix, bd_series, bd_windows = "", None, None
    if args.gap_matched:
        tracks, models, stat_models, suffix = ["ghcn_temperate"], stat_models, stat_models, "_gapmatched"
        bd_series = load_track("ghcn_bangladesh")
        bd_windows = windows[windows.track == "ghcn_bangladesh"]

    for track in tracks:
        series = load_track(track)
        for var in TRACK_VARIABLES[track]:
            w = windows[(windows.track == track) & (windows["var"] == var)]
            sub = {k: v for k, v in series.items() if k[0] == var}
            nonneg = var in NON_NEGATIVE
            todo = [m for m in stat_models if not cache_path(m + suffix, track, var).exists()]
            masks = None
            if args.gap_matched and todo:
                donors = donor_masks(bd_series, bd_windows, var, length=args.context)
                masks = dict(zip(w.window_id.to_numpy(), donors[assign_donors(w.window_id.to_numpy(), len(donors), seed=0)]))
            if todo:
                t0 = time.time()
                chunks = [run_stat_models(sub, w.iloc[i:i + args.chunk], TRAIN_END, H, tuple(todo), context_length=args.context,
                                          non_negative=nonneg, n_jobs=args.n_jobs, window_masks=masks) for i in range(0, len(w), args.chunk)]
                out = {m: pd.concat([c[m] for c in chunks], ignore_index=True) for m in todo}
                for m, p in out.items():
                    path = cache_path(m + suffix, track, var)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    p.to_parquet(path, index=False)
                print(f"{','.join(todo):22s} {track:28s} {var:18s} {len(w):6d} windows {time.time() - t0:7.1f}s", flush=True)
            if "lightgbm" in models and not args.gap_matched and not cache_path("lightgbm", track, var).exists():
                t0 = time.time()
                m = GlobalLGBM(horizon=H, non_negative=nonneg).fit(sub, TRAIN_END, *VAL)
                p = m.predict_windows(sub, w)
                path = cache_path("lightgbm", track, var)
                path.parent.mkdir(parents=True, exist_ok=True)
                p.to_parquet(path, index=False)
                print(f"{'lightgbm':22s} {track:28s} {var:18s} {len(w):6d} windows {time.time() - t0:7.1f}s "
                      f"(best_iter={m.booster_.best_iteration})", flush=True)


if __name__ == "__main__":
    main()
