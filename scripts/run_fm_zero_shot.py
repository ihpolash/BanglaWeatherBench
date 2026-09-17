"""Zero-shot foundation-model forecasts over the daily window index, written to the shared prediction cache.

Usage (inside a model-specific environment, e.g. on Kaggle):
    PYTHONPATH=src python scripts/run_fm_zero_shot.py --model chronos2 [--tracks bmd,...] [--context 1024] [--device cuda]

Contexts end exactly at each window origin (same slicing as the Week-3 harness); nothing after the origin is passed.
Output: data/predictions_cache/<model>/<track>__<var>.parquet with window_id, lead, mean, q0.1..q0.9 (resumable).
Run records (reports/fm_run_<name>.jsonl) carry wall time, batch size, peak GPU memory and parameter count per task.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bwb.data.store import NON_NEGATIVE, TRACK_VARIABLES, load_track
from bwb.eval.harness import QCOLS, cache_path
from bwb.models.fm_base import count_parameters
from bwb.models.fm_registry import REGISTRY, PeakMemory, load_adapter

CFG = yaml.safe_load(open("configs/splits.yaml"))
H = max(CFG["rolling_origin"]["daily"]["horizons_days"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--tracks", default=",".join(TRACK_VARIABLES))
    ap.add_argument("--context", type=int, default=1024, help="context length in days (capped at the model's maximum)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit-windows", type=int, default=0, help="smoke test / profiling: only the first N windows per task, no cache write")
    ap.add_argument("--tag", default="", help="suffix for the cache model name, e.g. _ctx512 for ablations")
    ap.add_argument("--gap-matched", action="store_true",
                    help="ablation: impose GHCN-Bangladesh context missingness on ghcn_temperate contexts (cache tag _gapmatched)")
    args = ap.parse_args()

    adapter = load_adapter(args.model, args.device, args.batch_size, context=args.context)
    peak = PeakMemory()
    name = args.model + args.tag + ("_gapmatched" if args.gap_matched else "")
    windows = pd.read_parquet("data/processed/windows_daily.parquet")
    tracks = args.tracks.split(",")
    if args.gap_matched:
        from bwb.eval.gap_matching import apply_mask, assign_donors, donor_masks

        tracks = ["ghcn_temperate"]
        bd_series = load_track("ghcn_bangladesh")
        bd_windows = windows[windows.track == "ghcn_bangladesh"]
    log = []
    for track in tracks:
        series = load_track(track)
        for var in TRACK_VARIABLES[track]:
            path = cache_path(name, track, var)
            if path.exists() and not args.limit_windows:
                continue
            w = windows[(windows.track == track) & (windows["var"] == var)]
            if args.limit_windows:
                w = w.head(args.limit_windows)
            t0 = time.time()
            peak.reset()
            masks = donors = None
            if args.gap_matched:
                masks = donor_masks(bd_series, bd_windows, var, length=min(args.context, adapter.max_context))
                donors = assign_donors(w.window_id.to_numpy(), len(masks), seed=0)
            contexts = []
            for i, (sid, origin) in enumerate(zip(w.series_id, w.origin)):
                s = series[(var, sid)]
                ctx = s.loc[:origin].to_numpy(dtype=np.float32)
                if masks is not None:
                    ctx = apply_mask(ctx, masks[donors[i]])
                contexts.append(adapter.prepare(ctx, args.context))
            mean, q = adapter.predict_batch(contexts, H)
            if var in NON_NEGATIVE:
                mean, q = np.maximum(mean, 0), np.maximum(q, 0)
            out = pd.DataFrame(q.transpose(0, 2, 1).reshape(-1, len(QCOLS)), columns=QCOLS)
            out.insert(0, "mean", mean.reshape(-1))
            out.insert(0, "lead", np.tile(np.arange(1, H + 1), len(w)))
            out.insert(0, "window_id", np.repeat(w.window_id.to_numpy(), H))
            if not args.limit_windows:
                path.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(path, index=False)
            secs = time.time() - t0
            log.append({"model": name, "track": track, "var": var, "windows": len(w), "seconds": round(secs, 1),
                        "context": min(args.context, adapter.max_context), "nan_policy": adapter.nan_policy,
                        "batch_size": adapter.batch_size, "peak_vram_mb": peak.read(), "gpu": peak.device})
            print(f"{name:14s} {track:28s} {var:18s} {len(w):6d} windows {secs:7.1f}s peak {peak.read()} MiB", flush=True)
    params = count_parameters(adapter)  # after the runs: some adapters build their networks lazily
    Path("reports").mkdir(exist_ok=True)
    with open(f"reports/fm_run_{name}.jsonl", "a") as f:
        for r in log:
            f.write(json.dumps({**r, "params": params}) + "\n")
    print(f"{name}: {params:,} parameters", flush=True)


if __name__ == "__main__":
    main()
