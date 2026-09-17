"""Zero-shot foundation-model forecasts for the CHIRPS dekadal rainfall task (64 districts, 6-dekad horizon).

Usage (inside a model-specific environment): PYTHONPATH=src python scripts/run_fm_dekadal.py --model chronos2 \
    [--contexts 36,72,144,1024] [--device cuda]

Windows and step positions come from data/processed/windows_dekadal.parquet (scripts/run_dekadal_baselines.py); a
context is every dekad up to and including the origin, capped at the context length. Nothing after the origin is passed.
Context lengths 36/72/144 dekads are pre-registered in configs/splits.yaml; the longest (144) is the headline and is
cached as <model>, the others as <model>_ctx<N>. 1,024 dekads (all history, ~28 years) is an extra full-history run.
TTM has no 10-day resolution; it receives freq="W", the nearest supported resolution of at least one day.
Output: data/predictions_cache/<name>/chirps__rfh.parquet; run records reports/fm_dekadal_<model>.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bwb.eval.dekadal import QCOLS
from bwb.eval.harness import CACHE
from bwb.models.fm_base import count_parameters
from bwb.models.fm_registry import REGISTRY, PeakMemory, load_adapter

CFG = yaml.safe_load(open("configs/splits.yaml"))
RO = CFG["rolling_origin"]["dekadal"]
H = max(RO["horizons_dekads"])
HEADLINE = max(RO["context_lengths"])


def cache_name(model: str, context: int) -> str:
    return model if context == HEADLINE else f"{model}_ctx{context}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--contexts", default="36,72,144,1024")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit-windows", type=int, default=0, help="smoke test: first N windows, no cache write")
    args = ap.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]
    adapter = load_adapter(args.model, args.device, args.batch_size, context=max(contexts), freq="W")
    peak = PeakMemory()
    df = pd.read_parquet("data/processed/chirps_dekadal_adm2.parquet")
    series = {pc: g.set_index("date").rfh.sort_index().to_numpy(dtype=np.float32) for pc, g in df.groupby("PCODE")}
    windows = pd.read_parquet("data/processed/windows_dekadal.parquet")
    if args.limit_windows:
        windows = windows.head(args.limit_windows)
    log = []
    for context in contexts:
        name = cache_name(args.model, context)
        path = CACHE / name / "chirps__rfh.parquet"
        if path.exists() and not args.limit_windows:
            continue
        t0 = time.time()
        peak.reset()
        ctxs = [adapter.prepare(series[sid][:pos + 1], context) for sid, pos in zip(windows.series_id, windows.origin_pos)]
        mean, q = adapter.predict_batch(ctxs, H)
        mean, q = np.maximum(mean, 0), np.maximum(q, 0)
        out = pd.DataFrame(q.transpose(0, 2, 1).reshape(-1, len(QCOLS)), columns=QCOLS)
        out.insert(0, "mean", mean.reshape(-1))
        out.insert(0, "lead", np.tile(np.arange(1, H + 1), len(windows)))
        out.insert(0, "window_id", np.repeat(windows.window_id.to_numpy(), H))
        if not args.limit_windows:
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path, index=False)
        secs = time.time() - t0
        log.append({"model": name, "track": "chirps", "var": "rfh", "windows": len(windows), "seconds": round(secs, 1),
                    "context": min(context, adapter.max_context), "nan_policy": adapter.nan_policy,
                    "batch_size": adapter.batch_size, "peak_vram_mb": peak.read(), "gpu": peak.device})
        print(f"{name:18s} chirps rfh {len(windows):6d} windows {secs:7.1f}s peak {peak.read()} MiB", flush=True)
    params = count_parameters(adapter)
    Path("reports").mkdir(exist_ok=True)
    with open(f"reports/fm_dekadal_{args.model}.jsonl", "a") as f:
        for r in log:
            f.write(json.dumps({**r, "params": params}) + "\n")


if __name__ == "__main__":
    main()
