"""Deep-learning baselines (NHITS, DLinear, PatchTST) trained on the training split, run over the test windows.

Runs in the isolated environment:
    PYTHONPATH=src uv run --project envs/neural python scripts/run_neural_baselines.py [--models NHITS,DLinear,PatchTST]

One global model per (track, variable, architecture). Series are modelled as anomalies from their training
climatology, scaled by the training anomaly standard deviation. Gaps are zero-filled and excluded from the loss
through neuralforecast's `available_mask`. `--gap-matched` is the Week-5 equity control: temperate contexts carry
GHCN-Bangladesh missingness from the same donor masks the foundation models received (cache tag `_gapmatched`). Output quantiles come from a multi-quantile loss (0.1..0.9).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bwb.data.store import NON_NEGATIVE, TRACK_VARIABLES, load_track
from bwb.eval.gap_matching import apply_mask, assign_donors, donor_masks
from bwb.eval.harness import cache_path
from bwb.models.baselines import QUANTILES, Climatology, doy_noleap

CFG = yaml.safe_load(open("configs/splits.yaml"))
TRAIN_END = CFG["daily"]["train"]["end"]
H = max(CFG["rolling_origin"]["daily"]["horizons_days"])
INPUT = 365
QCOLS = [f"q{q:g}" for q in QUANTILES]


def build_models(names, max_steps, accelerator):
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import DLinear, NHITS, PatchTST

    loss = MQLoss(quantiles=list(QUANTILES))
    common = dict(h=H, input_size=INPUT, loss=loss, max_steps=max_steps, batch_size=32, windows_batch_size=256,
                  scaler_type="identity", random_seed=0, accelerator=accelerator, devices=1, enable_progress_bar=False,
                  logger=False, enable_checkpointing=False)
    make = {
        "NHITS": lambda: NHITS(**common),
        "DLinear": lambda: DLinear(**common),
        "PatchTST": lambda: PatchTST(**{**common, "windows_batch_size": 128}, patch_len=16, stride=8),
    }
    return [make[n]() for n in names]


def normalised(series: dict):
    """{key: (norm anomaly Series, clim, std)} using training-period statistics only."""
    out = {}
    for key, s in series.items():
        clim = Climatology().fit(s.loc[:TRAIN_END])
        anom = s - clim.mean_[doy_noleap(s.index)]
        sd = float(anom.loc[:TRAIN_END].std()) or 1.0
        out[key] = (anom / sd, clim, sd)
    return out


def to_long(uid, s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": uid, "ds": s.index, "y": s.fillna(0.0).to_numpy(),
                         "available_mask": s.notna().astype(float).to_numpy()})


def quantile_columns(fc: pd.DataFrame, name: str) -> list[str]:
    cols = [c for c in fc.columns if c.startswith(f"{name}-")]
    def level(c):
        tag = c.removeprefix(f"{name}-")
        if tag == "median":
            return 0.5
        kind, pct = tag.split("-")
        p = float(pct) / 100
        return 0.5 - p / 2 if kind == "lo" else 0.5 + p / 2
    cols = sorted(cols, key=level)
    if len(cols) != len(QUANTILES):
        raise ValueError(f"expected {len(QUANTILES)} quantile columns for {name}, got {cols}")
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="NHITS,DLinear,PatchTST")
    ap.add_argument("--tracks", default=",".join(TRACK_VARIABLES))
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--gap-matched", action="store_true",
                    help="equity-test control: impose GHCN-Bangladesh context missingness on ghcn_temperate contexts")
    args = ap.parse_args()
    from neuralforecast import NeuralForecast

    names = args.models.split(",")
    windows = pd.read_parquet("data/processed/windows_daily.parquet")
    tracks = args.tracks.split(",")
    suffix, bd_series, bd_windows = "", None, None
    if args.gap_matched:
        tracks, suffix = ["ghcn_temperate"], "_gapmatched"
        bd_series = load_track("ghcn_bangladesh")
        bd_windows = windows[windows.track == "ghcn_bangladesh"]
    for track in tracks:
        series = load_track(track)
        for var in TRACK_VARIABLES[track]:
            todo = [n for n in names if not cache_path(n + suffix, track, var).exists()]
            if not todo:
                continue
            t0 = time.time()
            sub = normalised({k: v for k, v in series.items() if k[0] == var})
            train = pd.concat([to_long(f"{k[1]}", a.loc[:TRAIN_END].dropna().pipe(lambda x: a.loc[x.index.min():TRAIN_END]))
                               for k, (a, _, _) in sub.items() if a.loc[:TRAIN_END].notna().sum() > INPUT + H], ignore_index=True)
            nf = NeuralForecast(models=build_models(todo, args.max_steps, args.accelerator), freq="D")
            nf.fit(df=train)
            w = windows[(windows.track == track) & (windows["var"] == var)]
            masks = None
            if args.gap_matched:
                donors = donor_masks(bd_series, bd_windows, var, length=INPUT)
                masks = dict(zip(w.window_id.to_numpy(), donors[assign_donors(w.window_id.to_numpy(), len(donors), seed=0)]))
            ctx = []
            for wid, sid, origin in zip(w.window_id, w.series_id, w.origin):
                a = sub[(var, sid)][0]
                window = a.loc[:origin].iloc[-INPUT:].reindex(pd.date_range(origin - pd.Timedelta(days=INPUT - 1), origin, freq="D"))
                if masks is not None:
                    window = pd.Series(apply_mask(window.to_numpy(dtype=np.float32), masks[wid]), index=window.index)
                ctx.append(to_long(wid, window))
            fc = nf.predict(df=pd.concat(ctx, ignore_index=True)).reset_index()
            fc = fc.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            fc["lead"] = fc.groupby("unique_id").cumcount() + 1
            meta = w.set_index("window_id")
            clim_t = np.concatenate([sub[(var, meta.loc[u, "series_id"])][1].predict(None, meta.loc[u, "origin"], H)[0]
                                     for u in fc.unique_id.drop_duplicates()])
            sd_t = np.repeat([sub[(var, meta.loc[u, "series_id"])][2] for u in fc.unique_id.drop_duplicates()], H)
            for n in todo:
                qc = quantile_columns(fc, n)
                q = np.sort(fc[qc].to_numpy() * sd_t[:, None] + clim_t[:, None], axis=1)
                mean = q[:, 4].copy()
                if var in NON_NEGATIVE:
                    q, mean = np.maximum(q, 0), np.maximum(mean, 0)
                p = pd.DataFrame(q, columns=QCOLS)
                p.insert(0, "mean", mean)
                p.insert(0, "lead", fc["lead"].to_numpy())
                p.insert(0, "window_id", fc["unique_id"].astype(int).to_numpy())
                path = cache_path(n + suffix, track, var)
                path.parent.mkdir(parents=True, exist_ok=True)
                p.to_parquet(path, index=False)
            print(f"{','.join(todo):22s} {track:28s} {var:18s} {len(w):6d} windows {time.time() - t0:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
