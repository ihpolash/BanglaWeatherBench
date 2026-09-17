"""Smoke test: Chronos-2 zero-shot on one BMD station -> prediction cache parquet.
Run inside envs/chronos2. Same script runs on Kaggle/Colab with device='cuda'."""
import sys, time, pathlib
import pandas as pd
import torch
from chronos import BaseChronosPipeline

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
df = pd.read_csv("data/external/bmd_mendeley/stations/Dhaka.csv")
df["timestamp"] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day))
ctx = df[(df.timestamp < "2016-01-01")].tail(2048)[["timestamp", "Temperature"]].rename(columns={"Temperature": "target"})
ctx["id"] = "Dhaka"
t0 = time.time()
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)
load_s = time.time() - t0
t0 = time.time()
pred = pipe.predict_df(ctx, prediction_length=30, quantile_levels=[0.1, 0.5, 0.9], id_column="id", timestamp_column="timestamp", target="target")
infer_s = time.time() - t0
out = pathlib.Path("data/predictions_cache/chronos2/smoke_Dhaka_Temperature_h30.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
pred.to_parquet(out)
print(f"load {load_s:.1f}s infer {infer_s:.2f}s -> {out}")
print(pred.head(3).to_string())
