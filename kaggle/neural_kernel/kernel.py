"""Kaggle GPU runner for the BanglaWeatherBench deep-learning baselines (NHITS, DLinear, PatchTST).

Reads the private dataset bundle (code + processed parquet), trains/forecasts with the same script and settings as
the local run, splitting tracks across the available GPUs (one process per GPU), and leaves only the prediction
cache and logs in /kaggle/working for download.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
ROOT = WORK / "bwb"
GROUPS = [["bmd", "ghcn_bangladesh", "ghcn_temperate"],
          ["nasa_power_bmd", "nasa_power_ghcn_bangladesh", "nasa_power_ghcn_temperate"]]


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def locate_bundle():
    scripts = glob.glob("/kaggle/input/**/scripts/run_neural_baselines.py", recursive=True)
    if scripts:
        shutil.copytree(Path(scripts[0]).parents[1], ROOT, dirs_exist_ok=True)
        return
    zips = glob.glob("/kaggle/input/**/bwb_bundle.zip", recursive=True)
    if not zips:
        raise FileNotFoundError(f"bundle not found; /kaggle/input contains: {glob.glob('/kaggle/input/**', recursive=True)[:50]}")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(ROOT)


def main():
    sh([sys.executable, "-m", "pip", "install", "-q", "neuralforecast==3.2.2"])
    locate_bundle()
    import torch

    n_gpu = torch.cuda.device_count()
    print(f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}, gpus={n_gpu}", flush=True)
    for i in range(n_gpu):
        print(f"  gpu{i}: {torch.cuda.get_device_name(i)}", flush=True)
    groups = GROUPS if n_gpu >= 2 else [sum(GROUPS, [])]
    accel = "gpu" if n_gpu >= 1 else "cpu"

    procs = []
    for i, tracks in enumerate(groups):
        env = dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1")
        if n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        log = open(WORK / f"neural_gpu{i}.log", "w")
        cmd = [sys.executable, "-W", "ignore", "scripts/run_neural_baselines.py", "--tracks", ",".join(tracks),
               "--max-steps", "1000", "--accelerator", accel]
        print(f"launch group {i}: {tracks}", flush=True)
        procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log, i))
    codes = []
    for p, log, i in procs:
        codes.append(p.wait())
        log.close()
        print(f"group {i} exit {codes[-1]}", flush=True)
        print("".join(l for l in open(WORK / f"neural_gpu{i}.log") if " windows " in l or "Error" in l or "Traceback" in l), flush=True)

    out = WORK / "predictions_cache"
    shutil.copytree(ROOT / "data" / "predictions_cache", out, dirs_exist_ok=True)
    shutil.rmtree(ROOT)  # keep the kernel output small: predictions + logs only
    print("outputs:", sorted(str(p.relative_to(WORK)) for p in out.rglob("*.parquet")), flush=True)
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
