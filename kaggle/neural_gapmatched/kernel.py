"""Week-5 equity control: the trained neural baselines re-run on gap-matched temperate contexts.

The Week-4 equity test compares foundation models on GHCN-Bangladesh (82-86% observed contexts) with GHCN-temperate
(99-100%), using trained baselines as the control. The gap-matched ablation so far exists only for the foundation
models, so the control is not like-for-like. This notebook closes that: NHITS, DLinear and PatchTST are trained
exactly as in Week 3 and then forecast temperate windows whose contexts carry Bangladesh missingness, drawn from the
same donor masks with the same deterministic assignment (seed 0) the foundation models received.

Two tasks only (ghcn_temperate Rainfall and Tavg), so this is a short run. Outputs in /kaggle/working:
predictions_cache/{NHITS,DLinear,PatchTST}_gapmatched/ghcn_temperate__*.parquet plus the run log.
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


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def locate_bundle():
    """Kaggle unpacks zips uploaded to a dataset, so the bundle usually arrives as extracted files."""
    scripts = glob.glob("/kaggle/input/**/scripts/run_neural_baselines.py", recursive=True)
    if scripts:
        shutil.copytree(Path(scripts[0]).parents[1], ROOT, dirs_exist_ok=True)
        return
    zips = glob.glob("/kaggle/input/**/bwb_bundle.zip", recursive=True)
    if not zips:
        raise FileNotFoundError("bundle not found under /kaggle/input")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(ROOT)


def main():
    sh([sys.executable, "-m", "pip", "install", "-q", "neuralforecast==3.2.2"])
    locate_bundle()
    import torch

    n_gpu = torch.cuda.device_count()
    print(f"torch {torch.__version__}, gpus={n_gpu}", flush=True)
    env = dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-W", "ignore", "scripts/run_neural_baselines.py", "--models", "NHITS,DLinear,PatchTST",
           "--gap-matched", "--max-steps", "1000", "--accelerator", "gpu" if n_gpu else "cpu"]
    log = open(WORK / "neural_gapmatched.log", "w")
    code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    log.close()
    print((WORK / "neural_gapmatched.log").read_text()[-4000:], flush=True)
    src = ROOT / "data" / "predictions_cache"
    if src.exists():
        for d in src.glob("*_gapmatched"):
            shutil.copytree(d, WORK / "predictions_cache" / d.name, dirs_exist_ok=True)
    print("exit", code, "| files:", sorted(p.name for p in (WORK / "predictions_cache").rglob("*.parquet")), flush=True)
    shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
