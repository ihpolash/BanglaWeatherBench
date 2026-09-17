"""Week-5: gap-matched temperate runs at a 2,048-day context, to separate two explanations of the tropical penalty.

Week 4 found (R3) a long-lead temperature deficit on Bangladesh stations, and (R7) that the same deficit nearly
vanishes when models are given 2,048 days of context instead of 1,024. The gap-matched ablation so far exists only at
1,024, so "short context" and "gappy context" are still confounded: at 1,024 a Bangladesh context is both shorter in
effective observations and gappier.

This notebook runs Chronos-2, TiRex and TimesFM 2.5 on ghcn_temperate at context 2,048 with GHCN-Bangladesh
missingness imposed (same donor masks, seed 0). Combined with the existing 2,048 unmasked runs, it gives a DiD at
2,048 with and without gap matching. Forecasts stay in /tmp; only the scores are shipped (scripts/score_fm_runs.py),
as in kaggle/fm_week4b.
Outputs: /kaggle/working/gm2048_results.json, records/*.jsonl, ablation/board_*.csv, ablation/scores/**.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

WORK, ROOT, ENVS = Path("/kaggle/working"), Path("/tmp/bwb"), Path("/tmp/envs")
INSTALL_TIMEOUT, RUN_TIMEOUT, CONTEXT = 25 * 60, 3 * 60 * 60, 2048
PACKAGES = {"chronos2": ["chronos-forecasting>=2.0"], "tirex": ["tirex-ts"],
            "timesfm25": ["timesfm[torch] @ git+https://github.com/google-research/timesfm.git"]}
GROUPS = [["chronos2", "timesfm25"], ["tirex"]]
COMMON = ["pandas", "pyarrow", "pyyaml", "numpy"]
LOCK, OVERRIDES, SYSTEM_TORCH = threading.Lock(), Path("/tmp/torch_overrides.txt"), {}


def say(tag, msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def stream(tag, cmd, timeout, **kw):
    say(tag, "+ " + " ".join(map(str, cmd)))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    deadline = time.time() + timeout
    for line in proc.stdout:
        say(tag, line.rstrip()[:400])
        if time.time() > deadline:
            proc.kill()
            say(tag, f"TIMEOUT after {timeout}s")
            return 124
    return proc.wait()


def locate_bundle():
    scripts = glob.glob("/kaggle/input/**/scripts/run_fm_zero_shot.py", recursive=True)
    if scripts:
        shutil.copytree(Path(scripts[0]).parents[1], ROOT, dirs_exist_ok=True)
        return
    zips = glob.glob("/kaggle/input/**/bwb_bundle.zip", recursive=True)
    if not zips:
        raise FileNotFoundError("bundle not found under /kaggle/input")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(ROOT)


def run_model(gpu, model):
    tag, env_dir = f"gpu{gpu}:{model}", ENVS / model
    py = env_dir / "bin" / "python"
    code = stream(tag, ["uv", "venv", "--system-site-packages", "--python", sys.executable, str(env_dir)], 300)
    if code == 0:
        code = stream(tag, ["uv", "pip", "install", "--python", str(py), "--override", str(OVERRIDES), *COMMON, *PACKAGES[model]], INSTALL_TIMEOUT)
    if code != 0:
        return {"stage": "install", "exit": code}
    env = dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu))
    name = f"{model}_ctx{CONTEXT}_gapmatched"
    code = stream(tag, [str(py), "-W", "ignore", "scripts/run_fm_zero_shot.py", "--model", model, "--device", "cuda",
                        "--context", str(CONTEXT), "--tag", f"_ctx{CONTEXT}", "--gap-matched"], RUN_TIMEOUT, cwd=ROOT, env=env)
    if code != 0:
        return {"stage": "run", "exit": code}
    code = stream(tag, [str(py), "-W", "ignore", "scripts/score_fm_runs.py", "--models", name,
                        "--tracks", "ghcn_temperate", "--out", str(WORK / "ablation")], RUN_TIMEOUT, cwd=ROOT, env=env)
    shutil.rmtree(ROOT / "data" / "predictions_cache" / name, ignore_errors=True)
    return {"stage": "score", "exit": code}


def run_group(gpu, models, results):
    for model in models:
        try:
            results[model] = {"gpu": gpu, **run_model(gpu, model)}
        except Exception as e:
            results[model] = {"gpu": gpu, "stage": "exception", "exit": -1, "error": f"{type(e).__name__}: {e}"}
        finally:
            shutil.rmtree(ENVS / model, ignore_errors=True)
        say(f"gpu{gpu}:{model}", f"RESULT {results[model]}")


def main():
    results = {}
    try:
        stream("setup", [sys.executable, "-m", "pip", "install", "-q", "uv"], 600)
        locate_bundle()
        import importlib.metadata as md

        import torch

        for pkg in ("torch", "torchvision", "torchaudio"):
            try:
                SYSTEM_TORCH[pkg] = md.version(pkg).split("+")[0]
            except md.PackageNotFoundError:
                pass
        OVERRIDES.write_text("".join(f"{k}=={v}\n" for k, v in SYSTEM_TORCH.items()))
        n_gpu = torch.cuda.device_count()
        say("setup", f"torch {torch.__version__} gpus={n_gpu} overrides={SYSTEM_TORCH}")
        (WORK / "ablation").mkdir(parents=True, exist_ok=True)
        groups = GROUPS if n_gpu >= 2 else [sum(GROUPS, [])]
        threads = [threading.Thread(target=run_group, args=(g, models, results)) for g, models in enumerate(groups)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except Exception as e:
        results["_kernel"] = {"stage": "setup", "exit": -1, "error": f"{type(e).__name__}: {e}"}
        say("setup", f"EXCEPTION {type(e).__name__}: {e}")
    finally:
        (WORK / "gm2048_results.json").write_text(json.dumps(results, indent=2))
        (WORK / "records").mkdir(exist_ok=True)
        for rec in ROOT.glob("reports/fm_run_*.jsonl") if ROOT.exists() else []:
            shutil.copy(rec, WORK / "records" / rec.name)
        say("setup", f"summary: { {m: (r.get('stage'), r.get('exit')) for m, r in results.items()} }")


if __name__ == "__main__":
    main()
