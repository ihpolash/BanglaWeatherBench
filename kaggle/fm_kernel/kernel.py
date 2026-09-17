"""Kaggle GPU runner for Week-4 zero-shot foundation models (hardened after the first smoke attempt died silently).

Each model gets its own virtual environment layered on Kaggle's system packages, so conflicting pins (e.g. Sundial's
transformers==4.40.1) never meet. Two model groups run in parallel, one per GPU.

Hardening (smoke v1 ended in ERROR with no logs and a half-built venv in the output):
- every install/run line is echoed to the notebook's own stdout (Kaggle's log), prefixed by model, so output survives
  even if files in /kaggle/working are lost;
- virtual environments live in /tmp (never exported, never counted as output) and are deleted after each model;
- disk and memory are printed before and after every install;
- each model has a time limit; a results file is written in a finally-block no matter what fails.
MODE="smoke" runs every model on a few windows of one track; MODE="full" runs all 16 daily tasks plus the
gap-matched temperate ablation.
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

MODE = "full"  # "smoke" or "full"
CONTEXT = 1024
WORK = Path("/kaggle/working")
ROOT = Path("/tmp/bwb")
ENVS = Path("/tmp/envs")
INSTALL_TIMEOUT = 25 * 60
RUN_TIMEOUT = 30 * 60 if MODE == "smoke" else 5 * 60 * 60

PACKAGES = {
    "chronos2": ["chronos-forecasting>=2.0"],
    "chronos_bolt": ["chronos-forecasting>=2.0"],
    "timesfm25": ["timesfm[torch] @ git+https://github.com/google-research/timesfm.git"],
    "ttm_r2": ["granite-tsfm"],
    "toto2": ["toto-models"],
    "tirex": ["tirex-ts"],
    "moirai2": ["uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git"],
    "sundial": ["transformers==4.40.1"],
}
# TimesFM runs last on its GPU: if it is still slow, its per-model time limit stops only TimesFM (smoke v3 projection before
# the batch-size fix: ~11.6 h). Sundial (~2.4 h projected) runs last on the other GPU for the same reason.
GROUPS = [["chronos2", "chronos_bolt", "ttm_r2", "timesfm25"], ["toto2", "tirex", "moirai2", "sundial"]]
COMMON = ["pandas", "pyarrow", "pyyaml", "numpy"]
LOCK = threading.Lock()
OVERRIDES = Path("/tmp/torch_overrides.txt")
SYSTEM_TORCH = {}


def say(tag, msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def resources(tag):
    du = shutil.disk_usage("/kaggle/working")
    tmp = shutil.disk_usage("/tmp")
    try:
        mem = open("/proc/meminfo").read().split("\n")[:3]
        mem = " ".join(l.split()[1] for l in mem)
    except Exception:
        mem = "n/a"
    say(tag, f"disk /kaggle/working free {du.free / 1e9:.1f} GB | /tmp free {tmp.free / 1e9:.1f} GB | meminfo total/free/avail kB {mem}")


def stream(tag, cmd, timeout, **kw):
    """Run a command, echoing every output line to stdout with a tag; returns the exit code (124 on timeout)."""
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
    tag = f"gpu{gpu}:{model}"
    env_dir = ENVS / model
    py = env_dir / "bin" / "python"
    resources(tag)
    code = stream(tag, ["uv", "venv", "--system-site-packages", "--python", sys.executable, str(env_dir)], 300)
    if code == 0:
        # --override pins torch/torchvision/torchaudio to Kaggle's system builds, so no package can pull a torch whose
        # CUDA build mismatches the system torchvision (smoke v2: 5 of 8 models failed this way).
        code = stream(tag, ["uv", "pip", "install", "--python", str(py), "--override", str(OVERRIDES), *COMMON, *PACKAGES[model]], INSTALL_TIMEOUT)
    resources(tag)
    if code != 0:
        return {"stage": "install", "exit": code}
    # Version equality alone is not enough: a same-version torch re-downloaded into the venv can carry a different CUDA
    # build. Touching torch.ops.torchvision.nms raises in exactly the smoke-v2 failure state ("operator does not exist").
    check = ("import torch, torchvision; print('torch in env:', torch.__version__, torch.__file__, '| torchvision', torchvision.__version__, '| cuda', torch.cuda.is_available()); "
             "_ = torch.ops.torchvision.nms; print('torchvision ops OK'); "
             f"import sys; sys.exit(0 if torch.__version__.split('+')[0] == '{SYSTEM_TORCH.get('torch', '')}' else 3)")
    if stream(tag, [str(py), "-c", check], 120) != 0:
        return {"stage": "torch_mismatch", "exit": 3}
    env = dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [str(py), "-W", "ignore", "scripts/run_fm_zero_shot.py", "--model", model, "--context", str(CONTEXT), "--device", "cuda"]
    if MODE == "smoke":
        cmd += ["--tracks", "bmd", "--limit-windows", "256"]
    code = stream(tag, cmd, RUN_TIMEOUT, cwd=ROOT, env=env)
    if code == 0 and MODE == "full":
        code = stream(tag, cmd + ["--gap-matched"], RUN_TIMEOUT, cwd=ROOT, env=env)
    return {"stage": "run", "exit": code}


def run_group(gpu, models, results):
    for model in models:
        try:
            results[model] = {"gpu": gpu, **run_model(gpu, model)}
        except Exception as e:
            results[model] = {"gpu": gpu, "stage": "exception", "exit": -1, "error": f"{type(e).__name__}: {e}"}
            say(f"gpu{gpu}:{model}", f"EXCEPTION {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(ENVS / model, ignore_errors=True)
        say(f"gpu{gpu}:{model}", f"RESULT {results[model]}")


def main():
    results = {}
    try:
        resources("setup")
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
        say("setup", f"torch overrides: {SYSTEM_TORCH}")
        n_gpu = torch.cuda.device_count()
        say("setup", f"MODE={MODE} system torch {torch.__version__} gpus={n_gpu} "
            + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)))
        groups = GROUPS if n_gpu >= 2 else [sum(GROUPS, [])]
        threads = [threading.Thread(target=run_group, args=(gpu, models, results)) for gpu, models in enumerate(groups)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except Exception as e:
        results["_kernel"] = {"stage": "setup", "exit": -1, "error": f"{type(e).__name__}: {e}"}
        say("setup", f"EXCEPTION {type(e).__name__}: {e}")
    finally:
        (WORK / f"fm_{MODE}_results.json").write_text(json.dumps(results, indent=2))
        if (ROOT / "data" / "predictions_cache").exists():
            shutil.copytree(ROOT / "data" / "predictions_cache", WORK / "predictions_cache", dirs_exist_ok=True)
        for rec in (ROOT / "reports").glob("fm_run_*.jsonl") if ROOT.exists() else []:
            shutil.copy(rec, WORK / rec.name)
        say("setup", f"summary: { {m: (r.get('stage'), r.get('exit')) for m, r in results.items()} }")


if __name__ == "__main__":
    main()
