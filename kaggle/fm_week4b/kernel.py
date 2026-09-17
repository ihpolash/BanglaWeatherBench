"""Kaggle GPU runner for the remaining Week-4 foundation-model work (notebook ipolas/bwb-fm-week4b).

Same environment hardening and torch pinning as kaggle/fm_kernel (see reports/week4_fm_report.md, "Operational log").
Per model, in its own virtual environment:
1. profile  - first 4,096 windows of each BMD variable at context 1,024: peak GPU memory and parameter count (no cache write);
2. dekadal  - CHIRPS dekadal rainfall at contexts 36/72/144/1,024 dekads (forecasts are small and shipped);
3. ctx<N>   - daily context-length ablation (Chronos-2, TiRex, TimesFM 2.5) on all 16 tasks, then score_ctx<N>: the run is
              verified and scored here (scripts/score_fm_runs.py) and only scores are shipped. Full daily forecasts are
              ~250 MB per run, and downloading the main run's 2.2 GB was the slowest step of Week 4.
Outputs in /kaggle/working: week4b_results.json, records/*.jsonl, dekadal_cache/<name>/chirps__rfh.parquet,
ablation/board_<name>.csv, ablation/scores/<name>/<track>__<var>.parquet.
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

WORK = Path("/kaggle/working")
ROOT = Path("/tmp/bwb")
ENVS = Path("/tmp/envs")
INSTALL_TIMEOUT = 25 * 60
RUN_TIMEOUT = 5 * 60 * 60

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
DAILY_CONTEXTS = {"chronos2": [96, 336, 512, 2048, 4096], "tirex": [96, 336, 512, 2048], "timesfm25": [96, 336, 512, 2048]}
DEKADAL_CONTEXTS = "36,72,144,1024"
# Long ablations run last on each GPU, so a per-job time limit can only cut the longest context.
GROUPS = [["ttm_r2", "chronos_bolt", "toto2", "tirex", "timesfm25"], ["moirai2", "sundial", "chronos2"]]
# Re-run selection: {model: [job labels]} restricts the notebook to those jobs (used to redo one failed job after a fix);
# {} runs everything. v3 re-ran only Toto-2.0's dekadal job, which v2 failed on a patch-size mismatch.
ONLY = {}
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
        mem = " ".join(l.split()[1] for l in open("/proc/meminfo").read().split("\n")[:3])
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
    """Kaggle unpacks zip files uploaded to a dataset, so the bundle usually arrives as extracted files (as in kaggle/fm_kernel)."""
    scripts = glob.glob("/kaggle/input/**/scripts/run_fm_zero_shot.py", recursive=True)
    if scripts:
        shutil.copytree(Path(scripts[0]).parents[1], ROOT, dirs_exist_ok=True)
        return
    zips = glob.glob("/kaggle/input/**/bwb_bundle.zip", recursive=True)
    if not zips:
        raise FileNotFoundError("bundle not found under /kaggle/input")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(ROOT)


def jobs(model):
    daily = ["scripts/run_fm_zero_shot.py", "--model", model, "--device", "cuda"]
    out = [("profile", daily + ["--tracks", "bmd", "--context", "1024", "--limit-windows", "4096", "--tag", "_profile"]),
           ("dekadal", ["scripts/run_fm_dekadal.py", "--model", model, "--contexts", DEKADAL_CONTEXTS, "--device", "cuda"])]
    for c in DAILY_CONTEXTS.get(model, []):
        out.append((f"ctx{c}", daily + ["--context", str(c), "--tag", f"_ctx{c}"]))
        out.append((f"score_ctx{c}", ["scripts/score_fm_runs.py", "--models", f"{model}_ctx{c}", "--out", str(WORK / "ablation")]))
    return out


def run_model(gpu, model):
    tag = f"gpu{gpu}:{model}"
    env_dir = ENVS / model
    py = env_dir / "bin" / "python"
    resources(tag)
    code = stream(tag, ["uv", "venv", "--system-site-packages", "--python", sys.executable, str(env_dir)], 300)
    if code == 0:
        code = stream(tag, ["uv", "pip", "install", "--python", str(py), "--override", str(OVERRIDES), *COMMON, *PACKAGES[model]], INSTALL_TIMEOUT)
    resources(tag)
    if code != 0:
        return {"stage": "install", "exit": code}
    check = ("import torch, torchvision; print('torch in env:', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.cuda.is_available()); "
             "_ = torch.ops.torchvision.nms; print('torchvision ops OK'); "
             f"import sys; sys.exit(0 if torch.__version__.split('+')[0] == '{SYSTEM_TORCH.get('torch', '')}' else 3)")
    if stream(tag, [str(py), "-c", check], 120) != 0:
        return {"stage": "torch_mismatch", "exit": 3}
    env = dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu))
    exits = {}
    for label, args in jobs(model):
        if ONLY.get(model) and label not in ONLY[model]:
            continue
        if label.startswith("score_") and exits.get(label[len("score_"):]) != 0:
            exits[label] = "skipped"
            continue
        exits[label] = stream(f"{tag}:{label}", [str(py), "-W", "ignore", *args], RUN_TIMEOUT, cwd=ROOT, env=env)
        if label.startswith("score_"):  # scores are shipped; the ~250 MB of daily forecasts are not
            shutil.rmtree(ROOT / "data" / "predictions_cache" / f"{model}_{label[len('score_'):]}", ignore_errors=True)
    return {"stage": "run", "exit": 0 if all(v in (0, "skipped") for v in exits.values()) else 1, "jobs": exits}


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
        n_gpu = torch.cuda.device_count()
        say("setup", f"torch overrides {SYSTEM_TORCH} | system torch {torch.__version__} gpus={n_gpu} "
            + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)))
        (WORK / "ablation").mkdir(parents=True, exist_ok=True)
        groups = GROUPS if n_gpu >= 2 else [sum(GROUPS, [])]
        groups = [[m for m in g if not ONLY or m in ONLY] for g in groups]
        threads = [threading.Thread(target=run_group, args=(gpu, models, results)) for gpu, models in enumerate(groups)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except Exception as e:
        results["_kernel"] = {"stage": "setup", "exit": -1, "error": f"{type(e).__name__}: {e}"}
        say("setup", f"EXCEPTION {type(e).__name__}: {e}")
    finally:
        (WORK / "week4b_results.json").write_text(json.dumps(results, indent=2))
        (WORK / "records").mkdir(exist_ok=True)
        for rec in [*ROOT.glob("reports/fm_run_*.jsonl"), *ROOT.glob("reports/fm_dekadal_*.jsonl")] if ROOT.exists() else []:
            shutil.copy(rec, WORK / "records" / rec.name)
        for f in ROOT.glob("data/predictions_cache/*/chirps__rfh.parquet") if ROOT.exists() else []:
            dest = WORK / "dekadal_cache" / f.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest / f.name)
        say("setup", f"summary: { {m: (r.get('stage'), r.get('exit'), r.get('jobs')) for m, r in results.items()} }")


if __name__ == "__main__":
    main()
