"""Kaggle CPU runner for the AutoARIMA baseline (Week-3 carry-over), one of two parts (PART below).

Identical code and settings to the local AutoETS/AutoTheta run: scripts/run_stat_lgbm.py on climatological anomalies,
730-day context, season_length=1, StatsForecast 2.0.1 (the local version). Local timing on 200 BMD windows: ~0.1 s per
window on 4 workers, ~5.5 h for all 170,571 windows, so the 16 tasks are split over two CPU notebooks (no GPU quota).
Resumable: finished tasks are shipped even if the notebook stops early.
Outputs in /kaggle/working: predictions_cache/AutoARIMA/<track>__<var>.parquet, arima_<PART>_results.json, run log.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PART = "b"
TRACKS = {"a": "bmd,ghcn_bangladesh,ghcn_temperate",
          "b": "nasa_power_bmd,nasa_power_ghcn_bangladesh,nasa_power_ghcn_temperate"}[PART]
WORK, ROOT = Path("/kaggle/working"), Path("/tmp/bwb")
TIMEOUT = int(11.3 * 3600)


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stream(cmd, timeout, **kw):
    say("+ " + " ".join(map(str, cmd)))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    deadline = time.time() + timeout
    for line in proc.stdout:
        say(line.rstrip()[:400])
        if time.time() > deadline:
            proc.kill()
            say(f"TIMEOUT after {timeout}s")
            return 124
    return proc.wait()


def locate_bundle():
    """Kaggle unpacks zip files uploaded to a dataset, so the bundle usually arrives as extracted files (as in kaggle/fm_kernel)."""
    scripts = glob.glob("/kaggle/input/**/scripts/run_stat_lgbm.py", recursive=True)
    if scripts:
        shutil.copytree(Path(scripts[0]).parents[1], ROOT, dirs_exist_ok=True)
        return
    zips = glob.glob("/kaggle/input/**/bwb_bundle.zip", recursive=True)
    if not zips:
        raise FileNotFoundError("bundle not found under /kaggle/input")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(ROOT)


def main():
    result = {"part": PART, "tracks": TRACKS}
    t0 = time.time()
    try:
        say(f"cpus={os.cpu_count()}")
        result["install"] = stream([sys.executable, "-m", "pip", "install", "-q", "statsforecast==2.0.1"], 1200)
        locate_bundle()
        env = dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1")
        result["run"] = stream([sys.executable, "-W", "ignore", "scripts/run_stat_lgbm.py", "--models", "AutoARIMA", "--tracks", TRACKS,
                                "--n-jobs", str(os.cpu_count() or 4)], TIMEOUT, cwd=ROOT, env=env)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        say(f"EXCEPTION {result['error']}")
    finally:
        src = ROOT / "data" / "predictions_cache" / "AutoARIMA"
        if src.exists():
            shutil.copytree(src, WORK / "predictions_cache" / "AutoARIMA", dirs_exist_ok=True)
        result["files"] = sorted(p.name for p in (WORK / "predictions_cache" / "AutoARIMA").glob("*.parquet")) if (WORK / "predictions_cache").exists() else []
        result["minutes"] = round((time.time() - t0) / 60, 1)
        (WORK / f"arima_{PART}_results.json").write_text(json.dumps(result, indent=2))
        say(f"summary: {result}")


if __name__ == "__main__":
    main()
