"""Week-5 diagnostic: is Sundial's under-dispersion the model or our adapter?

Week 4 found Sundial's 80% intervals cover 32% of outcomes (nominal 80%) while its point forecasts are competitive,
and its mean 80% width on BMD temperature at lead 1 is 0.78 C against 2.5-3.0 C for the other foundation models.
Three explanations are separable on one GPU:
  A. sample count   - too few trajectories would make quantiles noisy, not systematically narrow: widths should be
                      flat in num_samples (20 / 100 / 500). If width grows with samples, our 100 was too few.
  B. reshape bug    - `generate` returns (batch, num_samples, horizon); if that were mis-ordered, samples would be
                      near-identical. Reported as the spread across samples vs across the batch.
  C. genuine model  - widths flat in num_samples and samples genuinely diverse => Sundial is simply overconfident
                      here, and the calibration table stands as written.
Also reports empirical 80% coverage against the real targets, so the number is comparable to reports/leaderboard_daily.csv.
Output: /kaggle/working/sundial_check.json and the printed log.
"""
import glob
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORK, ROOT, ENV = Path("/kaggle/working"), Path("/tmp/bwb"), Path("/tmp/envs/sundial")
CONTEXT, HORIZON, N_WINDOWS = 1024, 30, 512


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, **kw).returncode


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


PROBE = '''
import json, numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM
import sys
sys.path.insert(0, "src")
from bwb.data.store import load_track
from bwb.models.fm_base import interpolate_gaps, trim_leading_nan

CONTEXT, HORIZON, N = %d, %d, %d
windows = pd.read_parquet("data/processed/windows_daily.parquet")
w = windows[(windows.track == "bmd") & (windows["var"] == "Temperature")].head(N)
series = load_track("bmd")
ctx, target = [], []
for sid, origin in zip(w.series_id, w.origin):
    s = series[("Temperature", sid)]
    c = trim_leading_nan(s.loc[:origin].to_numpy(dtype=np.float32)[-CONTEXT:])
    ctx.append(interpolate_gaps(c))
    target.append(s.loc[origin + pd.Timedelta(days=1): origin + pd.Timedelta(days=HORIZON)].to_numpy())
L = max(len(c) for c in ctx)
x = torch.tensor(np.stack([np.pad(c, (L - len(c), 0), mode="edge") for c in ctx]), dtype=torch.float32).cuda()
y = np.stack(target)
model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).cuda().eval()
out = {}
for n_samples in (20, 100, 500):
    with torch.no_grad():
        raw = model.generate(x, max_new_tokens=HORIZON, num_samples=n_samples)
    arr = torch.as_tensor(raw).float().cpu().numpy()
    shape = list(arr.shape)
    s = arr.reshape(len(ctx), n_samples, -1)[:, :, :HORIZON]
    lo, hi = np.quantile(s, [0.1, 0.9], axis=1)
    width = float((hi - lo).mean())
    cover = float(((y >= lo) & (y <= hi)).mean())
    out[str(n_samples)] = {
        "raw_shape": shape, "mean_80_width": width, "empirical_80_coverage": cover,
        "sample_spread_within_window": float(s.std(axis=1).mean()),   # spread across trajectories
        "spread_across_windows": float(s.mean(axis=1).std()),          # spread across series
        "target_sd": float(y.std()), "rmse_of_sample_mean": float(np.sqrt(((s.mean(axis=1) - y) ** 2).mean())),
    }
    print(n_samples, out[str(n_samples)], flush=True)
json.dump(out, open("/kaggle/working/sundial_check.json", "w"), indent=2)
''' % (CONTEXT, HORIZON, N_WINDOWS)


def main():
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    locate_bundle()
    import importlib.metadata as md

    overrides = Path("/tmp/torch_overrides.txt")
    overrides.write_text("".join(f"{p}=={md.version(p).split('+')[0]}\n" for p in ("torch", "torchvision", "torchaudio")))
    sh(["uv", "venv", "--system-site-packages", "--python", sys.executable, str(ENV)])
    py = ENV / "bin" / "python"
    sh(["uv", "pip", "install", "--python", str(py), "--override", str(overrides), "transformers==4.40.1", "pandas", "pyarrow", "numpy", "pyyaml"])
    (ROOT / "probe.py").write_text(PROBE)
    code = sh([str(py), "-W", "ignore", "probe.py"], cwd=ROOT)
    print("probe exit", code, flush=True)
    f = WORK / "sundial_check.json"
    if f.exists():
        print(f.read_text(), flush=True)


if __name__ == "__main__":
    main()
