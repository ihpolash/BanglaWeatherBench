"""Remaining Week-4 analyses: compute cost, daily context-length ablation, CHIRPS dekadal zero-shot results.

Inputs (installed by scripts/install_kaggle_week4b.sh)
  reports/fm_run_<model>.jsonl, fm_run_<model>_profile.jsonl   wall time (main run); peak GPU memory, parameters (profile)
  reports/ablation/board_<model>_ctx<N>.csv                    daily scores at other context lengths (scored on Kaggle)
  reports/leaderboard_daily.csv                                main run (context 1,024) and same-track climatology
  reports/leaderboard_dekadal.csv                              CHIRPS dekadal, all models and contexts
Outputs: reports/compute_cost.csv, reports/context_ablation.csv, printed tables (tee to reports/week4b_analysis.log).
Skill = CRPSS vs same-track climatology (higher is better). Works with whatever subset of inputs is present.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
FMS = ["chronos2", "chronos_bolt", "timesfm25", "toto2", "tirex", "moirai2", "ttm_r2", "sundial"]
LEADS = (1, 7, 30)


def records(pattern):
    rows = [json.loads(l) for f in sorted(Path("reports").glob(pattern)) for l in open(f) if l.strip()]
    return pd.DataFrame(rows)


# 1. Compute cost (Kaggle Tesla T4)
main = records("fm_run_*.jsonl")
main = main[main.model.isin(FMS)] if len(main) else main
prof = records("fm_run_*_profile.jsonl")
cost = []
for m in FMS:
    r = main[main.model == m] if len(main) else pd.DataFrame()
    p = prof[prof.model == f"{m}_profile"] if len(prof) else pd.DataFrame()
    cost.append({"model": m,
                 "params_M": round(p.params.max() / 1e6, 1) if len(p) and "params" in p else np.nan,
                 "peak_vram_MiB": p.peak_vram_mb.max() if len(p) and "peak_vram_mb" in p else np.nan,
                 "batch_size": p.batch_size.max() if len(p) and "batch_size" in p else np.nan,
                 "context": int(r.context.max()) if len(r) else np.nan,
                 "daily_minutes_170k_windows": round(r.seconds.sum() / 60, 1) if len(r) else np.nan,
                 "windows_per_second": round(r.windows.sum() / r.seconds.sum(), 1) if len(r) else np.nan})
cost = pd.DataFrame(cost)
cost.to_csv("reports/compute_cost.csv", index=False)
print("1. Compute cost on a Kaggle Tesla T4 (main daily run; memory and parameters from the profile pass)")
print(cost.to_string(index=False))

# 2. Daily context-length ablation
b = pd.read_csv("reports/leaderboard_daily.csv")
clim = b[b.model == "climatology"].set_index(["track", "var", "lead"]).sCRPS
boards = [pd.read_csv(f) for f in sorted(Path("reports/ablation").glob("board_*_ctx*.csv"))]
if boards:
    abl = pd.concat(boards, ignore_index=True)
    abl["base_model"] = abl.model.str.replace(r"_ctx\d+$", "", regex=True)
    abl["context"] = abl.model.str.extract(r"_ctx(\d+)$")[0].astype(int)
    ref = b[b.model.isin(abl.base_model.unique())].assign(base_model=lambda d: d.model, context=1024)
    ctx = pd.concat([abl, ref[abl.columns.intersection(ref.columns)]], ignore_index=True)
    ctx["CRPSS_vs_clim"] = 1 - ctx.sCRPS / clim.reindex(pd.MultiIndex.from_frame(ctx[["track", "var", "lead"]])).to_numpy()
    # every run must be scored on the main leaderboard's windows
    nw = b.drop_duplicates(["track", "var"]).set_index(["track", "var"]).n_windows
    mism = ctx[ctx.n_windows.to_numpy() != nw.reindex(pd.MultiIndex.from_frame(ctx[["track", "var"]])).to_numpy()]
    print(f"\nwindow-count mismatches vs main leaderboard: {len(mism)}")
    ctx.to_csv("reports/context_ablation.csv", index=False)
    c = ctx[ctx.lead.isin(LEADS)].copy()
    c["group"] = np.where(c.track.str.startswith("nasa"), "reanalysis", "observations")
    c["kind"] = np.where(c["var"].isin(["Rainfall", "PRECTOTCORR"]), "rain", "non-rain")
    print("\n2a. Mean CRPSS over the 16 daily tasks by context length (1,024 = main run)")
    print(c.pivot_table(index=["base_model", "context"], columns="lead", values="CRPSS_vs_clim").round(3).to_string())
    print("\n2b. By track group and rain / non-rain (observation tracks vs reanalysis tracks)")
    print(c.pivot_table(index=["base_model", "context"], columns=["group", "kind", "lead"], values="CRPSS_vs_clim").round(3).to_string())
    print("\n2c. Best context per model x task x lead (count of tasks)")
    best = c.loc[c.groupby(["base_model", "track", "var", "lead"]).sCRPS.idxmin()]
    print(best.pivot_table(index=["base_model", "lead"], columns="context", values="track", aggfunc="count", fill_value=0).to_string())
    g = c[c.track.isin(["ghcn_bangladesh", "ghcn_temperate"])].pivot_table(index=["base_model", "context", "var", "lead"], columns="track", values="CRPSS_vs_clim")
    g["gap_temperate_minus_bd"] = g.ghcn_temperate - g.ghcn_bangladesh
    print("\n2d. Tropical-vs-temperate gap (GHCN tracks) by context length")
    print(g.gap_temperate_minus_bd.unstack(["var", "lead"]).round(3).to_string())
else:
    print("\n2. no context-ablation boards found")

# 3. CHIRPS dekadal
d = pd.read_csv("reports/leaderboard_dekadal.csv")
if "base_model" in d:
    print("\n3a. CHIRPS dekadal, headline context (144 dekads) and baselines: CRPSS vs climatology / 80% coverage")
    h = d[~d.model.str.contains("_ctx")]
    t = h.pivot_table(index="model", columns="lead", values="CRPSS_vs_clim").round(3)
    t["cov80_mean"] = h.groupby("model").coverage80.mean().round(3)
    t["MASE_l1"] = h[h.lead == 1].set_index("model").MASE.round(3)
    print(t.sort_values(1, ascending=False).to_string())
    f = d[d.base_model.isin(FMS)]
    print("\n3b. CHIRPS dekadal context sensitivity: CRPSS vs climatology by context (dekads)")
    print(f.pivot_table(index=["base_model", "context"], columns="lead", values="CRPSS_vs_clim").round(3).to_string())
