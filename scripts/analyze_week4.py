"""Week-4 contrasts from reports/leaderboard_daily.csv (skill = CRPSS vs same-track climatology; higher is better).

A. Equity test (GHCN vs GHCN): gap = CRPSS(temperate) - CRPSS(Bangladesh), per model. The control is the per-track
   *trained* baselines, whose gap reflects intrinsic predictability, not pretraining. Difference-in-differences
   (DiD) = model gap - control gap: positive DiD means the model loses more skill in the tropics than trained models
   do (consistent with pretraining imbalance). Gap-matched rows replace temperate skill with the model's
   `<model>_gapmatched` run (temperate contexts carrying Bangladesh-like missingness).
B. Observations vs reanalysis: CRPSS on BMD observations vs NASA POWER at the same 35 stations.
C. Foundation models vs the best Week-3 baseline per task.
Writes reports/week4_contrasts.csv. Works with any subset of models present (e.g. baselines only).
"""
import numpy as np
import pandas as pd

LEADS = (1, 7, 30)
TRAINED = ["AutoETS", "AutoTheta", "AutoARIMA", "lightgbm", "NHITS", "DLinear", "PatchTST"]
NAIVE = ["naive", "seasonal_naive", "climatology"]
BASELINES = NAIVE + TRAINED
FMS = ["chronos2", "chronos_bolt", "timesfm25", "toto2", "tirex", "moirai2", "ttm_r2", "sundial"]
OBS_REANALYSIS = [("Rainfall", "PRECTOTCORR"), ("Temperature", "T2M"), ("Humidity", "RH2M"), ("Sunshine", "ALLSKY_SFC_SW_DWN")]

b = pd.read_csv("reports/leaderboard_daily.csv")
present = set(b.model)
fms = [m for m in FMS if m in present]
trained = [m for m in TRAINED if m in present]


def skill(track, var, lead, model):
    r = b[(b.track == track) & (b["var"] == var) & (b.lead == lead) & (b.model == model)]
    return float(r.CRPSS_vs_clim.iloc[0]) if len(r) else np.nan


rows = []
# A. Equity test
for var in ("Rainfall", "Tavg"):
    for lead in LEADS:
        ctrl_bd = max(skill("ghcn_bangladesh", var, lead, m) for m in trained) if trained else np.nan
        ctrl_te = max(skill("ghcn_temperate", var, lead, m) for m in trained) if trained else np.nan
        control_gap = ctrl_te - ctrl_bd
        for m in trained + fms:
            bd, te = skill("ghcn_bangladesh", var, lead, m), skill("ghcn_temperate", var, lead, m)
            te_gm = skill("ghcn_temperate", var, lead, f"{m}_gapmatched")
            rows.append({"contrast": "A_equity", "var": var, "lead": lead, "model": m, "family": "FM" if m in fms else "trained",
                         "skill_bangladesh": bd, "skill_temperate": te, "gap": te - bd,
                         "control_gap_best_trained": control_gap, "DiD_vs_control": (te - bd) - control_gap,
                         "skill_temperate_gapmatched": te_gm, "gap_gapmatched": te_gm - bd})
# B. Observations vs reanalysis
for obs_var, re_var in OBS_REANALYSIS:
    for lead in LEADS:
        for m in trained + fms:
            o, r = skill("bmd", obs_var, lead, m), skill("nasa_power_bmd", re_var, lead, m)
            rows.append({"contrast": "B_obs_vs_reanalysis", "var": obs_var, "lead": lead, "model": m,
                         "family": "FM" if m in fms else "trained", "skill_obs": o, "skill_reanalysis": r, "reanalysis_minus_obs": r - o})
# C. Best FM vs best baseline per task
for (track, var), g in b[b.lead.isin(LEADS)].groupby(["track", "var"]):
    for lead in LEADS:
        gl = g[g.lead == lead]
        bl, fm = gl[gl.model.isin(BASELINES)], gl[gl.model.isin(fms)]
        if bl.empty:
            continue
        best_bl = bl.loc[bl.sCRPS.idxmin()]
        row = {"contrast": "C_fm_vs_baseline", "track": track, "var": var, "lead": lead,
               "best_baseline": best_bl.model, "best_baseline_CRPSS": best_bl.CRPSS_vs_clim}
        if not fm.empty:
            best_fm = fm.loc[fm.sCRPS.idxmin()]
            row.update(best_fm=best_fm.model, best_fm_CRPSS=best_fm.CRPSS_vs_clim,
                       fm_beats_best_baseline=bool(best_fm.sCRPS < best_bl.sCRPS))
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv("reports/week4_contrasts.csv", index=False)
pd.set_option("display.width", 250)
print(f"models present: FMs={fms or 'none yet'} | trained baselines={trained}")
a = out[out.contrast == "A_equity"]
print("\nA. Equity test (gap = temperate - Bangladesh CRPSS; control = best trained baseline gap)")
print(a[["var", "lead", "model", "family", "skill_bangladesh", "skill_temperate", "gap", "control_gap_best_trained", "DiD_vs_control", "gap_gapmatched"]].round(3).to_string(index=False))
bb = out[out.contrast == "B_obs_vs_reanalysis"]
print("\nB. Reanalysis minus observation skill (positive = reanalysis looks more predictable)")
print(bb.pivot_table(index=["var", "lead"], columns="model", values="reanalysis_minus_obs").round(3).to_string())
c = out[out.contrast == "C_fm_vs_baseline"]
print("\nC. Best baseline per task" + ("" if "best_fm" in c else " (no FM results yet)"))
cols = [k for k in ["track", "var", "lead", "best_baseline", "best_baseline_CRPSS", "best_fm", "best_fm_CRPSS", "fm_beats_best_baseline"] if k in c]
print(c[cols].round(3).to_string(index=False))
