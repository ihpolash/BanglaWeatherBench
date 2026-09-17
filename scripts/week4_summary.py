"""Week-4 summary tables behind reports/week4_fm_report.md (Results): overall ranks, obs vs reanalysis family margins,
calibration, equity DiD summaries, BMD monsoon-phase skill, TTM/Sundial diagnostics, and BMD active/break-spell skill
(writes reports/leaderboard_bmd_spell.csv). Run after build_leaderboard.py and analyze_week4.py."""
import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
b = pd.read_csv("reports/leaderboard_daily.csv")
FMS = ["chronos2","chronos_bolt","timesfm25","toto2","tirex","moirai2","ttm_r2","sundial"]
TR = [m for m in ["AutoETS","AutoTheta","AutoARIMA","lightgbm","NHITS","DLinear","PatchTST"] if m in set(pd.read_csv("reports/leaderboard_daily.csv").model)]; NV=["naive","seasonal_naive","climatology"]
main = b[~b.model.str.endswith("_gapmatched")]
print("n_windows check (per track/var, unique):"); print(main.groupby(["track","var"]).n_windows.agg(["min","max"]).to_string())
print("n_models_compared:", sorted(b.n_models_compared.unique()))
L = main[main.lead.isin([1,7,30])].copy()
L["rank"] = L.groupby(["track","var","lead"]).sCRPS.rank()
print("\nMean CRPSS vs climatology over 16 tasks, and mean rank (of 17)")
t = L.pivot_table(index="model", columns="lead", values="CRPSS_vs_clim", aggfunc="mean").round(3)
t["rank_mean"] = L.groupby("model")["rank"].mean().round(2)
for lead in (1,7,30):
    t[f"wins_l{lead}"] = L[L.lead==lead].loc[L[L.lead==lead].groupby(["track","var"]).sCRPS.idxmin()].model.value_counts()
print(t.sort_values("rank_mean").fillna(0).to_string())
print("\nBy track group, mean CRPSS all leads 1/7/30 (obs tracks vs reanalysis tracks)")
L["grp"] = np.where(L.track.str.startswith("nasa"), "reanalysis", "observations")
print(L.pivot_table(index="model", columns=["grp","lead"], values="CRPSS_vs_clim", aggfunc="mean").round(3).loc[FMS+TR+["seasonal_naive","naive"]].to_string())
print("\nRainfall-only / temperature-like-only mean CRPSS (obs tracks)")
O = L[L.grp=="observations"].copy(); O["vk"] = np.where(O["var"].isin(["Rainfall"]), "rain", "other")
print(O.pivot_table(index="model", columns=["vk","lead"], values="CRPSS_vs_clim", aggfunc="mean").round(3).loc[FMS+TR].to_string())
print("\n80% interval coverage (mean over tasks, leads 1/7/30) — nominal 0.80")
cov = L.pivot_table(index="model", columns=["grp"], values="coverage80", aggfunc="mean").round(3)
cov["rain_obs"] = L[(L.grp=="observations")&(L["var"]=="Rainfall")].groupby("model").coverage80.mean().round(3)
cov["min"] = L.groupby("model").coverage80.min().round(3); cov["max"] = L.groupby("model").coverage80.max().round(3)
print(cov.loc[FMS+TR+NV].to_string())
print("\nFM family vs trained family: reanalysis minus obs skill (contrast B), mean over 4 vars")
c = pd.read_csv("reports/week4_contrasts.csv"); B = c[c.contrast=="B_obs_vs_reanalysis"]
print(B.groupby(["family","lead"]).reanalysis_minus_obs.agg(["mean","median","min","max"]).round(3).to_string())
print(B.groupby(["var","family"]).reanalysis_minus_obs.mean().unstack().round(3).to_string())
A = c[c.contrast=="A_equity"]
print("\nEquity DiD summary: FM mean/median DiD by var x lead (main and gap-matched DiD using same control)")
A = A.assign(DiD_gm = A.gap_gapmatched - A.control_gap_best_trained)
print(A[A.family=="FM"].groupby(["var","lead"])[["gap","DiD_vs_control","gap_gapmatched","DiD_gm"]].agg(["mean","median"]).round(3).to_string())
print(A[A.family=="trained"].groupby(["var","lead"])[["gap","DiD_vs_control"]].agg(["mean","median"]).round(3).to_string())
p = pd.read_csv("reports/leaderboard_bmd_phase.csv")
print("\nphase n:"); print(p[(p.model=="climatology")].pivot_table(index=["var","monsoon_phase"], columns="lead", values="n").to_string())
clim = p[p.model=="climatology"][["var","lead","monsoon_phase","sCRPS"]].rename(columns={"sCRPS":"clim"})
p = p.merge(clim, on=["var","lead","monsoon_phase"]); p["CRPSS"] = 1 - p.sCRPS/p.clim
for var in ["Rainfall","Temperature","Humidity","Sunshine"]:
    for lead in (1,30):
        q = p[(p["var"]==var)&(p.lead==lead)&(~p.model.str.endswith("_gapmatched"))].pivot_table(index="model", columns="monsoon_phase", values="CRPSS").round(3)
        print(f"\nBMD {var} lead {lead}: CRPSS vs climatology by phase of target day"); print(q.loc[[m for m in FMS+TR+["seasonal_naive"] if m in q.index]].to_string())

# ---- diagnostics and active/break spells ----
import pandas as pd, numpy as np
pd.set_option("display.width", 250)
C = "data/predictions_cache"
p0 = pd.read_parquet(f"{C}/chronos2/bmd__Temperature.parquet"); print("cache columns:", list(p0.columns))
qcols = [c for c in p0.columns if c not in ("window_id","lead","mean")]
lo, hi = [c for c in qcols if "0.1" in c and "0.15" not in c][0], [c for c in qcols if "0.9" in c][0]
print("interval columns:", lo, hi)
rows = []
for m in ["climatology","PatchTST","NHITS","chronos2","chronos_bolt","timesfm25","toto2","tirex","moirai2","ttm_r2","sundial"]:
    for var in ["Temperature","Rainfall"]:
        p = pd.read_parquet(f"{C}/{m}/bmd__{var}.parquet")
        for lead in (1,30):
            q = p[p.lead==lead]
            rows.append({"model":m,"var":var,"lead":lead,"width80_mean":(q[hi]-q[lo]).mean(),"width80_median":(q[hi]-q[lo]).median(),
                         "mean_minus_median":(q["mean"]-q[[c for c in qcols if "0.5" in c][0]]).abs().mean()})
print(pd.DataFrame(rows).pivot_table(index="model", columns=["var","lead"], values=["width80_mean","mean_minus_median"]).round(2).to_string())
b = pd.read_csv("reports/leaderboard_daily.csv")
print("\nPoint skill (MAE of median vs climatology) at lead 1 / 30, BMD + GHCN tracks")
t = b[b.track.isin(["bmd","ghcn_bangladesh","ghcn_temperate","nasa_power_bmd"]) & b.model.isin(["naive","PatchTST","chronos2","tirex","ttm_r2","sundial"]) & b.lead.isin([1,30])]
print(t.pivot_table(index=["track","var"], columns=["lead","model"], values="skill_MAE_vs_clim").round(3).to_string())
# active/break spells, BMD
lab = pd.read_parquet("data/processed/bmd_labels.parquet", columns=["station","date","spell"])
print("\nspell values:", lab.spell.value_counts(dropna=False).to_dict())
w = pd.read_parquet("data/processed/windows_daily.parquet"); w = w[w.track=="bmd"][["window_id","var","series_id","origin"]]
s0 = pd.read_parquet("reports/scores/chronos2/bmd__Rainfall.parquet"); print("score columns:", list(s0.columns))
out = []
for var in ["Rainfall","Temperature","Humidity","Sunshine"]:
    wv = w[w["var"]==var]
    for m in ["climatology","NHITS","PatchTST","chronos2","chronos_bolt","timesfm25","toto2","tirex","moirai2","sundial"]:
        s = pd.read_parquet(f"reports/scores/{m}/bmd__{var}.parquet"); s = s[s.lead.isin([1,7,30])].merge(wv, on="window_id")
        s["date"] = s.origin + pd.to_timedelta(s.lead, unit="D")
        s = s.merge(lab.rename(columns={"station":"series_id"}), on=["series_id","date"], how="inner")
        s = s[s.spell.notna() & (s.spell.astype(str)!="none") & (s.spell.astype(str)!="")]
        g = s.groupby(["lead","spell"]).agg(sCRPS=("scaled_crps","mean"), n=("scaled_crps","size")).reset_index()
        out.append(g.assign(var=var, model=m))
o = pd.concat(out); cl = o[o.model=="climatology"][["var","lead","spell","sCRPS"]].rename(columns={"sCRPS":"clim"})
o = o.merge(cl, on=["var","lead","spell"]); o["CRPSS"] = 1 - o.sCRPS/o.clim
o.to_csv("reports/leaderboard_bmd_spell.csv", index=False)
print(o[o.model=="climatology"].pivot_table(index=["var","spell"], columns="lead", values="n").to_string())
for var in ["Rainfall","Temperature","Humidity","Sunshine"]:
    print(f"\nBMD {var}: CRPSS vs climatology by spell of target day (Jul-Aug)")
    print(o[o["var"]==var].pivot_table(index="model", columns=["spell","lead"], values="CRPSS").round(3).to_string())
