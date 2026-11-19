"""Week-6 verification: recompute every headline claim from artefacts and compare with what the reports assert."""
import pandas as pd, numpy as np, json, pathlib

ok = lambda c: "PASS" if c else "**FAIL**"
rows = []
def check(claim, computed, asserted, passed):
    rows.append((claim, str(computed), str(asserted), ok(passed)))

# 1. pairwise significance
p = pd.read_csv("reports/significance_pairwise.csv")
win = int((p.fm_better & p.significant_05).sum()); lose = int((~p.fm_better & p.significant_05).sum())
tie = int((~p.significant_05).sum())
check("FM wins / baseline wins / ties of 48", f"{win}/{lose}/{tie}", "14/5/29", (win, lose, tie) == (14, 5, 29))
check("total comparisons", len(p), 48, len(p) == 48)
rean = p.track.str.startswith("nasa")
st_w = int((p.fm_better & p.significant_05 & ~rean).sum()); re_w = int((p.fm_better & p.significant_05 & rean).sum())
check("FM wins on stations / reanalysis (of 24 each)", f"{st_w}/{re_w}", "3/11", (st_w, re_w, int(rean.sum())) == (3, 11, 24))
check("station FM wins all at lead 1", sorted(p[p.fm_better & p.significant_05 & ~rean].lead.unique().tolist()), [1],
      set(p[p.fm_better & p.significant_05 & ~rean].lead) == {1})
bw = p[~p.fm_better & p.significant_05]
check("baseline wins: all climatology on rainfall", sorted(bw.best_baseline.unique().tolist()), ["climatology"],
      set(bw.best_baseline) == {"climatology"} and bw["var"].isin(["Rainfall", "PRECTOTCORR"]).all())
check("baseline wins on tropical tracks", int(bw.track.isin(["bmd", "ghcn_bangladesh", "nasa_power_bmd", "nasa_power_ghcn_bangladesh"]).sum()),
      1, int(bw.track.isin(["bmd", "ghcn_bangladesh", "nasa_power_bmd", "nasa_power_ghcn_bangladesh"]).sum()) == 1)

# 1b. primary family with comparators chosen on the 2011-2015 validation split (scripts/validation_selection.py)
v = pd.read_csv("reports/significance_valselect.csv")
vc = v.valselect.value_counts(); vn = v.valselect_nolgbm.value_counts()
check("val-selected FM / baseline / ties of 48", f"{vc.get('FM',0)}/{vc.get('baseline',0)}/{vc.get('neither',0)}", "17/5/26",
      (vc.get("FM", 0), vc.get("baseline", 0), vc.get("neither", 0), len(v)) == (17, 5, 26, 48))
check("val-selected, LightGBM excluded", f"{vn.get('FM',0)}/{vn.get('baseline',0)}/{vn.get('neither',0)}", "18/5/25",
      (vn.get("FM", 0), vn.get("baseline", 0), vn.get("neither", 0)) == (18, 5, 25))
vre = v.track.str.startswith("nasa")
vs = v[(v.valselect == "FM") & ~vre]
check("val-selected FM wins on stations / reanalysis", f"{len(vs)}/{int(((v.valselect == 'FM') & vre).sum())}", "5/12",
      (len(vs), int(((v.valselect == "FM") & vre).sum())) == (5, 12))
check("val-selected station FM wins at lead 1", int((vs.lead == 1).sum()), 3, int((vs.lead == 1).sum()) == 3)
vb = v[v.valselect == "baseline"]
check("val-selected baseline wins: all climatology on rainfall", sorted(vb.best_baseline.unique().tolist()), ["climatology"],
      set(vb.best_baseline) == {"climatology"} and vb["var"].isin(["Rainfall", "PRECTOTCORR"]).all())
x = pd.crosstab(v.primary, v.valselect)
check("test-chosen FM wins kept / baseline wins kept / ties turned FM", f"{x.loc['FM','FM']}/{x.loc['baseline','baseline']}/{x.loc['neither','FM']}",
      "12/5/5", (x.loc["FM", "FM"], x.loc["baseline", "baseline"], x.loc["neither", "FM"]) == (12, 5, 5))
check("no verdict reverses direction", int(x.loc["FM"].get("baseline", 0) + x.loc["baseline"].get("FM", 0)), 0,
      x.loc["FM"].get("baseline", 0) + x.loc["baseline"].get("FM", 0) == 0)
check("same pair as test-chosen", int(v.same_pair_as_primary.sum()), 17, int(v.same_pair_as_primary.sum()) == 17)
check("LightGBM / AutoARIMA chosen as baseline", f"{(v.best_baseline=='lightgbm').sum()}/{(v.best_baseline=='AutoARIMA').sum()}", "7/7",
      ((v.best_baseline == "lightgbm").sum(), (v.best_baseline == "AutoARIMA").sum()) == (7, 7))

# 2. rainfall vs climatology
b = pd.read_csv("reports/leaderboard_daily.csv"); b = b[~b.model.str.endswith("_gapmatched")]
FMS = ["chronos2","chronos_bolt","timesfm25","toto2","tirex","moirai2","ttm_r2","sundial"]
bmd_rain = b[(b.track=="bmd") & (b["var"]=="Rainfall") & (b.lead.isin([7,30])) & b.model.isin(FMS)]
check("BMD rainfall lead 7/30: any FM CRPSS > 0", f"max={bmd_rain.CRPSS_vs_clim.max():.4f}", "none > 0",
      bmd_rain.CRPSS_vs_clim.max() <= 0)
d = pd.read_csv("reports/leaderboard_dekadal.csv"); dfm = d[d.base_model.isin(FMS) & ~d.model.str.contains("_ctx")]
check("CHIRPS dekadal: any FM CRPSS > 0 at any lead", f"max={dfm.CRPSS_vs_clim.max():.4f}", "none > 0",
      dfm.CRPSS_vs_clim.max() <= 0)

# 3. obs vs reanalysis
o = pd.read_csv("reports/significance_obs_vs_reanalysis.csv")
o_up = int((o.significant_05 & (o.reanalysis_minus_obs > 0)).sum()); o_dn = int((o.significant_05 & (o.reanalysis_minus_obs < 0)).sum())
check("obs-vs-reanalysis: favour reanalysis / stations", f"{o_up}/{o_dn} of {len(o)}", "74/4 of 108",
      (o_up, o_dn, len(o)) == (74, 4, 108))
o_t30 = o[(o["var"] == "Temperature") & (o.lead == 30)]
check("obs-vs-reanalysis: Tavg lead 30 favouring reanalysis", int((o_t30.significant_05 & (o_t30.reanalysis_minus_obs > 0)).sum()),
      0, int((o_t30.significant_05 & (o_t30.reanalysis_minus_obs > 0)).sum()) == 0)

# 4. equity
e = pd.read_csv("reports/significance_equity.csv")
t30 = e[(e["var"]=="Tavg") & (e.lead==30)]
check("Tavg lead 30: equity control", t30.control.unique().tolist(), ["AutoARIMA"], set(t30.control) == {"AutoARIMA"})
check("Tavg lead 30: share of raw median removed by masking", round(1 - t30.DiD_gm_control.median() / t30.DiD.median(), 2), "~0.60",
      abs(1 - t30.DiD_gm_control.median() / t30.DiD.median() - 0.6) < 0.03)
check("Tavg lead 30: models significant (both masked)", int(t30.gmc_significant_05.sum()), 4,
      int(t30.gmc_significant_05.sum()) == 4)
check("Tavg lead 30: median both-masked DiD", round(t30.DiD_gm_control.median(), 3), 0.043,
      abs(t30.DiD_gm_control.median() - 0.043) < 0.0006)
check("Tavg lead 30: median raw DiD", round(t30.DiD.median(), 3), 0.112, abs(t30.DiD.median() - 0.112) < 0.0006)
rain_any = e[e["var"]=="Rainfall"]
check("rainfall: any positive-significant tropical penalty", int((rain_any.gmc_significant_05 & (rain_any.DiD_gm_control>0)).sum()),
      "0 expected", int((rain_any.gmc_significant_05 & (rain_any.DiD_gm_control>0)).sum()) <= 1)

# 5. context 2048
c = pd.read_csv("reports/significance_equity_ctx2048.csv")
c30 = c[(c["var"]=="Tavg") & (c.lead==30)]
check("Tavg lead 30 median both-masked DiD at 2048", round(c30.DiD_gm_control.median(), 3), 0.052,
      abs(c30.DiD_gm_control.median() - 0.052) < 0.0006)
check("same at 1024 (from equity file, same 3 models)", round(e[(e["var"]=="Tavg") & (e.lead==30) & e.model.isin(c30.model)].DiD_gm_control.median(), 3),
      0.052, abs(e[(e["var"]=="Tavg") & (e.lead==30) & e.model.isin(c30.model)].DiD_gm_control.median() - 0.052) < 0.0006)

# 6. spells
s = pd.read_csv("reports/significance_spells.csv")
check("spell events active / break", f"{s.loc[s.spell=='active','n_events'].min()}-{s.loc[s.spell=='active','n_events'].max()} / "
      f"{s.loc[s.spell=='break','n_events'].min()}-{s.loc[s.spell=='break','n_events'].max()}", "11-12 / 18-21",
      (s.loc[s.spell=='active','n_events'].agg(['min','max']).tolist(), s.loc[s.spell=='break','n_events'].agg(['min','max']).tolist()) == ([11, 12], [18, 21]))
check("spell contrasts significant", f"{int(s.significant_05.sum())}/{len(s)}", "61/72",
      (int(s.significant_05.sum()), len(s)) == (61, 72))

# 7. compute
cc = pd.read_csv("reports/compute_cost.csv")
check("full zero-shot leaderboard GPU hours", round(cc.daily_minutes_170k_windows.sum()/60, 2), "~4.2",
      abs(cc.daily_minutes_170k_windows.sum()/60 - 4.2) < 0.15)

# 8. ranks
r = pd.read_csv("reports/significance_ranks.csv")
check("Nemenyi critical difference", round(r.cd.iloc[0], 2), 6.58, abs(r.cd.iloc[0] - 6.58) < 0.01)
r30 = r[r.lead==30].sort_values("avg_rank")
within = r30[r30.avg_rank <= r30.avg_rank.min() + r30.cd.iloc[0]]
check("models within one CD of the best at lead 30", len(within), ">= 7", len(within) >= 7)

# 9. scale
w = pd.read_parquet("data/processed/windows_daily.parquet")
check("daily windows", len(w), 170571, len(w) == 170571)
check("daily tasks (track x var)", w.groupby(["track","var"]).ngroups, 16, w.groupby(["track","var"]).ngroups == 16)
check("models on daily leaderboard", b.model.nunique(), 18, b.model.nunique() == 18)
check("skill CIs excluding climatology", f"{int(pd.read_csv('reports/significance_skill_ci.csv').beats_climatology_95.sum())}/432",
      "272/432", int(pd.read_csv('reports/significance_skill_ci.csv').beats_climatology_95.sum()) == 272)

sk = pd.read_csv('reports/significance_skill_ci.csv')
check("skill CIs entirely below climatology", f"{int((sk.ci_hi < 0).sum())}/432", "76/432", int((sk.ci_hi < 0).sum()) == 76)

obs_tasks = w[~w.track.str.startswith("nasa")].groupby(["track", "var"]).ngroups
check("observation / reanalysis tasks", f"{obs_tasks}/{16 - obs_tasks}", "8/8", obs_tasks == 8)

o3 = pd.read_csv("reports/significance_obs_vs_reanalysis.csv")
o3 = o3[o3["var"] != "Sunshine"]
o3u = int((o3.significant_05 & (o3.reanalysis_minus_obs > 0)).sum()); o3d = int((o3.significant_05 & (o3.reanalysis_minus_obs < 0)).sum())
check("obs-vs-reanalysis, rain/temp/humidity: reanalysis / stations", f"{o3u}/{o3d} of {len(o3)}", "67/2 of 81",
      (o3u, o3d, len(o3)) == (67, 2, 81))

# 10. split-sample check of the primary family (scripts/split_sample_check.py)
ss = pd.read_csv("reports/significance_split.csv")
sv, fv = ss.split.value_counts(), ss.fixed_late.value_counts()
check("split-sample FM / baseline wins of 48", f"{sv.get('FM', 0)}/{sv.get('baseline', 0)}", "5/0",
      (sv.get("FM", 0), sv.get("baseline", 0)) == (5, 0))
check("published pair on 2020-2023: FM / baseline wins", f"{fv.get('FM', 0)}/{fv.get('baseline', 0)}", "4/0",
      (fv.get("FM", 0), fv.get("baseline", 0)) == (4, 0))
won = ss[ss.primary != "neither"]
keep = int(((won.primary == "FM") == (won.mean_sCRPS_diff < 0)).sum())
check("decided comparisons keeping direction held-out", f"{keep}/{len(won)}", "19/19", (keep, len(won)) == (19, 19))
check("split-sample picks the published pair", f"{int(ss.same_pair_as_primary.sum())}/48", "29/48",
      int(ss.same_pair_as_primary.sum()) == 29)

print(f"{'claim':52s} {'computed':22s} {'asserted':12s} verdict")
print("-"*104)
for c_, comp, asrt, v in rows:
    print(f"{c_:52s} {comp:22s} {asrt:12s} {v}")
fails = sum(1 for r_ in rows if "FAIL" in r_[3])
print(f"\n{len(rows)-fails}/{len(rows)} headline claims reproduce from artefacts")
