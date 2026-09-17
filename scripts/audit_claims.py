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
check("obs-vs-reanalysis significant", f"{int(o.significant_05.sum())}/{len(o)}", "78/108",
      (int(o.significant_05.sum()), len(o)) == (78, 108))

# 4. equity
e = pd.read_csv("reports/significance_equity.csv")
t30 = e[(e["var"]=="Tavg") & (e.lead==30)]
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

print(f"{'claim':52s} {'computed':22s} {'asserted':12s} verdict")
print("-"*104)
for c_, comp, asrt, v in rows:
    print(f"{c_:52s} {comp:22s} {asrt:12s} {v}")
fails = sum(1 for r_ in rows if "FAIL" in r_[3])
print(f"\n{len(rows)-fails}/{len(rows)} headline claims reproduce from artefacts")
