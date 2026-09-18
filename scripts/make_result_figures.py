"""Week-5 result figures F6-F12 (print, arXiv). Style from bwb.viz.style; every figure also writes its table view.

Inputs: reports/leaderboard_daily.csv, leaderboard_dekadal.csv, leaderboard_bmd_phase.csv, context_ablation.csv,
compute_cost.csv and the significance_*.csv written by scripts/run_significance.py.
Usage: PYTHONPATH=src python scripts/make_result_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bwb.viz.style import CAT, DATA, GRID, INK, INK2, MUTED, NEUTRAL_1, S1, S2, SEQ, save

LEADS = (1, 3, 7, 14, 30)
FMS = ["chronos2", "chronos_bolt", "timesfm25", "toto2", "tirex", "moirai2", "ttm_r2", "sundial"]
LABEL = {"chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt", "timesfm25": "TimesFM 2.5", "toto2": "Toto-2.0",
         "tirex": "TiRex", "moirai2": "Moirai-2", "ttm_r2": "TTM r2", "sundial": "Sundial", "PatchTST": "PatchTST",
         "NHITS": "NHITS", "AutoARIMA": "AutoARIMA", "lightgbm": "LightGBM", "climatology": "Climatology",
         "AutoETS": "AutoETS", "AutoTheta": "AutoTheta", "DLinear": "DLinear", "naive": "Persistence",
         "seasonal_naive": "Seasonal naive"}
Path(DATA).mkdir(parents=True, exist_ok=True)
board = pd.read_csv("reports/leaderboard_daily.csv")
board = board[~board.model.str.endswith("_gapmatched")]


def table(df, name):
    df.to_csv(f"{DATA}/{name}.csv", index=False)


# ---------- F6 critical-difference diagram (Friedman + Nemenyi) ----------
ranks = pd.read_csv("reports/significance_ranks.csv")
fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)  # 18 models per panel need the height
for ax, lead in zip(axes, sorted(ranks.lead.unique())):
    r = ranks[ranks.lead == lead].sort_values("avg_rank")
    cd, best = r.cd.iloc[0], r.avg_rank.min()
    y = np.arange(len(r))
    colour = [S1 if m in FMS else MUTED for m in r.model]
    ax.barh(y, r.avg_rank, color=colour, height=0.6, zorder=3)
    ax.axvspan(best, best + cd, color=NEUTRAL_1, zorder=0, lw=0)  # models inside the band are not separable
    ax.set_yticks(y, [LABEL.get(m, m) for m in r.model], fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.invert_yaxis()
    ax.set_title(f"lead {lead} d   (Friedman p = {r.friedman_p.iloc[0]:.1e}, CD = {cd:.2f})")
    ax.grid(axis="x", zorder=0)
axes[-1].set_xlabel("average rank over the 16 daily tasks (1 = best)")
for ax in axes:  # the shaded band marks ranks within one critical difference of the best model
    ax.text(ax.patches[0].get_width() + ranks.cd.iloc[0] / 2, -0.85, "not separable from the best",
            color=MUTED, fontsize=6.5, ha="center", va="center")
    ax.set_ylim(len(ranks[ranks.lead == 1]) - 0.4, -1.3)  # headroom for the band label
fig.suptitle("Critical-difference ranking: foundation models (blue) vs baselines (grey)", x=0.01, ha="left",
             fontsize=10, fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "F6_critical_difference")
table(ranks, "F6_ranks")

# ---------- F7 skill vs lead, daily tracks and CHIRPS dekadal ----------
obs = board[~board.track.str.startswith("nasa")]
show = ["chronos2", "tirex", "timesfm25", "PatchTST", "AutoARIMA", "NHITS"]
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), sharey=True)
panels = [("Station temperature / Tavg", obs[obs["var"].isin(["Temperature", "Tavg"])]),
          ("Station rainfall", obs[obs["var"] == "Rainfall"]),
          ("CHIRPS dekadal rainfall", None)]
dek = pd.read_csv("reports/leaderboard_dekadal.csv")
dek = dek[~dek.model.str.contains("_ctx")]
for ax, (title, df) in zip(axes, panels):
    if df is None:
        for i, m in enumerate([m for m in ["chronos2", "toto2", "timesfm25", "tirex"] if m in set(dek.model)]):
            d = dek[dek.model == m].sort_values("lead")
            ax.plot(d.lead, d.CRPSS_vs_clim, color=CAT[i], lw=2, marker="o", ms=3.5, label=LABEL.get(m, m))
        ax.set_xlabel("lead (dekads)")
    else:
        for i, m in enumerate(show):
            d = df[df.model == m].groupby("lead").CRPSS_vs_clim.mean().reindex(LEADS)
            ax.plot(d.index, d.to_numpy(), color=CAT[i], lw=2, marker="o", ms=3.5, label=LABEL.get(m, m))
        ax.set_xlabel("lead (days)")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_title(title)
    ax.grid(axis="y")
axes[0].set_ylabel("CRPSS vs climatology")
axes[0].legend(fontsize=7, ncol=2, loc="upper right")
axes[2].legend(fontsize=7, loc="lower left")
fig.suptitle("Skill against climatology falls to zero on tropical rainfall at every horizon", x=0.01, ha="left",
             fontsize=10, fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "F7_skill_vs_lead")
table(board[board.model.isin(show + FMS)][["track", "var", "lead", "model", "CRPSS_vs_clim"]], "F7_skill")

# ---------- F8 equity difference-in-differences with CIs ----------
eq = pd.read_csv("reports/significance_equity.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
for ax, (var, lead) in zip(axes, [("Tavg", 7), ("Tavg", 30)]):
    e = eq[(eq["var"] == var) & (eq.lead == lead)].set_index("model").reindex(FMS).dropna(subset=["DiD"])
    y = np.arange(len(e))
    # Third series only exists once the control baselines have gap-matched runs of their own (like-for-like).
    series = [(S1, "as observed", "DiD", "ci_lo", "ci_hi"),
              (S2, "model gap-matched", "DiD_gapmatched", "gm_ci_lo", "gm_ci_hi")]
    if "DiD_gm_control" in e.columns and e["DiD_gm_control"].notna().any():
        series.append((CAT[2], "model + control gap-matched", "DiD_gm_control", "gmc_ci_lo", "gmc_ci_hi"))
    for off, (col, lab, pc, lc, hc) in zip(np.linspace(-0.2, 0.2, len(series)), series):
        point, lo, hi = e[pc], e[lc], e[hc]
        ax.errorbar(point, y + off, xerr=[point - lo, hi - point], fmt="o", ms=4, lw=1.6, color=col,
                    capsize=2, label=lab)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [LABEL[m] for m in e.index], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title(f"temperature, lead {lead} d")
    ax.set_xlabel("DiD vs trained-baseline control")
    ax.grid(axis="x")
handles, labels = axes[0].get_legend_handles_labels()  # legend outside the axes: three series per row crowd them
fig.legend(handles, labels, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
fig.suptitle("Tropical temperature penalty, and how much of it is missing context", x=0.01, ha="left",
             fontsize=10, fontweight="semibold", color=INK)
fig.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, "F8_equity_did")
table(eq, "F8_equity")

# ---------- F9 context length ----------
ctx = pd.read_csv("reports/context_ablation.csv")
c = ctx[ctx.lead.isin((1, 7, 30))].groupby(["base_model", "context", "lead"]).CRPSS_vs_clim.mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
for i, m in enumerate(sorted(c.base_model.unique())):
    for lead, ls in [(1, "-"), (7, "--"), (30, ":")]:
        d = c[(c.base_model == m) & (c.lead == lead)].sort_values("context")
        axes[0].plot(d.context, d.CRPSS_vs_clim, ls, color=CAT[i], lw=1.8,
                     label=f"{LABEL.get(m, m)}, lead {lead}" if lead != 7 else None)
axes[0].set_xscale("log", base=2)
axes[0].axhline(0, color=INK2, lw=1)
axes[0].axvline(365, color=MUTED, lw=1, ls="--")
_lo, _hi = axes[0].get_ylim()  # empty band between the lead-1 and lead-30 curves
axes[0].text(378, _lo + 0.80 * (_hi - _lo), "one year", color=MUTED, fontsize=7, va="center")
axes[0].set_xlabel("context length (days)")
axes[0].set_ylabel("mean CRPSS over 16 tasks")
axes[0].set_title("skill vs available history")
# bottom-right is the only empty region; "best" placement puts the legend on top of the lead-30 curves
axes[0].legend(fontsize=6.2, ncol=1, loc="lower right", handlelength=1.6, borderaxespad=0.4)
axes[0].grid(axis="y")
gap = ctx[(ctx["var"] == "Tavg") & (ctx.lead == 30) & ctx.track.isin(["ghcn_bangladesh", "ghcn_temperate"])]
gap = gap.pivot_table(index=["base_model", "context"], columns="track", values="CRPSS_vs_clim")
gap["gap"] = gap.ghcn_temperate - gap.ghcn_bangladesh
for i, m in enumerate(sorted(gap.index.get_level_values(0).unique())):
    d = gap.loc[m].reset_index().sort_values("context")
    axes[1].plot(d.context, d.gap, color=CAT[i], lw=2, marker="o", ms=3.5, label=LABEL.get(m, m))
axes[1].axhline(0, color=INK2, lw=1)
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("context length (days)")
axes[1].set_ylabel("temperate − Bangladesh skill")
axes[1].set_title("tropical gap vs history (control unmasked)")
axes[1].legend(fontsize=7)
axes[1].grid(axis="y")
fig.suptitle("How much history a zero-shot model needs", x=0.01, ha="left", fontsize=10,
             fontweight="semibold", color=INK)
fig.text(0.01, -0.02, "Right panel compares unmasked runs; with a gap-matched control the penalty does not shrink "
         "with context (week5 report, S8).", fontsize=6.5, color=MUTED, ha="left")
fig.tight_layout()
save(fig, "F9_context_length")
table(c, "F9_context")

# ---------- F10 monsoon regimes ----------
ph = pd.read_csv("reports/leaderboard_bmd_phase.csv")
clim = ph[ph.model == "climatology"][["var", "lead", "monsoon_phase", "sCRPS"]].rename(columns={"sCRPS": "clim"})
ph = ph.merge(clim, on=["var", "lead", "monsoon_phase"])
ph["CRPSS"] = 1 - ph.sCRPS / ph.clim
order = ["pre_monsoon", "onset", "peak", "withdrawal", "dry"]
models = ["chronos2", "tirex", "timesfm25", "toto2", "PatchTST", "NHITS"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
sub = ph[(ph["var"] == "Temperature") & (ph.lead == 30) & ph.model.isin(models)]
piv = sub.pivot_table(index="model", columns="monsoon_phase", values="CRPSS").reindex(models)[order]
im = axes[0].imshow(piv.to_numpy(), cmap=SEQ, aspect="auto", vmin=-0.2, vmax=0.2)
axes[0].set_xticks(range(len(order)), [o.replace("_", " ") for o in order], fontsize=7, rotation=30, ha="right")
axes[0].set_yticks(range(len(piv)), [LABEL[m] for m in piv.index], fontsize=7)
for (i, j), v in np.ndenumerate(piv.to_numpy()):
    axes[0].text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                 color=INK if abs(v) < 0.12 else "#ffffff")
axes[0].set_title("temperature, lead 30 d, by monsoon phase")
sp = pd.read_csv("reports/significance_spells.csv")
sp = sp[(sp["var"] == "Temperature") & (sp.lead == 30)]
mods = [m for m in models if m in set(sp.model)]
y = np.arange(len(mods))
for off, spell, col in [(-0.16, "active", S2), (0.16, "break", S1)]:
    d = sp[sp.spell == spell].set_index("model").reindex(mods)
    axes[1].errorbar(d.CRPSS, y + off, xerr=[d.CRPSS - d.ci_lo, d.ci_hi - d.CRPSS], fmt="o", ms=4, lw=1.6,
                     color=col, capsize=2, label=f"{spell} spell")
axes[1].axvline(0, color=INK2, lw=1)
axes[1].set_yticks(y, [LABEL[m] for m in mods], fontsize=7)
axes[1].invert_yaxis()
axes[1].set_xlabel("CRPSS vs climatology (95% CI, blocked by spell event)")
axes[1].set_title("July–August active vs break spells")
axes[1].legend(fontsize=7, loc="lower right")
axes[1].grid(axis="x")
fig.suptitle("Skill is regime-dependent: strong in break spells, negative in active spells", x=0.01,
             ha="left", fontsize=10, fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "F10_monsoon_regimes")
table(sp, "F10_spells")

# ---------- F11 compute vs skill Pareto ----------
cost = pd.read_csv("reports/compute_cost.csv")
mean_skill = board[(board.lead == 7) & (~board.track.str.startswith("nasa"))].groupby("model").CRPSS_vs_clim.mean()
cost["skill"] = cost.model.map(mean_skill)
fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.scatter(cost.daily_minutes_170k_windows, cost.skill, s=np.sqrt(cost.params_M) * 14 + 14, color=S1, alpha=0.85,
           edgecolor="#ffffff", linewidth=0.8, zorder=3)
# Labels are placed away from the cluster: below-right for the dense mid-range, above-left for the extremes.
offsets = {"chronos2": (5, 6), "tirex": (5, -10), "timesfm25": (6, 4), "toto2": (6, -11), "moirai2": (-6, 8),
           "chronos_bolt": (-14, -12), "ttm_r2": (6, 2), "sundial": (-8, 8)}
for _, r in cost.iterrows():
    dx, dy = offsets.get(r.model, (5, 5))
    ax.annotate(LABEL.get(r.model, r.model), (r.daily_minutes_170k_windows, r.skill), fontsize=6.8, color=INK2,
                xytext=(dx, dy), textcoords="offset points", ha="right" if dx < 0 else "left")
best_bl = float(mean_skill.get("PatchTST", np.nan))
ax.axhline(best_bl, color=S2, lw=1.4, ls="--")
ax.text(0.45, best_bl - 0.016, "best trained baseline (PatchTST)", color=S2, fontsize=7, va="top")
ax.axhline(0, color=INK2, lw=1)
ax.set_xscale("log")
ax.set_xlim(0.35, 320)
ax.set_xlabel("GPU minutes for all 170,571 daily windows (Tesla T4, log scale)")
ax.set_ylabel("mean CRPSS, observation tracks, lead 7 d")
ax.set_title("Compute vs skill (marker area scales with parameter count)", fontsize=9.5)
ax.grid(axis="y")
fig.tight_layout()
save(fig, "F11_compute_pareto")
table(cost, "F11_compute")

# ---------- F12 observations vs reanalysis ----------
ov = pd.read_csv("reports/significance_obs_vs_reanalysis.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.0))
vars_ = ["Humidity", "Rainfall", "Temperature", "Sunshine"]
width = 0.26
for i, lead in enumerate((1, 7, 30)):
    d = ov[ov.lead == lead].groupby("var").agg(mean=("reanalysis_minus_obs", "mean"), lo=("ci_lo", "mean"),
                                               hi=("ci_hi", "mean")).reindex(vars_)
    x = np.arange(len(vars_)) + (i - 1) * width
    ax.bar(x, d["mean"], width=width * 0.9, color=CAT[i], zorder=3, label=f"lead {lead} d")
    ax.errorbar(x, d["mean"], yerr=[d["mean"] - d.lo, d.hi - d["mean"]], fmt="none", ecolor=INK2, lw=1, capsize=2, zorder=4)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(np.arange(len(vars_)), vars_, fontsize=8)
ax.set_ylabel("reanalysis − observation CRPSS")
ax.set_title("Reanalysis looks more predictable than the stations it represents", fontsize=9.5)
ax.legend(fontsize=7)
ax.grid(axis="y")
fig.tight_layout()
save(fig, "F12_obs_vs_reanalysis")
table(ov, "F12_obs_vs_reanalysis")

print("wrote F6-F12 to reports/figures (pdf + png) with table views in reports/figures/data")
