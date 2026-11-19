"""Emit the paper's generated tables as LaTeX from the stored scores.

Regenerate with:  uv run python scripts/make_leaderboard_table.py
Writes, under paper/tables/:
  leaderboard.tex   main leaderboard (inputs into sections/05_results.tex)
  comparators.tex   best FM and best baseline chosen on the validation split per task and lead, with the Holm-adjusted DM
                    result on the test split and the outcome when the pair is chosen on the test windows
  evidence.tex      evidence status of every kind of statistical statement (inputs into sections/06_discussion.tex)

Ranks come from reports/significance_ranks.csv so that the table, the
critical-difference figure and the prose cannot drift apart: all three read the
same Friedman/Nemenyi output. "Overall" is the mean of the three per-lead ranks,
which is the figure the prose quotes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FMS = ["chronos2", "chronos_bolt", "timesfm25", "toto2", "tirex", "moirai2", "ttm_r2", "sundial"]
TRAINED = ["PatchTST", "NHITS", "AutoARIMA", "lightgbm", "DLinear", "AutoETS", "AutoTheta"]
NAIVE = ["climatology", "naive", "seasonal_naive"]
LABEL = {"chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt", "timesfm25": "TimesFM 2.5",
         "toto2": "Toto-2.0", "tirex": "TiRex", "moirai2": "Moirai-2", "ttm_r2": "TTM r2",
         "sundial": "Sundial", "PatchTST": "PatchTST", "NHITS": "NHITS", "AutoARIMA": "AutoARIMA",
         "lightgbm": "LightGBM", "DLinear": "DLinear", "AutoETS": "AutoETS", "AutoTheta": "AutoTheta",
         "climatology": "Climatology", "naive": "Persistence", "seasonal_naive": "Seasonal naive"}

rk = pd.read_csv("reports/significance_ranks.csv").pivot_table(index="model", columns="lead", values="avg_rank")
board = pd.read_csv("reports/leaderboard_daily.csv")
board = board[~board.model.str.endswith("_gapmatched")]
board["kind"] = board.track.str.startswith("nasa").map({True: "reanalysis", False: "station"})
cr = board[board.lead == 7].pivot_table(index="model", columns="kind", values="CRPSS_vs_clim")
cost = pd.read_csv("reports/compute_cost.csv").set_index("model")


def overall(m):
    return (rk.loc[m, 1] + rk.loc[m, 7] + rk.loc[m, 30]) / 3


def row(m):
    params = f"\\num{{{cost.params_M[m]:.1f}}}" if m in cost.index else "n/a"
    return (f"    {LABEL[m]} & {params} & \\num{{{rk.loc[m, 1]:.2f}}} & \\num{{{rk.loc[m, 7]:.2f}}} & "
            f"\\num{{{rk.loc[m, 30]:.2f}}} & \\num{{{overall(m):.2f}}} & "
            f"\\num{{{cr.loc[m, 'station']:+.3f}}} & \\num{{{cr.loc[m, 'reanalysis']:+.3f}}} \\\\")


out = [
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \footnotesize",
    r"  \setlength{\tabcolsep}{4.5pt}",
    r"  \caption{Main leaderboard over the 16 daily tasks. Ranks are Friedman average ranks across the tasks at "
    r"each lead (1 is best, 18 models); \emph{Overall} is the mean of the three. CRPSS is mean skill against the "
    r"same track's day-of-year climatology at a 7-day lead, so a positive value beats climatology and zero is "
    r"climatology, averaged over the 8 station-observation tasks and the 8 reanalysis tasks separately. The "
    r"Nemenyi critical difference is 6.58, so rank gaps smaller than that are not separable. Parameter counts "
    r"apply to the zero-shot models; the baselines are trained on each track's training split. TTM r2 is point-only and is scored on "
    r"point skill.}",
    r"  \label{tab:leaderboard}",
    r"  \begin{tabular}{lrrrrrrr}",
    r"    \toprule",
    r"    & Params & \multicolumn{4}{c}{Average rank (1 = best)} & \multicolumn{2}{c}{Mean CRPSS, lead 7} \\",
    r"    \cmidrule(lr){3-6} \cmidrule(lr){7-8}",
    r"    Model & (M) & Lead 1 & Lead 7 & Lead 30 & Overall & Station & Reanalysis \\",
    r"    \midrule",
    r"    \multicolumn{8}{l}{\emph{Zero-shot foundation models}} \\",
]
out += [row(m) for m in sorted(FMS, key=overall)]
out += [r"    \addlinespace", r"    \multicolumn{8}{l}{\emph{Trained and statistical baselines}} \\"]
out += [row(m) for m in sorted(TRAINED, key=overall)]
out += [r"    \addlinespace", r"    \multicolumn{8}{l}{\emph{Naive references}} \\"]
out += [row(m) for m in sorted(NAIVE, key=overall)]
out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

Path("paper/tables/leaderboard.tex").write_text("\n".join(out) + "\n")
print(f"wrote paper/tables/leaderboard.tex ({len(FMS) + len(TRAINED) + len(NAIVE)} models)")
print("prose check: chronos2 overall %.2f (paper says 2.77), PatchTST %.2f (paper says 6.96)"
      % (overall("chronos2"), overall("PatchTST")))


# ---------- comparators: which model was selected on each side of the 48 primary comparisons ----------
TRACK = {"bmd": "BMD", "ghcn_bangladesh": "GHCN-BD", "ghcn_temperate": "GHCN-Temp.",
         "nasa_power_bmd": "POWER at BMD", "nasa_power_ghcn_bangladesh": "POWER at GHCN-BD",
         "nasa_power_ghcn_temperate": "POWER at GHCN-Temp."}
VAR = {"Rainfall": "Rainfall", "Temperature": "Temperature", "Humidity": "Humidity", "Sunshine": "Sunshine",
       "Tavg": "Mean temp.", "PRECTOTCORR": "Rainfall", "T2M": "Temperature", "RH2M": "Humidity",
       "ALLSKY_SFC_SW_DWN": "Shortwave"}
pair = pd.read_csv("reports/significance_pairwise.csv")  # comparators chosen on the test windows
val = pd.read_csv("reports/significance_valselect.csv")  # chosen on 2011-2015; written by scripts/validation_selection.py
order = list(TRACK)
for d in (pair, val):
    d["o"] = d.track.map(order.index)
pair = pair.sort_values(["o", "var", "lead"])
val = val.sort_values(["o", "var", "lead"])
TIE = {"FM": "FM", "baseline": "baseline", "neither": "tie"}


def outcome(r):
    if not r.significant_05:
        return "tie"
    return "FM" if r.fm_better else "baseline"


def pval(p):
    return r"$<\num{0.001}$" if p < 0.001 else f"\\num{{{p:.3f}}}"


cmp_rows = [f"    {TRACK[r.track]} & {VAR[r['var']]} & {r.lead} & {LABEL[r.best_fm]} & {LABEL[r.best_baseline]} & "
            f"\\num{{{r.mean_sCRPS_diff:+.4f}}} & {pval(r.DM_p_holm)} & {TIE[r.valselect]} & {TIE[r.primary]} \\\\"
            for _, r in val.iterrows()]
cmp = [
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \scriptsize",
    r"  \setlength{\tabcolsep}{2.5pt}",
    r"  \caption{The 48 primary comparisons. For each task and lead the best foundation model and the best "
    r"baseline (trained, statistical or naive) are the models with the lowest mean scaled CRPS on the 2011--2015 "
    r"validation windows; the pair is then frozen and tested on the 2016--2023 test windows. $\Delta$ is the mean "
    r"per-origin test scaled-CRPS difference, foundation model minus baseline, so a negative value favours the "
    r"foundation model. $p$ is the Diebold--Mariano $p$-value with HAC variance and the "
    r"Harvey--Leybourne--Newbold correction, Holm-adjusted across all 48 rows. The last column (Test-sel.) gives the outcome "
    r"when both comparators are instead chosen on the test windows. GHCN-Temp.\ is the European "
    r"temperate reference; POWER is NASA POWER reanalysis at the named station set.}",
    r"  \label{tab:comparators}",
    r"  \begin{tabular}{llrllrrll}",
    r"    \toprule",
    r"    Track & Variable & Lead & Foundation model & Baseline & $\Delta$ & Holm $p$ & Outcome & Test-sel. \\",
    r"    \midrule",
] + cmp_rows + [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
Path("paper/tables/comparators.tex").write_text("\n".join(cmp) + "\n")

# ---------- evidence status: one row per kind of statistical statement ----------
ranks = pd.read_csv("reports/significance_ranks.csv")
skill = pd.read_csv("reports/significance_skill_ci.csv")
obs = pd.read_csv("reports/significance_obs_vs_reanalysis.csv")
eq = pd.read_csv("reports/significance_equity.csv")
eq30 = eq[(eq["var"] == "Tavg") & (eq.lead == 30)]
ctx = pd.read_csv("reports/significance_equity_ctx2048.csv")
ctx30 = ctx[(ctx["var"] == "Tavg") & (ctx.lead == 30)]
sp = pd.read_csv("reports/significance_spells.csv")
split = pd.read_csv("reports/significance_split.csv")  # written by scripts/split_sample_check.py

v_fm, v_bl, v_tie = (int((val.valselect == k).sum()) for k in ("FM", "baseline", "neither"))
n_fm = int((pair.fm_better & pair.significant_05).sum())
n_bl = int((~pair.fm_better & pair.significant_05).sum())
n_tie = int((~pair.significant_05).sum())
obs_up = int((obs.significant_05 & (obs.reanalysis_minus_obs > 0)).sum())
obs_dn = int((obs.significant_05 & (obs.reanalysis_minus_obs < 0)).sum())
obs3 = obs[obs["var"] != "Sunshine"]  # sunshine vs shortwave radiation is not a like-for-like quantity
obs3_up = int((obs3.significant_05 & (obs3.reanalysis_minus_obs > 0)).sum())
obs3_dn = int((obs3.significant_05 & (obs3.reanalysis_minus_obs < 0)).sum())
sk_up, sk_dn = int((skill.ci_lo > 0).sum()), int((skill.ci_hi < 0).sum())
ev_a, ev_b = sp[sp.spell == "active"].n_events, sp[sp.spell == "break"].n_events
max_p = ranks.friedman_p.max()

ev_rows = [
    ("Best foundation model vs best baseline (pair chosen on 2011--2015, tested on 2016--2023)",
     f"DM, HAC + HLN; Holm over {len(val)}",
     f"{v_fm} favour the foundation model, {v_bl} the baseline, {v_tie} ties; LightGBM excluded from selection: "
     f"{int((val.valselect_nolgbm == 'FM').sum())}, {int((val.valselect_nolgbm == 'baseline').sum())}, "
     f"{int((val.valselect_nolgbm == 'neither').sum())}"),
    ("Same, pair chosen on the test windows", f"DM, HAC + HLN; Holm over {len(pair)}",
     f"{n_fm} favour the foundation model, {n_bl} the baseline, {n_tie} ties"),
    ("Same, pair chosen on 2016--2019, tested on 2020--2023", f"DM, HAC + HLN; Holm over {len(split)}",
     f"{int((split.split == 'FM').sum())} favour the foundation model, {int((split.split == 'baseline').sum())} "
     f"the baseline; the test-chosen pair on the same origins: {int((split.fixed_late == 'FM').sum())} and "
     f"{int((split.fixed_late == 'baseline').sum())}"),
    ("Rank equality across 18 models", f"Friedman, {int(ranks.n_tasks.iloc[0])} tasks, per lead",
     f"rejected at every lead ($p < 10^{{{int(np.floor(np.log10(max_p))) + 1}}}$); Nemenyi CD {ranks.cd.iloc[0]:.2f}"),
    None,
    ("Reanalysis vs co-located stations", f"{len(obs)} block-bootstrap 95\\% CIs; unadjusted",
     f"{obs_up} favour reanalysis, {obs_dn} the stations, {len(obs) - obs_up - obs_dn} include zero; rainfall, "
     f"temperature, humidity: {obs3_up} of {len(obs3)} and {obs3_dn}"),
    ("Active vs break spells (BMD)", f"{len(sp)} CIs blocked by spell event; unadjusted",
     f"{int(sp.significant_05.sum())} exclude zero; {ev_a.min()}--{ev_a.max()} active and {ev_b.min()}--{ev_b.max()} break events"),
    ("Skill against climatology", f"{len(skill)} block-bootstrap 95\\% CIs; unadjusted",
     f"{sk_up} above zero, {sk_dn} below"),
    ("Tropical temperature deficit, lead 30", f"{len(eq30)} DiD CIs, control masked; unadjusted",
     f"{int(eq30.gmc_significant_05.sum())} exclude zero; 10 GHCN-BD stations"),
    ("Same, with 2{,}048-day context", f"{len(ctx30)} DiD CIs; unadjusted",
     f"{int(ctx30.gmc_significant_05.sum())} exclude zero; median {ctx30.DiD_gm_control.median():+.3f}"),
]
ev = [
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \footnotesize",
    r"  \caption{Evidence status of each kind of statistical statement in the paper. Rows above the rule are the "
    r"primary analyses, corrected for multiplicity. Rows below it are per-comparison 95\% moving-block bootstrap intervals with no "
    r"multiplicity adjustment: they are reported descriptively, and an interval that excludes zero is not a "
    r"significance claim after correction. Ordered from strongest to weakest evidence within each block.}",
    r"  \label{tab:evidence}",
    r"  \begin{tabular}{p{0.27\linewidth}p{0.29\linewidth}p{0.36\linewidth}}",
    r"    \toprule",
    r"    Finding & Test and scope & Result \\",
    r"    \midrule",
]
for r_ in ev_rows:
    ev.append(r"    \midrule" if r_ is None else f"    {r_[0]} & {r_[1]} & {r_[2]} \\\\")
ev += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
Path("paper/tables/evidence.tex").write_text("\n".join(ev) + "\n")
print("wrote paper/tables/comparators.tex (%d rows) and paper/tables/evidence.tex" % len(pair))
