"""Emit the main leaderboard table as LaTeX from the stored scores.

Regenerate with:  uv run python scripts/make_leaderboard_table.py
Writes paper/tables/leaderboard.tex, which paper/sections/05_results.tex inputs.

Ranks come from reports/significance_ranks.csv so that the table, the
critical-difference figure and the prose cannot drift apart: all three read the
same Friedman/Nemenyi output. "Overall" is the mean of the three per-lead ranks,
which is the figure the prose quotes.
"""
from pathlib import Path

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
    r"apply to the zero-shot models; the baselines are fitted per series. TTM r2 is point-only and is scored on "
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
