"""Does a longer context remove the tropical temperature penalty, or does it just hide gappier context?

Week-4 R7 showed the tropical-vs-temperate Tavg gap at lead 30 shrinks sharply from 1,024 to 2,048 days of context;
Week-5 S2 showed that at 1,024 days most of the penalty is explained by context missingness. The two explanations are
confounded, because a Bangladesh context at 1,024 days is both shorter in real observations and gappier.

This script combines the Week-4b 2,048-day runs (scored on Kaggle, reports/ablation/scores/<model>_ctx2048/) with the
Week-5 gap-matched 2,048-day runs (notebook ipolas/bwb-fm-gm2048) and recomputes the equity DiD at 2,048 days, with
the control masked exactly as the model is. Trained baselines keep their own fixed context (730 days statistical,
365 days neural), so the context change applies to the foundation models only - stated in the report.

Usage: PYTHONPATH=src python scripts/analyze_gm2048.py [--n-boot 2000] [--block 8]
Writes reports/significance_equity_ctx2048.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bwb.eval.significance import bootstrap_ci_multi, per_origin_loss

MODELS = ["chronos2", "tirex", "timesfm25"]
TRAINED = ["AutoETS", "AutoTheta", "AutoARIMA", "lightgbm", "NHITS", "DLinear", "PatchTST"]
LEADS = (7, 30)
SCORES, ABL = Path("reports/scores"), Path("reports/ablation/scores")


def load(root: Path, model: str, track: str, var: str):
    f = root / model / f"{track}__{var}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def crpss(mat):
    return 1 - mat[:, 0].mean() / mat[:, 1].mean()


def paired(*series):
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    return np.column_stack([s.loc[idx].to_numpy(float) for s in series])


def did(s):
    return ((crpss(s["te"][:, :2]) - crpss(s["bd"][:, :2])) - (crpss(s["te"][:, [2, 1]]) - crpss(s["bd"][:, [2, 1]])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=8)
    args = ap.parse_args()
    boot = dict(block=args.block, n_boot=args.n_boot)
    windows = pd.read_parquet("data/processed/windows_daily.parquet")
    board = pd.read_csv("reports/leaderboard_daily.csv")
    board = board[~board.model.str.endswith("_gapmatched")]
    at1024 = pd.read_csv("reports/significance_equity.csv")

    rows = []
    for var in ("Tavg", "Rainfall"):
        wb = windows[(windows.track == "ghcn_bangladesh") & (windows["var"] == var)]
        wt = windows[(windows.track == "ghcn_temperate") & (windows["var"] == var)]
        cb, ct = load(SCORES, "climatology", "ghcn_bangladesh", var), load(SCORES, "climatology", "ghcn_temperate", var)
        for lead in LEADS:
            g = board[(board.track == "ghcn_bangladesh") & (board["var"] == var) & (board.lead == lead) & board.model.isin(TRAINED)]
            if g.empty or cb is None or ct is None:
                continue
            control = g.loc[g.sCRPS.idxmin()].model
            cb_ctrl, ct_ctrl = load(SCORES, control, "ghcn_bangladesh", var), load(SCORES, control, "ghcn_temperate", var)
            ct_ctrl_gm = load(SCORES, f"{control}_gapmatched", "ghcn_temperate", var)
            for model in MODELS:
                mb = load(ABL, f"{model}_ctx2048", "ghcn_bangladesh", var)
                mt = load(ABL, f"{model}_ctx2048", "ghcn_temperate", var)
                mg = load(ABL, f"{model}_ctx2048_gapmatched", "ghcn_temperate", var)
                if any(x is None for x in (mb, mt, cb_ctrl, ct_ctrl)):
                    print(f"skip {model} {var} lead {lead}: missing 2,048-day scores")
                    continue
                bd = paired(per_origin_loss(mb, wb, lead), per_origin_loss(cb, wb, lead), per_origin_loss(cb_ctrl, wb, lead))
                te = paired(per_origin_loss(mt, wt, lead), per_origin_loss(ct, wt, lead), per_origin_loss(ct_ctrl, wt, lead))
                point, lo, hi = bootstrap_ci_multi({"bd": bd, "te": te}, did, seed=7, **boot)
                row = {"var": var, "lead": lead, "model": model, "control": control, "context": 2048,
                       "DiD": point, "ci_lo": lo, "ci_hi": hi, "significant_05": lo > 0 or hi < 0}
                if mg is not None and ct_ctrl_gm is not None:
                    tg = paired(per_origin_loss(mg, wt, lead), per_origin_loss(ct, wt, lead), per_origin_loss(ct_ctrl_gm, wt, lead))
                    p2, l2, h2 = bootstrap_ci_multi({"bd": bd, "te": tg}, did, seed=8, **boot)
                    row.update(DiD_gm_control=p2, gmc_ci_lo=l2, gmc_ci_hi=h2, gmc_significant_05=l2 > 0 or h2 < 0)
                prev = at1024[(at1024["var"] == var) & (at1024.lead == lead) & (at1024.model == model)]
                if len(prev):
                    row.update(DiD_at_1024=float(prev.DiD.iloc[0]), DiD_gm_control_at_1024=float(prev.DiD_gm_control.iloc[0]))
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("reports/significance_equity_ctx2048.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nEquity DiD at 2,048-day context (positive = model loses more skill in the tropics than the control)")
    cols = [c for c in ["var", "lead", "model", "control", "DiD_at_1024", "DiD", "ci_lo", "ci_hi", "significant_05",
                        "DiD_gm_control_at_1024", "DiD_gm_control", "gmc_ci_lo", "gmc_ci_hi", "gmc_significant_05"] if c in out]
    print(out[cols].round(3).to_string(index=False))
    if "DiD_gm_control" in out:
        print("\nmedians by var x lead (1,024 -> 2,048, both masked):")
        print(out.groupby(["var", "lead"])[["DiD_at_1024", "DiD", "DiD_gm_control_at_1024", "DiD_gm_control"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
