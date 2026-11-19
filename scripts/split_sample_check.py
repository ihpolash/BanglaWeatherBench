"""Split-sample check of the primary family (best FM vs best baseline per task x lead).

`run_significance.py` picks the best foundation model and the best baseline on the test windows and then tests that
pair on the same windows, so its Diebold-Mariano p-values do not account for the selection. Here the test period is
split at 2020-01-01: the pair is selected on origins in 2016-2019 (lowest mean per-origin scaled CRPS) and tested,
with the same Diebold-Mariano test (HAC lag h-1, HLN correction), on origins in 2020-2023 only. Selection and
inference then use disjoint windows. Holm adjustment is over the same 48 comparisons. As a power reference, the
published pair is also tested, unchanged, on the same 2020-2023 origins: if that loses as many verdicts as the
split-sample pair, the loss comes from halving the sample, not from selection.

Output: reports/significance_split.csv
Usage: PYTHONPATH=src python scripts/split_sample_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bwb.eval.significance import diebold_mariano, holm

LEADS = (1, 7, 30)
SPLIT = pd.Timestamp("2020-01-01")
FMS = ["chronos2", "chronos_bolt", "timesfm25", "toto2", "tirex", "moirai2", "ttm_r2", "sundial"]
BASELINES = ["AutoETS", "AutoTheta", "AutoARIMA", "lightgbm", "NHITS", "DLinear", "PatchTST",
             "naive", "seasonal_naive", "climatology"]
SCORES = Path("reports/scores")


def origin_losses(model, track, var, w):
    """Per-origin mean scaled CRPS at each primary lead, as {lead: Series indexed by origin}."""
    f = SCORES / model / f"{track}__{var}.parquet"
    if not f.exists():
        return None
    s = pd.read_parquet(f, columns=["window_id", "lead", "scaled_crps"])
    s = s[s.lead.isin(LEADS)].merge(w, on="window_id")
    return {lead: g.groupby("origin").scaled_crps.mean().sort_index() for lead, g in s.groupby("lead")}


def verdict(sig, diff):
    return ("FM" if diff < 0 else "baseline") if sig else "neither"


def main():
    windows = pd.read_parquet("data/processed/windows_daily.parquet", columns=["window_id", "track", "var", "origin"])
    primary = pd.read_csv("reports/significance_pairwise.csv")
    rows = []
    for (track, var), _ in primary.groupby(["track", "var"]):
        w = windows[(windows.track == track) & (windows["var"] == var)][["window_id", "origin"]]
        losses = {m: origin_losses(m, track, var, w) for m in FMS + BASELINES}
        losses = {m: v for m, v in losses.items() if v is not None}
        for lead in LEADS:
            mat = pd.concat({m: v[lead] for m, v in losses.items() if lead in v}, axis=1, join="inner")
            sel, test = mat[mat.index < SPLIT], mat[mat.index >= SPLIT]
            mu = sel.mean()
            a = mu[[c for c in mat.columns if c in FMS]].idxmin()
            b = mu[[c for c in mat.columns if c in BASELINES]].idxmin()
            dm = diebold_mariano(test[a].to_numpy(float), test[b].to_numpy(float), horizon=lead)
            ref = primary[(primary.track == track) & (primary["var"] == var) & (primary.lead == lead)].iloc[0]
            fixed = diebold_mariano(test[ref.best_fm].to_numpy(float), test[ref.best_baseline].to_numpy(float),
                                    horizon=lead)
            rows.append({"track": track, "var": var, "lead": lead, "n_select": len(sel), "n_test": len(test),
                         "best_fm": a, "best_baseline": b,
                         "same_pair_as_primary": (a == ref.best_fm) and (b == ref.best_baseline),
                         "mean_sCRPS_diff": dm.mean_diff, "DM_p": dm.p_value,
                         "fixed_diff": fixed.mean_diff, "fixed_p": fixed.p_value,
                         "primary": verdict(ref.significant_05, ref.mean_sCRPS_diff)})
    out = pd.DataFrame(rows)
    out["key"] = out.track + "|" + out["var"] + "|" + out.lead.astype(str)
    out["DM_p_holm"] = out.key.map(holm(dict(zip(out.key, out.DM_p))))
    out["split"] = [verdict(p < 0.05, d) for p, d in zip(out.DM_p_holm, out.mean_sCRPS_diff)]
    out["fixed_p_holm"] = out.key.map(holm(dict(zip(out.key, out.fixed_p))))
    out["fixed_late"] = [verdict(p < 0.05, d) for p, d in zip(out.fixed_p_holm, out.fixed_diff)]
    out.drop(columns="key").to_csv("reports/significance_split.csv", index=False)
    print(out[["track", "var", "lead", "best_fm", "best_baseline", "mean_sCRPS_diff", "DM_p_holm", "split",
               "primary"]].round(4).to_string(index=False))
    c = out.split.value_counts()
    print(f"\nsplit-sample, Holm: FM better {c.get('FM', 0)} | baseline better {c.get('baseline', 0)} | "
          f"not distinguishable {c.get('neither', 0)}")
    f = out.fixed_late.value_counts()
    print(f"published pair on the same 2020-2023 origins: FM better {f.get('FM', 0)} | "
          f"baseline better {f.get('baseline', 0)} | not distinguishable {f.get('neither', 0)}")
    won = out[out.primary != "neither"]
    same_sign = ((won.primary == "FM") == (won.mean_sCRPS_diff < 0)).sum()
    print(f"primary comparisons with a winner whose held-out difference keeps the sign: {same_sign} of {len(won)}")
    print("same pair selected as primary:", int(out.same_pair_as_primary.sum()), "of", len(out))
    print("verdict changes vs primary:", int((out.split != out.primary).sum()), "of", len(out))
    print(pd.crosstab(out.primary, out.split, rownames=["primary"], colnames=["split"]))
    print("median origins: select", int(np.median(out.n_select)), "test", int(np.median(out.n_test)))


if __name__ == "__main__":
    main()
