"""Primary family with comparators chosen on the validation split (2011-2015) and frozen before testing.

For each task and lead in (1, 7, 30), every model's validation forecasts (data/predictions_cache_val, produced with
BWB_SPLIT=val) are scored with the same harness and training-period scales as the test set. The foundation model and
the baseline with the lowest mean scaled CRPS on validation windows are frozen, and that pair is then tested on the
2016-2023 test windows (reports/scores) with the same Diebold-Mariano test (HAC lag h-1, HLN correction) and Holm
adjustment over the 48 comparisons. No test-period number enters the selection.

LightGBM stops early on the validation split and takes its quantiles from validation residuals, so its validation
scores are optimistic. The selection is therefore repeated with LightGBM excluded from the candidates
(`*_nolgbm` columns); if the two disagree, the version without LightGBM is the clean one.

Output: reports/significance_valselect.csv
Usage: PYTHONPATH=src python scripts/validation_selection.py
"""
from pathlib import Path

import pandas as pd
import yaml

from bwb.data.store import TRACK_VARIABLES, load_track
from bwb.eval.harness import score, train_scales
from bwb.eval.significance import diebold_mariano, holm, per_origin_loss

LEADS = (1, 7, 30)
FMS = ["chronos2", "chronos_bolt", "timesfm25", "toto2", "tirex", "moirai2", "ttm_r2", "sundial"]
BASELINES = ["AutoETS", "AutoTheta", "AutoARIMA", "lightgbm", "NHITS", "DLinear", "PatchTST",
             "naive", "seasonal_naive", "climatology"]
VAL_CACHE = Path("data/predictions_cache_val")
TEST_SCORES = Path("reports/scores")
CFG = yaml.safe_load(open("configs/splits.yaml"))
TRAIN_END = CFG["daily"]["train"]["end"]
H = max(CFG["rolling_origin"]["daily"]["horizons_days"])


def verdict(p_holm, diff):
    return ("FM" if diff < 0 else "baseline") if p_holm < 0.05 else "neither"


def val_means(track, var, w_val, series, scales):
    """Mean scaled CRPS per model and lead on the validation windows every available model covers."""
    avail = [m for m in FMS + BASELINES if (VAL_CACHE / m / f"{track}__{var}.parquet").exists()]
    missing = sorted(set(FMS + BASELINES) - set(avail))
    preds = {m: pd.read_parquet(VAL_CACHE / m / f"{track}__{var}.parquet") for m in avail}
    common = set(w_val.window_id)
    for p in preds.values():
        common &= set(p.window_id.unique())
    wc = w_val[w_val.window_id.isin(common)]
    sub = {k: v for k, v in series.items() if k[0] == var}
    rows = []
    for m, p in preds.items():
        sc = score(p[p.window_id.isin(common)], wc, sub, scales, H)
        sc = sc[sc.lead.isin(LEADS)]
        rows += [{"model": m, "lead": lead, "val_sCRPS": g.scaled_crps.mean()} for lead, g in sc.groupby("lead")]
    return pd.DataFrame(rows), missing, len(wc)


def test_pair(a, b, track, var, w_test, lead):
    la = per_origin_loss(pd.read_parquet(TEST_SCORES / a / f"{track}__{var}.parquet"), w_test, lead)
    lb = per_origin_loss(pd.read_parquet(TEST_SCORES / b / f"{track}__{var}.parquet"), w_test, lead)
    idx = la.index.intersection(lb.index)
    return diebold_mariano(la.loc[idx].to_numpy(float), lb.loc[idx].to_numpy(float), horizon=lead)


def main():
    w_val = pd.read_parquet("data/processed/windows_daily_val.parquet")
    w_test = pd.read_parquet("data/processed/windows_daily.parquet")
    primary = pd.read_csv("reports/significance_pairwise.csv")
    rows = []
    for track, variables in TRACK_VARIABLES.items():
        series = load_track(track)
        scales = train_scales(series, TRAIN_END)
        for var in variables:
            wv = w_val[(w_val.track == track) & (w_val["var"] == var)]
            wt = w_test[(w_test.track == track) & (w_test["var"] == var)]
            means, missing, n_val = val_means(track, var, wv, series, scales)
            if missing:
                print(f"WARNING {track} {var}: no validation forecasts for {missing}")
            for lead in LEADS:
                m = means[means.lead == lead].set_index("model").val_sCRPS
                fm = m[m.index.isin(FMS)].idxmin()
                bl = m[m.index.isin(BASELINES)].idxmin()
                bl_nolgbm = m[m.index.isin([b for b in BASELINES if b != "lightgbm"])].idxmin()
                dm = test_pair(fm, bl, track, var, wt, lead)
                dm2 = dm if bl_nolgbm == bl else test_pair(fm, bl_nolgbm, track, var, wt, lead)
                ref = primary[(primary.track == track) & (primary["var"] == var) & (primary.lead == lead)].iloc[0]
                rows.append({"track": track, "var": var, "lead": lead, "n_val_windows": n_val,
                             "best_fm": fm, "best_baseline": bl, "best_baseline_nolgbm": bl_nolgbm,
                             "same_pair_as_primary": (fm == ref.best_fm) and (bl == ref.best_baseline),
                             "mean_sCRPS_diff": dm.mean_diff, "DM_p": dm.p_value,
                             "mean_sCRPS_diff_nolgbm": dm2.mean_diff, "DM_p_nolgbm": dm2.p_value,
                             "primary": verdict(ref.DM_p_holm, ref.mean_sCRPS_diff)})
    out = pd.DataFrame(rows)
    out["key"] = out.track + "|" + out["var"] + "|" + out.lead.astype(str)
    for sfx in ("", "_nolgbm"):
        out["DM_p_holm" + sfx] = out.key.map(holm(dict(zip(out.key, out["DM_p" + sfx]))))
        out["valselect" + sfx] = [verdict(p, d) for p, d in zip(out["DM_p_holm" + sfx], out["mean_sCRPS_diff" + sfx])]
    out.drop(columns="key").to_csv("reports/significance_valselect.csv", index=False)
    print(out[["track", "var", "lead", "best_fm", "best_baseline", "best_baseline_nolgbm", "mean_sCRPS_diff",
               "DM_p_holm", "valselect", "valselect_nolgbm", "primary"]].round(4).to_string(index=False))
    for sfx, name in (("", "validation-selected"), ("_nolgbm", "validation-selected, LightGBM excluded")):
        c = out["valselect" + sfx].value_counts()
        print(f"\n{name}, Holm: FM better {c.get('FM', 0)} | baseline better {c.get('baseline', 0)} | "
              f"not distinguishable {c.get('neither', 0)}")
    print("same pair as the test-selected primary:", int(out.same_pair_as_primary.sum()), "of", len(out))
    print(pd.crosstab(out.primary, out.valselect, rownames=["test-selected"], colnames=["validation-selected"]))


if __name__ == "__main__":
    main()
