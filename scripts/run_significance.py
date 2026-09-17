"""Week-5 significance tests for the headline contrasts (point estimates come from build_leaderboard/analyze_week4).

Unit of resampling is the forecast origin, because neighbouring origins overlap (stride 7 days, horizon 30) and
stations within an origin are correlated; per-origin means collapse the station dimension, moving blocks absorb the
temporal overlap. Paired series (a model and its climatology reference, or a control baseline) are resampled with the
same blocks; separate tracks are resampled independently.

Outputs (all in reports/)
  significance_pairwise.csv   best FM vs best baseline per task x lead: DM (HAC, HLN) + Wilcoxon over stations + Holm
  significance_skill_ci.csv   CRPSS with 95% block-bootstrap CI for the headline models
  significance_equity.csv     equity DiD with CIs, control uncertainty included; gap-matched variants:
                              `DiD_gapmatched` masks the model only, `DiD_gm_control` masks the control too
                              (like-for-like, the honest version - available where <control>_gapmatched exists)
  significance_obs_vs_reanalysis.csv   reanalysis-minus-observation skill with CIs
  significance_spells.csv     active / break / normal spell skill with CIs blocked by spell event
  significance_ranks.csv      Friedman + Nemenyi average ranks and critical difference per lead
Usage: PYTHONPATH=src python scripts/run_significance.py [--n-boot 2000] [--block 8]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bwb.eval.significance import (bootstrap_ci, bootstrap_ci_multi, diebold_mariano, friedman_nemenyi, holm,
                                   per_origin_loss, per_station_loss, wilcoxon_paired)

LEADS = (1, 7, 30)
FMS = ["chronos2", "chronos_bolt", "timesfm25", "toto2", "tirex", "moirai2", "ttm_r2", "sundial"]
TRAINED = ["AutoETS", "AutoTheta", "AutoARIMA", "lightgbm", "NHITS", "DLinear", "PatchTST"]
NAIVE = ["naive", "seasonal_naive", "climatology"]
HEADLINE = ["chronos2", "tirex", "timesfm25", "toto2", "moirai2", "chronos_bolt", "PatchTST", "AutoARIMA", "NHITS"]
OBS_REANALYSIS = [("bmd", "Rainfall", "nasa_power_bmd", "PRECTOTCORR"), ("bmd", "Temperature", "nasa_power_bmd", "T2M"),
                  ("bmd", "Humidity", "nasa_power_bmd", "RH2M"), ("bmd", "Sunshine", "nasa_power_bmd", "ALLSKY_SFC_SW_DWN")]
SCORES = Path("reports/scores")


def load_scores(model, track, var):
    f = SCORES / model / f"{track}__{var}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def crpss(mat):  # columns: [model, climatology]
    return 1 - mat[:, 0].mean() / mat[:, 1].mean()


def paired(*series):
    """Align loss series on their common index and stack as columns."""
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    return np.column_stack([s.loc[idx].to_numpy(float) for s in series])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=8, help="moving-block length in origins (8 x 7 days = 56 days)")
    args = ap.parse_args()
    windows = pd.read_parquet("data/processed/windows_daily.parquet")
    board = pd.read_csv("reports/leaderboard_daily.csv")
    board = board[~board.model.str.endswith("_gapmatched")]
    labels = pd.read_parquet("data/processed/bmd_labels.parquet", columns=["station", "date", "spell"])
    boot = dict(block=args.block, n_boot=args.n_boot)

    # ---------- 1. best FM vs best baseline per task and lead ----------
    rows, pvals = [], {}
    for (track, var), g in board[board.lead.isin(LEADS)].groupby(["track", "var"]):
        w = windows[(windows.track == track) & (windows["var"] == var)]
        cache = {}
        for lead in LEADS:
            gl = g[g.lead == lead]
            fm = gl[gl.model.isin(FMS)]
            bl = gl[gl.model.isin(TRAINED + NAIVE)]
            if fm.empty or bl.empty:
                continue
            a, b = fm.loc[fm.sCRPS.idxmin()].model, bl.loc[bl.sCRPS.idxmin()].model
            for m in (a, b):
                cache.setdefault(m, load_scores(m, track, var))
            if cache[a] is None or cache[b] is None:
                continue
            la, lb = per_origin_loss(cache[a], w, lead), per_origin_loss(cache[b], w, lead)
            dm = diebold_mariano(la, lb, horizon=lead)
            wil = wilcoxon_paired(per_station_loss(cache[a], w, lead), per_station_loss(cache[b], w, lead))
            key = f"{track}|{var}|{lead}"
            pvals[key] = dm.p_value
            rows.append({"track": track, "var": var, "lead": lead, "best_fm": a, "best_baseline": b,
                         "n_origins": dm.n, "n_stations": wil.n, "mean_sCRPS_diff": dm.mean_diff,
                         "DM_stat": dm.statistic, "DM_p": dm.p_value, "wilcoxon_p": wil.p_value, "key": key})
    pair = pd.DataFrame(rows)
    if not pair.empty:
        adj = holm(pvals)
        pair["DM_p_holm"] = pair.key.map(adj)
        pair["fm_better"] = pair.mean_sCRPS_diff < 0
        pair["significant_05"] = pair.DM_p_holm < 0.05
        pair.drop(columns="key").to_csv("reports/significance_pairwise.csv", index=False)
        print(f"1. Best FM vs best baseline ({len(pair)} task-leads); Holm-adjusted DM across the family")
        print(pair[["track", "var", "lead", "best_fm", "best_baseline", "mean_sCRPS_diff", "DM_stat", "DM_p_holm",
                    "wilcoxon_p", "fm_better", "significant_05"]].round(4).to_string(index=False))
        print(f"   FM significantly better: {int((pair.fm_better & pair.significant_05).sum())} | "
              f"baseline significantly better: {int((~pair.fm_better & pair.significant_05).sum())} | "
              f"not distinguishable: {int((~pair.significant_05).sum())}")

    # ---------- 2. CRPSS with CIs for the headline models ----------
    rows = []
    for (track, var), _ in board.groupby(["track", "var"]):
        w = windows[(windows.track == track) & (windows["var"] == var)]
        clim = load_scores("climatology", track, var)
        if clim is None:
            continue
        for model in HEADLINE:
            sc = load_scores(model, track, var)
            if sc is None:
                continue
            for lead in LEADS:
                mat = paired(per_origin_loss(sc, w, lead), per_origin_loss(clim, w, lead))
                point, lo, hi = bootstrap_ci_multi({"x": mat}, lambda s: crpss(s["x"]), seed=1, **boot)
                rows.append({"track": track, "var": var, "lead": lead, "model": model, "CRPSS": point,
                             "ci_lo": lo, "ci_hi": hi, "beats_climatology_95": lo > 0})
    skill = pd.DataFrame(rows)
    skill.to_csv("reports/significance_skill_ci.csv", index=False)
    print("\n2. CRPSS with 95% moving-block bootstrap CIs (observation tracks, lead 30)")
    s30 = skill[(skill.lead == 30) & (~skill.track.str.startswith("nasa"))]
    print(s30.pivot_table(index=["track", "var"], columns="model", values="CRPSS").round(3).to_string())
    print("   beats climatology at 95% (all tracks/leads):",
          f"{int(skill.beats_climatology_95.sum())} of {len(skill)} model-task-leads")

    # ---------- 3. equity DiD with CIs ----------
    rows = []
    for var in ("Rainfall", "Tavg"):
        wb = windows[(windows.track == "ghcn_bangladesh") & (windows["var"] == var)]
        wt = windows[(windows.track == "ghcn_temperate") & (windows["var"] == var)]
        cb, ct = load_scores("climatology", "ghcn_bangladesh", var), load_scores("climatology", "ghcn_temperate", var)
        for lead in LEADS:
            ctrl = board[(board.track == "ghcn_bangladesh") & (board["var"] == var) & (board.lead == lead) & board.model.isin(TRAINED)]
            if ctrl.empty or cb is None or ct is None:
                continue
            control = ctrl.loc[ctrl.sCRPS.idxmin()].model
            sb_ctrl, st_ctrl = load_scores(control, "ghcn_bangladesh", var), load_scores(control, "ghcn_temperate", var)
            for model in FMS:
                sb, st = load_scores(model, "ghcn_bangladesh", var), load_scores(model, "ghcn_temperate", var)
                sg = load_scores(f"{model}_gapmatched", "ghcn_temperate", var)
                ctrl_gm = load_scores(f"{control}_gapmatched", "ghcn_temperate", var)
                if sb is None or st is None or sb_ctrl is None or st_ctrl is None:
                    continue
                bd = paired(per_origin_loss(sb, wb, lead), per_origin_loss(cb, wb, lead), per_origin_loss(sb_ctrl, wb, lead))
                te = paired(per_origin_loss(st, wt, lead), per_origin_loss(ct, wt, lead), per_origin_loss(st_ctrl, wt, lead))
                did = lambda s: ((crpss(s["te"][:, :2]) - crpss(s["bd"][:, :2]))
                                 - (crpss(s["te"][:, [2, 1]]) - crpss(s["bd"][:, [2, 1]])))
                point, lo, hi = bootstrap_ci_multi({"bd": bd, "te": te}, did, seed=2, **boot)
                row = {"var": var, "lead": lead, "model": model, "control": control, "DiD": point, "ci_lo": lo,
                       "ci_hi": hi, "significant_05": lo > 0 or hi < 0}
                if sg is not None:
                    tg = paired(per_origin_loss(sg, wt, lead), per_origin_loss(ct, wt, lead), per_origin_loss(st_ctrl, wt, lead))
                    p2, l2, h2 = bootstrap_ci_multi({"bd": bd, "te": tg}, did, seed=3, **boot)
                    row.update(DiD_gapmatched=p2, gm_ci_lo=l2, gm_ci_hi=h2, gm_significant_05=l2 > 0 or h2 < 0)
                    # Like-for-like: the control sees the same Bangladesh-like missingness as the model. Without this,
                    # a masked model is compared with an unmasked control, which overstates the remaining penalty.
                    row["control_gapmatched"] = ctrl_gm is not None
                    if ctrl_gm is not None:
                        tgc = paired(per_origin_loss(sg, wt, lead), per_origin_loss(ct, wt, lead),
                                     per_origin_loss(ctrl_gm, wt, lead))
                        p3, l3, h3 = bootstrap_ci_multi({"bd": bd, "te": tgc}, did, seed=6, **boot)
                        row.update(DiD_gm_control=p3, gmc_ci_lo=l3, gmc_ci_hi=h3, gmc_significant_05=l3 > 0 or h3 < 0)
                rows.append(row)
    equity = pd.DataFrame(rows)
    equity.to_csv("reports/significance_equity.csv", index=False)
    print("\n3. Equity DiD with 95% CIs (positive = model loses more skill in the tropics than the control does)")
    print(equity.round(3).to_string(index=False))
    if "DiD_gm_control" in equity:
        got = equity[equity.control_gapmatched.fillna(False)]
        print(f"   like-for-like (control also gap-matched): {len(got)} of {len(equity)} rows; "
              f"significant {int(got.gmc_significant_05.sum())}")
        print(got.groupby(["var", "lead"])[["DiD", "DiD_gapmatched", "DiD_gm_control"]].median().round(3).to_string())

    # ---------- 4. observations vs reanalysis ----------
    rows = []
    for obs_track, obs_var, re_track, re_var in OBS_REANALYSIS:
        wo = windows[(windows.track == obs_track) & (windows["var"] == obs_var)]
        wr = windows[(windows.track == re_track) & (windows["var"] == re_var)]
        co, cr = load_scores("climatology", obs_track, obs_var), load_scores("climatology", re_track, re_var)
        for model in HEADLINE:
            so, sr = load_scores(model, obs_track, obs_var), load_scores(model, re_track, re_var)
            if so is None or sr is None or co is None or cr is None:
                continue
            for lead in LEADS:
                o = paired(per_origin_loss(so, wo, lead), per_origin_loss(co, wo, lead))
                r = paired(per_origin_loss(sr, wr, lead), per_origin_loss(cr, wr, lead))
                stat = lambda s: crpss(s["r"]) - crpss(s["o"])
                point, lo, hi = bootstrap_ci_multi({"o": o, "r": r}, stat, seed=4, **boot)
                rows.append({"var": obs_var, "lead": lead, "model": model, "reanalysis_minus_obs": point,
                             "ci_lo": lo, "ci_hi": hi, "significant_05": lo > 0 or hi < 0})
    obs = pd.DataFrame(rows)
    obs.to_csv("reports/significance_obs_vs_reanalysis.csv", index=False)
    print("\n4. Reanalysis minus observation skill, with CIs (positive = reanalysis looks more predictable)")
    print(obs.groupby(["var", "lead"]).agg(mean=("reanalysis_minus_obs", "mean"), n_significant=("significant_05", "sum"),
                                           n=("significant_05", "size")).round(3).to_string())

    # ---------- 5. monsoon spells, blocked by spell event ----------
    spell_rows = []
    lab = labels.rename(columns={"station": "series_id"})
    ev = lab[["date", "spell"]].drop_duplicates().sort_values("date")
    ev["event"] = (ev.spell != ev.spell.shift()).cumsum()
    for var in ("Rainfall", "Temperature", "Humidity"):
        w = windows[(windows.track == "bmd") & (windows["var"] == var)]
        clim = load_scores("climatology", "bmd", var)
        for model in ["chronos2", "chronos_bolt", "timesfm25", "tirex", "PatchTST", "NHITS"]:
            sc = load_scores(model, "bmd", var)
            if sc is None or clim is None:
                continue
            for lead in (7, 30):
                m = sc[sc.lead == lead].merge(w[["window_id", "series_id", "origin"]], on="window_id")
                c = clim[clim.lead == lead][["window_id", "scaled_crps"]].rename(columns={"scaled_crps": "clim"})
                m = m.merge(c, on="window_id")
                m["date"] = m.origin + pd.to_timedelta(lead, unit="D")
                m = m.merge(ev[["date", "spell", "event"]], on="date", how="inner")
                for spell in ("active", "break"):
                    sub = m[m.spell == spell]
                    if sub.event.nunique() < 5:
                        continue
                    per_event = sub.groupby("event")[["scaled_crps", "clim"]].mean()
                    mat = per_event.to_numpy()
                    point, lo, hi = bootstrap_ci_multi({"x": mat}, lambda s: crpss(s["x"]), block=1,
                                                       n_boot=args.n_boot, seed=5)
                    spell_rows.append({"var": var, "lead": lead, "model": model, "spell": spell,
                                       "n_events": int(per_event.shape[0]), "CRPSS": point, "ci_lo": lo, "ci_hi": hi,
                                       "significant_05": lo > 0 or hi < 0})
    spells = pd.DataFrame(spell_rows)
    spells.to_csv("reports/significance_spells.csv", index=False)
    print("\n5. Active vs break spells (BMD), CIs blocked by spell event")
    if not spells.empty:
        print(spells.pivot_table(index=["var", "lead", "model"], columns="spell",
                                 values=["CRPSS", "ci_lo", "ci_hi"]).round(3).to_string())

    # ---------- 6. Friedman + Nemenyi across the 16 tasks ----------
    rank_rows = []
    for lead in LEADS:
        piv = board[board.lead == lead].pivot_table(index=["track", "var"], columns="model", values="sCRPS")
        piv = piv.dropna(axis=1, how="any")
        out = friedman_nemenyi(piv)
        print(f"\n6. Friedman across {out['n_tasks']} tasks x {out['n_models']} models at lead {lead}: "
              f"chi2={out['chi2']:.1f} p={out['p_value']:.2e} | Nemenyi CD={out['cd']:.2f}")
        print("   average ranks:", ", ".join(f"{m} {r:.2f}" for m, r in out["avg_ranks"].items()))
        for model, rank in out["avg_ranks"].items():
            rank_rows.append({"lead": lead, "model": model, "avg_rank": rank, "cd": out["cd"],
                              "n_tasks": out["n_tasks"], "n_models": out["n_models"],
                              "friedman_chi2": out["chi2"], "friedman_p": out["p_value"]})
    pd.DataFrame(rank_rows).to_csv("reports/significance_ranks.csv", index=False)


if __name__ == "__main__":
    main()
