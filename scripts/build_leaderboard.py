"""Score every cached model on the daily test windows and write leaderboards.

Outputs
  reports/leaderboard_daily.csv       track, var, model, lead -> MASE, sCRPS, RMSE, coverage80, n_windows
  reports/leaderboard_bmd_phase.csv   BMD only: model x var x lead x monsoon phase of the target day
  reports/scores/<model>/<track>__<var>.parquet   per window x lead scores (for significance tests in Week 5)
Only windows every listed model has forecast are compared (common-window rule), so rankings are like for like.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from bwb.data.store import TRACK_VARIABLES, load_track
from bwb.eval.harness import CACHE, aggregate, score, train_scales

CFG = yaml.safe_load(open("configs/splits.yaml"))
TRAIN_END = CFG["daily"]["train"]["end"]
H = max(CFG["rolling_origin"]["daily"]["horizons_days"])
LEADS = tuple(CFG["rolling_origin"]["daily"]["horizons_days"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list; default = every model directory in the cache")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else sorted(p.name for p in CACHE.iterdir() if p.is_dir())
    windows = pd.read_parquet("data/processed/windows_daily.parquet")
    labels = pd.read_parquet("data/processed/bmd_labels.parquet", columns=["station", "date", "monsoon_phase", "spell"])

    boards, phase_rows = [], []
    for track, variables in TRACK_VARIABLES.items():
        series = load_track(track)
        scales = train_scales(series, TRAIN_END)
        for var in variables:
            avail = [m for m in models if (CACHE / m / f"{track}__{var}.parquet").exists()]
            if not avail:
                continue
            w = windows[(windows.track == track) & (windows["var"] == var)]
            preds = {m: pd.read_parquet(CACHE / m / f"{track}__{var}.parquet") for m in avail}
            common = set(w.window_id)
            for p in preds.values():
                common &= set(p.window_id.unique())
            wc = w[w.window_id.isin(common)]
            sub = {k: v for k, v in series.items() if k[0] == var}
            for m, p in preds.items():
                sc = score(p[p.window_id.isin(common)], wc, sub, scales, H)
                out = Path("reports/scores") / m / f"{track}__{var}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                sc.to_parquet(out, index=False)
                boards.append(aggregate(sc, wc, leads=LEADS).assign(model=m, n_models_compared=len(avail)))
                if track == "bmd":
                    t = sc.merge(wc[["window_id", "series_id", "origin"]], on="window_id")
                    t["date"] = t.origin + pd.to_timedelta(t.lead, unit="D")
                    t = t.merge(labels.rename(columns={"station": "series_id"}), on=["series_id", "date"], how="left")
                    t = t[t.lead.isin([1, 7, 30])]
                    g = t.groupby(["lead", "monsoon_phase"])
                    phase_rows.append(pd.DataFrame({"MASE": g.ase.mean(), "sCRPS": g.scaled_crps.mean(), "n": g.size()})
                                      .reset_index().assign(var=var, model=m))
            print(f"scored {track:28s} {var:18s} models={len(avail)} common_windows={len(common)}", flush=True)

    board = pd.concat(boards, ignore_index=True)
    # Cross-track comparisons use skill relative to climatology: MASE/sCRPS scales differ between tracks
    # (e.g. smooth reanalysis drizzle vs sparse GHCN pairs), but skill vs the same-track climatology does not.
    clim = board[board.model == "climatology"].set_index(["track", "var", "lead"])[["MASE", "sCRPS"]]
    ref = clim.reindex(pd.MultiIndex.from_frame(board[["track", "var", "lead"]]))
    board["skill_MAE_vs_clim"] = 1 - board["MASE"].to_numpy() / ref["MASE"].to_numpy()
    board["CRPSS_vs_clim"] = 1 - board["sCRPS"].to_numpy() / ref["sCRPS"].to_numpy()
    board.to_csv("reports/leaderboard_daily.csv", index=False)
    if phase_rows:
        pd.concat(phase_rows, ignore_index=True).to_csv("reports/leaderboard_bmd_phase.csv", index=False)
    pd.set_option("display.width", 250)
    for lead in (1, 30):
        piv = board[board.lead == lead].pivot_table(index=["track", "var"], columns="model", values="MASE").round(3)
        print(f"\nMASE at lead {lead} (lower is better):\n{piv.to_string()}")
        best = piv.idxmin(axis=1)
        print("best model per task:", best.value_counts().to_dict())
        sk = board[board.lead == lead].pivot_table(index=["track", "var"], columns="model", values="CRPSS_vs_clim").round(3)
        print(f"CRPSS vs climatology at lead {lead} (higher is better, 0 = climatology):\n{sk.to_string()}")


if __name__ == "__main__":
    main()
