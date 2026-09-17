"""CHIRPS dekadal rainfall task: windows for 64 districts, reference baselines, leaderboard.

Outputs
  data/processed/windows_dekadal.parquet
  data/predictions_cache/<model>/chirps__rfh.parquet
  reports/leaderboard_dekadal.csv   model x lead (dekads) -> MASE, sCRPS, coverage80, n_windows, CRPSS_vs_clim, context
Every model folder in the cache with a chirps__rfh.parquet is scored (baselines and zero-shot foundation models from
scripts/run_fm_dekadal.py), after the integrity checks of check_predictions. Foundation-model context lengths: the
headline run (<model>) uses the longest pre-registered context, <model>_ctx<N> the others. Exit 1 if a file fails.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

from bwb.eval.dekadal import BASELINES, build_windows, check_predictions, run, score
from bwb.models.fm_registry import REGISTRY

CFG = yaml.safe_load(open("configs/splits.yaml"))
D = CFG["dekadal"]
RO = CFG["rolling_origin"]["dekadal"]
H = max(RO["horizons_dekads"])

df = pd.read_parquet("data/processed/chirps_dekadal_adm2.parquet")
series = {pc: g.set_index("date").rfh.sort_index() for pc, g in df.groupby("PCODE")}
windows = build_windows(series, D["test"]["start"], D["test"]["end"], horizon=H, stride=RO["stride_dekads"])
windows.to_parquet("data/processed/windows_dekadal.parquet", index=False)
print(f"dekadal windows: {len(windows)} across {windows.series_id.nunique()} districts")

for name, cls in BASELINES.items():
    path = Path("data/predictions_cache") / name / "chirps__rfh.parquet"
    if not path.exists():
        preds = run(cls, series, windows, D["train"]["end"], H)
        path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(path, index=False)

HEADLINE = max(RO["context_lengths"])
rows, problems = [], []
for path in sorted(Path("data/predictions_cache").glob("*/chirps__rfh.parquet")):
    name = path.parent.name
    preds = pd.read_parquet(path)
    bad = check_predictions(preds, windows, H)
    if bad:
        problems.append((name, bad))
        continue
    sc = score(preds, windows, series, D["train"]["end"])
    sc = sc[sc.lead.isin(RO["horizons_dekads"])]
    g = sc.groupby("lead")
    m = re.fullmatch(r"(.+)_ctx(\d+)", name)
    base, context = (m.group(1), int(m.group(2))) if m else (name, HEADLINE if name in REGISTRY else None)
    rows.append(pd.DataFrame({"MASE": g.ase.mean(), "sCRPS": g.scaled_crps.mean(), "coverage80": g.hit80.mean(),
                              "n_windows": g.window_id.nunique()}).reset_index().assign(model=name, base_model=base, context=context))
board = pd.concat(rows, ignore_index=True)
clim = board[board.model == "climatology"].set_index("lead").sCRPS
board["CRPSS_vs_clim"] = 1 - board.sCRPS / board.lead.map(clim)
board.to_csv("reports/leaderboard_dekadal.csv", index=False)
pd.set_option("display.width", 250)
head = board[~board.model.str.contains("_ctx")]
print(head.pivot_table(index="model", columns="lead", values=["CRPSS_vs_clim", "MASE", "coverage80"]).round(3).to_string())
for name, bad in problems:
    print("PROBLEM:", name, bad)
sys.exit(1 if problems else 0)
