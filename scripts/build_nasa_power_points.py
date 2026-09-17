"""Combine NASA POWER point downloads into one parquet and compare reanalysis with BMD observations."""
import glob, pathlib
import numpy as np
import pandas as pd

parts = []
for f in sorted(glob.glob("data/external/nasa_power_points/*.csv")):
    track, key = pathlib.Path(f).stem.split("_", 1)
    d = pd.read_csv(f, parse_dates=["date"]).assign(track=track, station_id=key)
    parts.append(d)
npw = pd.concat(parts, ignore_index=True)
temperate = set(pd.read_csv("reports/temperate_reference_stations.csv").id)
npw["region"] = np.where(npw.station_id.isin(temperate), "temperate", "bangladesh")
cols = ["track", "station_id", "region", "date"] + [c for c in npw.columns if c not in ("track", "station_id", "region", "date")]
npw = npw[cols]
npw.to_parquet("data/processed/nasa_power_points.parquet", index=False)
print(f"nasa_power_points: {len(npw):,} rows, {npw.station_id.nunique()} points, {npw.date.min().date()}..{npw.date.max().date()}")
print("missing % by variable:", npw.drop(columns=cols[:4]).isna().mean().mul(100).round(2).to_dict())

# Reanalysis vs observation at BMD stations, 2016-2023 test period, clean BMD values only
bmd = pd.read_parquet("data/processed/bmd_daily.parquet")
bmd = bmd[bmd.date.between("2016-01-01", "2023-12-31")]
j = bmd.merge(npw[npw.track == "bmd"].rename(columns={"station_id": "station"}), on=["station", "date"])
rows = []
for st, g in j.groupby("station"):
    r = g[~g.Rainfall_flag][["Rainfall", "PRECTOTCORR"]].dropna()
    t = g[~g.Temperature_flag][["Temperature", "T2M"]].dropna()
    h = g[~g.Humidity_flag][["Humidity", "RH2M"]].dropna()
    best = max((-1, 0, 1), key=lambda k: r.Rainfall.corr(r.PRECTOTCORR.shift(k)))
    rows.append({"station": st,
                 "rain_r": r.Rainfall.corr(r.PRECTOTCORR), "rain_best_lag": best,
                 "rain_r_best": r.Rainfall.corr(r.PRECTOTCORR.shift(best)),
                 "rain_total_ratio": r.PRECTOTCORR.sum() / r.Rainfall.sum(),
                 "wet_agree": ((r.Rainfall >= 1) == (r.PRECTOTCORR >= 1)).mean(),
                 "heavy50_capture": (r.PRECTOTCORR[r.Rainfall >= 50] >= 25).mean(),
                 "temp_r": t.Temperature.corr(t.T2M), "temp_bias": (t.T2M - t.Temperature).mean(),
                 "rh_r": h.Humidity.corr(h.RH2M), "rh_bias": (h.RH2M - h.Humidity).mean()})
cmp_ = pd.DataFrame(rows).round(3)
cmp_.to_csv("reports/nasa_power_vs_bmd_test_period.csv", index=False)
print("\nNASA POWER vs BMD, 2016-2023 (median [min, max] across 35 stations):")
for c in cmp_.columns[1:]:
    print(f"  {c:18s} {cmp_[c].median():7.3f}  [{cmp_[c].min():.3f}, {cmp_[c].max():.3f}]")
print("best rain lag counts:", cmp_.rain_best_lag.value_counts().to_dict())
