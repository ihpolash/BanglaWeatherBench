"""Shortlist temperate GHCN-Daily reference stations from the same WMO/GSOD-derived network ('M')
as the 10 Bangladesh GHCN stations, and measure 2016-2023 completeness for PRCP and TAVG."""
import time, pathlib
import pandas as pd
import requests

from bwb.data.ghcn import load_ghcn_station

D = pathlib.Path("data/external/ghcnd")
inv = pd.read_fwf(D / "ghcnd-inventory.txt", colspecs=[(0,11),(12,20),(21,30),(31,35),(36,40),(41,45)], names=["id","lat","lon","elem","first","last"])
st = pd.read_fwf(D / "ghcnd-stations.txt", colspecs=[(0,11),(41,71),(72,75)], names=["id","name","gsn"], dtype=str)
p = inv[inv.elem.isin(["PRCP","TAVG"])].pivot_table(index=["id","lat","lon"], columns="elem", values=["first","last"]).dropna()
ok = p[(p["first"].max(axis=1) <= 1985) & (p["last"].min(axis=1) >= 2024)].reset_index()
ok.columns = ["_".join(map(str, c)).strip("_") for c in ok.columns]
ok = ok.merge(st, on="id", how="left")
ok = ok[(ok.id.str[2] == "M") & ok.lat.between(35, 60) & (ok.lon.abs() > 0)]
ok["cc"] = ok.id.str[:2]
# Spread across countries: up to 6 per country, GSN stations first, 90 downloads max.
ok["gsn_rank"] = ok.gsn.fillna("").str.contains("GSN").map({True: 0, False: 1})
pick = ok.sort_values(["gsn_rank", "id"]).groupby("cc").head(6).head(90)
out = D / "by_station"
rows = []
for sid in pick.id:
    f = out / f"{sid}.csv.gz"
    if not f.exists():
        r = requests.get(f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{sid}.csv.gz", timeout=120)
        if r.status_code != 200:
            continue
        f.write_bytes(r.content)
        time.sleep(0.5)
    g = load_ghcn_station(f).loc["2016-01-01":"2023-12-31"]
    rows.append({"id": sid, "prcp_pct": round(100 * g.Rainfall.notna().sum() / 2922, 1), "tavg_pct": round(100 * g.Tavg.notna().sum() / 2922, 1)})
res = pick.merge(pd.DataFrame(rows), on="id")[["id", "cc", "name", "lat", "lon", "gsn", "prcp_pct", "tavg_pct"]]
res.to_csv("reports/temperate_reference_candidates.csv", index=False)
print("downloaded/evaluated:", len(res), "countries:", res.cc.nunique())
print(res[["prcp_pct", "tavg_pct"]].describe().round(1).to_string())
for thr in (60, 65, 80, 90):
    print(f"stations with PRCP & TAVG >= {thr}% in 2016-2023:", int(((res.prcp_pct >= thr) & (res.tavg_pct >= thr)).sum()))
