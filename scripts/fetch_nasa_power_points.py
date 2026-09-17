"""Fetch NASA POWER daily point data (incl. PRECTOTCORR) at every observation station location:
35 BMD stations, 10 GHCN Bangladesh stations, and the temperate GHCN reference stations.
Resumable: skips stations already saved. Local solar time days (time-standard=LST)."""
import json, pathlib, sys, time
import pandas as pd
import requests

from bwb.data.ghcn import BG_TO_BMD

PARAMS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M,T2MDEW,PS"
START, END = "19810101", "20251231"
OUT = pathlib.Path("data/external/nasa_power_points")
OUT.mkdir(parents=True, exist_ok=True)

bmd = pd.read_csv("data/processed/bmd_station_coords.csv")
pts = [("bmd", r.station, r.lat, r.lon) for r in bmd.itertuples()]
gh = pd.read_fwf("data/external/ghcnd/ghcnd-stations.txt", colspecs=[(0, 11), (12, 20), (21, 30)], names=["id", "lat", "lon"])
pts += [("ghcn", r.id, r.lat, r.lon) for r in gh[gh.id.isin(BG_TO_BMD)].itertuples()]
tr = pd.read_csv("reports/temperate_reference_stations.csv")
pts += [("ghcn", r.id, r.lat, r.lon) for r in tr.itertuples()]

failed = []
for track, key, lat, lon in pts:
    f = OUT / f"{track}_{key}.csv"
    if f.exists():
        continue
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters={PARAMS}&community=AG&longitude={lon:.4f}&latitude={lat:.4f}&start={START}&end={END}&format=JSON&time-standard=LST")
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200:
                p = r.json()["properties"]["parameter"]
                df = pd.DataFrame(p)
                df.index = pd.to_datetime(df.index, format="%Y%m%d")
                df = df.mask(df <= -999)
                df.index.name = "date"
                df.to_csv(f)
                print(f"ok {track}_{key} {len(df)} rows", flush=True)
                break
            print(f"http {r.status_code} {key}: {r.text[:200]}", flush=True)
        except Exception as e:  # network hiccup: back off and retry
            print(f"error {key}: {e}", flush=True)
        time.sleep(10 * (attempt + 1))
    else:
        failed.append(key)
    time.sleep(1.5)
print(f"done: {len(pts)} points, failed: {failed}")
sys.exit(1 if failed else 0)
