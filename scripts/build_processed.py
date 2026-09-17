"""Build processed parquet files for the observational tracks.

Outputs (data/processed/):
  bmd_daily.parquet          station, date, 4 variables, <var>_flag (suspected imputation)
  ghcn_daily.parquet         station_id, region, date, Rainfall, Tavg, Tmax, Tmin (GHCN date convention, unshifted)
  chirps_dekadal_adm2.parquet
  nasa_power_districts_daily.parquet  (legacy 38-district file, reanalysis covariates)
"""

from pathlib import Path

import pandas as pd

from bwb.data.bmd import VARIABLES, load_bmd_daily
from bwb.data.chirps import load_chirps_dekadal
from bwb.data.ghcn import BG_TO_BMD, load_ghcn_station
from bwb.data.nasa_power import load_nasa_power
from bwb.data.qc import imputation_mask

OUT = Path("data/processed")
PRECISION = {"Rainfall": 1, "Temperature": 1, "Humidity": 0, "Sunshine": 1}


def build_bmd() -> pd.DataFrame:
    df = load_bmd_daily("data/external/bmd_mendeley/combined/BD_weather.csv")
    parts = []
    for station, g in df.groupby(level="station"):
        g = g.droplevel("station")
        out = g.copy()
        for v in VARIABLES:
            out[f"{v}_flag"] = imputation_mask(g[v], v, decimals=PRECISION[v])["flag_any"].to_numpy()
        parts.append(out.reset_index().assign(station=station))
    res = pd.concat(parts, ignore_index=True)[["station", "date", *VARIABLES, *[f"{v}_flag" for v in VARIABLES]]]
    res.to_parquet(OUT / "bmd_daily.parquet", index=False)
    return res


def build_ghcn() -> pd.DataFrame:
    temperate = pd.read_csv("reports/temperate_reference_stations.csv").id.tolist()
    parts = []
    for region, ids in (("bangladesh", list(BG_TO_BMD)), ("temperate", temperate)):
        for sid in ids:
            g = load_ghcn_station(f"data/external/ghcnd/by_station/{sid}.csv.gz").asfreq("D")
            g.index.name = "date"
            parts.append(g.reset_index().assign(station_id=sid, region=region, bmd_station=BG_TO_BMD.get(sid)))
    res = pd.concat(parts, ignore_index=True)
    res = res[["station_id", "region", "bmd_station", "date", "Rainfall", "Tavg", "Tmax", "Tmin"]]
    res.to_parquet(OUT / "ghcn_daily.parquet", index=False)
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bmd = build_bmd()
    flagged = bmd[[f"{v}_flag" for v in VARIABLES]].mean().mul(100).round(2).to_dict()
    print(f"bmd_daily: {len(bmd):,} rows, {bmd.station.nunique()} stations, % flagged {flagged}")

    ghcn = build_ghcn()
    print(f"ghcn_daily: {len(ghcn):,} rows, stations per region {ghcn.groupby('region').station_id.nunique().to_dict()}")

    chirps = load_chirps_dekadal("data/bgd-rainfall-subnat-full.csv", adm_level=2)
    chirps.to_parquet(OUT / "chirps_dekadal_adm2.parquet", index=False)
    print(f"chirps_dekadal_adm2: {len(chirps):,} rows, {chirps.PCODE.nunique()} units")

    npw = load_nasa_power("data/All_Districts_Combined.csv").reset_index()
    npw.to_parquet(OUT / "nasa_power_districts_daily.parquet", index=False)
    print(f"nasa_power_districts_daily: {len(npw):,} rows, {npw.district.nunique()} districts")


if __name__ == "__main__":
    main()
