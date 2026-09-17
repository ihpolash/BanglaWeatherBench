"""Spatial crosswalk: BMD stations, GHCN Bangladesh stations and NASA POWER districts -> admin1/admin2 PCODEs
(HDX COD-AB v03), plus each station's nearest NASA POWER district centroid."""
import numpy as np
import pandas as pd
import geopandas as gpd

from bwb.data.ghcn import BG_TO_BMD

UTM = "EPSG:32646"  # UTM zone 46N covers Bangladesh
adm2 = gpd.read_file("data/external/boundaries/bgd_admin2.geojson")[["adm2_name", "adm2_pcode", "adm1_name", "adm1_pcode", "geometry"]]

bmd = pd.read_csv("data/processed/bmd_station_coords.csv").assign(track="bmd", station_id=lambda d: d.station)
gh = pd.read_fwf("data/external/ghcnd/ghcnd-stations.txt", colspecs=[(0, 11), (12, 20), (21, 30), (41, 71)], names=["station_id", "lat", "lon", "source_name"])
gh = gh[gh.station_id.isin(BG_TO_BMD)].assign(track="ghcn_bangladesh", station=lambda d: d.station_id.map(BG_TO_BMD))
npw = pd.read_csv("data/All_Districts_Combined.csv", usecols=["Region", "Latitude", "Longitude"]).drop_duplicates()
npw = npw.rename(columns={"Region": "station", "Latitude": "lat", "Longitude": "lon"}).assign(track="nasa_power_district", station_id=lambda d: d.station)

pts = pd.concat([bmd[["track", "station_id", "station", "lat", "lon"]], gh[["track", "station_id", "station", "lat", "lon"]], npw[["track", "station_id", "station", "lat", "lon"]]], ignore_index=True)
g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs="EPSG:4326").to_crs(UTM)
a = adm2.to_crs(UTM)
within = gpd.sjoin(g, a, how="left", predicate="within").drop(columns="index_right")
miss = within.adm2_pcode.isna()
within["match"] = np.where(miss, "nearest", "within")
within["dist_to_polygon_km"] = 0.0
if miss.any():
    near = gpd.sjoin_nearest(g[miss.to_numpy()], a, how="left", distance_col="d").drop(columns="index_right")
    for c in ["adm2_name", "adm2_pcode", "adm1_name", "adm1_pcode"]:
        within.loc[miss, c] = near[c].to_numpy()
    within.loc[miss, "dist_to_polygon_km"] = (near["d"] / 1000).round(2).to_numpy()

# Nearest NASA POWER district centroid for every observation station
dist = within[within.track == "nasa_power_district"]
obs = within[within.track != "nasa_power_district"].copy()
d = np.sqrt(((obs.geometry.x.to_numpy()[:, None] - dist.geometry.x.to_numpy()[None]) ** 2) + ((obs.geometry.y.to_numpy()[:, None] - dist.geometry.y.to_numpy()[None]) ** 2)) / 1000
obs["nearest_nasa_district"] = dist.station.to_numpy()[d.argmin(1)]
obs["nearest_nasa_district_km"] = d.min(1).round(1)
out = pd.concat([obs, within[within.track == "nasa_power_district"]], ignore_index=True).drop(columns="geometry")
out.to_csv("data/processed/crosswalk_spatial.csv", index=False)

print(out.groupby(["track", "match"]).size().to_string())
print("\nnearest-polygon fallbacks:\n", out[out.match == "nearest"][["track", "station", "adm2_name", "dist_to_polygon_km"]].to_string(index=False))
print("\nstations whose nearest NASA POWER district is > 25 km away:")
print(obs[obs.nearest_nasa_district_km > 25][["track", "station", "nearest_nasa_district", "nearest_nasa_district_km"]].to_string(index=False))
print("\nadm2 districts containing a BMD station:", out[out.track == "bmd"].adm2_pcode.nunique(), "of 64")
