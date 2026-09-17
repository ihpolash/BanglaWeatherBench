"""Station-level cyclone days from IBTrACS v04r01 (North Indian basin).

A station-day is a cyclone day when a storm fix with sustained wind >= 34 kt (tropical-storm strength,
max of WMO and USA agency winds) lies within 300 km of the station. Fix times are converted from UTC to
Bangladesh local date (UTC+6)."""
import numpy as np
import pandas as pd

RADIUS_KM, MIN_WIND_KT = 300, 34
tr = pd.read_csv("data/external/ibtracs/ibtracs.NI.list.v04r01.csv", skiprows=[1], low_memory=False,
                 usecols=["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "USA_WIND"])
tr = tr[tr.SEASON >= 1961].copy()
tr["wind_kt"] = pd.concat([pd.to_numeric(tr.WMO_WIND, errors="coerce"), pd.to_numeric(tr.USA_WIND, errors="coerce")], axis=1).max(axis=1)
tr = tr[tr.wind_kt >= MIN_WIND_KT]
tr["local_date"] = (pd.to_datetime(tr.ISO_TIME) + pd.Timedelta(hours=6)).dt.normalize()
st = pd.read_csv("data/processed/bmd_station_coords.csv")

la1, lo1 = np.radians(st.lat.to_numpy())[:, None], np.radians(st.lon.to_numpy())[:, None]
la2, lo2 = np.radians(tr.LAT.astype(float).to_numpy())[None], np.radians(tr.LON.astype(float).to_numpy())[None]
d = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
si, fi = np.nonzero(d <= RADIUS_KM)
hits = pd.DataFrame({"station": st.station.to_numpy()[si], "date": tr.local_date.to_numpy()[fi], "sid": tr.SID.to_numpy()[fi],
                     "name": tr.NAME.to_numpy()[fi], "dist_km": d[si, fi].round(1), "wind_kt": tr.wind_kt.to_numpy()[fi]})
days = hits.groupby(["station", "date"]).agg(sid=("sid", "first"), name=("name", "first"), min_dist_km=("dist_km", "min"), max_wind_kt=("wind_kt", "max")).reset_index()
days.to_parquet("data/processed/cyclone_station_days.parquet", index=False)

test = days[days.date.between("2016-01-01", "2023-12-31")]
print("station cyclone-days total:", len(days), "| storms:", days.sid.nunique(), "| in test 2016-2023:", len(test), "station-days,", test.sid.nunique(), "storms")
print(test.groupby("name").agg(stations=("station", "nunique"), days=("date", "nunique"), max_wind=("max_wind_kt", "max")).sort_values("stations", ascending=False).to_string())
print("\nstations with >= 5 test cyclone-days:", int((test.groupby("station").size() >= 5).sum()), "of 35")
