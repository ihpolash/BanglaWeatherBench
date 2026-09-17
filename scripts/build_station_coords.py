"""Station coordinate table for the 35 BMD stations (Mendeley names), from NOAA ISD history (WMO block 41).

BMD stations are WMO synoptic stations; the Mendeley dataset ships no coordinates. For each station we
take the ISD record (USAF id = WMO id * 10) that is active through 2025, falling back to GHCN-Daily
station coordinates where ISD lacks the record.
"""
import pandas as pd

# Mendeley station -> WMO station id (BMD synoptic network)
WMO = {
    "Ambaganctg": 41977, "Barisal": 41950, "Bhola": 41951, "Bogra": 41883, "Chandpur": 41941,
    "Chittagong": 41978, "Chuadanga": 41926, "Comilla": 41933, "Coxsbazar": 41992, "Dhaka": 41923,
    "Dinajpur": 41863, "Faridpur": 41929, "Feni": 41943, "Hatiya": 41963, "Ishurdi": 41907,
    "Jessore": 41936, "Khepupara": 41984, "Khulna": 41947, "Kutubdia": 41989, "Madaripur": 41939,
    "Mcourt": 41953, "Mongla": 41958, "Mymensingh": 41886, "Patuakhali": 41960, "Rajshahi": 41895,
    "Rangamati": 41966, "Rangpur": 41859, "Sandwip": 41964, "Satkhira": 41946, "Sitakunda": 41965,
    "Srimangal": 41915, "Sydpur": 41858, "Sylhet": 41891, "Tangail": 41909, "Teknaf": 41998,
}

isd = pd.read_csv("data/external/isd/isd-history.csv", dtype={"USAF": str, "WBAN": str})
isd = isd[isd.CTRY == "BG"].copy()
isd["wmo"] = isd.USAF.str[:5].astype(int)
isd = isd.sort_values("END").groupby("wmo").tail(1).set_index("wmo")

gh = pd.read_fwf("data/external/ghcnd/ghcnd-stations.txt", colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (41, 71), (80, 85)],
                 names=["id", "lat", "lon", "elev", "name", "wmo"], dtype={"wmo": str})
gh = gh[gh.id.str.startswith("BG")].dropna(subset=["wmo"]).assign(wmo=lambda d: d.wmo.astype(int)).set_index("wmo")

rows = []
for st, w in WMO.items():
    if w in isd.index:
        r = isd.loc[w]
        rows.append({"station": st, "wmo": w, "lat": r.LAT, "lon": r.LON, "elev_m": r["ELEV(M)"] if r["ELEV(M)"] > -999 else None,
                     "source_name": r["STATION NAME"], "coord_source": "ISD", "isd_end": int(r.END)})
    elif w in gh.index:
        r = gh.loc[w]
        rows.append({"station": st, "wmo": w, "lat": r.lat, "lon": r.lon, "elev_m": r.elev, "source_name": r["name"], "coord_source": "GHCN", "isd_end": None})
    else:
        rows.append({"station": st, "wmo": w, "lat": None, "lon": None, "elev_m": None, "source_name": None, "coord_source": "MISSING", "isd_end": None})
out = pd.DataFrame(rows)
out.to_csv("data/processed/bmd_station_coords.csv", index=False)
print(out.to_string(index=False))
print("\nmissing:", out[out.coord_source == "MISSING"].station.tolist())
print("inside BD bbox (20.5-26.7N, 88-92.7E):", int(out.lat.between(20.5, 26.7).mul(out.lon.between(88, 92.7)).sum()), "of", len(out))
