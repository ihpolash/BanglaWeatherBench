"""Per station-day evaluation strata for BMD: season, monsoon phase, active/break spell, extremes, cyclone day.

Active/break spells are a large-scale property: following Ferdoushi et al. (2023), they are identified on the
all-station mean daily rainfall (days with >= MIN_STATIONS clean reports) and the same label is applied to every
station. Extremes and cyclone days are per station."""
import numpy as np
import pandas as pd
import yaml

from bwb.data.labels import active_break_spells, extreme_flags, extreme_thresholds, monsoon_phase, season

TRAIN_END = yaml.safe_load(open("configs/splits.yaml"))["daily"]["train"]["end"]
MIN_STATIONS = 25
bmd = pd.read_parquet("data/processed/bmd_daily.parquet")
cyc = pd.read_parquet("data/processed/cyclone_station_days.parquet")[["station", "date"]].assign(cyclone_day=True)

clean_rain = bmd.assign(r=bmd.Rainfall.mask(bmd.Rainfall_flag)).pivot(index="date", columns="station", values="r").asfreq("D")
n_rep = clean_rain.notna().sum(axis=1)
national = clean_rain.mean(axis=1).where(n_rep >= MIN_STATIONS)
spell = active_break_spells(national, TRAIN_END).reindex(national.index).fillna("")
pd.DataFrame({"national_rain_mm": national, "n_stations": n_rep, "spell": spell}).to_parquet("data/processed/national_spells.parquet")

parts, thr_rows = [], []
for st, g in bmd.groupby("station"):
    g = g.set_index("date").sort_index()
    clean = g.assign(Rainfall=g.Rainfall.mask(g.Rainfall_flag), Temperature=g.Temperature.mask(g.Temperature_flag))
    th = extreme_thresholds(clean, TRAIN_END)
    thr_rows.append({"station": st, "rain_p95_mm": round(th["rain_p95"], 1), "rain_p99_mm": round(th["rain_p99"], 1), "train_start": g.index.min().date()})
    lab = pd.DataFrame({"season": season(g.index), "monsoon_phase": monsoon_phase(g.index), "spell": spell.reindex(g.index).fillna("").to_numpy()}, index=g.index)
    lab = lab.join(extreme_flags(clean, th))
    parts.append(lab.reset_index().assign(station=st))
labels = pd.concat(parts, ignore_index=True).merge(cyc, on=["station", "date"], how="left")
labels["cyclone_day"] = labels.cyclone_day.fillna(False).astype(bool)
labels.to_parquet("data/processed/bmd_labels.parquet", index=False)
pd.DataFrame(thr_rows).to_csv("data/processed/bmd_extreme_thresholds.csv", index=False)


def phases(s: pd.Series, kind: str) -> tuple[int, int]:
    m = (s == kind).to_numpy()
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    return len(starts), int(m.sum())


ref = spell.loc["1988-01-01":"2017-12-31"]
a_ph, a_d = phases(ref, "active")
b_ph, b_d = phases(ref, "break")
print(f"national series: first date with >= {MIN_STATIONS} stations = {national.first_valid_index().date()}")
print(f"1988-2017 ours: active {a_ph} phases / {a_d} days; break {b_ph} phases / {b_d} days; break/active days {b_d / a_d:.2f}")
print("1988-2017 Ferdoushi et al. 2023: active 65 phases / 271 days; break 104 phases / 517 days; ratio 1.91")
test = labels[labels.date.between("2016-01-01", "2023-12-31")]
print("test Jul-Aug spell share %:", test[test.spell != ""].spell.value_counts(normalize=True).mul(100).round(1).to_dict())
print("test flag rates %:", test[["heavy_rain", "very_heavy_rain", "hot_day", "cold_day", "cyclone_day"]].mean().mul(100).round(2).to_dict())
