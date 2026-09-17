"""EDA statistics behind the Week-2 figures. Each output CSV is the table-view twin of a figure."""
import numpy as np
import pandas as pd

from bwb.data.indices import read_psl_monthly

OUT = "reports/figures/data"
bmd = pd.read_parquet("data/processed/bmd_daily.parquet")
ghcn = pd.read_parquet("data/processed/ghcn_daily.parquet")

# F3 coverage: share of days with a non-flagged observation, per station-year (BMD) and per station-year (GHCN)
b = bmd.assign(year=bmd.date.dt.year, ok=~bmd.Rainfall_flag & bmd.Rainfall.notna())
cov_b = b.groupby(["station", "year"]).ok.mean().mul(100).round(1).rename("pct").reset_index().assign(track="BMD")
g = ghcn.assign(year=ghcn.date.dt.year, ok=ghcn.Rainfall.notna())
cov_g = g[g.year >= 1961].groupby(["region", "station_id", "year"]).ok.mean().mul(100).round(1).rename("pct").reset_index()
cov_g["track"] = "GHCN " + cov_g.region
pd.concat([cov_b.rename(columns={"station": "station_id"}), cov_g.drop(columns="region")]).to_csv(f"{OUT}/f3_coverage_station_year.csv", index=False)

# F2 seasonal cycle (train period, clean values): station-median of monthly mean rainfall total & temperature
tr = bmd[bmd.date <= "2010-12-31"].copy()
tr["Rainfall"] = tr.Rainfall.mask(tr.Rainfall_flag)
tr["Temperature"] = tr.Temperature.mask(tr.Temperature_flag)
tr["month"] = tr.date.dt.month
monthly_rain = tr.groupby(["station", tr.date.dt.year, "month"]).Rainfall.sum(min_count=25).groupby(["station", "month"]).mean()
monthly_temp = tr.groupby(["station", "month"]).Temperature.mean()
cyc = pd.DataFrame({
    "rain_mm_median": monthly_rain.groupby("month").median(), "rain_mm_p25": monthly_rain.groupby("month").quantile(0.25), "rain_mm_p75": monthly_rain.groupby("month").quantile(0.75),
    "temp_c_median": monthly_temp.groupby("month").median(), "temp_c_p25": monthly_temp.groupby("month").quantile(0.25), "temp_c_p75": monthly_temp.groupby("month").quantile(0.75),
}).round(2)
cyc.to_csv(f"{OUT}/f2_seasonal_cycle.csv")

# F4 rainfall distribution: all clean daily values in train; zero share + wet-day histogram on log bins
r = tr.Rainfall.dropna()
zero_share = float((r < 0.1).mean())
wet = r[r >= 1.0]
bins = np.unique(np.round(np.logspace(0, np.log10(wet.max() + 1), 32)))  # integer-mm edges: BMD reports small totals in whole mm
hist, edges = np.histogram(wet, bins=bins)
pd.DataFrame({"bin_lo_mm": edges[:-1].round(2), "bin_hi_mm": edges[1:].round(2), "days": hist}).to_csv(f"{OUT}/f4_wetday_histogram.csv", index=False)
pd.DataFrame({"stat": ["dry_day_share(<0.1mm)", "wet_day_share(>=1mm)", "wet_median_mm", "wet_p95_mm", "wet_p99_mm", "max_mm"],
              "value": [zero_share, float((r >= 1).mean()), wet.median(), wet.quantile(.95), wet.quantile(.99), wet.max()]}).round(3).to_csv(f"{OUT}/f4_rain_summary.csv", index=False)

# F5 teleconnections: national (station-mean) monsoon-season (JJAS) rainfall anomaly vs ONI/DMI by lead month, 1961-2010 (train)
oni = read_psl_monthly("data/external/indices/oni.data", "ONI")
dmi = read_psl_monthly("data/external/indices/dmi.had.long.data", "DMI")
full = bmd.copy()
full["Rainfall"] = full.Rainfall.mask(full.Rainfall_flag)
mon = full.groupby(["station", full.date.dt.to_period("M")]).Rainfall.sum(min_count=25).unstack(0)
anom = mon.groupby(mon.index.month).transform(lambda x: (x - x.loc[:"2010-12"].mean()) / x.loc[:"2010-12"].std())
nat = anom.mean(axis=1)
jjas = nat[nat.index.month.isin([6, 7, 8, 9])]
jjas = jjas.groupby(jjas.index.year).mean()
jjas = jjas.loc[1961:2010]
rows = []
for name, idx in (("ONI", oni), ("DMI", dmi)):
    for lead in range(0, 13):  # index month = August minus lead months
        m = pd.Timestamp("2000-08-01") - pd.DateOffset(months=lead)
        vals = idx[idx.index.month == m.month]
        vals.index = vals.index.year + (1 if m.month > 8 else 0)
        x = pd.concat([jjas, vals], axis=1).dropna()
        rr = x.corr().iloc[0, 1]
        n = len(x)
        rows.append({"index": name, "lead_months": lead, "index_month": m.strftime("%b"), "r": round(rr, 3), "n_years": n,
                     "abs_r_sig95": round(1.96 / np.sqrt(n - 3), 3)})
pd.DataFrame(rows).to_csv(f"{OUT}/f5_teleconnection_lag_corr.csv", index=False)

print("wrote:", "f2_seasonal_cycle, f3_coverage_station_year, f4_wetday_histogram, f4_rain_summary, f5_teleconnection_lag_corr")
print(cyc[["rain_mm_median", "temp_c_median"]].T.to_string())
print(pd.read_csv(f"{OUT}/f4_rain_summary.csv").to_string(index=False))
print(pd.DataFrame(rows).pivot(index="lead_months", columns="index", values="r").T.to_string())
print("approx |r| for p<0.05:", rows[0]["abs_r_sig95"])
