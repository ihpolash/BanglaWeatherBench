"""Week-1 gate: how much of the BMD daily dataset looks imputed, especially in the proposed test period?"""
import numpy as np
import pandas as pd

from bwb.data.qc import imputation_mask

SRC = "data/external/bmd_mendeley/combined/BD_weather.csv"
VARS = ["Rainfall", "Temperature", "Humidity", "Sunshine"]
TEST = ("2016-01-01", "2023-12-31")

df = pd.read_csv(SRC)
df["date"] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day), errors="coerce")
print("invalid dates:", int(df.date.isna().sum()), "| duplicate (station,date):", int(df.duplicated(["Station", "date"]).sum()))

# Coverage: calendar gaps would mean dropped days rather than filled ones.
cov = df.groupby("Station").date.agg(["min", "max", "count"])
cov["expected"] = (cov["max"] - cov["min"]).dt.days + 1
cov["missing_days"] = cov["expected"] - cov["count"]
print("\nstart-year distribution:", cov["min"].dt.year.value_counts().sort_index().to_dict())
print("end dates:", cov["max"].dt.strftime("%Y-%m-%d").value_counts().to_dict())
print("stations with calendar gaps:", int((cov.missing_days > 0).sum()), "| total missing days:", int(cov.missing_days.sum()))

# Precision anomaly: BMD reports humidity as integers and rain/temp/sunshine at 0.1; finer values imply interpolation/mean fill.
def off_grid(s, decimals):
    return ~np.isclose(s, s.round(decimals), atol=1e-9)

prec = {"Humidity": 0, "Rainfall": 1, "Temperature": 1, "Sunshine": 1}
rows = []
for st, g in df.groupby("Station"):
    g = g.set_index("date").sort_index()
    test = (g.index >= TEST[0]) & (g.index <= TEST[1])
    for v in VARS:
        m = imputation_mask(g[v], v)
        m["flag_offgrid"] = off_grid(g[v], prec[v])
        m["flag_any"] = m[["flag_constant", "flag_linear", "flag_offgrid"]].any(axis=1)
        rec = {"station": st, "var": v, "n_test": int(test.sum())}
        for c in ["flag_constant", "flag_linear", "flag_offgrid", "flag_any"]:
            rec[f"{c}_all_pct"] = 100 * m[c].mean()
            rec[f"{c}_test_pct"] = 100 * m.loc[test, c].mean() if test.any() else np.nan
        rows.append(rec)
res = pd.DataFrame(rows)
res.to_csv("reports/bmd_imputation_audit.csv", index=False)

pd.set_option("display.width", 200)
print("\n% of days flagged, by variable (median / max across stations):")
summ = res.groupby("var")[[c for c in res.columns if c.endswith("_pct")]].agg(["median", "max"]).round(2).T
print(summ.to_string())
print("\nstations with no test-period data:", sorted(res.loc[res.n_test == 0, "station"].unique()))
worst = res.sort_values("flag_any_test_pct", ascending=False).head(10)[["station", "var", "n_test", "flag_constant_test_pct", "flag_linear_test_pct", "flag_offgrid_test_pct", "flag_any_test_pct"]]
print("\nworst 10 station-variables in test period:\n", worst.round(2).to_string(index=False))
print("\nstation-variables with >10% flagged in test:", int((res.flag_any_test_pct > 10).sum()), "of", int(res.n_test.gt(0).sum()))
