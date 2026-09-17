"""Pick the final temperate reference set from reports/temperate_reference_candidates.csv.

Rules: PRCP and TAVG >= 90% complete over 2016-2023; Global-North only (Algeria and Iran excluded);
GSN stations first, then completeness; at most 5 stations per country."""
import pandas as pd

GLOBAL_SOUTH = {"AG", "IR"}
MIN_PCT, PER_COUNTRY = 90, 5

c = pd.read_csv("reports/temperate_reference_candidates.csv")
q = c[(c.prcp_pct >= MIN_PCT) & (c.tavg_pct >= MIN_PCT) & ~c.cc.isin(GLOBAL_SOUTH)].copy()
q["gsn_flag"] = q.gsn.fillna("").str.contains("GSN")
q = q.sort_values(["gsn_flag", "prcp_pct"], ascending=[False, False]).groupby("cc").head(PER_COUNTRY)
q.drop(columns="gsn_flag").to_csv("reports/temperate_reference_stations.csv", index=False)
print(f"selected {len(q)} stations in {q.cc.nunique()} countries: {q.cc.value_counts().to_dict()}")
