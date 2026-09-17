# GHCN-Daily tracks: Bangladesh second observational track + temperate reference track

Date: 2026-09-15. Code: `src/bwb/data/ghcn.py`, `scripts/select_temperate_reference.py`. Tests: `tests/test_ghcn.py`.

## 1. Bangladesh GHCN track (10 stations, co-located with BMD)

| GHCN ID | GHCN name | BMD station |
|---|---|---|
| BGM00041859 | RANGPUR | Rangpur |
| BGM00041883 | BOGRA | Bogra |
| BGM00041891 | OSMANY INTL | Sylhet |
| BGM00041907 | ISHURDI | Ishurdi |
| BGM00041923 | TEJGAON | Dhaka |
| BGM00041936 | JESSORE | Jessore |
| BGM00041943 | FENI | Feni |
| BGM00041950 | BARISAL | Barisal |
| BGM00041978 | SHAH AMANAT INTL | Chittagong |
| BGM00041992 | COXS BAZAR | Cox's Bazar |

All come from the WMO/GSOD-derived network (ID character 3 = `M`, source flag `S`). QC failure rate is 0.01–0.03%.

### Coverage — sparse
- **2016–2023 completeness:**
  - PRCP ≈ 65% (Chittagong 70%)
  - TAVG ≈ 67% (Chittagong 99%)
  - TMAX/TMIN ≈ 44–51%
- **Training period 1985–2010:** PRCP 17–59%. Feni and Barisal are the weakest.
- **Gap years:** 2011–2013 are half-empty, 2019 has ~220 days, **2021 only ~25 days**, and 2022 ~100 days.
- **Past BMD's 2023 cutoff:** PRCP runs to 2024-05-30 at Bogra, 2024-12-31 at Rangpur, Feni and Barisal, and 2025-08-24 at the other six. That is 137–577 extra days per station.

### Date convention — rainfall is shifted by one day
GHCN labels a 24-h rainfall total by its start day; BMD labels it by its end day. Daily correlation with BMD (2000–2023):

| Pairing | r range |
|---|---|
| GHCN day t vs BMD day t (lag 0) | 0.31–0.63 |
| **GHCN day t vs BMD day t+1** | **0.68–0.90** |
| GHCN day t vs BMD day t−1 | 0.11–0.38 |

`align_to_bmd_convention()` applies the +1-day relabel to rainfall only. Temperature needs no shift.

### Agreement with BMD after alignment
| Station | Rain r | Spearman | Wet-day agreement | BMD wet, GHCN 0 | BMD ≥50 mm captured (GHCN ≥25) | Tavg r | Tavg MAE °C |
|---|---|---|---|---|---|---|---|
| Rangpur | 0.825 | 0.830 | 91.8% | 11.2% | 77.5% | 0.992 | 0.32 |
| Bogra | 0.818 | 0.827 | 91.7% | 11.6% | 81.4% | 0.990 | 0.34 |
| Sylhet | 0.782 | 0.855 | 90.5% | 8.1% | 82.1% | 0.989 | 0.33 |
| Ishurdi | 0.807 | 0.854 | 93.6% | 11.0% | 85.5% | 0.990 | 0.39 |
| Dhaka | 0.772 | 0.851 | 91.2% | 8.4% | 88.4% | 0.992 | 0.31 |
| Jessore | 0.676 | 0.848 | 90.9% | 12.1% | 65.8% | 0.989 | 0.40 |
| Feni | 0.830 | 0.853 | 92.1% | 12.4% | 85.3% | 0.984 | 0.38 |
| Barisal | 0.839 | 0.885 | 93.5% | 7.4% | 89.8% | 0.988 | 0.36 |
| Chittagong | 0.749 | 0.709 | 86.6% | 25.6% | 72.3% | 0.945 | 0.86 |
| Cox's Bazar | 0.904 | 0.874 | 92.7% | 11.7% | 87.9% | 0.987 | 0.30 |

Monthly-total correlation is 0.80–0.99, and the GHCN/BMD total-rainfall ratio is 0.85–1.00.

**Interpretation:**
- Temperature independently validates the BMD record: r ≈ 0.99, MAE ≈ 0.3 °C. BMD's "Temperature" column is a true daily mean.
- Rainfall agrees in amount but carries ~8–12% wet-day disagreement. This is the kind of data a global station corpus (ISD/GSOD, e.g. WEATHER-5K) holds for Bangladesh.
- Chittagong is an outlier (airport vs city site). Treat it as a different location, not a duplicate of BMD Chittagong.

## 2. Temperate reference track (Global-North mid-latitude)

**Selection:**
- Same WMO/GSOD-derived network (`M`) as the Bangladesh GHCN stations, so both tracks share the same data pipeline.
- Latitude 35–60°N; PRCP and TAVG records span 1985–2024.
- ≥90% complete for both PRCP and TAVG over 2016–2023.
- GSN stations first, at most 5 per country.
- Algeria and Iran were excluded so the reference is Global North.

**Result:** 32 stations in 14 countries, latitude 36.2–60.0°N, median completeness PRCP 98.9% / TAVG 99.9%. List: `reports/temperate_reference_stations.csv`. Full 90-station shortlist: `reports/temperate_reference_candidates.csv`.

**Known limitations:**
- Europe-heavy: Russia 5, France 5, Czechia 5, Bulgaria 4, Romania 3, plus 9 others.
- No North American or East Asian stations, because the `M` network stations there do not meet the TAVG requirement. Week 2 could add national-network US/JP stations as a sensitivity set.

## 3. Design decisions for the benchmark
1. **Completeness asymmetry is a finding, not noise.** Over 2016–2023, the Bangladesh GHCN stations are ~65% complete and the temperate stations ~99%. Report it as data-divide evidence.
2. **Fair skill contrast:** score only rolling-origin windows whose context and target are both complete, in every track. As a sensitivity ablation, impose Bangladesh-like gap masks on the temperate series.
3. **Three observational views of Bangladesh:** BMD national network (primary), GHCN/GSOD (second track), and NASA POWER reanalysis. The same models run on each give a direct measurement of how the data source changes apparent foundation-model skill.
4. **The equity contrast uses GHCN vs GHCN** — Bangladesh `M` stations against temperate `M` stations — so the difference cannot be explained by data source. BMD vs temperate is reported as a secondary comparison.
