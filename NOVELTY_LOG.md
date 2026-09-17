# Novelty log (anti-scoop)

Re-run schedule: project start (done 2026-09-15) → end of Week 3 → 3 days before arXiv submission.
Full evidence tables: see the Week 1 plan (`~/.claude/plans/i-have-shared-my-effervescent-blum.md`), sections "Novelty verification" and "Literature review".

## Current verdict (2026-09-15): GO

| Claim | Status | Closest prior work |
|---|---|---|
| Standardized, released Bangladesh weather forecasting benchmark | OPEN | Data-only releases: Zubair et al. 2024 (Mendeley), ENACTS-BMD, Kaggle, BARC Figshare. No tasks/splits/leaderboard anywhere |
| TSFMs evaluated on Bangladesh / South Asian weather | OPEN | Most advanced Bangladesh comparison: Autoformer/FEDformer/TimesNet/Pyraformer (Kabir & Chakma, PLOS One 2026) |
| Monsoon-phase-stratified TSFM evaluation | OPEN (narrowed) | Masiwal et al. arXiv 2602.03767 benchmark monsoon *onset* for AIWP models over India |
| Geographic skill disparity measurement | CLOSED as general claim | SAFE arXiv 2510.26099 (AIWP, ERA5, T850/Z500). Our claim: TSFMs × station obs × precipitation × monsoon regime, vs temperate reference track |
| Full significance protocol | NARROWED | None of GIFT-Eval, Monash, TSFM-Bench, WB2, ChaosBench, WeatherReal, EWB run significance tests; some Bangladesh papers use paired t-tests |
| Reanalysis-vs-observation gap for TSFMs | OPEN | MAUSAM arXiv 2509.01879 (AIWP only) |
| Regime-stratified TSFM evaluation as a *method* | CLOSED (cite) | arXiv 2606.18367 (traffic); Sun & Sun 2026 ML:Earth (US streamflow hydroclimate clusters) |
| Climatic-regime stratification of ML *weather* forecast skill (added R1) | NARROWED | arXiv 2606.06348 claims "the first regime-stratified, sub-regional benchmark of an operational MLWP model over Brazil" (GraphCast; IFS analysis; Z500/T850/Q850; no precipitation). Our claim must be scoped to TSFMs × station observations incl. precipitation × monsoon phases |
| Leakage-free zero-shot TSFM evaluation (added R1) | NARROWED | TIME, arXiv 2602.12147 (ICML 2026): 50 fresh datasets, strict leakage-free, 12 TSFMs, no geographic/climate strata. Cite; do not present leakage-freeness as unique |
| Reanalysis-vs-observation gap (re-checked R1) | OPEN for TSFMs | MAUSAM (JAMES 2026) already shows AIWP errors 15–45% larger vs stations than vs reanalysis — cite as the AIWP precedent for our TSFM/baseline finding |

## Query log

| Date | ID | Query (abridged) | Top hits | Verdict |
|---|---|---|---|---|
| 2026-09-15 | A1 | TSFM benchmark monsoon / tropical stratified | 2606.18367 (traffic), TSFM-Bench, "It's TIME" 2602.12147 | OPEN |
| 2026-09-15 | A2 | "BanglaWeatherBench"; Bangladesh weather benchmark ML | Name unused; Zubair 2024 dataset; Tamim 2026 (classification) | OPEN |
| 2026-09-15 | A3 | foundation model monsoon onset / active-break ML | Masiwal 2602.03767 (AIWP onset, India) | OPEN for TSFMs |
| 2026-09-15 | A4 | AI weather models South Asia / Bay of Bengal | MAUSAM 2509.01879; Pangu on Dana/Remal (Nat. Hazards 2026) | NARROWED |
| 2026-09-15 | A5 | geographic bias / Global South AI weather | SAFE 2510.26099; Mozaffari 2603.05710 (position) | NARROWED |
| 2026-09-15 | A6 | NASA POWER / CHIRPS / BMD Bangladesh ML | Joy 2510.10702; Sci. Rep. 2025 South Asia; Hasan PLOS One 2024 | OPEN (no TSFM, no benchmark) |
| 2026-09-15 | A7 | TSFM zero-shot on geoscience variables | Sun & Sun 2026 (streamflow, US); WEATHER-5K (ICML 2026); Sundial LAI 2511.20004 | OPEN for tropics |
| 2026-09-15 | A8 | Full-text grep of 7 held benchmark PDFs | "monsoon" 0, "Bangladesh" 0, "South Asia" 1 (WeatherReal) | Supports OPEN |
| 2026-09-15 | R1 (mid-project, Week 3, full re-check) | All A1–A8 families re-run for 2026, plus new Week-3 claims (reanalysis vs observations, calibration, zero-shot rainfall) and watch-list follow-ups — see "R1 mid-project re-check" below | 11 new items verified from primary sources; none closes the core gap; 2 claims narrowed (regime stratification, leakage-free evaluation) | OPEN at our intersection — GO, with tightened wording |

## Watch list (re-check these authors/threads)
- envfluids/monsoon-benchmark (Masiwal, Hassanzadeh et al.) — could extend to TSFMs or Bangladesh
- SAFE (Masi & Balestriero) — could add stations/TSFMs
- Sun & Sun — could extend streamflow TSFM work beyond the US
- WEATHER-5K — could add precipitation or regional strata
- Rowell & Kupssinskü (Brazil GraphCast, 2606.06348) — promise "tropicalization" follow-ups; could extend to precipitation or stations
- TIME benchmark (2602.12147) — a v2 could add weather-station data or geographic strata
- Luitel, Mukhopadhyay, Singh, Juneja, Dhanuka (GraphCast ISM, 2607.11905) — could add onset/active/break strata or IMD stations

## R1 mid-project re-check (2026-09-15, before Week 4)

**Scope.** Two sets of searches:
- **Re-runs:** all Week-1 query families (A1–A8), phrased for 2026.
- **New:** 2026-specific queries on:
  - TSFMs × weather stations × precipitation;
  - Bangladesh × TSFMs;
  - monsoon onset/active/break × ML skill;
  - South Asia AIWP × station observations;
  - Global-South skill disparity;
  - regime-stratified TSFM/weather evaluation;
  - current TSFM roster × tropical/monsoon stations;
  - ML-ready Bangladesh benchmarks.

**Also searched, for claims that emerged in Week 3:**
- reanalysis vs observations overstating ML skill;
- probabilistic interval calibration for DL weather forecasts;
- zero-shot foundation models for daily rainfall;
- watch-list follow-ups (envfluids monsoon benchmark, SAFE, WEATHER-5K, Sun & Sun).

**Method.** 17 searches. Every new candidate was verified from its arXiv abstract or full text. The SAFE thesis PDF was text-extracted locally and term-counted.

### New items found and verified
| Item | What it is | Verdict for us |
|---|---|---|
| **arXiv 2606.06348** — Rowell & Kupssinskü, *Performance Evaluation of GraphCast … over Brazil* (Jun 2026) | GraphCast vs IFS HRES. Ground truth = IFS operational analysis. Z500/T850/Q850 only (precipitation explicitly not evaluated). 4 climatic sub-regions × 4 seasonal months. Full text: "the first regime-stratified, sub-regional benchmark of an operational MLWP model over Brazil" | **NARROWS** "no one stratifies ML forecast skill by climatic regime". Distinct from us: NWP emulator, analysis not stations, no rainfall, no TSFMs, no monsoon phases, Brazil |
| **arXiv 2607.11905** — Luitel et al., *GraphCast Skill and Systematic Biases in Indian Summer Monsoon Forecasts* (Jun/Jul 2026) | GraphCast vs ERA5 and IMERG; JUN–SEP 2021–2024; no IMD stations; no onset/active/break stratification in the abstract; Bangladesh not named | Neighbour; cite. Gap open |
| **arXiv 2602.12147** — Qiao et al., *It's TIME* (ICML 2026) | 50 fresh datasets, 98 tasks, strict leakage-free zero-shot evaluation of 12 TSFMs; "pattern-level" analysis; no weather-station, geographic or climate stratification | **NARROWS** leakage-free novelty (cite); supports "TSFM benchmarks do not stratify by climate" |
| **arXiv 2603.07893** — Aitken, Masiwal et al., *Designing probabilistic AI monsoon forecasts …* (Mar 2026) | Blends benchmarked AIWP models with a Bayesian "evolving farmer expectations" model for monsoon-onset probabilities; India; deployed to 38M farmers | Same group as 2602.03767; onset only, India, no TSFMs. Cite |
| **arXiv 2604.06567** — Pallotta et al., *PMP-inspired Evaluation Framework for DL Earth System Models* | Climate-simulation diagnostics (ACE2, NeuralGCM) incl. monsoon metrics | Climate simulation, not forecasting. Low risk |
| **arXiv 2603.23043** — Agana Navarro et al., *Robustness of Climate Foundation Models under No-Analog Distribution Shifts* | U-Net / ConvLSTM / ClimaX under emission-scenario shifts | Different robustness axis; no geography. Low risk |
| **arXiv 2606.19363** — Dey et al., *Guard: multi-foundation-model distillation* (Jun 2026) | TimesFM/Chronos(/Moirai) teachers; meteorology = Jena, Germany; states TSFMs "suffer from severe distributional misalignment when applied zero-shot to specific scientific domains"; no persistence/climatology comparison | Low risk; cite as motivation |
| **arXiv 2609.03763** — Partio et al., *From Nowcasting to Forecasting* (Sep 2026) | CloudCast v2: European cloud-cover model trained on CERRA reanalysis, adapted to satellite fields | Irrelevant |
| **Chronos-2 paper** (arXiv 2510.15821; rainfall–runoff figure) | Zero-shot Chronos-2 with meteorological covariates: median NSE 0.68 (univariate 0.15) vs LSTM ensemble 0.90; target is **discharge** | Hydrology neighbour alongside Sun & Sun 2026; not a rainfall-forecasting benchmark. Cite |
| **SAFE workshop/thesis version** — Masi & Balestriero (Brown) | Same scope as arXiv 2510.26099. Text search: 0 hits for station, TSFM, monsoon, season, Bangladesh, South Asia, India; precipitation only in references | No change (general disparity claim stays CLOSED; our scoped claim stays open) |
| **MAUSAM** — Gupta et al., *JAMES* 2026, doi 10.1029/2025MS005568 | Now formally published; 458 stations + rain gauges + satellite; AIWP errors 15–45% larger vs observations than vs reanalysis | Update citation. Precedent for our reanalysis-vs-observation finding (AIWP, not TSFMs) |

Snippet-only, not verified as neighbours (different region or task): arXiv 2512.01965 (West African monsoon onset, classic ML); arXiv 2607.07879 (ML reanalysis from observations). Calibration literature (arXiv 2606.19642 conformal AIWP; 2605.10297 QuantWeather) is general, and calibration is reported as a finding, not claimed as novel.

### Nothing found (searched explicitly)
- Any TSFM (Chronos, TimesFM, Moirai, Toto, TiRex, Sundial) evaluated on Bangladesh or South Asian station weather.
- Any released, ML-ready Bangladesh weather forecasting benchmark with tasks, splits and a leaderboard.
- Any monsoon-phase-stratified (pre-monsoon / onset / active–break / withdrawal) evaluation of TSFMs.
- Any quantified reanalysis-vs-observation skill gap for TSFMs.

### Verdict and wording changes
**GO — the core intersection remains OPEN.** Wording rules going forward:
1. Never claim "first regime-stratified evaluation of AI/ML weather forecasts" or "first in the Global South" unqualified (2606.06348 over Brazil). Scope it: **first for time-series foundation models, on station observations including precipitation, stratified by monsoon phase, in Bangladesh/South Asia.**
2. Do not present leakage-free zero-shot evaluation as unique; cite TIME (2602.12147). Our point is specific: a national station archive is unlikely to appear in TSFM pretraining corpora.
3. Cite MAUSAM (JAMES 2026) as the AIWP precedent for the reanalysis-vs-observation gap; our contribution is the TSFM/baseline analogue on identical station locations.

Next re-check: 3 days before arXiv submission (R2).

## R2 pre-submission re-check (2026-09-17, Week 6)

**Scope.** Targeted searches at the exact intersection (TSFM x station observations x tropics/monsoon x precipitation),
plus primary-source verification of the two papers our claim is *scoped against*. Verdict: **GAP STILL OPEN — GO.**

| Checked | What it actually is (verified from the abstract page, not a summary) | Effect on our claim |
|---|---|---|
| **TIME**, arXiv 2602.12147 | "It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks", Qiao et al., ICML 2026. 50 fresh datasets, 98 tasks, 12 foundation models, strict leakage-free zero-shot protocol, pattern-level evaluation. No weather-station, tropical or precipitation stratification stated. | Confirms rule 2: do **not** present leakage-free zero-shot evaluation as novel. Our narrower point stands: a national station archive is unlikely to appear in TSFM pretraining corpora, and we verify that per model. |
| **Brazil / GraphCast**, arXiv 2606.06348 | "Performance Evaluation of GraphCast for Medium-Range Weather Forecasting over Brazil", Rowell & Kupssinku0301. GraphCast vs IFS HRES, four Brazilian climatic sub-regions, ground truth = **operational IFS analysis**, variables T850/Q850/Z500, **precipitation excluded**, seasonal strata, days 2-7. | Weaker overlap than assumed at R1. It is an NWP-emulator study against gridded analysis without precipitation, so our scoped claim (TSFMs, station observations **including precipitation**, monsoon-phase strata, Bangladesh/South Asia) is comfortably distinct. Keep citing it; keep the claim scoped. |
| **Indian monsoon onset**, arXiv 2603.07893 | Aitken, Masiwal et al., "Designing probabilistic AI monsoon forecasts to inform agricultural decision-making". AI weather-prediction models blended with a Bayesian "evolving farmer expectations" model; monsoon **onset** timing for planting decisions; operationally deployed 2025. | Neighbour, not a scoop: onset timing rather than daily station forecasting, AIWP rather than TSFMs, no released station benchmark with fixed splits. Cite as decision-relevant monsoon forecasting. |
| **RainfallBench**, arXiv 2509.25263 | GNSS-based precipitation **nowcasting** (0-6 h), 140+ GNSS stations, PWV + ERA5-Land + IMERG truth, 17 supervised models across MLP/RNN/Transformer/CNN/GNN/KAN. | Neighbour: sub-daily nowcasting with supervised architectures, not zero-shot TSFMs, no monsoon-phase strata. Worth citing as the precipitation-specific benchmark that exists. |

**Searches run (2026-09-17):** TSFM benchmark x station observations x monsoon/Bangladesh; monsoon active/break spell
stratified foundation-model skill; zero-shot TSFM x tropical station precipitation vs climatology; Chronos/TimesFM/
Moirai evaluated on a national meteorological network in a developing country. No work found at our intersection.

**Wording rules (unchanged from R1, both re-confirmed against primary sources):**
1. Scope the novelty claim: first for **time-series foundation models, on station observations including
   precipitation, stratified by monsoon phase, in Bangladesh/South Asia**. Never "first regime-stratified" or
   "first in the Global South" unqualified.
2. Cite TIME for leakage-free zero-shot evaluation; do not claim it as ours.
3. Cite MAUSAM (JAMES 2026) as the AIWP precedent for the reanalysis-vs-observation gap.
4. New: cite 2606.06348 accurately - it excludes precipitation and scores against IFS analysis, so do not describe
   it as a station-based or precipitation benchmark.

**Next re-check:** none scheduled before submission; re-run if submission slips more than two weeks.
