# BanglaWeatherBench: A Complete Week-by-Week Research & Publication Plan

> **Important tooling caveat (read first).** Both my live web-search tools and my research subagent's search tools returned hard errors in this environment ("tool was not provided, cannot call tool"). I could **not** run live 2025–2026 searches against arXiv, Semantic Scholar, Papers with Code, OpenReview, Hugging Face, or GitHub. Everything below is reconstructed from expert domain knowledge reliable through roughly mid-2025. **Every novelty determination, model version suffix, 2026/2027 deadline, and impact factor marked below must be re-verified live before you rely on it in the paper.** I flag confidence throughout and give you the exact queries to run yourself in Section 3. Treat the novelty verification (Section 2) as *provisional-strong* rather than *confirmed*. The one externally confirmable fact I was able to validate through enrichment — Journal of Hydrology's 2024 Impact Factor of 5.9 — is marked as verified.

---

## TL;DR
- **The core thesis is genuinely novel and publishable.** Based on all available knowledge, no existing time-series benchmark (GIFT-Eval, Monash, TFB, FoundTS) stratifies zero-shot/few-shot foundation-model skill by **tropical-monsoon region**, no weather-foundation-model evaluation (WeatherBench 2, ChaosBench, WeatherReal) **isolates Bangladesh/Bay-of-Bengal** performance, and there is **no published Bangladesh-specific ML-ready weather benchmark**. Your regional-equity framing ("FMs underperform in the Global South tropics due to geographic training-data imbalance") is unclaimed for time-series/weather models. **Re-verify live before submission** (queries in §3).
- **A 6-week single-GPU (T4/P100/16 GB) path to an arXiv preprint is realistic** if you scope tightly (2–3 target variables, 4 horizons, ~6 foundation models zero-shot, ~10 baselines). An **optional 8–10 week extension** upgrades the identical work toward NeurIPS Datasets & Benchmarks / *Scientific Data* / *Environmental Data Science* quality by adding fine-tuning ablations, WBGT heat-stress targets, FFWC operational comparison, and statistical rigor (critical-difference diagrams, bootstrap CIs).
- **This report delivers all 15 requested components**: goal/contribution statement, defensible novelty + gap table, week-1 literature protocol, full data-pipeline/EDA plan, benchmark design, complete model roster with HF IDs and VRAM notes, evaluation protocol, granular week-by-week schedule (6-week core + 7–10 extension), engineering practices, figure/table manifest, section-by-section paper outline, release/submission mechanics for a cs.AI-endorsed user, impact-maximization venue strategy, a risk register (including how to publish a *negative* result), and a one-page printable checklist.

---

## Key Findings (decision-ready)

1. **Publish arXiv-first, target NeurIPS D&B second.** A benchmark+dataset paper accrues citations primarily through (a) being the standard others compare against and (b) an easy-to-use dataset release. Get the arXiv preprint out in 6 weeks to plant the flag on "BanglaWeatherBench," then spend the extension weeks hardening it for a peer-reviewed venue. This dual-track maximizes both priority (anti-scoop) and eventual impact factor.
2. **Design for a possibly-negative headline and it becomes un-scoopable.** The most likely empirical outcome is that **zero-shot foundation models do *not* beat well-tuned classical baselines (seasonal-naive, SARIMA, climatology) on monsoon precipitation**, and are only competitive on smooth variables (T2M, RH2M). This is a *feature*: a rigorous negative/mixed result that quantifies *where and why* FMs fail in the tropics is exactly the novel contribution. Pre-commit to this framing so a "disappointing" number is still a headline.
3. **The dataset contribution is your durable citation engine.** Classical NASA POWER + CHIRPS are openly licensed and redistributable; BMD PDFs and FFWC scrapes are **not** cleanly redistributable — release derived features/normals, not raw copyrighted tables. Ship the ML-ready benchmark on Hugging Face Datasets **plus** a Zenodo DOI (for citeability and a *Scientific Data* descriptor later).
4. **Compute is not the bottleneck; disciplined engineering is.** Every model in your roster runs zero-shot on a single 16 GB GPU. The real risks are Colab/Kaggle session timeouts, dependency conflicts between forecasting libraries, and data-leakage in temporal splits. Solve these with checkpointed prediction caching, per-library conda/venv isolation, and strict rolling-origin backtesting.
5. **arXiv mechanics for handle `ipolas`:** with a **cs.AI** endorsement only, the safe move is **primary = cs.AI**, cross-list **cs.LG** and **physics.ao-ph**. Endorsement gates the *primary* archive; cross-lists generally do not need separate endorsement. If you want cs.LG *primary* (slightly better for an ML benchmark), request a cs.LG endorsement first — verify current rules at arxiv.org/help/endorsement.

---

## Details

### 1. Goal and Contribution Statement

**Goal (one paragraph, paper-ready).**
> *BanglaWeatherBench* is the first systematic, reproducible benchmark for evaluating time-series foundation models against classical statistical and modern deep-learning baselines on tropical-monsoon meteorological data from Bangladesh, one of the most climate-vulnerable and observationally under-represented regions on Earth. We assemble an ML-ready, openly licensed dataset from NASA POWER daily agroclimatology, CHIRPS-derived dekadal rainfall, BMD climatological normals, and hazard-event records, augmented with global teleconnection indices (ONI, IOD/DMI, MJO RMM). We define standardized forecasting tasks across multiple variables and horizons with strict temporal, leakage-free splits, and we evaluate zero-shot and few-shot foundation models (Chronos-Bolt, TimesFM 2.0, Moirai, Granite TTM, MOMENT, Lag-Llama) alongside naive, statistical, tree-based, and deep baselines, stratifying performance by monsoon phase (pre-monsoon / active monsoon / break / dry) and by extreme-event conditioning. Our central question is whether the "train once, forecast anywhere" promise of foundation models holds in a data-scarce tropical-monsoon regime, or whether geographic imbalance in their pretraining corpora produces systematic skill degradation in the Global South.

**Claimed contributions (numbered, as they would appear in the Introduction).**
1. **A new open benchmark and ML-ready dataset** for Bangladesh weather/climate forecasting, harmonizing dekadal CHIRPS rainfall, daily NASA POWER variables, BMD normals, and hazard labels into leakage-free tasks with fixed splits, released under CC-BY-4.0 on Hugging Face Datasets with a Zenodo DOI.
2. **The first monsoon-phase-stratified, zero-shot/few-shot evaluation of time-series foundation models** in a tropical-monsoon regime, reporting skill separately for pre-monsoon, active-monsoon, break, and dry seasons and for extreme-event windows.
3. **Quantitative evidence on the Global-South geographic-bias hypothesis** for foundation forecasting: we measure whether FM skill relative to local baselines degrades specifically on tropical convective precipitation and during monsoon onset/active-break transitions.
4. **A single-GPU-reproducible protocol** — every result runs on ≤16 GB VRAM within Colab/Kaggle limits — with compute cost (runtime, VRAM, parameters) reported as a first-class metric alongside accuracy.
5. **Open, one-command-reproducible code**, cached predictions, and per-model configs, lowering the barrier for future work on Global-South weather ML.

---

### 2. Novelty Statement and Gap Analysis

**Defensible novelty claim (the "what has NOT been done" framing).**
> Existing time-series foundation-model benchmarks stratify by *domain* and *frequency* but never by *climatic regime*; existing weather-ML benchmarks report *global* aggregates (at most a coarse tropics-vs-extratropics latitude band) and never isolate the Bengal delta or the South Asian summer monsoon; and no ML-ready Bangladesh weather benchmark exists. Consequently, **no prior work has measured whether foundation models' forecast skill degrades specifically under tropical-monsoon dynamics relative to their strong temperate-region performance.** BanglaWeatherBench fills exactly this intersection.

**Table of closest prior work** (all determinations *provisional pending live re-verification* — see §3; confidence noted):

| Paper (author, year, venue) | What it did | What it did NOT do | How BanglaWeatherBench differs |
|---|---|---|---|
| **GIFT-Eval** (Aksu et al., 2024, arXiv 2410.10393; Salesforce) | ~23 datasets / 144k+ series across 7 domains, 10 frequencies; HF leaderboard for TS-FMs | No geographic/climatic stratification; weather subset is global/temperate-dominant; no monsoon cut | Stratifies by monsoon phase & extremeness; Bangladesh-specific; Global-South focus |
| **Monash TSF Archive** (Godahewa et al., 2021, NeurIPS D&B, arXiv 2105.06643) | ~30 datasets by domain for forecasting eval | No regional/monsoon stratification; predates most TS-FMs | Purpose-built for zero/few-shot FM eval in one climatic regime |
| **TFB / TSFM-Bench** (Qiu et al., 2024, VLDB, arXiv 2403.20150) | Fair benchmarking of TSF methods, domain-stratified | No monsoon/Global-South cut | Regional-equity framing; weather-specific tasks |
| **FoundTS** (Li et al., 2024, arXiv 2410.11802) | Benchmarks TS foundation + deep models | Dataset/domain-stratified only | Climatic-regime stratification, station data |
| **WeatherBench 2** (Rasp et al., 2023, arXiv 2308.15560, JAMES) | Global gridded NWP-ML benchmark; RMSE/ACC by variable, lead-time, latitude band | Global aggregates; tropics band only; not Bangladesh/Bay-of-Bengal-specific; not station-level TS-FMs | Station-level, monsoon-phase, TS-FM (not NWP-FM) focus |
| **ChaosBench** (Nathaniel et al., 2024, NeurIPS D&B, arXiv 2402.00712) | Subseasonal-to-seasonal physics-based metrics, global | No monsoon isolation; not TS-FM zero-shot | S2S-adjacent but regionally specific & FM-centric |
| **WeatherReal** (Microsoft, 2024, arXiv 2409.09371) | Station-observation benchmark, global stations | Not monsoon-stratified; not TS-FM zero-shot | Bangladesh stations + FM equity analysis |
| **ExtremeWeatherBench** (2025) | Extreme-event (heatwave/cyclone) focus | *Verify live*: may include Bay-of-Bengal cyclones but no dedicated monsoon/Bangladesh cut | Systematic monsoon-phase + extreme conditioning for TS-FMs |
| Applied Bangladesh rainfall/flood ML (various, LSTM/ARIMA/RF studies) | Point applications for BD rainfall/flood | No standardized benchmark, no FM zero-shot, no release | Standardized, released, FM-centric benchmark |

**Confidence:** HIGH that the *intersection* (monsoon-stratified TS-FM zero-shot benchmark for Bangladesh) is open; MEDIUM on exhaustiveness because live 2025–2026 search failed. Highest residual risk items to check: A2 (any named "Bangladesh weather benchmark" on HF/PwC), A5 (any 2025–26 "Global South FM equity" TS paper), A6 (2025–26 NASA POWER + CHIRPS Bangladesh ML), and ExtremeWeatherBench's regional coverage.

**Three-to-five specific research gaps no existing paper covers:**
1. **Monsoon-phase-conditioned FM skill.** No paper reports zero-shot FM error separately for pre-monsoon / active / break / dry phases.
2. **Bengal-delta isolation.** No weather-FM eval isolates Bangladesh/Bay-of-Bengal failure modes from global aggregates.
3. **Global-South geographic-bias quantification for time-series/weather FMs.** The theme exists in vision/NLP fairness (e.g., DeVries et al., "Does Object Recognition Work for Everyone?", ICCV-W 2019) but is unquantified for forecasting FMs.
4. **Dekadal-vs-daily multi-source harmonization** into an ML-ready released benchmark for the region.
5. **Compute-as-first-class-metric on consumer hardware** for TS-FMs on tropical data (a reproducibility/equity angle: what a Global-South researcher can actually run).

---

### 3. Literature Review Plan (Week 1)

**Databases & tools:** arXiv (listing + full-text), Semantic Scholar (+ its free API `api.semanticscholar.org`), Google Scholar (for citation-chasing only), Papers with Code (datasets + leaderboards), OpenReview (NeurIPS/ICLR submissions incl. rejected — good for scoop-detection), Hugging Face (Datasets + Models search), GitHub (topic search), Connected Papers and Litmaps (citation-graph exploration around 2–3 seed papers, e.g., GIFT-Eval and WeatherBench 2).

**Exact search queries to run (copy-paste):**
- Novelty A1: `time series foundation model benchmark monsoon`, `zero-shot forecasting benchmark South Asia`, `GIFT-Eval regional stratification`, `foundation model forecasting Global South`
- A2: `BanglaWeather`, `Bangladesh weather benchmark machine learning`, `Bangladesh climate dataset ML-ready`, `Bengal delta forecasting deep learning`, HF Datasets search `bangladesh weather`, PwC search `weather forecasting Bangladesh`
- A3: `foundation model monsoon onset forecast skill`, `tropical convective precipitation deep learning skill`, `active break monsoon machine learning`
- A4: `WeatherBench 2 tropics`, `Aurora Bay of Bengal`, `GraphCast monsoon evaluation`, `weather model regional bias South Asia`, `ExtremeWeatherBench cyclone`
- A5: `foundation model geographic bias tropics`, `time series model equity Global South`, `weather AI regional disparity`
- A6: `NASA POWER CHIRPS Bangladesh machine learning`, `CHIRPS rainfall LSTM Bangladesh 2025`

**Volume & depth:** read **~40–60 papers** at abstract level, **~15–20 deeply**. Prioritize: (i) the 8 benchmark papers in the §2 table; (ii) the primary paper for each foundation model in §6; (iii) 5–8 Bangladesh/South-Asia applied weather-ML papers; (iv) 3–5 FM-fairness/Global-South papers.

**Related-work taxonomy (use as the paper's §2 subsection headers):**
1. Time-series foundation models (Chronos, TimesFM, Moirai, TTM, MOMENT, Lag-Llama).
2. Time-series forecasting benchmarks (GIFT-Eval, Monash, TFB, FoundTS).
3. Weather/climate ML benchmarks (WeatherBench 2, ChaosBench, WeatherReal, ClimateLearn) and weather FMs (Aurora, GraphCast, Pangu, GenCast, NeuralGCM).
4. Regional / Global-South / equity evaluation in ML.
5. Bangladesh & South-Asian climate ML applications.

**Reference-manager workflow:** Zotero (with the Better BibTeX plugin) → auto-export `references.bib` into the LaTeX repo; use citation keys `firstauthorYEARkeyword`. Tag each entry with the taxonomy bucket.

**Living novelty-check log (anti-scoop):** keep `NOVELTY_LOG.md` in the repo. For each of A1–A6 record: date checked, query, top hits, verdict (OPEN/CLOSED/AT-RISK), and a one-line differentiation. Re-run the A-queries at **project start, mid-project (end of week 3), and 3 days before arXiv submission.** Set arXiv email alerts for `cs.LG` + `physics.ao-ph` with keywords "foundation model" + "weather/monsoon."

---

### 4. Data Pipeline and EDA Plan

**4.1 Ingestion & parsing**
- `All_Districts_Combined.csv` (171,190×16, NASA POWER daily): construct a proper date from `YEAR`+`DOY` (`pd.to_datetime(df.YEAR*1000+df.DOY, format='%Y%j')`), which handles leap years correctly. NASA POWER sentinel for missing is **-999**; replace with `NaN`. Variables are UTC/solar-day aggregates — document the convention; no per-row timezone conversion needed for daily data, but note Bangladesh is UTC+6.
- `bgd-rainfall-subnat-full.csv` (121,656×15, CHIRPS/WFP dekadal): parse `date` (dekad start), split by `adm_level` (admin-1 vs admin-2), key on `PCODE`. `rfh`=10-day rainfall, `rfq`=anomaly ratio vs `rfh_avg`. Keep `n_pixels` for QC (drop dekads with anomalously few pixels).
- `Hazard_Weather.xls`: read with `pandas.read_excel` (openpyxl/xlrd). Extract event date ranges + hazard type → build a binary/categorical **extreme-event label** time series per district for conditional evaluation.
- **BMD normals PDFs:** extract tables with `camelot-py` (lattice mode for ruled tables, stream mode otherwise) or `tabula-py`; fall back to `pdfplumber` for stubborn pages. Manually validate 2–3 extracted tables against the PDF. Store normals as tidy `station, month/day, variable, normal_value`. **Do not redistribute the raw BMD tables** (see §12) — use them only to define/validate anomalies.

**4.2 District↔station↔grid alignment.** Build a crosswalk table mapping NASA POWER district centroids, CHIRPS PCODEs, and BMD station coordinates. Match by nearest-neighbor on lat/lon (`scipy.cKDTree`) with a distance threshold; log unmatched units. This crosswalk is itself a dataset artifact worth releasing.

**4.3 Reconciling dekadal CHIRPS with daily NASA POWER.** Two supported modes, both shipped:
- **Native-frequency tasks** (recommended primary): forecast NASA POWER variables at **daily** frequency; forecast CHIRPS `rfh` at **dekadal** frequency. Do not upsample rainfall to daily (fabricates information).
- **Aligned aggregate tasks** (secondary): aggregate NASA POWER precip/derived variables to dekadal to co-model with CHIRPS. Always aggregate the finer series *down*, never interpolate the coarse series *up*.

**4.4 QC / cleaning checklist.** Missing-value heatmaps per district/variable; gap-length distribution; impute short gaps (≤3 days) by time interpolation, leave long gaps as `NaN` and mask in metrics; physical-range clipping (RH2M∈[0,100], negative rainfall→0, PS plausibility); duplicate-date detection; monotonic-date assertion; leap-year spot checks (DOY 366).

**4.5 Anomaly construction.** Standardized anomaly `z = (x − normal) / std`, where `normal` comes from BMD daily/monthly normals where available, else from the CHIRPS long-term `*_avg` columns, else from a within-sample climatology on the *training* period only (never use test-period statistics — leakage).

**4.6 Full EDA checklist (scripts → `notebooks/01_eda.ipynb`):**
- Distributions & skew per variable (rainfall is zero-inflated + heavy-tailed → log1p for plots).
- Seasonality decomposition (STL via `statsmodels`) per district; annual + intra-annual monsoon cycle.
- **Monsoon onset detection**: rolling rainfall threshold method (e.g., first dekad exceeding a rainfall threshold sustained N dekads) — produce onset-date time series per district.
- ACF/PACF per variable (informs context length and ARIMA orders).
- Stationarity: ADF + KPSS tests per variable/district.
- Spatial correlation matrix between districts (drives the "is cross-district transfer easy?" discussion).
- Extreme-value analysis: block-maxima/GEV or POT for rainfall; heatwave days (T2M threshold).
- Missing-data heatmap (matplotlib/seaborn).
- Teleconnection lag-correlation: ONI/IOD/MJO vs rainfall (cross-correlation function).

**EDA figures that go in the paper (keep ~4–5):** (F1) study-area map with districts/stations; (F2) monsoon seasonal cycle of rainfall + T2M with phase shading; (F3) missing-data / coverage heatmap; (F4) rainfall distribution (zero-inflation + tail); (F5) teleconnection lag-correlation panel. The rest go in the appendix.

---

### 5. Benchmark Design

**Target variables (tiered):**
- **Tier 1 (core, ship in v1):** `T2M` (daily mean temperature), `RH2M` (relative humidity), `rfh` (dekadal rainfall).
- **Tier 2 (extension):** `WS2M` (wind), `ALLSKY_SFC_SW_DWN` (solar), and a **derived WBGT-style heat-stress index** from T2M+RH2M (approximate WBGT via a documented empirical formula; label as heat-stress proxy).

**Forecast horizons:**
- Daily variables: **h ∈ {1, 3, 7, 14, 30} days.**
- Dekadal rainfall: **h ∈ {1, 2, 3, 6} dekads** (10, 20, 30, 60 days).

**Context lengths:** test **{96, 336, 512}** for daily (bounded by each model's max context, §6); **{36, 72}** dekads for rainfall. Report sensitivity to context length as a mini-ablation.

**Splits (strictly temporal, no leakage):** e.g., **train ≤ 2017, validation 2018–2020, test 2021–2024** (adjust to actual coverage; NASA POWER/CHIRPS both go back to ~1981). Zero-shot FMs use no train split (only context windows drawn from the pre-test history); baselines requiring fitting use train+val only.

**Backtesting protocol:** **rolling-origin evaluation** with a fixed stride (e.g., every 7 days for daily, every dekad for rainfall) over the test period → aim for **≥100 evaluation windows per district/variable/horizon** so significance tests have power. Cache the exact window index list for reproducibility.

**Frequency-mismatch handling:** evaluate daily and dekadal tasks in separate leaderboards; provide the aligned-aggregate task as a bridge (see §4.3).

**Strata & subsets:**
- **Seasonal strata:** pre-monsoon (Mar–May), active monsoon (Jun–Sep), retreat/break, dry (Nov–Feb) — define phase per date using onset detection.
- **Extreme subset:** windows whose target period overlaps a `Hazard_Weather` event or exceeds a percentile threshold (e.g., rainfall > 95th pct) → "extreme-conditional" leaderboard.
- **Onset/transition subset:** windows spanning detected monsoon onset dates.

---

### 6. Models and Baselines

All models below run **zero-shot inference on a single ≤16 GB GPU**; the largest (TimesFM 2.0 500M, Moirai-large ~300M) fit comfortably. Inference for ~20–40 districts × 3 variables × 5 horizons × ~100 windows is minutes-to-low-hours per model on a T4. **Verify all version suffixes live on each model card** (versions below are as of ~mid-2025).

**(i) Naive / statistical baselines** — via **Nixtla `statsforecast`** (fast, CPU) and **Darts**:
- Seasonal-naive, climatology (day-of-year mean from train), drift, ARIMA/`AutoARIMA`, SARIMA, ETS, Theta.

**(ii) ML baselines** — **LightGBM** / **XGBoost** with engineered lag + calendar + teleconnection features (via `mlforecast` for clean lag pipelines).

**(iii) Deep-learning baselines** — **Nixtla `neuralforecast`** and/or **Darts**:
- LSTM, N-BEATS, N-HiTS, DLinear, PatchTST, TiDE, TFT. Train on one GPU; each is small.

**(iv) Time-series foundation models (zero-shot):**

| Model | HF ID / repo | Max context | Probabilistic? | Fine-tune on 1 GPU? | Install |
|---|---|---|---|---|---|
| **Chronos-Bolt** | `amazon/chronos-bolt-{tiny,mini,small,base}` | 2048 ctx, ≤64 horizon (direct) | Yes (quantiles) | Yes | `pip install chronos-forecasting` |
| **Chronos (T5)** | `amazon/chronos-t5-{tiny…large}` | 512 | Yes (sampling) | Yes | `pip install chronos-forecasting` |
| **TimesFM 2.0** | `google/timesfm-2.0-500m-pytorch` | up to 2048 | Yes (10 quantile heads) | Yes | `pip install timesfm` |
| **Moirai / Moirai-MoE** | `Salesforce/moirai-1.1-R-{small,base,large}`, `Salesforce/moirai-moe-1.0-R-{small,base}` | ~5000 | Yes (distributional, any-variate) | Yes | `pip install uni2ts` |
| **Granite TTM** | `ibm-granite/granite-timeseries-ttm-r2` (also r1) | short fixed (e.g., 512→96, config-specific) | point (+ some quantile via config) | Yes (designed for cheap fine-tune) | `pip install granite-tsfm` |
| **MOMENT** | `AutonLab/MOMENT-1-{small,base,large}` | 512 | point (forecast head) | Yes | `pip install momentfm` |
| **Lag-Llama** | HF `time-series-foundation-models/Lag-Llama`; GitHub `time-series-foundation-models/lag-llama` | configurable lags | Yes (Student-t head) | Yes | clone repo |

Optional/commercial or extra: **TimeGPT** (Nixtla API, not open weights — use only if you accept API dependence), **Timer** (`thuml/Large-Time-Series-Model`), **UniTS** (`mims-harvard/UniTS`), **TOTEM**, **VisionTS** (`Keytoyze/VisionTS`).

**(v) Few-shot / LoRA fine-tuning (extension track):** Chronos-Bolt, TimesFM 2.0, Moirai, and especially **Granite TTM** support lightweight fine-tuning that fits in 16 GB. Use LoRA/PEFT where the repo supports it; TTM is explicitly built for few-shot fine-tuning on modest hardware. Report zero-shot vs few-shot as an ablation.

**Library recommendations & version pitfalls:**
- **GluonTS** underpins several loaders; pin versions — GluonTS + newer NumPy/pandas frequently break.
- **Nixtla `statsforecast`/`neuralforecast`/`mlforecast`** are the cleanest for baselines; keep them in one env.
- **`uni2ts`** (Moirai) and **`granite-tsfm`** (TTM) have conflicting torch/transformers pins vs `timesfm` — **isolate each foundation model in its own venv/conda env** and cache predictions to disk so they never need to coexist.
- **`autogluon-timeseries`** can run Chronos + baselines together and is a good sanity-check harness, but for control use direct APIs.
- Pin `transformers`, `torch`, `numpy`, `pandas` per env; record `pip freeze` per model in `env/`.

---

### 7. Evaluation Protocol

**Metrics and rationale:**
- **MASE** (scaled by in-sample seasonal-naive error) — primary point metric; scale-free, comparable across variables/districts, penalizes beating the naive baseline.
- **RMSE, MAE** — familiar, per-variable in physical units.
- **sMAPE** — reported but de-emphasized for rainfall (unstable near zero; zero-inflation makes it noisy).
- **CRPS** — primary probabilistic metric for models with distributional output (Chronos, TimesFM, Moirai, Lag-Llama).
- **Pinball/quantile loss** at {0.1, 0.5, 0.9} and **prediction-interval coverage** (nominal vs empirical, e.g., 80/90% PI) — calibration is a key story for FMs.

**Aggregation:** report per (variable, horizon, season, model); aggregate across districts by **mean and median of MASE** (median resists outlier districts). Never average RMSE across variables of different units — aggregate only scale-free metrics.

**Statistical significance:**
- **Diebold–Mariano** tests for pairwise forecast-accuracy comparison per series.
- **Wilcoxon signed-rank** across districts for a given model pair.
- **Bootstrap 95% CIs** on aggregate MASE (resample windows/districts).
- **Friedman test + Nemenyi post-hoc → critical-difference (CD) diagram** across all models over all tasks (the standard TS-benchmark visualization; use `scikit-posthocs` / `autorank`).

**Compute cost as a first-class result:** for every model log wall-clock inference time, peak VRAM (`torch.cuda.max_memory_allocated`), parameter count, and (for fitted models) training time. Present a **compute-vs-accuracy Pareto plot** — a core equity message ("what can a Global-South researcher actually afford to run?").

**Stratified reporting:** every headline table is repeated per season and for the extreme subset; the *gap* between overall and monsoon/extreme performance is the paper's punchline.

---

### 8. Week-by-Week Plan

**Core track (6 weeks).**

**Week 1 — Scoping, literature, novelty lock, repo skeleton.**
- Objectives: finalize novelty (run all §3 queries; fill `NOVELTY_LOG.md`), read ~40 abstracts / 15 deep, set up repo + Zotero + `references.bib`.
- Deliverables: §2 gap table populated with live citations; repo skeleton (§9); dataset licensing memo (§12).
- Scripts: repo bootstrap, `env/` files.
- Risk: discover a scooping paper → pivot framing (see risk register). **Go/no-go checkpoint: is the gap still open?**

**Week 2 — Data pipeline + EDA.**
- Objectives: ingest all sources, build district↔station↔grid crosswalk, QC, anomalies, EDA.
- Deliverables: `data/processed/` parquet files; `notebooks/01_eda.ipynb`; EDA figures F1–F5 drafts; QC report.
- Scripts: `src/data/load_*.py`, `build_crosswalk.py`, `make_anomalies.py`, unit tests.
- Checkpoint: clean, leakage-audited splits committed with a data card.

**Week 3 — Baselines + benchmark harness.**
- Objectives: implement rolling-origin harness; run naive/statistical/ML/DL baselines; establish the numbers FMs must beat.
- Deliverables: `src/eval/backtest.py`, cached baseline predictions, first draft leaderboard (baselines only).
- Checkpoint: harness validated (a deliberate leakage test *fails* correctly); mid-project novelty re-check.

**Week 4 — Foundation models zero-shot.**
- Objectives: run all 6 FMs zero-shot across variables/horizons/seasons; log compute.
- Deliverables: cached FM predictions; full leaderboard tables; compute log.
- Risk: dependency conflicts → per-model venvs, cache to disk. Checkpoint: all models produce valid predictions on all tasks.

**Week 5 — Analysis, figures, significance.**
- Objectives: seasonal/extreme stratification, CD diagram, Pareto plot, skill-vs-horizon, per-district maps; DM/Wilcoxon/bootstrap.
- Deliverables: all paper tables/figures (§10) final; results narrative bullet points.
- Checkpoint: headline result characterized (positive/negative/mixed) → choose framing.

**Week 6 — Writing, release, arXiv.**
- Objectives: write full paper (§11), prepare HF dataset + Zenodo + GitHub, submit arXiv.
- Deliverables: PDF, dataset DOI, tagged GitHub release, arXiv preprint (cs.AI primary; cs.LG + physics.ao-ph cross-list).
- Checkpoint: final novelty re-check 3 days pre-submission; reproducibility dry-run on a fresh Colab.

**Extension track (Weeks 7–10, for a stronger venue).**

- **Week 7 — Few-shot/LoRA fine-tuning** of Chronos-Bolt, TimesFM 2.0, Moirai, Granite TTM; zero-shot vs few-shot ablation.
- **Week 8 — More targets & variables:** WBGT heat-stress proxy, wind, solar; context-length and covariate (teleconnection) ablations; per-district transfer analysis.
- **Week 9 — Operational relevance:** integrate ERA5/ERA5-Land and IMERG as independent validation; optional FFWC operational-forecast comparison (river levels) as a case study; human-impact/exposure discussion using hazard labels.
- **Week 10 — Hardening for peer review:** datasheet for datasets, NeurIPS D&B checklist, expanded limitations/broader-impact, extra reviewers/proofreading, camera-ready formatting for target venue.

---

### 9. Experiment Engineering Practices

**Repo structure:**
```
banglaweatherbench/
  README.md  LICENSE(CC-BY-4.0 data, MIT/Apache code)  requirements.txt
  env/            # per-model pip-freeze files
  configs/        # Hydra YAML: data, splits, models, eval
  src/
    data/         # loaders, crosswalk, anomalies, QC
    models/       # wrappers: baselines, fms
    eval/         # backtest, metrics, significance
    viz/          # figures
  notebooks/      # 01_eda, 02_baselines, 03_fms, 04_analysis
  scripts/        run_all.sh
  tests/          # pytest: date parsing, no-leakage, metric sanity
  data/{raw,processed,predictions_cache}/
  paper/          # LaTeX + references.bib
  NOVELTY_LOG.md
```
**Config management:** **Hydra + YAML**; every run is fully specified by a composed config; log the config hash with results.
**Experiment tracking:** **Weights & Biases** free tier (or MLflow local) — log metrics, configs, compute; use W&B Tables for leaderboards.
**Reproducibility:** global seed set for numpy/torch/python; deterministic flags; record `pip freeze` per env; pin dataset version hashes.
**Prediction caching:** every model writes predictions to `data/predictions_cache/{model}/{var}_{h}.parquet`; evaluation reads only cache → models never need to coexist, and re-running analysis is instant.
**Colab/Kaggle session limits:** checkpoint after each (model, variable) to Drive/Kaggle output; make runs *resumable* (skip already-cached tasks on restart); keep single runs < ~2–3 h. Prefer Kaggle (fixed weekly GPU quota, longer sessions) for long batches.
**Unit tests (pytest):** DOY→date correctness (incl. leap years), no train/test temporal overlap, MASE = 1.0 for seasonal-naive on its own reference, sentinel (-999) removal, monotonic dates.
**One-command reproducibility:** `bash scripts/run_all.sh` (or `make all`) → downloads/loads data, runs baselines + FMs (per-env), builds all tables/figures. Document the exact env-switching in the README.

---

### 10. Results and Figures Plan (full manifest, in paper order)

**Tables:**
- **T1 — Main leaderboard:** MASE (+ RMSE, CRPS) per model, averaged over districts, per variable & horizon. *Supports: overall skill ranking.*
- **T2 — Seasonal stratification:** MASE per model × season (pre-monsoon/active/break/dry). *Supports: monsoon degradation claim.*
- **T3 — Extreme-conditional:** metrics on the extreme/hazard subset. *Supports: tail-failure claim.*
- **T4 — Compute cost:** params, VRAM, inference/train time per model. *Supports: equity/affordability.*
- **T5 — Zero-shot vs few-shot** (extension). *Supports: does adaptation close the gap?*
- **T6 — Dataset summary/datasheet stats** (coverage, districts, variables, span).

**Figures:**
- **F1** study-area map; **F2** monsoon seasonal cycle; **F3** coverage/missing heatmap; **F4** rainfall distribution; **F5** teleconnection lag-corr (EDA).
- **F6 — Skill vs horizon** curves per model/variable. *Supports: horizon degradation.*
- **F7 — Seasonal skill bars** (overall vs active-monsoon gap). *Supports: headline.*
- **F8 — Extreme-conditional performance** plot.
- **F9 — Compute-vs-accuracy Pareto** scatter. *Supports: equity message.*
- **F10 — Per-district skill map** (choropleth of best-model MASE).
- **F11 — Critical-difference diagram** (Friedman-Nemenyi). *Supports: significance of ranking.*
- **F12 — Qualitative forecast examples** (a good and a failure case around monsoon onset with prediction intervals).

---

### 11. Paper Format and Writing Plan

**Template:** use the **NeurIPS Datasets & Benchmarks LaTeX style** even for the arXiv-first version (it forces the datasheet/checklist discipline reviewers want and reformats cleanly). Alternative: generic arXiv article for speed, then reformat for the venue.

**Section-by-section outline (target ~8–9 pages main + appendix):**
1. **Abstract** (~200 words): one-line problem → dataset+benchmark → models evaluated → headline finding (esp. monsoon/extreme gap) → release. Write last.
2. **Introduction** (~1.25 pp): climate vulnerability + observational gap in Bangladesh; the "forecast anywhere" FM promise; the Global-South geographic-bias hypothesis; the 5 numbered contributions (§1).
3. **Related Work** (~1 pp): the 5-bucket taxonomy (§3).
4. **Dataset** (~1.5 pp): sources, harmonization, crosswalk, anomalies, licensing, **datasheet reference**; T6.
5. **Benchmark Design** (~1 pp): tasks, horizons, splits, strata, backtesting.
6. **Models** (~0.75 pp): baselines + FMs + configs (table).
7. **Evaluation Protocol** (~0.5 pp): metrics, significance, compute.
8. **Results** (~1.75 pp): T1–T4, F6–F11; lead with the monsoon/extreme gap.
9. **Discussion** (~0.75 pp): why FMs degrade in the tropics; implications for Global-South deployment.
10. **Limitations & Broader Impact** (~0.5 pp).
11. **Conclusion** (~0.25 pp).
12. **Appendix:** full EDA, per-district tables, hyperparameters, reproducibility & datasheet.

**Strong novelty paragraph (drop in Intro):** state the intersection explicitly ("to our knowledge, no prior benchmark evaluates zero-shot foundation forecasters stratified by monsoon phase, nor isolates Bengal-delta performance, nor quantifies Global-South geographic bias for time-series/weather FMs") and cite the §2 table to show each neighbor's blind spot.

**Checklists to complete:** Datasheet for Datasets (Gebru et al.), NeurIPS D&B checklist, a reproducibility checklist, an explicit Limitations section, and a Broader Impact statement (dual-use minimal; positive climate-adaptation framing).

**Author/affiliation:** Ismail Hossain Polas, Jahangirnagar University, Bangladesh; single or advisor-added author; include ORCID; corresponding email.

**Writing schedule:** Week 5 draft Methods/Dataset (stable early); Week 6 days 1–3 Results+Intro+Related; day 4 Abstract+Limitations+polish; day 5 internal proofread + submit.

---

### 12. Release and Submission Plan

**Dataset release:**
- **Primary:** Hugging Face Datasets (`ipolas/banglaweatherbench`) with a `README`/dataset card + loading script → maximizes ML adoption.
- **Archival DOI:** Zenodo release (citeable, versioned) → required for a later *Scientific Data* descriptor.
- **License:** **CC-BY-4.0** for derived products from **NASA POWER** (public domain, attribution) and **CHIRPS** (open, cite Funk et al. 2015, *Scientific Data*). ERA5/IMERG derivatives are redistributable with attribution ("Generated using Copernicus Climate Change Service information"; NASA public-domain attribution).
- **Do NOT redistribute:** raw **BMD** PDF tables (copyright/terms unclear) and **FFWC** scraped operational data — instead release *derived* anomaly baselines/features and provide code + instructions for users to fetch BMD/FFWC themselves. Document this clearly in the data card.

**Code release:** GitHub `ipolas/banglaweatherbench`, MIT or Apache-2.0 license, `README` with quickstart, `requirements.txt`/`env/`, notebooks, `run_all.sh`, tagged release matched to the arXiv version.

**arXiv mechanics (handle `ipolas`, cs.AI endorsement):**
- **Recommended:** **primary = cs.AI**, **cross-list cs.LG + physics.ao-ph.** This is safe given your endorsement and appropriately reaches ML + atmospheric-science audiences.
- **If you prefer cs.LG primary** (marginally better for an ML benchmark): request a **cs.LG endorsement** first; cross-lists themselves generally don't need separate endorsement. Verify at arxiv.org/help/endorsement.
- **Metadata/abstract:** ≤1920-char arXiv abstract mirroring the paper abstract; include dataset + code URLs in the comments field; add MSC/ACM classes optionally.

**Post-publication promotion:** X/Twitter + LinkedIn thread (map figure + headline gap plot); submit to **Papers with Code** (link dataset + leaderboard); pin the HF dataset; post on ResearchGate; email the authors of GIFT-Eval, WeatherBench 2, and 2–3 Bangladesh climate-ML groups with a short note; submit to the CCAI (Tackling Climate Change with ML) community.

---

### 13. Impact Maximization

**Strategy:** **arXiv-first (priority) → workshop for early visibility → main venue → journal data descriptor.** A benchmark/dataset paper accrues citations by (a) becoming the comparison standard and (b) frictionless dataset reuse — so invest disproportionately in a clean HF dataset + leaderboard.

**Target venues (verify all 2026/2027 dates live on official sites — my search tools failed):**
- **NeurIPS 2026 Datasets & Benchmarks Track** — best fit; deadline historically ~May 2026 (verify at neurips.cc). Prestigious, cited.
- **Tackling Climate Change with ML (CCAI)** workshop at NeurIPS/ICLR/ICML (climatechange.ai) — non-archival, fast visibility; deadlines per host conference.
- **AAAI AI for Social Impact** (aaai.org; AAAI-27 ~ early 2027; deadlines ~Aug 2026 — verify).
- **Environmental Data Science** (Cambridge University Press) — open access; strong fit for ML-ready dataset/benchmark.
- **Artificial Intelligence for the Earth Systems (AIES, AMS)** — open access; earth-systems ML audience.
- **Journal of Hydrology** (Elsevier) — **2024 Impact Factor 5.9 (Clarivate JCR 2024; CiteScore 11.9)** — strong for the hydro-met/rainfall angle.
- **Water Resources Research** (AGU) — verify current IF; good for the water/flood angle.
- **Scientific Data** (Nature) — ideal for a formal **data descriptor** of the released dataset; verify current IF.

**Recommended combo:** arXiv → CCAI workshop (visibility) → **NeurIPS D&B** (flagship) and, in parallel, a **Scientific Data / Environmental Data Science** descriptor for the dataset artifact. Two papers from one body of work, cross-citing each other.

---

### 14. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Scooped mid-project** | Medium | High | `NOVELTY_LOG.md` re-checks at start/wk3/pre-submit; arXiv-first to plant flag fast; if partial overlap, sharpen the monsoon-phase + Bangladesh + compute-equity angle that others lack |
| **FM API/version breakage** | High | Medium | Per-model venvs; pin versions; cache predictions to disk so a later break doesn't invalidate results; record `pip freeze` |
| **Colab/Kaggle timeouts** | High | Medium | Resumable, cached, checkpointed runs < 2–3 h; prefer Kaggle for long batches |
| **Data licensing problems** | Medium | High | Redistribute only NASA POWER/CHIRPS/ERA5/IMERG derivatives (CC-BY-4.0 + attribution); ship code, not raw BMD/FFWC; licensing memo in wk1 |
| **Weak/negative headline** (FMs don't beat baselines) | High | Low→Positive | Pre-commit to negative-result framing (below) — it becomes the contribution |
| **Reviewer: "just an application"** | Medium | Medium | Emphasize benchmark rigor (rolling-origin, CD diagrams, significance), the released dataset, and the novel equity hypothesis |
| **Reviewer: "too few models/variables"** | Medium | Medium | Extension track adds fine-tuning + WBGT + more FMs; state scope explicitly |
| **BMD PDF extraction fails** | Medium | Low | Fall back to CHIRPS `*_avg` and train-period climatology for normals |

**How to publish a NEGATIVE result (this is the paper's strength, not weakness):** If zero-shot foundation models do *not* beat seasonal-naive/SARIMA/climatology on monsoon rainfall, frame the paper as: *"Foundation models inherit a geographic blind spot: strong on smooth temperate-like signals (T2M, RH2M) but no better than a 1981-baseline climatology on tropical convective precipitation, and worst during monsoon onset/active-break transitions."* This is publishable because: (1) it's a rigorous, quantified, falsifiable finding on an important region; (2) it directly tests the field's "forecast anywhere" claim and finds a boundary; (3) the dataset + leaderboard are contributions independent of who wins; (4) negative/limitation findings on FMs are actively sought at D&B tracks and CCAI. Report *where* FMs *do* win (which variables/horizons/seasons) so the result is nuanced, not merely dismissive.

---

## Recommendations (staged, with thresholds)

**Stage 0 — Now (Week 1):** Re-run all §3 novelty queries yourself and complete `NOVELTY_LOG.md`. **Go/no-go threshold:** if you find a paper that already does *monsoon-stratified zero-shot TS-FM evaluation for Bangladesh*, pivot to the differentiator (compute-equity + WBGT + FFWC operational comparison) or narrow to the dataset-descriptor contribution. Otherwise proceed.

**Stage 1 — Weeks 2–4 (build the core):** Ship the data pipeline, baselines, and zero-shot FMs. **Threshold to continue to writing:** you have a valid, leakage-audited leaderboard with ≥100 windows/task and characterized seasonal strata. If the harness fails its leakage unit test, stop and fix before generating any results.

**Stage 2 — Weeks 5–6 (analyze + publish):** Produce all figures/tables, decide the framing based on the actual headline (positive/mixed/negative), write, release dataset+code, submit arXiv (cs.AI primary + cs.LG/physics.ao-ph cross-list). **Threshold:** fresh-Colab reproducibility dry-run passes before you press submit.

**Stage 3 — Weeks 7–10 (only if targeting a top venue):** Add fine-tuning ablations, WBGT, ERA5/IMERG validation, optional FFWC case study, datasheet + D&B checklist. **Threshold to submit to NeurIPS D&B / a journal:** you can defend statistical significance (CD diagram + bootstrap CIs) and have a complete datasheet. If the extension doesn't materially strengthen the story by end of Week 9, submit the workshop version and iterate.

**Venue sequencing:** arXiv → CCAI workshop → NeurIPS D&B (flagship) + *Scientific Data*/*Environmental Data Science* data descriptor in parallel.

---

## Caveats

- **Live verification failed in this environment.** All novelty determinations (A1–A6) are *provisional-strong*, not confirmed against 2025–2026 literature. You **must** re-run the §3 queries — prioritize checking for any named "Bangladesh weather benchmark," any 2025–26 "Global South / geographic-bias" time-series/weather-FM paper, any 2025–26 NASA POWER + CHIRPS Bangladesh ML paper, and ExtremeWeatherBench's regional coverage.
- **Model version suffixes** (e.g., `moirai-1.1-R`, `granite-timeseries-ttm-r2`, `timesfm-2.0-500m-pytorch`) are as of ~mid-2025 — confirm the current IDs, context limits, and license on each model card before coding.
- **Dataset access specifics** (Copernicus CDS migrated to a new endpoint/token system in 2024; NASA Earthdata login required for IMERG) should be re-confirmed against current docs; the `cdsapi` client and endpoints changed recently.
- **Deadlines and impact factors** for 2026/2027 cycles must be verified on official sites; the only externally confirmed figure here is **Journal of Hydrology's 2024 Impact Factor of 5.9 (Clarivate JCR 2024)**. Newer journals (*Environmental Data Science*, *AIES*) may have provisional or no assigned IF yet.
- **arXiv endorsement** behavior can change; confirm at arxiv.org/help/endorsement whether your cs.AI endorsement extends to a cs.LG *primary* submission before choosing your primary category.

---

## One-Page Condensed Checklist (print this)

**WEEK 1 — Scope & novelty**
- [ ] Run all §3 novelty queries; fill `NOVELTY_LOG.md`; verdict OPEN?
- [ ] Read ~40 abstracts / 15 deep; build `references.bib` (Zotero)
- [ ] Repo skeleton + per-model `env/` files + licensing memo
- [ ] GO/NO-GO: gap still open?

**WEEK 2 — Data & EDA**
- [ ] Parse NASA POWER (YEAR+DOY, −999→NaN), CHIRPS dekadal, Hazard xls, BMD PDFs (camelot)
- [ ] District↔station↔grid crosswalk; QC; anomalies (train-only stats)
- [ ] EDA + figures F1–F5; leakage-audited temporal splits committed

**WEEK 3 — Baselines & harness**
- [ ] Rolling-origin backtest (≥100 windows/task); pass leakage unit test
- [ ] Run naive/climatology/SARIMA/ETS/Theta + LightGBM + LSTM/PatchTST/N-HiTS/DLinear/TiDE/TFT
- [ ] Cache predictions; baseline leaderboard; mid-project novelty re-check

**WEEK 4 — Foundation models (zero-shot)**
- [ ] Chronos-Bolt, TimesFM 2.0, Moirai, Granite TTM, MOMENT, Lag-Llama (each own venv)
- [ ] Log VRAM/time/params; cache all predictions

**WEEK 5 — Analysis**
- [ ] Seasonal + extreme stratification; DM/Wilcoxon/bootstrap; CD diagram; Pareto; per-district map
- [ ] Finalize T1–T4, F6–F12; decide framing (positive/mixed/NEGATIVE)

**WEEK 6 — Write & release**
- [ ] Full paper (NeurIPS D&B template); abstract last
- [ ] HF dataset + Zenodo DOI + GitHub tagged release (CC-BY-4.0 data; no raw BMD/FFWC)
- [ ] Final novelty re-check (−3 days); fresh-Colab reproducibility run
- [ ] arXiv submit: primary cs.AI, cross-list cs.LG + physics.ao-ph

**EXTENSION 7–10 (top venue)**
- [ ] Few-shot/LoRA (Chronos-Bolt, TimesFM, Moirai, TTM) ablation
- [ ] WBGT heat-stress target; wind/solar; context & covariate ablations
- [ ] ERA5/IMERG validation; optional FFWC operational case study
- [ ] Datasheet + D&B checklist; submit CCAI → NeurIPS D&B + Scientific Data