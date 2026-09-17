# Week 4 — Zero-shot time-series foundation models

Date: 2026-09-16. Status: **Week 4 complete.** All 8 foundation models ran on the 16 daily tasks, the gap-matched ablation, the compute-cost profile, the context-length ablation and the CHIRPS dekadal task, and the AutoARIMA baseline finished the Week-3 roster. Every forecast file passed verification before install.

Results R1–R5 cover the main daily run; R6–R9 the completion runs. All numbers are point estimates: confidence intervals and significance tests are Week 5.

## Protocol
- **Same harness, windows and scoring as Week 3.** 16 daily tasks and 170,571 windows; 30-day horizon; median point rule; CRPS headline; skill against same-track climatology.
- **Zero-shot.** No fitting on BanglaWeatherBench data. Contexts end exactly at each origin.
- **Main context length: 1,024 days**, capped at each model's maximum (below). The context-length ablation (R7) covers 96 / 336 / 512 / 1,024 / 2,048 days, and 4,096 for Chronos-2.
- **Missing values.** Models that accept gaps natively receive NaN; the others receive linearly interpolated contexts (edges filled with the nearest value). History before a series' first observation is trimmed. The policy is recorded per task in `reports/fm_run_<model>.jsonl`.
- **Physical constraint.** Rainfall, sunshine and radiation forecasts are clipped at 0, as for the baselines.
- **Compute.** Kaggle, 2× Tesla T4, one isolated virtual environment per model (`kaggle/fm_kernel/kernel.py`), two model groups in parallel. The private dataset `ipolas/bwb-bundle` holds the code and processed tracks.
- **Code.**
  - `src/bwb/models/fm_base.py`: common interface `predict_batch(contexts, horizon) → (mean [B,H], quantiles [B,9,H])` and the gap helpers; 5 unit tests.
  - `src/bwb/models/fm_adapters.py`: the 8 adapters.
  - `scripts/run_fm_zero_shot.py`: runner writing the shared prediction cache.

## Roster and verified API facts (2026-09-15)
| Model | Weights (license) | Max context used | Gaps | Output | Verified from |
|---|---|---|---|---|---|
| Chronos-2 | `amazon/chronos-2` (Apache-2.0) | 1,024 (model allows 8,192) | native | 9 quantiles + mean via `predict_quantiles` | README + **local test** (list output, NaN accepted, no NaN out) |
| Chronos-Bolt | `amazon/chronos-bolt-base` (Apache-2.0) | 1,024 (model allows 2,048) | native | 9 quantiles + mean | **local test** (batched tensor, NaN accepted) |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` (Apache-2.0) | 1,024 (config cap) | interpolated | `quantile_forecast[...,0]` mean, `[...,1:10]` = q0.1–q0.9 | HF model card code |
| Toto-2.0 | `Datadog/Toto-2.0-313m` (Apache-2.0) | 1,024 | native (`target_mask`) | quantiles (9, B, V, H) | HF model card code |
| TiRex | `NX-AI/TiRex` (NX-AI community license) | 1,024 | interpolated | quantiles + mean; `backend="torch"` | README + `load_model` source |
| Moirai-2.0 | `Salesforce/moirai-2.0-R-small` (CC-BY-NC-4.0) | 1,024 (model allows 1,680) | native (GluonTS) | quantiles via GluonTS forecast objects | uni2ts README |
| TTM r2 | `ibm-granite/granite-timeseries-ttm-r2` (Apache-2.0) | **512** (daily limit) | interpolated | **point only** | HF repo refs + `get_model` source |
| Sundial | `thuml/sundial-base-128m` (Apache-2.0) | 1,024 (model allows 2,880) | interpolated | 100 sampled paths → quantiles and mean | HF model card code |

## Model-specific constraints that affect comparability
- **TTM r2 cannot use the common 1,024-day context on daily data.** The only daily-capable variants are release r2.1 (`512-96`, `512-48`, `360-60`, `180-60`, `90-30`, `52-16`). `get_model(freq="D")` selects the 512-day model and truncates its output to 30 days. TTM therefore runs at 512 days, which is flagged in every table.
- **TTM is point-only.** Its quantiles equal the point forecast, so its CRPS reduces to absolute error. It is excluded from calibration comparisons.
- **TiRex runs without its fast path.** Its xLSTM CUDA kernels need compute capability ≥ 8.0; the T4 is 7.5, so it uses the PyTorch backend. Accuracy is unaffected; only speed is.
- **Sundial needs `transformers==4.40.1`,** which conflicts with the others. Hence the per-model environments.
- **Licenses.** Moirai-2 (CC-BY-NC-4.0) and TiRex (NX-AI community license) are used for non-commercial academic evaluation only; no weights are redistributed (see `LICENSING.md`).

## Local validation before Kaggle
The runner and the Chronos-2 and Chronos-Bolt adapters ran end to end on CPU: all 4 BMD variables, 16 windows each, valid outputs and run records. Smoke runs (`--limit-windows`) never write to the prediction cache.

Two issues were found and fixed before upload:
1. The TTM adapter originally built 1,024- or 1,536-day inputs. That is incompatible with the daily r2.1 variants; it is now capped at 512 with a daily frequency token for frequency-tuned variants.
2. A Week-1 smoke file sat in a `chronos2` cache folder, and `build_leaderboard.py` excluded any folder with that name. It would have silently dropped the real Chronos-2 results. The file was moved to `data/interim/week1_smoke/` and the exclusion removed.

## Gap-matched ablation (equity-test control)
**Why.** Over the last 365 days of each forecast context, GHCN-Bangladesh contexts are ~82–86% observed, against ~99–100% for temperate stations. A tropical-vs-temperate skill gap could therefore come from missing context data rather than climate or pretraining imbalance.

**How** (`src/bwb/eval/gap_matching.py`, runner flag `--gap-matched`, cache tag `_gapmatched`):
1. For every temperate window, draw a real missing-day pattern: the NaN mask of the last *context-length* days of a GHCN-Bangladesh window, same variable.
2. Apply it right-aligned at the origin. Donor assignment is deterministic (seed 0), so every model sees identical masks.
3. Targets are never modified.
4. Each model's normal gap policy then applies: native NaN handling or interpolation.

**Validation on the real data** (median observed share of the last 365 context days):

| Variable | Temperate before | Temperate after masking | GHCN-Bangladesh reference |
|---|---|---|---|
| Rainfall | 0.995 | **0.808** | 0.819 |
| Tavg | 1.000 | **0.863** | 0.863 |

**Checks:**
- 4 unit tests cover mask length and missingness, short-history handling, right alignment without mutating the input, and deterministic donors.
- A local end-to-end run with Chronos-2 confirmed the flag touches only `ghcn_temperate` and never writes the cache in smoke mode.
- The full-mode Kaggle notebook runs it after each model's main run.

**Reading.** If a foundation model's temperate advantage survives gap matching, context completeness does not explain it.

## Operational log
- **Smoke attempt 1 (Kaggle notebook `ipolas/bwb-fm-zero-shot`, v1) ended in `ERROR` with no diagnostics.**
  - No notebook log, no per-model logs and no results file were produced. The Kaggle API reports only `status: ERROR`, with no failure message.
  - The downloaded output (37 MB) held the bundle copy plus a half-built Chronos-2 virtual environment (7.2 MB, only a `cuda` bindings package). The notebook therefore died during the first model's environment install, before any log file was written.
- **Ruled out.** A duplicated PyTorch install. `chronos-forecasting` 2.3.2 requires `torch>=2.2,<3`, which Kaggle's system PyTorch satisfies, and the partial environment contained no torch or NVIDIA libraries.
- **Monitoring bug found and fixed.** An earlier watcher treated a local DNS failure (`api.kaggle.com` unresolvable) as a terminal status and falsely reported "finished". The watcher now accepts only explicit `COMPLETE` / `ERROR` / `CANCEL` and retries network failures.
- **Hardening for attempt 2** (`kaggle/fm_kernel/kernel.py`):
  - every install and run line is streamed into Kaggle's own notebook log, prefixed by GPU and model;
  - environments are built in `/tmp` (never exported) and deleted after each model;
  - disk and memory are printed before and after each install;
  - per-model install and run time limits;
  - a results file is always written in a `finally` block.
- **Bundle.** The private dataset was re-versioned so the smoke test exercises exactly the code the full run uses, including `gap_matching.py`.
- **Smoke attempt 2 (v2) completed, with 32 windows × 4 BMD variables per model.** The streamed logs made the result fully diagnosable.
  - **3 of 8 models ran cleanly:** TiRex, TimesFM 2.5 and Sundial. This validates their adapters on real data: TiRex's shape normalisation, TimesFM's mean/quantile channel mapping, and Sundial's sampled-quantile path.
  - **5 failed with one shared cause, a PyTorch/torchvision CUDA mismatch.**
    - `uv pip install` pulled a different PyTorch into those environments: 2.14.0+cu130 for Chronos-2, Chronos-Bolt and Toto-2.0; 2.11.0+cu130 for TTM r2; 2.4.1+cu121 for Moirai-2.
    - The environments still saw Kaggle's system `torchvision`, built for torch 2.10.0+cu128.
    - `transformers` (and uni2ts) import torchvision and fail with "operator torchvision::nms does not exist", a circular-import error, or TTM's explicit "PyTorch has CUDA Version=13.0 and torchvision has CUDA Version=12.8".
    - Sundial kept the system torch 2.10.0+cu128. TiRex and TimesFM never import torchvision.
    - The adapters for these 5 models are therefore still untested on Kaggle.
  - **Resources were not a factor:** 20.9 GB free in `/kaggle/working`, 1.1 TB in `/tmp`, around 30 GB RAM available throughout.
  - **Timing signal:**
    - TimesFM 2.5 took about 8–9 s per 32-window call, which could make a full run very slow if it holds at scale.
    - Sundial took about 1.4 s per 32 windows after the first call; TiRex about 0.6 s.
    - The next smoke uses 256 windows per variable to measure throughput at realistic batch sizes.
- **Fix for attempt 3.**
  - At start-up the notebook reads Kaggle's system `torch`, `torchvision` and `torchaudio` versions and writes a uv override file (`torch==2.10.0`, etc.). Every environment installs with `--override`, so no model package can pull a mismatched torch.
  - Each environment runs three checks before any model loads:
    1. its torch version equals the system version;
    2. torchvision imports;
    3. the compiled operator `torch.ops.torchvision.nms` is registered.

    The operator probe matters because version equality alone would not catch a same-version torch re-downloaded with a different CUDA build, which is the exact smoke-v2 failure state. Any failure is recorded as stage `torch_mismatch`, and the model is skipped instead of crashing mid-run.
  - Known remaining risks, accepted for a smoke test:
    - the per-command timeout only triggers when the process prints output, so a silent hang would run until Kaggle's session limit;
    - uni2ts may pin numpy or gluonts versions that differ from the system packages inside Moirai-2's environment.
  - Moirai-2 (uni2ts pins torch<2.5) will run on torch 2.10 via the override. Whether uni2ts works on it is exactly what attempt 3 tests.

- **Smoke attempt 3 (v3) completed: all 8 models ran (exit 0), 256 windows × 4 BMD variables each.**
  - The torch pinning fix worked. Every environment reports `torch 2.10.0+cu128 | torchvision 0.25.0+cu128 | cuda True`, and the `torchvision::nms` operator probe passed in all 8.
  - The five models that failed in v2 now run, so all 8 adapters are validated on Kaggle.
  - Steady-state throughput on a Tesla T4, per 256 windows, projected to the 170,571-window daily set:

    | Model | Per 256 windows | Projected full run |
    |---|---|---|
    | TTM r2 | 0.1 s | ~1 min |
    | Moirai-2 | 0.4 s (15.8 s first task, warm-up) | ~5 min |
    | Chronos-Bolt | 1.1 s | ~12 min |
    | Chronos-2 | 1.6 s | ~18 min |
    | TiRex | 1.7 s | ~19 min |
    | Toto-2.0 | 1.9 s | ~21 min |
    | Sundial (100 samples) | ~12.8 s | ~2.4 h |
    | **TimesFM 2.5** | **~63 s** | **~11.6 h** |

    The gap-matched temperate runs add ~22.8k windows (~13%).
  - **Blocker:** as configured, TimesFM 2.5 would push its GPU group to ~14 h. That is beyond the 5 h per-model limit and Kaggle's 12 h session limit. At ~0.25 s per series it appears to process series one at a time. The next fix targets its batch configuration before the full run.

- **TimesFM slowdown: cause found and fixed.**
  - The cause, confirmed in `timesfm/configs.py`: `ForecastConfig.per_core_batch_size` defaults to **1**, and the adapter never set it. Each 256-window call therefore forecast series one at a time, which matches ~0.25 s per series.
  - The adapter now passes `per_core_batch_size=batch_size` (256). The bundle was re-versioned with the fix.
- **Scheduling decision for the full run: no extra timing-only round trip.**
  - The notebook runs in full mode with the smoke-proven structure.
  - TimesFM and Sundial run *last* on their respective GPUs. If either is still slow, its per-model limit (5 h) stops only that model.
  - Worst case stays around 5.5 h, within Kaggle's 12 h session limit. TimesFM's gap-matched pass is skipped automatically if its main pass does not finish.

- **Full run (notebook v4) completed on Kaggle in 232 min. All 8 models exited 0**, including their gap-matched passes. From the run records:

  | Run | Tasks | Windows | Minutes | Context | Gaps |
  |---|---|---|---|---|---|
  | TTM r2 | 16 | 170,571 | 0.6 | 512 | interpolated |
  | Moirai-2 | 16 | 170,571 | 4.2 | 1,024 | native |
  | Chronos-Bolt | 16 | 170,571 | 14.5 | 1,024 | native |
  | TiRex | 16 | 170,571 | 19.9 | 1,024 | interpolated |
  | Chronos-2 | 16 | 170,571 | 20.3 | 1,024 | native |
  | Toto-2.0 | 16 | 170,571 | 22.5 | 1,024 | native |
  | TimesFM 2.5 | 16 | 170,571 | 37.4 | 1,024 | interpolated |
  | Sundial | 16 | 170,571 | 130.3 | 1,024 | interpolated |
  | each `_gapmatched` run | 2 | 22,773 | 0.1–17.5 | as main | as main |

  The TimesFM batch fix took throughput from ~63 s to **3.4 s per 256 windows** (~18× faster).
- **The output download hung, and nothing was installed.**
  - The first `kaggle kernels output` call broke with `BrokenPipeError`.
  - The retry sat at 0% CPU for over an hour, holding only the run records and 6 of Chronos-2's 16 files. A retry loop cannot rescue a transfer that never fails.
  - The process was stopped. The live prediction cache was confirmed to contain no foundation-model files.
  - Fix in `scripts/install_kaggle_fm_outputs.sh`:
    - one download per model folder via `--file-pattern`;
    - each attempt wall-clock limited to 20 min (`scripts/_run_with_timeout.py`, since macOS lacks `timeout`);
    - up to 5 attempts, keeping files across attempts;
    - a folder counts as done at 16 files (main) or 2 (gap-matched).

  Verification still gates installation.
- **Root cause of every download failure: the Mac was sleeping.**
  - `pmset -g log` shows repeated `Maintenance Sleep` / `DarkWake` cycles every 5–15 min, on battery. The power setting is `sleep 1`, i.e. system sleep after 1 min idle.
  - Sleep drops open TCP connections, which produced `BrokenPipeError` and `Response ended prematurely`.
  - It also pauses the monotonic timer that `_run_with_timeout.py` relies on, while wall-clock time keeps running. That is why a probe with a 120 s limit ran 941 s by the wall clock and returned the CLI's own exit code instead of 124.
  - The "hang at 0% CPU" was the process frozen by sleep. The network itself was healthy when awake (kaggle.com in 0.46 s).
  - Two genuine bugs were found and fixed along the way:
    - the timeout helper now kills the whole process group, since killing only `uv run` left the kaggle grandchild holding the pipe (verified: a nested 30 s sleep is stopped at 3 s);
    - the `--file-pattern` path match is confirmed, because the probe saved `predictions_cache/ttm_r2_gapmatched/ghcn_temperate__Rainfall.parquet`.
  - **Fix:** the install runs under `caffeinate -dimsu`, which blocks idle sleep without changing system settings. The machine must stay awake with the lid open for long transfers.

- **Install completed (22:22).**
  - All 144 forecast files (8 models × 16 tasks, plus 8 × 2 gap-matched) passed `scripts/verify_predictions.py`: 0 missing, 0 with problems.
  - Only then were they installed into `data/predictions_cache/`, and the leaderboard and contrasts rebuilt.
  - Four parallel per-folder download lanes, running under `caffeinate`, completed the folders the main installer's own attempts had left short.

## Results

**How to read these numbers.**
- Skill is CRPSS against the same track's climatology: 0 means no better than climatology, and higher is better.
- Every model is scored on the same windows (common-window rule).
- "Tasks" means the 16 track × variable combinations, reported at leads 1, 7 and 30 days.
- No confidence intervals yet. The GHCN-Bangladesh track in particular has only 10 stations (1,154 rainfall and 1,478 Tavg windows), so small differences there must not be read as findings until the Week-5 block bootstrap.

**Where the numbers come from.**
- `reports/leaderboard_daily.csv`, `reports/leaderboard_bmd_phase.csv`, `reports/leaderboard_bmd_spell.csv`
- `reports/week4_contrasts.csv` and `reports/week4_contrasts.log`
- `reports/week4_summary.log`, produced by `scripts/week4_summary.py`

### R1. Overall: zero-shot foundation models lead, but not uniformly
Mean CRPSS over the 16 tasks, and mean rank among the 18 models on the daily leaderboard (lower rank is better). The last three columns count how many tasks each model wins, at leads 1, 7 and 30.

| Model | Lead 1 | Lead 7 | Lead 30 | Mean rank | Wins L1 / L7 / L30 |
|---|---|---|---|---|---|
| **Chronos-2** | 0.296 | **0.081** | **0.063** | **2.77** | 1 / 7 / 8 |
| TiRex | **0.304** | 0.073 | 0.045 | 2.90 | 7 / 2 / 2 |
| TimesFM 2.5 | 0.290 | 0.073 | 0.048 | 4.19 | 1 / 0 / 0 |
| Toto-2.0 | 0.283 | 0.051 | 0.011 | 5.38 | 7 / 1 / 0 |
| Moirai-2 | 0.292 | 0.056 | 0.005 | 6.02 | 0 / 1 / 2 |
| Chronos-Bolt | 0.286 | 0.062 | 0.017 | 6.06 | 0 |
| *PatchTST (best trained)* | 0.270 | 0.044 | 0.018 | 6.96 | 0 / 2 / 0 |
| *AutoARIMA* | 0.203 | −0.018 | −0.053 | 8.38 | 0 / 0 / 1 |
| *NHITS* | 0.266 | 0.010 | −0.011 | 8.65 | 0 |
| *LightGBM* | 0.145 | −0.010 | −0.037 | 9.92 | 0 |
| *Climatology* | 0 | 0 | 0 | 10.12 | 0 / 3 / 3 |
| Sundial | 0.218 | −0.029 | −0.055 | 10.69 | 0 |
| *DLinear* | 0.145 | −0.065 | −0.091 | 12.25 | 0 |
| *AutoETS* | 0.157 | −0.218 | −0.472 | 13.96 | 0 |
| *AutoTheta* | 0.137 | −0.228 | −0.483 | 14.67 | 0 |
| *Persistence* | 0.123 | −0.361 | −0.651 | 15.48 | 0 |
| TTM r2 (512 context, point only) | −0.003 | −0.310 | −0.387 | 15.85 | 0 |
| *Seasonal naive* | −0.366 | −0.377 | −0.366 | 16.77 | 0 |

- **Best FM vs best baseline, per task and lead:** the best foundation model beats the best baseline in **39 of 48** cases (`contrast C`).
  - 6 of the 9 losses are rainfall at leads 7 and 30, where *climatology* is best and no model beats it: BMD, GHCN-temperate, and NASA POWER at the temperate stations.
  - 2 more are temperature at lead 7, where PatchTST wins: BMD Temperature and GHCN-Bangladesh Tavg.
  - The last is GHCN-Bangladesh Tavg at lead 30, where **AutoARIMA (0.046) beats every foundation model** (best: Chronos-2, 0.038).
  - Which baseline is hardest to beat varies by task: PatchTST in 15 of 48, AutoARIMA in 12, NHITS in 11, climatology in 7, LightGBM in 3.
- **Margins are small on observations.**
  - BMD rainfall at lead 1: TiRex 0.028 vs NHITS 0.017.
  - BMD temperature at lead 1: TiRex 0.361 vs PatchTST 0.342.
- Six of the eight FMs rank above every trained baseline. Sundial and TTM r2 do not.
- **TTM r2's low rank is mostly a format effect.** It produces point forecasts only, so its CRPS is its absolute error.
  - On point skill (MAE of the median vs climatology) it is competitive at lead 1 on temperature, humidity and sunshine. On BMD Temperature it scores 0.301, against 0.349 for PatchTST.
  - On zero-inflated rainfall it is poor even on point skill (−0.30 at lead 1), because its point forecast behaves like a conditional mean rather than a median.
  - It also runs with half the context of the others.
- **Sundial is sharp but overconfident** (see R5). Its point skill is among the best (BMD Temperature, lead 1: 0.365), but its probabilistic score suffers.

### R2. Observations vs reanalysis (contrast B): reanalysis inflates apparent skill, for every model family
Skill on NASA POWER minus skill on BMD observations, at identical station coordinates and dates. Positive values mean the reanalysis looks more predictable.

| Family | Lead 1 | Lead 7 | Lead 30 |
|---|---|---|---|
| Foundation models, mean over 8 models × 4 variables | **+0.238** | **+0.162** | **+0.123** |
| Trained baselines, mean over 6 models × 4 variables | +0.225 | +0.122 | +0.073 |

- **By variable** (mean over leads), foundation models vs trained baselines:

  | Variable | Foundation models | Trained baselines |
  |---|---|---|
  | Humidity | +0.355 | +0.314 |
  | Rainfall | +0.209 | +0.203 |
  | Temperature | +0.094 | +0.096 |
  | Sunshine vs radiation | +0.038 | −0.054 |

  The sunshine row is weakest, and those two series are not the same physical quantity.
- **The foundation-model advantage is itself inflated on reanalysis.**
  - Compare the best foundation model's mean skill with the best trained model's, over the matched 8 observation tasks and 8 reanalysis tasks.
  - On observations the margin is +0.028, +0.026 and +0.033 at leads 1, 7 and 30.
  - On reanalysis it is +0.034, +0.048 and +0.056, which is **~1.2–1.8× larger**.
  - A reanalysis-only benchmark would therefore overstate both absolute skill and how far foundation models lead.

  This is the TSFM counterpart of the reanalysis critique MAUSAM makes for AI weather-prediction (AIWP) models (JAMES 2026, doi 10.1029/2025MS005568). Cite it as that, not as a new critique.

### R3. Equity test (contrast A): no tropical penalty for rainfall; a temperature penalty at 7–30 days, partly caused by missing data
- **Definitions.**
  - Gap = CRPSS on GHCN-temperate minus CRPSS on GHCN-Bangladesh.
  - Control gap = the same quantity for the best trained baseline on each track.
  - DiD = model gap − control gap. A positive DiD means the model loses more skill in the tropics than trained models do.
- **Gap-matched DiD** replaces the temperate skill with the model's run on temperate contexts carrying Bangladesh-like missingness. It uses the same control.

Median over the 8 foundation models (control = best trained baseline per track, AutoARIMA included):

| Variable | Lead | FM gap | Control gap | DiD | Gap-matched DiD |
|---|---|---|---|---|---|
| Rainfall | 1 | −0.061 | −0.044 | −0.017 | −0.025 |
| Rainfall | 7 | −0.022 | −0.035 | +0.013 | +0.006 |
| Rainfall | 30 | −0.018 | −0.015 | −0.003 | −0.009 |
| Tavg | 1 | +0.116 | +0.117 | −0.001 | −0.012 |
| Tavg | 7 | +0.054 | −0.011 | **+0.065** | +0.043 |
| Tavg | 30 | +0.135 | +0.011 | **+0.123** | +0.075 |

- **Rainfall.**
  - Foundation models are, if anything, *more* skilful relative to climatology in Bangladesh than at temperate stations (negative gaps).
  - Trained baselines show the same gap (DiD ≈ 0 at every lead).
  - **No foundation-model-specific tropical penalty for rainfall.**
- **Temperature at lead 1:** no penalty (DiD ≈ 0).
- **Temperature at leads 7 and 30:** a foundation-model-specific penalty appears.
  - By model, DiD at lead 7 / lead 30, and the same after gap matching:

    | Model | DiD L7 | DiD L30 | Gap-matched DiD L7 | Gap-matched DiD L30 |
    |---|---|---|---|---|
    | Toto-2.0 | +0.144 | +0.290 | +0.087 | +0.120 |
    | Moirai-2 | +0.128 | +0.278 | +0.058 | +0.102 |
    | TTM r2 | +0.125 | +0.264 | +0.109 | +0.232 |
    | Chronos-Bolt | +0.073 | +0.128 | +0.060 | +0.105 |
    | TimesFM 2.5 | +0.055 | +0.119 | +0.024 | +0.047 |
    | TiRex | +0.057 | +0.092 | +0.029 | +0.042 |
    | Sundial | +0.022 | +0.055 | +0.010 | +0.019 |
    | Chronos-2 | +0.020 | **+0.017** | +0.014 | **+0.004** |

  - **Gap matching removes about a third (lead 7) to ~40% (lead 30) of the median penalty**, and more for individual models: TimesFM's lead-30 DiD falls from +0.119 to +0.047, TiRex's from +0.092 to +0.042, Toto-2.0's from +0.290 to +0.120.
    - Context missingness therefore explains a large share, but a residual penalty remains for most models.
  - **The missing-data policy does not sort the models.**
    - Toto-2.0 and Moirai-2, both with native NaN handling, have the largest penalties.
    - Chronos-2, also native, has essentially none.
- **Status of the claim.** This is suggestive, not established.
  1. There are no confidence intervals yet, and GHCN-Bangladesh has 10 stations.
  2. The control is not itself gap-matched: trained baselines were not re-run on masked temperate contexts.
  3. The control uses the best trained model per track, which can be a different model on each side.
  4. Whether GHCN-Daily series overlap any foundation model's pretraining corpus is **unverified**. Overlap would favour the temperate track.

  Week-5 items address all four.
- **Wording for the paper until then:** "a lead-dependent temperature skill deficit on Bangladesh stations for most zero-shot TSFMs, partly attributable to context missingness". Not a "geographic bias" or "pretraining imbalance" claim.

### R4. Monsoon regimes (BMD, CRPSS vs climatology on target days in each phase)
- **Pre-monsoon rainfall (Kalbaishakhi season) is not predictable beyond climatology by any model, even at day 1.**
  - The best score is −0.015 (Chronos-2 and Moirai-2).
  - Dry-season rainfall is negative for every model too, because climatology is near zero.
  - Only onset and peak-monsoon rainfall carry day-1 skill: TiRex 0.046, Toto-2.0 0.041, Chronos-2 0.040, NHITS 0.043 in the peak phase.
  - At lead 30 no model beats climatology in any phase: the best is +0.003.
- **Temperature.**
  - At lead 1, skill is lowest at monsoon onset for every model: foundation models 0.20–0.24, against 0.37–0.40 in the dry season.
  - At lead 30, only the **peak monsoon** retains skill: Chronos-2 0.147, TiRex 0.148, TimesFM 0.145, against PatchTST 0.080. Dry and pre-monsoon are negative for nearly all models.
- **Active vs break spells**, July–August target days. Spells are defined on the national mean series (Ferdoushi et al. 2023).
  - At leads 7 and 30, **every model loses to climatology on temperature in active spells**, and foundation models lose more:

    | Model | Active-spell Temperature CRPSS, lead 30 |
    |---|---|
    | Chronos-Bolt | −0.997 |
    | TimesFM 2.5 | −0.600 |
    | Chronos-2 | −0.518 |
    | PatchTST | −0.193 |

  - In **break spells** the same models are strongly skilful, and foundation models lead by a wide margin:

    | Model | Break-spell Temperature CRPSS, lead 30 |
    |---|---|
    | Chronos-Bolt | 0.447 |
    | TimesFM 2.5 | 0.383 |
    | Chronos-2 | 0.379 |
    | PatchTST | 0.106 |
    | NHITS | 0.199 |

  - Humidity shows the same active-vs-break reversal.
  - For rainfall in break spells, several foundation models beat climatology at lead 30 while trained baselines do not: TimesFM 0.138, Moirai-2 0.120, Chronos-Bolt 0.118, against NHITS −0.015.
  - **A plausible mechanism, not tested:** long-context models extrapolate the persistent warm, dry state, which pays off when a break continues and fails when an active spell arrives.
  - **Sample size.** Samples are station-days: 378–414 active and 653–862 break per lead for rainfall. They are strongly correlated, because all 35 stations share one national spell label, so the effective sample is the number of spell events. Treat this as the most interesting lead for Week 5, to be tested with an event-block bootstrap, not as a result.
- These are regime-dependent reversals of the kind arXiv 2606.18367 reports for traffic, here with physically defined, externally validated regimes. Cite that paper as precedent for regime-stratified TSFM evaluation.

### R5. Calibration (80% central interval coverage; nominal 0.80; mean over tasks and leads 1/7/30)
| Model | Observation tracks | Reanalysis tracks | Observed rainfall | Min–max over tasks |
|---|---|---|---|---|
| TimesFM 2.5 | 0.831 | 0.827 | 0.854 | 0.777–0.879 |
| Toto-2.0 | 0.826 | 0.810 | 0.853 | 0.777–0.871 |
| TiRex | 0.797 | 0.810 | 0.784 | 0.673–0.847 |
| Chronos-Bolt | 0.781 | 0.807 | 0.740 | 0.584–0.866 |
| Moirai-2 | 0.756 | 0.780 | 0.641 | 0.465–0.875 |
| Chronos-2 | 0.732 | 0.748 | **0.629** | 0.424–0.852 |
| **Sundial** | **0.323** | 0.349 | 0.231 | 0.162–0.460 |
| *PatchTST* | 0.822 | 0.810 | 0.833 | 0.723–0.879 |
| *AutoARIMA* | 0.843 | 0.855 | 0.915 | 0.761–0.934 |
| *Climatology* | 0.838 | 0.798 | 0.897 | 0.698–0.927 |
| TTM r2 | not applicable (point only) | | | |

- **TimesFM 2.5, Toto-2.0 and TiRex are close to nominal.**
- **Chronos-2**, the best model by CRPS, is too sharp on rainfall: its intervals cover only 63% instead of 80%. Moirai-2 is similar.
- **Sundial is severely under-dispersed.**
  - Its mean 80% interval width on BMD Temperature at lead 1 is 0.78 °C, against 2.5–3.0 °C for the other foundation models.
  - On rainfall it is 9.0 mm, against 14–18 mm.
  - The adapter follows the model card: `generate(..., num_samples=100)` → quantiles over samples.
  - **Diagnosed on a GPU (notebook `ipolas/bwb-sundial-check`, 512 BMD temperature windows, context 1,024): the under-dispersion is the model, not the adapter.**

    | Samples | Raw output shape | Mean 80% width (°C) | 80% coverage | Trajectory spread (°C) | RMSE of sample mean (°C) |
    |---|---|---|---|---|---|
    | 20 | (512, 20, 30) | 1.44 | 0.350 | 0.61 | 1.52 |
    | 100 | (512, 100, 30) | 1.56 | 0.386 | 0.63 | 1.51 |
    | 500 | (512, 500, 30) | 1.59 | 0.391 | 0.64 | 1.51 |

    - Widths are flat in the sample count (+1.5% from 100 to 500), so 100 samples was sufficient and sampling noise is not the cause.
    - The raw shape is `(batch, samples, horizon)`, exactly what the adapter's reshape assumes, so the trajectories are not being mixed up.
    - Sundial's own trajectories differ by 0.64 °C while its point forecast is wrong by 1.51 °C, so its predictive distribution is about 2.4× too narrow relative to its own accuracy. That is overconfidence in the model.
    - Sundial's calibration numbers therefore stand as reported, and should be described as a property of the model.

### Open items carried to Week 5
1. **Significance.**
   - Station- and event-block bootstrap CIs for every contrast (A, B, C, spells).
   - Diebold–Mariano and Wilcoxon tests for the best-FM vs best-baseline pairs.
   - Friedman–Nemenyi critical-difference diagrams.
2. **Gap-matched control.** Re-run the trained baselines (at least NHITS and PatchTST on Kaggle; LightGBM and ETS locally) on the gap-matched temperate contexts, so the DiD compares like with like.
3. **Pretraining-overlap audit.** Check each model's published corpus for GHCN-Daily, NASA POWER and ERA5-derived series. BMD station data is unlikely to overlap; the temperate GHCN track is the risk.
4. **Sundial dispersion check** (R5): compare sample spread against an official Sundial evaluation wrapper before the paper. TTM r2 keeps its footnote in every table: 512-day context, point only.
5. **Gap-matched runs at 2,048 days.** R7 shows the long-lead tropical temperature deficit nearly disappears with longer context, but the gap-matched ablation has so far run only at 1,024, so the two explanations — short context and gappy context — are not yet separated.

Week-4 ablations are complete: compute cost (R6), context length (R7), CHIRPS dekadal (R8) and AutoARIMA (R9). The highest-value Week-5 target is the active/break-spell reversal in R4, blocked by spell event rather than station-day.

## Week 4 completion runs: compute cost, context length, CHIRPS dekadal, AutoARIMA
Notebooks `ipolas/bwb-fm-week4b` (2× T4) and `ipolas/bwb-autoarima-a` / `-b` (CPU only, no GPU quota), bundle version with the dekadal data and new scripts. Installed by `scripts/install_kaggle_week4b.sh` (verify before install) and analysed by `scripts/analyze_week4b.py`.

### Protocol
- **Compute cost.** Every run record now carries batch size, peak GPU memory (`torch.cuda.max_memory_allocated`, reset per task) and the parameter count.
  - Parameters are counted by walking each adapter's attributes for torch modules, counting tied weights once (`count_parameters`). Checked locally: Chronos-2 = 119,477,664, against the published ~120M.
  - Peak memory comes from a profile pass: the first 4,096 windows of each BMD variable, context 1,024, batch 256, no cache write.
  - Throughput comes from the main run's wall times.
- **Daily context-length ablation.**
  - Chronos-2 at 96 / 336 / 512 / 2,048 / 4,096 days; TiRex and TimesFM 2.5 at 96 / 336 / 512 / 2,048. The 1,024-day point is the main run.
  - All 16 tasks, identical windows.
  - 96 and 336 days are shorter than one annual cycle; 4,096 days (~11 years) is possible only for Chronos-2.
  - TimesFM 2.5 now compiles its context to the requested length. Its limit is 16,384 steps, context plus horizon; main-run behaviour is unchanged at 1,024.
  - Each run is verified and scored on Kaggle (`scripts/score_fm_runs.py`, the same checks as `verify_predictions.py` and the same harness as `build_leaderboard.py`), and only the scores are shipped.
  - Before upload, the scorer reproduced the main leaderboard exactly for Chronos-2 and TiRex: 40 rows, maximum absolute difference 0.0, identical window counts.
- **CHIRPS dekadal zero-shot.**
  - All 8 models; 18,112 windows over 64 districts; horizon 6 dekads; `scripts/run_fm_dekadal.py`.
  - Context lengths are 36 / 72 / 144 dekads, as pre-registered in `configs/splits.yaml`. The longest, 144 dekads (4 years), is the headline, cached as `<model>`.
  - An extra full-history run at 1,024 dekads (~28 years) is cached as `<model>_ctx1024`.
  - TTM r2 has no 10-day resolution. It runs with `freq="W"`, the nearest supported resolution of at least one day; any such resolution selects the r2.1 branches.
  - `scripts/run_dekadal_baselines.py` now scores every cached dekadal model, after `check_predictions`, and adds CRPSS vs climatology.
- **AutoARIMA** (Week-3 carry-over).
  - Same code and settings as AutoETS/AutoTheta: climatological anomalies, 730-day context, `season_length=1`, StatsForecast 2.0.1.
  - Local timing on 200 BMD windows: 0.10–0.13 s per window on 4 workers, against 0.02 s for AutoETS. That projects to ~5.5 h for all 170,571 windows.
  - It therefore runs on two Kaggle CPU notebooks: observation tracks (part a) and NASA POWER tracks (part b).
  - Once it passes verification it joins the main daily leaderboard, and the Week-4 contrasts and summary are rebuilt with it (`TRAINED` now includes AutoARIMA).

### Operational log
- **Version 1 of all three notebooks failed within seconds.**
  - The GPU notebook reported `FileNotFoundError: bundle not found`; both AutoARIMA notebooks hit `IndexError` from the same lookup.
  - Cause: Kaggle unpacks zip files uploaded to a dataset, so `bwb_bundle.zip` does not exist in `/kaggle/input`. The main-run notebook already searched for the extracted files first; the new notebooks only searched for the zip.
  - Fixed by reusing that lookup. Version 2 of all three was pushed at 23:02.
- **Version 2 (GPU notebook) ran 4 h and every job exited 0 except Toto-2.0's dekadal pass.**
  - `einops.EinopsError: can't divide axis of length 36 in chunks of 32`: Toto reduces its context in patches of 32, and the dekadal contexts (36 / 72 / 144) are not multiples of 32. The daily runs never hit it because 1,024 is.
  - Fixed in the adapter by rounding the padded common length up to the patch size, with the extra left padding masked as unobserved (`round_up_to` in `fm_base.py`, unit-tested on 36/72/144/1,024).
  - The bundle was re-versioned and **version 3 re-ran only that job** (`ONLY = {"toto2": ["dekadal"]}` in the kernel, a documented re-run switch, reset to `{}` afterwards). It completed in ~3 min; all 4 contexts verified and installed.
  - Ordering mattered: v3's output replaces the notebook's downloadable output, so v2's results were downloaded first (28 dekadal runs, 13 score boards, 208 score files, 678 MB) before v3 was pushed.

### R6. Compute cost on one free Tesla T4
Batch 256, context 1,024 (TTM 512). Memory and parameters are from the profile pass; minutes and throughput are the main daily run over all 16 tasks (170,571 windows).

| Model | Parameters (M) | Peak GPU memory (MiB) | Minutes for 170,571 windows | Windows/s | Overall rank (R1) |
|---|---|---|---|---|---|
| TTM r2 | **0.8** | **34** | **0.6** | **4,916** | 17 of 17 |
| Moirai-2 | 11.4 | 369 | 4.2 | 679 | 5 |
| TiRex | 35.3 | 568 | 19.9 | 143 | 2 |
| Chronos-2 | 119.5 | 1,345 | 20.3 | 140 | **1** |
| Sundial | 128.3 | 2,802 | 130.3 | 22 | 11 |
| Chronos-Bolt | 205.3 | 2,182 | 14.5 | 197 | 6 |
| TimesFM 2.5 | 231.3 | 4,181 | 37.4 | 76 | 3 |
| Toto-2.0 | 312.7 | 1,960 | 22.5 | 127 | 4 |

- **The whole zero-shot benchmark is affordable.** Every model fits in 4.2 GiB, well inside a free T4's 15 GiB, and the complete 16-task run costs between 0.6 minutes and 2.2 hours of GPU time.
- **Skill does not follow size.** TiRex ranks second overall with 35M parameters and 568 MiB; Toto-2.0 (313M) ranks fourth and Sundial (128M) eleventh. The best model, Chronos-2, is mid-sized.
- **Cost per unit of skill differs by two orders of magnitude.** Sundial needs 6.4× Chronos-2's wall time and 2× its memory to score worse.
- This is the affordability evidence for the equity framing: a Global-South researcher on free-tier hardware can reproduce the entire zero-shot leaderboard in about 4 GPU-hours.

### R7. How much history the models need (context-length ablation)
Mean CRPSS over the 16 daily tasks. 1,024 days is the main run; all runs use identical windows (0 window-count mismatches).

| Model | Context (days) | Lead 1 | Lead 7 | Lead 30 |
|---|---|---|---|---|
| Chronos-2 | 96 | 0.267 | −0.063 | −0.451 |
| | 336 | 0.284 | 0.018 | −0.114 |
| | 512 | 0.292 | 0.059 | 0.011 |
| | **1,024** | 0.296 | 0.081 | 0.063 |
| | 2,048 | 0.298 | 0.091 | 0.072 |
| | 4,096 | 0.298 | **0.092** | **0.073** |
| TimesFM 2.5 | 96 | 0.251 | −0.071 | −0.495 |
| | 512 | 0.283 | 0.048 | −0.021 |
| | **1,024** | 0.290 | 0.073 | 0.048 |
| | 2,048 | 0.292 | 0.083 | 0.066 |
| TiRex | 96 | 0.275 | −0.061 | −0.429 |
| | 512 | 0.299 | 0.047 | −0.029 |
| | **1,024** | 0.304 | 0.073 | 0.045 |
| | 2,048 | 0.306 | 0.083 | 0.058 |

- **At lead 1 context barely matters** (Chronos-2: 0.267 → 0.298 across a 43× range of context).
- **At lead 30 it decides everything.** With 96 days of history all three models are far worse than climatology (−0.43 to −0.50); they only reach climatology-level skill at 512–1,024 days. A model must see at least one full annual cycle before a monthly forecast is worth anything, which is a concrete requirement for station records in data-sparse regions.
- **Returns saturate.** 1,024 → 2,048 days adds about +0.01; Chronos-2's 2,048 → 4,096 adds +0.001. Per task, 2,048 is best in 15–16 of 16 tasks for TimesFM and TiRex; for Chronos-2, 4,096 wins 8–11 of 16.
- **The main run's 1,024 days is therefore slightly conservative but near-optimal**, and the ranking in R1 is not an artefact of context choice.
- **This changes how R3 should be read.** The tropical-vs-temperate Tavg gap at lead 30 depends strongly on context length:

  | Model | 336 days | 1,024 days | 2,048 days |
  |---|---|---|---|
  | Chronos-2 | 0.408 | 0.029 | **0.008** |
  | TimesFM 2.5 | 0.355 | 0.130 | **−0.004** |
  | TiRex | 0.422 | 0.104 | **0.060** |

  Against the trained-baseline control gap of +0.026, the foundation-model-specific penalty at 2,048 days is −0.018 for Chronos-2, −0.030 for TimesFM and +0.034 for TiRex: essentially gone for two of the three. Rainfall gaps stay near zero at every context length.

  **Reading (superseded by Week 5, S8).** Taken alone, this table suggests the long-lead temperature deficit is largely a context-length effect. Week 5 shows that holds only while the *control* keeps unmasked contexts: once the trained baseline is re-run on the same Bangladesh-like gaps, the median penalty is +0.052 at 1,024 days **and** +0.052 at 2,048 days, so doubling the history changes nothing. What the unmasked comparison reads as a context-length effect is mostly the missingness effect. Chronos-2 and TiRex keep a significant, context-independent penalty; TimesFM 2.5's is genuinely context-related and disappears by 2,048 days. Quote `reports/week5_significance_report.md` S2 and S8 for this claim, not this table.

### R8. CHIRPS dekadal rainfall (10-day, 18,112 windows, 64 districts)
CRPSS vs climatology; headline context 144 dekads (4 years).

| Model | Lead 1 | Lead 2 | Lead 3 | Lead 6 | 80% coverage |
|---|---|---|---|---|---|
| *Climatology* | 0 | 0 | 0 | 0 | 0.758 |
| Chronos-2 | **−0.063** | **−0.083** | **−0.080** | **−0.086** | 0.717 |
| Toto-2.0 | −0.086 | −0.115 | −0.114 | −0.113 | 0.793 |
| TimesFM 2.5 | −0.162 | −0.150 | −0.137 | −0.135 | 0.771 |
| TiRex | −0.169 | −0.219 | −0.257 | −0.323 | 0.794 |
| Moirai-2 | −0.196 | −0.290 | −0.347 | −0.504 | 0.814 |
| Chronos-Bolt | −0.260 | −0.397 | −0.518 | −0.864 | 0.757 |
| Sundial | −0.288 | −0.364 | −0.438 | −0.655 | 0.396 |
| *Persistence* | −0.478 | −0.635 | −0.763 | −1.158 | 0.819 |
| *Seasonal naive* | −0.514 | −0.511 | −0.511 | −0.509 | 0.807 |
| TTM r2 (point only) | −0.682 | −0.868 | −1.002 | −1.529 | n/a |

- **No zero-shot foundation model beats climatology on tropical 10-day rainfall at any lead.** This is the benchmark's cleanest negative result and matches daily rainfall at leads ≥ 7.
- **With all available history (1,024 dekads, ~28 years) they close most of the gap but still do not cross it:** Chronos-2 −0.001 to −0.011, Toto-2.0 −0.004 to −0.017, TiRex −0.030 to −0.044, TimesFM −0.055 to −0.062, Moirai-2 −0.073 to −0.094, Chronos-Bolt −0.105 to −0.215, Sundial −0.147 to −0.168. The best achievable zero-shot result is *matching* a dekad-of-year climatology.
- **Context matters here even more than on daily data** (Chronos-2 at lead 6: −0.999 at 36 dekads, −0.086 at 144, −0.011 at 1,024). Between 36 and 72 dekads several models get slightly worse before improving, so one to two years of dekadal history is not enough to recover the annual cycle.
- **TTM r2 is the exception and degrades with more context** (−0.565 at 36 dekads to −0.846 at 1,024), consistent with its 512-step cap and point-only output.
- **Sundial is under-dispersed again** (coverage 0.396 against a nominal 0.80), matching R5.
- **Toto-2.0 is the second-best model here and the best calibrated of the strong ones** (coverage 0.793 against a nominal 0.80), after its patch-size fix and re-run. Its context curve is the steepest of any model: −0.815 at 36 dekads to −0.015 at 1,024 (lead 6).
- All 8 models therefore have complete dekadal results: 32 runs (8 models × 4 contexts), each verified before install.

### R9. AutoARIMA
All 16 tasks completed on Kaggle CPU workers: part a (observation tracks) in 8.8 h, part b (NASA POWER tracks) in 11.9 h. Part b's runner was killed by its own 11.3 h limit, but every one of its 8 tasks had already been written, and all 16 files passed verification before install. Total cost ~20.7 CPU-hours, against ~4 GPU-hours for all 8 foundation models over the same 16 tasks.

- **Mean CRPSS 0.203 / −0.018 / −0.053 at leads 1 / 7 / 30; mean rank 8.38 of 18.** It is the third-best trained baseline overall, behind PatchTST and NHITS, and clearly ahead of AutoETS and AutoTheta, which collapse at long leads.
- **It is the strongest baseline at day 1 on smooth variables**, and is the best baseline in 12 of 48 task-and-lead cases: BMD Temperature at leads 1 and 30, BMD Humidity at lead 1, Tavg at lead 1 on both GHCN tracks, and NASA POWER radiation at lead 1.
- **It beats every foundation model in one case:** GHCN-Bangladesh Tavg at lead 30 (0.046 against Chronos-2's 0.038).
- **Calibration is good** (coverage 0.843 on observations, 0.855 on reanalysis), unlike AutoETS/AutoTheta which over-cover at 0.90+.
- Adding it to the control set slightly *strengthens* R3: the trained-baseline control gap for Tavg at lead 30 falls from +0.026 to +0.011, so the foundation-model-specific penalty at 1,024 days rises from a median +0.109 to **+0.123**. The context-length result in R7 is unaffected, since it compares gaps within each model.
- **Why it matters for the paper:** the strongest classical baseline is the one the plan originally flagged as too expensive to run. It costs ~5 CPU-hours per 100k windows and still loses to zero-shot foundation models on 39 of 48 tasks, which is the affordability argument in R6 from the other direction.
