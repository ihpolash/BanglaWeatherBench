# Claim set (written before prose; every claim names its evidence and its strongest objection)

Rule: a sentence in the paper may not say more than the row below it. Where a claim is scoped, the scope is part
of the claim, not a caveat added later.

## C1. Zero-shot foundation models lead the leaderboard, but most margins do not survive correction
- **Evidence**: `reports/significance_pairwise.csv` (14 wins / 5 losses / 29 ties of 48, Holm-adjusted DM with HAC);
  `reports/significance_ranks.csv` (Nemenyi CD 6.58; top six FMs plus PatchTST inside one CD at leads 7 and 30);
  `reports/leaderboard_daily.csv` (mean ranks).
- **Scope**: 16 daily tasks, 170,571 rolling-origin windows, 2016--2023 test split.
- **Strongest objection**: "You are reporting a null because your test is underpowered." Answer: the protocol is
  stated in advance, the same test finds 14 significant wins, and the wins concentrate exactly where series are
  smooth (lead 1, reanalysis tracks) --- a pattern, not an absence of power.

## C2. Nothing beats climatology on tropical rainfall, at the horizons that matter
- **Evidence**: `reports/leaderboard_daily.csv` (BMD rainfall, all FMs $\le -0.004$ CRPSS at leads 7 and 30);
  `reports/leaderboard_dekadal.csv` (CHIRPS dekadal, best FM $-0.063$ at 144 dekads, $-0.001$ with 28 years of
  context); all 5 baseline wins in C1 are climatology on rainfall.
- **Scope**: the 35-station BMD record beyond day 1, and CHIRPS dekadal at every lead. **Not** unscoped: on the
  10-station GHCN-Bangladesh track the best FM holds positive point estimates at every lead (Toto-2.0 $+0.095$ at
  day 1; Chronos-2 $+0.019$ at day 30), none significant (Holm $p = 0.88, 1.00, 1.00$).
- **Strongest objection**: "Daily station rainfall is unpredictable, so this is trivial." Answer: it is exactly the
  variable the zero-shot claim is sold on, and the result is a boundary, not a triviality --- stated together with
  where the models *do* win.

## C3. Reanalysis looks more predictable than the stations it represents
- **Evidence**: `reports/significance_obs_vs_reanalysis.csv` (78 of 108 model-variable-lead comparisons
  significant; humidity significant at every lead for all 9 models; rainfall at leads 1 and 30 for all 9).
- **Scope**: NASA POWER against co-located stations, identical coordinates and dates, skill measured against each
  track's own climatology.
- **Strongest objection**: "You are comparing different quantities." Answer: same coordinates, same dates, same
  protocol, skill relative to same-track climatology; and the sunshine-vs-radiation row, where the quantities truly
  differ, is the weakest and is not leaned on. Precedent for AIWP models: MAUSAM (JAMES 2026).

## C4. The tropical temperature deficit is real for half the roster and mostly a data-availability effect
- **Evidence**: `reports/significance_equity.csv` (Tavg lead 30, control gap-matched like-for-like: 4 of 8 models
  significant --- TTM $+0.187$, Chronos-Bolt $+0.138$, Chronos-2 $+0.060$, TiRex $+0.052$; median $+0.112 \to
  +0.043$); `reports/significance_equity_ctx2048.csv` (median $+0.052$ at both 1,024 and 2,048 days).
- **Scope**: temperature at 30 days. At lead 1 no model is significant. Rainfall shows **no systematic** penalty ---
  four models at lead 1 are significant in the opposite direction; the one exception is Toto-2.0 at lead 7
  ($+0.028$, CI $0.005$--$0.053$).
- **Strongest objection**: "Ten GHCN-Bangladesh stations cannot carry this." **This is the paper's weakest point.**
  Answer: state $n$ beside every equity number; the contrast uses a same-network temperate control, a gap-matched
  control applied to models *and* baselines, and a context-length test; and the direction is corroborated by the
  BMD 35-station track for the same variable.

## C5. Skill reverses by monsoon regime, which an aggregate leaderboard hides
- **Evidence**: `reports/significance_spells.csv` (61 of 72 contrasts significant, intervals blocked by spell
  *event*); break spells: Chronos-Bolt $+0.440$ at lead 30; active spells: Chronos-Bolt $-0.989$, all models worse
  than climatology at leads 7 and 30.
- **Scope**: BMD stations, July--August, spells defined on the national mean series after Ferdoushi et al. (2023).
- **Strongest objection**: "Your effective sample is the number of spells, not station-days." Answer: agreed ---
  which is why intervals are blocked by event, and the count of events is reported.

## Supporting, not headline
- **Leakage audit**: no pretraining corpus names GHCN-Daily, BMD or any Bangladesh source; the station corpora in
  play (USHCN, SubseasonalClimateUSA) are US-only and all 32 temperate reference stations are European.
- **Cost**: the full zero-shot leaderboard reproduces in 4.16 GPU-hours on a free Tesla T4; AutoARIMA alone costs
  ~20.7 CPU-hours.
- **Calibration**: TimesFM, Toto and TiRex near nominal; Chronos-2 sharp on rainfall (0.63 coverage); Sundial
  under-dispersed by a factor of ~2.4, verified as a model property rather than an adapter artefact.

## Claim wording constraints (from NOVELTY_LOG R1/R2 --- these are binding)
1. Novelty is scoped: **first for time-series foundation models, on station observations including precipitation,
   stratified by monsoon phase, in Bangladesh/South Asia.** Never "first regime-stratified" or "first in the Global
   South" unqualified.
2. Leakage-free zero-shot evaluation is **not** claimed as novel; cite TIME (arXiv 2602.12147, ICML 2026).
3. Cite MAUSAM (JAMES 2026, doi 10.1029/2025MS005568) as the AIWP precedent for C3.
4. Describe arXiv 2606.06348 accurately: GraphCast vs IFS HRES over Brazil, scored against IFS *analysis*, with
   precipitation **excluded**.
5. Report nulls in the same voice as positives. No ordering claimed inside the Nemenyi critical difference.
