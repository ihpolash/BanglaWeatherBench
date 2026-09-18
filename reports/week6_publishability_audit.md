# Week 6, Verification: are these results worth publishing?

Date: 2026-09-17. Written before drafting the paper, because if the results do not hold the writing is wasted.

## 1. Do the numbers reproduce? Yes, 19 of 19

Every headline claim was recomputed from the artefacts and compared with what the reports assert
(`/tmp/audit_claims.py`, rerunnable):

| Claim | Recomputed | Asserted |
|---|---|---|
| FM wins / baseline wins / ties, of 48 | 14 / 5 / 29 | 14 / 5 / 29 |
| BMD rainfall lead 7 & 30: any FM above climatology | max −0.0040 | none |
| CHIRPS dekadal: any FM above climatology | max −0.0631 | none |
| Reanalysis-vs-observation significant | 78 / 108 | 78 / 108 |
| Tavg lead 30, both-masked: models significant | 4 of 8 | 4 of 8 |
| Tavg lead 30, both-masked median DiD | +0.043 | +0.043 |
| Same at 2,048-day context | +0.052 (vs +0.052 at 1,024) | unchanged |
| Spell contrasts significant | 61 / 72 | 61 / 72 |
| Nemenyi critical difference | 6.58 | 6.58 |
| Full zero-shot leaderboard compute | 4.16 GPU-hours | ~4.2 |
| Daily windows / tasks / models | 170,571 / 16 / 18 | same |

**Two overstatements were found in my own reports and corrected**, both now scoped in `week5_significance_report.md`
and `README.md`:

- *"No model beats climatology on tropical rainfall"*, true for the 35-station BMD record and CHIRPS, but on the
  10-station GHCN-Bangladesh track the best model holds positive point estimates at every lead (Toto-2.0 +0.095 at
  day 1; Chronos-2 +0.019 at day 30), none significant (Holm p = 0.88, 1.00, 1.00).
- *"Rainfall shows no tropical penalty at any lead, in any variant"*, four models at lead 1 are significant in the
  *opposite* direction, and Toto-2.0 at lead 7 is significant *with* a penalty (+0.028, CI 0.005–0.053). The honest
  statement is "no systematic penalty", with that exception named.

Neither changes a headline. Both would have been caught by a referee.

## 2. Is the data spine sound? Yes

Week 1 set a second go/no-go: if the published BMD record is pervasively imputed, the observational spine collapses.
Re-checked on the test period (2016–2023, 102,270 observed station-days per variable):

| Variable | Flagged as possibly imputed |
|---|---|
| Rainfall | 0.08% |
| Temperature | 1.26% |
| Sunshine | 1.75% |
| Humidity | 4.44% |

Worst single station-variable is 14.1%; only 11 of 140 station-variables exceed 5%. Scored windows require a
complete **and unflagged** target, so flagged values are excluded from scoring rather than silently included.
**Go/no-go passes.**

## 3. Is the gap still open? Yes (R2 re-check, `NOVELTY_LOG.md`)

Both papers the claim is scoped against were verified from primary sources, and one turned out weaker than assumed:

- **TIME** (ICML 2026): 50 fresh datasets, 98 tasks, 12 foundation models, leakage-free by construction, no
  station, tropical or precipitation stratification. Confirms we must not claim leakage-free evaluation as novel.
- **Brazil / GraphCast** (2606.06348): GraphCast vs IFS HRES over four Brazilian sub-regions, ground truth is
  operational IFS *analysis*, variables T850/Q850/Z500, **precipitation excluded**. Not station-based, no
  precipitation, our scoped claim is comfortably distinct.
- Near misses checked and cleared: Indian monsoon **onset** forecasting (2603.07893, AIWP + Bayesian farmer model,
  operational); **RainfallBench** (2509.25263, 0–6 h GNSS nowcasting, supervised architectures).

No work found at the intersection: TSFMs × station observations including precipitation × monsoon-phase strata.

## 4. What will referees attack, and can it be defended?

| Objection | Strength | Defence |
|---|---|---|
| "Most of your results are null" | Fair, and it is the point | Nulls are the contribution: a boundary on the zero-shot claim, established with a significance protocol none of the seven reference benchmarks runs. Frame as mixed, mechanism-led |
| "Only 10 GHCN-Bangladesh stations carry the equity claim" | **Strongest objection** | Honest limitation. Mitigations in hand: the BMD 35-station track is the primary spine; the equity contrast uses a same-network temperate control, a gap-matched control on both sides, and a 2,048-day context test. State the n explicitly next to every equity number |
| "One country" | Fair | Scope the title and claims to Bangladesh/South Asia; the protocol is the transferable contribution |
| "Zero-shot only, why no fine-tuning?" | Fair | Scope is zero-shot; say so, and note few-shot is the obvious extension |
| "Model roster will age" | Fair for any benchmark | Released harness + cache; adding a model is one run |
| "CHIRPS dekadal is coarse" | Minor | It is an independent-source check, not a headline |
| "Imputation in the published BMD data" | Minor, now quantified | Section 2: ≤4.4% flagged, excluded from scoring |
| "Sunshine vs radiation are not the same quantity" | Minor | Already flagged in the report; do not lean on that row |

## 5. Verdict

**GO.** The evidence is reproducible, the data spine is sound, the gap is open at the stated intersection, and the
claims survive multiplicity correction once scoped. The paper's strength is not a leaderboard win, it is that the
negative and mechanism results are defensible: nothing beats climatology on tropical rainfall; reanalysis flatters
models against the stations they represent; skill reverses by monsoon regime; and the tropical temperature deficit
is mostly a data-availability effect with a residual for two models.

**Venue.** arXiv now (cs.AI primary, cs.LG + physics.ao-ph cross-list), then a benchmarks/datasets track. The
significance protocol and the controls are what make it more than a regional leaderboard.

## 6. Blockers to clear before submission

1. **Citations.** 25 arXiv IDs are cited across the reports; only 10 are held as PDFs. Every reference must be
   opened and verified before it enters the bibliography, no exceptions.
2. **Provenance.** The repository had zero commits until today; the manuscript must be committed incrementally.
3. **Similarity check** runs *before* posting the preprint, not after.
