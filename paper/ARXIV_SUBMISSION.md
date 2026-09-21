# arXiv submission sheet

Upload `arxiv-submission.tar.gz` (TeX source). Do not upload the PDF: arXiv builds it,
and PDFs produced from TeX may be declined.

## Title
BanglaWeatherBench: Monsoon-Stratified Evaluation of Zero-Shot Time-Series Foundation Models on Station Observations

## Authors
Ismail Hossain Polas

## Affiliation
Jahangirnagar University

## Abstract (plain text, paste as one paragraph)
Time-series foundation models promise that you can pretrain once and then forecast any series without fitting. The evidence comes almost entirely from mid-latitude, data-rich places, and often from reanalysis rather than from instruments. BanglaWeatherBench asks what that promise is worth in a tropical monsoon climate, measured against what stations actually recorded. It scores 8 zero-shot foundation models against 10 trained and statistical baselines on 16 daily tasks over 170,571 rolling-origin windows, targeting 35 Bangladesh Meteorological Department stations. Three controls make the result readable: 32 European stations from the same instrument network; NASA POWER reanalysis at the same 77 coordinates; and a context-gap control applied to models and baselines alike. We resample at the forecast origin, because overlapping windows make ordinary tests overconfident. The result is mixed, and the nulls are the useful part. Foundation models are the strongest family, yet the best one is statistically indistinguishable from the best baseline in 29 of 48 comparisons. On station observations it wins 3 of 24; on reanalysis at the same coordinates, 11 of the same 24. Against a day-of-year climatology on tropical rainfall nothing wins at all: not beyond a one-day lead at the stations, nor at any horizon on ten-day satellite rainfall. Reanalysis looks more predictable than the stations it stands in for in 78 of 108 comparisons, so benchmarks built on it flatter these models. At a 30-day lead 4 of 8 models lose significantly more temperature skill in Bangladesh than a gap-matched control; roughly 60% of that gap is sparser station context, not climate. Skill also reverses inside the monsoon: strong in break spells, worse than climatology in active ones, when forecasts matter most. The whole leaderboard reproduces in 4.16 GPU-hours on a free Tesla T4. Compute is not the barrier.

## Categories
- Primary: cs.AI
- Cross-list after it posts: cs.LG, physics.ao-ph

Endorsement is per archive, so request the cross-lists after announcement rather than
at submission. That way you only need standing in cs.AI.

## Comments field
18 pages, 7 figures, 2 tables. Code and benchmark: https://github.com/ihpolash/BanglaWeatherBench ; archived release: https://doi.org/10.5281/zenodo.22825319

## License
CC BY 4.0 is the consistent choice: the BMD source record is CC-BY-4.0 and the repository
already releases under it. Note that arXiv licenses cannot be changed after announcement.

## Endorsement
No institutional email is available, so auto-endorsement will not apply. Start the
submission, let arXiv issue the six-character endorsement code, and send that code plus
the URL to the cs.AI endorser identified in Week 1. The endorser acts at
arxiv.org/auth/endorse. They must have submitted a qualifying number of papers to cs.AI
roughly three to five years prior.

## Before clicking submit
- Read section "Declaration on the use of generative AI" and confirm it describes your
  own practice. It is your declaration.
- Re-check arXiv's current policy on generative AI; it has changed more than once.
- The Zenodo DOI is in the paper (10.5281/zenodo.22825319); no replacement version is needed for it.
