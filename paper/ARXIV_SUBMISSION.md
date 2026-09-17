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
Time-series foundation models are presented as zero-shot forecasters for arbitrary series, but the evidence is drawn overwhelmingly from mid-latitude, data-rich and frequently reanalysis-derived sources. We introduce BanglaWeatherBench, a benchmark of daily weather forecasting on tropical-monsoon station observations: 16 tasks and 170,571 rolling-origin windows over 35 Bangladesh Meteorological Department stations, 10 co-located GHCN-Daily stations, 32 European GHCN-Daily stations as a temperate reference, and NASA POWER reanalysis at all 77 station coordinates, with an independent CHIRPS dekadal rainfall task. We score 8 zero-shot foundation models against 10 trained and statistical baselines, resampling by forecast origin with Diebold-Mariano tests and Holm correction across 48 comparisons. Foundation models are the strongest family, yet the best one separates from the best baseline in only 14 of 48 comparisons, loses 5, and is indistinguishable in 29; on station observations it wins 3 of 24, all at a one-day lead, against 11 of 24 on reanalysis at the same coordinates. No model beats a day-of-year climatology on rainfall at the 35 stations beyond a one-day lead, or on dekadal rainfall at any horizon. Reanalysis looks more predictable than the stations it represents in 78 of 108 comparisons. At a 30-day lead, 4 of 8 models lose significantly more temperature skill in Bangladesh than a gap-matched trained control, with roughly 60% of that penalty attributable to sparser station context rather than to climate. Skill also reverses by monsoon regime, and the full leaderboard reproduces in 4.16 GPU-hours on a free GPU.

## Categories
- Primary: cs.AI
- Cross-list after it posts: cs.LG, physics.ao-ph

Endorsement is per archive, so request the cross-lists after announcement rather than
at submission. That way you only need standing in cs.AI.

## Comments field
17 pages, 7 figures, 1 tables. Code, benchmark definition and cached forecasts: https://github.com/ihpolash/BanglaWeatherBench

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
- The Zenodo DOI is not in the paper. Add it in a replacement version once a release
  is tagged.
