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
Time-series foundation models promise that you can pretrain once and then forecast any series without fitting. The evidence comes almost entirely from mid-latitude, data-rich places, and often from reanalysis rather than from instruments. BanglaWeatherBench asks what that promise is worth in a tropical monsoon climate, measured against what stations actually recorded. We compare 8 zero-shot foundation models with 10 trained and statistical baselines at 35 stations of the Bangladesh Meteorological Department. Three controls help distinguish climatic effects from differences in the observational data: European stations from the same network, reanalysis at the same coordinates, and contexts given Bangladesh's gaps in the record. Foundation models generally rank above the untuned, fixed-configuration baselines. Yet with the best of each chosen on earlier years and the tests corrected for multiple comparisons, the two cannot be told apart in 26 of 48 tests. No model beats a day-of-year climatology on tropical rainfall at these stations beyond a one-day lead. For rainfall, temperature and humidity, reanalysis looks more predictable than the stations it stands in for, so benchmarks built on it overstate skill. A long-lead temperature deficit in Bangladesh is mostly explained by sparser station records rather than by climate, though it rests on ten stations. Within the monsoon, across about 30 spell events, skill is strong in break spells and worse than climatology in active ones, when forecasts matter most. These results come from one country's stations. The protocol transfers; the numbers may not. The whole leaderboard reproduces in 4.16 GPU-hours on a Tesla T4.

## Categories
- Primary: cs.AI
- Cross-list after it posts: cs.LG, physics.ao-ph

Endorsement is per archive, so request the cross-lists after announcement rather than
at submission. That way you only need standing in cs.AI.

## Comments field
23 pages, 7 figures, 4 tables. Code and benchmark: https://github.com/ihpolash/BanglaWeatherBench ; archived release: https://doi.org/10.5281/zenodo.22825319

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
- The November 2026 revision changed code the paper cites (scripts/audit_claims.py,
  scripts/make_leaderboard_table.py) and added generated tables. The cited archive (v1.0.1,
  10.5281/zenodo.22825319) predates these changes. Publish a v1.0.2 GitHub release, take the new
  version DOI Zenodo mints, and replace the DOI and version label in sections/09_availability.tex,
  CITATION.cff and the README before uploading. Then rebuild arxiv-submission.tar.gz.
