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
Time-series foundation models promise that you can pretrain once and then forecast any series without fitting. The evidence comes almost entirely from mid-latitude, data-rich places, and often from reanalysis rather than from instruments. BanglaWeatherBench tests that promise in a tropical monsoon climate, against what stations actually recorded. We compare 8 zero-shot foundation models with 10 trained and statistical baselines at 35 stations of the Bangladesh Meteorological Department, using European stations, co-located reanalysis and gap-matched contexts as controls. Foundation models generally rank above the untuned baselines. Yet with the best of each chosen on earlier years, 26 of 48 corrected tests cannot tell them apart. No model beats a day-of-year climatology on tropical rainfall beyond a one-day lead. For rainfall, temperature and humidity, reanalysis looks more predictable than the stations it stands in for, so benchmarks built on it overstate skill. A long-lead temperature deficit at ten Bangladeshi stations is mostly explained by sparser records rather than climate. Across about 30 monsoon spells, skill is strong in breaks and worse than climatology in active spells, when forecasts matter most. The protocol transfers; the numbers, from one country, may not.

## Categories
- Primary: cs.AI
- Cross-list after it posts: cs.LG, physics.ao-ph

Endorsement is per archive, so request the cross-lists after announcement rather than
at submission. That way you only need standing in cs.AI.

## Comments field
23 pages, 7 figures, 4 tables. Code and benchmark: https://github.com/ihpolash/BanglaWeatherBench ; archived release: https://doi.org/10.5281/zenodo.24358236

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
- The paper cites the v1.0.2 archive (10.5281/zenodo.24358236). If code the paper cites changes
  again before upload, publish a new release and replace that DOI in sections/09_availability.tex
  and CITATION.cff, then rebuild arxiv-submission.tar.gz.
