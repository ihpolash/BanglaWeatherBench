# Licensing memo

Status: Week 1 draft (2026-09-15). Re-verify each source's terms before the public release.

## Code
Apache-2.0.

## Released data (benchmark derivatives): CC-BY-4.0

| Source | License / terms | Redistribute? | Required attribution |
|---|---|---|---|
| BMD 35-station daily observations, Zubair et al. 2024 — Mendeley Data, DOI 10.17632/tbrhznpwg9.1 | CC-BY-4.0 | Yes | Cite Zubair et al. (2024), *Data in Brief*, DOI 10.1016/j.dib.2024.111156, and the Mendeley DOI |
| NASA POWER (MERRA-2-based) | Public domain / open, attribution requested | Yes | "Data obtained from the NASA Langley Research Center POWER Project" |
| CHIRPS (via WFP/HDX subnational dekadal rainfall) | Open; cite Funk et al. 2015 | Yes | Funk et al. (2015), *Scientific Data*; HDX dataset page |
| GHCN-Daily (temperate reference track) | Public domain (NOAA NCEI) | Yes | Menne et al. (2012), GHCN-Daily |
| IBTrACS v04r01 North Indian basin (cyclone-conditional subset) | Public domain (NOAA NCEI) | Yes | Knapp et al. (2010) |
| NOAA ISD station history (station coordinates) | Public domain (NOAA NCEI) | Yes | NOAA NCEI Integrated Surface Database |
| Bangladesh admin boundaries (HDX COD-AB, `cod-ab-bgd`) | CC BY-IGO | Yes, with attribution | OCHA / HDX, Bangladesh Subnational Administrative Boundaries |
| ONI (ERSST v6) and DMI (HadISST) indices, NOAA PSL | Public domain (NOAA) | Yes | NOAA PSL; cite ERSST v6 / HadISST |

## Not redistributed
| Source | Why | What we ship instead |
|---|---|---|
| BMD normals PDFs (`data/*.pdf`) | Terms of use not stated | Derived per-station normals/anomaly baselines + extraction code |
| FFWC scrapes (if ever used) | No open-data license | Fetch code only |

## Evaluated model weights (not redistributed; used for non-commercial academic research)
| Model | License |
|---|---|
| amazon/chronos-2, amazon/chronos-bolt-base | Apache-2.0 |
| google/timesfm-2.5-200m-pytorch | Apache-2.0 |
| Datadog/Toto-2.0-313m | Apache-2.0 |
| ibm-granite/granite-timeseries-ttm-r2 | Apache-2.0 (verify on model card) |
| thuml/sundial-base-128m | Apache-2.0 |
| Salesforce/moirai-2.0-R-small | CC-BY-NC-4.0 |
| NX-AI/TiRex | NX-AI community license (non-permissive) |

TimesFM 3.0 is excluded (non-commercial weights).
