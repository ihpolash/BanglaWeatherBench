#!/usr/bin/env bash
# Download public external sources used by the pipeline (idempotent: skips files already present).
# BMD Mendeley data: scripts/download_bmd_mendeley.py. NASA POWER points: scripts/fetch_nasa_power_points.py.
# Temperate GHCN station files: scripts/select_temperate_reference.py.
set -euo pipefail
cd "$(dirname "$0")/.."
X=data/external

fetch() {  # fetch <url> <dest>
  if [ ! -s "$2" ]; then
    mkdir -p "$(dirname "$2")"
    curl -sSL --fail -m 900 -o "$2" "$1"
    echo "downloaded $2"
  fi
}

# GHCN-Daily metadata + the 10 Bangladesh stations
fetch https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt "$X/ghcnd/ghcnd-inventory.txt"
fetch https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt "$X/ghcnd/ghcnd-stations.txt"
for id in BGM00041859 BGM00041883 BGM00041891 BGM00041907 BGM00041923 BGM00041936 BGM00041943 BGM00041950 BGM00041978 BGM00041992; do
  fetch "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/$id.csv.gz" "$X/ghcnd/by_station/$id.csv.gz"
done

# NOAA ISD station history (BMD station coordinates)
fetch https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv "$X/isd/isd-history.csv"

# HDX COD-AB Bangladesh admin boundaries (CC BY-IGO)
fetch "https://data.humdata.org/dataset/401d3fae-4262-48c9-891f-461fd776d49b/resource/cec2abe3-d8b7-4025-9362-9f7e780f2a07/download/bgd_admin_boundaries.geojson.zip" "$X/boundaries/bgd_admin_boundaries.geojson.zip"
[ -s "$X/boundaries/bgd_admin2.geojson" ] || unzip -o -q "$X/boundaries/bgd_admin_boundaries.geojson.zip" -d "$X/boundaries"

# IBTrACS v04r01 North Indian basin
fetch "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv" "$X/ibtracs/ibtracs.NI.list.v04r01.csv"

# Climate indices (NOAA PSL)
fetch https://psl.noaa.gov/data/correlation/oni.data "$X/indices/oni.data"
fetch https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data "$X/indices/dmi.had.long.data"

# Natural Earth 1:110m countries (figure basemap, public domain)
fetch https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip "$X/naturalearth/ne_110m_admin_0_countries.zip"
echo "external downloads complete"
