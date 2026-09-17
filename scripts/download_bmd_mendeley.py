"""Download the BMD 35-station daily dataset (Zubair et al. 2024, Mendeley DOI 10.17632/tbrhznpwg9.1, CC-BY-4.0)
and verify every file against the SHA-256 published in the Mendeley metadata."""
import hashlib
import json
import pathlib
import sys

import requests

DATASET = "tbrhznpwg9"
OUT = pathlib.Path("data/external/bmd_mendeley")


def main() -> int:
    meta = requests.get(f"https://data.mendeley.com/public-api/datasets/{DATASET}", timeout=60).json()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mendeley_metadata.json").write_text(json.dumps(meta, indent=2))
    bad = []
    for f in meta["files"]:
        name, cd = f["filename"], f["content_details"]
        sub = "combined" if name == "BD_weather.csv" else "stations"
        path = OUT / sub / name
        path.parent.mkdir(exist_ok=True)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != cd["sha256_hash"]:
            r = requests.get(cd["download_url"], timeout=300)
            r.raise_for_status()
            path.write_bytes(r.content)
        ok = hashlib.sha256(path.read_bytes()).hexdigest() == cd["sha256_hash"]
        if not ok:
            bad.append(name)
        print(f"{'OK ' if ok else 'BAD'} {sub}/{name} {path.stat().st_size}")
    print(f"{len(meta['files'])} files, {len(bad)} failed checksum: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
