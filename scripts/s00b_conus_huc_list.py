"""
Source the full CONUS-wide HUC8 list -- the successor to s00_study_boundary.py's
small Excel-driven study area, for the CONUS-scale run.

Queries the HUC8_Boundaries ArcGIS FeatureServer directly for attributes only
(no geometry, no spatial filter) and paginates through resultOffset until all
rows are collected, then filters to true CONUS HUC2 regions (01-18) via the
HUC8 code prefix.

Why not fimbox.HUC8Finder.from_boundary()?
Tried it first with a CONUS-wide bounding box: the server-side spatial
intersects computation against ~2,400 large polygons times out (504) for a
query this broad, and each polygon's full-resolution geometry is enormous
(~560KB/feature -- 500 features = 281MB), so returning geometry for the whole
country is both slow and unnecessary. The actual pipeline already fetches each
HUC's own boundary individually when it processes that HUC
(HUC8Finder.from_huc8() inside getAllInputData), so this list only needs the
flat HUC8 codes -- an attributes-only, no-spatial-filter query is dramatically
lighter and avoids the timeout entirely. (The pagination fix added to
HUC8Finder.from_boundary in preprocessing/download_data/utils.py is still
useful for its normal regional-AOI use case; it's just the wrong tool for a
whole-country attribute pull.)

Output:
    /media/share/fimbox-conus-data/conus_hucs.txt  (flat list, one HUC8 per
    line, sorted -- feed directly to scripts/run_shard.py --shard)

Run:
    .venv/bin/python scripts/s00b_conus_huc_list.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

# ── CONFIG ────────────────────────────────────────────────────────
OUT_DIR = Path("/media/share/fimbox-conus-data")
LIST_OUT = OUT_DIR / "conus_hucs.txt"

_URL = (
    "https://services.arcgis.com/ts4gk3YgS68yLGFl/arcgis/rest/services"
    "/HUC8_Boundaries/FeatureServer/0/query"
)
PAGE_SIZE = 2000
CONUS_HUC2_PREFIXES = {f"{i:02d}" for i in range(1, 19)}  # '01'..'18'
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def fetch_all_huc8_attributes() -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "f": "json",
            "where": "1=1",
            "outFields": "HUC8,states,name,areasqkm",
            "returnGeometry": "false",
            "orderByFields": "FID",
            "resultRecordCount": PAGE_SIZE,
            "resultOffset": offset,
        }
        resp = requests.post(_URL, data=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"FeatureServer error: {data['error']}")
        feats = [f["attributes"] for f in data.get("features", [])]
        rows.extend(feats)
        log.info("  page at offset %d: %d rows", offset, len(feats))
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Fetching all HUC8 attributes from FeatureServer ...")
    rows = fetch_all_huc8_attributes()
    log.info("Total features service-wide: %d", len(rows))

    seen = set()
    conus_hucs = []
    excluded_prefixes = set()
    for row in rows:
        huc8 = str(row["HUC8"]).zfill(8)
        prefix = huc8[:2]
        if prefix in CONUS_HUC2_PREFIXES:
            if huc8 not in seen:
                seen.add(huc8)
                conus_hucs.append(huc8)
        else:
            excluded_prefixes.add(prefix)

    conus_hucs.sort()
    log.info("CONUS HUC8s after HUC2 01-18 filter: %d", len(conus_hucs))
    if excluded_prefixes:
        log.info("Excluded HUC2 prefixes (non-CONUS, e.g. AK/HI/PR/territories): %s",
                 sorted(excluded_prefixes))

    LIST_OUT.write_text("\n".join(conus_hucs) + "\n")
    log.info("Flat HUC8 list (%d codes) --> %s", len(conus_hucs), LIST_OUT)


if __name__ == "__main__":
    main()
