"""
Method HE — stage-banded Horton–Einstein composite-n calibration.

Zones: n_bathy (solved from Q0, h0 and the CONUS mean-flow width) | 0→Tr2 |
Tr2→Tr50 | Tr50→Tr100 (or the highest observed flow when no Tr100 is
available).  See s05g_he_composite_core.py for the equations.

READ-ONLY on the pipeline outputs: hydroTables, rasters and usgs_elev_table
are only read.  Everything is written to a NEW folder (HE_OUT_DIR):

    <HE_OUT_DIR>/
        calibration_results_all.csv            every gauge of every finished HUC
        he_status_<list>.txt                   PASS/FAIL per HUC (resumable)
        HUC<huc8>/calibration_results.csv      one row per gauge (calibrated or skip reason)
        HUC<huc8>/curves/curve_<location_id>_<HydroID>.csv
        HUC<huc8>/branches/<bid>/hydroTable_<bid>.csv
              copy of the branch hydroTable; calibrated HydroIDs get the HE
              discharge_cms (+ exact rows at h1, h2) and he_* columns

Paths follow the "ali" branch CONUS layout (Jetstream2); override with env vars:
    FIMBOX_CONUS_ROOT     default /media/share/fimbox-conus-data
    FIMBOX_OUT_DIR        default $FIMBOX_CONUS_ROOT/out           (pipeline outputs, HUC<huc8>/...)
    FIMBOX_HE_OUT_DIR     default $FIMBOX_CONUS_ROOT/out_calib_hecomposite
    FIMBOX_BANKFULL_FILE  default $FIMBOX_CONUS_ROOT/reference/Bankfull_Meanflow_CONUS.txt
    FIMBOX_TR100_FILE     optional csv/parquet with feature_id + a 100-yr flow column
    FIMBOX_TR100_COL      default 100_0_year_recurrence_flow_17C
    FIMBOX_TR100_UNITS    cfs (default) or cms
    FIMBOX_INCLUDE_BRANCH0  0 (default, same as s05_ncalib_core) or 1
    FIMBOX_HE_MONOTONE    cummax (default: free fit + running max) or constrain
USGS rating curves and NWM recurrence flows come from the repo data/ folder,
exactly as s05_ncalib_core.load_shared reads them.

Run (from the repo root):
    .venv/bin/python scripts/s05g_calibrate_he_composite.py                    # all HUCs found in OUT_DIR
    .venv/bin/python scripts/s05g_calibrate_he_composite.py --huc 12090301     # one HUC
    .venv/bin/python scripts/s05g_calibrate_he_composite.py --huc-list shards/shard_00.txt
    add --rerun to ignore the status file
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from s05_ncalib_core import load_shared
from s05g_he_composite_core import (
    MONOTONE_MODE, N_MAX, N_MIN, ReachGeometry, calibrate_gauge, geometry_from_hydrotable,
    read_branch_cells,
)

# ── CONFIG (ali / Jetstream2 layout) ──────────────────────────────
CONUS_ROOT    = Path(os.environ.get("FIMBOX_CONUS_ROOT", "/media/share/fimbox-conus-data"))
OUT_DIR       = Path(os.environ.get("FIMBOX_OUT_DIR", CONUS_ROOT / "out"))
HE_OUT_DIR    = Path(os.environ.get("FIMBOX_HE_OUT_DIR", CONUS_ROOT / "out_calib_hecomposite"))
BANKFULL_FILE = Path(os.environ.get("FIMBOX_BANKFULL_FILE",
                                    CONUS_ROOT / "reference" / "Bankfull_Meanflow_CONUS.txt"))
TR100_FILE    = os.environ.get("FIMBOX_TR100_FILE") or None
TR100_COL     = os.environ.get("FIMBOX_TR100_COL", "100_0_year_recurrence_flow_17C")
TR100_UNITS   = os.environ.get("FIMBOX_TR100_UNITS", "cfs").lower()
INCLUDE_BRANCH0 = os.environ.get("FIMBOX_INCLUDE_BRANCH0", "0") == "1"
METHOD = "HE_composite"
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("he_calib")

# B3 / s05 columns that may sit on a calibrated hydroTable — dropped from the copy
_S05_COLS = ["zonal_n_applied", "discharge_cms_wholesection", "whole_section_n"]


# ══════════════════════════════════════════════════════════════════
# Shared inputs
# ══════════════════════════════════════════════════════════════════

def _sniff_sep(path: Path) -> str | None:
    with path.open("r", errors="replace") as f:
        head = f.readline()
    if "\t" in head:
        return "\t"
    if "," in head:
        return ","
    if ";" in head:
        return ";"
    if "|" in head:
        return "|"
    return None  # whitespace


def load_mf_width(path: Path, feature_ids: set[int]) -> dict[int, float]:
    """COMID → mf_width (m) for the gauged feature_ids only."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Bankfull_Meanflow_CONUS.txt not found: {path} "
            f"(set FIMBOX_BANKFULL_FILE to its location)")
    sep = _sniff_sep(path)
    sep = sep or r"\s+"
    header = pd.read_csv(path, sep=sep, nrows=0).columns
    lower = {c.strip().lower(): c for c in header}
    if "comid" not in lower or "mf_width" not in lower:
        raise ValueError(f"{path.name} needs COMID and mf_width columns; found {list(header)}")
    cols = [lower["comid"], lower["mf_width"]]
    df = pd.read_csv(path, sep=sep, usecols=cols)
    df.columns = ["COMID", "mf_width"]
    df["COMID"] = pd.to_numeric(df["COMID"], errors="coerce")
    df["mf_width"] = pd.to_numeric(df["mf_width"], errors="coerce")
    df = df.dropna(subset=["COMID"])
    df["COMID"] = df["COMID"].astype(np.int64)
    df = df[df["COMID"].isin(feature_ids)].drop_duplicates("COMID")
    log.info("mf_width: %d of %d gauged feature_ids matched in %s",
             len(df), len(feature_ids), path.name)
    return dict(zip(df["COMID"].tolist(), df["mf_width"].tolist()))


def load_tr100() -> dict[int, float]:
    if not TR100_FILE:
        return {}
    p = Path(TR100_FILE)
    df = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p)
    df = df[["feature_id", TR100_COL]].dropna()
    f = 0.028317 if TR100_UNITS == "cfs" else 1.0
    log.info("Tr100: %d flows from %s (%s)", len(df), p.name, TR100_UNITS)
    return dict(zip(df["feature_id"].astype(np.int64), df[TR100_COL].astype(float) * f))


def discover_hucs() -> list[str]:
    return sorted(p.parent.name[3:] for p in OUT_DIR.glob("HUC*/usgs_elev_table.csv"))


def read_elev(huc8: str) -> pd.DataFrame | None:
    p = OUT_DIR / f"HUC{huc8}" / "usgs_elev_table.csv"
    if not p.is_file():
        return None
    e = pd.read_csv(p, dtype={"location_id": str, "feature_id": "Int64"})
    if not INCLUDE_BRANCH0:
        e = e[e["levpa_id"].astype(str) != "0"]
    return e


# ══════════════════════════════════════════════════════════════════
# Per branch / per HUC
# ══════════════════════════════════════════════════════════════════

def _baseline_hydrotable(branch_dir: Path, bid: str) -> tuple[pd.DataFrame, str] | None:
    pre = branch_dir / f"hydroTable_{bid}.pre_n_calib.csv"     # s05 baseline backup
    cur = branch_dir / f"hydroTable_{bid}.csv"
    path = pre if pre.is_file() else cur
    if not path.is_file():
        return None
    ht = pd.read_csv(path, low_memory=False)
    ht["HydroID"] = ht["HydroID"].astype(int)
    return ht.drop(columns=[c for c in _S05_COLS if c in ht.columns]), path.name


def process_branch(huc8: str, bid: str, gauges: pd.DataFrame, rc: pd.DataFrame,
                   recur: pd.DataFrame, mfw: dict, tr100: dict,
                   out_huc: Path) -> list[dict]:
    branch_dir = OUT_DIR / f"HUC{huc8}" / "watershed-data" / "branches" / bid
    rows: list[dict] = []
    base = _baseline_hydrotable(branch_dir, bid)

    def skip(g, reason, extra=None):
        r = _id_cols(huc8, bid, g)
        r.update(extra or {})
        r["status"] = reason
        rows.append(r)

    if base is None:
        for _, g in gauges.iterrows():
            skip(g, "hydrotable_not_found")
        return rows
    ht, ht_name = base

    # pre-checks without geometry → which HydroIDs need rasters
    todo: list[tuple] = []
    seen: set[int] = set()
    for _, g in gauges.iterrows():
        hid = int(g["HydroID"])
        if pd.isna(g["feature_id"]):
            skip(g, "no_feature_id"); continue
        if hid in seen:
            skip(g, "hydroid_already_calibrated"); continue
        hdf = ht[ht["HydroID"] == hid]
        if hdf.empty:
            skip(g, "hydroid_not_in_hydrotable"); continue
        gage_rc = rc[rc["location_id"] == str(g["location_id"]).zfill(8)]
        if gage_rc.empty:
            skip(g, "no_rating_curve"); continue
        fid = int(g["feature_id"])
        fr = recur[recur["feature_id"] == fid]
        q_rec = {2: float(fr["Q_2yr_cms"].iloc[0]) if not fr.empty else None,
                 50: float(fr["Q_50yr_cms"].iloc[0]) if not fr.empty else None,
                 100: tr100.get(fid)}
        slope = float(hdf["SLOPE"].iloc[0])
        d, _ = calibrate_gauge(None, slope, gage_rc, float(g["dem_adj_elevation"]),
                               q_rec, mfw.get(fid), check_only=True)
        if d["status"] != "ok":
            skip(g, d["status"], d); continue
        seen.add(hid)
        todo.append((g, hid, gage_rc, q_rec, slope, hdf))

    if not todo:
        return rows

    cells = read_branch_cells(branch_dir, bid, [t[1] for t in todo])
    if cells is None:
        log.warning("  Branch %s: HAND/catchment rasters not found — table interpolation", bid)

    calibrated: dict[int, pd.DataFrame] = {}
    for g, hid, gage_rc, q_rec, slope, hdf in todo:
        geom: ReachGeometry = geometry_from_hydrotable(hdf)
        if cells is not None:
            geom.attach_raster(cells[0].get(hid, np.array([])), cells[1])
        hs = hdf.sort_values("stage")
        d, curve = calibrate_gauge(geom, slope, gage_rc, float(g["dem_adj_elevation"]),
                                   q_rec, mfw.get(int(g["feature_id"])),
                                   base_stage=hs["stage"].to_numpy(float),
                                   base_q=hs["discharge_cms"].to_numpy(float))
        r = _id_cols(huc8, bid, g)
        r.update(d)
        r["baseline_hydrotable"] = ht_name
        rows.append(r)
        if curve is not None:
            cdir = out_huc / "curves"
            cdir.mkdir(parents=True, exist_ok=True)
            curve.insert(0, "HydroID", hid)
            curve.insert(1, "location_id", r["location_id"])
            curve.to_csv(cdir / f"curve_{r['location_id']}_{hid}.csv", index=False)
            calibrated[hid] = curve
            log.info("  Gauge %s HydroID %d: n_b=%s n1=%.4f n2=%.4f n3=%.4f  rmse_logQ=%.3f",
                     r["location_id"], hid,
                     f"{d['n_bathy']:.4f}" if d["bathy_band"] else "—",
                     d["n1"], d["n2"], d["n3"], d.get("err_rmse_logQ", np.nan))
        else:
            log.info("  Gauge %s HydroID %d: skipped (%s)", r["location_id"], hid, d["status"])

    if calibrated:
        _write_hydrotable_copy(ht, calibrated, out_huc / "branches" / bid, bid)
    return rows


def _id_cols(huc8, bid, g) -> dict:
    return {"method": METHOD, "huc8": huc8, "branch": bid,
            "location_id": str(g["location_id"]).zfill(8),
            "HydroID": int(g["HydroID"]),
            "feature_id": int(g["feature_id"]) if pd.notna(g["feature_id"]) else -1,
            "dem_adj_elevation_m": float(g["dem_adj_elevation"]),
            "n_min": N_MIN, "n_max": N_MAX}


def _write_hydrotable_copy(ht: pd.DataFrame, calibrated: dict[int, pd.DataFrame],
                           out_dir: Path, bid: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ht[~ht["HydroID"].isin(list(calibrated))].copy()
    keep["he_calibrated"] = False
    parts = [keep]
    for hid, curve in calibrated.items():
        tmpl = ht[ht["HydroID"] == hid].sort_values("stage")
        st = curve["stage"].to_numpy(float)
        # constant per-HydroID columns from the nearest original row
        idx = np.clip(np.searchsorted(tmpl["stage"].to_numpy(float), st, side="right") - 1,
                      0, len(tmpl) - 1)
        new = tmpl.iloc[idx].reset_index(drop=True).copy()
        L = float(tmpl["LENGTHKM"].iloc[0]) * 1000.0
        new["stage"] = st
        new["SurfaceArea (m2)"] = curve["SurfaceArea (m2)"].to_numpy()
        new["BedArea (m2)"] = curve["BedArea (m2)"].to_numpy()
        new["Volume (m3)"] = curve["Volume (m3)"].to_numpy()
        if "Number of Cells" in new.columns:
            new["Number of Cells"] = np.nan        # not reconstructed for inserted rows
            orig = curve["inserted_row"].to_numpy() == False  # noqa: E712
            new.loc[orig, "Number of Cells"] = tmpl.set_index("stage").loc[
                st[orig], "Number of Cells"].to_numpy()
        new["TopWidth (m)"] = new["SurfaceArea (m2)"] / L
        new["WettedPerimeter (m)"] = new["BedArea (m2)"] / L
        new["WetArea (m2)"] = new["Volume (m3)"] / L
        new["HydraulicRadius (m)"] = (new["WetArea (m2)"] / new["WettedPerimeter (m)"]).fillna(0.0)
        new["discharge_cms_baseline"] = curve.get("discharge_cms_baseline")
        new["discharge_cms"] = curve["discharge_cms"].to_numpy()
        for c in ("A_total_m2", "P_total_m", "P_bathy_m", "P1_m", "P2_m", "P3_m", "n_c",
                  "inserted_row"):
            new[f"he_{c}"] = curve[c].to_numpy()
        new["he_calibrated"] = True
        parts.append(new)
    out = pd.concat(parts, ignore_index=True).sort_values(["HydroID", "stage"])
    out.to_csv(out_dir / f"hydroTable_{bid}.csv", index=False)


def run_huc(huc8: str, rc, recur, mfw, tr100) -> int:
    elev = read_elev(huc8)
    if elev is None:
        raise FileNotFoundError(f"usgs_elev_table.csv not found for HUC {huc8}")
    out_huc = HE_OUT_DIR / f"HUC{huc8}"
    out_huc.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for bid, gauges in elev.groupby(elev["levpa_id"].astype(str)):
        rows += process_branch(huc8, bid, gauges, rc, recur, mfw, tr100, out_huc)
    res = pd.DataFrame(rows)
    res.to_csv(out_huc / "calibration_results.csv", index=False)
    n_ok = int((res.get("status") == "calibrated").sum()) if not res.empty else 0
    log.info("  HUC %s: %d/%d gauges calibrated", huc8, n_ok, len(res))
    return n_ok


# ══════════════════════════════════════════════════════════════════
# Batch driver (resumable, shard-friendly like ali's run_shard.py)
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--huc", action="append", help="HUC8 to run (repeatable)")
    ap.add_argument("--huc-list", type=Path, help="text file, one HUC8 per line")
    ap.add_argument("--rerun", action="store_true", help="ignore PASS lines in the status file")
    a = ap.parse_args()

    if a.huc:
        hucs, tag = [h.zfill(8) for h in a.huc], "manual"
    elif a.huc_list:
        hucs = [l.strip().zfill(8) for l in a.huc_list.read_text().splitlines() if l.strip()]
        tag = a.huc_list.stem
    else:
        hucs, tag = discover_hucs(), "all"
    HE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = HE_OUT_DIR / f"he_status_{tag}.txt"
    done = set()
    if status.exists() and not a.rerun:
        done = {p[1] for l in status.read_text().splitlines()
                if len(p := l.split()) >= 3 and p[2] == "PASS"}
    todo = [h for h in hucs if h not in done]
    log.info("HE composite calibration: %d HUCs | %d done | %d to run | out → %s | "
             "n ∈ [%.3f, %.2f] | monotone=%s", len(hucs), len(hucs) - len(todo), len(todo),
             HE_OUT_DIR, N_MIN, N_MAX, MONOTONE_MODE)
    if not todo:
        return

    fids: set[int] = set()
    for h in todo:
        e = read_elev(h)
        if e is not None:
            fids |= set(e["feature_id"].dropna().astype(np.int64).tolist())
    mfw = load_mf_width(BANKFULL_FILE, fids)
    tr100 = load_tr100()
    rc, recur = load_shared()

    t0, failed = time.time(), []
    for i, h in enumerate(todo, 1):
        t = time.time()
        log.info("[%d/%d] HUC8 %s", i, len(todo), h)
        try:
            run_huc(h, rc, recur, mfw, tr100)
            with status.open("a") as f:
                f.write(f"he {h} PASS\n")
            log.info("  [%s] PASS | %.0fs", h, time.time() - t)
        except Exception:
            err = traceback.format_exc().splitlines()[-1]
            with status.open("a") as f:
                f.write(f"he {h} FAIL {err}\n")
            failed.append(h)
            log.error("  [%s] FAIL | %s", h, err)

    parts = [pd.read_csv(p, dtype={"location_id": str, "huc8": str, "branch": str})
             for p in sorted(HE_OUT_DIR.glob("HUC*/calibration_results.csv"))]
    if parts:
        allr = pd.concat(parts, ignore_index=True)
        allr.to_csv(HE_OUT_DIR / "calibration_results_all.csv", index=False)
        log.info("calibration_results_all.csv: %d gauges, %d calibrated",
                 len(allr), int((allr["status"] == "calibrated").sum()))
    log.info("Done in %.0fs | failed: %s", time.time() - t0, failed or "none")


if __name__ == "__main__":
    main()
