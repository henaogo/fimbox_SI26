"""
Run Stage 1 (download) + Stage 2 (HAND) for one shard (a static subset) of
HUC8s -- the concurrency-safe unit for scaling this pipeline across multiple
processes on one VM, and eventually across multiple VMs.

Why this exists (and why it's not just s02_stage1_all_hucs.py + s03_stage2_all_hucs.py)
-----------------------------------------------------------------------------
Those scripts read the FULL HUC list from study_area.xlsx and keep one shared
TASK_LOG. Two copies started at once would grab the same next HUC and race
writing to the same output directory -- not safe to run concurrently.

This script instead:
  - reads its HUC list from a plain-text shard file (one HUC8 per line),
    partitioned ahead of time by --split -- no runtime coordination between
    shards is needed, so any number can run at once (same VM or different
    VMs) without ever touching each other's HUCs.
  - writes to a status log named after the shard file, so concurrent shards
    never share a log (and thus never race on the same file).
  - Dask worker count per HUC is controlled by FIMBOX_DASK_WORKERS. When
    running N shards concurrently on one VM, set this LOWER than the
    single-shard auto-sized default (see _resolve_n_workers() in
    fimbox._dask) so the shards don't oversubscribe the machine's RAM/CPU
    together.

Usage
-----
Split a flat HUC list into N shards (round-robin, so each shard gets a mix
rather than a contiguous geographic block):
    python scripts/run_shard.py --split all_hucs.txt --n-shards 4 \\
        --shard-dir /media/share/fimbox-conus-data/shards

Run one shard (reduce FIMBOX_DASK_WORKERS when running multiple concurrently):
    FIMBOX_DASK_WORKERS=3 python scripts/run_shard.py \\
        --shard /media/share/fimbox-conus-data/shards/shard_00.txt
"""
from __future__ import annotations

import argparse
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────
OUT_DIR    = Path("/media/share/fimbox-conus-data/out")
IDENTIFIER = "nwmmr"
BUFFER_M   = 5000
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ── Sharding ────────────────────────────────────────────────────────

def split_into_shards(all_hucs_path: Path, n_shards: int, shard_dir: Path) -> None:
    """Round-robin partition a flat HUC8 list (one per line) into n_shards
    files. Round-robin (not contiguous blocks) so each shard gets a mix of
    HUC sizes/regions rather than risking one shard landing all-large or
    all-small HUCs from whatever order the source list happens to be in."""
    hucs = [line.strip().zfill(8) for line in all_hucs_path.read_text().splitlines()
            if line.strip()]
    if not hucs:
        raise ValueError(f"No HUCs found in {all_hucs_path}")

    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    for i, huc8 in enumerate(hucs):
        shards[i % n_shards].append(huc8)

    for i, shard_hucs in enumerate(shards):
        out = shard_dir / f"shard_{i:02d}.txt"
        out.write_text("\n".join(shard_hucs) + "\n")
        log.info("  shard_%02d: %d HUCs -> %s", i, len(shard_hucs), out)

    log.info("Split %d HUCs into %d shards in %s", len(hucs), n_shards, shard_dir)


# ── Per-shard status log (one file per shard -- never shared) ───────

def _log_path_for(shard_path: Path) -> Path:
    return shard_path.with_name(f"{shard_path.stem}_status.txt")


def _skiplist_path_for(shard_path: Path) -> Path:
    return shard_path.with_name(f"{shard_path.stem}_skiplist.txt")


def _inprogress_path_for(shard_path: Path) -> Path:
    return shard_path.with_name(f"{shard_path.stem}_inprogress.txt")


def _crashcount_path_for(shard_path: Path) -> Path:
    return shard_path.with_name(f"{shard_path.stem}_crashcount.txt")


_CRASH_SKIPLIST_THRESHOLD = 2


def _reconcile_crash(shard_path: Path, log_path: Path, skiplist_path: Path) -> None:
    """A HUC that gets kernel-OOM-killed (SIGKILL) dies mid-run with no chance
    to log FAIL or reach the graceful except-block that adds it to the
    skip-list -- so on its own it would crash-loop forever, re-attempted on
    every restart and blocking every HUC behind it in the list (this is
    exactly what happened to 04190000 on 2026-08-06: ~19 back-to-back OOM
    kills, ~124GB RSS each time, before this function existed).

    Detects that case from the *outside*: `_inprogress_path_for` holds the
    one HUC currently being attempted, written before the attempt and cleared
    on any clean exit (PASS or a caught exception). If it's still present at
    the next startup, the last attempt never reached a clean exit -- almost
    certainly a SIGKILL. After `_CRASH_SKIPLIST_THRESHOLD` such crashes for
    the same HUC, auto-skiplist it so the queue keeps moving.
    """
    inprogress_path = _inprogress_path_for(shard_path)
    if not inprogress_path.exists():
        return
    huc8 = inprogress_path.read_text().strip()
    inprogress_path.unlink(missing_ok=True)
    if not huc8:
        return

    crashcount_path = _crashcount_path_for(shard_path)
    counts: dict[str, int] = {}
    if crashcount_path.exists():
        for line in crashcount_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                counts[parts[0]] = int(parts[1])
    counts[huc8] = counts.get(huc8, 0) + 1

    if counts[huc8] >= _CRASH_SKIPLIST_THRESHOLD:
        log.warning("  [%s] crashed without a clean exit %d times (likely OOM/SIGKILL) "
                    "-- auto-skiplisting", huc8, counts[huc8])
        _log_result(log_path, huc8, "FAIL",
                    f"auto-skiplisted after {counts[huc8]} crashes with no clean exit "
                    "(likely repeated OOM/SIGKILL, never raised a catchable exception)")
        _add_to_skiplist(skiplist_path, huc8)
        del counts[huc8]
    else:
        log.warning("  [%s] crashed without a clean exit (likely OOM/SIGKILL) -- "
                    "%d/%d, will retry once more before skiplisting",
                    huc8, counts[huc8], _CRASH_SKIPLIST_THRESHOLD)

    crashcount_path.write_text("".join(f"{h} {c}\n" for h, c in counts.items()))


def _load_done(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    done = set()
    for line in log_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] == "PASS":
            done.add(parts[1])
    return done


def _load_skiplist(skiplist_path: Path) -> set[str]:
    if not skiplist_path.exists():
        return set()
    return {line.strip() for line in skiplist_path.read_text().splitlines() if line.strip()}


def _add_to_skiplist(skiplist_path: Path, huc8: str) -> None:
    """Set a HUC aside after its first failure so restarts stop re-attempting
    it (some failures take up to 30 min to time out). This is a parking lot,
    not a verdict -- failed HUCs stay logged in the status file with their
    error, and can be retried later (dedicated cleanup pass) by removing
    their line from this file."""
    skiplist_path.parent.mkdir(parents=True, exist_ok=True)
    with skiplist_path.open("a") as f:
        f.write(f"{huc8}\n")


def _log_result(log_path: Path, huc8: str, status: str, note: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(f"shard {huc8} {status}{(' ' + note) if note else ''}\n")


# ── Per-HUC Stage 1 + Stage 2 (same logic as s02/s03, inlined so a shard
# runs both stages back-to-back per HUC without depending on those scripts'
# Excel-driven, single-shared-log main loops) ────────────────────────────

def run_stage1(huc8: str) -> None:
    import fimbox

    pp = fimbox.getAllInputData(
        huc8=huc8,
        out_dir=str(OUT_DIR),
        buffer_m=BUFFER_M,
        headwater_buffer_cells=8,
        get_flowlines=True,
        get_catchments=True,
        resolution="medium",
        identifier=IDENTIFIER,
    )
    pp.run()


def run_stage2(huc8: str) -> None:
    from fimbox import AOIProcessingConfig, BranchDerivation, calculate_allbranches
    from fimbox._dask import _resolve_n_workers

    aoi_dir = OUT_DIR / f"HUC{huc8}" / "watershed-data"
    if not aoi_dir.exists():
        raise FileNotFoundError(f"watershed-data not found — Stage 1 must run first: {aoi_dir}")

    def _opt(p: Path):
        return p if p.exists() else None

    BranchDerivation(
        out_dir=aoi_dir,
        branch_id_attribute="levpa_id",
        reach_id_attribute="ID",
        branch_buffer_distance_meters=7000.0,
    ).run()

    cfg = AOIProcessingConfig(
        aoi_dir=aoi_dir,
        branch_list_path=aoi_dir / "branch_ids.lst",
        dem_path=aoi_dir / "dem.tif",
        streams_gpkg=aoi_dir / f"{IDENTIFIER}_subset_streams.gpkg",
        boundary_gpkg=aoi_dir / "wbd_buffered.gpkg",
        bridge_elev_diff_path=_opt(aoi_dir / "bridge_elev_diff.tif"),
        levee_gpkg_path=_opt(aoi_dir / "3d_nld_subset_levees_burned.gpkg"),
        headwaters_gpkg=_opt(aoi_dir / f"{IDENTIFIER}_headwater_points_subset.gpkg"),
        levelpaths_extended_gpkg=_opt(
            aoi_dir / f"{IDENTIFIER}_subset_streams_levelPaths_extended.gpkg"
        ),
        agree_buffer_m=15.0,
        agree_smooth_drop=10.0,
        agree_sharp_drop=1000.0,
        cost_distance_tolerance=50.0,
        lateral_elevation_threshold=10,
        max_split_distance_m=1500.0,
        slope_min=0.0001,
        lakes_buffer_dist_m=100.0,
        mannings_n=0.06,
        stage_min_m=0.0,
        stage_interval_m=0.3048,
        stage_max_m=25.2984,
        min_catchment_area=0.25,
        min_stream_length=0.5,
        crosswalk_max_distance_m=100.0,
        src_slope_source="iris_sword",
        iris_slope_csv=None,
        hfab_slope_column=None,
        n_workers=_resolve_n_workers(),
        keep_failed_branches=True,
        delete_deny_list=True,
    )

    calculate_allbranches(
        cfg,
        run_branch_zero=True,
        delete_deny_list=True,
        deny_unit_list=CONFIG_DIR / "deny_unit.lst",
        branch_ids_csv=aoi_dir / "branch_ids.csv",
    )


def run_shard(shard_path: Path) -> None:
    log_path = _log_path_for(shard_path)
    skiplist_path = _skiplist_path_for(shard_path)
    inprogress_path = _inprogress_path_for(shard_path)

    _reconcile_crash(shard_path, log_path, skiplist_path)

    hucs = [line.strip().zfill(8) for line in shard_path.read_text().splitlines()
            if line.strip()]
    done = _load_done(log_path)
    skipped = _load_skiplist(skiplist_path)
    remaining = [h for h in hucs if h not in done and h not in skipped]

    log.info("Shard %s: %d total | %d already done | %d skipped (known-bad) | %d to run",
             shard_path.name, len(hucs), len(done), len(skipped), len(remaining))
    log.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    batch_start = time.time()
    passed, failed = list(done), []

    for i, huc8 in enumerate(remaining, 1):
        huc_start = time.time()
        log.info("  [%d/%d] HUC8 = %s", i, len(remaining), huc8)
        # Written before the attempt, cleared on any clean exit (PASS or a
        # caught exception below). If the process dies mid-attempt (e.g.
        # kernel OOM SIGKILL) this is left behind for _reconcile_crash() to
        # find on the next startup -- see its docstring.
        inprogress_path.write_text(huc8)
        try:
            run_stage1(huc8)
            run_stage2(huc8)
            elapsed = time.time() - huc_start
            _log_result(log_path, huc8, "PASS")
            passed.append(huc8)
            log.info("  [%s] PASS  |  %s", huc8, _fmt(elapsed))
        except Exception:
            elapsed = time.time() - huc_start
            err = traceback.format_exc().splitlines()[-1]
            _log_result(log_path, huc8, "FAIL", err)
            _add_to_skiplist(skiplist_path, huc8)
            failed.append(huc8)
            log.error("  [%s] FAIL after %s  |  %s  (skipped on future restarts)",
                       huc8, _fmt(elapsed), err)
        inprogress_path.unlink(missing_ok=True)

    total = time.time() - batch_start
    log.info("Shard %s complete  |  total: %s  |  passed: %d  |  failed: %d",
             shard_path.name, _fmt(total), len(passed), len(failed))
    if failed:
        log.warning("Failed HUCs: %s", failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path,
                        help="Path to a flat HUC8 list (one per line) to partition into shards")
    parser.add_argument("--n-shards", type=int, default=4,
                        help="Number of shards to split into (with --split)")
    parser.add_argument("--shard-dir", type=Path,
                        default=Path("/media/share/fimbox-conus-data/shards"),
                        help="Directory to write/read shard files")
    parser.add_argument("--shard", type=Path,
                        help="Path to a shard file to run (one HUC8 per line)")
    args = parser.parse_args()

    if args.split:
        split_into_shards(args.split, args.n_shards, args.shard_dir)
    elif args.shard:
        run_shard(args.shard)
    else:
        parser.error("pass either --split <hucs.txt> or --shard <shard_file>")


if __name__ == "__main__":
    main()
