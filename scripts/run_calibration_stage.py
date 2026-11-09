"""
CONUS calibration stage runner — chains s03b -> s04 -> s05d in sequence.

Why this exists
---------------
s03b_fix_nonmonotonic_srcs.py, s04_usgs_gage_crosswalk_all_hucs.py, and
s05d_calibrate_B3_wedge_continuity.py (via s05_ncalib_core.run_batch) are each
independently resumable -- every one keeps its own PASS/FAIL status file under
OUT_DIR and skips already-PASSed HUCs on a re-run. That makes it safe to chain
them as three sequential subprocesses under one systemd service: if the
service is killed or crashes partway through (say, mid-s04), a restart re-runs
s03b first, which fast-skips every already-done HUC via its own status file in
seconds, then resumes s04 exactly where it left off.

study_area.xlsx (which all three read via EXCEL_PATH/HUC_CODE_COL) was
regenerated from conus_hucs_status.txt's PASS lines -- the 2,114 HUC8s that
came out of the Stage 1+2 CONUS run -- so this stage runs over that same set,
not the original ~24-HUC test area.

Run:
    .venv/bin/python scripts/run_calibration_stage.py
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
PYTHON = sys.executable

STEPS = [
    "s03b_fix_nonmonotonic_srcs.py",
    "s04_usgs_gage_crosswalk_all_hucs.py",
    "s05d_calibrate_B3_wedge_continuity.py",
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main() -> None:
    for step in STEPS:
        script_path = SCRIPTS_DIR / step
        log.info("=== starting %s ===", step)
        result = subprocess.run([PYTHON, str(script_path)], cwd=REPO_ROOT)
        if result.returncode != 0:
            log.error("=== %s exited with code %d -- stopping chain ===",
                      step, result.returncode)
            sys.exit(result.returncode)
        log.info("=== %s complete ===", step)
    log.info("Calibration stage (s03b -> s04 -> s05d) complete.")


if __name__ == "__main__":
    main()
