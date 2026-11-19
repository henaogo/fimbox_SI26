# FIMbox pipeline scripts

Run everything with `.venv\Scripts\python.exe scripts/<name>.py` **from the
repo root** (several scripts resolve `data/` and other paths relative to the
working directory).

## 1. Pipeline run order

| # | Script | What it does |
|---|---|---|
| 0 | `s00_study_boundary.py` | Builds the study-area boundary from `data/study_area.xlsx` (run once) |
| 1 | `s01_download_dem_fast.py` | Downloads the 10 m 3DEP DEM for the study area |
| 2 | `s02_stage1_all_hucs.py` | Stage 1 — data download, per HUC8 |
| 3 | `s03_stage2_all_hucs.py` | Stage 2 — HAND processing, produces SRCs + hydroTables per branch |
| 3b | `s03b_fix_nonmonotonic_srcs.py` | **Run right after s03.** Forces every rating curve monotone (raster artifacts from HAND geometry can otherwise make discharge dip with rising stage). Fixed ~3,000 curves study-wide in the last full run. |
| 4 | `s04_usgs_gage_crosswalk_all_hucs.py` | Crosswalks USGS gauges onto branches/HydroIDs; writes `usgs_elev_table.csv` per HUC |
| 5 | `s05d_calibrate_B3_wedge_continuity.py` | **Manning's n calibration — the production method.** See §2. |
| 6 | `s06_stage3_all_hucs.py` | Streamflow / feature-ID extraction for a forecast event |
| 7 | `s07_stage4_all_hucs.py` | **Ablation baseline only** — OWP's own channel/overbank calibration. Overwrites the B3 hydroTables. Do not run this in the main sequence; only run it (on a restored baseline) when producing the L0/L1/L2 comparison arms against B3. |
| 8 | `s08_stage5_all_hucs.py` | FIM generation — inundation depth/extent rasters |

## 2. Calibration — how n is calibrated and why B3 was chosen

- **`s05_ncalib_core.py`** — the shared engine. Not run directly; everything
  below imports it. Handles the starting-point match (datum/bathymetry
  correction), conveyance smoothing, and both n-application schemes
  (whole-section and incremental).
- **`s05d_calibrate_B3_wedge_continuity.py`** — **the method actually applied
  in production.** Continuity-wedge starting-point match + incremental
  (slice) n application. Writes both n definitions to every calibrated
  hydroTable: `zonal_n_applied`/`n_{yr}yr` (slice n — drives `discharge_cms`,
  what FIM consumes) and `whole_section_n`/`n_eff_{yr}yr` (an independently
  calibrated whole-section curve, `discharge_cms_wholesection`, kept for
  comparison with OWP's 0.06/0.12 defaults — not used by FIM).
- **`s05a`/`s05b`/`s05c`** (methods A, B1, B2) — the three *rejected*
  starting-point corrections. Kept only so the method comparison is
  reproducible — **do not use for production runs.**
- **`s05e_evaluate_methods.py`** — reruns A/B1/B2/B3 offline (reads baseline
  backups, writes nothing to the pipeline) and produces the evidence for
  choosing B3. Output: `E:/SI/out/calibration_analysis/method_evaluation/`.
- **`s05f_evaluate_schemes.py`** — same idea for whole-section vs incremental
  n application; justifies the incremental choice. Output:
  `E:/SI/out/calibration_analysis/scheme_evaluation/`.
- Full narrative + evidence: `docs/method_decision_deck.pptx`.

### Method HE — stage-banded Horton–Einstein composite n (`s05g_*`)

- **`s05g_calibrate_he_composite.py`** (driver) + **`s05g_he_composite_core.py`**
  (equations). Zones: `n_bathy` (solved from Q0, h0 and `mf_width` of
  `Bankfull_Meanflow_CONUS.txt`, rectangular section replacing the HAND-0
  channel bed) | 0→Tr2 | Tr2→Tr50 | Tr50→Tr100 (or the highest observed
  rating-curve flow — no 100-yr flow ships with the repo). Composite
  n_c(h) = [Σ P_k·n_k^1.5 / P]^(2/3) on the whole section; P_k from the
  cumulative BedArea/L table with exact rows at h1, h2 (rebuilt from the HAND
  and catchment rasters). Each zone n is fitted to 3 stages (1/3, 2/3, top)
  minimising log-Q error, n ∈ [0.007, 0.36]. Default monotone handling: free
  fit + running maximum (`FIMBOX_HE_MONOTONE=constrain` for the alternative).
- Skips a gauge when Tr2 or Tr50 is missing/outside the observed curve, the
  PZF fit fails, `mf_width` is missing, or Q0/(w·h0) > 1 m/s. Q0 < 0 → Q0 = 0,
  and with Q0 = 0 the bathymetry zone is dropped for that gauge.
- **Read-only** on the pipeline outputs; writes to
  `/media/share/fimbox-conus-data/out_calib_hecomposite/` (ali/Jetstream2
  layout; every path has an env-var override — see the script docstring).
  Put `Bankfull_Meanflow_CONUS.txt` in
  `/media/share/fimbox-conus-data/reference/` (or set `FIMBOX_BANKFULL_FILE`).
- Run: `.venv/bin/python scripts/s05g_calibrate_he_composite.py [--huc H | --huc-list F] [--rerun]`.
  Tests: `tests/test_he_composite.py`.

## 3. Analysis & export

| Script | Produces | Output folder |
|---|---|---|
| `plot_v2_diagnostics.py` | Datum-gap, clip-rate, n-profile, anchor-residual, hold-out validation, and starting-point figures + one RC panel per calibrated gauge | `E:/SI/out/calibration_analysis/v2_diagnostics/` |
| `plot_calibration_analysis.py` | Study-area aggregate plots (n distributions, Q scatter, coverage, metrics) + per-branch RC plots | `E:/SI/out/calibration_analysis/` and `E:/SI/out/HUC{huc8}/src_plots/` |
| `plot_src_ensemble.py` | SRC ensemble comparisons (side-by-side, overlaid, normalized) across all calibrated gauges | `E:/SI/out/calibration_analysis/` |
| `export_rc_panels_and_videos.py` | Consolidates all per-gauge RC panels into one folder in three annotation variants (slice n / whole-section n / both), plus a slideshow video per variant | `E:/SI/out/calibration_analysis/rc_panels_export/` |

## `legacy/`

Retired scripts from earlier phases of the project (single-HUC prototypes,
the old monolithic calibration script, earlier plotting tools) — kept for
history, not maintained. See `legacy/README.md` for what superseded each one.
