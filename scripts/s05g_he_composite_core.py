"""
Stage-banded Horton–Einstein composite-n calibration (method "HE").

Calibration zones (by HAND stage h above the DEM water surface)
---------------------------------------------------------------
    bathy   below HAND 0: rectangular channel, width w = mf_width (CONUS
            Bankfull_Meanflow table, matched COMID = feature_id), depth
            h0 = δ (PZF datum gap, same fit as s05_ncalib_core.fit_pzf).
            n_bathy is SOLVED (not fitted) so the rectangle carries Q0:
                Q0 = (1/n_b)·(w·h0)·R_b^(2/3)·√S,  R_b = w·h0 / (w + 2·h0)
    zone 1  0  → h1 = HAND stage of Tr2
    zone 2  h1 → h2 = HAND stage of Tr50
    zone 3  h2 → h_top = Tr100 if supplied and observed, else the highest
            observed rating-curve flow above Tr50 (n3 also applies above h_top)

Geometry (from the hydroTable; P = BedArea/L, A = Volume/L)
-----------------------------------------------------------
With the bathymetry rectangle (it REPLACES the HAND-0 channel bed P(0)):
    A*(h) = A(h) + w·h0
    P*(h) = (w + 2·h0) + P(h) − P(0)
    P_b   = w + 2·h0                                   (roughness n_b)
    P_1   = P(min(h, h1)) − P(0)
    P_2   = max(0, P(min(h, h2)) − P(h1))
    P_3   = max(0, P(h) − P(h2))
Without it (Q0 = 0): A* = A, P* = P, P_b = 0, P_1 = P(min(h, h1)).
Σ P_k = P* at every stage.

Composite roughness and discharge (whole section, equal-velocity assumption)
    n_c(h) = [ Σ_k P_k(h)·n_k^1.5 / P*(h) ]^(2/3)
    Q(h)   = (1/n_c)·A*·(A*/P*)^(2/3)·√S

Exact rows at h1 and h2
    P(h1), P(h2), A(h1), A(h2) come from the branch HAND + catchment rasters:
    the wet-cell set (HAND < h, zero-HAND always wet; same rule as
    build_src.build_src_base) is exact.  The slope raster that build_src used
    for the bed factor √(1+s²) is deleted by the pipeline cleanup, so the bed
    area of the cells added between the grid row below h and h uses that
    1-ft interval's mean bed factor (ΔBedArea/ΔSurfaceArea from the table).
    If the rasters are missing or don't reproduce the table's cell counts,
    the geometry falls back to linear interpolation of the table
    (geom_source = "table_interp").

Fitting
    Zones are fitted in order (n1, then n2 with n1 fixed, then n3).  Each n_k
    minimises Σ (ln Q_model − ln Q_obs)² over 3 stages at 1/3, 2/3 and the top
    of its zone (Q_obs interpolated from the USGS rating curve), bounded to
    [N_MIN, N_MAX].  Whole-section H–E can make Q(h) dip where the floodplain
    starts to wet (P jumps, A barely grows) even with the true n, so by
    default (FIMBOX_HE_MONOTONE=cummax) the fit is unconstrained and the
    written discharge_cms is the running maximum of the raw composite curve
    (discharge_cms_raw kept; dips counted in nonmonotone_rows).  With
    FIMBOX_HE_MONOTONE=constrain, n_k is restricted to values that keep Q
    non-decreasing over its zone (h2 → table top for zone 3).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from s05_ncalib_core import baseflow_q0, fit_pzf

log = logging.getLogger(__name__)

# ── Calibration constants ─────────────────────────────────────────
N_MIN = 0.007
N_MAX = 0.36
V_MAX = 1.0                     # m/s — bathymetry velocity Q0/(w·h0) above this → skip gauge
FIT_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0, 1.0)
N_GRID = np.geomspace(N_MIN, N_MAX, 241)
GEOM_MATCH_TOL = 0.005          # max relative cell-count mismatch raster vs table
MIN_ZONE3_THICKNESS = 0.01      # m — thinner top zone is not fitted (n3 = n2)
MONO_TOL = 1e-9
# How a non-decreasing Q(h) is obtained (FIMBOX_HE_MONOTONE):
#   "cummax"     (default) n_k is the unconstrained best fit; discharge_cms is the
#                running maximum of the raw curve (discharge_cms_raw kept), like s03b/OWP
#   "constrain"  each n_k is chosen only among values that keep Q(h) non-decreasing
#                over its zone; if none does, best fit + monotone_ok = False
MONOTONE_MODE = os.environ.get("FIMBOX_HE_MONOTONE", "cummax").lower()


# ══════════════════════════════════════════════════════════════════
# Geometry
# ══════════════════════════════════════════════════════════════════

@dataclass
class ReachGeometry:
    """Cumulative HAND geometry of one HydroID, evaluable at any stage."""
    stages: np.ndarray            # table stage grid, ascending (m)
    surf: np.ndarray              # SurfaceArea (m2) at grid
    bed: np.ndarray               # BedArea (m2) at grid
    vol: np.ndarray               # Volume (m3) at grid
    ncells: np.ndarray            # Number of Cells at grid
    length_m: float
    pixel_area: float | None = None
    hand_pos: np.ndarray | None = None      # sorted positive HAND of the reach's cells
    cum_hand: np.ndarray | None = None      # cumsum of hand_pos
    zero_count: int = 0
    source: str = "table_interp"
    cell_mismatch: float = np.nan

    @property
    def h_max(self) -> float:
        return float(self.stages[-1])

    # raster helpers --------------------------------------------------
    def _raster_counts(self, h: np.ndarray):
        n_pos = np.searchsorted(self.hand_pos, h, side="left")
        sum_hand = np.where(n_pos > 0, self.cum_hand[np.maximum(n_pos - 1, 0)], 0.0)
        return n_pos, sum_hand

    def attach_raster(self, hand_cells: np.ndarray, pixel_area: float) -> None:
        """Use the reach's HAND cells for exact geometry if they reproduce the table."""
        hand_cells = np.asarray(hand_cells, float)
        is_zero = np.abs(hand_cells) < 1e-6
        pos = np.sort(hand_cells[~is_zero])
        self.hand_pos = pos
        self.cum_hand = np.cumsum(pos, dtype=np.float64)
        self.zero_count = int(is_zero.sum())
        self.pixel_area = float(pixel_area)
        n_pos, _ = self._raster_counts(self.stages)
        n_r = n_pos + self.zero_count
        denom = max(float(np.max(self.ncells)), 1.0)
        self.cell_mismatch = float(np.max(np.abs(n_r - self.ncells)) / denom)
        if self.cell_mismatch <= GEOM_MATCH_TOL:
            self.source = "raster_exact"
        else:
            self.source = "table_interp"
            self.hand_pos = self.cum_hand = None

    # evaluation ------------------------------------------------------
    def at(self, h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(SurfaceArea m2, BedArea m2, Volume m3) at stages h (0 ≤ h ≤ h_max)."""
        h = np.clip(np.atleast_1d(np.asarray(h, float)), 0.0, self.h_max)
        if self.source != "raster_exact":
            return (np.interp(h, self.stages, self.surf),
                    np.interp(h, self.stages, self.bed),
                    np.interp(h, self.stages, self.vol))
        pa, zc = self.pixel_area, self.zero_count
        n_pos, sum_hand = self._raster_counts(h)
        surf = (n_pos + zc) * pa
        vol = np.maximum(h * n_pos * pa - sum_hand * pa + h * zc * pa, 0.0)
        # bed: table value at the grid row below + new cells × that interval's bed factor
        j = np.clip(np.searchsorted(self.stages, h, side="right") - 1, 0, len(self.stages) - 2)
        s_lo, s_hi = self.stages[j], self.stages[j + 1]
        n_lo, _ = self._raster_counts(s_lo)
        n_hi, _ = self._raster_counts(s_hi)
        d_surf_r = (n_hi - n_lo) * pa
        d_bed_t = self.bed[j + 1] - self.bed[j]
        factor = np.where(d_surf_r > 0, d_bed_t / np.where(d_surf_r > 0, d_surf_r, 1.0), 1.0)
        bed = self.bed[j] + (n_pos - n_lo) * pa * factor
        return surf, bed, vol

    def AP(self, h) -> tuple[np.ndarray, np.ndarray]:
        """Reach-averaged wet area A (m2) and wetted perimeter P (m)."""
        _, bed, vol = self.at(h)
        return vol / self.length_m, bed / self.length_m


def geometry_from_hydrotable(hdf: pd.DataFrame) -> ReachGeometry:
    hdf = hdf.sort_values("stage").drop_duplicates("stage")
    return ReachGeometry(
        stages=hdf["stage"].to_numpy(float),
        surf=hdf["SurfaceArea (m2)"].to_numpy(float),
        bed=hdf["BedArea (m2)"].to_numpy(float),
        vol=hdf["Volume (m3)"].to_numpy(float),
        ncells=hdf["Number of Cells"].to_numpy(float),
        length_m=float(hdf["LENGTHKM"].iloc[0]) * 1000.0,
    )


# ══════════════════════════════════════════════════════════════════
# Composite Horton–Einstein discharge
# ══════════════════════════════════════════════════════════════════

@dataclass
class Bands:
    h1: float
    h2: float
    P0: float
    Ph1: float
    Ph2: float
    bathy: tuple[float, float, float] | None   # (w, h0, n_bathy) or None


def he_curve(A: np.ndarray, P: np.ndarray, h: np.ndarray, bands: Bands,
             n1: float, n2: float, n3: float, slope: float) -> dict[str, np.ndarray]:
    """Banded perimeters, composite n and Q at stages h given HAND A(h), P(h)."""
    h = np.asarray(h, float)
    P_min_h1 = np.where(h <= bands.h1, P, bands.Ph1)
    P_min_h2 = np.where(h <= bands.h2, P, bands.Ph2)
    P2 = np.maximum(0.0, P_min_h2 - bands.Ph1)
    P3 = np.maximum(0.0, P - bands.Ph2)
    if bands.bathy is not None:
        w, d, nb = bands.bathy
        Pb = np.full_like(h, w + 2.0 * d)
        P1 = np.maximum(0.0, P_min_h1 - bands.P0)
        A_s = A + w * d
        P_s = Pb + P - bands.P0
    else:
        nb = 0.0
        Pb = np.zeros_like(h)
        P1 = P_min_h1
        A_s = A
        P_s = P
    num = Pb * nb ** 1.5 + P1 * n1 ** 1.5 + P2 * n2 ** 1.5 + P3 * n3 ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        nc = np.where(P_s > 0, (num / np.where(P_s > 0, P_s, 1.0)) ** (2.0 / 3.0), n1)
        R = np.where(P_s > 0, A_s / np.where(P_s > 0, P_s, 1.0), 0.0)
        Q = np.where((A_s > 0) & (P_s > 0),
                     A_s * np.power(np.maximum(R, 0.0), 2.0 / 3.0) * np.sqrt(slope) / nc, 0.0)
    return {"A": A_s, "P": P_s, "P_bathy": Pb, "P1": P1, "P2": P2, "P3": P3,
            "n_c": nc, "Q": Q}


def solve_n_bathy(Q0: float, w: float, h0: float, slope: float) -> float:
    """n that makes a w × h0 rectangle carry Q0 (Manning, full perimeter)."""
    A = w * h0
    R = A / (w + 2.0 * h0)
    return A * R ** (2.0 / 3.0) * np.sqrt(slope) / Q0


# ══════════════════════════════════════════════════════════════════
# Observed rating curve helpers
# ══════════════════════════════════════════════════════════════════

class ObservedRC:
    def __init__(self, gage_rc: pd.DataFrame, dem_adj: float):
        g = gage_rc.sort_values("elev_m").drop_duplicates("elev_m")
        self.elev = g["elev_m"].to_numpy(float)
        self.flow_by_elev = g["flow_cms"].to_numpy(float)
        f = gage_rc.sort_values("flow_cms").drop_duplicates("flow_cms")
        self.flow = f["flow_cms"].to_numpy(float)
        self.elev_by_flow = f["elev_m"].to_numpy(float)
        self.dem_adj = dem_adj
        self.hand = self.elev - dem_adj

    @property
    def q_max(self) -> float:
        return float(self.flow.max())

    def in_range(self, q: float) -> bool:
        return bool(self.flow.min() <= q <= self.flow.max())

    def stage_of(self, q: float) -> float:
        return float(np.interp(q, self.flow, self.elev_by_flow)) - self.dem_adj

    def q_at(self, h) -> np.ndarray:
        """Observed Q at HAND stages h; NaN outside the observed elevation range."""
        e = np.atleast_1d(np.asarray(h, float)) + self.dem_adj
        q = np.interp(e, self.elev, self.flow_by_elev)
        return np.where((e >= self.elev[0]) & (e <= self.elev[-1]), q, np.nan)


# ══════════════════════════════════════════════════════════════════
# Zone fit
# ══════════════════════════════════════════════════════════════════

def _golden(f, a: float, b: float, iters: int = 40) -> float:
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def fit_zone(k: int, h_lo: float, h_hi: float, check_hi: float,
             geom: ReachGeometry, bands: Bands, n_fixed: dict, slope: float,
             orc: ObservedRC, constrain: bool = True) -> dict:
    """Fit n_k on 3 stages of zone k under a monotone-Q constraint."""
    pts = h_lo + (h_hi - h_lo) * np.asarray(FIT_FRACTIONS)
    q_obs = orc.q_at(pts)
    ok = np.isfinite(q_obs) & (q_obs > 0)
    out = {"n": np.nan, "n_pts": int(ok.sum()), "clip": "", "monotone_ok": True,
           "fit_rmse_logQ": np.nan}
    if not ok.any():
        return out
    pts, q_obs = pts[ok], q_obs[ok]

    chk = np.unique(np.concatenate([
        [h_lo, h_hi, check_hi],
        geom.stages[(geom.stages > h_lo) & (geom.stages < check_hi)],
    ]))
    chk = chk[chk <= geom.h_max]
    A_p, P_p = geom.AP(pts)
    A_c, P_c = geom.AP(chk)

    def ns(nk):
        n = dict(n_fixed)
        n[k] = nk
        # zones above k are not fitted yet: give them n_k (no effect below h_k)
        for j in range(k + 1, 4):
            n.setdefault(j, nk)
        return n[1], n[2], n[3]

    def obj(nk):
        q = he_curve(A_p, P_p, pts, bands, *ns(nk), slope)["Q"]
        if np.any(q <= 0):
            return 1e9
        return float(np.sum((np.log(q) - np.log(q_obs)) ** 2))

    def feasible(nk):
        q = he_curve(A_c, P_c, chk, bands, *ns(nk), slope)["Q"]
        return bool(np.all(np.diff(q) >= -MONO_TOL * max(float(q.max()), 1.0)))

    objs = np.array([obj(n) for n in N_GRID])
    feas = np.array([feasible(n) for n in N_GRID]) if constrain else np.ones(len(N_GRID), bool)
    if feas.any():
        cand = np.where(feas, objs, np.inf)
    else:
        cand = objs
        out["monotone_ok"] = False
    i = int(np.argmin(cand))
    n_best = float(N_GRID[i])
    lo, hi = N_GRID[max(i - 1, 0)], N_GRID[min(i + 1, len(N_GRID) - 1)]
    if hi > lo:
        n_ref = _golden(obj, float(lo), float(hi))
        if obj(n_ref) < obj(n_best) and (not constrain or feasible(n_ref)
                                         or not out["monotone_ok"]):
            n_best = n_ref
    n_best = float(np.clip(n_best, N_MIN, N_MAX))
    out["n"] = n_best
    out["fit_rmse_logQ"] = float(np.sqrt(obj(n_best) / len(pts)))
    if n_best <= N_MIN * (1 + 1e-6):
        out["clip"] = "low"
    elif n_best >= N_MAX * (1 - 1e-6):
        out["clip"] = "high"
    return out


# ══════════════════════════════════════════════════════════════════
# One gauge
# ══════════════════════════════════════════════════════════════════

def calibrate_gauge(geom: ReachGeometry, slope: float, gage_rc: pd.DataFrame,
                    dem_adj: float, q_recur: dict, mf_width: float | None,
                    base_stage: np.ndarray | None = None,
                    base_q: np.ndarray | None = None,
                    check_only: bool = False,
                    monotone_mode: str | None = None) -> tuple[dict, pd.DataFrame | None]:
    """
    Calibrate one gauged HydroID.  q_recur: {2: Q2, 50: Q50, 100: Q100 or None} (cms).
    Returns (diag, curve).  diag["status"] is "calibrated" or a skip reason.
    check_only=True runs only the pre-checks that need no geometry
    (geom may be None) and returns status "ok" when they pass.
    """
    d: dict = {"status": "", "Q0_raw_cms": np.nan, "Q0_cms": np.nan, "h0_m": np.nan,
               "h0_elev_m": np.nan, "pzf_b": np.nan, "pzf_r2": np.nan,
               "mf_width_m": mf_width if mf_width is not None else np.nan,
               "V0_ms": np.nan, "bathy_band": False, "n_bathy": np.nan,
               "n_bathy_raw": np.nan, "n_bathy_clip": "",
               "Q_tr2_cms": np.nan, "h_tr2_m": np.nan,
               "Q_tr50_cms": np.nan, "h_tr50_m": np.nan,
               "last_zone_top": "", "Q_top_cms": np.nan, "h_top_m": np.nan,
               "top_clipped_to_table": False, "slope": slope}
    orc = ObservedRC(gage_rc, dem_adj)
    mode = (monotone_mode or MONOTONE_MODE).lower()
    if mode not in ("constrain", "cummax"):
        raise ValueError(f"monotone mode must be 'constrain' or 'cummax', got {mode!r}")
    d["monotone_mode"] = mode

    # Tr2 / Tr50 must exist and be on the observed curve
    for yr in (2, 50):
        q = q_recur.get(yr)
        if q is None or not np.isfinite(q) or q <= 0:
            d["status"] = f"missing_Tr{yr}"
            return d, None
        if not orc.in_range(q):
            d["status"] = f"Tr{yr}_outside_observed_RC"
            return d, None
        d[f"Q_tr{yr}_cms"] = float(q)
        d[f"h_tr{yr}_m"] = orc.stage_of(q)
    h1, h2 = d["h_tr2_m"], d["h_tr50_m"]
    if h1 <= 0:
        d["status"] = "Tr2_at_or_below_HAND0"
        return d, None
    if h2 <= h1:
        d["status"] = "Tr50_not_above_Tr2"
        return d, None

    # h0 (PZF) and Q0
    pzf = fit_pzf(gage_rc, dem_adj)
    if pzf is None:
        d["status"] = "pzf_fit_failed"
        return d, None
    h0 = float(pzf["delta_m"])
    d.update(h0_m=h0, h0_elev_m=pzf["h0_elev_m"], pzf_b=pzf["b"], pzf_r2=pzf["r2"])
    q0_raw = float(baseflow_q0(gage_rc, dem_adj))
    Q0 = max(q0_raw, 0.0)
    d.update(Q0_raw_cms=q0_raw, Q0_cms=Q0)

    if Q0 > 0:
        if mf_width is None or not np.isfinite(mf_width) or mf_width <= 0:
            d["status"] = "no_mf_width"
            return d, None
        V0 = Q0 / (mf_width * h0)
        d["V0_ms"] = V0
        if V0 > V_MAX:
            d["status"] = "V0_gt_1ms"
            return d, None
        if slope <= 0:
            d["status"] = "nonpositive_slope"
            return d, None
        nb_raw = solve_n_bathy(Q0, mf_width, h0, slope)
        nb = float(np.clip(nb_raw, N_MIN, N_MAX))
        d.update(bathy_band=True, n_bathy_raw=nb_raw, n_bathy=nb,
                 n_bathy_clip="low" if nb_raw < N_MIN else ("high" if nb_raw > N_MAX else ""))
        bathy = (float(mf_width), h0, nb)
    else:
        bathy = None

    # top of zone 3
    q100 = q_recur.get(100)
    if q100 is not None and np.isfinite(q100) and q100 > d["Q_tr50_cms"] and orc.in_range(q100):
        q_top, top_lab = float(q100), "Tr100"
    else:
        q_top, top_lab = orc.q_max, "observed_RC_max"
    if check_only:
        d["status"] = "ok"
        return d, None

    h_top = orc.stage_of(q_top)
    if h2 >= geom.h_max:
        d["status"] = "Tr50_above_table_max_stage"
        return d, None
    if h_top > geom.h_max:
        h_top = geom.h_max
        d["top_clipped_to_table"] = True
    fit_zone3 = (h_top - h2) >= MIN_ZONE3_THICKNESS
    if not fit_zone3:
        top_lab, h_top, q_top = "Tr50 (no data above)", h2, d["Q_tr50_cms"]
    d.update(last_zone_top=top_lab, Q_top_cms=q_top, h_top_m=h_top)

    P0 = float(geom.AP(0.0)[1][0])
    _, Ph = geom.AP(np.array([h1, h2]))
    bands = Bands(h1=h1, h2=h2, P0=float(P0), Ph1=float(Ph[0]), Ph2=float(Ph[1]),
                  bathy=bathy)
    if bathy is None and P0 <= 0 and Ph[0] <= 0:
        d["status"] = "no_wetted_perimeter"
        return d, None

    # sequential zone fits
    n_fixed: dict = {}
    zones = [(1, 0.0, h1, h1), (2, h1, h2, h2)]
    if fit_zone3:
        zones.append((3, h2, h_top, geom.h_max))
    mono_all = True
    for k, lo, hi, chk_hi in zones:
        r = fit_zone(k, lo, hi, chk_hi, geom, bands, n_fixed, slope, orc,
                     constrain=(mode == "constrain"))
        if not np.isfinite(r["n"]):
            d["status"] = f"no_observed_points_zone{k}"
            return d, None
        n_fixed[k] = r["n"]
        d[f"n{k}"] = r["n"]
        d[f"n{k}_clip"] = r["clip"]
        d[f"n{k}_fit_pts"] = r["n_pts"]
        d[f"n{k}_fit_rmse_logQ"] = r["fit_rmse_logQ"]
        mono_all &= r["monotone_ok"]
    if not fit_zone3:
        n_fixed[3] = n_fixed[2]
        d.update(n3=n_fixed[2], n3_clip="", n3_fit_pts=0, n3_fit_rmse_logQ=np.nan)
    n1, n2, n3 = n_fixed[1], n_fixed[2], n_fixed[3]

    # calibrated curve on the table grid + exact rows at h1, h2
    st = np.unique(np.concatenate([geom.stages, [h1, h2]]))
    surf, bed, vol = geom.at(st)
    A, P = vol / geom.length_m, bed / geom.length_m
    c = he_curve(A, P, st, bands, n1, n2, n3, slope)
    q_raw = c["Q"]
    q_fin = np.maximum.accumulate(q_raw) if mode == "cummax" else q_raw
    curve = pd.DataFrame({
        "stage": st, "inserted_row": ~np.isin(st, geom.stages),
        "SurfaceArea (m2)": surf, "BedArea (m2)": bed, "Volume (m3)": vol,
        "A_hand_m2": A, "P_hand_m": P,
        "A_total_m2": c["A"], "P_total_m": c["P"], "P_bathy_m": c["P_bathy"],
        "P1_m": c["P1"], "P2_m": c["P2"], "P3_m": c["P3"],
        "n_c": c["n_c"], "discharge_cms_raw": q_raw, "discharge_cms": q_fin,
    })
    curve["Q_obs_cms"] = orc.q_at(st)
    if base_stage is not None and base_q is not None:
        curve["discharge_cms_baseline"] = np.interp(st, base_stage, base_q)
    dq = np.diff(q_raw)
    d["nonmonotone_rows"] = int((dq < -MONO_TOL * max(float(q_raw.max()), 1.0)).sum())
    d["monotone_ok"] = bool(mono_all and d["nonmonotone_rows"] == 0)
    d["Q_at_HAND0_cms"] = float(q_fin[0])
    d["geom_source"] = geom.source
    d["geom_cell_mismatch"] = geom.cell_mismatch

    # errors of the FINAL curve vs the observed rating curve, 0 ≤ h ≤ h_top
    sel = (orc.hand >= 0) & (orc.hand <= h_top) & (orc.flow_by_elev > 0)
    hs, qo = orc.hand[sel], orc.flow_by_elev[sel]
    d["err_n_obs"] = int(sel.sum())
    for key in ("err_rmse_logQ", "err_pbias_pct", "err_rmse_stage_m", "err_rmse_logQ_baseline"):
        d[key] = np.nan
    if sel.sum() >= 2:
        st_all = np.unique(np.concatenate([st, hs]))
        A_a, P_a = geom.AP(st_all)
        q_all = he_curve(A_a, P_a, st_all, bands, n1, n2, n3, slope)["Q"]
        if mode == "cummax":
            q_all = np.maximum.accumulate(q_all)
        qm = np.interp(hs, st_all, q_all)
        good = qm > 0
        if good.any():
            d["err_rmse_logQ"] = float(np.sqrt(np.mean((np.log(qm[good]) - np.log(qo[good])) ** 2)))
        d["err_pbias_pct"] = float(100.0 * (qm.sum() - qo.sum()) / qo.sum())
        q_mono = np.maximum.accumulate(q_all)
        inside = (qo >= q_mono[0]) & (qo <= q_mono[-1])
        if inside.any():
            h_mod = np.interp(qo[inside], q_mono, st_all)
            d["err_rmse_stage_m"] = float(np.sqrt(np.mean((h_mod - hs[inside]) ** 2)))
        if base_stage is not None and base_q is not None:
            qb = np.interp(hs, base_stage, base_q)
            gb = qb > 0
            if gb.any():
                d["err_rmse_logQ_baseline"] = float(np.sqrt(np.mean((np.log(qb[gb]) - np.log(qo[gb])) ** 2)))
    d["status"] = "calibrated"
    return d, curve


# ══════════════════════════════════════════════════════════════════
# Raster access (rem_zeroed_masked + catchment raster, as build_src)
# ══════════════════════════════════════════════════════════════════

def read_branch_cells(branch_dir: Path, bid: str, hydroids: list[int]
                      ) -> tuple[dict[int, np.ndarray], float] | None:
    """HAND values of every valid cell for the given HydroIDs, plus pixel area.
    Handles float rasters and the Int16-encoded variant (convert_to_int16)."""
    import rasterio

    hand_p = branch_dir / f"rem_zeroed_masked_{bid}.tif"
    cat_p = branch_dir / f"gw_catchments_reaches_filtered_addedAttributes_{bid}.tif"
    if not (hand_p.is_file() and cat_p.is_file()):
        return None
    with rasterio.open(hand_p) as ds:
        hand = ds.read(1)
        h_nod = ds.nodata
        pa = abs(ds.transform.a * ds.transform.e)
    with rasterio.open(cat_p) as ds:
        cat = ds.read(1).astype(np.int64)
        c_nod = ds.nodata
    if hand.dtype == np.int16:
        valid_h = hand != 32767
        hand = hand.astype(np.float64) / 1000.0
    else:
        hand = hand.astype(np.float64)
        valid_h = ~np.isnan(hand)
        if h_nod is not None:
            valid_h &= hand != h_nod
    valid_c = cat > 0
    if c_nod is not None:
        valid_c &= cat != c_nod
    prefix_f = branch_dir / "hydroid_prefix.txt"
    if prefix_f.is_file():
        cat = np.where(valid_c, cat + int(prefix_f.read_text().strip()) * 10000, cat)
    m = valid_h & valid_c & np.isin(cat, hydroids)
    cats, hands = cat[m], hand[m]
    out = {int(h): hands[cats == h] for h in hydroids}
    return out, float(pa)
