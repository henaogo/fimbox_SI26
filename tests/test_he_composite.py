"""Tests for the stage-banded Horton–Einstein calibration (scripts/s05g_*)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import s05g_he_composite_core as core  # noqa: E402

PA = 100.0           # 10 m cells
NX = 100             # cells along the reach → L = 1000 m
L = NX * 10.0
STAGES = np.round(np.arange(0.0, 25.2984 + 0.3048, 0.3048), 4)
SLOPE = 5e-4
W = 20.0             # rectangle width = channel strip width


def synthetic_cells():
    """Lateral HAND profile (per 10 m column), repeated NX times along the reach.
    Channel strip 20 m (HAND 0), steep banks, then a flat-ish floodplain."""
    lat = []
    slp = []
    lat += [0.0, 0.0]; slp += [0.01, 0.01]                       # channel strip
    for i in range(1, 9):                                         # banks, 0.4 m per cell
        lat += [0.4 * i] * 2; slp += [0.04 * 10 / 10] * 2
    for i in range(1, 60):                                        # floodplain, 0.05 m per cell
        lat += [3.2 + 0.05 * i] * 2; slp += [0.005] * 2
    for i in range(1, 40):                                        # valley walls
        lat += [6.15 + 0.6 * i] * 2; slp += [0.06] * 2
    hand = np.repeat(np.array(lat)[None, :], NX, axis=0).ravel()
    s = np.repeat(np.array(slp)[None, :], NX, axis=0).ravel()
    # realistic scatter: real HAND is continuous, not terraced
    rng = np.random.default_rng(0)
    pos = hand > 0
    hand[pos] = np.maximum(hand[pos] + rng.uniform(-0.2, 0.2, pos.sum()), 0.01)
    return hand, s


def build_table(hand, s):
    """Replicates build_src.build_src_base for one catchment."""
    bed_c = PA * np.sqrt(1 + s ** 2)
    rows = []
    zero = np.abs(hand) < 1e-6
    for h in STAGES:
        wet = (hand < h) & ~zero
        n = wet.sum() + zero.sum()
        rows.append({"stage": h, "Number of Cells": n, "SurfaceArea (m2)": n * PA,
                     "BedArea (m2)": bed_c[wet].sum() + bed_c[zero].sum(),
                     "Volume (m3)": ((h - hand[wet]).sum() + h * zero.sum()) * PA,
                     "LENGTHKM": L / 1000.0, "SLOPE": SLOPE})
    return pd.DataFrame(rows)


def exact_geom_ref(hand, s, h):
    """True A, P at stage h straight from the cells (slope factor per cell)."""
    bed_c = PA * np.sqrt(1 + s ** 2)
    zero = np.abs(hand) < 1e-6
    wet = (hand < h) & ~zero
    vol = ((h - hand[wet]).sum() + h * zero.sum()) * PA
    return vol / L, (bed_c[wet].sum() + bed_c[zero].sum()) / L


@pytest.fixture(scope="module")
def reach():
    hand, s = synthetic_cells()
    tab = build_table(hand, s)
    g = core.geometry_from_hydrotable(tab)
    g.attach_raster(hand, PA)
    return hand, s, tab, g


TRUE_N = dict(nb=0.035, n1=0.045, n2=0.09, n3=0.14)
DELTA = 1.2
DEM_ADJ = 100.0


def synth_rc(g, h1, h2):
    """Observed rating curve: rectangle below the DEM surface, HE model above."""
    P0 = float(g.AP(0.0)[1][0])
    Ph = g.AP(np.array([h1, h2]))[1]
    bands = core.Bands(h1, h2, P0, float(Ph[0]), float(Ph[1]), (W, DELTA, TRUE_N["nb"]))
    hb = np.linspace(0.02, DELTA, 40)
    qb = (W * hb) * ((W * hb) / (W + 2 * hb)) ** (2 / 3) * np.sqrt(SLOPE) / TRUE_N["nb"]
    ha = np.linspace(0.05, 20.0, 400)
    A, P = g.AP(ha)
    qa = core.he_curve(A, P, ha, bands, TRUE_N["n1"], TRUE_N["n2"], TRUE_N["n3"], SLOPE)["Q"]
    elev = np.concatenate([DEM_ADJ - DELTA + hb, DEM_ADJ + ha])
    q = np.concatenate([qb, qa])
    return pd.DataFrame({"elev_m": elev, "flow_cms": q}), bands


# ── geometry ──────────────────────────────────────────────────────

def test_raster_reproduces_table(reach):
    hand, s, tab, g = reach
    assert g.source == "raster_exact"
    surf, bed, vol = g.at(STAGES)
    np.testing.assert_allclose(surf, tab["SurfaceArea (m2)"], rtol=1e-12)
    np.testing.assert_allclose(bed, tab["BedArea (m2)"], rtol=1e-12)
    np.testing.assert_allclose(vol, tab["Volume (m3)"], rtol=1e-9, atol=1e-6)


def test_inserted_rows_exact(reach):
    hand, s, tab, g = reach
    for h in (0.77, 3.33, 3.4167, 5.01):
        A, P = g.AP(h)
        A_ref, P_ref = exact_geom_ref(hand, s, h)
        assert A[0] == pytest.approx(A_ref, rel=1e-9)
        # wet-cell set is exact; the bed factor of the partial interval is the
        # interval mean (slope raster not available) → tiny error at most
        assert P[0] == pytest.approx(P_ref, rel=2e-3)


def test_P_cumulative_monotone(reach):
    _, _, _, g = reach
    _, P = g.AP(np.linspace(0, 25, 3000))
    assert np.all(np.diff(P) >= -1e-12)


def test_table_interp_fallback_on_mismatch(reach):
    hand, s, tab, _ = reach
    g = core.geometry_from_hydrotable(tab)
    g.attach_raster(hand[: len(hand) // 2], PA)    # wrong cell set
    assert g.source == "table_interp"


# ── composite n properties ────────────────────────────────────────

def test_band_sum_and_continuity(reach):
    # continuity needs a continuous P(h): use the table geometry (linear between
    # rows).  With raster geometry P steps whenever a cell becomes wet, exactly
    # like the SRC table itself.
    _, _, tab, _ = reach
    g = core.geometry_from_hydrotable(tab)
    h1, h2 = 2.0, 4.5
    P0 = float(g.AP(0.0)[1][0]); Ph = g.AP(np.array([h1, h2]))[1]
    for bathy in ((W, DELTA, 0.03), None):
        b = core.Bands(h1, h2, P0, float(Ph[0]), float(Ph[1]), bathy)
        h = np.linspace(0, 12, 6001)
        A, P = g.AP(h)
        c = core.he_curve(A, P, h, b, 0.05, 0.1, 0.15, SLOPE)
        np.testing.assert_allclose(c["P_bathy"] + c["P1"] + c["P2"] + c["P3"], c["P"], rtol=1e-12)
        # n_c continuous across h1, h2 (no jump bigger than the grid change)
        assert np.max(np.abs(np.diff(c["n_c"]))) < 1e-3
        if bathy is None:
            np.testing.assert_allclose(c["n_c"][h <= h1], 0.05)
        else:
            assert c["n_c"][0] == pytest.approx(0.03)


def test_Q_at_hand0_equals_Q0(reach):
    _, _, _, g = reach
    Q0 = 3.0
    nb = core.solve_n_bathy(Q0, W, DELTA, SLOPE)
    P0 = float(g.AP(0.0)[1][0]); Ph = g.AP(np.array([2.0, 4.5]))[1]
    b = core.Bands(2.0, 4.5, P0, float(Ph[0]), float(Ph[1]), (W, DELTA, nb))
    A, P = g.AP(np.array([0.0]))
    assert core.he_curve(A, P, np.array([0.0]), b, 0.05, 0.1, 0.15, SLOPE)["Q"][0] == pytest.approx(Q0)


# ── calibration ───────────────────────────────────────────────────

def test_recovers_known_n(reach):
    _, _, tab, g = reach
    h1_true, h2_true = 2.6, 5.4
    rc, bands = synth_rc(g, h1_true, h2_true)
    orc = core.ObservedRC(rc, DEM_ADJ)
    q2 = float(orc.q_at(h1_true)[0]); q50 = float(orc.q_at(h2_true)[0])
    # unconstrained fit: the true-n curve itself dips at floodplain onset
    d, curve = core.calibrate_gauge(g, SLOPE, rc, DEM_ADJ, {2: q2, 50: q50, 100: None}, W,
                                    base_stage=tab["stage"].to_numpy(),
                                    base_q=np.full(len(tab), 1.0),
                                    monotone_mode="cummax")
    assert d["status"] == "calibrated", d["status"]
    assert d["h0_m"] == pytest.approx(DELTA, abs=0.06)
    assert d["n_bathy"] == pytest.approx(TRUE_N["nb"], rel=0.08)
    assert d["n1"] == pytest.approx(TRUE_N["n1"], rel=0.05)
    assert d["n2"] == pytest.approx(TRUE_N["n2"], rel=0.05)
    assert d["n3"] == pytest.approx(TRUE_N["n3"], rel=0.05)
    assert d["last_zone_top"] == "observed_RC_max"
    assert d["err_rmse_logQ"] < 0.05
    # exact rows at h1 and h2 are present
    assert np.isclose(curve["stage"], d["h_tr2_m"]).any()
    assert np.isclose(curve["stage"], d["h_tr50_m"]).any()
    assert np.all(np.diff(curve["discharge_cms"]) >= -1e-9)
    assert curve["discharge_cms"].iloc[0] == pytest.approx(d["Q0_cms"], rel=1e-6)


def test_skip_rules(reach):
    _, _, _, g = reach
    rc, _ = synth_rc(g, 2.6, 5.4)
    orc = core.ObservedRC(rc, DEM_ADJ)
    q2 = float(orc.q_at(2.6)[0]); q50 = float(orc.q_at(5.4)[0])
    run = lambda qr, w: core.calibrate_gauge(g, SLOPE, rc, DEM_ADJ, qr, w)[0]["status"]
    assert run({2: None, 50: q50}, W) == "missing_Tr2"
    assert run({2: q2, 50: np.nan}, W) == "missing_Tr50"
    assert run({2: q2, 50: q50}, 0.2) == "V0_gt_1ms"
    assert run({2: q2, 50: q50}, None) == "no_mf_width"
    assert run({2: q2, 50: 1e7}, W) == "Tr50_outside_observed_RC"


def test_monotone_constraint_on_terraced_geometry():
    """Terraced HAND (all cells of a bank row at one height) makes Q dip for
    some n; the fit must still return a non-decreasing curve or flag it."""
    lat = [0.0, 0.0] + [0.5 * i for i in range(1, 30) for _ in range(2)]
    hand = np.repeat(np.array(lat)[None, :], NX, axis=0).ravel()
    s = np.full_like(hand, 0.05)
    tab = build_table(hand, s)
    g = core.geometry_from_hydrotable(tab)
    g.attach_raster(hand, PA)
    rc, _ = synth_rc(g, 2.6, 5.4)
    orc = core.ObservedRC(rc, DEM_ADJ)
    d, curve = core.calibrate_gauge(
        g, SLOPE, rc, DEM_ADJ,
        {2: float(orc.q_at(2.6)[0]), 50: float(orc.q_at(5.4)[0])}, W,
        monotone_mode="constrain")
    assert d["status"] == "calibrated"
    if d["monotone_ok"]:
        assert np.all(np.diff(curve["discharge_cms"]) >= -1e-9)
    else:
        assert d["nonmonotone_rows"] > 0


def test_n_limits():
    assert core.N_MIN == 0.007 and core.N_MAX == 0.36
