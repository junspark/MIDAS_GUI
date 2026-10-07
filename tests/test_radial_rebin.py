"""Uniform-Q and uniform-2θ OUTPUT rebinning (``workers.rebin_*``).

Neither is a backend binning mode. ``IntegrationSpec`` does carry
QMin/QMax/QBinSize, but ``workers.apply_q_uniform`` is deliberately unused
and there is no 2θ equivalent at all — so both units are produced the same
way: integrate R-uniform, then resample the finished 1-D profile. These
tests pin that arithmetic, and in particular the thing that went wrong
once (4cb7aaa): the axis stored alongside a rebinned profile is **R in
pixels**, not the grid it was rebinned onto.

The arithmetic half is pure numpy, so it runs in the headless ``batch_cli``
world too; the last section drives a real ``BatchWorker`` to pin the wiring
and imports Qt inside a fixture per STATE.md.
"""
import math

import numpy as np
import pytest

from midas_gui.workers import (REBIN_UNITS, rebin_cfg_parts, rebin_grid_and_r,
                               rebin_R_to_grid)

# One real geometry, from the run that exposed the Q-axis bug.
LSD, PX, WL = 13866027.0, 75.0, 0.13060


# ── the config shape ────────────────────────────────────────────────────────

def test_canonical_cfg_round_trips():
    assert rebin_cfg_parts(
        {"unit": "2th", "min": 1.0, "max": 9.0, "step": 0.01}) == \
        ("2th", 1.0, 9.0, 0.01)


def test_legacy_q_only_spelling_is_read_as_Q():
    """Projects saved before 2θ existed carry QMin/QMax/QBinSize and no
    "unit" key. They must keep loading, and keep meaning Q."""
    assert rebin_cfg_parts({"QMin": 0.5, "QMax": 8.0, "QBinSize": 0.01}) == \
        ("Q", 0.5, 8.0, 0.01)


def test_both_spellings_give_the_same_grid():
    legacy = rebin_grid_and_r({"QMin": 0.5, "QMax": 8.0, "QBinSize": 0.01},
                              LSD, PX, WL)
    modern = rebin_grid_and_r({"unit": "Q", "min": 0.5, "max": 8.0, "step": 0.01},
                              LSD, PX, WL)
    for a, b in zip(legacy, modern):
        assert np.array_equal(a, b)


def test_no_cfg_is_all_None():
    assert rebin_cfg_parts(None) == (None, None, None, None)


def test_an_unknown_unit_is_refused():
    with pytest.raises(ValueError, match="unknown output rebin unit"):
        rebin_cfg_parts({"unit": "d", "min": 1.0, "max": 2.0, "step": 0.1})


# ── the grids ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("unit,lo,hi,step", [("Q", 0.5, 8.0, 0.01),
                                             ("2th", 0.5, 9.0, 0.01)])
def test_grid_is_uniform_in_its_own_unit_and_not_in_R(unit, lo, hi, step):
    grid, r_of_grid = rebin_grid_and_r(
        {"unit": unit, "min": lo, "max": hi, "step": step}, LSD, PX, WL)
    assert np.allclose(np.diff(grid), step)
    # ...and correspondingly NOT uniform in pixels — that is the whole point.
    dr = np.diff(r_of_grid)
    assert dr.max() / dr.min() > 1.01
    assert np.all(dr > 0)


def test_the_2theta_grid_lands_where_the_geometry_says():
    grid, r_of_grid = rebin_grid_and_r(
        {"unit": "2th", "min": 0.5, "max": 9.0, "step": 0.01}, LSD, PX, WL)
    assert np.allclose(np.degrees(np.arctan(r_of_grid * PX / LSD)), grid)


def test_2theta_needs_no_wavelength():
    """R = Lsd·tan(2θ)/px has no λ in it. A caller with an unknown
    wavelength still gets a correct 2θ grid."""
    cfg = {"unit": "2th", "min": 1.0, "max": 5.0, "step": 0.01}
    a = rebin_grid_and_r(cfg, LSD, PX, WL)[1]
    b = rebin_grid_and_r(cfg, LSD, PX, WL * 3.0)[1]
    assert np.array_equal(a, b)


def test_the_second_return_value_is_pixels_in_every_unit():
    """The regression behind 4cb7aaa, stated once for all units: what
    travels with a rebinned profile is R in pixels. Both grids here span
    the same detector radii, so neither can be mistaken for its own unit."""
    for cfg in ({"unit": "Q", "min": 0.5, "max": 8.0, "step": 0.01},
                {"unit": "2th", "min": 0.5, "max": 9.0, "step": 0.01}):
        _grid, r_of_grid = rebin_grid_and_r(cfg, LSD, PX, WL)
        assert 1_000.0 < r_of_grid.min() < r_of_grid.max() < 100_000.0


# ── agreement with the display-side converter ───────────────────────────────

def test_grid_math_agrees_with_widgets_convert_radial():
    """``workers`` keeps its own copy of the unit conversions so that
    ``batch_cli`` never has to import Qt. This is the guard that stops the
    two from drifting apart."""
    from midas_gui.widgets import _convert_radial
    for unit in REBIN_UNITS:
        lo, hi, step = (0.5, 8.0, 0.01) if unit == "Q" else (0.5, 9.0, 0.01)
        grid, r_of_grid = rebin_grid_and_r(
            {"unit": unit, "min": lo, "max": hi, "step": step}, LSD, PX, WL)
        # pixels → unit, the direction the views convert in
        assert np.allclose(
            _convert_radial(r_of_grid, LSD, PX, WL, "R", unit), grid, rtol=1e-12)
        # unit → pixels, the direction rebin_grid_and_r built them in
        assert np.allclose(
            _convert_radial(grid, LSD, PX, WL, unit, "R"), r_of_grid, rtol=1e-12)


# ── the rebin itself ────────────────────────────────────────────────────────

#: Ring width for the synthetic lineouts, in pixels. Comfortably wider than
#: the ~16-19 px an output bin spans at these settings, so the rings are
#: resolved on the rebinned grid rather than falling between samples.
RING_PX = 150.0


def _ring_profile(r_ax, centres, width=RING_PX):
    """A synthetic lineout: unit-height Gaussians at the given radii."""
    prof = np.zeros_like(r_ax)
    for c in centres:
        prof += np.exp(-0.5 * ((r_ax - c) / width) ** 2)
    return prof


@pytest.mark.parametrize("unit,lo,hi,step", [("Q", 0.5, 8.0, 0.005),
                                             ("2th", 0.5, 9.0, 0.005)])
def test_peaks_keep_their_position_through_the_rebin(unit, lo, hi, step):
    """Rings must come out at the right Q / 2θ — the reason to rebin at all."""
    from midas_gui.widgets import _convert_radial
    r_ax = np.arange(500.0, 32_000.0, 1.0)
    peaks_px = np.array([4_000.0, 11_000.0, 23_500.0])
    prof = _ring_profile(r_ax, peaks_px)

    grid, _r_of_grid = rebin_grid_and_r(
        {"unit": unit, "min": lo, "max": hi, "step": step}, LSD, PX, WL)
    out, _ = rebin_R_to_grid(r_ax, prof, None, grid, LSD, PX, WL, unit)

    expected = _convert_radial(peaks_px, LSD, PX, WL, "R", unit)
    for want in expected:
        near = np.abs(grid - want) < 40 * step
        assert out[near].max() == pytest.approx(1.0, abs=0.02)
        assert grid[near][np.argmax(out[near])] == pytest.approx(want, abs=3 * step)


def test_Q_rebin_matches_what_it_was_before_the_generalisation():
    """The old rebin_R_to_Q, inlined. Q runs must not have changed.

    Agreement is to ~1 ulp rather than exact: the old expression computed
    ``radians(degrees(arctan(...)))`` — a round-trip through degrees that
    cancels mathematically but not in floating point. The generalised form
    stays in radians, so it differs in the last bit (rel. 2.6e-16 over this
    whole axis) and is, if anything, the more accurate of the two."""
    r_ax = np.arange(500.0, 32_000.0, 1.0)
    prof = _ring_profile(r_ax, [4_000.0, 11_000.0])
    sigma = np.sqrt(prof + 1.0)
    qgrid, _ = rebin_grid_and_r({"unit": "Q", "min": 0.5, "max": 8.0,
                                 "step": 0.01}, LSD, PX, WL)

    q_of_r = 4 * math.pi * np.sin(
        np.radians(np.degrees(np.arctan(r_ax * PX / LSD))) / 2) / WL
    order = np.argsort(q_of_r)
    want_prof = np.interp(qgrid, q_of_r[order], prof[order])
    want_sig = np.interp(qgrid, q_of_r[order], sigma[order])

    got_prof, got_sig = rebin_R_to_grid(r_ax, prof, sigma, qgrid, LSD, PX, WL, "Q")
    assert np.allclose(got_prof, want_prof, rtol=1e-12, atol=1e-12)
    assert np.allclose(got_sig, want_sig, rtol=1e-12, atol=1e-12)


def test_sigma_rides_along_and_None_stays_None():
    r_ax = np.arange(500.0, 32_000.0, 1.0)
    prof = _ring_profile(r_ax, [9_000.0])
    grid, _ = rebin_grid_and_r({"unit": "2th", "min": 0.5, "max": 9.0,
                                "step": 0.01}, LSD, PX, WL)

    _, sig = rebin_R_to_grid(r_ax, prof, np.sqrt(prof + 1.0), grid,
                             LSD, PX, WL, "2th")
    assert sig is not None and sig.shape == grid.shape and np.all(sig > 0)

    _, none_sig = rebin_R_to_grid(r_ax, prof, None, grid, LSD, PX, WL, "2th")
    assert none_sig is None


def test_Q_and_2theta_put_the_same_ring_in_the_same_place():
    """Two units, one detector: a ring at a known radius has to land at the
    2θ and Q that correspond to each other."""
    from midas_gui.widgets import _convert_radial
    r_ax = np.arange(500.0, 32_000.0, 1.0)
    prof = _ring_profile(r_ax, [13_000.0])

    peaks = {}
    for unit, lo, hi, step in (("Q", 0.5, 8.0, 0.002), ("2th", 0.5, 9.0, 0.002)):
        grid, _ = rebin_grid_and_r(
            {"unit": unit, "min": lo, "max": hi, "step": step}, LSD, PX, WL)
        out, _ = rebin_R_to_grid(r_ax, prof, None, grid, LSD, PX, WL, unit)
        peaks[unit] = grid[int(np.argmax(out))]

    assert _convert_radial(np.array([peaks["2th"]]), LSD, PX, WL, "2th", "Q")[0] \
        == pytest.approx(peaks["Q"], abs=0.01)


# ── the deliberately-unused native hook ─────────────────────────────────────

def test_apply_q_uniform_refuses_2theta():
    """The backend's native spec mode exists for Q only. If anyone ever does
    wire it up, a 2θ config must fail loudly rather than be read as Q."""
    from midas_gui.workers import apply_q_uniform

    class _Spec:
        QMin = QMax = QBinSize = None

    spec = apply_q_uniform(_Spec(), {"unit": "Q", "min": 0.5, "max": 8.0,
                                     "step": 0.01})
    assert (spec.QMin, spec.QMax, spec.QBinSize) == (0.5, 8.0, 0.01)

    with pytest.raises(ValueError, match="native spec binning exists for Q only"):
        apply_q_uniform(_Spec(), {"unit": "2th", "min": 1.0, "max": 9.0,
                                  "step": 0.01})


# ── end to end, through BatchWorker ─────────────────────────────────────────
#
# The unit tests above pin the arithmetic. These two pin the wiring: that the
# grid built once per run and the unit passed per frame actually reach
# rebin_R_to_grid, and that what comes out of a real run is a profile on the
# requested grid with an R-px axis beside it.

@pytest.fixture(scope="module")
def app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _run(app, tmp_path, q_cfg):
    from types import SimpleNamespace
    pytest.importorskip("torch")
    pytest.importorskip("midas_integrate_v2")
    tifffile = pytest.importorskip("tifffile")
    import midas_gui.workers as wk
    from midas_gui.helpers import _build_spec

    rng = np.random.default_rng(0)
    paths = []
    for i in range(2):
        p = tmp_path / f"frame_{i:04d}.tif"
        tifffile.imwrite(str(p), (rng.random((64, 64)) * 100 + 10).astype(np.float32))
        paths.append(str(p))
    calib = SimpleNamespace(
        Lsd=200000.0, BC_y=32.0, BC_z=32.0, tx=0.0, ty=0.0, tz=0.0,
        distortion={}, pxY=200.0, pxZ=200.0,
        NrPixelsY=64, NrPixelsZ=64, wavelength_A=0.1729)
    spec = _build_spec(calib, r_bin=2.0, eta_bin=5.0)

    worker = wk.BatchWorker(
        spec, {"type": "tiff_list", "paths": paths}, None, None, [], "subpixel2",
        (None, None), None, q_cfg=q_cfg)
    results, failures = {}, []
    worker.finished.connect(lambda d: results.update(d))
    worker.failed.connect(failures.append)
    worker.run()           # direct call, not .start() — no real QThread
    assert not failures, failures[0]
    return results, spec


@pytest.mark.parametrize("unit,lo,hi,step", [("Q", 0.5, 4.0, 0.05),
                                             ("2th", 1.0, 6.0, 0.05)])
def test_a_real_run_lands_on_the_requested_grid(app, tmp_path, unit, lo, hi, step):
    cfg = {"unit": unit, "min": lo, "max": hi, "step": step}
    results, spec = _run(app, tmp_path, cfg)

    grid, r_of_grid = rebin_grid_and_r(
        cfg, float(spec.Lsd), float(spec.pxY), float(spec.Wavelength))
    assert results["profiles"].shape == (2, len(grid))
    # ...and the axis stored beside it is R in pixels, not the grid.
    assert np.allclose(results["r_axis_px"], r_of_grid)


def test_radial_keeps_the_kernels_own_grid(app, tmp_path):
    """No rebin configured: the profile stays exactly as integrated."""
    from midas_gui.workers import compute_r_axis
    results, spec = _run(app, tmp_path, None)
    assert results["profiles"].shape == (2, spec.n_r_bins)
    assert np.allclose(results["r_axis_px"], compute_r_axis(spec))
