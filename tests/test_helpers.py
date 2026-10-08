"""Unit tests for lightweight, profile-scoped widget helpers in helpers.py.

Covers the pieces added to fix "profile switch doesn't refresh option lists":
refresh_combo_items (used by the Calibrant dropdown) and the pixel-size /
K-edge-foil popup menus rebuilding their entries from live constants each
time they're opened, instead of freezing them at construction.
"""
import math

import numpy as np
import pytest


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_refresh_combo_items_preserves_existing_selection(app):
    from midas_gui.helpers import _NoScrollComboBox, refresh_combo_items

    combo = _NoScrollComboBox()
    combo.addItems(["A", "B", "C"])
    combo.setCurrentText("B")

    refresh_combo_items(combo, ["X", "B", "Y"])

    assert [combo.itemText(i) for i in range(combo.count())] == ["X", "B", "Y"]
    assert combo.currentText() == "B"


def test_refresh_combo_items_falls_back_when_selection_gone(app):
    from midas_gui.helpers import _NoScrollComboBox, refresh_combo_items

    combo = _NoScrollComboBox()
    combo.addItems(["A", "B"])
    combo.setCurrentText("B")

    refresh_combo_items(combo, ["X", "Y"])

    assert combo.currentText() == "X"


def test_pixel_label_menu_rebuilds_from_current_constants(app, monkeypatch):
    """make_pixel_label's popup menu must reflect constants.PIXEL_PRESETS as
    of when it's opened, not as of when the label was constructed — this is
    what makes it survive a profile switch with no extra wiring."""
    import midas_gui.constants as C
    from midas_gui.helpers import make_pixel_label, _fspin

    monkeypatch.setattr(C, "PIXEL_PRESETS", [("Before", 100.0)])
    px_spin = _fspin(1.0, 1000.0, 3, 50.0)
    btn = make_pixel_label(px_spin)
    assert [a.text() for a in btn.menu().actions()] == ["Before  (100 µm)"]

    monkeypatch.setattr(C, "PIXEL_PRESETS", [("After1", 75.0), ("After2", 150.0)])
    btn.menu().aboutToShow.emit()
    labels = [a.text() for a in btn.menu().actions()]
    assert labels == ["After1  (75 µm)", "After2  (150 µm)"]

    btn.menu().actions()[1].trigger()
    assert px_spin.value() == 150.0


def test_kedge_label_menu_rebuilds_from_current_constants(app, monkeypatch):
    import midas_gui.constants as C
    from midas_gui.helpers import make_kedge_label, _fspin

    monkeypatch.setattr(C, "K_EDGE_FOILS", [("Fe", 7.11)])
    wl_spin = _fspin(0.01, 5.0, 5, 0.2)
    btn = make_kedge_label(wl_spin)
    foil_labels_before = [a.text() for a in btn.menu().actions()
                          if a.text() and not a.isSeparator()]
    assert len(foil_labels_before) == 1
    assert foil_labels_before[0].startswith("Fe")

    monkeypatch.setattr(C, "K_EDGE_FOILS", [("Cu", 8.98), ("Ni", 8.33)])
    btn.menu().aboutToShow.emit()
    foil_labels_after = [a.text() for a in btn.menu().actions()
                         if a.text() and not a.isSeparator()]
    assert len(foil_labels_after) == 2
    assert foil_labels_after[0].startswith("Cu")
    assert foil_labels_after[1].startswith("Ni")


# ── detect_geometry_from_path (auto pxY/wavelength_A on load) ─────────────────
# Detector-from-filename and the HDF5 energy-metadata location are specific to
# the APS 1-ID-E / 20-ID-D / 20-ID-E beamlines, so every case below pins an
# explicit `profile=` rather than depending on whatever profile is active on
# the machine running the tests.

@pytest.mark.parametrize("name,expected", [
    ("scan.ge1", "ge"), ("scan.GE3.h5", "ge"), ("scan_ge4_001.ge4", "ge"),
    ("scan.vrx", "vrx"), ("scan.VRX.h5", "vrx"),
    ("scan.pxrd", "pxrd"),
    # 1-ID-E writes Pixirad frames as ".pixi", not ".pxrd".
    ("scan.pixi", "pxrd"), ("AgBeh_80p725keV_3s_003510.pixi.h5", "pxrd"),
    ("silver_behenate_72keV_001027.pmg.h5", "pimega"), ("scan.PMG", "pimega"),
    ("scan.tif", None), ("scan.h5", None),
])
def test_detect_detector_from_filename(name, expected):
    from midas_gui.helpers import detect_detector_from_filename
    assert detect_detector_from_filename(name) == expected


def test_detect_geometry_gated_to_known_beamline_profiles():
    from midas_gui.helpers import detect_geometry_from_path

    assert detect_geometry_from_path("scan.ge2", profile="Default") == {}
    assert detect_geometry_from_path("scan.ge2", profile="Some Other Profile") == {}


@pytest.mark.parametrize("profile", ["1-ID-E", "20-ID-D", "20-ID-E"])
def test_detect_geometry_pixel_size_from_filename(profile):
    from midas_gui.helpers import detect_geometry_from_path

    assert detect_geometry_from_path("scan.ge1", profile=profile) == {"pxY": 200.0}
    assert detect_geometry_from_path("scan.vrx", profile=profile) == {"pxY": 150.0}
    assert detect_geometry_from_path("scan.pmg.h5", profile=profile) == {"pxY": 55.0}
    # Pixirad used to be identified without a pixel size, so loading one left
    # the px box holding whatever the previously-loaded calibration had put
    # there -- a GE's 200 um, in the 1-ID-E SAXS case that surfaced this.
    assert detect_geometry_from_path("scan.pxrd", profile=profile) == {"pxY": 62.0}
    assert detect_geometry_from_path("scan.pixi", profile=profile) == {"pxY": 62.0}
    assert detect_geometry_from_path("scan.tif", profile=profile) == {}


def test_detect_geometry_wavelength_from_h5_energy_metadata(tmp_path):
    h5py = pytest.importorskip("h5py")
    from midas_gui import constants as C
    from midas_gui.helpers import detect_geometry_from_path

    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("instrument/HEM/Energy", data=[10.0])

    detected = detect_geometry_from_path(str(path), profile="20-ID-D")
    assert detected == pytest.approx({"wavelength_A": C.HC_KEV_A / 10.0})


def test_detect_geometry_wavelength_absent_when_dataset_missing(tmp_path):
    h5py = pytest.importorskip("h5py")
    from midas_gui.helpers import detect_geometry_from_path

    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("exchange/data", data=[[1, 2], [3, 4]])

    assert detect_geometry_from_path(str(path), profile="1-ID-E") == {}


def test_detect_geometry_combines_filename_and_h5_metadata(tmp_path):
    """A real Hydra frame file's name carries the detector tag AND its own
    HDF5 metadata carries the energy — both should be detected together."""
    h5py = pytest.importorskip("h5py")
    from midas_gui import constants as C
    from midas_gui.helpers import detect_geometry_from_path

    path = tmp_path / "dark_scan_002030.ge1.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("instrument/HEM/Energy", data=[80.61])

    detected = detect_geometry_from_path(str(path), profile="1-ID-E")
    assert detected == pytest.approx({"pxY": 200.0, "wavelength_A": C.HC_KEV_A / 80.61})


# ── Integration tab Rmin/Rmax presets + bin-grid thinning (pure logic, no Qt) ──

def test_rmax_corner_px_centered_beam():
    from midas_gui.helpers import rmax_corner_px
    import math
    # Centered beam on a 100x100 detector: corner distance is the half-diagonal.
    assert rmax_corner_px(49.5, 49.5, 100, 100) == pytest.approx(
        math.hypot(49.5, 49.5))


def test_rmax_corner_px_off_center_beam():
    from midas_gui.helpers import rmax_corner_px
    import math
    # Beam near the bottom-left corner of a 100x100 detector: farthest corner
    # is the top-right one, at distance (99-10, 99-20) away.
    assert rmax_corner_px(10, 20, 100, 100) == pytest.approx(math.hypot(89, 79))


def test_rmax_edge_px_off_center_beam():
    from midas_gui.helpers import rmax_edge_px
    # Farthest straight edge is whichever perpendicular distance is largest:
    # left=10, right=89, bottom=20, top=79 -> the right edge, at 89.
    assert rmax_edge_px(10, 20, 100, 100) == pytest.approx(89)


def test_rmax_edge_never_exceeds_corner():
    from midas_gui.helpers import rmax_corner_px, rmax_edge_px
    for bc_y, bc_z in [(10, 20), (49.5, 49.5), (5, 990)]:
        assert rmax_edge_px(bc_y, bc_z, 1000, 1000) <= rmax_corner_px(bc_y, bc_z, 1000, 1000)


def test_thinned_bin_edges_no_thinning_needed():
    from midas_gui.helpers import _thinned_bin_edges
    edges = _thinned_bin_edges(0.0, 10.0, 2.0, max_count=50)
    np.testing.assert_allclose(edges, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


def test_thinned_bin_edges_caps_dense_bins():
    from midas_gui.helpers import _thinned_bin_edges
    edges = _thinned_bin_edges(0.0, 1000.0, 0.5, max_count=50)
    assert len(edges) <= 50
    assert edges[0] == pytest.approx(0.0)


def test_thinned_bin_edges_degenerate_range_is_empty():
    from midas_gui.helpers import _thinned_bin_edges
    assert len(_thinned_bin_edges(10.0, 10.0, 1.0, max_count=50)) == 0
    assert len(_thinned_bin_edges(10.0, 0.0, 1.0, max_count=50)) == 0
    assert len(_thinned_bin_edges(0.0, 10.0, 0.0, max_count=50)) == 0


# ═════════════════════════════════════════════════════════════════════════════
# VAREX-style multi-frame HDF5 stack combining (read_hdf5_stack_combined)
# ═════════════════════════════════════════════════════════════════════════════

def _stack_file(tmp_path, n, h=3, w=4, name="s.h5"):
    h5py = pytest.importorskip("h5py")
    data = np.arange(n * h * w, dtype=np.float32).reshape(n, h, w)
    with h5py.File(tmp_path / name, "w") as f:
        f.create_dataset("exchange/data", data=data)
    return tmp_path / name, data


def test_read_hdf5_stack_combined_whole_file_by_default(tmp_path):
    """chunk_size=None collapses the whole (N,H,W) stack to one frame — the
    VAREX case where every raw sub-frame is the same scan point."""
    from midas_gui.helpers import read_hdf5_stack_combined
    path, data = _stack_file(tmp_path, 5)
    out = read_hdf5_stack_combined(path, "exchange/data")
    assert len(out) == 1
    np.testing.assert_allclose(out[0], data.mean(axis=0), rtol=1e-6)
    assert out[0].dtype == np.float32


@pytest.mark.parametrize("op, reducer", [
    ("mean", np.mean), ("sum", np.sum), ("max", np.max), ("median", np.median),
])
def test_read_hdf5_stack_combined_ops(tmp_path, op, reducer):
    from midas_gui.helpers import read_hdf5_stack_combined
    path, data = _stack_file(tmp_path, 5)
    out = read_hdf5_stack_combined(path, "exchange/data", op=op)
    np.testing.assert_allclose(out[0], reducer(data, axis=0), rtol=1e-6)


def test_read_hdf5_stack_combined_unknown_op_falls_back_to_mean(tmp_path):
    from midas_gui.helpers import read_hdf5_stack_combined
    path, data = _stack_file(tmp_path, 4)
    out = read_hdf5_stack_combined(path, "exchange/data", op="nonsense")
    np.testing.assert_allclose(out[0], data.mean(axis=0), rtol=1e-6)


def test_read_hdf5_stack_combined_chunks_contiguously(tmp_path):
    from midas_gui.helpers import read_hdf5_stack_combined
    path, data = _stack_file(tmp_path, 6)
    out = read_hdf5_stack_combined(path, "exchange/data", chunk_size=2, op="sum")
    assert len(out) == 3
    for k in range(3):
        np.testing.assert_allclose(out[k], data[2 * k:2 * k + 2].sum(axis=0),
                                   rtol=1e-6)


def test_read_hdf5_stack_combined_ragged_last_chunk(tmp_path):
    """A stack that doesn't divide evenly keeps its short trailing chunk."""
    from midas_gui.helpers import read_hdf5_stack_combined
    path, data = _stack_file(tmp_path, 5)
    out = read_hdf5_stack_combined(path, "exchange/data", chunk_size=2)
    assert len(out) == 3
    np.testing.assert_allclose(out[-1], data[4:].mean(axis=0), rtol=1e-6)


def test_read_hdf5_stack_combined_raw_start_end_filters_before_chunking(tmp_path):
    """raw_start/raw_end restrict which raw sub-frames are combined at all,
    applied BEFORE chunk_size splits whatever survives — frames outside the
    range are never read into a chunk."""
    from midas_gui.helpers import read_hdf5_stack_combined
    path, data = _stack_file(tmp_path, 10)
    out = read_hdf5_stack_combined(path, "exchange/data", chunk_size=3,
                                   op="sum", raw_start=2, raw_end=7)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], data[2:5].sum(axis=0), rtol=1e-6)
    np.testing.assert_allclose(out[1], data[5:8].sum(axis=0), rtol=1e-6)


def test_read_hdf5_stack_combined_raw_start_end_default_unbounded(tmp_path):
    """A one-sided bound leaves the other side unbounded (start defaults to
    0, end defaults to the dataset's last frame)."""
    from midas_gui.helpers import read_hdf5_stack_combined
    path, data = _stack_file(tmp_path, 6)
    out_start = read_hdf5_stack_combined(path, "exchange/data", raw_start=4)
    np.testing.assert_allclose(out_start[0], data[4:].mean(axis=0), rtol=1e-6)
    out_end = read_hdf5_stack_combined(path, "exchange/data", raw_end=1)
    np.testing.assert_allclose(out_end[0], data[:2].mean(axis=0), rtol=1e-6)


def test_read_hdf5_stack_combined_raw_start_past_end_is_empty(tmp_path):
    """An out-of-range/empty effective slice returns [] rather than raising
    (e.g. a ZeroDivisionError-shaped `range(..., step=0)`)."""
    from midas_gui.helpers import read_hdf5_stack_combined
    path, _data = _stack_file(tmp_path, 5)
    assert read_hdf5_stack_combined(path, "exchange/data", raw_start=10) == []
    assert read_hdf5_stack_combined(path, "exchange/data", chunk_size=2, raw_start=10) == []


def test_read_hdf5_stack_combined_raw_bounds_ignored_for_2d(tmp_path):
    """A plain (H,W) dataset has nothing to sub-select — raw_start/raw_end
    are ignored, same as chunk_size/op already are."""
    h5py = pytest.importorskip("h5py")
    from midas_gui.helpers import read_hdf5_stack_combined
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(tmp_path / "flat2.h5", "w") as f:
        f.create_dataset("exchange/data", data=img)
    out = read_hdf5_stack_combined(tmp_path / "flat2.h5", "exchange/data",
                                   raw_start=5, raw_end=9)
    assert len(out) == 1
    np.testing.assert_allclose(out[0], img)


def test_read_hdf5_stack_combined_passes_2d_through(tmp_path):
    """A plain (H,W) dataset is already one frame; chunk_size/op are ignored."""
    h5py = pytest.importorskip("h5py")
    from midas_gui.helpers import read_hdf5_stack_combined
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(tmp_path / "flat.h5", "w") as f:
        f.create_dataset("exchange/data", data=img)
    out = read_hdf5_stack_combined(tmp_path / "flat.h5", "exchange/data",
                                   chunk_size=2, op="sum")
    assert len(out) == 1
    np.testing.assert_allclose(out[0], img)


# ═════════════════════════════════════════════════════════════════════════════
# Tilt-aware overlay geometry (tilted_ring_xy / tilted_spoke_xy)
# ═════════════════════════════════════════════════════════════════════════════

_GEO = dict(Lsd_um=500000.0, bc_y=1024.0, bc_z=980.0, pxY_um=200.0, pxZ_um=200.0)


def _expected_radius_px(two_theta_deg):
    """Plain flat-detector radius: Lsd*tan(2θ) converted to pixels."""
    return (_GEO["Lsd_um"] * np.tan(np.radians(two_theta_deg))) / _GEO["pxY_um"]


def test_tilted_ring_reduces_to_a_plain_circle_at_zero_tilt():
    """The documented invariant: with tx=ty=tz=0 the forward projection must
    collapse to bc + r*(sin eta, cos eta), the untilted circle every overlay
    used to draw directly."""
    from midas_gui.helpers import tilted_ring_xy
    tt, n = 8.0, 360
    Y, Z = tilted_ring_xy(tt, 0.0, 0.0, 0.0, n=n, **_GEO)

    r = _expected_radius_px(tt)
    # endpoint=True: the ring grid closes its polyline (last point == first),
    # so the n samples span [0, 360] inclusive at 360/(n-1) spacing.
    eta = np.radians(np.linspace(0.0, 360.0, n, endpoint=True))
    np.testing.assert_allclose(Y, _GEO["bc_y"] + r * np.sin(eta), atol=1e-6)
    np.testing.assert_allclose(Z, _GEO["bc_z"] + r * np.cos(eta), atol=1e-6)


def test_tilted_ring_polyline_is_closed():
    """The ring is fed straight to pg.PlotDataItem, which never auto-closes a
    polyline — so the last sample must repeat the first, or the overlay shows
    a visible seam at eta=0."""
    from midas_gui.helpers import tilted_ring_xy
    for tilt in ((0.0, 0.0, 0.0), (0.5, 2.0, -1.0)):
        Y, Z = tilted_ring_xy(7.0, *tilt, n=180, **_GEO)
        np.testing.assert_allclose([Y[0], Z[0]], [Y[-1], Z[-1]], atol=1e-12)


def test_tilted_ring_is_centred_and_round_at_zero_tilt():
    from midas_gui.helpers import tilted_ring_xy
    tt = 5.0
    Y, Z = tilted_ring_xy(tt, 0.0, 0.0, 0.0, n=180, **_GEO)
    radii = np.hypot(Y - _GEO["bc_y"], Z - _GEO["bc_z"])
    np.testing.assert_allclose(radii, _expected_radius_px(tt), rtol=1e-9)


def test_tilted_ring_becomes_eccentric_under_tilt():
    """A non-zero tilt must actually change the projection — otherwise the
    'overlay offset from the real rings' bug this replaced would be back."""
    from midas_gui.helpers import tilted_ring_xy
    tt = 8.0
    Y0, Z0 = tilted_ring_xy(tt, 0.0, 0.0, 0.0, n=180, **_GEO)
    Y1, Z1 = tilted_ring_xy(tt, 0.0, 3.0, 0.0, n=180, **_GEO)
    assert not np.allclose(Y0, Y1, atol=1e-3) or not np.allclose(Z0, Z1, atol=1e-3)
    # ...and the tilted ring is no longer a constant-radius circle.
    radii = np.hypot(Y1 - _GEO["bc_y"], Z1 - _GEO["bc_z"])
    assert radii.ptp() > 1e-3


def test_tilted_ring_returns_n_points():
    from midas_gui.helpers import tilted_ring_xy
    Y, Z = tilted_ring_xy(6.0, 1.0, 2.0, 3.0, n=57, **_GEO)
    assert Y.shape == Z.shape == (57,)


def test_tilted_spoke_is_radial_at_zero_tilt():
    """A fixed-eta spoke at zero tilt is a straight radial line through the
    beam centre, sampled between the two 2θ limits."""
    from midas_gui.helpers import tilted_spoke_xy
    eta, n = 30.0, 24
    Y, Z = tilted_spoke_xy(2.0, 10.0, eta, 0.0, 0.0, 0.0, n=n, **_GEO)
    assert Y.shape == Z.shape == (n,)

    r = np.hypot(Y - _GEO["bc_y"], Z - _GEO["bc_z"])
    np.testing.assert_allclose(r[0], _expected_radius_px(2.0), rtol=1e-9)
    np.testing.assert_allclose(r[-1], _expected_radius_px(10.0), rtol=1e-9)
    assert np.all(np.diff(r) > 0), "must run outward monotonically"

    # Every sample sits on the same azimuth.
    ang = np.degrees(np.arctan2(Y - _GEO["bc_y"], Z - _GEO["bc_z"]))
    np.testing.assert_allclose(ang, eta, atol=1e-6)


def test_tilted_spoke_endpoints_match_the_ring_at_the_same_eta():
    """Spoke and ring are two slices of one projection, so where they meet
    they must agree exactly — the property that keeps a bin-grid overlay's
    cells closed."""
    from midas_gui.helpers import tilted_ring_xy, tilted_spoke_xy
    tx, ty, tz, tt = 0.5, 2.0, -1.0, 7.0
    n = 360
    Yr, Zr = tilted_ring_xy(tt, tx, ty, tz, n=n, **_GEO)
    idx = 45                                  # some eta on the closed ring grid
    eta = np.linspace(0.0, 360.0, n, endpoint=True)[idx]
    Ys, Zs = tilted_spoke_xy(tt, tt, eta, tx, ty, tz, n=2, **_GEO)
    np.testing.assert_allclose([Ys[0], Zs[0]], [Yr[idx], Zr[idx]], atol=1e-9)
def test_simulate_rings_from_dspacings_matches_braggs_law():
    import math
    from midas_gui.helpers import simulate_rings_from_dspacings
    d = 58.380
    wavelength_A = 0.1729
    lsd_um, px_um = 200000.0, 200.0
    rings = simulate_rings_from_dspacings([d], wavelength_A, lsd_um, px_um, max_2theta_deg=30.0)
    assert len(rings) == 1
    r = rings[0]
    expected_two_theta = 2.0 * math.degrees(math.asin(wavelength_A / (2.0 * d)))
    assert r["two_theta_deg"] == pytest.approx(expected_two_theta)
    assert r["d_spacing"] == pytest.approx(d)
    assert r["hkl"] is None
    assert r["order"] == 1
    expected_radius = lsd_um * math.tan(math.radians(expected_two_theta)) / px_um
    assert r["radius_px"] == pytest.approx(expected_radius)


def test_simulate_rings_from_dspacings_drops_orders_past_max_two_theta():
    from midas_gui.helpers import simulate_rings_from_dspacings
    d_list = [58.380 / n for n in range(1, 11)]
    rings = simulate_rings_from_dspacings(d_list, 0.1729, 200000.0, 200.0, max_2theta_deg=1.0)
    assert rings
    assert all(r["two_theta_deg"] <= 1.0 for r in rings)
    assert len(rings) < len(d_list)


def test_simulate_rings_from_dspacings_skips_non_positive_d():
    from midas_gui.helpers import simulate_rings_from_dspacings
    rings = simulate_rings_from_dspacings([58.380, 0.0, -1.0], 0.1729, 200000.0, 200.0)
    assert len(rings) == 1
    assert rings[0]["d_spacing"] == pytest.approx(58.380)


def test_coerce_material_dspacing_valid():
    from midas_gui.constants import _coerce_material
    m = _coerce_material({"kind": "dspacing", "d_list": [58.38, "29.19"]})
    assert m == {"kind": "dspacing", "d_list": [58.38, 29.19]}


def test_coerce_material_dspacing_empty_list_raises():
    from midas_gui.constants import _coerce_material
    with pytest.raises(ValueError):
        _coerce_material({"kind": "dspacing", "d_list": []})


# ── Manual d-spacing ring-picking calibration (Calibrate tab) ────────────────

def test_parse_dspacing_text_drops_blank_and_invalid_tokens():
    from midas_gui.helpers import parse_dspacing_text
    assert parse_dspacing_text("58.38, 29.19  19.46 0 -1 abc") == [58.38, 29.19, 19.46]


def test_parse_dspacing_text_empty_string():
    from midas_gui.helpers import parse_dspacing_text
    assert parse_dspacing_text("   ") == []


def test_fit_circle_algebraic_recovers_known_circle():
    from midas_gui.helpers import fit_circle_algebraic
    cx0, cy0, r0 = 123.4, -50.0, 200.0
    angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    pts = [(cx0 + r0 * math.cos(a), cy0 + r0 * math.sin(a)) for a in angles]
    cx, cy, r = fit_circle_algebraic(pts)
    assert (cx, cy, r) == pytest.approx((cx0, cy0, r0))


def test_fit_circle_algebraic_returns_none_for_too_few_points():
    from midas_gui.helpers import fit_circle_algebraic
    assert fit_circle_algebraic([(0, 0), (1, 1)]) is None


def test_fit_geometry_from_ring_picks_recovers_known_geometry():
    from midas_gui.helpers import fit_geometry_from_ring_picks, simulate_rings_from_dspacings

    wavelength_A = 0.1729
    px_um = 200.0
    lsd_um, bc_y, bc_z = 300000.0, 512.3, 498.7
    d_list = [58.380, 29.190, 19.460]

    rings = simulate_rings_from_dspacings(d_list, wavelength_A, lsd_um, px_um)
    rng = np.random.default_rng(0)
    picks = []
    for ring in rings:
        r_px = ring["radius_px"]
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            y = bc_y - r_px * math.cos(angle)
            z = bc_z + r_px * math.sin(angle)
            picks.append((y, z, ring["d_spacing"]))

    fit = fit_geometry_from_ring_picks(picks, wavelength_A, px_um, px_um)
    assert fit["success"]
    assert fit["Lsd"] == pytest.approx(lsd_um, rel=1e-4)
    assert fit["BC_y"] == pytest.approx(bc_y, abs=0.05)
    assert fit["BC_z"] == pytest.approx(bc_z, abs=0.05)
    assert fit["residual_deg_rms"] < 1e-3


def test_fit_geometry_from_ring_picks_refine_none_matches_legacy_lsd_bc_only():
    from midas_gui.helpers import fit_geometry_from_ring_picks, simulate_rings_from_dspacings

    wavelength_A = 0.1729
    px_um = 200.0
    lsd_um, bc_y, bc_z = 300000.0, 512.3, 498.7
    d_list = [58.380, 29.190, 19.460]

    rings = simulate_rings_from_dspacings(d_list, wavelength_A, lsd_um, px_um)
    picks = []
    for ring in rings:
        r_px = ring["radius_px"]
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            y = bc_y - r_px * math.cos(angle)
            z = bc_z + r_px * math.sin(angle)
            picks.append((y, z, ring["d_spacing"]))

    legacy = fit_geometry_from_ring_picks(picks, wavelength_A, px_um, px_um)
    explicit = fit_geometry_from_ring_picks(
        picks, wavelength_A, px_um, px_um,
        refine={"Lsd": True, "BC": True, "tx": False, "ty": False,
                "tz": False, "Wavelength": False})

    assert legacy["tx"] == pytest.approx(0.0)
    assert legacy["ty"] == pytest.approx(0.0)
    assert legacy["tz"] == pytest.approx(0.0)
    assert legacy["wavelength_A"] == pytest.approx(wavelength_A)
    assert legacy["n_free"] == 3
    for key in ("Lsd", "BC_y", "BC_z", "tx", "ty", "tz", "wavelength_A",
                "residual_deg_rms", "n_free"):
        assert legacy[key] == pytest.approx(explicit[key])


def test_fit_geometry_from_ring_picks_recovers_tilt_with_fixed_tx():
    from midas_gui.helpers import fit_geometry_from_ring_picks, tilted_ring_xy

    # A short wavelength / large-d combo (as in the AgBH test above) puts
    # every ring at a tiny 2theta, where tilt's effect on ring shape is
    # second-order and numerically degenerate with Lsd/BC — not a realistic
    # stand-in for a tilt-refinement scenario. Use d-spacings/wavelength
    # that spread 2theta across ~20-60 degrees so tilt is well identified.
    wavelength_A = 1.0
    px_um = 200.0
    lsd_um, bc_y, bc_z = 300000.0, 512.3, 498.7
    tx_true, ty_true, tz_true = 0.0, 1.7, -0.9
    d_list = [3.0, 1.5, 1.0]

    picks = []
    for d in d_list:
        s = wavelength_A / (2.0 * d)
        two_theta = 2.0 * math.degrees(math.asin(s))
        Y_px, Z_px = tilted_ring_xy(two_theta, tx_true, ty_true, tz_true,
                                     lsd_um, bc_y, bc_z, px_um, px_um, n=16)
        for y, z in zip(Y_px[:-1], Z_px[:-1]):
            picks.append((float(y), float(z), d))

    # tx is held fixed at its (correct) seed of 0.0 — only the free ty/tz
    # get a deliberately wrong seed, to make sure the fit actually moves
    # them rather than just reporting the seed back.
    wrong_seed_tilt = (0.0, 0.5, 0.5)
    refine = {"Lsd": True, "BC": True, "tx": False, "ty": True, "tz": True,
              "Wavelength": False}
    fit = fit_geometry_from_ring_picks(
        picks, wavelength_A, px_um, px_um,
        tilt_seed=wrong_seed_tilt, refine=refine)

    assert fit["success"]
    assert fit["n_free"] == 5
    assert fit["Lsd"] == pytest.approx(lsd_um, rel=1e-3)
    assert fit["BC_y"] == pytest.approx(bc_y, abs=0.1)
    assert fit["BC_z"] == pytest.approx(bc_z, abs=0.1)
    assert fit["ty"] == pytest.approx(ty_true, abs=1e-2)
    assert fit["tz"] == pytest.approx(tz_true, abs=1e-2)
    # tx was not selected to refine — it must stay pinned exactly at its
    # seed value, never nudged toward compensating for ty/tz residuals.
    assert fit["tx"] == pytest.approx(wrong_seed_tilt[0])
    assert fit["residual_deg_rms"] < 1e-3


def test_auto_seed_from_picks_falls_back_when_no_ring_has_enough_points():
    from midas_gui.helpers import _auto_seed_from_picks
    picks = [(10.0, 20.0, 58.38), (30.0, 40.0, 29.19)]   # 1 pt per ring, can't circle-fit
    lsd, bc_y, bc_z, quality = _auto_seed_from_picks(picks, 0.1729, 200.0, 200.0)
    assert quality == "fallback"
    assert bc_y == pytest.approx(20.0)
    assert bc_z == pytest.approx(30.0)


def test_predict_ring_radii_uses_d_list_branch_not_crystalline_fallback():
    from types import SimpleNamespace
    from midas_gui.helpers import _predict_ring_radii, simulate_rings_from_dspacings

    d_list = [58.380, 29.190]
    wavelength_A, lsd_um, px_um = 0.1729, 300000.0, 200.0
    result = SimpleNamespace(
        _d_list=d_list, wavelength_A=wavelength_A, Lsd=lsd_um, pxY=px_um,
        _calibrant_name="AgBH (silver behenate)")

    radii = _predict_ring_radii(result)
    expected = sorted({round(r["radius_px"], 3)
                       for r in simulate_rings_from_dspacings(d_list, wavelength_A, lsd_um, px_um)})
    assert radii == expected


def test_predict_ring_radii_uses_detector_coverage_not_fixed_30deg():
    """A short-Lsd / wide-detector geometry reaches well past the old
    hardcoded 30 deg cutoff — a ring out there must still be predicted,
    since it is a real ring that lies within the frame."""
    from types import SimpleNamespace
    from midas_gui.helpers import _predict_ring_radii, max_two_theta_deg

    wavelength_A, lsd_um, px_um = 0.1729, 100000.0, 200.0
    ny = nz = 2000
    bc_y = bc_z = 999.5  # centered beam

    two_theta_target_deg = 45.0
    d = wavelength_A / (2.0 * math.sin(math.radians(two_theta_target_deg / 2.0)))

    coverage = max_two_theta_deg(bc_y, bc_z, ny, nz, lsd_um, px_um)
    assert coverage > two_theta_target_deg > 30.0  # this geometry really reaches past 30 deg

    result = SimpleNamespace(
        _d_list=[d], wavelength_A=wavelength_A, Lsd=lsd_um, pxY=px_um, pxZ=px_um,
        BC_y=bc_y, BC_z=bc_z, NrPixelsY=ny, NrPixelsZ=nz, _calibrant_name="custom")

    radii = _predict_ring_radii(result)
    expected_radius_px = round(lsd_um * math.tan(math.radians(two_theta_target_deg)) / px_um, 3)
    assert radii == [expected_radius_px]


def test_apply_field_corrections_skips_mismatched_shape_instead_of_raising():
    """Regression test for a crash hit loading a saved session whose dark/
    bright/background paths were computed against a different detector
    (e.g. a 2880x2880 WAXS dark reused with a 512x3072 SAXS image): a plain
    ``out - d`` used to raise a numpy broadcast ValueError. Mismatched
    fields are now skipped (treated as None) with a warning, mirroring
    MaskSelector.composite_mask()'s "skip + warn" handling of a mismatched
    mask source."""
    import warnings
    from midas_gui.helpers import apply_field_corrections

    img = np.full((512, 3072), 10.0)
    stale_dark = np.ones((2880, 2880))
    matching_dark = np.full((512, 3072), 1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = apply_field_corrections(img, dark=stale_dark)
    assert np.array_equal(out, img)   # mismatched dark skipped, image unchanged
    assert any("dark" in str(w.message) and "skipped" in str(w.message) for w in caught)

    out2 = apply_field_corrections(img, dark=matching_dark)
    assert np.allclose(out2, 9.0)   # matching-shape dark still applies normally


# ── browse_start_dir / warn_if_path_missing ──────────────────────────────

def test_browse_start_dir_existing_directory_is_used_directly(tmp_path):
    from midas_gui.helpers import browse_start_dir

    assert browse_start_dir(str(tmp_path)) == str(tmp_path)


def test_browse_start_dir_existing_file_falls_back_to_its_parent(tmp_path):
    from midas_gui.helpers import browse_start_dir

    f = tmp_path / "data.h5"
    f.write_text("x")
    assert browse_start_dir(str(f)) == str(tmp_path)


def test_browse_start_dir_missing_leaf_falls_back_to_existing_parent(tmp_path):
    from midas_gui.helpers import browse_start_dir

    assert browse_start_dir(str(tmp_path / "not_there.h5")) == str(tmp_path)


def test_browse_start_dir_nothing_existing_returns_fallback(tmp_path):
    from midas_gui.helpers import browse_start_dir

    bogus = tmp_path / "nope" / "also_nope" / "x.h5"
    assert browse_start_dir(str(bogus), fallback="/some/default") == "/some/default"


def test_browse_start_dir_empty_text_returns_fallback():
    from midas_gui.helpers import browse_start_dir

    assert browse_start_dir("", fallback="/some/default") == "/some/default"
    assert browse_start_dir("   ") == ""


def test_warn_if_path_missing_ignores_empty_text(app, monkeypatch):
    from PyQt5 import QtWidgets
    from midas_gui.helpers import warn_if_path_missing

    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: calls.append(("warning", a)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                         lambda *a, **k: calls.append(("information", a)))

    edit = QtWidgets.QLineEdit()
    parent = QtWidgets.QWidget()
    warn_if_path_missing(edit, parent)
    edit.setText("")
    edit.returnPressed.emit()
    assert calls == []


def test_warn_if_path_missing_says_nothing_for_an_existing_path(app, monkeypatch, tmp_path):
    from PyQt5 import QtWidgets
    from midas_gui.helpers import warn_if_path_missing

    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: calls.append(a))

    edit = QtWidgets.QLineEdit(str(tmp_path))
    parent = QtWidgets.QWidget()
    warn_if_path_missing(edit, parent)
    edit.returnPressed.emit()
    assert calls == []


def test_warn_if_path_missing_warns_for_an_input_field(app, monkeypatch, tmp_path):
    from PyQt5 import QtWidgets
    from midas_gui.helpers import warn_if_path_missing

    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: calls.append(a))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                         lambda *a, **k: (_ for _ in ()).throw(
                             AssertionError("should warn, not inform")))

    missing = tmp_path / "nope.h5"
    edit = QtWidgets.QLineEdit(str(missing))
    parent = QtWidgets.QWidget()
    warn_if_path_missing(edit, parent, is_output_dir=False)
    edit.returnPressed.emit()
    assert len(calls) == 1
    assert str(missing) in calls[0][-1]


def test_warn_if_path_missing_informs_for_an_output_dir(app, monkeypatch, tmp_path):
    from PyQt5 import QtWidgets
    from midas_gui.helpers import warn_if_path_missing

    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                         lambda *a, **k: calls.append(a))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: (_ for _ in ()).throw(
                             AssertionError("should inform, not warn")))

    missing = tmp_path / "new_output_dir"
    edit = QtWidgets.QLineEdit(str(missing))
    parent = QtWidgets.QWidget()
    warn_if_path_missing(edit, parent, is_output_dir=True)
    edit.returnPressed.emit()
    assert len(calls) == 1


def test_warn_if_path_missing_does_not_shadow_an_existing_return_pressed_handler(app, monkeypatch, tmp_path):
    from PyQt5 import QtWidgets
    from midas_gui.helpers import warn_if_path_missing

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)

    edit = QtWidgets.QLineEdit(str(tmp_path / "missing"))
    seen = []
    edit.returnPressed.connect(lambda: seen.append(True))
    parent = QtWidgets.QWidget()
    warn_if_path_missing(edit, parent)
    edit.returnPressed.emit()
    assert seen == [True]


def test_path_is_missing_returns_a_guard_flag_for_a_handlers_own_return_pressed(app, monkeypatch, tmp_path):
    # path_is_missing() is the building block a field with its own Enter
    # handler (e.g. DataLoaderPanel._load) uses as an early-return guard, so
    # a missing path shows only this one friendly dialog instead of stacking
    # a second, less friendly error from the handler's own load attempt.
    from PyQt5 import QtWidgets
    from midas_gui.helpers import path_is_missing

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    parent = QtWidgets.QWidget()

    ok_edit = QtWidgets.QLineEdit(str(tmp_path))
    assert path_is_missing(ok_edit, parent) is False

    missing_edit = QtWidgets.QLineEdit(str(tmp_path / "missing"))
    assert path_is_missing(missing_edit, parent) is True
