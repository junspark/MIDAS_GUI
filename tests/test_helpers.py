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
    # Pixirad is identified but has no known pixel size to auto-populate.
    assert detect_geometry_from_path("scan.pxrd", profile=profile) == {}
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
