"""Unit tests for the image viewers' 2θ / Q / d / η pixel readout.

These cover the pure, Qt-free half of the feature: the geometry maths in
``helpers.pixel_readout_text`` / ``pixel_eta_deg`` and the point-wise
transform mapping in ``helpers.im_trans_map_point`` that the Mask Builder
needs to carry a raw-frame hover into the calibration's frame.

The Qt-side wiring (``ImageViewer.set_radial_readout_fn``) is covered in
tests/test_viewer_origin_and_readout.py.
"""
import math

import numpy as np
import pytest

from midas_gui.helpers import (
    _apply_im_trans, _pixel_to_two_theta_deg, im_trans_map_point,
    pixel_eta_deg, pixel_readout_text, tilted_ring_xy)

# A plausible single-detector geometry: 2048², 200 µm pixels, Lsd 200 mm.
GEOM = {"Lsd": 200_000.0, "BC_y": 1024.0, "BC_z": 1024.0,
        "pxY": 200.0, "wavelength_A": 0.1729}


def _field(text, name):
    """Pull one '<name> = <value><unit>' field out of a readout string."""
    for part in text.split("    "):
        part = part.strip()
        if part.startswith(f"{name} ="):
            val = part.split("=", 1)[1].strip()
            # drop a trailing unit ("Å", "Å⁻¹"); the degree sign is glued on
            return val.split(" ")[0]
    return None


def _num(text, name):
    raw = _field(text, name)
    if raw is None or raw.startswith("—"):
        return None
    return float(raw.split()[0].rstrip("°"))


# ── 2θ ────────────────────────────────────────────────────────────────────

def test_two_theta_at_zero_tilt_is_the_plain_arctan():
    """With no tilt the readout must reduce to degrees(atan(r·px/Lsd))."""
    col, row = 1024 + 300, 1024 + 400           # r = 500 px exactly
    expected = math.degrees(math.atan(500 * 200.0 / 200_000.0))
    assert _num(pixel_readout_text(col, row, GEOM), "2θ") == pytest.approx(
        expected, abs=1e-4)


def test_two_theta_is_zero_at_the_beam_centre():
    assert _num(pixel_readout_text(1024, 1024, GEOM), "2θ") == pytest.approx(0.0)


def test_tilt_changes_two_theta():
    """A tilted geometry must not silently return the flat answer."""
    flat = _num(pixel_readout_text(1500, 1024, GEOM), "2θ")
    tilted = _num(pixel_readout_text(
        1500, 1024, {**GEOM, "ty": 4.0, "tz": -3.0}), "2θ")
    assert abs(tilted - flat) > 1e-3


@pytest.mark.parametrize("tx,ty,tz", [(0, 0, 0), (0, 5, 0), (0, 0, -4),
                                      (2, 3, -1.5)])
def test_two_theta_round_trips_against_the_forward_ring_projection(tx, ty, tz):
    """Forward-project a ring at a known 2θ through the tilt geometry, then
    read each of its pixels back — every one must return that same 2θ.

    This is the real correctness proof: it ties the readout to the exact
    projection the ring overlays are drawn with, at non-zero tilt.
    """
    tt0 = 7.5
    Y, Z = tilted_ring_xy(tt0, tx, ty, tz, GEOM["Lsd"], GEOM["BC_y"],
                          GEOM["BC_z"], GEOM["pxY"], GEOM["pxY"], n=60)
    geom = {**GEOM, "tx": tx, "ty": ty, "tz": tz}
    back = _pixel_to_two_theta_deg(
        Y, Z, GEOM["Lsd"], GEOM["BC_y"], GEOM["BC_z"], tx, ty, tz,
        GEOM["pxY"], GEOM["pxY"])
    assert np.allclose(back, tt0, atol=1e-6)
    # and through the formatted readout, at one point on that ring
    assert _num(pixel_readout_text(float(Y[0]), float(Z[0]), geom),
                "2θ") == pytest.approx(tt0, abs=1e-3)


# ── η ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dcol,drow,expected", [
    (0,  300,    0.0),      # +Z, straight up
    (300,   0,   90.0),     # +Y
    (0, -300,  180.0),      # -Z, straight down
    (-300,  0,  -90.0),     # -Y
])
def test_eta_cardinal_directions(dcol, drow, expected):
    """η = 0 is straight up (+Z) and increases towards +Y — the backend's
    pixel_to_REta convention. Swapping sin/cos puts every value 90° out,
    which is exactly the failure this pins.

    Compared as an angle, not a number: straight down comes back as -180°
    rather than +180° (atan2 of a negative-signed zero), and the two name
    the same direction.
    """
    eta = float(pixel_eta_deg(1024 + dcol, 1024 + drow, 1024.0, 1024.0,
                              200.0, 200.0))
    diff = (eta - expected + 180.0) % 360.0 - 180.0
    assert diff == pytest.approx(0.0, abs=1e-9)


def test_eta_appears_in_the_readout():
    assert _num(pixel_readout_text(1024 + 300, 1024, GEOM), "η") == pytest.approx(
        90.0, abs=1e-6)


# ── Q and d ───────────────────────────────────────────────────────────────

def test_q_and_d_are_mutually_consistent():
    """d = 2π/Q for every pixel off the beam axis."""
    for col, row in [(1300, 1024), (1024, 1400), (1500, 1500)]:
        text = pixel_readout_text(col, row, GEOM)
        q, d = _num(text, "Q"), _num(text, "d")
        assert d == pytest.approx(2 * math.pi / q, rel=1e-3)


def test_q_matches_braggs_law():
    col, row = 1024 + 500, 1024
    tt = _num(pixel_readout_text(col, row, GEOM), "2θ")
    expected_q = 4 * math.pi * math.sin(math.radians(tt) / 2) / GEOM["wavelength_A"]
    assert _num(pixel_readout_text(col, row, GEOM), "Q") == pytest.approx(
        expected_q, rel=1e-3)


def test_d_is_an_em_dash_on_the_beam_axis():
    """d diverges as 2θ → 0; it must render as a dash, not inf or a crash."""
    text = pixel_readout_text(1024, 1024, GEOM)
    assert _field(text, "d") == "—"
    assert "inf" not in text


# ── Graceful degradation ──────────────────────────────────────────────────

def test_without_a_wavelength_two_theta_and_eta_still_show():
    geom = {k: v for k, v in GEOM.items() if k != "wavelength_A"}
    text = pixel_readout_text(1300, 1300, geom)
    assert _num(text, "2θ") is not None and _num(text, "η") is not None
    assert "Q =" not in text and "d =" not in text


@pytest.mark.parametrize("missing", ["Lsd", "BC_y", "BC_z", "pxY"])
def test_missing_placement_geometry_yields_no_clause(missing):
    """Without these the pixel cannot be placed at all — the readout must be
    empty so a viewer with no calibration renders exactly as it did before."""
    geom = {k: v for k, v in GEOM.items() if k != missing}
    assert pixel_readout_text(1300, 1300, geom) == ""


def test_empty_or_none_geometry_yields_no_clause():
    assert pixel_readout_text(10, 10, {}) == ""
    assert pixel_readout_text(10, 10, None) == ""


def test_nonpositive_lsd_or_pixel_size_yields_no_clause():
    assert pixel_readout_text(10, 10, {**GEOM, "Lsd": 0.0}) == ""
    assert pixel_readout_text(10, 10, {**GEOM, "pxY": 0.0}) == ""


def test_non_square_pixels_are_honoured():
    """pxZ differing from pxY must change the answer off the Z axis."""
    square = _num(pixel_readout_text(1024, 1424, GEOM), "2θ")
    tall = _num(pixel_readout_text(1024, 1424, {**GEOM, "pxZ": 400.0}), "2θ")
    assert tall > square


# ── im_trans_map_point (the Mask Builder's correctness proof) ─────────────

@pytest.mark.parametrize("codes", [
    (), (1,), (2,), (3,), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2),
    (1, 2, 3), (3, 2, 1), (2, 3, 1),
])
def test_im_trans_map_point_agrees_with_the_array_transform(codes):
    """The mapped point must index the same value in the transformed array
    that the original point indexed in the raw one — for every ordering of
    the transform codes, including the non-square case where a transpose
    swaps the shape."""
    raw = np.arange(7 * 5, dtype=float).reshape(7, 5)      # non-square on purpose
    xf = _apply_im_trans(raw, codes)
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            c2, r2 = im_trans_map_point(col, row, raw.shape, codes)
            assert xf[r2, c2] == raw[row, col], (
                f"codes={codes} ({col},{row}) -> ({c2},{r2})")


def test_im_trans_map_point_with_no_codes_is_identity():
    assert im_trans_map_point(3, 9, (20, 10), ()) == (3, 9)
    assert im_trans_map_point(3, 9, (20, 10), None) == (3, 9)


def test_im_trans_map_point_tracks_shape_through_a_transpose():
    """After a transpose the bounds swap; a later flip must use the new
    width/height, not the original."""
    raw = np.arange(4 * 6, dtype=float).reshape(4, 6)   # 4 rows, 6 cols
    for codes in [(3, 1), (3, 2)]:
        xf = _apply_im_trans(raw, codes)
        c2, r2 = im_trans_map_point(5, 3, raw.shape, codes)
        assert 0 <= r2 < xf.shape[0] and 0 <= c2 < xf.shape[1]
        assert xf[r2, c2] == raw[3, 5]


# ── Mask Builder: raw frame vs calibration frame ─────────────────────────

@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Result:
    """Minimal stand-in for an AutoCalibrationResult."""
    def __init__(self, **kw):
        self.Lsd = 200_000.0
        self.BC_y = 1024.0
        self.BC_z = 1024.0
        self.pxY = 200.0
        self.pxZ = 200.0
        self.tx = self.ty = self.tz = 0.0
        self.wavelength_A = 0.1729
        self.NrPixelsY = 2048
        self.NrPixelsZ = 2048
        self.im_trans = []
        self.__dict__.update(kw)


@pytest.mark.forked
def test_mask_readout_maps_the_raw_hover_into_the_calibration_frame(app):
    """The Mask Builder shows the RAW image while the calibration lives in
    the transformed frame. Hovering a raw pixel must report the 2θ of where
    that pixel actually sits in the geometry — not of the same (col, row)
    read naively against the transformed beam centre."""
    from midas_gui.tab_mask import MaskTab
    tab = MaskTab()
    tab._image = np.zeros((2048, 2048), dtype=np.float32)
    tab.set_calibration(_Result(im_trans=[1]))          # flipY

    col, row = 300, 1024
    got = _num(tab._radial_readout(col, row), "2θ")
    # flipY sends raw col 300 to transformed col 2048-1-300 = 1747
    expected = _num(pixel_readout_text(2048 - 1 - 300, 1024, GEOM), "2θ")
    assert got == pytest.approx(expected, abs=1e-9)
    # and that is genuinely different from ignoring the transform
    assert got != pytest.approx(_num(pixel_readout_text(300, 1024, GEOM), "2θ"),
                                abs=1e-6)


@pytest.mark.forked
def test_mask_readout_is_blank_when_the_detector_shape_disagrees(app):
    """A mask built on one detector with a calibration from another must
    show nothing rather than a plausible wrong number."""
    from midas_gui.tab_mask import MaskTab
    tab = MaskTab()
    tab._image = np.zeros((1024, 1024), dtype=np.float32)   # not 2048²
    tab.set_calibration(_Result())
    assert tab._radial_readout(500, 500) == ""


@pytest.mark.forked
def test_mask_readout_is_blank_without_a_calibration_or_image(app):
    from midas_gui.tab_mask import MaskTab
    tab = MaskTab()
    assert tab._radial_readout(10, 10) == ""
    tab._image = np.zeros((2048, 2048), dtype=np.float32)
    assert tab._radial_readout(10, 10) == ""


@pytest.mark.forked
def test_mask_readout_with_no_transform_needs_no_mapping(app):
    from midas_gui.tab_mask import MaskTab
    tab = MaskTab()
    tab._image = np.zeros((2048, 2048), dtype=np.float32)
    tab.set_calibration(_Result(im_trans=[]))
    assert _num(tab._radial_readout(1300, 1300), "2θ") == pytest.approx(
        _num(pixel_readout_text(1300, 1300, GEOM), "2θ"), abs=1e-9)
