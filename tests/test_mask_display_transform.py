"""Mask Builder: the displayed image follows the Data Viewer's flips, while
the mask it produces stays in raw detector space.

The load-bearing guarantee is the last one: Calibrate/Batch/Integrate
pre-flip the mask themselves before handing it to the backend (DECISIONS
2026-08-25), so a display-only change must leave `maskReady`'s payload
byte-identical. Everything else here exists to keep what is *drawn* over the
image anchored to the same detector pixels while the picture moves.

`map_roi_state` is checked by rasterisation equivalence rather than by
re-deriving its arithmetic: map the ROI, rasterise it, and compare against
`_apply_im_trans` of the original raster. That compares the thing that
actually matters — which pixels the shape covers.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked

from midas_gui.helpers import _apply_im_trans, map_point_xy, map_roi_state

# Every ordered subset of (flipY, flipZ, transpose) — the fixed composition
# order im_trans_codes_from_checkboxes emits.
ALL_CODES = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

# Non-square on purpose: a transpose that fails to swap the extent only
# shows up when the two differ.
RAW_SHAPE = (70, 50)          # (n_rows, n_cols)


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _raster(roi, shape):
    """Rasterise an ROI into a bool array of `shape` — the same path
    tab_mask._raster_roi uses, minus the viewer."""
    from PyQt5 import QtGui, QtCore
    n_rows, n_cols = shape
    path = roi.shape()
    path = roi.mapToParent(path)
    qimg = QtGui.QImage(n_cols, n_rows, QtGui.QImage.Format_Grayscale8)
    qimg.fill(0)
    painter = QtGui.QPainter(qimg)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(255, 255, 255))
    painter.drawPath(path)
    painter.end()
    bpl = qimg.bytesPerLine()
    ptr = qimg.constBits(); ptr.setsize(n_rows * bpl)
    return np.frombuffer(ptr, np.uint8).reshape(n_rows, bpl)[:, :n_cols] > 127


def _assert_same_region(got, expected, label):
    """Assert two rasters cover the same region.

    Rendering the same shape twice through Qt is not bit-exact along a
    diagonal edge: a reflection reverses the polygon's winding, and Qt's
    scanline fill breaks ties differently, which moves a 45°-rotated
    rectangle's apparent centroid by half a pixel without moving the shape
    at all. So coverage is checked structurally — each raster eroded by one
    pixel must sit inside the other — and *placement* is pinned exactly by
    test_mapped_roi_corners_match_the_point_map, which compares geometry and
    is immune to rasterisation entirely.
    """
    from scipy.ndimage import binary_erosion
    assert expected.any() and got.any(), f"{label}: empty raster"
    assert binary_erosion(expected)[~got].sum() == 0, \
        f"{label}: expected interior not covered"
    assert binary_erosion(got)[~expected].sum() == 0, \
        f"{label}: got interior outside expected"


def _disp_shape(shape, codes):
    n_rows, n_cols = shape
    if sum(1 for c in codes if c == 3) % 2:
        return (n_cols, n_rows)
    return (n_rows, n_cols)


# ── map_point_xy ─────────────────────────────────────────────────────────

def test_map_point_xy_is_identity_without_codes():
    assert map_point_xy(3.5, 9.25, RAW_SHAPE, ())[:2] == (3.5, 9.25)


def test_map_point_xy_reports_the_swapped_extent_after_a_transpose():
    _x, _y, n_cols, n_rows = map_point_xy(1.0, 2.0, RAW_SHAPE, (3,))
    assert (n_rows, n_cols) == (50, 70)


def test_map_point_xy_uses_edge_not_index_arithmetic():
    """A flip sends the continuous coordinate x to W - x. Using W - 1 - x
    (the whole-pixel index form) would shift every shape half a pixel."""
    x, _y, _w, _h = map_point_xy(0.0, 0.0, RAW_SHAPE, (1,))
    assert x == 50.0            # n_cols, not n_cols - 1


# ── map_roi_state, by rasterisation ──────────────────────────────────────

def _roi_cases(pg):
    """(label, factory) for each ROI kind the Mask Builder can create."""
    return [
        ("rect", lambda pos, size, ang: pg.RectROI(pos, size, angle=ang,
                                                   rotatable=True)),
        ("ellipse", lambda pos, size, ang: pg.EllipseROI(pos, size, angle=ang,
                                                         rotatable=True)),
        ("circle", lambda pos, size, _a: pg.CircleROI(pos, size)),
    ]


@pytest.mark.parametrize("codes", ALL_CODES)
@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, -45.0])
@pytest.mark.parametrize("kind", ["rect", "ellipse", "circle"])
def test_mapped_roi_covers_the_same_detector_pixels(app, codes, angle, kind):
    """The whole point of anchoring shapes to pixels: rasterise, transform
    the raster, and the mapped ROI must land on exactly that."""
    import pyqtgraph as pg
    factory = dict((k, f) for k, f in _roi_cases(pg))[kind]
    pos, size = (12.0, 18.0), (20.0, 12.0)
    if kind == "circle":
        size = (16.0, 16.0)             # CircleROI is square and unrotatable
        angle = 0.0

    before = factory(pos, size, angle)
    raw_raster = _raster(before, RAW_SHAPE)
    expected = _apply_im_trans(raw_raster.astype(np.uint8), codes).astype(bool)

    new_pos, new_size, new_angle = map_roi_state(pos, size, angle,
                                                 RAW_SHAPE, codes)
    after = factory(tuple(new_pos), tuple(new_size), new_angle)
    after.setAngle(new_angle)
    got = _raster(after, _disp_shape(RAW_SHAPE, codes))

    assert got.shape == expected.shape
    _assert_same_region(got, expected, f"{kind} angle={angle} codes={codes}")


@pytest.mark.parametrize("codes", ALL_CODES)
def test_mapped_roi_preserves_size(codes):
    """Every MIDAS transform is orthogonal, so the extent cannot change —
    not even across a transpose."""
    _p, size, _a = map_roi_state((5.0, 7.0), (20.0, 11.0), 23.0,
                                 RAW_SHAPE, codes)
    assert size == [20.0, 11.0]


@pytest.mark.parametrize("codes", ALL_CODES)
def test_mapping_round_trips_back_to_the_original(codes):
    """Reversing the code order is the inverse (each op is self-inverse), so
    mapping out and back must return the original placement."""
    pos, size, ang = (12.0, 18.0), (20.0, 12.0), 30.0
    p1, s1, a1 = map_roi_state(pos, size, ang, RAW_SHAPE, codes)
    disp = _disp_shape(RAW_SHAPE, codes)
    p2, s2, a2 = map_roi_state(p1, s1, a1, disp, tuple(reversed(codes)))
    assert p2 == pytest.approx(list(pos), abs=1e-6)
    assert s2 == pytest.approx(list(size), abs=1e-6)
    assert (a2 - ang + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-6)


# ── map_roi_state, exactly, by geometry ──────────────────────────────────

def _corners(roi, size):
    """The ROI's four corners in parent (image) coordinates, via Qt's own
    transform — no assumption about how pyqtgraph composes pos/size/angle."""
    from PyQt5 import QtCore
    w, h = float(size[0]), float(size[1])
    out = []
    for ux, uy in [(0, 0), (w, 0), (w, h), (0, h)]:
        p = roi.mapToParent(QtCore.QPointF(ux, uy))
        out.append((round(p.x(), 6), round(p.y(), 6)))
    return sorted(out)


@pytest.mark.parametrize("codes", ALL_CODES)
@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, -45.0])
@pytest.mark.parametrize("kind", ["rect", "ellipse", "circle"])
def test_mapped_roi_corners_match_the_point_map(app, codes, angle, kind):
    """The exact placement proof, independent of any rasteriser.

    The mapped ROI must occupy precisely the four corners you get by
    pushing the original corners through map_point_xy — same set, to
    floating-point. Rasterisation can dither an outline; this cannot.
    """
    import pyqtgraph as pg
    factory = dict((k, f) for k, f in _roi_cases(pg))[kind]
    pos, size = (12.0, 18.0), (20.0, 12.0)
    if kind == "circle":
        size, angle = (16.0, 16.0), 0.0

    before = factory(pos, size, angle)
    before.setAngle(angle)
    want = sorted((round(map_point_xy(x, y, RAW_SHAPE, codes)[0], 6),
                   round(map_point_xy(x, y, RAW_SHAPE, codes)[1], 6))
                  for x, y in _corners(before, size))

    new_pos, new_size, new_angle = map_roi_state(pos, size, angle,
                                                 RAW_SHAPE, codes)
    after = factory(tuple(new_pos), tuple(new_size), new_angle)
    after.setAngle(new_angle)
    got = _corners(after, new_size)

    assert got == pytest.approx(want, abs=1e-6), (
        f"{kind} angle={angle} codes={codes}")


@pytest.mark.parametrize("codes", ALL_CODES)
def test_mapped_roi_stays_inside_the_transformed_frame(codes):
    """A shape on the detector must still be on the detector afterwards —
    the check that catches a transpose that forgot to swap the extent."""
    pos, size, ang = (12.0, 18.0), (20.0, 12.0), 30.0
    new_pos, new_size, new_angle = map_roi_state(pos, size, ang,
                                                 RAW_SHAPE, codes)
    n_rows, n_cols = _disp_shape(RAW_SHAPE, codes)
    import math
    a = math.radians(new_angle)
    xs, ys = [], []
    for ux, uy in [(0, 0), (new_size[0], 0), (new_size[0], new_size[1]),
                   (0, new_size[1])]:
        xs.append(new_pos[0] + ux * math.cos(a) - uy * math.sin(a))
        ys.append(new_pos[1] + ux * math.sin(a) + uy * math.cos(a))
    assert min(xs) >= -1e-6 and max(xs) <= n_cols + 1e-6, f"{codes}: x {min(xs)}..{max(xs)} vs {n_cols}"
    assert min(ys) >= -1e-6 and max(ys) <= n_rows + 1e-6, f"{codes}: y {min(ys)}..{max(ys)} vs {n_rows}"


@pytest.mark.parametrize("codes", ALL_CODES)
def test_polygon_vertices_map_pointwise(codes):
    """PolyLineROI carries explicit vertices, so it maps with map_point_xy
    directly — no pos/size/angle involved."""
    pts = [(10.0, 12.0), (30.0, 14.0), (22.0, 40.0), (8.0, 33.0)]
    mapped = [map_point_xy(x, y, RAW_SHAPE, codes)[:2] for x, y in pts]
    n_rows, n_cols = _disp_shape(RAW_SHAPE, codes)
    for x, y in mapped:
        assert 0.0 <= x <= n_cols and 0.0 <= y <= n_rows
    back_shape = (n_rows, n_cols)
    back = [map_point_xy(x, y, back_shape, tuple(reversed(codes)))[:2]
            for x, y in mapped]
    assert back == pytest.approx(pts, abs=1e-9)


# ── MaskTab: display follows, mask stays raw ─────────────────────────────

def _tab_with_image(shape=RAW_SHAPE, seed=0):
    """A MaskTab holding a raw frame, without going through file loading."""
    from midas_gui.tab_mask import MaskTab
    tab = MaskTab()
    img = np.random.default_rng(seed).random(shape).astype(np.float32)
    tab._image = img
    return tab, img


@pytest.mark.parametrize("codes", ALL_CODES)
def test_display_shape_tracks_the_transpose(app, codes):
    tab, _img = _tab_with_image()
    tab.set_display_transform(codes)
    assert tab._disp_shape() == _disp_shape(RAW_SHAPE, codes)


@pytest.mark.parametrize("codes", ALL_CODES)
def test_emitted_mask_is_identical_whatever_the_display_transform(app, codes):
    """The load-bearing guarantee. Downstream pre-flips this mask itself, so
    a display-only change must not alter one bit of it."""
    emitted = []
    tab, img = _tab_with_image()
    tab.maskReady.connect(lambda m: emitted.append(np.array(m)))

    computed = np.zeros(RAW_SHAPE, dtype=bool)
    computed[5:9, 3:30] = True                 # an asymmetric "module gap"
    tab.set_display_transform(codes)
    tab._set_mask(computed)

    assert emitted, "maskReady never fired"
    assert emitted[-1].shape == RAW_SHAPE, "emitted mask left raw space"
    np.testing.assert_array_equal(emitted[-1].astype(bool), computed)


@pytest.mark.parametrize("codes", ALL_CODES)
def test_overlay_is_painted_in_the_displayed_frame(app, codes):
    """The mask stays raw; the picture of it must follow the image."""
    seen = {}
    tab, _img = _tab_with_image()
    tab._viewer.set_mask_overlay = lambda m: seen.update(m=np.array(m))

    computed = np.zeros(RAW_SHAPE, dtype=bool)
    computed[5:9, 3:30] = True
    tab.set_display_transform(codes)
    tab._set_mask(computed)

    expected = _apply_im_trans(computed.astype(np.uint8), codes).astype(bool)
    assert seen["m"].shape == _disp_shape(RAW_SHAPE, codes)
    np.testing.assert_array_equal(seen["m"].astype(bool), expected)
    # and the tab's own mask is untouched
    np.testing.assert_array_equal(tab._mask.astype(bool), computed)


@pytest.mark.parametrize("codes", [(1,), (2,), (3,), (1, 2, 3)])
def test_a_shape_drawn_after_a_flip_masks_the_pixels_it_covers(app, codes):
    """Draw in the transformed view; the raw mask must come back inverse-
    mapped, so it marks the detector pixels that are actually under the
    shape on screen."""
    import pyqtgraph as pg
    tab, _img = _tab_with_image()
    tab.set_display_transform(codes)

    n_rows, n_cols = tab._disp_shape()
    roi = pg.RectROI((4.0, 6.0), (10.0, 8.0), pen=None)
    tab._viewer._iv.addItem(roi)
    tab._shapes.append({"kind": "shape", "roi": roi})
    tab._apply_shapes()

    assert tab._drawn_mask.shape == RAW_SHAPE, "drawn mask left raw space"
    # Forward-transforming it must reproduce what the ROI covers on screen.
    shown = _apply_im_trans(tab._drawn_mask.astype(np.uint8), codes).astype(bool)
    direct = _raster(roi, (n_rows, n_cols))
    _assert_same_region(shown, direct, f"codes={codes}")


def test_changing_the_transform_keeps_a_shape_on_the_same_pixels(app):
    """The anchoring guarantee: flip the view, and an already-drawn ROI must
    still mask the same detector pixels."""
    import pyqtgraph as pg
    tab, _img = _tab_with_image()
    roi = pg.RectROI((8.0, 11.0), (14.0, 9.0), pen=None)
    tab._viewer._iv.addItem(roi)
    tab._shapes.append({"kind": "shape", "roi": roi})
    tab._apply_shapes()
    before = tab._drawn_mask.copy()

    tab.set_display_transform((1, 2))
    tab._apply_shapes()

    _assert_same_region(tab._drawn_mask, before, "after flip")


def test_picked_points_follow_the_transform(app):
    tab, _img = _tab_with_image()
    import pyqtgraph as pg
    tab._points = [(3, 9)]
    tab._point_items = [pg.ScatterPlotItem([3], [9])]
    tab._apply_shapes()
    before = tab._drawn_mask.copy()

    tab.set_display_transform((1,))
    tab._apply_shapes()

    np.testing.assert_array_equal(tab._drawn_mask, before)
