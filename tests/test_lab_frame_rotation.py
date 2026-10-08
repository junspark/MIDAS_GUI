"""The frame is drawn as the detector is mounted, once tx is non-zero.

tx is a panel's installation azimuth about the beam (0-360 deg), not a small
tilt, so a rolled panel's array is a rolled picture of the lab. Drawn
unrotated it disagrees with the coordinate system every other number on
screen is quoted in -- reported from the beamline as "the display does not
match the actual coordinate system and how the detector is mounted".

The rotation is a DISPLAY transform: nothing stored, emitted or fitted
leaves panel (row, col) space. It is applied to the pyqtgraph ImageItem,
which is what lets the Data Viewer's ROI machinery keep working untouched --
``roi.getArrayRegion(data.T, imgitem)`` and ``_raster_roi_mask``'s
``imgitem.mapFromScene(...)`` both route through that item's transform.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import math
import numpy as np
import pytest

pytestmark = pytest.mark.forked

BCY, BCZ, LSD, PX = 1024.0, 1000.0, 2.39e6, 200.0


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ── the convention itself, against the backend's own projection ──────────

def test_the_display_map_is_the_inverse_of_the_panel_roll():
    """Not derived on paper: ``_tilt_project_YZ`` gives panel pixels for a
    ray, so the same ray at tx=0 gives the lab position of that panel point.
    The display map is whatever takes one to the other, and it must hold at
    every azimuth, not just the one that was eyeballed.
    """
    from midas_gui.helpers import _tilt_project_YZ
    tt = np.full(16, 5.0)
    eta = np.linspace(0, 337.5, 16)
    lab_Y, lab_Z = _tilt_project_YZ(tt, eta, 0.0, 0.0, 0.0, LSD, BCY, BCZ, PX, PX)
    for tx in (15.0, 30.0, 120.0, 210.0, 300.0):
        pan_Y, pan_Z = _tilt_project_YZ(tt, eta, tx, 0.0, 0.0,
                                         LSD, BCY, BCZ, PX, PX)
        c, s = math.cos(math.radians(tx)), math.sin(math.radians(tx))
        dY, dZ = pan_Y - BCY, pan_Z - BCZ
        assert np.allclose(BCY + dY * c + dZ * s, lab_Y, atol=1e-9), tx
        assert np.allclose(BCZ - dY * s + dZ * c, lab_Z, atol=1e-9), tx


def test_the_viewers_transform_implements_exactly_that_map(app):
    from PyQt5 import QtCore
    from midas_gui.widgets import ImageViewer
    v = ImageViewer()
    v.set_image(np.zeros((2048, 2048), dtype=float))
    for tx in (30.0, 120.0, 210.0):
        v.set_lab_rotation(tx, BCY, BCZ)
        tf = v.lab_transform()
        c, s = math.cos(math.radians(tx)), math.sin(math.radians(tx))
        for Y, Z in [(0, 0), (2047, 0), (0, 2047), (1500, 700), (BCY, BCZ)]:
            got = tf.map(QtCore.QPointF(Y, Z))
            dY, dZ = Y - BCY, Z - BCZ
            assert got.x() == pytest.approx(BCY + dY * c + dZ * s, abs=1e-6)
            assert got.y() == pytest.approx(BCZ - dY * s + dZ * c, abs=1e-6)


def test_the_beam_centre_is_the_one_point_that_cannot_move(app):
    """A roll about the beam leaves the beam where it is. If the centre
    drifted, every overlay anchored to it would be wrong by that drift."""
    from PyQt5 import QtCore
    from midas_gui.widgets import ImageViewer
    v = ImageViewer()
    v.set_image(np.zeros((512, 512), dtype=float))
    for tx in (1.0, 90.0, 180.0, 270.0, 359.0):
        v.set_lab_rotation(tx, BCY, BCZ)
        p = v.lab_transform().map(QtCore.QPointF(BCY, BCZ))
        assert p.x() == pytest.approx(BCY, abs=1e-6), tx
        assert p.y() == pytest.approx(BCZ, abs=1e-6), tx


# ── the unrolled detector must be provably untouched ─────────────────────

def test_an_unrolled_detector_gets_no_transform_at_all(app):
    """Every detector that has ever worked here is unrolled, so tx=0 has to
    be the identical code path -- not merely a rotation by zero."""
    from midas_gui.widgets import ImageViewer
    v = ImageViewer()
    v.set_image(np.zeros((256, 256), dtype=float))
    for tx in (0.0, 360.0, -360.0):
        v.set_lab_rotation(tx, BCY, BCZ)
        assert v.lab_transform() is None, tx
        assert v._iv.getImageItem().transform().isIdentity(), tx
    # ...and with no beam centre to rotate about there is nothing to do.
    v.set_lab_rotation(120.0, None, None)
    assert v.lab_transform() is None


# ── the input boundary: everything downstream stays in panel space ───────

def test_view_coordinates_come_back_as_panel_pixels(app):
    """The single inverse in the codebase. Picks and the pixel readout both
    go through it, so a pixel's identity never depends on the display."""
    from PyQt5 import QtCore
    from midas_gui.widgets import ImageViewer
    v = ImageViewer()
    v.set_image(np.zeros((2048, 2048), dtype=float))
    for tx in (0.0, 30.0, 210.0):
        v.set_lab_rotation(tx, BCY, BCZ)
        tf = v.lab_transform()
        for Y, Z in [(10.0, 20.0), (1500.0, 700.0), (BCY, BCZ)]:
            pt = QtCore.QPointF(Y, Z)
            lab = tf.map(pt) if tf is not None else pt
            back = v.panel_xy(lab)
            assert back[0] == pytest.approx(Y, abs=1e-6), (tx, Y, Z)
            assert back[1] == pytest.approx(Z, abs=1e-6), (tx, Y, Z)


def test_a_pick_on_a_rolled_frame_reports_the_pixel_under_the_cursor(app):
    """bcPicked feeds a fit that works in panel space throughout, so a click
    must name the panel pixel, not the place on screen it landed."""
    from PyQt5 import QtCore
    from midas_gui.widgets import PickableImageViewer
    v = PickableImageViewer()
    v.set_image(np.zeros((2048, 2048), dtype=float))
    v.set_lab_rotation(120.0, BCY, BCZ)
    got = []
    v.bcPicked.connect(lambda y, z: got.append((y, z)))
    # Where panel pixel (1500, 700) is drawn...
    lab = v.lab_transform().map(QtCore.QPointF(1500.0, 700.0))
    # ...is what a click there hands back, in panel coordinates.
    y, z = v.panel_xy(lab)
    v._set_bc_marker(y, z)
    v.bcPicked.emit(y, z)
    assert got and got[0][0] == pytest.approx(1500.0, abs=1e-6)
    assert got[0][1] == pytest.approx(700.0, abs=1e-6)
    # The marker is placed in panel space and carried over by the transform,
    # so it lands back under the click.
    assert v._bc_click_item.transform() == v.lab_transform()


def test_markers_already_on_screen_are_restamped_when_the_roll_changes(app):
    """Picks survive a tx edit, so they must move with the picture rather
    than stay behind on the pixels they were drawn at."""
    from midas_gui.widgets import PickableImageViewer
    v = PickableImageViewer()
    v.set_image(np.zeros((2048, 2048), dtype=float))
    v._set_bc_marker(1500.0, 700.0)
    for x, y in [(100.0, 100.0), (200.0, 300.0), (400.0, 150.0)]:
        v._add_ring_point(x, y)
    assert v._bc_click_item.transform().isIdentity()

    v.set_lab_rotation(120.0, BCY, BCZ)
    tf = v.lab_transform()
    assert v._bc_click_item.transform() == tf
    assert all(it.transform() == tf for it in v._ring_pt_items)
    assert v._ring_fit_item is not None
    assert v._ring_fit_item.transform() == tf


# ── the two tabs ─────────────────────────────────────────────────────────

def _calib_tab():
    from midas_gui.tab_calibrate import CalibrationTab
    t = CalibrationTab()
    t._cal.setCurrentText("CeO2")
    t._wl.setValue(0.1729); t._pxY.setValue(PX)
    t._image = np.zeros((2048, 2048), dtype=float)
    for cb in t._seed_enables:
        cb.setChecked(True)
    t._seed_bcy.setValue(BCY); t._seed_bcz.setValue(BCZ)
    t._seed_lsd.setValue(2390.0)
    return t


def test_calibrate_rings_are_panel_space_and_so_must_rotate(app):
    """_ring_curves projects through ring_xy_corrected with the REAL tx, so
    its output is in the detector panel's own basis. Leaving those behind
    while the picture turned would be a worse mismatch than not rotating."""
    t = _calib_tab()
    t._seed_tx.setValue(120.0)
    t._draw_seed_rings()
    tf = t._img_view.lab_transform()
    assert tf is not None
    assert t._ring_items, "the seed preview should draw rings here"
    assert all(it.transform() == tf for it in t._ring_items)
    assert t._img_view._iv.getImageItem().transform() == tf


def test_calibrate_rings_carry_no_transform_when_unrolled(app):
    t = _calib_tab()
    t._seed_tx.setValue(0.0)
    t._draw_seed_rings()
    assert t._img_view.lab_transform() is None
    assert t._ring_items
    assert all(it.transform().isIdentity() for it in t._ring_items)


def test_the_data_viewer_rotates_when_its_tx_is_edited(app):
    """Regression: tx was added to DetectorGeometryCard without being wired
    to _on_bc_changed, so it fed get_geometry() while notifying nobody --
    the roll reached the saved geometry but never the display."""
    from midas_gui.tab_view import DataViewerTab
    t = DataViewerTab()
    c = t._geom_card
    c._bcy.setValue(BCY); c._bcz.setValue(BCZ)
    assert t._viewer.lab_transform() is None

    c._tx.setValue(210.0)
    assert t._viewer.lab_transform() is not None
    assert t._viewer._iv.getImageItem().transform() == t._viewer.lab_transform()

    c._tx.setValue(0.0)
    assert t._viewer.lab_transform() is None
