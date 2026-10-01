"""Data Viewer image-viewer display origin + live pixel readout.

Two behaviours that are easy to break silently:

* **Origin** is display-only. Flipping it must move nothing but the painted
  rows — the readout, the stored frame and every overlay stay in the same
  (row, col) frame, because MIDAS geometry (BC_y/BC_z, ring overlays) is fit
  in that frame regardless of which corner the user chose to look at it from.
* **The pixel readout must survive a new frame arriving under a stationary
  cursor** — during live acquisition ``set_image`` is called continuously while
  the mouse never moves, and it used to overwrite the readout with its
  "Move cursor over image" placeholder on every single frame.
* **vmin%/vmax% are off the toolbar but still live.** They were removed from
  the visible row, not from the viewer: ``_redisplay`` still reads them for the
  auto-level window and they still round-trip through the project state, so
  hiding them must not have quietly reverted the 30/99 default or unhooked the
  percentile recompute.

Builds pyqtgraph widgets, hence forked, with every Qt-pulling import deferred
into a fixture — see STATE.md.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def viewer(app):
    from midas_gui.widgets import ImageViewer
    return ImageViewer()


# Asymmetric so a flip in either axis is detectable.
FRAME = np.arange(12, dtype=np.float32).reshape(3, 4)


# ── origin ───────────────────────────────────────────────────────────

def test_default_origin_is_bottom_left(viewer):
    """MIDAS convention — the calibration the other tabs fit assumes it."""
    from midas_gui.widgets import ORIGIN_BOTTOM_LEFT
    assert viewer.origin() == ORIGIN_BOTTOM_LEFT
    assert viewer._iv.getView().getViewBox().yInverted() is False


def test_top_left_inverts_the_y_axis_and_back(viewer):
    from midas_gui.widgets import ORIGIN_BOTTOM_LEFT, ORIGIN_TOP_LEFT
    vb = viewer._iv.getView().getViewBox()
    viewer.set_origin(ORIGIN_TOP_LEFT)
    assert viewer.origin() == ORIGIN_TOP_LEFT
    assert vb.yInverted() is True
    viewer.set_origin(ORIGIN_BOTTOM_LEFT)
    assert vb.yInverted() is False


def test_unknown_origin_falls_back_to_bottom_left(viewer):
    from midas_gui.widgets import ORIGIN_BOTTOM_LEFT
    viewer.set_origin("sideways")
    assert viewer.origin() == ORIGIN_BOTTOM_LEFT
    assert viewer._iv.getView().getViewBox().yInverted() is False


def test_origin_does_not_touch_the_data_or_the_readout(viewer):
    """Display-only: same stored array, same value at the same (row, col)."""
    from midas_gui.widgets import ORIGIN_TOP_LEFT
    viewer.set_image(FRAME)
    viewer._hover_xy = (1.4, 2.7)          # col 1, row 2 → FRAME[2, 1] == 9
    before = viewer._coord_text()
    viewer.set_origin(ORIGIN_TOP_LEFT)
    np.testing.assert_array_equal(viewer._data, FRAME)
    assert viewer._coord_text() == before
    assert "intensity = 9" in before


def test_origin_round_trips_through_display_state(viewer, app):
    from midas_gui.widgets import ImageViewer, ORIGIN_BOTTOM_LEFT, ORIGIN_TOP_LEFT
    viewer.set_origin(ORIGIN_TOP_LEFT)
    state = viewer.display_state()
    assert state["origin"] == ORIGIN_TOP_LEFT

    other = ImageViewer()
    assert other.origin() == ORIGIN_BOTTOM_LEFT
    other.set_display_state(state)
    assert other.origin() == ORIGIN_TOP_LEFT
    assert other._iv.getView().getViewBox().yInverted() is True


def test_state_without_origin_leaves_it_alone(viewer):
    """Projects saved before this existed must not flip anything."""
    from midas_gui.widgets import ORIGIN_TOP_LEFT
    viewer.set_origin(ORIGIN_TOP_LEFT)
    viewer.set_display_state({"log": False, "vmin": 10, "vmax": 90})
    assert viewer.origin() == ORIGIN_TOP_LEFT


def test_origin_button_tracks_and_drives_the_viewer(viewer):
    from midas_gui.widgets import OriginToolButton, ORIGIN_BOTTOM_LEFT, ORIGIN_TOP_LEFT
    btn = OriginToolButton(viewer)
    assert btn.text() == "Origin: BL"
    assert "Bottom-Left" in btn.toolTip()          # the MIDAS-calibration hint
    assert "Currently: Bottom-left" in btn.toolTip()
    assert [a.text() for a in btn._actions.values()] == ["Bottom-left", "Top-left"]

    top_left_action = btn._actions[ORIGIN_TOP_LEFT]
    top_left_action.trigger()
    assert viewer.origin() == ORIGIN_TOP_LEFT
    assert btn.text() == "Origin: TL"
    assert "Currently: Top-left" in btn.toolTip()
    assert top_left_action.isChecked()

    # Changed behind the button's back (project-state restore) → sync() catches up.
    viewer.set_origin(ORIGIN_BOTTOM_LEFT)
    btn.sync()
    assert btn.text() == "Origin: BL"
    assert btn._actions[ORIGIN_BOTTOM_LEFT].isChecked()


# ── live pixel readout ───────────────────────────────────────────────

def test_readout_starts_as_the_placeholder(viewer):
    viewer.set_image(FRAME)
    text = viewer._coord_bar.text()
    assert "Move cursor over image" in text
    assert "4×3 px" in text


def test_readout_follows_incoming_frames_without_the_mouse_moving(viewer):
    """The live-acquisition bug: set_image used to reset the bar every frame."""
    viewer.set_image(FRAME)
    viewer._hover_xy = (1.2, 2.9)                       # col 1, row 2
    viewer._refresh_coord_bar()
    assert "intensity = 9" in viewer._coord_bar.text()

    for i in range(3):                                  # frames streaming in
        viewer.set_image(FRAME + 100 * (i + 1), autorange=False, reset_levels=False)
        text = viewer._coord_bar.text()
        assert "Move cursor over image" not in text
        assert f"intensity = {9 + 100 * (i + 1)}" in text
        assert "x (col) = 1" in text and "y (row) = 2" in text


def test_readout_reverts_to_the_placeholder_once_the_cursor_leaves(viewer):
    viewer.set_image(FRAME)
    viewer._hover_xy = (1.2, 2.9)
    viewer._refresh_coord_bar()
    assert "intensity = 9" in viewer._coord_bar.text()

    viewer.leaveEvent(None)
    assert viewer._hover_xy is None
    assert "intensity" in viewer._coord_bar.text()      # text stands until replaced
    viewer.set_image(FRAME)                             # next frame clears it
    assert "Move cursor over image" in viewer._coord_bar.text()


def test_hover_outside_the_frame_shows_the_placeholder(viewer):
    """A shrinking frame must not report a stale out-of-bounds pixel."""
    viewer.set_image(FRAME)
    viewer._hover_xy = (3.5, 2.5)                       # inside 4×3
    viewer._refresh_coord_bar()
    assert "intensity = 11" in viewer._coord_bar.text()

    viewer.set_image(FRAME[:1, :1])                     # now 1×1 — hover is off-frame
    assert "Move cursor over image" in viewer._coord_bar.text()


# ── vmin%/vmax% removed from the toolbar, kept as state ──────────────

def _toolbar_texts(viewer):
    bar = viewer._toolbar_layout
    out = []
    for i in range(bar.count()):
        w = bar.itemAt(i).widget()
        if w is not None and hasattr(w, "text") and w.text():
            out.append(w.text())
    return out


@pytest.mark.parametrize("factory", ["ImageViewer", "CakeViewer"])
def test_percentile_spinboxes_are_off_the_toolbar(app, factory):
    import midas_gui.widgets as W
    v = getattr(W, factory)()
    joined = " ".join(_toolbar_texts(v))
    assert "vmin" not in joined and "vmax" not in joined
    for spin in (v._vmin, v._vmax):
        assert spin.isHidden()
        assert v._toolbar_layout.indexOf(spin) == -1


@pytest.mark.parametrize("factory", ["ImageViewer", "CakeViewer"])
def test_percentile_defaults_still_30_and_99(app, factory):
    import midas_gui.widgets as W
    v = getattr(W, factory)()
    assert (v._vmin.value(), v._vmax.value()) == (30, 99)


def test_percentiles_still_drive_the_auto_levels(viewer):
    """Hidden, not disconnected — the levels must still follow the values."""
    ramp = np.linspace(0.0, 1000.0, 10000, dtype=np.float32).reshape(100, 100)
    viewer._log.setChecked(False)
    viewer.set_image(ramp)
    wide = viewer._iv.getHistogramWidget().getLevels()
    viewer._vmin.setValue(45)          # narrows the window; also clears manual levels
    viewer._vmax.setValue(55)
    narrow = viewer._iv.getHistogramWidget().getLevels()
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
    assert viewer.display_state()["vmin"] == 45


def test_percentiles_still_round_trip_through_display_state(viewer, app):
    from midas_gui.widgets import ImageViewer
    viewer._vmin.setValue(5); viewer._vmax.setValue(95)
    other = ImageViewer()
    other.set_display_state(viewer.display_state())
    assert (other._vmin.value(), other._vmax.value()) == (5, 95)


# ── the button is wired up wherever a detector frame is shown ────────

@pytest.mark.parametrize("page, viewer_attr", [
    ("midas_gui.tab_calibrate:CalibrationTab", "_img_view"),
    ("midas_gui.hydra_calib_page:HydraCalibrationPage", "_img_view"),
    ("midas_gui.tab_batch:BatchTab", "_det_view"),
    ("midas_gui.hydra_batch_page:HydraBatchPage", "_det_view"),
])
def test_origin_button_present_and_state_round_trips(app, page, viewer_attr):
    import importlib
    from midas_gui.widgets import OriginToolButton, ORIGIN_TOP_LEFT
    mod_name, cls_name = page.split(":")
    tab = getattr(importlib.import_module(mod_name), cls_name)()
    viewer = getattr(tab, viewer_attr)
    bar = viewer._toolbar_layout
    assert any(isinstance(bar.itemAt(i).widget(), OriginToolButton)
               for i in range(bar.count()))

    tab._origin_btn._actions[ORIGIN_TOP_LEFT].trigger()
    assert viewer.origin() == ORIGIN_TOP_LEFT

    restored = getattr(importlib.import_module(mod_name), cls_name)()
    restored.set_state(tab.get_state())
    assert getattr(restored, viewer_attr).origin() == ORIGIN_TOP_LEFT
    assert restored._origin_btn.text() == "Origin: TL"


# ── lab-frame compass is invariant to the display origin ─────────────
#
# The compass describes the hutch, not the frame: +Y_Lab is vertically up in
# the real world and the beam goes into the screen, whatever the user has done
# to the picture. A reference that flipped along with the image could not be
# used to check the image, which is the overlay's entire job.

def _compass_screen_directions(app, viewer, shape=(200, 200), bc=(100.0, 100.0)):
    """Render the compass into ``viewer`` and report, in *screen* terms, where
    each arrow points and which side of the beam centre each label sits on."""
    import pyqtgraph as pg
    from midas_gui.widgets import build_lab_frame_axes_items
    NAMES = {"#ff3b30": "X_Lab", "#34c759": "Y_Lab"}

    items = build_lab_frame_axes_items(viewer._iv, shape, *bc)
    for it in items:
        viewer._iv.addItem(it)
    app.processEvents()
    vb = viewer._iv.getView().getViewBox()
    p0 = vb.mapViewToScene(pg.Point(*bc))
    out = {}
    for it in items:
        if isinstance(it, pg.PlotDataItem):
            name = NAMES.get(it.opts["pen"].color().name().lower())
            if name is None:
                continue                     # η tick, not an axis arrow
            x, y = it.getData()
            tip = int(np.argmax(np.hypot(x - bc[0], y - bc[1])))
            p1 = vb.mapViewToScene(pg.Point(float(x[tip]), float(y[tip])))
            dx, dy = p1.x() - p0.x(), p1.y() - p0.y()
            out[name] = (("left" if dx < 0 else "right") if abs(dx) > abs(dy)
                         else ("up" if dy < 0 else "down"))   # scene y grows down
        elif isinstance(it, pg.TextItem):
            text = it.toPlainText().replace("\n", " ")
            key = ("beam" if "beam" in text else
                   "eta180" if "180" in text else
                   "Y_Lab box" if "ZMIDAS" in text else None)
            if key:
                cy = it.mapRectToScene(it.boundingRect()).center().y()
                out[key] = "below" if cy > p0.y() else "above"
    for it in items:
        viewer._iv.removeItem(it)
    return out


def test_compass_points_the_same_way_in_both_origins(app, viewer):
    from midas_gui.widgets import ORIGIN_BOTTOM_LEFT, ORIGIN_TOP_LEFT
    viewer.resize(500, 500); viewer.show()
    viewer.set_raw_frame(np.zeros((200, 200), dtype=np.float32), [])
    app.processEvents()

    viewer.set_origin(ORIGIN_BOTTOM_LEFT)
    app.processEvents()
    bl = _compass_screen_directions(app, viewer)
    viewer.set_origin(ORIGIN_TOP_LEFT)
    app.processEvents()
    tl = _compass_screen_directions(app, viewer)

    assert bl == tl, "the lab frame moved when only the display origin changed"
    # ...and it points where the lab actually is, not merely consistently.
    assert bl["X_Lab"] == "left" and bl["Y_Lab"] == "up"
    assert bl["Y_Lab box"] == "above"      # label rides the arrow it names
    assert bl["beam"] == "below"           # ⊗ caption hangs under the centre
    assert bl["eta180"] == "below"         # η=180° is straight down


def test_origin_change_emits_so_overlays_can_rebuild(viewer):
    """The compass has to be *re-derived*, not merely re-painted, so the
    viewer announces the flip. Only a real change fires."""
    from midas_gui.widgets import ORIGIN_BOTTOM_LEFT, ORIGIN_TOP_LEFT
    seen = []
    viewer.originChanged.connect(seen.append)

    viewer.set_origin(ORIGIN_BOTTOM_LEFT)          # already there
    assert seen == []
    viewer.set_origin(ORIGIN_TOP_LEFT)
    assert seen == [ORIGIN_TOP_LEFT]
    viewer.set_origin(ORIGIN_TOP_LEFT)             # no-op
    assert seen == [ORIGIN_TOP_LEFT]
    viewer.set_origin(ORIGIN_BOTTOM_LEFT)
    assert seen == [ORIGIN_TOP_LEFT, ORIGIN_BOTTOM_LEFT]


@pytest.mark.parametrize("tab_cls, viewer_attr", [
    ("midas_gui.tab_view:DataViewerTab", "_viewer"),
    ("midas_gui.tab_calibrate:CalibrationTab", "_img_view"),
    ("midas_gui.tab_batch:BatchTab", "_det_view"),
])
def test_every_tab_with_a_compass_listens_for_the_origin_flip(app, tab_cls, viewer_attr):
    """Each tab that can draw the compass must be subscribed to its viewer's
    ``originChanged``; without that the overlay keeps the stale orientation
    until something else happens to redraw it."""
    import importlib
    # BatchTab's loader auto-starts a background StreamPreviewWorker on
    # construction (the default nickel_tifs preview read) — see
    # widgets.DataLoaderPanel._start_preview_worker. That worker lazily
    # imports midas_integrate_v2 (which pulls in torch) on first use, and if
    # this is the first time anything in the process has imported torch, it
    # is now happening on a background thread racing the main thread's own
    # import machinery — an intermittent SIGSEGV/SIGBUS in torch's own
    # extension-module init, not anything specific to this test. Reproduced
    # standalone (without this import) at roughly 50% failure rate across 10
    # runs of this file; a real app never hits it because some earlier tab
    # (Calibrate, via midas_calibrate_v2) has always imported torch on the
    # main thread first by the time Batch Integrate's loader runs. Import it
    # here too, before constructing anything, so every parametrized tab class
    # gets the same guarantee a real app gives them.
    pytest.importorskip("midas_integrate_v2")
    mod_name, cls_name = tab_cls.split(":")
    tab = getattr(importlib.import_module(mod_name), cls_name)()
    viewer = getattr(tab, viewer_attr)
    assert viewer.receivers(viewer.originChanged) > 0, \
        f"{cls_name} never connected to originChanged"
