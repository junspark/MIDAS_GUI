"""An empty frame must say it is empty, not just render black.

Reported from 1-ID: "pixirad h5 file does not load in the calibration
window for some reason. data viewer loads it fine." The file loaded
perfectly well. A Pixirad arms one frame before it starts counting, so
frame 1 of every .pixi.h5 is all zeros, and Calibrate opens on frame 1,
rendering a correct and entirely black picture of nothing. The Data Viewer
looked fine only because its projection was set to skip one frame.

Measured on the reported file (AgBeh_80p725keV_3s_003510.pixi.h5)::

    frame 0: max=0      nonzero=0
    frame 1: max=32766  nonzero=411436

Nothing was broken except that the GUI had no way to say "this frame is
blank" -- the same failure mode as the rest of this session: a display that
is literally correct and still misleads.

``forked`` per .context/DECISIONS.md -- builds a CalibrationTab (pyqtgraph).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(app):
    from midas_gui.tab_calibrate import CalibrationTab
    return CalibrationTab()


def _stack(tmp_path, name, frames):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / name
    with h5py.File(str(p), "w") as f:
        f.create_dataset("exchange/data", data=np.asarray(frames, np.float32))
    return str(p)


def _pixirad_like(tmp_path):
    """A blank arming frame followed by real ones, as a Pixirad writes it."""
    blank = np.zeros((32, 64), np.float32)
    live = np.full((32, 64), 500.0, np.float32)
    return _stack(tmp_path, "s_00001.pixi.h5", [blank, live, live])


def test_a_blank_first_frame_is_named_not_just_drawn(tab, tmp_path, app):
    tab._loader.set_path(_pixirad_like(tmp_path))
    app.processEvents()
    assert tab._image is not None, "the file must still load"
    assert not np.any(tab._image), "frame 1 is the blank one"

    txt = tab._blank_note.text()
    assert "entirely zero" in txt
    # Names the frame, so it reads as one bad frame and not a bad file.
    assert "Frame 1 of 3" in txt
    # ...and points at the controls that fix it, which sit at opposite ends
    # of the tab from each other.
    assert "Mean of frames" in txt and "start=1" in txt


def test_the_notice_clears_once_a_real_frame_is_shown(tab, tmp_path, app):
    tab._loader.set_path(_pixirad_like(tmp_path))
    app.processEvents()
    assert tab._blank_note.text()

    tab._loader._frame_spin.setValue(1)   # as the frame stepper does
    app.processEvents()
    assert np.any(tab._image)
    assert not tab._blank_note.isVisible(), \
        "a standing warning on a good frame is its own kind of lie"


def test_an_ordinary_file_never_sees_the_notice(tab, tmp_path, app):
    live = np.full((32, 64), 500.0, np.float32)
    tab._loader.set_path(_stack(tmp_path, "s_00002.h5", [live, live]))
    app.processEvents()
    assert not tab._blank_note.isVisible()
