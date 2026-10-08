"""A blank opening frame must not look like a file that failed to load.

Reported from 1-ID, twice: "pixirad h5 file does not load in the
calibration window for some reason. data viewer loads it fine", then again
on the SAXS file after a warning had been added. The file loads perfectly
well both times. A Pixirad arms one frame before it starts counting, so
frame 1 of every .pixi.h5 is all zeros and Calibrate opened on it,
rendering a correct and entirely black picture of nothing. The Data Viewer
looked fine only because its projection was set to skip one frame.

Measured on the reported file (AgBeh_80p725keV_3s_003510.pixi.h5)::

    frame 0: max=0      nonzero=0
    frame 1: max=32766  nonzero=411436

The second report is why the tab now steps off the dud frame itself rather
than only naming it: a warning still left the user to do the work, and an
empty canvas is a bad first impression of a file that is fine. It is
announced, not silent, and it happens once per file so deliberately
stepping back is respected.

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


def _blank(): return np.zeros((32, 64), np.float32)
def _live(v=500.0): return np.full((32, 64), v, np.float32)


def _pixirad_like(tmp_path, name="s_00001.pixi.h5"):
    """A blank arming frame followed by real ones, as a Pixirad writes it."""
    return _stack(tmp_path, name, [_blank(), _live(), _live()])


# -- the common case: land on something you can see ----------------------

def test_a_blank_opening_frame_is_stepped_over_and_announced(tab, tmp_path, app):
    tab._loader.set_path(_pixirad_like(tmp_path))
    app.processEvents()
    assert tab._loader.frame_index() == 1, "should not open on the dud frame"
    assert np.any(tab._image), "and the image should actually have data in it"

    txt = tab._blank_note.text()
    assert tab._blank_note.isVisibleTo(tab)
    # Announced, not silent: a tab quietly showing frame 2 while the stepper
    # reads 2/10 is honest only if it says why it is not on 1.
    assert "Frame 1 was entirely zero" in txt and "showing frame 2" in txt
    assert "step back" in txt


def test_an_ordinary_file_is_left_on_its_first_frame(tab, tmp_path, app):
    tab._loader.set_path(_stack(tmp_path, "s_00002.h5", [_live(), _live()]))
    app.processEvents()
    assert tab._loader.frame_index() == 0
    assert not tab._blank_note.isVisibleTo(tab)
    assert tab._blank_note.text() == ""


# -- the narrow guards, which are what keep this from being annoying -----

def test_stepping_back_onto_the_blank_frame_is_respected(tab, tmp_path, app):
    """Once per file. Re-skipping would fight the frame control, which is
    worse than the blank canvas it is avoiding."""
    tab._loader.set_path(_pixirad_like(tmp_path))
    app.processEvents()
    tab._loader.set_frame(0)
    app.processEvents()
    assert tab._loader.frame_index() == 0, "the user asked for this frame"
    assert not np.any(tab._image)
    # ...and falls back to naming the problem rather than saying nothing.
    txt = tab._blank_note.text()
    assert "Frame 1 of 3 is entirely zero" in txt
    assert "Mean of frames" in txt and "start=1" in txt


def test_a_wholly_empty_stack_is_not_chased_frame_by_frame(tab, tmp_path, app):
    """Only ever one frame forward: the dud is the first readout, not an
    arbitrary run, and scanning would read an unbounded part of a big stack
    on every load. If frame 2 is blank too, say so and stop."""
    tab._loader.set_path(_stack(tmp_path, "s_00003.pixi.h5",
                                [_blank(), _blank(), _blank()]))
    app.processEvents()
    assert tab._loader.frame_index() == 1, "one step, then give up"
    assert "entirely zero" in tab._blank_note.text()


def test_the_frame_mean_is_left_alone(tab, tmp_path, app):
    """With the mean on, what is displayed is not a frame at all and the
    start/end boxes are the control that matters -- moving the frame index
    underneath it would change nothing visible and confuse the readout."""
    tab._loader.set_path(_stack(tmp_path, "s_00004.h5", [_live(), _live()]))
    app.processEvents()
    tab._avg_check.setChecked(True)
    app.processEvents()
    at = tab._loader.frame_index()
    tab._loader.set_path(_pixirad_like(tmp_path, "s_00005.pixi.h5"))
    app.processEvents()
    assert tab._loader.frame_index() == at
