"""The frame range starts at 1 when frame 0 is a blank arming frame.

A Pixirad arms one frame before it starts counting, so frame 0 of every
.pixi.h5 is all zeros. Combining it scales a mean by (n-1)/n -- measured on
1-ID-E air_80p725keV_3s_003512, the reduced output came out exactly 10/9 low
(902.36 against a correct 1002.53).

Keyed on the frame's own content, not the detector name. The GE panels in the
same experiment record ten real frames, and defaulting those to 1 would throw
a good frame away on every run -- the mirror-image of the bug being fixed.

Only the DEFAULT moves; typing 0 gets the frame back.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked

SHAPE = (8, 6)


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _h5(path, n=6, blank_first=False, ndim=3):
    h5py = pytest.importorskip("h5py")
    data = np.stack([np.full(SHAPE, float(i + 1), np.float32) for i in range(n)])
    if blank_first:
        data[0] = 0.0
    if ndim == 2:
        data = data[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("exchange/data", data=data)
    return path


def _blank(path):
    from midas_gui.widgets import DataLoaderPanel
    return DataLoaderPanel._first_raw_frame_is_blank(str(path), "exchange/data")


def test_a_blank_opening_frame_is_detected(tmp_path):
    assert _blank(_h5(tmp_path / "pixi.h5", blank_first=True)) is True


def test_a_real_opening_frame_is_not(tmp_path):
    assert _blank(_h5(tmp_path / "ge.h5", blank_first=False)) is False


def test_a_single_frame_file_is_never_called_blank(tmp_path):
    """Nothing to step onto -- dropping the only frame leaves no data."""
    assert _blank(_h5(tmp_path / "one.h5", n=1, blank_first=True)) is False


def test_a_2d_dataset_is_not_a_stack(tmp_path):
    assert _blank(_h5(tmp_path / "flat.h5", ndim=2)) is False


def test_an_unreadable_file_says_nothing(tmp_path):
    p = tmp_path / "junk.h5"
    p.write_bytes(b"not an hdf5 file")
    assert _blank(p) is False


def test_a_missing_file_says_nothing(tmp_path):
    assert _blank(tmp_path / "absent.h5") is False


def test_the_loader_defaults_past_a_blank_frame(app, tmp_path):
    from midas_gui.tab_batch_correct import BatchCorrectionTab
    tab = BatchCorrectionTab()
    tab._loader.set_path(str(_h5(tmp_path / "pixi.h5", blank_first=True)))
    app.processEvents()
    assert tab._loader._fr_start.value() == 1
    assert tab._loader._fr_end.value() == 5


def test_the_loader_keeps_frame_zero_when_it_holds_data(app, tmp_path):
    """The case that must NOT move: a detector with no arming frame."""
    from midas_gui.tab_batch_correct import BatchCorrectionTab
    tab = BatchCorrectionTab()
    tab._loader.set_path(str(_h5(tmp_path / "ge.h5", blank_first=False)))
    app.processEvents()
    assert tab._loader._fr_start.value() == 0
