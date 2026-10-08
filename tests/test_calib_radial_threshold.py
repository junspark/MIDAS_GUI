"""The Calibrate tab's "radially adaptive threshold" card: an interactive
drag-point curve (``widgets.RadialThresholdEditor``) centered on the current
beam centre, replacing the scalar cutoff, then the Gaussian-in-r floor, then
the power-law step this went through earlier the same day — see
.context/DECISIONS.md. Also pins that Pick BC / Pick Ring only populate the
BC value — they must never re-activate Manual seed, which the user must
tick themselves.

Its own module (and ``forked``) because building a second ``CalibrationTab``
in the same process segfaults inside pyqtgraph's ViewBox teardown — see
``tests/test_calibrate_state_restore.py``.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.forked
def test_threshold_toggle_enables_the_curve_editor(app):
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    assert tab._thr_editor.isEnabled() is False

    tab._thr_check.setChecked(True)
    assert tab._thr_editor.isEnabled() is True

    tab._thr_check.setChecked(False)
    assert tab._thr_editor.isEnabled() is False


@pytest.mark.forked
def test_calib_image_zeros_more_aggressively_near_bc_than_far_from_it(app):
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    img = np.full((200, 300), 5.0, dtype=np.float32)
    tab._image = img
    tab._seed_bcy.setValue(150.0)
    tab._seed_bcz.setValue(100.0)

    tab._thr_editor.set_domain(100.0)
    tab._thr_editor.set_points([0.0, 10.0], [10.0, 0.0])
    tab._thr_check.setChecked(True)

    out = tab._calib_image()
    assert out[100, 150] == 0.0     # at BC: 5 < peak of 10
    assert out[0, 0] == 5.0         # far corner: curve has decayed to its floor (1.0) < 5

    # Off: the raw image is returned unchanged.
    tab._thr_check.setChecked(False)
    assert np.array_equal(tab._calib_image(), img)


@pytest.mark.forked
def test_pick_bc_sets_value_but_does_not_activate_manual_seed(app):
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    assert tab._seed_en_bc.isChecked() is False
    style_before = tab._seed_btn.styleSheet()

    tab._on_bc_picked(123.4, 56.7)

    assert tab._seed_bcy.value() == pytest.approx(123.4)
    assert tab._seed_bcz.value() == pytest.approx(56.7)
    assert tab._seed_en_bc.isChecked() is False
    assert tab._seed_btn.styleSheet() == style_before   # still not "active" (green)


@pytest.mark.forked
def test_ring_fit_bc_sets_value_but_does_not_activate_manual_seed(app):
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    style_before = tab._seed_btn.styleSheet()

    tab._on_ring_fit_bc(10.0, 20.0, 99.0)

    assert tab._seed_bcy.value() == pytest.approx(10.0)
    assert tab._seed_bcz.value() == pytest.approx(20.0)
    assert tab._seed_en_bc.isChecked() is False
    assert tab._seed_btn.styleSheet() == style_before


@pytest.mark.forked
def test_picked_bc_is_used_as_the_radial_threshold_origin_even_while_inactive(app):
    """The threshold's BC comes from the live Seed-card spinboxes regardless
    of whether Manual seed's BC checkbox is ticked."""
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    img = np.full((100, 100), 5.0, dtype=np.float32)
    tab._image = img

    tab._on_bc_picked(42.0, 43.0)
    assert tab._seed_en_bc.isChecked() is False   # confirm still inactive

    tab._thr_editor.set_domain(100.0)
    tab._thr_editor.set_points([0.0, 10.0], [10.0, 0.0])
    tab._thr_check.setChecked(True)

    out = tab._calib_image()
    assert out[43, 42] == 0.0       # zeroed right at the picked BC
    assert out[0, 0] == 5.0         # untouched far away
