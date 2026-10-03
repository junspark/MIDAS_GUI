"""The Calibrate tab must be able to *show* which pixels the fit will not see.

The Mask card already said how many pixels were excluded ("Tab 1 mask
(3,801,171 px)"), but a count cannot answer the question actually being asked
at the beamline — whether a ring arc about to be picked is half excluded, or
a module gap sits across the beam centre. ImageViewer.set_mask_overlay had
existed for the Mask Builder and Data Viewer all along; this tab just never
called it.

``forked`` per .context/DECISIONS.md — builds a CalibrationTab (pyqtgraph).
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
    t = CalibrationTab()
    assert t._image is not None, "bundled demo image should have loaded"
    return t


def _mask_for(tab, cols=10):
    m = np.zeros(np.asarray(tab._image).shape, dtype=np.uint8)
    m[:, :cols] = 1
    return m


def test_no_mask_shows_nothing(tab):
    assert tab._mask_status.text() == ""


def test_mask_count_is_reported_without_ticking_the_overlay(tab):
    """The count answers "is a mask applied at all?", which is worth having
    whether or not the user wants the image shaded."""
    tab.set_mask_from_tab1(_mask_for(tab))
    assert not tab._show_mask_check.isChecked()      # off by default
    txt = tab._mask_status.text()
    assert "excluded" in txt and f"{10 * np.asarray(tab._image).shape[0]:,}" in txt


def test_ticking_the_box_paints_the_overlay(tab):
    tab.set_mask_from_tab1(_mask_for(tab))
    tab._show_mask_check.setChecked(True)
    assert tab._img_view._overlay.isVisible()
    rgba = tab._img_view._overlay.image
    assert rgba is not None and rgba.shape[2] == 4
    # Masked pixels are the red, non-transparent ones; the image is shown
    # transposed, so the masked columns land in rows of the overlay.
    assert rgba[:10, :, 3].min() > 0        # excluded -> opaque
    assert rgba[10:, :, 3].max() == 0       # kept     -> fully transparent


def test_unticking_clears_it_but_keeps_the_count(tab):
    tab.set_mask_from_tab1(_mask_for(tab))
    tab._show_mask_check.setChecked(True)
    tab._show_mask_check.setChecked(False)
    assert tab._img_view._overlay.image.shape[:2] == (1, 1)   # cleared
    assert "excluded" in tab._mask_status.text()


def test_a_mask_that_does_not_fit_the_image_is_refused_not_painted(tab):
    """A mask built on another detector would otherwise paint a
    plausible-looking overlay that lines up with nothing."""
    tab._show_mask_check.setChecked(True)
    tab.set_mask_from_tab1(np.ones((7, 9), dtype=np.uint8))
    assert "≠ image" in tab._mask_status.text()
    assert tab._img_view._overlay.image.shape[:2] == (1, 1)   # nothing painted


def test_removing_the_mask_clears_the_status(tab):
    tab.set_mask_from_tab1(_mask_for(tab))
    tab.set_mask_from_tab1(None)
    assert tab._mask_status.text() == ""


# ── The run log must state the mask the run actually got ──

def test_run_log_line_reports_no_mask(tab):
    line = tab._mask_log_line()
    assert line.startswith("Mask: none")
    assert f"{np.asarray(tab._image).size:,} px included" in line


def test_run_log_line_reports_the_excluded_count(tab):
    tab.set_mask_from_tab1(_mask_for(tab))
    n = 10 * np.asarray(tab._image).shape[0]
    line = tab._mask_log_line()
    assert f"{n:,} /" in line and "excluded" in line
