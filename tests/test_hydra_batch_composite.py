"""Batch Integrate's Hydra Detector view can show all four panels at once.

Asked for at 1-ID-E: "hydra view does not show the four panels. we need to fix
this too. This view is similar to the data viewer but with the four detector
calibrations applied appropriately."

The view showed one panel at a time -- deliberately, since drawing another
panel onto the shared viewer would show it under the wrong geometry -- so there
was no way to check that the four calibrations register against each other
before committing to a batch run.

Almost all of the machinery already existed: ``HydraDetectorToolbar`` has always
carried a Composite button (the Data Viewer's Hydra page uses it) and
``hydra.build_windmill_composite`` does the remapping. This page opted out with
``include_composite=False``, and four of its methods parsed the toolbar key as
``int(key[2])`` -- which reads the digit out of "ge3" and raises ValueError on
"composite", whose third character is 'm'.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets. Page
instances are the scarce resource here (HydraBatchPage is the heaviest in the
repo, see tests/test_hydra_batch_ui.py), so each test below builds at most one
and the key-parsing tests build none at all.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.forked

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "test_data" / "gui_synthetic" / "hydra"


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def fixture_available():
    if not FIXTURE_DIR.exists():
        pytest.skip("test_data/gui_synthetic/hydra/ fixture not present — "
                    "run make_hydra_test_data.py")
    return FIXTURE_DIR


def _calibrated_page(app, fx, panels=(1, 2, 3, 4)):
    """A page with the synthetic four-panel set loaded and each panel's REAL
    calibration applied — the thing the composite is supposed to honour."""
    from midas_gui.hydra_batch_page import HydraBatchPage
    from midas_gui.helpers import geometry_fields_from_file
    page = HydraBatchPage()
    page._loader.set_path(str(fx / "ge1" / "panel.ge1.h5"))
    app.processEvents()
    for n in panels:
        fields = geometry_fields_from_file(str(fx / f"ge{n}" / f"ps_ge{n}.txt"))
        page.set_panel_calibration(n, SimpleNamespace(**fields))
    app.processEvents()
    return page


# ── key parsing: no page needed ──────────────────────────────────────────

def test_the_toolbar_key_parses_without_assuming_a_panel_number():
    """`"composite"[2]` is 'm'. Every consumer used to do int(key[2]), so the
    button could not be enabled until they all went through one helper."""
    from midas_gui.hydra_batch_page import HydraBatchPage as P
    assert P._panel_num("ge1") == 1
    assert P._panel_num("ge4") == 4
    assert P._panel_num("composite") is None
    assert P._panel_num("") is None
    assert P._panel_num(None) is None


# ── the view itself ──────────────────────────────────────────────────────

def test_selecting_composite_builds_it_from_the_panels_own_calibrations(app, fixture_available):
    """The whole point: each panel placed by ITS calibration, not by the
    bundled default geometry build_windmill_composite falls back to."""
    from midas_gui import hydra
    page = _calibrated_page(app, fixture_available)

    # Lazy: nothing built while a single panel is on screen.
    assert page._composite_img is None

    page._toolbar.set_current("composite")
    app.processEvents()
    img = page._composite_img
    if img is None:
        pytest.skip("composite did not build in this environment")

    # Decimated, and the shape follows the canvas rather than a magic number.
    step = hydra.COMPOSITE_DISPLAY_STEP
    assert img.shape[0] == (page._big_det_size + step - 1) // step
    assert np.count_nonzero(np.nan_to_num(img)) > 0

    # The synthetic fixture's beam centre is 128; the bundled 1-ID-E default
    # is ~2300. Seeing 128 is what proves the cards were read.
    for n in (1, 2, 3, 4):
        assert page._states[n].bc_y == pytest.approx(128.0)


def test_a_new_calibration_does_not_leave_a_stale_composite(app, fixture_available):
    """The invalidation with no hydra_page precedent, and the most damaging
    to miss: the Calibrate hand-off is how real geometry arrives, and a
    composite built before it would keep placing that panel wrongly while
    looking entirely plausible."""
    from midas_gui.helpers import geometry_fields_from_file
    page = _calibrated_page(app, fixture_available)
    page._toolbar.set_current("composite")
    app.processEvents()
    if page._composite_img is None:
        pytest.skip("composite did not build in this environment")
    first = page._composite_build_id

    moved = geometry_fields_from_file(str(fixture_available / "ge2" / "ps_ge2.txt"))
    moved["BC_y"] = 99.0
    page.set_panel_calibration(2, SimpleNamespace(**moved))
    app.processEvents()

    assert page._states[2].bc_y == pytest.approx(99.0)
    assert page._composite_build_id > first, "composite was not rebuilt"


def test_fewer_than_two_calibrated_panels_builds_nothing(app, fixture_available):
    """A single panel is not a composite. Uncalibrated panels are skipped
    rather than placed from bundled defaults."""
    page = _calibrated_page(app, fixture_available, panels=(1,))
    page._toolbar.set_current("composite")
    app.processEvents()
    assert page._composite_img is None


def test_the_composite_carries_no_single_panel_overlay(app, fixture_available):
    """Rmin/Rmax and the bin grid are drawn around ONE panel's beam centre in
    that panel's pixel frame. On the BigDet canvas those coordinates mean
    nothing, so they are cleared — an overlay in a plausible wrong place is
    worse than none. The Rmax presets go dead for the same reason."""
    page = _calibrated_page(app, fixture_available)
    before = page._r_max.value()
    page._toolbar.set_current("composite")
    app.processEvents()

    assert page._bin_overlay_items == []
    assert not page._rmax_corner_btn.isEnabled()
    assert not page._rmax_edge_btn.isEnabled()
    from midas_gui.helpers import rmax_corner_px
    page._apply_rmax_preset(rmax_corner_px)
    assert page._r_max.value() == before, "a composite canvas filled a per-panel Rmax"


def test_switching_to_composite_keeps_the_per_panel_stacks_put(app, fixture_available):
    """Neither stack has a composite entry. Holding the last panel keeps its
    calibration card editable — which is the thing that rebuilds the view."""
    page = _calibrated_page(app, fixture_available)
    page._toolbar.set_current("ge3")
    app.processEvents()
    page._toolbar.set_current("composite")
    app.processEvents()
    assert page._card_stack.currentWidget() is page._cards[3]
    assert page._viewer_stack.currentWidget() is page._viewer_pairs[3]
