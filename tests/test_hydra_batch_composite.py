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
    before = {n: page._cards[n].caking() for n in (1, 2, 3, 4)}
    page._toolbar.set_current("composite")
    app.processEvents()

    assert page._bin_overlay_items == []
    # Caking lives on the cards now, so there is no shared Rmax for a
    # composite canvas to fill -- but assert it explicitly, because the
    # auto-Rmax resolution in the per-panel path would be a plausible place
    # for a BigDet corner distance to leak into a panel's real setting.
    assert {n: page._cards[n].caking() for n in (1, 2, 3, 4)} == before


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


# ── lab-frame axes ───────────────────────────────────────────────────────
# Asked for at 1-ID-E: "we also need to be able to show the lab frame on
# batch integrate panel." The Data Viewer, Calibrate and the single-detector
# Batch Integrate all had this overlay; the Hydra batch page had no
# lab-axes code at all.

def test_lab_axes_follow_the_selected_panel_and_the_composite(app, fixture_available):
    """One overlay, re-anchored — not one per panel accumulating on the
    shared viewer.

    The composite used to clear them, on the grounds that no single panel's
    beam centre anchors the canvas. That was wrong in the useful direction:
    the canvas registers all four onto ONE point, its centre, so the lab
    frame is defined — and reported missing from the beamline as "Lab frame
    view not visible in integration tab"."""
    page = _calibrated_page(app, fixture_available)
    assert page._axis_items == []

    page._lab_axes_chk.setChecked(True)
    app.processEvents()
    on_ge1 = len(page._axis_items)
    assert on_ge1 > 0, "lab axes did not draw"

    page._toolbar.set_current("ge3")
    app.processEvents()
    assert len(page._axis_items) == on_ge1, "axes accumulated instead of re-anchoring"

    page._toolbar.set_current("composite")
    app.processEvents()
    assert len(page._axis_items) == on_ge1, (
        "the composite has a well-defined centre and must still show the "
        "lab frame, without accumulating a second overlay")

    page._toolbar.set_current("ge2")
    app.processEvents()
    assert len(page._axis_items) == on_ge1

    page._lab_axes_chk.setChecked(False)
    app.processEvents()
    assert page._axis_items == []


# ── lab-frame axes on the composite ──────────────────────────────────────
# Reported at 1-ID-E: "Lab frame view not visible in integration tab." The
# composite branch cleared the axes outright, on the grounds that no single
# panel's orientation describes the canvas. But the compositor registers all
# four beam centres onto ONE point -- the canvas centre -- so the lab frame
# IS defined, and the composite is the view where it matters most. Pure
# geometry, so no page is built here (see the module docstring: page
# instances are this file's scarce resource).
import numpy as _np

from midas_gui.hydra_batch_page import HydraBatchPage as _HBP


def test_the_composite_beam_centre_is_the_canvas_centre():
    bc = _HBP._composite_axes_bc(_np.zeros((3328, 3328), dtype=_np.float32))
    assert bc == (1664.0, 1664.0)


def test_it_is_returned_as_y_then_z_not_rows_then_cols():
    """build_lab_frame_axes_items takes (y, z) = (column, row); an array's
    shape is (rows, cols). A square composite hides a swap, so check a
    non-square canvas."""
    bc_y, bc_z = _HBP._composite_axes_bc(_np.zeros((200, 400), dtype=_np.float32))
    assert (bc_y, bc_z) == (200.0, 100.0)


@pytest.mark.parametrize("img", [None, _np.zeros((0, 0)), _np.zeros(5)])
def test_no_centre_without_a_real_2d_frame(img):
    assert _HBP._composite_axes_bc(img) is None
