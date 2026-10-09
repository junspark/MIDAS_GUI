"""The composite card's geometry must follow the panels it composites.

Reported from 1-ID-E WAXS: "I loaded the calibrations for the four individual
files. but when I simulate the rings, they do not line up with the observed
rings at all. Also, intensity vs. 2theta plot seems to indicate that the CeO2
peaks do not line up."

The composite card read **Lsd 3071.610 mm**, which is exactly the mean of
ge1's real calibration (2392.223) and three bundled ps_ge{2,3,4} defaults
(3298587, 3297582, 3298047 um):

    (2392223 + 3298587 + 3297582 + 3298047) / 4 = 3071609.75 um

_reseed_composite_card_if_needed was keyed on the canvas size alone and
documented as firing "once per distinct canvas size". Loading four panel
calibrations changes every panel's Lsd but not the canvas extent, so the
composite kept what it was seeded with partway through that load -- after
ge1 had landed and before the rest. Every simulated ring then sat at the
wrong radius and the 2-theta axis was scaled by 28%, silently.

hydra._big_det_size_cache already keys on the panels' real geometry for this
exact reason ("loading a different calibration file for one panel"); this
makes the card's seeding agree.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import pytest

pytestmark = pytest.mark.forked

PANELS = {1: "a.h5", 2: "b.h5", 3: "c.h5", 4: "d.h5"}
REAL_LSD_UM = 2392223.0          # the connolly_oct26 ge{1..4} calibrations
CANVAS = 6656                    # unchanged across the load, which is the trap


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def page(app):
    from midas_gui.hydra_page import HydraViewerPage
    p = HydraViewerPage()
    p._ensure_states_for_siblings(dict(PANELS))
    p._reseed_composite_card_if_needed(CANVAS, dict(PANELS))
    return p


def _lsd_mm(page):
    return page._cards["composite"].get_geometry()["Lsd"] / 1000.0


def _load_calibration(page, n, lsd_um, app):
    """Put a real calibration on panel ``n``'s CARD, which is how one
    actually arrives.

    Deliberately not ``page._states[n].lsd = ...``: the cards are the source
    of truth and ``_on_card_geometry_changed`` syncs them INTO the states, so
    a state poked directly is reverted the next time any card emits. The
    sync also needs a full geometry (NrPixels/distortion present), since
    ``get_full_geometry`` returns None otherwise and silently syncs nothing.
    """
    card = page._cards[f"ge{n}"]
    g = dict(card.get_geometry())
    g.update({"Lsd": float(lsd_um), "NrPixelsY": 2048, "NrPixelsZ": 2048,
              "pxY": 200.0, "pxZ": 200.0, "distortion": {}, "im_trans": []})
    card.set_geometry(g)
    app.processEvents()


def test_the_bundled_defaults_seed_the_composite(page):
    assert _lsd_mm(page) == pytest.approx(3298.309, abs=1e-3)


def test_loading_every_panel_calibration_moves_the_composite(page, app):
    """The canvas size does not change, which is exactly why this was
    missed: the old key was the canvas size alone."""
    for n in PANELS:
        _load_calibration(page, n, REAL_LSD_UM, app)
    assert [round(page._states[n].lsd, 1) for n in PANELS] == [REAL_LSD_UM] * 4
    page._reseed_composite_card_if_needed(CANVAS, dict(PANELS))
    assert _lsd_mm(page) == pytest.approx(2392.223, abs=1e-3)


def test_a_part_way_load_does_not_stick(page, app):
    """3071.610 mm -- the number reported from the beamline -- is a
    one-panel-loaded average. It must be transient, not final."""
    _load_calibration(page, 1, REAL_LSD_UM, app)
    page._reseed_composite_card_if_needed(CANVAS, dict(PANELS))
    assert _lsd_mm(page) == pytest.approx(3071.610, abs=1e-3)   # as reported
    for n in (2, 3, 4):
        _load_calibration(page, n, REAL_LSD_UM, app)
    page._reseed_composite_card_if_needed(CANVAS, dict(PANELS))
    assert _lsd_mm(page) == pytest.approx(2392.223, abs=1e-3)   # and recovers


def test_an_unchanged_geometry_does_not_reseed(page):
    """Cheap no-op on every refresh, and it must not fight a hand-edit that
    nothing else has invalidated."""
    page._cards["composite"]._lsd.setValue(1234.5)
    page._reseed_composite_card_if_needed(CANVAS, dict(PANELS))
    assert page._cards["composite"]._lsd.value() == pytest.approx(1234.5)
