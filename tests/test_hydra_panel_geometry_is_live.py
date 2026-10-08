"""A Hydra panel's geometry boxes must mean something.

``DetectorGeometryCard._export_geom`` used to return ``self._calib_geom``
wholesale whenever a calibration had been loaded, ignoring every geometry
widget on the card. On the Hydra page that is *always*: each panel loads a
bundled ``ps_ge{n}.txt`` at startup, so tx/ty/tz and the beam centre were
display-only for the whole page -- and ``set_geometry`` did not even write
tx into its box, so what was displayed was not what was used either.

Measured before the fix, panel 1 on a fresh page::

    tx box        = 0.0        <- what the user sees
    _calib_geom tx= 296.885    <- what the composite places the panel at
    ...and typing 138 into the box changed neither.

That is two failures at once: a control that silently does nothing, and a
readout that names a value nothing uses. Both matter more here than on the
single-detector tab, because per-panel geometry is the entire point of the
Hydra composite.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import pytest

pytestmark = pytest.mark.forked

PANELS = {1: "a.h5", 2: "b.h5", 3: "c.h5", 4: "d.h5"}


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def page(app):
    from midas_gui.hydra_page import HydraViewerPage
    p = HydraViewerPage()
    p._ensure_states_for_siblings(dict(PANELS))
    return p


def test_the_box_shows_the_roll_the_panel_is_actually_placed_at(page):
    """set_geometry wrote ty and tz into their boxes and skipped tx, so the
    one angle that differs per Hydra panel was the one never displayed."""
    for n in PANELS:
        card = page._cards[f"ge{n}"]
        used = (card._calib_geom or {}).get("tx")
        assert used is not None, n
        # The box is 2-decimal, so it shows the value to its own resolution.
        assert card._tx.value() == pytest.approx(used, abs=0.005), n
    # ...and the panels really do differ, so this is not vacuously true.
    rolls = {round((page._cards[f'ge{n}']._calib_geom or {}).get("tx"), 3)
             for n in PANELS}
    assert len(rolls) > 1, "a windmill composite needs distinct per-panel rolls"


def test_editing_a_panels_roll_reaches_the_composite(page):
    """The whole point of the card. This changed nothing at all before."""
    card = page._cards["ge1"]
    card._tx.setValue(138.0)
    assert card.get_full_geometry()["tx"] == pytest.approx(138.0)
    assert page._states[1].tx == pytest.approx(138.0)
    assert page._composite_img is None, "the canvas must be rebuilt"


def test_editing_the_beam_centre_and_tilts_reaches_the_composite(page):
    card = page._cards["ge2"]
    card._bcy.setValue(1111.0); card._bcz.setValue(2222.0)
    card._ty.setValue(0.75); card._tz.setValue(-0.25)
    g = card.get_full_geometry()
    assert (g["BC_y"], g["BC_z"]) == (pytest.approx(1111.0), pytest.approx(2222.0))
    assert (g["ty"], g["tz"]) == (pytest.approx(0.75), pytest.approx(-0.25))
    st = page._states[2]
    assert (st.bc_y, st.bc_z) == (pytest.approx(1111.0), pytest.approx(2222.0))


def test_an_untouched_box_does_not_round_the_calibration(page):
    """The boxes are fixed-decimal; tx has two. Letting the widget win
    unconditionally would quietly truncate a fitted 296.885 to 296.88 --
    0.005 deg, about 0.15 px at the edge of a 2048 px panel. An untouched
    box must contribute nothing at all.
    """
    for n in PANELS:
        card = page._cards[f"ge{n}"]
        for key in ("tx", "ty", "tz", "Lsd", "BC_y", "BC_z"):
            assert card.get_full_geometry()[key] == (card._calib_geom or {})[key], (n, key)


def test_the_file_keeps_what_no_widget_can_carry(page):
    """Distortion coefficients and the detector size have no box on this
    card, so the loaded calibration must still supply them after an edit."""
    card = page._cards["ge1"]
    before = card.get_full_geometry()
    assert before.get("distortion"), "the bundled panel geometry carries distortion"
    card._tx.setValue(17.0)
    after = card.get_full_geometry()
    assert after["distortion"] == before["distortion"]
    assert (after["NrPixelsY"], after["NrPixelsZ"]) == \
           (before["NrPixelsY"], before["NrPixelsZ"])


def test_one_pixel_box_does_not_flatten_a_non_square_pair(page):
    """The card has a single px field. Leaving it alone must not collapse a
    file's distinct pxY/pxZ onto one value; changing it means "square"."""
    card = page._cards["ge1"]
    card._calib_geom = dict(card._calib_geom or {}, pxY=200.0, pxZ=150.0)
    card._snapshot_calib_baseline()
    card._px.blockSignals(True); card._px.setValue(200.0); card._px.blockSignals(False)
    assert card.get_full_geometry()["pxZ"] == pytest.approx(150.0)

    card._px.setValue(62.0)
    g = card.get_full_geometry()
    assert (g["pxY"], g["pxZ"]) == (pytest.approx(62.0), pytest.approx(62.0))
