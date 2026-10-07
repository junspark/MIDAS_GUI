"""``tx`` is a full-circle installation azimuth, not a small alignment tilt.

Reported at the beamline: "In the hydra calibration mode, I need to have
large Tx values but there seems to be a limit." The seed spins were built
as ``_fspin(-180, 180, ...)`` for all three of tx/ty/tz, which is right for
ty/tz -- sub-degree panel tilts -- and wrong for tx, which places the panel
around the beam. The bundled real fitted Hydra geometry is proof: three of
its four panels sit outside +/-180.

The silent half is the one worth a test. ``QDoubleSpinBox.setValue``
clamps rather than raising, so loading such a geometry did not fail, warn,
or look wrong -- it just quietly recorded 180 and went on to calibrate a
panel at the wrong azimuth. So these tests assert on the value that comes
back out, not on the range, which is what a reader would actually have to
check to notice the bug.

``forked`` per .context/DECISIONS.md -- builds Qt widgets.
"""
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _bundled_tx():
    """The tx of each bundled default panel, read from the files themselves
    so this test keeps describing the real geometry if those are refitted."""
    from midas_gui import hydra
    out = {}
    for n in (1, 2, 3, 4):
        for line in hydra.default_param_file(n).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].lower() == "tx":
                out[n] = float(parts[1])
    return out


def test_the_bundled_geometry_really_does_go_past_180():
    """If this ever fails the rest of the file is arguing about nothing."""
    tx = _bundled_tx()
    assert sorted(tx) == [1, 2, 3, 4]
    assert [n for n, v in sorted(tx.items()) if not -180 <= v <= 180] == [1, 4]


# -- the Hydra seed card (where it was reported) -------------------------

@pytest.fixture
def hydra_card(app):
    from midas_gui.hydra_calib_widgets import HydraCalibPanelCard
    return HydraCalibPanelCard(1)


def test_every_bundled_panel_azimuth_survives_being_seeded(hydra_card):
    """The whole bug in one assertion: a real panel geometry must come back
    out of the seed field as the angle that went in."""
    for n, tx in _bundled_tx().items():
        hydra_card._seed_tx.setValue(tx)
        assert hydra_card._seed_tx.value() == pytest.approx(tx), \
            f"ge{n}: tx {tx} was clamped to {hydra_card._seed_tx.value()}"


def test_the_far_side_of_the_circle_is_reachable(hydra_card):
    for tx in (190.0, 270.0, 359.9, -270.0):
        hydra_card._seed_tx.setValue(tx)
        assert hydra_card._seed_tx.value() == pytest.approx(tx)


def test_the_small_tilts_are_left_alone(hydra_card):
    """ty/tz are alignment tilts of a fraction of a degree; widening them
    would only make a fat-fingered entry harder to notice."""
    for w in (hydra_card._seed_ty, hydra_card._seed_tz):
        assert (w.minimum(), w.maximum()) == (-180.0, 180.0)


# -- the single-detector seed card has the same field --------------------

def test_the_single_detector_card_takes_them_too(app):
    """A panel pulled out of a Hydra array is calibrated on the ordinary
    Calibrate tab, with the same tx out of the same file."""
    from midas_gui.tab_calibrate import CalibrationTab
    tab = CalibrationTab()
    for n, tx in _bundled_tx().items():
        tab._seed_tx.setValue(tx)
        assert tab._seed_tx.value() == pytest.approx(tx, abs=5e-3), \
            f"ge{n}: tx {tx} was clamped to {tab._seed_tx.value()}"
