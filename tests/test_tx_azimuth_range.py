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


# -- feeding a result back must not erase the azimuth --------------------
#
# The second half of the same bug, and the one that actually degraded
# calibrations at the beamline: the range fix let 296.885 be typed in, and
# then "Feed result back to seed" -- on by default -- wrote it straight back
# out again as 0. No pipeline refines tx (it is frozen in
# midas_calibrate_v2/compat/from_v1.py), and first_time never even carries
# the seed, so result.tx is 0 for a panel that is physically at 296.885.
# Each Run moved the panel another step away from where it is: observed as
# Lsd drifting 2768.895 -> 2384.979 mm and post-refine strain 364 -> ~1400
# microstrain across three attempts, with the overview line quietly reading
# "ge1 tx 0 deg (tx not set)".

from types import SimpleNamespace


def _crystalline_result(**kw):
    """What a powder pipeline hands back: no fit_sigma, tx echoed as 0."""
    base = dict(Lsd=2_768_895.0, BC_y=1024.0, BC_z=1024.0,
                tx=0.0, ty=0.1, tz=-0.2, distortion={},
                pxY=200.0, pxZ=200.0, NrPixelsY=2048, NrPixelsZ=2048,
                wavelength_A=0.1729, post_residual_strain_uE=364.0)
    base.update(kw)
    return SimpleNamespace(**base)


#: What a crystalline (powder) pipeline actually refines. tx is NOT in it --
#: no crystalline pipeline fits the azimuth -- which is the whole point of
#: these tests. Since the upstream merge, seed_from_result takes this
#: explicitly instead of inferring "did the fit refine tx?" from the result,
#: so the tests now state the premise they always relied on.
_CRYSTALLINE_REFINE = {"BC": True, "Lsd": True, "ty": True, "tz": True}


def test_a_fit_round_trip_leaves_the_seeded_azimuth_alone(hydra_card):
    """Set the real ge1 azimuth, run, feed the result back: tx must still be
    the azimuth the panel is installed at."""
    hydra_card._seed_tx.setValue(296.885)
    hydra_card._seed_en_tx.setChecked(True)
    hydra_card.seed_from_result(_crystalline_result(), _CRYSTALLINE_REFINE)
    assert hydra_card._seed_tx.value() == pytest.approx(296.885), \
        "feeding a result back erased the panel's installation azimuth"
    assert hydra_card._seed_en_tx.isChecked(), \
        "tx went unticked, so the next fit would silently run at tx=0"


def test_the_repeated_run_does_not_walk_the_azimuth_away(hydra_card):
    """Three Runs in a row, as at the beamline. The damage was cumulative."""
    hydra_card._seed_tx.setValue(296.885)
    hydra_card._seed_en_tx.setChecked(True)
    for _ in range(3):
        hydra_card.seed_from_result(_crystalline_result(), _CRYSTALLINE_REFINE)
    assert hydra_card.seed_tx_value() == pytest.approx(296.885)


def test_the_parameters_the_fit_does_refine_still_come_back(hydra_card):
    """The fix must not turn feedback off wholesale -- Lsd/BC/ty/tz are
    refined, and feeding them back is the point of the checkbox."""
    hydra_card.seed_from_result(
        _crystalline_result(Lsd=2_700_000.0, BC_y=1030.0), _CRYSTALLINE_REFINE)
    assert hydra_card._seed_lsd.value() == pytest.approx(2700.0)
    assert hydra_card._seed_bcy.value() == pytest.approx(1030.0)
    assert hydra_card._seed_ty.value() == pytest.approx(0.1)


def test_the_one_fit_that_does_refine_tx_still_feeds_it_back(app):
    """The manual d-spacing fit refines tx as an ordinary free parameter when
    asked (helpers.fit_geometry_from_ring_picks), and reports a sigma for it
    only then. That result's tx is real and must reach the seed."""
    from midas_gui.tab_calibrate import CalibrationTab
    tab = CalibrationTab()
    tab._seed_tx.setValue(12.0)
    refined = _crystalline_result(tx=31.25)
    refined.fit_sigma = {"Lsd": 120.0, "tx": 0.04}
    # The manual d-spacing fit DOES refine tx when asked, and the tab reads
    # what the last run refined rather than being told per call.
    tab._last_refine_flags = {**_CRYSTALLINE_REFINE, "tx": True}
    tab._seed_from_result(refined)
    assert tab._seed_tx.value() == pytest.approx(31.25, abs=5e-3)


def test_a_crystalline_result_does_not_touch_the_single_detector_tx(app):
    """Same tab, same field, no sigma for tx -- so the fit did not refine it."""
    from midas_gui.tab_calibrate import CalibrationTab
    tab = CalibrationTab()
    tab._seed_tx.setValue(207.5)
    # Explicit: the crystalline fit refined everything EXCEPT tx, so this is
    # the real case, not the vacuous one where nothing is promoted at all.
    tab._last_refine_flags = dict(_CRYSTALLINE_REFINE)
    tab._seed_from_result(_crystalline_result())
    assert tab._seed_tx.value() == pytest.approx(207.5, abs=5e-3)
