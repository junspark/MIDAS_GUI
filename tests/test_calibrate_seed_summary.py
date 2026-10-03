"""The Calibrate tab's seed summary must show the seed *values*, not just
which parameters are seeded.

The seed spin boxes live inside ``ManualSeedDialog``, which is non-modal and
normally closed, so from the tab itself the starting Lsd / BC / tilts were
invisible — while the Refine card right below promised "± a window around its
seed value" and the crystalline footer "centred on the seed", neither of which
showed the value they meant. Reported from the beamline on an AgBH fit.

``forked`` per .context/DECISIONS.md — builds a CalibrationTab (pyqtgraph).
"""
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


def test_unseeded_summary_says_what_the_fit_will_do_instead(tab):
    for cb in tab._seed_enables:
        cb.setChecked(False)
    txt = tab._seed_summary_lbl.text()
    assert "Fully automatic" in txt
    # The point of the line: say where the numbers come from when they are
    # not the user's.
    assert "auto-seeded from the image" in txt and "0°" in txt


def test_seeded_parameters_show_their_values(tab):
    tab._seed_bcy.setValue(1024.5)
    tab._seed_bcz.setValue(1030.25)
    tab._seed_lsd.setValue(1382.4)
    tab._seed_ty.setValue(0.37)
    tab._seed_en_bc.setChecked(True)
    tab._seed_en_lsd.setChecked(True)
    tab._seed_en_ty.setChecked(True)

    txt = tab._seed_summary_lbl.text()
    assert "BC 1024.5, 1030.25 px" in txt      # full 3-dp precision kept
    assert "Lsd 1382.4 mm" in txt              # trailing zeros trimmed
    assert "ty 0.37°" in txt
    # Unseeded ones are named, not silently dropped.
    assert "auto: tx, tz, Distortion" in txt


def test_summary_follows_a_value_edited_after_the_tick(tab):
    """Pick BC, the arrow keys and result feedback all move the spin boxes
    without touching the enable ticks the summary used to hang off."""
    tab._seed_en_lsd.setChecked(True)
    tab._seed_lsd.setValue(500.0)
    assert "Lsd 500 mm" in tab._seed_summary_lbl.text()
    tab._seed_lsd.setValue(612.125)
    assert "Lsd 612.125 mm" in tab._seed_summary_lbl.text()


def test_distortion_seed_count_reaches_the_summary(tab):
    tab._seed_dist = {"iso_R2": 1e-6, "iso_R4": -2e-9}
    tab._seed_en_dist.setChecked(True)
    tab._update_seed_dist_label()
    assert "Distortion 2 coeff" in tab._seed_summary_lbl.text()
