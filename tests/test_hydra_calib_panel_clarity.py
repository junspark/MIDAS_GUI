"""Which panel, which parameters, which tx — answerable without clicking.

Reported at the beamline, in one breath: "it is unclear what parameters
are dedicated to each GE panel. Also it is unclear which panel is being
fitted ... And I do not know what Tx they are getting."

All three come from the same shape: the per-panel card is a QStackedWidget
showing one panel at a time, and the seed summary named the seeded
parameters without their values. So the only way to read four panels' tx
was to click through four cards and open four dialogs.

tx is the one that matters most, because it is an input the fit never
refines -- a wrong or unset tx produces a confident result at the wrong
azimuth. It is therefore reported whether or not it is ticked: unticked
means the fit uses 0, and four panels at 0 is exactly the state that piles
every panel onto one wedge.

``forked`` per .context/DECISIONS.md — builds Qt widgets.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def card(app):
    from midas_gui.hydra_calib_widgets import HydraCalibPanelCard
    return HydraCalibPanelCard(1)


# ── the per-panel seed summary carries values, not just names ──────────

def test_the_summary_shows_the_tx_value_not_just_the_word(card):
    card._seed_en_tx.setChecked(True)
    card._seed_tx.setValue(296.885)
    assert "296.885" in card._seed_summary_lbl.text()


def test_the_summary_follows_a_value_edit(card):
    """It used to be driven only by the ticks, so a value change left a
    stale number on screen."""
    card._seed_en_tx.setChecked(True)
    card._seed_tx.setValue(117.8)
    assert "117.8" in card._seed_summary_lbl.text()
    card._seed_tx.setValue(207.5)
    assert "207.5" in card._seed_summary_lbl.text()
    assert "117.8" not in card._seed_summary_lbl.text()


def test_an_untypable_angle_is_no_longer_untypable(card):
    """Ties this to the +/-180 clamp: the two panels that could not be
    entered at all are the two this summary most needs to show."""
    card._seed_en_tx.setChecked(True)
    for tx in (296.885, 207.5):
        card._seed_tx.setValue(tx)
        assert f"{tx:g}" in card._seed_summary_lbl.text()


def test_a_tx_that_is_set_but_not_ticked_is_called_out(card):
    """The quietest way to get a wrong answer: type the angle, forget the
    tick, and the fit silently uses 0."""
    card._seed_tx.setValue(296.885)
    card._seed_en_tx.setChecked(False)
    assert card.seed_tx_value() == 0.0
    assert "NOT ticked" in card._seed_summary_lbl.text()


def test_tx_unset_says_so(card):
    card._seed_en_tx.setChecked(False)
    card._seed_tx.setValue(0.0)
    assert "not set" in card._seed_summary_lbl.text().lower()


def test_the_effective_tx_is_what_the_fit_will_use(card):
    card._seed_tx.setValue(27.3)
    card._seed_en_tx.setChecked(True)
    assert card.seed_tx_value() == pytest.approx(27.3)


# ── the page-level overview answers all four at once ───────────────────

def _page_with_panels(tmp_path, panels=(1, 2, 3, 4)):
    h5py = pytest.importorskip("h5py")
    from midas_gui.hydra_calib_page import HydraCalibrationPage
    for n in panels:
        with h5py.File(str(tmp_path / f"c_00001.ge{n}.h5"), "w") as f:
            f.create_dataset("exchange/data",
                             data=np.zeros((2, 32, 32), np.float32))
    page = HydraCalibrationPage()
    page._loader._set_path(str(tmp_path / f"c_00001.ge{min(panels)}.h5"))
    return page


def test_the_overview_names_every_panels_tx(app, tmp_path):
    page = _page_with_panels(tmp_path)
    for n, tx in ((1, 296.885), (2, 27.3), (3, 117.8), (4, 207.5)):
        page._cards[n]._seed_en_tx.setChecked(True)
        page._cards[n]._seed_tx.setValue(tx)
    page._update_panels_overview()
    text = page._panels_lbl.text()
    for n, tx in ((1, 296.885), (2, 27.3), (3, 117.8), (4, 207.5)):
        assert f"ge{n}" in text and f"{tx:g}" in text


def test_the_overview_says_how_many_panels_will_be_fitted(app, tmp_path):
    """Directly the reported confusion: one panel ticked, four fitted."""
    page = _page_with_panels(tmp_path)
    for n in (2, 3, 4):
        page._loader._status_lbls[n].setChecked(False)
    page._update_panels_overview()
    assert "Fitting 1 panel" in page._panels_lbl.text()


def test_an_excluded_panel_is_named_as_excluded(app, tmp_path):
    """Omitting it would make "left out on purpose" look like "not found"."""
    page = _page_with_panels(tmp_path)
    page._loader._status_lbls[3].setChecked(False)
    page._update_panels_overview()
    text = page._panels_lbl.text()
    assert "ge3 excluded" in text
    assert "Fitting 3 panels" in text


def test_the_running_panel_is_marked(app, tmp_path):
    page = _page_with_panels(tmp_path)
    page._update_panels_overview(running=2)
    text = page._panels_lbl.text()
    assert "fitting" in text
    assert text.index("ge2") < text.index("fitting") < text.index("ge3")


def test_the_card_header_names_the_panel_on_show(app, tmp_path):
    page = _page_with_panels(tmp_path)
    page._on_panel_changed("ge3")
    assert "ge3" in page._panel_hdr.text()


# ── pseudo-strain, in the units the beamline judges it in ──────────────

class _Res:
    """Stand-in for an AutoCalibrationResult's strain field."""
    def __init__(self, uE):
        self.post_residual_strain_uE = uE


def test_strain_is_reported_in_parts_per_1e4_as_well(app, tmp_path):
    """The working figure of merit from the old mpe_wf workflow is "a few
    parts in 1e-4"; the GUI reported microstrain only, so every comparison
    against that baseline needed a mental factor of 100."""
    page = _page_with_panels(tmp_path)
    page._cards[1].result = _Res(237.0)
    page._update_panels_overview()
    text = page._panels_lbl.text()
    assert "237 µε" in text
    assert "2.37e-4" in text


def test_every_panels_strain_is_on_one_line(app, tmp_path):
    """Judging a Hydra calibration means comparing panels, and the
    per-panel card shows one at a time."""
    page = _page_with_panels(tmp_path)
    for n, uE in ((1, 210.0), (2, 185.0), (3, 402.0), (4, 196.0)):
        page._cards[n].result = _Res(uE)
    page._update_panels_overview()
    text = page._panels_lbl.text()
    for uE in (210, 185, 402, 196):
        assert f"{uE} µε" in text, f"ge with {uE} µε missing from the overview"


def test_an_unfitted_panel_shows_no_strain(app, tmp_path):
    """Absent is not zero — a panel that has not run must not read as a
    perfect calibration."""
    page = _page_with_panels(tmp_path)
    page._update_panels_overview()
    assert "µε" not in page._panels_lbl.text()
