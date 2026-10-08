"""The crystalline ± window is a per-iteration trust region, not a bound.

Reported from the beamline on CeO2 at 1-ID: "Lsd was set to 2390mm with
+/-2 mm but the value converged to 2382 mm."  The window was applied --
midas_calibrate_v2 ``pipelines/single.py`` re-centres each parameter's
bounds on that iteration's result before the next LM call ("Bounds move
with the value"), so a fit that rails at its window walks one full width
per E-M iteration.  With the tab's default 4 iterations that is 8 mm, and
the observed answer was 2382.051 mm against a 2390.051 mm seed -- the
predicted floor to the micron.

Nothing in the GUI was wrong except what it *said*: "Always applied,
centred on the seed".  These tests pin the honest wording, because the
mismatch is the whole bug -- a reader who believes the note concludes the
limits are broken and stops trusting the card.

``forked`` per .context/DECISIONS.md -- builds a CalibrationTab (pyqtgraph).
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
    t = CalibrationTab()
    t._cal.setCurrentText("CeO2")
    return t


def _set_lsd_window(tab, mm):
    cb, spin, combo = tab._limit_widgets["Lsd"]
    combo.setCurrentText("mm")
    spin.setValue(mm)
    cb.setChecked(True)


def test_the_note_no_longer_claims_the_window_is_centred_on_the_seed(tab):
    """The one phrase that made a correct fit look broken."""
    txt = tab._limits_note.text()
    assert "centred on the seed" not in txt
    # ...but these windows really are unconditional, unlike the manual fit's,
    # so that half of the sentence has to survive.
    assert txt.startswith("Always applied")
    assert "re-centred every E-M iteration" in txt


def test_the_note_states_the_envelope_the_run_can_actually_reach(tab):
    """+-2 mm x 4 iterations = +-8 mm, which is what the beamline saw."""
    tab._n_iter.setValue(4)
    _set_lsd_window(tab, 2.0)
    txt = tab._limits_note.text()
    assert "Lsd ±2 mm" in txt          # per iteration, still quoted
    assert "4× this from the seed" in txt
    assert "Lsd ±8 mm" in txt          # ...and the total, spelled out


def test_the_envelope_follows_the_iteration_count(tab):
    """The drift is n_iter x tol, so the note is stale unless it tracks the
    Advanced spin box -- which lives in a collapsed group two cards away."""
    _set_lsd_window(tab, 2.0)
    tab._n_iter.setValue(10)
    assert "Lsd ±20 mm" in tab._limits_note.text()
    tab._n_iter.setValue(1)
    txt = tab._limits_note.text()
    # A single iteration cannot drift, so promising an envelope would be its
    # own small lie.
    assert "from the seed" not in txt
    assert "Lsd ±2 mm" in txt


def test_the_run_log_still_gets_the_window_list_without_the_prefix(tab):
    """The log line re-uses the note's tail; renaming the prefix must not
    leave the prefix embedded in it."""
    _set_lsd_window(tab, 2.0)
    line = tab._limits_note.text().replace(tab._XTAL_NOTE_PREFIX, "")
    assert not line.startswith("Always applied")
    assert line.startswith("Lsd ±2 mm")


# -- the at-limit warning must survive "Feed result back to seed" ------------

def _armed(tab):
    """A tab set up the way the beamline run was: CeO2, seeded, Lsd +-2 mm."""
    tab._manual_seed_check.setChecked(True)
    tab._seed_lsd.setValue(2390.051)
    tab._seed_bcy.setValue(2282.4); tab._seed_bcz.setValue(2115.6)
    tab._seed_ty.setValue(0.23); tab._seed_tz.setValue(0.11)
    _cb, spin, combo = tab._limit_widgets["Lsd"]
    combo.setCurrentText("mm"); spin.setValue(2.0)
    tab._capture_limit_ctx(tab._crystalline_tols())
    return tab


def _result(tab, **over):
    from types import SimpleNamespace
    base = dict(Lsd=2_390_051.0, BC_y=2282.4, BC_z=2115.6, ty=0.23, tz=0.11,
                wavelength_A=tab._wl.value())
    base.update(over)
    return SimpleNamespace(**base)


def test_the_run_that_walked_eight_millimetres_is_flagged(tab):
    """attempt_0016: seed 2390.051 mm, +-2 mm, converged 2382.051 mm.

    Four E-M iterations x 2 mm, railing every round. The answer is the
    limit's, not the data's, and the log said nothing.
    """
    _armed(tab)
    assert tab._crystalline_at_limit(_result(tab)) == set()
    assert tab._crystalline_at_limit(_result(tab, Lsd=2_382_051.0)) == {"Lsd"}


def test_feeding_the_result_back_does_not_silence_the_warning(tab):
    """The actual defect. ``_on_done`` applies the feedback BEFORE it asks
    whether the fit railed, and the question used to be put to the seed
    boxes -- which the feedback had just overwritten with this very result.
    Result vs itself is 0, so the check could never fire, and the default is
    for feedback to be on.
    """
    _armed(tab)
    bad = _result(tab, Lsd=2_382_051.0)
    assert tab._crystalline_at_limit(bad) == {"Lsd"}

    # ...now the feedback lands, exactly as _on_done does it.
    tab._feedback_check.setChecked(True)
    tab._seed_from_result(bad)
    assert tab._seed_lsd.value() == pytest.approx(2382.051)   # seed is gone

    assert tab._crystalline_at_limit(bad) == {"Lsd"}


def test_the_window_width_is_read_from_the_run_not_the_card(tab):
    """Same trap one step over: widening the window after the fact must not
    retroactively declare the finished run healthy."""
    _armed(tab)
    bad = _result(tab, Lsd=2_382_051.0)
    _cb, spin, _combo = tab._limit_widgets["Lsd"]
    spin.setValue(50.0)                      # user loosens the card afterwards
    assert tab._crystalline_at_limit(bad) == {"Lsd"}


# -- the hard cap -----------------------------------------------------------

def _tol_lsd(tab):
    from midas_gui.calib import tol_defaults
    return {**tol_defaults(), **(tab._crystalline_tols() or {})}["tolLsd"]


def test_the_cap_divides_the_window_by_the_iteration_count(tab):
    """The Hydra constraint: four panels share one frame, so a panel's Lsd
    cannot really wander. "+-5 mm" has to mean 5 mm total, and the only exact
    way to get that from a window the backend re-centres each round is to
    hand it 1/n of the cap.
    """
    cb, spin, combo = tab._limit_widgets["Lsd"]
    combo.setCurrentText("mm"); spin.setValue(5.0); cb.setChecked(True)
    tab._n_iter.setValue(4)

    assert _tol_lsd(tab) == pytest.approx(5000.0)        # uncapped: per round
    tab._hard_cap.setChecked(True)
    assert _tol_lsd(tab) == pytest.approx(1250.0)        # capped: per round
    # ...which is the point: n rounds of 1.25 mm is the 5 mm that was asked for.
    assert _tol_lsd(tab) * tab._n_iter.value() == pytest.approx(5000.0)

    tab._n_iter.setValue(10)
    assert _tol_lsd(tab) == pytest.approx(500.0)
    assert _tol_lsd(tab) * 10 == pytest.approx(5000.0)


def test_the_capped_note_quotes_the_cap_not_the_divided_window(tab):
    """The user typed 5 mm and must keep reading 5 mm; the 1.25 is an
    implementation detail and is labelled as one."""
    cb, spin, combo = tab._limit_widgets["Lsd"]
    combo.setCurrentText("mm"); spin.setValue(5.0); cb.setChecked(True)
    tab._n_iter.setValue(4)
    tab._hard_cap.setChecked(True)
    txt = tab._limits_note.text()
    assert txt.startswith("Hard cap on the whole run: ")
    assert "Lsd ±5 mm" in txt
    assert "±1.25 mm per iteration" in txt
    assert "can drift" not in txt          # the uncapped warning must be gone


def test_the_cap_is_hidden_for_a_manual_dspacing_calibrant(tab):
    """That path is a single bounded LM solve -- its window already bounds
    the answer, so offering to cap it would imply a defect that isn't there."""
    assert tab._hard_cap.isVisible() or not tab.isVisible()
    agbh = next(tab._cal.itemText(i) for i in range(tab._cal.count())
                if "AgBH" in tab._cal.itemText(i))
    tab._cal.setCurrentText(agbh)
    assert not tab._hard_cap.isVisibleTo(tab)
    tab._cal.setCurrentText("CeO2")
    assert tab._hard_cap.isVisibleTo(tab)


def test_under_a_cap_only_the_cap_counts_as_at_limit(tab):
    """Uncapped, one window away means the fit railed once -- worth saying.
    Capped, the per-round window is deliberately 1/n of the cap, so railing
    a round or two is just how the fit crosses the allowed span. Warning
    there would cry wolf on every healthy capped run.
    """
    _armed(tab)                       # Lsd +-2 mm, seed 2390.051 mm
    tab._n_iter.setValue(4)
    tab._hard_cap.setChecked(True)
    tab._capture_limit_ctx(tab._crystalline_tols())

    # 3 mm out: past a single 0.5 mm round, well inside the 2 mm cap.
    assert tab._crystalline_at_limit(_result(tab, Lsd=2_391_051.0)) == set()
    # 2 mm out: the cap itself.
    assert tab._crystalline_at_limit(_result(tab, Lsd=2_388_051.0)) == {"Lsd"}


def test_the_cap_survives_a_project_round_trip(tab, app):
    """A capped calibration that reopens uncapped would quietly widen the
    geometry constraint by 4x."""
    from midas_gui.tab_calibrate import CalibrationTab
    tab._hard_cap.setChecked(True)
    state = tab.get_state()
    other = CalibrationTab()
    assert not other._hard_cap.isChecked()
    other.set_state(state)
    assert other._hard_cap.isChecked()
