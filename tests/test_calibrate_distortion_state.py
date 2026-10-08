"""Saving and restoring the Distortion refine selection.

Reported from the beamline: unticking Distortion, saving the project and
reopening it brought the tick back on with all fifteen harmonics — the
constructor default, not the saved state. Three separate faults conspired,
one test class each.

Tabs are built per test rather than shared, because the bug only shows across
*instances*: a live tab already holds the right ``_dist_coeffs``, so a
round-trip through the same object hides the fact that it was never written
out. ``--forked`` gives each test its own process, and no test here builds
more than two tabs (three in one process has segfaulted before — see
tests/test_calibrate_panel_save.py).
"""
import pytest


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def make_tab(app):
    from midas_gui.tab_calibrate import CalibrationTab
    return CalibrationTab


def _saved(make_tab, checked, coeffs):
    """State from a tab where the user set the Distortion row by hand."""
    t = make_tab()
    t._ref_dist.setChecked(checked)
    t._dist_coeffs = set(coeffs)
    t._update_dist_label()
    return t.get_state()


class TestWorkspaceRoundTrip:
    def test_the_coefficient_selection_survives_into_a_fresh_tab(self, make_tab):
        state = _saved(make_tab, True, {"iso_R2", "a1", "phi1"})
        t = make_tab()
        assert len(t._dist_coeffs) == 15          # the default it must overwrite
        t.set_state(state)
        assert t._dist_coeffs == {"iso_R2", "a1", "phi1"}

    def test_an_unticked_distortion_row_stays_unticked(self, make_tab):
        state = _saved(make_tab, False, {"iso_R2", "a1", "phi1"})
        t = make_tab()
        assert t._ref_dist.isChecked()            # the default it must overwrite
        t.set_state(state)
        assert not t._ref_dist.isChecked()
        assert "Fixed:" in t._refine_summary_text()
        assert "Distortion" in t._refine_summary_text().split("Fixed:")[1]

    def test_the_checkbox_caption_is_resynced_not_left_stale(self, make_tab):
        """apply_dict_to_widgets restores ref_dist with signals blocked, so
        _update_dist_label — the only writer of the "(n/15)" caption — never
        fired and the caption contradicted the tick beside it."""
        state = _saved(make_tab, False, {"iso_R2", "a1", "phi1"})
        t = make_tab()
        assert t._dist_btn.text() == "15/15"
        t.set_state(state)
        assert t._dist_btn.text() == "0/15"

    def test_a_project_saved_before_this_key_existed_keeps_its_default(self, make_tab):
        state = _saved(make_tab, True, {"iso_R2"})
        state.pop("dist_coeffs")
        t = make_tab()
        t.set_state(state)
        assert len(t._dist_coeffs) == 15
        assert t._dist_btn.text() == "15/15"


_ATTEMPT = {"cfg": {"refine": {"Lsd": True, "BC": True, "Distortion": True,
                               "distortion_coeffs": ["a1", "iso_R2", "phi1"]}}}


class TestAttemptReplay:
    def test_the_attempt_restores_the_coefficients_it_actually_used(self, make_tab):
        t = make_tab()
        t.apply_project_calibration({"single": _ATTEMPT})
        assert t._ref_dist.isChecked()
        assert t._dist_coeffs == {"iso_R2", "a1", "phi1"}
        assert t._dist_btn.text() == "3/15"

    def test_a_restored_workspace_wins_over_the_staler_attempt(self, make_tab):
        """The workspace was saved at Ctrl+S; the attempt was recorded when the
        fit ran. Replaying the attempt over it reverted the user's later edits,
        which is how the unticked Distortion row came back ticked."""
        t = make_tab()
        t.set_state(_saved(make_tab, False, {"iso_R2"}))
        t.apply_project_calibration({"single": _ATTEMPT}, restore_fields=False)
        assert not t._ref_dist.isChecked()
        assert t._dist_coeffs == {"iso_R2"}

    def test_an_attempt_predating_the_selector_leaves_the_default_alone(self, make_tab):
        from midas_gui import project
        assert project.calib_attempt_dist_coeffs({"cfg": {"refine": {"Distortion": True}}}) is None
        t = make_tab()
        t.apply_project_calibration({"single": {"cfg": {"refine": {"Distortion": True}}}})
        assert len(t._dist_coeffs) == 15   # not narrowed to "refine nothing"

    def test_coefficients_are_read_back_sorted_and_as_strings(self):
        from midas_gui import project
        got = project.calib_attempt_dist_coeffs(
            {"cfg": {"refine": {"distortion_coeffs": {"phi1", "a1", "iso_R2"}}}})
        assert got == ["a1", "iso_R2", "phi1"]


class TestHydraPage:
    def test_the_shared_recipe_persists_its_coefficient_selection(self, app):
        from midas_gui.hydra_calib_page import HydraCalibrationPage
        a = HydraCalibrationPage()
        a._ref_dist.setChecked(False)
        a._dist_coeffs = {"iso_R2", "a1"}
        a._update_dist_label()
        state = a.get_state()

        b = HydraCalibrationPage()
        assert len(b._dist_coeffs) == 15
        b.set_state(state)
        assert b._dist_coeffs == {"iso_R2", "a1"}
        assert not b._ref_dist.isChecked()
        # Hydra still captions the checkbox; only the single-detector tab
        # moved the count onto its button, where the parameter table names
        # the row in column 0.
        assert b._ref_dist.text() == "Distortion (0/15)"


class TestRealProjectFile:
    """Through an actual project ``.h5``, not just the in-memory state dict.

    The dict-level tests above would still pass if the selection were saved as
    something JSON could not carry — a ``set`` is the obvious trap here, since
    that is what ``_dist_coeffs`` is in memory and what ``_refine_flags``
    already hands the backend. This walks the real path the GUI uses on
    Ctrl+S and File > Open Project.
    """

    def test_the_selection_survives_a_save_and_reopen(self, make_tab, tmp_path):
        from midas_gui import project

        proj = tmp_path / "dist.h5"
        project.create_project(proj, name="dist")

        a = make_tab()
        a._ref_dist.setChecked(False)
        a._dist_coeffs = {"iso_R2", "a1", "phi1"}
        a._update_dist_label()
        project.write_gui_workspace(proj, tabs={"Calibrate": a.get_state()})

        state, _sidecars = project.read_workspace_tab(proj, "Calibrate")
        assert state["dist_coeffs"] == ["a1", "iso_R2", "phi1"]   # a list, not a set

        b = make_tab()
        b.set_state(state)
        assert not b._ref_dist.isChecked()
        assert b._dist_coeffs == {"iso_R2", "a1", "phi1"}
        # The caption counts what the *run* would refine, so an unticked row
        # reads (0/15) while still holding the three underneath — re-tick it
        # and the selection is there, which is the whole point of saving it.
        assert b._dist_btn.text() == "0/15"
        b._ref_dist.setChecked(True)
        assert b._dist_btn.text() == "3/15"

    def test_a_ticked_subset_survives_too(self, make_tab, tmp_path):
        """The unticked case can pass for the wrong reason — nothing to refine
        either way. This one has to carry the subset through to be right."""
        from midas_gui import project

        proj = tmp_path / "dist2.h5"
        project.create_project(proj, name="dist2")

        a = make_tab()
        a._ref_dist.setChecked(True)
        a._dist_coeffs = {"iso_R2", "iso_R4"}
        a._update_dist_label()
        project.write_gui_workspace(proj, tabs={"Calibrate": a.get_state()})

        b = make_tab()
        b.set_state(project.read_workspace_tab(proj, "Calibrate")[0])
        assert b._ref_dist.isChecked()
        assert b._dist_coeffs == {"iso_R2", "iso_R4"}
        assert b._dist_btn.text() == "2/15"
        # and the run would honour it, not silently widen back out to 15
        assert b._refine_flags()["distortion_coeffs"] == {"iso_R2", "iso_R4"}
