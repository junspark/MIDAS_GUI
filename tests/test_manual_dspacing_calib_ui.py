"""Qt-level wiring test for the Calibrate tab's manual d-spacing ring-picking
fit mode (non-crystalline calibrants like AgBH) — exercises the
PICK_DSPACING pick mode, the live per-ring summary, the Fit button gating,
and that a fitted result flows through the same _on_done pipeline as a
regular midas-calibrate-v2 result (ring overlay uses the new `_d_list`
branch of `_predict_ring_radii`, not the CeO2 fallback).
"""
from types import SimpleNamespace

import pytest

# Each test here builds at least one full CalibrationTab (a pyqtgraph
# ImageView plus several PlotWidgets); a couple build a second one for a
# get_state()/set_state() round trip. Per .context/DECISIONS.md's
# pyqtgraph-teardown-crash entry (see test_hydra_calib_ui.py/test_project.py
# for the same fix), enough accumulated pyqtgraph instances across a whole
# pytest run's garbage collection reliably segfaults the interpreter — run
# each test in its own forked subprocess (pytest-forked) so that aborts only
# that subprocess, reported as a normal FAILED with signal info, instead of
# crashing the whole pytest run.
pytestmark = pytest.mark.forked


_QT_LOADED = False

# Bound by _load_qt() at fixture time, declared here so static analysis
# (and the pyflakes diff in the review recipe) can still resolve them.
QtCore = QtWidgets = None
_FakeManualDspacingCalibWorker = _FakeIntegrationWorker = None


def _load_qt():
    """Import Qt and the QObject-subclass fakes, publishing them as module
    globals.

    Deliberately NOT done at module level. pytest imports this module during
    collection, in the *parent* process, while pytest-forked (see
    ``pytestmark`` above) runs each test in a forked child. Importing PyQt5
    in the parent initialises macOS CoreFoundation, which a forked child may
    not use — all 17 tests then die with SIGSEGV ("The process has forked and
    you cannot use this CoreFoundation functionality safely") before their
    bodies run. Importing here means each child does its own first-time
    init, which is legal.

    The two fakes subclass ``QtCore.QObject`` and declare ``pyqtSignal``
    class attributes, so they cannot be defined at module scope either —
    that alone would force the import at collection time. They are defined
    here and published into globals() so the test bodies below can go on
    referring to them by bare name. See .context/STATE.md.
    """
    global _QT_LOADED
    if _QT_LOADED:
        return
    from PyQt5 import QtCore, QtWidgets

    class _FakeManualDspacingCalibWorker(QtCore.QObject):
        """No-op-thread fake: finishes on the next event-loop tick with a fixed
        known result, instead of actually running least_squares."""
        log_line = QtCore.pyqtSignal(str)
        finished = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, picks, wavelength_A, pxY, pxZ, seed, NY, NZ,
                     material_name, d_list, parent=None,
                     refine=None, tilt_seed=(0.0, 0.0, 0.0), bounds=None):
            super().__init__(parent)
            self.refine = refine
            self.bounds = bounds
            self._wavelength_A = wavelength_A
            self._pxY = pxY
            self._pxZ = pxZ
            self._NY = NY
            self._NZ = NZ
            self._material_name = material_name
            self._d_list = d_list
            self._refine = refine
            self._tilt_seed = tilt_seed

        def start(self):
            QtCore.QTimer.singleShot(0, self._finish)

        def isRunning(self) -> bool:
            return False

        def requestInterruption(self):
            pass

        def _finish(self):
            result = SimpleNamespace(
                Lsd=300000.0, BC_y=512.0, BC_z=498.0, tx=0.0, ty=0.0, tz=0.0,
                distortion={}, pxY=self._pxY, pxZ=self._pxZ or self._pxY,
                NrPixelsY=self._NY, NrPixelsZ=self._NZ,
                wavelength_A=self._wavelength_A, post_residual_strain_uE=None,
                _calibrant_name=self._material_name, _d_list=list(self._d_list))
            self.finished.emit(result)


    class _FakeIntegrationWorker(QtCore.QObject):
        """No-op-thread fake for the real ``IntegrationWorker`` that ``_on_done``
        kicks off automatically after any successful fit (manual or not) — the
        Calibrate tab auto-loads ``DEFAULT_CALIBRANT_TIF`` at construction, so
        ``_calib_image()`` is never None and a real background thread (importing
        torch/midas_calibrate_v2) would otherwise be left running past test/
        interpreter teardown."""
        log_line = QtCore.pyqtSignal(str)
        finished = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, result, image, dark, im_trans, r_bin, eta_bin, mask=None,
                     parent=None, bright=None, background=None, bright_mode="divide",
                     weighted=True):
            super().__init__(parent)
            self._result = result

        def start(self):
            QtCore.QTimer.singleShot(0, self._finish)

        def isRunning(self) -> bool:
            return False

        def requestInterruption(self):
            pass

        def _finish(self):
            import numpy as np
            r_axis = np.linspace(0, 100, 50)
            profile = np.ones_like(r_axis)
            eta_axis = np.linspace(-180, 180, 36)
            cake = np.ones((len(eta_axis), len(r_axis)))
            self.finished.emit({"r_axis_px": r_axis, "profile": profile,
                                "wavelength_A": self._result.wavelength_A,
                                "lsd_um": self._result.Lsd, "px_um": self._result.pxY,
                                "cake_2d": cake, "eta_axis_deg": eta_axis})

    _QT_LOADED = True
    globals().update(
        QtCore=QtCore, QtWidgets=QtWidgets,
        _FakeManualDspacingCalibWorker=_FakeManualDspacingCalibWorker,
        _FakeIntegrationWorker=_FakeIntegrationWorker,
    )


@pytest.fixture(scope="module")
def app():
    _load_qt()
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_manual_fit_pick_summary_and_button_gating(app, monkeypatch):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()

    # isVisible() reflects on-screen visibility (requires the window to be
    # shown); isHidden() reflects the explicit setVisible()/hide() state
    # independent of ancestors, which is what these mode-switch toggles set.
    idx = tab._cal.findText("CeO2")
    assert idx >= 0
    tab._cal.setCurrentIndex(idx)
    assert not tab._refc_card.isHidden()
    assert tab._manual_card.isHidden()

    idx = tab._cal.findText("AgBH (silver behenate)")
    assert idx >= 0
    tab._cal.setCurrentIndex(idx)
    # The Refine parameters card is now shared with the crystalline path (same
    # checkboxes drive both) — only the distortion/residual-map row, which the
    # manual point-pick fit has no forward model for, is hidden for it.
    assert not tab._refc_card.isHidden()
    assert tab._dist_row.isHidden()
    assert tab._build_rc.isHidden()
    assert not tab._manual_card.isHidden()
    assert tab._dsp_custom_ed.isHidden()
    assert not tab._run_btn.isEnabled()

    # ty/tz default on (shared with CeO2's defaults) would raise the minimum
    # pick count past 3 — uncheck them here so the pick-count assertions below
    # exercise the historical Lsd+BC-only 3-point gate; tilt-refinement's
    # higher threshold is covered separately.
    tab._ref_ty.setChecked(False)
    tab._ref_tz.setChecked(False)

    view = tab._img_view
    view._dsp_ring_spin.setValue(1)
    for x, y in [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0)]:
        view._add_dspacing_point(512.0 + x, 498.0 + y)

    # 3 points on a single ring already satisfies the ">=3 total" gate.
    assert tab._run_btn.isEnabled()
    assert "Ring 1" in tab._dsp_summary.text()
    assert "58.380" in tab._dsp_summary.text()

    view._dsp_ring_spin.setValue(2)
    for x, y in [(50.0, 0.0), (0.0, 50.0), (-50.0, 0.0)]:
        view._add_dspacing_point(512.0 + x, 498.0 + y)
    assert "Ring 2" in tab._dsp_summary.text()
    assert "29.190" in tab._dsp_summary.text()

    # A ring index beyond the material's d-spacing count is flagged, not silently dropped.
    view._dsp_ring_spin.setValue(20)
    view._add_dspacing_point(700.0, 700.0)
    assert "invalid" in tab._dsp_summary.text()


def test_manual_fit_min_picks_rises_with_more_refine_flags(app, monkeypatch):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()

    idx = tab._cal.findText("AgBH (silver behenate)")
    assert idx >= 0
    tab._cal.setCurrentIndex(idx)

    # Lsd+BC only: 3 free parameters, so 3 picks are enough. Set every flag
    # explicitly rather than relying on the d-spacing defaults — those are
    # deliberately BC-only now (see test_dspacing_calibrant_defaults_to_bc_only),
    # and this test is about the gate tracking the flags, not about the default.
    tab._ref_lsd.setChecked(True)
    tab._ref_bc.setChecked(True)
    tab._ref_tx.setChecked(False)
    tab._ref_ty.setChecked(False)
    tab._ref_tz.setChecked(False)
    tab._ref_wl.setChecked(False)
    assert tab._manual_min_picks() == 3

    view = tab._img_view
    view._dsp_ring_spin.setValue(1)
    for x, y in [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0)]:
        view._add_dspacing_point(512.0 + x, 498.0 + y)
    assert tab._run_btn.isEnabled()

    # Ticking tz adds a free parameter — the same 3 picks are no longer
    # enough, and the button gate must reflect that live.
    tab._ref_tz.setChecked(True)
    assert tab._manual_min_picks() == 4
    assert not tab._run_btn.isEnabled()

    # A 4th pick on the same ring satisfies the new threshold.
    view._add_dspacing_point(512.0, 498.0 + 40.0)
    assert tab._run_btn.isEnabled()


def test_manual_fit_result_flows_through_on_done_with_correct_rings(app, monkeypatch):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    from midas_gui.helpers import simulate_rings_from_dspacings
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()

    idx = tab._cal.findText("AgBH (silver behenate)")
    tab._cal.setCurrentIndex(idx)
    view = tab._img_view
    for ring_idx, r in ((1, 100.0), (2, 60.0)):
        view._dsp_ring_spin.setValue(ring_idx)
        for x, y in [(r, 0.0), (0.0, r), (-r, 0.0)]:
            view._add_dspacing_point(512.0 + x, 498.0 + y)

    assert tab._run_btn.isEnabled()
    tab._run_manual_fit()
    assert tab._worker is not None
    assert not tab._run_btn.isEnabled()   # disabled while running

    loop = QtCore.QEventLoop()
    tab.calibrationDone.connect(lambda *_: loop.quit())
    QtCore.QTimer.singleShot(2000, loop.quit)   # safety timeout
    loop.exec_()

    assert tab._result is not None
    assert tab._result._calibrant_name == "AgBH (silver behenate)"
    assert tab._save_json_btn.isEnabled()
    assert tab._save_ps_btn.isEnabled()
    assert tab._run_btn.isEnabled()   # re-enabled after the fake worker finishes

    d_list = sorted(tab._result._d_list, reverse=True)
    expected_radii = sorted({round(r["radius_px"], 3) for r in simulate_rings_from_dspacings(
        d_list, tab._result.wavelength_A, tab._result.Lsd, tab._result.pxY)})
    from midas_gui.helpers import _predict_ring_radii
    assert _predict_ring_radii(tab._result) == expected_radii


def _param_grid_labels(tab):
    """Text of every key-label QLabel currently laid out in the results grid."""
    grid = tab._param_grid
    labels = []
    for i in range(grid.count()):
        w = grid.itemAt(i).widget()
        if isinstance(w, QtWidgets.QLabel) and w.text().endswith(":"):
            labels.append(w.text())
    return labels


def test_manual_fit_marks_unrefined_geometry_params_as_fixed_in_results_grid(app, monkeypatch):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()

    idx = tab._cal.findText("AgBH (silver behenate)")
    tab._cal.setCurrentIndex(idx)
    # Only Lsd/BC/ty/tz refined; tx and Wavelength stay fixed. Set all six
    # explicitly — a d-spacing calibrant now defaults to BC-only, and this
    # test is about how the grid renders the flags, not what they default to.
    tab._ref_lsd.setChecked(True)
    tab._ref_bc.setChecked(True)
    tab._ref_tx.setChecked(False)
    tab._ref_wl.setChecked(False)
    tab._ref_ty.setChecked(True)
    tab._ref_tz.setChecked(True)

    view = tab._img_view
    for ring_idx, r in ((1, 100.0), (2, 60.0)):
        view._dsp_ring_spin.setValue(ring_idx)
        for x, y in [(r, 0.0), (0.0, r), (-r, 0.0)]:
            view._add_dspacing_point(512.0 + x, 498.0 + y)

    assert tab._run_btn.isEnabled()
    tab._run_manual_fit()

    loop = QtCore.QEventLoop()
    tab.calibrationDone.connect(lambda *_: loop.quit())
    QtCore.QTimer.singleShot(2000, loop.quit)
    loop.exec_()

    assert tab._result is not None
    labels = _param_grid_labels(tab)
    assert any(lbl.startswith("tx") and "(fixed)" in lbl for lbl in labels)
    assert any(lbl.startswith("Wavelength") and "(fixed)" in lbl for lbl in labels)
    assert not any(lbl.startswith("Lsd") and "(fixed)" in lbl for lbl in labels)
    assert not any(lbl.startswith("ty") and "(fixed)" in lbl for lbl in labels)
    assert not any(lbl.startswith("tz") and "(fixed)" in lbl for lbl in labels)


def test_dspacing_picks_round_trip_through_project_state(app, monkeypatch):
    """Regression test for a project/session save-load losing every
    manually-picked d-spacing point (reported as "project file does not
    save the calibration tab state accurately" — a saved-and-reloaded
    session showed "No points picked yet." even though points had been
    picked before saving, so re-running Fit could not reproduce the prior
    result or its ring overlay)."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()
    idx = tab._cal.findText("AgBH (silver behenate)")
    tab._cal.setCurrentIndex(idx)

    view = tab._img_view
    view._dsp_ring_spin.setValue(1)
    for x, y in [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0)]:
        view._add_dspacing_point(512.0 + x, 498.0 + y)
    view._dsp_ring_spin.setValue(2)
    for x, y in [(50.0, 0.0), (0.0, 50.0), (-50.0, 0.0)]:
        view._add_dspacing_point(512.0 + x, 498.0 + y)
    assert tab._run_btn.isEnabled()

    state = tab.get_state()

    tab2 = tab_calibrate_mod.CalibrationTab()
    assert tab2._img_view.dspacing_picks() == []
    tab2.set_state(state)

    assert tab2._cal.currentText() == "AgBH (silver behenate)"
    picks = tab2._img_view.dspacing_picks()
    assert sorted(p[2] for p in picks) == [1, 1, 1, 2, 2, 2]
    assert tab2._run_btn.isEnabled()   # a single click reproduces the result
    assert tab2._dsp_summary.text() == tab._dsp_summary.text()


def test_manual_fit_result_persists_through_project_state_without_rerunning_fit(app, monkeypatch):
    """The project/session state must round-trip the fitted ``_result``
    itself, not just the seed picks (see the pick-only round trip above) —
    reloading a saved project/session should immediately show the fitted
    rings/parameters again, with no need to click Fit a second time."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()
    idx = tab._cal.findText("AgBH (silver behenate)")
    tab._cal.setCurrentIndex(idx)

    view = tab._img_view
    for ring_idx, r in ((1, 100.0), (2, 60.0)):
        view._dsp_ring_spin.setValue(ring_idx)
        for x, y in [(r, 0.0), (0.0, r), (-r, 0.0)]:
            view._add_dspacing_point(512.0 + x, 498.0 + y)

    tab._run_manual_fit()
    loop = QtCore.QEventLoop()
    tab.calibrationDone.connect(lambda *_: loop.quit())
    QtCore.QTimer.singleShot(2000, loop.quit)
    loop.exec_()
    assert tab._result is not None

    state = tab.get_state()

    tab2 = tab_calibrate_mod.CalibrationTab()
    assert tab2._result is None
    tab2.set_state(state)

    assert tab2._result is not None
    assert tab2._result.Lsd == pytest.approx(tab._result.Lsd)
    assert tab2._result.BC_y == pytest.approx(tab._result.BC_y)
    assert tab2._result._calibrant_name == "AgBH (silver behenate)"
    assert tab2._result._d_list == tab._result._d_list
    assert tab2._save_json_btn.isEnabled()
    assert tab2._save_ps_btn.isEnabled()


# ── BC-only default + parameter limits (manual fit conditioning) ────────
#
# See tests/test_manual_fit_conditioning.py for the numerics these UI
# defaults exist to serve: on a d-spacing calibrant the picks are often a
# single short ring arc, where floating Lsd alongside BC is badly
# conditioned and tilt is not identifiable at all.


def _set_limit(tab, name, value, unit):
    """Tick one inline limit row and set its window."""
    cb, spin, combo = tab._limit_widgets[name]
    cb.setChecked(True); spin.setValue(value); combo.setCurrentText(unit)


def _flags(tab):
    return {k: tab._refine_box(k).isChecked() for k in tab._REFINE_BOXES}


def test_dspacing_calibrant_defaults_to_bc_only(app):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()

    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    assert _flags(tab) == {"Lsd": True, "BC": True, "tx": False,
                           "ty": True, "tz": True, "Wavelength": False}

    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    assert _flags(tab) == {"Lsd": False, "BC": True, "tx": False,
                           "ty": False, "tz": False, "Wavelength": False}
    summary = tab._refine_summary_lbl.text()
    assert "Refining: BC" in summary
    assert "Lsd" in summary.split("Fixed:")[1]
    # BC alone is 2 free parameters, so the 3-pick floor still governs.
    assert tab._manual_min_picks() == 3


def test_each_calibrant_kind_keeps_its_own_refine_flags(app):
    """A choice made on one kind of calibrant must survive a round trip
    through the other — otherwise switching to check something would
    silently reset the user's refinement scope."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    agbh = tab._cal.findText("AgBH (silver behenate)")
    ceo2 = tab._cal.findText("CeO2")

    tab._cal.setCurrentIndex(agbh)
    tab._ref_lsd.setChecked(True)                 # user opts Lsd back in
    tab._cal.setCurrentIndex(ceo2)
    assert _flags(tab)["ty"] is True              # crystalline set restored
    tab._ref_ty.setChecked(False)                 # and edited
    tab._cal.setCurrentIndex(agbh)
    assert _flags(tab)["Lsd"] is True             # d-spacing choice remembered
    assert _flags(tab)["ty"] is False
    tab._cal.setCurrentIndex(ceo2)
    assert _flags(tab)["ty"] is False             # crystalline edit remembered


def test_limits_column_is_shaped_per_calibrant_kind(app):
    """Both kinds bound their fit, at different granularity.

    ``CalibrationParams`` carries one window for both centre coordinates and
    one for both refined tilts, and never refines tx, so those rows are hidden
    for a crystalline calibrant and a distortion row (which the manual fit has
    no model for) takes their place.
    """
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()

    def shown():
        return {n for n, cells in tab._limit_row_cells.items()
                if not cells[2].isHidden()}

    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    assert shown() == {"Lsd", "BC_y", "ty", "wavelength_A", "distortion"}
    assert not tab._dist_row.isHidden()
    # Always applied, so the opt-in checkbox would state something false —
    # including on the distortion row, which used to keep a disabled box
    # visible purely to caption itself and so read as a live, ticked control.
    assert all(tab._limit_widgets[n][0].isHidden()
               for n in ("Lsd", "BC_y", "ty", "wavelength_A", "distortion"))
    assert tab._limit_name_lbls["distortion"].text() == "Distortion"
    assert not tab._limit_name_lbls["distortion"].isHidden()
    assert all(tab._limit_widgets[n][0].isChecked()
               for n in ("Lsd", "BC_y", "ty", "wavelength_A", "distortion"))

    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    assert shown() == {"Lsd", "BC_y", "BC_z", "ty", "tz", "tx", "wavelength_A"}
    assert tab._dist_row.isHidden()
    # The manual fit's own rows do get a live box, and BC_y/BC_z keep the
    # sub-labels that tell the two apart under the single "BC" refine box.
    assert not tab._limit_widgets["BC_y"][0].isHidden()
    assert tab._limit_name_lbls["BC_y"].text() == "BC_y"
    # ...and opt-in again, so an untouched card leaves the manual fit unbounded.
    assert not any(cb.isChecked() for cb, _s, _c in tab._limit_widgets.values())
    assert tab._limit_bounds() == (None, [])


def test_crystalline_limits_show_the_windows_actually_in_force(app):
    """The crystalline windows always apply, so the card is prefilled from the
    backend's own defaults and reports no override until one is edited."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    from midas_gui.calib import tol_defaults
    tab = tab_calibrate_mod.CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    d = tol_defaults()

    assert tab._limit_widgets["Lsd"][1].value() == pytest.approx(d["tolLsd"] / 1000.0)
    assert tab._limit_widgets["BC_y"][1].value() == pytest.approx(d["tolBC"])
    assert tab._limit_widgets["ty"][1].value() == pytest.approx(d["tolTilts"])
    # Untouched defaults ask the backend for nothing it would not already do,
    # which is what keeps run_pipeline on the plain calibrate() path.
    assert tab._crystalline_tols() is None
    assert "Always applied" in tab._limits_note.text()

    tab._manual_seed_check.setChecked(True)
    tab._seed_lsd.setValue(1000.0)
    _set_limit(tab, "Lsd", 2.0, "mm")
    tols = tab._crystalline_tols()
    assert tols is not None
    assert tols["tolLsd"] == pytest.approx(2000.0)        # mm entered, µm stored
    assert tols["tolBC"] == pytest.approx(d["tolBC"])     # untouched rows ride along


def test_seed_step_follows_the_limit_window(app):
    """A window states how far a value can sensibly move, so it is a better
    arrow step than a fixed constant."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    from midas_gui.constants import DEFAULT_STEP_LSD_MM
    tab = tab_calibrate_mod.CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    tab._manual_seed_check.setChecked(True)
    tab._seed_lsd.setValue(1000.0)

    _set_limit(tab, "Lsd", 5.0, "mm")
    # 10% of the full ±5 mm span.
    assert tab._seed_lsd.singleStep() == pytest.approx(1.0)
    # The merged crystalline BC window drives both centre boxes.
    _set_limit(tab, "BC_y", 20.0, "px")
    assert tab._seed_bcy.singleStep() == pytest.approx(4.0)
    assert tab._seed_bcz.singleStep() == pytest.approx(4.0)

    # No window in force -> back to the configured constant.
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    assert tab._seed_lsd.singleStep() == pytest.approx(DEFAULT_STEP_LSD_MM)


def test_limit_bounds_conversion_and_zero_value_guard(app):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    # _limit_bounds() feeds the manual d-spacing fit, which only runs for a
    # d-spacing calibrant — crystalline windows go through _crystalline_tols().
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    tab._manual_seed_check.setChecked(True)
    tab._seed_lsd.setValue(13500.0)     # mm in the UI, µm in the fit
    tab._seed_bcy.setValue(129.0)
    tab._seed_ty.setValue(0.0)

    assert tab._limit_bounds() == (None, [])            # nothing enabled
    assert "No limits set" in tab._limits_note.text()

    _set_limit(tab, "Lsd", 5.0, "%")
    _set_limit(tab, "BC_y", 50.0, "px")
    _set_limit(tab, "ty", 5.0, "%")                     # % of a 0 deg seed
    assert tab._n_limits_set() == 3

    b, skipped = tab._limit_bounds()
    assert skipped == []
    assert set(b) == {"Lsd", "BC_y", "ty"}              # 'off' rows excluded
    assert b["Lsd"] == pytest.approx((13500e3 * 0.95, 13500e3 * 1.05))
    assert b["BC_y"] == pytest.approx((79.0, 179.0))
    # A percentage of zero would pin the parameter exactly; it must not.
    assert b["ty"][0] < 0.0 < b["ty"][1]


def test_seed_relative_limits_are_dropped_when_the_manual_seed_is_off(app):
    """Without 'Use manual seed' the fit auto-seeds Lsd/BC from the picks, so
    a window centred on the (unused) seed spin boxes would bound the fit
    around a value it never starts from. Those rows are dropped and named;
    the wavelength, which is always live, is kept."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    tab._manual_seed_check.setChecked(False)
    tab._wl.setValue(0.173)
    _set_limit(tab, "Lsd", 5.0, "%")
    _set_limit(tab, "BC_y", 50.0, "px")
    _set_limit(tab, "wavelength_A", 1.0, "%")

    b, skipped = tab._limit_bounds()
    assert sorted(skipped) == ["BC_y", "Lsd"]
    assert set(b) == {"wavelength_A"}
    assert b["wavelength_A"] == pytest.approx((0.173 * 0.99, 0.173 * 1.01))

    # With the seed enabled they all come back.
    tab._manual_seed_check.setChecked(True)
    b, skipped = tab._limit_bounds()
    assert skipped == []
    assert set(b) == {"Lsd", "BC_y", "wavelength_A"}


def test_limits_reach_the_worker_and_survive_a_project_round_trip(app, monkeypatch):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    tab._manual_seed_check.setChecked(True)
    tab._seed_lsd.setValue(13500.0)
    _set_limit(tab, "Lsd", 5.0, "%")

    view = tab._img_view
    view._dsp_ring_spin.setValue(1)
    for x, y in [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0)]:
        view._add_dspacing_point(512.0 + x, 498.0 + y)
    assert tab._run_btn.isEnabled()
    tab._run_manual_fit()

    assert tab._worker.bounds is not None
    assert tab._worker.bounds["Lsd"] == pytest.approx((13500e3 * 0.95, 13500e3 * 1.05))
    assert tab._worker.refine["Lsd"] is False        # BC-only default still in force

    state = tab.get_state()
    tab2 = tab_calibrate_mod.CalibrationTab()
    tab2.set_state(state)
    assert tab2._limits["Lsd"] == {"on": True, "value": 5.0, "unit": "%"}
    assert tab2._n_limits_set() == 1
    assert _flags(tab2) == _flags(tab)
    assert tab2._cal.currentText() == "AgBH (silver behenate)"


def test_param_grid_annotates_refined_rows_with_their_uncertainty(app):
    """A converged fit is not the same as a determined one, so the results
    grid carries the 1-sigma next to each refined geometry value."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    from midas_gui.helpers import paramstest_pairs
    tab = tab_calibrate_mod.CalibrationTab()

    result = SimpleNamespace(
        Lsd=13500000.0, BC_y=129.0, BC_z=124.0, tx=0.0, ty=0.0, tz=0.0,
        distortion={}, pxY=55.0, pxZ=55.0, NrPixelsY=3072, NrPixelsZ=512,
        wavelength_A=0.173, _calibrant_name="AgBH (silver behenate)",
        _d_list=[58.38], im_trans=[])
    refine = {"Lsd": False, "BC": True, "tx": False, "ty": False, "tz": False,
              "Wavelength": False}
    sigma = {"Lsd": 0.0, "BC_y": 0.264, "BC_z": 0.868,
             "tx": 0.0, "ty": 0.0, "tz": 0.0, "wavelength_A": 0.0}
    tab._populate_param_grid(paramstest_pairs(result, selected=set()),
                             refine_flags=refine, sigma=sigma, at_limit=set())

    grid = tab._param_grid
    texts = [grid.itemAt(i).widget().text() for i in range(grid.count())
             if isinstance(grid.itemAt(i).widget(), QtWidgets.QLabel)]
    labels = [t for t in texts if t.endswith(":")]
    values = [t for t in texts if not t.endswith(":")]
    assert any(l.startswith("Lsd") and "(fixed)" in l for l in labels)
    assert any("± 0.264, 0.868" in v for v in values)     # the refined BC row
    # A held-fixed row carries no uncertainty.
    assert not any(v.startswith("13500000") and "±" in v for v in values)


def test_param_grid_marks_a_bound_pinned_row_instead_of_a_sigma(app):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    from midas_gui.helpers import paramstest_pairs
    tab = tab_calibrate_mod.CalibrationTab()
    result = SimpleNamespace(
        Lsd=13635000.0, BC_y=129.0, BC_z=124.0, tx=0.0, ty=0.0, tz=0.0,
        distortion={}, pxY=55.0, pxZ=55.0, NrPixelsY=3072, NrPixelsZ=512,
        wavelength_A=0.173, _calibrant_name="AgBH (silver behenate)",
        _d_list=[58.38], im_trans=[])
    refine = {"Lsd": True, "BC": True, "tx": False, "ty": False, "tz": False,
              "Wavelength": False}
    tab._populate_param_grid(
        paramstest_pairs(result, selected=set()), refine_flags=refine,
        sigma={"Lsd": 0.0, "BC_y": 0.3, "BC_z": 0.4, "tx": 0.0, "ty": 0.0,
               "tz": 0.0, "wavelength_A": 0.0},
        at_limit={"Lsd"})
    values = [tab._param_grid.itemAt(i).widget().text()
              for i in range(tab._param_grid.count())
              if isinstance(tab._param_grid.itemAt(i).widget(), QtWidgets.QLabel)]
    assert any("(at limit)" in v for v in values)


def _ring_extents(tab):
    """(y-extent, z-extent) of each drawn ring curve — a plain circle has equal
    extents, a tilt-distorted one does not.

    A curve's off-detector points are drawn as NaN (see
    ``helpers.ring_on_image_mask``), so the extent is taken over the finite
    (on-image) points only — otherwise a ring that partly leaves the frame
    would report an all-NaN extent instead of the visible arc's."""
    import numpy as np
    out = []
    for it in tab._ring_items:
        d = it.getData() if hasattr(it, "getData") else None
        if d and d[0] is not None and len(d[0]) > 10:
            x, y = d
            fin = np.isfinite(x) & np.isfinite(y)
            if fin.sum() > 10:
                out.append((float(np.ptp(x[fin])), float(np.ptp(y[fin]))))
    return out


def _curve_extents(tab, result):
    """(y-extent, z-extent) of each *unclipped* predicted ring curve — the
    underlying forward-model geometry (circular vs. tilt-stretched), not what
    actually lands on the detector image. ``_draw_rings`` clips its rendered
    items to the image (see ``helpers.ring_on_image_mask``), which is the
    right thing on screen but means a ring far larger than this test's tiny
    512x3072 detector would report a near-empty visible extent instead of
    the shape this test is actually checking."""
    import numpy as np
    import midas_gui.tab_calibrate as tab_calibrate_mod
    max_r = max(result.NrPixelsY, result.NrPixelsZ)
    radii = [r for r in tab_calibrate_mod._predict_ring_radii(result) if 0 < r < max_r]
    return [(float(np.ptp(ys)), float(np.ptp(zs))) for ys, zs in tab._ring_curves(result, radii)]


def test_seed_card_never_draws_rings(app):
    """The Calibrate tab overlays calibration results only. Previewing a
    hand-dialled geometry belongs to the Data Viewer's Ring simulation card;
    doing it here too meant ticking "Use manual seed" painted rings no
    calibration had endorsed, which reads as a calibrated overlay."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    tab._wl.setValue(0.1730); tab._pxY.setValue(55.0)
    tab._show_rings_check.setChecked(True)
    tab._manual_seed_check.setChecked(True)
    tab._seed_lsd.setValue(13500.0)
    tab._seed_bcy.setValue(129.0); tab._seed_bcz.setValue(124.0)
    assert tab._ring_items == []
    assert tab._ring_status.text() == ""

    # Only a result puts rings on the image — and editing the seed afterwards
    # leaves them where the fit put them.
    result = SimpleNamespace(
        Lsd=13500e3, BC_y=129.0, BC_z=124.0, tx=0.0, ty=0.0, tz=0.0,
        distortion={}, pxY=55.0, pxZ=55.0, NrPixelsY=3072, NrPixelsZ=512,
        wavelength_A=0.1730, _calibrant_name="AgBH (silver behenate)",
        _d_list=sorted([58.380 / n for n in range(1, 11)], reverse=True),
        im_trans=[])
    tab._draw_rings(result)
    rendered = _ring_extents(tab)
    assert rendered, "a fitted result should draw rings"
    # Circles, to within the 512-point sampling of the curve — checked on the
    # unclipped model, since these test rings are far larger than the tiny
    # detector and their *rendered* (image-clipped) extents are not circular
    # (see _curve_extents).
    for y_ext, z_ext in _curve_extents(tab, result):
        assert y_ext == pytest.approx(z_ext, rel=1e-3)

    tab._seed_ty.setValue(-82.0)
    assert _ring_extents(tab) == pytest.approx(rendered, rel=1e-9)


def test_fitted_tilt_reaches_the_overlay_with_the_seed_card_on(app):
    """"Use manual seed" used to divert the overlay through the seed fields.
    A result's own tilt must reach the image whether it is ticked or not."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    tab._wl.setValue(0.1730); tab._pxY.setValue(55.0)
    tab._show_rings_check.setChecked(True)
    tab._manual_seed_check.setChecked(True)
    tab._feedback_check.setChecked(True)

    tilted = SimpleNamespace(
        Lsd=13500e3, BC_y=129.0, BC_z=124.0, tx=0.0, ty=-82.0, tz=0.0,
        distortion={}, pxY=55.0, pxZ=55.0, NrPixelsY=3072, NrPixelsZ=512,
        wavelength_A=0.1730, _calibrant_name="AgBH (silver behenate)",
        _d_list=sorted([58.380 / n for n in range(1, 11)], reverse=True),
        im_trans=[])
    tab._result = tilted
    tab._seed_from_result(tilted)
    tab._draw_rings(tilted)

    # The tilt is both visible in the seed card and applied to the overlay.
    assert tab._seed_ty.value() == pytest.approx(-82.0)
    assert "ty=-82" in tab._seed_note.text()
    stretched = _curve_extents(tab, tilted)
    assert stretched[0][1] > 3 * stretched[0][0]            # stretched, not circular
    assert "tilt applied" in tab._ring_status.text()


def test_ring_labels_land_on_the_visible_arc(app):
    """The old anchor was the ring's twelve o'clock point, which is off-frame
    for every ring when the beam centre sits near the edge of a wide, short
    strip — so every label vanished."""
    import numpy as np
    from midas_gui.hydra_geometry_card import _ring_label_pos
    from midas_gui.helpers import simulate_rings_from_dspacings

    bc_y, bc_z, shape = 129.0, 124.0, (512, 3072)
    th = np.linspace(0, 2 * np.pi, 400)
    rings = simulate_rings_from_dspacings([58.380 / n for n in range(1, 11)],
                                          0.1730, 13.5e6, 55.0)
    on_image = 0
    for r in rings[:4]:
        rad = r["radius_px"]
        ys, zs = bc_y + rad * np.cos(th), bc_z + rad * np.sin(th)
        assert bc_z - rad < 0                       # old anchor: below the frame
        ly, lz, _anchor = _ring_label_pos(ys, zs, shape)
        if 0 <= ly < shape[1] and 0 <= lz < shape[0]:
            on_image += 1
    assert on_image == 4

    # A ring entirely off the frame gets no label at all: a label out in the
    # empty canvas beside the detector describes nothing the user can see.
    ys, zs = bc_y + 9000 * np.cos(th), bc_z + 9000 * np.sin(th)
    assert _ring_label_pos(ys, zs, shape) is None




# ── Calibrant-driven controls, compact Refine card, accurate rings ────────


def test_dspacing_pick_controls_only_appear_for_a_dspacing_calibrant(app):
    """"Pick d-spacing pts" and its "Ring #" selector tag a point with the ring
    it belongs to — a question only a d-spacing-list calibrant asks. Leaving
    them on the toolbar for CeO2 offered a workflow that leads nowhere."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    view = tab._img_view
    controls = (view._pick_dsp_btn, view._dsp_ring_lbl, view._dsp_ring_spin)

    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    assert all(w.isHidden() for w in controls)

    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    assert not any(w.isHidden() for w in controls)

    tab._cal.setCurrentIndex(tab._cal.findText("Custom d-spacings…"))
    assert not any(w.isHidden() for w in controls)


def test_leaving_a_dspacing_calibrant_cancels_an_active_pick_mode(app):
    """Hiding the button while it is checked would strand the viewer in
    PICK_DSPACING with no way to leave it — the next click on a CeO2 image
    would silently add a d-spacing point."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    tab._img_view._pick_dsp_btn.setChecked(True)
    assert tab._img_view._pick_mode == tab._img_view.PICK_DSPACING

    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    assert not tab._img_view._pick_dsp_btn.isChecked()
    assert tab._img_view._pick_mode != tab._img_view.PICK_DSPACING


def _grid_rows(grid):
    """{row: [widget, …]} for a QGridLayout, in column order."""
    rows = {}
    for i in range(grid.count()):
        it = grid.itemAt(i)
        r, c, _, _ = grid.getItemPosition(i)
        w = it.widget()
        if w is not None:
            rows.setdefault(r, []).append((c, w))
    return {r: [w for _c, w in sorted(cells)] for r, cells in rows.items()}


def test_refine_card_interleaves_each_flag_with_its_window(app):
    """One row per parameter: the "refine?" checkbox in column 0 and the +/-
    window bounding that same parameter on the rest of the line, so the two
    decisions about one parameter read together.

    This replaces an earlier compact 2x3 refine grid with the limits in a
    separate block below it. That block had nowhere to hang a per-parameter
    window, and it captioned the crystalline case "the MIDAS calibrate backend
    takes no bounds arguments" -- which is false: CalibrationParams.tol* become
    hard box constraints in midas_calibrate/param_vector.py:bounds().
    """
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    rows = _grid_rows(tab._refine_grid)
    # Derived, not restated: this card builds its rows from
    # PARAMETER_LIMIT_ROWS so that it, the Manual seed dialog and
    # _REFINE_BOXES cannot drift into different orders (they had). A fourth
    # hand-written copy here would just be the same trap one level out --
    # what this test is about is the INTERLEAVING, not the order, which
    # test_calibrate_seed_summary pins directly.
    from midas_gui.dialogs import PARAMETER_LIMIT_ROWS
    order = tuple(row[0] for row in PARAMETER_LIMIT_ROWS)
    # Header, one row per parameter, then the trailing note.
    assert sorted(rows) == list(range(len(order) + 2)), f"got {sorted(rows)}"
    assert rows[0] == [tab._limits_hdr]
    assert rows[len(order) + 1] == [tab._limits_note]

    for r, name in enumerate(order, start=1):
        assert tab._limit_row_index[name] == r, f"{name} on row {r}?"
        _cb, spin, combo = tab._limit_widgets[name]
        assert spin in rows[r] and combo in rows[r], \
            f"{name}'s window is not on its own row"

    # The refine flag leads its parameter's row. BC's single box frees both
    # centre coordinates, so it spans the pair and anchors on BC_y; distortion
    # and BC_z have no box of their own in this grid.
    for name, box in (("Lsd", tab._ref_lsd), ("BC_y", tab._ref_bc),
                      ("ty", tab._ref_ty), ("tz", tab._ref_tz),
                      ("tx", tab._ref_tx), ("wavelength_A", tab._ref_wl)):
        assert rows[tab._limit_row_index[name]][0] is box, f"{name} unflagged"

    # Distortion's refine box stays below the grid: it carries the "..."
    # button for the per-coefficient dialog, and Residual map is an output,
    # not a fit parameter, so neither belongs on a parameter row.
    bottom_rows = _grid_rows(tab._refine_grid_bottom)
    assert len(bottom_rows) == 1, f"expected 1 row, got {sorted(bottom_rows)}"
    assert bottom_rows[0] == [tab._dist_row, tab._build_rc]


def _row_of(w):
    """Grid row of ``w`` within whichever grid layout holds it, descending
    through nested layouts starting at its parent widget's own layout."""
    def walk(layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is w and isinstance(layout, QtWidgets.QGridLayout):
                return layout.getItemPosition(i)[0]
            sub = item.layout()
            if sub is not None:
                found = walk(sub)
                if found is not None:
                    return found
        return None
    row = walk(w.parentWidget().layout())
    assert row is not None, f"{w} not found in any grid under its parent"
    return row


def _grid_of(w):
    """The QGridLayout instance holding ``w`` directly, descending through
    nested layouts starting at its parent widget's own layout."""
    def walk(layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is w and isinstance(layout, QtWidgets.QGridLayout):
                return layout
            sub = item.layout()
            if sub is not None:
                found = walk(sub)
                if found is not None:
                    return found
        return None
    grid = walk(w.parentWidget().layout())
    assert grid is not None, f"{w} not found in any grid under its parent"
    return grid


def _layout_of(w):
    """The immediate QLayout holding ``w``, descending through nested
    layouts starting at its parent widget's own layout."""
    def walk(layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is w:
                return layout
            sub = item.layout()
            if sub is not None:
                found = walk(sub)
                if found is not None:
                    return found
        return None
    layout = walk(w.parentWidget().layout())
    assert layout is not None, f"{w} not found in any layout under its parent"
    return layout


def test_advanced_card_packs_three_controls_per_row(app):
    """E-M iters/LM iters share one tightly-packed line; Device sits on its
    own line below. Each is a plain QHBoxLayout (with a trailing stretch)
    rather than a Form()/QGridLayout row: giving a fixed-width field its own
    stretched grid column left a wide gap before the next label instead of
    packing the fields close together."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    assert _layout_of(tab._n_iter) is _layout_of(tab._lm_iter)
    assert _layout_of(tab._device) is not _layout_of(tab._n_iter)


def test_manual_seed_dialog_lays_out_one_row_per_parameter(app):
    """The per-parameter seed panel (ManualSeedDialog, behind the "Manual
    seed…" button): BC_y and BC_z share the "Beam centre" row (the backend
    only takes them as a pair — see calib._resolve_seed); Lsd/tx/ty/tz each
    get their own row so each can be ticked independently."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()

    assert tab._seed_bcy.parentWidget() is tab._seed_dialog
    assert _row_of(tab._seed_bcy) == _row_of(tab._seed_bcz)
    for w in (tab._seed_lsd, tab._seed_tx, tab._seed_ty, tab._seed_tz):
        assert _row_of(w) not in (_row_of(tab._seed_bcy),)
    rows = {_row_of(w) for w in
            (tab._seed_lsd, tab._seed_tx, tab._seed_ty, tab._seed_tz)}
    assert len(rows) == 4, "Lsd/tx/ty/tz must each get their own row"


def test_rings_account_for_distortion_with_no_toggle_to_find(app):
    """The overlay is always the full forward model. Previously the honest
    rings lived behind a "Corrected" tick that defaulted off — and even ticked,
    it applied tilt only, so a refined-distortion calibration still drew rings
    that sat off the measured ones.
    """
    import numpy as np
    import midas_gui.tab_calibrate as tab_calibrate_mod

    tab = tab_calibrate_mod.CalibrationTab()
    assert not hasattr(tab, "_corrected_check")
    tab._show_rings_check.setChecked(True)

    common = dict(Lsd=1_000_000.0, BC_y=1024.0, BC_z=1024.0, tx=0.0, ty=0.0, tz=0.0,
                  pxY=200.0, pxZ=200.0, NrPixelsY=2048, NrPixelsZ=2048,
                  wavelength_A=0.1729, _calibrant_name="CeO2", im_trans=[])

    def _curves(**over):
        tab._result = None
        tab._manual_seed_check.setChecked(False)
        tab._draw_rings(SimpleNamespace(**dict(common, **over)))
        return [it.getData() for it in tab._ring_items
                if it.getData()[0] is not None and len(it.getData()[0]) > 10]

    plain = _curves(distortion={})
    assert plain, "a flat, undistorted geometry should still draw rings"

    distorted = _curves(distortion={"iso_R2": 5e-3, "a2": 4e-3, "phi2": 30.0})
    assert len(distorted) == len(plain)
    shifts = [float(np.abs(np.hypot(dy - 1024.0, dz - 1024.0)
                           - np.hypot(py - 1024.0, pz - 1024.0)).max())
              for (py, pz), (dy, dz) in zip(plain, distorted)]
    assert max(shifts) > 1.0, (
        f"distortion moved the drawn rings by at most {max(shifts):.3f} px — "
        "it is not reaching the overlay")

    assert "distortion applied" in tab._ring_status.text()


def test_ring_status_flags_the_residual_map_it_cannot_draw(app):
    """The one term the overlay leaves out says so, rather than going missing."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    tab._show_rings_check.setChecked(True)
    tab._manual_seed_check.setChecked(False)
    tab._draw_rings(SimpleNamespace(
        Lsd=1_000_000.0, BC_y=1024.0, BC_z=1024.0, tx=0.0, ty=0.4, tz=0.0,
        pxY=200.0, pxZ=200.0, NrPixelsY=2048, NrPixelsZ=2048, wavelength_A=0.1729,
        distortion={}, _calibrant_name="CeO2", im_trans=[],
        residual_corr_bin_path="/tmp/resid.bin"))
    assert "tilt applied" in tab._ring_status.text()
    assert "residual map not drawn" in tab._ring_status.text()


def test_reloaded_result_keeps_its_distortion_for_the_next_seed(app):
    """The seed card has no distortion widgets to be restored from project
    fields, so the coefficients have to be carried across explicitly — else
    re-running the fit from a reloaded project seeds it with none, quietly
    discarding the harmonics the stored result was refined with."""
    import midas_gui.tab_calibrate as tab_calibrate_mod
    tab = tab_calibrate_mod.CalibrationTab()
    assert tab._seed_dist == {}
    tab._display_stored_result(SimpleNamespace(
        Lsd=1_000_000.0, BC_y=1024.0, BC_z=1024.0, tx=0.0, ty=0.0, tz=0.0,
        pxY=200.0, pxZ=200.0, NrPixelsY=2048, NrPixelsZ=2048, wavelength_A=0.1729,
        distortion={"iso_R2": 5e-3}, _calibrant_name="CeO2", im_trans=[],
        post_residual_strain_uE=None), reintegrate_if_missing=False)
    assert tab._seed_dist == {"iso_R2": 5e-3}
