"""Offscreen UI test for the Calibrate tab's Hydra page: mode wiring, panel
switching (incl. the Pick BC/Pick Ring cross-panel leak regression, same
shape as `eadb14d`'s fix for the Data Viewer's Hydra page), and Sequential/
Parallel run orchestration.

Each ``HydraCalibrationPage`` builds 1 ``PickableImageViewer`` (pg.ImageView)
plus 5 pg.PlotWidgets (4 ``ResidualBarChart`` + 1 ``HydraProfileViewer``) —
enough pyqtgraph instances per page that, per .context/DECISIONS.md's
pyqtgraph-teardown-crash entry, building one per test function reliably
segfaults the whole file's run (reproduced while writing this file: 2 tests
in, then a hard abort in teardown). Folded into just 2 test functions, each
building exactly ONE page and reusing it for every assertion, mirroring
that entry's fix for the Data Viewer's own Hydra UI tests.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import h5py

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "test_data" / "gui_synthetic" / "hydra"

# Run each test in this file in its own forked subprocess (pytest-forked):
# the pyqtgraph-teardown segfault documented above and in .context/STATE.md
# then aborts only that subprocess, reported as a normal FAILED with signal
# info, instead of crashing the whole pytest run.
pytestmark = pytest.mark.forked


_QT_LOADED = False

# Bound by _load_qt() at fixture time, declared here so static analysis
# (and the pyflakes diff in the review recipe) can still resolve them.
QtCore = QtWidgets = None
hydra_calib_page_mod = project = None
geometry_fields_from_file = HydraCalibrationPage = None
_FakeWorker = _FakeCalibrationWorker = _FakeIntegrationWorker = None
SCRATCH_DIRNAME = None


def _load_qt():
    """Import Qt, the GUI modules under test, and the QObject-subclass fakes,
    publishing them all as module globals.

    Deliberately NOT done at module level. pytest imports this module during
    collection, in the *parent* process, while pytest-forked runs each test
    in a forked child. Importing PyQt5 in the parent initialises macOS
    CoreFoundation, which a forked child may not use — every test then dies
    with SIGSEGV ("The process has forked and you cannot use this
    CoreFoundation functionality safely") before its body runs. Importing
    here means each child does its own first-time init, which is legal.

    The fakes below subclass ``QtCore.QObject``, so they cannot be defined at
    module scope either — that alone would force the import at collection
    time. See .context/STATE.md.
    """
    global _QT_LOADED
    if _QT_LOADED:
        return
    from PyQt5 import QtCore, QtWidgets
    import midas_gui.hydra_calib_page as hydra_calib_page_mod
    from midas_gui import project
    from midas_gui.helpers import geometry_fields_from_file, SCRATCH_DIRNAME
    from midas_gui.hydra_calib_page import HydraCalibrationPage

    class _FakeWorker(QtCore.QObject):
        """Shared no-op-thread shape for both CalibrationWorker and
        IntegrationWorker fakes: finishes on the next event-loop tick instead of
        a real background thread. The synthetic Hydra fixture wasn't built to
        produce enough calibrant rings for a real fit to converge (verified
        directly against midas_gui.calib — a pre-existing fixture/data
        characteristic, not a page bug), so these tests exercise the GUI's
        sequencing/routing logic rather than the fit numerics."""
        log_line = QtCore.pyqtSignal(str)
        finished = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def start(self):
            QtCore.QTimer.singleShot(0, self._finish)

        def isRunning(self) -> bool:
            return False

        def requestInterruption(self):
            pass


    class _FakeCalibrationWorker(_FakeWorker):
        def __init__(self, mode, image, dark, cfg, parent=None, bright=None, background=None,
                     bright_mode="divide", capture_stdout=True):
            super().__init__(parent)
            self._cfg = cfg

        def _finish(self):
            seed = self._cfg.get("manual_seed") or {}
            result = SimpleNamespace(
                BC_y=seed.get("BC_y", 128.0), BC_z=seed.get("BC_z", 128.0),
                Lsd=seed.get("Lsd", 1_000_000.0), wavelength_A=self._cfg["wavelength"],
                pxY=self._cfg["pxY"], pxZ=self._cfg["pxY"], NrPixelsY=256, NrPixelsZ=256,
                tx=0.0, ty=0.0, tz=0.0, distortion={}, post_residual_strain_uE=0.0)
            self.finished.emit(result)


    class _FakeIntegrationWorker(_FakeWorker):
        def __init__(self, result, image, dark, im_trans, r_bin, eta_bin, mask=None, parent=None,
                     bright=None, background=None, bright_mode="divide", weighted=True):
            super().__init__(parent)
            self._result = result

        def _finish(self):
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
        hydra_calib_page_mod=hydra_calib_page_mod,
        project=project,
        geometry_fields_from_file=geometry_fields_from_file,
        SCRATCH_DIRNAME=SCRATCH_DIRNAME,
        HydraCalibrationPage=HydraCalibrationPage,
        _FakeWorker=_FakeWorker,
        _FakeCalibrationWorker=_FakeCalibrationWorker,
        _FakeIntegrationWorker=_FakeIntegrationWorker,
    )


@pytest.fixture(scope="module")
def app():
    _load_qt()
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _qt_teardown(app):
    yield
    app.processEvents()


@pytest.fixture()
def fixture_available():
    if not FIXTURE_DIR.exists():
        pytest.skip("test_data/gui_synthetic/hydra/ fixture not present — "
                    "run make_hydra_test_data.py")
    return FIXTURE_DIR


@pytest.fixture(autouse=True)
def _stub_workers(app, monkeypatch):   # `app` so _load_qt() has run first
    monkeypatch.setattr(hydra_calib_page_mod, "CalibrationWorker", _FakeCalibrationWorker)
    monkeypatch.setattr(hydra_calib_page_mod, "IntegrationWorker", _FakeIntegrationWorker)


def _load_and_seed(page: HydraCalibrationPage, fixture: Path):
    page._loader.set_path(str(fixture / "ge1" / "panel.ge1.h5"))
    last = None
    for n in (1, 2, 3, 4):
        g = geometry_fields_from_file(str(fixture / f"ge{n}" / f"ps_ge{n}.txt"))
        page._cards[n].seed_from_geometry(g)
        last = g
    page._wl.setValue(last["wavelength_A"])
    page._pxY.setValue(last["pxY"])


def _pump(app, condition, max_iters=200):
    for _ in range(max_iters):
        if condition():
            return True
        app.processEvents()
    return False


def test_hydra_calib_page_wiring_and_pick_isolation(app, fixture_available):
    """Toolbar has no Composite button, siblings load correctly, and Pick
    BC/Pick Ring in-progress state on the shared viewer is cleared when the
    active panel switches (the eadb14d regression shape)."""
    page = HydraCalibrationPage()
    assert "composite" not in page._toolbar._buttons
    _load_and_seed(page, fixture_available)
    assert set(page._loader.siblings()) == {1, 2, 3, 4}

    page._toolbar.set_current("ge1")
    page._img_view._ring_pts.append((1.0, 2.0))
    page._toolbar.set_current("ge2")
    assert page._img_view._ring_pts == []
    assert page._active_card.panel_number == 2

    # "Use manual seed" / "Feed result back to seed" are one shared choice
    # across all 4 panels; the seed VALUES stay independent (already set
    # per-panel above by _load_and_seed via seed_from_geometry).
    for card in page._cards.values():
        assert card._manual_seed_check.isChecked()   # seed_from_geometry sets it
    page._cards[2]._manual_seed_check.setChecked(False)
    assert all(not page._cards[n]._manual_seed_check.isChecked() for n in (1, 2, 3, 4))
    page._cards[3]._manual_seed_check.setChecked(True)
    assert all(page._cards[n]._manual_seed_check.isChecked() for n in (1, 2, 3, 4))
    # The synthetic fixture's ps_ge{1..4}.txt files share one nominal BC by
    # design; give each panel a distinct seed value here to prove the shared
    # checkbox above didn't also link the VALUE fields.
    for n in (1, 2, 3, 4):
        page._cards[n]._seed_bcy.setValue(100.0 + n)
    bcs = {n: page._cards[n]._seed_bcy.value() for n in (1, 2, 3, 4)}
    assert len(set(bcs.values())) == 4, "seed VALUES must stay independent per panel"

    page._cards[4]._feedback_check.setChecked(False)
    assert all(not page._cards[n]._feedback_check.isChecked() for n in (1, 2, 3, 4))


def test_hydra_calib_run_orchestration_and_results_switching(app, fixture_available, tmp_path):
    """Sequential run: independent per-panel BCs. Parallel run: all 4
    workers started up-front. Results/Ring-Residuals tabs switch with the
    active panel. All against ONE page instance (see module docstring).
    Also verifies each panel's completed run is logged to a project file
    when one is open (FAIR provenance — see midas_gui/project.py)."""
    page = HydraCalibrationPage()
    proj_path = str(tmp_path / "proj.h5")
    project.create_project(proj_path)
    page.set_project_context(project.ProjectContext())
    page._project_ctx.path = proj_path
    _load_and_seed(page, fixture_available)
    # Pin the working directory rather than leaning on whatever the autofill
    # ladder derives from the fixture tree — this test is about scratch
    # separation, not about where the default comes from.
    work = tmp_path / "wd"
    work.mkdir()
    page._out_ed.setText(str(work))
    # The synthetic fixture's ps_ge{1..4}.txt files all share the same
    # nominal BC (128, 128) by design (built for geometry/composite tests,
    # not distinct panel positions) — give each panel a distinct seed BC
    # here so the fake worker's echoed-back result can actually distinguish
    # "routed to the right panel" from "coincidentally identical seed data".
    for n in (1, 2, 3, 4):
        page._cards[n]._seed_bcy.setValue(100.0 + n)
        page._cards[n]._seed_bcz.setValue(100.0 + n)

    page._run_mode_combo.setCurrentIndex(0)   # Sequential
    page._run_all()
    ok = _pump(app, lambda: not page._workers and not page._pending_panels)
    assert ok, "sequential run did not complete"
    results = {n: page._cards[n].result for n in (1, 2, 3, 4)}
    assert all(r is not None for r in results.values())
    bcs = {n: (r.BC_y, r.BC_z) for n, r in results.items()}
    assert len(set(bcs.values())) > 1, "panels should not all share the same fitted BC"
    assert page._run_btn.isEnabled() and not page._abort_btn.isEnabled()

    with h5py.File(proj_path, "r") as f:
        for n in (1, 2, 3, 4):
            att = f[f"analysis/calibrate/ge{n}/attempt_0001"]
            assert att.attrs["BC_y"] == pytest.approx(bcs[n][0])
    for n in (1, 2, 3, 4):
        assert getattr(results[n], "_project_attempt_ref", None) == f"/analysis/calibrate/ge{n}/attempt_0001"

    # Each panel's completed integration (fired from _on_done, one event-loop
    # tick after the calibration itself) also populates its own Eta vs R
    # cake view (built from IntegrationWorker's return_cake=True output).
    ok = _pump(app, lambda: not page._int_workers)
    assert ok, "integration did not complete for all panels"
    for n in (1, 2, 3, 4):
        assert page._cake_views[n]._cake is not None

    page._toolbar.set_current("ge1")
    assert page._results_stack.currentWidget() is page._cards[1].results_widget
    assert page._resid_stack.currentWidget() is page._cards[1].residual_chart
    assert page._cake_stack.currentWidget() is page._cake_views[1]
    page._toolbar.set_current("ge3")
    assert page._results_stack.currentWidget() is page._cards[3].results_widget
    assert page._resid_stack.currentWidget() is page._cards[3].residual_chart
    assert page._cake_stack.currentWidget() is page._cake_views[3]

    for n in (1, 2, 3, 4):
        page._cards[n].result = None
    page._run_mode_combo.setCurrentIndex(1)   # Parallel
    page._run_all()
    assert len(page._workers) == 4, "parallel mode should start every panel at once"
    ok = _pump(app, lambda: not page._workers and not page._pending_panels)
    assert ok, "parallel run did not complete"
    assert all(page._cards[n].result is not None for n in (1, 2, 3, 4))
    # Four workers in flight against one working directory: every file the
    # backend writes there is generically named (residual_corr.bin,
    # calibration.json), so a shared scratch folder is a race. Each panel gets
    # its own leaf, and all four sit under the one run folder inside
    # .midas_scratch/ so the run is still deletable as a unit.
    scratches = {n: Path(page._last_cfgs[n]["scratch_dir"]) for n in (1, 2, 3, 4)}
    assert len(set(scratches.values())) == 4
    assert len({d.parent for d in scratches.values()}) == 1
    assert all(d.parent.parent == work / SCRATCH_DIRNAME
               for d in scratches.values())

    with h5py.File(proj_path, "r") as f:
        for n in (1, 2, 3, 4):
            grp = f[f"analysis/calibrate/ge{n}"]
            assert set(grp.keys()) == {"attempt_0001", "attempt_0002"}
            assert grp.attrs["latest"] == "attempt_0002"
