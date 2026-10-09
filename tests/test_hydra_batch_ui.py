"""Offscreen UI test for the Batch Integrate tab's Hydra page: per-panel
calibration hand-off (auto from Calibrate + manual "From file"), per-panel
mask wiring, output subfolder-per-panel naming, and Sequential/Parallel run
orchestration + per-panel result-viewer switching.

``HydraBatchPage`` builds 4 ``WaterfallViewer`` + 4 ``StackedProfileViewer``
+ 1 shared Detector-view ``ImageViewer`` (9 pyqtgraph widgets total — more
than the Calibrate Hydra page's 6). Per .context/DECISIONS.md's
pyqtgraph-teardown-crash entry (reproduced there building one
``HydraCalibrationPage`` per test function), this file follows the same
fix: exactly 2 test functions, each building exactly ONE page and reusing
it for every assertion — see ``tests/test_hydra_calib_ui.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "test_data" / "gui_synthetic" / "hydra"

# Run each test in this file in its own forked subprocess (pytest-forked):
# the Detector-view ImageViewer added alongside the existing 8 pyqtgraph
# widgets tipped this file into the pyqtgraph-teardown segfault documented
# above and in .context/STATE.md — forking contains it to one clean FAILED
# with signal info per test, instead of crashing the whole pytest run.
pytestmark = pytest.mark.forked


_QT_LOADED = False

# Bound by _load_qt() at fixture time, declared here so static analysis
# (and the pyflakes diff in the review recipe) can still resolve them.
QtCore = QtWidgets = None
hydra_batch_page_mod = hydra_batch_widgets_mod = project = None
HydraBatchPage = _FakeBatchWorker = None


def _load_qt():
    """Import Qt, the GUI modules under test, and the QObject-subclass fake,
    publishing them all as module globals.

    Deliberately NOT done at module level. pytest imports this module during
    collection, in the *parent* process, while pytest-forked runs each test
    in a forked child. Importing PyQt5 in the parent initialises macOS
    CoreFoundation, which a forked child may not use — every test then dies
    with SIGSEGV ("The process has forked and you cannot use this
    CoreFoundation functionality safely") before its body runs. Importing
    here means each child does its own first-time init, which is legal.

    ``_FakeBatchWorker`` subclasses ``QtCore.QObject``, so it cannot be
    defined at module scope either. See .context/STATE.md.
    """
    global _QT_LOADED
    if _QT_LOADED:
        return
    from PyQt5 import QtCore, QtWidgets
    import midas_gui.hydra_batch_page as hydra_batch_page_mod
    import midas_gui.hydra_batch_widgets as hydra_batch_widgets_mod
    from midas_gui import project
    from midas_gui.hydra_batch_page import HydraBatchPage

    class _FakeBatchWorker(QtCore.QObject):
        """No-op-thread shape for ``BatchRunCoordinator``: finishes on the next
        event-loop tick instead of a real background thread/file source."""
        progress = QtCore.pyqtSignal(int, int)
        frame_done = QtCore.pyqtSignal(str, object, object, object)
        finished = QtCore.pyqtSignal(dict)
        failed = QtCore.pyqtSignal(str)
        log_line = QtCore.pyqtSignal(str)
        geom_ready = QtCore.pyqtSignal(object)

        #: (panel key inferred from out_dir) -> out_dir passed at construction,
        #: appended by every instance — tests reset this list before each run.
        calls: list = []
        MIN_FRAMES_PER_WORKER = 10   # duck-typed to match BatchRunCoordinator

        def __init__(self, spec, source_cfg, mask, out_dir, fmts, kernel, corrections,
                     variance_cfg, q_cfg=None, frame_range=None, monitor_file=None,
                     drift_traj=None, parent=None, dark=None, bright=None, background=None,
                     bright_mode="divide", weighted=True, context=None, im_trans=(),
                     run_mode="sequential", n_workers=1):
            super().__init__(parent)
            self.out_dir = out_dir
            _FakeBatchWorker.calls.append(out_dir)

        def isRunning(self) -> bool:
            return False

        def requestInterruption(self):
            pass

        def start(self):
            QtCore.QTimer.singleShot(0, self._finish)

        def _finish(self):
            r_axis = np.linspace(0, 100, 10)
            profile = np.ones_like(r_axis)
            self.progress.emit(1, 1)
            self.frame_done.emit("0", r_axis, profile, None)
            self.finished.emit({
                "n": 1, "r_axis_px": r_axis, "profiles": profile.reshape(1, -1),
                "sigmas": profile.reshape(1, -1), "frame_ids": ["0"],
                "out_paths": [f"{self.out_dir}/f0.csv"] if self.out_dir else []})

    _QT_LOADED = True
    globals().update(
        QtCore=QtCore, QtWidgets=QtWidgets,
        hydra_batch_page_mod=hydra_batch_page_mod,
        hydra_batch_widgets_mod=hydra_batch_widgets_mod,
        project=project,
        HydraBatchPage=HydraBatchPage,
        _FakeBatchWorker=_FakeBatchWorker,
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


def _fake_spec(lsd=200_000.0, pxY=200.0, wavelength_A=0.1729):
    return SimpleNamespace(Lsd=lsd, pxY=pxY, Wavelength=wavelength_A)


@pytest.fixture(autouse=True)
def _stub_spec_builders(app, monkeypatch):   # `app` so _load_qt() has run first
    """Spec-building goes through midas_calibrate_v2's real geometry math —
    not what this file tests (it tests GUI wiring/orchestration, mirroring
    tests/test_hydra_calib_ui.py's choice to stub CalibrationWorker/
    IntegrationWorker rather than exercise a real fit). Stub it with a
    lightweight object carrying just the attributes ``_start_panel_worker``
    reads (``Lsd``, ``pxY``, ``Wavelength``)."""
    # **kw rather than the four named optionals the stub used to list: the
    # real builders have always taken eta_min/eta_max too, and the card
    # started passing them when caking went per panel. A stub that models a
    # narrower signature than the thing it replaces fails the caller for a
    # reason that has nothing to do with the caller.
    monkeypatch.setattr(hydra_batch_widgets_mod, "_build_spec",
                        lambda result, r_bin, e_bin, **kw: _fake_spec(
                            getattr(result, "Lsd", 200_000.0), getattr(result, "pxY", 200.0),
                            getattr(result, "wavelength_A", 0.1729)))
    monkeypatch.setattr(hydra_batch_widgets_mod, "spec_from_geometry_file",
                        lambda path, r_bin, e_bin, **kw: _fake_spec())


@pytest.fixture(autouse=True)
def _stub_worker(app, monkeypatch):   # `app` so _load_qt() has run first
    _FakeBatchWorker.calls = []
    monkeypatch.setattr(hydra_batch_page_mod, "BatchRunCoordinator", _FakeBatchWorker)


def _mk_result(**overrides):
    fields = dict(BC_y=128.0, BC_z=128.0, Lsd=200_000.0, wavelength_A=0.1729,
                  pxY=200.0, pxZ=200.0, NrPixelsY=256, NrPixelsZ=256,
                  tx=0.0, ty=0.0, tz=0.0, distortion={})
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _pump(app, condition, max_iters=200):
    for _ in range(max_iters):
        if condition():
            return True
        app.processEvents()
    return False


def test_hydra_batch_calibration_sources_and_masks(app, fixture_available):
    """Auto hand-off (set_panel_calibration) and manual "From file" both
    populate a panel's calibration-values grid; per-panel mask selectors
    exist and are independent; a panel with no calibration is skipped and
    the ones that do run write to their own ge{n}/ output subfolder."""
    page = HydraBatchPage()
    page._loader.set_path(str(fixture_available / "ge1" / "panel.ge1.h5"))
    assert set(page._loader.siblings()) == {1, 2, 3, 4}

    # Per-panel masks: independent MaskSelector instances, no cross-talk.
    assert set(page._loader._mask_sels) == {1, 2, 3, 4}
    assert len({id(s) for s in page._loader._mask_sels.values()}) == 4

    # Panel 1: auto hand-off from the Calibrate tab.
    page.set_panel_calibration(1, _mk_result(BC_y=111.0))
    card1 = page._cards[1]
    assert card1._use_calib_btn.isChecked()
    _, note1 = card1._calib_fields_in_use()
    assert "Calibrate tab" in note1

    # Panel 2: manual "From file".
    card2 = page._cards[2]
    card2._json_ed.setText(str(fixture_available / "ge2" / "ps_ge2.txt"))
    assert card2._use_file_btn.isChecked()
    _, note2 = card2._calib_fields_in_use()
    assert "From file" in note2

    # Panels 3 & 4: no calibration configured at all.
    page._out_ed.setText("/tmp/hydra_batch_test_out")
    page._run_mode_combo.setCurrentIndex(0)   # Sequential
    page._run_all()
    ok = _pump(app, lambda: not page._workers and not page._pending_panels)
    assert ok, "sequential run did not complete"

    assert _FakeBatchWorker.calls == [
        "/tmp/hydra_batch_test_out/ge1", "/tmp/hydra_batch_test_out/ge2"], _FakeBatchWorker.calls
    for n in (3, 4):
        status = page._cards[n]._status_lbl.text().lower()
        assert "calibration error" in status or "no data" in status
    assert page._run_btn.isEnabled() and not page._abort_btn.isEnabled()


def test_hydra_batch_run_orchestration_and_viewer_switching(app, fixture_available, tmp_path):
    """Sequential drains the queue one panel at a time; Parallel starts all
    4 workers up front; the per-panel Waterfall/Stacked viewer stack and
    card stack both switch with the active-panel toolbar. Also verifies each
    panel's completed run is logged to a project file, linked back to its
    (fake) calibration attempt, when one is open."""
    page = HydraBatchPage()
    proj_path = str(tmp_path / "proj.h5")
    project.create_project(proj_path)
    page.set_project_context(project.ProjectContext())
    page._project_ctx.path = proj_path
    page._loader.set_path(str(fixture_available / "ge1" / "panel.ge1.h5"))
    for n in (1, 2, 3, 4):
        result = _mk_result(BC_y=100.0 + n)
        result._project_attempt_ref = f"/analysis/calibrate/ge{n}/attempt_0001"
        page.set_panel_calibration(n, result)
    page._out_ed.setText("/tmp/hydra_batch_test_out2")

    page._run_mode_combo.setCurrentIndex(0)   # Sequential
    page._run_all()
    ok = _pump(app, lambda: not page._workers and not page._pending_panels)
    assert ok, "sequential run did not complete"
    assert _FakeBatchWorker.calls == [f"/tmp/hydra_batch_test_out2/ge{n}" for n in (1, 2, 3, 4)]
    for n in (1, 2, 3, 4):
        assert "Complete" in page._cards[n]._status_lbl.text()

    with h5py.File(proj_path, "r") as f:
        for n in (1, 2, 3, 4):
            att = f[f"analysis/integrate/ge{n}/attempt_0001"]
            assert att.attrs["n_frames"] == 1
            assert att.attrs["calib_attempt_ref"] == f"/analysis/calibrate/ge{n}/attempt_0001"

    _FakeBatchWorker.calls = []
    page._run_mode_combo.setCurrentIndex(1)   # Parallel
    page._run_all()
    assert len(page._workers) == 4, "parallel mode should start every panel at once"
    ok = _pump(app, lambda: not page._workers and not page._pending_panels)
    assert ok, "parallel run did not complete"
    assert sorted(_FakeBatchWorker.calls) == sorted(
        f"/tmp/hydra_batch_test_out2/ge{n}" for n in (1, 2, 3, 4))

    page._toolbar.set_current("ge1")
    assert page._card_stack.currentWidget() is page._cards[1]
    assert page._viewer_stack.currentWidget() is page._viewer_pairs[1]
    page._toolbar.set_current("ge3")
    assert page._card_stack.currentWidget() is page._cards[3]
    assert page._viewer_stack.currentWidget() is page._viewer_pairs[3]
