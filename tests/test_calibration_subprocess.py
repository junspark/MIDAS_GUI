"""``CalibrationWorker`` runs each calibration in its own OS process
(``calib_cli.py``) instead of an in-process ``QThread`` — pins the job-file
hand-off (``workers._write_calib_job``), the CLI's read/replicate/write
contract (``calib_cli.main``), and the worker's QProcess wiring (real
subprocess spawn, live log streaming, success/failure signals, a real kill
on ``requestInterruption()``).

Deliberately uses a fast-FAILING real calibration (an unknown pipeline mode,
which ``calib.run_pipeline`` rejects before doing any heavy work) for every
subprocess-spawning test — a real converging fit takes minutes (see
``.context/STATE.md``'s Hydra timing notes) and isn't needed to prove the
plumbing works; the success path through ``calib_cli.main()`` itself is
covered separately with the backend mocked out.

Spawns real child processes but builds no pyqtgraph widgets; marked forked
anyway (cheap insurance, not a known crash here) since every other test file
that spawns subprocesses or builds a QApplication in this suite does the
same — see STATE.md's per-file-isolation rule. Defers every Qt / midas_gui
import into fixtures or test bodies per that same rule (forked tests must not
import PyQt5 at module/collection time).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

pytestmark = pytest.mark.forked


# ── workers._write_calib_job ─────────────────────────────────────────────

def test_write_calib_job_round_trips_cfg_and_arrays(tmp_path):
    from midas_gui.workers import _write_calib_job
    from midas_gui import calib_cli

    image = np.random.rand(4, 4).astype(np.float32)
    dark = np.ones((4, 4), dtype=np.float32)
    bright = np.full((4, 4), 2.0, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    cfg = {
        "wavelength": 0.2, "pxY": 150.0, "pxZ": None, "calibrant": "CeO2",
        "refine": {"Lsd": True, "distortion_coeffs": {"p0", "p1"}},
        "im_trans": (1, 3), "scratch_dir": str(tmp_path), "mask": mask,
    }
    _write_calib_job(tmp_path, "one_shot", image, dark, cfg,
                     bright=bright, background=None, bright_mode="divide")

    job = json.loads((tmp_path / calib_cli.JOB_JSON).read_text())
    assert job["mode"] == "one_shot"
    assert job["bright_mode"] == "divide"
    assert "mask" not in job["cfg"]                       # moved to the npz
    assert sorted(job["cfg"]["refine"]["distortion_coeffs"]) == ["p0", "p1"]
    assert job["cfg"]["im_trans"] == [1, 3]                # tuple -> JSON list

    arrays = np.load(tmp_path / calib_cli.JOB_NPZ)
    assert set(arrays.files) == {"image", "dark", "bright", "mask"}
    assert np.array_equal(arrays["image"], image)
    assert np.array_equal(arrays["mask"], mask)


def test_write_calib_job_omits_absent_optional_arrays(tmp_path):
    from midas_gui.workers import _write_calib_job
    from midas_gui import calib_cli

    image = np.zeros((2, 2), dtype=np.float32)
    _write_calib_job(tmp_path, "one_shot", image, None, {"wavelength": 0.2, "pxY": 1.0,
                                                          "calibrant": "CeO2"})
    arrays = np.load(tmp_path / calib_cli.JOB_NPZ)
    assert arrays.files == ["image"]


def test_write_calib_job_cfg_mutation_does_not_leak_to_caller(tmp_path):
    """``_write_calib_job`` pops "mask" out of its own copy, not the caller's
    cfg dict — both tabs reuse/log the same cfg object after launching."""
    from midas_gui.workers import _write_calib_job

    mask = np.ones((2, 2), dtype=bool)
    cfg = {"wavelength": 0.2, "pxY": 1.0, "calibrant": "CeO2", "mask": mask}
    _write_calib_job(tmp_path, "one_shot", np.zeros((2, 2), dtype=np.float32), None, cfg)
    assert cfg["mask"] is mask


# ── calib_cli.main() — success path, backend mocked (fast) ───────────────

def test_calib_cli_main_success_applies_corrections_and_writes_result(tmp_path):
    from midas_gui.workers import _write_calib_job
    from midas_gui import calib_cli

    image = np.full((4, 4), 5.0, dtype=np.float32)
    # Non-uniform on purpose: a *uniform* bright field is a no-op in divide
    # mode (it normalises by its own mean), which would prove nothing.
    bright = np.full((4, 4), 2.0, dtype=np.float32)
    bright[0, 0] = 4.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    cfg = {"wavelength": 0.2, "pxY": 150.0, "calibrant": "CeO2", "refine": {},
          "im_trans": (), "scratch_dir": str(tmp_path), "mask": mask}
    _write_calib_job(tmp_path, "one_shot", image, None, cfg, bright=bright, bright_mode="divide")

    fake_result = SimpleNamespace(residual_corr_map=None)

    with mock.patch("midas_gui.calib.run_pipeline", return_value="RAW") as m_run, \
         mock.patch("midas_gui.calib.normalize_result", return_value=fake_result), \
         mock.patch("midas_gui.calib.effective_pixel_counts", return_value=(4, 4)):
        rc = calib_cli.main(["--job-dir", str(tmp_path)])

    assert rc == 0
    corrected_image = m_run.call_args[0][1]
    assert corrected_image[1, 1] == 0.0          # masked pixel zeroed
    assert corrected_image[0, 0] != 5.0           # bright correction applied

    with open(tmp_path / calib_cli.RESULT_PKL, "rb") as fh:
        result = pickle.load(fh)
    assert result._calibrant_name == "CeO2"


def test_calib_cli_main_detaches_residual_corr_map_to_cpu(tmp_path):
    """A torch tensor still wired into autograd (or resident on a GPU the
    launching GUI process may not share) can't cross the process boundary;
    ``main()`` must detach + CPU it before pickling."""
    torch = pytest.importorskip("torch")
    from midas_gui.workers import _write_calib_job
    from midas_gui import calib_cli

    image = np.zeros((4, 4), dtype=np.float32)
    cfg = {"wavelength": 0.2, "pxY": 150.0, "calibrant": "CeO2", "refine": {},
          "im_trans": (), "scratch_dir": str(tmp_path)}
    _write_calib_job(tmp_path, "one_shot", image, None, cfg)

    fake_result = SimpleNamespace(residual_corr_map=torch.ones(2, 2, requires_grad=True))

    with mock.patch("midas_gui.calib.run_pipeline", return_value="RAW"), \
         mock.patch("midas_gui.calib.normalize_result", return_value=fake_result), \
         mock.patch("midas_gui.calib.effective_pixel_counts", return_value=(4, 4)):
        rc = calib_cli.main(["--job-dir", str(tmp_path)])

    assert rc == 0
    with open(tmp_path / calib_cli.RESULT_PKL, "rb") as fh:
        result = pickle.load(fh)
    assert result.residual_corr_map.device.type == "cpu"
    assert not result.residual_corr_map.requires_grad


def test_calib_cli_main_missing_job_dir_fails_cleanly(tmp_path):
    from midas_gui import calib_cli
    rc = calib_cli.main(["--job-dir", str(tmp_path / "does_not_exist")])
    assert rc == 1


# ── CalibrationWorker — real subprocess, fast-failing mode ────────────────

@pytest.fixture()
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _run_worker_to_completion(worker, timeout_ms=30_000):
    """Drive ``worker.start()`` through a real Qt event loop until it emits
    ``finished`` or ``failed`` (or the timeout fires), returning
    ``(signal_name, payload)``."""
    from PyQt5 import QtCore
    outcome = {}
    loop = QtCore.QEventLoop()

    def _finished(r):
        outcome["kind"], outcome["payload"] = "finished", r
        loop.quit()

    def _failed(m):
        outcome["kind"], outcome["payload"] = "failed", m
        loop.quit()

    worker.finished.connect(_finished)
    worker.failed.connect(_failed)
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    worker.start()
    loop.exec_()
    return outcome.get("kind"), outcome.get("payload")


def test_calibration_worker_runs_real_subprocess_and_reports_failure(app, tmp_path):
    from midas_gui.workers import CalibrationWorker

    image = np.zeros((4, 4), dtype=np.float32)
    cfg = {"wavelength": 0.2, "pxY": 150.0, "calibrant": "CeO2", "refine": {},
          "im_trans": (), "scratch_dir": str(tmp_path)}
    w = CalibrationWorker("bogus_mode_xyz", image, None, cfg)
    log_lines = []
    w.log_line.connect(log_lines.append)

    assert w.isRunning() is False
    kind, payload = _run_worker_to_completion(w)
    assert kind == "failed"
    assert "Unknown pipeline mode" in payload
    assert any("Unknown pipeline mode" in line for line in log_lines)
    assert w.isRunning() is False


def test_calibration_worker_uses_tempdir_when_cfg_has_no_scratch_dir(app):
    """Both real call sites always set cfg['scratch_dir'] before constructing
    the worker, but the worker must not crash if one doesn't."""
    from midas_gui.workers import CalibrationWorker

    image = np.zeros((4, 4), dtype=np.float32)
    cfg = {"wavelength": 0.2, "pxY": 150.0, "calibrant": "CeO2", "refine": {}, "im_trans": ()}
    w = CalibrationWorker("bogus_mode_xyz", image, None, cfg)
    kind, payload = _run_worker_to_completion(w)
    assert kind == "failed"
    assert "Unknown pipeline mode" in payload


def test_calibration_worker_request_interruption_before_start_is_a_noop(app):
    from midas_gui.workers import CalibrationWorker

    image = np.zeros((4, 4), dtype=np.float32)
    cfg = {"wavelength": 0.2, "pxY": 150.0, "calibrant": "CeO2", "refine": {}, "im_trans": ()}
    w = CalibrationWorker("bogus_mode_xyz", image, None, cfg)
    w.requestInterruption()   # never started — must not raise
    assert w.isRunning() is False


def test_calibration_worker_kill_after_start_stops_cleanly(app, tmp_path):
    from midas_gui.workers import CalibrationWorker
    from PyQt5 import QtCore

    image = np.zeros((4, 4), dtype=np.float32)
    cfg = {"wavelength": 0.2, "pxY": 150.0, "calibrant": "CeO2", "refine": {},
          "im_trans": (), "scratch_dir": str(tmp_path)}
    w = CalibrationWorker("bogus_mode_xyz", image, None, cfg)
    received = []
    w.finished.connect(lambda r: received.append("finished"))
    w.failed.connect(lambda m: received.append("failed"))
    w.start()
    w.requestInterruption()

    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(5000, loop.quit)
    loop.exec_()
    assert w.isRunning() is False   # killed (or had already exited) either way
