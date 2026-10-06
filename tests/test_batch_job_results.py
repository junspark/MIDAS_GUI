""""Run as background job" results reaching the Batch Integrate tab.

A background job is a detached `screen` session running `batch_cli.py` in a
separate process — no Qt signals from it reach the GUI that launched it. Two
things close that gap:

* `batch_cli._write_results_sidecar` persists the same r_axis_px/profiles/
  frame_ids/eta_axis arrays a live run's `finished` payload carries, as
  `_bg_job_results.npz` in the job's `--out-dir` — regardless of which
  `--fmts` the user picked (plain csv doesn't round-trip that shape).
* `job_queue.JobQueuePanel`'s `on_job_done` callback fires once a job is
  detected `Done` (and again from a "Load results" click), and
  `BatchTab._on_job_done` reads that sidecar back through the same replay
  path `_populate_plots_from_attempt` already uses for a restored project
  attempt.

Builds pyqtgraph widgets via BatchTab/JobQueuePanel, hence forked, and defers
every Qt / midas_gui GUI import into fixtures — see STATE.md's rule for new
Qt test files (``tests/test_set_raw_frame.py`` is the reference).
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.forked

N_ETA, N_R = 6, 8


def _axes():
    return (np.arange(N_R, dtype=float), np.linspace(-180.0, 180.0, N_ETA))


def _1d_payload(n_frames=3):
    r_axis, _ = _axes()
    return {"n": n_frames, "out_paths": ["x"], "aborted": False,
            "r_axis_px": r_axis,
            "profiles": np.random.default_rng(0).random((n_frames, N_R)).astype(np.float32),
            "frame_ids": list(range(n_frames))}


def _cake_payload(n_frames=4):
    r_axis, eta_axis = _axes()
    return {"n": n_frames, "out_paths": ["x"], "aborted": False,
            "r_axis_px": r_axis, "eta_axis": eta_axis,
            "profiles": np.random.default_rng(1).random((n_frames, N_ETA, N_R)).astype(np.float32),
            "frame_ids": [f"f{i}" for i in range(n_frames)]}


# ── batch_cli._source_cfg (argv -> source dict, pure logic, no Qt) ──────

def _parse_source_args(extra):
    from midas_gui import batch_cli
    parser = batch_cli._build_arg_parser()
    base = ["--calib-file", "c.json", "--out-dir", "o"]
    return parser.parse_args(base + extra)


def test_source_cfg_hdf5_forwards_frame_start_end():
    """The "hdf5" branch was the only one of the four source types that
    didn't forward --frame-start/--frame-end into the returned cfg dict —
    needed now that a single-file HDF5 source's start/end mean a raw
    sub-frame range within that one file (see
    widgets.DataLoaderPanel.source_cfg's "hdf5" branch), matching how the
    other three source types already forward them (as scan-number bounds)."""
    from midas_gui import batch_cli
    args = _parse_source_args([
        "--source-type", "hdf5", "--source-path", "scan.h5",
        "--frame-start", "2", "--frame-end", "7"])
    cfg = batch_cli._source_cfg(args)
    assert cfg["type"] == "hdf5"
    assert (cfg["frame_start"], cfg["frame_end"]) == (2, 7)


def test_source_cfg_hdf5_frame_bounds_default_to_none():
    from midas_gui import batch_cli
    args = _parse_source_args(["--source-type", "hdf5", "--source-path", "scan.h5"])
    cfg = batch_cli._source_cfg(args)
    assert cfg["frame_start"] is None and cfg["frame_end"] is None


# ── batch_cli._write_results_sidecar (pure logic, no Qt) ────────────────

def test_write_results_sidecar_round_trips_1d(tmp_path):
    from midas_gui import batch_cli
    batch_cli._write_results_sidecar(str(tmp_path), _1d_payload())
    with np.load(tmp_path / "_bg_job_results.npz") as npz:
        assert npz["profiles"].shape == (3, N_R)
        assert list(npz["frame_ids"]) == ["0", "1", "2"]
        assert "eta_axis_deg" not in npz.files


def test_write_results_sidecar_round_trips_cakes(tmp_path):
    from midas_gui import batch_cli
    batch_cli._write_results_sidecar(str(tmp_path), _cake_payload())
    with np.load(tmp_path / "_bg_job_results.npz") as npz:
        assert npz["profiles"].shape == (4, N_ETA, N_R)
        assert npz["eta_axis_deg"] == pytest.approx(_axes()[1])
        assert list(npz["frame_ids"]) == ["f0", "f1", "f2", "f3"]


def test_write_results_sidecar_is_a_noop_with_nothing_to_plot(tmp_path):
    from midas_gui import batch_cli
    batch_cli._write_results_sidecar(str(tmp_path), {"n": 0, "aborted": False})
    assert not (tmp_path / "_bg_job_results.npz").exists()


# ── job_queue.JobQueuePanel ───────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def job_queue(app):
    from midas_gui import job_queue
    return job_queue


@pytest.fixture
def log_widget(app):
    from midas_gui.widgets import LogPanel
    return LogPanel()


def test_launch_records_out_dir_in_meta(job_queue, log_widget, tmp_path, monkeypatch):
    monkeypatch.setattr(job_queue, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(job_queue, "screen_available", lambda: True)
    monkeypatch.setattr(job_queue, "_screen_session_alive", lambda session: False)
    monkeypatch.setattr(job_queue.QtCore.QProcess, "execute", staticmethod(lambda *a, **k: 0))

    panel = job_queue.JobQueuePanel(log_widget)
    out_dir = str(tmp_path / "results")
    job = panel.launch(["true"], name="myjob", total_frames=5, out_dir=out_dir)
    assert job is not None
    assert job.out_dir == out_dir
    import json
    meta = json.loads(job_queue.Path(job.meta_path).read_text())
    assert meta["out_dir"] == out_dir


def test_adopt_existing_sessions_recovers_out_dir(job_queue, log_widget, tmp_path, monkeypatch):
    monkeypatch.setattr(job_queue, "JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir()
    session = f"{job_queue._SESSION_PREFIX}old_job"
    meta_path = tmp_path / "jobs" / f"{session}.meta.json"
    import json
    meta_path.write_text(json.dumps({"name": "old_job", "total_frames": 2,
                                     "out_dir": str(tmp_path / "old_out")}))

    def fake_run(cmd, **kw):
        if cmd[:2] == ["screen", "-ls"]:
            return SimpleNamespace(stdout=f"\t123.{session}\t(Detached)\n")
        return SimpleNamespace(stdout="")
    monkeypatch.setattr(job_queue.subprocess, "run", fake_run)
    monkeypatch.setattr(job_queue, "_screen_session_alive", lambda s: True)

    panel = job_queue.JobQueuePanel(log_widget)
    assert len(panel._jobs) == 1
    assert panel._jobs[0].out_dir == str(tmp_path / "old_out")


def test_finalize_job_done_calls_on_job_done_and_enables_button(job_queue, log_widget, tmp_path):
    calls = []
    panel = job_queue.JobQueuePanel(log_widget, on_job_done=calls.append)
    logfile = tmp_path / "x.screenlog"
    logfile.write_text("[launcher] DONE exit=0\n")
    job = job_queue.Job(session="s1", logfile=str(logfile), meta_path="",
                        name="job1", out_dir=str(tmp_path))
    panel._jobs.append(job)
    panel._add_job_row(job)
    assert job.load_results_btn.isEnabled() is False

    panel._finalize_job(job)

    assert job.status == "Done"
    assert job.load_results_btn.isEnabled() is True
    assert calls == [job]


def test_finalize_job_failed_does_not_call_on_job_done(job_queue, log_widget, tmp_path):
    calls = []
    panel = job_queue.JobQueuePanel(log_widget, on_job_done=calls.append)
    logfile = tmp_path / "x.screenlog"
    logfile.write_text("[launcher] DONE exit=1\n")
    job = job_queue.Job(session="s2", logfile=str(logfile), meta_path="",
                        name="job2", out_dir=str(tmp_path))
    panel._jobs.append(job)
    panel._add_job_row(job)

    panel._finalize_job(job)

    assert job.status == "Failed (1)"
    assert job.load_results_btn.isEnabled() is False
    assert calls == []


def test_load_results_button_click_invokes_callback_again(job_queue, log_widget, tmp_path):
    calls = []
    panel = job_queue.JobQueuePanel(log_widget, on_job_done=calls.append)
    job = job_queue.Job(session="s3", logfile="", meta_path="", name="job3",
                        out_dir=str(tmp_path), status="Done")
    panel._jobs.append(job)
    panel._add_job_row(job)

    job.load_results_btn.click()

    assert calls == [job]


# ── BatchTab._on_job_done ──────────────────────────────────────────────

@pytest.fixture
def tab(app):
    from midas_gui.tab_batch import BatchTab
    return BatchTab()


def test_on_job_done_populates_waterfall_from_1d_sidecar(tab, tmp_path):
    from midas_gui import batch_cli
    batch_cli._write_results_sidecar(str(tmp_path), _1d_payload())
    job = SimpleNamespace(session="sess-1d", out_dir=str(tmp_path))

    tab._on_job_done(job)

    assert tab._waterfall._nrows == 3
    assert tab._cake_stack_view.frame_count() == 0


def test_on_job_done_populates_cake_stack_from_multi_azimuth_sidecar(tab, tmp_path):
    from midas_gui import batch_cli
    batch_cli._write_results_sidecar(str(tmp_path), _cake_payload())
    job = SimpleNamespace(session="sess-cake", out_dir=str(tmp_path))

    tab._on_job_done(job)

    assert tab._cake_stack_view.frame_count() == 4
    assert tab._cake_stack_view._eta_axis == pytest.approx(_axes()[1])
    assert tab._cake_stack_view._frame_ids == ["f0", "f1", "f2", "f3"]
    # 1-D views got the η-collapse, not raw cakes.
    assert tab._waterfall._nrows == 4
    assert tab._waterfall._buf.shape[1] == N_R


def test_on_job_done_with_no_out_dir_logs_and_does_not_crash(tab):
    job = SimpleNamespace(session="sess-none", out_dir="")
    tab._on_job_done(job)
    assert "no recorded" in tab._log.toPlainText()


def test_on_job_done_with_missing_sidecar_logs_and_does_not_crash(tab, tmp_path):
    job = SimpleNamespace(session="sess-missing", out_dir=str(tmp_path))
    tab._on_job_done(job)
    assert "No results file found" in tab._log.toPlainText()


def test_job_queue_panel_wired_into_batch_tab_calls_populate(tab, tmp_path):
    """End-to-end wiring: BatchTab passes its own _on_job_done as the
    JobQueuePanel's on_job_done callback."""
    from midas_gui import batch_cli
    from midas_gui.job_queue import Job
    batch_cli._write_results_sidecar(str(tmp_path), _1d_payload())
    job = Job(session="sess-e2e", logfile="", meta_path="", name="j",
             out_dir=str(tmp_path))
    tab._job_queue._jobs.append(job)
    tab._job_queue._add_job_row(job)

    tab._job_queue._on_job_done(job)  # simulate what _finalize_job would call

    assert tab._waterfall._nrows == 3


# ── the finished job's screenlog reaches the shared log tree ────────────
#
# A background job already had a log, but in ~/.midas_gui/jobs — a fixed
# location job adoption scans after a GUI restart, so it cannot move. It is
# copied into ~/midas_runs/midas_screen_logs/<beamline>/<expid>/ instead,
# where the in-process run writes its own. Without that, which run mode you
# happened to use would decide whether your log was findable.

def test_a_finished_job_log_is_copied_into_the_screen_log_tree(tab, tmp_path,
                                                               monkeypatch):
    from midas_gui import run_log
    root = tmp_path / "midas_screen_logs"
    monkeypatch.setattr(run_log, "LOG_ROOT", root)

    live = tmp_path / "jobs" / "midasgui_batch_42.screenlog"
    live.parent.mkdir(parents=True)
    live.write_text("[launcher] DONE exit=0\n")

    tab.set_expid_provider(lambda: "brown_sep26")
    tab._archive_job_log(SimpleNamespace(logfile=str(live),
                                         session="midasgui_batch_42"))

    copies = list(root.rglob("*.screenlog"))
    assert len(copies) == 1
    assert copies[0].read_text() == "[launcher] DONE exit=0\n"
    assert "midasgui_batch_42" in copies[0].name
    assert "brown_sep26" in copies[0].parts
    assert live.is_file(), "the live log must stay put for job adoption"


def test_archiving_a_missing_job_log_is_a_no_op(tab, tmp_path, monkeypatch):
    """A log is a record of work, not the work."""
    from midas_gui import run_log
    root = tmp_path / "midas_screen_logs"
    monkeypatch.setattr(run_log, "LOG_ROOT", root)
    tab.set_expid_provider(lambda: "e")
    tab._archive_job_log(SimpleNamespace(logfile="/nonexistent/x.screenlog",
                                         session="s"))      # must not raise
    assert not root.exists()
