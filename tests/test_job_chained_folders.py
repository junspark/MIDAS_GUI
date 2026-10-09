"""A background job must be able to span source folders.

``batch_cli`` takes a single ``--out-dir``, so a selection spanning folders
needs one invocation per folder. They are chained into ONE screen session
rather than launched as N jobs: N sessions would compete for the same disk
and clutter the queue with N entries for what the user asked for once.

Qt import discipline per STATE.md: import inside fixtures, never at module
scope.
"""
from pathlib import Path

import pytest

pytest.importorskip("PyQt5.QtWidgets")

pytestmark = pytest.mark.forked


class _Log:
    """Duck-typed stand-in for the panel's log widget (it wants
    ``append_html``, which QTextEdit does not have)."""

    def __init__(self):
        self.lines = []

    def append_html(self, html):
        self.lines.append(html)

    def append(self, text):
        self.lines.append(text)

    def clear(self):
        self.lines.clear()


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(qapp, monkeypatch, tmp_path):
    from PyQt5 import QtCore, QtWidgets
    from midas_gui import job_queue as J
    monkeypatch.setattr(J, "screen_available", lambda: True)
    monkeypatch.setattr(J, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(J, "_screen_session_alive", lambda s: False)
    captured = {}
    monkeypatch.setattr(QtCore.QProcess, "execute",
                        staticmethod(lambda prog, args: captured.setdefault("args", args) and 0 or 0))
    p = J.JobQueuePanel(_Log())
    p.captured = captured
    return p


def _wrapped(panel):
    """The job script the panel wrote for screen to run."""
    return Path(panel.captured["args"][-1]).read_text()


def test_a_single_argv_is_unchanged_in_shape(panel):
    panel.launch(["python", "-m", "midas_gui.batch_cli", "--out-dir", "/o/a"],
                 name="one", total_frames=5, out_dir="/o/a")
    w = _wrapped(panel)
    assert "--out-dir /o/a" in w
    assert "step 1/" not in w, "a lone run should not be announced as a step"
    assert "DONE exit=$rc" in w


def test_every_folder_gets_its_own_invocation(panel):
    base = ["python", "-m", "midas_gui.batch_cli"]
    panel.launch(base + ["--out-dir", "/o/a"], name="many", total_frames=9,
                 out_dir="/o/a",
                 extra_argvs=[base + ["--out-dir", "/o/b"],
                              base + ["--out-dir", "/o/c"]])
    w = _wrapped(panel)
    for d in ("/o/a", "/o/b", "/o/c"):
        assert f"--out-dir {d}" in w, f"{d} missing from the chained command"
    assert w.count("midas_gui.batch_cli") == 3


def test_the_folders_run_in_order(panel):
    base = ["python", "-m", "midas_gui.batch_cli"]
    panel.launch(base + ["--out-dir", "/o/a"], name="many", total_frames=9,
                 out_dir="/o/a",
                 extra_argvs=[base + ["--out-dir", "/o/b"]])
    w = _wrapped(panel)
    assert w.index("/o/a") < w.index("/o/b")
    assert "step 1/2" in w and "step 2/2" in w


def test_one_failing_folder_does_not_cancel_the_rest(panel):
    """`|| rc=$?` rather than `&&` -- folder 7 dying must not silently drop
    folders 8..23, but the job must still report failure."""
    base = ["python", "-m", "midas_gui.batch_cli"]
    panel.launch(base + ["--out-dir", "/o/a"], name="many", total_frames=9,
                 out_dir="/o/a", extra_argvs=[base + ["--out-dir", "/o/b"]])
    w = _wrapped(panel)
    assert "&&" not in w.split("rc=0;", 1)[1], "a chain must not short-circuit"
    assert w.count("|| rc=$?") == 2, "every step must contribute to the exit code"
    assert "rc=0;" in w


def test_a_large_chain_does_not_blow_the_argv_limit(panel):
    """Linux caps ONE argv element at MAX_ARG_STRLEN (128 KiB). Passing the
    chain as `bash -c <string>` made the whole thing a single element, so a
    big enough fan-out failed to exec and surfaced only as "`screen -dmS`
    exited -2" -- which is what 83 source folders did at 1-ID-E. The script
    goes in a file, so nothing on the command line grows with the run."""
    paths = [f"/data/s1c/expid/pixirad/step/frame_{i:06d}.pixi.h5" for i in range(40)]
    base = ["python", "-m", "midas_gui.batch_cli", "--source-paths", *paths]
    extras = [base + ["--out-dir", f"/o/step_{i}"] for i in range(1, 83)]

    job = panel.launch(base + ["--out-dir", "/o/step_0"], name="big",
                       total_frames=900, out_dir="/o/step_0", extra_argvs=extras)
    assert job is not None, "the launch must not fail on a big fan-out"

    script = _wrapped(panel)
    assert len(script) > 128 * 1024, (
        "fixture too small to exercise the cap it is guarding "
        f"({len(script)} bytes)")
    assert script.count("--out-dir") == 83
    # Nothing handed to execve may scale with the number of folders.
    for a in panel.captured["args"]:
        assert len(str(a)) < 4096, f"argv element carries the script: {len(str(a))} bytes"


# ── progress across the chain ────────────────────────────────────────────
def _job(qapp, **kw):
    from PyQt5 import QtWidgets
    from midas_gui.job_queue import Job
    j = Job(session="s", logfile="l", meta_path="m", name="n",
            total_frames=kw.pop("total_frames", 1))
    j.progress = QtWidgets.QProgressBar()
    for k, v in kw.items():
        setattr(j, k, v)
    return j


def test_mid_chain_is_not_rendered_as_finished(qapp):
    """Reported from the beamline: the bar read "15 / 15 frames, Running" at
    folder 3 of 84, because each folder reports PROGRESS k/15 and the bar was
    driven straight off that -- filling and resetting once per folder."""
    from midas_gui.job_queue import JobQueuePanel
    j = _job(qapp, n_steps=84, step=3, seen_frames=15, step_total=15)
    JobQueuePanel._apply_progress(j)
    assert j.progress.value() < j.progress.maximum(), (
        "folder 3 of 84 complete must not render as a full bar")
    pct = 100.0 * j.progress.value() / j.progress.maximum()
    assert 3.0 < pct < 4.0, f"3 of 84 folders is ~3.6%, got {pct:.1f}%"


def test_the_bar_names_the_folder_it_is_on(qapp):
    from midas_gui.job_queue import JobQueuePanel
    j = _job(qapp, n_steps=84, step=3, seen_frames=7, step_total=15)
    JobQueuePanel._apply_progress(j)
    assert "folder 3/84" in j.progress.format()
    assert "7/15 frames" in j.progress.format()


def test_the_final_folder_completes_the_bar(qapp):
    from midas_gui.job_queue import JobQueuePanel
    j = _job(qapp, n_steps=84, step=84, seen_frames=15, step_total=15)
    JobQueuePanel._apply_progress(j)
    assert j.progress.value() == j.progress.maximum()


def test_a_single_folder_job_renders_exactly_as_before(qapp):
    from midas_gui.job_queue import JobQueuePanel
    j = _job(qapp, total_frames=877, n_steps=1, seen_frames=300, step_total=877)
    JobQueuePanel._apply_progress(j)
    assert j.progress.format() == "%v / %m frames"
    assert j.progress.maximum() == 877
    assert j.progress.value() == 300
