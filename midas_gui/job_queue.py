"""Linux-only background job queue for long Batch Integrate runs.

Ported from ``mpe_wf_saxs_waxs/gui_caking_launcher.py``'s ``CakingLauncher``
(``Job`` dataclass + the ``screen``/``tail -F`` job-supervision mechanics),
adapted to supervise ``python -m midas_gui.batch_cli`` (see ``batch_cli.py``)
instead of that project's caking shell script. A job is a detached ``screen``
session — it keeps running (and can be re-adopted) after this GUI closes,
which an in-process ``QThread`` (``BatchRunCoordinator``) cannot do.

Departures from the source, and why:
  - Progress is parsed with a trivial regex (``[batch|correct] PROGRESS d/n``)
    instead of mpe_wf's detector-context/frame-number log-scraping — we
    control our own CLI's stdout format, so there's nothing to scrape.
  - A prior GUI's still-running jobs are re-discovered via a JSON sidecar
    (``<session>.meta.json``) instead of decoding fields out of the session
    name — simpler and doesn't constrain the session-naming scheme.
  - No detector selection / DM-tree / beamline-root handling — Batch
    Integrate's single-detector tab has no such concepts.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets

# Every job's screenlog + JSON sidecar lives here, regardless of the job's
# own --out-dir — so adoption (after a GUI restart) can scan one fixed
# location instead of needing to already know each job's output folder.
JOBS_DIR = Path.home() / ".midas_gui" / "jobs"

_SESSION_PREFIX = "midasgui_batch_"
#: Both headless entry points report the same way: batch_cli tags its lines
#: [batch], correct_cli tags them [correct]. One regex serves both so a
#: Batch Correction job gets a progress bar too.
_PROGRESS_RE = re.compile(r"\[(?:batch|correct)\] PROGRESS (\d+)/(\d+)")
_DONE_RE = re.compile(r"\[launcher\] DONE exit=(\d+)")

_LOG_RULES = [
    (re.compile(r"\[launcher\]"), "#b78bff", True),
    (re.compile(r"\bERROR\b"), "#ff5f5f", True),
    (re.compile(r"\b(WARNING|WARN)\b"), "#d7861f", False),
    (re.compile(r"\[(?:batch|correct)\] FINISHED"), "#2fa84f", True),
    (re.compile(r"\[(?:batch|correct)\] PROGRESS"), "#5f87d7", False),
]


def _classify_line(line: str):
    for pat, color, bold in _LOG_RULES:
        if pat.search(line):
            return color, bold
    return "#3a3a3a", False


@dataclass
class Job:
    """One background Batch Integrate run: a detached ``screen`` session +
    its tail-streamed logfile."""
    session: str
    logfile: str
    meta_path: str
    name: str
    total_frames: int = 1
    out_dir: str = ""                # where the job writes results — see
                                      # batch_cli._write_results_sidecar and
                                      # JobQueuePanel's on_job_done callback
    start_time: float = field(default_factory=time.time)
    status: str = "Running"          # Running / Cancelling / Done / Failed(n) / Ended / Cancelled
    exit_code: Optional[int] = None
    seen_frames: int = 0
    log_offset: int = 0
    # UI widgets, populated by JobQueuePanel._add_job_row
    row_widget: Optional[QtWidgets.QWidget] = None
    label: Optional[QtWidgets.QLabel] = None
    progress: Optional[QtWidgets.QProgressBar] = None
    status_label: Optional[QtWidgets.QLabel] = None
    reattach_btn: Optional[QtWidgets.QPushButton] = None
    cancel_btn: Optional[QtWidgets.QPushButton] = None
    load_results_btn: Optional[QtWidgets.QPushButton] = None


def screen_available() -> bool:
    import shutil
    return shutil.which("screen") is not None


def _screen_session_alive(session: str) -> bool:
    try:
        out = subprocess.run(["screen", "-ls"], capture_output=True,
                             text=True, timeout=5)
    except Exception:
        return False
    return any(re.match(rf"\s*\d+\.{re.escape(session)}\s+\(", ln)
              for ln in (out.stdout or "").splitlines())


def _sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name.strip()) or "job"
    return name[:80]


class JobQueuePanel(QtWidgets.QWidget):
    """Embeddable panel: a "Refresh" bar and a scrollable list of active-job
    rows (most recent first). Log output for the most recently focused job's
    logfile is streamed into ``log_widget`` (a ``widgets.LogPanel``, owned by
    the caller) rather than a log widget of this panel's own — the caller
    shares one log surface across this panel and its own in-process run
    output.

    ``on_job_done``, if given, is called with the ``Job`` as soon as it's
    detected ``Done`` (screen session exited 0) — both right after that
    happens and again whenever "Load results" is clicked on an already-Done
    row (including one adopted from a prior GUI instance). A background
    ``screen`` job runs in a separate process with no Qt signals reaching
    this one, so this callback is the only way its results ever reach the
    caller; the caller reads ``job.out_dir`` for whatever results sidecar it
    left there."""

    def __init__(self, log_widget, parent=None, *, on_job_done=None):
        super().__init__(parent)
        self._log = log_widget
        self._on_job_done = on_job_done
        self._jobs: list[Job] = []
        self._active_log_job: Optional[Job] = None
        self._tail_proc: Optional[QtCore.QProcess] = None
        self._tail_buf: str = ""
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._build_ui()

        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(2000)
        self._poll.timeout.connect(self._poll_jobs)
        self._poll.start()
        self._adopt_existing_sessions()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4); v.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Background jobs"))
        top.addStretch(1)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.setToolTip(
            "Re-scan `screen -ls` for jobs launched by a prior GUI instance "
            "(or another one running now) and re-validate already-tracked jobs.")
        refresh_btn.clicked.connect(self._adopt_existing_sessions)
        top.addWidget(refresh_btn)
        v.addLayout(top)

        self._jobs_container = QtWidgets.QWidget()
        self._jobs_layout = QtWidgets.QVBoxLayout(self._jobs_container)
        self._jobs_layout.setContentsMargins(0, 0, 0, 0); self._jobs_layout.setSpacing(4)
        self._jobs_layout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._jobs_container)
        scroll.setMinimumHeight(90)
        scroll.setMaximumHeight(220)
        v.addWidget(scroll)

        self._empty_lbl = QtWidgets.QLabel("No background jobs.")
        self._empty_lbl.setStyleSheet("color:#888;font-size:10px;")
        self._jobs_layout.insertWidget(0, self._empty_lbl)

    def _refresh_empty_label(self):
        self._empty_lbl.setVisible(not self._jobs)

    # ── Launch ───────────────────────────────────────────────────────

    def launch(self, argv: list, *, name: str, total_frames: int,
              out_dir: str = "") -> Optional[Job]:
        """Launch ``argv`` (e.g. ``[sys.executable, "-m", "midas_gui.batch_cli", ...]``)
        in a fresh detached ``screen`` session. Returns the new ``Job``, or
        ``None`` if ``screen`` isn't available or a same-named session is
        already running (both cases show a message box). ``out_dir`` is the
        job's own ``--out-dir`` — recorded so ``on_job_done`` (and a later
        "Load results" click) knows where to find whatever results sidecar
        the job leaves behind."""
        if not screen_available():
            QtWidgets.QMessageBox.warning(
                self, "screen not found",
                "The `screen` command isn't available — background jobs "
                "need it (Linux only). Use the regular 'Start Integration' "
                "button for an in-process run instead.")
            return None

        session = f"{_SESSION_PREFIX}{_sanitize(name)}_{time.strftime('%Y%m%d-%H%M%S')}"
        if _screen_session_alive(session):
            QtWidgets.QMessageBox.warning(
                self, "Session exists",
                f'A screen session named "{session}" already exists.')
            return None

        logfile = str(JOBS_DIR / f"{session}.screenlog")
        meta_path = str(JOBS_DIR / f"{session}.meta.json")
        try:
            open(logfile, "w").close()
            Path(meta_path).write_text(json.dumps(
                {"name": name, "total_frames": total_frames,
                 "out_dir": out_dir, "started": time.time()}))
        except OSError as e:
            QtWidgets.QMessageBox.warning(
                self, "Cannot write log", f"Could not create job files:\n{e}")
            return None

        # `argv` runs `python -m midas_gui.batch_cli`, which only resolves
        # the `midas_gui` package when the process's cwd is the repo root
        # (that's what puts the repo root on sys.path[0] for `-m`) — the
        # spawned screen/bash session otherwise inherits whatever cwd the
        # GUI itself happened to have, so pin it explicitly here rather
        # than relying on that.
        repo_root = Path(__file__).resolve().parent.parent
        inner = " ".join(shlex.quote(str(c)) for c in argv)
        wrapped = (f'cd {shlex.quote(str(repo_root))} && {inner}; '
                   f'rc=$?; echo "[launcher] DONE exit=$rc"; sleep 3')
        screen_cmd = ["screen", "-dmS", session, "-L", "-Logfile", logfile,
                      "bash", "-c", wrapped]
        rc = QtCore.QProcess.execute("screen", screen_cmd[1:])
        if rc != 0:
            QtWidgets.QMessageBox.warning(
                self, "Launch failed", f"`screen -dmS` exited {rc}.")
            return None

        job = Job(session=session, logfile=logfile, meta_path=meta_path,
                 name=name, total_frames=max(1, total_frames), out_dir=out_dir)
        self._jobs.append(job)
        self._add_job_row(job)
        self._set_active_log_job(job, header_cmd=argv)
        self._refresh_empty_label()
        return job

    # ── Adoption (Refresh / startup) ─────────────────────────────────

    def _adopt_existing_sessions(self) -> None:
        try:
            out = subprocess.run(["screen", "-ls"], capture_output=True,
                                 text=True, timeout=5)
        except Exception:
            return
        line_re = re.compile(r"\s*\d+\.([^\s]+)\s+\(")
        for raw_line in (out.stdout or "").splitlines():
            m = line_re.match(raw_line)
            if not m:
                continue
            session = m.group(1)
            if not session.startswith(_SESSION_PREFIX):
                continue
            if any(j.session == session for j in self._jobs):
                continue
            logfile = str(JOBS_DIR / f"{session}.screenlog")
            meta_path = str(JOBS_DIR / f"{session}.meta.json")
            name, total_frames, out_dir = session, 1, ""
            try:
                meta = json.loads(Path(meta_path).read_text())
                name = meta.get("name", session)
                total_frames = int(meta.get("total_frames", 1))
                out_dir = meta.get("out_dir", "")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            job = Job(session=session, logfile=logfile, meta_path=meta_path,
                     name=name, total_frames=max(1, total_frames), out_dir=out_dir)
            if os.path.isfile(logfile):
                try:
                    job.log_offset = os.path.getsize(logfile)
                    job.seen_frames = self._last_progress_count(logfile)
                except OSError:
                    pass
            self._jobs.append(job)
            self._add_job_row(job)
            if job.progress is not None:
                job.progress.setValue(job.seen_frames)

        for job in list(self._jobs):
            if job.status in ("Running", "Cancelling") and \
                    not _screen_session_alive(job.session):
                try:
                    self._finalize_job(job)
                except Exception:
                    pass
        self._refresh_empty_label()

    @staticmethod
    def _last_progress_count(logfile: str) -> int:
        try:
            text = Path(logfile).read_text(errors="replace")
        except OSError:
            return 0
        last = 0
        for m in _PROGRESS_RE.finditer(text):
            last = int(m.group(1))
        return last

    # ── Active-jobs panel rows ────────────────────────────────────────

    def _add_job_row(self, job: Job) -> None:
        row = QtWidgets.QWidget()
        row.setAutoFillBackground(True)
        row.setStyleSheet("QWidget { background-color: #2a2d33; border-radius: 4px; }")
        hl = QtWidgets.QHBoxLayout(row)
        hl.setContentsMargins(6, 4, 6, 4); hl.setSpacing(6)

        label = QtWidgets.QLabel(job.name)
        label.setMinimumWidth(160)
        label.setStyleSheet("font-weight:bold;color:#eee;")
        label.setToolTip(f"session: {job.session}\nlogfile: {job.logfile}")

        progress = QtWidgets.QProgressBar()
        progress.setRange(0, max(1, job.total_frames))
        progress.setValue(job.seen_frames)
        progress.setFormat("%v / %m frames")
        progress.setMinimumWidth(160)

        status_label = QtWidgets.QLabel(job.status)
        status_label.setMinimumWidth(80)
        status_label.setStyleSheet("color:#d7861f;font-weight:bold;")

        show_log_btn = QtWidgets.QPushButton("Show log")
        show_log_btn.clicked.connect(lambda _=False, j=job: self._focus_log(j))
        reattach_btn = QtWidgets.QPushButton("Reattach")
        reattach_btn.setToolTip("Copy `screen -r <session>` to the clipboard.")
        reattach_btn.clicked.connect(lambda _=False, j=job: self._reattach_job(j))
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(lambda _=False, j=job: self._cancel_job(j))
        load_results_btn = QtWidgets.QPushButton("Load results")
        load_results_btn.setToolTip(
            "Populate the Waterfall/Stacked-profiles/Eta-R cakes tabs from "
            "this job's results (enabled once it finishes successfully).")
        load_results_btn.setEnabled(job.status == "Done")
        load_results_btn.clicked.connect(lambda _=False, j=job: self._load_results(j))

        for w in (label, progress, status_label, show_log_btn, reattach_btn,
                 cancel_btn, load_results_btn):
            hl.addWidget(w)
        hl.setStretchFactor(label, 1)

        job.row_widget = row; job.label = label; job.progress = progress
        job.status_label = status_label
        job.reattach_btn = reattach_btn; job.cancel_btn = cancel_btn
        job.load_results_btn = load_results_btn
        self._jobs_layout.insertWidget(0, row)

    def _load_results(self, job: Job) -> None:
        if self._on_job_done is not None:
            self._on_job_done(job)
        else:
            QtWidgets.QMessageBox.information(
                self, "No handler", "Nothing is registered to load this job's results.")

    def _remove_job_row(self, job: Job) -> None:
        if job.row_widget is not None:
            self._jobs_layout.removeWidget(job.row_widget)
            job.row_widget.deleteLater()
            job.row_widget = None
        if job in self._jobs:
            self._jobs.remove(job)
        if self._active_log_job is job:
            self._active_log_job = None
            if self._tail_proc is not None:
                self._tail_proc.kill()
                self._tail_proc.waitForFinished(500)
                self._tail_proc = None
        self._refresh_empty_label()

    # ── Log tailing ───────────────────────────────────────────────────

    def _focus_log(self, job: Job) -> None:
        if self._active_log_job is not job:
            self._set_active_log_job(job)

    def _set_active_log_job(self, job: Job, *, header_cmd=None) -> None:
        if self._tail_proc is not None:
            try:
                self._tail_proc.readyReadStandardOutput.disconnect()
            except TypeError:
                pass
            self._tail_proc.kill()
            self._tail_proc.waitForFinished(500)
            self._tail_proc = None
        self._tail_buf = ""
        self._active_log_job = job
        self._log.clear()
        if header_cmd is not None:
            self._append_log(f"$ {' '.join(shlex.quote(str(c)) for c in header_cmd)}")
        self._append_log(f"[launcher] screen session: {job.session}")
        self._append_log(f"[launcher] logfile:        {job.logfile}")
        self._append_log(f"[launcher] reattach with:  screen -r {job.session}")
        self._append_log("[launcher] job will survive closing this GUI.")

        for j in self._jobs:
            if j.row_widget is None:
                continue
            j.row_widget.setStyleSheet(
                "QWidget { background-color: #3d3517; border: 1px solid #d7861f; "
                "border-radius: 4px; }" if j is job else
                "QWidget { background-color: #2a2d33; border-radius: 4px; }")

        self._tail_proc = QtCore.QProcess(self)
        self._tail_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._tail_proc.readyReadStandardOutput.connect(self._on_tail_stdout)
        self._tail_proc.start("tail", ["-n", "2000", "-F", job.logfile])

    def _append_log(self, line: str) -> None:
        import html
        color, bold = _classify_line(line)
        weight = "bold" if bold else "normal"
        escaped = html.escape(line).replace(" ", "&nbsp;")
        self._log.append_html(f'<span style="color:{color};font-weight:{weight};">{escaped}</span>')

    def _on_tail_stdout(self) -> None:
        if self._tail_proc is None:
            return
        data = bytes(self._tail_proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._tail_buf += data
        while "\n" in self._tail_buf:
            line, self._tail_buf = self._tail_buf.split("\n", 1)
            self._append_log(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Reattach / cancel ─────────────────────────────────────────────

    def _reattach_job(self, job: Job) -> None:
        cmd = f"screen -r {job.session}"
        try:
            QtWidgets.QApplication.clipboard().setText(cmd)
        except Exception:
            pass
        QtWidgets.QMessageBox.information(
            self, "Reattach", f"Copied to clipboard:\n{cmd}\n\nPaste it in a terminal.")

    def _cancel_job(self, job: Job) -> None:
        if job.status != "Running":
            return
        QtCore.QProcess.execute("screen", ["-S", job.session, "-X", "quit"])
        job.status = "Cancelling"
        if job.status_label is not None:
            job.status_label.setText("Cancelling")

    # ── Polling ───────────────────────────────────────────────────────

    def _poll_jobs(self) -> None:
        for job in list(self._jobs):
            try:
                self._update_job_progress(job)
                if job.status in ("Running", "Cancelling") and \
                        not _screen_session_alive(job.session):
                    self._finalize_job(job)
            except Exception as e:
                if self._active_log_job is job:
                    self._append_log(f"[launcher] poll error: {e}")

    def _update_job_progress(self, job: Job) -> None:
        if not os.path.isfile(job.logfile):
            return
        try:
            with open(job.logfile, "rb") as f:
                f.seek(job.log_offset)
                data = f.read()
                job.log_offset += len(data)
        except OSError:
            return
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        last_done = last_total = None
        for m in _PROGRESS_RE.finditer(text):
            last_done, last_total = int(m.group(1)), int(m.group(2))
        if last_done is not None:
            job.seen_frames = last_done
            if last_total and last_total != job.total_frames:
                job.total_frames = last_total
                if job.progress is not None:
                    job.progress.setRange(0, max(1, job.total_frames))
            if job.progress is not None:
                job.progress.setValue(job.seen_frames)

    def _finalize_job(self, job: Job) -> None:
        was_cancelling = job.status == "Cancelling"
        exit_code = self._parse_exit_from_logfile(job.logfile)
        job.exit_code = exit_code
        if was_cancelling:
            job.status, color = "Cancelled", "#999"
        elif exit_code is None:
            job.status, color = "Ended", "#999"
        elif exit_code == 0:
            job.status, color = "Done", "#2fa84f"
        else:
            job.status, color = f"Failed ({exit_code})", "#e05252"
        if job.status_label is not None:
            job.status_label.setText(job.status)
            job.status_label.setStyleSheet(f"color:{color};font-weight:bold;")
        if job.cancel_btn is not None:
            job.cancel_btn.setEnabled(False)
        if job.load_results_btn is not None:
            job.load_results_btn.setEnabled(job.status == "Done")
        self._update_job_progress(job)
        if self._active_log_job is job:
            self._append_log(f"[launcher] session ended ({job.status}).")
        if job.status == "Done" and self._on_job_done is not None:
            self._on_job_done(job)

    def _parse_exit_from_logfile(self, logfile: str) -> Optional[int]:
        if not logfile or not os.path.isfile(logfile):
            return None
        try:
            with open(logfile) as f:
                tail = f.readlines()[-30:]
        except OSError:
            return None
        for line in reversed(tail):
            m = _DONE_RE.search(line)
            if m:
                return int(m.group(1))
        return None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Stop polling/tailing on app close. Jobs' screen sessions keep
        running — that's the whole point of a detached screen session."""
        self._poll.stop()
        if self._tail_proc is not None:
            self._tail_proc.kill()
            self._tail_proc.waitForFinished(500)
            self._tail_proc = None
