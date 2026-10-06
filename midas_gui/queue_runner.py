"""Runs a Batch Queue: many samples, several at a time, one detector map per
calibration.

The scheduling decisions live in :mod:`queue_policy` (pure, unit-tested); this
is the Qt layer that owns the workers and translates their signals into policy
transitions. Per in-flight sample it starts one
``workers.BatchRunCoordinator`` in ``"sequential"`` mode, so the parallelism is
*across* samples rather than across frames within one — that keeps each
sample's progress, log lines and results cleanly attributable, and one sample
per core beats N cores fighting over one sample's frames when there are dozens
of samples to get through.

Modelled on ``HydraBatchPage``'s multi-run driver (``_run_all`` /
``_start_panel_worker`` / ``_on_panel_done`` / ``_maybe_finish_run``), which
has been doing per-run calibration, per-run source and per-run output directory
for the four GE panels for a while. That driver is keyed on ``ge1..ge4`` and
reaches straight into the page's cards, viewers and loader, so it is the shape
that is reused here, not the code.
"""
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore

from midas_gui.queue_policy import CANCELLED, DONE, FAILED, QueuePolicy
from midas_gui.workers import BatchRunCoordinator


def default_max_concurrent() -> int:
    """A deliberately conservative default: each in-flight sample holds its own
    frames and results, so the ceiling is memory, not cores."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


@dataclass
class RunItem:
    """Everything one sample needs to run — the queue's model resolved against
    its calibration and the queue-wide integration settings.

    ``group`` is the calibration node's key: every item sharing it shares a
    detector map. Built by the tab (see ``tab_queue``), not by this module,
    because resolving a calibration snapshot into an ``IntegrationSpec`` needs
    the geometry helpers and the tab already holds them."""
    key: str
    group: str
    label: str
    spec: object
    source_cfg: dict
    out_dir: str
    mask: object = None
    mask_is_file_backed: bool = False
    fmts: tuple = ("csv",)
    kernel: str = "subpixel4"
    corrections: tuple = (None, None)
    variance_cfg: Optional[dict] = None
    q_cfg: Optional[dict] = None
    frame_range: Optional[tuple] = None
    dark: object = None
    bright: object = None
    background: object = None
    bright_mode: str = "divide"
    weighted: bool = True
    im_trans: tuple = ()
    multi_azimuth: bool = False
    #: Optional column groups for the per-frame beam-monitor CSV
    #: ("env", "motors"). A tuple, not a set, so RunItem stays hashable-ish
    #: and serialises predictably like every other field here.
    ion_csv_extras: tuple = ()
    #: Filled in by the tab for the per-sample project write.
    calibration_snapshot: Optional[dict] = None
    extra: dict = field(default_factory=dict)


class SampleRunScheduler(QtCore.QObject):
    """Drives a list of :class:`RunItem` to completion.

    Signals carry the sample key so the tab can route progress to the right
    tree row; ``logLine`` is already prefixed with the sample label, since with
    several samples in flight unprefixed output is unreadable.
    """

    sampleStarted = QtCore.pyqtSignal(str)              # key
    sampleProgress = QtCore.pyqtSignal(str, int, int)   # key, done, total
    sampleFinished = QtCore.pyqtSignal(str, dict)       # key, worker payload
    sampleFailed = QtCore.pyqtSignal(str, str)          # key, message
    logLine = QtCore.pyqtSignal(str)
    runFinished = QtCore.pyqtSignal(dict)               # QueuePolicy.counts()

    def __init__(self, items, max_concurrent: int = 1, parent=None):
        super().__init__(parent)
        self._items = {it.key: it for it in items}
        self._order = [it.key for it in items]
        self._policy = QueuePolicy([(it.key, it.group) for it in items],
                                   max_concurrent=max_concurrent)
        self._workers: dict = {}       # key -> BatchRunCoordinator
        self._contexts: dict = {}      # group -> detector map
        self._finished_emitted = False

    # ── lifecycle ────────────────────────────────────────────────

    def start(self):
        if not self._order:
            self._finish()
            return
        self.logLine.emit(
            f"Starting {len(self._order)} sample(s), up to "
            f"{self._policy.max_concurrent} at a time.")
        self._pump()

    def cancel(self):
        """Stop issuing work and interrupt whatever is running.

        The interrupted coordinators still emit ``finished`` (with
        ``aborted``), which drains the in-flight set through the normal path —
        so completion is reported once, from one place."""
        dropped = self._policy.cancel()
        if dropped:
            self.logLine.emit(f"Cancelled — {len(dropped)} sample(s) not started.")
        for key, worker in list(self._workers.items()):
            try:
                worker.requestInterruption()
            except Exception:
                pass
        self._maybe_finish()

    def is_running(self) -> bool:
        return bool(self._workers)

    def counts(self) -> dict:
        return self._policy.counts()

    # ── the pump ─────────────────────────────────────────────────

    def _pump(self):
        """Start everything the policy currently allows, then check for the
        end. Called after every state change — start, context-ready, finish."""
        for key in self._policy.take_next():
            self._start_one(key)
        self._maybe_finish()

    def _start_one(self, key: str):
        item = self._items[key]
        context = self._contexts.get(item.group)
        try:
            Path(item.out_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._on_failed(key, f"could not create output directory: {e}")
            return
        try:
            worker = BatchRunCoordinator(
                item.spec, item.source_cfg, item.mask, item.out_dir, item.fmts,
                item.kernel, item.corrections, item.variance_cfg,
                q_cfg=item.q_cfg, frame_range=item.frame_range, parent=self,
                dark=item.dark, bright=item.bright, background=item.background,
                bright_mode=item.bright_mode, weighted=item.weighted,
                context=context, im_trans=item.im_trans,
                multi_azimuth=item.multi_azimuth,
                ion_csv_extras=item.ion_csv_extras,
                run_mode="sequential", n_workers=1)
        except Exception:
            self._on_failed(key, traceback.format_exc())
            return

        worker.progress.connect(
            lambda done, total, k=key: self.sampleProgress.emit(k, done, total))
        worker.finished.connect(lambda data, k=key: self._on_finished(k, data))
        worker.failed.connect(lambda msg, k=key: self._on_failed(k, msg))
        worker.log_line.connect(
            lambda line, k=key: self.logLine.emit(f"[{self._items[k].label}] {line}"))
        # Only the pilot's context matters — a later sample of the same group
        # was handed one already and re-emits the same object.
        worker.geom_ready.connect(lambda ctx, g=item.group: self._on_context(g, ctx))

        self._workers[key] = worker
        self.sampleStarted.emit(key)
        self.logLine.emit(f"[{item.label}] starting…")
        worker.start()

    def _on_context(self, group: str, ctx):
        if group in self._contexts:
            return
        self._contexts[group] = ctx
        self._policy.mark_context_ready(group)
        self.logLine.emit(f"Detector map ready for '{group}' — releasing its samples.")
        self._pump()

    def _on_finished(self, key: str, data: dict):
        self._workers.pop(key, None)
        item = self._items.get(key)
        aborted = bool(data.get("aborted"))
        state = CANCELLED if aborted else DONE
        self._policy.mark_finished(key, state)
        n = data.get("n", 0)
        if item is not None:
            verb = "aborted after" if aborted else "done —"
            self.logLine.emit(f"[{item.label}] {verb} {n} frame(s)")
        # Emitted even when aborted: a partial run still wrote real files and
        # the tab should still record them in that sample's project.
        self.sampleFinished.emit(key, data)
        self._pump()

    def _on_failed(self, key: str, msg: str):
        self._workers.pop(key, None)
        self._policy.mark_finished(key, FAILED)
        label = self._items[key].label if key in self._items else key
        self.logLine.emit(f"[{label}] ERROR:\n{msg}")
        self.sampleFailed.emit(key, msg)
        self._pump()

    def _maybe_finish(self):
        if self._workers or not self._policy.is_complete():
            return
        self._finish()

    def _finish(self):
        if self._finished_emitted:
            return
        self._finished_emitted = True
        counts = self._policy.counts()
        self.logLine.emit(
            "Queue complete — {done} done, {failed} failed, "
            "{cancelled} not run.".format(**counts))
        self.runFinished.emit(counts)
