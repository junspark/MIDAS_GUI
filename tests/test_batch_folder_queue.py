"""Chaining one batch run per source folder.

A recursive "Full folder" pick spans many source folders, each implying its
own output dir, so Batch Integrate runs them as a queue of separate batches
(see ``helpers.group_paths_by_output_dir``). This pins the queue bookkeeping
in ``_on_done``: a mid-queue finish must chain silently, and only the last
one may report and pop a dialog.

Qt import discipline per STATE.md: import inside fixtures, never at module
scope.
"""
from pathlib import Path

import pytest

pytest.importorskip("PyQt5.QtWidgets")

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(qapp, monkeypatch):
    from PyQt5 import QtCore, QtWidgets
    from midas_gui.tab_batch import BatchTab
    t = BatchTab()
    # Everything a finish does besides the queue bookkeeping.
    monkeypatch.setattr(t, "_update_cake_stack", lambda d: None)
    monkeypatch.setattr(t, "_log_to_project", lambda d: None)
    t.shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: t.shown.append(a[-1])))
    t.scheduled = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot",
                        staticmethod(lambda ms, fn: t.scheduled.append(fn)))
    return t


def test_a_plain_single_folder_run_is_untouched(tab):
    """No queue -> the original finish, dialog and all."""
    assert tab._pending_groups == [] and tab._group_tally is None
    tab._on_done({"n": 7})
    assert len(tab.shown) == 1
    assert tab.scheduled == [], "nothing to chain to"


def test_a_mid_queue_finish_chains_without_a_dialog(tab):
    tab._pending_groups = [(Path("/out/step_1"), ["b.h5"])]
    tab._group_tally = {"total": 2, "done": 0, "frames": 0}
    tab._on_done({"n": 5})
    assert tab.shown == [], "23 folders must not pop 23 dialogs"
    assert tab.scheduled == [tab._run], "next folder scheduled via the event loop"
    assert tab._group_tally["done"] == 1
    assert tab._group_tally["frames"] == 5


def test_the_last_folder_reports_the_whole_sweep(tab):
    tab._pending_groups = []
    tab._group_tally = {"total": 3, "done": 2, "frames": 10}
    tab._on_done({"n": 4})
    assert len(tab.shown) == 1
    msg = tab.shown[0]
    assert "14 frames" in msg, f"should sum every folder, got: {msg}"
    assert "3 folders" in msg
    assert tab._pending_groups == [] and tab._group_tally is None


def test_an_abort_ends_the_sweep_rather_than_the_folder(tab):
    tab._pending_groups = [(Path("/out/step_2"), ["c.h5"])]
    tab._group_tally = {"total": 3, "done": 1, "frames": 6}
    tab._on_done({"n": 2, "aborted": True})
    assert tab.scheduled == [], "an aborted sweep must not start the next folder"
    assert len(tab.shown) == 1
    assert tab._group_tally is None


def test_pressing_abort_clears_the_queue(tab):
    tab._pending_groups = [(Path("/out/step_1"), ["b.h5"])]
    tab._abort()                       # no worker running -> early return
    assert tab._pending_groups == [], "the queue must not outlive an abort"
