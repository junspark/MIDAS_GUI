"""Shutdown must stop every background thread, wherever it is parked.

`QThread: Destroyed while thread is still running` is Qt telling you a
worker outlived its owner; at interpreter exit it is a common hard
crash. It was reported from the beamline against a Hydra session.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dummy_thread():
    from PyQt5 import QtCore

    class _Dummy(QtCore.QThread):
        def run(self):
            for _ in range(400):
                if self.isInterruptionRequested():
                    return
                self.msleep(25)
    return _Dummy()


def test_a_worker_in_a_dict_on_a_nested_page_is_still_stopped(app):
    """The reported case, reproduced by construction.

    The old scan walked a hardcoded tab list, one level deep, and
    traversed lists only. The Hydra pages are at _batch_tab._hydra_page
    -- a level further down -- and HydraLoaderPanel keeps its
    ProjectionWorkers in a DICT. So every Hydra worker was invisible to
    shutdown, on a tab list that also never mentioned _bcorr_tab,
    _queue_tab, _dexp_tab or _zarr_tab.
    """
    from midas_gui.app import MainWindow
    w = MainWindow()
    page = w._batch_tab._ensure_hydra_page()
    th = _dummy_thread()
    page._loader._proj_workers[9] = th
    th.start()
    try:
        assert any(x is th for x in w._live_threads()), \
            "a dict-held worker on a nested page was not found"
        w._stop_all_workers()
        assert not th.isRunning(), "shutdown left a worker running"
    finally:
        th.requestInterruption()
        th.wait(2000)


def test_the_walk_is_breadth_first_so_the_depth_limit_means_something(app):
    """Depth-first plus an identity set loses objects: one reached first
    by a long path is marked seen and its children pushed past the
    limit, so a later short path to it is refused. The first version of
    this walk failed exactly that way -- it missed a worker five hops
    away with the limit at six."""
    from PyQt5 import QtCore
    from midas_gui.app import MainWindow
    w = MainWindow()

    # A chain long enough that a DFS could reach the tail the wrong way
    # round, with the thread hung off a short branch near the root.
    node = QtCore.QObject()
    w._probe_chain = node
    for _ in range(12):
        nxt = QtCore.QObject()
        node._next = nxt
        node = nxt
    th = _dummy_thread()
    w._probe_shallow = {"w": th}
    th.start()
    try:
        assert any(x is th for x in w._live_threads())
    finally:
        th.requestInterruption()
        th.wait(2000)


def test_every_tab_is_discovered_without_a_hardcoded_list(app):
    """A list has to be updated by whoever adds the next tab, and the
    failure is silent until someone closes the GUI mid-run."""
    from midas_gui.app import MainWindow
    w = MainWindow()
    names = {t.objectName() or type(t).__name__ for t in w._all_tabs()}
    assert len(w._all_tabs()) >= 10, names
    # the four the old hardcoded list forgot
    attrs = {n for n in vars(w) if n.endswith("_tab")}
    for missed in ("_bcorr_tab", "_queue_tab", "_zarr_tab"):
        if missed in attrs:
            assert any(getattr(w, missed) is t for t in w._all_tabs()), missed
