"""Several Hydra panels streaming at once — ``live_sources.HydraLiveMux``.

Asked for at the beamline: a live viewer for Hydra, "take inspiration from
the single panel but instead stream the hydra panels concurrently (but
within reason so that we do not make the gui so much slower)."

That parenthesis is the design constraint, so most of this file is about
it. Four detectors at full rate would otherwise cost four repaints plus a
composite rebuild per arrival, all on the GUI thread. The mux therefore
decouples arrival rate from paint rate, keeps only the newest frame per
panel, and refuses to stall on a panel that stops reporting.

No PV server and no Qt painting: the sources are stubbed, so this runs
anywhere and tests the assembly rules rather than EPICS.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def fake_sources(monkeypatch):
    """Replace create_live_source with a stub whose frames we drive by hand."""
    from PyQt5 import QtCore
    from midas_gui import live_sources

    class _Fake(QtCore.QObject):
        frameReady = QtCore.pyqtSignal(np.ndarray, int)
        connectionChanged = QtCore.pyqtSignal(bool)
        error = QtCore.pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.pv = None
            self.stopped = False

        def start(self, pv):
            self.pv = pv
            return True

        def stop(self):
            self.stopped = True

        def is_active(self):
            return self.pv is not None and not self.stopped

    made = []
    monkeypatch.setattr(live_sources, "create_live_source",
                        lambda backend, parent=None: made.append(_Fake()) or made[-1])
    return made


def _mux(**kw):
    from midas_gui.live_sources import HydraLiveMux
    return HydraLiveMux(**kw)


def _img(v):
    return np.full((4, 3), float(v), np.float32)


def _start(mux, panels=(1, 2, 3, 4)):
    from midas_gui.live_sources import hydra_pv_for_panel
    # The spelling the beamline actually publishes.
    pattern = "GE1:Pva1:Image"
    return mux.start({n: hydra_pv_for_panel(pattern, n) for n in panels})


# ── PV naming ───────────────────────────────────────────────────────────

def test_the_pv_pattern_substitutes_the_panel_token():
    """PVs follow the same geN convention the files do, so one pattern
    describes all four channels."""
    from midas_gui.live_sources import hydra_pv_for_panel
    assert hydra_pv_for_panel("20iddE:ge1:Pva1:Image", 3) == "20iddE:ge3:Pva1:Image"
    assert hydra_pv_for_panel("GE2:Pva1:Image", 4) == "GE4:Pva1:Image"


def test_the_pv_keeps_the_case_it_was_typed_in():
    """The real PVs are upper case (GE2:Pva1:Image) while the files are
    lower case (.ge2.h5), and PV names are case-sensitive -- lower-casing
    the substitution would produce a PV that does not exist, which looks
    exactly like a detector being switched off."""
    from midas_gui.live_sources import hydra_pv_for_panel
    assert hydra_pv_for_panel("GE2:Pva1:Image", 3) == "GE3:Pva1:Image"
    assert hydra_pv_for_panel("20iddE:GE1:Pva1:Image", 2) == "20iddE:GE2:Pva1:Image"
    assert hydra_pv_for_panel("det:ge4:img", 2) == "det:ge2:img"
    assert hydra_pv_for_panel("Ge1:img", 4) == "Ge4:img"


def test_a_pattern_with_no_token_is_left_alone():
    from midas_gui.live_sources import hydra_pv_for_panel
    assert hydra_pv_for_panel("plain:Image", 2) == "plain:Image"


# ── starting and stopping ───────────────────────────────────────────────

def test_one_source_per_panel_each_on_its_own_pv(app, fake_sources):
    mux = _mux()
    assert _start(mux) == {1: True, 2: True, 3: True, 4: True}
    assert mux.panels() == [1, 2, 3, 4]
    assert sorted(s.pv for s in fake_sources) == [
        f"GE{n}:Pva1:Image" for n in (1, 2, 3, 4)]


def test_only_the_panels_asked_for_are_started(app, fake_sources):
    """Part 1's selection decides this — an unticked panel gets no stream."""
    mux = _mux()
    _start(mux, panels=(1, 2, 4))
    assert mux.panels() == [1, 2, 4]
    assert len(fake_sources) == 3


def test_stop_tears_down_every_source(app, fake_sources):
    mux = _mux()
    _start(mux)
    mux.stop()
    assert all(s.stopped for s in fake_sources)
    assert mux.panels() == [] and not mux.is_active()


def test_the_sources_are_not_qt_children_of_the_mux(app, fake_sources):
    """A QObject child dies with its parent via the C++ cascade; for a live
    source mid-callback that is a crash, not an exception."""
    mux = _mux()
    _start(mux)
    assert all(s.parent() is None for s in fake_sources)


def test_one_bad_pv_does_not_take_the_other_panels_down(app, fake_sources,
                                                        monkeypatch):
    from midas_gui import live_sources
    real = live_sources.create_live_source
    calls = {"n": 0}

    def _flaky(backend, parent=None):
        src = real(backend, parent)
        calls["n"] += 1
        if calls["n"] == 2:                      # the second panel is broken
            src.start = lambda pv: (_ for _ in ()).throw(RuntimeError("no such PV"))
        return src
    monkeypatch.setattr(live_sources, "create_live_source", _flaky)

    mux = _mux()
    errs = []
    mux.error.connect(lambda n, msg: errs.append(n))
    res = _start(mux)
    assert res[2] is False and res[1] is True and res[4] is True
    assert errs == [2]
    assert 2 not in mux.panels()


# ── rule 1: arrivals do not drive repaints ──────────────────────────────

def test_an_arrival_emits_nothing_on_its_own(app, fake_sources):
    """The whole point: a frame landing must not cost a repaint. N panels
    cost one emission per timer tick, not N."""
    mux = _mux()
    _start(mux)
    got = []
    mux.framesReady.connect(lambda *a: got.append(a))
    for i, src in enumerate(fake_sources, start=1):
        src.frameReady.emit(_img(i), 7)
    assert got == [], "a frame arrival emitted without waiting for the timer"


def test_one_tick_emits_one_matched_set(app, fake_sources):
    mux = _mux()
    _start(mux)
    got = []
    mux.framesReady.connect(lambda imgs, fid, missing: got.append((sorted(imgs), fid, missing)))
    for src in fake_sources:
        src.frameReady.emit(_img(1), 7)
    mux._tick()
    assert got == [([1, 2, 3, 4], 7, [])]


def test_a_quiet_tick_emits_nothing(app, fake_sources):
    mux = _mux()
    _start(mux)
    got = []
    mux.framesReady.connect(lambda *a: got.append(a))
    mux._tick(); mux._tick()
    assert got == []


# ── rule 2: newest frame only, never a backlog ──────────────────────────

def test_only_the_newest_frame_per_panel_survives(app, fake_sources):
    """A detector outrunning the display must drop frames, not queue them:
    a backlog in a live view is both useless and unbounded."""
    mux = _mux()
    _start(mux, panels=(1, 2))
    got = []
    mux.framesReady.connect(lambda imgs, fid, missing: got.append((imgs, fid)))
    for fid in (1, 2, 3):
        for src in fake_sources:
            src.frameReady.emit(_img(fid * 10), fid)
    mux._tick()
    assert len(got) == 1
    imgs, fid = got[0]
    assert fid == 3
    assert all(np.allclose(im, 30.0) for im in imgs.values()), \
        "an older frame survived into the emitted set"


# ── rule 3: match when possible, never stall ────────────────────────────

def test_a_partial_set_is_held_back(app, fake_sources):
    """Panels are not in step; a half-updated composite would tear."""
    mux = _mux(sync_timeout_ms=10_000)
    _start(mux)
    got = []
    mux.framesReady.connect(lambda *a: got.append(a))
    fake_sources[0].frameReady.emit(_img(1), 5)
    fake_sources[1].frameReady.emit(_img(1), 5)
    mux._tick()
    assert got == [], "an incomplete set was emitted immediately"


def test_mismatched_ids_are_held_back(app, fake_sources):
    mux = _mux(sync_timeout_ms=10_000)
    _start(mux, panels=(1, 2))
    got = []
    mux.framesReady.connect(lambda *a: got.append(a))
    fake_sources[0].frameReady.emit(_img(1), 5)
    fake_sources[1].frameReady.emit(_img(1), 6)      # one frame behind
    mux._tick()
    assert got == []


def test_the_set_goes_out_once_the_straggler_catches_up(app, fake_sources):
    mux = _mux(sync_timeout_ms=10_000)
    _start(mux, panels=(1, 2))
    got = []
    mux.framesReady.connect(lambda imgs, fid, missing: got.append((fid, missing)))
    fake_sources[0].frameReady.emit(_img(1), 5)
    mux._tick()
    fake_sources[1].frameReady.emit(_img(1), 5)
    mux._tick()
    assert got == [(5, [])]


def test_a_dead_panel_does_not_freeze_the_others(app, fake_sources):
    """One detector dropping out must not stop the view updating. After the
    timeout, what is held goes out and the missing panel is named."""
    mux = _mux(sync_timeout_ms=0)        # expire immediately
    _start(mux, panels=(1, 2, 3))
    got = []
    mux.framesReady.connect(lambda imgs, fid, missing: got.append((sorted(imgs), fid, missing)))
    fake_sources[0].frameReady.emit(_img(1), 9)
    fake_sources[1].frameReady.emit(_img(1), 9)
    mux._tick()      # starts the clock
    mux._tick()      # timeout already elapsed
    assert len(got) == 1
    panels, fid, missing = got[0]
    assert panels == [1, 2]
    assert fid is None, "a timed-out set must not claim a matched frame id"
    assert missing == [3]


def test_recovery_after_a_timeout_emits_a_matched_set_again(app, fake_sources):
    mux = _mux(sync_timeout_ms=0)
    _start(mux, panels=(1, 2))
    got = []
    mux.framesReady.connect(lambda imgs, fid, missing: got.append((fid, missing)))
    fake_sources[0].frameReady.emit(_img(1), 1)
    mux._tick(); mux._tick()                     # times out, emits partial
    for src in fake_sources:
        src.frameReady.emit(_img(2), 2)
    mux._tick()
    assert got[-1] == (2, []), "a matched set did not resume after a timeout"


# ── repaint budget ──────────────────────────────────────────────────────

def test_the_repaint_interval_follows_max_fps(app):
    assert _mux(max_fps=10.0)._timer.interval() == 100
    assert _mux(max_fps=5.0)._timer.interval() == 200


def test_the_interval_is_floored_so_a_silly_fps_cannot_flood_the_loop(app):
    assert _mux(max_fps=1000.0)._timer.interval() >= 20
