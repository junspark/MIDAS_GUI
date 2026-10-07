"""Live detector-stream sources for the Data Viewer's "Live Data" card.

Two EPICS backends are supported, selected per-device via each device
dict's ``backend`` field ("pva" | "ca", default "pva" -- see
``constants.DEFAULT_DEVICES``/``settings.BUNDLED_PROFILES``):

- :class:`PvaLiveSource` -- pvAccess (PVA), for beamlines whose areaDetector
  IOC runs an NDPluginPva plugin (20-ID-D/E, 1-ID-E).
- :class:`CaLiveSource` -- plain Channel Access (CA), for beamlines whose
  IOC has no PVA plugin and only exposes images via the areaDetector
  NDPluginStdArrays plugin (e.g. 17-BM's Varex detector).

Both classes share one signal/method contract so ``widgets.DataLoaderPanel``
can treat them interchangeably:

    frameReady(np.ndarray, int)   -- decoded frame, frame/unique id
    connectionChanged(bool)       -- underlying channel connected/dropped
    error(str)                   -- human-readable failure message
    start(pv_name: str) -> bool
    stop() -> None
    is_active() -> bool

Use :func:`create_live_source` to pick the right class from a device's
``backend`` field.
"""
from __future__ import annotations

import numpy as np
from PyQt5 import QtCore


class PvaLiveSource(QtCore.QObject):
    """Subscribes to an EPICS PVA image PV (NTNDArray) and emits decoded
    numpy frames.

    pvapy's ``Channel.monitor()`` delivers callbacks on its own internal
    thread; this class never touches Qt widgets directly, only emits
    signals — Qt auto-queues those onto the receiving (GUI) thread."""

    frameReady = QtCore.pyqtSignal(np.ndarray, int)      # image, uniqueId
    connectionChanged = QtCore.pyqtSignal(bool)
    error = QtCore.pyqtSignal(str)

    _REQUEST = "field(value,dimension,uniqueId,attribute,codec,uncompressedSize)"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channel = None
        self._AdImageUtility = None

    def start(self, pv_name: str) -> bool:
        self.stop()
        try:
            import pvapy as pva
            from pvapy.utility.adImageUtility import AdImageUtility
        except ImportError as e:
            self.error.emit(f"pvapy not installed: {e}")
            return False
        self._AdImageUtility = AdImageUtility
        try:
            self._channel = pva.Channel(pv_name)
            self._channel.setConnectionCallback(self._on_connection)
            self._channel.monitor(self._on_value, self._REQUEST)
        except Exception as e:
            self.error.emit(str(e))
            self._channel = None
            return False
        return True

    def _on_connection(self, is_connected):
        self.connectionChanged.emit(bool(is_connected))

    def _on_value(self, pv_object):
        try:
            image_id, image, *_ = self._AdImageUtility.reshapeNtNdArray(pv_object)
        except Exception as e:
            self.error.emit(f"Frame decode failed: {e}")
            return
        if image is not None:
            self.frameReady.emit(np.asarray(image, dtype=np.float32), int(image_id))

    def stop(self):
        if self._channel is not None:
            try:
                self._channel.stopMonitor()
            except Exception:
                pass
            self._channel = None

    def is_active(self) -> bool:
        return self._channel is not None


def _reshape_std_arrays(raw, size0, size1, size2, color_mode: int) -> np.ndarray:
    """Reshape a flat NDPluginStdArrays ``ArrayData`` waveform into a
    ``(rows, cols[, 3])`` float32 array, using the areaDetector NDArray
    dims convention (``dims[0]`` is the fastest-varying axis, i.e. the
    *last* numpy axis once reshaped).

    ``size0``/``size1``/``size2`` are ``ArraySize0_RBV``/``1_RBV``/``2_RBV``;
    ``color_mode`` is ``ColorMode_RBV`` (0/1=Mono/Bayer, 2=RGB1, 3=RGB2,
    4=RGB3).

    Mono is the fully-verified path -- Varex-class flat-panel detectors are
    always single-plane Mono. The RGB1/2/3 branches are implemented per the
    areaDetector convention but are best-effort/untested without a real
    color CA source; a Bayer-pattern sensor (color_mode 1) is treated as
    raw Mono data (no de-Bayer step) since no such device is in scope here.
    """
    raw = np.asarray(raw, dtype=np.float32).ravel()
    color_mode = int(color_mode)
    if color_mode in (0, 1):
        # dims = [NX, NY] -> numpy shape (NY, NX)
        width, height = int(size0), int(size1)
        return raw.reshape(height, width)
    if color_mode == 2:
        # RGB1: dims = [3, NX, NY] -> numpy shape (NY, NX, 3) directly
        width, height = int(size1), int(size2)
        return raw.reshape(height, width, 3)
    if color_mode == 3:
        # RGB2: dims = [NX, 3, NY] -> numpy shape (NY, 3, NX) -> move color last
        width, height = int(size0), int(size2)
        return raw.reshape(height, 3, width).transpose(0, 2, 1)
    if color_mode == 4:
        # RGB3: dims = [NX, NY, 3] -> numpy shape (3, NY, NX) -> move color last
        width, height = int(size0), int(size1)
        return raw.reshape(3, height, width).transpose(1, 2, 0)
    raise ValueError(f"unsupported ColorMode_RBV value: {color_mode}")


class CaLiveSource(QtCore.QObject):
    """Subscribes to an EPICS Channel Access (CA) image readout published
    by an areaDetector NDPluginStdArrays plugin (conventionally named
    ``"image1:"``) and emits decoded numpy frames.

    Unlike PVA's single NTNDArray channel, the NDPluginStdArrays convention
    spreads a frame across several plain CA records under one plugin
    prefix: an ``ArrayData`` waveform plus scalar metadata PVs
    (``ArraySize0_RBV``/``ArraySize1_RBV``/``ArraySize2_RBV``/
    ``ColorMode_RBV``/``UniqueId_RBV``). ``pv_name`` passed to
    :meth:`start` is that plugin *prefix* (e.g. ``"17bmVarex:image1:"``),
    not a single PV -- this class derives the individual record names from
    it internally so its external signature stays identical to
    :class:`PvaLiveSource`.

    pyepics delivers monitor/connection callbacks on its own CA
    event-processing thread, exactly like pvapy does for PVA; this class
    never touches Qt widgets directly, only emits signals — Qt auto-queues
    those onto the receiving (GUI) thread."""

    frameReady = QtCore.pyqtSignal(np.ndarray, int)      # image, uniqueId
    connectionChanged = QtCore.pyqtSignal(bool)
    error = QtCore.pyqtSignal(str)

    _SUF_DATA = "ArrayData"
    _SUF_SIZE0 = "ArraySize0_RBV"
    _SUF_SIZE1 = "ArraySize1_RBV"
    _SUF_SIZE2 = "ArraySize2_RBV"
    _SUF_COLOR = "ColorMode_RBV"
    _SUF_UID = "UniqueId_RBV"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pvs = []          # every epics.PV opened by start(), for stop()
        self._data_pv = None
        self._size0 = self._size1 = self._size2 = None
        self._color_mode = 0
        # UniqueId_RBV isn't guaranteed present on every IOC; fall back to a
        # local monotonically-incrementing counter so frameReady's contract
        # (a real int id) is never violated even when it's missing.
        self._uid_ok = False
        self._local_uid = 0

    def start(self, pv_name: str) -> bool:
        self.stop()
        try:
            import epics
        except ImportError as e:
            self.error.emit(f"pyepics not installed: {e}")
            return False
        base = pv_name
        try:
            size0_pv = epics.PV(base + self._SUF_SIZE0, callback=self._on_size0, auto_monitor=True)
            size1_pv = epics.PV(base + self._SUF_SIZE1, callback=self._on_size1, auto_monitor=True)
            size2_pv = epics.PV(base + self._SUF_SIZE2, callback=self._on_size2, auto_monitor=True)
            color_pv = epics.PV(base + self._SUF_COLOR, callback=self._on_color, auto_monitor=True)
            uid_pv = epics.PV(base + self._SUF_UID, callback=self._on_uid, auto_monitor=True)
            data_pv = epics.PV(base + self._SUF_DATA, callback=self._on_array,
                                connection_callback=self._on_connection, auto_monitor=True)
        except Exception as e:
            self.error.emit(str(e))
            self.stop()
            return False
        self._pvs = [size0_pv, size1_pv, size2_pv, color_pv, uid_pv, data_pv]
        self._data_pv = data_pv
        return True

    def _on_connection(self, pvname=None, conn=None, **kw):
        self.connectionChanged.emit(bool(conn))

    def _on_size0(self, value=None, **kw):
        self._size0 = None if value is None else int(value)

    def _on_size1(self, value=None, **kw):
        self._size1 = None if value is None else int(value)

    def _on_size2(self, value=None, **kw):
        self._size2 = None if value is None else int(value)

    def _on_color(self, value=None, **kw):
        self._color_mode = 0 if value is None else int(value)

    def _on_uid(self, value=None, **kw):
        if value is not None:
            self._uid_ok = True
            self._local_uid = int(value)

    def _on_array(self, value=None, **kw):
        if value is None or self._size0 is None or self._size1 is None:
            return  # metadata not ready yet -- drop this frame rather than guess a shape
        try:
            frame = _reshape_std_arrays(value, self._size0, self._size1,
                                         self._size2, self._color_mode)
        except Exception as e:
            self.error.emit(f"Frame decode failed: {e}")
            return
        if not self._uid_ok:
            self._local_uid += 1
        self.frameReady.emit(frame, int(self._local_uid))

    def stop(self):
        for pv in self._pvs:
            try:
                pv.disconnect()
            except Exception:
                pass
        self._pvs = []
        self._data_pv = None
        self._size0 = self._size1 = self._size2 = None
        self._color_mode = 0
        self._uid_ok = False
        self._local_uid = 0

    def is_active(self) -> bool:
        return self._data_pv is not None


def create_live_source(backend, parent=None):
    """Return the live-source instance for ``backend`` ("pva" | "ca").

    Anything falsy or unrecognized defaults to PVA -- this is what keeps
    every existing device/profile (which predate the ``backend`` field
    entirely) behaving identically to before this field existed."""
    if (backend or "pva").strip().lower() == "ca":
        return CaLiveSource(parent)
    return PvaLiveSource(parent)


# ─────────────────────────────────────────────────────────────────────────────
#  Hydra: several panels streaming at once
# ─────────────────────────────────────────────────────────────────────────────

def hydra_pv_for_panel(pattern: str, n: int) -> str:
    """``pattern`` with every ``geN`` token repointed at panel ``n``.

    The same token :func:`helpers.hydra_siblings` substitutes in file
    paths, applied to PV names -- the detectors are named the same way in
    both places, so one pattern describes all four channels and is typed
    once. A pattern with no ``geN`` token comes back unchanged, which is
    the "one PV for everything" case and is the caller's to handle.

    **Only the digit changes; the case of the letters is kept exactly as
    typed.** PV names are case-sensitive and the real ones are upper case
    (``GE2:Pva1:Image``), while the files are lower case (``.ge2.h5``) --
    lower-casing the substitution here would silently produce a PV that
    does not exist, which looks identical to a detector being off.
    """
    from midas_gui.helpers import _HYDRA_PANEL_RE
    return _HYDRA_PANEL_RE.sub(lambda m: m.group(0)[:-1] + str(int(n)), pattern)


class HydraLiveMux(QtCore.QObject):
    """Several :class:`PvaLiveSource`/:class:`CaLiveSource` as one stream.

    Hydra needs 2-4 detectors live at once, and its consumers want a
    ``{panel: image}`` set per frame rather than one image. This owns one
    ordinary single-panel source per panel -- ``create_live_source``
    unchanged -- and does three things on top, all of which exist to keep
    the GUI responsive with four detectors running:

    1. **Arrival rate is decoupled from paint rate.** A frame arriving only
       updates a latest-frame map; a single repaint timer emits
       ``framesReady``. N panels therefore cost ONE repaint per tick, not
       N, and a detector running faster than the display cannot drive the
       event loop.
    2. **Never queue, always replace.** Only the newest frame per panel is
       kept. A GUI that falls behind drops frames instead of accumulating a
       backlog it can never clear -- for a live view, the newest frame is
       the only one anybody wants.
    3. **Emit a matched set when one exists, but never stall.** Panels are
       not in step, so a set is held back until every live panel reports
       the same ``image_id``. If that has not happened within
       ``sync_timeout_ms``, whatever is held goes out anyway and
       ``framesReady`` reports which panels were missing -- one dead panel
       must not freeze the other three.

    Sources are deliberately UNPARENTED and stopped in :meth:`stop`: a
    QObject child is destroyed by Qt's parent-owns-children cascade the
    moment its parent is, and for a live source mid-callback that is a
    crash rather than an exception. See ``DataLoaderPanel._start_preview_worker``.
    """

    #: ``{panel: image}``, the frame id they share (or ``None`` when the set
    #: went out on the staleness timeout), and the panels that were missing.
    framesReady = QtCore.pyqtSignal(dict, object, list)
    #: panel, connected
    connectionChanged = QtCore.pyqtSignal(int, bool)
    #: panel, message
    error = QtCore.pyqtSignal(int, str)

    def __init__(self, parent=None, *, max_fps: float = 10.0,
                 sync_timeout_ms: int = 2000):
        super().__init__(parent)
        self._sources: dict = {}        # panel -> live source
        self._latest: dict = {}         # panel -> (image, image_id)
        self._dirty = False
        self._sync_timeout_ms = int(sync_timeout_ms)
        self._waiting_since = None      # monotonic ms when the set went partial
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(max(20, int(1000.0 / max(0.5, float(max_fps)))))
        self._timer.timeout.connect(self._tick)

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self, pv_by_panel: dict, backend: str = "pva") -> dict:
        """Open one channel per panel. Returns ``{panel: started?}``."""
        self.stop()
        results = {}
        for n, pv in sorted(pv_by_panel.items()):
            src = create_live_source(backend)        # unparented, on purpose
            src.frameReady.connect(
                lambda img, fid, n=n: self._on_frame(n, img, fid))
            src.connectionChanged.connect(
                lambda ok, n=n: self.connectionChanged.emit(n, bool(ok)))
            src.error.connect(lambda msg, n=n: self.error.emit(n, str(msg)))
            try:
                ok = bool(src.start(pv))
            except Exception as exc:                 # a bad PV must not take
                self.error.emit(n, str(exc))         # the other panels down
                ok = False
            results[n] = ok
            if ok:
                self._sources[n] = src
        if self._sources:
            self._timer.start()
        return results

    def stop(self):
        self._timer.stop()
        for src in self._sources.values():
            try:
                src.stop()
            except Exception:
                pass
        self._sources.clear()
        self._latest.clear()
        self._dirty = False
        self._waiting_since = None

    def is_active(self) -> bool:
        return bool(self._sources)

    def panels(self) -> list:
        """The panels currently streaming."""
        return sorted(self._sources)

    def latest(self) -> dict:
        """``{panel: image}`` as last received -- whatever is held right
        now, matched or not."""
        return {n: img for n, (img, _fid) in self._latest.items()}

    # ── arrival / emission ──────────────────────────────────────────────
    def _on_frame(self, n: int, image, image_id: int):
        """Runs per arrival, so it stays cheap: store and mark dirty. All
        the work happens on the timer."""
        self._latest[n] = (image, int(image_id))
        self._dirty = True

    def _matched_id(self):
        """The frame id every streaming panel has reported, or ``None``."""
        live = set(self._sources)
        if not live or set(self._latest) != live:
            return None
        ids = {fid for _img, fid in self._latest.values()}
        return ids.pop() if len(ids) == 1 else None

    def _tick(self):
        if not self._dirty or not self._sources:
            return
        now = QtCore.QDateTime.currentMSecsSinceEpoch()
        fid = self._matched_id()
        if fid is not None:
            self._waiting_since = None
            self._emit(fid, [])
            return
        # Partial set. Give the stragglers until the timeout, then go
        # without them rather than freezing the view on one dead panel.
        if self._waiting_since is None:
            self._waiting_since = now
        elif now - self._waiting_since >= self._sync_timeout_ms:
            missing = sorted(set(self._sources) - set(self._latest))
            self._waiting_since = None
            self._emit(None, missing)

    def _emit(self, frame_id, missing: list):
        self._dirty = False
        self.framesReady.emit(self.latest(), frame_id, missing)
