"""Application shell: MainWindow, dark palette, Dioptas-inspired stylesheet, main()."""
from __future__ import annotations

import faulthandler
import hashlib
import inspect
import json
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import midas_gui._paths  # noqa: F401  (KMP_DUPLICATE_LIB_OK env var)
from midas_gui import __version__
from PyQt5 import QtCore, QtGui, QtWidgets

# ── Crash diagnostics ─────────────────────────────────────────────────────────
# On Windows a startup error (or an exception raised inside a Qt slot — which
# PyQt5 turns into a hard abort) makes the window "pop up and die" with no visible
# traceback, especially when launched by double-click (no console).  We log every
# uncaught Python exception and native fault to a file and, if a QApplication is
# up, show it in a dialog.  Installing our own excepthook also stops PyQt5 from
# aborting the process on a slot exception, so the app survives non-fatal errors.

_LOG_FILE = Path.home() / "midas_gui_error.log"


def _log(text: str) -> None:
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now().isoformat()} =====\n{text}\n")
    except Exception:
        pass


def _excepthook(exc_type, exc, tb) -> None:
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    _log(msg)
    try:
        sys.stderr.write(msg)
    except Exception:
        pass
    try:
        if QtWidgets.QApplication.instance() is not None:
            QtWidgets.QMessageBox.critical(
                None, "MIDAS GUI — unexpected error",
                f"{exc_type.__name__}: {exc}\n\nFull traceback written to:\n{_LOG_FILE}")
    except Exception:
        pass


def _install_diagnostics() -> None:
    sys.excepthook = _excepthook
    try:
        faulthandler.enable(open(_LOG_FILE, "a", encoding="utf-8"))
    except Exception:
        pass
    _log(f"MIDAS GUI v{__version__} starting — Python {sys.version.split()[0]}, "
         f"Qt {QtCore.QT_VERSION_STR}, PyQt {QtCore.PYQT_VERSION_STR}, platform {sys.platform}")

from midas_gui.helpers import _make_checkmark_svg, _make_arrow_svg
from midas_gui import style as S
from midas_gui import bridge_server
from midas_gui.auto_attenuation import refresh_server as auto_att_refresh_server
from midas_gui.tab_view import DataViewerTab
from midas_gui.tab_mask import MaskTab
from midas_gui.tab_calibrate import CalibrationTab
from midas_gui.tab_batch import BatchTab
from midas_gui.tab_zarrviewer import ZarrViewerTab
from midas_gui.tab_queue import BatchQueueTab
from midas_gui.tab_refine import RefinementTab
from midas_gui.tab_corrections import CorrectionsTab
from midas_gui.tab_pdf import PDFTab
from midas_gui.tab_texture import TextureTab
from midas_gui.tab_pumpprobe import PumpProbeTab
from midas_gui.tab_export import ExportTab
from midas_gui import constants as C
from midas_gui import project
from midas_gui import settings

_CHECKMARK_SVG = _make_checkmark_svg()
_ARROW_UP_SVG = _make_arrow_svg("up")
_ARROW_DOWN_SVG = _make_arrow_svg("down")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1600, 950)
        self._workspace_dirty = False
        self._last_saved_hash: Optional[str] = None
        self._project_ctx = project.ProjectContext()  # currently-open project (path also
                                                        # doubles as "the file Ctrl+S targets")
        self._build_ui()
        self._update_window_title()
        # Baseline for the dirty-check timer below — an untouched, freshly
        # built window must never report itself as having unsaved changes.
        self._last_saved_hash = self._hash_workspace_state(self._serialize_workspace()[0])

        # Periodic, cheap (no sidecar file I/O — see _serialize_workspace)
        # dirty-state check driving the window-title "*" / Close-confirm
        # prompt, plus a separate, much less frequent autosave of a
        # crash-recovery draft. Neither ever touches a Project's `.h5` file.
        self._dirty_timer = QtCore.QTimer(self)
        self._dirty_timer.setInterval(7_000)
        self._dirty_timer.timeout.connect(self._check_workspace_dirty)
        self._dirty_timer.start()
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(5 * 60_000)
        self._autosave_timer.timeout.connect(self._autosave_tick)
        self._autosave_timer.start()

        # Lets B-PILOT (a separate Bluesky plan-runner GUI) auto-start Live
        # Data on a detector's PVA channel when it launches a scan — no-op
        # if B-PILOT never connects. See midas_gui/bridge_server.py.
        self._bridge_server = bridge_server.BridgeServer(
            self._resolve_and_start_live, log_fn=_log)
        self._bridge_server.start()

        # Lets a still-open, detached Auto Attenuation popup ask for a fresh
        # data/dark/mask snapshot without this GUI ever handing over live
        # objects — see midas_gui/auto_attenuation/refresh_server.py.
        self._auto_att_refresh_server = auto_att_refresh_server.AutoAttenuationRefreshServer(
            self._on_auto_attenuation_refresh_request, log_fn=_log)
        self._auto_att_refresh_server.start()

    def _resolve_and_start_live(self, prefix: str) -> None:
        resolved = bridge_server.resolve_pv(prefix, C.DEVICES)
        if resolved is None:
            _log(f"MIDAS bridge: no DEVICES entry for prefix {prefix!r}")
            return
        pv, backend = resolved
        self._view_tab.start_live_pv(pv, backend)

    def _build_ui(self):
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Profile selector, pinned to the same row as the tab bar (top-left
        # corner widget) so it reads "Profile: <dropdown> | <tabs>" as one
        # header row without disturbing centralWidget() (tests assert
        # win.centralWidget().count() directly — see test_smoke.py).
        profile_corner = QtWidgets.QWidget()
        profile_row = QtWidgets.QHBoxLayout(profile_corner)
        profile_row.setContentsMargins(8, 0, 6, 0)
        profile_row.setSpacing(6)
        profile_label = QtWidgets.QLabel("Profile:")
        profile_font = profile_label.font()
        profile_font.setPointSize(profile_font.pointSize() + 3)
        profile_font.setBold(True)
        profile_label.setFont(profile_font)
        self._profile_combo = QtWidgets.QComboBox()
        self._profile_combo.setFont(profile_font)
        self._profile_combo.setMinimumWidth(140)
        profile_row.addWidget(profile_label)
        profile_row.addWidget(self._profile_combo)

        # Exp ID, mpe_wf_saxs_waxs style (e.g. "park_may26") — pinned here
        # rather than living inside any one tab so every tab's own output-path
        # logic can share it (today just Batch Integrate's "Suggest" button,
        # via set_expid_provider() below). Persists across restarts as a
        # plain last-used value (settings.get/set_last_expid); once a Project
        # is open its own saved Exp ID travels with it instead — see
        # _serialize_workspace/_apply_workspace_state/save_project.
        expid_label = QtWidgets.QLabel("Exp ID:")
        expid_label.setFont(profile_font)
        self._expid_ed = QtWidgets.QLineEdit()
        self._expid_ed.setFont(profile_font)
        self._expid_ed.setMaximumWidth(120)
        self._expid_ed.setPlaceholderText("park_may26")
        self._expid_ed.setToolTip(
            "Experiment ID (mpe_wf_saxs_waxs style, e.g. 'park_may26'). "
            "Shared across every tab — Batch Integrate's 'Suggest' output-"
            "folder button inserts it as <expid>_bc/. Saved with the Project "
            "once one is open; otherwise remembered as your last-used value.")
        self._expid_ed.setText(settings.get_last_expid())
        self._expid_ed.editingFinished.connect(
            lambda: settings.set_last_expid(self._expid_ed.text().strip()))
        profile_row.addWidget(expid_label)
        profile_row.addWidget(self._expid_ed)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        profile_row.addWidget(sep)
        tabs.setCornerWidget(profile_corner, QtCore.Qt.TopLeftCorner)
        self._refresh_profile_combo()
        self._profile_combo.activated[str].connect(self._on_header_profile_changed)

        # Active-project indicator, pinned to the opposite (top-right) corner
        # of the same tab bar row — deliberately separate from the muted
        # status-bar "Project: none" label (kept as-is), since this one needs
        # to be prominent/high-contrast rather than a quiet status readout.
        self._header_project_lbl = QtWidgets.QLabel("")
        proj_font = self._header_project_lbl.font()
        proj_font.setPointSize(proj_font.pointSize() + 2)
        proj_font.setBold(True)
        self._header_project_lbl.setFont(proj_font)
        self._header_project_lbl.setStyleSheet("color: #3ddc84; padding-right: 8px;")
        tabs.setCornerWidget(self._header_project_lbl, QtCore.Qt.TopRightCorner)

        # Build each tab in isolation: a single tab that fails on this platform
        # becomes an error placeholder instead of taking the whole window down.
        def _tab(factory, name):
            try:
                return factory()
            except Exception:
                _log(f"Tab '{name}' failed to build:\n{traceback.format_exc()}")
                w = QtWidgets.QWidget()
                lay = QtWidgets.QVBoxLayout(w)
                lbl = QtWidgets.QLabel(
                    f"{name} failed to load.\n\nSee the error log:\n{_LOG_FILE}")
                lbl.setWordWrap(True)
                lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                lay.addWidget(lbl); lay.addStretch(1)
                return w

        self._view_tab   = _tab(DataViewerTab,   "Data Viewer")
        self._mask_tab   = _tab(MaskTab,         "Mask Builder")
        self._cal_tab    = _tab(CalibrationTab,  "Calibrate")
        self._batch_tab  = _tab(BatchTab,        "Batch Integrate")
        self._queue_tab  = _tab(BatchQueueTab,   "Batch Queue")
        self._zarr_tab   = _tab(ZarrViewerTab,   "Zarr Viewer")
        self._refine_tab = _tab(RefinementTab,   "Calib. Refinement")
        self._corr_tab   = _tab(CorrectionsTab,  "Corrections")
        self._pdf_tab    = _tab(PDFTab,          "PDF Analysis")
        self._tex_tab    = _tab(TextureTab,      "Texture")
        self._pump_tab   = _tab(PumpProbeTab,    "Pump Probe")
        self._export_tab = _tab(ExportTab,       "Results & Export")

        # All tabs are always CONSTRUCTED (cross-tab wiring below relies on the
        # attributes existing); modular visibility only controls which are ADDED to
        # the QTabWidget. `always` tabs are pinned; the rest are toggled from
        # Preferences (see apply_tab_visibility / constants.DEFAULT_VISIBLE_TABS).
        self._tabs = tabs
        self._tab_specs = [
            (self._view_tab,   "Data Viewer",       True),
            (self._mask_tab,   "Mask Builder",      True),
            (self._cal_tab,    "Calibrate",         True),
            (self._refine_tab, "Calib. Refinement", False),
            (self._batch_tab,  "Batch Integrate",   True),
            (self._queue_tab,  "Batch Queue",       False),
            (self._zarr_tab,   "Zarr Viewer",       False),
            (self._corr_tab,   "Corrections",       False),
            (self._pdf_tab,    "PDF Analysis",      False),
            (self._tex_tab,    "Texture",           False),
            (self._pump_tab,   "Pump Probe",        False),
            (self._export_tab, "Results & Export",  False),
        ]
        # An ✕ on every optional tab. Closing one is the same act as unchecking
        # it in Preferences ▸ Tabs: the widget survives (cross-tab wiring holds
        # references to all twelve), it just leaves the bar, and the new set is
        # written to the active profile so it sticks across restarts. Closable
        # is a property of the bar rather than of one tab, so the four pinned
        # tabs have their button stripped in apply_tab_visibility.
        tabs.setTabsClosable(True)
        tabs.tabCloseRequested.connect(self._close_optional_tab)
        self.apply_tab_visibility()
        self.apply_hydra_visibility()

        # Cross-tab data sharing: every tab's DataLoaderPanel (plus Mask Builder's
        # bespoke loader) registers itself so any other tab's "Import from…" menu
        # can pull its currently-loaded file/folder/buffer. Defensive `getattr`
        # guards mirror `_connect()` below, in case a tab failed to build.
        from midas_gui.data_bridge import DataSourceRegistry
        self._data_registry = registry = DataSourceRegistry()
        for tab, label in (
            (self._view_tab, "Data Viewer"), (self._cal_tab, "Calibrate"),
            (self._refine_tab, "Calib. Refinement"), (self._batch_tab, "Batch Integrate"),
            (self._pump_tab, "Pump Probe"),
        ):
            loader = getattr(tab, "_loader", None)
            if loader is not None:
                try:
                    loader.bind_registry(registry, label)
                except Exception:
                    _log(f"Registry bind failed for {label}:\n{traceback.format_exc()}")
        bind_mask = getattr(self._mask_tab, "bind_registry", None)
        if bind_mask is not None:
            try:
                bind_mask(registry)
            except Exception:
                _log(f"Registry bind failed for Mask Builder:\n{traceback.format_exc()}")

        # Same registry, Hydra mode: Data Viewer / Calibrate / Batch Integrate
        # each also own a HydraLoaderPanel (a separate page, not the same
        # object as their single-detector `_loader`) — label them distinctly
        # so Import-from lists don't conflate a Hydra anchor path with its
        # single-detector counterpart.
        for tab, label in (
            (self._view_tab, "Data Viewer (Hydra)"), (self._cal_tab, "Calibrate (Hydra)"),
            (self._batch_tab, "Batch Integrate (Hydra)"),
        ):
            bind_hydra = getattr(tab, "bind_hydra_registry", None)
            if bind_hydra is not None:
                try:
                    bind_hydra(registry, label)
                except Exception:
                    _log(f"Hydra registry bind failed for {label}:\n{traceback.format_exc()}")

        # Header Exp ID field: any tab that wants it for output-path
        # suggestions reads it live via this callback rather than keeping its
        # own copy — Batch Integrate names its output folder from it, Calibrate
        # names the files it saves.
        for tab in (self._batch_tab, self._cal_tab):
            set_provider = getattr(tab, "set_expid_provider", None)
            if set_provider is not None:
                try:
                    set_provider(self.expid)
                except Exception:
                    _log(f"Exp ID provider wiring failed:\n{traceback.format_exc()}")

        # Wire cross-tab signals defensively (skip any placeholder tab).
        def _connect(src, signal_name, targets, slot_name):
            sig = getattr(src, signal_name, None)
            if sig is None:
                return
            for t in targets:
                slot = getattr(t, slot_name, None)
                if slot is not None:
                    try:
                        sig.connect(slot)
                    except Exception:
                        _log(f"Signal wiring {signal_name}->{slot_name} failed:\n"
                             f"{traceback.format_exc()}")

        # Mask propagation
        _connect(self._mask_tab, "maskReady",
                 (self._view_tab, self._cal_tab, self._batch_tab, self._refine_tab,
                  self._corr_tab, self._pdf_tab, self._tex_tab, self._pump_tab,
                  self._export_tab),
                 "set_mask_from_tab1")
        # Display orientation: the Mask Builder paints the detector the way
        # the Data Viewer does, so the two never disagree about which way up
        # it is. Display only — the mask stays raw-frame (see
        # tab_mask.set_display_transform).
        _connect(self._view_tab, "imTransChanged", (self._mask_tab,),
                 "set_display_transform")
        # Calibration propagation (Tab 2 result → consumers)
        _connect(self._cal_tab, "calibrationDone",
                 (self._batch_tab, self._queue_tab, self._mask_tab, self._refine_tab,
                  self._corr_tab, self._pdf_tab, self._tex_tab, self._pump_tab,
                  self._export_tab),
                 "set_calibration")
        # Refined geometry (Tab 4) re-broadcasts to the calibration consumers
        _connect(self._refine_tab, "refinedResult",
                 (self._batch_tab, self._queue_tab, self._mask_tab, self._corr_tab,
                  self._pdf_tab, self._tex_tab, self._pump_tab, self._export_tab),
                 "set_calibration")

        # Geometry hand-off between Data Viewer (Tab 0) and Calibrate (Tab 2):
        #   Data Viewer "→ Send geometry to Calibrate" pushes its values;
        #   Calibrate "← Data Viewer" pulls them. Both land in Calibrate.apply_geometry.
        try:
            self._view_tab.pushGeometry.connect(self._cal_tab.apply_geometry)
            self._cal_tab.pullGeometry.connect(
                lambda: self._cal_tab.apply_geometry(self._view_tab.get_geometry()))
            # Calibrate → Data Viewer: push calibrated geometry into the Viewer
            # fields — either Calibrate's own "→ Send to Data Viewer", or the
            # Viewer's "← Get" pulling the same geometry on demand.
            self._cal_tab.sendGeometryToViewer.connect(self._view_tab.set_geometry)
            self._view_tab.pullGeometry.connect(self._pull_geometry_from_calibrate)
            # Same hand-off, Hydra mode: per-panel geometry, keyed by panel number.
            self._cal_tab.pullHydraFromViewer.connect(
                lambda: self._cal_tab.import_hydra_from_viewer(self._view_tab.get_hydra_export()))
            self._cal_tab.sendHydraGeometryToViewer.connect(self._view_tab.set_hydra_panel_geometry)
            # Calibrate (Hydra mode) → Batch Integrate (Hydra mode): a panel's
            # fit auto-populates that panel's calibration source as soon as it
            # finishes, mirroring the single-detector calibrationDone wiring.
            self._cal_tab.hydraPanelCalibrationDone.connect(self._batch_tab.set_hydra_panel_calibration)
        except Exception:
            _log(f"Geometry hand-off wiring failed:\n{traceback.format_exc()}")

        # Batch Queue's "Copy from Batch Integrate" reads that tab's current
        # binning/format choices live, rather than snapshotting them at startup.
        try:
            provider = getattr(self._batch_tab, "integration_settings", None)
            setter = getattr(self._queue_tab, "set_settings_provider", None)
            if provider is not None and setter is not None:
                setter(provider)
        except Exception:
            _log(f"Batch Queue settings-provider wiring failed:\n{traceback.format_exc()}")

        # FAIR provenance: hand the (initially closed) project context to the
        # tabs that log attempts to it. Defensive, like the wiring above —
        # a placeholder tab (failed to build) simply has no such method.
        for tab in (self._cal_tab, self._batch_tab, self._queue_tab, self._mask_tab,
                    self._export_tab):
            setter = getattr(tab, "set_project_context", None)
            if setter is not None:
                try:
                    setter(self._project_ctx)
                except Exception:
                    _log(f"Project-context wiring failed:\n{traceback.format_exc()}")

        self._build_file_menu()
        self._build_menu()
        self._build_tools_menu()
        self.statusBar().showMessage(
            "Tip: mask → calibrate → (refine) → batch integrate")
        self._project_lbl = QtWidgets.QLabel("Project: none")
        self._project_lbl.setToolTip(
            "The currently-open FAIR provenance project file (File → New/Open Project…).\n"
            "Calibrate and Batch Integrate runs are logged to it automatically while open.")
        self.statusBar().addPermanentWidget(self._project_lbl)

    def _pull_geometry_from_calibrate(self):
        """Data Viewer's "← Get" — copy the Calibrate tab's latest calibrated
        geometry into the Viewer's fields. Same payload as Calibrate's own
        "→ Send to Data Viewer"; says so plainly when there is no result to
        pull, rather than silently doing nothing."""
        g = self._cal_tab.geometry_for_viewer()
        if not g:
            QtWidgets.QMessageBox.information(
                self, "No calibration yet",
                "The Calibrate tab has no calibration result to send.\n\n"
                "Run a calibration there first (or load a calibration file "
                "directly with the Data Viewer's Load/save calibration card).")
            return
        self._view_tab.set_geometry(g)

    # ── modular tab visibility ─────────────────────────────────────
    _NUMERALS = "⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭"

    def apply_tab_visibility(self, visible=None):
        """Rebuild the tab bar from ``self._tab_specs``, showing every pinned tab
        plus the optional tabs named in ``visible`` (default:
        ``constants.DEFAULT_VISIBLE_TABS``). Numerals are renumbered so they stay
        contiguous. All tab widgets already exist, so this applies live (no
        restart) and preserves each tab's in-memory state."""
        if visible is None:
            visible = getattr(C, "DEFAULT_VISIBLE_TABS", None) or []
        visible = set(visible)
        # Remembered in tab_specs order so _close_optional_tab can subtract one
        # name from the set that is actually on screen, rather than re-reading a
        # config file that a live apply (Preferences' "apply now") may predate.
        self._visible_optional = [name for _w, name, always in self._tab_specs
                                  if not always and name in visible]
        current = self._tabs.currentWidget()
        self._tabs.clear()
        i = 0
        for widget, name, always in self._tab_specs:
            if not (always or name in visible):
                continue
            prefix = self._NUMERALS[i] if i < len(self._NUMERALS) else str(i)
            idx = self._tabs.addTab(widget, f"{prefix}  {name}")
            if always:
                self._strip_close_button(idx)
            else:
                self._attach_close_button(idx, name)
            i += 1
        # keep the previously-selected tab focused if it is still shown
        idx = self._tabs.indexOf(current) if current is not None else -1
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def _strip_close_button(self, index: int) -> None:
        """Take the ✕ off a pinned tab, so the four ALWAYS_TABS can't be closed.

        Which side the style puts the button on is platform-dependent (left on
        macOS, right elsewhere), so clear both rather than guessing at one."""
        bar = self._tabs.tabBar()
        for side in (QtWidgets.QTabBar.RightSide, QtWidgets.QTabBar.LeftSide):
            if bar.tabButton(index, side) is not None:
                bar.setTabButton(index, side, None)

    # Flat, borderless, and coloured from the palette so it reads the same on
    # whatever style the desktop supplies. Kept muted until hover so a row of
    # them doesn't compete with the tab labels for attention.
    _CLOSE_BTN_QSS = (
        "QToolButton { border: none; background: transparent; padding: 0px;"
        " color: palette(mid); font-size: 12px; font-weight: bold; }"
        "QToolButton:hover { color: palette(highlight); }"
        "QToolButton:pressed { color: palette(bright-text); }"
    )

    def _attach_close_button(self, index: int, name: str) -> None:
        """Put our own ✕ on an optional tab, replacing the style's close icon.

        With ``setTabsClosable(True)`` alone, Qt draws whatever pixmap the
        active style supplies for ``SP_TabCloseButton``. On this desktop that
        comes out as a heavy angular mark that doesn't read as "close" at tab
        size. A text ✕ in a flat QToolButton looks the same on every style
        and at every DPI, for the cost of one small widget per optional tab.

        The click closes by NAME, not by the index passed in here:
        ``apply_tab_visibility`` rebuilds the whole bar, so an index baked into
        the connection goes stale as soon as any other tab is shown or hidden.
        It is also deferred a tick — handling it inline would have
        ``apply_tab_visibility`` delete this very button while its own
        ``clicked`` signal is still being emitted."""
        bar = self._tabs.tabBar()
        side = bar.style().styleHint(
            QtWidgets.QStyle.SH_TabBar_CloseButtonPosition, None, bar)
        try:
            side = QtWidgets.QTabBar.ButtonPosition(side)
        except Exception:
            side = QtWidgets.QTabBar.RightSide
        btn = QtWidgets.QToolButton(bar)
        btn.setText("\u2715")
        btn.setToolTip(f"Close {name}  \u2014  turn it back on in Preferences \u25b8 Tabs")
        btn.setAutoRaise(True)
        btn.setFocusPolicy(QtCore.Qt.NoFocus)
        btn.setCursor(QtCore.Qt.ArrowCursor)
        btn.setFixedSize(14, 14)
        btn.setStyleSheet(self._CLOSE_BTN_QSS)
        btn.clicked.connect(
            lambda _checked=False, n=name: QtCore.QTimer.singleShot(
                0, lambda: self._close_optional_tab_by_name(n)))
        self._strip_close_button(index)
        bar.setTabButton(index, side, btn)

    def _close_optional_tab_by_name(self, name: str) -> None:
        """Hide the optional tab called ``name``. See ``_close_optional_tab``."""
        always = next((a for _w, n, a in self._tab_specs if n == name), True)
        if always:
            return
        remaining = [n for n in getattr(self, "_visible_optional", []) if n != name]
        self.apply_tab_visibility(remaining)
        self._persist_visible_tabs(remaining)

    def _close_optional_tab(self, index: int) -> None:
        """Hide the optional tab whose ✕ was clicked, and remember the choice.

        Nothing is destroyed — this is exactly ``apply_tab_visibility`` with one
        name removed, so the tab's in-memory state (a loaded file, a running
        queue) is still there when it is turned back on from Preferences ▸ Tabs.

        Pinned tabs have no ✕ to click, but the guard stays: a stripped button
        is a per-index cosmetic, and anything that reaches ``tabCloseRequested``
        another way (a re-styled bar, a future shortcut) must not be able to
        close one of the four."""
        widget = self._tabs.widget(index)
        match = next(((n, a) for w, n, a in self._tab_specs if w is widget), None)
        if match is None:
            return
        name, always = match
        if always:
            return
        self._close_optional_tab_by_name(name)

    def _persist_visible_tabs(self, visible) -> None:
        """Write the optional-tab set to the active profile's ``ui.visible_tabs``
        — the same key Preferences ▸ Tabs writes, so the two stay one setting
        rather than two that can disagree.

        Read-modify-write, because ``save_user_config`` replaces the whole
        overlay file and the rest of it belongs to other settings. Best-effort:
        a read-only config directory must not turn closing a tab into an error
        dialog, so a failure is logged and the close still holds for this
        session."""
        try:
            cfg = dict(settings.load_config() or {})
            ui = dict(cfg.get("ui") or {})
            ui["visible_tabs"] = list(visible)
            cfg["ui"] = ui
            settings.save_user_config(cfg)
            C.reload_from_config()
        except Exception as e:
            _log(f"tab close: could not persist ui.visible_tabs ({e})")

    # ── header Exp ID field ─────────────────────────────────────────
    def expid(self) -> str:
        """Current Exp ID (mpe_wf_saxs_waxs style, e.g. 'park_may26'), or ''
        if unset. Shared read path for any tab's output-path logic — see
        set_expid_provider() wiring in _build_ui."""
        return self._expid_ed.text().strip()

    def set_expid(self, value: str) -> None:
        self._expid_ed.setText(value or "")

    # ── header profile selector ─────────────────────────────────────
    def _refresh_profile_combo(self):
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItems(settings.list_profiles())
        idx = self._profile_combo.findText(settings.active_profile())
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.blockSignals(False)

    def _on_header_profile_changed(self, name: str):
        if name == settings.active_profile():
            return
        try:
            settings.set_active_profile(name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Switch failed", str(e))
            self._refresh_profile_combo()
            return
        C.reload_from_config()
        self.on_profile_changed()

    def on_profile_changed(self):
        """Common tail for any profile switch, whichever UI triggered it
        (header combo or Settings ▸ Preferences ▸ Profile): sync the header
        combo's selection and re-apply profile-dependent live UI state."""
        self._refresh_profile_combo()
        self.apply_tab_visibility()
        self.apply_hydra_visibility()
        self._refresh_profile_scoped_widgets()

    def _refresh_profile_scoped_widgets(self):
        """Repopulate option-list widgets whose choices come from the active
        profile's config: the Data Viewer's Live PV device dropdown and the
        Calibrant dropdown (single-detector + every Hydra panel). The
        pixel-size-preset / K-edge-foil popup menus and the Materials dialog
        rebuild themselves lazily on open and need no wiring here; seeded
        default *values* (wavelength, pixel size, Lsd, beam-centre, default
        paths, ...) are deliberately left alone — see .context/DECISIONS.md."""
        for tab, method in ((self._view_tab, "refresh_devices"),
                            (self._cal_tab, "refresh_calibrants")):
            fn = getattr(tab, method, None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    _log(f"Profile refresh ({method}) failed:\n{traceback.format_exc()}")

    def apply_hydra_visibility(self):
        """Hydra (4-panel GE detector) mode is only offered at the 1-ID-E
        beamline profile — no other bundled profile has that detector."""
        enabled = settings.active_profile() == "1-ID-E"
        for tab in (self._view_tab, self._cal_tab, self._batch_tab):
            setter = getattr(tab, "set_hydra_available", None)
            if setter is not None:
                try:
                    setter(enabled)
                except Exception:
                    _log(f"Hydra-visibility update failed:\n{traceback.format_exc()}")

    # ── File menu: one Project (.h5) holds both the session snapshot and
    # the append-only FAIR-provenance calibration/integration history ──
    def _build_file_menu(self):
        m = self.menuBar().addMenu("&File")
        act_save = m.addAction("Save Project")
        act_save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._save_project_dialog)
        act_save_as = m.addAction("Save Project As…")
        act_save_as.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._save_project_as_dialog)
        act_open = m.addAction("Open Project…")
        act_open.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._open_project_dialog)
        self._open_last_act = m.addAction("Open Last Project")
        self._open_last_act.setToolTip(
            "Open the most recently-opened project — same confirmation "
            "dialog as picking the top entry in Recent Projects below, "
            "just one click instead of two.")
        self._open_last_act.triggered.connect(self._open_last_project)
        self._recent_projects_menu = m.addMenu("Recent Projects")

        m.addSeparator()
        self._project_history_act = m.addAction("Project History…")
        self._project_history_act.setEnabled(False)
        self._project_history_act.triggered.connect(self._show_project_history)
        self._close_proj_act = m.addAction("Close Project")
        self._close_proj_act.setEnabled(False)
        self._close_proj_act.triggered.connect(self._close_project)

        m.addSeparator()
        act_import_legacy = m.addAction("Import Legacy Workspace (.json)…")
        act_import_legacy.triggered.connect(self._import_legacy_workspace_dialog)

        m.addSeparator()
        act_quit = m.addAction("Quit")
        act_quit.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)

        m.aboutToShow.connect(self._refresh_recent_menus)

    def _refresh_recent_menus(self) -> None:
        self._populate_recent_menu(
            self._recent_projects_menu, "project", self._open_project_path)

    @staticmethod
    def _format_recent_when(when_utc) -> str:
        try:
            return datetime.fromisoformat(when_utc).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""

    def _populate_recent_menu(self, menu, kind: str, open_fn) -> None:
        menu.clear()
        entries = settings.get_recent(kind)
        if not entries:
            act = menu.addAction("(none yet)")
            act.setEnabled(False)
            return
        for entry in entries:
            when = self._format_recent_when(entry.get("last_opened_utc"))
            label = f"{entry['name']}   —   {when}" if when else entry["name"]
            act = menu.addAction(label)
            act.setToolTip(entry["path"])
            act.triggered.connect(lambda checked=False, p=entry["path"]: open_fn(p))

    def _set_project_path(self, path) -> None:
        self._project_ctx.path = path
        self._project_lbl.setText(f"Project: {Path(path).name}" if path else "Project: none")
        self._header_project_lbl.setText(f"● Project: {Path(path).name}" if path else "")
        self._close_proj_act.setEnabled(path is not None)
        self._project_history_act.setEnabled(path is not None)

    def _confirm_ok_to_switch_project(self) -> bool:
        """Shared by ``closeEvent``/``_close_project``/``_open_project_dialog``/
        ``_open_project_path``: if there are unsaved session changes, prompt
        Save/Discard/Cancel exactly like closing the app does. Returns True
        if it's fine to proceed (nothing unsaved, or the user explicitly
        saved or discarded), False if the caller should abort (Cancel, or a
        Save/Save-As prompt raised from here was itself cancelled)."""
        if not self._workspace_dirty:
            return True
        resp = QtWidgets.QMessageBox.question(
            self, "Unsaved changes",
            "This project has unsaved session changes. Save before continuing?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save)
        if resp == QtWidgets.QMessageBox.Cancel:
            return False
        if resp == QtWidgets.QMessageBox.Save:
            self._save_project_dialog()
            if self._workspace_dirty:   # user cancelled the Save/Save-As prompt
                return False
        else:   # Discard
            self._clear_autosave_draft()
        return True

    def _open_project_dialog(self):
        """File ▸ Open Project…: browse to a project ``.h5`` and, the moment
        one is clicked, preview exactly what it contains (which
        ``gui_workspace`` tabs, which mask/calibrate/integrate attempts) as a
        checkbox tree — everything checked by default; uncheck what
        shouldn't be restored, then Open."""
        if not self._confirm_ok_to_switch_project():
            return
        from midas_gui.dialogs import ProjectOpenDialog
        dlg = ProjectOpenDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._open_project_selection(dlg.selected_path(), dlg.selection())

    def _open_last_project(self) -> None:
        """File ▸ Open Last Project: shortcut for the top entry of Recent
        Projects — goes through the exact same ``_open_project_path`` flow
        (confirmation dialog included), just without opening the submenu
        first."""
        entries = settings.get_recent("project")
        if not entries:
            QtWidgets.QMessageBox.information(
                self, "No recent project", "No recently-opened project found.")
            return
        self._open_project_path(entries[0]["path"])

    def _open_project_path(self, path) -> None:
        """Recent Projects menu entries: the path is already known, so skip
        the file-browser pane and go straight to the same checkbox-tree
        preview (``ProjectSelectionDialog``, wrapping the same
        ``ProjectContentsPicker`` widget ``ProjectOpenDialog`` uses) for this
        one file."""
        if not self._confirm_ok_to_switch_project():
            return
        from midas_gui.dialogs import ProjectSelectionDialog
        dlg = ProjectSelectionDialog(path, self)
        if not dlg.has_content():
            QtWidgets.QMessageBox.critical(
                self, "Open Project failed",
                dlg.error_message() or f"{path} has nothing this version can restore.")
            return
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._open_project_selection(path, dlg.selection())

    def _open_project_selection(self, path, sel: dict) -> None:
        """Shared by File ▸ Open Project… and the Recent Projects menu: given
        a validated project path and a checkbox-tree selection (see
        ``dialogs.ProjectContentsPicker.selection``), opens the project for
        future Calibrate/Batch-Integrate/Mask-Builder logging and restores
        only what was checked."""
        try:
            project.open_project(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Project failed", str(e))
            return
        self._set_project_path(path)
        settings.record_recent(path, "project")

        tab_names = sel.get("workspace_tabs") or []
        if tab_names:
            try:
                meta = project.read_workspace_meta(path)
                data = {"__midas_gui_state__": True, "tabs": {},
                        "active_tab": meta.get("active_tab"),
                        "active_profile": meta.get("active_profile"),
                        "expid": meta.get("expid")}
                sidecars = {}
                for name in tab_names:
                    tstate, tsidecars = project.read_workspace_tab(path, name)
                    data["tabs"][name] = tstate
                    if tsidecars:
                        sidecars[name] = tsidecars
                self._apply_workspace_state(data, sidecars)
            except Exception:
                _log(f"Restoring gui_workspace failed:\n{traceback.format_exc()}")
        else:
            # Nothing checked under GUI Workspace (or none saved) — fall
            # back to the creation-time profile so at least Profile-scoped
            # defaults line up.
            try:
                self._restore_active_profile(project.project_active_profile(path))
            except Exception:
                _log(f"Could not read project's active profile:\n{traceback.format_exc()}")
            try:
                expid = project.read_workspace_meta(path).get("expid")
                if expid:
                    self.set_expid(expid)
            except Exception:
                _log(f"Could not read project's Exp ID:\n{traceback.format_exc()}")

        restored = []
        try:
            calib_attempts = {}
            for k, ref in (sel.get("calib_refs") or {}).items():
                meta = project.read_attempt(path, ref)
                meta["_results_arrays"] = project.read_calib_attempt_results(path, ref)
                # A multi-panel calibration's refined shifts are embedded in
                # the project (see append_calibration_attempt) rather than
                # relying on whatever file path was live when the attempt was
                # saved — that path is often an ephemeral tempfile (no Output
                # folder set during Fit) and may no longer exist, especially
                # after moving the project to another machine. Re-materialize
                # it next to the project file and point the restored result
                # at that instead, so integration keeps seeing real panel
                # shifts rather than silently falling back to zero.
                panel_arr = project.read_calib_attempt_panel_shifts(path, ref)
                if panel_arr is not None and meta.get("result") is not None:
                    ps_path = project.materialize_panel_shifts(path, ref, panel_arr)
                    if ps_path:
                        meta["result"]["panel_shifts_path"] = ps_path
                calib_attempts[k] = meta
            integrate_attempts = {}
            for k, ref in (sel.get("integrate_refs") or {}).items():
                meta = project.read_attempt(path, ref)
                meta["_results_arrays"] = project.read_attempt_results(path, ref)
                integrate_attempts[k] = meta
            if calib_attempts:
                # Fields only when the workspace didn't already restore them —
                # see apply_project_calibration's restore_fields.
                self._cal_tab.apply_project_calibration(
                    calib_attempts, restore_fields="Calibrate" not in tab_names)
                restored.append("Calibrate: " + ", ".join(sorted(calib_attempts)))
                # apply_project_calibration only redraws the Calibrate tab
                # itself — it doesn't emit calibrationDone (that signal is
                # reserved for a just-finished Fit), so consumers wired only
                # to that signal (Batch Queue chiefly) never learn a result
                # now exists, and report "No calibration available from the
                # Calibrate tab yet" even though the tab plainly shows one.
                # Propagate the restored single-detector result by hand, to
                # the same consumer list the live signal wiring uses.
                restored_result = self._cal_tab.get_result()
                if restored_result is not None:
                    for t in (self._batch_tab, self._queue_tab, self._mask_tab,
                              self._refine_tab, self._corr_tab, self._pdf_tab,
                              self._tex_tab, self._pump_tab, self._export_tab):
                        setter = getattr(t, "set_calibration", None)
                        if setter is not None:
                            try:
                                setter(restored_result)
                            except Exception:
                                _log("Calibration propagation to "
                                     f"{t} failed:\n{traceback.format_exc()}")
            if integrate_attempts:
                self._batch_tab.apply_project_integration(integrate_attempts)
                restored.append("Batch Integrate: " + ", ".join(sorted(integrate_attempts)))
            mask_ref = sel.get("mask_ref")
            if mask_ref:
                mask_meta = project.read_attempt(path, mask_ref)
                mask_arr = project.read_mask_attempt_array(path, mask_ref)
                self._mask_tab.apply_project_mask(mask_meta, mask_arr)
                restored.append(f"Mask: {mask_ref.rsplit('/', 1)[-1]}")
        except Exception:
            _log(f"Populate from project failed:\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(
                self, "Populate from project failed",
                f"Could not populate the GUI from this project's records.\n"
                f"See the error log:\n{_LOG_FILE}")
            return

        if restored:
            QtWidgets.QMessageBox.information(
                self, "Open Project",
                "Restored from this project's recorded analysis:\n\n" + "\n".join(restored))

    def _restore_active_profile(self, name) -> None:
        """Switch to ``name`` (a Workspace/Project's recorded active Profile)
        via the exact same path the header combo/Preferences dialog use, so
        every side effect (tab visibility, calibrant/device dropdowns, ...)
        applies identically. Silently does nothing if ``name`` is falsy,
        unknown locally, or already active."""
        if not name or name not in settings.list_profiles() or name == settings.active_profile():
            return
        try:
            settings.set_active_profile(name)
        except Exception:
            _log(f"Could not restore active profile ({name!r}):\n{traceback.format_exc()}")
            return
        C.reload_from_config()
        self.on_profile_changed()

    def _close_project(self):
        """Unlike closing the whole app, this detaches from the project file
        (stops future Calibrate/Batch-Integrate logging and the Ctrl+S
        target) without touching any tab's live field values — but since
        Ctrl+S now writes the session into this same file, an unsaved
        session would otherwise be silently lost on Close, so offer to save
        first exactly like ``closeEvent`` does."""
        if not self._confirm_ok_to_switch_project():
            return
        self._set_project_path(None)

    def _show_project_history(self) -> None:
        path = self._project_ctx.path
        if not path:
            return
        from midas_gui.dialogs import ProjectHistoryDialog
        ProjectHistoryDialog(path, self).exec_()


    def _save_project_dialog(self):
        """Ctrl+S: overwrite the currently-open project's ``/workspace``
        slot, if any; otherwise fall back to a Save-As prompt (first save)."""
        if self._project_ctx.path:
            self.save_project(self._project_ctx.path)
            return
        self._save_project_as_dialog()

    def _save_project_as_dialog(self):
        """Save Project As… always targets a *fresh* project file: an
        existing target is completely overwritten (after explicit
        confirmation, and a ``.bak`` copy — see
        ``project.backup_before_overwrite``) rather than adopting its own
        analysis history. If a project is currently open and has any
        ``/analysis`` provenance, the user separately picks how much of it
        (none / latest-only / all) to carry into the new file — see
        ``project.copy_analysis_history``."""
        src_path = self._project_ctx.path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project As", src_path or "midas_project.h5",
            "MIDAS Project (*.h5)")
        if not path:
            return
        if not path.lower().endswith(".h5"):
            path += ".h5"
        if Path(path).exists():
            if QtWidgets.QMessageBox.question(
                    self, "Save Project As",
                    f"{Path(path).name} already exists and will be completely "
                    "overwritten, replacing any project data it currently "
                    "holds. Continue?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return
            try:
                project.create_project(path, overwrite=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Project As failed", str(e))
                return
        else:
            try:
                project.create_project(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Project As failed", str(e))
                return

        scope = "none"
        src_summary = {}
        if src_path and str(src_path) != str(path):
            src_summary = project.analysis_summary(src_path)
            if src_summary:
                from midas_gui.dialogs import SaveAsHistoryDialog
                dlg = SaveAsHistoryDialog(src_summary, self)
                if dlg.exec_() != QtWidgets.QDialog.Accepted:
                    return
                scope = dlg.selected_scope()

        self.save_project(path)

        if scope != "none" and src_path and str(src_path) != str(path):
            try:
                copied = project.copy_analysis_history(src_path, path, scope)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Copying analysis history failed",
                    f"The workspace was saved, but copying analysis history "
                    f"from the previous project failed:\n{e}")
                return
            if copied:
                lines = []
                if copied.get("mask"):
                    lines.append(f"Mask: {copied['mask']} attempt(s)")
                for key, label in (("calibrate", "Calibrate"), ("integrate", "Batch Integrate")):
                    per_panel = copied.get(key) or {}
                    if per_panel:
                        detail = ", ".join(f"{p}: {n}" for p, n in sorted(per_panel.items()))
                        lines.append(f"{label}: {detail}")
                QtWidgets.QMessageBox.information(
                    self, "Save Project As",
                    "Copied analysis history from the previous project:\n\n"
                    + "\n".join(lines))

    def _import_legacy_workspace_dialog(self):
        """Reads a standalone Workspace JSON file from before this feature
        merged Workspace and Project into one ``.h5`` file, and applies it
        into whichever project is (or isn't) currently open."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Legacy Workspace", "", "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import failed", str(e))
            return
        if not data.get("__midas_gui_state__"):
            QtWidgets.QMessageBox.critical(
                self, "Import failed", "This file is not a MIDAS GUI workspace file.")
            return
        if QtWidgets.QMessageBox.question(
                self, "Import Legacy Workspace",
                "Importing this workspace will overwrite the current values "
                "in every tab. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        self._apply_workspace_state(data, sidecars={})

    @staticmethod
    def _accepts_sidecar_stem(fn) -> bool:
        try:
            return "sidecar_stem" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    def _serialize_workspace(self, sidecar_dir=None, *, quiet: bool = False):
        """Build the ``{"__midas_gui_state__": ..., "tabs": {...}}`` dict that
        :meth:`save_project` writes into the project's ``/gui_workspace``
        header, one child group per tab (see ``project.write_gui_workspace``).
        Shared with the periodic dirty-check and autosave: passing
        ``sidecar_dir=None`` (their case) skips every tab's sidecar file I/O
        (mask/calibration export — see each tab's own ``get_state``), so it's
        cheap and side-effect-free to call on every timer tick. When given, each
        tab that accepts ``sidecar_stem`` gets its own subdirectory under
        ``sidecar_dir`` (named after the tab) so its exported sidecar files
        stay unambiguously scoped to that tab — ``save_project`` reads them
        back out per-subdirectory. Returns ``(state, errors)``."""
        current = self._tabs.currentWidget()
        state = {"__midas_gui_state__": True, "version": 1, "tabs": {},
                 "active_profile": settings.active_profile(),
                 "expid": self.expid()}
        for widget, name, _always in self._tab_specs:
            if widget is current:
                state["active_tab"] = name
        errors = []
        for widget, name, _always in self._tab_specs:
            get_state = getattr(widget, "get_state", None)
            if get_state is None:
                continue
            try:
                if sidecar_dir is not None and self._accepts_sidecar_stem(get_state):
                    tab_dir = Path(sidecar_dir) / name
                    tab_dir.mkdir(parents=True, exist_ok=True)
                    state["tabs"][name] = get_state(sidecar_stem=str(tab_dir / "sidecar"))
                else:
                    state["tabs"][name] = get_state()
            except Exception:
                if not quiet:
                    _log(f"Serialize workspace failed for tab '{name}':\n"
                         f"{traceback.format_exc()}")
                errors.append(name)
        return state, errors

    @staticmethod
    def _hash_workspace_state(state: dict) -> str:
        blob = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _set_workspace_dirty(self, dirty: bool) -> None:
        self._workspace_dirty = dirty
        self.setWindowModified(dirty)

    def _check_workspace_dirty(self) -> None:
        """QTimer tick: cheap hash-diff against the last saved/loaded state
        (see _serialize_workspace) drives the window title's Qt-native
        ``[*]`` unsaved-changes marker — no per-widget change signals wired."""
        if self._last_saved_hash is None:
            return
        state, _errors = self._serialize_workspace(quiet=True)
        self._set_workspace_dirty(self._hash_workspace_state(state) != self._last_saved_hash)

    def _update_window_title(self) -> None:
        name = Path(self._project_ctx.path).stem if self._project_ctx.path else "Untitled"
        self.setWindowTitle(f"MIDAS GUI v{__version__} — {name}[*]")

    def _autosave_tick(self) -> None:
        """QTimer tick (every few minutes): if the session is dirty, write a
        crash-recovery draft JSON — reusing the exact serialization
        save_project uses, just pointed at a fixed internal path instead of
        prompting. Never touches the open Project's `.h5` file itself."""
        if not self._workspace_dirty:
            return
        state, _errors = self._serialize_workspace(quiet=True)
        draft = settings.autosave_draft_path()
        try:
            draft.parent.mkdir(parents=True, exist_ok=True)
            draft.write_text(json.dumps(state, indent=2))
        except Exception:
            _log(f"Session autosave failed:\n{traceback.format_exc()}")

    def _clear_autosave_draft(self) -> None:
        try:
            settings.autosave_draft_path().unlink(missing_ok=True)
        except Exception:
            pass

    def maybe_offer_restore_autosave(self) -> None:
        """Called once from main(), after the window is shown — deliberately
        NOT from __init__/_build_ui, so plain ``MainWindow()`` construction
        (as every test does) never risks blocking on a modal dialog because
        of a leftover draft from an earlier crashed/killed session."""
        draft = settings.autosave_draft_path()
        if not draft.is_file():
            return
        try:
            when = datetime.fromtimestamp(draft.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            when = "an earlier session"
        if QtWidgets.QMessageBox.question(
                self, "Restore unsaved session?",
                f"MIDAS GUI found an autosaved session from {when} that was "
                "never explicitly saved (likely from a crash or a forced "
                "quit).\n\nRestore it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes) == QtWidgets.QMessageBox.Yes:
            try:
                data = json.loads(draft.read_text())
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Restore failed", str(e))
                data = None
            if data is not None:
                self._apply_workspace_state(data, sidecars={})
                self._set_project_path(None)      # recovered draft has no "real" home yet
                self._update_window_title()
                self._set_workspace_dirty(True)   # force Ctrl+S to prompt for a destination
        self._clear_autosave_draft()

    def save_project(self, path):
        """Overwrite this project's ``/gui_workspace`` header (see
        ``project.write_gui_workspace``) with every tab's current field
        state, one modular group per tab. Tabs that hold derived data not
        yet exported to a file of its own (MaskTab, CalibrationTab) still
        write small sidecars via their existing ``get_state(sidecar_stem=...)``
        contract; those are harvested from a scratch temp dir (one
        subdirectory per tab — see ``_serialize_workspace``) and embedded
        into that tab's own group instead of being left as loose files on
        disk."""
        with tempfile.TemporaryDirectory(prefix="midas_gui_sidecar_") as td:
            state, errors = self._serialize_workspace(sidecar_dir=td)
            sidecars = {}
            for sub in Path(td).iterdir():
                if sub.is_dir():
                    files = {f.name: f.read_bytes() for f in sub.iterdir()}
                    if files:
                        sidecars[sub.name] = files
        meta = {"active_tab": state.get("active_tab"), "active_profile": state.get("active_profile"),
                "expid": state.get("expid")}
        try:
            project.write_gui_workspace(path, tabs=state["tabs"], sidecars=sidecars, meta=meta)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
            return
        self._set_project_path(path)
        self._last_saved_hash = self._hash_workspace_state(state)
        self._set_workspace_dirty(False)
        self._update_window_title()
        settings.record_recent(path, "project")
        self._clear_autosave_draft()
        msg = f"Saved project to:\n{path}"
        if errors:
            msg += "\n\nThe following tabs could not be saved:\n" + "\n".join(errors)
        QtWidgets.QMessageBox.information(self, "Save Project", msg)

    def _apply_workspace_state(self, data: dict, sidecars: Optional[dict] = None) -> None:
        """Restore every tab's fields from a previously saved session
        snapshot — one or more of a project's ``/gui_workspace`` tab groups
        (see ``_open_project_dialog``), a legacy standalone Workspace JSON
        (``_import_legacy_workspace_dialog``), or an autosave crash-recovery
        draft (``maybe_offer_restore_autosave``). ``sidecars`` is nested,
        ``{tab_name: {filename: data}}`` (matching ``_serialize_workspace``'s
        per-tab subdirectories) — empty/omitted for callers with no sidecars
        of their own. Path-backed fields (images, masks, dark/bright/
        background, HDF5 datasets) are re-loaded automatically; long-running
        pipelines (Fit, Batch Integrate, PDF transform, Refinement) are not
        re-run — their inputs are restored so a single click of the tab's
        own Run/Fit button reproduces the result."""
        if not data.get("__midas_gui_state__"):
            QtWidgets.QMessageBox.critical(
                self, "Load failed", "This is not a valid MIDAS GUI session snapshot.")
            return
        try:
            self._restore_active_profile(data.get("active_profile"))
        except Exception:
            _log(f"Could not restore session's active profile:\n{traceback.format_exc()}")
        if data.get("expid"):
            self.set_expid(data["expid"])
        name_to_widget = {name: widget for widget, name, _always in self._tab_specs}
        errors, skipped = [], []
        with tempfile.TemporaryDirectory(prefix="midas_gui_sidecar_") as td:
            for name, tstate in data.get("tabs", {}).items():
                widget = name_to_widget.get(name)
                set_state = getattr(widget, "set_state", None) if widget is not None else None
                if set_state is None:
                    skipped.append(name)
                    continue
                try:
                    if self._accepts_sidecar_stem(set_state):
                        tab_dir = Path(td) / name
                        tab_dir.mkdir(parents=True, exist_ok=True)
                        for fname, blob in (sidecars or {}).get(name, {}).items():
                            if isinstance(blob, (bytes, bytearray)):
                                (tab_dir / fname).write_bytes(bytes(blob))
                            else:
                                (tab_dir / fname).write_text(blob)
                        set_state(tstate, sidecar_stem=str(tab_dir / "sidecar"))
                    else:
                        set_state(tstate)
                except Exception:
                    _log(f"Session restore failed for tab '{name}':\n{traceback.format_exc()}")
                    errors.append(name)
        # Tabs restore in dict order, so the Mask Builder may have been
        # rebuilt before the Data Viewer knew its transforms. Push them once
        # now rather than depending on that order.
        try:
            self._mask_tab.set_display_transform(self._view_tab._im_trans_codes())
        except Exception:
            _log(f"Display-transform sync after restore failed:\n"
                 f"{traceback.format_exc()}")
        active = data.get("active_tab")
        widget = name_to_widget.get(active) if active else None
        idx = self._tabs.indexOf(widget) if widget is not None else -1
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)
        state, _errors = self._serialize_workspace(quiet=True)
        self._last_saved_hash = self._hash_workspace_state(state)
        self._set_workspace_dirty(False)
        self._update_window_title()
        msg = ("Restored session.\n\n"
               "Path-backed fields (images, masks, dark/bright/background) were "
               "reloaded. Fit / Batch Integrate / PDF transform / Refinement "
               "results are not recomputed — re-run each tab's own action to "
               "reproduce them.")
        if skipped:
            msg += "\n\nTabs in the file no longer present:\n" + "\n".join(skipped)
        if errors:
            msg += "\n\nThe following tabs failed to restore:\n" + "\n".join(errors)
        QtWidgets.QMessageBox.information(self, "Open Project", msg)

    def _build_menu(self):
        """Settings menu: preferences, open config folder, reload."""
        m = self.menuBar().addMenu("&Settings")
        act_pref = m.addAction("Preferences…")
        act_pref.triggered.connect(self._open_preferences)
        act_scale = m.addAction("Interface scaling…")
        act_scale.triggered.connect(self._open_scaling)
        m.addSeparator()
        act_open = m.addAction("Open config folder")
        act_open.triggered.connect(self._open_config_folder)
        act_reload = m.addAction("Reload config")
        act_reload.triggered.connect(self._reload_config)

    def _build_tools_menu(self):
        """Tools menu: standalone utilities that operate on the current
        session's data but run as their own process."""
        m = self.menuBar().addMenu("&Tools")
        act = m.addAction("Auto Attenuation…")
        act.triggered.connect(self._open_auto_attenuation)

    def _write_auto_attenuation_snapshot(self, path):
        """Gather the Data Viewer's current buffer/loaded-data/dark/mask/
        geometry and write it to *path* as an Auto Attenuation snapshot.

        Shared by the initial launch (``_open_auto_attenuation``) and a
        running popup's "Refresh from Data Viewer" request (see
        ``midas_gui/auto_attenuation/refresh_server.py``) — the popup itself
        never touches these live objects, only this method does. Returns
        ``(ok, message)``; ``message`` is set only when ``ok`` is False."""
        loader = getattr(self._view_tab, "_loader", None)
        if loader is None or loader.n_frames() == 0:
            return False, "Capture or load a buffer/stack in the Data Viewer first."
        frames = loader.full_stack()
        source = loader.data_source_kind()
        dark = loader.dark()
        dark_stack = loader.dark_raw_stack()
        mask = loader.composite_mask()
        geometry = self._view_tab.get_geometry() or {}
        wl = float(geometry.get("wavelength_A") or 0.0)
        energy_keV = (C.HC_KEV_A / wl) if wl > 0 else None

        from midas_gui.auto_attenuation.snapshot import write_snapshot
        write_snapshot(
            path, frames=frames, dark=dark, dark_stack=dark_stack,
            mask=mask, energy_keV=energy_keV, geometry=geometry, source=source,
        )
        return True, None

    def _on_auto_attenuation_refresh_request(self, path):
        """Callback for AutoAttenuationRefreshServer — see that module."""
        try:
            return self._write_auto_attenuation_snapshot(path)
        except Exception:
            _log(f"Auto Attenuation refresh failed:\n{traceback.format_exc()}")
            return False, "Refresh failed — see the main GUI's log."

    def _open_auto_attenuation(self):
        """Snapshot the Data Viewer's buffer/dark/mask/geometry to a temp
        file and launch the Auto Attenuation popup as a separate, detached
        process (see midas_gui.auto_attenuation) — it only ever reads that
        one snapshot, never the live GUI objects, so it can't be crashed by
        or crash the main GUI, and survives the main GUI closing."""
        import os
        fd, path = tempfile.mkstemp(suffix=".npz", prefix="midas_auto_att_")
        os.close(fd)
        try:
            ok, message = self._write_auto_attenuation_snapshot(path)
            if not ok:
                os.unlink(path)
                QtWidgets.QMessageBox.information(self, "Auto Attenuation", message)
                return

            # QProcess.startDetached(sys.executable, args) is not enough on
            # its own: sys.executable is the right interpreter, but if this
            # process can only import midas_gui because its launcher put the
            # repo root on sys.path (e.g. an un-pip-installed source
            # checkout run via cwd), a *fresh* child interpreter started with
            # `-m` has no such head start and fails with "No module named
            # 'midas_gui'" — seen on a beamline workstation. Prepending
            # midas_gui's own parent directory to PYTHONPATH makes the child
            # able to import it however this process could; on a proper pip
            # install it's already on sys.path, so this is a harmless no-op.
            import midas_gui as _mg
            pkg_root = str(Path(_mg.__file__).resolve().parent.parent)
            env = QtCore.QProcessEnvironment.systemEnvironment()
            old_pp = env.value("PYTHONPATH")
            env.insert("PYTHONPATH", pkg_root + (os.pathsep + old_pp if old_pp else ""))

            proc = QtCore.QProcess()
            proc.setProgram(sys.executable)
            proc.setArguments(["-m", "midas_gui.auto_attenuation.app", "--snapshot", path])
            proc.setProcessEnvironment(env)
            if not proc.startDetached():
                QtWidgets.QMessageBox.critical(
                    self, "Auto Attenuation",
                    "Failed to launch the Auto Attenuation process.")
        except Exception:
            _log(f"Auto Attenuation launch failed:\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(
                self, "Auto Attenuation",
                "Failed to launch Auto Attenuation — see log.")

    def _open_preferences(self):
        try:
            from midas_gui.prefs_dialog import PreferencesDialog
            PreferencesDialog(self).exec_()
        except Exception:
            _log(f"Preferences dialog failed:\n{traceback.format_exc()}")

    def _open_scaling(self):
        """Quick interface-scale picker (whole-app zoom for HiDPI / 4K monitors).
        Persists ui.ui_scale and offers to relaunch so the new scale takes effect."""
        cur = float(getattr(C, "DEFAULT_UI_SCALE", 1.0) or 1.0)
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Interface scaling",
            "Whole-interface scale (layout + fonts).\n"
            "1.0 ≈ 1080p,  ~1.5 ≈ 1440p,  ~2.0 ≈ 4K.\nApplies after a restart.",
            cur, 0.5, 4.0, 2)
        if not ok or abs(val - cur) < 1e-6:
            return
        try:
            cfg = settings.load_config()
            cfg.setdefault("ui", {})["ui_scale"] = float(val)
            settings.save_user_config(cfg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e)); return
        self._offer_restart(f"Interface scale set to {val:g}×.")

    def _offer_restart(self, msg):
        """Ask to relaunch now so a startup-only setting (e.g. UI scale) applies."""
        if QtWidgets.QMessageBox.question(
                self, "Restart to apply",
                f"{msg}\n\nRestart the GUI now to apply?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes) == QtWidgets.QMessageBox.Yes:
            self.restart_app()

    def restart_app(self):
        """Relaunch a fresh instance (which re-reads the scale) and close this one.
        Uses QProcess so the new process is detached from this one."""
        try:
            self._stop_all_workers()
        except Exception:
            pass
        import os
        prog = sys.executable
        if os.path.basename(sys.argv[0] or "") == "__main__.py":
            args = ["-m", "midas_gui", *sys.argv[1:]]
        else:
            args = list(sys.argv)
        try:
            QtCore.QProcess.startDetached(prog, args)
        except Exception:
            _log(f"Restart failed:\n{traceback.format_exc()}")
        QtWidgets.QApplication.quit()

    def _open_config_folder(self):
        folder = settings.user_config_path().parent
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _reload_config(self):
        settings.reload()
        QtWidgets.QMessageBox.information(
            self, "Config reloaded",
            "Config re-read from disk. Restart the GUI for changes to take full effect.")

    # ── clean shutdown ─────────────────────────────────────────────
    def _stop_all_workers(self):
        """Interruption-request + bounded-wait every background QThread on every
        tab (and any parked orphans), so nothing is left running at teardown — a
        live QThread at interpreter exit is a common hard-crash cause."""
        tab_attrs = ("_view_tab", "_mask_tab", "_cal_tab", "_refine_tab",
                     "_batch_tab", "_corr_tab", "_pdf_tab", "_tex_tab",
                     "_pump_tab", "_export_tab")
        threads = []
        for a in tab_attrs:
            tab = getattr(self, a, None)
            if tab is None:
                continue
            for val in list(vars(tab).values()):
                if isinstance(val, QtCore.QThread):
                    threads.append(val)
                elif isinstance(val, list):
                    threads.extend(x for x in val if isinstance(x, QtCore.QThread))
        for th in threads:                       # ask everyone to stop first
            try:
                if th.isRunning():
                    th.requestInterruption()
            except Exception:
                pass
        for th in threads:                       # then bounded wait
            try:
                if th.isRunning():
                    th.wait(2000)
            except Exception:
                pass
        for a in tab_attrs:                      # then any tab-level explicit shutdown
            tab = getattr(self, a, None)
            if tab is None:
                continue
            fn = getattr(tab, "shutdown", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def closeEvent(self, event):
        if getattr(self, "_workspace_dirty", False) and not self._confirm_ok_to_switch_project():
            event.ignore()
            return
        try:
            self._dirty_timer.stop()
            self._autosave_timer.stop()
        except Exception:
            pass
        try:
            self._stop_all_workers()
        except Exception:
            _log(f"Worker shutdown on close failed:\n{traceback.format_exc()}")
        try:
            self._bridge_server.stop()
        except Exception:
            _log(f"Bridge server shutdown failed:\n{traceback.format_exc()}")
        try:
            self._auto_att_refresh_server.stop()
        except Exception:
            _log(f"Auto Attenuation refresh server shutdown failed:\n{traceback.format_exc()}")
        super().closeEvent(event)


def _apply_ui_scale():
    """Set Qt's whole-application scale factor from the configured ui.ui_scale BEFORE
    the QApplication is created, so layout + fonts scale uniformly on HiDPI / 4K
    screens. Must run before any QApplication instance exists."""
    from midas_gui import helpers
    # In-app setting is authoritative (overwrites any inherited value on restart).
    scale = helpers.apply_ui_scale()
    _log(f"UI scale (QT_SCALE_FACTOR) = {scale:.4g}")


def main():
    _install_diagnostics()
    _apply_ui_scale()   # must precede QApplication construction
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("MIDAS GUI")
    S.apply_theme(app, _CHECKMARK_SVG, _ARROW_UP_SVG, _ARROW_DOWN_SVG)

    win = MainWindow()
    win.show()
    # Only the real app entry point offers to restore a crash-recovery draft —
    # never bare `MainWindow()` construction (every test does exactly that),
    # so a leftover draft on disk can't block a headless test run on a modal
    # dialog. See MainWindow.maybe_offer_restore_autosave.
    win.maybe_offer_restore_autosave()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
