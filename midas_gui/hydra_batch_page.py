"""Hydra (4-panel GE detector) page for the Batch Integrate tab.

``HydraBatchPage`` mirrors ``hydra_calib_page.HydraCalibrationPage``'s
composition: a shared ``HydraLoaderPanel`` (in streaming mode) for data, one
small ``HydraBatchPanelCard`` per GE panel (its own calibration source +
values + progress — each panel is integrated with its own independently
fitted geometry), and shared "recipe" cards (Integration, Corrections,
Monitor normalisation, Output) applied identically to every panel's run,
since it's the same integration settings for all 4.

Runs currently-present panels Sequentially (one ``BatchRunCoordinator`` at a
time) or in Parallel (all started at once) — panel-level concurrency.
Independently, each panel's own ``BatchRunCoordinator`` can further split
that panel's frames across N concurrent chunk workers ("Batch Parallel"),
sharing one detector map — see ``workers.BatchRunCoordinator``.
``BatchWorker``/``BatchRunCoordinator`` never touch process-global state
(they log via the ``log_line`` signal only), so no ``capture_stdout``-style
flag is needed for safe concurrent runs — the same is now true of
``CalibrationWorker``, which runs each panel's calibration in its own OS
process (see ``workers.py``/``calib_cli.py``) rather than redirecting
``sys.stdout``.

Deliberately has no Drift-correction or live-MONITOR (folder-watch) support
for Hydra mode in this first pass — both exist on the single-detector Batch
tab and can be added later if needed; see ``.context/DECISIONS.md``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui.constants import (KERNELS, ERROR_MODELS,
                                 DEFAULT_KERNEL, DEFAULT_ERROR_MODEL)
from midas_gui.helpers import (
    _fspin, _browse, _NoScrollComboBox, _NoScrollSpinBox,
    widgets_to_dict, apply_dict_to_widgets,
    _load_image, rmax_corner_px, rmax_edge_px, draw_polar_bin_overlay,
    browse_start_dir, warn_if_path_missing, suggest_panel_base_output_dir, check_output_dir_writable)
from midas_gui.widgets import (LogPanel, CorrectionFlagsWidget, WaterfallViewer,
                               StackedProfileViewer, OutputFormatSelector, ImageViewer,
                               OriginToolButton)
from midas_gui.hydra_widgets import HydraLoaderPanel, HydraDetectorToolbar
from midas_gui.hydra_batch_widgets import HydraBatchPanelCard
from midas_gui.workers import BatchRunCoordinator, write_all_profiles
from midas_gui import hydra
from midas_gui import project
from midas_gui import settings
from midas_gui import style as S


class _PanelViewerPair(QtWidgets.QWidget):
    """One panel's own Waterfall/Stacked-profiles tab pair — a small copy of
    the single-detector Batch tab's RIGHT-pane viewer, one instance per GE
    panel so each panel's results display independently.

    The Detector-view preview is deliberately NOT part of this pair — see
    ``HydraBatchPage``'s own docstring comment (in ``_build_ui``, RIGHT
    section) for why it's one page-level ``ImageViewer`` refreshed in place
    rather than a per-panel widget reparented on every switch.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lv = QtWidgets.QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        self.waterfall = WaterfallViewer()
        self.stack_view = StackedProfileViewer()
        self.tabs.addTab(self.waterfall, "Waterfall")
        self.tabs.addTab(self.stack_view, "Stacked profiles")
        lv.addWidget(self.tabs)
        self.wf_started = False


class HydraBatchPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict = {}           # panel_num -> HydraBatchPanelCard
        self._viewer_pairs: dict = {}    # panel_num -> _PanelViewerPair
        self._expid_provider = None      # () -> str, forwarded by BatchTab
        #: Composite view state, mirroring hydra_page's. `_states` is the
        #: persistent per-panel DetectorState cache build_windmill_composite
        #: mutates in place; `_composite_img` doubles as the dirty flag --
        #: None means "rebuild on next display", which is what keeps a
        #: 44-megapixel remap off every siblings/frame/calibration signal.
        self._states: dict = {}          # panel number -> hydra.DetectorState
        self._composite_img = None
        self._big_det_size = None
        #: Bumped on every successful build. The display guard keys on this
        #: rather than on the array's shape or id(): a rebuild from changed
        #: geometry very often lands on the SAME canvas size, and a freed
        #: array's replacement can reuse its address -- either way the new
        #: image would be judged "not fresh" and never reach the viewer.
        self._composite_build_id = 0
        self._composite_note = None      # last reason, for log de-duplication
        #: Output path this page last filled in itself; see
        #: _maybe_autofill_output_dir. Set before the UI is built, since
        #: loading data fires siblingsChanged, which auto-fills.
        self._autofilled_out = None
        self._workers: dict = {}         # panel_num -> BatchRunCoordinator (running)
        self._orphans: list = []         # aborted workers kept alive until they wind down
        self._pending_panels: list = []  # sequential-mode queue
        self._run_cancelled = False
        self._integrated_fids: dict = {}  # panel_num -> set of frame ids
        self._geom_cache: dict = {}      # panel_num -> cached integration context
        self._geom_sig: dict = {}        # panel_num -> signature the cache was built for
        self._last_run_inputs: dict = {}  # panel_num -> JSON-safe run inputs (provenance)
        self._last_run_fields: dict = {}  # panel_num -> {mask, mask_is_file_backed}
        self._last_results: dict = {}     # panel_num -> {r_axis_px, profiles, sigmas, frame_ids}
        self._last_axis_ctx: dict = {}    # panel_num -> (lsd, px, wl) — for the Save button
        self._project_ctx = None
        # Single shared, page-level Detector-view widget + its overlay
        # items — its content (not the widget itself) refreshes per the
        # toolbar-selected panel (see _PanelViewerPair's docstring for why
        # it's never reparented between panels).
        self._det_view = ImageViewer()
        self._origin_btn = OriginToolButton(self._det_view)
        self._det_view._toolbar_layout.addWidget(self._origin_btn)
        self._bin_overlay_items: list = []
        # (panel, shape, im_trans) the shared Detector view is framed for —
        # see _refresh_active_detector_preview.
        self._det_view_framed_for = None
        self._build_ui()
        self._on_panel_changed(self._toolbar.current())

    def set_project_context(self, ctx):
        self._project_ctx = ctx

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(0)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False); split.setHandleWidth(6)
        root.addWidget(split)

        # ── LEFT: the Hydra data loader (streaming: frame range/stride +
        #    per-panel dark/bright/background/mask) ──
        self._loader = HydraLoaderPanel(mode="stream")
        self._loader.setMinimumWidth(200)
        self._loader.siblingsChanged.connect(self._on_siblings_changed)
        self._loader.frameChanged.connect(
            lambda *_: (self.invalidate_composite(),
                        self._refresh_active_detector_preview()))
        split.addWidget(self._loader)

        # ── MIDDLE: shared "recipe" cards + per-panel card stack ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setMinimumWidth(260)
        inner = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(inner); lv.setContentsMargins(2, 2, 2, 2); lv.setSpacing(8)
        scroll.setWidget(inner)

        # Integration (shared)
        integ = S.make_card("Integration  (shared across ge1–ge4)")
        self._kernel = _NoScrollComboBox()
        for label, key in KERNELS.items():
            self._kernel.addItem(label, key)
        _ki = self._kernel.findData(DEFAULT_KERNEL)
        if _ki >= 0:
            self._kernel.setCurrentIndex(_ki)
        self._r_bin = _fspin(0.1, 20.0, 2, 1.0, "px")
        self._e_bin = _fspin(0.5, 360.0, 1, 5.0, "°")
        self._azim = _NoScrollComboBox()
        self._azim.addItem("Pixel-weighted", True)
        self._azim.addItem("η-bin mean (legacy)", False)
        intf = S.Form()
        intf.row(("Kernel:", self._kernel))
        intf.row(("R bin:", self._r_bin), ("η bin:", self._e_bin))
        intf.row(("Azim. mean:", self._azim))
        integ.body.addLayout(intf)
        # Rmin/Rmax — shared across panels like R bin/η bin; Corner/Edge
        # presets compute from whichever panel is currently selected in the
        # toolbar (see BatchTab's single-detector counterpart for the 0.0
        # "auto" sentinel convention).
        self._r_min = _fspin(0.0, 1_000_000.0, 2, 0.0, "px")
        self._r_max = _fspin(0.0, 1_000_000.0, 2, 0.0, "px")
        self._r_max.setToolTip(
            "0 = auto (farthest detector corner from the beam centre).\n"
            "Use the Corner/Edge buttons to fill in a value, or type your own.")
        self._rmax_corner_btn = QtWidgets.QPushButton("Corner")
        self._rmax_corner_btn.setToolTip(
            "Set Rmax to the farthest detector CORNER from the beam centre "
            "of the currently-selected panel.")
        self._rmax_corner_btn.clicked.connect(lambda: self._apply_rmax_preset(rmax_corner_px))
        self._rmax_edge_btn = QtWidgets.QPushButton("Edge")
        self._rmax_edge_btn.setToolTip(
            "Set Rmax to the farthest detector EDGE from the beam centre "
            "of the currently-selected panel.")
        self._rmax_edge_btn.clicked.connect(lambda: self._apply_rmax_preset(rmax_edge_px))
        rmax_row = QtWidgets.QHBoxLayout(); rmax_row.setSpacing(4)
        rmax_row.addWidget(self._r_max)
        rmax_row.addWidget(self._rmax_corner_btn); rmax_row.addWidget(self._rmax_edge_btn)
        rf = S.Form()
        rf.row(("Rmin:", self._r_min))
        rf.row(("Rmax:", rmax_row))
        integ.body.addLayout(rf)
        self._grid_chk = QtWidgets.QCheckBox("Show bin grid")
        self._grid_chk.setToolTip(
            "Overlay the Rmin/Rmax boundary circles and the full (R, η) "
            "integration bin grid on each panel's Detector view tab — "
            "concentric circles at each R-bin edge, spokes at each η-bin "
            "edge. Thinned to at most ~50 rings / ~72 spokes for "
            "legibility with fine bin sizes. Unchecking this hides the "
            "overlay entirely, including Rmin/Rmax.")
        integ.body.addWidget(self._grid_chk)
        for w in (self._r_min, self._r_max, self._r_bin, self._e_bin):
            w.valueChanged.connect(self._refresh_active_detector_preview)
        self._grid_chk.toggled.connect(self._refresh_active_detector_preview)
        self._var_check = QtWidgets.QCheckBox("Per-bin variance (σ)")
        self._var_check.setToolTip(
            "Compute per-bin σ via the chosen error model.\n"
            "Mutually exclusive with corrections (corrections win; σ→√I).")
        self._err_model = _NoScrollComboBox(); self._err_model.addItems(ERROR_MODELS)
        self._err_model.setEnabled(False)
        if DEFAULT_ERROR_MODEL in ERROR_MODELS:
            self._err_model.setCurrentText(DEFAULT_ERROR_MODEL)
        self._var_check.toggled.connect(self._err_model.setEnabled)
        vrow = QtWidgets.QHBoxLayout(); vrow.setSpacing(6)
        vrow.addWidget(self._var_check); vrow.addWidget(self._err_model, 1)
        integ.body.addLayout(vrow)
        self._q_check = QtWidgets.QCheckBox("Q-uniform bins")
        self._q_check.setToolTip("Bin uniformly in Q (Å⁻¹) instead of R (px).")
        integ.body.addWidget(self._q_check)
        self._q_min = _fspin(0.0, 100.0, 3, 0.5, "Å⁻¹")
        self._q_max = _fspin(0.0, 100.0, 3, 8.0, "Å⁻¹")
        self._q_bin = _fspin(0.0001, 1.0, 4, 0.01, "Å⁻¹")
        for w in (self._q_min, self._q_max, self._q_bin):
            w.setEnabled(False)
        self._q_check.toggled.connect(lambda c: [w.setEnabled(c) for w in
                                                 (self._q_min, self._q_max, self._q_bin)])
        qf = S.Form(); qf.row(("Qmin:", self._q_min), ("Qmax:", self._q_max)); qf.row(("ΔQ:", self._q_bin))
        integ.body.addLayout(qf)
        lv.addWidget(integ)

        # Corrections (shared)
        self._corr_widget = CorrectionFlagsWidget()
        lv.addWidget(self._corr_widget)

        # Monitor normalisation (shared — a property of the beam/scan, not the panel)
        mon = S.make_card("Monitor normalisation (optional, shared)")
        self._mon_ed = QtWidgets.QLineEdit()
        self._mon_ed.setPlaceholderText("monitor.txt  (one value per line)")
        monr = QtWidgets.QHBoxLayout(); monr.setSpacing(4); monr.addWidget(self._mon_ed, 1)
        bmon = QtWidgets.QPushButton("…"); bmon.setFixedWidth(30)
        bmon.clicked.connect(lambda: self._mon_ed.setText(
            _browse(self, "Open monitor file", "Text (*.txt *.dat *.csv);;All (*)") or ""))
        monr.addWidget(bmon)
        mon.body.addLayout(monr)
        lv.addWidget(mon)

        # Output (shared base dir — each panel writes to its own ge{n}/ subfolder)
        out = S.make_card("Output  (shared — each panel writes to its own ge{n}/ subfolder)")
        self._out_ed = QtWidgets.QLineEdit(); self._out_ed.setPlaceholderText("Output directory…")
        warn_if_path_missing(self._out_ed, self, is_output_dir=True)
        orow = QtWidgets.QHBoxLayout(); orow.setSpacing(4); orow.addWidget(self._out_ed, 1)
        bou = QtWidgets.QPushButton("…"); bou.setFixedWidth(30)
        bou.clicked.connect(lambda: self._out_ed.setText(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Output directory", browse_start_dir(self._out_ed.text())) or "")); orow.addWidget(bou)
        self._suggest_out_btn = QtWidgets.QPushButton("Suggest")
        self._suggest_out_btn.setToolTip(
            "Fill in <outroot>/<expid>_bc/<froot>/, the SHARED base each "
            "panel writes its own ge{n}/ subfolder into — so it is Batch "
            "Integrate's convention without the <detector> segment, which "
            "the per-panel subfolder supplies.\n\n"
            "Read positionally off whichever panel's source is loaded; every "
            "panel of a set shares a froot and an outroot and differs only "
            "in the detector, so any one of them gives the same answer. The "
            "detector tag is stripped off the froot too, so a shared base is "
            "not named after panel 1.")
        self._suggest_out_btn.clicked.connect(self._apply_suggested_output_dir)
        orow.addWidget(self._suggest_out_btn)
        out.body.addLayout(S.Form().row(("Folder:", orow)))
        self._fmt = OutputFormatSelector()
        out.body.addWidget(self._fmt)
        lv.addWidget(out)

        # Run controls — two independent levels of parallelism:
        #  (1) panel-level Sequential/Parallel (below, unchanged): how many of
        #      ge1..ge4 integrate concurrently.
        #  (2) frame-level Sequential/Batch-Parallel (new): whether *each*
        #      panel's own frames are further split across chunk workers —
        #      same BatchRunCoordinator single-detector Batch Integrate uses.
        run_card = S.make_card("Run  (Hydra: ge1–ge4, one recipe)")
        mode_row = QtWidgets.QHBoxLayout(); mode_row.setSpacing(6)
        mode_row.addWidget(S.LabelRight("Panels:"))
        self._run_mode_combo = _NoScrollComboBox()
        self._run_mode_combo.addItem("Sequential", "sequential")
        self._run_mode_combo.addItem("Parallel", "parallel")
        self._run_mode_combo.setToolTip(
            "Sequential: one panel integrated at a time.\n"
            "Parallel: all present panels integrated at once — faster, but "
            "shares CPU/GPU across concurrent runs.")
        mode_row.addWidget(self._run_mode_combo); mode_row.addStretch(1)
        run_card.body.addLayout(mode_row)
        frame_mode_row = QtWidgets.QHBoxLayout(); frame_mode_row.setSpacing(6)
        _frame_mode_tip = (
            "Sequential: frames integrated one at a time.\n"
            "Batch Parallel: splits each running panel's own frames across N "
            "workers sharing one detector map. The worker count auto-shrinks "
            f"so each worker gets ≥{BatchRunCoordinator.MIN_FRAMES_PER_WORKER} frames.")
        _frame_mode_lbl = S.LabelRight("Per panel:")
        _frame_mode_lbl.setToolTip(_frame_mode_tip)
        frame_mode_row.addWidget(_frame_mode_lbl)
        self._frame_run_mode = _NoScrollComboBox()
        self._frame_run_mode.addItem("Sequential", "sequential")
        self._frame_run_mode.addItem("Batch Parallel", "batch_parallel")
        self._frame_run_mode.setToolTip(_frame_mode_tip)
        frame_mode_row.addWidget(self._frame_run_mode)
        frame_mode_row.addWidget(S.LabelRight("Workers:"))
        _max_workers = os.cpu_count() or 8
        self._n_workers = _NoScrollSpinBox()
        self._n_workers.setRange(1, _max_workers)
        self._n_workers.setValue(min(4, _max_workers))
        self._n_workers.setEnabled(False)
        frame_mode_row.addWidget(self._n_workers)
        self._frame_run_mode.currentIndexChanged.connect(
            lambda *_: self._n_workers.setEnabled(
                self._frame_run_mode.currentData() == "batch_parallel"))
        run_card.body.addLayout(frame_mode_row)
        self._run_btn = S.primary_btn("Start Integration")
        self._run_btn.clicked.connect(self._run_all)
        self._abort_btn = QtWidgets.QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.setToolTip("Stop each running panel after its current frame, "
                                   "keeping frames already integrated.")
        self._abort_btn.clicked.connect(self._abort_all)
        self._abort_btn.setStyleSheet(S.DANGER_BTN_QSS)
        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip(
            "Write every panel's lineouts already computed this run to disk "
            "(each into its own ge{n}/ subfolder), in the checked format(s) "
            "above — works even if no Output folder was set before running.")
        self._save_btn.clicked.connect(self._save_results)
        self._save_btn.setStyleSheet(S.SUCCESS_BTN_QSS)
        run_row = QtWidgets.QHBoxLayout(); run_row.setSpacing(6)
        run_row.addWidget(self._run_btn, 1)
        run_row.addWidget(self._abort_btn, 1)
        run_row.addWidget(self._save_btn, 1)
        run_card.body.addLayout(run_row)
        self._prog = QtWidgets.QProgressBar(); self._prog.setRange(0, 0); self._prog.setVisible(False)
        run_card.body.addWidget(self._prog)
        lv.addWidget(run_card)

        # Per-panel toggle + card stack (calibration source/values/progress)
        # Composite ON: the toolbar has always supported it (the Data Viewer's
        # Hydra page uses it); this page opted out. Everything that reads the
        # toolbar key now goes through _panel_num, which returns None here.
        self._toolbar = HydraDetectorToolbar(include_composite=True)
        self._toolbar.panelChanged.connect(self._on_panel_changed)
        lv.addWidget(self._toolbar)
        self._card_stack = QtWidgets.QStackedWidget()
        for n in (1, 2, 3, 4):
            card = HydraBatchPanelCard(n)
            self._cards[n] = card
            self._card_stack.addWidget(card)
            # A panel's calibration feeds BOTH its own overlay and the
            # composite's placement of that panel, so invalidate either way:
            # the composite may not be on screen now, but it must not be
            # redrawn later from the geometry this just replaced.
            card._use_calib_btn.toggled.connect(
                lambda *_, n=n: self._on_panel_calibration_changed(n))
            card._json_ed.textChanged.connect(
                lambda *_, n=n: self._on_panel_calibration_changed(n))
        lv.addWidget(self._card_stack)
        lv.addStretch(1)
        split.addWidget(scroll)

        # ── RIGHT: per-panel Waterfall/Stacked-profiles (switched with the
        #    active-panel toggle) + a page-level Detector-view tab, + a
        #    shared Log. ONE shared ImageViewer (not one per panel) — a 5th
        #    pyqtgraph ImageView here pushes this page's already-heavy
        #    widget count (4x Waterfall + 4x StackedProfileViewer) further
        #    into the pyqtgraph-teardown segfault risk documented in
        #    .context/STATE.md (worse under an explicit gc.collect(), fine
        #    in normal use — see tests/test_hydra_batch_ui.py's pytestmark).
        #    Its content (not the widget itself) refreshes per the
        #    toolbar-selected panel, mirroring ``HydraCalibrationPage``'s
        #    shared ``_img_view``.
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._viewer_stack = QtWidgets.QStackedWidget()
        for n in (1, 2, 3, 4):
            pair = _PanelViewerPair()
            self._viewer_pairs[n] = pair
            self._viewer_stack.addWidget(pair)
        self._top_tabs = QtWidgets.QTabWidget()
        self._top_tabs.addTab(self._viewer_stack, "Per-panel results")
        self._top_tabs.addTab(self._det_view, "Detector view")
        # Build the composite on the way IN to the Detector view, and not at
        # all while Per-panel results is showing: a ~1 s GUI-thread remap for
        # pixels behind another tab is the cost this gate exists to avoid.
        self._top_tabs.currentChanged.connect(
            lambda *_: self._refresh_active_detector_preview())
        right.addWidget(self._top_tabs)
        self._log = LogPanel()
        self._log.setMaximumHeight(16_777_215)
        right.addWidget(self._log)
        right.setStretchFactor(0, 4); right.setStretchFactor(1, 1)
        right.setMinimumWidth(320)
        split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 0); split.setStretchFactor(2, 1)
        split.setSizes([286, 361, 950])

    # ── Loader signal handlers ──────────────────────────────────────

    def _on_siblings_changed(self, siblings: dict):
        self._toolbar.set_available(siblings.keys())
        self.invalidate_composite()      # different panels / different files
        self._maybe_autofill_output_dir()
        self._refresh_active_detector_preview()

    # ── Output folder (same contract as BatchTab's — see
    # BatchTab._maybe_autofill_output_dir for why the test is provenance
    # rather than emptiness) ──────────────────────────────────────────
    def set_expid_provider(self, provider) -> None:
        """Header Exp ID, read live for the output suggestion. Forwarded by
        BatchTab, which is what app.py wires — this page is built lazily, so
        it cannot be in that loop itself."""
        self._expid_provider = provider

    def _suggest_output_dir(self):
        """``<outroot>/<expid>_bc/<froot>`` — the shared base, no detector
        segment: each panel appends its own ``ge{n}/`` in ``_run_panel``."""
        siblings = self._loader.siblings() or {}
        rep = next((siblings[n] for n in sorted(siblings) if siblings.get(n)), None)
        if not rep:
            return None
        expid = self._expid_provider().strip() if self._expid_provider else ""
        return suggest_panel_base_output_dir(rep, expid_fallback=expid)

    def _set_output_dir_suggested(self, suggested) -> None:
        self._out_ed.setText(str(suggested))
        self._autofilled_out = str(suggested)
        reason = check_output_dir_writable(suggested)
        if reason:
            self._log.append(f"[hydra] Warning: {reason}")

    def _apply_suggested_output_dir(self):
        """The Suggest button: overwrite whatever is there, on request."""
        suggested = self._suggest_output_dir()
        if suggested is None:
            self._log.append("[hydra] No panel data loaded yet — nothing to suggest.")
            return
        self._set_output_dir_suggested(suggested)

    def _maybe_autofill_output_dir(self):
        """Replace a path we filled in; never one the user chose."""
        current = self._out_ed.text().strip()
        if current and current != (getattr(self, "_autofilled_out", None) or ""):
            return
        suggested = self._suggest_output_dir()
        if suggested is None or str(suggested) == current:
            return
        self._set_output_dir_suggested(suggested)

    @staticmethod
    def _panel_num(key) -> "Optional[int]":
        """Panel number from a toolbar key, or None for "composite".

        Every consumer here used to do ``int(key[2])``, which reads the digit
        out of "ge3" -- and raises ValueError on "composite", whose third
        character is 'm'. One helper rather than four guards, because a
        missed site is a crash the moment the button is pressed.
        """
        if not key or not key.startswith("ge"):
            return None
        try:
            return int(key[2:])
        except ValueError:
            return None

    def _on_panel_changed(self, key: str):
        n = self._panel_num(key)
        # Corner/Edge fill the SHARED Rmax from one panel's beam centre;
        # there is no such thing on a BigDet canvas, so show them as dead
        # rather than let them look available and do nothing.
        for btn in (self._rmax_corner_btn, self._rmax_edge_btn):
            btn.setEnabled(n is not None)
        if n is None:                      # composite
            # The card stack and the Per-panel results stack have no
            # composite entry, and inventing a blank one would make both
            # read as "this panel has nothing". They hold their last panel;
            # only the shared detector image follows the selection. Holding
            # the last panel also leaves its calibration card editable while
            # the composite is up -- which is the thing that rebuilds it.
            #
            # Raise the Detector view: selecting Composite has no other
            # meaning on this page, and the build is gated on that tab being
            # visible, so without this the click would appear to do nothing.
            self._top_tabs.setCurrentWidget(self._det_view)
            self._refresh_detector_preview(None)
            return
        self._card_stack.setCurrentWidget(self._cards[n])
        self._viewer_stack.setCurrentWidget(self._viewer_pairs[n])
        self._refresh_detector_preview(n)

    # ── Detector-view preview (Rmin/Rmax + bin-grid overlay) ─────────

    def _apply_rmax_preset(self, formula) -> None:
        """Corner/Edge button handler — resolves the currently-selected
        panel's calibration and fills the shared Rmax field."""
        n = self._panel_num(self._toolbar.current())
        card = self._cards.get(n) if n else None
        if card is None:
            # Composite selected: Rmin/Rmax are per-panel integration
            # settings and there is no single beam centre to read.
            if self._toolbar.current() == "composite":
                self._log.append(
                    "[hydra batch] Rmax presets need a panel selected — "
                    "they fill that panel's calibration into the shared "
                    "Rmax, and the composite has no single beam centre.")
            return
        fields, note = card._calib_fields_in_use()
        if not fields or fields.get("BC_y") is None or fields.get("NrPixelsY") is None:
            self._log.append(f"[hydra batch] Can't set Rmax preset: {note}")
            return
        value = formula(fields["BC_y"], fields["BC_z"], fields["NrPixelsY"], fields["NrPixelsZ"])
        self._r_max.setValue(value)

    def _refresh_active_detector_preview(self, *_args) -> None:
        key = self._toolbar.current()
        if key:
            self._refresh_detector_preview(self._panel_num(key))

    def _on_panel_calibration_changed(self, n: int) -> None:
        """One panel's calibration source changed: its own overlay is stale,
        and so is any cached composite that placed it."""
        self.invalidate_composite()
        self._refresh_detector_preview(n)
        if self._toolbar.current() == "composite":
            self._refresh_detector_preview(None)

    # ── Composite (all panels, each under its own calibration) ──────

    def invalidate_composite(self) -> None:
        """Mark the cached composite stale. Cheap: the rebuild itself is
        deferred to the next display of the composite view, so a siblings /
        frame / calibration change costs nothing while a single panel is on
        screen. Same dirty-flag discipline as ``hydra_page``."""
        self._composite_img = None

    def _build_composite_if_needed(self) -> None:
        """Build the four-panel canvas, once, using each panel's ACTIVE
        calibration.

        Seeding ``self._states`` from the cards first is not optional:
        ``build_windmill_composite`` silently falls back to the bundled
        *default* 1-ID-E geometry for any panel missing from the dict, which
        would show a plausible composite built from the wrong numbers (the
        same trap ``tests/test_hydra_geometry.py`` warns about).

        Decimated to ``hydra.COMPOSITE_DISPLAY_STEP`` — this is a
        registration preview, not a measurement surface. Failures leave the
        cache None and the viewer showing whatever it had; the per-panel
        views remain the authority.
        """
        if self._composite_img is not None:
            return
        siblings = self._loader.siblings() or {}
        # Only panels with a REAL calibration take part. Handing a panel to
        # build_windmill_composite without seeding its state first makes it
        # fall back to the bundled default 1-ID-E geometry -- which would
        # place that panel plausibly and wrongly in a view whose entire
        # purpose is "the four detector calibrations applied appropriately".
        # Same validity test the per-panel overlay already uses.
        active, skipped = {}, []
        for n, path in sorted(siblings.items()):
            card = self._cards.get(n)
            fields = None
            if card is not None:
                fields, note = card._calib_fields_in_use()
            if (not fields or fields.get("BC_y") is None
                    or fields.get("NrPixelsY") is None):
                skipped.append(f"ge{n}")
                continue
            st = self._states.setdefault(n, hydra.DetectorState())
            # Re-applied on every build rather than tracked for dirtiness:
            # correctness then does not depend on the invalidation list being
            # complete, only on WHETHER we rebuild. It is a dict read, and
            # get_inv_coords still cache-hits on unchanged values.
            st.load_from_geometry_dict(fields)
            st.data_file = path
            active[n] = path
        if len(active) < 2:
            self._composite_img = None
            self._note_composite(
                f"Composite needs at least 2 calibrated panels — have "
                f"{len(active)}" + (f", missing {', '.join(skipped)}" if skipped else ""))
            return
        # Deliberately NOT applying dark/bright/background: this page's
        # per-panel preview reads the raw frame off disk, and a composite
        # that silently disagreed with the panel beside it would be worse
        # than both being raw. (hydra_page._apply_field_selectors is the
        # thing not being copied here. Change both paths together or
        # neither.)
        try:
            comp, big = hydra.build_windmill_composite(
                active, self._loader.frame_index(),
                self._loader.dataset(), self._states,
                op="max", step=hydra.COMPOSITE_DISPLAY_STEP)
        except Exception as exc:
            self._composite_img = None
            self._note_composite(f"Composite could not be built: {exc}")
            return
        self._composite_img = comp
        self._big_det_size = big
        self._composite_build_id += 1
        self._composite_note = None
        # Which panels actually made it in is otherwise invisible -- "why is
        # ge3 missing" with nothing in the log is a support call.
        self._log.append(
            f"[hydra batch] Composite: {'+'.join(f'ge{n}' for n in active)}, "
            f"{comp.shape[1]}x{comp.shape[0]} px "
            f"(canvas {big}, step {hydra.COMPOSITE_DISPLAY_STEP})"
            + (f" — skipped {', '.join(skipped)} (no calibration)" if skipped else ""))

    def _note_composite(self, msg: str) -> None:
        """Log a composite problem once per distinct message — this runs on
        every refresh, and a line per keystroke would bury the log."""
        if msg != self._composite_note:
            self._composite_note = msg
            self._log.append(f"[hydra batch] {msg}")

    def _show_composite(self) -> None:
        """Draw the composite, with no polar overlay.

        ``draw_polar_bin_overlay`` places Rmin/Rmax and the bin grid around
        ONE panel's beam centre in that panel's pixel frame; on the BigDet
        canvas those coordinates mean nothing, so it is cleared rather than
        drawn somewhere plausible and wrong.

        ``set_raw_frame`` is passed an EMPTY transform list on purpose:
        ``DetectorState.get_remapped_frame`` already applied each panel's own
        ImTransOpt codes before remapping it, so applying a card's codes here
        would transform twice.
        """
        if self._top_tabs.currentWidget() is not self._det_view:
            return                       # not on screen; do not pay for it
        self._build_composite_if_needed()
        img = self._composite_img
        if img is None:
            return
        framed_for = ("composite", tuple(img.shape), self._composite_build_id)
        fresh = framed_for != self._det_view_framed_for
        try:
            self._det_view.set_raw_frame(img, (), autorange=fresh,
                                         reset_levels=fresh)
        except Exception:
            return
        self._det_view_framed_for = framed_for
        draw_polar_bin_overlay(
            self._det_view, self._bin_overlay_items,
            bc_y=0.0, bc_z=0.0, r_min=0.0, r_max=0.0, r_bin=1.0, e_bin=5.0)

    def _refresh_detector_preview(self, n: int) -> None:
        """Refresh the (single, page-level) Detector-view tab's frame +
        Rmin/Rmax/bin-grid overlay for panel ``n`` — a no-op unless ``n`` is
        the panel currently selected in the toolbar, since drawing onto the
        shared viewer for a panel that isn't the active one would show the
        wrong geometry. Reads the frame straight off disk since the Hydra
        loader is stream-mode (see ``hydra_calib_page._panel_raw_image``
        for the identical pattern).

        ``n`` is None for the composite view, which draws all four panels
        remapped onto one BC/tilt-registered canvas instead — see
        :meth:`_build_composite_if_needed`.
        """
        key = self._toolbar.current()
        if not key or n != self._panel_num(key):
            return
        if n is None:
            self._show_composite()
            return
        card = self._cards.get(n)
        if card is None:
            return
        # The Rmax auto-fill/overlay below only need calibration fields, not
        # a loaded frame — calibration commonly arrives before data does.
        path = self._loader.siblings().get(n)
        if path is not None:
            try:
                frame = _load_image(path, self._loader.dataset(), self._loader.frame_index())
                # Only re-frame when the picture itself changes (panel switch,
                # detector size, transform) — not on every Rmin/Rmax/bin or
                # Show-bin-grid toggle, all of which route through here and
                # would otherwise autoRange() the user's pan/zoom away. Same
                # fix as the single-detector Batch Integrate tab.
                codes = tuple(card.resolved_im_trans() or ())
                framed_for = (n, tuple(frame.shape), codes)
                fresh = framed_for != self._det_view_framed_for
                self._det_view.set_raw_frame(frame, card.resolved_im_trans(),
                                              autorange=fresh, reset_levels=fresh)
                self._det_view_framed_for = framed_for
            except Exception:
                pass
        fields, _ = card._calib_fields_in_use()
        if not fields or fields.get("BC_y") is None or fields.get("NrPixelsY") is None:
            draw_polar_bin_overlay(
                self._det_view, self._bin_overlay_items,
                bc_y=0.0, bc_z=0.0, r_min=0.0, r_max=0.0, r_bin=1.0, e_bin=5.0)
            return
        if self._r_max.value() == 0.0:
            self._r_max.blockSignals(True)
            self._r_max.setValue(rmax_corner_px(
                fields["BC_y"], fields["BC_z"], fields["NrPixelsY"], fields["NrPixelsZ"]))
            self._r_max.blockSignals(False)
        draw_polar_bin_overlay(
            self._det_view, self._bin_overlay_items,
            bc_y=fields["BC_y"], bc_z=fields["BC_z"],
            r_min=self._r_min.value(), r_max=self._r_max.value(),
            r_bin=self._r_bin.value(), e_bin=self._e_bin.value(),
            show_grid=self._grid_chk.isChecked(),
            tx=fields.get("tx") or 0.0, ty=fields.get("ty") or 0.0,
            tz=fields.get("tz") or 0.0, lsd_um=fields.get("Lsd"),
            pxY_um=fields.get("pxY"), pxZ_um=fields.get("pxZ"))

    # ── Run ────────────────────────────────────────────────────────

    def _run_mode(self) -> str:
        return self._run_mode_combo.currentData() or "sequential"

    def _integration_signature(self, n, src_cfg, kernel, corrections, weighted, mask):
        """Signature identifying a reusable detector map for panel ``n``'s
        current settings — mirrors ``BatchTab._integration_signature``."""
        card = self._cards[n]
        if not card.using_file():
            calib = ("calib", id(card.result))
        else:
            calib = ("file", card.file_path())
        mask_id = None if mask is None else (tuple(mask.shape), int(np.count_nonzero(mask)))
        pol, sa = corrections
        return (calib, kernel, round(self._r_bin.value(), 4), round(self._e_bin.value(), 4),
                round(self._r_min.value(), 4), round(self._r_max.value(), 4),
                bool(weighted), pol is not None, sa is not None, mask_id,
                src_cfg.get("path"), src_cfg.get("type"), src_cfg.get("dataset"))

    def _cache_geom(self, n, sig, ctx):
        self._geom_cache[n] = ctx
        self._geom_sig[n] = sig

    def _run_all(self):
        siblings = self._loader.siblings()
        if not siblings:
            QtWidgets.QMessageBox.warning(self, "No data", "Load Hydra panel data first.")
            return
        if self._workers:
            return
        self._orphans = [o for o in self._orphans if o.isRunning()]
        self._run_cancelled = False
        self._run_btn.setEnabled(False); self._abort_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._last_results = {}
        self._prog.setVisible(True)
        mode = self._run_mode()
        self._log.append("─" * 40 + f"\nStarting Hydra batch integration ({mode})…")
        panels = sorted(siblings)
        self._pending_panels = list(panels)
        if mode == "parallel":
            pending, self._pending_panels = list(self._pending_panels), []
            for n in pending:
                self._start_panel_worker(n)
        else:
            self._start_next_sequential()

    def _start_next_sequential(self):
        if not self._pending_panels:
            self._maybe_finish_run()
            return
        n = self._pending_panels.pop(0)
        self._start_panel_worker(n)

    def _skip_panel(self, n: int, reason: str):
        self._log.append(f"[ge{n}] {reason} — skipped")
        self._cards[n].set_status(reason)
        if self._run_mode() == "sequential":
            self._start_next_sequential()
        self._maybe_finish_run()

    def _start_panel_worker(self, n: int):
        card = self._cards[n]
        try:
            spec = card.resolved_spec(self._r_bin.value(), self._e_bin.value(),
                                      r_min=self._r_min.value(), r_max=self._r_max.value() or None)
        except Exception as e:
            self._skip_panel(n, f"calibration error: {e}")
            return
        src_cfg = self._loader.source_cfg(n)
        if not src_cfg.get("path"):
            self._skip_panel(n, "no data")
            return

        kernel = self._kernel.currentData()
        corrections = self._corr_widget.build_corrections()
        variance_cfg = ({"error_model": self._err_model.currentText()}
                        if self._var_check.isChecked() else None)
        if variance_cfg and self._corr_widget.any_enabled():
            variance_cfg = None
        base_out = self._out_ed.text().strip()
        out_dir = str(Path(base_out) / f"ge{n}") if base_out else None
        fmts = self._fmt.checked_keys()
        q_cfg = ({"QMin": self._q_min.value(), "QMax": self._q_max.value(),
                  "QBinSize": self._q_bin.value()} if self._q_check.isChecked() else None)

        pair = self._viewer_pairs[n]
        lsd, px, wl = float(spec.Lsd), float(spec.pxY), float(spec.Wavelength)
        self._last_axis_ctx[n] = (lsd, px, wl)
        # "R" even in Q mode: Q-uniform is a REBIN of an R-uniform
        # integration (workers.rebin_R_to_grid) and the axis travelling with
        # the rebinned profile is the R in pixels of each Q bin, not the Q
        # grid. Saying "Q" here makes _convert_radial read pixels as A^-1,
        # which clips every bin to 2theta = 180 deg. Same bug, same fix as
        # tab_batch.py -- see 4cb7aaa and
        # tests/test_batch_cake_stack.py::test_q_uniform_profiles_carry_an_R_axis_not_a_Q_one
        _axctx = (lsd, px, wl, "R")
        pair.waterfall.set_axis_context(*_axctx)
        pair.stack_view.set_axis_context(*_axctx)
        pair.wf_started = False
        self._integrated_fids[n] = set()

        dark = self._loader.dark(n); bright = self._loader.bright(n)
        background = self._loader.background(n); bright_mode = self._loader.bright_mode()
        mask = self._loader.composite_mask(n)
        frame_range = self._loader.frame_range()
        monitor_file = self._mon_ed.text().strip() or None
        weighted = bool(self._azim.currentData())

        sig = self._integration_signature(n, src_cfg, kernel, corrections, weighted, mask)
        context = self._geom_cache.get(n) if self._geom_sig.get(n) == sig else None

        self._last_run_inputs[n] = {
            "src_cfg": src_cfg, "kernel": kernel, "fmt": fmts,
            "frame_range": frame_range, "monitor_file": monitor_file,
            "q_cfg": q_cfg, "weighted": weighted, "bright_mode": bright_mode,
            "mask_sources": self._loader.get_state().get("masks", {}).get(n),
        }
        self._last_run_fields[n] = {
            "mask": mask,
            "mask_is_file_backed": mask is not None and not self._loader.has_live_mask_source(n),
        }

        worker = BatchRunCoordinator(
            spec, src_cfg, mask, out_dir, fmts, kernel, corrections, variance_cfg,
            q_cfg=q_cfg, frame_range=frame_range, monitor_file=monitor_file, parent=self,
            dark=dark, bright=bright, background=background, bright_mode=bright_mode,
            weighted=weighted, context=context, im_trans=card.resolved_im_trans(),
            run_mode=self._frame_run_mode.currentData(), n_workers=self._n_workers.value())
        worker.progress.connect(lambda done, total, n=n: self._cards[n].set_progress(done, total))
        worker.frame_done.connect(
            lambda fid, r_ax, prof, sigma, n=n: self._on_frame(n, fid, r_ax, prof, sigma))
        worker.finished.connect(lambda data, n=n: self._on_panel_done(n, data))
        worker.failed.connect(lambda msg, n=n: self._on_panel_fail(n, msg))
        worker.log_line.connect(lambda line, n=n: self._log.append(f"[ge{n}] {line}"))
        worker.geom_ready.connect(lambda ctx, n=n, s=sig: self._cache_geom(n, s, ctx))
        self._workers[n] = worker
        card.set_status("Starting…")
        self._log.append(f"[ge{n}] starting…")
        worker.start()

    def _on_frame(self, n, fid, r_ax, prof, sigma):
        pair = self._viewer_pairs[n]
        if not pair.wf_started:
            pair.waterfall.reset(r_ax)
            pair.stack_view.reset(r_ax)
            pair.wf_started = True
        pair.waterfall.add_profile(prof)
        pair.stack_view.add_profile(r_ax, prof, label=fid)
        self._integrated_fids[n].add(str(fid))

    def _on_panel_done(self, n: int, data: dict):
        if self._run_cancelled:
            return
        count = data["n"]; out = data.get("out_paths", [])
        aborted = data.get("aborted", False)
        self._cards[n].set_status(f"{'Aborted' if aborted else 'Complete'}: {count} frames")
        verb = "aborted after" if aborted else "done —"
        msg = f"[ge{n}] {verb} {count} frames integrated"
        if out:
            msg += f"\n  saved to: {Path(out[0]).parent}"
        self._log.append(msg)
        if count and data.get("r_axis_px") is not None:
            self._last_results[n] = {
                "r_axis_px": data["r_axis_px"], "profiles": data.get("profiles"),
                "sigmas": data.get("sigmas"), "frame_ids": data.get("frame_ids"),
            }
        self._log_to_project(n, data)
        self._workers.pop(n, None)
        if self._run_mode() == "sequential":
            self._start_next_sequential()
        self._maybe_finish_run()

    def _log_to_project(self, n: int, data: dict):
        if not self._project_ctx or not self._project_ctx.path:
            return
        card = self._cards[n]
        # The whole calibration, not the display subset — an attempt has to be
        # able to reconstruct the geometry it ran under.
        calib_fields, _note = card.full_calib_snapshot()
        calib_ref = None
        if not card.using_file():
            calib_ref = getattr(card.result, "_project_attempt_ref", None)
        try:
            ref = project.append_integration_attempt(
                self._project_ctx.path, f"ge{n}",
                inputs=self._last_run_inputs.get(n, {}), finished_payload=data,
                calibration_snapshot=calib_fields, calib_attempt_ref=calib_ref,
                extra={"active_profile": settings.active_profile()},
                **self._last_run_fields.get(n, {}))
            self._log.append(f"[ge{n}] logged to project: {ref}")
        except Exception:
            import traceback as _tb
            self._log.append(f"[ge{n}] could not log to project file:\n" + _tb.format_exc())

    def _on_panel_fail(self, n: int, msg: str):
        if self._run_cancelled:
            return
        self._cards[n].set_status("Error")
        self._log.append(f"[ge{n}] ERROR:\n{msg}")
        self._workers.pop(n, None)
        if self._run_mode() == "sequential":
            self._start_next_sequential()
        self._maybe_finish_run()

    def _maybe_finish_run(self):
        if self._workers or self._pending_panels:
            return
        self._run_btn.setEnabled(True); self._abort_btn.setEnabled(False)
        self._save_btn.setEnabled(bool(self._last_results))
        self._prog.setVisible(False)
        self._log.append("Hydra batch integration run complete.")

    def _save_results(self):
        """Write every panel's already-computed lineouts to disk — mirrors
        ``BatchTab._save_results``, but writes each panel into its own
        ``ge{n}/`` subfolder under the chosen directory (same layout a run
        with an Output folder set already uses)."""
        if not self._last_results:
            return
        fmts = self._fmt.checked_keys()
        if not fmts:
            QtWidgets.QMessageBox.warning(
                self, "No format", "Check at least one output format first."); return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Save lineouts to…", self._out_ed.text().strip())
        if not out_dir:
            return
        if "2d_csv" in fmts:
            self._log.append(
                "[batch] Note: 2D CSV (cake) isn't saved by Save — per-frame "
                "cakes aren't kept in memory after a run; re-run with an "
                "Output folder set to get that format.")
        total = 0
        for n, results in sorted(self._last_results.items()):
            axis_ctx = self._last_axis_ctx.get(n)
            if axis_ctx is None:
                continue
            lsd, px, wl = axis_ctx
            try:
                paths = write_all_profiles(
                    Path(out_dir) / f"ge{n}", fmts, results["r_axis_px"],
                    results["profiles"], results["sigmas"], results["frame_ids"],
                    lsd, px, wl, omegas=results.get("omegas"))
                total += len(paths)
            except Exception:
                import traceback as _tb
                self._log.append(f"[ge{n}] save failed:\n" + _tb.format_exc())
        self._log.append(f"Saved {total} file(s) to {out_dir}")

    def _abort_all(self):
        """Stop every running panel. Mirrors ``HydraCalibrationPage._abort_all``:
        disconnect + orphan immediately (no per-worker wait), since waiting on
        several concurrent (Parallel-mode) workers in turn would freeze the UI
        for their combined grace periods. Frames already integrated were
        written to disk as each panel's run went."""
        if not self._workers:
            return
        self._run_cancelled = True
        for n, w in list(self._workers.items()):
            for sig in (w.progress, w.frame_done, w.finished, w.failed, w.log_line, w.geom_ready):
                try:
                    sig.disconnect()
                except Exception:
                    pass
            w.requestInterruption()
            self._cards[n].set_status("Aborting…")
            self._orphans.append(w)
        self._workers.clear()
        self._pending_panels = []
        self._run_btn.setEnabled(True); self._abort_btn.setEnabled(False)
        self._prog.setVisible(False)
        self._log.append("Hydra batch integration aborted — a background thread per panel "
                         "may still be winding down. Frames already integrated were saved.")

    def shutdown(self):
        """Interrupt + orphan any still-running per-panel workers on app
        close (picked up by ``MainWindow._stop_all_workers``'s generic
        ``tab.shutdown()`` hook via ``BatchTab.shutdown``)."""
        self._abort_all()

    # ── GUI state ────────────────────────────────────────────────────

    def _state_widgets(self) -> dict:
        return {
            "kernel": self._kernel, "r_bin": self._r_bin, "e_bin": self._e_bin,
            "r_min": self._r_min, "r_max": self._r_max, "grid_chk": self._grid_chk,
            "azim": self._azim, "var_check": self._var_check, "err_model": self._err_model,
            "q_check": self._q_check, "q_min": self._q_min, "q_max": self._q_max,
            "q_bin": self._q_bin, "mon_ed": self._mon_ed, "out_ed": self._out_ed,
            "run_mode": self._run_mode_combo, "frame_run_mode": self._frame_run_mode,
            "n_workers": self._n_workers,
        }

    def get_state(self) -> dict:
        return {
            "anchor_path": self._loader.current_path(),
            "active_panel": self._toolbar.current(),
            "fields": widgets_to_dict(self._state_widgets()),
            "corr": self._corr_widget.get_state(),
            "fmt": self._fmt.get_state(),
            "loader": self._loader.get_state(),
            "det_view": self._det_view.display_state(),
            "waterfalls": {n: pair.waterfall.display_state()
                           for n, pair in self._viewer_pairs.items()},
            "cards": {n: widgets_to_dict(card.state_widgets())
                     for n, card in self._cards.items()},
        }

    def set_state(self, state: dict):
        if not state:
            return
        fields = dict(state.get("fields", {}))
        # A project-attempt's "fmt_keys" (see project.integrate_attempt_gui_fields)
        # rides along in "fields" but isn't a plain widget — see BatchTab.set_state.
        fmt_keys = fields.pop("fmt_keys", None)
        apply_dict_to_widgets(self._state_widgets(), fields)
        self._corr_widget.set_state(state.get("corr") or {})
        self._fmt.set_state(fmt_keys if fmt_keys is not None else state.get("fmt"))
        self._loader.set_state(state.get("loader") or {})
        self._det_view.set_display_state(state.get("det_view"))
        self._origin_btn.sync()
        for n_key, wf_state in (state.get("waterfalls") or {}).items():
            pair = self._viewer_pairs.get(int(n_key))
            if pair is not None:
                pair.waterfall.set_display_state(wf_state)
        for n_key, fields in (state.get("cards") or {}).items():
            card = self._cards.get(int(n_key))
            if card is None:
                continue
            apply_dict_to_widgets(card.state_widgets(), fields)
        anchor = state.get("anchor_path")
        if anchor and Path(anchor).exists():
            self._loader.set_path(anchor)
        panel = state.get("active_panel")
        if panel:
            self._toolbar.set_current(panel)

    # ── Cross-tab hand-off ───────────────────────────────────────────

    def set_panel_calibration(self, n: int, result):
        """A Hydra panel's fit finished on the Calibrate tab — auto-populate
        that panel's calibration source here (mirrors ``BatchTab.set_calibration``)."""
        card = self._cards.get(n)
        if card is not None:
            card.set_calibration(result)
            # The Calibrate hand-off is the main route by which REAL geometry
            # arrives, so it is the invalidation most likely to be missed and
            # the most damaging to miss: the composite would keep placing this
            # panel at whatever geometry it was first built with, looking
            # entirely plausible. Same class as the 1-ID-E incident recorded
            # in hydra_page._reseed_composite_card_if_needed's comment.
            self._on_panel_calibration_changed(n)

    def populate_panel_plots(self, n: int, meta: dict) -> None:
        """Fill panel ``n``'s own Waterfall/Stacked-profiles views from a
        project integration attempt's embedded arrays — mirrors
        ``BatchTab._populate_plots_from_attempt`` for one Hydra panel. Each
        panel keeps its own independent viewer pair (``_viewer_pairs``), so
        populating several panels' worth here is what makes the toolbar
        selection (GE1/GE2/…) actually show panel-specific results."""
        arrays = meta.get("_results_arrays") or {}
        r_axis = arrays.get("r_axis_px")
        profiles = arrays.get("profiles")
        if r_axis is None or profiles is None or len(profiles) == 0:
            return
        pair = self._viewer_pairs.get(n)
        if pair is None:
            return
        try:
            spec = self._cards[n].resolved_spec(self._r_bin.value(), self._e_bin.value())
            axctx = (float(spec.Lsd), float(spec.pxY), float(spec.Wavelength))
            pair.waterfall.set_axis_context(*axctx)
            pair.stack_view.set_axis_context(*axctx)
        except Exception:
            pass
        pair.waterfall.reset(r_axis)
        pair.stack_view.reset(r_axis)
        frame_ids = arrays.get("frame_ids") or list(range(len(profiles)))
        self._integrated_fids[n] = set()
        for fid, prof in zip(frame_ids, profiles):
            pair.waterfall.add_profile(prof)
            pair.stack_view.add_profile(r_axis, prof, label=fid)
            self._integrated_fids[n].add(str(fid))
        pair.wf_started = True
