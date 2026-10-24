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
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui.constants import (KERNELS, ERROR_MODELS,
                                 DEFAULT_KERNEL, DEFAULT_ERROR_MODEL)
from midas_gui.helpers import (collapse_cake_eta,
    _fspin, _browse, _NoScrollComboBox, _NoScrollSpinBox,
    widgets_to_dict, apply_dict_to_widgets,
    _load_image, rmax_corner_px, rmax_edge_px, draw_polar_bin_overlay,
    browse_start_dir, warn_if_path_missing, suggest_panel_base_output_dir, check_output_dir_writable,
    suggest_working_dir, bc_path_parts, source_kind, is_h5,
    list_h5_1d_datasets, _folder_output_dir)
from midas_gui.widgets import (LogPanel, CorrectionFlagsWidget, WaterfallViewer, CakeStackViewer,
                               StackedProfileViewer, OutputFormatSelector, ImageViewer,
                               OriginToolButton, build_lab_frame_axes_items)
from midas_gui.hydra_widgets import (HydraLoaderPanel, HydraDetectorToolbar,
                                     panel_color)
from midas_gui.hydra_batch_widgets import (HydraBatchPanelCard,
                                           HydraCakingSummary)
from midas_gui.job_queue import JobQueuePanel
from midas_gui.dark_policy import DarkArgBuilder
from midas_gui import hydra_zarr_combine
from midas_gui.workers import BatchRunCoordinator, write_all_profiles, REBIN_UNITS
from midas_gui import cake_params
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
        self._composite_panels: list = []  # panels actually on the canvas
        #: Multi-scan batch (a recursive folder pick). Empty / None for the
        #: ordinary one-scan run, which then behaves exactly as before.
        self._scan_queue: list = []
        #: True while a parallel scan's panels are still being started.
        self._launching = False
        self._scan_total = 0
        self._scan_pos = 0
        self._active_scan: "Optional[dict]" = None
        self._active_out_base: "Optional[str]" = None
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
        # Same slot the single-detector tab uses: origin, grid, lab axes.
        # Both are GLOBAL, view-level toggles -- they change what the
        # Detector view draws, not what the integration computes -- so they
        # belong together, and in one fixed place.
        #
        # This used to be reparented into whichever caking card was showing,
        # on the reasoning that it draws the caking values. That made a
        # single shared toggle LOOK per-panel while actually being global,
        # and made the widget physically migrate across the page on every
        # ge1->ge2 switch. Toolbar instead (2026-10-12), matching
        # tab_batch.py and `Lab-frame axes` right beside it.
        self._grid_chk = QtWidgets.QCheckBox("Show bin grid")
        self._grid_chk.setToolTip(
            "Overlay the Rmin/Rmax boundary circles and the full (R, η) "
            "integration bin grid on each panel's Detector view tab — "
            "concentric circles at each R-bin edge, spokes at each η-bin "
            "edge. Thinned to at most ~50 rings / ~72 spokes for "
            "legibility with fine bin sizes. Unchecking this hides the "
            "overlay entirely, including Rmin/Rmax.")
        self._grid_chk.toggled.connect(self._refresh_active_detector_preview)
        self._det_view._toolbar_layout.addWidget(self._grid_chk)
        self._lab_axes_chk = QtWidgets.QCheckBox("Lab-frame axes")
        self._lab_axes_chk.setToolTip(
            "Overlay MIDAS lab-frame axes (X_Lab/Y_Lab), the beam-direction "
            "\u2297 glyph, and an \u03b7 sweep arc, anchored at the selected "
            "panel's beam centre \u2014 same overlay as the Data Viewer, "
            "Calibrate and single-detector Batch Integrate.\n\n"
            "Not drawn on the Composite: the canvas registers every panel's "
            "beam centre onto one point, so there is no single panel whose "
            "orientation the compass would be describing.")
        self._lab_axes_chk.toggled.connect(self._on_lab_axes_toggled)
        # Flipping the display origin inverts the ViewBox's Y axis; the
        # compass points at the hutch, not the pixel grid, so it is
        # re-derived rather than carried along.
        self._det_view.originChanged.connect(self._on_origin_changed)
        self._det_view._toolbar_layout.addWidget(self._lab_axes_chk)
        # Last in the toolbar, as on the single-detector tab.
        self._det_view._toolbar_layout.addWidget(
            QtWidgets.QLabel("Preview: sum first"))
        self._preview_sum_n = _NoScrollSpinBox()
        self._preview_sum_n.setRange(1, 999); self._preview_sum_n.setValue(1)
        self._preview_sum_n.setFixedWidth(50)
        self._preview_sum_n.setToolTip(
            "Detector-view preview only — never affects the real batch run. "
            "Sums this many frames from the current index before display, "
            "to boost signal enough to see the actual diffraction pattern "
            "beneath detector-readout artifacts that dominate a single "
            "frame.\n\n"
            "Read synchronously here, one frame at a time, so keep it small "
            "on an NFS-mounted share.")
        self._preview_sum_n.valueChanged.connect(self._on_preview_sum_changed)
        self._det_view._toolbar_layout.addWidget(self._preview_sum_n)
        self._axis_items: list = []
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
        self._azim = _NoScrollComboBox()
        self._azim.addItem("Pixel-weighted", True)
        self._azim.addItem("η-bin mean (legacy)", False)
        intf = S.Form()
        intf.row(("Kernel:", self._kernel))
        intf.row(("Azim. mean:", self._azim))
        integ.body.addLayout(intf)
        # Rmin/Rmax/R bin/eta bin are NOT here: caking is per panel, on each
        # HydraBatchPanelCard (see its module docstring). What stays shared
        # is the processing recipe -- kernel, azimuthal weighting, variance,
        # Q-uniform bins -- none of which mpe_wf models per detector either.
        # Rotation. SHARED, not per panel: there is one goniometer, and all
        # four panels see the same omega for a given frame. mpe_wf's
        # cake_parameters CSV carries OME_* per detector, redundantly.
        #
        # This page had no rotation controls at all, so it never passed an
        # omega_cfg and BatchWorker fell through to start=step=0 -- every
        # frame recorded at omega=0, which is precisely the all-zero /Omegas
        # the single-detector tab goes out of its way to warn about. A
        # single-frame scan never notices; a rotation scan is simply wrong.
        # In a container, not loose in this card: it is MOVED to sit under
        # whichever caking card is showing (_place_shared_cake_widgets), the
        # same treatment the bin-grid toggle got. Asked for twice while
        # looking at the caking table -- "omega bin size is not available
        # here" -- and a control the run depends on being two cards away
        # from where you look for it is not available in any useful sense.
        self._ome_box = QtWidgets.QWidget()
        ome_v = QtWidgets.QVBoxLayout(self._ome_box)
        ome_v.setContentsMargins(0, 0, 0, 0); ome_v.setSpacing(2)
        ome_row = QtWidgets.QHBoxLayout(); ome_row.setSpacing(4)
        self._ome_start = _fspin(-100000.0, 100000.0, 4, 0.0, "°")
        self._ome_step = _fspin(-100000.0, 100000.0, 6, 0.0, "°")
        self._ome_start.setToolTip(
            "OME_START — ω of raw sub-frame 0.\n\nShared across ge1–ge4: "
            "one rotation stage, one angle per frame.")
        self._ome_step.setToolTip(
            "OME_STEP — ω advanced per raw sub-frame (the omega bin size).\n\n"
            "Left at 0 with OME_START 0, every frame is written to the zarr's "
            "/Omegas as 0°, which is right only for a stationary scan.")
        # Measured, rather than assumed from a ramp. "What those frames are
        # in angle or position": a 1-D dataset in the source HDF5 holding
        # the real rotation (or a motor position) per frame. Read per frame
        # INSTEAD of the OME_START/OME_STEP ramp, which only describes a
        # scan that stepped perfectly. Editable because the list is a
        # convenience, not the set of legal answers.
        self._ome_channel = _NoScrollComboBox()
        self._ome_channel.setEditable(True)
        self._ome_channel.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._ome_channel.setToolTip(
            "A 1-D dataset in the source HDF5 holding the measured rotation "
            "or position, read per frame instead of the ω start / ω step "
            "ramp.\n\nBlank falls back to the ramp.")
        # "one ω for ALL frames", which is what it does -- tab_batch's
        # wording. The first label, "one ω per frame (averaged)", described
        # chunk averaging, which is a different thing the run does anyway;
        # this is the collapse OVERRIDE, a single constant for the whole
        # run. And it is tested before the channel in _omega_resolver, so
        # ticking it silently discards a chosen omega channel.
        self._ome_collapse = QtWidgets.QCheckBox("one ω for all frames")
        self._ome_collapse.setToolTip(
            "These images were averaged or summed outside this app, so no "
            "per-frame angle describes them: give EVERY frame the one ω at "
            "the middle of the run.\n\nOverrides the ω channel — tick it "
            "only when there is genuinely one exposure.")
        ome_row.addWidget(QtWidgets.QLabel("ω start:"))
        ome_row.addWidget(self._ome_start)
        ome_row.addWidget(QtWidgets.QLabel("ω step:"))
        ome_row.addWidget(self._ome_step)
        ome_row.addStretch(1)
        ome_v.addLayout(ome_row)
        chan_row = QtWidgets.QHBoxLayout(); chan_row.setSpacing(4)
        chan_row.addWidget(QtWidgets.QLabel("ω from channel:"))
        chan_row.addWidget(self._ome_channel, 1)
        ome_v.addLayout(chan_row)
        ome_v.addWidget(self._ome_collapse)
        self._ome_lbl = QtWidgets.QLabel("")
        self._ome_lbl.setWordWrap(True)
        self._ome_lbl.setStyleSheet("color:#8AB4F8;font-size:10px")
        ome_v.addWidget(self._ome_lbl)
        for w in (self._ome_start, self._ome_step):
            w.valueChanged.connect(lambda *_: self._refresh_omega_note())
        self._ome_collapse.toggled.connect(lambda *_: self._refresh_omega_note())
        self._ome_channel.currentTextChanged.connect(
            lambda *_: self._refresh_omega_note())
        # The chunk lives on the loader, two columns away, and decides what
        # each frame's ω averages over -- "so in this case, how are the
        # frames chunked?" could not be answered from this card.
        self._loader._combine_chunk.valueChanged.connect(
            lambda *_: self._refresh_omega_note())
        self._refresh_omega_note()

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
        # Output rebin. A three-way Bin type, matching tab_batch.py --
        # this was a Q-only checkbox, so the Hydra page could not produce
        # 2theta-uniform output AT ALL while the single-detector tab could
        # (added there with the "one output rebin, two units" work; see
        # DECISIONS 2026-10-06). Same vocabulary, same semantics: the
        # integration grid is always R-uniform and these resample the
        # finished profile (workers.rebin_R_to_grid).
        self._bin_type = _NoScrollComboBox()
        self._bin_type.addItem("Radial", "R")
        self._bin_type.addItem("Q", "Q")
        self._bin_type.addItem("2θ", "2th")
        self._bin_type.setToolTip(
            "Radial — bin uniformly in R (detector pixels).\n"
            "Q — additionally rebin the OUTPUT uniformly in Q (Å⁻¹).\n"
            "2θ — additionally rebin the OUTPUT uniformly in 2θ (degrees).\n"
            "In all three the underlying integration grid is R-uniform; Q "
            "and 2θ resample that result, and the axis stored alongside a "
            "profile stays R in pixels.\n\n"
            "Shared by all four panels, like the rest of this card.")
        self._bin_type.currentIndexChanged.connect(self._on_bin_type_changed)
        bt = S.Form(); bt.row(("Bin type:", self._bin_type))
        integ.body.addLayout(bt)
        self._q_min = _fspin(0.0, 100.0, 3, 0.5, "Å⁻¹")
        self._q_max = _fspin(0.0, 100.0, 3, 8.0, "Å⁻¹")
        self._q_bin = _fspin(0.0001, 1.0, 4, 0.01, "Å⁻¹")
        self._tth_min = _fspin(0.0, 89.999, 4, 0.0, "°")
        self._tth_max = _fspin(0.0, 89.999, 4, 10.0, "°")
        self._tth_bin = _fspin(0.0001, 10.0, 4, 0.01, "°")
        self._q_form = S.Form()
        self._q_form.row(("Qmin:", self._q_min), ("Qmax:", self._q_max))
        self._q_form.row(("ΔQ:", self._q_bin))
        self._tth_form = S.Form()
        self._tth_form.row(("2θmin:", self._tth_min), ("2θmax:", self._tth_max))
        self._tth_form.row(("Δ2θ:", self._tth_bin))
        integ.body.addLayout(self._q_form)
        integ.body.addLayout(self._tth_form)
        self._on_bin_type_changed()
        # Same control, same wording as tab_batch.py -- it was simply absent
        # here, and nothing about four panels prevents it (each panel writes
        # its own cake into its own ge{n}/ folder, exactly as it writes its
        # own profile).
        self._multi_azimuth_chk = QtWidgets.QCheckBox("Multi-azimuth output (cake)")
        self._multi_azimuth_chk.setToolTip(
            "Keep every azimuthal (η) sector as a SEPARATE output profile "
            "instead of collapsing to one mean profile per frame. Reuses "
            "each panel's own η bin/η range to define the sectors.\n\n"
            "Off by default — η bin already exists to control "
            "collapse-weighting resolution; turning this on repurposes that "
            "same setting to also define real output granularity, so every "
            "existing run's result size is unaffected unless you opt in.\n\n"
            "Needed for per-azimuth GSAS-II/texture analysis. Not supported "
            "together with Q- or 2θ-uniform bins.")
        integ.body.addWidget(self._multi_azimuth_chk)
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
        # The box is AUTHORITATIVE for a single scan, so a stale one does
        # not merely look wrong -- it silently writes this dataset's output
        # into another's folder. Loading 4130_D_load_step_0 with the box
        # still on APS_4130_C_load_step_0 did exactly that.
        self._out_mismatch_lbl = QtWidgets.QLabel("")
        self._out_mismatch_lbl.setWordWrap(True)
        self._out_mismatch_lbl.setStyleSheet("color:#FFA500;font-size:10px")
        self._out_mismatch_lbl.setVisible(False)
        out.body.addWidget(self._out_mismatch_lbl)
        self._fmt = OutputFormatSelector()
        out.body.addWidget(self._fmt)
        # How many integrated frames share one archive. The Hydra page never
        # passed this, so BatchWorker fell through to its "frame" default
        # and wrote ONE .zarr.zip per frame: a 834-frame scan came out as
        # 834 archives per panel, which is both not what anyone wants and
        # slow -- every archive is opened, written and closed on its own,
        # and the four-panel combine then has 834 sets to merge instead of
        # one. Reported as "I would like these frames saved into 1 zarr
        # file ... the job might be stuck on the first one".
        self._zarr_grouping = _NoScrollComboBox()
        for _label, _key in (("One zarr for the whole scan", "run"),
                             ("One zarr per source file", "file"),
                             ("One zarr per output frame", "frame")):
            self._zarr_grouping.addItem(_label, _key)
        self._zarr_grouping.setToolTip(
            "How many integrated frames share one .zarr.zip.\n\n"
            "Per source file groups by ROTATION — the same unit omega is "
            "measured from, so one HDF5 sub-frame stack becomes one "
            "archive.\n\n"
            "Per output frame is one archive per frame: rarely wanted on a "
            "multi-frame scan, and the reason a 834-frame run looked stuck.")
        # "run" first and therefore the default, unlike the single-detector
        # tab: a Hydra scan is a rotation and its frames belong together,
        # and the four-panel combine wants one archive per panel to merge.
        out.body.addLayout(S.Form().row(("Zarr grouping:", self._zarr_grouping)))
        self._combine_zarr_chk = QtWidgets.QCheckBox(
            "Combine panels into one zarr (eta-concatenated)")
        self._combine_zarr_chk.setChecked(True)
        self._combine_zarr_chk.setToolTip(
            "After every panel finishes, merge each frame's four per-panel "
            "zarr archives into one, concatenated along eta — the same "
            "archive mpe_wf_saxs_waxs's combine_hydra_zarr.py produces "
            "(ported; see midas_gui/hydra_zarr_combine.py).\n\n"
            "Written beside the ge1 archive it came from, with the panel "
            "token replaced: <froot>_<NNNNNN>.ge1.zarr.zip becomes "
            "<froot>_<NNNNNN>.ge1234.zarr.zip.\n\n"
            "Needs Zarr among the output formats and an Output folder. A "
            "frame any panel did not write is skipped rather than merged "
            "from a partial set, which would be silently short on eta.")
        out.body.addWidget(self._combine_zarr_chk)
        # Optional beam-monitor CSV columns. Written per panel, beside the
        # per-panel CSV this page already produces (<froot>_ge{n}_bi.csv) --
        # the control was simply missing here, so those extra columns could
        # not be asked for on a Hydra run at all.
        ion_note = QtWidgets.QLabel(
            "Ion-chamber CSV — written automatically, one row per frame "
            "(frame, I0, I, transmission). Also include:")
        ion_note.setWordWrap(True)
        ion_note.setStyleSheet(f"color:{S.MUTED};font-size:10px;")
        out.body.addWidget(ion_note)
        self._ion_env_chk = QtWidgets.QCheckBox("Ring current / T / P")
        self._ion_env_chk.setToolTip(
            "Add storage-ring current (mA), temperature and pressure "
            "columns.\n"
            "Older 20-ID files carry placeholder PVs that read NaN for "
            "temperature and pressure; a column that is NaN on every frame "
            "is dropped rather than written as a wall of 'nan'.")
        self._ion_motors_chk = QtWidgets.QCheckBox("Sample motors")
        self._ion_motors_chk.setToolTip(
            "Add one column per sample-stage channel.\n"
            "E hutch has two coexisting sub-configs (HL/HR) with no reliable "
            "flag for which is in use; both are read, and the inactive one "
            "is all-NaN and therefore dropped. The data says which was live.")
        ion_row = QtWidgets.QHBoxLayout(); ion_row.setSpacing(6)
        ion_row.addWidget(self._ion_env_chk)
        ion_row.addWidget(self._ion_motors_chk)
        ion_row.addStretch(1)
        out.body.addLayout(ion_row)
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
        # Same control, same wording, same mechanism as the single-detector
        # tab's (tab_batch.BatchTab._run_as_job) — the deployment interface is
        # deliberately identical between the two pages.
        self._run_job_btn = QtWidgets.QPushButton("Run as background job")
        self._run_job_btn.setToolTip(
            "Linux only. Runs this same integration in a detached `screen` "
            "session instead of in-process — survives closing this GUI. Each "
            "scan runs its panels one after another (python -m "
            "midas_gui.batch_cli, once per panel) and then the four-panel "
            "combine (python -m midas_gui.hydra_combine_cli), all in the one "
            "session. Needs an Output folder (calibration snapshots, masks "
            "and dark/bright are written there for the job to read).\n"
            "Progress/log/reattach/cancel are tracked in the 'Background "
            "jobs' panel below.")
        self._run_job_btn.clicked.connect(self._run_as_job)
        run_card.body.addWidget(self._run_job_btn)
        # How many SCANS a background job runs at once. Panels within a scan
        # always run together (they are independent and a scan's combine
        # waits only for its own), so the process count is roughly this x4.
        #
        # Background-job only, and deliberately separate from the two
        # in-process Run-mode controls above: those drive threads inside
        # this GUI, this drives separate OS processes in the detached
        # session, where there is no GIL and no shared Qt state.
        jp_row = QtWidgets.QHBoxLayout(); jp_row.setSpacing(6)
        jp_row.addWidget(QtWidgets.QLabel("Background job — scans at once:"))
        self._job_scans = _NoScrollSpinBox()
        self._job_scans.setRange(1, 64)
        self._job_scans.setValue(1)
        self._job_scans.setToolTip(
            "How many scans a background job processes concurrently. Each "
            "scan runs all of its panels at once, so the number of "
            "processes is about this times the panel count.\n\n"
            "1 keeps the old strictly-sequential behaviour. Raising it is "
            "the main reason to use a background job on a big sweep: the "
            "in-process run is limited to one scan at a time.\n\n"
            "The ceiling is usually the output filesystem, not CPU — a scan "
            "writes a few thousand small files, so try 4 before 16 and "
            "watch whether the rate actually improves.")
        jp_row.addWidget(self._job_scans)
        jp_row.addStretch(1)
        run_card.body.addLayout(jp_row)
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
            # The caking spins drive BOTH this panel's overlay and its
            # region on the composite. cakingChanged was emitted all along
            # and nothing on this page connected it, so every Rmin/Rmax/eta
            # edit left the Detector view showing the previous values until
            # some unrelated refresh happened to fire.
            #
            # No invalidate_composite(): caking does not change the PICTURE,
            # only what is drawn over it, and a rebuild is a ~1 s GUI-thread
            # remap per keystroke.
            card.cakingChanged.connect(self._on_panel_caking_changed)
            card.cakeLoadRequested.connect(self._load_cake_csv)
            card.cakeSaveRequested.connect(self._save_cake_csv)
        # A 5th page in the stack, for the Composite selection. The four
        # panel cards each answer "what is THIS panel caked to"; the one
        # view that is about all four had no card of its own and simply
        # held whichever panel was last open.
        self._cake_summary = HydraCakingSummary()
        self._load4_btn = QtWidgets.QPushButton("Load 4 panel calibrations…")
        self._load4_btn.setToolTip(
            "Pick ONE panel's calibration file. The other three are found "
            "beside it by rewriting the ge<n> in its path — the same rule "
            "the cake_parameters CSVs follow, and the one the files are "
            "written under.\n\n"
            "Any panel whose file is not there is named rather than skipped "
            "quietly: a composite built from three real calibrations and one "
            "stale one looks plausible and is wrong.")
        self._load4_btn.clicked.connect(self._load_four_calibrations)
        self._cake_summary.top_extras_layout().addWidget(self._load4_btn)
        self._load4_lbl = QtWidgets.QLabel("")
        self._load4_lbl.setStyleSheet("color:#9a9a9a;font-size:10px")
        self._load4_lbl.setWordWrap(True)
        self._cake_summary.top_extras_layout().addWidget(self._load4_lbl)
        # The table forwards edits rather than holding values: the panel's
        # own spin box is still what the run reads, and what clamps.
        self._cake_summary.cakingEdited.connect(self._on_summary_cake_edited)
        self._card_stack.addWidget(self._cake_summary)
        lv.addWidget(self._card_stack)
        self._place_shared_cake_widgets(self._cards[1])
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
        # The MERGED ge1234 result, which had no view anywhere: this page
        # showed four per-panel results, and the archive a Hydra run
        # actually produces -- the only place all four panels' eta coverage
        # appears on one continuous axis, ge1's +/-180 crossing included --
        # had to be opened outside the GUI. The per-panel tab stays: one
        # panel at a time is still how you check a single panel's
        # calibration, which is a different question from what the run
        # produced.
        self._merged_waterfall = WaterfallViewer()
        self._merged_stack = StackedProfileViewer()
        self._merged_cakes = CakeStackViewer()
        self._top_tabs.addTab(self._merged_waterfall, "Waterfall")
        self._top_tabs.addTab(self._merged_stack, "Stacked profiles")
        self._top_tabs.addTab(self._merged_cakes, "Eta-R cakes")
        #: path of the merged archive currently on display, if any
        self._merged_shown = None
        # Build the composite on the way IN to the Detector view, and not at
        # all while Per-panel results is showing: a ~1 s GUI-thread remap for
        # pixels behind another tab is the cost this gate exists to avoid.
        self._top_tabs.currentChanged.connect(
            lambda *_: self._refresh_active_detector_preview())
        right.addWidget(self._top_tabs)
        self._log = LogPanel()
        self._log.setMaximumHeight(16_777_215)
        # Background-job rows above the shared log, exactly as the
        # single-detector tab lays its Logs tab out: one row per job, with
        # progress, "Reattach" (copies `screen -r <session>`), Cancel and
        # live tailing of that job's screenlog into the log below.
        self._job_queue = JobQueuePanel(self._log)
        right.addWidget(self._job_queue)
        right.addWidget(self._log)
        right.setStretchFactor(0, 4); right.setStretchFactor(1, 1)
        right.setMinimumWidth(320)
        split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 0); split.setStretchFactor(2, 1)
        split.setSizes([286, 361, 950])

    # ── Loader signal handlers ──────────────────────────────────────

    def _on_siblings_changed(self, siblings: dict):
        self._toolbar.set_available(siblings.keys())
        self._populate_omega_channels()
        self._refresh_output_mismatch()
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
        self._out_state_was_legacy = False   # we own the box again
        self._refresh_output_mismatch()
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
        """Replace a path we filled in; never one the user chose.

        With one exception, learned the hard way twice. A project saved
        BEFORE _autofilled_out was persisted restores a box with no marker,
        so the box looks hand-typed and the autofill backs off forever --
        and because the box is authoritative for a single scan, the run
        then writes this dataset's output into the previous dataset's
        folder. That is how APS_4130_C_load_step_0 came to hold
        4130_D_load_step_0's ge1..ge4.

        So when there is NO marker at all (only a legacy restore produces
        that; a real edit in this session always sets one) and the box
        disagrees with what the loaded data implies, the box is replaced
        and the swap is logged. A path that matches the data is left alone,
        and so is any box in a session where the marker exists.
        """
        current = self._out_ed.text().strip()
        marker = getattr(self, "_autofilled_out", None)
        if current and current != (marker or ""):
            # ONLY a legacy restore, not merely "no marker": a fresh page
            # starts with marker None too, so a path typed into one would
            # otherwise be overwritten -- which is the opposite of the rule
            # this function exists to keep.
            if not getattr(self, "_out_state_was_legacy", False):
                return
            suggested = self._suggest_output_dir()
            if suggested is None or Path(current).name == Path(str(suggested)).name:
                return
            self._log.append(
                f"[hydra batch] Output folder '{Path(current).name}' does not "
                f"match the loaded data — replaced with "
                f"'{Path(str(suggested)).name}'. (A project saved before this "
                f"was recorded cannot say whether the old path was chosen or "
                f"auto-filled; re-type it if it was deliberate.)")
        suggested = self._suggest_output_dir()
        if suggested is None or str(suggested) == current:
            return
        self._set_output_dir_suggested(suggested)

    def _refresh_output_mismatch(self) -> None:
        """Say so when the Output box does not describe the loaded data.

        Checked regardless of how the box got its value -- a restored
        project, a hand-typed path, or an autofill left over from the last
        dataset all land the same way, and the box being authoritative for
        a single scan means any of them silently redirects the output.
        """
        lbl = getattr(self, "_out_mismatch_lbl", None)
        if lbl is None:
            return
        current = self._out_ed.text().strip()
        suggested = self._suggest_output_dir()
        bad = bool(current and suggested
                   and Path(current).name != Path(str(suggested)).name)
        lbl.setVisible(bad)
        if bad:
            lbl.setText(
                f"⚠ Output folder is named for '{Path(current).name}' but "
                f"the loaded data implies '{Path(str(suggested)).name}'. "
                f"The run writes where this box says — clear it to re-fill.")

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

    def _place_shared_cake_widgets(self, host) -> None:
        """Move the rotation row into ``host``'s caking card.

        The rotation row is page-level state that belongs WITH the caking
        values rather than with the processing recipe -- it is the ninth,
        tenth and eleventh cake_parameters fields. It is one widget,
        reparented to follow the visible card rather than copied per card:
        it is persisted by identity in ``_state_widgets``, and five copies
        would have to be kept in step with each other for no benefit.

        The bin-grid toggle used to ride along here too. It does not any
        more -- it is a view-level toggle, not a caking value, and it now
        lives in the Detector view toolbar beside `Lab-frame axes`, where
        the single-detector tab has always kept it (2026-10-12).

        Qt reparents on addWidget, but the OLD layout would still hold an
        item pointing at the widget, so it is removed explicitly first.
        """
        for w in (self._ome_box,):
            old = w.parentWidget().layout() if w.parentWidget() else None
            if old is not None:
                old.removeWidget(w)
            host.cake_extras_layout().addWidget(w)
            w.show()

    def _on_panel_changed(self, key: str):
        n = self._panel_num(key)
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
            self._card_stack.setCurrentWidget(self._cake_summary)
            self._place_shared_cake_widgets(self._cake_summary)
            self._refresh_cake_summary()
            self._top_tabs.setCurrentWidget(self._det_view)
            self._refresh_detector_preview(None)
            return
        self._card_stack.setCurrentWidget(self._cards[n])
        self._place_shared_cake_widgets(self._cards[n])
        self._viewer_stack.setCurrentWidget(self._viewer_pairs[n])
        self._refresh_detector_preview(n)

    # ── Detector-view preview (Rmin/Rmax + bin-grid overlay) ─────────

    def _refresh_active_detector_preview(self, *_args) -> None:
        key = self._toolbar.current()
        if key:
            self._refresh_detector_preview(self._panel_num(key))

    # ── Lab-frame axes (same overlay as the Data Viewer / Calibrate /
    # single-detector Batch Integrate — widgets.build_lab_frame_axes_items) ──
    def _on_lab_axes_toggled(self, checked: bool) -> None:
        if checked:
            self._refresh_active_detector_preview()
        else:
            self._clear_lab_axes()

    def _on_origin_changed(self, *_args) -> None:
        """Display origin flipped — the compass is drawn in screen terms, so
        it has to be re-derived rather than carried along by the ViewBox's
        now-inverted Y axis."""
        if self._lab_axes_chk.isChecked():
            self._refresh_active_detector_preview()

    def _redraw_lab_axes_if_on(self, bc_y: float, bc_z: float,
                               tx: float = 0.0) -> None:
        """``tx`` is the panel's installation roll. Without it the compass
        puts η=0 straight up on a panel whose η=0 is a full tx round from
        there -- and at 1-ID-E the four Hydra panels are rolled 300/30/120/
        210, so all four read wrong, each by a different amount, while the
        caking sector beside them (tx-aware) was right."""
        if not self._lab_axes_chk.isChecked():
            return
        self._clear_lab_axes()
        img = self._det_view._data
        if img is None:
            return
        items = build_lab_frame_axes_items(self._det_view._iv, img.shape,
                                           bc_y, bc_z, tx=tx)
        for it in items:
            self._det_view._iv.addItem(it)
        self._axis_items.extend(items)

    @staticmethod
    def _composite_axes_bc(img):
        """``(bc_y, bc_z)`` for the composite canvas, or ``None``.

        The compositor registers every panel's beam centre onto the centre of
        the BigDet canvas, so on the built (decimated) image that is simply
        its middle. ``build_lab_frame_axes_items`` wants (y, z) = (column,
        row), while an array's shape is (rows, cols).
        """
        if img is None:
            return None
        shape = getattr(img, "shape", None)
        if not shape or len(shape) < 2:
            return None
        n_rows, n_cols = int(shape[0]), int(shape[1])
        if n_rows <= 0 or n_cols <= 0:
            return None
        return n_cols / 2.0, n_rows / 2.0

    def _clear_lab_axes(self) -> None:
        for it in self._axis_items:
            self._det_view._iv.removeItem(it)
        self._axis_items.clear()

    def _on_summary_cake_edited(self, n: int, key: str, value: float) -> None:
        """Apply an edit made in the all-panels table to that panel's card.

        set_caking emits cakingChanged, which repaints this table from the
        cards -- so the clamped value is what ends up on screen, and the
        overlay and the coverage line follow in the same step.
        """
        card = self._cards.get(n)
        if card is None:
            return
        card.set_caking({key: value})
        self._refresh_cake_summary()     # also covers a clamped no-op

    def _load_four_calibrations(self) -> None:
        """One pick, four panels — from the Composite view.

        Asked for at 1-ID-E: "we should be able to load the calibration file
        from composite. Pick one for one panel and find others automatically
        by pattern. If file that matches the pattern does not exist alert."

        The Composite view is the one that needs all four calibrations at
        once -- it cannot be built without them -- and it was the one view
        with no way to supply them, since the per-panel cards are hidden
        behind it.

        A missing sibling is an ALERT, not a log line. Loading three of four
        and carrying on leaves the fourth panel on whatever it had before,
        which is how a composite ends up built from one stale calibration
        and still looks entirely plausible.
        """
        start = ""
        try:
            start = browse_start_dir(self._loader.current_path() or "")
        except Exception:
            pass
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Pick ONE panel's calibration — the rest are matched by name",
            start, "Calibration (*.json *.poni *.txt);;All (*)")
        if not path:
            return
        picked = hydra.panel_for_file(path)
        if picked is None:
            QtWidgets.QMessageBox.warning(
                self, "Which panel is this?",
                f"{Path(path).name}\n\nhas no 'ge1'–'ge4' in its path, so it "
                f"cannot be matched to a panel or used to name the other "
                f"three. Load it from that panel's own card instead.")
            return
        wanted = sorted(self._loader.siblings() or {}) or [1, 2, 3, 4]
        found, missing = {}, []
        for n in wanted:
            cand = path if n == picked else hydra.sibling_for_panel(path, n)
            if cand and Path(cand).exists():
                found[n] = cand
            else:
                missing.append((n, cand))
        for n, p in sorted(found.items()):
            card = self._cards.get(n)
            if card is not None:
                card.set_calibration_file(p)
        self._load4_lbl.setText(
            f"from ge{picked}: loaded " + ", ".join(f"ge{n}" for n in sorted(found))
            + (f"  ·  missing " + ", ".join(f"ge{n}" for n, _ in missing)
               if missing else ""))
        self._log.append(
            f"[hydra batch] calibration from ge{picked}: loaded "
            + ", ".join(f"ge{n}" for n in sorted(found))
            + (f"; no file for " + ", ".join(f"ge{n}" for n, _ in missing)
               if missing else ""))
        if missing:
            lines = "\n".join(
                f"  ge{n}:  {c or '(no ge token to rewrite)'}" for n, c in missing)
            QtWidgets.QMessageBox.warning(
                self, "Some panel calibrations were not found",
                f"Picked ge{picked}:\n  {path}\n\n"
                f"Loaded: {', '.join(f'ge{n}' for n in sorted(found)) or 'none'}\n\n"
                f"These were expected beside it and are not there:\n{lines}\n\n"
                f"Those panels keep whatever calibration they already had — "
                f"check them before building the composite or running.")

    def _refresh_cake_summary(self) -> None:
        """Re-read all four cards into the summary table.

        Pulled from the cards on every refresh rather than pushed on change:
        the table has no state of its own to drift, and four spin boxes'
        worth of signals into one read is cheaper to reason about than four
        incremental updates.
        """
        cakes = {n: card.caking() for n, card in sorted(self._cards.items())}
        on_canvas = set(self._composite_panels or ())
        missing = {n for n in cakes if on_canvas and n not in on_canvas}
        self._cake_summary.refresh(cakes, missing=missing)

    def _on_panel_caking_changed(self, n: int) -> None:
        """One panel's caking moved: redraw whichever view is showing it.

        The composite draws all four regions at once, so it is stale after a
        change to ANY panel -- including one that is not the selected panel.
        """
        key = self._toolbar.current()
        if key and self._panel_num(key) is None:
            self._refresh_cake_summary()
            self._refresh_detector_preview(None)
        else:
            self._refresh_detector_preview(n)

    def _on_panel_calibration_changed(self, n: int) -> None:
        """One panel's calibration source changed: its own overlay is stale,
        and so is any cached composite that placed it."""
        self.invalidate_composite()
        self._refresh_detector_preview(n)
        if self._toolbar.current() == "composite":
            self._refresh_detector_preview(None)

    # ── Per-panel cake_parameters CSV (mpe_wf interchange) ──────────
    def _cake_csv_path(self, n: int):
        """``<analysis root>/cake_parameters.<beamline>.ge{n}.csv`` — the
        exact name ``run_midas_for_cakes_gui.sh`` looks up under
        ``<expid>_bc/``, so a file written here is the one the workflow
        reads, and vice versa.

        A *suggestion* for the dialog, nothing more. mpe_wf's own editor
        writes this path silently, but it runs as the beamline user and we
        do not; showing the path in a dialog first means nobody creates a
        directory in a shared tree by accident. Mirrors
        ``BatchTab._suggest_cake_csv_path``, with ge{n} in place of the
        positional detector — on Hydra the panel IS the detector.
        """
        siblings = self._loader.siblings() or {}
        rep = next((siblings[k] for k in sorted(siblings) if siblings.get(k)), None)
        expid = self._expid_provider().strip() if self._expid_provider else ""
        # mpe_wf spells the beamline with neither dashes nor case: 1-ID-E is
        # "1ide" in every filename it writes.
        beamline = settings.active_profile().lower().replace("-", "")
        candidates = [suggest_working_dir(rep, expid_fallback=expid) if rep else None,
                      browse_start_dir(self._out_ed.text(), fallback="") or None,
                      Path(rep).parent if rep else None]
        directory = next((Path(c) for c in candidates if c and Path(c).is_dir()),
                         Path.home())
        return directory / f"cake_parameters.{beamline}.ge{n}.csv"

    def _load_cake_csv(self, n: int) -> None:
        card = self._cards.get(n)
        if card is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load ge{n} cake parameters", str(self._cake_csv_path(n)),
            "CSV (*.csv);;All (*)")
        if not path:
            return
        values = cake_params.parse_cake_csv(path)
        if not values:
            self._log.append(f"[hydra batch] ge{n}: no usable values in {path}")
            return
        card.set_caking(values)
        # After set_caking, never before: the spin boxes' valueChanged marks
        # the card "edited since", and setting the source clears that.
        card.set_cake_source(Path(path).name)
        self._load_sibling_cake_csvs(n, path)
        # OME_* describe the rotation, which this page has no controls for
        # (its loader takes a frame RANGE, not a combine count). Saying so
        # beats dropping them silently -- the file carries them and the user
        # may be expecting them to land somewhere.
        # OME_START/OME_STEP now have somewhere to land -- the shared
        # rotation row -- so they are APPLIED rather than reported and
        # dropped. OME_SUM still has none: this loader takes a frame range,
        # not a combine count, so it is carried through on save but cannot
        # be honoured.
        applied, ignored = [], []
        for key, spin in (("OME_START", self._ome_start),
                          ("OME_STEP", self._ome_step)):
            v = values.get(key)
            if v is not None:
                spin.setValue(float(v))
                applied.append(f"{key}={float(v):g}")
        self._cake_ome_sum = values.get("OME_SUM")
        if values.get("OME_SUM") is not None:
            # It has somewhere to land now -- the loader's combine spin --
            # so it is applied rather than reported as unhonoured.
            self._loader.set_combine_chunk(values["OME_SUM"])
            applied.append(f"OME_SUM={float(values['OME_SUM']):g}")
        self._refresh_omega_note()
        note = ""
        if applied:
            note += f"  (rotation: {', '.join(applied)})"
        if ignored:
            note += (f"  (ignored {', '.join(ignored)} — this page takes a "
                     f"frame range, not a combine count)")
        self._log.append(f"[hydra batch] ge{n}: caking loaded from "
                         f"{Path(path).name}{note}")

    def _load_sibling_cake_csvs(self, n: int, path) -> None:
        """Load the other panels from the files beside the one just picked.

        mpe_wf names these ``cake_parameters.<beamline>.ge{n}.csv``, so the
        file the user designated names its siblings too -- picking ge4's is
        enough to find ge1/ge2/ge3. Asked for at 1-ID-E: "the gui should be
        able to automatically find these files if user designates one."

        Only panels whose file actually exists and parses are touched; the
        rest keep whatever they had, and the log says which were found so a
        half-populated set is never silent.
        """
        p = Path(path)
        token = f"ge{n}"
        if token not in p.name:
            self._log.append(
                f"[hydra batch] ge{n}: '{p.name}' does not contain '{token}', "
                f"so the other panels' files cannot be derived from it — "
                f"load them individually.")
            return
        found, missing = [], []
        for m in sorted(self._cards):
            if m == n:
                continue
            cand = p.with_name(p.name.replace(token, f"ge{m}"))
            vals = cake_params.parse_cake_csv(str(cand)) if cand.is_file() else None
            if not vals:
                missing.append(m)
                continue
            card = self._cards[m]
            card.set_caking(vals)
            card.set_cake_source(cand.name)
            self.invalidate_composite()
            found.append(m)
        if found:
            self._log.append(
                f"[hydra batch] also loaded caking for "
                f"{', '.join(f'ge{m}' for m in found)} from the same folder.")
        if missing:
            self._log.append(
                f"[hydra batch] no usable cake file beside it for "
                f"{', '.join(f'ge{m}' for m in missing)} — left unchanged.")

    def _save_cake_csv(self, n: int) -> None:
        card = self._cards.get(n)
        if card is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Save ge{n} cake parameters", str(self._cake_csv_path(n)),
            "CSV (*.csv);;All (*)")
        if not path:
            return
        try:
            # The six caking fields PLUS the rotation. write_cake_csv fills
            # anything absent with 0, which is what mpe_wf's reader needs (a
            # blank cell is a ValueError there) -- but writing 0 for a
            # rotation the user has set would hand the workflow back a file
            # saying the stage never moved. OME_SUM is carried through from
            # whatever was loaded, since this page cannot produce one.
            values = dict(card.caking())
            values["OME_START"] = self._ome_start.value()
            values["OME_STEP"] = self._ome_step.value()
            values["OME_SUM"] = self._loader.combine_chunk()
            cake_params.write_cake_csv(path, values)
        except Exception as exc:
            self._log.append(f"[hydra batch] ge{n}: could not write {path}: {exc}")
            return
        card.set_cake_source(Path(path).name, verb="saved to")
        self._log.append(f"[hydra batch] ge{n}: caking saved to {Path(path).name}")

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
        # Which panels are ON the canvas, in draw order -- the caking
        # overlay must not draw a region for a panel the compositor
        # skipped for want of a calibration.
        self._composite_panels = sorted(active)
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
        self._refresh_cake_summary()     # _composite_panels is known now
        self._draw_composite_caking(getattr(self._det_view, "_data", None))
        # All four beam centres register onto ONE canvas point, and for a
        # registered composite that point IS the canvas centre -- the same
        # rule hydra_page._reseed_composite_card_if_needed seeds its
        # composite card with (BigDetSize/2, in built pixels). So the lab
        # frame is well defined here; this used to clear the axes on the
        # grounds that no single panel's orientation applies, which threw
        # away the one view where all four panels share a frame and the lab
        # axes mean the most.
        bc = self._composite_axes_bc(getattr(self._det_view, "_data", None))
        if bc is None:
            self._clear_lab_axes()
        else:
            # tx=0 and not a panel's roll: the compositor's remap already
            # takes each panel's roll out, which is what makes one compass
            # legal for all four here.
            self._redraw_lab_axes_if_on(*bc, tx=0.0)

    def _draw_composite_caking(self, img) -> None:
        """Every panel's caking region, in that panel's colour, on one canvas.

        This is the view the per-panel ones cannot give you: what the four
        regions cover BETWEEN them, in one frame, rather than as four pairs
        of spin boxes to hold in your head. Whether they meet, overlap or
        leave a wedge is then something you can see -- which is a question
        about the experiment, not about file integrity: a non-tiling set
        merges and integrates perfectly well (see
        hydra_zarr_combine.eta_tiling_problem).

        The geometry is shared, which is what makes one overlay frame legal
        here: the compositor registers every beam centre onto the canvas
        centre, remaps every panel at the same pixel pitch, and takes the
        in-plane rotation tx out. So a panel's R in its own pixels is just
        R / COMPOSITE_DISPLAY_STEP in built pixels -- the same scaling
        hydra_page._reseed_composite_card_if_needed applies to the composite
        card's own geometry. Eta needs no mapping at all: the canvas IS the
        lab frame the eta convention is defined in, which is why the lab
        axes drawn just below this land where they do.

        Drawn UNTILTED on purpose. The remap removes tx but not ty/tz, and
        four panels do not share one tilt to forward-project rings through,
        so there is no single tilt-aware answer to give. This is the
        registration preview _build_composite_if_needed already says it is;
        the per-panel views remain the authority on where a bin edge truly
        falls.
        """
        bc = self._composite_axes_bc(img)
        if bc is None:
            self._clear_bin_overlay()
            return
        bc_y, bc_z = bc
        step = float(max(1, int(hydra.COMPOSITE_DISPLAY_STEP)))
        drawn = False
        for n in self._composite_panels:
            card = self._cards.get(n)
            if card is None:
                continue
            cake = card.caking()
            if not cake["R_MAX"]:
                # 0 = "auto, resolve per panel from its own corner". That
                # answer is a per-panel pixel count and the composite is not
                # the place to invent one; the panel's own view fills it in.
                continue
            draw_polar_bin_overlay(
                self._det_view, self._bin_overlay_items,
                bc_y=bc_y, bc_z=bc_z,
                r_min=cake["R_MIN"] / step, r_max=cake["R_MAX"] / step,
                r_bin=max(cake["R_STEP"] / step, 1e-9),
                e_bin=cake["ETA_STEP"],
                eta_min=cake["ETA_MIN"], eta_max=cake["ETA_MAX"],
                show_grid=self._grid_chk.isChecked(), color=panel_color(n),
                # Only the FIRST clears: the four regions stack into the one
                # items list that the next call still tears down whole.
                clear=not drawn)
            drawn = True
        if not drawn:
            self._clear_bin_overlay()

    def _clear_bin_overlay(self) -> None:
        """Tear down the bin overlay via the drawing call itself.

        ``r_max=0`` returns before drawing anything but AFTER the clear, so
        the items list is emptied by the same code path that fills it --
        there is no second place that has to know what was added.
        """
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
                frame = self._preview_frame(path)
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
        cake = card.caking()
        if cake["R_MAX"] == 0.0:          # 0 = auto, resolved per panel
            card.set_caking({"R_MAX": rmax_corner_px(
                fields["BC_y"], fields["BC_z"],
                fields["NrPixelsY"], fields["NrPixelsZ"])})
            cake = card.caking()
        draw_polar_bin_overlay(
            self._det_view, self._bin_overlay_items,
            bc_y=fields["BC_y"], bc_z=fields["BC_z"],
            r_min=cake["R_MIN"], r_max=cake["R_MAX"],
            r_bin=cake["R_STEP"], e_bin=cake["ETA_STEP"],
            eta_min=cake["ETA_MIN"], eta_max=cake["ETA_MAX"],
            show_grid=self._grid_chk.isChecked(), color=panel_color(n),
            tx=fields.get("tx") or 0.0, ty=fields.get("ty") or 0.0,
            tz=fields.get("tz") or 0.0, lsd_um=fields.get("Lsd"),
            pxY_um=fields.get("pxY"), pxZ_um=fields.get("pxZ"))
        self._redraw_lab_axes_if_on(fields["BC_y"], fields["BC_z"],
                                    tx=fields.get("tx") or 0.0)

    # ── Run ────────────────────────────────────────────────────────

    def _populate_omega_channels(self) -> None:
        """Offer the anchor file's 1-D datasets as ω sources.

        Whatever is typed survives the rebuild: a channel named by hand, or
        one from a file that is not loaded right now, is still a legal
        answer -- the list is a convenience.
        """
        typed = self._ome_channel.currentText()
        self._ome_channel.blockSignals(True)
        try:
            self._ome_channel.clear()
            self._ome_channel.addItem("")      # blank = use the ramp
            path = self._loader.current_path()
            if path and is_h5(path):
                try:
                    for name, _n in list_h5_1d_datasets(path):
                        self._ome_channel.addItem(name)
                except Exception:
                    pass                        # unreadable: typing still works
            self._ome_channel.setCurrentText(typed)
        finally:
            self._ome_channel.blockSignals(False)
        self._refresh_omega_note()

    def _refresh_omega_note(self) -> None:
        """Say what ω the run will record — all-zero is the quiet failure."""
        chunk = self._loader.combine_chunk()
        per = (f"{chunk} raw sub-frames per output frame"
               if chunk and chunk != 1 else
               "every raw sub-frame" if chunk == 1 else
               "every sub-frame of a file combined into one")
        if self._ome_collapse.isChecked():
            self._ome_lbl.setText(
                "one ω for ALL frames (the middle of the run) — this "
                "overrides the ω channel and the ramp.")
            return
        chan = self._ome_channel.currentText().strip()
        if chan:
            self._ome_lbl.setText(
                f"ω read per frame from '{chan}', averaged over {per}.")
            return
        start, step = self._ome_start.value(), self._ome_step.value()
        if not start and not step:
            self._ome_lbl.setText(
                "ω 0° on every frame — set ω start / ω step for a rotation "
                "scan, or the zarr's /Omegas is all zeros.")
        else:
            self._ome_lbl.setText(
                f"ω {start:g}° at raw sub-frame 0, {step:g}° per sub-frame"
                f"  ·  {per}")

    def omega_cfg(self) -> dict:
        """The rotation, in BatchWorker's vocabulary. Shared by all panels."""
        return {"start": self._ome_start.value(),
                "step": self._ome_step.value(),
                "channel": self._ome_channel.currentText().strip(),
                "collapse": self._ome_collapse.isChecked()}

    #: unit -> the (min, max, step) spin trio that configures it. Mirrors
    #: ``tab_batch.BatchTab._REBIN_SPINS`` so the two pages cannot drift
    #: apart on what a unit means.
    _REBIN_SPINS = {
        "Q":   lambda s: (s._q_min, s._q_max, s._q_bin),
        "2th": lambda s: (s._tth_min, s._tth_max, s._tth_bin),
    }

    def _rebin_unit(self):
        """The OUTPUT rebin unit the Bin type dropdown selects — ``"Q"``,
        ``"2th"``, or ``None`` for plain Radial.

        "Radial" means no rebin at all: the profile stays on the R-uniform
        grid the kernels integrated onto. The other two rebin that result
        (``workers.rebin_R_to_grid``); neither changes the underlying
        integration grid, which is always R-uniform. Shared across the four
        panels, like everything else on the recipe card."""
        unit = self._bin_type.currentData()
        return unit if unit in REBIN_UNITS else None

    def _rebin_cfg(self):
        """This page's output-rebin config for the workers, or ``None``.

        The self-describing ``{"unit", "min", "max", "step"}`` form that
        ``workers.rebin_cfg_parts`` reads — NOT the legacy Q-only
        ``QMin``/``QMax``/``QBinSize`` spelling this page used to emit.
        Both are still accepted by the workers, but only the new one can
        express 2theta."""
        unit = self._rebin_unit()
        if unit is None:
            return None
        lo, hi, step = self._REBIN_SPINS[unit](self)
        return {"unit": unit, "min": lo.value(), "max": hi.value(),
                "step": step.value()}

    def _rebin_label(self) -> str:
        """Display name of the selected rebin unit, for user-facing text."""
        return {None: "Radial", "Q": "Q", "2th": "2θ"}[self._rebin_unit()]

    def _on_bin_type_changed(self, *_args) -> None:
        """Enable only the trio that the selected unit actually uses.

        Both trios stay VISIBLE rather than one being hidden: hiding a form
        row relayouts the card and makes it jump height as you change unit,
        and a greyed row still tells you what the other unit would ask for.
        """
        unit = self._rebin_unit()
        for u, pick in self._REBIN_SPINS.items():
            for w in pick(self):
                w.setEnabled(u == unit)

    def _preview_frame(self, path):
        """The Detector-view preview image for ``path`` — one frame, or the
        sum of the first N from the current index when "Preview: sum first"
        asks for more.

        Display aid only; it never touches the real batch run, exactly as
        on the single-detector tab. A single weak frame can be dominated by
        detector-readout artifacts that hide the diffraction pattern, and
        summing a handful boosts the real signal enough to see it.

        Read synchronously, unlike the single-detector tab, which pushes
        this onto a StreamPreviewWorker. That page's loader owns a preview
        pipeline with dark/bright correction to apply; this one already
        reads its single frame straight off disk on every panel switch
        (see _refresh_detector_preview), so summing N just does that N
        times. Keep N small over NFS -- each frame is a separate read.
        """
        ds, i0 = self._loader.dataset(), self._loader.frame_index()
        n = max(1, int(self._preview_sum_n.value()))
        total = None
        for i in range(i0, i0 + n):
            try:
                f = _load_image(path, ds, i)
            except Exception:
                break          # ran off the end of the stack; sum what we got
            total = np.asarray(f, dtype=np.float64) if total is None else total + f
        if total is None:      # not even the first frame read -- let the
            return _load_image(path, ds, i0)   # caller's handler report it
        return total

    def _on_preview_sum_changed(self, _n: int) -> None:
        self._refresh_active_detector_preview()

    def _ion_csv_extras(self) -> tuple:
        """Optional beam-monitor CSV column groups, as the tuple
        ``ion_csv.write_ion_csv`` expects — same accessor, same vocabulary
        as ``tab_batch.BatchTab._ion_csv_extras``. Collapsed into ONE value
        rather than a kwarg per checkbox: each extra kwarg has to be
        threaded through the in-process run AND the as-job argv."""
        return tuple(k for k, chk in (("env", self._ion_env_chk),
                                      ("motors", self._ion_motors_chk))
                     if chk.isChecked())

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
        cake = tuple(round(v, 4) for v in card.caking().values())
        return (calib, kernel, cake,
                bool(weighted), pol is not None, sa is not None, mask_id,
                src_cfg.get("path"), src_cfg.get("type"), src_cfg.get("dataset"))

    def _cache_geom(self, n, sig, ctx):
        self._geom_cache[n] = ctx
        self._geom_sig[n] = sig

    def _run_all(self):
        scans = self._scan_plan()
        if not scans:
            QtWidgets.QMessageBox.warning(self, "No data", "Load Hydra panel data first.")
            return
        if self._workers:
            return
        # Same guard the single-detector tab applies: the two are not
        # supported together yet, and discovering that per panel mid-run
        # (or, worse, in a detached job's log) is no way to find out.
        if self._multi_azimuth_chk.isChecked() and self._rebin_cfg():
            QtWidgets.QMessageBox.warning(
                self, "Incompatible options",
                f"Multi-azimuth output isn't supported together with "
                f"{self._rebin_label()}-uniform bins yet. Uncheck "
                f"Multi-azimuth output, or set Bin type back to Radial.")
            return
        self._orphans = [o for o in self._orphans if o.isRunning()]
        self._run_cancelled = False
        self._run_btn.setEnabled(False); self._abort_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._prog.setVisible(True)
        # One queue entry per scan. A single-scan pick makes a queue of one
        # and takes the identical path, which is what keeps every existing
        # run byte-for-byte what it was.
        self._scan_queue = [(panels, out_dir) for out_dir, panels in scans]
        self._scan_total = len(self._scan_queue)
        self._scan_pos = 0
        if self._scan_total > 1:
            self._log.append(
                "─" * 40 + f"\n{self._scan_total} folders selected — each is "
                f"integrated as one scan, to its own output directory:")
            for panels, out_dir in self._scan_queue:
                src = Path(panels[sorted(panels)[0]][0]).parent
                n_f = sum(len(v) for v in panels.values())
                self._log.append(
                    f"    {src}  ({n_f} files, {len(panels)} panels)  →  {out_dir}")
        self._start_next_scan()

    def _run_as_job(self):
        """Launch this same Hydra integration as a detached background
        `screen` job (see job_queue.JobQueuePanel) instead of running
        in-process — survives closing this GUI.

        The single-detector counterpart is ``tab_batch.BatchTab._run_as_job``
        and this deliberately mirrors it: same button, same queue panel, same
        one-session-many-steps mechanism. The only structural difference is
        what a step is. There, a selection spanning source folders becomes one
        ``batch_cli`` invocation per folder; here each scan additionally
        becomes one invocation PER PANEL (``batch_cli`` integrates a single
        detector and knows nothing about Hydra) followed by one
        ``hydra_combine_cli`` call that merges that scan's four archives::

            scan 1:  batch_cli --out-dir <scan1>/ge1  (ge1 calib, ge1 caking)
                     batch_cli --out-dir <scan1>/ge2
                     batch_cli --out-dir <scan1>/ge3
                     batch_cli --out-dir <scan1>/ge4
                     hydra_combine_cli --out-dir <scan1>
            scan 2:  ...

        all chained into ONE session. Panels run sequentially within a scan,
        which is what makes this work with no new concurrency machinery —
        and the in-process "Panels: parallel" mode stays available for
        interactive runs.

        Everything the detached process needs is written to disk first: it
        has no access to this GUI's live state, and Hydra needs four of most
        things (calibration, mask, dark/bright) rather than one.
        """
        scans = self._scan_plan()
        if not scans:
            QtWidgets.QMessageBox.warning(
                self, "No data", "Load Hydra panel data first.")
            return
        panels = sorted({n for _d, pl in scans for n in pl})
        if not panels:
            QtWidgets.QMessageBox.warning(
                self, "No data", "No panel has any files selected.")
            return

        fmts = self._fmt.checked_keys()
        if not fmts:
            QtWidgets.QMessageBox.warning(
                self, "No format", "Check at least one output format first.")
            return
        # Same guard the single-detector tab applies: the two are not
        # supported together yet, and discovering that per panel mid-run
        # (or, worse, in a detached job's log) is no way to find out.
        if self._multi_azimuth_chk.isChecked() and self._rebin_cfg():
            QtWidgets.QMessageBox.warning(
                self, "Incompatible options",
                f"Multi-azimuth output isn't supported together with "
                f"{self._rebin_label()}-uniform bins yet. Uncheck "
                f"Multi-azimuth output, or set Bin type back to Radial.")
            return

        # Resolve every panel's calibration up front — a detached job that
        # dies on frame 1 fails where nobody is watching for it.
        #
        # A panel that cannot resolve is SKIPPED, not fatal, because that is
        # what the in-process run does (_start_panel_worker -> _skip_panel):
        # loading one panel's file pulls in all four siblings, so three
        # uncalibrated panels is the ordinary state of a two-panel run, and
        # refusing the whole job for it would make the two paths disagree
        # about the same setup. Only "no panel at all" is an error.
        skipped = []
        usable = []
        for n in panels:
            try:
                self._cards[n].resolved_spec()
            except Exception as e:
                skipped.append((n, str(e)))
            else:
                usable.append(n)
        if not usable:
            QtWidgets.QMessageBox.critical(
                self, "Calibration error",
                "No panel has a usable calibration:\n\n"
                + "\n".join(f"ge{n}: {msg}" for n, msg in skipped))
            return
        panels = usable

        missing = [d for d, _pl in scans if not d]
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Output folder required",
                "Background jobs need an Output folder — the job writes its "
                "calibration snapshots and results there.")
            return

        # Shared sidecars live at the common root of the scan directories:
        # for one scan that is the scan's own folder (where the
        # single-detector tab puts them), for many it is the tree they fan
        # out beneath. Deliberately NOT the Output box, which may hold a
        # stale autofill equal to one group's own directory (see _scan_plan)
        # and would then bury the shared files inside the first load step.
        dirs = [str(d) for d, _pl in scans]
        root = Path(os.path.commonpath(dirs) if len(dirs) > 1 else dirs[0])
        reason = check_output_dir_writable(str(root))
        if reason:
            QtWidgets.QMessageBox.critical(
                self, "Output folder not writable", reason)
            return
        root.mkdir(parents=True, exist_ok=True)

        import tifffile
        from midas_gui.helpers import write_standalone_paramstest

        # ── per-panel argv prefixes (everything except source and out-dir) ──
        base: dict = {}
        for n in panels:
            card = self._cards[n]
            if card.using_file():
                calib_path = card.file_path()
                if not calib_path or not Path(calib_path).exists():
                    QtWidgets.QMessageBox.critical(
                        self, "Calibration error",
                        f"ge{n}: calibration file not found: {calib_path}")
                    return
            else:
                if card.result is None:
                    QtWidgets.QMessageBox.critical(
                        self, "Calibration error",
                        f"ge{n}: no calibration from the Calibrate tab. "
                        "Run its Hydra fit first.")
                    return
                calib_path = root / f"_bg_job_calibration_ge{n}.txt"
                write_standalone_paramstest(card.result, calib_path)

            c = card.caking()
            a = [sys.executable, "-m", "midas_gui.batch_cli",
                 "--calib-file", str(calib_path),
                 "--r-bin", str(c["R_STEP"]), "--eta-bin", str(c["ETA_STEP"])]
            if c["R_MIN"]:
                a += ["--r-min", str(c["R_MIN"])]
            if c["R_MAX"]:            # 0.0 is the "auto" sentinel, not a radius
                a += ["--r-max", str(c["R_MAX"])]
            # Always sent, unlike R: a panel's wedge is the whole point of
            # Hydra, and an omitted flag means the full circle rather than
            # "unchanged" — ge1's range even straddles the ±180 branch cut.
            a += ["--eta-min", str(c["ETA_MIN"]), "--eta-max", str(c["ETA_MAX"])]
            a += ["--fmts", ",".join(fmts), "--kernel", self._kernel.currentData()]
            a += self._shared_job_args()

            mask = self._loader.composite_mask(n)
            if mask is not None:
                p = root / f"_bg_job_mask_ge{n}.tif"
                tifffile.imwrite(str(p), mask.astype("uint8"))
                a += ["--mask", str(p)]
            # Bright and background are properties of the RUN, so one per
            # panel is right. The dark is not: it belongs to the scan, and
            # is attached per (scan, panel) in the units loop below.
            for name, arr in (("bright", self._loader.bright(n)),
                              ("background", self._loader.background(n))):
                if arr is not None:
                    p = root / f"_bg_job_{name}_ge{n}.tif"
                    tifffile.imwrite(str(p), np.asarray(arr, dtype=np.float32))
                    a += [f"--{name}", str(p)]
            if self._loader.dark(n) is not None or self._loader.bright(n) is not None:
                a += ["--bright-mode", self._loader.bright_mode()]
            base[n] = a

        # ── one step per (scan, panel), plus a combine per scan ──
        combine = (getattr(self, "_combine_zarr_chk", None) is not None
                   and self._combine_zarr_chk.isChecked()
                   and "zarr" in fmts)
        # One UNIT per scan: its panels run concurrently, then its combine.
        # That is the real dependency shape -- four independent panels, and
        # a merge that needs all four -- so the launcher can overlap them
        # without any ordering risk (job_queue._parallel_script).
        # The dark is resolved HERE, at curation, rather than inside the
        # job. The GUI already holds the policy code and already emits one
        # batch_cli invocation per (scan, panel), so teaching batch_cli to
        # resolve darks would put dark_policy in two places that can
        # drift -- which is the bug this whole series fixed. Resolving now
        # also means an unresolvable dark is reported at SUBMIT time
        # instead of hours later in a screenlog. See .context/ROADMAP.md,
        # "Background jobs and the dark".
        dark_cache = DarkArgBuilder(root)

        units, argvs = [], []
        for out_dir, panel_paths in scans:
            ran, par = [], []
            for n in sorted(panel_paths):
                paths = list(panel_paths.get(n) or [])
                if not paths or n not in base:
                    continue
                darg = dark_cache.arg_for(
                    paths[0], self._loader.dark_request(n), tag=f"ge{n}")
                if darg is None:
                    continue          # no valid dark — this panel is skipped
                par.append(base[n] + darg + self._source_job_args(n, paths)
                           + ["--out-dir", str(Path(out_dir) / f"ge{n}")])
                ran.append(n)
            if not par:
                continue
            tail = []
            # Only the panels this scan actually integrates. Naming a skipped
            # panel would make find_panel_zarr_sets look for an archive that
            # is never written, and every frame would then be "incomplete"
            # and silently merge nothing.
            if combine and len(ran) >= 2:
                tail.append([sys.executable, "-m", "midas_gui.hydra_combine_cli",
                             "--out-dir", str(out_dir),
                             "--panels", *[f"ge{n}" for n in ran]])
            units.append({"parallel": par, "then": tail})
            argvs.extend(par + tail)
        if not argvs:
            QtWidgets.QMessageBox.warning(
                self, "No data", "No panel of any selected scan has files.")
            return

        # Progress-bar denominator only. JobQueuePanel reads the LAST
        # PROGRESS line's numerator, which resets at each step, so the
        # meaningful total is ONE panel-step's frame count — the same choice
        # the single-detector tab makes for a multi-folder job.
        total = 1
        try:
            from midas_gui.workers import _open_source_cfg
            first_n = sorted(scans[0][1])[0]
            total = int(_open_source_cfg(
                self._source_cfg_for_paths(first_n, scans[0][1][first_n])
            ).n_frames) or 1
        except Exception:
            pass

        # Before launching, not after: the whole reason the dark is
        # resolved at curation is that an unresolvable one should cost a
        # dialog now rather than a surprise in a screenlog in three hours.
        if dark_cache.skipped and not self._confirm_dark_skips(
                dark_cache.skipped):
            return

        n_par = max(1, int(self._job_scans.value()))
        job = self._job_queue.launch(
            argvs[0], name=root.name or "hydra", total_frames=total,
            out_dir=str(Path(scans[0][0]) / f"ge{sorted(scans[0][1])[0]}"),
            units=units, max_parallel=n_par)
        if job is None:
            return
        self._log.append(
            f"[hydra] Launched background job: {job.session}\n"
            f"[hydra]   reattach with:  screen -r {job.session}")
        self._log.append(
            f"[hydra] {len(units)} scan(s) x {len(panels)} panel(s)"
            + (" + combine" if combine else "")
            + f" = {len(argvs)} step(s) in that one session.")
        self._log.append(
            f"[hydra] {n_par} scan(s) at a time, panels within a scan all at "
            f"once — up to ~{n_par * max(1, len(panels))} processes. Each "
            f"scan's combine waits for its own four panels, not for the "
            f"whole run.")
        for n, msg in skipped:
            # Same outcome as an in-process run's _skip_panel, and said
            # before the job starts rather than discovered in the screenlog.
            self._log.append(f"[hydra] ge{n} skipped — {msg}")
        self._log.append(f"[hydra] shared sidecars (calibration/mask/fields): {root}")
        for out_dir, panel_paths in scans[:10]:
            n_f = sum(len(v) for v in panel_paths.values())
            self._log.append(
                f"[hydra]     {n_f:>5} file(s), {len(panel_paths)} panel(s) → {out_dir}")
        if len(scans) > 10:
            self._log.append(f"[hydra]     … and {len(scans) - 10} more")
        self._write_job_command_log(job, argvs, scans, panels, skipped, root)

    def _write_job_command_log(self, job, argvs, scans, panels, skipped, root) -> None:
        """Record WHAT WAS RUN, in the two places someone will look for it.

        ``JobQueuePanel.launch`` already writes the chain to
        ``~/.midas_gui/jobs/<session>.sh`` and its output to
        ``<session>.screenlog``, but both are keyed by session name in a
        directory nobody browses — answering "what did I actually run on this
        dataset?" months later means knowing the session name first. So the
        same command list also goes to:

        * the run-log tree (``~/midas_runs/midas_screen_logs/<beamline>/
          <expid>/``), which is where this GUI's other long runs are filed
          and is organised by experiment rather than by session; and
        * ``<root>/logs/``, beside the data, which is the convention the MPE
          SAXS/WAXS workflow established (a scan folder ends up as
          ``xy/ zarr/ mat/ logs/``).

        Best effort throughout — ``run_log.open_log`` returns a discarding
        log rather than raising, and the ``logs/`` copy is wrapped, because a
        job that has already been launched must not be reported as failed
        over a log file.
        """
        import shlex
        from midas_gui import run_log

        lines = [f"[{i}/{len(argvs)}] " + " ".join(shlex.quote(str(c)) for c in a)
                 for i, a in enumerate(argvs, 1)]
        fields = {
            "Session": job.session,
            "Reattach": f"screen -r {job.session}",
            "Scans": len(scans),
            "Panels": ", ".join(f"ge{n}" for n in panels) or "(none)",
            "Skipped": ", ".join(f"ge{n}" for n, _ in skipped) or "(none)",
            "Steps": len(argvs),
            "Output root": str(root),
            "Job script": f"~/.midas_gui/jobs/{job.session}.sh",
            "Job stdout": f"~/.midas_gui/jobs/{job.session}.screenlog",
        }
        title = "MIDAS GUI — Hydra Batch Integrate (background job)"

        expid = self._expid_provider().strip() if self._expid_provider else ""
        log = run_log.open_log("hydra_integrate", profile=settings.active_profile(),
                               expid=expid, stem=root.name)
        if log.path is None:
            self._log.append("[hydra] Note: no run log written — "
                             + (log.error or "could not open it"))
        else:
            log.header(title, fields)
            for line in lines:
                log.write(line)
            log.close()
            self._log.append(f"[hydra] Run log: {log.path}")
            if not expid:
                self._log.append(f"[hydra] Note: Exp ID is blank, so this run's "
                                 f"log is filed under {run_log.NO_EXPID}/.")

        try:
            logs_dir = Path(root) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            beside = logs_dir / f"{job.session}.commands.log"
            with open(beside, "w", encoding="utf-8") as fh:
                fh.write("=" * 40 + f"\n{title}\n" + "=" * 40 + "\n")
                for k, v in fields.items():
                    fh.write(f"{k:<13}: {v}\n")
                fh.write("=" * 40 + "\n")
                fh.write("\n".join(lines) + "\n")
            self._log.append(f"[hydra] Commands: {beside}")
        except OSError as exc:
            self._log.append(f"[hydra] Note: could not write {root}/logs/ "
                             f"command log ({exc}); the job is unaffected.")

    def _confirm_dark_skips(self, skipped) -> bool:
        """Show what will NOT be integrated, and let the user stop.

        Returns True to go ahead with the rest. A skip is the deliberate
        outcome when no valid dark exists -- integrating with another
        scan's dark produces a plausible and wrong answer -- but
        submitting a job that quietly covers less than the selection
        would be its own silent failure.
        """
        from midas_gui import settings
        from midas_gui.mail import dark_skip_report, send_mail
        self._log.append(f"[hydra] {len(skipped)} (scan, panel) step(s) have "
                         f"no valid dark and will NOT be integrated:")
        for path, why in skipped[:20]:
            self._log.append(f"[hydra]     {Path(path).name}: {why}")
        if len(skipped) > 20:
            self._log.append(f"[hydra]     ... and {len(skipped) - 20} more.")
        try:
            addr = settings.get_notify_email()
        except Exception:
            addr = ""
        if addr:
            subject, body = dark_skip_report(skipped, context="Hydra batch job")
            if send_mail(subject, body, addr):
                self._log.append(f"[hydra] Skip report sent to {addr}.")
        from midas_gui.mail import summarise_skips
        lines, why, n_scans = summarise_skips(skipped)
        btn = QtWidgets.QMessageBox.warning(
            self, "Some scans have no valid dark",
            f"{n_scans} scan(s) will NOT be integrated:\n\n{lines}\n\n"
            f"{why}.\n\n"
            f"They are skipped rather than corrected with another scan's "
            f"dark, which would look plausible and be wrong. The full "
            f"reason for each file is in the Log.\n\n"
            f"Submit the job for the remaining scans?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes)
        return btn == QtWidgets.QMessageBox.Yes

    def _shared_job_args(self) -> list:
        """The run settings that are the same for every panel and every scan
        — corrections, rotation, binning mode, frame subsetting. Split out of
        ``_run_as_job`` only so the per-panel loop there stays readable.

        Every one of these is sent explicitly, even at its default: the
        detached process cannot see the widgets, so an omitted flag silently
        falls back to the CLI's own default and the two run paths disagree
        about the same run. That class of bug is why the single-detector tab
        sends its omega and zarr-grouping flags unconditionally too.
        """
        a = []
        rebin = self._rebin_cfg()
        if rebin:
            a += ["--rebin-unit", rebin["unit"],
                  "--rebin-min", str(rebin["min"]),
                  "--rebin-max", str(rebin["max"]),
                  "--rebin-step", str(rebin["step"])]
        a += ["--weighted"] if bool(self._azim.currentData()) else ["--no-weighted"]
        ome = self.omega_cfg()
        a += ["--ome-start", str(ome["start"]), "--ome-step", str(ome["step"])]
        if ome.get("channel"):
            a += ["--ome-channel", ome["channel"]]
        if ome.get("collapse"):
            a += ["--ome-collapse"]
        a += ["--zarr-grouping", self._zarr_grouping.currentData() or "run"]
        if self._multi_azimuth_chk.isChecked():
            a += ["--multi-azimuth"]
        # Same reasoning as the flags above: the job cannot see the
        # checkboxes, so an omitted flag would quietly drop the optional
        # columns on this path only and the two run modes would disagree.
        extras = self._ion_csv_extras()
        if extras:
            a += ["--ion-csv-extras", ",".join(extras)]
        if self._corr_widget.polar_check.isChecked():
            a += ["--polarization",
                  "--pol-fraction", str(self._corr_widget.pol_fraction.value()),
                  "--pol-plane", str(self._corr_widget.pol_plane.value())]
        if self._corr_widget.solid_check.isChecked():
            a += ["--solid-angle"]
        if self._var_check.isChecked() and not self._corr_widget.any_enabled():
            a += ["--variance", "--error-model", self._err_model.currentText()]
        # The loader's Frames start/end/stride subsets the OUTPUT frames a
        # source yields, which is batch_cli's --frame-range-* trio, not the
        # --frame-start/--frame-end file-number filter. Sent only when it is
        # not the "everything" default, so an ordinary job's command line
        # stays as short as the single-detector one.
        fr = self._loader.frame_range() or (0, None, 1)
        if tuple(fr) != (0, None, 1):
            a += ["--frame-range-start", str(int(fr[0] or 0))]
            if fr[1] is not None:
                a += ["--frame-range-end", str(int(fr[1]))]
            if (fr[2] or 1) != 1:
                a += ["--frame-range-stride", str(int(fr[2]))]
        mon = self._mon_ed.text().strip()
        if mon:
            a += ["--monitor-file", mon]
        return a

    def _source_cfg_for_paths(self, n: int, paths: list) -> dict:
        """``_source_cfg_for``'s body with the path list supplied rather than
        read from the active scan — the background path plans every scan up
        front, so it cannot go through ``_active_scan``."""
        saved = getattr(self, "_active_scan", None)
        self._active_scan = {n: list(paths)}
        try:
            return self._source_cfg_for(n)
        finally:
            self._active_scan = saved

    def _source_job_args(self, n: int, paths: list) -> list:
        """``--source-type`` and friends for one panel of one scan, derived
        from the very same ``_source_cfg_for`` the in-process run uses, so
        both paths read the identical files the identical way."""
        cfg = self._source_cfg_for_paths(n, paths)
        kind = cfg.get("type")
        if kind == "hdf5":
            a = ["--source-type", "hdf5", "--source-path", cfg["path"],
                 "--dataset", cfg.get("dataset", "frames")]
        elif kind == "hdf5_stack_glob":
            a = ["--source-type", "hdf5_stack_glob",
                 "--source-paths", *cfg["paths"],
                 "--dataset", cfg.get("dataset", "exchange/data")]
        else:
            a = ["--source-type", "tiff_list", "--source-paths", *cfg["paths"]]
        # "Combine sub-frames", baked into the cfg by the loader. chunk_size
        # None means "combine everything selected into one frame", which the
        # CLI spells 0 (see batch_cli's --chunk-size).
        chunk = cfg.get("chunk_size")
        a += ["--chunk-size", str(chunk if chunk is not None else 0),
              "--combine-op", cfg.get("combine_op") or "mean"]
        return a

    def _start_next_scan(self) -> None:
        """Begin the next queued scan, or wind the run up."""
        if self._run_cancelled or not self._scan_queue:
            self._finish_whole_run()
            return
        self._active_scan, self._active_out_base = self._scan_queue.pop(0)
        self._scan_pos += 1
        self._last_results = {}
        mode = self._run_mode()
        where = (f" [scan {self._scan_pos}/{self._scan_total}]"
                 if self._scan_total > 1 else "")
        rep = self._active_scan[sorted(self._active_scan)[0]][0]
        self._log.append(
            "─" * 40 + f"\nStarting Hydra batch integration ({mode}){where}…"
            + (f"\n  {Path(rep).parent}" if self._scan_total > 1 else ""))
        self._pending_panels = sorted(self._active_scan)
        if mode == "parallel":
            pending, self._pending_panels = list(self._pending_panels), []
            # Held until every panel has been launched. _start_panel_worker
            # can finish synchronously (a skip), and _maybe_finish_run then
            # saw no workers and no pending and declared the scan done --
            # combining, and with a queue behind it starting the NEXT scan
            # while this one's ge2/ge3/ge4 had not been started yet.
            self._launching = True
            try:
                for n in pending:
                    self._start_panel_worker(n)
            finally:
                self._launching = False
            self._maybe_finish_run()
        else:
            self._start_next_sequential()

    def _scan_plan(self) -> list:
        """``[(out_dir, {panel: [paths]}), ...]`` — one entry per SOURCE
        FOLDER, in the layout picking that folder by hand would produce.

        Delegates the hard half to ``helpers.group_paths_by_output_dir``,
        which the single-detector tab already uses for exactly this and
        which gets three things right that a hand-rolled version did not:

        * a folder is ONE batch. Grouping per ge1/ge2/ge3/ge4 quartet made
          every frame file its own scan -- thirteen output directories for
          one load step.
        * names come from ``_folder_output_dir``'s convention rather than
          from disambiguating derived names against each other, which
          produced
          ``APS_4130_C_load_step_0_APS_4130_C_load_step_0_APS_4130_C_load_step_0_004057.ge1``.
        * an Output box holding a value one of the groups already derives
          is a STALE autofill, not a chosen root, so each group uses its own
          directory. Treating it as a root is what put load_step_1's output
          inside load_step_0.
        """
        files = self._loader.selected_files()
        if not files:
            return []
        typed = self._out_ed.text().strip()
        expid = self._expid_provider().strip() if self._expid_provider else ""

        # Group by the folder of each file's ge1-NORMALISED path, not by its
        # own parent. Both Hydra layouts are real: all four panels together
        # in a scan folder (connolly_oct26's load steps) and one folder per
        # panel (.../ge1/panel.ge1.h5). Grouping on the raw parent is right
        # for the first and splits a single scan into four for the second.
        # Normalising collapses both to one key per scan.
        groups: dict = {}
        for f in files:
            norm = hydra.sibling_for_panel(f, 1) or f
            groups.setdefault(str(Path(norm).parent), []).append(f)

        if len(groups) == 1:
            # The Output box is authoritative for a single scan, as it has
            # always been: each panel appends its own ge{n} to it.
            only = next(iter(groups.values()))
            out = typed or _folder_output_dir(
                [hydra.sibling_for_panel(f, 1) or f for f in only],
                expid, suggest_panel_base_output_dir)
            return [(str(out or ""), hydra.split_by_panel(only))]

        derived = {k: _folder_output_dir(
                       [hydra.sibling_for_panel(f, 1) or f for f in v],
                       expid, suggest_panel_base_output_dir)
                   for k, v in groups.items()}
        # A box equal to one group's own derived dir is a STALE AUTOFILL,
        # not a chosen root -- reading it as a root is what nested
        # load_step_1's output inside load_step_0.
        keys = [str(d) for d in derived.values() if d is not None]
        custom = bool(typed) and typed not in keys
        plan = []
        for k in sorted(groups):
            d = derived.get(k)
            if custom and d is not None and keys:
                common = (Path(os.path.commonpath(keys)) if len(keys) > 1
                          else Path(keys[0]).parent)
                out = Path(typed) / Path(str(d)).relative_to(common)
            elif d is not None:
                out = Path(str(d))
            else:
                out = Path(typed) if typed else None
            plan.append((str(out or ""), hydra.split_by_panel(groups[k])))
        return plan

    def _source_cfg_for(self, n: int) -> dict:
        """Streaming descriptor for panel ``n`` of the scan being run.

        Falls back to the loader when no scan is active (an interactive
        preview, or anything that runs before the queue exists), so the
        loader stays the one place that knows how to read a path.
        """
        scan = getattr(self, "_active_scan", None)
        if not scan:
            return self._loader.source_cfg(n)
        paths = list(scan.get(n) or [])
        if not paths:
            return {"path": ""}
        base = self._loader.source_cfg(n)      # chunk_size / combine_op / dataset
        chunk = base.get("chunk_size")
        op = base.get("combine_op", "mean")
        if len(paths) == 1 and source_kind(paths[0]) == "hdf5":
            return {"type": "hdf5", "path": paths[0],
                    "dataset": self._loader.DATASET,
                    "chunk_size": chunk, "combine_op": op}
        if all(source_kind(p) == "hdf5" for p in paths):
            # Several HDF5 files are a stack glob, exactly as the
            # single-detector loader describes a multi-file pick -- one
            # scan point per file, each with its own sub-frame stack.
            return {"type": "hdf5_stack_glob", "paths": paths,
                    "dataset": self._loader.DATASET,
                    "chunk_size": chunk, "combine_op": op}
        return {"type": "tiff_list", "paths": paths,
                "chunk_size": chunk, "combine_op": op}

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
            spec = card.resolved_spec()   # this panel's own caking
        except Exception as e:
            self._skip_panel(n, f"calibration error: {e}")
            return
        src_cfg = self._source_cfg_for(n)
        # `paths` OR `path`: a folder's worth of frames is an
        # hdf5_stack_glob/tiff_list, which carries a LIST and no "path" at
        # all. Testing only "path" skipped every multi-file panel as
        # "no data" -- the whole point of the folder fan-out.
        if not (src_cfg.get("path") or src_cfg.get("paths")):
            self._skip_panel(n, "no data")
            return

        kernel = self._kernel.currentData()
        corrections = self._corr_widget.build_corrections()
        variance_cfg = ({"error_model": self._err_model.currentText()}
                        if self._var_check.isChecked() else None)
        if variance_cfg and self._corr_widget.any_enabled():
            variance_cfg = None
        base_out = (self._active_out_base
                    if self._active_out_base is not None
                    else self._out_ed.text().strip())
        out_dir = str(Path(base_out) / f"ge{n}") if base_out else None
        fmts = self._fmt.checked_keys()
        # Self-describing {"unit", "min", "max", "step"} now, not the
        # legacy Q-only dict -- it is the only spelling that can say 2theta.
        # workers.rebin_cfg_parts reads both, so nothing downstream changed.
        q_cfg = self._rebin_cfg()

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
        # Telling the two viewers what units to label is cosmetic, but it
        # sits on the QUEUE's critical path: this method is reached from a
        # worker's finished signal via _maybe_finish_run -> _start_next_scan,
        # so an exception raised here does not just spoil a plot -- it stops
        # the remaining panels launching and the run never advances. A
        # 189-folder batch died after ONE folder when a profile with no
        # finite points made pyqtgraph's autoRange resolve to [nan, nan]
        # (fixed in StackedProfileViewer._restack too, but the queue must
        # not depend on every view bug having been found first).
        for _view in (pair.waterfall, pair.stack_view):
            try:
                _view.set_axis_context(*_axctx)
            except Exception as exc:            # noqa: BLE001 — see above
                self._log.append(
                    f"[ge{n}] axis units not applied to the viewer ({exc}); "
                    f"integration continues")
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
            q_cfg=q_cfg, omega_cfg=self.omega_cfg(),
            frame_range=frame_range, monitor_file=monitor_file, parent=self,
            dark=dark, bright=bright, background=background, bright_mode=bright_mode,
            weighted=weighted, context=context, im_trans=card.resolved_im_trans(),
            run_mode=self._frame_run_mode.currentData(), n_workers=self._n_workers.value(),
            multi_azimuth=self._multi_azimuth_chk.isChecked(),
            ion_csv_extras=list(self._ion_csv_extras()),
            zarr_grouping=self._zarr_grouping.currentData() or "run",
            # PER PANEL. The request's search_root and explicit path both
            # carry the panel token, so ge2 looks for ge2's dark in ge2's
            # tree — applying ge1's dark to ge2 data is the one thing this
            # must never do.
            dark_request=self._loader.dark_request(n))
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
        if self._workers or self._pending_panels or getattr(self, "_launching", False):
            return
        # One scan's four panels are done: combine THIS scan, then move on.
        # The combine has to run per scan, not once at the end, because each
        # scan writes into its own folder and the merge is a property of a
        # folder's ge1..ge4.
        self._combine_panel_zarrs_if_wanted()
        if self._scan_queue and not self._run_cancelled:
            self._start_next_scan()
            return
        self._finish_whole_run()

    def _finish_whole_run(self) -> None:
        self._active_scan = None
        self._active_out_base = None
        self._run_btn.setEnabled(True); self._abort_btn.setEnabled(False)
        self._save_btn.setEnabled(bool(self._last_results))
        self._prog.setVisible(False)
        if self._scan_total > 1:
            self._log.append(
                f"Hydra batch integration complete — {self._scan_pos}/"
                f"{self._scan_total} scan(s).")
        else:
            self._log.append("Hydra batch integration run complete.")
        self._scan_total = 0
        self._scan_pos = 0

    def _combine_panel_zarrs_if_wanted(self) -> None:
        """Merge each frame's four per-panel zarrs into one, eta-concatenated.

        Runs after every panel is done, because it needs all four. Synchronous
        on the GUI thread: it is pure array concatenation over archives that
        were just written, and it only runs once per run rather than per
        frame. Any failure is reported per frame and does not stop the rest --
        a bad frame should not cost the run its other output.
        """
        if not getattr(self, "_combine_zarr_chk", None) \
                or not self._combine_zarr_chk.isChecked():
            return
        if "zarr" not in (self._fmt.checked_keys() or []):
            return
        out_dir = (self._active_out_base
                   if self._active_out_base
                   else self._out_ed.text().strip())
        if not out_dir:
            return
        panels = sorted(self._cards)
        if len(panels) < 2:
            return
        try:
            sets = hydra_zarr_combine.find_panel_zarr_sets(
                out_dir, [f"ge{n}" for n in panels])
        except Exception as exc:
            self._log.append(f"[combine] could not scan for panel zarrs: {exc}")
            return
        if not sets:
            self._log.append(
                "[combine] no frame had an archive from every panel — "
                "nothing to combine.")
            return
        self._log.append(f"[combine] merging {len(sets)} frame(s) across "
                         f"{len(panels)} panels…")
        done = 0
        last_out = None
        # Collected rather than logged per frame: the caking is one setting
        # for the whole run, so a non-tiling set says the same thing once per
        # frame and would bury the rest of the log on a long scan.
        warned = set()
        for group in sets:
            try:
                out = hydra_zarr_combine.combine_panel_zarrs(
                    group,
                    hydra_zarr_combine.default_combined_path(group, out_dir),
                    warn=lambda m: warned.add(m))
            except Exception as exc:
                self._log.append(
                    f"[combine] {Path(group[sorted(group)[0]]).name}: {exc}")
                continue
            done += 1
            if done == 1 or done == len(sets):
                self._log.append(f"[combine]   → {out}")
            last_out = out
        self._log.append(f"[combine] wrote {done}/{len(sets)} combined "
                         f"archive(s).")
        # Show the last one written. With the default "one zarr per scan"
        # grouping there is exactly one, which is the natural thing to
        # display; with per-file grouping the others are on disk and named
        # in the log above.
        if done:
            self._populate_merged_views(last_out)
        for msg in sorted(warned):
            # Deliberately a note, not a warning. The archive is correct:
            # eta travels per column in REtaMap, so an uncovered wedge is
            # absent rather than misindexed, and I vs 2θ/q/d is unaffected
            # for every eta the file does have. The only consequence is
            # that cake_params carries the per-panel ranges instead of one
            # EtaMin/EtaMax span.
            self._log.append(f"[combine] note: {msg}. Merged as usual; "
                             f"cake_params records the per-panel eta ranges "
                             f"rather than one span.")

    def _merged_axis_context(self):
        """``((lsd, px, wl), "")`` for converting the merged view's R
        axis, or ``(None, reason)`` when it cannot be offered.

        The merged archive's R rows are shared across panels (``validate``
        requires it), but 2theta at a given R depends on that panel's own
        Lsd. With the four panels at genuinely different distances there is
        no single conversion that is right for all of them, so the view
        stays in pixels rather than offering a 2theta axis that is correct
        for one quarter of the data and quietly wrong for the rest.
        """
        lsds, pxs, wls = [], [], []
        for n in sorted(self._cards):
            try:
                spec = self._cards[n].resolved_spec()
            except Exception:
                continue
            lsds.append(float(spec.Lsd)); pxs.append(float(spec.pxY))
            wls.append(float(spec.Wavelength))
        if not lsds:
            return None, "no panel calibration is loaded"
        # 0.1% of the distance -- well inside a real panel-to-panel offset
        # and well outside fit noise on the same nominal distance.
        if max(lsds) - min(lsds) > 1e-3 * max(lsds):
            return None, (f"the panels are at different distances "
                          f"({min(lsds):.0f}–{max(lsds):.0f} µm)")
        return (lsds[0], pxs[0], wls[0]), ""

    def _populate_merged_views(self, path) -> None:
        """Fill the Waterfall / Stacked profiles / Eta-R cakes tabs from a
        merged ge1234 archive. Best effort -- a view that cannot be drawn
        costs a log line, never the run."""
        try:
            cakes, r_axis, eta_axis, ids, truncated = \
                hydra_zarr_combine.read_merged_cakes(path)
        except Exception as exc:
            self._log.append(f"[combine] could not read back {Path(path).name} "
                             f"for display: {exc}")
            return
        profiles = collapse_cake_eta(cakes)
        ctx, why = self._merged_axis_context()
        for view in (self._merged_waterfall, self._merged_stack, self._merged_cakes):
            try:
                view.clear()
            except Exception:
                pass
            if ctx is not None:
                try:
                    # "R": the stored axis is R in pixels in every case, and
                    # ctx is only how to convert it -- same contract as the
                    # per-panel views and tab_batch.
                    view.set_axis_context(*ctx, "R")
                except Exception:
                    pass
        self._merged_waterfall.reset(r_axis)
        self._merged_stack.reset(r_axis)
        for prof, fid in zip(profiles, ids):
            self._merged_waterfall.add_profile(prof)
            self._merged_stack.add_profile(r_axis, prof, label=fid)
        self._merged_cakes.set_cakes(cakes, r_axis, eta_axis, frame_ids=ids)
        self._merged_shown = str(path)
        note = (f"  (first {len(ids)} of {len(ids) + truncated} frames; "
                f"the rest are on disk)" if truncated else "")
        self._log.append(
            f"[combine] showing {Path(path).name} in Waterfall / Stacked "
            f"profiles / Eta-R cakes{note}")
        if ctx is None:
            # Said once per populate, and it says WHICH reason: "no
            # calibration loaded" and "the panels really are at different
            # distances" look the same from the viewer and mean quite
            # different things.
            self._log.append(
                f"[combine] note: {why}, so the merged view stays in R (px) "
                f"— no single 2θ axis would be right for all four panels.")

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
        # The combine runs from _maybe_finish_run, which an abort never
        # reaches: the workers' signals are disconnected and they are
        # orphaned, so `finished` never arrives. That is the right call --
        # merging a set whose panels stopped at different frames would
        # produce a quietly uneven archive -- but it has to be SAID, or a
        # cancelled run just looks like the combine silently failed.
        if (getattr(self, "_combine_zarr_chk", None)
                and self._combine_zarr_chk.isChecked()
                and "zarr" in (self._fmt.checked_keys() or [])):
            self._log.append(
                "[combine] skipped — an aborted scan's panels stop at "
                "different frames, so there is no complete set to merge. "
                "Re-run the scan to get its ge1234 archive.")

    def shutdown(self):
        """Interrupt + orphan any still-running per-panel workers on app
        close (picked up by ``MainWindow._stop_all_workers``'s generic
        ``tab.shutdown()`` hook via ``BatchTab.shutdown``)."""
        self._abort_all()
        # Stops polling/tailing only. Background jobs' screen sessions keep
        # running — that is the entire point of launching them detached.
        try:
            self._job_queue.shutdown()
        except Exception:
            pass

    # ── GUI state ────────────────────────────────────────────────────

    def _state_widgets(self) -> dict:
        return {
            "kernel": self._kernel, "grid_chk": self._grid_chk,
            "ome_start": self._ome_start, "ome_step": self._ome_step,
            "ome_collapse": self._ome_collapse, "ome_channel": self._ome_channel,
            "zarr_grouping": self._zarr_grouping,
            "lab_axes_chk": self._lab_axes_chk,
            "azim": self._azim, "var_check": self._var_check, "err_model": self._err_model,
            "multi_azimuth": self._multi_azimuth_chk,
            "preview_sum_n": self._preview_sum_n,
            "job_scans": self._job_scans,
            "ion_env_chk": self._ion_env_chk,
            "ion_motors_chk": self._ion_motors_chk,
            "bin_type": self._bin_type, "q_min": self._q_min, "q_max": self._q_max,
            "tth_min": self._tth_min, "tth_max": self._tth_max,
            "tth_bin": self._tth_bin,
            "q_bin": self._q_bin, "mon_ed": self._mon_ed, "out_ed": self._out_ed,
            "zarr_grouping": self._zarr_grouping,
            "run_mode": self._run_mode_combo, "frame_run_mode": self._frame_run_mode,
            "n_workers": self._n_workers,
        }

    def get_state(self) -> dict:
        return {
            "anchor_path": self._loader.current_path(),
            "active_panel": self._toolbar.current(),
            "fields": widgets_to_dict(self._state_widgets()),
            # NOT a widget, and without it every restored Output box reads
            # as a path the user chose -- so _maybe_autofill_output_dir
            # returns early forever and the box stays on the last dataset.
            "autofilled_out": self._autofilled_out,
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
        # Bin type was a Q-only checkbox ("q_check") until 2026-10-12. A
        # project saved then carries no "bin_type", and dropping its q_check
        # on the floor would silently reopen a Q-uniform run as Radial --
        # same numbers on screen, different output. Translate instead.
        # The combo's DISPLAY TEXT, not its userData: apply_dict_to_widgets
        # matches combos with findText.
        legacy_q = fields.pop("q_check", None)
        if legacy_q is not None and "bin_type" not in fields:
            fields["bin_type"] = "Q" if legacy_q else "Radial"
        apply_dict_to_widgets(self._state_widgets(), fields)
        # apply_dict_to_widgets blocks signals, so the handler that greys the
        # unused spin trio never ran -- a restored 2θ project would show its
        # Q boxes live and its 2θ boxes dead.
        self._on_bin_type_changed()
        # A project saved before the marker existed has no key at all --
        # that absence, not a None value, is what marks it legacy.
        self._out_state_was_legacy = "autofilled_out" not in state
        self._autofilled_out = state.get("autofilled_out")
        self._corr_widget.set_state(state.get("corr") or {})
        self._fmt.set_state(fmt_keys if fmt_keys is not None else state.get("fmt"))
        self._loader.set_state(state.get("loader") or {})
        self._det_view.set_display_state(state.get("det_view"))
        self._origin_btn.sync()
        for n_key, wf_state in (state.get("waterfalls") or {}).items():
            pair = self._viewer_pairs.get(int(n_key))
            if pair is not None:
                pair.waterfall.set_display_state(wf_state)
        # Projects saved before caking became per panel carry ONE shared set
        # at page level. Give every panel that set rather than dropping it:
        # it is what those runs actually used, so it restores the project as
        # it was, and the panels diverge only once somebody edits one.
        legacy = {k: state.get(k) for k in ("r_min", "r_max", "r_bin", "e_bin")
                  if state.get(k) is not None}
        if legacy:
            shared = {"R_MIN": legacy.get("r_min"), "R_MAX": legacy.get("r_max"),
                      "R_STEP": legacy.get("r_bin"), "ETA_STEP": legacy.get("e_bin")}
            for card in self._cards.values():
                card.set_caking({k: v for k, v in shared.items() if v is not None})
            self._log.append(
                "[hydra batch] This project predates per-panel caking — its one "
                "shared Rmin/Rmax/R bin/η bin was applied to all four panels.")
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
            spec = self._cards[n].resolved_spec()
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
