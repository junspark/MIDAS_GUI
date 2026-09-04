"""Tab 2 — Calibrate.

Ports the v3 calibration tab and adds Phase-1 features:
  - pipeline dropdown (one-shot / first-time / four-stage)
  - refine-flags group (Lsd, BC, ty, tz, tx, Wavelength, Distortion)
  - read-only distortion-coefficient table
  - per-ring radial-residual bar chart (new bottom tab)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from midas_gui.constants import (
    CALIBRANTS, PIPELINES, DEFAULT_PIPELINE, _SG, _LC, DEFAULT_WAVELENGTH, DEFAULT_PIXEL_UM,
    DEFAULT_LSD_UM, DEFAULT_BC_Y, DEFAULT_BC_Z, DEFAULT_CALIBRANT_TIF,
    DISTORTION_NAMES, MATERIALS)
from midas_gui.helpers import (
    _fspin, _NoScrollSpinBox, _predict_ring_radii, _NoScrollComboBox,
    make_kedge_label, make_pixel_label, tilted_ring_xy, refresh_combo_items,
    widgets_to_dict, apply_dict_to_widgets, im_trans_codes_from_checkboxes,
    paramstest_pairs, parse_dspacing_text)
from midas_gui.widgets import (
    PickableImageViewer, ProfileViewer, LogPanel, DataLoaderPanel, CakeViewer,
    RingResidualViewer, build_lab_frame_axes_items, ring_azimuth_residual)
from midas_gui.workers import CalibrationWorker, IntegrationWorker, ManualDspacingCalibWorker
from midas_gui.dialogs import _SaveParamstestDialog, DistortionRefineDialog, show_error
from midas_gui.hydra_widgets import HydraModeRibbon
from midas_gui.hydra_calib_page import HydraCalibrationPage
from midas_gui import project
from midas_gui import settings
from midas_gui import style as S


class CalibrationTab(QtWidgets.QWidget):
    calibrationDone = QtCore.pyqtSignal(object)   # AutoCalibrationResult
    pullGeometry = QtCore.pyqtSignal()            # request geometry from Data Viewer
    sendGeometryToViewer = QtCore.pyqtSignal(dict)  # push calibrated geometry → Data Viewer
    pullHydraFromViewer = QtCore.pyqtSignal()     # Hydra page's "← Data Viewer" clicked
    sendHydraGeometryToViewer = QtCore.pyqtSignal(int, dict)  # panel_num, geometry
    hydraPanelCalibrationDone = QtCore.pyqtSignal(int, object)  # panel_num, AutoCalibrationResult

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._dark: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._result = None
        self._worker = None
        self._int_worker = None
        self._calib_cancelled = False
        self._orphans: list = []       # aborted workers kept alive until they wind down
        self._ring_items: list = []
        self._corrected_ring_items: list = []
        self._calib_result = None
        self._dist_coeffs = set(DISTORTION_NAMES)   # distortion coeffs to refine
        self._seed_dist: dict = {}                  # seed distortion carried from a result
        self._last_dist_coeffs: Optional[set] = None  # coeffs selected for the last run
        self._last_cfg: Optional[dict] = None          # cfg used for the last run (provenance)
        self._last_bright: Optional[np.ndarray] = None
        self._last_background: Optional[np.ndarray] = None
        self._project_ctx: Optional[project.ProjectContext] = None
        self._pending_log_result = None   # result awaiting _log_to_project once integration finishes
        self._build_ui()
        self._loader.set_path(DEFAULT_CALIBRANT_TIF)

    def set_mask_from_tab1(self, mask: Optional[np.ndarray]):
        self._loader.set_tab1_mask(mask)

    def set_project_context(self, ctx: "project.ProjectContext"):
        self._project_ctx = ctx
        self._hydra_page.set_project_context(ctx)

    def import_hydra_from_viewer(self, data: dict):
        self._hydra_page.import_from_viewer(data)

    def _on_mode_changed(self, mode: str):
        """Leftmost ribbon switched between "single" and "hydra" — swap the
        visible page, mirroring DataViewerTab's identical split."""
        self._mode_stack.setCurrentWidget(self._hydra_page if mode == "hydra" else self._hsplit)

    def set_hydra_available(self, enabled: bool) -> None:
        """Show/hide the Hydra option on the mode ribbon (only meaningful at
        the 1-ID-E beamline profile — see MainWindow.apply_hydra_visibility)."""
        self._mode_ribbon.set_hydra_enabled(enabled)

    def bind_hydra_registry(self, registry, label: str):
        """Same role as `widgets.DataLoaderPanel.bind_registry`, for this
        tab's Hydra loader (the Hydra page is built eagerly, unlike Batch
        Integrate's, so no deferred binding is needed)."""
        self._hydra_page._loader.bind_registry(registry, label)

    def refresh_calibrants(self) -> None:
        """Repopulate the Calibrant dropdown (single-detector view and every
        Hydra panel) from the just-activated profile's constants.CALIBRANTS."""
        refresh_combo_items(self._cal, CALIBRANTS)
        self._hydra_page.refresh_calibrants()

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(0)

        # Leftmost mode ribbon: "Single detector" (this tab's existing view)
        # vs. "Hydra" (4-panel GE detector calibration) — same pattern as
        # the Data Viewer tab's split.
        self._mode_ribbon = HydraModeRibbon()
        self._mode_ribbon.modeChanged.connect(self._on_mode_changed)
        root.addWidget(self._mode_ribbon)

        self._mode_stack = QtWidgets.QStackedWidget()
        root.addWidget(self._mode_stack, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False); split.setHandleWidth(6)
        self._mode_stack.addWidget(split); self._hsplit = split

        # ── LEFT: data loader ──
        self._loader = DataLoaderPanel(mode="single")
        self._loader.setMinimumWidth(200)
        self._loader.dataChanged.connect(self._on_loader_data)
        self._loader.fieldsChanged.connect(self._on_fields_changed)
        self._loader.metadataDetected.connect(self._on_metadata_detected)
        split.addWidget(self._loader)

        # ── MIDDLE: parameters ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setMinimumWidth(260)
        inner = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(inner); lv.setContentsMargins(2, 2, 2, 2); lv.setSpacing(8)
        scroll.setWidget(inner)

        def _br(w=30):
            b = QtWidgets.QPushButton("…"); b.setFixedWidth(w); return b

        # ── Pipeline ──
        pipe = S.make_card("Pipeline")
        self._pipeline = _NoScrollComboBox()
        for label, key, enabled in PIPELINES:
            self._pipeline.addItem(label, key)
            if not enabled:
                self._pipeline.model().item(self._pipeline.count() - 1).setEnabled(False)
        _pi = self._pipeline.findData(DEFAULT_PIPELINE)
        if _pi >= 0 and self._pipeline.model().item(_pi).isEnabled():
            self._pipeline.setCurrentIndex(_pi)
        self._pipeline.setToolTip(
            "Lsd & beam-centre are recovered well by every pipeline.\n"
            "For trustworthy TILTS / strain, prefer Four-stage or First-time —\n"
            "validation found One-shot / Bayesian can report a spurious tilt on\n"
            "weakly-tilted data (it is self-compensated, so integration is still fine).")
        pipe.body.addWidget(self._pipeline)
        guide = QtWidgets.QLabel("Lsd/BC: any · tilt/strain: Four-stage or First-time")
        guide.setStyleSheet(f"color:{S.MUTED};font-size:10px"); guide.setWordWrap(True)
        pipe.body.addWidget(guide)
        lv.addWidget(pipe)

        # ── Detector & Calibrant ──
        det = S.make_card("Detector & Calibrant")
        self._load_calib_btn = QtWidgets.QPushButton("Load calibration file…")
        self._load_calib_btn.setToolTip(
            "Load geometry from a MIDAS paramstest (.txt), a calibration .json, "
            "or a pyFAI .poni — sets λ, pixel size, and the seed BC + Lsd.")
        self._load_calib_btn.clicked.connect(self._load_calib_file)
        self._from_view_btn = QtWidgets.QPushButton("← Data Viewer")
        self._from_view_btn.setToolTip(
            "Pull λ, pixel size, Lsd and beam centre from the Data Viewer tab "
            "into the detector + seed fields here.")
        self._from_view_btn.clicked.connect(self.pullGeometry.emit)
        _lrow = QtWidgets.QHBoxLayout(); _lrow.setSpacing(4)
        _lrow.addWidget(self._load_calib_btn); _lrow.addWidget(self._from_view_btn)
        _lrow.addStretch(1)
        det.body.addLayout(_lrow)
        self._wl = _fspin(0.001, 10.0, 5, DEFAULT_WAVELENGTH, "Å")
        self._cal = _NoScrollComboBox(); self._cal.addItems(CALIBRANTS); self._cal.setMaximumWidth(150)
        det.body.addLayout(S.Form().row(
            (make_kedge_label(self._wl, "λ:"), self._wl), ("Calibrant:", self._cal)))
        self._pxY = _fspin(1.0, 5000.0, 2, DEFAULT_PIXEL_UM, "µm")
        self._pxZ_check = QtWidgets.QCheckBox("pxZ")
        self._pxZ_spin = _fspin(1.0, 5000.0, 2, DEFAULT_PIXEL_UM, "µm"); self._pxZ_spin.setEnabled(False)
        self._pxZ_check.toggled.connect(self._pxZ_spin.setEnabled)
        prow = QtWidgets.QHBoxLayout(); prow.setSpacing(4)
        prow.addWidget(self._pxY, 1); prow.addWidget(self._pxZ_check); prow.addWidget(self._pxZ_spin, 1)
        det.body.addLayout(S.Form().row(
            (make_pixel_label(self._pxY, "Pixel:", also=self._pxZ_spin), prow)))
        self._flip_y = QtWidgets.QCheckBox("Flip Y"); self._flip_z = QtWidgets.QCheckBox("Flip Z")
        self._transp = QtWidgets.QCheckBox("Transpose")
        for cb in (self._flip_y, self._flip_z, self._transp):
            cb.toggled.connect(self._on_im_trans_changed)
        tb2 = QtWidgets.QHBoxLayout(); tb2.setSpacing(8)
        tb2.addWidget(self._flip_y); tb2.addWidget(self._flip_z); tb2.addWidget(self._transp); tb2.addStretch(1)
        det.body.addWidget(S.LabelRight("Transforms:")); det.body.addLayout(tb2)
        lv.addWidget(det)

        # ── Manual ring-picking (non-crystalline calibrants) ──
        manual = S.make_card("Manual ring-picking (non-crystalline calibrants)")
        manual_hint = QtWidgets.QLabel(
            "For calibrants with no space group (e.g. AgBH): pick points on the "
            "image with 'Pick d-spacing pts' below, tag each with its Ring #, "
            "then fit Lsd + beam center directly from Bragg's law. Bypasses the "
            "Calibrant dropdown above; tilt is fixed at 0.")
        manual_hint.setStyleSheet(f"color:{S.MUTED};font-size:10px"); manual_hint.setWordWrap(True)
        manual.body.addWidget(manual_hint)
        self._dsp_material = _NoScrollComboBox()
        self._dsp_material.addItems(
            [n for n, m in MATERIALS.items() if m.get("kind") == "dspacing"]
            + ["Custom d-spacings…"])
        manual.body.addLayout(S.Form().row(("Material:", self._dsp_material)))
        self._dsp_custom_ed = QtWidgets.QLineEdit()
        self._dsp_custom_ed.setPlaceholderText(
            "Custom d-spacings (Å), comma/space-separated, e.g. 58.38 29.19 19.46")
        self._dsp_custom_ed.setVisible(False)
        manual.body.addWidget(self._dsp_custom_ed)
        self._dsp_material.currentTextChanged.connect(self._on_dsp_material_changed)
        self._dsp_custom_ed.textChanged.connect(self._on_dspacing_picks_changed)
        self._dsp_summary = QtWidgets.QLabel("No points picked yet.")
        self._dsp_summary.setStyleSheet(f"color:{S.MUTED};font-size:10px"); self._dsp_summary.setWordWrap(True)
        manual.body.addWidget(self._dsp_summary)
        self._dsp_fit_btn = QtWidgets.QPushButton("Fit Geometry (manual)")
        self._dsp_fit_btn.setEnabled(False)
        self._dsp_fit_btn.setToolTip(
            "Fit Lsd + beam center from the picked d-spacing points (need ≥3 valid points).")
        self._dsp_fit_btn.clicked.connect(self._run_manual_fit)
        manual.body.addWidget(self._dsp_fit_btn)
        lv.addWidget(manual)

        # ── Threshold (calibration image only) ──
        thr = S.make_card("Threshold  (pixels below → 0, calibration image)")
        self._thr_check = QtWidgets.QCheckBox("Apply threshold to calibration image")
        self._thr_check.setToolTip(
            "When on, pixels dimmer than the slider value are set to 0 in the image\n"
            "fed to the calibration pipeline (and the live preview). Useful to drop\n"
            "background / weak pixels before calibrating.")
        thr.body.addWidget(self._thr_check)
        self._thr_min = _fspin(-1e9, 1e9, 1, 0.0)
        self._thr_max = _fspin(-1e9, 1e9, 1, 65535.0)
        thr.body.addLayout(S.Form().row(("slider min:", self._thr_min), ("max:", self._thr_max)))
        self._thr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._thr_slider.setRange(0, 1000); self._thr_slider.setValue(0)
        self._thr_val = QtWidgets.QLabel("threshold = —")
        self._thr_val.setStyleSheet(f"color:{S.ACCENT};font-size:11px")
        srow = QtWidgets.QHBoxLayout(); srow.setSpacing(6)
        srow.addWidget(self._thr_slider, 1); srow.addWidget(self._thr_val)
        thr.body.addLayout(srow)
        for w in (self._thr_min, self._thr_max, self._thr_slider, self._thr_val):
            w.setEnabled(False)
        self._thr_check.toggled.connect(self._on_threshold_toggled)
        self._thr_slider.valueChanged.connect(self._on_threshold_changed)
        self._thr_min.valueChanged.connect(self._on_threshold_changed)
        self._thr_max.valueChanged.connect(self._on_threshold_changed)
        lv.addWidget(thr)

        # ── Average frames (hdf5 / folder) ──
        avgc = S.make_card("Average frames")
        self._avg_check = QtWidgets.QCheckBox("Average frames into a single image")
        self._avg_check.setToolTip(
            "Average a range of frames into one image used for calibration. "
            "Requires a multi-frame source.")
        avgc.body.addWidget(self._avg_check)
        self._avg_start = _NoScrollSpinBox(); self._avg_start.setRange(0, 999999)
        self._avg_end = _NoScrollSpinBox(); self._avg_end.setRange(0, 999999)
        self._avg_end.setToolTip("Last frame (exclusive). 0 = all frames.")
        for w in (self._avg_start, self._avg_end):
            w.setEnabled(False)
        afm = S.Form(); afm.row(("start:", self._avg_start), ("end(0=all):", self._avg_end))
        avgc.body.addLayout(afm)
        self._avg_note = QtWidgets.QLabel("")
        self._avg_note.setStyleSheet("color:#9a9a9a;font-size:10px"); self._avg_note.setWordWrap(True)
        avgc.body.addWidget(self._avg_note)
        self._avg_card = avgc
        self._avg_check.toggled.connect(self._on_avg_toggled)
        for w in (self._avg_start, self._avg_end):
            w.valueChanged.connect(self._on_avg_changed)
        lv.addWidget(avgc)

        # ── Initial seed ──
        seed = S.make_card("Initial seed  (Pick tools on image)")
        self._manual_seed_check = QtWidgets.QCheckBox("Use manual seed")
        self._manual_seed_check.setToolTip(
            "Enable BC + Lsd as the LM starting point.\n"
            "Use Pick BC / Pick Ring on the image to populate BC automatically.")
        seed.body.addWidget(self._manual_seed_check)
        self._seed_bcy = _fspin(-99999, 99999, 2, DEFAULT_BC_Y, "px")
        self._seed_bcz = _fspin(-99999, 99999, 2, DEFAULT_BC_Z, "px")
        # Lsd shown/entered in mm; calculations & files use µm.
        self._seed_lsd = _fspin(0.001, 1e5, 4, DEFAULT_LSD_UM / 1000.0, " mm")
        # Seed tilts (deg). Honoured by the four-stage / advanced pipelines; the
        # one-shot / first-time paths seed tilts only if the installed backend
        # exposes initial-tilt kwargs (otherwise they start at 0).
        self._seed_tx = _fspin(-180, 180, 4, 0.0, "°")
        self._seed_ty = _fspin(-180, 180, 4, 0.0, "°")
        self._seed_tz = _fspin(-180, 180, 4, 0.0, "°")
        self._seed_tilts = (self._seed_tx, self._seed_ty, self._seed_tz)
        for w in self._seed_tilts:
            w.setToolTip(
                "Honoured by Four-stage / Bayesian / Joint (always), and by "
                "One-shot when Multi-panel is enabled or Distortion refinement "
                "is restricted to a subset of coefficients. Plain One-shot / "
                "First-time only honour this if the installed calibrate() "
                "backend exposes initial-tilt kwargs — a warning is logged "
                "before Run if it doesn't and this value is non-zero.")
        for w in (self._seed_bcy, self._seed_bcz, self._seed_lsd, *self._seed_tilts):
            w.setEnabled(False)
        for sig in (self._seed_bcy, self._seed_bcz, self._seed_lsd, *self._seed_tilts):
            self._manual_seed_check.toggled.connect(sig.setEnabled)
        sfm = S.Form(); sfm.row(("BC_y:", self._seed_bcy), ("BC_z:", self._seed_bcz)); sfm.row(("Lsd:", self._seed_lsd))
        sfm.row(("tx:", self._seed_tx), ("ty:", self._seed_ty)); sfm.row(("tz:", self._seed_tz))
        seed.body.addLayout(sfm)
        self._feedback_check = QtWidgets.QCheckBox("Feed result back to seed")
        self._feedback_check.setChecked(True)
        self._feedback_check.setToolTip(
            "After a calibration, copy the optimized BC / Lsd / tilts / distortion "
            "back into these seed fields so the next run starts from them.")
        seed.body.addWidget(self._feedback_check)
        self._seed_note = QtWidgets.QLabel("")
        self._seed_note.setStyleSheet(f"color:{S.ACCENT};font-size:10px"); self._seed_note.setWordWrap(True)
        seed.body.addWidget(self._seed_note)
        lv.addWidget(seed)

        # ── Refine parameters ──
        refc = S.make_card("Refine parameters")
        rfl = QtWidgets.QGridLayout(); rfl.setSpacing(4)
        self._ref_lsd = QtWidgets.QCheckBox("Lsd"); self._ref_lsd.setChecked(True)
        self._ref_bc = QtWidgets.QCheckBox("BC"); self._ref_bc.setChecked(True)
        self._ref_ty = QtWidgets.QCheckBox("ty"); self._ref_ty.setChecked(True)
        self._ref_tz = QtWidgets.QCheckBox("tz"); self._ref_tz.setChecked(True)
        self._ref_tx = QtWidgets.QCheckBox("tx")
        self._ref_wl = QtWidgets.QCheckBox("Wavelength")
        self._ref_dist = QtWidgets.QCheckBox("Distortion"); self._ref_dist.setChecked(True)
        self._build_rc = QtWidgets.QCheckBox("Residual map"); self._build_rc.setChecked(True)
        for i, w in enumerate((self._ref_lsd, self._ref_bc, self._ref_ty, self._ref_tz,
                               self._ref_tx, self._ref_wl)):
            rfl.addWidget(w, i // 2, i % 2)
        # Distortion gets a companion "…" button opening the per-coefficient dialog.
        self._dist_btn = QtWidgets.QToolButton(); self._dist_btn.setText("…")
        self._dist_btn.setToolTip("Choose which distortion coefficients to refine "
                                  "(η-fold presets available).")
        self._dist_btn.clicked.connect(self._edit_distortion_coeffs)
        drow = QtWidgets.QHBoxLayout(); drow.setSpacing(4)
        drow.addWidget(self._ref_dist); drow.addWidget(self._dist_btn); drow.addStretch(1)
        rfl.addLayout(drow, 3, 0)
        rfl.addWidget(self._build_rc, 3, 1)
        self._ref_dist.toggled.connect(lambda _=0: self._update_dist_label())
        refc.body.addLayout(rfl)
        lv.addWidget(refc)
        self._update_dist_label()

        # ── Advanced ──
        grp_adv = QtWidgets.QGroupBox("Advanced")
        grp_adv.setCheckable(True); grp_adv.setChecked(False)
        av = QtWidgets.QVBoxLayout(grp_adv); av.setContentsMargins(8, 6, 8, 6); av.setSpacing(5)
        self._n_iter = _NoScrollSpinBox(); self._n_iter.setRange(1, 1_000_000); self._n_iter.setValue(4)
        self._lm_iter = _NoScrollSpinBox(); self._lm_iter.setRange(1, 1_000_000); self._lm_iter.setValue(200)
        self._device = _NoScrollComboBox(); self._device.addItems(["cpu", "cuda"])
        av.addLayout(S.Form().row(("E-M iters:", self._n_iter), ("LM iters:", self._lm_iter)))
        self._out_ed = QtWidgets.QLineEdit(); self._out_ed.setPlaceholderText("Output dir…")
        bou = _br(); bou.clicked.connect(lambda: self._out_ed.setText(
            QtWidgets.QFileDialog.getExistingDirectory(self, "Output dir") or ""))
        outr = QtWidgets.QHBoxLayout(); outr.setSpacing(4); outr.addWidget(self._out_ed, 1); outr.addWidget(bou)
        av.addLayout(S.Form().row(("Device:", self._device)))
        av.addLayout(S.Form().row(("Output:", outr)))
        lv.addWidget(grp_adv)

        # ── Multi-panel ──
        grp_panel = QtWidgets.QGroupBox("Multi-panel detector")
        grp_panel.setCheckable(True); grp_panel.setChecked(False)
        grp_panel.setToolTip("Refine per-module rigid shifts for tiled detectors (px).")
        pv = QtWidgets.QVBoxLayout(grp_panel); pv.setContentsMargins(8, 6, 8, 6); pv.setSpacing(5)
        self._pn_y = _NoScrollSpinBox(); self._pn_y.setRange(1, 1_000_000); self._pn_y.setValue(3)
        self._pn_z = _NoScrollSpinBox(); self._pn_z.setRange(1, 1_000_000); self._pn_z.setValue(8)
        self._ps_y = _NoScrollSpinBox(); self._ps_y.setRange(1, 1_000_000); self._ps_y.setValue(487)
        self._ps_z = _NoScrollSpinBox(); self._ps_z.setRange(1, 1_000_000); self._ps_z.setValue(195)
        self._pg_y = _NoScrollSpinBox(); self._pg_y.setRange(0, 1_000_000); self._pg_y.setValue(7)
        self._pg_z = _NoScrollSpinBox(); self._pg_z.setRange(0, 1_000_000); self._pg_z.setValue(17)
        pf2 = S.Form()
        pf2.row(("panels Y:", self._pn_y), ("panels Z:", self._pn_z))
        pf2.row(("size Y:", self._ps_y), ("size Z:", self._ps_z))
        pf2.row(("gap Y:", self._pg_y), ("gap Z:", self._pg_z))
        pv.addLayout(pf2)
        self._panel_grp = grp_panel
        lv.addWidget(grp_panel)

        # ── Run + Save ──
        self._run_btn = S.primary_btn("Run Calibration")
        self._run_btn.clicked.connect(self._run)
        self._abort_btn = QtWidgets.QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.setToolTip("Cancel: returns control immediately and discards the "
                                   "result. The running computation finishes in the background.")
        self._abort_btn.clicked.connect(self._abort)
        run_row = QtWidgets.QHBoxLayout(); run_row.setSpacing(6)
        run_row.addWidget(self._run_btn, 1); run_row.addWidget(self._abort_btn)
        lv.addLayout(run_row)
        self._prog = QtWidgets.QProgressBar(); self._prog.setRange(0, 0); self._prog.setVisible(False)
        lv.addWidget(self._prog)
        self._save_json_btn = QtWidgets.QPushButton("Save .json"); self._save_json_btn.setEnabled(False)
        self._save_json_btn.clicked.connect(self._save_json)
        self._save_ps_btn = QtWidgets.QPushButton("Save paramstest.txt"); self._save_ps_btn.setEnabled(False)
        self._save_ps_btn.clicked.connect(self._save_paramstest)
        lv.addLayout(S.button_grid([self._save_json_btn, self._save_ps_btn], 2))

        lv.addStretch(1)
        split.addWidget(scroll)

        # Right: image + bottom tabs
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._img_view = PickableImageViewer()
        self._img_view.bcPicked.connect(self._on_bc_picked)
        self._img_view.ringFitBC.connect(self._on_ring_fit_bc)
        self._img_view.dspacingPicksChanged.connect(self._on_dspacing_picks_changed)
        tb = self._img_view._toolbar_layout
        self._show_rings_check = QtWidgets.QCheckBox("Show rings"); self._show_rings_check.setChecked(True)
        self._show_rings_check.toggled.connect(self._on_show_rings_toggled)
        tb.addWidget(self._show_rings_check)
        self._corrected_check = QtWidgets.QCheckBox("Corrected")
        self._corrected_check.setToolTip("Redraw the rings reflecting the fitted tilt correction.")
        self._corrected_check.toggled.connect(self._on_corrected_rings_toggled)
        tb.addWidget(self._corrected_check)
        self._corr_status = QtWidgets.QLabel("")
        self._corr_status.setStyleSheet(f"color:{S.ACCENT};font-size:10px")
        tb.addWidget(self._corr_status)
        self._lab_axes_on = QtWidgets.QCheckBox("Lab-frame axes")
        self._lab_axes_on.setToolTip(
            "Overlay MIDAS lab-frame axes (X_Lab/Y_Lab), the beam-direction ⊗ "
            "glyph, and an η sweep arc, anchored at the current seed beam "
            "centre — lets you verify orientation/ImTransOpt at a glance.")
        self._lab_axes_on.toggled.connect(self._on_lab_axes_toggled)
        tb.addWidget(self._lab_axes_on)
        self._axis_items: list = []
        for sig in (self._seed_bcy.valueChanged, self._seed_bcz.valueChanged):
            sig.connect(self._redraw_lab_axes_if_on)
        right.addWidget(self._img_view)

        bot = QtWidgets.QTabWidget()
        self._prof_view = ProfileViewer()
        ptb = self._prof_view._toolbar_layout
        self._cal_r_bin = _fspin(0.1, 20.0, 2, 1.0, "px"); self._cal_r_bin.setFixedWidth(78)
        self._cal_eta_bin = _fspin(0.5, 30.0, 1, 5.0, "°"); self._cal_eta_bin.setFixedWidth(64)
        self._cal_azim = _NoScrollComboBox()
        self._cal_azim.addItem("Pixel-weighted", True)
        self._cal_azim.addItem("η-bin mean", False)
        self._cal_azim.setToolTip(
            "1-D profile from the (η, R) cake: pixel-weighted mean (robust to partial\n"
            "azimuthal coverage / off-detector beam centre) vs unweighted η-bin mean.")
        reint_btn = QtWidgets.QPushButton("Re-integrate"); reint_btn.clicked.connect(self._reintegrate)
        ptb.insertWidget(3, reint_btn)
        ptb.insertWidget(3, self._cal_azim)
        ptb.insertWidget(3, self._cal_eta_bin)
        ptb.insertWidget(3, QtWidgets.QLabel("η:"))
        ptb.insertWidget(3, self._cal_r_bin)
        ptb.insertWidget(3, QtWidgets.QLabel("  R bin:"))
        bot.addTab(self._prof_view, "Radial Profile")
        self._cake_view = CakeViewer()
        bot.addTab(self._cake_view, "Eta vs R Cake")
        self._resid_cake_view = RingResidualViewer()
        bot.addTab(self._resid_cake_view, "Ring Residual")
        # Results tab: the full parameter set exactly as written to paramstest.txt,
        # laid out across several columns (the panel is wide but short) as plain text —
        # including the distortion coefficients (no table) — plus a button to push the
        # geometry to the Data Viewer.
        res_w = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(res_w)
        rl.setContentsMargins(10, 8, 10, 8); rl.setSpacing(8)
        hdr = QtWidgets.QHBoxLayout()
        hdr.addWidget(QtWidgets.QLabel("<b>Calibration parameters</b> "
                                       "(as written to <code>paramstest.txt</code>)"))
        hdr.addStretch(1)
        self._to_view_btn = QtWidgets.QPushButton("→ Send to Data Viewer")
        self._to_view_btn.setEnabled(False)
        self._to_view_btn.setToolTip(
            "Replace the Data Viewer tab's geometry fields (λ, pixel size, Lsd, beam "
            "centre) with these calibrated values.")
        self._to_view_btn.clicked.connect(self._send_to_viewer)
        hdr.addWidget(self._to_view_btn)
        rl.addLayout(hdr)

        self._param_grid = QtWidgets.QGridLayout()
        self._param_grid.setHorizontalSpacing(28); self._param_grid.setVerticalSpacing(9)
        _pg_host = QtWidgets.QWidget(); _pg_host.setLayout(self._param_grid)
        rl.addWidget(_pg_host)

        self._r_diag = QtWidgets.QLabel("Run a calibration to see the parameters.")
        self._r_diag.setStyleSheet(f"color:{S.MUTED};font-size:12px")
        self._r_diag.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        rl.addWidget(self._r_diag)
        rl.addStretch(1)
        bot.addTab(res_w, "Results")
        self._log = LogPanel()
        bot.addTab(self._log, "Log")
        # Let the Log text fill its whole tab page (no small 120px cap here).
        self._log.setMaximumHeight(16_777_215)
        right.addWidget(bot)
        right.setChildrenCollapsible(False)
        right.setStretchFactor(0, 3); right.setStretchFactor(1, 1)
        right.setSizes([680, 320])
        self._bot_tabs = bot
        right.setMinimumWidth(320)
        split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 0); split.setStretchFactor(2, 1)
        split.setSizes([286, 361, 950])

        # Page 1: Hydra (4-panel GE detector) calibration.
        self._hydra_page = HydraCalibrationPage()
        self._hydra_page.pullFromViewer.connect(self.pullHydraFromViewer.emit)
        self._hydra_page.sendGeometryToViewer.connect(self.sendHydraGeometryToViewer.emit)
        self._hydra_page.panelCalibrationDone.connect(self.hydraPanelCalibrationDone.emit)
        self._mode_stack.addWidget(self._hydra_page)

    # ── Data (from the loader panel) ──────────────────────────────

    def _on_loader_data(self):
        """New frame / data from the loader — refresh the calibration image, the
        threshold-slider range, and the display."""
        self._sync_avg_controls()
        self._image = self._source_image()
        if self._image is None:
            return
        lo, hi = float(np.nanmin(self._image)), float(np.nanmax(self._image))
        for w in (self._thr_min, self._thr_max, self._thr_slider):
            w.blockSignals(True)
        self._thr_min.setValue(max(0.0, lo)); self._thr_max.setValue(hi)
        self._thr_slider.setValue(0)
        for w in (self._thr_min, self._thr_max, self._thr_slider):
            w.blockSignals(False)
        self._update_threshold_label()
        self._show_calib_image(autorange=True)

    def _on_fields_changed(self):
        """Dark/bright/background changed — refresh the calibration preview
        (no autorange, matching Data Viewer's _on_fields_changed). The raw
        ``self._image`` is untouched; only the displayed, corrected render
        changes (see ``_show_calib_image``)."""
        if self._image is None:
            return
        self._show_calib_image(autorange=False)

    def _on_metadata_detected(self, detected: dict):
        """Best-effort pxY/wavelength_A auto-detected from the just-loaded
        file (see helpers.detect_geometry_from_path) — only the fields
        actually present are applied."""
        if "wavelength_A" in detected:
            self._wl.setValue(float(detected["wavelength_A"]))
        if "pxY" in detected:
            self._pxY.setValue(float(detected["pxY"]))

    # ── Threshold (calibration image only) ────────────────────────

    def _threshold_value(self) -> float:
        lo, hi = self._thr_min.value(), self._thr_max.value()
        if hi <= lo:
            return hi
        return lo + (self._thr_slider.value() / 1000.0) * (hi - lo)

    def _update_threshold_label(self):
        self._thr_val.setText(f"< {self._threshold_value():.4g} → 0")

    def _calib_image(self):
        """Image fed to the calibration pipeline: thresholded copy if enabled."""
        if self._image is None:
            return None
        if self._thr_check.isChecked():
            thr = self._threshold_value()
            out = self._image.copy()
            out[self._image < thr] = 0.0
            return out
        return self._image

    # ── Average frames ────────────────────────────────────────────

    def _source_image(self):
        """Base image for calibration: averaged frames if enabled, else current.

        Deliberately raw (uncorrected, untransformed) — this feeds both the
        on-screen preview (via ``_show_calib_image``, which applies dark/
        bright/background and the Transforms checkboxes for display only) and
        the actual pipeline run, where ``CalibrationWorker`` applies bright/
        background, the same transform, and the backend subtracts dark
        internally. Pre-correcting here would double-apply them for the real
        run."""
        if self._avg_check.isChecked() and self._loader.n_frames() > 1:
            avg = self._loader.average_frames(
                self._avg_start.value(), self._avg_end.value())
            if avg is not None:
                return avg
        return self._loader.current_frame()

    def _sync_avg_controls(self):
        """Enable/disable the averaging card and clamp spin ranges to the source."""
        n = self._loader.n_frames()
        multi = n > 1
        self._avg_card.setEnabled(multi)
        if not multi and self._avg_check.isChecked():
            self._avg_check.blockSignals(True); self._avg_check.setChecked(False)
            self._avg_check.blockSignals(False)
        hi = max(0, n)
        for w in (self._avg_start, self._avg_end):
            w.blockSignals(True); w.setRange(0, hi); w.blockSignals(False)
        self._update_avg_note()

    def _update_avg_note(self):
        n = self._loader.n_frames()
        if n <= 1:
            self._avg_note.setText("Single-frame source — averaging unavailable.")
            return
        start = self._avg_start.value()
        end = self._avg_end.value() or n
        end = min(end, n)
        cnt = len(range(max(0, start), end))
        self._avg_note.setText(f"{cnt} of {n} frames averaged (start={start}, "
                               f"end={end}).")

    def _on_avg_toggled(self, on):
        for w in (self._avg_start, self._avg_end):
            w.setEnabled(on)
        self._update_avg_note()
        self._image = self._source_image()
        if self._image is not None:
            self._show_calib_image(autorange=False)

    def _on_avg_changed(self, *_):
        self._update_avg_note()
        if self._avg_check.isChecked():
            self._image = self._source_image()
            if self._image is not None:
                self._show_calib_image(autorange=False)

    # ── Distortion coefficient selection ──────────────────────────

    def _edit_distortion_coeffs(self):
        dlg = DistortionRefineDialog(self._dist_coeffs, self)
        if dlg.exec_():
            self._dist_coeffs = dlg.selected()
            if self._dist_coeffs and not self._ref_dist.isChecked():
                self._ref_dist.setChecked(True)
            self._update_dist_label()

    def _update_dist_label(self):
        n = len(self._dist_coeffs) if self._ref_dist.isChecked() else 0
        self._ref_dist.setText(f"Distortion ({n}/15)")

    # ── Seed feedback from a result ───────────────────────────────

    def _seed_from_result(self, result):
        """Copy optimized geometry from a result into the seed fields."""
        self._manual_seed_check.setChecked(True)
        self._seed_bcy.setValue(float(result.BC_y))
        self._seed_bcz.setValue(float(result.BC_z))
        self._seed_lsd.setValue(float(result.Lsd) / 1000.0)   # µm → mm
        self._seed_tx.setValue(float(getattr(result, "tx", 0.0) or 0.0))
        self._seed_ty.setValue(float(getattr(result, "ty", 0.0) or 0.0))
        self._seed_tz.setValue(float(getattr(result, "tz", 0.0) or 0.0))
        if getattr(result, "wavelength_A", None):
            self._wl.setValue(float(result.wavelength_A))
        self._seed_dist = dict(getattr(result, "distortion", {}) or {})
        self._seed_note.setText("Seed updated from the last calibration result.")

    def _im_trans_codes(self) -> list:
        """Ordered MIDAS ImTransOpt codes from the Transforms checkboxes."""
        return im_trans_codes_from_checkboxes(self._flip_y, self._flip_z, self._transp)

    def _on_im_trans_changed(self, *_):
        """Transform checkbox toggled — refresh the preview to match. Beam-center
        and ring-fit picks are read straight off the displayed array, so once it's
        transformed here, picks land in transformed-pixel space automatically."""
        if self._image is not None:
            self._show_calib_image(autorange=False)

    def _show_calib_image(self, autorange: bool = False):
        """Render the preview: dark/bright/background-corrected and Transforms-
        applied for display only (``self._image``/``_calib_image()`` stay raw —
        see ``_source_image``). ``CalibrationWorker`` applies the same transform
        to the array actually fed to the calibration pipeline."""
        if self._image is not None:
            img = self._loader.corrected(self._calib_image())
            self._img_view.set_raw_frame(img, self._im_trans_codes(),
                                          autorange=autorange, reset_levels=autorange)
        self._redraw_lab_axes_if_on()

    # ── Lab-frame axes overlay ───────────────────────────────────────
    # Same overlay as the Data Viewer tab (see tab_view.py / widgets.py
    # build_lab_frame_axes_items) — anchored here at the current seed BC
    # (rather than a "geometry card") since that's what's always present
    # and kept up to date (Pick BC, Load calibration file, the post-run
    # feedback loop, and typing all funnel through _seed_bcy/_seed_bcz).

    def _on_lab_axes_toggled(self, checked: bool):
        if checked:
            self._draw_lab_axes()
        else:
            self._clear_lab_axes()

    def _redraw_lab_axes_if_on(self, *_args):
        if getattr(self, "_lab_axes_on", None) is not None and self._lab_axes_on.isChecked():
            self._draw_lab_axes()

    def _clear_lab_axes(self):
        for it in self._axis_items:
            self._img_view._iv.removeItem(it)
        self._axis_items.clear()

    def _draw_lab_axes(self):
        self._clear_lab_axes()
        img = self._img_view._data
        if img is None:
            return
        items = build_lab_frame_axes_items(
            self._img_view._iv, img.shape, self._seed_bcy.value(), self._seed_bcz.value())
        for it in items:
            self._img_view._iv.addItem(it)
        self._axis_items.extend(items)

    def _on_threshold_toggled(self, on: bool):
        for w in (self._thr_min, self._thr_max, self._thr_slider, self._thr_val):
            w.setEnabled(on)
        self._update_threshold_label()
        self._show_calib_image(autorange=False)

    def _on_threshold_changed(self, *_):
        self._update_threshold_label()
        if self._thr_check.isChecked():
            self._show_calib_image(autorange=False)

    # ── Seed from picks ───────────────────────────────────────────

    def _load_calib_file(self):
        """Load geometry from a paramstest/.json/.poni into the seed + detector fields."""
        from midas_gui.helpers import geometry_fields_from_file
        from midas_gui.constants import DEFAULT_CALIB_FILE
        start = DEFAULT_CALIB_FILE if Path(DEFAULT_CALIB_FILE).exists() else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration file", start,
            "Calibration (*.json *.txt *.poni);;All files (*)")
        if not path:
            return
        try:
            g = geometry_fields_from_file(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e)); return
        self._wl.setValue(float(g["wavelength_A"]))
        self._pxY.setValue(float(g["pxY"]))
        if abs(float(g["pxZ"]) - float(g["pxY"])) > 1e-9:
            self._pxZ_check.setChecked(True); self._pxZ_spin.setValue(float(g["pxZ"]))
        else:
            self._pxZ_check.setChecked(False)
        self._manual_seed_check.setChecked(True)
        self._seed_bcy.setValue(float(g["BC_y"]))
        self._seed_bcz.setValue(float(g["BC_z"]))
        self._seed_lsd.setValue(float(g["Lsd"]) / 1000.0)   # µm → mm display
        self._seed_tx.setValue(float(g.get("tx") or 0.0))
        self._seed_ty.setValue(float(g.get("ty") or 0.0))
        self._seed_tz.setValue(float(g.get("tz") or 0.0))
        im_trans = g.get("im_trans") or []
        self._flip_y.setChecked(1 in im_trans)
        self._flip_z.setChecked(2 in im_trans)
        self._transp.setChecked(3 in im_trans)
        self._seed_note.setText(
            f"Loaded {Path(path).name}: λ={g['wavelength_A']:.5f} Å, px={g['pxY']:.2f} µm, "
            f"BC=({g['BC_y']:.2f}, {g['BC_z']:.2f}), Lsd={g['Lsd']/1000:.3f} mm, "
            f"tx={g.get('tx') or 0.0:.3f}°, ty={g.get('ty') or 0.0:.3f}°, "
            f"tz={g.get('tz') or 0.0:.3f}°.")
        self._log.append(f"Calibration file loaded: {path}")
        if im_trans and im_trans != [c for c in (1, 2, 3) if c in im_trans]:
            self._log.append(
                f"Note: ImTransOpt order in file ({im_trans}) differs from the "
                "fixed Flip Y → Flip Z → Transpose order used here; checkboxes "
                "were set but may not exactly reproduce the file's composition.")

    def apply_geometry(self, g: dict):
        """Set λ / pixel size / seed BC + Lsd from a geometry dict (Data Viewer)."""
        if not g:
            return
        if g.get("wavelength_A"):
            self._wl.setValue(float(g["wavelength_A"]))
        if g.get("pxY"):
            self._pxY.setValue(float(g["pxY"]))
        self._manual_seed_check.setChecked(True)
        if g.get("BC_y") is not None:
            self._seed_bcy.setValue(float(g["BC_y"]))
        if g.get("BC_z") is not None:
            self._seed_bcz.setValue(float(g["BC_z"]))
        if g.get("Lsd"):
            self._seed_lsd.setValue(float(g["Lsd"]) / 1000.0)   # µm → mm display
        if g.get("tx") is not None:
            self._seed_tx.setValue(float(g["tx"]))
        if g.get("ty") is not None:
            self._seed_ty.setValue(float(g["ty"]))
        if g.get("im_trans") is not None:
            im_trans = g["im_trans"] or []
            self._flip_y.setChecked(1 in im_trans)
            self._flip_z.setChecked(2 in im_trans)
            self._transp.setChecked(3 in im_trans)
        if g.get("tz") is not None:
            self._seed_tz.setValue(float(g["tz"]))
        self._seed_note.setText(
            f"Geometry from Data Viewer: λ={g.get('wavelength_A', 0):.5f} Å, "
            f"px={g.get('pxY', 0):.2f} µm, "
            f"BC=({g.get('BC_y', 0):.2f}, {g.get('BC_z', 0):.2f}), "
            f"Lsd={g.get('Lsd', 0)/1000:.3f} mm, "
            f"tx={g.get('tx', 0):.3f}°, ty={g.get('ty', 0):.3f}°, tz={g.get('tz', 0):.3f}°.")
        self._log.append("Geometry pulled from Data Viewer tab.")

    def _on_bc_picked(self, bc_y, bc_z):
        self._manual_seed_check.setChecked(True)
        self._seed_bcy.setValue(bc_y); self._seed_bcz.setValue(bc_z)
        self._seed_note.setText("BC set from click — also set Lsd before running.")
        self._log.append(f"BC set by click: ({bc_y:.2f}, {bc_z:.2f}) px — manual seed enabled")

    def _on_ring_fit_bc(self, bc_y, bc_z, r_px):
        self._manual_seed_check.setChecked(True)
        self._seed_bcy.setValue(bc_y); self._seed_bcz.setValue(bc_z)
        self._seed_note.setText(f"BC from ring fit (R={r_px:.1f} px). Set Lsd before running.")
        self._log.append(
            f"Ring fit: BC=({bc_y:.2f}, {bc_z:.2f}) px  R={r_px:.1f} px — manual seed enabled")

    # ── Manual d-spacing ring-picking fit (non-crystalline calibrants) ──

    def _manual_d_list(self) -> list:
        """Current material's d-spacings (Å), sorted descending — Ring #1 is
        the largest d-spacing, matching ``simulate_rings_from_dspacings``'s
        ``order`` numbering."""
        name = self._dsp_material.currentText()
        if name == "Custom d-spacings…":
            d_list = parse_dspacing_text(self._dsp_custom_ed.text())
        else:
            d_list = list(MATERIALS.get(name, {}).get("d_list", []))
        return sorted(d_list, reverse=True)

    def _on_dsp_material_changed(self, text: str):
        self._dsp_custom_ed.setVisible(text == "Custom d-spacings…")
        self._on_dspacing_picks_changed()

    def _on_dspacing_picks_changed(self, *_args):
        picks = self._img_view.dspacing_picks()
        d_list = self._manual_d_list()
        counts: dict = {}
        invalid = 0
        for _, _, ring_idx in picks:
            if 1 <= ring_idx <= len(d_list):
                counts[ring_idx] = counts.get(ring_idx, 0) + 1
            else:
                invalid += 1
        parts = [f"Ring {i} (d={d_list[i-1]:.3f} Å): {n} pts"
                 for i, n in sorted(counts.items())]
        if invalid:
            parts.append(f"{invalid} pt(s) on a ring # beyond this material's "
                         f"{len(d_list)} d-spacings (invalid)")
        self._dsp_summary.setText("   ".join(parts) if parts else "No points picked yet.")
        self._dsp_fit_btn.setEnabled(bool(d_list) and (len(picks) - invalid) >= 3)

    def _run_manual_fit(self):
        if self._worker and self._worker.isRunning():
            return
        d_list = self._manual_d_list()
        if not d_list:
            show_error(self, "Manual fit",
                       "No d-spacing list — pick a Material or enter custom d-spacings.")
            return
        picks = [(x, y, d_list[ring_idx - 1])
                for x, y, ring_idx in self._img_view.dspacing_picks()
                if 1 <= ring_idx <= len(d_list)]
        if len(picks) < 3:
            show_error(self, "Manual fit", "Need at least 3 valid picked points.")
            return
        self._orphans = [o for o in self._orphans if o.isRunning()]
        pxY = self._pxY.value()
        pxZ = self._pxZ_spin.value() if self._pxZ_check.isChecked() else pxY
        seed = None
        if self._manual_seed_check.isChecked():
            seed = (self._seed_lsd.value() * 1000.0,   # mm display → µm
                    self._seed_bcy.value(), self._seed_bcz.value())
        img = self._img_view._data
        NZ, NY = img.shape[:2] if img is not None else (0, 0)
        material_name = self._dsp_material.currentText()

        self._calib_cancelled = False
        self._run_btn.setEnabled(False); self._dsp_fit_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._prog.setVisible(True)
        self._bot_tabs.setCurrentWidget(self._log)
        self._log.append("─" * 40 + "\nStarting manual d-spacing fit…")

        self._last_dist_coeffs = set()
        self._worker = ManualDspacingCalibWorker(
            picks, self._wl.value(), pxY, pxZ, seed, NY, NZ, material_name, d_list,
            parent=self)
        self._worker.log_line.connect(self._log.append)
        self._worker.finished.connect(self._on_manual_fit_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_manual_fit_done(self, result):
        self._dsp_fit_btn.setEnabled(True)
        self._on_done(result)

    # ── Run ────────────────────────────────────────────────────────

    def _refine_flags(self) -> dict:
        coeffs = set(self._dist_coeffs) if self._ref_dist.isChecked() else set()
        return {
            "Lsd": self._ref_lsd.isChecked(),
            "BC": self._ref_bc.isChecked(),
            "ty": self._ref_ty.isChecked(),
            "tz": self._ref_tz.isChecked(),
            "tx": self._ref_tx.isChecked(),
            "Wavelength": self._ref_wl.isChecked(),
            "Distortion": self._ref_dist.isChecked(),   # legacy/back-compat
            "distortion_coeffs": coeffs,
        }

    def _run(self):
        self._image = self._source_image()
        if self._image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load a calibrant image first."); return
        if self._worker and self._worker.isRunning():
            return
        self._orphans = [o for o in self._orphans if o.isRunning()]   # drop finished ones
        # Dark / bright / background fields (from the loader)
        for sel in self._loader.has_pending_fields():
            QtWidgets.QMessageBox.warning(
                self, "Field not computed",
                f"'{sel.title()}' is enabled but not computed. "
                "Click 'Compute field' in that box first."); return
        self._dark = self._loader.dark()
        bright = self._loader.bright()
        background = self._loader.background()
        bright_mode = self._loader.bright_mode()

        mode = self._pipeline.currentData()
        self._calib_cancelled = False
        self._run_btn.setEnabled(False); self._abort_btn.setEnabled(True)
        self._prog.setVisible(True)
        self._bot_tabs.setCurrentWidget(self._log)
        self._log.append("─" * 40 + f"\nStarting calibration ({mode})…")

        trans = im_trans_codes_from_checkboxes(self._flip_y, self._flip_z, self._transp)

        cfg = {
            "wavelength": self._wl.value(),
            "pxY": self._pxY.value(),
            "pxZ": self._pxZ_spin.value() if self._pxZ_check.isChecked() else None,
            "calibrant": self._cal.currentText(),
            "refine": self._refine_flags(),
            "n_iter": self._n_iter.value(),
            "lm_max_iter": self._lm_iter.value(),
            "device": self._device.currentText(),
            "build_residual_corr": self._build_rc.isChecked(),
            "output_dir": self._out_ed.text().strip() or None,
            "im_trans": trans,
            "mask": self._loader.composite_mask(),
        }
        self._last_dist_coeffs = cfg["refine"]["distortion_coeffs"]
        if self._manual_seed_check.isChecked():
            cfg["manual_seed"] = {
                "BC_y": self._seed_bcy.value(),
                "BC_z": self._seed_bcz.value(),
                "Lsd":  self._seed_lsd.value() * 1000.0,   # mm display → µm
                "tx": self._seed_tx.value(),
                "ty": self._seed_ty.value(),
                "tz": self._seed_tz.value(),
                "distortion": dict(self._seed_dist),
            }
        if self._panel_grp.isChecked():
            cfg["panel_layout"] = {
                "n_y": self._pn_y.value(), "n_z": self._pn_z.value(),
                "sy": self._ps_y.value(), "sz": self._ps_z.value(),
                "gap_y": self._pg_y.value(), "gap_z": self._pg_z.value(),
            }

        manual = cfg.get("manual_seed")
        if manual and any(manual.get(k) for k in ("tx", "ty", "tz")):
            from midas_gui.calib import tilt_seed_effective
            if not tilt_seed_effective(mode, panel_layout=cfg.get("panel_layout"),
                                       refine=cfg["refine"]):
                self._log.append(
                    f"⚠ Tilt seed (tx={manual['tx']:.3f}°, ty={manual['ty']:.3f}°, "
                    f"tz={manual['tz']:.3f}°) will NOT be used: the '{self._pipeline.currentText()}' "
                    "pipeline with these settings ignores an initial tilt guess — tilts "
                    "start from 0° instead. Four-stage / Bayesian / Joint pipelines (or "
                    "One-shot with Multi-panel, or with Distortion refinement restricted "
                    "to a coefficient subset) do honour a tilt seed.")

        self._last_cfg = dict(cfg)
        self._last_bright = bright
        self._last_background = background

        self._worker = CalibrationWorker(
            mode, self._calib_image(), self._dark, cfg, parent=self,
            bright=bright, background=background, bright_mode=bright_mode)
        self._worker.log_line.connect(self._log.append)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _abort(self):
        """Abort the running calibration and free the slot immediately.

        The pipeline is one uninterruptible library call, so we cannot stop it
        cleanly mid-flight — and ``terminate()`` on a thread inside native
        torch/scipy code can corrupt the interpreter.  So we *detach* instead:
        disconnect the worker's signals (its result is discarded), orphan the thread
        (kept alive so its QObject isn't GC'd while the C thread winds down on its
        own), and clear ``self._worker`` so a fresh run can start right away. The
        worker restores stdout/stderr itself, guarded so it won't clobber a new run."""
        w = self._worker
        if not (w and w.isRunning()):
            return
        self._calib_cancelled = True
        for sig in (w.log_line, w.finished, w.failed):
            try:
                sig.disconnect()
            except Exception:
                pass
        w.requestInterruption()       # honoured if/when the library call yields
        self._orphans.append(w)
        self._worker = None           # free the slot so _run can start again now
        self._run_btn.setEnabled(True)
        self._on_dspacing_picks_changed()   # restore correct manual-fit button state
        self._abort_btn.setEnabled(False); self._abort_btn.setText("Abort")
        self._prog.setVisible(False)
        self._log.append("Calibration aborted — you can start a new run now "
                         "(a background thread may still be winding down).")

    def _populate_param_grid(self, pairs, ncols=3):
        """Lay (key, value) pairs into ``ncols`` columns as plain text, filled
        column-major so each column reads top-to-bottom in file order. The paramstest
        distortion slots p0–p14 are relabelled with their coefficient names
        (iso_R2, a1, …) so the distortion reads clearly without a separate table."""
        import math
        from midas_gui.widgets import _mono_font
        from midas_gui.helpers import _PARAMSTEST_DISTORTION   # p#-slot → v2 name
        grid = self._param_grid
        while grid.count():                       # clear previous run
            item = grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        mono = _mono_font(12)
        klbl = "font-weight:600; font-size:12px;"
        n = len(pairs); nrows = max(1, math.ceil(n / ncols))
        for idx, (key, val) in enumerate(pairs):
            col, row = idx // nrows, idx % nrows
            label = _PARAMSTEST_DISTORTION.get(key, key)   # name distortion slots
            k = QtWidgets.QLabel(f"{label}:"); k.setStyleSheet(klbl)
            v = QtWidgets.QLabel(val); v.setFont(mono)
            v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            grid.addWidget(k, row, col * 2, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(v, row, col * 2 + 1, QtCore.Qt.AlignVCenter)
        grid.setColumnStretch(ncols * 2 + 1, 1)

    def _send_to_viewer(self):
        """Push the calibrated geometry to the Data Viewer (µm internal; the Viewer
        converts Lsd to its mm display)."""
        r = self._result
        if r is None:
            return
        self.sendGeometryToViewer.emit({
            "wavelength_A": float(r.wavelength_A), "pxY": float(r.pxY),
            "pxZ": float(getattr(r, "pxZ", r.pxY) or r.pxY),
            "Lsd": float(r.Lsd), "BC_y": float(r.BC_y), "BC_z": float(r.BC_z),
            "tx": float(getattr(r, "tx", 0.0) or 0.0),
            "ty": float(getattr(r, "ty", 0.0) or 0.0),
            "tz": float(getattr(r, "tz", 0.0) or 0.0),
            "NrPixelsY": int(getattr(r, "NrPixelsY", 0) or 0),
            "NrPixelsZ": int(getattr(r, "NrPixelsZ", 0) or 0),
            "distortion": dict(getattr(r, "distortion", {}) or {}),
            "im_trans": list(getattr(r, "im_trans", []) or [])})
        self._log.append("Sent calibrated geometry (incl. tilts + distortion) "
                         "to the Data Viewer.")

    def _on_done(self, result):
        if self._calib_cancelled:
            return   # user aborted — ignore the late result
        result.im_trans = im_trans_codes_from_checkboxes(
            self._flip_y, self._flip_z, self._transp)
        self._result = result
        if self._feedback_check.isChecked():
            try:
                self._seed_from_result(result)
            except Exception:
                pass   # feedback is best-effort; never block the result display
        self._run_btn.setEnabled(True); self._abort_btn.setEnabled(False)
        self._prog.setVisible(False)
        try:
            self._populate_param_grid(
                paramstest_pairs(result, selected=self._last_dist_coeffs))
        except Exception:
            import traceback as _tb
            self._log.append("Could not render parameter grid:\n" + _tb.format_exc())
        s = result.post_residual_strain_uE
        seed_s = getattr(result, "seed_seconds", 0.0) or 0.0
        ref_s  = getattr(result, "refine_seconds", 0.0) or 0.0
        strain_txt = f"{s:.1f} µε" if s else "n/a"
        self._r_diag.setText(f"Post-refine strain: {strain_txt}    ·    "
                             f"timing: seed={seed_s:.1f} s, refine={ref_s:.1f} s")
        self._to_view_btn.setEnabled(True)
        self._save_json_btn.setEnabled(True)
        self._save_ps_btn.setEnabled(True)
        self._log.append(f"Done — Lsd={result.Lsd/1000:.3f} mm"
                         + (f"  strain={s:.0f} µε" if s else ""))
        # Bayesian: report per-parameter σ if present
        lap = getattr(result, "_laplace_sigma", None)
        if lap:
            self._log.append("Laplace 1σ per parameter:")
            for name, sigma in lap.items():
                self._log.append(f"    {name:12s} ± {sigma:.4g}")
        self._draw_rings(result)
        self._bot_tabs.setCurrentWidget(self._prof_view)
        self._pending_log_result = result
        self._run_integration(result)
        if not (self._int_worker and self._int_worker.isRunning()):
            # No image loaded (or integration otherwise didn't start) — log
            # now, without cake/profile results.
            self._pending_log_result = None
            self._log_to_project(result)
        self.calibrationDone.emit(result)

    def _log_to_project(self, result, results: Optional[dict] = None):
        """Append a provenance record to the currently-open project file, if
        any — a no-op (never blocks the result display) when no project is
        open or the write fails for any reason. ``results`` (when given) is
        the last IntegrationWorker payload — cake/profile arrays get
        embedded alongside the calibration record."""
        if not self._project_ctx or not self._project_ctx.path:
            return
        try:
            mask = (self._last_cfg or {}).get("mask")
            mask_is_file_backed = mask is not None and not self._loader.has_live_mask_source()
            ref = project.append_calibration_attempt(
                self._project_ctx.path, "single",
                cfg=self._last_cfg, result=result,
                loader_state=self._loader.get_state(),
                mask_is_file_backed=mask_is_file_backed,
                results=results,
                extra={"active_profile": settings.active_profile()})
            result._project_attempt_ref = ref
            self._log.append(f"Logged to project: {ref}")
        except Exception:
            import traceback as _tb
            self._log.append("Could not log to project file:\n" + _tb.format_exc())

    def _on_fail(self, msg):
        if self._calib_cancelled:
            return   # user aborted — ignore the late failure
        self._run_btn.setEnabled(True); self._abort_btn.setEnabled(False)
        self._prog.setVisible(False)
        show_error(self, "Calibration failed", msg, log=self._log, log_prefix="\nERROR:\n")

    # ── Rings ──────────────────────────────────────────────────────

    def _draw_rings(self, result):
        self._calib_result = result
        for item in self._ring_items:
            self._img_view._iv.removeItem(item)
        self._ring_items.clear()
        self._clear_corrected_rings()
        radii = _predict_ring_radii(result)
        visible = self._show_rings_check.isChecked()
        th = np.linspace(0, 2 * math.pi, 512)
        pen = pg.mkPen("lime", width=1.2)
        max_r = max(result.NrPixelsY, result.NrPixelsZ)
        for r in radii:
            if 0 < r < max_r:
                item = pg.PlotDataItem(result.BC_y + r * np.cos(th),
                                       result.BC_z + r * np.sin(th), pen=pen)
                item.setVisible(visible)
                self._img_view._iv.addItem(item); self._ring_items.append(item)
        bc = pg.ScatterPlotItem([result.BC_y], [result.BC_z], symbol="o", size=10,
                                pen=pg.mkPen("yellow", width=2), brush=pg.mkBrush("red"))
        bc.setVisible(visible)
        self._img_view._iv.addItem(bc); self._ring_items.append(bc)
        if self._corrected_check.isChecked():
            self._draw_corrected_rings(radii)

    def _on_show_rings_toggled(self, visible):
        active = (self._corrected_ring_items
                  if (self._corrected_check.isChecked() and self._corrected_ring_items)
                  else self._ring_items)
        for item in active:
            item.setVisible(visible)

    def _on_corrected_rings_toggled(self, checked):
        if self._calib_result is None:
            return
        if checked:
            for item in self._ring_items:
                item.setVisible(False)
            if self._corrected_ring_items:
                vis = self._show_rings_check.isChecked()
                for item in self._corrected_ring_items:
                    item.setVisible(vis)
            else:
                self._draw_corrected_rings(_predict_ring_radii(self._calib_result))
        else:
            for item in self._corrected_ring_items:
                item.setVisible(False)
            vis = self._show_rings_check.isChecked()
            for item in self._ring_items:
                item.setVisible(vis)
            self._corr_status.setText("")

    def _clear_corrected_rings(self):
        for item in self._corrected_ring_items:
            self._img_view._iv.removeItem(item)
        self._corrected_ring_items.clear()

    def _draw_corrected_rings(self, radii_px):
        if self._calib_result is None:
            return
        result = self._calib_result
        self._clear_corrected_rings()
        vis = self._show_rings_check.isChecked()
        pen = pg.mkPen("lime", width=1.2)
        n = 0
        for r in radii_px:
            try:
                two_theta_deg = math.degrees(math.atan(r * result.pxY / result.Lsd))
                ys, zs = tilted_ring_xy(two_theta_deg, result.tx, result.ty, result.tz,
                                        result.Lsd, result.BC_y, result.BC_z,
                                        result.pxY, result.pxZ)
            except Exception:
                import traceback
                self._log.append(f"Corrected ring error:\n{traceback.format_exc()}")
                continue
            item = pg.PlotDataItem(ys, zs, pen=pen)
            item.setVisible(vis)
            self._img_view._iv.addItem(item); self._corrected_ring_items.append(item)
            n += 1
        self._corr_status.setText(f"Corrected rings: {n} shown")
        for item in self._ring_items:
            item.setVisible(False)

    # ── Integration / residual chart ───────────────────────────────

    def _run_integration(self, result):
        image = self._calib_image()
        if image is None:
            return
        if self._int_worker and self._int_worker.isRunning():
            return
        im_trans = tuple(im_trans_codes_from_checkboxes(
            self._flip_y, self._flip_z, self._transp))
        self._int_worker = IntegrationWorker(
            result, image, self._loader.dark(), im_trans,
            r_bin=self._cal_r_bin.value(), eta_bin=self._cal_eta_bin.value(),
            mask=self._loader.composite_mask(), parent=self,
            bright=self._loader.bright(), background=self._loader.background(),
            bright_mode=self._loader.bright_mode(),
            weighted=bool(self._cal_azim.currentData()))
        self._int_worker.log_line.connect(self._log.append)
        self._int_worker.finished.connect(self._on_int_done)
        self._int_worker.failed.connect(self._on_int_failed)
        self._int_worker.start()

    def _reintegrate(self):
        if self._result is not None:
            self._run_integration(self._result)

    def _flush_pending_log(self, results: Optional[dict]):
        if self._pending_log_result is not None:
            pending, self._pending_log_result = self._pending_log_result, None
            self._log_to_project(pending, results=results)

    def _on_int_failed(self, msg: str):
        self._log.append(f"Integration error: {msg}")
        self._flush_pending_log(None)

    def _on_int_done(self, data):
        self._prof_view.set_profile(
            data["r_axis_px"], data["profile"],
            wavelength_A=data["wavelength_A"], lsd_um=data["lsd_um"], px_um=data["px_um"])
        if data.get("cake_2d") is not None:
            self._cake_view.set_cake(data["cake_2d"], data["r_axis_px"], data["eta_axis_deg"])
        radii = _predict_ring_radii(self._result) if self._result else []
        if data.get("cake_2d") is not None and radii:
            ring_grid, kept_radii = ring_azimuth_residual(
                data["cake_2d"], data["r_axis_px"], radii)
            self._resid_cake_view.set_data(
                data.get("resid_cake"), ring_grid, kept_radii,
                data["r_axis_px"], data["eta_axis_deg"],
                profile=data["profile"], all_ring_radii_px=radii)
        else:
            self._resid_cake_view.clear()
        if self._result:
            self._prof_view.set_ring_markers(
                [{"radii": radii, "color": "#f0c060"}],
                data["lsd_um"], data["px_um"], data["wavelength_A"])
        self._flush_pending_log(data)

    # ── Save ───────────────────────────────────────────────────────

    def _save_json(self):
        if not self._result: return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save calibration.json", "calibration.json", "JSON (*.json)")
        if not path: return
        import json
        d = {k: v for k, v in vars(self._result).items()
             if not k.startswith("_") and not hasattr(v, "numpy")}
        d.pop("residual_corr_map", None); d.pop("iter_history", None)
        panel_u = getattr(self._result, "_panel_unpacked", None)
        ps_note = ""
        if panel_u and d.get("panel_layout"):
            # Re-write the shifts sidecar next to wherever this JSON actually
            # lands — d["panel_shifts_path"] may still point at the tempfile
            # calib._attach_panel_result fell back to when no Output folder
            # was set during Fit, which is not guaranteed to persist or to
            # travel with this file. Same convention as MIDAS's own
            # write_v1_paramstest: <stem>_panelshifts.txt beside the file
            # that describes the instrument. Best-effort: a result restored
            # from a project attempt carries a JSON round-tripped (string,
            # not tensor) ``_panel_unpacked``, which this can't rewrite from
            # — fall back to whatever panel_shifts_path is already on the
            # result (already resolved to a real file by the project-open
            # flow in that case) rather than failing the whole save.
            try:
                from midas_calibrate_v2.compat.to_v1 import write_panel_shifts_file
                ps_path = Path(path).with_name(Path(path).stem + "_panelshifts.txt")
                write_panel_shifts_file(panel_u, ps_path)
                d["panel_shifts_path"] = str(ps_path)
                ps_note = f"\npanel shifts saved: {ps_path}"
                self._log.append(f"Panel shifts saved: {ps_path}")
            except Exception:
                import traceback
                self._log.append(f"Panel shifts save error (kept existing "
                                  f"panel_shifts_path):\n{traceback.format_exc()}")
        Path(path).write_text(json.dumps(d, indent=2, default=str))
        self._log.append(f"Saved: {path}")
        QtWidgets.QMessageBox.information(
            self, "Saved", f"calibration.json saved:\n{path}{ps_note}")

    def _save_paramstest(self):
        if not self._result:
            return
        dlg = _SaveParamstestDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        out_path = dlg.out_path()
        if not out_path:
            QtWidgets.QMessageBox.warning(self, "No output", "Please specify an output file."); return
        tmpl_path = dlg.template_path()
        panel_u = getattr(self._result, "_panel_unpacked", None)
        panel_layout = getattr(self._result, "panel_layout", None)
        # <stem>_panelshifts.txt, not a generic "panel_shifts.txt" — saving
        # more than one paramstest into the same output folder would
        # otherwise have every one of them share (and overwrite) a single
        # sidecar, silently swapping in whichever calibration saved last.
        ps_path = (Path(out_path).with_name(Path(out_path).stem + "_panelshifts.txt")
                   if panel_u else None)

        def _gap_str(g, n):
            # Same uniform-gap expansion as PanelLayout.regular/helpers._apply_panel_fields.
            vals = g if isinstance(g, (list, tuple)) else [int(g)] * max(int(n) - 1, 0)
            return " ".join(str(int(v)) for v in vals)

        panel_grid_lines = []
        panel_grid_extra = {}
        if panel_layout:
            n_y, n_z = int(panel_layout["n_y"]), int(panel_layout["n_z"])
            panel_grid_extra = {
                "NPanelsY": n_y, "NPanelsZ": n_z,
                "PanelSizeY": int(panel_layout["sy"]), "PanelSizeZ": int(panel_layout["sz"]),
                "PanelGapsY": _gap_str(panel_layout.get("gap_y", 0), n_y),
                "PanelGapsZ": _gap_str(panel_layout.get("gap_z", 0), n_z),
            }
            panel_grid_lines = [f"{k} {v}" for k, v in panel_grid_extra.items()]
        try:
            if tmpl_path:
                if not Path(tmpl_path).exists():
                    raise FileNotFoundError(f"Template not found: {tmpl_path}")
                from midas_calibrate_v2.compat.to_v1 import ff_paramstest_from_auto_result
                ff_paramstest_from_auto_result(self._result, tmpl_path, out_path)
                # Append panel grid + PanelShiftsFile so downstream tools (the GUI's own
                # spec builder, or midas_integrate_v2 standalone) know where each panel
                # sits, not just its refined shift.
                if panel_grid_lines:
                    with open(out_path, "a") as _f:
                        for line in panel_grid_lines:
                            _f.write(line + "\n")
                if ps_path:
                    with open(out_path, "a") as _f:
                        _f.write(f"PanelShiftsFile {ps_path}\n")
                im_trans = getattr(self._result, "im_trans", None)
                if im_trans:
                    with open(out_path, "a") as _f:
                        for code in im_trans:
                            _f.write(f"ImTransOpt {int(code)}\n")
                mode = "from template"
            else:
                from midas_calibrate.params import CalibrationParams
                from midas_gui.helpers import write_standalone_paramstest
                result = self._result
                extra = dict(panel_grid_extra)
                rcm = getattr(result, "residual_corr_bin_path", None)
                if rcm and getattr(result, "residual_corr_map", None) is not None:
                    extra["ResidualCorrectionMap"] = rcm
                if ps_path:
                    extra["PanelShiftsFile"] = str(ps_path)
                write_standalone_paramstest(result, out_path, extra=extra)
                mode = "standalone"
            self._log.append(f"paramstest.txt saved ({mode}): {out_path}")
            # Save companion panel_shifts.txt if calibration refined panel shifts
            ps_saved = ""
            if panel_u and ps_path:
                try:
                    from midas_calibrate_v2.compat.to_v1 import write_panel_shifts_file
                    write_panel_shifts_file(panel_u, ps_path)
                    self._log.append(f"Panel shifts saved: {ps_path}")
                    ps_saved = f"\npanel_shifts.txt: {ps_path}"
                except Exception:
                    import traceback
                    self._log.append(f"Panel shifts save error:\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.information(
                self, "Saved",
                f"paramstest.txt saved ({mode}):\n{out_path}{ps_saved}")
        except Exception as e:
            import traceback
            self._log.append(f"Save paramstest error:\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def get_result(self):
        return self._result

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def _state_widgets(self) -> dict:
        return {
            "pipeline": self._pipeline,
            "wl": self._wl,
            "cal": self._cal,
            "pxY": self._pxY,
            "pxZ_check": self._pxZ_check,
            "pxZ_spin": self._pxZ_spin,
            "flip_y": self._flip_y,
            "flip_z": self._flip_z,
            "transp": self._transp,
            "thr_check": self._thr_check,
            "thr_min": self._thr_min,
            "thr_max": self._thr_max,
            "avg_check": self._avg_check,
            "avg_start": self._avg_start,
            "avg_end": self._avg_end,
            "manual_seed_check": self._manual_seed_check,
            "seed_bcy": self._seed_bcy,
            "seed_bcz": self._seed_bcz,
            "seed_lsd": self._seed_lsd,
            "seed_tx": self._seed_tx,
            "seed_ty": self._seed_ty,
            "seed_tz": self._seed_tz,
            "feedback_check": self._feedback_check,
            "ref_lsd": self._ref_lsd,
            "ref_bc": self._ref_bc,
            "ref_ty": self._ref_ty,
            "ref_tz": self._ref_tz,
            "ref_tx": self._ref_tx,
            "ref_wl": self._ref_wl,
            "ref_dist": self._ref_dist,
            "build_rc": self._build_rc,
            "n_iter": self._n_iter,
            "lm_iter": self._lm_iter,
            "device": self._device,
            "out_ed": self._out_ed,
            "pn_y": self._pn_y,
            "pn_z": self._pn_z,
            "ps_y": self._ps_y,
            "ps_z": self._ps_z,
            "pg_y": self._pg_y,
            "pg_z": self._pg_z,
            "show_rings_check": self._show_rings_check,
            "corrected_check": self._corrected_check,
            "cal_r_bin": self._cal_r_bin,
            "cal_eta_bin": self._cal_eta_bin,
            "cal_azim": self._cal_azim,
        }

    def get_state(self, sidecar_stem: Optional[str] = None) -> dict:
        """``sidecar_stem`` (if given) is the state file's path without its
        extension. A fitted result that hasn't been exported via "Save JSON" is
        written to ``<sidecar_stem>_calibration.json`` for the record and to
        reseed the geometry fields on load — but is NOT reconstructed into a
        live ``self._result`` object (there is no loader for that; re-running
        Fit with the restored seed fields reproduces it)."""
        state = {"fields": widgets_to_dict(self._state_widgets()),
                 "loader": self._loader.get_state(),
                 "img_view": self._img_view.display_state(),
                 "cake_view": self._cake_view.display_state(),
                 "hydra": {"active_mode": self._mode_ribbon.mode(),
                           "page": self._hydra_page.get_state()}}
        if self._result is not None and sidecar_stem:
            try:
                import json
                d = {k: v for k, v in vars(self._result).items()
                     if not k.startswith("_") and not hasattr(v, "numpy")}
                d.pop("residual_corr_map", None); d.pop("iter_history", None)
                Path(f"{sidecar_stem}_calibration.json").write_text(
                    json.dumps(d, indent=2, default=str))
            except Exception:
                pass
        return state

    def set_state(self, state: dict, sidecar_stem: Optional[str] = None) -> None:
        apply_dict_to_widgets(self._state_widgets(), state.get("fields", {}))
        self._loader.set_state(state.get("loader") or {})
        self._img_view.set_display_state(state.get("img_view"))
        self._cake_view.set_display_state(state.get("cake_view"))
        hydra_state = state.get("hydra") or {}
        self._mode_ribbon.set_mode(hydra_state.get("active_mode", "single"))
        self._hydra_page.set_state(hydra_state.get("page") or {})

    # ── File > Open Project… ─────────────────────────────────────────

    def _display_stored_result(self, result, results_arrays: Optional[dict] = None) -> None:
        """Redraw rings + the radial profile/cake for a result recovered
        from a project attempt — same visual effects as a live Fit's
        ``_on_done``, without re-running Fit. When ``results_arrays`` (the
        attempt's embedded cake/profile, see
        ``project.read_calib_attempt_results``) is available, the plots are
        populated directly from it — no recompute needed. Otherwise, falls
        back to live re-integration if an image happens to be loaded.
        Best-effort per step so a partially-available result (e.g. the
        source image no longer on disk) still shows whatever it can."""
        self._result = result
        try:
            self._populate_param_grid(paramstest_pairs(result))
            self._to_view_btn.setEnabled(True)
            self._save_json_btn.setEnabled(True)
            self._save_ps_btn.setEnabled(True)
        except Exception:
            pass
        try:
            self._draw_rings(result)
        except Exception:
            pass
        if results_arrays and results_arrays.get("profile") is not None:
            try:
                self._bot_tabs.setCurrentWidget(self._prof_view)
                self._prof_view.set_profile(
                    results_arrays["r_axis_px"], results_arrays["profile"],
                    wavelength_A=results_arrays.get("wavelength_A"),
                    lsd_um=results_arrays.get("lsd_um"), px_um=results_arrays.get("px_um"))
                if results_arrays.get("cake_2d") is not None:
                    self._cake_view.set_cake(
                        results_arrays["cake_2d"], results_arrays["r_axis_px"],
                        results_arrays["eta_axis_deg"])
                radii = _predict_ring_radii(result)
                if results_arrays.get("cake_2d") is not None and radii:
                    ring_grid, kept_radii = ring_azimuth_residual(
                        results_arrays["cake_2d"], results_arrays["r_axis_px"], radii)
                    self._resid_cake_view.set_data(
                        results_arrays.get("resid_cake"), ring_grid, kept_radii,
                        results_arrays["r_axis_px"], results_arrays["eta_axis_deg"],
                        profile=results_arrays["profile"], all_ring_radii_px=radii)
                else:
                    self._resid_cake_view.clear()
                self._prof_view.set_ring_markers(
                    [{"radii": radii, "color": "#f0c060"}],
                    results_arrays.get("lsd_um"), results_arrays.get("px_um"),
                    results_arrays.get("wavelength_A"))
            except Exception:
                pass
        elif self._image is not None:
            self._bot_tabs.setCurrentWidget(self._prof_view)
            try:
                self._run_integration(result)
            except Exception:
                pass

    def apply_project_calibration(self, attempts: dict) -> None:
        """``attempts`` maps panel key (``"single"`` or ``"ge1"``..``"ge4"``)
        to that panel's calibration-attempt metadata (``project.read_attempt``)
        — called after File > Open Project… when the user opts to populate
        this tab. Reuses ``set_state()``'s existing field-restore machinery
        (widget keys are shared across the single-detector tab, the Hydra
        page's shared recipe, and a Hydra panel card's seed fields — see
        ``project.calib_attempt_gui_fields``), and switches the mode ribbon
        to match what was found."""
        if not attempts:
            return
        single_meta = attempts.get("single")
        hydra_metas = {k: v for k, v in attempts.items() if k != "single"}
        state = {}
        if single_meta is not None:
            state["fields"] = project.calib_attempt_gui_fields(single_meta)
            state["loader"] = project.calib_attempt_loader_state(single_meta)
        if hydra_metas:
            cards, page_fields, anchor_path = {}, {}, None
            for panel_key, meta in sorted(hydra_metas.items()):
                fields = project.calib_attempt_gui_fields(meta)
                cards[int(panel_key[2:])] = fields
                page_fields = fields   # same recipe on every panel; last one wins
                if anchor_path is None:
                    anchor_path = project.calib_attempt_loader_state(meta).get("path")
            state["hydra"] = {"active_mode": "hydra",
                               "page": {"fields": page_fields, "cards": cards,
                                        "anchor_path": anchor_path}}
        elif single_meta is not None:
            state["hydra"] = {"active_mode": "single"}
        self.set_state(state)

        if single_meta is not None and single_meta.get("result"):
            self._display_stored_result(
                project.calibration_namespace(single_meta["result"]),
                single_meta.get("_results_arrays"))
        for panel_key, meta in hydra_metas.items():
            if meta.get("result"):
                self._hydra_page.display_stored_result(
                    int(panel_key[2:]), project.calibration_namespace(meta["result"]),
                    meta.get("_results_arrays"))
