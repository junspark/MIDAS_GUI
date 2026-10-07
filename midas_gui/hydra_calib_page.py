"""Hydra (4-panel GE detector) page for the Calibrate tab.

``HydraCalibrationPage`` mirrors ``hydra_page.HydraViewerPage``'s
composition: a shared ``HydraLoaderPanel`` for data, one small
``HydraCalibPanelCard`` per GE panel (Transforms + seed + fitted result,
each genuinely independent per physical panel — see that module's
docstring), and shared "recipe" cards (Pipeline, Detector & Calibrant,
Threshold, Mean of frames, Refine parameters, Advanced) applied identically
to every panel's fit, since it's the same beam and the same choice of what
to refine for all 4.

Calibration for the panels currently loaded can run Sequentially (one
``CalibrationWorker`` at a time — full per-line log capture, safe) or in
Parallel (all workers started at once — see ``workers.CalibrationWorker``'s
``capture_stdout`` flag for why parallel runs skip fine-grained log capture).

Deliberately does NOT surface ``HydraLoaderPanel.projection_card()`` — the
single-detector Calibrate tab has no stack-projection feature either (it
only offers a frame mean), and ``HydraLoaderPanel.projected(n)`` returns
an *already* dark/bright/background-corrected frame, which would be
double-corrected if handed to ``CalibrationWorker`` (which expects a raw
frame and applies bright/background itself). Taking a frame mean instead
(the "Mean of frames" card below) mirrors the single-detector tab exactly.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui.constants import (
    CALIBRANTS, PIPELINES, DEFAULT_PIPELINE, DEFAULT_WAVELENGTH, DEFAULT_PIXEL_UM,
    DISTORTION_NAMES)
from midas_gui.helpers import (
    _fspin, _NoScrollSpinBox, _NoScrollComboBox, make_kedge_label, make_pixel_label,
    _load_image, apply_field_corrections, average_field, source_kind,
    widgets_to_dict, apply_dict_to_widgets, _predict_ring_radii, refresh_combo_items,
    browse_start_dir, warn_if_path_missing, suggest_working_dir,
    check_output_dir_writable, scratch_dir, SCRATCH_DIRNAME)
from midas_gui.widgets import (PickableImageViewer, LogPanel, CakeViewer, _convert_radial,
                               OriginToolButton)
from midas_gui.hydra_widgets import HydraLoaderPanel, HydraDetectorToolbar, HydraProfileViewer
from midas_gui.hydra_calib_widgets import HydraCalibPanelCard
from midas_gui.workers import CalibrationWorker, IntegrationWorker
from midas_gui.dialogs import DistortionRefineDialog
from midas_gui import project
from midas_gui import settings
from midas_gui import style as S


def _resample_rows_to_eta_grid(cake: np.ndarray, src_eta: np.ndarray,
                               dst_eta: np.ndarray) -> np.ndarray:
    """Redistribute ``cake``'s rows from ``src_eta`` bin centres onto
    ``dst_eta`` bin centres via per-column linear interpolation, periodic in
    η so it's correct across the ±180° seam. Purely a defensive alignment
    step for ``_compose_overall_cake`` (see its docstring): every panel's
    cake there is already expressed in the shared/world η frame by the
    backend, so in the normal case (every panel integrated with the same
    page-level ``EtaMin``/``EtaMax``/``EtaBinSize``) ``src_eta`` and
    ``dst_eta`` are numerically identical and this is a no-op; it only does
    real work if two panels' cakes were captured at different times with
    different η-bin settings. Unlike the R-axis resampling below, no
    NaN-outside-range guard is needed here: a panel's cake already spans the
    *full* -180°..180° grid with real zero-count bins wherever it saw no
    data (``midas_integrate_v2``'s binning zero-fills, never NaN-fills — see
    DECISIONS.md 2026-08-26), so every output row has a genuine value to
    interpolate from."""
    src_eta = np.asarray(src_eta, dtype=np.float64)
    order = np.argsort(src_eta)
    src_sorted = src_eta[order]
    cake_sorted = np.asarray(cake, dtype=np.float64)[order, :]
    ext_eta = np.concatenate([src_sorted - 360.0, src_sorted, src_sorted + 360.0])
    ext_cake = np.concatenate([cake_sorted, cake_sorted, cake_sorted], axis=0)
    out = np.empty((len(dst_eta), cake.shape[1]), dtype=np.float32)
    for j in range(cake.shape[1]):
        out[:, j] = np.interp(dst_eta, ext_eta, ext_cake[:, j])
    return out


def _compose_overall_cake(panels: dict) -> Optional[tuple]:
    """Sum the available panels' (η, R) cakes into one "Overall" cake,
    covering the full -180°..180° η range instead of each panel's own
    narrower wedge landing on top of the others.

    ``panels`` maps panel number -> ``(cake_2d, r_axis_px, eta_axis_deg,
    lsd_um, px_um, wavelength_A)`` (see ``HydraCalibrationPage._on_int_done``,
    which caches this per panel). Returns ``(cake, r_axis, eta_axis)`` or
    ``None`` if no panel has been integrated yet.

    **No η rotation is applied here, deliberately.** Each panel's own cake
    is already expressed in the shared/world η frame at the point
    ``IntegrationWorker`` produces it: ``helpers._build_spec`` always sets
    ``spec.tx = result.tx`` (calibration never refines tx — see
    ``calib._refine_dict``, which doesn't even include it — so ``result.tx``
    is exactly whatever was seeded: 0.0 by default, or a panel's real,
    distinct installation angle if one was loaded/pulled from the Data
    Viewer before Fit), and the backend's
    ``midas_calibrate_v2.forward.geometry.pixel_to_REta`` applies that
    ``tx`` (via ``build_tilt_matrix``, the same rotation convention —
    verified same sign/handedness — as ``hydra.compute_inv_coords``, which
    places panels into the Data Viewer's windmill composite) *before*
    computing ``eta = atan2(-Y, Z)``. So a panel calibrated with its real
    tx already lands in the same world frame the composite image uses; a
    second rotation here would double-count it. (A prior version of this
    function did exactly that — see DECISIONS.md.)

    A practical consequence: Overall is only physically meaningful once
    each panel has been calibrated with its own true, distinct installation
    ``tx`` fixed as a (possibly-unrefined) seed. If every panel is left at
    the 0.0 default, every cake is genuinely computed as if unrotated, and
    Overall correctly shows one wedge with all panels' signal piled onto
    it — that reflects missing placement information, not a bug this
    function can compensate for after the fact.

    Every panel already shares one η bin grid too (``EtaMin``/``EtaMax``/
    ``EtaBinSize`` are one page-level control used for every panel's Fit),
    so ``_resample_rows_to_eta_grid`` below is normally a no-op; it only
    does real work if two panels' cakes were captured at different times
    with different η-bin settings. The existing R-axis handling is
    unchanged and still needed: each panel's R axis is converted to 2θ
    using **that panel's own** geometry (lsd/px/wavelength can differ per
    panel — 2θ itself is tx-invariant when ty=tz=0, so this part was never
    affected by the rotation bug), resampled onto one shared 2θ grid, and
    ``np.nansum``'d — the R axis genuinely differs per panel (each panel's
    own beam-centre/detector-corner distance) so NaN-outside-range tracking
    still applies there."""
    if not panels:
        return None
    eta_axis = np.asarray(next(iter(panels.values()))[2], dtype=np.float64)
    n_eta = len(eta_axis)
    tth_grids, cakes, refs = [], [], []
    for cake, r_px, eta, lsd, px, wl in panels.values():
        eta_aligned_cake = _resample_rows_to_eta_grid(np.asarray(cake), np.asarray(eta), eta_axis)
        tth = _convert_radial(np.asarray(r_px), lsd, px, wl, "R", "2th")
        order = np.argsort(tth)
        tth_grids.append(tth[order])
        cakes.append(eta_aligned_cake[:, order])
        refs.append((lsd, px, wl))
    lo = min(g.min() for g in tth_grids)
    hi = max(g.max() for g in tth_grids)
    n_common = max(c.shape[1] for c in cakes)
    common = np.linspace(lo, hi, n_common)
    resampled = []
    for tth, cake in zip(tth_grids, cakes):
        rows = np.full((n_eta, n_common), np.nan, dtype=np.float32)
        for i in range(n_eta):
            rows[i, :] = np.interp(common, tth, cake[i, :], left=np.nan, right=np.nan)
        resampled.append(rows)
    stacked = np.stack(resampled, axis=0)
    all_nan = np.all(np.isnan(stacked), axis=0)
    summed = np.where(all_nan, np.nan, np.nansum(stacked, axis=0))
    ref_lsd, ref_px, _ref_wl = refs[0]
    r_ref = ref_lsd * np.tan(np.radians(common)) / ref_px
    return summed, r_ref, eta_axis


class HydraCalibrationPage(QtWidgets.QWidget):
    pullFromViewer = QtCore.pyqtSignal()               # "← Data Viewer" clicked
    sendGeometryToViewer = QtCore.pyqtSignal(int, dict)  # panel_num, geometry
    panelCalibrationDone = QtCore.pyqtSignal(int, object)  # panel_num, AutoCalibrationResult

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict = {}          # panel_num -> HydraCalibPanelCard
        self._workers: dict = {}        # panel_num -> CalibrationWorker (running)
        self._int_workers: dict = {}    # panel_num -> IntegrationWorker (running)
        self._orphans: list = []        # aborted workers kept alive until they wind down
        self._pending_panels: list = []  # sequential-mode queue
        self._calib_cancelled = False
        self._dist_coeffs = set(DISTORTION_NAMES)
        self._last_dist_coeffs: Optional[set] = None
        self._active_card: Optional[HydraCalibPanelCard] = None
        self._disp_key = None
        self._last_cfgs: dict = {}      # panel_num -> cfg used for its last run (provenance)
        self._pending_log_results: dict = {}  # panel_num -> result awaiting _log_to_project
        self._composite_log_pending = False   # True during a live run, until it fully finishes
        self._project_ctx: Optional[project.ProjectContext] = None
        self._expid_provider = None     # forwarded by CalibrationTab; see there
        self._run_scratch_id = ""       # one per Run All, shared by its panels
        self._wd_declined = ""          # last unwritable candidate, logged once
        self._build_ui()
        self._on_panel_changed(self._toolbar.current())

    def set_project_context(self, ctx: "project.ProjectContext"):
        self._project_ctx = ctx

    def refresh_calibrants(self) -> None:
        """Repopulate this page's shared Calibrant dropdown from the
        just-activated profile's constants.CALIBRANTS."""
        refresh_combo_items(self._cal, CALIBRANTS)

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(0)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False); split.setHandleWidth(6)
        root.addWidget(split)

        # ── LEFT: "← Data Viewer" + the Hydra data loader ──
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(4)
        self._from_view_btn = QtWidgets.QPushButton("← Data Viewer")
        self._from_view_btn.setToolTip(
            "Pull the Hydra panel data path and each panel's fitted geometry "
            "(BC/Lsd/tilts/transforms) from the Data Viewer's Hydra page.")
        self._from_view_btn.clicked.connect(self.pullFromViewer.emit)
        ll.addWidget(self._from_view_btn)
        self._loader = HydraLoaderPanel()
        self._loader.setMinimumWidth(200)
        self._loader.siblingsChanged.connect(self._on_siblings_changed)
        self._loader.frameChanged.connect(self._on_frame_changed)
        self._loader.fieldsChanged.connect(self._on_fields_changed)
        ll.addWidget(self._loader, 1)
        split.addWidget(left)

        # ── MIDDLE: shared "recipe" cards + per-panel card stack ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setMinimumWidth(260)
        inner = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(inner); lv.setContentsMargins(2, 2, 2, 2); lv.setSpacing(8)
        scroll.setWidget(inner)

        # The model, stated rather than implied. Asked for at the beamline:
        # "what does it mean shared parameters? Are ge1 - GE4 in hydra
        # sharing one set of parameters?" -- the cards below were already
        # marked "(shared)", and that was exactly what misled: "shared"
        # reads as one value for all four detectors, i.e. a joint fit, when
        # it means one recipe driving four independent fits.
        banner = QtWidgets.QLabel(
            "<b>Four independent fits — one per panel.</b> The panels are "
            "NOT treated as a single detector. Cards marked <i>same for all "
            "4 fits</i> set the inputs each fit uses; the fitted geometry "
            "(BC, Lsd, ty, tz, distortion) is per-panel and lives on the "
            "ge1–ge4 card below, as does each panel's result.")
        banner.setWordWrap(True)
        banner.setStyleSheet(
            f"color:{S.MUTED};font-size:10px;border:1px solid {S.ACCENT};"
            "border-radius:3px;padding:5px;")
        lv.addWidget(banner)

        # Pipeline
        pipe = S.make_card("Pipeline  (same for all 4 fits)")
        self._pipeline = _NoScrollComboBox()
        for label, key, enabled in PIPELINES:
            self._pipeline.addItem(label, key)
            if not enabled:
                self._pipeline.model().item(self._pipeline.count() - 1).setEnabled(False)
        _pi = self._pipeline.findData(DEFAULT_PIPELINE)
        if _pi >= 0 and self._pipeline.model().item(_pi).isEnabled():
            self._pipeline.setCurrentIndex(_pi)
        pipe.body.addWidget(self._pipeline)
        lv.addWidget(pipe)

        # Detector & Calibrant (shared — same beam/detector model for all 4 panels)
        det = S.make_card("Detector & Calibrant  (same for all 4 fits)")
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
        lv.addWidget(det)

        # Threshold (shared value; applied to whichever panel's own image is active/fit)
        thr = S.make_card("Threshold  (pixels below → 0, shared)")
        self._thr_check = QtWidgets.QCheckBox("Apply threshold to calibration image")
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

        # Mean of frames (shared range — panels are synchronized frames of one scan)
        avgc = S.make_card("Mean of frames  (shared range)")
        self._avg_check = QtWidgets.QCheckBox("Combine frames into a single mean image")
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

        # Refine parameters (shared)
        refc = S.make_card("Refine parameters  (same for all 4 fits)")
        rfl = QtWidgets.QGridLayout(); rfl.setSpacing(4)
        self._ref_lsd = QtWidgets.QCheckBox("Lsd"); self._ref_lsd.setChecked(True)
        self._ref_bc = QtWidgets.QCheckBox("BC"); self._ref_bc.setChecked(True)
        self._ref_ty = QtWidgets.QCheckBox("ty"); self._ref_ty.setChecked(True)
        self._ref_tz = QtWidgets.QCheckBox("tz"); self._ref_tz.setChecked(True)
        self._ref_tx = QtWidgets.QCheckBox("tx")
        # Not refinable from a powder calibrant, and not by any Hydra route:
        # a Debye-Scherrer pattern is azimuthally symmetric about the beam,
        # so rotating a panel about it maps the rings onto themselves. The
        # backend states the same thing in the gauge language -- with the
        # azimuthal harmonics free, (tx, phi_k) -> (tx + d, phi_k + k*d) is
        # an exact gauge orbit -- and freezes tx in compat/from_v1.py for
        # EVERY powder pipeline. Hydra has no manual d-spacing route (the
        # single-detector tab does, which is why its own tx tick stays
        # live), so here the box could never do anything. Left visible and
        # disabled rather than removed: the parameter is real and seeded
        # per panel, and a silently-missing row invites the question of
        # whether tx is being refined behind your back.
        self._ref_tx.setEnabled(False)
        self._ref_tx.setToolTip(
            "tx cannot be refined from a powder calibrant — the ring pattern "
            "is unchanged by rotating a panel about the beam, so there is "
            "nothing in the data to fit. Supply it per panel in Manual seed "
            "from the installation geometry; it is held at that value. "
            "(Determine it from grain spots or Friedel-pair omega splitting "
            "if you need to measure it.)")
        self._ref_wl = QtWidgets.QCheckBox("Wavelength")
        self._ref_dist = QtWidgets.QCheckBox("Distortion"); self._ref_dist.setChecked(True)
        self._build_rc = QtWidgets.QCheckBox("Residual map"); self._build_rc.setChecked(True)
        for i, w in enumerate((self._ref_lsd, self._ref_bc, self._ref_ty, self._ref_tz,
                               self._ref_tx, self._ref_wl)):
            rfl.addWidget(w, i // 2, i % 2)
        self._dist_btn = QtWidgets.QToolButton(); self._dist_btn.setText("…")
        self._dist_btn.setToolTip("Choose which distortion coefficients to refine.")
        self._dist_btn.clicked.connect(self._edit_distortion_coeffs)
        drow = QtWidgets.QHBoxLayout(); drow.setSpacing(4)
        drow.addWidget(self._ref_dist); drow.addWidget(self._dist_btn); drow.addStretch(1)
        rfl.addLayout(drow, 3, 0)
        rfl.addWidget(self._build_rc, 3, 1)
        self._ref_dist.toggled.connect(lambda _=0: self._update_dist_label())
        refc.body.addLayout(rfl)
        lv.addWidget(refc)
        self._update_dist_label()

        # Advanced (shared)
        grp_adv = QtWidgets.QGroupBox("Advanced")
        grp_adv.setCheckable(True); grp_adv.setChecked(False)
        av = QtWidgets.QVBoxLayout(grp_adv); av.setContentsMargins(8, 6, 8, 6); av.setSpacing(5)
        self._n_iter = _NoScrollSpinBox(); self._n_iter.setRange(1, 1_000_000); self._n_iter.setValue(4)
        self._lm_iter = _NoScrollSpinBox(); self._lm_iter.setRange(1, 1_000_000); self._lm_iter.setValue(200)
        self._device = _NoScrollComboBox(); self._device.addItems(["cpu", "cuda"])
        av.addLayout(S.Form().row(("E-M iters:", self._n_iter), ("LM iters:", self._lm_iter)))
        # Attribute and state key stay `_out_ed` / "out_ed" across the rename
        # to "Working dir" — see CalibrationTab for the same reasoning.
        self._out_ed = QtWidgets.QLineEdit()
        self._out_ed.setPlaceholderText("Working directory…")
        self._out_ed.setToolTip(
            "Working directory shared by all four panels.\n\n"
            f"Each panel's intermediates go in their own {SCRATCH_DIRNAME}/"
            "<run>/ge<N>/ subfolder here, so a parallel run can't have four "
            "panels overwriting one another's files. Delete the subfolder "
            "whenever you like — nothing saved through a Save button is in it."
            "\n\nDefaults to the <expid>_bc analysis folder derived from the "
            "loaded data path.")
        warn_if_path_missing(self._out_ed, self, is_output_dir=True)
        bou = QtWidgets.QPushButton("…"); bou.setFixedWidth(30)
        bou.clicked.connect(lambda: self._out_ed.setText(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Working directory", browse_start_dir(self._out_ed.text())) or ""))
        self._suggest_out_btn = QtWidgets.QPushButton("Suggest")
        self._suggest_out_btn.clicked.connect(self._apply_suggested_working_dir)
        outr = QtWidgets.QHBoxLayout(); outr.setSpacing(4)
        outr.addWidget(self._out_ed, 1); outr.addWidget(bou)
        outr.addWidget(self._suggest_out_btn)
        av.addLayout(S.Form().row(("Device:", self._device)))
        av.addLayout(S.Form().row(("Working dir:", outr)))
        lv.addWidget(grp_adv)

        # Run controls
        run_card = S.make_card("Run  (Hydra: ge1–ge4, one recipe)")
        mode_row = QtWidgets.QHBoxLayout(); mode_row.setSpacing(6)
        mode_row.addWidget(S.LabelRight("Run mode:"))
        self._run_mode_combo = _NoScrollComboBox()
        self._run_mode_combo.addItem("Sequential", "sequential")
        self._run_mode_combo.addItem("Parallel", "parallel")
        self._run_mode_combo.setToolTip(
            "Sequential: one panel fit at a time — full per-line log capture.\n"
            "Parallel: all present panels fit at once — faster, but each panel's "
            "fine-grained progress prints go to the console rather than the Log "
            "tab (only start/finish/error lines appear there); a fit's stdout "
            "redirect is process-global, so it can't safely be shared across "
            "concurrent threads.")
        mode_row.addWidget(self._run_mode_combo); mode_row.addStretch(1)
        run_card.body.addLayout(mode_row)
        self._run_btn = S.primary_btn("Run Calibration")
        self._run_btn.clicked.connect(self._run_all)
        self._abort_btn = QtWidgets.QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._abort_all)
        run_row = QtWidgets.QHBoxLayout(); run_row.setSpacing(6)
        run_row.addWidget(self._run_btn, 1); run_row.addWidget(self._abort_btn)
        run_card.body.addLayout(run_row)
        self._prog = QtWidgets.QProgressBar(); self._prog.setRange(0, 0); self._prog.setVisible(False)
        run_card.body.addWidget(self._prog)
        lv.addWidget(run_card)

        # Which panels are in play, what tx each will use, and which one is
        # running. Asked for at the beamline in one breath: "it is unclear
        # what parameters are dedicated to each GE panel ... which panel is
        # being fitted ... and I do not know what Tx they are getting."
        # The per-panel card below shows one panel at a time, so none of
        # those three were answerable without clicking through all four.
        self._panels_lbl = QtWidgets.QLabel("")
        self._panels_lbl.setWordWrap(True)
        self._panels_lbl.setTextFormat(QtCore.Qt.RichText)
        self._panels_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px;")
        lv.addWidget(self._panels_lbl)

        # Per-panel: Transforms + Initial seed, switched with the active panel
        self._panel_hdr = QtWidgets.QLabel("")
        self._panel_hdr.setStyleSheet(
            f"color:{S.ACCENT};font-size:11px;font-weight:bold;padding-top:4px;")
        lv.addWidget(self._panel_hdr)
        self._card_stack = QtWidgets.QStackedWidget()
        for n in (1, 2, 3, 4):
            card = HydraCalibPanelCard(n)
            card.set_log_fn(self._log_append_raw)
            card.imTransChanged.connect(lambda n=n: self._on_card_transform_changed(n))
            card.calibFileLoaded.connect(self._on_card_calib_file_loaded)
            card.sendToViewer.connect(self.sendGeometryToViewer.emit)
            card._manual_seed_check.toggled.connect(
                lambda checked, n=n: self._sync_seed_checkbox("_manual_seed_check", n, checked))
            card._feedback_check.toggled.connect(
                lambda checked, n=n: self._sync_seed_checkbox("_feedback_check", n, checked))
            for attr in ("_seed_en_bc", "_seed_en_lsd", "_seed_en_tx",
                        "_seed_en_ty", "_seed_en_tz"):
                getattr(card, attr).toggled.connect(
                    lambda checked, n=n, a=attr: self._sync_seed_checkbox(a, n, checked, block=False))
            card._seed_tx.valueChanged.connect(lambda *_: self._update_panels_overview())
            card._seed_en_tx.toggled.connect(lambda *_: self._update_panels_overview())
            self._cards[n] = card
            self._card_stack.addWidget(card)
        lv.addWidget(self._card_stack)
        lv.addStretch(1)
        split.addWidget(scroll)

        # ── RIGHT: image viewer + bottom tabs ──
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._img_view = PickableImageViewer()
        self._toolbar = HydraDetectorToolbar(include_composite=False)
        self._toolbar.panelChanged.connect(self._on_panel_changed)
        tb = self._img_view._toolbar_layout
        self._origin_btn = OriginToolButton(self._img_view)
        tb.addWidget(self._origin_btn)
        tb.addWidget(self._toolbar)
        self._show_rings_check = QtWidgets.QCheckBox("Show rings"); self._show_rings_check.setChecked(True)
        self._show_rings_check.setToolTip(
            "Overlay the active panel's predicted rings, drawn through its full "
            "fitted geometry — tilts and refined distortion both applied.")
        self._show_rings_check.toggled.connect(
            lambda c: self._active_card and self._active_card.set_show_rings(c))
        tb.addWidget(self._show_rings_check)
        right.addWidget(self._img_view)

        bot = QtWidgets.QTabWidget()
        self._profile_view = HydraProfileViewer(composite_as_button=True)
        self._profile_view.compositeVisibilityChanged.connect(
            lambda _active: self._refresh_composite_curve())
        ptb = self._profile_view._toolbar_layout
        self._cal_r_bin = _fspin(0.1, 20.0, 2, 1.0, "px"); self._cal_r_bin.setFixedWidth(78)
        self._cal_eta_bin = _fspin(0.5, 360.0, 1, 5.0, "°"); self._cal_eta_bin.setFixedWidth(64)
        self._cal_azim = _NoScrollComboBox()
        self._cal_azim.addItem("Pixel-weighted", True)
        self._cal_azim.addItem("η-bin mean", False)
        reint_btn = QtWidgets.QPushButton("Re-integrate"); reint_btn.clicked.connect(self._reintegrate_all)
        insert_at = ptb.count() - 1
        ptb.insertWidget(insert_at, reint_btn)
        ptb.insertWidget(insert_at, self._cal_azim)
        ptb.insertWidget(insert_at, self._cal_eta_bin)
        ptb.insertWidget(insert_at, QtWidgets.QLabel("η:"))
        ptb.insertWidget(insert_at, self._cal_r_bin)
        ptb.insertWidget(insert_at, QtWidgets.QLabel("  R bin:"))
        bot.addTab(self._profile_view, "Radial Profile")

        cake_tab = QtWidgets.QWidget()
        cake_layout = QtWidgets.QVBoxLayout(cake_tab)
        cake_layout.setContentsMargins(0, 0, 0, 0); cake_layout.setSpacing(2)
        cake_bar = QtWidgets.QHBoxLayout()
        self._cake_checks: dict = {}
        for n in (1, 2, 3, 4):
            chk = QtWidgets.QCheckBox(f"GE{n}")
            chk.setChecked(n == 1)
            chk.toggled.connect(lambda checked, n=n: self._on_cake_panel_toggled(n, checked))
            cake_bar.addWidget(chk)
            self._cake_checks[n] = chk
        self._cake_overall_btn = QtWidgets.QPushButton("Overall")
        self._cake_overall_btn.setCheckable(True)
        self._cake_overall_btn.setStyleSheet(
            "QPushButton { border: 1px solid #666; border-radius: 4px; padding: 3px 10px; }")
        self._cake_overall_btn.toggled.connect(self._on_cake_overall_toggled)
        cake_bar.addWidget(self._cake_overall_btn)
        cake_bar.addStretch(1)
        cake_layout.addLayout(cake_bar)

        self._cake_views: dict = {}     # panel_num -> CakeViewer
        self._last_cake_data: dict = {}  # panel_num -> (cake_2d, r_axis_px, eta_axis_deg, lsd_um, px_um, wavelength_A)
        self._cake_stack = QtWidgets.QStackedWidget()
        for n in (1, 2, 3, 4):
            cake_view = CakeViewer()
            self._cake_views[n] = cake_view
            self._cake_stack.addWidget(cake_view)
        self._overall_cake_view = CakeViewer()
        self._cake_stack.addWidget(self._overall_cake_view)
        cake_layout.addWidget(self._cake_stack)
        bot.addTab(cake_tab, "Eta vs R Cake")

        self._resid_stack = QtWidgets.QStackedWidget()
        for n in (1, 2, 3, 4):
            self._resid_stack.addWidget(self._cards[n].residual_chart)
        bot.addTab(self._resid_stack, "Ring Residuals")

        self._results_stack = QtWidgets.QStackedWidget()
        for n in (1, 2, 3, 4):
            self._results_stack.addWidget(self._cards[n].results_widget)
        bot.addTab(self._results_stack, "Results")

        self._log = LogPanel()
        bot.addTab(self._log, "Log")
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

    # ── Loader signal handlers ──────────────────────────────────────

    def _suggest_working_dir(self):
        path = self._loader.current_path()
        if not path:
            return None
        try:
            expid = (self._expid_provider() or "").strip() if self._expid_provider else ""
        except Exception:
            expid = ""
        return suggest_working_dir(path, expid_fallback=expid)

    def _set_working_dir(self, d) -> None:
        self._out_ed.setText(str(d))
        reason = check_output_dir_writable(d)
        if reason:
            self._log.append(f"[hydra] Warning: {reason}")

    def _apply_suggested_working_dir(self):
        d = self._suggest_working_dir()
        if d is None:
            self._log.append("[hydra] Can't derive a working directory from the "
                             "loaded data path — pick one with the … button.")
            return
        self._set_working_dir(d)

    def _maybe_autofill_working_dir(self):
        """Never overwrites a directory the user already chose.

        Nor pre-fills one that can't be written to — see CalibrationTab's
        copy for why, and why the reason is logged only once per candidate.
        """
        if self._out_ed.text().strip():
            return
        d = self._suggest_working_dir()
        if d is None:
            return
        reason = check_output_dir_writable(d)
        if reason:
            if self._wd_declined != str(d):
                self._wd_declined = str(d)
                self._log.append(
                    f"[hydra] No working directory filled in — {reason}")
            return
        self._set_working_dir(d)

    def _update_panels_overview(self, running: Optional[int] = None):
        """One line naming every panel that will be fitted and the tx it
        will use, with the running one marked.

        ``running`` is the panel currently being fitted, or None. Panels
        found on disk but unticked are listed as excluded rather than
        omitted -- "ge3 is missing from this list" and "ge3 was left out on
        purpose" must not look the same.
        """
        if not hasattr(self, "_panels_lbl"):
            return
        selected = self._loader.siblings()
        found = self._loader.found_siblings()
        if not found:
            self._panels_lbl.setText("")
            return
        bits = []
        for n in sorted(found):
            if n not in selected:
                bits.append(f"<span style='color:#777'>ge{n} excluded</span>")
                continue
            tx = self._cards[n].seed_tx_value()
            mark = " ▶ fitting" if running == n else ""
            warn = "" if abs(tx) > 1e-9 else " <b>(tx not set)</b>"
            bits.append(f"ge{n} tx {tx:g}°{warn}{mark}")
        n_sel = len(selected)
        head = (f"<b>Fitting {n_sel} panel{'' if n_sel == 1 else 's'} "
                f"independently</b> — ")
        self._panels_lbl.setText(head + " &nbsp;·&nbsp; ".join(bits))

    def _on_siblings_changed(self, siblings: dict):
        self._toolbar.set_available(siblings.keys())
        self._update_panels_overview()
        # Safe to autofill straight off the load signal here, unlike
        # CalibrationTab: this page never loads a bundled demo image in
        # __init__, so there is no packaged path to derive a bogus default from.
        self._maybe_autofill_working_dir()
        self._sync_avg_controls()
        detected = self._loader.detected_geometry()
        if "wavelength_A" in detected:
            self._wl.setValue(float(detected["wavelength_A"]))
        if "pxY" in detected:
            self._pxY.setValue(float(detected["pxY"]))
        self._refresh_display()

    def _on_frame_changed(self, _idx: int):
        self._refresh_display()

    def _on_fields_changed(self):
        self._refresh_display()

    def _on_panel_changed(self, key: str):
        n = int(key[2])
        if self._active_card is not None:
            self._active_card.bind_viewer(None)
        if hasattr(self, "_panel_hdr"):
            self._panel_hdr.setText(
                f"ge{n} — this panel only (transforms, seed, result)")
        self._card_stack.setCurrentWidget(self._cards[n])
        self._resid_stack.setCurrentWidget(self._cards[n].residual_chart)
        self._results_stack.setCurrentWidget(self._cards[n].results_widget)
        self._cake_stack.setCurrentWidget(self._cake_views[n])
        self._cake_overall_btn.blockSignals(True)
        self._cake_overall_btn.setChecked(False)
        self._cake_overall_btn.blockSignals(False)
        for cn, chk in self._cake_checks.items():
            chk.blockSignals(True); chk.setChecked(cn == n); chk.blockSignals(False)
        self._active_card = self._cards[n]
        self._active_card.bind_viewer(self._img_view)
        chk = self._show_rings_check
        chk.blockSignals(True); chk.setChecked(self._active_card.show_rings_checked())
        chk.blockSignals(False)
        self._refresh_display()

    def _on_card_transform_changed(self, n: int):
        if self._active_card is self._cards.get(n):
            self._refresh_display()

    def _on_card_calib_file_loaded(self, g: dict):
        if g.get("wavelength_A"):
            self._wl.setValue(float(g["wavelength_A"]))
        if g.get("pxY"):
            self._pxY.setValue(float(g["pxY"]))

    def _sync_seed_checkbox(self, attr: str, src_panel: int, checked: bool,
                            block: bool = True):
        """"Use manual seed" / "Feed result back to seed" / each per-parameter
        seed-enable flag are one shared choice across all 4 GE panels (only
        the seed VALUES — BC/Lsd/tilts — stay independent per panel), so
        mirror a change on one panel's checkbox onto the other three.

        ``block`` (default True, matching the original "Use manual seed" /
        "Feed result back" behaviour) blocks the destination's own
        ``toggled`` handlers while mirroring. The five per-parameter
        ``_seed_en_*`` flags pass ``block=False`` instead: each drives that
        card's own spin-box enable state and seed summary label
        (``_on_seed_enable_changed``), which must actually run on every
        mirrored panel, not just the one the user clicked. The equality
        guard below still prevents runaway recursion — a card whose flag
        already matches ``checked`` is a no-op, so the cascade this can
        trigger (each card's own ``toggled`` re-enters this method) settles
        in at most one pass per panel."""
        for n, card in self._cards.items():
            if n == src_panel:
                continue
            cb = getattr(card, attr)
            if cb.isChecked() != checked:
                if block:
                    cb.blockSignals(True)
                    cb.setChecked(checked)
                    cb.blockSignals(False)
                else:
                    cb.setChecked(checked)

    # ── Per-panel frame sourcing ─────────────────────────────────────

    def _avg_index_range(self, n_frames: int) -> tuple:
        """(start, end_inclusive) for ``helpers.average_field``, translated
        from the start/end(0=all, exclusive) spinbox convention shared with
        the single-detector Calibrate tab's Mean-of-frames card."""
        end = n_frames if self._avg_end.value() <= 0 else min(self._avg_end.value(), n_frames)
        start = max(0, self._avg_start.value())
        return start, max(start, end - 1)

    def _panel_raw_image(self, n: int) -> Optional[np.ndarray]:
        """Raw (uncorrected, untransformed) source image for panel ``n`` —
        the mean over a frame range if enabled, else the current frame.
        Deliberately raw: ``CalibrationWorker`` applies bright/background/
        transforms itself (see module docstring)."""
        path = self._loader.siblings().get(n)
        if path is None:
            return None
        ds = self._loader.dataset()
        try:
            if self._avg_check.isChecked() and self._loader.n_frames() > 1:
                start, end = self._avg_index_range(self._loader.n_frames())
                return average_field(source_kind(path), path, ds, start, end)
            return _load_image(path, ds, self._loader.frame_index())
        except Exception:
            return None

    def _threshold_value(self) -> float:
        lo, hi = self._thr_min.value(), self._thr_max.value()
        if hi <= lo:
            return hi
        return lo + (self._thr_slider.value() / 1000.0) * (hi - lo)

    def _update_threshold_label(self):
        self._thr_val.setText(f"< {self._threshold_value():.4g} → 0")

    def _calib_image_for(self, img):
        if img is None:
            return None
        if self._thr_check.isChecked():
            thr = self._threshold_value()
            out = img.copy()
            out[img < thr] = 0.0
            return out
        return img

    def _sync_avg_controls(self):
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
            self._avg_note.setText("Single-frame source — frame mean unavailable.")
            return
        start = self._avg_start.value(); end = self._avg_end.value() or n; end = min(end, n)
        cnt = len(range(max(0, start), end))
        self._avg_note.setText(f"mean of {cnt} of {n} frames (start={start}, end={end}).")

    def _on_threshold_toggled(self, on: bool):
        for w in (self._thr_min, self._thr_max, self._thr_slider, self._thr_val):
            w.setEnabled(on)
        self._update_threshold_label()
        self._refresh_display()

    def _on_threshold_changed(self, *_):
        self._update_threshold_label()
        if self._thr_check.isChecked():
            self._refresh_display()

    def _on_avg_toggled(self, on):
        for w in (self._avg_start, self._avg_end):
            w.setEnabled(on)
        self._update_avg_note()
        self._refresh_display()

    def _on_avg_changed(self, *_):
        self._update_avg_note()
        if self._avg_check.isChecked():
            self._refresh_display()

    def _refresh_display(self):
        if self._active_card is None:
            return
        n = self._active_card.panel_number
        raw = self._panel_raw_image(n)
        if raw is None:
            return
        lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
        for w in (self._thr_min, self._thr_max, self._thr_slider):
            w.blockSignals(True)
        self._thr_min.setValue(max(0.0, lo)); self._thr_max.setValue(hi)
        for w in (self._thr_min, self._thr_max, self._thr_slider):
            w.blockSignals(False)
        self._update_threshold_label()
        img = self._calib_image_for(raw)
        img = apply_field_corrections(
            img, dark=self._loader.dark(n), bright=self._loader.bright(n),
            bright_mode=self._loader.bright_mode(), background=self._loader.background(n))
        im_trans = tuple(self._active_card.im_trans_codes())
        # Cheap post-transform shape (only `3`=transpose changes it — see
        # helpers._apply_im_trans) so the "fresh view" check below doesn't
        # need to flip the array twice just to know its displayed shape.
        disp_shape = img.shape[::-1] if 3 in im_trans else img.shape
        fresh = (self._disp_key != (disp_shape, n))
        self._disp_key = (disp_shape, n)
        self._img_view.set_raw_frame(img, im_trans, autorange=fresh, reset_levels=fresh)

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

    # ── Run ────────────────────────────────────────────────────────

    def _run_mode(self) -> str:
        return self._run_mode_combo.currentData() or "sequential"

    def _refine_flags(self) -> dict:
        coeffs = set(self._dist_coeffs) if self._ref_dist.isChecked() else set()
        return {
            "Lsd": self._ref_lsd.isChecked(), "BC": self._ref_bc.isChecked(),
            "ty": self._ref_ty.isChecked(), "tz": self._ref_tz.isChecked(),
            "tx": self._ref_tx.isChecked(), "Wavelength": self._ref_wl.isChecked(),
            "Distortion": self._ref_dist.isChecked(), "distortion_coeffs": coeffs,
        }

    def _build_cfg(self, card: HydraCalibPanelCard) -> dict:
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
            "work_dir": self._out_ed.text().strip() or None,
            # No "scratch_dir" here on purpose: _build_cfg has no panel number,
            # and all four panels share this cfg. It is resolved per panel in
            # _start_panel_worker, which is where the cfg forks.
            "im_trans": card.im_trans_codes(),
            "mask": None,
        }
        seed = card.manual_seed()
        if seed is not None:
            seed = dict(seed); seed["distortion"] = {}
            cfg["manual_seed"] = seed
        return cfg

    def _log_append_raw(self, line: str):
        self._log.append(line)

    def _run_all(self):
        siblings = self._loader.siblings()
        if not siblings:
            QtWidgets.QMessageBox.warning(self, "No data", "Load Hydra panel data first.")
            return
        if self._workers:
            return
        work_dir = self._out_ed.text().strip() or None
        if work_dir:
            reason = check_output_dir_writable(work_dir)
            if reason:
                QtWidgets.QMessageBox.critical(
                    self, "Working directory not writable", reason)
                return
        # One id for the whole Run All, so its four panels land side by side
        # under a single run folder rather than four timestamps apart.
        self._run_scratch_id = "calib_" + time.strftime("%Y%m%d-%H%M%S")
        self._orphans = [o for o in self._orphans if o.isRunning()]
        self._calib_cancelled = False
        self._composite_log_pending = True
        self._last_dist_coeffs = set(self._dist_coeffs) if self._ref_dist.isChecked() else set()
        self._run_btn.setEnabled(False); self._abort_btn.setEnabled(True)
        self._prog.setVisible(True)
        self._bot_tabs.setCurrentWidget(self._log)
        mode = self._run_mode()
        self._log.append("─" * 40 + f"\nStarting Hydra calibration ({mode}, "
                         f"{self._pipeline.currentData()})…")
        panels = sorted(siblings)
        self._pending_panels = list(panels)
        if mode == "parallel":
            pending, self._pending_panels = list(self._pending_panels), []
            for n in pending:
                self._start_panel_worker(n, capture_stdout=False)
        else:
            self._start_next_sequential()

    def _start_next_sequential(self):
        if not self._pending_panels:
            self._maybe_finish_run()
            return
        n = self._pending_panels.pop(0)
        self._start_panel_worker(n, capture_stdout=True)

    def _start_panel_worker(self, n: int, capture_stdout: bool):
        self._update_panels_overview(running=n)
        card = self._cards[n]
        raw = self._panel_raw_image(n)
        if raw is None:
            self._log.append(f"[ge{n}] no data — skipped")
            if self._run_mode() == "sequential":
                self._start_next_sequential()
            return
        image = self._calib_image_for(raw)
        dark = self._loader.dark(n)
        bright = self._loader.bright(n)
        background = self._loader.background(n)
        bright_mode = self._loader.bright_mode()
        cfg = self._build_cfg(card)
        # Each panel gets its own scratch leaf. In parallel mode all four
        # workers are in flight at once against one working directory, and
        # every file the backend writes there is generically named
        # (residual_corr.bin, calibration.json) — one shared folder is a race,
        # and in sequential mode it is last-writer-wins, which silently gives
        # panels 1-3 panel 4's residual map.
        try:
            cfg["scratch_dir"] = str(scratch_dir(
                cfg.get("work_dir"),
                self._run_scratch_id or "calib_" + time.strftime("%Y%m%d-%H%M%S"),
                f"ge{n}"))
        except OSError as e:
            # _on_panel_fail logs it and unwinds the run state; don't log twice.
            self._on_panel_fail(n, str(e))
            return
        cfg["save_stem"] = f"ge{n}"
        self._last_cfgs[n] = dict(cfg)
        mode = self._pipeline.currentData()
        worker = CalibrationWorker(
            mode, image, dark, cfg, parent=self, bright=bright, background=background,
            bright_mode=bright_mode, capture_stdout=capture_stdout)
        worker.log_line.connect(lambda line, n=n: self._log.append(f"[ge{n}] {line}"))
        worker.finished.connect(lambda result, n=n: self._on_panel_done(n, result))
        worker.failed.connect(lambda msg, n=n: self._on_panel_fail(n, msg))
        self._workers[n] = worker
        self._log.append(f"[ge{n}] starting…")
        worker.start()

    def _on_panel_done(self, n: int, result):
        if self._calib_cancelled:
            return
        card = self._cards[n]
        result.im_trans = card.im_trans_codes()
        result._calibrant_name = self._cal.currentText()
        card.on_result(result)
        self._log.append(f"[ge{n}] done — Lsd={result.Lsd/1000:.3f} mm")
        self._pending_log_results[n] = result
        self._run_integration(n, result)
        if not (self._int_workers.get(n) is not None and self._int_workers[n].isRunning()):
            # No image loaded (or integration otherwise didn't start) — log
            # now, without cake/profile results.
            self._pending_log_results.pop(n, None)
            self._log_to_project(n, result)
        self.panelCalibrationDone.emit(n, result)
        self._workers.pop(n, None)
        if self._run_mode() == "sequential":
            self._start_next_sequential()
        self._maybe_finish_run()

    def _log_to_project(self, n: int, result, results: Optional[dict] = None):
        if not self._project_ctx or not self._project_ctx.path:
            return
        siblings = self._loader.siblings()
        loader_state = {
            "path": siblings.get(n), "dataset": self._loader.dataset(),
            "frame_index": self._loader.frame_index(),
        }
        try:
            ref = project.append_calibration_attempt(
                self._project_ctx.path, f"ge{n}",
                cfg=self._last_cfgs.get(n, {}), result=result,
                loader_state=loader_state,
                results=results,
                extra={"active_profile": settings.active_profile()})
            result._project_attempt_ref = ref
            self._log.append(f"[ge{n}] logged to project: {ref}")
        except Exception:
            import traceback as _tb
            self._log.append(f"[ge{n}] could not log to project file:\n" + _tb.format_exc())

    def _on_panel_fail(self, n: int, msg: str):
        if self._calib_cancelled:
            return
        self._log.append(f"[ge{n}] ERROR:\n{msg}")
        self._workers.pop(n, None)
        if self._run_mode() == "sequential":
            self._start_next_sequential()
        self._maybe_finish_run()

    def _maybe_finish_run(self):
        if self._workers or self._pending_panels:
            return
        self._run_btn.setEnabled(True); self._abort_btn.setEnabled(False)
        self._prog.setVisible(False)
        self._log.append("Hydra calibration run complete.")

    def _abort_all(self):
        if not self._workers:
            return
        self._calib_cancelled = True
        for n, w in list(self._workers.items()):
            for sig in (w.log_line, w.finished, w.failed):
                try:
                    sig.disconnect()
                except Exception:
                    pass
            w.requestInterruption()
            self._orphans.append(w)
        self._workers.clear()
        self._pending_panels = []
        self._run_btn.setEnabled(True); self._abort_btn.setEnabled(False)
        self._prog.setVisible(False)
        self._log.append("Hydra calibration aborted — a background thread per panel "
                         "may still be winding down.")

    # ── Integration / residual chart ───────────────────────────────

    def _run_integration(self, n: int, result):
        image = self._calib_image_for(self._panel_raw_image(n))
        if image is None:
            return
        if self._int_workers.get(n) is not None and self._int_workers[n].isRunning():
            return
        card = self._cards[n]
        im_trans = tuple(card.im_trans_codes())
        w = IntegrationWorker(
            result, image, self._loader.dark(n), im_trans,
            r_bin=self._cal_r_bin.value(), eta_bin=self._cal_eta_bin.value(),
            mask=None, parent=self, bright=self._loader.bright(n),
            background=self._loader.background(n), bright_mode=self._loader.bright_mode(),
            weighted=bool(self._cal_azim.currentData()))
        w.finished.connect(lambda data, n=n, result=result: self._on_int_done(n, result, data))
        w.failed.connect(lambda m, n=n: self._on_int_failed(n, m))
        self._int_workers[n] = w
        w.start()

    def _reintegrate_all(self):
        for n, card in self._cards.items():
            if card.result is not None:
                self._run_integration(n, card.result)

    def _flush_pending_log(self, n: int, results: Optional[dict]):
        pending = self._pending_log_results.pop(n, None)
        if pending is not None:
            self._log_to_project(n, pending, results=results)

    def _on_int_failed(self, n: int, msg: str):
        self._log.append(f"[ge{n}] integration error: {msg}")
        self._int_workers.pop(n, None)
        self._flush_pending_log(n, None)

    def _on_int_done(self, n: int, result, data: dict):
        self._profile_view.set_curve(
            f"ge{n}", data["r_axis_px"], data["profile"],
            lsd_um=data["lsd_um"], px_um=data["px_um"], wavelength_A=data["wavelength_A"])
        radii = _predict_ring_radii(result)
        self._cards[n].residual_chart.set_data(data["r_axis_px"], data["profile"], radii)
        if data.get("cake_2d") is not None:
            self._cake_views[n].set_cake(data["cake_2d"], data["r_axis_px"], data["eta_axis_deg"])
            self._last_cake_data[n] = (
                data["cake_2d"], data["r_axis_px"], data["eta_axis_deg"],
                data["lsd_um"], data["px_um"], data["wavelength_A"])
        self._int_workers.pop(n, None)
        self._refresh_composite_curve()
        self._flush_pending_log(n, data)
        self._maybe_log_composite_attempt()

    def _maybe_log_composite_attempt(self):
        """Once a live run's calibration AND integration have both fully
        finished for every panel (not on a manual Re-integrate), append the
        Overall/composite profile as its own lightweight attempt — only if
        Overall was active and something was actually computed."""
        if self._workers or self._pending_panels or self._int_workers:
            return
        if not self._composite_log_pending:
            return
        self._composite_log_pending = False
        if not (self._project_ctx and self._project_ctx.path):
            return
        if not self._profile_view.composite_visible():
            return
        native = self._profile_view.get_native("composite")
        if native is None:
            return
        r_ref, summed = native[0], native[1]
        try:
            from types import SimpleNamespace
            ref = project.append_calibration_attempt(
                self._project_ctx.path, "hydra_composite",
                cfg={}, result=SimpleNamespace(), loader_state={},
                results={"profile": summed, "r_axis_px": r_ref},
                extra={"active_profile": settings.active_profile()})
            self._log.append(f"Logged Overall profile to project: {ref}")
        except Exception:
            import traceback as _tb
            self._log.append(
                "Could not log Overall profile to project:\n" + _tb.format_exc())

    def _refresh_composite_curve(self):
        """Recompute the toggleable "Overall" radial-profile curve: each
        available panel's own (already-computed) profile, converted to a
        shared 2theta axis, resampled onto one common grid, and NaN-aware
        summed — not a re-integration of a composited image (would double-
        count any panel overlap). Mirrors
        ``hydra_page.HydraViewerPage._refresh_composite_curve``."""
        if not self._profile_view.composite_visible():
            self._profile_view.clear_curve("composite")
            return
        natives = []
        for n in (1, 2, 3, 4):
            data = self._profile_view.get_native(f"ge{n}")
            if data is not None and None not in data[2:]:   # need lsd, px, wl
                natives.append(data)
        if not natives:
            self._profile_view.clear_curve("composite")
            return
        grids = []
        for r_px, profile, lsd, px, wl in natives:
            tth = _convert_radial(r_px, lsd, px, wl, "R", "2th")
            order = np.argsort(tth)
            grids.append((tth[order], profile[order]))
        lo = min(g[0].min() for g in grids)
        hi = max(g[0].max() for g in grids)
        common = np.linspace(lo, hi, 500)
        resampled = [np.interp(common, tth, profile, left=np.nan, right=np.nan)
                    for tth, profile in grids]
        stacked = np.vstack(resampled)
        all_nan = np.all(np.isnan(stacked), axis=0)
        summed = np.where(all_nan, np.nan, np.nansum(stacked, axis=0))
        ref_lsd, ref_px, ref_wl = natives[0][2], natives[0][3], natives[0][4]
        r_ref = ref_lsd * np.tan(np.radians(common)) / ref_px
        self._profile_view.set_curve("composite", r_ref, summed,
                                     lsd_um=ref_lsd, px_um=ref_px, wavelength_A=ref_wl)

    # ── Overall cake ─────────────────────────────────────────────────

    def _on_cake_panel_toggled(self, n: int, checked: bool):
        if not checked:
            return
        for cn, chk in self._cake_checks.items():
            if cn != n:
                chk.blockSignals(True); chk.setChecked(False); chk.blockSignals(False)
        self._cake_overall_btn.blockSignals(True)
        self._cake_overall_btn.setChecked(False)
        self._cake_overall_btn.blockSignals(False)
        self._toolbar.set_current(f"ge{n}")

    def _on_cake_overall_toggled(self, active: bool):
        if active:
            self._cake_overall_btn.setStyleSheet(
                "QPushButton { background: #2e7d32; color: white; font-weight: bold; "
                "border: 1px solid #1b5e20; border-radius: 4px; padding: 3px 10px; }")
            for chk in self._cake_checks.values():
                chk.blockSignals(True); chk.setChecked(False); chk.blockSignals(False)
            composed = _compose_overall_cake(self._last_cake_data)
            if composed is not None:
                cake, r_axis, eta_axis = composed
                self._overall_cake_view.set_cake(cake, r_axis, eta_axis)
                self._cake_stack.setCurrentWidget(self._overall_cake_view)
            else:
                self._log.append("Overall cake: no panels integrated yet.")
        else:
            self._cake_overall_btn.setStyleSheet(
                "QPushButton { border: 1px solid #666; border-radius: 4px; padding: 3px 10px; }")
            n = int(self._toolbar.current()[2])
            self._cake_stack.setCurrentWidget(self._cake_views[n])
            self._cake_checks[n].blockSignals(True)
            self._cake_checks[n].setChecked(True)
            self._cake_checks[n].blockSignals(False)

    # ── File > Open Project… ─────────────────────────────────────────

    def display_stored_result(self, n: int, result, results_arrays: Optional[dict] = None) -> None:
        """Redraw panel ``n``'s rings + profile/cake for a result recovered
        from a project attempt — mirrors ``_on_panel_done``'s visual effects
        without re-running Fit. When ``results_arrays`` (the attempt's
        embedded cake/profile) is available, populates directly instead of
        re-integrating; otherwise falls back to live re-integration if an
        image happens to be loaded."""
        card = self._cards.get(n)
        if card is None:
            return
        card.on_result(result)
        if results_arrays and results_arrays.get("profile") is not None:
            self._profile_view.set_curve(
                f"ge{n}", results_arrays["r_axis_px"], results_arrays["profile"],
                lsd_um=results_arrays.get("lsd_um"), px_um=results_arrays.get("px_um"),
                wavelength_A=results_arrays.get("wavelength_A"))
            radii = _predict_ring_radii(result)
            card.residual_chart.set_data(
                results_arrays["r_axis_px"], results_arrays["profile"], radii)
            if results_arrays.get("cake_2d") is not None:
                self._cake_views[n].set_cake(
                    results_arrays["cake_2d"], results_arrays["r_axis_px"],
                    results_arrays["eta_axis_deg"])
                self._last_cake_data[n] = (
                    results_arrays["cake_2d"], results_arrays["r_axis_px"],
                    results_arrays["eta_axis_deg"], results_arrays.get("lsd_um"),
                    results_arrays.get("px_um"), results_arrays.get("wavelength_A"))
            self._refresh_composite_curve()
        else:
            self._run_integration(n, result)

    # ── Import from Data Viewer ──────────────────────────────────────

    def import_from_viewer(self, data: dict):
        data = data or {}
        anchor = data.get("anchor_path")
        if anchor:
            self._loader.set_path(anchor)
        for n_key, g in (data.get("geometries") or {}).items():
            n = int(n_key)
            card = self._cards.get(n)
            if card is None or not g:
                continue
            card.seed_from_geometry(g)
            if g.get("wavelength_A"):
                self._wl.setValue(float(g["wavelength_A"]))
            if g.get("pxY"):
                self._pxY.setValue(float(g["pxY"]))
        self._log.append("Geometry imported from Data Viewer (Hydra).")

    # ── GUI state ────────────────────────────────────────────────────

    def _state_widgets(self) -> dict:
        return {
            "pipeline": self._pipeline, "wl": self._wl, "cal": self._cal,
            "pxY": self._pxY, "pxZ_check": self._pxZ_check, "pxZ_spin": self._pxZ_spin,
            "thr_check": self._thr_check, "thr_min": self._thr_min, "thr_max": self._thr_max,
            "avg_check": self._avg_check, "avg_start": self._avg_start, "avg_end": self._avg_end,
            "ref_lsd": self._ref_lsd, "ref_bc": self._ref_bc, "ref_ty": self._ref_ty,
            "ref_tz": self._ref_tz, "ref_tx": self._ref_tx, "ref_wl": self._ref_wl,
            "ref_dist": self._ref_dist, "build_rc": self._build_rc,
            "n_iter": self._n_iter, "lm_iter": self._lm_iter, "device": self._device,
            "out_ed": self._out_ed, "run_mode": self._run_mode_combo,
            "cal_r_bin": self._cal_r_bin, "cal_eta_bin": self._cal_eta_bin,
            "cal_azim": self._cal_azim,
        }

    def get_state(self) -> dict:
        cards = {}
        for n, card in self._cards.items():
            fields = widgets_to_dict(card.state_widgets())
            fields["show_rings"] = card.show_rings_checked()
            cards[n] = fields
        return {
            "anchor_path": self._loader.current_path(),
            "active_panel": self._toolbar.current(),
            "fields": widgets_to_dict(self._state_widgets()),
            "img_view": self._img_view.display_state(),
            "cake_views": {n: cv.display_state() for n, cv in self._cake_views.items()},
            "overall_cake_view": self._overall_cake_view.display_state(),
            "cards": cards,
            # No widget of its own — see CalibrationTab.get_state's copy.
            "dist_coeffs": sorted(self._dist_coeffs),
        }

    def set_state(self, state: dict):
        if not state:
            return
        apply_dict_to_widgets(self._state_widgets(), state.get("fields", {}))
        dist_coeffs = state.get("dist_coeffs")
        if dist_coeffs is not None:
            self._dist_coeffs = set(dist_coeffs)
        self._update_dist_label()   # signals were blocked above; resync the caption
        self._img_view.set_display_state(state.get("img_view"))
        self._origin_btn.sync()
        for n_key, cv_state in (state.get("cake_views") or {}).items():
            cv = self._cake_views.get(int(n_key))
            if cv is not None:
                cv.set_display_state(cv_state)
        self._overall_cake_view.set_display_state(state.get("overall_cake_view"))
        for n_key, fields in (state.get("cards") or {}).items():
            card = self._cards.get(int(n_key))
            if card is None:
                continue
            card.apply_state_fields(fields)
            if "show_rings" in fields:
                card.set_show_rings(bool(fields["show_rings"]))
        anchor = state.get("anchor_path")
        if anchor and Path(anchor).exists():
            self._loader.set_path(anchor)
        panel = state.get("active_panel")
        if panel:
            self._toolbar.set_current(panel)
