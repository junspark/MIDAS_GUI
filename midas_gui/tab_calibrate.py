"""Tab 2 — Calibrate.

Ports the v3 calibration tab and adds Phase-1 features:
  - pipeline dropdown (one-shot / first-time / four-stage)
  - refine-flags group (Lsd, BC, ty, tz, tx, Wavelength, Distortion)
  - read-only distortion-coefficient table
  - per-ring radial-residual bar chart (new bottom tab)
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from midas_gui.constants import (
    CALIBRANTS, PIPELINES, DEFAULT_PIPELINE, _SG, _LC, DEFAULT_WAVELENGTH, DEFAULT_PIXEL_UM,
    DEFAULT_LSD_UM, DEFAULT_BC_Y, DEFAULT_BC_Z, DEFAULT_CALIBRANT_TIF,
    DEFAULT_STEP_BC, DEFAULT_STEP_LSD_MM, DEFAULT_STEP_TILT,
    DISTORTION_NAMES, MATERIALS, calibrant_combo_items, is_dspacing_calibrant)
from midas_gui.helpers import (
    _fspin, _NoScrollSpinBox, _predict_ring_radii, _NoScrollComboBox,
    make_kedge_label, make_pixel_label, ring_xy_corrected, distortion_rho_d_um,
    ring_on_image_mask, refresh_combo_items, rmax_corner_px, pixel_readout_text,
    widgets_to_dict, apply_dict_to_widgets, im_trans_codes_from_checkboxes,
    paramstest_pairs, parse_dspacing_text, browse_start_dir, warn_if_path_missing,
    result_refined_tx,
    suggest_working_dir, check_output_dir_writable, scratch_dir, SCRATCH_DIRNAME)
from midas_gui.widgets import (
    PickableImageViewer, ProfileViewer, LogPanel, DataLoaderPanel, CakeViewer,
    RingResidualViewer, OriginToolButton, build_lab_frame_axes_items,
    ring_azimuth_residual)
from midas_gui.workers import CalibrationWorker, IntegrationWorker, ManualDspacingCalibWorker
from midas_gui.dialogs import (_SaveParamstestDialog, DistortionRefineDialog,
                                DistortionSeedDialog, ManualSeedDialog,
                                PARAMETER_LIMIT_ROWS, limit_window, show_error)
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
        self._calib_result = None
        self._dist_coeffs = set(DISTORTION_NAMES)   # distortion coeffs to refine
        self._seed_dist: dict = {}                  # {v2 coeff name: value} — from a result's feedback or typed in via _edit_seed_distortion
        self._last_dist_coeffs: Optional[set] = None  # coeffs selected for the last run
        self._last_refine_flags: Optional[dict] = None  # refine flags used for the last run
        self._last_fit_sigma: Optional[dict] = None     # per-parameter 1σ from the last manual fit
        self._last_at_limit: set = set()                # params that hit a limit last run
        # The Refine checkboxes are shared by the crystalline and manual fits,
        # but their sensible defaults are not. A crystalline CeO2 pattern fills
        # the detector and constrains Lsd and tilt well; a d-spacing calibrant
        # is fit from a handful of hand-picked points, often on a single short
        # ring arc, where floating Lsd on top of BC is badly conditioned and
        # tilt is not identifiable at all. So each mode keeps its own flags and
        # they are swapped on the calibrant transition, rather than one set of
        # defaults being wrong for one of the two.
        self._refine_state_xtal: Optional[dict] = None
        self._refine_state_dsp: dict = {"Lsd": False, "BC": True, "tx": False,
                                        "ty": False, "tz": False, "Wavelength": False}
        self._refine_mode_is_dsp: Optional[bool] = None
        # The limits column is likewise per-calibrant-kind — different rows,
        # different granularity, and "off" means unbounded for the manual fit
        # but "backend default" for the crystalline one. See _sync_limits_mode.
        self._limit_state_xtal: Optional[dict] = None
        self._limit_state_dsp: Optional[dict] = None
        self._limits_mode_is_dsp: Optional[bool] = None
        self._last_cfg: Optional[dict] = None          # cfg used for the last run (provenance)
        self._last_bright: Optional[np.ndarray] = None
        self._last_background: Optional[np.ndarray] = None
        self._project_ctx: Optional[project.ProjectContext] = None
        self._pending_log_result = None   # result awaiting _log_to_project once integration finishes
        self._expid_provider = None       # set by app.py; see set_expid_provider
        self._wd_declined = ""            # last unwritable candidate, logged once
        self._build_ui()
        self._loader.set_path(DEFAULT_CALIBRANT_TIF)
        # Connected after the bundled demo image loads above, deliberately
        # unlike Batch Integrate (which connects before its own default load):
        # DEFAULT_CALIBRANT_TIF lives inside the installed package, so letting
        # it autofill would propose a working directory next to the source
        # tree on every cold start — a path nobody asked to write to.
        self._loader.dataChanged.connect(self._maybe_autofill_working_dir)

    def set_mask_from_tab1(self, mask: Optional[np.ndarray]):
        self._loader.set_tab1_mask(mask)

    def set_project_context(self, ctx: "project.ProjectContext"):
        self._project_ctx = ctx
        self._hydra_page.set_project_context(ctx)

    def set_expid_provider(self, provider) -> None:
        """Wired by app.py's MainWindow: ``provider()`` returns the header's
        current Exp ID (shared app-wide, not owned by this tab). Feeds the
        saved-calibration name (``_default_save_stem``) and, as a fallback for
        layouts too shallow to read it off the path, the working-directory
        suggestion (``_suggest_working_dir``)."""
        self._expid_provider = provider
        # Forwarded the same way set_project_context is: the Hydra page is a
        # child of this tab, not separately wired by app.py.
        self._hydra_page._expid_provider = provider

    # ── Default names for saved calibrations ──────────────────────

    def _default_save_stem(self) -> str:
        """``<expid>_<calibration image stem>`` for the Save dialogs.

        Both halves are best-effort: a blank Exp ID or an empty Data path
        simply drops that half, so the suggestion degrades to the image stem,
        to the Exp ID, or — with neither — to "calibration", rather than
        offering something like ``_.instr.txt``.
        """
        parts = []
        try:
            expid = (self._expid_provider() or "").strip() if self._expid_provider else ""
        except Exception:
            expid = ""
        if expid:
            parts.append(expid)
        data_path = self._loader.data_path()
        if data_path:
            # .h5/.tif alike: one suffix off is enough, and a doubled
            # extension (".ge3.edf") keeps its first half, which is the
            # distinguishing part of the name.
            stem = Path(data_path).name.rsplit(".", 1)[0]
            if stem:
                parts.append(stem)
        return "_".join(parts) if parts else "calibration"

    def _suggest_working_dir(self):
        """The ``<expid>_bc`` folder implied by the loaded data, or None."""
        path = self._loader.data_path()
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
            self._log.append(f"[calibrate] Warning: {reason}")

    def _apply_suggested_working_dir(self):
        """The Suggest button: overwrite whatever is there with the default."""
        d = self._suggest_working_dir()
        if d is None:
            self._log.append(
                "[calibrate] Can't derive a working directory from the loaded "
                "data path — pick one with the … button.")
            return
        self._set_working_dir(d)

    def _maybe_autofill_working_dir(self, *_a):
        """Fill the field on data load, but never overwrite the user's choice.

        Declines an unwritable candidate rather than pre-filling it: a path
        that looks accepted but fails at Run time is worse than an empty field
        that makes the user choose. The Suggest button still fills it — there
        the user asked, so they get the path and the warning.

        Silent when it can't derive one, unlike the Suggest button: this fires
        on every data change, and a log line per frame would be noise. The
        same reason a repeated unwritable candidate is only reported once.
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
                    f"[calibrate] No working directory filled in — {reason}")
            return
        self._set_working_dir(d)

    def _default_save_path(self, suffix: str) -> str:
        """``_default_save_stem()`` + *suffix*, under the Output dir if one is
        set (else beside the calibration image, else the process CWD).

        Returning a full path rather than a bare filename is what keeps the
        save dialog from opening on whatever directory the app happens to have
        been launched from.
        """
        out_dir = self._out_ed.text().strip()
        start = browse_start_dir(out_dir) if out_dir else browse_start_dir(self._loader.data_path())
        name = self._default_save_stem() + suffix
        return str(Path(start) / name) if start else name

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
        Hydra panel) from the just-activated profile's constants.CALIBRANTS /
        constants.MATERIALS."""
        refresh_combo_items(self._cal, calibrant_combo_items())
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
        self._loader = DataLoaderPanel(mode="single", hide_frame_field=True)
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
        self._pipeline.setMaximumWidth(208)   # ~50% of the unconstrained width
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
        lv.addWidget(pipe)

        # ── Detector & Calibrant ──
        det = S.make_card("Define experiment")
        self._load_calib_btn = QtWidgets.QPushButton("Load calibration file…")
        self._load_calib_btn.setToolTip(
            "Load geometry from a MIDAS paramstest (.txt), a calibration .json, "
            "or a pyFAI .poni — sets λ, pixel size, and the seed BC + Lsd.")
        self._load_calib_btn.clicked.connect(self._load_calib_file)
        self._from_view_btn = QtWidgets.QPushButton("→ Get from Data Viewer")
        self._from_view_btn.setToolTip(
            "Pull λ, pixel size, Lsd and beam centre from the Data Viewer tab "
            "into the detector + seed fields here.")
        self._from_view_btn.clicked.connect(self.pullGeometry.emit)
        _lrow = QtWidgets.QHBoxLayout(); _lrow.setSpacing(4)
        _lrow.addWidget(self._load_calib_btn); _lrow.addWidget(self._from_view_btn)
        _lrow.addStretch(1)
        det.body.addLayout(_lrow)
        self._wl = _fspin(0.001, 10.0, 5, DEFAULT_WAVELENGTH, "Å")
        self._cal = _NoScrollComboBox(); self._cal.addItems(calibrant_combo_items()); self._cal.setMaximumWidth(75)
        # A Form().row() stretches every field column equally, which spreads
        # "Calibrant:" arbitrarily far from λ as the panel widens. Build this
        # row by hand instead so the gap between the two fields stays small
        # and fixed, with the leftover width pushed past Calibrant.
        wl_row = QtWidgets.QHBoxLayout(); wl_row.setSpacing(4)
        wl_row.addWidget(make_kedge_label(self._wl, "λ:")); wl_row.addWidget(self._wl)
        wl_row.addSpacing(10)
        wl_row.addWidget(S.LabelRight("Calibrant:")); wl_row.addWidget(self._cal)
        wl_row.addStretch(1)
        det.body.addLayout(wl_row)
        self._pxY = _fspin(1.0, 5000.0, 2, DEFAULT_PIXEL_UM, "µm")
        self._pxZ_check = QtWidgets.QCheckBox("pxZ")
        self._pxZ_spin = _fspin(1.0, 5000.0, 2, DEFAULT_PIXEL_UM, "µm"); self._pxZ_spin.setEnabled(False)
        self._pxZ_check.toggled.connect(self._pxZ_spin.setEnabled)
        prow = QtWidgets.QHBoxLayout(); prow.setSpacing(4)
        # No stretch on any of these: a stretched spinbox grows well past its
        # digits (up to its 104px cap), which visually reads as "far from" the
        # checkbox beside it even though it starts right after it. Pack all
        # three at their natural width and push the leftover space to a
        # trailing stretch instead.
        prow.addWidget(self._pxY); prow.addWidget(self._pxZ_check); prow.addWidget(self._pxZ_spin)
        prow.addStretch(1)
        det.body.addLayout(S.Form().row(
            (make_pixel_label(self._pxY, "Pixel:", also=self._pxZ_spin), prow)))
        self._flip_y = QtWidgets.QCheckBox("Flip Y"); self._flip_z = QtWidgets.QCheckBox("Flip Z")
        self._transp = QtWidgets.QCheckBox("Transpose")
        for cb in (self._flip_y, self._flip_z, self._transp):
            cb.toggled.connect(self._on_im_trans_changed)

        def _vsep():
            f = QtWidgets.QFrame()
            f.setFrameShape(QtWidgets.QFrame.VLine)
            f.setFrameShadow(QtWidgets.QFrame.Sunken)
            return f

        tb2 = QtWidgets.QHBoxLayout(); tb2.setSpacing(8)
        tb2.addWidget(QtWidgets.QLabel("Transforms:"))
        tb2.addWidget(self._flip_y); tb2.addWidget(_vsep())
        tb2.addWidget(self._flip_z); tb2.addWidget(_vsep())
        tb2.addWidget(self._transp); tb2.addStretch(1)
        det.body.addLayout(tb2)
        lv.addWidget(det)

        # ── Manual ring-picking (non-crystalline calibrants) ──
        manual = S.make_card("Manual ring-picking (non-crystalline calibrants)")
        self._manual_card = manual
        manual_hint = QtWidgets.QLabel(
            "Shown because a calibrant with no space group is selected above "
            "(e.g. AgBH, or Custom d-spacings…): pick points on the image with "
            "'Pick d-spacing pts', tag each with its Ring #, then Run fits the "
            "geometry directly from Bragg's law, refining whichever parameters "
            "are ticked under Refine parameters.\n"
            "Defaults to beam centre only. Points spread over one short ring arc "
            "cannot separate Lsd from the beam centre, and constrain tilt hardly "
            "at all — pick across more rings and a wider arc before freeing "
            "those, and check the ± uncertainty reported for each fitted value.")
        manual_hint.setStyleSheet(f"color:{S.MUTED};font-size:10px"); manual_hint.setWordWrap(True)
        manual.body.addWidget(manual_hint)
        self._dsp_custom_ed = QtWidgets.QLineEdit()
        self._dsp_custom_ed.setPlaceholderText(
            "Custom d-spacings (Å), comma/space-separated, e.g. 58.38 29.19 19.46")
        self._dsp_custom_ed.setVisible(False)
        manual.body.addWidget(self._dsp_custom_ed)
        self._cal.currentTextChanged.connect(self._on_calibrant_changed)
        self._dsp_custom_ed.textChanged.connect(self._on_dspacing_picks_changed)
        self._dsp_summary = QtWidgets.QLabel("No points picked yet.")
        self._dsp_summary.setStyleSheet(f"color:{S.MUTED};font-size:10px"); self._dsp_summary.setWordWrap(True)
        manual.body.addWidget(self._dsp_summary)
        lv.addWidget(manual)

        # ── Threshold (calibration image only) ──
        thr = S.make_card("Apply threshold to calibration image")
        thr.setCheckable(True); thr.setChecked(False)
        thr.setToolTip(
            "When on, pixels dimmer than the slider value are set to 0 in the image\n"
            "fed to the calibration pipeline (and the live preview). Useful to drop\n"
            "background / weak pixels before calibrating.")
        self._thr_check = thr
        self._thr_min = _fspin(-1e9, 1e9, 0, 0.0)
        self._thr_min.setMaximumWidth(52)
        self._thr_max = _fspin(-1e9, 1e9, 0, 65535.0)
        self._thr_max.setMaximumWidth(83)
        self._thr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._thr_slider.setRange(0, 1000); self._thr_slider.setValue(0)
        self._thr_val = QtWidgets.QLabel("threshold = —")
        self._thr_val.setStyleSheet(f"color:{S.ACCENT};font-size:11px")
        srow = QtWidgets.QHBoxLayout(); srow.setSpacing(6)
        srow.addWidget(self._thr_min); srow.addWidget(self._thr_slider, 9); srow.addStretch(1)
        srow.addWidget(self._thr_max)
        thr.body.addLayout(srow)
        thr.body.addWidget(self._thr_val)
        for w in (self._thr_min, self._thr_max, self._thr_slider, self._thr_val):
            w.setEnabled(False)
        thr.toggled.connect(self._on_threshold_toggled)
        self._thr_slider.valueChanged.connect(self._on_threshold_changed)
        self._thr_min.valueChanged.connect(self._on_threshold_changed)
        self._thr_max.valueChanged.connect(self._on_threshold_changed)
        lv.addWidget(thr)

        # ── Mean of frames (hdf5 / folder) ──
        # Checkable QGroupBox (like "Multi-panel detector" below): the on/off
        # toggle lives in the heading itself rather than a separate checkbox
        # in the body. `_avg_check` is the groupbox itself — it already has
        # isChecked()/setChecked()/toggled, so every existing caller (state
        # save/restore included, see helpers.widgets_to_dict) works unchanged.
        avgc = QtWidgets.QGroupBox("Mean of frames in single image")
        avgc.setCheckable(True); avgc.setChecked(False)
        avgc.setToolTip(
            "Take the mean of a range of frames into one image used for calibration. "
            "Requires a multi-frame source.")
        avgc_body = QtWidgets.QVBoxLayout(avgc)
        avgc_body.setContentsMargins(8, 6, 8, 6); avgc_body.setSpacing(5)
        avgc.body = avgc_body
        self._avg_check = avgc
        self._avg_start = _NoScrollSpinBox(); self._avg_start.setRange(0, 999999)
        self._avg_end = _NoScrollSpinBox(); self._avg_end.setRange(0, 999999)
        self._avg_end.setToolTip("Last frame (exclusive). 0 = all frames.")
        for w in (self._avg_start, self._avg_end):
            w.setEnabled(False)
            w.setFixedWidth(69)   # ~50% narrower than the default rendered width
        # Plain grid (not S.Form) with the stretch pushed onto a trailing
        # empty column, so start/end stay packed left with just the row's
        # own spacing between them instead of being spread across the card.
        afm = QtWidgets.QGridLayout(); afm.setHorizontalSpacing(6); afm.setVerticalSpacing(5)
        afm.setContentsMargins(0, 0, 0, 0)
        afm.addWidget(S.LabelRight("start:"), 0, 0)
        afm.addWidget(self._avg_start, 0, 1)
        afm.addWidget(S.LabelRight("end(0=all):"), 0, 2)
        afm.addWidget(self._avg_end, 0, 3)
        afm.setColumnStretch(4, 1)
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
        # "Manual seed…" opens a non-modal per-parameter panel (ManualSeedDialog):
        # BC (as one pair — the backend only accepts BC_y/BC_z together, see
        # calib._resolve_seed), Lsd, tx, ty, tz can each be independently
        # ticked "include in seed"; an unticked parameter behaves exactly as
        # if seeding were off for it alone (auto-seed for BC/Lsd, 0° default
        # for a tilt). `_manual_seed_check` is kept only as a derived bulk
        # on/off convenience — see `_on_seed_master_toggled`/
        # `_on_seed_enable_changed` — for project-file backward compatibility
        # and the Hydra cross-panel sync (hydra_calib_page._sync_seed_checkbox).
        seed = S.make_card("Initial seed  (Pick tools on image)")
        self._manual_seed_check = QtWidgets.QCheckBox("Use manual seed")
        self._manual_seed_check.setTristate(True)
        self._manual_seed_check.setVisible(False)   # superseded by the dialog; kept as internal/legacy state only
        self._syncing_seed_master = False
        # Beam-centre seed accepts up to 3 decimal places — a precisely-known
        # centre (e.g. from an external optical/mechanical measurement, or
        # copied from a fit result's own sub-pixel precision) can be entered
        # exactly rather than rounded to the nearest tenth of a pixel.
        #
        # Explicit arrow steps, from the same constants (and the same user
        # preference) the Data Viewer's geometry card uses. Without them these
        # spinboxes fall back to _fspin's adaptive stepping, which scales with
        # the value's magnitude: at BC_z ~ 1340 px one click moved 100 px, and
        # at Lsd = 1000 mm one click moved 50 mm — useless for nudging a seed.
        # _sync_seed_steps() overrides these per-box from the limit windows.
        self._seed_bcy = _fspin(-99999, 99999, 3, DEFAULT_BC_Y, "px",
                                step=DEFAULT_STEP_BC)
        self._seed_bcz = _fspin(-99999, 99999, 3, DEFAULT_BC_Z, "px",
                                step=DEFAULT_STEP_BC)
        # Lsd shown/entered in mm; calculations & files use µm.
        self._seed_lsd = _fspin(0.001, 1e5, 3, DEFAULT_LSD_UM / 1000.0, " mm",
                                step=DEFAULT_STEP_LSD_MM)
        # Seed tilts (deg). Honoured by the four-stage / advanced pipelines; the
        # one-shot / first-time paths seed tilts only if the installed backend
        # exposes initial-tilt kwargs (otherwise they start at 0).
        # tx is the panel's installation azimuth about the beam, not a small
        # alignment tilt like ty/tz: real fitted Hydra geometries run the whole
        # way round (the bundled example is 27.3 / 117.8 / 207.5 / 296.885).  A
        # +/-180 range made three of those four untypable, and -- worse --
        # setValue clamps silently, so loading such a file turned 296.885 into
        # 180 with nothing said.  ty/tz stay at +/-180; they are sub-degree.
        self._seed_tx = _fspin(-360, 360, 2, 0.0, "°", step=DEFAULT_STEP_TILT)
        self._seed_ty = _fspin(-180, 180, 2, 0.0, "°", step=DEFAULT_STEP_TILT)
        self._seed_tz = _fspin(-180, 180, 2, 0.0, "°", step=DEFAULT_STEP_TILT)
        self._seed_tilts = (self._seed_tx, self._seed_ty, self._seed_tz)
        for w in self._seed_tilts:
            w.setToolTip(
                "Honoured by Four-stage / Bayesian / Joint (always), and by "
                "One-shot (with or without Multi-panel / a partial distortion "
                "selection). Plain One-shot / First-time only honour this if "
                "the installed calibrate() backend exposes initial-tilt "
                "kwargs — a warning is logged before Run if it doesn't and "
                "this value is non-zero.")
        self._seed_en_bc = QtWidgets.QCheckBox("Beam centre")
        self._seed_en_lsd = QtWidgets.QCheckBox("Lsd")
        self._seed_en_tx = QtWidgets.QCheckBox("tx")
        self._seed_en_ty = QtWidgets.QCheckBox("ty")
        self._seed_en_tz = QtWidgets.QCheckBox("tz")
        # Distortion seed: unlike BC/Lsd/tilts, its values (up to 15
        # coefficients) don't fit an inline spinbox, so they live behind
        # their own "…" dialog (DistortionSeedDialog) — self._seed_dist
        # (a plain {name: value} dict, set up in __init__) is what it edits.
        # Seeded independently of BC/Lsd/tilts (see _manual_seed_kwargs):
        # ticking only this lets a known detector distortion be fixed as a
        # starting point while BC/Lsd still auto-seed.
        self._seed_en_dist = QtWidgets.QCheckBox("Distortion")
        self._seed_dist_btn = QtWidgets.QToolButton(); self._seed_dist_btn.setText("…")
        self._seed_dist_btn.setToolTip(
            "Enter starting values for individual distortion coefficients "
            "(iso_R2/4/6, per-fold amplitude/phase).")
        self._seed_dist_btn.clicked.connect(self._edit_seed_distortion)
        self._seed_enables = (self._seed_en_bc, self._seed_en_lsd,
                              self._seed_en_tx, self._seed_en_ty, self._seed_en_tz,
                              self._seed_en_dist)
        #: Seed tick -> the widget(s) it gates. Driven by
        #: _sync_seed_enabled() rather than by per-checkbox toggled->setEnabled
        #: connections: a signal only fires when something *changes*, and
        #: apply_dict_to_widgets restores every tick with signals BLOCKED — so
        #: a project saved with seeds on reopened with the ticks checked and
        #: every value box greyed out. Ticked, and uneditable.
        self._seed_gated = (
            (self._seed_en_bc, (self._seed_bcy, self._seed_bcz)),
            (self._seed_en_lsd, (self._seed_lsd,)),
            (self._seed_en_tx, (self._seed_tx,)),
            (self._seed_en_ty, (self._seed_ty,)),
            (self._seed_en_tz, (self._seed_tz,)),
            (self._seed_en_dist, (self._seed_dist_btn,)),
        )

        for cb in self._seed_enables:
            cb.toggled.connect(self._on_seed_enable_changed)
        # The summary line carries the seed *values*, so it has to follow the
        # spin boxes too — they are edited in ManualSeedDialog, by Pick BC /
        # Pick Ring, by the Data Viewer's "Send →" and by the result feedback,
        # none of which touch the enable ticks that _on_seed_enable_changed
        # hangs off.
        #
        # _update_limits_label for the same reason: every ± window is centred
        # on a seed value, so the bounds it prints move when the seed does.
        # Only the limit widgets used to refresh it, so the line sat at
        # whichever seed was current when a ± box was last touched — a
        # "Send →" of Lsd 13900 mm left it still printing the window around
        # 13868, which reads as Send having been dropped or halved.
        for w in (self._seed_bcy, self._seed_bcz, self._seed_lsd, *self._seed_tilts):
            w.valueChanged.connect(self._update_seed_summary)
            w.valueChanged.connect(self._update_limits_label)
            w.valueChanged.connect(self._refresh_pixel_readout)
        self._wl.valueChanged.connect(self._update_limits_label)
        # The pixel readout is computed from the same seed geometry, and the
        # cursor does not move when a spin box does — pyqtgraph only emits
        # sigMouseMoved on actual motion, so without this the bar keeps
        # reporting 2θ for the geometry that was current when the mouse last
        # moved. Same staleness the Limits note above had.
        self._wl.valueChanged.connect(self._refresh_pixel_readout)
        self._pxY.valueChanged.connect(self._refresh_pixel_readout)
        self._pxZ_spin.valueChanged.connect(self._refresh_pixel_readout)
        self._pxZ_check.toggled.connect(self._refresh_pixel_readout)
        self._manual_seed_check.toggled.connect(self._on_seed_master_toggled)
        # Column 2 holds only two-decimal degree fields, so it does not need the
        # default numeric width — narrowing it is most of what keeps this card
        # (and with it the whole left column) from setting the panel width.
        for w in self._seed_tilts:
            w.setMaximumWidth(76)
        self._feedback_check = QtWidgets.QCheckBox("Feed result back to seed")
        self._feedback_check.setChecked(True)
        self._feedback_check.setToolTip(
            "After a calibration, copy the optimized BC / Lsd / tilts / distortion "
            "back into these seed fields so the next run starts from them.")
        self._seed_note = QtWidgets.QLabel("")
        self._seed_note.setStyleSheet(f"color:{S.ACCENT};font-size:10px"); self._seed_note.setWordWrap(True)
        self._seed_dialog = ManualSeedDialog(
            en_bc=self._seed_en_bc, bcy=self._seed_bcy, bcz=self._seed_bcz,
            en_lsd=self._seed_en_lsd, lsd=self._seed_lsd,
            en_tx=self._seed_en_tx, tx=self._seed_tx,
            en_ty=self._seed_en_ty, ty=self._seed_ty,
            en_tz=self._seed_en_tz, tz=self._seed_tz,
            en_dist=self._seed_en_dist, dist_btn=self._seed_dist_btn,
            feedback_check=self._feedback_check, note=self._seed_note, parent=self)
        self._seed_btn = QtWidgets.QPushButton("Manual seed…")
        self._seed_btn.setToolTip(
            "Choose which of BC / Lsd / tx / ty / tz / Distortion to seed "
            "the fit's starting point from. Use Pick BC / Pick Ring on the "
            "image to populate BC while this is open.")
        self._seed_btn.clicked.connect(self._open_seed_dialog)
        seed.body.addWidget(self._seed_btn)
        self._seed_summary_lbl = QtWidgets.QLabel("")
        self._seed_summary_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._seed_summary_lbl.setWordWrap(True)
        seed.body.addWidget(self._seed_summary_lbl)
        # BC without Lsd is a known-bad combination — Lsd/BC/tilt are near-
        # degenerate at small 2theta (see the Refine card / DECISIONS
        # 2026-09-09), so flag it rather than let a half-seeded geometry run away.
        self._seed_bc_lsd_warn = QtWidgets.QLabel("")
        self._seed_bc_lsd_warn.setStyleSheet("color:#d7861f;font-weight:bold;font-size:10px")
        self._seed_bc_lsd_warn.setWordWrap(True)
        self._seed_bc_lsd_warn.setVisible(False)
        seed.body.addWidget(self._seed_bc_lsd_warn)
        self._update_seed_dist_label()
        self._sync_seed_enabled()
        self._update_seed_summary()
        self._update_seed_bc_lsd_warning()
        self._update_seed_btn_style()
        lv.addWidget(seed)

        # ── Refine parameters ──
        refc = S.make_card("Refine parameters")
        self._refine_summary_lbl = QtWidgets.QLabel("")
        self._refine_summary_lbl.setStyleSheet(f"color:{S.ACCENT};font-size:10px")
        self._refine_summary_lbl.setWordWrap(True)
        refc.body.addWidget(self._refine_summary_lbl)
        # Three compact rows — the geometry scalars, the tilts, then the two
        # whole-image refinements. Each of these used to own a line of its own
        # (a hangover from the ± limits column that ran alongside them), which
        # made this the tallest card in a column that has to hold six.
        # Rows 0-1 (the scalars + tilts) get their own grid, separate from row
        # 2's (Distortion/Residual map need a wider "…" button column) — a
        # single shared grid ties every row's column widths together, which
        # is why row 2 stayed unchanged while rows 0-1 can go tighter here.
        rfl = QtWidgets.QGridLayout(); rfl.setHorizontalSpacing(2); rfl.setVerticalSpacing(4)
        rfl_bottom = QtWidgets.QGridLayout(); rfl_bottom.setSpacing(4)
        self._ref_lsd = QtWidgets.QCheckBox("Lsd"); self._ref_lsd.setChecked(True)
        self._ref_bc = QtWidgets.QCheckBox("BC"); self._ref_bc.setChecked(True)
        self._ref_ty = QtWidgets.QCheckBox("ty"); self._ref_ty.setChecked(True)
        self._ref_tz = QtWidgets.QCheckBox("tz"); self._ref_tz.setChecked(True)
        self._ref_tx = QtWidgets.QCheckBox("tx")
        self._ref_wl = QtWidgets.QCheckBox("Wavelength")
        self._ref_dist = QtWidgets.QCheckBox("Distortion"); self._ref_dist.setChecked(True)
        self._build_rc = QtWidgets.QCheckBox("Residual map"); self._build_rc.setChecked(True)
        for w in (self._ref_lsd, self._ref_bc, self._ref_ty, self._ref_tz,
                  self._ref_tx, self._ref_wl):
            w.toggled.connect(self._on_refine_flags_changed)
        # Let each Manual seed row say what the fit will actually do with
        # that parameter. The two panels are deliberately NOT wired together
        # — seeding something you are not refining is how you pin it to a
        # measured value — but without this they only look like they
        # disagree. Done here because this card is built after the dialog.
        self._seed_dialog.set_refine_boxes(
            {"BC": self._ref_bc, "Lsd": self._ref_lsd, "tx": self._ref_tx,
             "ty": self._ref_ty, "tz": self._ref_tz,
             "Distortion": self._ref_dist})

        # One row per parameter: the "refine?" checkbox on the left, and on the
        # right the ± window that bounds it — the two decisions about the same
        # parameter read together instead of living in separate blocks.
        # Column 1 is the limits column. Both calibrant kinds bound their fit,
        # but at different granularity and with different semantics — see
        # _LIMIT_ROWS_XTAL / _sync_limits_mode. The header says which is in
        # force, since "± window" means an opt-in bound for the manual fit and
        # an always-applied one for the crystalline backend.
        hdr = QtWidgets.QLabel("")
        hdr.setStyleSheet(f"color:{S.MUTED};font-size:10px"); hdr.setWordWrap(True)
        rfl.addWidget(hdr, 0, 0, 1, 5)
        self._limits_hdr = hdr
        #: limit slot -> (refine checkbox, sub-label). The single "BC" checkbox
        #: frees both centre coordinates, so it spans the BC_y/BC_z rows and
        #: those two rows carry their own sub-label; every other row is already
        #: named by its refine checkbox, so its limit checkbox has no text.
        #: "distortion" has no refine checkbox of its own here — the Distortion
        #: refine box lives in its own row below the grid — so it carries a
        #: sub-label too.
        limit_layout = {"Lsd": (self._ref_lsd, ""), "BC_y": (self._ref_bc, "BC_y"),
                        "BC_z": (None, "BC_z"), "ty": (self._ref_ty, ""),
                        "tz": (self._ref_tz, ""), "tx": (self._ref_tx, ""),
                        "wavelength_A": (self._ref_wl, ""),
                        "distortion": (None, "Distortion")}
        # Taken from PARAMETER_LIMIT_ROWS rather than restated, so this card,
        # the Manual seed dialog and _REFINE_BOXES cannot drift into three
        # different orders again (they had: this card ran ty, tz, tx and the
        # seed dialog led with BC).
        order = tuple(row[0] for row in PARAMETER_LIMIT_ROWS)
        #: slot -> (unit0, win0, abs_unit, decimals), dropping the label and
        #: fallback columns the dialog form of this block used.
        rows = {r[0]: (r[2], r[3], r[4], r[5]) for r in PARAMETER_LIMIT_ROWS}
        self._limit_widgets: dict = {}
        #: slot -> the QLabel naming the row (empty for rows already named by
        #: their refine checkbox in column 0).
        self._limit_name_lbls: dict = {}
        #: slot -> its manual-fit sub-label (see _XTAL_ROW_LABEL for the
        #: crystalline one, which differs because the rows merge).
        self._limit_row_sub: dict = {}
        self._limit_cells: list = [hdr]
        #: slot -> its own cells, so a row can be hidden on its own. The
        #: crystalline windows are coarser than the manual fit's (one for both
        #: centre coordinates, one for both tilts), so the surplus rows are
        #: hidden rather than shown as controls that would silently do nothing.
        self._limit_row_cells: dict = {}
        self._limit_row_index: dict = {}
        self._refine_grid = rfl
        for r, name in enumerate(order, start=1):
            unit0, win0, abs_unit, dec = rows[name]
            ref_box, sub = limit_layout[name]
            if ref_box is not None:
                # BC owns two rows; centring its box across them keeps it read
                # as the parent of both sub-rows rather than a peer of BC_y.
                span = 2 if name == "BC_y" else 1
                rfl.addWidget(ref_box, r, 0, span, 1)
            # The row's name is a QLabel beside the opt-in box, never the
            # box's own text. Crystalline rows hide the box (their window
            # always applies), and a checkbox kept visible only to caption
            # the row reads as a live control: it renders in the accent fill
            # whether or not it is enabled, so it looks ticked and clickable
            # while doing nothing. Held in one cell so the row still hides,
            # spans and aligns as a unit.
            cb = QtWidgets.QCheckBox("")
            lbl = QtWidgets.QLabel(sub)
            name_cell = QtWidgets.QWidget()
            nrow = QtWidgets.QHBoxLayout(name_cell)
            nrow.setContentsMargins(0, 0, 0, 0); nrow.setSpacing(4)
            nrow.addWidget(cb); nrow.addWidget(lbl); nrow.addStretch(1)
            spin = _fspin(0.0, 1e6, dec, win0, "")
            combo = _NoScrollComboBox()
            # A percentage of the seed is meaningless for the distortion
            # coefficients (they seed at 0), so that row is absolute-only.
            combo.addItems([u for u in ("%", abs_unit) if u])
            combo.setCurrentText(unit0 or abs_unit)
            spin.setEnabled(False); combo.setEnabled(False)
            cb.toggled.connect(spin.setEnabled)
            cb.toggled.connect(combo.setEnabled)
            cb.toggled.connect(self._on_limits_changed)
            spin.valueChanged.connect(self._on_limits_changed)
            combo.currentTextChanged.connect(self._on_limits_changed)
            # Placed straight into the outer grid rather than in a per-row
            # container, so the ± / value / unit columns line up down the card
            # even though the BC rows carry an extra sub-label.
            cells = (name_cell, QtWidgets.QLabel("±"), spin, combo)
            for c, w in enumerate(cells, start=1):
                rfl.addWidget(w, r, c)
            self._limit_widgets[name] = (cb, spin, combo)
            self._limit_name_lbls[name] = lbl
            self._limit_row_sub[name] = sub
            self._limit_row_cells[name] = list(cells)
            self._limit_row_index[name] = r
            self._limit_cells.extend(cells)
        self._limits_note = QtWidgets.QLabel("")
        self._limits_note.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._limits_note.setWordWrap(True)
        rfl.addWidget(self._limits_note, len(order) + 1, 0, 1, 5)
        self._limit_cells.append(self._limits_note)


        # Distortion gets a companion "…" button opening the per-coefficient dialog.
        # Held in a container widget (not a bare layout) so the whole row can be
        # hidden as one unit for calibrants that don't support distortion refinement.
        self._dist_btn = QtWidgets.QToolButton(); self._dist_btn.setText("…")
        self._dist_btn.setToolTip("Choose which distortion coefficients to refine "
                                  "(η-fold presets available).")
        self._dist_btn.clicked.connect(self._edit_distortion_coeffs)
        self._dist_row = QtWidgets.QWidget()
        drow = QtWidgets.QHBoxLayout(self._dist_row); drow.setContentsMargins(0, 0, 0, 0); drow.setSpacing(4)
        drow.addWidget(self._ref_dist); drow.addWidget(self._dist_btn); drow.addStretch(1)

        # Distortion needs two cells for its "…" button; Residual map takes the third.
        rfl_bottom.addWidget(self._dist_row, 0, 0, 1, 2)
        rfl_bottom.addWidget(self._build_rc, 0, 2)
        rfl_bottom.setColumnStretch(0, 1)
        rfl_bottom.setColumnStretch(1, 0)
        rfl_bottom.setColumnStretch(2, 1)
        # Spare width in the limits grid goes to an empty 5th column, not to
        # the unit combo — stretching a two-item combo across half the card
        # reads as an input you are meant to type into.
        rfl.setColumnStretch(4, 1)
        self._refine_grid = rfl
        self._refine_grid_bottom = rfl_bottom
        self._ref_dist.toggled.connect(lambda _=0: self._update_dist_label())
        self._ref_dist.toggled.connect(self._on_refine_flags_changed)
        refc.body.addLayout(rfl)
        refc.body.addLayout(rfl_bottom)

        lv.addWidget(refc)
        self._refc_card = refc
        self._update_dist_label()
        self._update_limits_label()
        self._update_refine_summary()

        # ── Advanced ──
        grp_adv = QtWidgets.QGroupBox("Advanced")
        grp_adv.setCheckable(True); grp_adv.setChecked(False)
        av = QtWidgets.QVBoxLayout(grp_adv); av.setContentsMargins(8, 6, 8, 6); av.setSpacing(5)
        self._n_iter = _NoScrollSpinBox(); self._n_iter.setRange(1, 1_000_000); self._n_iter.setValue(4)
        self._n_iter.setFixedWidth(53)    # ~50% narrower than the default sizeHint
        self._lm_iter = _NoScrollSpinBox(); self._lm_iter.setRange(1, 1_000_000); self._lm_iter.setValue(200)
        self._lm_iter.setFixedWidth(63)   # ~40% narrower than the default sizeHint
        self._device = _NoScrollComboBox(); self._device.addItems(["cpu", "cuda"])
        # 20% of the 72px sizeHint would be 14px — too narrow for a combo box
        # to show its text past the 18px drop-down arrow, so 50px is the
        # practical floor that still keeps "cuda" legible.
        self._device.setFixedWidth(50)
        iter_row = QtWidgets.QHBoxLayout(); iter_row.setSpacing(4)
        iter_row.addWidget(S.LabelRight("E-M iters:")); iter_row.addWidget(self._n_iter)
        iter_row.addSpacing(10)
        iter_row.addWidget(S.LabelRight("LM iters:")); iter_row.addWidget(self._lm_iter)
        iter_row.addStretch(1)
        av.addLayout(iter_row)
        device_row = QtWidgets.QHBoxLayout(); device_row.setSpacing(4)
        device_row.addWidget(S.LabelRight("Device:")); device_row.addWidget(self._device)
        device_row.addStretch(1)
        av.addLayout(device_row)
        lv.addWidget(grp_adv)
        self._adv_grp = grp_adv

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
        for w in (self._pn_y, self._pn_z, self._ps_y, self._ps_z, self._pg_y, self._pg_z):
            w.setFixedWidth(64)   # ~50% narrower than the default rendered width
        # Plain grid (not S.Form) with the stretch pushed onto a trailing
        # empty column, so the Y column stays flush left and the Z column
        # follows it with only the row's own spacing, instead of both field
        # columns being stretched apart across the card.
        pf2 = QtWidgets.QGridLayout(); pf2.setHorizontalSpacing(6); pf2.setVerticalSpacing(5)
        pf2.setContentsMargins(0, 0, 0, 0)
        for r, (lab_y, w_y, lab_z, w_z) in enumerate((
                ("panels Y:", self._pn_y, "panels Z:", self._pn_z),
                ("size Y:", self._ps_y, "size Z:", self._ps_z),
                ("gap Y:", self._pg_y, "gap Z:", self._pg_z))):
            pf2.addWidget(S.LabelRight(lab_y), r, 0)
            pf2.addWidget(w_y, r, 1)
            pf2.addWidget(S.LabelRight(lab_z), r, 2)
            pf2.addWidget(w_z, r, 3)
        pf2.setColumnStretch(4, 1)
        pv.addLayout(pf2)
        self._panel_grp = grp_panel
        lv.addWidget(grp_panel)

        lv.addStretch(1)

        # ── Output / Run / Save — a fixed footer, not part of the scrollable
        # content above, so it stays pinned to the bottom of this panel no
        # matter how many cards above it are expanded ──
        footer = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(footer); fv.setContentsMargins(2, 6, 2, 0); fv.setSpacing(6)
        fv.addWidget(S.hline())

        # Labelled "Working dir", but the attribute and state key stay
        # `_out_ed` / "out_ed" so projects saved before the rename still
        # restore into it — the directory means the same thing either way.
        self._out_ed = QtWidgets.QLineEdit()
        self._out_ed.setPlaceholderText("Working directory…")
        self._out_ed.setToolTip(
            "Working directory for this calibration.\n\n"
            "Intermediate files the fit produces (residual_corr.bin, the "
            "backend's calibration.json, panel shifts) go in a "
            f"{SCRATCH_DIRNAME}/ subfolder here, never next to your raw data. "
            "Delete that subfolder whenever you like — nothing you saved "
            "through a Save button lives in it.\n\n"
            "Defaults to the <expid>_bc analysis folder derived from the "
            "loaded data path. Also the folder the Save dialogs open on.")
        warn_if_path_missing(self._out_ed, self, is_output_dir=True)
        bou = _br(); bou.clicked.connect(lambda: self._out_ed.setText(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Working directory", browse_start_dir(self._out_ed.text())) or ""))
        # Autofill only fires into an empty field, by design, so without a
        # button there is no way back to the derived default after a project
        # restore or a typo.
        self._suggest_out_btn = QtWidgets.QPushButton("Suggest")
        self._suggest_out_btn.setToolTip(
            "Fill in the <expid>_bc analysis folder, read off the loaded data "
            "path — the _bc folder the data already sits in if there is one, "
            "otherwise derived from the <outroot>/<expid>/<detector>/<froot>/ "
            "layout. No need to type the Exp ID first.")
        self._suggest_out_btn.clicked.connect(self._apply_suggested_working_dir)
        outr = QtWidgets.QHBoxLayout(); outr.setSpacing(4)
        outr.addWidget(self._out_ed, 1); outr.addWidget(bou)
        outr.addWidget(self._suggest_out_btn)
        # A Form().row() stretches its field column to fill the footer's full
        # width, so the row grows/shrinks with the splitter instead of
        # staying pinned to the left like the Run/Save rows below it.
        out_row = QtWidgets.QHBoxLayout(); out_row.setSpacing(4)
        out_row.addWidget(S.LabelRight("Working dir:")); out_row.addLayout(outr, 1)
        fv.addLayout(out_row)

        # ── Run + Save ──
        self._run_btn = S.primary_btn("Run Calibration")
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._run_btn.setToolTip("Run Calibration")
        self._abort_btn = QtWidgets.QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.setToolTip("Cancel: returns control immediately and discards the "
                                   "result. The running computation finishes in the background.")
        self._abort_btn.clicked.connect(self._abort)
        # Full-width Run with Abort pinned to its right, and the two Save
        # buttons as equal halves below it — the arrangement this tab had
        # before the footer was introduced. Upstream centred all four at fixed
        # pixel widths; that leaves the block floating free of the Output field
        # it belongs with, and a fixed width cannot follow the splitter or a
        # different font scale.
        run_row = QtWidgets.QHBoxLayout(); run_row.setSpacing(6)
        run_row.addWidget(self._run_btn, 1); run_row.addWidget(self._abort_btn)
        fv.addLayout(run_row)
        self._prog = QtWidgets.QProgressBar(); self._prog.setRange(0, 0); self._prog.setVisible(False)
        fv.addWidget(self._prog)
        self._save_json_btn = QtWidgets.QPushButton("Save .json"); self._save_json_btn.setEnabled(False)
        self._save_json_btn.clicked.connect(self._save_json)
        self._save_json_btn.setToolTip("Save .json")
        self._save_ps_btn = QtWidgets.QPushButton("Save paramstest.txt"); self._save_ps_btn.setEnabled(False)
        self._save_ps_btn.clicked.connect(self._save_paramstest)
        fv.addLayout(S.button_grid([self._save_json_btn, self._save_ps_btn], 2))

        mid_col = QtWidgets.QWidget()
        mid_v = QtWidgets.QVBoxLayout(mid_col); mid_v.setContentsMargins(0, 0, 0, 0); mid_v.setSpacing(0)
        mid_v.addWidget(scroll, 1)
        mid_v.addWidget(footer)
        split.addWidget(mid_col)

        # Right: image + bottom tabs
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._img_view = PickableImageViewer()
        self._img_view.set_radial_readout_fn(self._radial_readout)
        self._img_view.bcPicked.connect(self._on_bc_picked)
        self._img_view.ringFitBC.connect(self._on_ring_fit_bc)
        self._img_view.dspacingPicksChanged.connect(self._on_dspacing_picks_changed)
        tb = self._img_view._toolbar_layout
        self._origin_btn = OriginToolButton(self._img_view)
        tb.addWidget(self._origin_btn)
        self._show_rings_check = QtWidgets.QCheckBox("Show rings"); self._show_rings_check.setChecked(True)
        self._show_rings_check.setToolTip(
            "Overlay the calibrant's predicted rings, drawn through the full "
            "forward model — fitted tilts and refined distortion both applied.")
        self._show_rings_check.toggled.connect(self._on_show_rings_toggled)
        tb.addWidget(self._show_rings_check)
        self._ring_status = QtWidgets.QLabel("")
        self._ring_status.setStyleSheet(f"color:{S.ACCENT};font-size:10px")
        tb.addWidget(self._ring_status)
        # Which pixels the fit will not see. The Mask card says how many are
        # masked, but a count cannot show *where* they are — whether a ring
        # arc you are about to pick points on is half excluded, or a module
        # gap sits across the beam centre. Off by default so it never hides
        # the rings on a tab whose whole job is picking them.
        self._show_mask_check = QtWidgets.QCheckBox("Mask")
        self._show_mask_check.setToolTip(
            "Shade the pixels excluded by the Mask card, in red. These are "
            "dropped from the fit and from every integration run from this "
            "tab.")
        self._show_mask_check.toggled.connect(self._on_show_mask_toggled)
        tb.addWidget(self._show_mask_check)
        self._mask_status = QtWidgets.QLabel("")
        self._mask_status.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        tb.addWidget(self._mask_status)
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
        # Flipping the display origin inverts the ViewBox's Y axis; the compass
        # points at the hutch, not the pixel grid, so it is re-derived rather
        # than carried along (widgets.build_lab_frame_axes_items).
        self._img_view.originChanged.connect(self._redraw_lab_axes_if_on)
        # Nothing here redraws rings. This tab overlays one geometry and one
        # only — the calibration it produced. Dialling a geometry in by eye and
        # watching rings follow is the Data Viewer's Ring simulation card; a
        # second, weaker copy of it driven by the seed fields meant ticking
        # "Use manual seed" painted rings that no calibration had endorsed.
        img_container = QtWidgets.QWidget()
        icl = QtWidgets.QVBoxLayout(img_container)
        icl.setContentsMargins(0, 0, 0, 0); icl.setSpacing(0)
        icl.addWidget(self._img_view, 1)
        icl.addWidget(self._build_frame_scrub_bar())
        right.addWidget(img_container)

        bot = QtWidgets.QTabWidget()
        self._prof_view = ProfileViewer()
        ptb = self._prof_view._toolbar_layout
        self._cal_r_bin = _fspin(0.1, 20.0, 2, 1.0, "px"); self._cal_r_bin.setFixedWidth(78)
        self._cal_eta_bin = _fspin(0.5, 360.0, 1, 5.0, "°"); self._cal_eta_bin.setFixedWidth(64)
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

        self._on_calibrant_changed(self._cal.currentText())

    # ── Data (from the loader panel) ──────────────────────────────

    def _build_frame_scrub_bar(self) -> QtWidgets.QWidget:
        """Prev/slider/next scrubber shown under the image viewer once the
        loaded source (HDF5 dataset / TIFF folder) has more than one frame —
        replaces the loader panel's compact "Frame:" spin, which this tab
        keeps hidden entirely (see ``DataLoaderPanel(hide_frame_field=True)``),
        mirroring ``CakeStackViewer``'s scrub bar in the Batch "Eta-R cakes"
        tab."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2); row.setSpacing(4)
        self._frame_prev_btn = QtWidgets.QToolButton(); self._frame_prev_btn.setText("◀")
        self._frame_prev_btn.setObjectName("frameNavBtn")
        self._frame_prev_btn.setToolTip("Previous frame")
        self._frame_prev_btn.clicked.connect(lambda: self._step_frame(-1))
        row.addWidget(self._frame_prev_btn)
        self._frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._frame_slider.setObjectName("frameNavSlider")
        self._frame_slider.setMinimum(0)
        self._frame_slider.setPageStep(1)
        self._frame_slider.valueChanged.connect(self._loader.set_frame)
        row.addWidget(self._frame_slider, 1)
        self._frame_next_btn = QtWidgets.QToolButton(); self._frame_next_btn.setText("▶")
        self._frame_next_btn.setObjectName("frameNavBtn")
        self._frame_next_btn.setToolTip("Next frame")
        self._frame_next_btn.clicked.connect(lambda: self._step_frame(1))
        row.addWidget(self._frame_next_btn)
        self._frame_lbl = QtWidgets.QLabel("")
        self._frame_lbl.setMinimumWidth(90)
        row.addWidget(self._frame_lbl)
        self._frame_scrub_bar = QtWidgets.QWidget()
        self._frame_scrub_bar.setLayout(row)
        self._frame_scrub_bar.setVisible(False)
        return self._frame_scrub_bar

    def _step_frame(self, delta: int):
        n = self._loader.n_frames()
        if n <= 1:
            return
        self._frame_slider.setValue(
            min(max(self._frame_slider.value() + delta, 0), n - 1))

    def _sync_frame_scrub_bar(self):
        n = self._loader.n_frames()
        self._frame_scrub_bar.setVisible(n > 1)
        if n <= 1:
            return
        idx = self._loader.frame_index()
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, n - 1)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)
        self._frame_lbl.setText(f"frame {idx + 1}/{n}")

    def _on_loader_data(self):
        """New frame / data from the loader — refresh the calibration image, the
        threshold-slider range, and the display."""
        self._sync_frame_scrub_bar()
        self._sync_avg_controls()
        self._image = self._source_image()
        # Before the early return, so loading a frame of a different shape
        # re-checks the mask against it either way.
        self._update_mask_overlay()
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

    # ── Mask overlay ─────────────────────────────────────────────

    def _on_show_mask_toggled(self, *_args):
        self._update_mask_overlay()

    def _update_mask_overlay(self):
        """Paint the excluded pixels over the calibration image.

        Shape-checked against what is actually on screen rather than assumed:
        a mask built on a differently-shaped frame (a different detector, or
        a file reloaded after the mask was made) would otherwise either throw
        inside pyqtgraph or, worse, paint a plausible-looking overlay that
        lines up with nothing. Says so in the status label instead.
        """
        if not hasattr(self, "_show_mask_check"):     # still building the UI
            return
        mask = self._loader.composite_mask()
        if mask is None:
            self._mask_status.setText("")
            self._img_view.clear_overlay()
            return
        mask = np.asarray(mask) != 0
        n = int(mask.sum())
        shape = None if self._image is None else np.asarray(self._image).shape
        if shape is not None and mask.shape != shape:
            self._mask_status.setText(
                f"mask {mask.shape} ≠ image {shape}")
            self._mask_status.setStyleSheet("color:#e0a030;font-size:10px")
            self._img_view.clear_overlay()
            return
        self._mask_status.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._mask_status.setText(
            f"{n:,} px ({100.0 * n / mask.size:.1f}%) excluded")
        if self._show_mask_check.isChecked():
            self._img_view.set_mask_overlay(mask)
            self._img_view.set_overlay_visible(True)
        else:
            self._img_view.clear_overlay()

    def _mask_log_line(self) -> str:
        """One line for the run log saying what the mask excluded.

        The Mask card states the mask the *tab* holds; this states the mask
        the *run* actually got (cfg["mask"] = composite_mask()), which is the
        question a log is asked afterwards. Refinement already logs its
        equivalent — see workers.py's "[refine] mask: none".
        """
        mask = self._loader.composite_mask()
        if mask is None:
            tot = "" if self._image is None else f" (all {np.asarray(self._image).size:,} px included)"
            return f"Mask: none{tot}"
        mask = np.asarray(mask) != 0
        n = int(mask.sum())
        return (f"Mask: {n:,} / {mask.size:,} px excluded "
                f"({100.0 * n / mask.size:.1f}%)")

    def _on_fields_changed(self):
        """Dark/bright/background changed — refresh the calibration preview
        (no autorange, matching Data Viewer's _on_fields_changed). The raw
        ``self._image`` is untouched; only the displayed, corrected render
        changes (see ``_show_calib_image``)."""
        self._update_mask_overlay()
        if self._image is None:
            return
        self._show_calib_image(autorange=False)

    def _on_metadata_detected(self, detected: dict):
        """Best-effort pxY/wavelength_A auto-detected from the just-loaded
        file (see helpers.detect_geometry_from_path) — only the fields
        actually present are applied.

        Suppressed while ``set_state()`` is restoring: that path re-loads the
        saved file, which re-fires this signal *after* the saved fields have
        been applied, so the file header would silently overwrite the
        wavelength/pixel size the user explicitly saved. A file's
        ``instrument/HEM/Energy`` is a best-effort hint for a fresh
        interactive load, and is routinely stale (e.g. an AgBH frame taken at
        72 keV whose header still reads 51 keV); an explicitly restored
        workspace value always wins over it."""
        if getattr(self, "_restoring_state", False):
            return
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

    # ── Mean of frames ────────────────────────────────────────────

    def _source_image(self):
        """Base image for calibration: the frame mean if enabled, else current.

        Deliberately raw (uncorrected, untransformed) — this feeds both the
        on-screen preview (via ``_show_calib_image``, which applies dark/
        bright/background and the Transforms checkboxes for display only) and
        the actual pipeline run, where ``CalibrationWorker`` applies bright/
        background, the same transform, and the backend subtracts dark
        internally. Pre-correcting here would double-apply them for the real
        run."""
        if self._avg_check.isChecked() and self._loader.n_frames() > 1:
            mean = self._loader.mean_frames(
                self._avg_start.value(), self._avg_end.value())
            if mean is not None:
                return mean
        return self._loader.current_frame()

    def _sync_avg_controls(self):
        """Enable/disable the frame-mean card and clamp spin ranges to the source."""
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
        start = self._avg_start.value()
        end = self._avg_end.value() or n
        end = min(end, n)
        cnt = len(range(max(0, start), end))
        self._avg_note.setText(f"mean of {cnt} of {n} frames (start={start}, "
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
        self._update_refine_summary()

    def _on_refine_flags_changed(self, *_args):
        self._update_refine_summary()
        self._on_dspacing_picks_changed()

    # ── Manual-fit parameter limits ───────────────────────────────

    def _limit_seed_values(self) -> dict:
        """Current seed geometry keyed by ``fit_geometry_from_ring_picks`` slot
        name, in *fit* units (Lsd in µm) — the centre each ± window is taken
        around."""
        return {"Lsd": self._seed_lsd.value() * 1000.0,
                "BC_y": self._seed_bcy.value(), "BC_z": self._seed_bcz.value(),
                "tx": self._seed_tx.value(), "ty": self._seed_ty.value(),
                "tz": self._seed_tz.value(), "wavelength_A": self._wl.value()}

    def _readout_geometry(self) -> tuple:
        """``(geom_dict, is_seed)`` for the pixel readout.

        Prefers the fitted result once there is one, so the bar agrees with
        the rings drawn over it; falls back to the live seed boxes, which is
        what makes the readout useful during picking — before any fit exists,
        which is when "what 2θ is this ring?" is actually being asked.
        """
        r = self._calib_result
        if r is not None:
            pxY = float(getattr(r, "pxY", 0.0) or self._pxY.value())
            return ({"Lsd": float(r.Lsd), "BC_y": float(r.BC_y),
                     "BC_z": float(r.BC_z), "pxY": pxY,
                     "pxZ": float(getattr(r, "pxZ", 0.0) or pxY),
                     "tx": float(getattr(r, "tx", 0.0) or 0.0),
                     "ty": float(getattr(r, "ty", 0.0) or 0.0),
                     "tz": float(getattr(r, "tz", 0.0) or 0.0),
                     "wavelength_A": float(getattr(r, "wavelength_A", 0.0)
                                           or self._wl.value())}, False)
        pxY = self._pxY.value()
        geom = dict(self._limit_seed_values())
        geom["pxY"] = pxY
        geom["pxZ"] = self._pxZ_spin.value() if self._pxZ_check.isChecked() else pxY
        return geom, True

    def _radial_readout(self, col, row) -> str:
        """2θ / Q / d / η under the cursor — see
        ``widgets.ImageViewer.set_radial_readout_fn``.

        Tagged ``(seed)`` while it comes from the seed boxes rather than a
        fit, so the number never quietly changes meaning the moment a
        calibration lands.
        """
        geom, is_seed = self._readout_geometry()
        text = pixel_readout_text(col, row, geom)
        return f"{text}  (seed)" if text and is_seed else text

    def _refresh_pixel_readout(self, *_):
        """Re-render the readout against the current geometry without the
        cursor having moved. See the connections in ``_build_ui``."""
        self._img_view._refresh_coord_bar()

    @property
    def _limits(self) -> dict:
        """``{slot: {"on", "value", "unit"}}`` read straight off the inline
        limit widgets — they are the single source of truth, so there is no
        separate copy to keep in sync (and project state persists them via
        ``_state_widgets`` like any other spin box)."""
        return {name: {"on": cb.isChecked(), "value": spin.value(),
                       "unit": combo.currentText()}
                for name, (cb, spin, combo) in self._limit_widgets.items()}

    #: Limit rows each calibrant kind actually has a bound for. The manual fit
    #: bounds every free parameter individually; ``CalibrationParams`` carries
    #: one window for both centre coordinates (``tolBC``), one for both refined
    #: tilts (``tolTilts``) and one for all fifteen distortion slots
    #: (``tolDistortion``), and never refines tx at all — so the surplus rows
    #: are hidden rather than left as controls that would do nothing.
    _LIMIT_ROWS_DSP  = ("Lsd", "BC_y", "BC_z", "tx", "ty", "tz", "wavelength_A")
    _LIMIT_ROWS_XTAL = ("Lsd", "BC_y", "ty", "wavelength_A", "distortion")
    #: Crystalline row -> the ``CalibrationParams`` window it drives.
    _XTAL_TOL_FIELD = {"Lsd": "tolLsd", "BC_y": "tolBC", "ty": "tolTilts",
                       "wavelength_A": "tolWavelength", "distortion": "tolDistortion"}
    #: Crystalline sub-labels. Only ``distortion`` needs one: every other
    #: crystalline row is named by the refine checkbox in column 0, and the
    #: manual fit's "BC_y"/"BC_z" would misname a window that covers both.
    _XTAL_ROW_LABEL = {"distortion": "Distortion"}

    def _limit_rows_for_mode(self, is_dsp: bool) -> tuple:
        return self._LIMIT_ROWS_DSP if is_dsp else self._LIMIT_ROWS_XTAL

    def _sync_limits_mode(self, is_dsp: bool) -> None:
        """Shape the limits column for the calibrant kind.

        The two kinds differ in more than which rows exist. For the manual fit
        a row that is off means *unbounded*, so rows are opt-in. For the
        crystalline backend the windows are **always** applied — an untouched
        fit already runs at ±15 mm / ±20 px / ±3° — so "off" there would state
        something false. Crystalline rows are therefore always on, their enable
        checkbox is hidden, and they are seeded with the values actually in
        force so the card shows the real constraint rather than an invitation
        to add one.
        """
        if self._limits_mode_is_dsp == is_dsp:
            return
        if self._limits_mode_is_dsp is not None:      # remember the outgoing mode
            snap = self._limits
            if self._limits_mode_is_dsp:
                self._limit_state_dsp = snap
            else:
                self._limit_state_xtal = snap
        want = set(self._limit_rows_for_mode(is_dsp))
        for name, cells in self._limit_row_cells.items():
            for w in cells:
                w.setVisible(name in want)
        for name, (cb, _spin, _combo) in self._limit_widgets.items():
            # Crystalline: the window always applies, so the opt-in box is
            # meaningless — hide it outright and hold it checked so _limits()
            # reports the row as live. The row keeps its name either way: that
            # is the QLabel next to the box, not the box's own text.
            cb.setVisible(is_dsp and name in want)
            cb.setEnabled(is_dsp)
            lbl = self._limit_name_lbls[name]
            lbl.setText(self._limit_row_sub[name] if is_dsp
                        else self._XTAL_ROW_LABEL.get(name, ""))
            lbl.setVisible(name in want and bool(lbl.text()))
        # The crystalline tilt window is one value for ty and tz (tolTilts), so
        # span it across both rows the way the BC refine box already spans its
        # pair — parked on the ty row alone it reads as bounding only ty.
        for w, col in zip(self._limit_row_cells["ty"], (1, 2, 3, 4)):
            self._refine_grid.removeWidget(w)
            self._refine_grid.addWidget(w, self._limit_row_index["ty"], col,
                                        1 if is_dsp else 2, 1)
        incoming = self._limit_state_dsp if is_dsp else self._limit_state_xtal
        if incoming is None:
            incoming = (self._dsp_default_limit_state() if is_dsp
                        else self._xtal_default_limit_state())
        self._apply_limit_state(incoming, force_on=not is_dsp, rows=want)
        self._limits_hdr.setText(
            "Limits — bound a refined parameter to ± a window around its seed "
            "value:" if is_dsp else
            "Limits — the MIDAS backend always bounds the fit to a ± window "
            "around the seed. These are the windows in force; edit to tighten "
            "or loosen them.")
        self._limits_mode_is_dsp = is_dsp
        self._update_limits_label()
        self._sync_seed_steps()

    def _dsp_default_limit_state(self) -> dict:
        """Manual-fit rows start off, at the ``PARAMETER_LIMIT_ROWS`` defaults —
        an untouched card must leave the fit unbounded. Spelled out rather than
        left to whatever the other mode happened to set, so arriving from a
        crystalline calibrant does not inherit its always-on rows."""
        return {r[0]: {"on": False, "value": r[3], "unit": r[2] or r[4]}
                for r in PARAMETER_LIMIT_ROWS}

    def _xtal_default_limit_state(self) -> dict:
        """Crystalline rows prefilled from the ``tol*`` defaults actually in
        force, read off the installed backend rather than hardcoded."""
        from midas_gui.calib import tol_defaults
        d = tol_defaults()
        # tolLsd is stored in µm; the row is entered in mm.
        vals = {"Lsd": d["tolLsd"] / 1000.0, "BC_y": d["tolBC"],
                "ty": d["tolTilts"], "wavelength_A": d["tolWavelength"],
                "distortion": d["tolDistortion"]}
        units = {"Lsd": "mm", "BC_y": "px", "ty": "°",
                 "wavelength_A": "Å", "distortion": ""}
        return {n: {"on": True, "value": v, "unit": units[n]}
                for n, v in vals.items()}

    def _apply_limit_state(self, state: dict, *, force_on: bool, rows) -> None:
        for name, (cb, spin, combo) in self._limit_widgets.items():
            row = (state or {}).get(name)
            if isinstance(row, dict):
                if row.get("value") is not None:
                    spin.setValue(float(row["value"]))
                idx = combo.findText(str(row.get("unit", "")))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                cb.setChecked(bool(row.get("on", False)))
            if force_on and name in rows:
                cb.setChecked(True)
            spin.setEnabled(cb.isChecked())
            combo.setEnabled(cb.isChecked())

    def _on_limits_changed(self, *_args):
        self._update_limits_label()
        self._sync_seed_steps()

    def _n_limits_set(self) -> int:
        rows = set(self._limit_rows_for_mode(bool(self._limits_mode_is_dsp)))
        return sum(1 for n, (cb, _s, _c) in self._limit_widgets.items()
                   if n in rows and cb.isChecked())

    #: Limit row -> the seed spin box its window is centred on. Used both to
    #: report the resulting range and to size that box's arrow step.
    _LIMIT_SEED_BOX = {"Lsd": "_seed_lsd", "BC_y": "_seed_bcy", "BC_z": "_seed_bcz",
                       "tx": "_seed_tx", "ty": "_seed_ty", "tz": "_seed_tz"}
    #: Crystalline rows that drive more than one seed box (merged windows).
    _XTAL_STEP_EXTRA = {"BC_y": ("_seed_bcz",), "ty": ("_seed_tz",)}
    #: Fraction of the *full* [lo, hi] span used as the arrow step, so a ±5 mm
    #: window steps 1 mm and a ±20 px window steps 4 px.
    _STEP_FRACTION_OF_RANGE = 0.10

    def _sync_seed_steps(self, *_args):
        """Size each seed box's arrow step from its own limit window.

        A window is a statement about how far the value can sensibly move, so
        it is a better step than a fixed constant: tightening Lsd to ±2 mm
        should also stop the arrow jumping 1 mm at a time. Rows without a live
        window fall back to the configured DEFAULT_STEP_* constants.
        """
        is_dsp = bool(self._limits_mode_is_dsp)
        rows = set(self._limit_rows_for_mode(is_dsp))
        fallback = {"_seed_lsd": DEFAULT_STEP_LSD_MM,
                    "_seed_bcy": DEFAULT_STEP_BC, "_seed_bcz": DEFAULT_STEP_BC,
                    "_seed_tx": DEFAULT_STEP_TILT, "_seed_ty": DEFAULT_STEP_TILT,
                    "_seed_tz": DEFAULT_STEP_TILT}
        steps = dict(fallback)
        seed = self._limit_seed_values()
        for name, attr in self._LIMIT_SEED_BOX.items():
            if name not in rows or name not in seed:
                continue
            cb, spin, combo = self._limit_widgets[name]
            if not cb.isChecked():
                continue
            try:
                lo, hi = limit_window(name, seed[name], spin.value(),
                                      combo.currentText())
            except KeyError:
                continue
            span = abs(hi - lo) * (1e-3 if name == "Lsd" else 1.0)   # µm → mm
            step = span * self._STEP_FRACTION_OF_RANGE
            if step <= 0:
                continue
            for target in (attr,) + (self._XTAL_STEP_EXTRA.get(name, ())
                                     if not is_dsp else ()):
                steps[target] = step
        for attr, step in steps.items():
            box = getattr(self, attr, None)
            if box is not None:
                box.setStepType(QtWidgets.QAbstractSpinBox.DefaultStepType)
                box.setSingleStep(step)

    #: Crystalline window -> (label, display scale from fit units, unit, fmt).
    _XTAL_NOTE_ROWS = (("tolLsd", "Lsd", 1e-3, "mm", ".4g"),
                       ("tolBC", "BC", 1.0, "px", ".4g"),
                       ("tolTilts", "tilts", 1.0, "°", ".4g"),
                       ("tolWavelength", "λ", 1.0, "Å", ".3g"),
                       ("tolDistortion", "distortion", 1.0, "", ".3g"))

    def _update_limits_label(self, *_args):
        if self._limits_mode_is_dsp is False:
            # The crystalline windows are always applied and are centred on
            # whatever seed the fit starts from, so the manual fit's
            # "unbounded"/"needs a manual seed" wording would both be wrong.
            from midas_gui.calib import tol_defaults
            tols = self._crystalline_tols() or {}
            eff = {**tol_defaults(), **tols}
            bits = [f"{lbl} ±{eff[f] * sc:{fmt}}{(' ' + u) if u else ''}"
                    for f, lbl, sc, u, fmt in self._XTAL_NOTE_ROWS if f in eff]
            tail = ("" if tols else
                    "  — backend defaults; edit any to tighten or loosen")
            self._limits_note.setText(
                "Always applied, centred on the seed: " + "   ".join(bits) + tail)
            return
        bounds, skipped = self._limit_bounds()
        if not bounds and not skipped:
            self._limits_note.setText("No limits set — the fit is unbounded.")
            return
        bits = [f"{n} ∈ [{lo * (1e-3 if n == 'Lsd' else 1):.5g}, "
                f"{hi * (1e-3 if n == 'Lsd' else 1):.5g}]"
                for n, (lo, hi) in sorted((bounds or {}).items())]
        if skipped:
            bits.append(f"{', '.join(sorted(skipped))} ignored (needs 'Use manual seed')")
        self._limits_note.setText("   ".join(bits))

    #: tol* field -> the result attributes it bounds, the seed slot each is
    #: centred on, and the refine flag that had to be on for the value to move
    #: at all. One window covers both centre coordinates and both tilts.
    _XTAL_AT_LIMIT = {"tolLsd": (("Lsd", "Lsd", "Lsd"),),
                      "tolBC": (("BC_y", "BC_y", "BC"), ("BC_z", "BC_z", "BC")),
                      "tolTilts": (("ty", "ty", "ty"), ("tz", "tz", "tz")),
                      "tolWavelength": (("wavelength_A", "wavelength_A",
                                         "Wavelength"),)}

    def _crystalline_at_limit(self, result) -> set:
        """Which crystalline parameters came back sitting on their window.

        A bounded fit that stops at its bound is reporting the bound, not a
        measurement — the same thing ``at_limit`` says for the manual fit, and
        the reason the windows being visible is not on its own enough.

        Only decidable when the window's centre is known, i.e. with "Use manual
        seed" on. An auto-seeded run is centred on a seed the GUI never sees,
        so this reports nothing rather than guessing.
        """
        if self._limits_mode_is_dsp is not False:
            return set()
        if not self._manual_seed_check.isChecked():
            return set()
        from midas_gui.calib import tol_defaults
        eff = {**tol_defaults(), **(self._crystalline_tols() or {})}
        seed = self._limit_seed_values()
        refine = self._last_refine_flags or self._refine_flags()
        out = set()
        for field, pairs in self._XTAL_AT_LIMIT.items():
            tol = eff.get(field)
            if not tol or not math.isfinite(tol):
                continue
            for attr, slot, flag in pairs:
                # A parameter that was held fixed never moved, so it cannot
                # have been stopped by its window — saying otherwise would be
                # noise on exactly the rows the user already knows are pinned.
                if not refine.get(flag, True):
                    continue
                centre, got = seed.get(slot), getattr(result, attr, None)
                if centre is None or got is None:
                    continue
                # Lsd is seeded in µm here and returned in µm, so no scaling.
                if abs(float(got) - float(centre)) >= tol * (1.0 - 1e-6):
                    out.add(slot)
        return out

    def _crystalline_tols(self) -> Optional[dict]:
        """The ``tol*`` overrides for a crystalline run, or ``None`` when every
        row still sits at the backend default (which keeps ``run_pipeline`` on
        the plain ``calibrate()`` path — see ``calib.tols_are_default``)."""
        from midas_gui.calib import tols_are_default
        if self._limits_mode_is_dsp:
            return None
        seed = self._limit_seed_values()
        out = {}
        for name in self._LIMIT_ROWS_XTAL:
            cb, spin, combo = self._limit_widgets[name]
            if not cb.isChecked():
                continue
            field = self._XTAL_TOL_FIELD[name]
            if name == "distortion":
                # Seeds at 0 and has no seed box, so the entered value is the
                # window itself rather than something to centre on a value.
                out[field] = float(spin.value())
                continue
            centre = seed.get(name, 0.0)
            lo, hi = limit_window(name, centre, spin.value(), combo.currentText())
            out[field] = abs(hi - lo) / 2.0
        return None if tols_are_default(out) else out

    #: Limit rows whose ± window is centred on a manual-seed field. Without
    #: "Use manual seed" the fit auto-seeds Lsd/BC from the picks themselves
    #: and starts the tilts at 0, so those spin boxes are not the centre the
    #: window would be taken around — only the wavelength is always live.
    _SEED_DEPENDENT_LIMITS = ("Lsd", "BC_y", "BC_z", "tx", "ty", "tz")

    def _limit_bounds(self):
        """``(bounds, skipped)`` — the enabled limit rows as the ``bounds``
        dict :func:`~midas_gui.helpers.fit_geometry_from_ring_picks` expects,
        plus the names of any dropped because their window has no defined
        centre. ``bounds`` is ``None`` when nothing is bounded, which keeps
        the fit on the unbounded Levenberg-Marquardt path it has always
        used.

        Crystalline calibrants never reach the manual fit, and their rows are
        always on and coarser (see :meth:`_crystalline_tols`), so this reports
        nothing for them rather than handing those windows to a solver that
        will not run."""
        if self._limits_mode_is_dsp is False:
            return None, []
        seed = self._limit_seed_values()
        out, skipped = {}, []
        rows = set(self._limit_rows_for_mode(True))
        for name, row in (self._limits or {}).items():
            if name not in rows:
                continue
            if not (isinstance(row, dict) and row.get("on")) or name not in seed:
                continue
            en = self._seed_enable_for_slot(name)
            if en is not None and not en.isChecked() and name in self._SEED_DEPENDENT_LIMITS:
                skipped.append(name)
                continue
            try:
                out[name] = limit_window(name, seed[name], float(row.get("value", 0.0)),
                                         str(row.get("unit", "")))
            except KeyError:
                continue                      # unknown slot in a stale project file
        return (out or None), skipped

    def _seed_enable_for_slot(self, name: str):
        """The granular "include in seed" checkbox governing limit-row
        ``name`` (``Lsd``/``BC_y``/``BC_z``/``tx``/``ty``/``tz``), or ``None``
        for a slot with no seed concept (``wavelength_A``)."""
        return {
            "Lsd": self._seed_en_lsd, "BC_y": self._seed_en_bc, "BC_z": self._seed_en_bc,
            "tx": self._seed_en_tx, "ty": self._seed_en_ty, "tz": self._seed_en_tz,
        }.get(name)

    # ── Per-parameter manual seed: enable flags, dialog, sparse dict ──

    def _open_seed_dialog(self):
        self._seed_dialog.show()
        self._seed_dialog.raise_()
        self._seed_dialog.activateWindow()

    def _on_seed_enable_changed(self, *_args):
        """A granular enable flag changed — resync the derived master
        tri-state (used for legacy project state + Hydra's cross-panel sync)
        and the visible summary, without re-firing ``_on_seed_master_toggled``."""
        self._syncing_seed_master = True
        try:
            n_on = sum(cb.isChecked() for cb in self._seed_enables)
            if n_on == 0:
                self._manual_seed_check.setCheckState(QtCore.Qt.Unchecked)
            elif n_on == len(self._seed_enables):
                self._manual_seed_check.setCheckState(QtCore.Qt.Checked)
            else:
                self._manual_seed_check.setCheckState(QtCore.Qt.PartiallyChecked)
        finally:
            self._syncing_seed_master = False
        self._sync_seed_enabled()
        self._update_seed_summary()
        self._update_seed_bc_lsd_warning()
        self._update_seed_btn_style()

    def _sync_seed_enabled(self):
        """A seed value box is editable exactly when its tick is on.

        Derived from each checkbox's CURRENT state every time, so it is right
        whether the tick was clicked, set programmatically, or restored from
        a project with signals blocked."""
        for cb, widgets in self._seed_gated:
            on = cb.isChecked()
            for w in widgets:
                w.setEnabled(on)

    def _update_seed_btn_style(self):
        """Green "Manual seed..." once at least one parameter is ticked, so an
        active (and easily forgotten) seed is visible without opening the dialog."""
        active = any(cb.isChecked() for cb in self._seed_enables)
        self._seed_btn.setStyleSheet(S.SUCCESS_BTN_QSS if active else "")

    def _update_seed_bc_lsd_warning(self):
        bad = self._seed_en_bc.isChecked() and not self._seed_en_lsd.isChecked()
        self._seed_bc_lsd_warn.setVisible(bad)
        if bad:
            self._seed_bc_lsd_warn.setText(
                "WARNING: BC guess is provided, but Lsd guess is missing. It is "
                "recommended to either provide a good guess for both BC and Lsd, "
                "or skip the guess entirely and let MIDAS run it's own auto-seed "
                "program.")

    def _on_seed_master_toggled(self, *_args):
        """``_manual_seed_check`` set programmatically or by a test/legacy
        project restore — bulk on/off over every granular flag. Ignored while
        ``_on_seed_enable_changed`` is itself updating the master's tri-state,
        so the two don't bounce off each other."""
        if self._syncing_seed_master:
            return
        want = self._manual_seed_check.isChecked()   # PartiallyChecked reads as True
        for cb in self._seed_enables:
            cb.setChecked(want)

    #: Seed slot -> how its current value reads in the summary line. Same
    #: units as the spin boxes in ManualSeedDialog (Lsd in mm, not the µm the
    #: fit uses), so the two always agree on screen.
    @staticmethod
    def _seed_num(v: float, dec: int) -> str:
        """``v`` at the spin box's own precision, trailing zeros trimmed — a
        BC entered to three decimals must not read back rounded to one, but
        a round 1382.4 mm should not pad out to 1382.400 either."""
        return f"{v:.{dec}f}".rstrip("0").rstrip(".") or "0"

    def _seed_value_text(self, name: str) -> str:
        if name == "BC":
            return (f"{self._seed_num(self._seed_bcy.value(), 3)}, "
                    f"{self._seed_num(self._seed_bcz.value(), 3)} px")
        if name == "Lsd":
            return f"{self._seed_num(self._seed_lsd.value(), 3)} mm"
        if name == "Distortion":
            return f"{len(self._seed_dist)} coeff"
        spin = {"tx": self._seed_tx, "ty": self._seed_ty, "tz": self._seed_tz}[name]
        return f"{self._seed_num(spin.value(), 2)}°"

    _SEED_SLOTS = ("BC", "Lsd", "tx", "ty", "tz", "Distortion")

    def _update_seed_summary(self, *_args):
        """Name *and* value every seeded parameter.

        The seed spin boxes live inside ManualSeedDialog, which is non-modal
        and normally closed, so these numbers were invisible from the tab —
        yet they are the starting point of the fit and the centre every ±
        window in the Refine card is taken around ("± a window around its
        seed value" with no value anywhere on screen). Unseeded parameters
        are listed as auto rather than dropped, so a glance says what the fit
        actually starts from rather than only what was overridden.
        """
        on, off = [], []
        for cb, name in zip(self._seed_enables, self._SEED_SLOTS):
            (on if cb.isChecked() else off).append(name)
        if not on:
            self._seed_summary_lbl.setText(
                "Fully automatic (no manual seed) — BC/Lsd auto-seeded from the "
                "image, tilts start at 0°")
            return
        txt = "Seeding: " + " · ".join(f"{n} {self._seed_value_text(n)}" for n in on)
        if off:
            txt += f"   (auto: {', '.join(off)})"
        self._seed_summary_lbl.setText(txt)

    def _enable_seed(self, **flags):
        """Tick specific granular seed-enable flags by slot name, e.g.
        ``self._enable_seed(BC=True)``. Never unticks — callers that populate
        a subset of fields (Pick BC, a partial geometry dict) should not
        silently disable a parameter the user already enabled."""
        by_name = dict(zip(("BC", "Lsd", "tx", "ty", "tz", "Distortion"), self._seed_enables))
        for name, on in flags.items():
            if on:
                by_name[name].setChecked(True)

    def _manual_seed_kwargs(self) -> dict:
        """Sparse seed dict for ``calib.run_pipeline``'s ``manual_seed`` cfg
        key: only the parameters actually ticked "include in seed" are
        present. BC_y/BC_z always travel together (see
        ``calib._resolve_seed``). Distortion is seeded independently of
        BC/Lsd/tilts — ticking only "Distortion" seeds a known detector
        distortion while BC/Lsd/tilts stay on the auto-seeder, and vice
        versa — from whichever coefficients are in ``self._seed_dist``
        (typed in via ``_edit_seed_distortion``, or carried forward from a
        prior result by "Feed result back to seed")."""
        manual: dict = {}
        if self._seed_en_bc.isChecked():
            manual["BC_y"] = self._seed_bcy.value()
            manual["BC_z"] = self._seed_bcz.value()
        if self._seed_en_lsd.isChecked():
            manual["Lsd"] = self._seed_lsd.value() * 1000.0   # mm display → µm
        for en, key, w in ((self._seed_en_tx, "tx", self._seed_tx),
                           (self._seed_en_ty, "ty", self._seed_ty),
                           (self._seed_en_tz, "tz", self._seed_tz)):
            if en.isChecked():
                manual[key] = w.value()
        if self._seed_en_dist.isChecked() and self._seed_dist:
            manual["distortion"] = dict(self._seed_dist)
        return manual

    def _edit_seed_distortion(self):
        dlg = DistortionSeedDialog(self._seed_dist, self)
        if dlg.exec_():
            self._seed_dist = dlg.values()
            if self._seed_dist and not self._seed_en_dist.isChecked():
                self._seed_en_dist.setChecked(True)
            self._update_seed_dist_label()

    def _update_seed_dist_label(self):
        n = len(self._seed_dist)
        self._seed_en_dist.setText(f"Distortion ({n}/15)" if n else "Distortion")
        # _seed_dist has no widget of its own, so this is the one place its
        # size changes — and the summary line quotes that count.
        self._update_seed_summary()

    # ── Seed feedback from a result ───────────────────────────────

    def _seed_from_result(self, result):
        """Copy optimized geometry from a result into the seed fields — a
        completed fit is a full geometry, so every parameter it refined is
        enabled and written back.

        ``tx`` is the exception: it is an input no crystalline pipeline
        refines, so feeding ``result.tx`` back can only repeat the seed or
        overwrite it with 0, and the second is what happened on a panel
        fitted out of a Hydra array. Only the manual d-spacing fit refines
        it, and only when asked — :func:`helpers.result_refined_tx` is the
        single place that distinction is made."""
        self._enable_seed(BC=True, Lsd=True, ty=True, tz=True)
        self._seed_bcy.setValue(float(result.BC_y))
        self._seed_bcz.setValue(float(result.BC_z))
        self._seed_lsd.setValue(float(result.Lsd) / 1000.0)   # µm → mm
        if result_refined_tx(result):
            self._enable_seed(tx=True)
            self._seed_tx.setValue(float(getattr(result, "tx", 0.0) or 0.0))
        self._seed_ty.setValue(float(getattr(result, "ty", 0.0) or 0.0))
        self._seed_tz.setValue(float(getattr(result, "tz", 0.0) or 0.0))
        if getattr(result, "wavelength_A", None):
            self._wl.setValue(float(result.wavelength_A))
        self._seed_dist = dict(getattr(result, "distortion", {}) or {})
        if self._seed_dist:
            self._enable_seed(Distortion=True)
        self._update_seed_dist_label()
        self._seed_note.setText(
            f"Seed updated from the last fit: BC=({result.BC_y:.1f}, {result.BC_z:.1f}) px, "
            f"Lsd={float(result.Lsd) / 1000:.3f} mm, "
            f"tx={float(getattr(result, 'tx', 0.0) or 0.0):.2f}°, "
            f"ty={float(getattr(result, 'ty', 0.0) or 0.0):.2f}°, "
            f"tz={float(getattr(result, 'tz', 0.0) or 0.0):.2f}°.")

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
        self._enable_seed(BC=True, Lsd=True, tx=True, ty=True, tz=True)
        self._seed_bcy.setValue(float(g["BC_y"]))
        self._seed_bcz.setValue(float(g["BC_z"]))
        self._seed_lsd.setValue(float(g["Lsd"]) / 1000.0)   # µm → mm display
        self._seed_tx.setValue(float(g.get("tx") or 0.0))
        self._seed_ty.setValue(float(g.get("ty") or 0.0))
        self._seed_tz.setValue(float(g.get("tz") or 0.0))
        im_trans = g.get("im_trans") or []
        if g.get("im_trans_in_file", True):
            self._flip_y.setChecked(1 in im_trans)
            self._flip_z.setChecked(2 in im_trans)
            self._transp.setChecked(3 in im_trans)
        # else: the file is silent on the transform (a backend-written
        # calibration.json records none; a .poni has no such concept). The
        # geometry in it is only meaningful in the frame it was fitted in, so
        # unticking the boxes here would quietly re-frame it — leave whatever
        # the user has set and say so in the log.
        # The distortion coefficients are part of the geometry too: dropping
        # them on load seeds the next fit from a distortion-free detector and
        # throws away the harmonics this calibration was refined with. Zeros
        # are skipped — a zero seed is the default, so carrying all fifteen
        # would only inflate the "Distortion (n)" count with empty slots.
        dist = {k: float(v) for k, v in (g.get("distortion") or {}).items()
                if float(v) != 0.0}
        if dist:
            self._seed_dist = dist
            self._enable_seed(Distortion=True)
            self._update_seed_dist_label()
        self._seed_note.setText(
            f"Loaded {Path(path).name}: λ={g['wavelength_A']:.5f} Å, px={g['pxY']:.2f} µm, "
            f"BC=({g['BC_y']:.2f}, {g['BC_z']:.2f}), Lsd={g['Lsd']/1000:.3f} mm, "
            f"tx={g.get('tx') or 0.0:.3f}°, ty={g.get('ty') or 0.0:.3f}°, "
            f"tz={g.get('tz') or 0.0:.3f}°.")
        self._log.append(f"Calibration file loaded: {path}")
        if dist:
            self._log.append(
                f"Distortion seeded from file: {len(dist)} non-zero coefficient(s) "
                f"({', '.join(sorted(dist))}).")
        if not g.get("im_trans_in_file", True):
            self._log.append(
                "Note: this file records no image transform, so Flip Y / Flip Z / "
                "Transpose were left as they are. The loaded geometry is only valid "
                "in the frame it was fitted in — check them against that run.")
        if im_trans and im_trans != [c for c in (1, 2, 3) if c in im_trans]:
            self._log.append(
                f"Note: ImTransOpt order in file ({im_trans}) differs from the "
                "fixed Flip Y → Flip Z → Transpose order used here; checkboxes "
                "were set but may not exactly reproduce the file's composition.")

    def apply_geometry(self, g: dict):
        """Set λ / pixel size / seed BC + Lsd from a geometry dict (Data Viewer)."""
        if not g:
            return
        if g.get("wavelength_A") is not None:
            self._wl.setValue(float(g["wavelength_A"]))
        if g.get("pxY") is not None:
            self._pxY.setValue(float(g["pxY"]))
        # Only the geometry keys this dict actually carries get enabled as
        # seed — a Data Viewer "Get" that supplies BC but not Lsd (say)
        # should not also silently seed Lsd from whatever was last typed.
        if g.get("BC_y") is not None and g.get("BC_z") is not None:
            self._seed_bcy.setValue(float(g["BC_y"]))
            self._seed_bcz.setValue(float(g["BC_z"]))
            self._enable_seed(BC=True)
        if g.get("Lsd") is not None:
            self._seed_lsd.setValue(float(g["Lsd"]) / 1000.0)   # µm → mm display
            self._enable_seed(Lsd=True)
        if g.get("tx") is not None:
            self._seed_tx.setValue(float(g["tx"]))
            self._enable_seed(tx=True)
        if g.get("ty") is not None:
            self._seed_ty.setValue(float(g["ty"]))
            self._enable_seed(ty=True)
        if g.get("im_trans") is not None and g.get("im_trans_in_file", True):
            # "im_trans_in_file" is only set by helpers.geometry_fields_from_file
            # and is False when the source file said nothing about the frame;
            # the Data Viewer builds its dict from live checkboxes, so it has
            # no such key and keeps overriding exactly as before.
            im_trans = g["im_trans"] or []
            self._flip_y.setChecked(1 in im_trans)
            self._flip_z.setChecked(2 in im_trans)
            self._transp.setChecked(3 in im_trans)
        if g.get("tz") is not None:
            self._seed_tz.setValue(float(g["tz"]))
            self._enable_seed(tz=True)
        self._seed_note.setText(
            f"Geometry from Data Viewer: λ={g.get('wavelength_A', 0):.5f} Å, "
            f"px={g.get('pxY', 0):.2f} µm, "
            f"BC=({g.get('BC_y', 0):.2f}, {g.get('BC_z', 0):.2f}), "
            f"Lsd={g.get('Lsd', 0)/1000:.3f} mm, "
            f"tx={g.get('tx', 0):.3f}°, ty={g.get('ty', 0):.3f}°, tz={g.get('tz', 0):.3f}°.")
        self._log.append("Geometry pulled from Data Viewer tab.")

    def _on_bc_picked(self, bc_y, bc_z):
        self._enable_seed(BC=True)
        self._seed_bcy.setValue(bc_y); self._seed_bcz.setValue(bc_z)
        self._seed_note.setText("BC set from click — Lsd is auto-seeded unless it's ticked too.")
        self._log.append(f"BC set by click: ({bc_y:.2f}, {bc_z:.2f}) px — BC seed enabled")

    def _on_ring_fit_bc(self, bc_y, bc_z, r_px):
        self._enable_seed(BC=True)
        self._seed_bcy.setValue(bc_y); self._seed_bcz.setValue(bc_z)
        self._seed_note.setText(
            f"BC from ring fit (R={r_px:.1f} px). Lsd is auto-seeded unless it's ticked too.")
        self._log.append(
            f"Ring fit: BC=({bc_y:.2f}, {bc_z:.2f}) px  R={r_px:.1f} px — BC seed enabled")

    # ── Manual d-spacing ring-picking fit (non-crystalline calibrants) ──

    def _manual_d_list(self) -> list:
        """Current calibrant's d-spacings (Å), sorted descending — Ring #1 is
        the largest d-spacing, matching ``simulate_rings_from_dspacings``'s
        ``order`` numbering."""
        name = self._cal.currentText()
        if name == "Custom d-spacings…":
            d_list = parse_dspacing_text(self._dsp_custom_ed.text())
        else:
            d_list = list(MATERIALS.get(name, {}).get("d_list", []))
        return sorted(d_list, reverse=True)

    _REFINE_BOXES = ("Lsd", "BC", "tx", "ty", "tz", "Wavelength")

    def _refine_box(self, key):
        return {"Lsd": self._ref_lsd, "BC": self._ref_bc, "tx": self._ref_tx,
                "ty": self._ref_ty, "tz": self._ref_tz, "Wavelength": self._ref_wl}[key]

    def _refine_box_state(self) -> dict:
        return {k: self._refine_box(k).isChecked() for k in self._REFINE_BOXES}

    def _set_refine_box_state(self, state: dict) -> None:
        """Apply a saved checkbox set without firing a signal storm — the
        summary/pick-count refresh is done once by the caller instead."""
        for key in self._REFINE_BOXES:
            if key not in state:
                continue
            box = self._refine_box(key)
            box.blockSignals(True)
            box.setChecked(bool(state[key]))
            box.blockSignals(False)

    def _sync_refine_mode(self, is_dsp: bool) -> None:
        """Swap the Refine checkboxes between the crystalline and d-spacing
        parameter sets when the calibrant kind changes.

        Only the *transition* swaps: while the user stays on one kind of
        calibrant their own choices stick, so ticking Lsd for an AgBH fit
        survives until they switch to a crystalline calibrant and back. See
        ``_refine_state_dsp`` in ``__init__`` for why the two defaults differ.
        """
        if self._refine_mode_is_dsp == is_dsp:
            return
        if self._refine_mode_is_dsp is not None:      # remember the outgoing mode
            if self._refine_mode_is_dsp:
                self._refine_state_dsp = self._refine_box_state()
            else:
                self._refine_state_xtal = self._refine_box_state()
        elif not is_dsp:
            self._refine_state_xtal = self._refine_box_state()
        incoming = self._refine_state_dsp if is_dsp else self._refine_state_xtal
        if incoming:
            self._set_refine_box_state(incoming)
        self._refine_mode_is_dsp = is_dsp

    def _on_calibrant_changed(self, text: str):
        is_dsp = is_dspacing_calibrant(text)
        self._sync_refine_mode(is_dsp)
        self._dsp_custom_ed.setVisible(text == "Custom d-spacings…")
        self._manual_card.setVisible(is_dsp)
        # "Pick d-spacing pts" + "Ring #" only mean anything for a calibrant
        # given as a d-spacing list (AgBH, Custom d-spacings…) — a crystalline
        # calibrant is fitted from the whole image, not from tagged points. Same
        # rule the Data Viewer's geometry card applies to the same two controls.
        self._img_view.set_dspacing_picking_visible(is_dsp)
        self._refc_card.setVisible(True)
        self._sync_limits_mode(is_dsp)
        self._dist_row.setVisible(not is_dsp)
        self._build_rc.setVisible(not is_dsp)
        self._refc_card.setToolTip(
            "Distortion and residual-map refinement need a full-image forward "
            "model — not available for manual point-pick fits." if is_dsp else "")
        self._panel_grp.setVisible(not is_dsp)
        self._adv_grp.setVisible(not is_dsp)
        run_label = "Fit Geometry (manual)" if is_dsp else "Run Calibration"
        self._run_btn.setText(run_label)
        self._run_btn.setToolTip(run_label)   # button is narrower than its own text
        self._on_dspacing_picks_changed()
        self._update_refine_summary()

    def _manual_min_picks(self) -> int:
        """Minimum valid picks needed for the manual fit given which
        parameters are currently selected to refine (BC counts as 2 free
        parameters — BC_y and BC_z)."""
        flags = self._refine_flags()
        n_free = ((1 if flags["Lsd"] else 0) + (2 if flags["BC"] else 0) +
                  (1 if flags["tx"] else 0) + (1 if flags["ty"] else 0) +
                  (1 if flags["tz"] else 0) + (1 if flags["Wavelength"] else 0))
        return max(3, n_free)

    def _on_dspacing_picks_changed(self, *_args):
        if not is_dspacing_calibrant(self._cal.currentText()):
            return
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
        self._run_btn.setEnabled(bool(d_list) and (len(picks) - invalid) >= self._manual_min_picks())

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
        min_picks = self._manual_min_picks()
        if len(picks) < min_picks:
            show_error(self, "Manual fit",
                       f"Need at least {min_picks} valid picked points to refine the "
                       f"selected parameters — have {len(picks)}.")
            return
        self._orphans = [o for o in self._orphans if o.isRunning()]
        pxY = self._pxY.value()
        pxZ = self._pxZ_spin.value() if self._pxZ_check.isChecked() else pxY
        # fit_geometry_from_ring_picks's ``seed`` is a single (Lsd, BC_y, BC_z)
        # starting point, not independently settable per this helper's own
        # API — so BC and Lsd both need to be ticked to supply it; otherwise
        # the fit self-seeds from the picked points, as before. Tilts stay
        # genuinely independent (each defaults to 0° unless its own flag is on).
        seed = None
        if self._seed_en_bc.isChecked() and self._seed_en_lsd.isChecked():
            seed = (self._seed_lsd.value() * 1000.0,   # mm display → µm
                    self._seed_bcy.value(), self._seed_bcz.value())
        tilt_seed = (
            self._seed_tx.value() if self._seed_en_tx.isChecked() else 0.0,
            self._seed_ty.value() if self._seed_en_ty.isChecked() else 0.0,
            self._seed_tz.value() if self._seed_en_tz.isChecked() else 0.0)
        img = self._img_view._data
        NZ, NY = img.shape[:2] if img is not None else (0, 0)
        material_name = self._cal.currentText()

        self._calib_cancelled = False
        self._run_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._prog.setVisible(True)
        self._bot_tabs.setCurrentWidget(self._log)
        self._log.append("─" * 40 + "\nStarting manual d-spacing fit…")
        self._log.append(self._refine_summary_text())
        self._log.append(self._mask_log_line())

        refine = self._refine_flags()
        bounds, skipped_limits = self._limit_bounds()
        if bounds:
            self._log.append(
                "Limits: " + ",  ".join(
                    f"{n} ∈ [{lo * (1e-3 if n == 'Lsd' else 1):.5g}, "
                    f"{hi * (1e-3 if n == 'Lsd' else 1):.5g}]"
                    for n, (lo, hi) in sorted(bounds.items())))
        if skipped_limits:
            self._log.append(
                f"Limits on {', '.join(sorted(skipped_limits))} ignored: they are "
                f"windows around a manual seed value, but that parameter isn't "
                f"ticked to be seeded (Manual seed…), so the fit is determining "
                f"it from the picked points instead.")
        self._last_dist_coeffs = set()
        self._last_refine_flags = refine
        self._worker = ManualDspacingCalibWorker(
            picks, self._wl.value(), pxY, pxZ, seed, NY, NZ, material_name, d_list,
            parent=self, refine=refine, tilt_seed=tilt_seed, bounds=bounds)
        self._worker.log_line.connect(self._log.append)
        self._worker.finished.connect(self._on_manual_fit_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_manual_fit_done(self, result):
        self._run_btn.setEnabled(True)
        self._on_done(result)

    def _on_run_clicked(self):
        if is_dspacing_calibrant(self._cal.currentText()):
            self._run_manual_fit()
        else:
            self._run()

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

    def _refine_summary_text(self) -> str:
        """One-line 'Refining: ... Fixed: ...' summary, so it's always
        obvious which geometry parameters a run will actually vary — shown
        live in the Refine parameters card and logged at the start of every
        run (crystalline or manual)."""
        flags = self._refine_flags()
        is_dsp = is_dspacing_calibrant(self._cal.currentText())
        rows = [("Lsd", "Lsd"), ("BC", "BC"), ("tx", "tx"), ("ty", "ty"),
                ("tz", "tz"), ("Wavelength", "Wavelength")]
        if not is_dsp:
            dist_label = (f"Distortion ({len(flags['distortion_coeffs'])}/15)"
                          if flags["Distortion"] else "Distortion")
            rows.append(("Distortion", dist_label))
        refining = [label for key, label in rows if flags.get(key)]
        fixed = [label for key, label in rows if not flags.get(key)]
        bits = []
        if refining:
            bits.append("Refining: " + ", ".join(refining))
        if fixed:
            bits.append("Fixed: " + ", ".join(fixed))
        return "   ".join(bits) if bits else "Nothing selected to refine."

    def _update_refine_summary(self):
        self._refine_summary_lbl.setText(self._refine_summary_text())

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
        if mode == "frozen_point" and self._panel_grp.isChecked():
            QtWidgets.QMessageBox.warning(
                self, "Multi-panel not supported",
                "'Frozen-point (high-tilt)' does not support Multi-panel "
                "detectors yet. Uncheck 'Multi-panel' or choose a "
                "different pipeline."); return
        # Resolve scratch before anything runs. Blocking (rather than warning
        # and carrying on) matches Batch Integrate and is the right call here:
        # a fit that runs for minutes and only then finds it can't record its
        # residual map has wasted the user's time and left them with a result
        # that silently lacks the refinement they asked for.
        work_dir = self._out_ed.text().strip() or None
        stem = self._default_save_stem()
        run_id = "calib_%s_%s" % (
            time.strftime("%Y%m%d-%H%M%S"), re.sub(r"[^\w.-]", "_", stem))
        try:
            scratch = scratch_dir(work_dir, run_id)
        except OSError as e:
            QtWidgets.QMessageBox.critical(
                self, "Working directory not writable", str(e))
            return

        self._calib_cancelled = False
        self._run_btn.setEnabled(False); self._abort_btn.setEnabled(True)
        self._prog.setVisible(True)
        self._bot_tabs.setCurrentWidget(self._log)
        self._log.append("─" * 40 + f"\nStarting calibration ({mode})…")
        self._log.append(self._refine_summary_text())
        self._log.append(self._mask_log_line())
        # The windows bound the answer, so they belong in the run's own record
        # next to what was refined — not only on the card, which shows whatever
        # is set now rather than what this run used.
        if self._limits_mode_is_dsp is False:
            self._log.append("Limits: " + self._limits_note.text()
                             .replace("Always applied, centred on the seed: ", ""))
        if mode == "frozen_point":
            self._log.append(
                "ℹ Frozen-point (high-tilt) always refines Lsd/BC/ty/tz; "
                "Distortion is refined too if ticked (less-validated for "
                "this pipeline than geometry-only — check results against "
                "a separate refit). tx is never refined (not identifiable "
                "from a single image). Runs on CPU regardless of the "
                "Device setting.")

        trans = im_trans_codes_from_checkboxes(self._flip_y, self._flip_z, self._transp)

        if work_dir:
            self._log.append(f"[calibrate] scratch: {scratch}")
        else:
            self._log.append(
                f"[calibrate] No working directory set — intermediates go to "
                f"{scratch} and are deleted when the GUI exits. Set one to keep them.")

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
            # Where machine-generated intermediates go. The user-facing
            # working directory is `work_dir`; `scratch_dir` is the per-run
            # leaf inside its .midas_scratch/, which keeps two fits in one
            # working directory from overwriting each other's generically
            # named output (residual_corr.bin, calibration.json, panel shifts).
            "work_dir": work_dir,
            "scratch_dir": str(scratch),
            "save_stem": stem,
            "im_trans": trans,
            "mask": self._loader.composite_mask(),
            # None while every window sits at the backend default, which keeps
            # run_pipeline on the plain calibrate() path.
            "tols": self._crystalline_tols(),
        }
        self._last_dist_coeffs = cfg["refine"]["distortion_coeffs"]
        self._last_refine_flags = cfg["refine"]
        manual = self._manual_seed_kwargs()   # sparse: only ticked parameters
        if manual:
            cfg["manual_seed"] = manual
        if self._panel_grp.isChecked():
            cfg["panel_layout"] = {
                "n_y": self._pn_y.value(), "n_z": self._pn_z.value(),
                "sy": self._ps_y.value(), "sz": self._ps_z.value(),
                "gap_y": self._pg_y.value(), "gap_z": self._pg_z.value(),
            }

        if manual and any(k in manual for k in ("tx", "ty", "tz")):
            from midas_gui.calib import tilt_seed_effective
            if not tilt_seed_effective(mode, panel_layout=cfg.get("panel_layout"),
                                       refine=cfg["refine"]):
                tilt_bits = ", ".join(f"{k}={manual[k]:.3f}°" for k in ("tx", "ty", "tz")
                                      if k in manual)
                self._log.append(
                    f"⚠ Tilt seed ({tilt_bits}) will NOT be used: the "
                    f"'{self._pipeline.currentText()}' pipeline with these settings "
                    "ignores an initial tilt guess — tilts start from 0° instead. "
                    "Four-stage / Bayesian / Joint pipelines (or One-shot with "
                    "Multi-panel) do honour a tilt seed.")

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

    _GEOMETRY_REFINE_KEYS = ("Lsd", "BC", "tx", "ty", "tz", "Wavelength")

    #: paramstest row key → the ``fit_geometry_from_ring_picks`` sigma slot(s)
    #: it displays. "BC" is one row holding both centre coordinates.
    _SIGMA_SLOTS_FOR_KEY = {"Lsd": ("Lsd",), "BC": ("BC_y", "BC_z"),
                            "tx": ("tx",), "ty": ("ty",), "tz": ("tz",),
                            "Wavelength": ("wavelength_A",)}

    def _sigma_suffix(self, key, sigma, at_limit) -> str:
        """`` ± σ`` (or ``(at limit)``) for a geometry row, in the row's own
        displayed units — empty for rows with no uncertainty to report."""
        slots = self._SIGMA_SLOTS_FOR_KEY.get(key)
        if not slots:
            return ""
        if any(s in (at_limit or set()) for s in slots):
            return "   (at limit)"
        vals = [sigma.get(s) for s in slots]
        if any(v is None or not v for v in vals):    # unrefined, or exactly 0
            return ""
        if any(not math.isfinite(v) for v in vals):
            return "   ± ∞ (unconstrained)"
        return "   ± " + ", ".join(f"{v:.4g}" for v in vals)

    def _populate_param_grid(self, pairs, ncols=3, refine_flags=None, sigma=None,
                             at_limit=None):
        """Lay (key, value) pairs into ``ncols`` columns as plain text, filled
        column-major so each column reads top-to-bottom in file order. The paramstest
        distortion slots p0–p14 are relabelled with their coefficient names
        (iso_R2, a1, …) so the distortion reads clearly without a separate table.

        ``refine_flags``, when given (the dict ``_refine_flags()`` produces
        for the run that generated ``pairs``), marks each geometry row
        (Lsd/BC/tx/ty/tz/Wavelength) that was held fixed with a muted
        "(fixed)" label — so it's never ambiguous which values were actually
        optimized vs. carried over from the seed.

        ``sigma``/``at_limit`` (from a manual d-spacing fit) annotate each
        *refined* geometry row with its 1σ uncertainty. A converged fit is not
        the same as a determined one — on a short ring arc the optimizer will
        happily report a beam centre it could not actually measure — so the
        uncertainty belongs next to the value, not only in the log. This is
        display-only: ``pairs`` still mirrors paramstest.txt exactly."""
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
        fixed_lbl = "font-weight:600; font-size:12px; color:#888;"
        fixed_keys = ({k for k in self._GEOMETRY_REFINE_KEYS if not refine_flags.get(k)}
                      if refine_flags is not None else set())
        n = len(pairs); nrows = max(1, math.ceil(n / ncols))
        for idx, (key, val) in enumerate(pairs):
            col, row = idx // nrows, idx % nrows
            label = _PARAMSTEST_DISTORTION.get(key, key)   # name distortion slots
            if key in fixed_keys:
                label += " (fixed)"
                k = QtWidgets.QLabel(f"{label}:"); k.setStyleSheet(fixed_lbl)
            else:
                k = QtWidgets.QLabel(f"{label}:"); k.setStyleSheet(klbl)
                if sigma:
                    val += self._sigma_suffix(key, sigma, at_limit)
            v = QtWidgets.QLabel(val); v.setFont(mono)
            v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            grid.addWidget(k, row, col * 2, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(v, row, col * 2 + 1, QtCore.Qt.AlignVCenter)
        grid.setColumnStretch(ncols * 2 + 1, 1)

    def geometry_for_viewer(self) -> Optional[dict]:
        """The calibrated geometry as a Data-Viewer geometry dict (µm internal;
        the Viewer converts Lsd to its mm display), or ``None`` if this tab has
        no result yet. Shared by this tab's "→ Send to Data Viewer" and the Data
        Viewer's own "← Get" pull."""
        r = self._result
        if r is None:
            return None
        return {
            "wavelength_A": float(r.wavelength_A), "pxY": float(r.pxY),
            "pxZ": float(getattr(r, "pxZ", r.pxY) or r.pxY),
            "Lsd": float(r.Lsd), "BC_y": float(r.BC_y), "BC_z": float(r.BC_z),
            "tx": float(getattr(r, "tx", 0.0) or 0.0),
            "ty": float(getattr(r, "ty", 0.0) or 0.0),
            "tz": float(getattr(r, "tz", 0.0) or 0.0),
            "NrPixelsY": int(getattr(r, "NrPixelsY", 0) or 0),
            "NrPixelsZ": int(getattr(r, "NrPixelsZ", 0) or 0),
            "distortion": dict(getattr(r, "distortion", {}) or {}),
            "im_trans": list(getattr(r, "im_trans", []) or [])}

    def _send_to_viewer(self):
        """Push the calibrated geometry to the Data Viewer."""
        g = self.geometry_for_viewer()
        if g is None:
            return
        self.sendGeometryToViewer.emit(g)
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
        # Only the manual d-spacing fit reports uncertainties (the crystalline
        # backend returns none), so sigma is absent for a crystalline run.
        self._last_fit_sigma = getattr(result, "fit_sigma", None)
        self._last_at_limit = (getattr(result, "fit_at_limit", None)
                               or self._crystalline_at_limit(result))
        if self._last_at_limit:
            self._log.append(
                "\nWARNING: " + ", ".join(sorted(self._last_at_limit)) +
                " came back on the edge of the allowed window — that value was "
                "set by the limit, not measured from the data. Widen the limit "
                "if the true value may lie outside it, or check the seed.")
        try:
            self._populate_param_grid(
                paramstest_pairs(result, selected=self._last_dist_coeffs),
                refine_flags=self._last_refine_flags,
                sigma=self._last_fit_sigma, at_limit=self._last_at_limit)
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
    # There is one overlay and it is always the accurate one. The old
    # "Corrected" tick redrew the same rings through the fitted tilt and was
    # off by default, which put the honest overlay behind a control most users
    # never found — and it never accounted for distortion at all, so even
    # ticked it could sit visibly off the measured rings on a detector whose
    # calibration refined harmonics. Correctness is not a display preference;
    # see ``helpers.ring_xy_corrected``, which reduces exactly to a circle when
    # there is no tilt and no distortion to apply.

    def _ring_curves(self, ns, radii_px):
        """``(Y, Z)`` polylines for each predicted ring radius, projected
        through the full forward model carried by ``ns`` (a result, or the
        seed namespace) — fitted tilts *and* refined distortion.

        ``radii_px`` are ideal Bragg radii; each is turned back into the 2θ it
        came from so the projector can re-solve where that cone actually lands.
        Per-ring try/except: one unprojectable radius (a grazing 2θ against a
        steep tilt) must not cost the whole overlay.
        """
        pxY = float(ns.pxY)
        pxZ = float(getattr(ns, "pxZ", 0.0) or pxY)
        lsd = float(ns.Lsd)
        dist = dict(getattr(ns, "distortion", {}) or {})
        rho_d = distortion_rho_d_um(getattr(ns, "NrPixelsY", 0),
                                    getattr(ns, "NrPixelsZ", 0),
                                    ns.BC_y, ns.BC_z, pxY, pxZ)
        tilts = tuple(float(getattr(ns, t, 0.0) or 0.0) for t in ("tx", "ty", "tz"))
        out = []
        for r in radii_px:
            try:
                two_theta = math.degrees(math.atan(r * pxY / lsd))
                out.append(ring_xy_corrected(
                    two_theta, *tilts, lsd, ns.BC_y, ns.BC_z, pxY, pxZ,
                    distortion=dist, rho_d_um=rho_d))
            except Exception:
                import traceback
                self._log.append(f"Ring projection error:\n{traceback.format_exc()}")
        return out

    @staticmethod
    def _ring_model_note(ns) -> str:
        """What the overlay accounts for, so "these rings are bent" reads as
        the geometry rather than as a drawing bug — and so the one term that
        is *not* drawn says so instead of silently going missing."""
        bits = []
        if max(abs(float(getattr(ns, t, 0.0) or 0.0)) for t in ("tx", "ty", "tz")) > 1e-9:
            bits.append("tilt")
        if any(float(v or 0.0) for v in (getattr(ns, "distortion", None) or {}).values()):
            bits.append("distortion")
        note = f"  ({' + '.join(bits)} applied)" if bits else ""
        if (getattr(ns, "residual_corr_map", None) is not None
                or getattr(ns, "residual_corr_bin_path", None)):
            note += "  · residual map not drawn"
        return note

    def _draw_rings(self, result):
        """Overlay ``result``'s predicted rings — the only rings this tab draws.

        Whether "Use manual seed" is ticked makes no difference: the seed card
        is an input to the fit, not a thing to preview. What is on screen is
        always the geometry a calibration actually produced (or one restored
        from a project attempt), so rings that sit off the measured ones mean
        the fit, not the drawing."""
        self._calib_result = result
        # The readout switches from the seed geometry to this one (and drops
        # its "(seed)" tag) the moment it lands, with the cursor stationary.
        self._img_view._refresh_coord_bar()
        for item in self._ring_items:
            self._img_view._iv.removeItem(item)
        self._ring_items.clear()
        max_r = rmax_corner_px(result.BC_y, result.BC_z, result.NrPixelsY, result.NrPixelsZ)
        radii = [r for r in _predict_ring_radii(result) if 0 < r <= max_r]
        visible = self._show_rings_check.isChecked()
        pen = pg.mkPen("lime", width=1.2)
        curves = self._ring_curves(result, radii)
        img_shape = (result.NrPixelsZ, result.NrPixelsY)
        for ys, zs in curves:
            # Confined to the detector image, same as the Data Viewer's ring
            # overlay: a ring that swings outside the frame is not a
            # prediction the fit's own measured rings can be checked against,
            # so points off the image become NaN and connect="finite" breaks
            # the polyline there rather than drawing a chord across empty
            # canvas (see helpers.ring_on_image_mask).
            on = ring_on_image_mask(ys, zs, img_shape)
            ys_clip = np.where(on, ys, np.nan)
            zs_clip = np.where(on, zs, np.nan)
            item = pg.PlotDataItem(ys_clip, zs_clip, pen=pen, connect="finite")
            item.setVisible(visible)
            self._img_view._iv.addItem(item); self._ring_items.append(item)
        bc = pg.ScatterPlotItem([result.BC_y], [result.BC_z], symbol="o", size=10,
                                pen=pg.mkPen("yellow", width=2), brush=pg.mkBrush("red"))
        bc.setVisible(visible)
        self._img_view._iv.addItem(bc); self._ring_items.append(bc)
        self._ring_status.setText(
            f"{len(curves)} ring(s) from the fitted geometry"
            + self._ring_model_note(result))

    def _on_show_rings_toggled(self, visible):
        for item in self._ring_items:
            item.setVisible(visible)
        if not self._ring_items:
            # Nothing on screen to describe — a leftover "N ring(s) …" beside
            # an unticked Show rings reads as an overlay that failed to appear.
            self._ring_status.setText("")

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
            self, "Save calibration.json", self._default_save_path(".instr.json"),
            "JSON (*.json)")
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
        dlg = _SaveParamstestDialog(self, default_out=self._default_save_path(".instr.txt"))
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
                    # Copy the map next to the paramstest and point at the
                    # copy, for the same reason the panel shifts get a
                    # <stem>_ sidecar above: the original lives in the
                    # working directory's .midas_scratch/, which the user is
                    # explicitly told they may delete at any time. A saved
                    # instrument file that silently stops working the first
                    # time someone tidies up is worse than no map at all —
                    # midas_integrate_v2 treats an unreadable one as fatal.
                    try:
                        import shutil
                        rcm_copy = Path(out_path).with_name(
                            Path(out_path).stem + "_residual_corr.bin")
                        if Path(rcm).resolve() != rcm_copy.resolve():
                            shutil.copyfile(rcm, rcm_copy)
                        rcm = str(rcm_copy)
                    except OSError as e:
                        self._log.append(
                            f"[calibrate] Warning: couldn't copy the residual map "
                            f"beside the paramstest ({e}); pointing at the scratch "
                            f"copy instead, which won't survive deleting "
                            f"{SCRATCH_DIRNAME}/.")
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
            "dsp_custom_ed": self._dsp_custom_ed,
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
            "manual_seed_check": self._manual_seed_check,   # derived bulk on/off — see _on_seed_enable_changed
            "seed_en_bc": self._seed_en_bc,
            "seed_en_lsd": self._seed_en_lsd,
            "seed_en_tx": self._seed_en_tx,
            "seed_en_ty": self._seed_en_ty,
            "seed_en_tz": self._seed_en_tz,
            "seed_en_dist": self._seed_en_dist,
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
            "cal_r_bin": self._cal_r_bin,
            "cal_eta_bin": self._cal_eta_bin,
            "cal_azim": self._cal_azim,
            **{f"limit_{n}_on": cb for n, (cb, _s, _c) in self._limit_widgets.items()},
            **{f"limit_{n}_val": sp for n, (_cb, sp, _c) in self._limit_widgets.items()},
            **{f"limit_{n}_unit": co for n, (_cb, _s, co) in self._limit_widgets.items()},
        }

    def get_state(self, sidecar_stem: Optional[str] = None) -> dict:
        """``sidecar_stem`` (if given) is the state file's path without its
        extension. A fitted result is embedded in the returned state (under
        ``"result"``, via ``project.sanitize_result_dict``) so ``set_state()``
        can restore the rings/param grid/Save-button state without
        re-running Fit — and is also, best-effort, written out to
        ``<sidecar_stem>_calibration.json`` as an external-facing record."""
        state = {"fields": widgets_to_dict(self._state_widgets()),
                 "loader": self._loader.get_state(),
                 "img_view": self._img_view.display_state(),
                 "img_picks": self._img_view.pick_state(),
                 "cake_view": self._cake_view.display_state(),
                 "hydra": {"active_mode": self._mode_ribbon.mode(),
                           "page": self._hydra_page.get_state()},
                 # Both remembered Refine sets, not just the live checkboxes —
                 # otherwise reloading a project saved on an AgBH calibrant
                 # would lose the crystalline flags entirely (and vice versa).
                 "refine_modes": {"xtal": self._refine_state_xtal,
                                   "dsp": self._refine_state_dsp},
                 # Same reasoning for the limit rows, which are likewise
                 # remembered per calibrant kind.
                 "limit_modes": {"xtal": self._limit_state_xtal,
                                  "dsp": self._limit_state_dsp},
                 # Distortion seed *values* — "seed_en_dist" (in "fields") is
                 # just the on/off tick; this is the {coeff: value} dict it
                 # gates, which has no widget of its own. Restoring a result
                 # (below) overwrites this with the result's own distortion,
                 # so this only matters when there's no result yet.
                 "seed_dist": dict(self._seed_dist),
                 # Which of the fifteen distortion harmonics are selected.
                 # Same reasoning as "seed_dist": "ref_dist" (in "fields") is
                 # only the on/off tick, and the set it gates has no widget of
                 # its own — so without this a reopened project came back
                 # refining all 15 whatever the user had picked.
                 "dist_coeffs": sorted(self._dist_coeffs),
                 "result": project.sanitize_result_dict(self._result)}
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
        self._restoring_state = True
        try:
            self._set_state(state, sidecar_stem)
        finally:
            self._restoring_state = False

    def _set_state(self, state: dict, sidecar_stem: Optional[str] = None) -> None:
        fields = state.get("fields", {})
        apply_dict_to_widgets(self._state_widgets(), fields)
        # A restored working directory is the user's stored choice, so it is
        # never silently rewritten to the current default — but it can have
        # gone stale (project opened on a host without that mount), and
        # finding that out at Run time is worse than at open time.
        restored_wd = self._out_ed.text().strip()
        if restored_wd:
            reason = check_output_dir_writable(restored_wd)
            if reason:
                self._log.append(
                    f"[calibrate] Warning: restored working directory — {reason}")
        if "seed_en_bc" not in fields and self._manual_seed_check.isChecked():
            # A project saved before the per-parameter seed panel existed:
            # "Use manual seed" meant BC+Lsd+tilts all together, so reproduce
            # that exactly rather than silently seeding nothing.
            self._enable_seed(BC=True, Lsd=True, tx=True, ty=True, tz=True)
        else:
            self._on_seed_enable_changed()   # resync the derived master tri-state
        self._update_limits_label()
        dist_coeffs = state.get("dist_coeffs")
        if dist_coeffs is not None:            # absent in pre-2026-09 projects
            self._dist_coeffs = set(dist_coeffs)
        # Unconditional, even when the key is absent: apply_dict_to_widgets
        # restores "ref_dist" with signals blocked, so the checkbox's own
        # "Distortion (n/15)" caption — which only _update_dist_label writes —
        # would otherwise keep whatever the freshly built tab defaulted to and
        # contradict the tick right next to it.
        self._update_dist_label()
        modes = state.get("refine_modes") or {}
        if isinstance(modes.get("xtal"), dict):
            self._refine_state_xtal = dict(modes["xtal"])
        if isinstance(modes.get("dsp"), dict):
            self._refine_state_dsp = dict(modes["dsp"])
        # The live checkboxes were just restored from "fields" and already
        # belong to the saved calibrant, so pin the mode first — otherwise
        # _on_calibrant_changed would treat this as a transition, file them
        # under the wrong mode, and overwrite them with the other one's set.
        limits = state.get("limit_modes") or {}
        if isinstance(limits.get("xtal"), dict):
            self._limit_state_xtal = dict(limits["xtal"])
        if isinstance(limits.get("dsp"), dict):
            self._limit_state_dsp = dict(limits["dsp"])
        self._refine_mode_is_dsp = is_dspacing_calibrant(self._cal.currentText())
        # The limits column cannot be pinned the same way. _sync_limits_mode
        # *shapes* the card as well as filling it — which rows exist, the
        # header wording, whether the tilt window spans ty+tz — and pinning
        # the mode made it early-return, so a project saved on AgBH opened
        # wearing the crystalline card it was built with: no tx/tz/BC_z
        # windows to edit, a Distortion row the manual fit ignores, and a
        # footer quoting backend tolerances that fit never sees.
        #
        # Instead park the just-restored live rows in this calibrant kind's
        # bucket and leave the mode unset, so the sync below runs for its
        # layout pass and then re-applies exactly those rows. The other
        # kind's bucket keeps the stash restored from "limit_modes" above.
        if self._refine_mode_is_dsp:
            self._limit_state_dsp = self._limits
        else:
            self._limit_state_xtal = self._limits
        self._limits_mode_is_dsp = None
        self._on_calibrant_changed(self._cal.currentText())
        self._sync_seed_steps()
        self._loader.set_state(state.get("loader") or {})
        self._img_view.set_display_state(state.get("img_view"))
        self._origin_btn.sync()
        self._img_view.set_pick_state(state.get("img_picks"))
        self._cake_view.set_display_state(state.get("cake_view"))
        hydra_state = state.get("hydra") or {}
        self._mode_ribbon.set_mode(hydra_state.get("active_mode", "single"))
        self._hydra_page.set_state(hydra_state.get("page") or {})
        # Restored first so it's in place even with no "result" below (a
        # project saved before a first fit, with only a manually-typed seed);
        # _display_stored_result overwrites it from the actual result when
        # one exists, which is the more authoritative source.
        seed_dist = state.get("seed_dist")
        if seed_dist:
            self._seed_dist = dict(seed_dist)
            self._update_seed_dist_label()
        result_state = state.get("result")
        if result_state:
            self._display_stored_result(project.calibration_namespace(result_state),
                                         reintegrate_if_missing=False)

    # ── File > Open Project… ─────────────────────────────────────────

    def _display_stored_result(self, result, results_arrays: Optional[dict] = None,
                                reintegrate_if_missing: bool = True) -> None:
        """Redraw rings + the radial profile/cake for a result recovered
        from a project attempt — same visual effects as a live Fit's
        ``_on_done``, without re-running Fit. When ``results_arrays`` (the
        attempt's embedded cake/profile, see
        ``project.read_calib_attempt_results``) is available, the plots are
        populated directly from it — no recompute needed. Otherwise, falls
        back to live re-integration if an image happens to be loaded, unless
        ``reintegrate_if_missing`` is False (``set_state()`` passes False —
        a generic project/session reload must not silently kick off a
        long-running background integration; see ``_apply_workspace_state``'s
        "long-running pipelines are not re-run" contract). Best-effort per
        step so a partially-available result (e.g. the source image no
        longer on disk) still shows whatever it can."""
        self._result = result
        # The seed spin boxes come back from "fields", but the distortion
        # *values* dict has no widget of its own to be restored into — so
        # without this, re-running the fit from a reloaded project would
        # seed it with no coefficients, quietly discarding the harmonics
        # the stored result was refined with. ``_enable_seed`` never
        # unticks, so this only ever turns Distortion seeding on to match,
        # never overrides an explicit off left over from "fields".
        self._seed_dist = dict(getattr(result, "distortion", {}) or {})
        if self._seed_dist:
            self._enable_seed(Distortion=True)
        self._update_seed_dist_label()
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
        elif reintegrate_if_missing and self._image is not None:
            self._bot_tabs.setCurrentWidget(self._prof_view)
            try:
                self._run_integration(result)
            except Exception:
                pass

    def apply_project_calibration(self, attempts: dict, *,
                                  restore_fields: bool = True) -> None:
        """``attempts`` maps panel key (``"single"`` or ``"ge1"``..``"ge4"``)
        to that panel's calibration-attempt metadata (``project.read_attempt``)
        — called after File > Open Project… when the user opts to populate
        this tab. Reuses ``set_state()``'s existing field-restore machinery
        (widget keys are shared across the single-detector tab, the Hydra
        page's shared recipe, and a Hydra panel card's seed fields — see
        ``project.calib_attempt_gui_fields``), and switches the mode ribbon
        to match what was found.

        ``restore_fields=False`` restores only the fitted results, leaving
        every input widget alone. File ▸ Open Project… passes it when the
        Calibrate tab's GUI Workspace was restored in the same action: the
        workspace holds all the same fields and is strictly newer (saved at
        Ctrl+S, whereas the attempt was recorded when the fit ran), so
        replaying the attempt over it only reverts the user's later edits —
        the Distortion tick being the case that surfaced it. What the
        attempt still has that the workspace doesn't is the embedded
        cake/profile arrays and the materialized panel shifts, which is why
        the result half runs either way.
        """
        if not attempts:
            return
        single_meta = attempts.get("single")
        hydra_metas = {k: v for k, v in attempts.items() if k != "single"}
        state = {}
        if single_meta is not None:
            state["fields"] = project.calib_attempt_gui_fields(single_meta)
            state["loader"] = project.calib_attempt_loader_state(single_meta)
            # The tick alone says only *that* distortion was refined; without
            # the set it gates the replay silently widened a 3-coefficient fit
            # back out to all 15.
            coeffs = project.calib_attempt_dist_coeffs(single_meta)
            if coeffs is not None:
                state["dist_coeffs"] = coeffs
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
        if restore_fields:
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
