"""Tab 0 — Data Viewer.

Plot single/stacked TIFF, HDF5 (2-D or 3-D), or a folder/glob of frames, with a
frame navigator for stacks.  Define one or more materials (each a dropdown-preset
or custom lattice + space group, independently toggled and colored) alongside a
shared wavelength + Lsd + pixel size, and overlay the simulated Debye-Scherrer
ring positions from every enabled material on the image and radial profile.

This tab is purely for inspection — it produces no shared state for other tabs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from midas_gui.constants import DEFAULT_NICKEL_H5
from midas_gui.helpers import (_fspin, _NoScrollSpinBox, _NoScrollComboBox,
                         widgets_to_dict, apply_dict_to_widgets, _apply_im_trans,
                         pixel_readout_text,
                         load_profile_file, profile_file_axis_kind,
                         native_axis_to_r_px, PROFILE_FILE_FILTER)
from midas_gui.widgets import (ProfileViewer, DataLoaderPanel, CakeViewer,
                              OriginToolButton, build_lab_frame_axes_items)
from midas_gui.dialogs import show_error
from midas_gui.roi_tools import ROIImageViewer, ROIRibbon
from midas_gui.hydra_widgets import HydraModeRibbon
from midas_gui.hydra_geometry_card import DetectorGeometryCard, CAKE_ETA_BIN_DEG
from midas_gui.hydra_page import HydraViewerPage
from midas_gui.workers import ProjectionWorker, AllFrameStatsWorker
from midas_gui import style as S

# Bounds for the intensity-mask pixel-< / pixel-> spin boxes (integer counts).
# Upper bound covers common 32-bit-detector overflow/dead-pixel sentinels
# (e.g. 2**32-1 = 4294967295), which a plain 32-bit QSpinBox cannot hold.
_IMASK_MIN = -1_000_000_000
_IMASK_MAX = 5_000_000_000


class DataViewerTab(QtWidgets.QWidget):
    pushGeometry = QtCore.pyqtSignal(dict)   # λ/px/Lsd/BC → Calibrate tab
    pullGeometry = QtCore.pyqtSignal()       # "← Get": pull Calibrate's geometry

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cur: Optional[np.ndarray] = None     # current 2-D frame (corrected, for display)
        self._disp_shape = None                    # last displayed shape (fresh-load detection)
        self._is_projection = False                # showing a projected stack?
        self._proj_raw: Optional[np.ndarray] = None  # projection output before ImTransOpt
        self._proj_worker = None
        self._stats_worker = None                  # AllFrameStatsWorker for "All frames" scope
        self._stats_all_frames_dirty = False        # a request arrived while one was running
        self._topn_items: list = []                # scatter overlays for Top-N pixels
        self._axis_items: list = []                # lab-frame axes overlay items
        # Radial Profile tab showing a loaded profile file instead of the live
        # detector frame — see _on_profile_source_changed/_render_loaded_profile.
        self._profile_file_mode = False
        self._loaded_profile = None                 # (x, y, sigma_or_None, x_kind) or None
        # Throttle rapid dataChanged bursts (slider drag, fast live streaming)
        # into one heavy refresh per interval. A single-shot QTimer is used as
        # a throttle, not a trailing-edge debounce: _on_data_changed() below
        # only (re)starts it while idle, so once armed it fires on schedule
        # instead of being restarted indefinitely and starved by a continuous
        # stream of events (e.g. live frames arriving faster than every 60ms).
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True); self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._on_loader_data)
        # Debounce the intensity-mask lo/hi spinboxes: a bounded human gesture
        # (typing a value, holding the spin arrows), so restarting on every
        # valueChanged and only recomputing once input settles is correct
        # here (unlike _refresh_timer above, which must guarantee periodic
        # firing against a continuous external stream).
        self._imask_debounce_timer = QtCore.QTimer(self)
        self._imask_debounce_timer.setSingleShot(True)
        self._imask_debounce_timer.setInterval(120)
        self._imask_debounce_timer.timeout.connect(self._on_imask_changed)
        self._build_ui()
        if self._loader.stats_panel is not None:
            self._loader.stats_panel.scopeChanged.connect(self._update_stats)
        self._loader.set_path(DEFAULT_NICKEL_H5, dataset="exchange/data")

    def set_mask_from_tab1(self, mask):
        self._loader.set_tab1_mask(mask)

    def _on_mode_changed(self, mode: str):
        """Leftmost ribbon switched between "single" and "hydra" — swap the
        visible page. The two modes are independent views; no data/geometry
        is shared between them."""
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

    def refresh_devices(self) -> None:
        """Repopulate the Live Data card's device dropdown from the
        just-activated profile (see MainWindow.on_profile_changed)."""
        self._loader.refresh_devices()

    def start_live_pv(self, pv: str) -> bool:
        """Programmatic equivalent of picking `pv` in the Live Data card and
        clicking Start — used by the MIDAS-bridge QLocalServer (app.py)."""
        return self._loader.start_live_pv(pv)

    def shutdown(self):
        """Called by MainWindow on app close to stop any live PV stream
        (and the Sim Detector server, if it was ever started)."""
        self._loader.stop_live()
        from midas_gui import sim_detector
        sim_detector.stop_all()

    def get_geometry(self) -> dict:
        """Current manual geometry — λ (Å), pixel (µm), Lsd (µm), beam centre (px).

        Lsd is entered in mm (display) but always returned/used in µm.
        Delegates to the extracted ``DetectorGeometryCard``."""
        return self._geom_card.get_geometry()

    def _im_trans_codes(self) -> list:
        """Ordered MIDAS ImTransOpt codes from the Transforms checkboxes."""
        return self._geom_card.im_trans_codes()

    def _radial_readout(self, col, row) -> str:
        """2θ / Q / d / η under the cursor — see
        ``widgets.ImageViewer.set_radial_readout_fn``.

        The displayed frame is already in the geometry's own orientation
        (every path here goes through ``set_raw_frame`` with these same
        ``im_trans`` codes), so the hovered pixel needs no remapping. The
        card carries no ``tx``; ``pixel_readout_text`` defaults it to 0.
        """
        return pixel_readout_text(col, row, self.get_geometry())

    def _on_im_trans_changed(self):
        """Transform checkbox toggled — re-apply to the current frame + refresh."""
        if self._is_projection and getattr(self, "_proj_raw", None) is not None:
            self._cur = self._viewer.set_raw_frame(
                self._proj_raw, self._im_trans_codes(), autorange=False)
            self._redraw_lab_axes_if_on()
            self._geom_card.maybe_auto_radial()
        elif self._loader.current_frame() is not None:
            self._on_fields_changed()

    def set_geometry(self, g: dict):
        """Replace the manual-geometry fields from a geometry dict (e.g. the Calibrate
        tab's result). Values are µm/Å/px; the Lsd field displays mm.
        Delegates to the extracted ``DetectorGeometryCard``."""
        self._geom_card.set_geometry(g)

    def get_hydra_export(self) -> dict:
        """Hydra page's anchor path + each present panel's geometry — for the
        Calibrate tab's Hydra "← Data Viewer" import."""
        return self._hydra_page.export_for_calibration()

    def set_hydra_panel_geometry(self, n: int, g: dict):
        """Push a calibrated geometry into one Hydra panel's card (the
        Calibrate tab's Hydra "→ Send to Data Viewer")."""
        card = self._hydra_page._cards.get(f"ge{n}")
        if card is not None:
            card.set_geometry(g)

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def _state_widgets(self) -> dict:
        widgets = {
            "proj_skip": self._proj_skip,
            "proj_nframes": self._proj_nframes,
            "imask_on": self._imask_on,
            "imask_lo": self._imask_lo,
            "imask_hi": self._imask_hi,
            "topn_spin": self._topn_spin,
            "rad_r_bin": self._rad_r_bin,
            "rad_auto": self._rad_auto,
            "rad_accurate": self._rad_accurate,
            "cake_r_bin": self._cake_r_bin,
            "cake_eta_bin": self._cake_eta_bin,
            "lab_axes_on": self._lab_axes_on,
        }
        widgets.update(self._geom_card.state_widgets())
        return widgets

    def get_state(self) -> dict:
        fields = widgets_to_dict(self._state_widgets())
        fields["materials"] = self._geom_card.materials_state()
        return {"fields": fields, "loader": self._loader.get_state(),
                "viewer": self._viewer.display_state(),
                "cake_view": self._cake_view.display_state(),
                "hydra": {"active_mode": self._mode_ribbon.mode(),
                          "page": self._hydra_page.get_state()}}

    def set_state(self, state: dict):
        """Restores the loader (which re-triggers its own data reload) and any
        calibration file (re-triggering the card's own calibration load) *before*
        applying the saved field values, so an explicitly saved value always wins
        over whatever a re-triggered load computed as a default."""
        hydra_state = state.get("hydra") or {}
        self._mode_ribbon.set_mode(hydra_state.get("active_mode", "single"))
        self._hydra_page.set_state(hydra_state.get("page") or {})
        self._loader.set_state(state.get("loader") or {})
        self._viewer.set_display_state(state.get("viewer"))
        self._origin_btn.sync()   # display_state carries "origin"; the button can't see it change
        self._cake_view.set_display_state(state.get("cake_view"))
        fields = state.get("fields", {})
        calib_path = fields.get("calib_ed")
        if calib_path:
            self._geom_card.set_calib_path(calib_path)
        materials = fields.get("materials")
        if materials:
            self._geom_card.set_materials(materials)
        apply_dict_to_widgets(self._state_widgets(), fields)
        self._geom_card.refresh_geometry()
        self._redraw_lab_axes_if_on()

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(0)

        # Leftmost mode ribbon: "Single detector" (this tab's existing view,
        # page 0 below) vs. "Hydra" (4-panel GE detector view, page 1).
        self._mode_ribbon = HydraModeRibbon()
        self._mode_ribbon.modeChanged.connect(self._on_mode_changed)
        root.addWidget(self._mode_ribbon)

        self._mode_stack = QtWidgets.QStackedWidget()
        root.addWidget(self._mode_stack, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False); split.setHandleWidth(6)
        self._mode_stack.addWidget(split); self._hsplit = split

        # ── LEFT: data loader (stack mode) ──
        # hide_frame_field=True: this tab shows its own scrubber under the
        # image viewer instead (see _build_frame_scrub_bar), mirroring
        # tab_calibrate.py. folder_format_filter=True: offer a "Format:"
        # filter when a loaded folder mixes file types.
        self._loader = DataLoaderPanel(mode="stack", allow_live=True,
                                       hide_frame_field=True, folder_format_filter=True)
        self._loader.setMinimumWidth(200)
        self._loader.dataChanged.connect(self._on_data_changed)
        self._loader.fieldsChanged.connect(self._on_fields_changed)
        split.addWidget(self._loader)

        # ── MIDDLE: parameters ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setMinimumWidth(260)
        inner = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(inner)
        lv.setContentsMargins(2, 2, 2, 2); lv.setSpacing(8)
        scroll.setWidget(inner)

        def _br(w=30):
            b = QtWidgets.QPushButton("…"); b.setFixedWidth(w); return b

        def _frow(ed, slot):
            r = QtWidgets.QHBoxLayout(); r.setSpacing(4)
            r.addWidget(ed); b = _br(); b.clicked.connect(slot); r.addWidget(b); return r

        self._info_lbl = QtWidgets.QLabel("")
        self._info_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px"); self._info_lbl.setWordWrap(True)

        # ── Projection card ──
        self._proj_grp = S.make_card("Projection")
        m_row = QtWidgets.QHBoxLayout(); m_row.setSpacing(8)
        m_row.addWidget(S.LabelRight("Method:"))
        self._proj_method = {}
        for meth in ("max", "sum", "mean"):
            rb = QtWidgets.QRadioButton(meth.capitalize()); m_row.addWidget(rb)
            self._proj_method[meth] = rb
        self._proj_method["max"].setChecked(True)
        m_row.addStretch(1)
        self._proj_grp.body.addLayout(m_row)
        self._proj_skip = _NoScrollSpinBox(); self._proj_skip.setRange(0, 1000000); self._proj_skip.setValue(1)
        self._proj_skip.setFixedWidth(72)
        self._proj_skip.setToolTip(
            "Ignore this many leading frames before projecting\n"
            "(1 = skip the first frame, 4 = skip the first four).")
        self._proj_nframes = _NoScrollSpinBox(); self._proj_nframes.setRange(0, 1000000)
        self._proj_nframes.setValue(0)
        self._proj_nframes.setFixedWidth(72)
        self._proj_nframes.setToolTip(
            "Number of frames to project after Skip frames\n"
            "(0 = use all remaining frames).")
        ax = S.Form()
        ax.row(("Skip frames:", self._proj_skip), ("N frames:", self._proj_nframes))
        self._proj_grp.body.addLayout(ax)
        self._proj_btn = QtWidgets.QPushButton("Project stack")
        self._proj_btn.clicked.connect(self._project)
        self._frame_btn = QtWidgets.QPushButton("Back to frames")
        self._frame_btn.clicked.connect(self._on_loader_data)
        self._proj_grp.body.addLayout(S.button_grid([self._proj_btn, self._frame_btn], 2))
        self._proj_grp.body.addWidget(self._info_lbl)
        self._proj_grp.setEnabled(False)
        self._apply_project_style(False)
        lv.addWidget(self._proj_grp)

        # ── Intensity range mask (compact — lives in the radial-plot toolbar,
        # see below where self._profile_view is built) ──
        self._imask_on = QtWidgets.QCheckBox("Exclude range")
        self._imask_on.setToolTip(
            "Pixels < min or > max are masked: drawn as a red overlay on the image\n"
            "and excluded from the radial integration (removes gaps / hot / overflow).")
        # Plain QSpinBox is 32-bit-int limited (overflows on detector sentinel
        # values like 2**32-1 for dead/overflow pixels), so these stay double
        # spin boxes with decimals=0 — displays/steps as a whole number with no
        # decimal point, but the range comfortably covers such sentinels.
        self._imask_lo = _fspin(_IMASK_MIN, _IMASK_MAX, 0, 0.0)
        self._imask_lo.setToolTip("Pixels < this value are masked (dead / gap / beam-stop).")
        self._imask_lo.setFixedWidth(112)
        self._imask_hi = _fspin(0, _IMASK_MAX, 0, 1_048_575)
        self._imask_hi.setToolTip("Pixels > this value are masked (hot / overflow).")
        self._imask_hi.setFixedWidth(112)
        for w in (self._imask_lo, self._imask_hi):
            w.setEnabled(False)
        self._imask_on.toggled.connect(
            lambda c: (self._imask_lo.setEnabled(c), self._imask_hi.setEnabled(c)))
        self._imask_on.toggled.connect(self._on_imask_changed)
        self._imask_lo.valueChanged.connect(self._on_imask_value_changed)
        self._imask_hi.valueChanged.connect(self._on_imask_value_changed)

        # ── Ring simulation + calibration load/save (extracted, reusable) ──
        self._geom_card = DetectorGeometryCard()
        self._geom_card.pushGeometry.connect(self.pushGeometry.emit)
        self._geom_card.pullGeometry.connect(self.pullGeometry.emit)
        self._geom_card.imTransChanged.connect(self._on_im_trans_changed)
        self._geom_card.set_image_source(lambda: self._cur, self._combined_bad_mask)
        self._loader.metadataDetected.connect(self._geom_card.apply_shared_fields)
        lv.addWidget(self._geom_card)
        lv.addStretch(1)
        split.addWidget(scroll)

        # Right: image (top) + radial-integration plot (bottom) in a splitter.
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.setHandleWidth(8)
        self._viewer = ROIImageViewer()
        self._viewer.set_radial_readout_fn(self._radial_readout)
        # The geometry card owns BC/Lsd/tilts; the cursor does not move when
        # one of them is edited, so the readout has to be told.
        self._geom_card.geometryChanged.connect(
            lambda: self._viewer._refresh_coord_bar())
        self._geom_card.set_viewer(self._viewer)
        vtb = self._viewer._toolbar_layout
        # Display-origin selector — bottom-left (MIDAS convention, the default)
        # or top-left (what most generic image viewers show).
        self._origin_btn = OriginToolButton(self._viewer)
        vtb.addWidget(self._origin_btn)
        origin_sep = QtWidgets.QFrame()
        origin_sep.setFrameShape(QtWidgets.QFrame.VLine)
        origin_sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        vtb.addWidget(origin_sep)
        # Top-N brightest-pixel locator (toggle on the image toolbar).
        self._topn_btn = QtWidgets.QPushButton("Top-N pixels"); self._topn_btn.setCheckable(True)
        self._topn_btn.setToolTip(
            "Mark the N highest-intensity pixels on the image (crosshair + circle) "
            "and show their statistics. Click again to turn off.")
        self._topn_spin = _NoScrollSpinBox(); self._topn_spin.setRange(1, 100000)
        self._topn_spin.setValue(20); self._topn_spin.setFixedWidth(70)
        self._topn_spin.setToolTip("Number of highest-intensity pixels to mark.")
        self._topn_thresh_on = QtWidgets.QCheckBox("I >")
        self._topn_thresh_on.setToolTip(
            "Only consider pixels above this intensity for Top-N — pixels at or "
            "below it are excluded from ranking and never marked.")
        self._topn_thresh = _fspin(-1e9, 1e9, 1, 0.0)
        self._topn_thresh.setFixedWidth(84)
        self._topn_thresh.setEnabled(False)
        vtb.addWidget(self._topn_btn)
        vtb.addWidget(QtWidgets.QLabel("N:"))
        vtb.addWidget(self._topn_spin)
        vtb.addWidget(self._topn_thresh_on)
        vtb.addWidget(self._topn_thresh)
        self._topn_btn.toggled.connect(self._on_topn_toggled)
        self._topn_spin.valueChanged.connect(
            lambda *_: self._show_topn() if self._topn_btn.isChecked() else None)
        self._topn_thresh_on.toggled.connect(self._topn_thresh.setEnabled)
        self._topn_thresh_on.toggled.connect(
            lambda *_: self._show_topn() if self._topn_btn.isChecked() else None)
        self._topn_thresh.valueChanged.connect(
            lambda *_: self._show_topn() if self._topn_btn.isChecked() else None)
        # Lab-frame axes overlay: APS/MIDAS lab-frame X/Y compass + beam-direction
        # glyph + eta-sweep arc, anchored at the beam centre — lets the user verify
        # orientation/ImTransOpt by checking that a feature lands in the quadrant
        # the overlay predicts.
        axes_sep = QtWidgets.QFrame()
        axes_sep.setFrameShape(QtWidgets.QFrame.VLine)
        axes_sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        vtb.addWidget(axes_sep)
        self._lab_axes_on = QtWidgets.QCheckBox("Lab-frame axes")
        self._lab_axes_on.setToolTip(
            "Overlay MIDAS lab-frame axes (X_Lab/Y_Lab), the beam-direction ⊗ "
            "glyph, and an η sweep arc, anchored at the beam centre.")
        self._lab_axes_on.toggled.connect(self._on_lab_axes_toggled)
        vtb.addWidget(self._lab_axes_on)
        self._geom_card.geometryChanged.connect(self._redraw_lab_axes_if_on)
        self._geom_card.geometryChanged.connect(self._on_profile_geometry_changed)
        self._viewer.originChanged.connect(self._redraw_lab_axes_if_on)
        # ROI popups are always-on-top (roi_tools.ROIStatsPopup) so they don't
        # get buried behind the main window; minimizing one tucks it into this
        # ribbon on the viewer's left edge instead of just closing it.
        self._roi_ribbon = ROIRibbon()
        self._viewer.set_ribbon(self._roi_ribbon)
        viewer_container = QtWidgets.QWidget()
        vc_layout = QtWidgets.QHBoxLayout(viewer_container)
        vc_layout.setContentsMargins(0, 0, 0, 0); vc_layout.setSpacing(0)
        vc_layout.addWidget(self._roi_ribbon)
        vc_layout.addWidget(self._viewer, 1)
        img_container = QtWidgets.QWidget()
        icl = QtWidgets.QVBoxLayout(img_container)
        icl.setContentsMargins(0, 0, 0, 0); icl.setSpacing(0)
        icl.addWidget(viewer_container, 1)
        icl.addWidget(self._build_frame_scrub_bar())
        right.addWidget(img_container)

        # Radial integration (azimuthal mean around the beam centre).
        self._profile_view = ProfileViewer()
        self._profile_view.radiusClicked.connect(self._on_radius_clicked)
        self._geom_card.set_profile_view(self._profile_view)
        self._cake_view = CakeViewer()
        self._geom_card.set_cake_view(self._cake_view)
        # The cake otherwise only appears as a by-product of a full-geometry
        # radial integration — this computes it on demand, for the current
        # frame, whether or not a calibration is loaded.
        self._cake_btn = QtWidgets.QPushButton("Calculate")
        self._cake_btn.setToolTip(
            "Compute the (η, R) cake for the current frame, using the loaded "
            "calibration / dialled-in tilt if there is one and a plain polar "
            "binning about the beam centre otherwise. Uses the R bin size set "
            "on the Radial Profile tab.")
        self._cake_btn.clicked.connect(self._geom_card.cake_integrate)
        # Cake binning is independent of the profile's: it is an on-demand
        # Calculate, not a live-view refresh, so it can afford finer bins.
        self._cake_r_bin = _fspin(0.1, 50.0, 2, 1.0, "px"); self._cake_r_bin.setFixedWidth(64)
        self._cake_r_bin.setToolTip("Radial (R) bin size for the cake, in detector pixels.")
        self._cake_eta_bin = _fspin(0.05, 45.0, 2, CAKE_ETA_BIN_DEG, "°")
        self._cake_eta_bin.setFixedWidth(64)
        self._cake_eta_bin.setToolTip(
            "Azimuthal (η) bin size for the cake, in degrees. Smaller bins give "
            "finer η resolution at the cost of counts per bin and compute time.")
        self._geom_card.set_cake_controls(self._cake_r_bin, self._cake_eta_bin)
        ctb = self._cake_view._toolbar_layout
        ctb.insertWidget(0, self._cake_btn)
        ctb.insertWidget(1, QtWidgets.QLabel("  R bin:"))
        ctb.insertWidget(2, self._cake_r_bin)
        ctb.insertWidget(3, QtWidgets.QLabel("η bin:"))
        ctb.insertWidget(4, self._cake_eta_bin)
        ptb = self._profile_view._toolbar_layout
        # Source: the live detector frame's own radial integration (default), or
        # a pre-integrated profile file (csv/xye/dat/fxye — Batch Integrate's own
        # output formats) loaded and plotted directly, with the image viewer left
        # empty — a calibration can still be attached for ring simulation on top.
        self._prof_source = _NoScrollComboBox()
        self._prof_source.addItem("Detector frame", "image")
        self._prof_source.addItem("Profile file…", "file")
        self._prof_source.setToolTip(
            "Detector frame: integrate the loaded image (default).\n"
            "Profile file…: load an existing radial-integration output file "
            "(.csv/.xye/.dat/.fxye) and plot it directly — no image needed.")
        self._prof_source.currentIndexChanged.connect(self._on_profile_source_changed)
        self._prof_source_lbl = QtWidgets.QLabel("")
        self._prof_source_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        prof_src_sep = QtWidgets.QFrame()
        prof_src_sep.setFrameShape(QtWidgets.QFrame.VLine)
        prof_src_sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        ptb.insertWidget(0, self._prof_source)
        ptb.insertWidget(1, self._prof_source_lbl)
        ptb.insertWidget(2, prof_src_sep)
        self._rad_r_bin = _fspin(0.1, 20.0, 2, 1.0, "px"); self._rad_r_bin.setFixedWidth(56)
        self._rad_r_bin.setToolTip("Radial bin size for the azimuthal mean.")
        self._rad_auto = QtWidgets.QCheckBox("Auto"); self._rad_auto.setChecked(True)
        self._rad_auto.setToolTip("Recompute the radial integration when the beam "
                                  "centre or frame changes.")
        self._rad_accurate = QtWidgets.QCheckBox("Accurate")
        self._rad_accurate.setToolTip(
            "Integrate through the full Batch-Integrate pipeline (MIDAS engine, "
            "subpixel K=2), so detector tilts, pixel size and distortion are all "
            "properly accounted for.\n"
            "Off (the default) uses the fast approximate profile, which ignores "
            "some of those but is quick enough to keep up with the live view.")
        self._geom_card.set_radial_controls(self._rad_r_bin, self._rad_auto,
                                            self._rad_accurate)
        self._rad_btn = QtWidgets.QPushButton("Integrate")
        self._rad_btn.clicked.connect(self._geom_card.radial_integrate)
        ptb.insertWidget(3, self._rad_btn)
        ptb.insertWidget(3, self._rad_auto)
        ptb.insertWidget(3, self._rad_accurate)
        ptb.insertWidget(3, self._rad_r_bin)
        ptb.insertWidget(3, QtWidgets.QLabel("  R bin:"))
        self._radial_help_btn = QtWidgets.QToolButton()
        self._radial_help_btn.setText("?")
        self._radial_help_btn.setFixedSize(18, 18)
        self._radial_help_btn.setToolTip("How is this profile calculated?")
        self._radial_help_btn.setStyleSheet(
            "QToolButton { border-radius: 9px; border: 1px solid #777; "
            "background: #333; color: #ddd; font-weight: bold; }"
            "QToolButton:hover { background: #444; }")
        self._radial_help_btn.clicked.connect(self._geom_card.show_radial_help)
        ptb.insertWidget(ptb.indexOf(self._rad_btn) + 1, self._radial_help_btn)
        # Exclude-out-of-range-pixels controls (moved here from the left panel
        # to share this row; the bins/max stats printout is hidden to make room).
        # Appended after everything else (including the row's stretch) so the
        # group sits pinned to the far right of the toolbar.
        self._profile_view._stat.hide()
        ptb.addWidget(self._imask_on)
        ptb.addWidget(QtWidgets.QLabel(" <"))
        ptb.addWidget(self._imask_lo)
        ptb.addWidget(QtWidgets.QLabel(">"))
        ptb.addWidget(self._imask_hi)
        # Bottom tab strip. The intensity-statistics panel is built by the
        # loader (it reports on the loader's frames) but lives here, alongside
        # the profile and cake, instead of squeezing the left card column — and
        # is the tab shown on open, since it is the first thing to look at on a
        # freshly loaded frame.
        bot = QtWidgets.QTabWidget()
        self._bottom_tabs = bot
        if self._loader.stats_panel is not None:
            bot.addTab(self._loader.stats_panel, "Statistics")
        bot.addTab(self._profile_view, "Radial Profile")
        bot.addTab(self._cake_view, "Eta vs R Cake")
        bot.setCurrentIndex(0)
        right.addWidget(bot)
        right.setStretchFactor(0, 3); right.setStretchFactor(1, 1)
        right.setMinimumWidth(320)
        split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 0); split.setStretchFactor(2, 1)
        split.setSizes([286, 361, 950])

        # Page 1: Hydra (4-panel GE detector) view.
        self._hydra_page = HydraViewerPage()
        self._mode_stack.addWidget(self._hydra_page)

    # ── Loading ───────────────────────────────────────────────────

    def _build_frame_scrub_bar(self) -> QtWidgets.QWidget:
        """Prev/slider/next scrubber shown under the image viewer once the
        loaded source (HDF5 dataset / TIFF folder / stack) has more than one
        frame — replaces the loader panel's own compact nav row, which this
        tab keeps hidden (``DataLoaderPanel(hide_frame_field=True)``),
        mirroring ``tab_calibrate.py``'s identical scrubber (itself modeled
        on ``widgets.CakeStackViewer``'s scrub bar in Batch Integrate's
        "Eta-R cakes" tab)."""
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

    def _on_data_changed(self):
        """Loader's dataChanged fired — throttle into at most one refresh per
        interval instead of restarting (and so indefinitely deferring) the
        timer on every event, which would starve the display during a fast
        continuous burst (e.g. live streaming faster than the timer interval)."""
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _on_loader_data(self):
        """Data loaded or frame changed in the loader — refresh the display,
        applying dark/bright/background corrections."""
        if self._profile_file_mode:
            self._frame_scrub_bar.setVisible(False)
            return
        self._sync_frame_scrub_bar()
        raw = self._loader.current_frame()
        if raw is None:
            return
        self._proj_grp.setEnabled(self._loader.n_frames() > 1)
        self._is_projection = False
        self._proj_raw = None
        self._apply_project_style(False)
        if self._loader.stats_panel is not None:
            self._loader.stats_panel.set_scope_enabled(True)
        fresh = (self._disp_shape != raw.shape)
        self._disp_shape = raw.shape
        self._cur = self._viewer.set_raw_frame(
            self._loader.corrected(raw), self._im_trans_codes(),
            autorange=fresh, reset_levels=fresh)
        if fresh:
            self._autofill_imask_max()
        if self._geom_card.bc_auto_enabled():
            NZ, NY = self._cur.shape
            self._geom_card.center_beam_on(NY / 2.0, NZ / 2.0)
        self._geom_card.refresh_rings_and_radial()
        self._redraw_lab_axes_if_on()
        self._update_intensity_overlay()
        # "All frames" stats don't depend on the current frame — skip on frame change
        # (avoids re-reading + re-correcting the whole stack on every slider tick).
        sp = self._loader.stats_panel
        if getattr(self, "_topn_btn", None) is not None and self._topn_btn.isChecked():
            self._show_topn()   # re-locate brightest pixels + refresh their stats
        elif sp is None or sp.scope() != "all":
            self._update_stats()

    def _on_fields_changed(self):
        """Dark/bright/background/mask changed — recompute the corrected image and
        refresh the overlay + radial integration (no autorange, no autofill)."""
        if self._profile_file_mode:
            return
        raw = self._loader.current_frame()
        if raw is None:
            return
        self._cur = self._viewer.set_raw_frame(
            self._loader.corrected(raw), self._im_trans_codes(), autorange=False)
        self._redraw_lab_axes_if_on()
        self._update_intensity_overlay()
        if getattr(self, "_topn_btn", None) is not None and self._topn_btn.isChecked():
            self._show_topn()
        else:
            self._update_stats()
        self._geom_card.maybe_auto_radial()

    def _apply_project_style(self, active: bool):
        """Green highlight on "Project stack" while a projection is being
        displayed, mirroring the "Use Buffer" ready-state precedent in
        widgets.py's DataLoaderPanel._apply_buffer_style."""
        if active:
            self._proj_btn.setStyleSheet(
                "QPushButton { background:#2e7d32; color:white; font-weight:bold; "
                "border:1px solid #1b5e20; border-radius:4px; padding:4px; }")
        else:
            self._proj_btn.setStyleSheet("")

    def _project(self):
        if self._loader.n_frames() <= 1:
            QtWidgets.QMessageBox.warning(self, "No stack", "Projection needs a stack."); return
        if self._proj_worker and self._proj_worker.isRunning():
            return
        method = next(m for m, b in self._proj_method.items() if b.isChecked())
        skip = self._proj_skip.value()
        nframes = self._proj_nframes.value()
        self._info_lbl.setText("Projecting stack…")
        self._proj_btn.setEnabled(False)
        self._proj_worker = ProjectionWorker(
            self._loader.full_stack, method, 0, skip, nframes,
            dark=self._loader.dark(), bright=self._loader.bright(),
            background=self._loader.background(), bright_mode=self._loader.bright_mode(),
            parent=self)
        self._proj_worker.finished.connect(self._on_projection_done)
        self._proj_worker.failed.connect(self._on_projection_fail)
        self._proj_worker.start()

    def _on_projection_done(self, img, info):
        self._proj_btn.setEnabled(True)
        self._proj_raw = img
        self._is_projection = True
        self._apply_project_style(True)
        if self._loader.stats_panel is not None:
            self._loader.stats_panel.set_scope_enabled(False)
        self._cur = self._viewer.set_raw_frame(img, self._im_trans_codes())
        if self._geom_card.bc_auto_enabled():
            NZ, NY = self._cur.shape
            self._geom_card.center_beam_on(NY / 2.0, NZ / 2.0)
        self._geom_card.refresh_after_projection()
        self._redraw_lab_axes_if_on()
        self._update_intensity_overlay()
        self._update_stats()
        self._info_lbl.setText(info)

    def _on_projection_fail(self, msg):
        self._proj_btn.setEnabled(True)
        self._info_lbl.setText("Projection failed.")
        show_error(self, "Projection error", msg)

    def _on_radius_clicked(self, r_px: float):
        """A radius was clicked on the profile — draw its ring on the image."""
        self._info_lbl.setText(self._geom_card.on_radius_clicked(r_px))

    # ── Radial Profile "Source" — a loaded profile file, in place of the
    # live detector frame's own integration ────────────────────────────

    def _on_profile_source_changed(self, _idx: int):
        if self._prof_source.currentData() != "file":
            self._exit_profile_file_mode()
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load profile file", "", PROFILE_FILE_FILTER)
        if not path:
            self._prof_source.blockSignals(True)
            self._prof_source.setCurrentIndex(0)
            self._prof_source.blockSignals(False)
            return
        self._load_profile_file(path)

    def _load_profile_file(self, path: str):
        try:
            x, y, sigma = load_profile_file(path)
        except Exception:
            import traceback
            show_error(self, "Profile load error", traceback.format_exc())
            self._prof_source.blockSignals(True)
            self._prof_source.setCurrentIndex(0)
            self._prof_source.blockSignals(False)
            return
        self._loaded_profile = (x, y, sigma, profile_file_axis_kind(path))
        self._profile_file_mode = True
        self._prof_source_lbl.setText(Path(path).name)
        # No image in this mode — blank the viewer (pyqtgraph's own
        # ImageView.clear(), which only touches the displayed image, not the
        # overlay/crosshair items added separately) and clear self._cur so
        # every existing image-dependent guard (radial_integrate, Top-N,
        # lab-frame axes, …) already no-ops correctly with no further changes.
        self._cur = None
        self._viewer._iv.clear()
        self._viewer._data = None
        self._viewer._refresh_coord_bar()
        self._frame_scrub_bar.setVisible(False)
        self._geom_card.simulate_rings_without_image()
        self._render_loaded_profile()

    def _exit_profile_file_mode(self):
        if not self._profile_file_mode:
            return
        self._loaded_profile = None
        self._profile_file_mode = False
        self._prof_source_lbl.setText("")
        self._on_loader_data()

    def _render_loaded_profile(self):
        """Plot the loaded profile file: straight through if it's already in
        R_px, converted (with full ring/unit-toggle support) if a calibration
        is attached, or in its own native axis (locked, no ring overlay)
        otherwise — see helpers.native_axis_to_r_px."""
        if self._loaded_profile is None:
            return
        x, y, sigma, x_kind = self._loaded_profile
        if x_kind == "r_px":
            self._profile_view.set_profile(x, y, sigma=sigma)
            return
        if self._geom_card.has_calibration():
            geo = self._geom_card.get_geometry()
            lsd_um, px_um, wl = geo["Lsd"], geo["pxY"], geo["wavelength_A"]
            r_px = native_axis_to_r_px(x, x_kind, lsd_um, px_um, wl)
            self._profile_view.set_profile(r_px, y, sigma=sigma,
                                           wavelength_A=wl, lsd_um=lsd_um, px_um=px_um)
        else:
            self._profile_view.set_profile(x, y, sigma=sigma, native_unit=x_kind)

    def _on_profile_geometry_changed(self):
        """Calibration/material/geometry changed while a profile file is
        loaded — immediately upgrade a native-axis plot to a converted,
        ring-annotated one (or vice versa if the calibration was cleared),
        and keep the ring markers themselves in sync (the normal image-driven
        "Simulate rings" flow never runs with no image loaded)."""
        if self._profile_file_mode:
            self._geom_card.simulate_rings_without_image()
            self._render_loaded_profile()

    # ── Lab-frame axes overlay ───────────────────────────────────────
    #
    # APS/MIDAS lab-frame X_Lab/Y_Lab compass, beam-direction (Z_Lab) glyph,
    # and an eta-sweep arc, anchored at the beam centre — lets the user
    # verify orientation/ImTransOpt by checking that a feature lands in the
    # quadrant the overlay predicts. All items are plain pyqtgraph scene
    # items added onto the viewer's ViewBox, so pan/zoom transforms them for
    # free; they only need rebuilding when the image/beam-centre changes — or
    # when the display origin flips, which inverts the ViewBox's Y axis and so
    # needs the compass re-derived to keep pointing at the real hutch (see
    # widgets.build_lab_frame_axes_items).

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
            self._viewer._iv.removeItem(it)
        self._axis_items.clear()

    def _draw_lab_axes(self):
        self._clear_lab_axes()
        if self._cur is None:
            return
        geo = self._geom_card.get_geometry()
        items = build_lab_frame_axes_items(
            self._viewer._iv, self._cur.shape, geo["BC_y"], geo["BC_z"])
        for it in items:
            self._viewer._iv.addItem(it)
        self._axis_items.extend(items)

    # ── Intensity mask ──────────────────────────────────────────────

    def _autofill_imask_max(self):
        """Default the intensity-mask upper bound to max(99.99th percentile, 100000)."""
        if self._cur is None:
            return
        fin = self._cur[np.isfinite(self._cur)]
        if not fin.size:
            return
        hi = max(int(round(np.percentile(fin, 99.99))), 100_000)
        self._imask_hi.blockSignals(True)
        self._imask_hi.setValue(hi)
        self._imask_hi.blockSignals(False)
        if self._imask_on.isChecked():
            self._update_intensity_overlay()
            self._geom_card.maybe_auto_radial()

    def _intensity_bad_mask(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Boolean mask (True = excluded) from the intensity-range controls, or None."""
        if not self._imask_on.isChecked() or img is None:
            return None
        lo, hi = self._imask_lo.value(), self._imask_hi.value()
        bad = ~np.isfinite(img)
        if lo > _IMASK_MIN:
            bad |= (img < lo)
        if hi > 0:
            bad |= (img > hi)
        return bad

    def _combined_bad_mask(self, img):
        """Union of the intensity-range mask and the loader's composite mask files."""
        if img is None:
            return None
        parts = []
        im = self._intensity_bad_mask(img)
        if im is not None:
            parts.append(im)
        cm = self._loader.composite_mask()
        if cm is not None:
            # composite_mask() is always raw detector-space; `img` (self._cur)
            # is shown in this tab's own transformed orientation — align them.
            codes = self._im_trans_codes()
            cm = _apply_im_trans(cm, codes) if codes else cm
            if cm.shape == img.shape:
                parts.append(cm != 0)
        if not parts:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = out | p
        return out

    def _update_intensity_overlay(self):
        bad = self._combined_bad_mask(self._cur)
        if bad is None or not bad.any():
            self._viewer.clear_overlay()
            self._viewer.set_bad_mask(None)
        else:
            self._viewer.set_mask_overlay(bad)
            self._viewer.set_bad_mask(bad)

    def _on_imask_changed(self, *_):
        self._update_intensity_overlay()
        if getattr(self, "_topn_btn", None) is not None and self._topn_btn.isChecked():
            self._show_topn()   # re-rank excluding the updated intensity mask
        else:
            self._update_stats()
        self._geom_card.maybe_auto_radial()

    def _on_imask_value_changed(self, *_):
        """Spinbox valueChanged fires on every keystroke/arrow-click; give
        immediate visual feedback but debounce the heavier stats/radial
        recompute until the value settles."""
        self._update_intensity_overlay()
        self._imask_debounce_timer.start()

    # ── Top-N brightest pixels ────────────────────────────────────────
    def _on_topn_toggled(self, on):
        if on:
            self._show_topn()
        else:
            self._clear_topn()
            self._update_stats()   # restore the normal stats readout

    def _clear_topn(self):
        for it in self._topn_items:
            self._viewer._iv.removeItem(it)
        self._topn_items.clear()

    def _show_topn(self):
        """Mark the N highest-intensity pixels and show their stats."""
        self._clear_topn()
        img = self._cur
        if img is None:
            return
        flat = np.nan_to_num(np.asarray(img, dtype=np.float64),
                             nan=-np.inf, posinf=-np.inf).ravel()
        # Exclude masked pixels (mask file + intensity-range mask) so they can
        # never rank in the Top-N or skew its statistics/plot.
        bad = self._combined_bad_mask(img)
        if bad is not None:
            flat[np.asarray(bad).ravel()] = -np.inf
        # Optional intensity floor: pixels at/below it are excluded from ranking
        # (and so never marked), regardless of whether they'd otherwise be Top-N.
        if self._topn_thresh_on.isChecked():
            flat[flat <= self._topn_thresh.value()] = -np.inf
        n_valid = int(np.count_nonzero(np.isfinite(flat)))
        n = min(int(self._topn_spin.value()), n_valid)
        if n <= 0:
            return
        idx = np.argpartition(flat, -n)[-n:]
        rows, cols = np.unravel_index(idx, img.shape)   # rows=Z, cols=Y
        # Image is displayed transposed, so plot (x=col=Y, y=row=Z); +0.5 centres.
        x = cols.astype(float) + 0.5
        y = rows.astype(float) + 0.5
        circ = pg.ScatterPlotItem(x=x, y=y, symbol="o", size=26,
                                  pen=pg.mkPen("#00cfff", width=1.5),
                                  brush=pg.mkBrush(0, 207, 255, 60))
        cross = pg.ScatterPlotItem(x=x, y=y, symbol="+", size=10,
                                   pen=pg.mkPen("#00cfff", width=1.5),
                                   brush=pg.mkBrush(0, 0, 0, 0))
        for it in (circ, cross):
            self._viewer._iv.addItem(it); self._topn_items.append(it)
        sp = getattr(self._loader, "stats_panel", None)
        if sp is not None:
            sp.set_data(flat[idx], scope=f"Top {n} pixels")

    # ── intensity statistics (left panel) ─────────────────────────────
    def _update_stats(self, *_):
        """Recompute the intensity-stats panel for the current scope/state."""
        sp = getattr(self._loader, "stats_panel", None)
        if sp is None:
            return
        if getattr(self, "_topn_btn", None) is not None and self._topn_btn.isChecked():
            return   # Top-N owns the stats panel while active
        try:
            if self._is_projection:
                img = self._cur
                if img is None:
                    return
                sp.set_data(self._unmasked_values(img), scope="Projected stack")
            elif sp.scope() == "all":
                self._update_stats_all_frames()
            else:
                img = self._cur
                if img is None:
                    return
                sp.set_data(self._unmasked_values(img),
                            scope=f"Frame {self._loader.frame_index()}")
        except Exception:
            import traceback
            traceback.print_exc()

    def _unmasked_values(self, img):
        """1-D array of an image's pixels, excluding intensity-range + file masks."""
        bad = self._combined_bad_mask(img)
        return img[~bad] if bad is not None else np.asarray(img).ravel()

    def _update_stats_all_frames(self):
        """Kick off (or flag for re-run) a background computation of the "All
        frames" stats scope. Reading + correcting an entire stack/live buffer
        synchronously here would freeze the UI for large data, unlike the
        per-frame scope which already has a corrected frame in hand."""
        if self._stats_worker is not None and self._stats_worker.isRunning():
            self._stats_all_frames_dirty = True
            return
        self._stats_all_frames_dirty = False
        self._stats_worker = AllFrameStatsWorker(
            self._loader.full_stack,
            dark=self._loader.dark(), bright=self._loader.bright(),
            background=self._loader.background(), bright_mode=self._loader.bright_mode(),
            composite_mask=self._loader.composite_mask(),
            imask_on=self._imask_on.isChecked(),
            imask_lo=self._imask_lo.value(), imask_hi=self._imask_hi.value(),
            parent=self)
        self._stats_worker.finished.connect(self._on_all_frame_stats_done)
        self._stats_worker.failed.connect(self._on_all_frame_stats_fail)
        self._stats_worker.start()

    def _on_all_frame_stats_done(self, vals, n):
        sp = getattr(self._loader, "stats_panel", None)
        if sp is not None and sp.scope() == "all" and not self._is_projection:
            sp.set_data(vals, scope=f"All frames ({n})")
        if self._stats_all_frames_dirty:
            self._update_stats_all_frames()

    def _on_all_frame_stats_fail(self, msg):
        print(msg)   # background stats readout — mirror the try/except's
                      # silent-to-the-user, printed-to-console handling.
        if self._stats_all_frames_dirty:
            self._update_stats_all_frames()
