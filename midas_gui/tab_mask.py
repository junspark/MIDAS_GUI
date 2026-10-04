"""Tab 1 — Mask Builder.

Threshold mask (always applied) + checkbox-gated statistical spatial-outlier
auto-mask, with TIFF save/load and a red bad-pixel overlay.  Ported from v3.
"""
from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from midas_gui.constants import _SENTINELS, H5_EXTS, DEFAULT_CALIBRANT_TIF
from midas_gui.helpers import (_load_image, _fspin, _NoScrollSpinBox, _browse, is_h5,
                               _NoScrollComboBox, list_h5_datasets,
                               widgets_to_dict, apply_dict_to_widgets,
                               new_temp_h5_path, save_stack_h5,
                               display_text_for_paths,
                               browse_start_dir, warn_if_path_missing,
                               pixel_readout_text, im_trans_map_point,
                               _apply_im_trans, map_roi_state, map_point_xy)
from midas_gui.widgets import ImageViewer
from midas_gui.dialogs import show_error
from midas_gui.workers import MaskComputeWorker
from midas_gui import project
from midas_gui import settings
from midas_gui import style as S


class MaskTab(QtWidgets.QWidget):
    maskReady = QtCore.pyqtSignal(object)   # np.ndarray (uint8) or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._orig_dtype: Optional[np.dtype] = None
        self._mask: Optional[np.ndarray] = None
        self._thresh_mask: Optional[np.ndarray] = None
        self._mask_worker = None
        self._calib_result = None
        # Mask composed of: computed (threshold/stat/…) OR hand-drawn shapes
        self._computed_mask: Optional[np.ndarray] = None   # bool (NZ, NY) or None
        self._drawn_mask: Optional[np.ndarray] = None       # bool (NZ, NY) or None
        self._shapes: list = []        # [{'kind':'shape'|'annulus', ...}]
        self._points: list = []        # [(col, row)] single-pixel picks
        self._point_items: list = []   # scatter markers for points
        self._point_mode = False
        self._click_proxy = None
        # ImTransOpt codes the viewer is currently *painting* under, mirrored
        # from the Data Viewer. Display only: self._image, every computed
        # mask and everything maskReady emits stay in raw detector space.
        self._disp_codes: tuple = ()
        # Freeform click-polygon state
        self._freeform_mode = False
        self._freeform_pts: list = []    # [(x, y)] image-coord vertices
        self._freeform_line = None       # pg.PlotDataItem — live edge preview
        self._freeform_vdots = None      # pg.ScatterPlotItem — vertex markers
        self._registry = None            # DataSourceRegistry, set by bind_registry()
        self._stack_snapshot_file = None  # temp .h5 from importing another tab's buffer
        self._stack_files: Optional[list] = None  # explicit multi-select stack files, or None
        # Multi-frame Image source (folder of frames / multi-page TIFF / 3-D HDF5
        # dataset / multi-frame .geN): None means the Image field is a plain 2-D
        # single image. See ``_detect_multiframe``/``_get_frame_array``.
        self._img_frames: Optional[dict] = None
        self._img_frame_idx: int = 0
        self._project_ctx = None
        self._build_ui()
        if Path(self._img_edit.text().strip() or "x").exists():
            self._load_image()

    def set_project_context(self, ctx):
        self._project_ctx = ctx

    def bind_registry(self, registry):
        """Register this tab as an importable ("path" kind only — Mask Builder
        has no in-memory buffer of its own) data source, and let its Image /
        Stack browse menus pull data loaded in other tabs."""
        self._registry = registry
        registry.register("Mask Builder", self)

    def describe_source(self):
        raw = self._img_edit.text().strip()
        if not raw:
            return None
        if self._img_frames is not None and self._img_frames["kind"] == "files":
            # A folder isn't itself loadable elsewhere — hand over the
            # specific frame file currently shown instead.
            raw = self._img_frames["paths"][self._img_frame_idx]
        return {"kind": "path", "path": raw,
                "dataset": self._h5loc_edit.currentText().strip() if is_h5(raw) else None,
                "field": "data", "label": "Mask Builder"}

    # ── Display transform (mirrored from the Data Viewer) ────────

    def _disp_shape(self) -> tuple:
        """``(n_rows, n_cols)`` of the frame *as painted*, which differs from
        ``self._image.shape`` whenever an odd number of transposes is active.

        Everything drawn over the image — ROI defaults, click bounds, shape
        rasterisation — is in this frame. The array, and every mask built
        from it, stay raw.
        """
        if self._image is None:
            return (0, 0)
        n_rows, n_cols = self._image.shape
        if sum(1 for c in self._disp_codes if c == 3) % 2:
            return (n_cols, n_rows)
        return (n_rows, n_cols)

    def set_display_transform(self, codes) -> None:
        """Paint the detector under ``codes`` — the Data Viewer's Transforms
        checkboxes, broadcast by ``app.py``.

        Display only. The mask this tab computes, saves and emits is always
        raw-frame, because Calibrate/Batch/Integrate apply the calibration's
        own ImTransOpt to it themselves (DECISIONS 2026-08-25). Changing the
        orientation here must not change a single bit of that mask.

        Shapes already drawn stay over the same detector pixels: each ROI is
        re-placed from the old painted frame into the new one, so flipping
        the view does not silently re-aim a mask you already built.
        """
        codes = tuple(codes or ())
        if codes == self._disp_codes:
            return
        old = self._disp_codes
        self._cancel_freeform()          # a half-drawn polygon has no anchor yet
        self._remap_drawn_items(old, codes)
        self._disp_codes = codes
        if self._image is not None:
            self._viewer.set_raw_frame(self._image, codes)
        self._paint_overlay()
        self._viewer._refresh_coord_bar()

    def _remap_drawn_items(self, old: tuple, new: tuple) -> None:
        """Move ROIs and picked points from the ``old`` painted frame to the
        ``new`` one so they keep covering the same detector pixels.

        The composite map is "undo old, then apply new" — reversed old codes
        followed by the new ones — starting from the old painted extent.
        """
        if self._image is None:
            return
        combo = tuple(reversed(old)) + tuple(new)
        if not combo:
            return
        shape = self._disp_shape()       # still the OLD frame at this point

        for item in self._shapes:
            rois = ([item["roi"]] if item["kind"] == "shape"
                    else [item["outer"], item["inner"]])
            for roi in rois:
                if isinstance(roi, pg.PolyLineROI):
                    pts = [roi.mapToParent(h.pos())
                           for _i, h in roi.getLocalHandlePositions()]
                    moved = [map_point_xy(p.x(), p.y(), shape, combo)[:2]
                             for p in pts]
                    roi.setPos((0, 0))
                    roi.setPoints(moved, closed=True)
                    continue
                pos, size, angle = map_roi_state(
                    (roi.pos().x(), roi.pos().y()),
                    (roi.size().x(), roi.size().y()),
                    roi.angle(), shape, combo)
                roi.setPos(pos[0], pos[1])
                roi.setSize(size)
                roi.setAngle(angle)

        moved_pts = []
        for (col, row), dot in zip(self._points, self._point_items):
            c2, r2 = im_trans_map_point(col, row, shape, combo)
            moved_pts.append((c2, r2))
            dot.setData([c2], [r2])
        self._points = moved_pts

    def _paint_overlay(self) -> None:
        """Repaint the bad-pixel overlay in the displayed frame.

        ``self._mask`` is raw; the picture under it may not be.
        """
        if self._mask is None:
            self._viewer.clear_overlay()
            return
        shown = (_apply_im_trans(self._mask, self._disp_codes)
                 if self._disp_codes else self._mask)
        self._viewer.set_mask_overlay(shown)
        self._viewer.set_overlay_visible(self._show_overlay_check.isChecked())

    def _radial_readout(self, col, row) -> str:
        """2θ / Q / d / η under the cursor — see
        ``widgets.ImageViewer.set_radial_readout_fn``.

        Two frames sit between the cursor and the geometry: what this tab
        *paints* (``_disp_codes``, mirrored from the Data Viewer) and what
        the calibration was *fit in* (``result.im_trans``). They are usually
        the same codes, but nothing guarantees it, so both legs are composed
        explicitly — undo the display transform back to raw, then apply the
        calibration's — rather than leaning on them matching.

        Nothing is shown unless the resulting shape matches the detector the
        calibration was fit on — a mask built against one detector and a
        calibration from another would otherwise read plausibly. Same guard,
        and the same reasoning, as the mask-overlay shape check.
        """
        r = self._calib_result
        if r is None or self._image is None:
            return ""
        combo = (tuple(reversed(self._disp_codes))
                 + tuple(getattr(r, "im_trans", ()) or ()))
        n_rows, n_cols = self._disp_shape()
        col, row = im_trans_map_point(col, row, (n_rows, n_cols), combo)
        # Each transpose swaps the frame's extent; an even number cancels.
        if sum(1 for c in combo if c == 3) % 2:
            n_rows, n_cols = n_cols, n_rows
        ny, nz = getattr(r, "NrPixelsY", None), getattr(r, "NrPixelsZ", None)
        if ny and nz and (int(ny) != n_cols or int(nz) != n_rows):
            return ""
        pxY = float(getattr(r, "pxY", 0.0) or 0.0)
        return pixel_readout_text(col, row, {
            "Lsd": getattr(r, "Lsd", None),
            "BC_y": getattr(r, "BC_y", None), "BC_z": getattr(r, "BC_z", None),
            "pxY": pxY, "pxZ": float(getattr(r, "pxZ", 0.0) or pxY),
            "tx": float(getattr(r, "tx", 0.0) or 0.0),
            "ty": float(getattr(r, "ty", 0.0) or 0.0),
            "tz": float(getattr(r, "tz", 0.0) or 0.0),
            "wavelength_A": getattr(r, "wavelength_A", None)})

    def set_calibration(self, result):
        """Receive calibration from Tab 2 — enables geometry-based mask methods."""
        self._calib_result = result
        # Geometry arriving does not move the cursor.
        if getattr(self, "_viewer", None) is not None:
            self._viewer._refresh_coord_bar()
        if result is not None:
            self._geom_group.setEnabled(True)
            self._geom_note.setText(
                f"Calibration available (Lsd={result.Lsd/1000:.2f} mm) — "
                "azimuthal & learnable masks enabled. The image is shown in the "
                "Data Viewer's orientation, but the mask is always built and "
                "saved against the raw detector image — Calibrate/Batch/Integrate "
                "apply the calibration's own ImTransOpt to it automatically.")

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setFixedWidth(468)
        inner = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(inner)
        lv.setContentsMargins(2, 2, 2, 2); lv.setSpacing(8)
        scroll.setWidget(inner)

        def _br(w=30):
            b = QtWidgets.QPushButton("…"); b.setFixedWidth(w); return b

        def _frow(edit, slot):
            r = QtWidgets.QHBoxLayout(); r.setSpacing(4)
            r.addWidget(edit); b = _br(); b.clicked.connect(slot); r.addWidget(b)
            return r

        # ── Image ──
        img = S.make_card("Image")
        self._img_edit = QtWidgets.QLineEdit(DEFAULT_CALIBRANT_TIF)
        self._img_edit.setPlaceholderText("Select image file…")
        _img_browse = QtWidgets.QToolButton(); _img_browse.setText("⋯"); _img_browse.setFixedWidth(30)
        _img_browse.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._img_menu = QtWidgets.QMenu(_img_browse)
        self._img_menu.addAction("File…", self._browse_img)
        self._img_menu.addAction("Folder…", self._browse_img_folder)
        self._img_menu.addSeparator()
        self._img_import_menu = self._img_menu.addMenu("Import from…")
        self._img_import_menu.aboutToShow.connect(self._populate_mask_image_import_menu)
        _img_browse.setMenu(self._img_menu)
        _img_row = QtWidgets.QHBoxLayout(); _img_row.setSpacing(4)
        _img_row.addWidget(self._img_edit); _img_row.addWidget(_img_browse)
        img.body.addLayout(_img_row)
        self._h5loc_edit = _NoScrollComboBox(); self._h5loc_edit.setEditable(True)
        self._h5loc_edit.setEditText("exchange/data")
        self._h5loc_lbl = S.LabelRight("Dataset:")
        self._h5loc_lbl.setVisible(False); self._h5loc_edit.setVisible(False)
        ds = QtWidgets.QHBoxLayout(); ds.setSpacing(4)
        ds.addWidget(self._h5loc_lbl); ds.addWidget(self._h5loc_edit, 1)
        img.body.addLayout(ds)
        # Frame navigator — shown only when the Image field resolves to more
        # than one frame (a folder of files, a multi-page TIFF, a 3-D HDF5
        # dataset, or a multi-frame .geN file).
        self._frame_nav_row = QtWidgets.QWidget()
        fnl = QtWidgets.QHBoxLayout(self._frame_nav_row)
        fnl.setContentsMargins(0, 0, 0, 0); fnl.setSpacing(4)
        fnl.addWidget(QtWidgets.QLabel("Frame:"))
        self._frame_prev_btn = QtWidgets.QPushButton("◀"); self._frame_prev_btn.setFixedWidth(28)
        self._frame_prev_btn.setObjectName("frameNavBtn")
        self._frame_prev_btn.clicked.connect(lambda: self._frame_spin.stepBy(-1))
        fnl.addWidget(self._frame_prev_btn)
        self._frame_spin = _NoScrollSpinBox(); self._frame_spin.setRange(1, 1)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        fnl.addWidget(self._frame_spin)
        self._frame_count_lbl = QtWidgets.QLabel("/ 1")
        fnl.addWidget(self._frame_count_lbl)
        self._frame_next_btn = QtWidgets.QPushButton("▶"); self._frame_next_btn.setFixedWidth(28)
        self._frame_next_btn.setObjectName("frameNavBtn")
        self._frame_next_btn.clicked.connect(lambda: self._frame_spin.stepBy(1))
        fnl.addWidget(self._frame_next_btn)
        fnl.addStretch(1)
        self._frame_nav_row.setVisible(False)
        img.body.addWidget(self._frame_nav_row)
        self._img_edit.textChanged.connect(self._on_img_path_changed)
        self._img_edit.returnPressed.connect(self._load_image)
        self._h5loc_edit.currentIndexChanged.connect(
            lambda _=0: self._image is not None and self._load_image())
        self._h5loc_edit.lineEdit().editingFinished.connect(
            lambda: self._image is not None and self._load_image())
        lv.addWidget(img)

        # ── 1 · Threshold ──
        thr = S.make_card("1 · Threshold mask")
        self._lower = _fspin(-1e9, 1e9, 1, 0.0)
        self._lower.setToolTip("Pixels ≤ this value are masked (dead / gap / beam-stop)")
        self._upper = _fspin(0, 5e9, 0, 1_048_575)
        self._upper.setToolTip("Pixels > this value are masked. Auto-filled from dtype on load.")
        thr.body.addLayout(S.Form().row(("pixel ≤", self._lower), ("pixel >", self._upper)))
        self._thresh_proj_combo = _NoScrollComboBox()
        self._thresh_proj_combo.addItems(["Current frame", "Average", "Sum", "Max"])
        self._thresh_proj_combo.setEnabled(False)
        self._thresh_proj_combo.setToolTip(
            "When the Image above is a folder or a multi-frame file, build the\n"
            "threshold mask from a projection across every frame instead of only\n"
            "the frame shown by the Frame navigator. 'Current frame' (the only\n"
            "choice for a plain single image) uses just that one frame.")
        thr.body.addLayout(S.Form().row(("Projection:", self._thresh_proj_combo)))
        lv.addWidget(thr)

        # ── 2 · Statistical auto-mask (spatial + temporal, independent) ──
        auto = S.make_card("2 · Statistical auto-mask")
        self._spatial_check = QtWidgets.QCheckBox("Spatial outlier  (local robust Z-score)")
        self._spatial_check.setToolTip(
            "Flags pixels whose local Z-score exceeds K_σ AND whose intensity\n"
            "passes the hot/dead magnitude gate. Uses the single image, or the\n"
            "temporal median if a stack is given. No MIDAS geometry required.")
        self._temporal_check = QtWidgets.QCheckBox("Temporal constancy  (frozen pixels)")
        self._temporal_check.setToolTip(
            "Flags pixels whose frame-to-frame std < Frozen × Q75(std):\n"
            "constant-value regions (dead modules, stuck pixels).\n"
            "Requires a stack of ≥2 frames.")
        auto.body.addWidget(self._spatial_check)
        auto.body.addWidget(self._temporal_check)

        self._auto_widget = QtWidgets.QWidget()
        awf = QtWidgets.QVBoxLayout(self._auto_widget); awf.setContentsMargins(0, 0, 0, 0); awf.setSpacing(5)
        self._k_sigma = _fspin(0.0, 1e9, 1, 6.0)
        self._hot_factor = _fspin(0.0, 1e9, 2, 1.5)
        self._dead_factor = _fspin(0.0, 1e9, 2, 0.5)
        self._frozen_frac = _fspin(0.0, 1e9, 3, 0.05)
        self._frozen_frac.setToolTip(
            "Temporal constancy: flag pixels whose frame-to-frame std < this × Q75(std).\n"
            "Catches constant-value regions (dead modules, stuck pixels).")
        # spatial-only parameters
        self._spatial_params = QtWidgets.QWidget()
        spv = QtWidgets.QVBoxLayout(self._spatial_params); spv.setContentsMargins(0, 0, 0, 0); spv.setSpacing(4)
        sf = S.Form(); sf.row(("K_σ:", self._k_sigma), ("Hot:", self._hot_factor)); sf.row(("Dead:", self._dead_factor))
        spv.addLayout(sf)
        awf.addWidget(self._spatial_params)
        # temporal-only parameter
        self._temporal_params = QtWidgets.QWidget()
        tpv = QtWidgets.QVBoxLayout(self._temporal_params); tpv.setContentsMargins(0, 0, 0, 0); tpv.setSpacing(4)
        tpv.addLayout(S.Form().row(("Frozen:", self._frozen_frac)))
        awf.addWidget(self._temporal_params)
        # Shared stack source (temporal median for spatial; frame stack for temporal).
        # Accepts a folder / *.tif glob, OR a single HDF5 file whose 3-D dataset is a
        # time sequence of images.
        self._stack_ed = QtWidgets.QLineEdit()
        self._stack_ed.setPlaceholderText(
            "folder / *.tif / .h5 (3-D dataset)   (blank = single image)")
        self._stack_ed.textChanged.connect(self._on_stack_path_changed)
        warn_if_path_missing(self._stack_ed, self)
        awf.addWidget(QtWidgets.QLabel("Stack (temporal median / temporal constancy):"))
        _sbrowse = QtWidgets.QToolButton(); _sbrowse.setText("⋯"); _sbrowse.setFixedWidth(28)
        _sbrowse.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._stack_menu = QtWidgets.QMenu(_sbrowse)
        self._stack_menu.aboutToShow.connect(self._populate_stack_menu)
        _sbrowse.setMenu(self._stack_menu)
        self._populate_stack_menu()
        _srow = QtWidgets.QHBoxLayout(); _srow.setSpacing(4)
        _srow.addWidget(self._stack_ed); _srow.addWidget(_sbrowse)
        awf.addLayout(_srow)
        # HDF5 dataset selector — shown only for a single .h5 file.
        self._stack_ds_row = QtWidgets.QWidget()
        _dsl = QtWidgets.QHBoxLayout(self._stack_ds_row)
        _dsl.setContentsMargins(0, 0, 0, 0); _dsl.setSpacing(4)
        self._stack_ds_combo = _NoScrollComboBox(); self._stack_ds_combo.setEditable(True)
        self._stack_ds_combo.setEditText("exchange/data")
        _dsl.addWidget(QtWidgets.QLabel("Dataset:")); _dsl.addWidget(self._stack_ds_combo, 1)
        self._stack_ds_row.setVisible(False)
        awf.addWidget(self._stack_ds_row)
        self._stride_spin = _NoScrollSpinBox(); self._stride_spin.setRange(1, 1_000_000); self._stride_spin.setValue(1)
        awf.addLayout(S.Form().row(("Stride:", self._stride_spin)))
        self._auto_widget.setVisible(False)

        def _update_auto_visibility(_=0):
            sp, tp = self._spatial_check.isChecked(), self._temporal_check.isChecked()
            self._spatial_params.setVisible(sp)
            self._temporal_params.setVisible(tp)
            self._auto_widget.setVisible(sp or tp)
        self._spatial_check.toggled.connect(_update_auto_visibility)
        self._temporal_check.toggled.connect(_update_auto_visibility)
        _update_auto_visibility()
        auto.body.addWidget(self._auto_widget)
        self._stat_prog = QtWidgets.QLabel("")
        self._stat_prog.setStyleSheet("color:#7fb8ff;font-size:10px"); self._stat_prog.setWordWrap(True)
        auto.body.addWidget(self._stat_prog)
        lv.addWidget(auto)

        # ── 3 · Spike rejection ──
        spike = S.make_card("3 · Spatial spike rejection")
        self._spike_check = QtWidgets.QCheckBox("Enable  (Laplacian spike detector)")
        self._spike_check.setToolTip(
            "Flags isolated single-pixel spikes via a Laplacian high-pass + robust σ.")
        spike.body.addWidget(self._spike_check)
        self._spike_sigma = _fspin(0.0, 1e9, 1, 5.0)
        spike.body.addLayout(S.Form().row(("n_σ:", self._spike_sigma)))
        lv.addWidget(spike)

        # ── 3b · Cosmic-ray rejection ──
        cosmic = S.make_card("3b · Cosmic-ray rejection (temporal)")
        self._cosmic_check = QtWidgets.QCheckBox(
            "Enable  (temporal σ-clip across frames)")
        self._cosmic_check.setToolTip(
            "Flags pixels that are statistical outliers in any frame compared\n"
            "to the per-pixel temporal median.\n"
            "Requires a stack of ≥3 frames in the same folder as section 2.\n"
            "Uses the stack folder specified in section 2 above.")
        cosmic.body.addWidget(self._cosmic_check)
        self._cosmic_sigma = _fspin(0.0, 1e9, 1, 5.0)
        self._cosmic_sigma.setToolTip("n_σ threshold — pixels deviating more than\n"
                                      "this many MAD-σ from the temporal median are flagged.")
        cosmic.body.addLayout(S.Form().row(("n_σ:", self._cosmic_sigma)))
        cosmic_note = QtWidgets.QLabel(
            "Uses the stack folder from section 2.  "
            "Needs ≥3 frames.  Flags cosmic-ray hits that look like isolated\n"
            "bright spikes in individual frames but are absent in other frames.")
        cosmic_note.setWordWrap(True)
        cosmic_note.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        cosmic.body.addWidget(cosmic_note)
        lv.addWidget(cosmic)

        # ── 4 · Calibration-based masks ──
        self._geom_group = S.make_card("4 · Calibration-based masks")
        self._geom_group.setEnabled(False)
        self._geom_note = QtWidgets.QLabel("Run Tab 2 calibration to enable these.")
        self._geom_note.setStyleSheet(f"color:{S.ACCENT};font-size:10px"); self._geom_note.setWordWrap(True)
        self._geom_group.body.addWidget(self._geom_note)
        self._azim_check = QtWidgets.QCheckBox("Azimuthal σ-clip")
        self._azim_check.setToolTip("Per-(R,η) ring-uniformity outlier clip (needs geometry).")
        self._azim_sigma = _fspin(0.0, 1e9, 1, 5.0)
        self._geom_group.body.addLayout(S.Form().row(("Azimuthal n_σ:", self._azim_sigma)))
        self._geom_group.body.addWidget(self._azim_check)
        self._learn_check = QtWidgets.QCheckBox("Learnable mask")
        self._learn_check.setToolTip(
            "Differentiable per-pixel weights optimised against ring η-uniformity.")
        self._geom_group.body.addWidget(self._learn_check)
        self._learn_steps = _NoScrollSpinBox(); self._learn_steps.setRange(1, 1_000_000); self._learn_steps.setValue(300)
        self._learn_lr = _fspin(0.0, 1e6, 2, 0.5)
        self._learn_sparsity = _fspin(0.0, 1e6, 5, 1e-4)
        lf = S.Form()
        lf.row(("steps:", self._learn_steps), ("lr:", self._learn_lr))
        lf.row(("sparsity:", self._learn_sparsity), (None, QtWidgets.QWidget()))
        self._geom_group.body.addLayout(lf)
        lv.addWidget(self._geom_group)

        # ── 5 · Post-processing ──
        post = S.make_card("5 · Post-processing")
        self._dilation_spin = _NoScrollSpinBox()
        self._dilation_spin.setRange(0, 50)
        self._dilation_spin.setValue(0)
        self._dilation_spin.setToolTip(
            "Grow bad-pixel regions by N pixels (8-neighbor morphological dilation).\n"
            "0 = no growth. 1 = the full 3×3 block around each bad pixel becomes bad.\n"
            "2 = the full 5×5 block, etc. (N → a (2N+1)×(2N+1) square per pixel).\n"
            "Applied to the computed mask (threshold/statistical/loaded) before\n"
            "combining with hand-drawn shapes.")
        post.body.addLayout(S.Form().row(("Dilation (px):", self._dilation_spin)))
        lv.addWidget(post)

        # ── Compute + stats ──
        compute_btn = S.primary_btn("Compute Mask")
        compute_btn.clicked.connect(self._compute)
        lv.addWidget(compute_btn)
        self._stat_lbl = QtWidgets.QLabel("No mask computed.")
        self._stat_lbl.setWordWrap(True); self._stat_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        lv.addWidget(self._stat_lbl)

        # ── Save / Load ──
        sl = S.make_card("Save / Load")
        self._save_edit = QtWidgets.QLineEdit(); self._save_edit.setPlaceholderText("mask.tif")
        srow = QtWidgets.QHBoxLayout(); srow.setSpacing(4)
        srow.addWidget(self._save_edit, 1)
        b2 = _br(); b2.clicked.connect(self._browse_save); srow.addWidget(b2)
        self._save_btn = QtWidgets.QPushButton("Save"); self._save_btn.setEnabled(False)
        self._save_btn.setFixedWidth(52); self._save_btn.clicked.connect(self._save); srow.addWidget(self._save_btn)
        self._log_project_btn = QtWidgets.QPushButton("Log to Project")
        self._log_project_btn.setEnabled(False)
        self._log_project_btn.setToolTip(
            "Record this mask (parameters + resulting mask) as a new FAIR-provenance "
            "attempt in the currently-open project — see File ▸ Open Project…")
        self._log_project_btn.clicked.connect(self._log_to_project)
        srow.addWidget(self._log_project_btn)
        sl.body.addLayout(srow)
        self._load_mask_edit = QtWidgets.QLineEdit(); self._load_mask_edit.setPlaceholderText("Load existing mask…")
        warn_if_path_missing(self._load_mask_edit, self)
        lrow = QtWidgets.QHBoxLayout(); lrow.setSpacing(4)
        lrow.addWidget(self._load_mask_edit, 1)
        b3 = _br(); b3.clicked.connect(self._browse_load_mask); lrow.addWidget(b3)
        sl.body.addLayout(lrow)
        lv.addWidget(sl)

        lv.addStretch(1)
        root.addWidget(scroll)

        # Right: overlay toggle + viewer
        right_panel = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right_panel)
        rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(2)
        ov_bar = QtWidgets.QHBoxLayout()
        self._show_overlay_check = QtWidgets.QCheckBox("Show bad pixels overlay")
        self._show_overlay_check.setChecked(True)
        self._show_overlay_check.toggled.connect(self._on_show_overlay)
        ov_bar.addWidget(self._show_overlay_check)
        self._draw_check = QtWidgets.QCheckBox("Draw mask")
        self._draw_check.setToolTip("Reveal tools to draw shapes that mark regions to mask out.")
        self._draw_check.toggled.connect(self._on_draw_toggled)
        ov_bar.addWidget(self._draw_check)
        ov_bar.addStretch(1)
        rv.addLayout(ov_bar)

        # Shape-drawing toolbar (hidden until "Draw mask" is on)
        self._draw_bar = QtWidgets.QWidget()
        db = QtWidgets.QHBoxLayout(self._draw_bar)
        db.setContentsMargins(0, 0, 0, 0); db.setSpacing(4)
        db.addWidget(QtWidgets.QLabel("Add:"))
        for label, slot in [("Rectangle", self._add_rect), ("Oval", self._add_oval),
                            ("Circle", self._add_circle), ("Polygon", self._add_polygon),
                            ("Annulus", self._add_annulus)]:
            b = QtWidgets.QPushButton(label); b.clicked.connect(slot)
            b.setFixedHeight(24); db.addWidget(b)
        self._freeform_btn = QtWidgets.QPushButton("Freeform"); self._freeform_btn.setCheckable(True)
        self._freeform_btn.setFixedHeight(24)
        self._freeform_btn.setToolTip(
            "Click to place vertices one by one.\n"
            "Lines connect each vertex to the next.\n"
            "Double-click OR press 'Close shape' to finish and seal the polygon.\n"
            "Need ≥ 3 vertices to close.")
        self._freeform_btn.toggled.connect(self._toggle_freeform_mode)
        db.addWidget(self._freeform_btn)
        self._close_shape_btn = QtWidgets.QPushButton("Close shape")
        self._close_shape_btn.setFixedHeight(24)
        self._close_shape_btn.setToolTip("Seal the freeform polygon and add it to the shape list.")
        self._close_shape_btn.setVisible(False)
        self._close_shape_btn.clicked.connect(self._close_freeform)
        db.addWidget(self._close_shape_btn)
        self._point_btn = QtWidgets.QPushButton("Point"); self._point_btn.setCheckable(True)
        self._point_btn.setFixedHeight(24)
        self._point_btn.setToolTip("Toggle, then click pixels to mask them individually.")
        self._point_btn.toggled.connect(self._toggle_point_mode)
        db.addWidget(self._point_btn)
        db.addStretch(1)
        self._apply_shapes_btn = QtWidgets.QPushButton("Apply shapes → mask")
        self._apply_shapes_btn.clicked.connect(self._apply_shapes); self._apply_shapes_btn.setFixedHeight(24)
        db.addWidget(self._apply_shapes_btn)
        clr = QtWidgets.QPushButton("Clear shapes"); clr.clicked.connect(self._clear_shapes); clr.setFixedHeight(24)
        db.addWidget(clr)
        self._draw_bar.setVisible(False)
        rv.addWidget(self._draw_bar)

        self._viewer = ImageViewer(title="")
        self._viewer.set_radial_readout_fn(self._radial_readout)
        rv.addWidget(self._viewer, stretch=1)
        root.addWidget(right_panel, stretch=1)

    # ── Actions ──────────────────────────────────────────────────

    def _browse_img(self):
        p = _browse(self, "Open Image",
                    "Images (*.tif *.tiff *.h5 *.hdf5 *.hdf *.nxs *.ge*);;All (*)",
                    start_dir=browse_start_dir(self._img_edit.text()))
        if p: self._img_edit.setText(p); self._load_image()

    def _browse_img_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select image folder", browse_start_dir(self._img_edit.text()))
        if d: self._img_edit.setText(d); self._load_image()

    def _browse_save(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Mask", self._save_edit.text().strip() or "mask.tif",
            "TIFF (*.tif);;All (*)")
        if p: self._save_edit.setText(p)

    def _browse_load_mask(self):
        p = _browse(self, "Open Mask", "TIFF (*.tif *.tiff);;All (*)",
                    start_dir=browse_start_dir(self._load_mask_edit.text()))
        if p: self._load_mask_edit.setText(p); self._load_existing_mask()

    def _browse_stack_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select stack folder", browse_start_dir(self._stack_ed.text()))
        if d: self._stack_ed.setText(d)

    def _browse_stack_file(self):
        p = _browse(self, "Select stack file (TIFF stack or HDF5)",
                    "Stacks (*.h5 *.hdf5 *.nxs *.tif *.tiff);;All (*)",
                    start_dir=browse_start_dir(self._stack_ed.text()))
        if p: self._stack_ed.setText(p)

    def _browse_stack_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select stack files (multi-select)",
            browse_start_dir(self._stack_ed.text()),
            "Stack frames (*.tif *.tiff *.h5 *.hdf5 *.ge*);;All (*)")
        if not files:
            return
        files = sorted(files)   # deterministic frame order regardless of click order
        self._stack_ed.setText(display_text_for_paths(files))
        self._stack_ed.setToolTip("\n".join(files))
        self._stack_files = files

    # ── cross-tab data sharing (data_bridge.DataSourceRegistry) ─────
    def _populate_stack_menu(self):
        menu = self._stack_menu
        menu.clear()
        menu.addAction("Folder…", self._browse_stack_folder)
        menu.addAction("File (TIFF / HDF5)…", self._browse_stack_file)
        menu.addAction("Files (multi-select)…", self._browse_stack_files)
        if self._registry is None:
            return
        buffers = self._registry.available(exclude=self, kind="buffer", field="data")
        if not buffers:
            return
        menu.addSeparator()
        for desc in buffers:
            n = len(desc["provider"]._buffer) if desc["provider"]._buffer else 0
            menu.addAction(f"Buffer: {desc['label']} ({n} frames)",
                            lambda d=desc: self._use_buffer_stack(d["provider"]))

    def _use_buffer_stack(self, provider):
        """Materialize another tab's ring buffer to a temp HDF5 file and point
        the Stack field at it — `MaskComputeWorker` only ever takes file paths
        / HDF5 tuples, never a raw array, so this reuses the existing stack
        loading path unchanged."""
        with provider._buffer_lock:
            frames = list(provider._buffer) if (provider._buffer_frozen and provider._buffer) else None
        if not frames:
            QtWidgets.QMessageBox.warning(self, "No buffer", "Source buffer is empty.")
            return
        old = self._stack_snapshot_file
        path = new_temp_h5_path()
        save_stack_h5(path, frames, dataset="buffer/data")
        self._stack_snapshot_file = path
        self._stack_ed.setText(path)
        if old is not None:
            import os
            try:
                os.unlink(old)
            except OSError:
                pass

    def _populate_mask_image_import_menu(self):
        menu = self._img_import_menu
        menu.clear()
        if self._registry is None:
            menu.addAction("(no other tabs loaded)").setEnabled(False)
            return
        sources = self._registry.available(exclude=self, kind="path", field="data")
        if not sources:
            menu.addAction("(nothing loaded elsewhere)").setEnabled(False)
            return
        for desc in sources:
            menu.addAction(f"{desc['label']}: {desc['path']}",
                            lambda d=desc: self._apply_imported_image_source(d))

    def _apply_imported_image_source(self, desc: dict):
        self._img_edit.setText(desc["path"])
        if desc.get("dataset") and is_h5(desc["path"]):
            self._h5loc_edit.setEditText(desc["dataset"])
        self._load_image()

    def _on_img_path_changed(self, txt: str):
        """Show the dataset selector + list datasets when the image is HDF5."""
        h5 = is_h5(txt) and Path(txt).is_file()
        self._h5loc_lbl.setVisible(h5); self._h5loc_edit.setVisible(h5)
        if not h5:
            return
        try:
            items = list_h5_datasets(txt)
        except Exception:
            items = []
        if not items:
            return
        keep = self._h5loc_edit.currentText().split("   ")[0].strip()
        self._h5loc_edit.blockSignals(True)
        self._h5loc_edit.clear()
        for name, shape in items:
            self._h5loc_edit.addItem(f"{name}   {tuple(shape)}", name)
        idx = next((i for i, (n, s) in enumerate(items) if len(s) >= 2), 0)
        for i in range(self._h5loc_edit.count()):
            if self._h5loc_edit.itemData(i) == keep:
                idx = i; break
        self._h5loc_edit.setCurrentIndex(idx)
        self._h5loc_edit.blockSignals(False)

    def _on_stack_path_changed(self, txt: str):
        """Show the dataset selector + list datasets when the stack is an HDF5 file."""
        self._stack_files = None
        self._stack_ed.setToolTip("")
        h5 = is_h5(txt) and Path(txt).is_file()
        self._stack_ds_row.setVisible(h5)
        if not h5:
            return
        try:
            items = list_h5_datasets(txt)
        except Exception:
            items = []
        if not items:
            return
        keep = self._stack_ds_combo.currentText().split("   ")[0].strip()
        self._stack_ds_combo.blockSignals(True)
        self._stack_ds_combo.clear()
        for name, shape in items:
            self._stack_ds_combo.addItem(f"{name}   {tuple(shape)}", name)
        # prefer a 3-D dataset (time sequence); keep the prior choice if still present
        idx = next((i for i, (n, s) in enumerate(items) if len(s) >= 3), 0)
        for i in range(self._stack_ds_combo.count()):
            if self._stack_ds_combo.itemData(i) == keep:
                idx = i; break
        self._stack_ds_combo.setCurrentIndex(idx)
        self._stack_ds_combo.blockSignals(False)

    def _load_image(self):
        path = self._img_edit.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "File not found."); return
        try:
            if Path(path).is_dir():
                self._load_image_folder(path)
            else:
                self._load_image_file(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def _load_image_folder(self, folder: str):
        """A folder of single-frame files — one frame per file, ordered like
        the section-2 Stack folder loader."""
        paths = []
        for ext in ("*.tif", "*.tiff", "*.h5", "*.hdf5", "*.ge*"):
            paths.extend(sorted(Path(folder).glob(ext)))
        paths = [str(p) for p in paths]
        if not paths:
            raise ValueError(
                "No image files (*.tif/*.tiff/*.h5/*.hdf5/*.ge*) found in folder.")
        self._img_frames = {"kind": "files", "paths": paths, "n": len(paths)}
        self._update_frame_nav(len(paths))
        self._load_frame(0)

    def _load_image_file(self, path: str):
        data_loc = self._h5loc_edit.currentText().split("   ")[0].strip() or "exchange/data"
        kind, n, extra = self._detect_multiframe(path, data_loc)
        if kind is not None and n > 1:
            if kind == "array_tiff":
                import tifffile
                self._img_frames = {"kind": "array", "data": np.asarray(tifffile.imread(path)), "n": n}
            elif kind == "h5":
                self._img_frames = {"kind": "h5", "path": path, "dataset": extra, "n": n}
            elif kind == "ge":
                self._img_frames = {"kind": "ge", "path": path, "n": n}
            self._update_frame_nav(n)
            self._load_frame(0)
            return
        self._img_frames = None
        self._update_frame_nav(1)
        import tifffile
        raw = tifffile.imread(path) if Path(path).suffix.lower() in (".tif", ".tiff") \
              else _load_image(path, data_loc=data_loc)
        self._set_loaded_image(raw)

    def _detect_multiframe(self, path: str, h5_dataset: Optional[str] = None):
        """Return ``(kind, n, extra)`` for a single file that holds more than
        one frame — a multi-page TIFF, a 3-D HDF5 dataset, or a multi-frame
        raw ``.geN`` file — or ``(None, 1, None)`` for a plain 2-D image.
        Only peeks at shape/page-count/file-size, never loads pixel data."""
        ext = Path(path).suffix.lower()
        if ext in (".tif", ".tiff"):
            import tifffile
            # tifffile may pack a small (N, H, W) stack into a single strip
            # under ONE page — len(tf.pages) then reports 1 even for N>1 — so
            # the series shape (metadata only, no pixel data read) is the
            # reliable frame count, not the page count.
            with tifffile.TiffFile(path) as tf:
                shape = tf.series[0].shape if tf.series else tf.pages[0].shape
            return ("array_tiff", int(shape[0]), None) if len(shape) >= 3 else (None, 1, None)
        if is_h5(path):
            import h5py
            dloc = h5_dataset or "exchange/data"
            try:
                with h5py.File(path, "r") as f:
                    if dloc not in f:
                        return (None, 1, None)
                    ndim, shape0 = f[dloc].ndim, f[dloc].shape[0]
            except Exception:
                return (None, 1, None)
            return ("h5", int(shape0), dloc) if ndim >= 3 else (None, 1, None)
        if ".ge" in Path(path).name.lower():
            n_bytes = Path(path).stat().st_size - 8192
            if n_bytes <= 0:
                return (None, 1, None)
            n_pixels = n_bytes // 2   # uint16
            for side in (2048, 4096, 1024, 512):
                if n_pixels >= side * side and n_pixels % (side * side) == 0:
                    return ("ge", n_pixels // (side * side), None)
            return (None, 1, None)
        return (None, 1, None)

    def _get_frame_array(self, idx: int) -> np.ndarray:
        """The raw (pre-``astype(float32)``-in-``_set_loaded_image``) 2-D
        array for frame ``idx`` of the current multi-frame Image source."""
        meta = self._img_frames
        kind = meta["kind"]
        if kind == "files":
            return _load_image(meta["paths"][idx])
        if kind == "array":
            return meta["data"][idx]
        if kind == "h5":
            return _load_image(meta["path"], data_loc=meta["dataset"], frame=idx)
        if kind == "ge":
            return _load_image(meta["path"], frame=idx)
        raise ValueError(f"unknown frame source kind {kind!r}")

    def _update_frame_nav(self, n: int):
        multi = n > 1
        self._frame_nav_row.setVisible(multi)
        self._frame_prev_btn.setEnabled(multi)
        self._frame_next_btn.setEnabled(multi)
        self._frame_spin.blockSignals(True)
        self._frame_spin.setRange(1, max(1, n))
        self._frame_spin.setValue(1)
        self._frame_spin.blockSignals(False)
        self._frame_count_lbl.setText(f"/ {n}")
        # Projection only means something across >1 frame — otherwise force
        # (and lock) "Current frame" so the threshold step is unambiguous.
        self._thresh_proj_combo.setEnabled(multi)
        if not multi:
            self._thresh_proj_combo.setCurrentText("Current frame")

    def _on_frame_spin_changed(self, val: int):
        if self._img_frames is None:
            return
        self._load_frame(val - 1)

    def _load_frame(self, idx: int):
        if self._img_frames is None:
            return
        n = self._img_frames["n"]
        idx = max(0, min(idx, n - 1))
        self._img_frame_idx = idx
        raw = self._get_frame_array(idx)
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(idx + 1)
        self._frame_spin.blockSignals(False)
        self._set_loaded_image(raw)

    def _set_loaded_image(self, raw: np.ndarray):
        self._orig_dtype = raw.dtype
        self._image = raw.astype(np.float32)
        sentinel = _SENTINELS.get(np.dtype(raw.dtype).name)
        if sentinel is not None:
            self._upper.setValue(float(sentinel))
        self._viewer.set_raw_frame(self._image, self._disp_codes)
        n = self._img_frames["n"] if self._img_frames is not None else 1
        frame_info = f"  frame {self._img_frame_idx + 1}/{n}" if n > 1 else ""
        self._stat_lbl.setText(
            f"Loaded: {self._image.shape[1]}×{self._image.shape[0]}  dtype={raw.dtype}  "
            f"range [{raw.min():.0f}, {raw.max():.0f}]{frame_info}")

    def _threshold_source_image(self) -> np.ndarray:
        """The 2-D image the threshold step (section 1) is computed against:
        a projection across every frame of the Image source when a folder/
        multi-frame source is loaded AND a projection is selected, otherwise
        just the currently-displayed frame (``self._image``)."""
        proj = self._thresh_proj_combo.currentText()
        if self._img_frames is None or proj == "Current frame" or self._img_frames["n"] <= 1:
            return self._image
        n = self._img_frames["n"]
        self._stat_prog.setText(f"Building {proj.lower()} projection across {n} frames…")
        QtWidgets.QApplication.processEvents()
        if proj == "Max":
            acc = np.asarray(self._get_frame_array(0), dtype=np.float32).copy()
            for i in range(1, n):
                np.maximum(acc, self._get_frame_array(i), out=acc)
            return acc
        acc = np.zeros(self._image.shape, dtype=np.float64)
        for i in range(n):
            acc += self._get_frame_array(i)
        if proj == "Average":
            acc /= n
        return acc.astype(np.float32)

    def _on_show_overlay(self, visible: bool):
        self._viewer.set_overlay_visible(visible)

    def _compute(self):
        if self._image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load an image first."); return
        if self._mask_worker and self._mask_worker.isRunning():
            return
        lower = self._lower.value(); upper = self._upper.value()
        thresh_img = self._threshold_source_image()
        thresh_mask = np.zeros(thresh_img.shape, dtype=np.uint8)
        if lower > -1e9:
            thresh_mask |= (thresh_img <= lower).astype(np.uint8)
        if upper > 0:
            thresh_mask |= (thresh_img > upper).astype(np.uint8)
        self._thresh_mask = thresh_mask

        methods = {}
        if self._spatial_check.isChecked() or self._temporal_check.isChecked():
            methods["stat"] = {
                "spatial": self._spatial_check.isChecked(),
                "temporal": self._temporal_check.isChecked(),
                "k_sigma": self._k_sigma.value(),
                "hot_factor": self._hot_factor.value(),
                "dead_factor": self._dead_factor.value(),
                "frozen_frac": self._frozen_frac.value(),
                "overflow": upper if upper > 0 else None,
            }
        if self._spike_check.isChecked():
            methods["spike"] = {"n_sigma": self._spike_sigma.value(), "method": "laplacian"}
        if self._cosmic_check.isChecked():
            methods["cosmic_ray"] = {"n_sigma": self._cosmic_sigma.value()}
        if self._azim_check.isChecked() and self._calib_result is not None:
            methods["azimuthal"] = {"n_sigma": self._azim_sigma.value()}
        if self._learn_check.isChecked() and self._calib_result is not None:
            methods["learnable"] = {
                "n_steps": self._learn_steps.value(), "lr": self._learn_lr.value(),
                "sparsity_weight": self._learn_sparsity.value(), "init_weight": 0.9,
            }

        if not methods:
            self._set_mask(thresh_mask); return

        self._stat_prog.setText("Computing mask…")
        stack_paths, stack_hdf5 = self._stack_source()
        self._mask_worker = MaskComputeWorker(
            self._image, thresh_mask, methods,
            stack_paths=stack_paths, stack_hdf5=stack_hdf5,
            calib_result=self._calib_result, parent=self)
        self._mask_worker.progress.connect(self._stat_prog.setText)
        self._mask_worker.finished.connect(self._on_mask_done)
        self._mask_worker.failed.connect(self._on_stat_fail)
        self._mask_worker.start()

    def _stack_source(self):
        """Resolve the stack source → (paths_list, hdf5_tuple).

        A single HDF5 file with a 3-D dataset returns ``([], (path, dataset, stride))``;
        a folder / glob / TIFF returns ``(paths_list, None)`` (each file = one frame)."""
        raw = self._stack_ed.text().strip()
        stride = max(1, self._stride_spin.value())
        if raw and is_h5(raw) and Path(raw).is_file():
            dset = self._stack_ds_combo.currentText().split("   ")[0].strip() or "exchange/data"
            return [], (raw, dset, stride)
        return self._collect_stack_paths(), None

    def _collect_stack_paths(self) -> list:
        stride = max(1, self._stride_spin.value())
        if self._stack_files:
            return self._stack_files[::stride]
        raw = self._stack_ed.text().strip()
        if not raw:
            return []
        p = Path(raw)
        if p.is_dir():
            paths = []
            for ext in ("*.tif", "*.tiff", "*.h5", "*.hdf5", "*.ge*"):
                paths.extend(sorted(p.glob(ext)))
        elif "*" in raw or "?" in raw:
            paths = sorted(Path(f) for f in _glob.glob(raw))
        elif p.is_file():
            paths = [p]
        else:
            return []
        return [str(x) for x in paths[::stride]]

    def _on_mask_done(self, mask: np.ndarray):
        self._set_mask(mask)

    def _on_stat_fail(self, msg: str):
        self._stat_prog.setText("Failed — check parameters.")
        show_error(self, "Mask error", f"Statistical outlier mask failed:\n\n{msg}")

    def _set_mask(self, mask):
        """Set the *computed* mask (threshold/stat/loaded), apply pixel dilation,
        and refresh the final mask."""
        if mask is None:
            self._computed_mask = None
        else:
            m = mask.astype(bool)
            n = self._dilation_spin.value()
            if n > 0:
                from scipy.ndimage import binary_dilation
                m = binary_dilation(m, structure=np.ones((3, 3), dtype=bool), iterations=n)
            self._computed_mask = m
        self._emit_final()

    def _emit_final(self):
        """Combine computed + hand-drawn masks, update overlay/stats, emit maskReady."""
        parts = [m for m in (self._computed_mask, self._drawn_mask) if m is not None]
        if not parts:
            self._mask = None
            self._viewer.clear_overlay()
            self._stat_lbl.setText("No mask computed.")
            return
        final = parts[0].copy()
        for m in parts[1:]:
            final |= m
        final = final.astype(np.uint8)
        self._mask = final
        n_bad = int(final.sum()); n_tot = final.size; pct = 100 * n_bad / n_tot
        drawn_n = int(self._drawn_mask.sum()) if self._drawn_mask is not None else 0
        self._stat_lbl.setText(
            f"Bad pixels: {n_bad:,} / {n_tot:,} ({pct:.2f}%)"
            + (f"   (incl. {drawn_n:,} hand-drawn)" if drawn_n else "")
            + f"\nGood pixels: {n_tot - n_bad:,} ({100 - pct:.2f}%)")
        self._paint_overlay()
        self._save_btn.setEnabled(True)
        self._log_project_btn.setEnabled(True)
        self.maskReady.emit(final)

    # ── Hand-drawn shape masks ────────────────────────────────────

    _SHAPE_PEN = None  # set lazily

    def _on_draw_toggled(self, on: bool):
        if on and self._image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load an image first.")
            self._draw_check.setChecked(False); return
        self._draw_bar.setVisible(on)
        if not on:
            if self._point_btn.isChecked():
                self._point_btn.setChecked(False)
            if self._freeform_btn.isChecked():
                self._freeform_btn.setChecked(False)  # triggers _cancel_freeform via toggled signal

    def _pen(self):
        return pg.mkPen("#ff5a5a", width=2)

    def _ipen(self):
        return pg.mkPen("#5ab0ff", width=2, style=QtCore.Qt.DashLine)

    def _default_geom(self):
        """Reasonable starting (pos, size) for a new ROI, in image (x=Y, y=Z) coords."""
        NZ, NY = self._disp_shape()
        sx, sy = NY * 0.25, NZ * 0.25
        return (NY * 0.5 - sx / 2, NZ * 0.5 - sy / 2), (sx, sy)

    def _register(self, roi):
        roi.setZValue(20)
        roi.removable = True
        roi.sigRemoveRequested.connect(self._remove_roi)
        self._viewer._iv.addItem(roi)
        return roi

    def _add_rect(self):
        if not self._guard_draw(): return
        pos, size = self._default_geom()
        roi = pg.RectROI(pos, size, pen=self._pen(), rotatable=True)
        self._shapes.append({"kind": "shape", "roi": self._register(roi)})

    def _add_oval(self):
        if not self._guard_draw(): return
        pos, size = self._default_geom()
        roi = pg.EllipseROI(pos, size, pen=self._pen(), rotatable=True)
        self._shapes.append({"kind": "shape", "roi": self._register(roi)})

    def _add_circle(self):
        if not self._guard_draw(): return
        NZ, NY = self._disp_shape()
        d = min(NY, NZ) * 0.25
        roi = pg.CircleROI((NY * 0.5 - d / 2, NZ * 0.5 - d / 2), (d, d), pen=self._pen())
        self._shapes.append({"kind": "shape", "roi": self._register(roi)})

    def _add_polygon(self):
        if not self._guard_draw(): return
        NZ, NY = self._disp_shape()
        cx, cy, r = NY * 0.5, NZ * 0.5, min(NY, NZ) * 0.18
        pts = [[cx + r, cy], [cx, cy + r], [cx - r, cy], [cx, cy - r]]
        roi = pg.PolyLineROI(pts, closed=True, pen=self._pen())
        self._shapes.append({"kind": "shape", "roi": self._register(roi)})

    def _add_annulus(self):
        if not self._guard_draw(): return
        NZ, NY = self._disp_shape()
        do = min(NY, NZ) * 0.4; di = do * 0.5
        cx, cy = NY * 0.5, NZ * 0.5
        outer = pg.EllipseROI((cx - do / 2, cy - do / 2), (do, do), pen=self._pen(), rotatable=True)
        inner = pg.EllipseROI((cx - di / 2, cy - di / 2), (di, di), pen=self._ipen(), rotatable=True)
        self._register(outer); self._register(inner)
        self._shapes.append({"kind": "annulus", "outer": outer, "inner": inner})

    def _guard_draw(self) -> bool:
        if self._image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load an image first.")
            return False
        return True

    def _remove_roi(self, roi):
        for item in list(self._shapes):
            if item["kind"] == "shape" and item["roi"] is roi:
                self._viewer._iv.removeItem(roi); self._shapes.remove(item); return
            if item["kind"] == "annulus" and roi in (item["outer"], item["inner"]):
                self._viewer._iv.removeItem(item["outer"]); self._viewer._iv.removeItem(item["inner"])
                self._shapes.remove(item); return

    # ── Single-pixel point picking ────────────────────────────────

    def _toggle_point_mode(self, on: bool):
        if on and not self._guard_draw():
            self._point_btn.setChecked(False); return
        self._point_mode = on
        self._ensure_click_proxy()

    # ── Freeform click-polygon ────────────────────────────────────

    def _toggle_freeform_mode(self, on: bool):
        if on and not self._guard_draw():
            self._freeform_btn.setChecked(False); return
        self._freeform_mode = on
        self._close_shape_btn.setVisible(on)
        if not on:
            self._cancel_freeform()
        else:
            self._ensure_click_proxy()

    def _ensure_click_proxy(self):
        if self._click_proxy is None:
            self._click_proxy = pg.SignalProxy(
                self._viewer._iv.scene.sigMouseClicked, rateLimit=60,
                slot=self._on_scene_click)

    def _update_freeform_preview(self):
        """Redraw the live edge-lines and vertex dots from _freeform_pts."""
        pts = self._freeform_pts
        if not pts:
            return

        # Build coordinate arrays for the preview: vertices + closing segment back to start
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]

        if self._freeform_line is None:
            self._freeform_line = pg.PlotDataItem(
                xs, ys,
                pen=pg.mkPen("#ffcc00", width=1.5),
                symbol=None)
            self._freeform_line.setZValue(22)
            self._viewer._iv.addItem(self._freeform_line)
        else:
            self._freeform_line.setData(xs, ys)

        vx = [p[0] for p in pts]
        vy = [p[1] for p in pts]
        if self._freeform_vdots is None:
            self._freeform_vdots = pg.ScatterPlotItem(
                vx, vy, symbol="o", size=7,
                pen=pg.mkPen("#ffcc00", width=1),
                brush=pg.mkBrush("#ffcc0088"))
            self._freeform_vdots.setZValue(23)
            self._viewer._iv.addItem(self._freeform_vdots)
        else:
            self._freeform_vdots.setData(vx, vy)

    def _close_freeform(self):
        """Seal the polygon: turn accumulated vertices into a PolyLineROI."""
        pts = self._freeform_pts
        if len(pts) < 3:
            QtWidgets.QMessageBox.information(
                self, "Too few vertices",
                "Place at least 3 vertices before closing the shape.")
            return
        # Remove live preview items
        self._cancel_freeform(keep_roi=True)
        # Create a proper closeable PolyLineROI from the accumulated vertices
        roi = pg.PolyLineROI(list(pts), closed=True, pen=self._pen())
        self._shapes.append({"kind": "shape", "roi": self._register(roi)})
        # Reset freeform state but leave the button toggled off
        self._freeform_mode = False
        self._freeform_btn.setChecked(False)
        self._close_shape_btn.setVisible(False)

    def _cancel_freeform(self, keep_roi=False):
        """Remove live preview graphics and reset vertex list."""
        if self._freeform_line is not None:
            self._viewer._iv.removeItem(self._freeform_line)
            self._freeform_line = None
        if self._freeform_vdots is not None:
            self._viewer._iv.removeItem(self._freeform_vdots)
            self._freeform_vdots = None
        self._freeform_pts = []

    def _on_scene_click(self, evt):
        event = evt[0]
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self._image is None:
            return

        imgitem = self._viewer._iv.getImageItem()
        p = imgitem.mapFromScene(event.scenePos())
        x, y = float(p.x()), float(p.y())
        NZ, NY = self._disp_shape()

        # ── Freeform polygon mode ──
        if self._freeform_mode:
            if not (0 <= y < NZ and 0 <= x < NY):
                return
            if event.double():
                # Double-click: close without adding another vertex
                # (the preceding single-click already added the final vertex)
                if len(self._freeform_pts) >= 3:
                    self._close_freeform()
                return
            self._freeform_pts.append((x, y))
            self._update_freeform_preview()
            return

        # ── Single-pixel point mode ──
        if self._point_mode:
            col, row = int(x), int(y)
            if 0 <= row < NZ and 0 <= col < NY:
                self._points.append((col, row))
                dot = pg.ScatterPlotItem([col], [row], symbol="s", size=6,
                                         pen=pg.mkPen(None), brush=pg.mkBrush("#ff5a5a"))
                dot.setZValue(21)
                self._viewer._iv.addItem(dot); self._point_items.append(dot)

    # ── Rasterise shapes → mask ───────────────────────────────────

    def _raster_roi(self, roi) -> np.ndarray:
        """Boolean (NZ, NY) mask of pixels inside a pyqtgraph ROI (any shape)."""
        NZ, NY = self._disp_shape()
        imgitem = self._viewer._iv.getImageItem()
        path = imgitem.mapFromScene(roi.mapToScene(roi.shape()))
        qimg = QtGui.QImage(NY, NZ, QtGui.QImage.Format_Grayscale8)
        qimg.fill(0)
        painter = QtGui.QPainter(qimg)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255))
        painter.drawPath(path)
        painter.end()
        bpl = qimg.bytesPerLine()
        ptr = qimg.constBits(); ptr.setsize(NZ * bpl)
        arr = np.frombuffer(ptr, np.uint8).reshape(NZ, bpl)[:, :NY]
        return arr > 127

    def _apply_shapes(self):
        if self._image is None:
            return
        if not self._shapes and not self._points:
            QtWidgets.QMessageBox.information(self, "No shapes", "Draw a shape or pick points first.")
            return
        NZ, NY = self._disp_shape()
        drawn = np.zeros((NZ, NY), dtype=bool)
        for item in self._shapes:
            if item["kind"] == "shape":
                drawn |= self._raster_roi(item["roi"])
            else:  # annulus: region between outer and inner
                drawn |= (self._raster_roi(item["outer"]) & ~self._raster_roi(item["inner"]))
        for col, row in self._points:
            drawn[row, col] = True
        # Rasterised against what is on screen; the mask itself is raw-frame,
        # so undo the display transform. Reversing the code order inverts it
        # (each op is self-inverse) — the same idiom workers.py uses to bring
        # an azimuthal-clip mask back to raw.
        if self._disp_codes:
            drawn = _apply_im_trans(drawn.astype(np.uint8),
                                    tuple(reversed(self._disp_codes))).astype(bool)
        self._drawn_mask = drawn
        self._emit_final()

    def _clear_shapes(self):
        # Cancel any in-progress freeform polygon first
        self._cancel_freeform()
        self._freeform_mode = False
        self._freeform_btn.setChecked(False)
        self._close_shape_btn.setVisible(False)
        for item in self._shapes:
            if item["kind"] == "shape":
                self._viewer._iv.removeItem(item["roi"])
            else:
                self._viewer._iv.removeItem(item["outer"]); self._viewer._iv.removeItem(item["inner"])
        self._shapes.clear()
        for dot in self._point_items:
            self._viewer._iv.removeItem(dot)
        self._point_items.clear(); self._points.clear()
        self._drawn_mask = None
        self._emit_final()

    def _save(self):
        if self._mask is None: return
        path = self._save_edit.text().strip()
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Mask", "mask.tif", "TIFF (*.tif)")
        if not path: return
        try:
            import tifffile
            tifffile.imwrite(str(path), self._mask)
            QtWidgets.QMessageBox.information(self, "Saved", f"Mask saved:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

    def _log_to_project(self):
        """Append this mask (full parameters + the resulting compressed
        array) as a new FAIR-provenance attempt under a currently-open
        project's ``/analysis/mask`` — see ``project.append_mask_attempt``.
        Unlike Calibrate/Batch-Integrate there's no single "run finished"
        moment to auto-log from (a mask can come from Compute, Load, hand-
        drawn shapes, or any combination), so this is an explicit, user-
        triggered action, same click-when-ready contract as ``_save``."""
        if self._mask is None:
            return
        if not self._project_ctx or not self._project_ctx.path:
            QtWidgets.QMessageBox.warning(
                self, "No project open",
                "Open or create a project first (File ▸ Open Project…) to log this mask.")
            return
        cfg = {"fields": widgets_to_dict(self._state_widgets())}
        loader_state = {
            "path": self._img_edit.text().strip() or None,
            "h5_dataset": self._h5loc_edit.currentText().strip() if is_h5(self._img_edit.text().strip()) else None,
            "stack_files": self._stack_files,
            "img_frame_idx": self._img_frame_idx if self._img_frames is not None else None,
        }
        try:
            ref = project.append_mask_attempt(
                self._project_ctx.path, cfg=cfg, mask=self._mask, loader_state=loader_state,
                extra={"active_profile": settings.active_profile()})
            QtWidgets.QMessageBox.information(self, "Logged to project", f"Mask attempt logged:\n{ref}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Log to project failed", str(e))

    def apply_project_mask(self, meta: dict, mask_array: Optional[np.ndarray]) -> None:
        """Populate this tab from a recorded ``/analysis/mask`` attempt (see
        ``project.append_mask_attempt``) — mirrors
        ``CalibrationTab.apply_project_calibration``/
        ``BatchTab.apply_project_integration``'s "load this analysis into
        the tab, including its result" contract. Called after File ▸ Open
        Project… when the user opts to restore a mask attempt."""
        cfg = meta.get("cfg") or {}
        fields = cfg.get("fields") or {}
        loader_state = meta.get("loader_state") or {}
        img_path = loader_state.get("path")
        if img_path and Path(img_path).exists():
            self._img_edit.setText(img_path)
            self._load_image()
            frame_idx = loader_state.get("img_frame_idx")
            if frame_idx is not None and self._img_frames is not None:
                self._load_frame(int(frame_idx))
        apply_dict_to_widgets(self._state_widgets(), fields)
        self._stack_files = list(loader_state.get("stack_files") or []) or None
        if mask_array is not None:
            # The embedded array is already the final, fully-processed mask
            # (dilation etc. already applied when it was first computed) —
            # set it directly as the "computed" half rather than routing
            # through _set_mask, which would dilate it a second time.
            self._computed_mask = mask_array.astype(bool)
            self._drawn_mask = None
            self._emit_final()

    def _load_existing_mask(self):
        path = self._load_mask_edit.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "File not found."); return
        try:
            import tifffile
            raw = tifffile.imread(path)
            self._set_mask((raw != 0).astype(np.uint8))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def get_mask(self) -> Optional[np.ndarray]:
        return self._mask

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def _state_widgets(self) -> dict:
        return {
            "img_path": self._img_edit,
            "h5_dataset": self._h5loc_edit,
            "lower": self._lower,
            "upper": self._upper,
            "thresh_proj_combo": self._thresh_proj_combo,
            "spatial_check": self._spatial_check,
            "temporal_check": self._temporal_check,
            "k_sigma": self._k_sigma,
            "hot_factor": self._hot_factor,
            "dead_factor": self._dead_factor,
            "frozen_frac": self._frozen_frac,
            "stack_ed": self._stack_ed,
            "stack_ds_combo": self._stack_ds_combo,
            "stride_spin": self._stride_spin,
            "spike_check": self._spike_check,
            "spike_sigma": self._spike_sigma,
            "cosmic_check": self._cosmic_check,
            "cosmic_sigma": self._cosmic_sigma,
            "azim_check": self._azim_check,
            "azim_sigma": self._azim_sigma,
            "learn_check": self._learn_check,
            "learn_steps": self._learn_steps,
            "learn_lr": self._learn_lr,
            "learn_sparsity": self._learn_sparsity,
            "dilation_spin": self._dilation_spin,
            "save_edit": self._save_edit,
            "load_mask_edit": self._load_mask_edit,
            "show_overlay_check": self._show_overlay_check,
            "draw_check": self._draw_check,
        }

    def get_state(self, sidecar_stem: Optional[str] = None) -> dict:
        """``sidecar_stem`` (if given) is the state file's path without its
        extension. A mask that's been drawn/computed but not yet exported via
        the Save button is auto-exported to ``<sidecar_stem>_mask.tif`` so it
        isn't silently lost."""
        state = {"fields": widgets_to_dict(self._state_widgets()),
                  "viewer": self._viewer.display_state()}
        if self._stack_files:
            state["stack_files"] = self._stack_files
        if self._img_frames is not None:
            state["img_frame_idx"] = self._img_frame_idx
        if self._mask is not None and sidecar_stem:
            try:
                import tifffile
                tifffile.imwrite(f"{sidecar_stem}_mask.tif", self._mask)
            except Exception:
                pass
        return state

    def set_state(self, state: dict, sidecar_stem: Optional[str] = None) -> None:
        # Load the image *before* applying the saved fields — `_load_image()`
        # auto-picks a dtype-based threshold sentinel, which must not clobber
        # an explicitly saved lower/upper threshold applied afterwards.
        fields = state.get("fields", {})
        img_path = fields.get("img_path")
        if img_path:
            self._img_edit.setText(img_path)
            if Path(img_path).exists():
                self._load_image()
                frame_idx = state.get("img_frame_idx")
                if frame_idx is not None and self._img_frames is not None:
                    self._load_frame(int(frame_idx))
        apply_dict_to_widgets(self._state_widgets(), fields)
        self._viewer.set_display_state(state.get("viewer"))
        self._stack_files = list(state["stack_files"]) if state.get("stack_files") else None
        if sidecar_stem:
            mask_path = Path(f"{sidecar_stem}_mask.tif")
            if mask_path.is_file():
                try:
                    import tifffile
                    raw = tifffile.imread(str(mask_path))
                    self._set_mask((raw != 0).astype(np.uint8))
                except Exception:
                    pass
