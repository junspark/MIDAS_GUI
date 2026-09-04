"""Shared display widgets.

ImageViewer / PickableImageViewer / ProfileViewer / LogPanel are ported verbatim
from midas_workflow_gui_v3.py (frozen template).  ResidualBarChart, DistortionTable
and CorrectionFlagsWidget are new Phase-1 additions.

pyqtgraph rules (see context/design_rules.md) preserved:
  - store pg.SignalProxy as instance var      (else GC'd, hover dies)
  - setColorMap() not setLookupTable()        (else reset on setImage)
  - setXRange() not autoRange(axes=)          (else TypeError on this pg version)
  - ring markers redrawn LAST inside _replot  (else don't render)
  - int(x) floor for pixel indexing           (not int(x+0.5))
"""
from __future__ import annotations

import math
import threading
from collections import deque
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from midas_gui.constants import COLORMAPS, DISTORTION_NAMES, DEFAULT_COLORMAP, DEVICES
from midas_gui.dialogs import show_error, BrowseFilesDialog
from midas_gui.helpers import fit_circle_algebraic
from midas_gui.sim_detector import DEFAULT_CHANNEL_NAME as _SIM_CHANNEL_NAME

# Default colormap: the configured one if it's a known option, else the first.
_DEFAULT_CMAP = DEFAULT_COLORMAP if DEFAULT_COLORMAP in COLORMAPS else COLORMAPS[0]
from midas_gui.helpers import (_NoScrollSpinBox, _NoScrollDoubleSpinBox, _fspin, _twocol,
                               _browse, is_h5, list_h5_datasets, _NoScrollComboBox,
                               _load_image, _collect_frame_paths, apply_field_corrections,
                               new_temp_h5_path, save_stack_h5, detect_geometry_from_path,
                               source_kind, display_text_for_paths, _apply_im_trans,
                               is_dark_like_name)
from midas_gui import style as S


def _mono_font(size: int) -> QtGui.QFont:
    """A fixed-width font at ``size`` pt using the real-family stack in style.py.

    Naming concrete families (Menlo/Consolas/…) instead of ``QFont("Monospace")``
    avoids Qt scanning and building font-family aliases at startup (the
    "qt.qpa.fonts: Populating font family aliases" warning) on macOS/Windows."""
    f = QtGui.QFont()
    try:
        f.setFamilies(S.MONO_FAMILIES)
    except Exception:
        f.setFamily(S.MONO_FAMILIES[0])
    f.setStyleHint(QtGui.QFont.Monospace)
    f.setPointSize(size)
    return f


def _resolve_cmap(name):
    """Return a pyqtgraph ColorMap for ``name`` — never None.

    The GUI's colormaps (hot / viridis / inferno / plasma / turbo / gray) are
    matplotlib maps: ``pg.colormap.get(name)`` returns None when matplotlib is not
    installed, and passing that None into pyqtgraph crashes viewer construction (see
    the Linux/Windows fresh-env bug). Fall back through matplotlib, then a
    pyqtgraph-native map, then a plain grayscale ramp, so a missing matplotlib can
    never take the tabs down."""
    for attempt in (
        lambda: pg.colormap.get(name),
        lambda: pg.colormap.get(name, source="matplotlib"),
        lambda: pg.colormap.getFromMatplotlib(name),
        lambda: pg.colormap.get("CET-L9"),   # native — no matplotlib needed
    ):
        try:
            cm = attempt()
        except Exception:
            cm = None
        if cm is not None:
            return cm
    return pg.ColorMap([0.0, 1.0], [(0, 0, 0, 255), (255, 255, 255, 255)])


# ═════════════════════════════════════════════════════════════════════════════
#  ImageViewer
# ═════════════════════════════════════════════════════════════════════════════

class ImageViewer(QtWidgets.QWidget):
    """pyqtgraph image viewer with log scale, colormap, vmin/vmax, crosshair,
    pixel-value status bar, and a mask overlay."""

    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        pg.setConfigOptions(background="k", foreground="w")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar
        bar = QtWidgets.QHBoxLayout()
        if title:
            bar.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))
        self._log = QtWidgets.QCheckBox("Log")
        self._log.setChecked(True)
        self._log.toggled.connect(self._on_log_toggled)
        bar.addWidget(self._log)
        bar.addWidget(QtWidgets.QLabel("cmap:"))
        self._cmap = _NoScrollComboBox()
        self._cmap.addItems(COLORMAPS)
        self._cmap.setCurrentText(_DEFAULT_CMAP)
        self._cmap.currentTextChanged.connect(self._set_cmap)
        self._cmap.setFixedWidth(90)
        bar.addWidget(self._cmap)
        bar.addWidget(QtWidgets.QLabel("vmin%:"))
        self._vmin = _NoScrollSpinBox()
        self._vmin.setRange(0, 99); self._vmin.setValue(30); self._vmin.setFixedWidth(45)
        self._vmin.valueChanged.connect(self._on_percentile_changed)
        bar.addWidget(self._vmin)
        bar.addWidget(QtWidgets.QLabel("vmax%:"))
        self._vmax = _NoScrollSpinBox()
        self._vmax.setRange(1, 100); self._vmax.setValue(99); self._vmax.setFixedWidth(45)
        self._vmax.valueChanged.connect(self._on_percentile_changed)
        bar.addWidget(self._vmax)
        bar.addStretch(1)
        self._toolbar_layout = bar   # exposed so subclasses can append widgets
        layout.addLayout(bar)

        # Image view
        self._iv = pg.ImageView(view=pg.PlotItem())
        self._iv.ui.roiBtn.hide(); self._iv.ui.menuBtn.hide()
        vb = self._iv.getView().getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.setMouseMode(pg.ViewBox.PanMode)
        # pg.ImageView.__init__ calls view.invertY() internally, putting
        # pixel row 0 at the top of the screen. MIDAS convention places
        # detector (0,0) at the bottom-left, so the on-screen image matches
        # the physical world view of the detector looking downstream from
        # the sample along the beam. Override that default here.
        vb.invertY(False)
        layout.addWidget(self._iv, stretch=1)

        # Crosshair
        self._vl = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("y", width=1))
        self._hl = pg.InfiniteLine(angle=0,  movable=False, pen=pg.mkPen("y", width=1))
        # ignoreBounds: without it, this item's position (which follows the
        # mouse every frame) feeds pyqtgraph's auto-range bounds calculation,
        # so once continuous auto-range is enabled (e.g. via the "A" button)
        # the view range keeps re-fitting itself to wherever the cursor is.
        self._iv.addItem(self._vl, ignoreBounds=True)
        self._iv.addItem(self._hl, ignoreBounds=True)
        self._mouse_proxy = pg.SignalProxy(
            self._iv.scene.sigMouseMoved, rateLimit=60, slot=self._mouse)

        # Overlay (for mask)
        self._overlay = pg.ImageItem()
        self._overlay.setZValue(10)
        self._iv.addItem(self._overlay)

        # Bottom status bar — pixel coordinates and raw value on hover
        self._coord_bar = QtWidgets.QLabel("Move cursor over image to inspect pixel values")
        self._coord_bar.setStyleSheet(
            f"color:#dddddd; background:#1a1a1a; font-family:{S.MONO_CSS};"
            "font-size:12px; padding:2px 6px; border-top:1px solid #444;")
        layout.addWidget(self._coord_bar)

        self._data: Optional[np.ndarray] = None
        self._manual_levels: Optional[tuple] = None
        self._manual_hist_range: Optional[tuple] = None
        self._suspend_level_track = False
        self._suspend_hist_range_track = False
        self._iv.getHistogramWidget().sigLevelsChanged.connect(self._on_hist_levels_changed)
        self._iv.getHistogramWidget().item.vb.sigRangeChanged.connect(self._on_hist_range_changed)
        self._set_cmap(_DEFAULT_CMAP)

    def set_image(self, data: np.ndarray, autorange: bool = True, reset_levels: bool = True):
        """`reset_levels`: drop any manual color-scale/histogram-zoom window and
        go back to the vmin%/vmax% percentile auto-levels. Pass False for a
        live-streaming frame update, where the window should stay put across
        incoming frames until the user chooses to move it."""
        self._data = data.astype(np.float32)
        if reset_levels:
            self._manual_levels = None
            self._manual_hist_range = None
        self._redisplay()
        self._apply_view_limits(data.shape[1], data.shape[0])
        if autorange:
            self._iv.getView().getViewBox().autoRange()
        self._coord_bar.setText(
            f"Image {data.shape[1]}×{data.shape[0]} px  |  "
            "Move cursor over image to inspect pixel values")

    def set_raw_frame(self, raw_frame: np.ndarray, im_trans, *,
                       autorange: bool = True, reset_levels: bool = True) -> np.ndarray:
        """Display ``raw_frame`` after applying ``im_trans`` (the same
        flip/transpose codes — 1=flipY, 2=flipZ, 3=transpose, see
        ``helpers._apply_im_trans`` — a calibration's BC_y/BC_z were fit
        under) and return the transformed array.

        This is the ONE place a caller that has both a raw detector frame
        and calibration geometry should go through, instead of calling
        ``_apply_im_trans`` itself and then ``set_image``. Every BC-based
        overlay (Rmin/Rmax circles, bin grids, lab-frame axes, calibration
        rings) is computed in the flipped frame's coordinate system, so a
        caller that forgets this step — as ``tab_batch.py``'s Detector view
        once did — silently shows the overlay offset from the real
        diffraction rings underneath. mpe_wf_saxs_waxs hit this same class
        of bug (see its ``apply_image_transforms`` docstring in
        ``ff_asym_qt.py``: "the 'data flipped on cols, dark flipped on
        rows' trap") and fixed it the same way — one shared function every
        display call goes through, rather than each call site remembering
        to flip on its own. Returns the transformed array so callers that
        keep their own reference to "the currently displayed frame" (for
        pixel readback, mask overlays, shape lookups, etc.) can do
        ``self._cur = viewer.set_raw_frame(raw, im_trans)`` in one line.

        Use plain ``set_image`` instead when the array is already in its
        final display orientation (e.g. Mask Builder, which deliberately
        works in raw-frame space throughout — see ``tab_mask.py``) or has
        no associated calibration geometry at all.
        """
        codes = tuple(im_trans or ())
        frame = _apply_im_trans(raw_frame, codes) if codes else raw_frame
        self.set_image(frame, autorange=autorange, reset_levels=reset_levels)
        return frame

    def display_state(self) -> dict:
        """cmap/log/vmin%/vmax% as a plain dict — the one place a caller
        that wants to persist "how this viewer is displayed" (e.g. a
        tab's project-state save) should read from, instead of reaching
        into ``_cmap``/``_log``/``_vmin``/``_vmax`` directly."""
        return {"cmap": self._cmap.currentText(), "log": self._log.isChecked(),
                "vmin": self._vmin.value(), "vmax": self._vmax.value()}

    def set_display_state(self, state: Optional[dict]) -> None:
        """Inverse of :meth:`display_state`. Restoring ``log``/``vmin``/
        ``vmax`` alone is enough to take effect on the next
        ``set_image``/``set_raw_frame`` (``_redisplay`` reads their
        current values fresh every time), but ``cmap`` is different: the
        colormap is only ever applied to the actual pyqtgraph view from
        ``_set_cmap``, which only runs off the combo's own
        ``currentTextChanged`` signal — restoring the combo's index with
        signals blocked (the same convention ``apply_dict_to_widgets``
        uses for every other widget) would leave the dropdown showing the
        saved colormap while the image stayed rendered in whatever
        colormap this viewer was constructed with. So this explicitly
        re-applies it. Missing/unrecognized keys are left untouched."""
        if not state:
            return
        cmap = state.get("cmap")
        if cmap and self._cmap.findText(str(cmap)) >= 0:
            self._cmap.blockSignals(True)
            self._cmap.setCurrentText(str(cmap))
            self._cmap.blockSignals(False)
            self._set_cmap(str(cmap))
        if "log" in state:
            self._log.blockSignals(True)
            self._log.setChecked(bool(state["log"]))
            self._log.blockSignals(False)
        for key, spin in (("vmin", self._vmin), ("vmax", self._vmax)):
            if key in state:
                spin.blockSignals(True)
                spin.setValue(state[key])
                spin.blockSignals(False)
        if self._data is not None:
            self._redisplay()

    def _apply_view_limits(self, w: int, h: int):
        """Bound pan/zoom to a sane region around the image so the user can't
        scroll/zoom out into an empty void or lose the image off-screen."""
        if w <= 0 or h <= 0:
            return
        span = max(w, h)
        pad = span * 0.5
        vb = self._iv.getView().getViewBox()
        vb.setLimits(
            xMin=-pad, xMax=w + pad,
            yMin=-pad, yMax=h + pad,
            minXRange=max(span * 0.01, 2.0),
            minYRange=max(span * 0.01, 2.0),
            maxXRange=w + 2 * pad,
            maxYRange=h + 2 * pad,
        )

    def set_mask_overlay(self, mask: Optional[np.ndarray]):
        if mask is None:
            self._overlay.setImage(np.zeros((1, 1, 4), dtype=np.uint8))
            return
        NZ, NY = mask.shape
        rgba = np.zeros((NY, NZ, 4), dtype=np.uint8)
        bad = mask.T.astype(bool)
        rgba[bad, 0] = 220  # red
        rgba[bad, 3] = 180  # alpha
        self._overlay.setImage(rgba)

    def clear_overlay(self):
        self._overlay.setImage(np.zeros((1, 1, 4), dtype=np.uint8))

    def set_overlay_visible(self, visible: bool):
        self._overlay.setVisible(visible)

    def _redisplay(self):
        if self._data is None:
            return
        d = self._data
        if self._log.isChecked():
            disp = np.log10(np.clip(d, 1e-10, None)).T
        else:
            disp = d.T
        if self._manual_levels is not None:
            lo, hi = self._manual_levels
        else:
            # Exclude exact-zero pixels from the percentile calc — on a
            # mostly-empty canvas (e.g. the Hydra composite's unfilled
            # BigDet background) they'd otherwise dominate and skew the
            # auto-level window. Masked on the raw (pre-log) data, since
            # log10(0) isn't 0. Falls back to the unfiltered set if the
            # whole frame is exactly zero (nothing loaded yet).
            nonzero = d.T != 0
            candidates = disp[np.isfinite(disp) & nonzero]
            fin = candidates if candidates.size else disp[np.isfinite(disp)]
            if fin.size:
                lo = float(np.percentile(fin, self._vmin.value()))
                hi = float(np.percentile(fin, self._vmax.value()))
            else:
                lo, hi = 0.0, 1.0
        # autoRange/autoHistogramRange default to True in pyqtgraph and would reset
        # the view on every redraw; set_image() handles framing explicitly via its
        # `autorange` flag, so the zoom/pan is preserved across frames and re-levels.
        # Levels are likewise pinned to `lo`/`hi` rather than pyqtgraph's autoLevels
        # so a manual histogram drag survives subsequent live frames.
        self._suspend_level_track = True
        self._suspend_hist_range_track = True
        self._iv.setImage(disp.astype(np.float32), autoLevels=False, levels=(lo, hi),
                          autoRange=False, autoHistogramRange=False)
        if self._manual_hist_range is None:
            # Zoom the colorbar/histogram's own axis to the percentile window by
            # default (bad pixels can otherwise stretch the full-range histogram
            # down to an unreadable sliver); a manual zoom/pan overrides this.
            self._iv.getHistogramWidget().item.setHistogramRange(lo, hi, padding=0.1)
        self._suspend_level_track = False
        self._suspend_hist_range_track = False

    def _on_hist_levels_changed(self, *_args):
        """User dragged the histogram LUT region — remember it across future frames."""
        if self._suspend_level_track:
            return
        self._manual_levels = tuple(self._iv.getHistogramWidget().getLevels())

    def _on_hist_range_changed(self, *_args):
        """User zoomed/panned the histogram's own axis — remember it across frames."""
        if self._suspend_hist_range_track:
            return
        self._manual_hist_range = tuple(self._iv.getHistogramWidget().item.getHistogramRange())

    def _on_percentile_changed(self, *_args):
        """User edited vmin%/vmax% — that's an explicit request to go back to auto levels."""
        self._manual_levels = None
        self._manual_hist_range = None
        self._redisplay()

    def _on_log_toggled(self, checked: bool):
        """Log/linear toggled — convert manual levels along with the display,
        so a manually-set color window keeps pointing at the same underlying
        data instead of the same raw numbers reinterpreted in the other scale
        (e.g. a log-space level of 2.0 becoming a linear level of 2.0 instead
        of 100.0)."""
        def convert(lo, hi):
            if checked:   # was linear, now log
                return (float(np.log10(max(lo, 1e-10))), float(np.log10(max(hi, 1e-10))))
            else:         # was log, now linear
                # A manual level can sit anywhere in a histogram view that was
                # itself dragged out to values far beyond any real data (e.g.
                # hi=400) — clamp before 10**x since float exponentiation
                # raises OverflowError rather than saturating at inf.
                lo = min(max(lo, -300.0), 300.0)
                hi = min(max(hi, -300.0), 300.0)
                return (float(10 ** lo), float(10 ** hi))
        if self._manual_levels is not None:
            self._manual_levels = convert(*self._manual_levels)
        # Don't carry a manually-panned/zoomed histogram window through the
        # same nonlinear transform: containment of the levels is preserved
        # mathematically, but they can end up squeezed into an imperceptible
        # sliver of the transformed window. Drop the pin instead so
        # _redisplay() reframes the view tightly around the (converted)
        # levels, same as it does for a fresh percentile-based window.
        self._manual_hist_range = None
        self._redisplay()

    def _set_cmap(self, name: str):
        self._iv.setColorMap(_resolve_cmap(name))

    def _mouse(self, evt):
        pos = evt[0]
        vb = self._iv.getView().getViewBox()
        if self._iv.getView().sceneBoundingRect().contains(pos):
            mp = vb.mapSceneToView(pos)
            x, y = mp.x(), mp.y()
            self._vl.setPos(x); self._hl.setPos(y)
            if self._data is not None:
                ix, iy = int(x), int(y)   # floor, not round (Bug 6)
                h, w = self._data.shape
                if 0 <= iy < h and 0 <= ix < w:
                    val = self._data[iy, ix]
                    self._coord_bar.setText(
                        f"  x (col) = {ix}    y (row) = {iy}    "
                        f"intensity = {val:.4g}    (image {w}×{h} px)")


# ═════════════════════════════════════════════════════════════════════════════
#  PickableImageViewer
# ═════════════════════════════════════════════════════════════════════════════

class PickableImageViewer(ImageViewer):
    """ImageViewer + beam-centre pick tools.

    Pick BC   — single click sets beam centre (bcPicked signal).
    Pick Ring — 3+ clicks; algebraic circle fit estimates BC (ringFitBC signal).
    """
    bcPicked  = QtCore.pyqtSignal(float, float)         # (BC_y, BC_z)
    ringFitBC = QtCore.pyqtSignal(float, float, float)  # (BC_y, BC_z, R_px)
    dspacingPicksChanged = QtCore.pyqtSignal()

    PICK_NONE      = 0
    PICK_BC        = 1
    PICK_RING      = 2
    PICK_DSPACING  = 3

    _DSP_COLORS = ["#e05656", "#56a8e0", "#7fd45a", "#e0c056",
                   "#c066e0", "#e08c40", "#40c8c0", "#c0c0c0"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pick_mode = self.PICK_NONE
        self._ring_pts:        list = []
        self._ring_pt_items:   list = []
        self._ring_fit_item    = None
        self._ring_fit_center  = None
        self._bc_click_item    = None
        self._dsp_pts:         list = []
        self._dsp_pt_items:    list = []

        _BTN = ("QPushButton{padding:2px 8px;border-radius:3px}"
                "QPushButton:checked{background:#2a7fd4;color:white;font-weight:bold}")
        pick_bar = QtWidgets.QHBoxLayout()
        pick_bar.setSpacing(4)

        self._pick_bc_btn = QtWidgets.QPushButton("Pick BC")
        self._pick_bc_btn.setCheckable(True)
        self._pick_bc_btn.setStyleSheet(_BTN)
        self._pick_bc_btn.setToolTip(
            "Click once on the image to set the beam center as the initial seed")
        self._pick_bc_btn.toggled.connect(self._on_pick_bc_toggled)
        pick_bar.addWidget(self._pick_bc_btn)

        self._pick_ring_btn = QtWidgets.QPushButton("Pick Ring")
        self._pick_ring_btn.setCheckable(True)
        self._pick_ring_btn.setStyleSheet(_BTN)
        self._pick_ring_btn.setToolTip(
            "Click 3+ points on a ring; algebraic circle fit estimates beam center")
        self._pick_ring_btn.toggled.connect(self._on_pick_ring_toggled)
        pick_bar.addWidget(self._pick_ring_btn)

        self._pick_dsp_btn = QtWidgets.QPushButton("Pick d-spacing pts")
        self._pick_dsp_btn.setCheckable(True)
        self._pick_dsp_btn.setStyleSheet(_BTN)
        self._pick_dsp_btn.setToolTip(
            "Click points on a ring, set Ring # per group, "
            "for manual Bragg's-law geometry fitting")
        self._pick_dsp_btn.toggled.connect(self._on_pick_dsp_toggled)
        pick_bar.addWidget(self._pick_dsp_btn)

        pick_bar.addWidget(QtWidgets.QLabel("Ring #"))
        self._dsp_ring_spin = QtWidgets.QSpinBox()
        self._dsp_ring_spin.setRange(1, 20)
        self._dsp_ring_spin.setValue(1)
        self._dsp_ring_spin.setToolTip("Ring group new d-spacing clicks are added to")
        pick_bar.addWidget(self._dsp_ring_spin)

        self._undo_btn = QtWidgets.QPushButton("Undo")
        self._undo_btn.setEnabled(False)
        self._undo_btn.setToolTip("Remove last ring point")
        self._undo_btn.clicked.connect(self._undo_ring_point)
        pick_bar.addWidget(self._undo_btn)

        self._clear_ring_btn = QtWidgets.QPushButton("Clear")
        self._clear_ring_btn.setEnabled(False)
        self._clear_ring_btn.clicked.connect(self._clear_ring_points)
        pick_bar.addWidget(self._clear_ring_btn)

        self._pick_status = QtWidgets.QLabel("")
        self._pick_status.setStyleSheet("color:#f0c060;font-size:11px")
        pick_bar.addWidget(self._pick_status)
        pick_bar.addStretch(1)

        self.layout().insertLayout(1, pick_bar)   # after main toolbar
        self._iv.scene.sigMouseClicked.connect(self._on_scene_clicked)

    def _on_pick_bc_toggled(self, checked: bool):
        if checked:
            self._pick_ring_btn.blockSignals(True)
            self._pick_ring_btn.setChecked(False)
            self._pick_ring_btn.blockSignals(False)
            self._pick_dsp_btn.blockSignals(True)
            self._pick_dsp_btn.setChecked(False)
            self._pick_dsp_btn.blockSignals(False)
            self._pick_mode = self.PICK_BC
            self._pick_status.setText("Click image to set BC")
        elif self._pick_mode == self.PICK_BC:
            self._pick_mode = self.PICK_NONE
            self._pick_status.setText("")

    def _on_pick_ring_toggled(self, checked: bool):
        if checked:
            self._pick_bc_btn.blockSignals(True)
            self._pick_bc_btn.setChecked(False)
            self._pick_bc_btn.blockSignals(False)
            self._pick_dsp_btn.blockSignals(True)
            self._pick_dsp_btn.setChecked(False)
            self._pick_dsp_btn.blockSignals(False)
            self._pick_mode = self.PICK_RING
            n = len(self._ring_pts)
            self._pick_status.setText(
                f"{n} pts — click ring to add" if n else
                "Click on a ring to pick points (need ≥3)")
        elif self._pick_mode == self.PICK_RING:
            self._pick_mode = self.PICK_NONE
            self._pick_status.setText(
                f"{len(self._ring_pts)} ring pts (mode off)"
                if self._ring_pts else "")

    def _on_pick_dsp_toggled(self, checked: bool):
        if checked:
            self._pick_bc_btn.blockSignals(True)
            self._pick_bc_btn.setChecked(False)
            self._pick_bc_btn.blockSignals(False)
            self._pick_ring_btn.blockSignals(True)
            self._pick_ring_btn.setChecked(False)
            self._pick_ring_btn.blockSignals(False)
            self._pick_mode = self.PICK_DSPACING
            n = len(self._dsp_pts)
            self._pick_status.setText(
                f"{n} pts — click ring to add (Ring #{self._dsp_ring_spin.value()})")
        elif self._pick_mode == self.PICK_DSPACING:
            self._pick_mode = self.PICK_NONE
            self._pick_status.setText(
                f"{len(self._dsp_pts)} d-spacing pts (mode off)"
                if self._dsp_pts else "")

    def _on_scene_clicked(self, event):
        if self._pick_mode == self.PICK_NONE:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        vb  = self._iv.getView().getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        x, y = pos.x(), pos.y()
        if self._pick_mode == self.PICK_BC:
            self._set_bc_marker(x, y)
            self.bcPicked.emit(x, y)
            self._pick_bc_btn.setChecked(False)   # one-shot
        elif self._pick_mode == self.PICK_RING:
            self._add_ring_point(x, y)
        elif self._pick_mode == self.PICK_DSPACING:
            self._add_dspacing_point(x, y)

    def _set_bc_marker(self, x: float, y: float):
        if self._bc_click_item is not None:
            self._iv.removeItem(self._bc_click_item)
        self._bc_click_item = pg.ScatterPlotItem(
            [x], [y], symbol="+", size=20,
            pen=pg.mkPen("#00aaff", width=2.5), brush=pg.mkBrush(0, 0, 0, 0))
        self._iv.addItem(self._bc_click_item)
        self._clear_ring_btn.setEnabled(True)

    def _add_ring_point(self, x: float, y: float):
        self._ring_pts.append((x, y))
        dot = pg.ScatterPlotItem(
            [x], [y], symbol="o", size=10,
            pen=pg.mkPen("#2a7fd4", width=1.5),
            brush=pg.mkBrush(42, 127, 212, 180))
        self._iv.addItem(dot)
        self._ring_pt_items.append(dot)
        self._undo_btn.setEnabled(True)
        self._clear_ring_btn.setEnabled(True)
        self._update_ring_fit()

    def _undo_ring_point(self):
        if self._pick_mode == self.PICK_DSPACING:
            self._undo_dspacing_point()
            return
        if not self._ring_pts:
            return
        self._ring_pts.pop()
        if self._ring_pt_items:
            self._iv.removeItem(self._ring_pt_items.pop())
        self._undo_btn.setEnabled(bool(self._ring_pts))
        self._clear_ring_btn.setEnabled(bool(self._ring_pts))
        self._update_ring_fit()

    def _clear_ring_points(self):
        if self._pick_mode == self.PICK_DSPACING:
            self._clear_dspacing_points()
            return
        for item in self._ring_pt_items:
            self._iv.removeItem(item)
        self._ring_pt_items.clear()
        self._ring_pts.clear()
        for item in (self._ring_fit_item, self._ring_fit_center):
            if item is not None:
                self._iv.removeItem(item)
        self._ring_fit_item = self._ring_fit_center = None
        if self._bc_click_item is not None:
            self._iv.removeItem(self._bc_click_item)
            self._bc_click_item = None
        self._undo_btn.setEnabled(False)
        self._clear_ring_btn.setEnabled(False)
        self._pick_status.setText(
            "Click on a ring to pick points (need ≥3)"
            if self._pick_mode == self.PICK_RING else
            "Click image to set BC" if self._pick_mode == self.PICK_BC else "")

    def _add_dspacing_point(self, x: float, y: float):
        ring_idx = self._dsp_ring_spin.value()
        self._dsp_pts.append((x, y, ring_idx))
        color = self._DSP_COLORS[(ring_idx - 1) % len(self._DSP_COLORS)]
        dot = pg.ScatterPlotItem(
            [x], [y], symbol="o", size=10,
            pen=pg.mkPen(color, width=1.5), brush=pg.mkBrush(color))
        self._iv.addItem(dot)
        self._dsp_pt_items.append(dot)
        self._undo_btn.setEnabled(True)
        self._clear_ring_btn.setEnabled(True)
        n = len(self._dsp_pts)
        self._pick_status.setText(f"{n} pts — click ring to add (Ring #{ring_idx})")
        self.dspacingPicksChanged.emit()

    def _undo_dspacing_point(self):
        if not self._dsp_pts:
            return
        self._dsp_pts.pop()
        if self._dsp_pt_items:
            self._iv.removeItem(self._dsp_pt_items.pop())
        self._undo_btn.setEnabled(bool(self._dsp_pts))
        self._clear_ring_btn.setEnabled(bool(self._dsp_pts))
        n = len(self._dsp_pts)
        self._pick_status.setText(
            f"{n} pts — click ring to add (Ring #{self._dsp_ring_spin.value()})"
            if n else "Click on a ring to pick points")
        self.dspacingPicksChanged.emit()

    def _clear_dspacing_points(self):
        for item in self._dsp_pt_items:
            self._iv.removeItem(item)
        self._dsp_pt_items.clear()
        self._dsp_pts.clear()
        self._undo_btn.setEnabled(False)
        self._clear_ring_btn.setEnabled(False)
        self._pick_status.setText("Click on a ring to pick points")
        self.dspacingPicksChanged.emit()

    def dspacing_picks(self) -> list:
        """Read-only snapshot of picked (x, y, ring_idx) points."""
        return list(self._dsp_pts)

    def _update_ring_fit(self):
        n = len(self._ring_pts)
        if n < 3:
            for item in (self._ring_fit_item, self._ring_fit_center):
                if item is not None:
                    self._iv.removeItem(item)
            self._ring_fit_item = self._ring_fit_center = None
            self._pick_status.setText(f"{n} pt{'s' if n != 1 else ''} — need {3-n} more")
            return
        fit = self._fit_circle(self._ring_pts)
        if fit is None:
            self._pick_status.setText(f"{n} pts — fit failed (collinear?)")
            return
        cx, cy, r = fit
        th  = np.linspace(0, 2 * math.pi, 512)
        xs  = cx + r * np.cos(th);  ys = cy + r * np.sin(th)
        pen = pg.mkPen("#2a7fd4", width=1.5, style=QtCore.Qt.DashLine)
        if self._ring_fit_item is not None:
            self._iv.removeItem(self._ring_fit_item)
        self._ring_fit_item = pg.PlotDataItem(xs, ys, pen=pen)
        self._iv.addItem(self._ring_fit_item)
        if self._ring_fit_center is not None:
            self._iv.removeItem(self._ring_fit_center)
        self._ring_fit_center = pg.ScatterPlotItem(
            [cx], [cy], symbol="+", size=18,
            pen=pg.mkPen("#2a7fd4", width=2.5), brush=pg.mkBrush(0, 0, 0, 0))
        self._iv.addItem(self._ring_fit_center)
        self._pick_status.setText(
            f"{n} pts | fit: BC=({cx:.1f}, {cy:.1f})  R={r:.1f} px → seed updated")
        self.ringFitBC.emit(cx, cy, r)

    @staticmethod
    def _fit_circle(pts: list) -> Optional[tuple]:
        """Algebraic least-squares circle fit.  Returns (cx, cy, r) or None."""
        return fit_circle_algebraic(pts)


def _add_auto_manual_buttons(plot_widget: "pg.PlotWidget", on_auto, on_manual):
    """Replace a PlotWidget's native "A" auto-range corner button with a small
    "A"/"M" (Auto/Manual) pair in the same bottom-left spot.

    ``on_auto``/``on_manual`` fire on *every* click of the respective button,
    including a reclick of whichever one is already active — a QButtonGroup
    blocks the checked button from being unchecked by its own click, but
    ``clicked`` still fires, so callers use a reclick to mean "reset the view
    to this mode's defaults now" (re-fit for Auto, snap back to the held
    range for Manual).

    Returns the ``(auto_button, manual_button)`` pair.
    """
    plot_widget.getPlotItem().hideButtons()
    grp = QtWidgets.QButtonGroup(plot_widget)
    grp.setExclusive(True)
    btn_a = QtWidgets.QPushButton("A", plot_widget)
    btn_m = QtWidgets.QPushButton("M", plot_widget)
    for b in (btn_a, btn_m):
        b.setCheckable(True)
        b.setFixedSize(18, 18)
        b.setStyleSheet(
            "QPushButton { font-size:9px; padding:0; background:#333; color:#ccc; "
            "border:1px solid #555; } "
            "QPushButton:checked { background:#4a7; color:#000; }")
        grp.addButton(b)
    btn_a.setToolTip("Auto-range: fit the view to the data automatically.\n"
                     "Click again to re-fit right now.")
    btn_m.setToolTip("Manual: hold the axis limits set via each axis's "
                     "right-click menu (“Manual” + min/max), even "
                     "during live acquisition.\nClick again to snap back to "
                     "those exact values.")
    btn_a.setChecked(True)
    btn_a.clicked.connect(on_auto)
    btn_m.clicked.connect(on_manual)

    def _reposition(*_):
        margin = 3
        y = plot_widget.height() - margin - btn_a.height()
        btn_a.move(margin, y)
        btn_m.move(margin + btn_a.width() + 2, y)
        btn_a.raise_(); btn_m.raise_()

    class _ResizeFilter(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if ev.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
                _reposition()
            return False

    rf = _ResizeFilter(plot_widget)
    plot_widget.installEventFilter(rf)
    plot_widget._auto_manual_resize_filter = rf   # keep alive (else GC'd)
    _reposition()
    return btn_a, btn_m


def _install_manual_axis_capture(plot_widget: "pg.PlotWidget", callback):
    """Call ``callback(xmin, xmax, ymin, ymax)`` with the ViewBox's current
    view range whenever the user commits a value into the PlotWidget's own
    native right-click "X axis" / "Y axis" > Manual min/max fields.

    pyqtgraph already applies a typed value to the view itself (see
    ``ViewBoxMenu.xRangeTextChanged``/``yRangeTextChanged``); this just lets a
    caller mirror the result, e.g. to remember it as the range an Auto/Manual
    toggle button pair (see :func:`_add_auto_manual_buttons`) should hold and
    restore.
    """
    vb = plot_widget.getViewBox()

    def _captured(*_):
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        callback(xmin, xmax, ymin, ymax)

    for axis in (0, 1):
        ctrl = vb.menu.ctrl[axis]
        ctrl.minText.editingFinished.connect(_captured)
        ctrl.maxText.editingFinished.connect(_captured)


# ═════════════════════════════════════════════════════════════════════════════
#  Lab-frame axes overlay  (shared builder — see tab_view.py / tab_calibrate.py)
# ═════════════════════════════════════════════════════════════════════════════

def build_lab_frame_axes_items(iv, image_shape, bc_y: float, bc_z: float) -> list:
    """MIDAS lab-frame axes overlay: X_Lab/Y_Lab compass, beam-direction
    (Z_Lab) ⊗ glyph, and an η sweep arc, anchored at (bc_y, bc_z) — lets a
    user verify orientation/ImTransOpt by checking a feature lands in the
    quadrant the overlay predicts.

    Returns a list of plain pyqtgraph scene items, NOT yet added to any
    view — the caller adds them (``iv.addItem(item)``) and owns removal;
    pan/zoom transforms them for free once added.

    ``iv`` must be a ``pg.ImageView`` whose ViewBox has ``invertY(False)``
    set, as every image viewer in this app already does
    (``ImageViewer.__init__``, ``CakeViewer.__init__``) — the MIDAS 'bl'
    lab-frame convention (+Y_MIDAS = display-left) assumes that.
    """
    nz, ny = image_shape
    y_sign = -1.0   # MIDAS 'bl' convention: +Y_MIDAS points display-left
    # invertY(False) is already applied on every viewer this is used with, so
    # +Z_MIDAS (increasing pixel row) already renders upward — no extra flip.
    V = 1.0

    xl_color, yl_color, zl_color, eta_color = "#FF3B30", "#34C759", "#0A84FF", "#FFA500"
    L = max(60.0, min(400.0, 0.15 * min(ny, nz)))
    head = max(15.0, L * 0.20)

    text_pen = pg.mkPen("w")
    text_fill = pg.mkBrush(0, 0, 0, 200)
    xl_pen = pg.mkPen(xl_color, width=3.5)
    yl_pen = pg.mkPen(yl_color, width=3.5)
    arc_pen = pg.mkPen(eta_color, width=2.5)
    label_font = QtGui.QFont(); label_font.setPointSize(13); label_font.setBold(False)
    glyph_font = QtGui.QFont(); glyph_font.setPointSize(17); glyph_font.setBold(True)

    px_w = px_h = 1.0
    try:
        pw, ph = iv.getView().getViewBox().viewPixelSize()
        if pw and ph and pw > 0 and ph > 0:
            px_w, px_h = pw, ph
    except Exception:
        pass
    px_iso = math.sqrt(px_w * px_h) if (px_w > 0 and px_h > 0) else 1.0

    items: list = []

    def add(item):
        items.append(item)

    def shaft_with_head(x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return [x0, x1], [y0, y1]
        ux, uy = dx / length, dy / length
        nx, ny_ = -uy, ux
        base_x, base_y = x1 - ux * head, y1 - uy * head
        wing = head * 0.55
        p1x, p1y = base_x + nx * wing, base_y + ny_ * wing
        p2x, p2y = base_x - nx * wing, base_y - ny_ * wing
        return [x0, x1, p1x, x1, p2x], [y0, y1, p1y, y1, p2y]

    # X_Lab arrow (MIDAS-native Y_MIDAS, display-LEFT) — unaffected by V.
    xs, ys = shaft_with_head(bc_y, bc_z, bc_y + y_sign * L, bc_z)
    add(pg.PlotDataItem(xs, ys, pen=xl_pen, connect="all"))
    # Y_Lab arrow (MIDAS-native Z_MIDAS, display-UP) — flipped by V.
    xs, ys = shaft_with_head(bc_y, bc_z, bc_y, bc_z + V * L)
    add(pg.PlotDataItem(xs, ys, pen=yl_pen, connect="all"))

    fm = QtGui.QFontMetrics(label_font)
    margin_px = 4.0
    label_specs = (
        ("h", "+X<sub>Lab</sub> (+Y<sub>MIDAS</sub>)", xl_color),
        ("v", "+Y<sub>Lab</sub> (+Z<sub>MIDAS</sub>)", yl_color))
    for axis_kind, html_body, axis_color in label_specs:
        html = f'<span style="color:{axis_color};">{html_body}</span>'
        if axis_kind == "h":
            arrow_label_R_h = L + head * 0.6
            dx, dy = y_sign * arrow_label_R_h, V * (-head * 0.9)
            anchor = (0.0 if dx > 0 else 1.0, 0.5)
        else:
            text_extent = min((fm.height() / 2.0 + margin_px) * px_iso, 0.5 * L)
            arrow_label_R_v = L + max(head * 0.6, text_extent)
            dx, dy = 0.0, V * arrow_label_R_v
            anchor = (0.5, 0.5)
        lbl = pg.TextItem(html=html, anchor=anchor, border=text_pen, fill=text_fill)
        lbl.setFont(label_font)
        lbl.setPos(bc_y + dx, bc_z + dy)
        add(lbl)

    # ⊗ glyph at BC — Z_Lab (MIDAS-native X_MIDAS), the beam direction.
    glyph = pg.TextItem("⊗", color=zl_color, anchor=(0.5, 0.5), border=text_pen, fill=text_fill)
    glyph.setFont(glyph_font)
    glyph.setPos(bc_y, bc_z)
    add(glyph)
    beam_html = f'<span style="color:{zl_color};">+Z<sub>Lab</sub> (+X<sub>MIDAS</sub>, beam)</span>'
    x_lbl = pg.TextItem(html=beam_html, anchor=(0.5, 0.0), border=text_pen, fill=text_fill)
    x_lbl.setFont(label_font)
    x_lbl.setPos(bc_y, bc_z + V * (-head * 1.2))
    add(x_lbl)

    # η reference marks at the four cardinal angles — 0°/+90°/−90°/180° —
    # using the same convention as pixel_to_REta (η=atan2(-Yc,Zc): η=0 is
    # +Z_MIDAS/+Y_Lab, straight up) and the same (-y_sign)/V flips as the
    # X_Lab/Y_Lab arrows above, so these track any lab-frame flip exactly.
    # A real caking ring/spoke overlay (draw_polar_bin_overlay) reduces to
    # this same dY=r·sinη, dZ=r·cosη formula at zero tilt — this is just
    # the always-visible compass, independent of any loaded geometry.
    R_arc = L * 0.85
    tick_inner, tick_outer, label_R = R_arc * 0.92, R_arc * 1.12, R_arc * 1.32
    eta_marks = ((0.0, "η=0°"), (90.0, "η=+90°"), (-90.0, "η=−90°"), (180.0, "η=180°"))
    for eta_deg, label in eta_marks:
        eta_rad = math.radians(eta_deg)
        ux = (-y_sign) * math.sin(eta_rad)
        uy = V * math.cos(eta_rad)
        add(pg.PlotDataItem([bc_y + ux * tick_inner, bc_y + ux * tick_outer],
                             [bc_z + uy * tick_inner, bc_z + uy * tick_outer],
                             pen=arc_pen))
        if abs(uy) >= abs(ux):
            anchor = (0.5, 1.0 if uy > 0 else 0.0)
        else:
            anchor = (0.0 if ux > 0 else 1.0, 0.5)
        html = f'<span style="color:{eta_color};">{label}</span>'
        lbl = pg.TextItem(html=html, anchor=anchor, border=text_pen, fill=text_fill)
        lbl.setFont(label_font)
        lbl.setPos(bc_y + ux * label_R, bc_z + uy * label_R)
        add(lbl)

    return items


# ═════════════════════════════════════════════════════════════════════════════
#  CakeViewer
# ═════════════════════════════════════════════════════════════════════════════

class CakeViewer(QtWidgets.QWidget):
    """2-D azimuthal-integration "cake" heatmap: R (px) on X, η (°) on Y.

    Unlike ImageViewer, the two axes are physical integration-bin coordinates
    (not detector row/column indices starting at 0) — the image is positioned
    with ``ImageItem.setRect()`` to the actual R/η bin-centre extent rather
    than assumed to start at the origin.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(background="k", foreground="w")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        bar = QtWidgets.QHBoxLayout()
        self._log = QtWidgets.QCheckBox("Log")
        self._log.toggled.connect(self._redisplay)
        bar.addWidget(self._log)
        bar.addWidget(QtWidgets.QLabel("cmap:"))
        self._cmap = _NoScrollComboBox()
        self._cmap.addItems(COLORMAPS)
        self._cmap.setCurrentText(_DEFAULT_CMAP)
        self._cmap.currentTextChanged.connect(self._set_cmap)
        self._cmap.setFixedWidth(90)
        bar.addWidget(self._cmap)
        bar.addWidget(QtWidgets.QLabel("vmin%:"))
        self._vmin = _NoScrollSpinBox()
        self._vmin.setRange(0, 99); self._vmin.setValue(30); self._vmin.setFixedWidth(45)
        self._vmin.valueChanged.connect(self._redisplay)
        bar.addWidget(self._vmin)
        bar.addWidget(QtWidgets.QLabel("vmax%:"))
        self._vmax = _NoScrollSpinBox()
        self._vmax.setRange(1, 100); self._vmax.setValue(99); self._vmax.setFixedWidth(45)
        self._vmax.valueChanged.connect(self._redisplay)
        bar.addWidget(self._vmax)
        bar.addStretch(1)
        self._toolbar_layout = bar   # exposed so subclasses/callers can append widgets
        layout.addLayout(bar)

        self._iv = pg.ImageView(view=pg.PlotItem(viewBox=pg.ViewBox()))
        self._iv.ui.roiBtn.hide(); self._iv.ui.menuBtn.hide()
        vb = self._iv.getView().getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.setMouseMode(pg.ViewBox.PanMode)
        vb.invertY(False)   # η increases upward, like a normal Cartesian plot
        # R (px) and η (°) are unrelated units with no physical aspect ratio to
        # preserve. pg.ImageView.__init__ force-locks the aspect on any view it's
        # given, and an aspect-locked ViewBox always couples X/Y on right-drag
        # zoom (ViewBox.scaleBy sets scale[0]=scale[1] whenever both axes are
        # given, and ViewBox.updateViewRange re-derives one axis from the other
        # on every range change regardless) — so a pure horizontal drag ends up
        # zooming both axes together (or not at all), never R alone. Unlocking
        # it lets right-drag zoom independently per axis, matching
        # ProfileViewer/HydraProfileViewer's plain (unlocked) PlotWidget.
        vb.setAspectLocked(False)
        self._iv.getView().setLabel("bottom", "R", units="px")
        self._iv.getView().setLabel("left", "η", units="°")
        layout.addWidget(self._iv, stretch=1)

        self._coord_bar = QtWidgets.QLabel("Run an integration to see the (η, R) cake")
        self._coord_bar.setStyleSheet(
            f"color:#dddddd; background:#1a1a1a; font-family:{S.MONO_CSS};"
            "font-size:12px; padding:2px 6px; border-top:1px solid #444;")
        layout.addWidget(self._coord_bar)
        self._mouse_proxy = pg.SignalProxy(self._iv.scene.sigMouseMoved, rateLimit=60, slot=self._mouse)

        self._cake: Optional[np.ndarray] = None      # (n_eta, n_r)
        self._r_axis: Optional[np.ndarray] = None
        self._eta_axis: Optional[np.ndarray] = None
        self._set_cmap(_DEFAULT_CMAP)

    def set_cake(self, cake_2d: np.ndarray, r_axis_px: np.ndarray, eta_axis_deg: np.ndarray):
        self._cake = np.asarray(cake_2d, dtype=np.float32)
        self._r_axis = np.asarray(r_axis_px, dtype=np.float64)
        self._eta_axis = np.asarray(eta_axis_deg, dtype=np.float64)
        self._redisplay()

    def clear(self):
        self._cake = None
        self._iv.clear()
        self._coord_bar.setText("Run an integration to see the (η, R) cake")

    def _redisplay(self, *_args):
        if self._cake is None or self._cake.size == 0:
            return
        cake = self._cake
        disp = np.log10(np.clip(cake, 1e-6, None)) if self._log.isChecked() else cake
        # Exclude exact-zero bins from the percentile calc (masked on the
        # pre-log data, like ImageViewer) — unfilled η/R bins would otherwise
        # skew the auto-level window toward zero.
        nonzero = cake != 0
        candidates = disp[np.isfinite(disp) & nonzero]
        finite = candidates if candidates.size else disp[np.isfinite(disp)]
        if finite.size:
            lo = float(np.percentile(finite, self._vmin.value()))
            hi = float(np.percentile(finite, self._vmax.value()))
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        img = disp.T   # (n_eta, n_r) -> (n_r, n_eta): pyqtgraph's first axis is X
        self._iv.setImage(img.astype(np.float32), autoLevels=False, levels=(lo, hi),
                          autoRange=False, autoHistogramRange=False)
        r0, r1 = float(self._r_axis[0]), float(self._r_axis[-1])
        e0, e1 = float(self._eta_axis[0]), float(self._eta_axis[-1])
        self._iv.getImageItem().setRect(
            QtCore.QRectF(r0, e0, max(r1 - r0, 1e-6), max(e1 - e0, 1e-6)))
        self._apply_view_limits(r0, r1, e0, e1)
        self._iv.getView().getViewBox().autoRange()
        self._iv.getHistogramWidget().item.setHistogramRange(lo, hi, padding=0.1)
        n_eta, n_r = cake.shape
        self._coord_bar.setText(
            f"cake {n_r} R-bins × {n_eta} η-bins  |  "
            f"R ∈ [{r0:.1f}, {r1:.1f}] px   η ∈ [{e0:.1f}, {e1:.1f}]°")

    def _apply_view_limits(self, r0: float, r1: float, e0: float, e1: float):
        """Bound pan/zoom to the current cake's (R, η) extent (+ margin), same
        intent as ``ImageViewer._apply_view_limits`` — stops the user
        scrolling/zooming out into an empty void or losing the cake off-screen."""
        if not all(math.isfinite(v) for v in (r0, r1, e0, e1)):
            return
        rmin, rmax = min(r0, r1), max(r0, r1)
        emin, emax = min(e0, e1), max(e0, e1)
        if rmax <= rmin:
            rmax = rmin + 1.0
        if emax <= emin:
            emax = emin + 1.0
        rpad = 0.5 * (rmax - rmin)
        epad = 0.5 * (emax - emin)
        vb = self._iv.getView().getViewBox()
        vb.setLimits(
            xMin=rmin - rpad, xMax=rmax + rpad,
            yMin=emin - epad, yMax=emax + epad,
            minXRange=max((rmax - rmin) * 0.01, 1e-6),
            minYRange=max((emax - emin) * 0.01, 1e-6),
            maxXRange=(rmax - rmin) + 2 * rpad,
            maxYRange=(emax - emin) + 2 * epad,
        )

    def _set_cmap(self, name: str):
        self._iv.setColorMap(_resolve_cmap(name))

    def display_state(self) -> dict:
        """Same contract as ``ImageViewer.display_state`` — see there for why
        this exists as its own method rather than reading the widgets
        directly (CakeViewer duplicates ImageViewer's cmap/log/vmin/vmax
        toolbar rather than subclassing it, since its axes are physical
        (R, η) bin coordinates rather than detector row/column indices)."""
        return {"cmap": self._cmap.currentText(), "log": self._log.isChecked(),
                "vmin": self._vmin.value(), "vmax": self._vmax.value()}

    def set_display_state(self, state: Optional[dict]) -> None:
        """See ``ImageViewer.set_display_state`` — identical reasoning
        (cmap needs an explicit re-apply; log/vmin/vmax take effect on the
        next ``set_cake`` regardless of whether their signal fired)."""
        if not state:
            return
        cmap = state.get("cmap")
        if cmap and self._cmap.findText(str(cmap)) >= 0:
            self._cmap.blockSignals(True)
            self._cmap.setCurrentText(str(cmap))
            self._cmap.blockSignals(False)
            self._set_cmap(str(cmap))
        if "log" in state:
            self._log.blockSignals(True)
            self._log.setChecked(bool(state["log"]))
            self._log.blockSignals(False)
        for key, spin in (("vmin", self._vmin), ("vmax", self._vmax)):
            if key in state:
                spin.blockSignals(True)
                spin.setValue(state[key])
                spin.blockSignals(False)
        self._redisplay()

    def _mouse(self, evt):
        if self._cake is None or self._r_axis is None:
            return
        pos = evt[0]
        vb = self._iv.getView().getViewBox()
        if not self._iv.getView().sceneBoundingRect().contains(pos):
            return
        mp = vb.mapSceneToView(pos)
        r, eta = mp.x(), mp.y()
        n_eta, n_r = self._cake.shape
        r0, r1 = float(self._r_axis[0]), float(self._r_axis[-1])
        e0, e1 = float(self._eta_axis[0]), float(self._eta_axis[-1])
        if not (min(r0, r1) <= r <= max(r0, r1) and min(e0, e1) <= eta <= max(e0, e1)):
            return
        ir = int((r - r0) / max(r1 - r0, 1e-9) * (n_r - 1))
        ie = int((eta - e0) / max(e1 - e0, 1e-9) * (n_eta - 1))
        ir = min(max(ir, 0), n_r - 1); ie = min(max(ie, 0), n_eta - 1)
        val = self._cake[ie, ir]
        self._coord_bar.setText(
            f"R = {r:.2f} px   η = {eta:.2f}°   intensity = {val:.4g}")


# ═════════════════════════════════════════════════════════════════════════════
#  RingResidualViewer  (NEW — per-ring residual, 1-D bar chart or 2-D
#  ring × azimuth pseudo-strain heatmap; ΔR (px) or strain (µε))
# ═════════════════════════════════════════════════════════════════════════════

class RingResidualViewer(CakeViewer):
    """Per-ring radial residual, in either of two sources and two views:

    * **Model** — ``AutoCalibrationResult.residual_corr_map`` (the backend's
      whole-detector RBF-smoothed residual model) rebinned onto the same
      (R, η) grid as the intensity cake. Dense, but interpolated — it has a
      value everywhere the detector has pixel coverage, not just at the
      calibrant's actual ring positions.
    * **Ring** — :func:`ring_azimuth_residual`: the same "local peak near the
      predicted radius" measurement, run independently per η row instead of
      once on the azimuthally-collapsed profile. This is the direct
      deviation from the calibrant's ideal ring position (tied to its real
      d-spacings) — no smoothing/interpolation involved. X-axis is ring
      index, not R, since rings aren't evenly spaced in radius.

    Both are a signed quantity centred on zero (a good calibration reads ~0
    everywhere) and NaN marks "no data" rather than 0 — so this overrides
    ``CakeViewer``'s log toggle, percentile windowing and mouse readout
    instead of reusing them as-is.

    **View** toggles between the full 2-D (ring/R × η) heatmap and a 1-D
    per-ring reduction. For the 1-D **Ring** reduction, **Ring 1-D method**
    picks how: "η-mean of cake" collapses the 2-D grid actually on screen
    (so 1-D and 2-D always agree on the same underlying measurement), or
    "Peak of collapsed profile" repeats the higher-SNR single-peak-search
    :func:`collapsed_profile_ring_residual` does on the pre-azimuthally-
    averaged profile (loses per-η detail, more robust for weak/spotty
    rings). The Model source only ever has an η-mean reduction — its cake
    is already a smoothed residual model, not raw intensity, so there's no
    "collapsed profile" for it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log.setVisible(False)
        self._cmap.insertItem(0, "coolwarm")
        self._cmap.setCurrentIndex(0)
        self._source = _NoScrollComboBox()
        self._source.addItem("Ring (η-resolved)", "ring")
        self._source.addItem("Model (whole detector)", "model")
        self._source.setToolTip(
            "Model: the backend's smoothed residual_corr_map, rebinned onto\n"
            "the (R, η) cake grid. Ring: the local-peak measurement run\n"
            "independently per azimuth row instead of on the azimuthally-\n"
            "collapsed profile.")
        self._source.currentIndexChanged.connect(self._on_source_changed)
        self._toolbar_layout.insertWidget(1, self._source)
        self._mode = _NoScrollComboBox()
        self._mode.addItem("ΔR (px)", "px")
        self._mode.addItem("strain (µε)", "strain")
        self._mode.currentIndexChanged.connect(self._redisplay)
        self._toolbar_layout.insertWidget(2, self._mode)
        self._view = _NoScrollComboBox()
        self._view.addItem("2-D (cake)", "2d")
        self._view.addItem("1-D (per ring)", "1d")
        self._view.currentIndexChanged.connect(self._redisplay)
        self._toolbar_layout.insertWidget(3, self._view)
        self._ring1d_method = _NoScrollComboBox()
        self._ring1d_method.addItem("η-mean of cake", "eta_mean")
        self._ring1d_method.addItem("Peak of collapsed profile", "collapsed_profile")
        self._ring1d_method.setToolTip(
            "η-mean of cake: average the per-η ring residuals shown in 2-D\n"
            "mode — 1-D and 2-D always agree.\n"
            "Peak of collapsed profile: one peak search on the azimuthally-\n"
            "averaged profile (higher SNR, no η resolution).")
        self._ring1d_method.currentIndexChanged.connect(self._redisplay)
        self._toolbar_layout.insertWidget(4, self._ring1d_method)

        self._model_cake: Optional[np.ndarray] = None   # (n_eta, n_r)
        self._ring_grid: Optional[np.ndarray] = None     # (n_eta, n_rings)
        self._ring_radii: Optional[np.ndarray] = None    # (n_rings,) px
        self._profile: Optional[np.ndarray] = None        # azimuthally-collapsed profile
        self._all_ring_radii: Optional[np.ndarray] = None  # unfiltered predicted radii

        # 1-D view: a sibling plot, stacked with the 2-D image view so
        # toggling View swaps which one is visible.
        self._plot1d = pg.PlotWidget(background="k")
        self._plot1d.showGrid(x=True, y=True, alpha=0.2)
        self._zero1d = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#888", width=1))
        self._plot1d.addItem(self._zero1d)
        self._plot1d_item = None
        layout = self.layout()
        idx = layout.indexOf(self._iv)
        layout.removeWidget(self._iv)
        self._view_stack = QtWidgets.QStackedWidget()
        self._view_stack.addWidget(self._iv)
        self._view_stack.addWidget(self._plot1d)
        layout.insertWidget(idx, self._view_stack, 1)
        self._update_ring1d_method_visibility()

    def set_data(self, model_cake, ring_grid, ring_radii_px, r_axis_px, eta_axis_deg,
                *, profile=None, all_ring_radii_px=None):
        """``model_cake``/``ring_grid`` may each be None (e.g. no
        residual_corr_map for a Multi-panel run, or no ring matched the
        cake) — the source combo simply has nothing to show for that entry.
        ``profile``/``all_ring_radii_px`` are optional and feed only the
        1-D "Peak of collapsed profile" Ring method."""
        self._model_cake = None if model_cake is None else np.asarray(model_cake, dtype=float)
        self._ring_grid = None if ring_grid is None else np.asarray(ring_grid, dtype=float)
        self._ring_radii = np.asarray(ring_radii_px, dtype=float)
        self._r_axis = np.asarray(r_axis_px, dtype=float)
        self._eta_axis = np.asarray(eta_axis_deg, dtype=float)
        self._profile = None if profile is None else np.asarray(profile, dtype=float)
        self._all_ring_radii = (None if all_ring_radii_px is None
                                else np.asarray(all_ring_radii_px, dtype=float))
        self._redisplay()

    def clear(self):
        self._model_cake = None
        self._ring_grid = None
        self._profile = None
        self._all_ring_radii = None
        self._iv.clear()
        if self._plot1d_item is not None:
            self._plot1d.removeItem(self._plot1d_item)
            self._plot1d_item = None
        self._coord_bar.setText(
            "No residual data for this attempt (Model needs "
            "residual_corr_map — unavailable with Multi-panel enabled or "
            "Distortion refinement off; Ring needs a matched ring in the "
            "cake's R range).")

    def _on_source_changed(self, *_args):
        self._redisplay()

    def _update_ring1d_method_visibility(self):
        show = (self._view.currentData() == "1d"
               and self._source.currentData() == "ring")
        self._ring1d_method.setVisible(show)

    def _active(self):
        """(cake, x_axis, x_label, x_units) for the selected source."""
        if self._source.currentData() == "ring":
            n = 0 if self._ring_radii is None else len(self._ring_radii)
            return self._ring_grid, np.arange(n, dtype=float), "ring index", None
        return self._model_cake, self._r_axis, "R", "px"

    def _display_cake(self, cake, x_axis):
        """The raw ΔR (px) cake, or the pseudo-strain (µε) — see
        :func:`radius_ratio_strain_ue`. Note: in Ring mode ``x_axis`` is the
        *plot* x-position (ring index, for even bar-chart-style spacing) —
        the physics needs the real radius, ``self._ring_radii``, not that
        index. Whichever radius array is real, (radius, radius+cake) is
        r_pred vs. r_obs depending on source, per each one's own sign
        convention:

        * Ring (`ring_azimuth_residual`): cake = r_obs - r_pred, axis = r_pred.
        * Model (backend's residual_corr_map, see forward/residual_corr.py):
          axis = r_obs (the raw/naive position), axis + cake = r_pred — its
          docstring: "storing -ΔR/px makes addition [axis + cake] the right
          operation" to go from observed to ideal.
        """
        if self._mode.currentData() != "strain":
            return cake
        is_ring = self._source.currentData() == "ring"
        r_axis = self._ring_radii if is_ring else x_axis
        r_grid = r_axis[np.newaxis, :]
        if is_ring:
            r_pred, r_obs = r_grid, r_grid + cake
        else:
            r_obs, r_pred = r_grid, r_grid + cake
        return radius_ratio_strain_ue(r_pred, r_obs)

    def _redisplay(self, *_args):
        self._update_ring1d_method_visibility()
        is_1d = self._view.currentData() == "1d"
        self._view_stack.setCurrentWidget(self._plot1d if is_1d else self._iv)
        if is_1d:
            self._redisplay_1d()
        else:
            self._redisplay_2d()

    def _redisplay_2d(self):
        cake, x_axis, x_label, x_units = self._active()
        self._iv.getView().setLabel("bottom", x_label, units=x_units)
        self._iv.getView().setLabel("left", "η", units="°")
        if cake is None or cake.size == 0 or self._eta_axis is None:
            self._iv.clear()
            return
        disp = self._display_cake(cake, x_axis)
        finite = disp[np.isfinite(disp)]
        if finite.size:
            v = float(np.percentile(np.abs(finite), self._vmax.value()))
            v = max(v, 1e-9)
        else:
            v = 1.0
        lo, hi = -v, v
        img = disp.T   # (n_eta, n_x) -> (n_x, n_eta): pyqtgraph's first axis is X
        self._iv.setImage(img.astype(np.float32), autoLevels=False, levels=(lo, hi),
                          autoRange=False, autoHistogramRange=False)
        x0, x1 = (float(x_axis[0]), float(x_axis[-1])) if x_axis.size else (0.0, 1.0)
        e0, e1 = float(self._eta_axis[0]), float(self._eta_axis[-1])
        # Ring mode: centre each ring's column on its index (bar-chart-style),
        # not a zero-width point at x_axis[0]==x_axis[-1]==0 for a single ring.
        rect_x0 = x0 - 0.5 if x_units is None else x0
        rect_w = max((x1 - x0) + (1.0 if x_units is None else 0.0), 1e-6)
        self._iv.getImageItem().setRect(
            QtCore.QRectF(rect_x0, e0, rect_w, max(e1 - e0, 1e-6)))
        self._apply_view_limits(rect_x0, rect_x0 + rect_w, e0, e1)
        self._iv.getView().getViewBox().autoRange()
        self._iv.getHistogramWidget().item.setHistogramRange(lo, hi, padding=0.1)
        n_eta, n_x = disp.shape
        unit = "µε" if self._mode.currentData() == "strain" else "px"
        src = "ring" if self._source.currentData() == "ring" else "R"
        self._coord_bar.setText(
            f"strain cake {n_x} {src}-bins × {n_eta} η-bins  |  "
            f"η ∈ [{e0:.1f}, {e1:.1f}]°   (±{v:.3g} {unit} shown)")

    def _redisplay_1d(self):
        is_ring = self._source.currentData() == "ring"
        is_strain = self._mode.currentData() == "strain"
        unit = "µε" if is_strain else "px"
        method = self._ring1d_method.currentData() if is_ring else "eta_mean"

        if method == "collapsed_profile":
            radii = list(self._all_ring_radii) if self._all_ring_radii is not None else []
            resid, kept = collapsed_profile_ring_residual(self._r_axis, self._profile, radii)
            if is_strain and kept:
                r_pred = np.asarray(kept, dtype=float)
                resid = radius_ratio_strain_ue(r_pred, r_pred + resid)
            x = np.arange(len(kept), dtype=float)
            y = resid
        else:
            cake, x_axis, _label, _units = self._active()
            if cake is None or cake.size == 0:
                x, y = np.array([]), np.array([])
            else:
                disp = self._display_cake(cake, x_axis)
                with np.errstate(invalid="ignore"):
                    y = np.nanmean(disp, axis=0)
                x = x_axis

        if self._plot1d_item is not None:
            self._plot1d.removeItem(self._plot1d_item)
            self._plot1d_item = None
        ylabel = "strain (µε)" if is_strain else "Δr (px)"
        self._plot1d.setLabel("left", ylabel)

        finite = np.isfinite(y) if y.size else np.array([], dtype=bool)
        if not np.any(finite):
            self._plot1d.setLabel("bottom", "ring index" if is_ring else "R", units=None if is_ring else "px")
            self._coord_bar.setText("no data")
            return

        xf, yf = x[finite], y[finite]
        if is_ring:
            self._plot1d.setLabel("bottom", "ring index")
            self._plot1d_item = pg.BarGraphItem(x=xf, height=yf, width=0.6,
                                                brush=pg.mkBrush("#5aa0e0"))
            self._plot1d.addItem(self._plot1d_item)
        else:
            self._plot1d.setLabel("bottom", "R", units="px")
            self._plot1d_item = self._plot1d.plot(xf, yf, pen=pg.mkPen("#5aa0e0", width=2))
        self._plot1d.autoRange()
        rms = float(np.sqrt(np.mean(yf ** 2)))
        n_label = f"{len(yf)} rings" if is_ring else f"{len(yf)} points"
        method_note = ("  |  peak-of-collapsed-profile" if method == "collapsed_profile"
                       else "  |  η-mean of cake")
        self._coord_bar.setText(f"{n_label} | RMS {unit} = {rms:.3f}{method_note}")

    def _mouse(self, evt):
        if self._view.currentData() != "2d":
            return
        cake, x_axis, _label, x_units = self._active()
        if cake is None or x_axis is None or x_axis.size == 0 or self._eta_axis is None:
            return
        pos = evt[0]
        vb = self._iv.getView().getViewBox()
        if not self._iv.getView().sceneBoundingRect().contains(pos):
            return
        mp = vb.mapSceneToView(pos)
        x, eta = mp.x(), mp.y()
        n_eta, n_x = cake.shape
        x0, x1 = float(x_axis[0]), float(x_axis[-1])
        e0, e1 = float(self._eta_axis[0]), float(self._eta_axis[-1])
        ring_mode = x_units is None
        x_lo = x0 - 0.5 if ring_mode else min(x0, x1)
        x_hi = x1 + 0.5 if ring_mode else max(x0, x1)
        if not (x_lo <= x <= x_hi and min(e0, e1) <= eta <= max(e0, e1)):
            return
        ix = int(round(x)) if ring_mode else int((x - x0) / max(x1 - x0, 1e-9) * (n_x - 1))
        ie = int((eta - e0) / max(e1 - e0, 1e-9) * (n_eta - 1))
        ix = min(max(ix, 0), n_x - 1); ie = min(max(ie, 0), n_eta - 1)
        dr = cake[ie, ix]
        loc = f"ring {ix}" if ring_mode else f"R = {x:.2f} px"
        if not np.isfinite(dr):
            self._coord_bar.setText(f"{loc}   η = {eta:.2f}°   (no data)")
            return
        # x_axis[ix] in Ring mode is the plot position (ring index), not the
        # real radius the physics needs — that's self._ring_radii[ix].
        r_val = float(self._ring_radii[ix]) if ring_mode else float(x_axis[ix])
        if ring_mode:
            r_pred, r_obs = r_val, r_val + dr
        else:
            r_obs, r_pred = r_val, r_val + dr
        eps_ue = float(radius_ratio_strain_ue(r_pred, r_obs))
        self._coord_bar.setText(
            f"{loc}   η = {eta:.2f}°   Δr = {dr:.4g} px (ε = {eps_ue:.4g} µε)")


# ═════════════════════════════════════════════════════════════════════════════
#  ProfileViewer
# ═════════════════════════════════════════════════════════════════════════════

class ProfileViewer(QtWidgets.QWidget):
    """1D radial profile viewer with x-axis unit switching and ring markers.

    Optionally shades a ±σ uncertainty band when sigma is supplied.
    Left-clicking the plot emits ``radiusClicked`` (radius in px), so a caller can
    draw the matching ring on the image; a marker line shows the picked position.
    """

    radiusClicked = QtCore.pyqtSignal(float)   # picked radius in px

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(2)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("X:"))
        self._xaxis = _NoScrollComboBox()
        self._xaxis.addItems(["R (px)", "2θ (°)", "Q (Å⁻¹)"])
        self._xaxis.currentIndexChanged.connect(self._on_xaxis_changed)
        bar.addWidget(self._xaxis)
        self._logy = QtWidgets.QCheckBox("Log Y")
        self._logy.toggled.connect(self._on_logy_toggled)
        bar.addWidget(self._logy)
        bar.addStretch(1)
        self._stat = QtWidgets.QLabel("")
        self._stat.setStyleSheet("color:#aaa;font-size:10px")
        bar.addWidget(self._stat)
        self._toolbar_layout = bar   # exposed for external widget insertion
        layout.addLayout(bar)

        # Manual axis limits (see the "A"/"M" buttons added over the plot's
        # bottom-left corner, below) reuse the axis's own native right-click
        # "Manual" min/max fields rather than a separate row of spin boxes.
        self._manual_mode = False
        self._manual_range: Optional[tuple] = None   # (xmin, xmax, ymin, ymax)

        self._plot = pg.PlotWidget(background="k")
        self._plot.setLabel("left", "Mean intensity")
        self._plot.setLabel("bottom", "R (px)")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._user_xrange: Optional[tuple] = None
        self._user_yrange: Optional[tuple] = None
        self._suspend_range_track = False
        vb0 = self._plot.getPlotItem().getViewBox()
        vb0.sigXRangeChanged.connect(self._on_xrange_changed)
        vb0.sigYRangeChanged.connect(self._on_yrange_changed)
        self._band_lo = pg.PlotDataItem([], [], pen=None)
        self._band_hi = pg.PlotDataItem([], [], pen=None)
        self._band = pg.FillBetweenItem(self._band_lo, self._band_hi,
                                        brush=pg.mkBrush(136, 204, 255, 60))
        self._band.setVisible(False)
        self._plot.addItem(self._band)
        self._curve = self._plot.plot([], [], pen=pg.mkPen("#88ccff", width=2))
        self._ring_lines: list = []
        layout.addWidget(self._plot, stretch=1)

        self._btn_auto, self._btn_manual = _add_auto_manual_buttons(
            self._plot, self._on_auto_clicked, self._on_manual_clicked)
        _install_manual_axis_capture(self._plot, self._on_manual_range_edited)

        # Click-to-pick a radius (drawn on the image by the caller).
        self._pick_line = None
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self._r_px = self._prof = self._sigma = None
        self._wl = self._lsd = self._px = None
        self._ring_groups: list = []   # [{"radii": [...], "color": "#hex"}, ...]
        self._ring_lsd = self._ring_px = self._ring_wl = None

    def set_profile(self, r_px, profile, *, sigma=None, wavelength_A=None,
                    lsd_um=None, px_um=None):
        self._r_px   = np.asarray(r_px)
        self._prof   = np.asarray(profile)
        self._sigma  = np.asarray(sigma) if sigma is not None else None
        self._wl     = wavelength_A
        self._lsd    = lsd_um
        self._px     = px_um
        self._replot()

    def set_ring_markers(self, groups, lsd_um=None, px_um=None, wl=None):
        """``groups``: list of ``{"radii": [r_px, ...], "color": "#rrggbb"}`` —
        one entry per material, each drawn in its own color."""
        self._ring_groups = list(groups)
        self._ring_lsd = lsd_um
        self._ring_px  = px_um
        self._ring_wl  = wl
        self._replot()

    def _r_to_x(self, r_px, idx, lsd, px, wl):
        if idx == 0 or lsd is None:
            return r_px
        two_theta = math.atan(r_px * px / lsd)
        if idx == 1:
            return math.degrees(two_theta)
        if idx == 2 and wl:
            return 4 * math.pi * math.sin(two_theta / 2) / wl
        return r_px

    def _on_xaxis_changed(self, *_):
        """Switching R/2θ/Q changes the X scale entirely — a remembered X zoom
        from the old unit is meaningless in the new one, so drop it."""
        self._user_xrange = None
        self._clear_pick_line()
        self._replot()

    def _on_logy_toggled(self, *_):
        """Log/linear Y have unrelated scales — drop any remembered Y zoom."""
        self._user_yrange = None
        self._replot()

    def _on_manual_range_edited(self, xmin, xmax, ymin, ymax):
        """The user set exact limits via an axis's native right-click
        "Manual" min/max fields (pyqtgraph already applied it to the view) —
        remember it as the held manual range: what "M" reapplies on a
        reclick, and what every live-acquisition redraw holds to while
        Manual is active."""
        self._manual_range = (xmin, xmax, ymin, ymax)
        if self._manual_mode:
            self._apply_manual_range()

    def _on_manual_clicked(self):
        """"M" clicked — switch to Manual, or (on a reclick) snap back to the
        exact limits held from the axes' native "Manual" min/max fields."""
        if self._manual_range is None:
            (xmin, xmax), (ymin, ymax) = self._plot.getViewBox().viewRange()
            self._manual_range = (xmin, xmax, ymin, ymax)
        self._manual_mode = True
        self._apply_manual_range()

    def _on_auto_clicked(self):
        """"A" clicked — switch to Auto, or (on a reclick) force an
        immediate re-fit to the current profile."""
        self._manual_mode = False
        self._user_xrange = self._user_yrange = None
        self._replot()

    def _apply_manual_range(self):
        """Force the exact held manual limits, unclamped by any pan/zoom bound."""
        if self._manual_range is None:
            return
        xmin, xmax, ymin, ymax = self._manual_range
        vb = self._plot.getPlotItem().getViewBox()
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None,
                     maxXRange=None, maxYRange=None)
        self._plot.setXRange(xmin, xmax, padding=0)
        self._plot.setYRange(ymin, ymax, padding=0)

    def _replot(self):
        if self._r_px is None:
            return
        idx = self._xaxis.currentIndex()
        if idx == 0 or self._lsd is None:
            x = self._r_px
            self._plot.setLabel("bottom", "R (px)")
        else:
            x = np.array([self._r_to_x(r, idx, self._lsd, self._px, self._wl)
                          for r in self._r_px])
            self._plot.setLabel("bottom", ["R (px)", "2θ (°)", "Q (Å⁻¹)"][idx])
        y = self._prof
        log = self._logy.isChecked()
        if log:
            y = np.where(y > 0, np.log10(np.maximum(y, 1e-30)), np.nan)
            self._plot.setLabel("left", "log₁₀(intensity)")
        else:
            self._plot.setLabel("left", "Mean intensity")

        # Everything below can trigger incidental view-range signals (pyqtgraph
        # autorange on setData, setLimits() clamping the current view, etc.) —
        # suspend user-zoom tracking for all of it so only genuine mouse-driven
        # pan/zoom ever gets remembered as "the user zoomed".
        self._suspend_range_track = True
        try:
            self._curve.setData(x, y)

            # Uncertainty band (linear scale only)
            if self._sigma is not None and not log:
                self._band_lo.setData(x, self._prof - self._sigma)
                self._band_hi.setData(x, self._prof + self._sigma)
                self._band.setCurves(self._band_lo, self._band_hi)
                self._band.setVisible(True)
            else:
                self._band.setVisible(False)

            fin = y[np.isfinite(y)]
            ymin = ymax = None
            if fin.size:
                ymin = float(fin.min() * 0.9) if not log else 1.0
                ymax = float(fin.max()) * 1.05

            x_arr = np.asarray(x)
            xmin = max(0.0, float(x_arr.min())) if x_arr.size else None
            xmax = float(x_arr.max()) if x_arr.size else None

            if self._manual_mode:
                # Manual mode is authoritative: never touch pan/zoom bounds or
                # the view range from the data here — just hold the user's
                # entered limits, even across live-acquisition redraws.
                self._apply_manual_range()
            else:
                if xmin is not None and ymin is not None:
                    self._apply_view_limits(xmin, xmax, ymin, ymax)

                if self._user_xrange is not None:
                    self._plot.setXRange(*self._user_xrange, padding=0)
                elif xmin is not None:
                    self._plot.setXRange(xmin, xmax, padding=0.02)

                if self._user_yrange is not None:
                    self._plot.setYRange(*self._user_yrange, padding=0)
                elif ymin is not None:
                    self._plot.setYRange(ymin, ymax, padding=0)

            self._stat.setText(f"{len(self._r_px)} bins | max={np.nanmax(self._prof):.1f}")

            # Ring markers redrawn LAST (after setXRange) so they always appear
            for ln in self._ring_lines:
                self._plot.removeItem(ln)
            self._ring_lines.clear()
            if self._ring_groups:
                lsd = self._ring_lsd or self._lsd
                px  = self._ring_px  or self._px
                wl  = self._ring_wl  or self._wl
                for group in self._ring_groups:
                    pen = pg.mkPen(group.get("color", "#f0c060"), width=1.5,
                                    style=QtCore.Qt.DotLine)
                    for r in group.get("radii", []):
                        x_pos = self._r_to_x(r, idx, lsd, px, wl)
                        if x_pos is None:
                            continue
                        ln = pg.InfiniteLine(pos=x_pos, angle=90, pen=pen, movable=False)
                        self._plot.addItem(ln)
                        self._ring_lines.append(ln)
        finally:
            self._suspend_range_track = False

    def _x_to_r(self, x, idx, lsd, px, wl):
        """Inverse of _r_to_x: current-axis value → radius in px (None if invalid)."""
        if idx == 0 or lsd is None or px in (None, 0):
            return x
        if idx == 1:                                    # 2θ (deg)
            return math.tan(math.radians(x)) * lsd / px
        if idx == 2 and wl:                             # Q (Å⁻¹)
            s = x * wl / (4 * math.pi)
            if abs(s) >= 1.0:
                return None
            two_theta = 2 * math.asin(s)
            return math.tan(two_theta) * lsd / px
        return x

    def _on_plot_clicked(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        vb = self._plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        x = vb.mapSceneToView(event.scenePos()).x()
        r = self._x_to_r(x, self._xaxis.currentIndex(), self._lsd, self._px, self._wl)
        if r is None or r <= 0:
            return
        if self._pick_line is None:
            self._pick_line = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen("#ff30ff", width=1.6))
            self._plot.addItem(self._pick_line)
        self._pick_line.setPos(x)
        self.radiusClicked.emit(float(r))

    def _clear_pick_line(self, *_):
        if self._pick_line is not None:
            self._plot.removeItem(self._pick_line)
            self._pick_line = None

    def _on_xrange_changed(self, _vb, xrange):
        """User zoomed/panned the X axis — remember it so a later replot (e.g.
        from a parameter change) doesn't reset the view back to full range."""
        if self._suspend_range_track:
            return
        self._user_xrange = (float(xrange[0]), float(xrange[1]))

    def _on_yrange_changed(self, _vb, yrange):
        """User zoomed/panned the Y axis — remember it (see _on_xrange_changed)."""
        if self._suspend_range_track:
            return
        self._user_yrange = (float(yrange[0]), float(yrange[1]))

    def _apply_view_limits(self, xmin, xmax, ymin, ymax):
        """Bound pan/zoom to the current profile (+ margin) so the user can't
        scroll/zoom arbitrarily far away from where the data actually is."""
        if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
            return
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        xpad = 0.15 * (xmax - xmin)
        ypad = 0.25 * (ymax - ymin)
        vb = self._plot.getPlotItem().getViewBox()
        vb.setLimits(xMin=max(0.0, xmin - xpad), xMax=xmax + xpad,
                     yMin=ymin - ypad, yMax=ymax + ypad,
                     maxXRange=(xmax - xmin) + 2 * xpad,
                     maxYRange=(ymax - ymin) + 2 * ypad)


def radius_ratio_strain_ue(r_pred, r_obs):
    """Pseudo-strain in microstrain (µε): ``eps = 1 - R_obs/R_pred`` — the
    exact residual the calibration's own LM fit minimizes, and what
    ``post_residual_strain_uE`` (shown elsewhere in this app, e.g. the
    Results tab and the calibration log) already reports. See
    ``midas_calibrate_v2/loss/pseudo_strain.py``: "Per-spot pseudo-strain
    residual: 1 - R_obs / R_pred. This is the v1 calibrant cost."

    R_obs already carries the full geometry + distortion + parallax model
    (``pixel_to_REta``), and R_pred is the exact ideal Bragg radius from the
    calibrant's known d-spacing — so this ratio is a sound, self-consistent
    strain measure, not a raw-pixel approximation. It is not identical to
    the true ``(d_obs - d_pred)/d_pred`` at large angles (departs by ~17% at
    this app's default 2θ_max of 28°, a magnitude difference — the SIGN
    matches at all angles, verified by Taylor expansion against the exact
    Bragg relation), but it matches what the calibration actually optimizes
    rather than introducing a second, differently-defined "strain".
    """
    r_pred = np.asarray(r_pred, dtype=float)
    r_obs = np.asarray(r_obs, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1e6 * (1.0 - r_obs / r_pred)


def ring_azimuth_residual(cake_2d, r_axis_px, ring_radii_px, window_px: float = 8.0):
    """Per-(η row, ring) radial residual Δr = r_obs − r_pred — the same
    windowed-local-peak algorithm as :class:`ResidualBarChart` (below), run
    independently for each η row of a cake instead of once on the
    azimuthally-collapsed profile. This is the actual per-ring deviation from
    the calibrant's ideal ring position (tied to its real d-spacings),
    resolved by azimuth — as opposed to :func:`StrainCakeViewer`'s other mode,
    which rebins the backend's whole-detector RBF-smoothed residual model.

    Returns ``(grid, kept_radii)``: ``grid`` has shape
    ``(n_eta, len(kept_radii))``, NaN where an η row had no signal in a
    ring's window (e.g. that azimuth falls outside the detector at that
    radius); ``kept_radii`` is the subset of ``ring_radii_px`` that had at
    least one in-window R-axis sample, matching the ring-dropping
    ``ResidualBarChart`` already does.

    The peak position is parabolic-interpolated around the discrete argmax
    (3-point quadratic fit through it and its two neighbors), not just the
    bin centre the plain argmax lands on. ``ResidualBarChart`` gets away
    with a raw argmax because it acts on the azimuthally-**averaged**
    profile — pooling the whole ring's signal smooths the result enough
    that R-bin quantization isn't visible. Averaged over only one η row's
    worth of pixels, that same discretization dominates: without subpixel
    refinement, adjacent η rows mostly round to the *same* R-bin (a flat,
    uniform-looking ring) with occasional whole-bin-width jumps between
    rows that cross a boundary — not genuine azimuthal structure.
    """
    r_axis = np.asarray(r_axis_px, dtype=float)
    cake = np.asarray(cake_2d, dtype=float)
    n_eta = cake.shape[0]
    sels, kept_radii = [], []
    for r_pred in ring_radii_px:
        sel = np.abs(r_axis - r_pred) <= window_px
        if sel.any():
            sels.append(sel)
            kept_radii.append(float(r_pred))
    if not sels:
        return np.zeros((n_eta, 0)), []
    grid = np.full((n_eta, len(sels)), np.nan)
    rows = np.arange(n_eta)
    for k, (sel, r_pred) in enumerate(zip(sels, kept_radii)):
        local_r = r_axis[sel]
        local_i = cake[:, sel]                    # (n_eta, n_sel)
        # A cake bin with no pixel coverage reads as exact 0 (integrate_*'s
        # normalize=True divides by clamp(min=1e-12) there), not NaN — same
        # convention CakeViewer._redisplay already uses to spot empty bins.
        has_signal = np.any(local_i != 0, axis=1)
        if not has_signal.any():
            continue
        n_sel = local_i.shape[1]
        arg = np.argmax(local_i, axis=1)
        if n_sel < 3:
            r_obs = local_r[arg]   # window too narrow to fit a parabola through
        else:
            # A neighbor on each side is needed for the fit — clip the centre
            # index inward by one bin at the window's edges (rare in an
            # 8 px-wide window) rather than skip refinement there.
            c = np.clip(arg, 1, n_sel - 2)
            y0, y1, y2 = local_i[rows, c - 1], local_i[rows, c], local_i[rows, c + 1]
            denom = y0 - 2 * y1 + y2
            with np.errstate(invalid="ignore", divide="ignore"):
                offset = np.where(denom != 0, 0.5 * (y0 - y2) / denom, 0.0)
            offset = np.clip(offset, -1.0, 1.0)   # guard a near-flat/noisy local max
            bin_width = local_r[1] - local_r[0]   # r_axis bins are uniform-width
            r_obs = local_r[c] + offset * bin_width
        grid[has_signal, k] = r_obs[has_signal] - r_pred
    return grid, kept_radii


def collapsed_profile_ring_residual(r_axis_px, profile, ring_radii_px,
                                    window_px: float = 8.0):
    """Δr = r_obs − r_pred per ring, from a single peak search on the
    azimuthally-**collapsed** profile (no η resolution) — higher SNR than
    :func:`ring_azimuth_residual`'s per-η peak search, at the cost of
    losing azimuthal detail. Shared by :class:`ResidualBarChart` and
    :class:`RingResidualViewer`'s 1-D "Peak of collapsed profile" method.

    Returns ``(resid_px, kept_radii)`` — a ring with no in-window sample
    is dropped from both, mirroring ``ring_azimuth_residual``'s contract.
    Plain argmax (no subpixel refinement): the collapsed profile already
    pools the whole ring's signal, smoothing the result enough that R-bin
    quantization isn't visible (see ``ring_azimuth_residual``'s docstring
    for why that's not true per-η-row).
    """
    r_axis = np.asarray(r_axis_px, dtype=float)
    prof = np.asarray(profile, dtype=float)
    if r_axis.size == 0 or not ring_radii_px:
        return np.array([]), []
    resid, kept_radii = [], []
    for r_pred in ring_radii_px:
        sel = np.abs(r_axis - r_pred) <= window_px
        if not sel.any():
            continue
        local_r = r_axis[sel]
        local_i = prof[sel]
        if not np.isfinite(local_i).any():
            continue
        r_obs = float(local_r[int(np.nanargmax(local_i))])
        resid.append(r_obs - float(r_pred))
        kept_radii.append(float(r_pred))
    return np.array(resid, dtype=float), kept_radii


# ═════════════════════════════════════════════════════════════════════════════
#  ResidualBarChart  (NEW — per-ring radial residual after calibration)
# ═════════════════════════════════════════════════════════════════════════════

class ResidualBarChart(QtWidgets.QWidget):
    """Bar chart of Δr = r_observed − r_predicted (px) for each predicted ring.

    Self-contained: the observed radius is the local profile peak within a window
    around each predicted radius.  No dependence on pipeline internals, so it
    works identically for every calibration pipeline.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(2)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Per-ring radial residual  Δr = r_obs − r_pred"))
        bar.addStretch(1)
        self._stat = QtWidgets.QLabel("")
        self._stat.setStyleSheet("color:#aaa;font-size:10px")
        bar.addWidget(self._stat)
        layout.addLayout(bar)

        self._plot = pg.PlotWidget(background="k")
        self._plot.setLabel("left", "Δr (px)")
        self._plot.setLabel("bottom", "ring index")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#888", width=1))
        self._plot.addItem(self._zero)
        self._bars = None
        layout.addWidget(self._plot, stretch=1)

    def set_data(self, r_axis_px, profile, ring_radii_px, window_px: float = 8.0):
        if self._bars is not None:
            self._plot.removeItem(self._bars)
            self._bars = None
        h, kept_radii = collapsed_profile_ring_residual(
            r_axis_px, profile, ring_radii_px, window_px)
        if not kept_radii:
            self._stat.setText("no data" if not ring_radii_px else
                               "no rings matched the profile")
            return

        # x = index into the ORIGINAL ring_radii_px list (gaps where a ring
        # was dropped), so ring spacing/ordering stays recognizable.
        idxs = [k for k, r_pred in enumerate(ring_radii_px) if r_pred in kept_radii]
        x = np.array(idxs, dtype=float)
        self._bars = pg.BarGraphItem(x=x, height=h, width=0.6,
                                     brush=pg.mkBrush("#5aa0e0"))
        self._plot.addItem(self._bars)
        rms = float(np.sqrt(np.mean(h ** 2)))
        self._stat.setText(f"{len(h)} rings | RMS Δr = {rms:.3f} px")


# ═════════════════════════════════════════════════════════════════════════════
#  DistortionTable  (NEW — read-only 15-coefficient grid)
# ═════════════════════════════════════════════════════════════════════════════

class DistortionTable(QtWidgets.QTableWidget):
    """Compact read-only display of the 15 distortion coefficients."""

    def __init__(self, parent=None):
        super().__init__(len(DISTORTION_NAMES), 2, parent)
        self.setHorizontalHeaderLabels(["coeff", "value"])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.horizontalHeader().setStretchLastSection(True)
        self.setColumnWidth(0, 70)
        self.setFixedHeight(150)
        for i, name in enumerate(DISTORTION_NAMES):
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.setItem(i, 1, QtWidgets.QTableWidgetItem("—"))

    def set_distortion(self, distortion: dict):
        for i, name in enumerate(DISTORTION_NAMES):
            val = distortion.get(name)
            txt = f"{val:.6g}" if val is not None else "—"
            self.item(i, 1).setText(txt)


# ═════════════════════════════════════════════════════════════════════════════
#  CorrectionFlagsWidget  (NEW — reusable physics-corrections panel)
# ═════════════════════════════════════════════════════════════════════════════

class CorrectionFlagsWidget(QtWidgets.QGroupBox):
    """Polarization + solid-angle correction toggles with sub-controls."""

    def __init__(self, parent=None):
        super().__init__("Physics corrections", parent)
        form = QtWidgets.QFormLayout(self); form.setSpacing(4)

        self.polar_check = QtWidgets.QCheckBox("Polarization")
        self.polar_check.setToolTip(
            "Apply the polarization correction (synchrotron horizontal plane).")
        self.pol_fraction = _fspin(0.0, 1.0, 3, 0.99)
        self.pol_fraction.setFixedWidth(80)
        self.pol_plane = _fspin(-180.0, 180.0, 1, 0.0, "°")
        self.pol_plane.setFixedWidth(80)
        form.addRow(self.polar_check)
        form.addRow(_twocol("frac:", self.pol_fraction, "plane η:", self.pol_plane))

        self.solid_check = QtWidgets.QCheckBox("Solid-angle (tilt-aware)")
        self.solid_check.setToolTip(
            "Divide by the per-pixel solid angle (accounts for detector tilt).")
        form.addRow(self.solid_check)

        for w in (self.pol_fraction, self.pol_plane):
            w.setEnabled(False)
        self.polar_check.toggled.connect(self.pol_fraction.setEnabled)
        self.polar_check.toggled.connect(self.pol_plane.setEnabled)

    def any_enabled(self) -> bool:
        return self.polar_check.isChecked() or self.solid_check.isChecked()

    def build_corrections(self):
        """Return (polarization, solid_angle) correction objects or None each."""
        pol = sa = None
        if self.polar_check.isChecked():
            from midas_integrate_v2 import PolarizationCorrection
            pol = PolarizationCorrection(
                pol_fraction=self.pol_fraction.value(),
                pol_plane_eta_deg=self.pol_plane.value())
        if self.solid_check.isChecked():
            from midas_integrate_v2 import SolidAngleCorrection
            sa = SolidAngleCorrection()
        return pol, sa

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def get_state(self) -> dict:
        return {
            "polar_check": self.polar_check.isChecked(),
            "pol_fraction": self.pol_fraction.value(),
            "pol_plane": self.pol_plane.value(),
            "solid_check": self.solid_check.isChecked(),
        }

    def set_state(self, state: dict):
        if not state:
            return
        if "pol_fraction" in state:
            self.pol_fraction.setValue(float(state["pol_fraction"]))
        if "pol_plane" in state:
            self.pol_plane.setValue(float(state["pol_plane"]))
        if "polar_check" in state:
            self.polar_check.setChecked(bool(state["polar_check"]))
        if "solid_check" in state:
            self.solid_check.setChecked(bool(state["solid_check"]))


class OutputFormatSelector(QtWidgets.QWidget):
    """Multi-select output-format picker for the batch tabs — one checkbox per
    ``constants.OUTPUT_FORMATS`` entry, so a run can write several formats at
    once instead of picking exactly one. ``DEFAULT_OUTPUT_FORMAT`` starts
    checked.

    The checkboxes live behind a clickable "Output format" button (a popup
    menu) rather than always taking up vertical space in the layout — same
    click-to-see-options interaction as ``helpers.make_calib_values_button``.
    The button's own text is kept in sync with the current selection so the
    choice is visible without opening the menu. Kept out of the generic
    ``_state_widgets()``/``widgets_to_dict`` save-state path (like
    ``CorrectionFlagsWidget``) since that only understands single-value
    widgets — callers wire ``get_state()``/``set_state()`` directly instead.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        from midas_gui.constants import OUTPUT_FORMATS, DEFAULT_OUTPUT_FORMAT
        # key -> short display name (label text before the parenthesised
        # description), for the compact button text.
        self._short_names = {key: label.split("(")[0].strip()
                              for label, key in OUTPUT_FORMATS.items()}
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)

        self._btn = QtWidgets.QToolButton()
        self._btn.setAutoRaise(True)
        self._btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn.setStyleSheet(
            "QToolButton { border: none; padding: 0 2px; color: #4da3ff; }"
            "QToolButton::menu-indicator { image: none; }")
        f = self._btn.font(); f.setUnderline(True); self._btn.setFont(f)

        menu = QtWidgets.QMenu(self._btn)
        host = QtWidgets.QWidget(menu)
        v = QtWidgets.QVBoxLayout(host)
        v.setContentsMargins(10, 8, 10, 8); v.setSpacing(4)
        self._checks: dict = {}   # key -> QCheckBox
        for label, key in OUTPUT_FORMATS.items():
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(key == DEFAULT_OUTPUT_FORMAT)
            cb.toggled.connect(self._sync_button_text)
            v.addWidget(cb)
            self._checks[key] = cb
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(host)
        menu.addAction(action)
        self._btn.setMenu(menu)

        h.addWidget(self._btn)
        h.addStretch(1)
        self._sync_button_text()

    def _sync_button_text(self):
        keys = self.checked_keys()
        if not keys:
            self._btn.setText("Output format: none selected ▾")
        else:
            names = ", ".join(self._short_names[k] for k in self._checks if k in keys)
            self._btn.setText(f"Output format: {names} ▾")

    def checked_keys(self) -> list:
        return [key for key, cb in self._checks.items() if cb.isChecked()]

    def get_state(self) -> dict:
        return {"checked": self.checked_keys()}

    def set_state(self, state) -> None:
        """``state`` is either the ``get_state()`` dict shape, or a bare
        ``list[str]`` of keys (as stored on a project integration attempt —
        see ``project.integrate_attempt_gui_fields``)."""
        if not state:
            return
        keys = state.get("checked") if isinstance(state, dict) else state
        if not keys:
            return
        keys = set(keys)
        for key, cb in self._checks.items():
            cb.blockSignals(True)
            cb.setChecked(key in keys)
            cb.blockSignals(False)
        self._sync_button_text()


def _fmt_source_desc(desc: dict) -> str:
    """Human-readable "Import from…" menu label for one
    ``data_bridge.DataSourceRegistry`` descriptor — shared by every
    Data/Dark/Bright/Background selector (single-detector and Hydra)."""
    if desc["kind"] == "buffer":
        n = len(desc["provider"]._buffer) if desc["provider"]._buffer else 0
        return f"{desc['label']}: Buffer ({n} frames)"
    path = desc["path"]
    if isinstance(path, list):
        return f"{desc['label']}: {len(path)} files"
    return f"{desc['label']}: {path}"


class FieldSelector(QtWidgets.QGroupBox):
    """Compact reusable dark / bright / background field picker.

    A checkable group; its body is hidden while unchecked so three of these stay
    compact.  Browsing a file or folder (⋯ menu) auto-computes the field: a single
    file, a folder / *.tif glob, or an HDF5 dataset averaged over an index range
    that is clamped to the number of frames available.  The bright variant adds a
    divide / subtract mode combo.  ``get_field()`` → computed field (or None);
    ``get_mode()`` → "divide" | "subtract".
    """
    fieldReady = QtCore.pyqtSignal()

    def __init__(self, title, parent=None, *, with_mode=False,
                 default_dataset="exchange/data"):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(False)
        self._with_mode = with_mode
        self._field = None
        self._worker = None
        self._registry = None          # DataSourceRegistry, set by set_registry()
        self._exclude_label = None     # owning panel's registry label — skip its own entry
        self._buffer_snapshot_file = None   # temp .h5 from importing another tab's buffer
        self._explicit_paths = None    # list[str], set by a Browse… "Multiple files"/"stem" pick

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 4); outer.setSpacing(2)
        self._body = QtWidgets.QWidget()
        self._body.setVisible(False)                       # collapsed until enabled
        self.toggled.connect(self._body.setVisible)
        outer.addWidget(self._body)
        v = QtWidgets.QVBoxLayout(self._body)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)

        # Path + browse (file/folder popup menu)
        self._path_ed = QtWidgets.QLineEdit()
        self._path_ed.setPlaceholderText("file / folder / .h5")
        self._path_ed.textChanged.connect(self._on_path_changed)
        self._path_ed.editingFinished.connect(self._update_frame_limit)
        browse = QtWidgets.QToolButton()
        browse.setText("⋯"); browse.setFixedWidth(28)
        browse.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(browse)
        menu.addAction("Browse…", self._open_browse_dialog)
        menu.addSeparator()
        self._import_menu = menu.addMenu("Import from…")
        self._import_menu.aboutToShow.connect(self._populate_import_menu)
        browse.setMenu(menu)
        pr = QtWidgets.QHBoxLayout(); pr.setSpacing(4)
        pr.addWidget(self._path_ed); pr.addWidget(browse)
        v.addLayout(pr)

        # HDF5 dataset dropdown (row hidden unless an HDF5 path is selected)
        self._ds_row = QtWidgets.QWidget()
        dr = QtWidgets.QHBoxLayout(self._ds_row); dr.setContentsMargins(0, 0, 0, 0); dr.setSpacing(4)
        self._ds_combo = _NoScrollComboBox()
        self._ds_combo.setEditable(True); self._ds_combo.setEditText(default_dataset)
        self._ds_combo.currentIndexChanged.connect(self._update_frame_limit)
        dr.addWidget(QtWidgets.QLabel("Dataset:")); dr.addWidget(self._ds_combo, 1)
        self._ds_row.setVisible(False)
        v.addWidget(self._ds_row)

        # Index range (clamped to available frames) + optional mode, on one row
        self._start = _NoScrollSpinBox(); self._start.setRange(0, 0); self._start.setFixedWidth(50)
        self._end = _NoScrollSpinBox(); self._end.setRange(0, 0); self._end.setFixedWidth(50)
        self._end.setToolTip("Last frame index to average (inclusive).")
        self._nfr_lbl = QtWidgets.QLabel("")
        self._nfr_lbl.setStyleSheet("color:#9a9a9a;font-size:10px")
        ir = QtWidgets.QHBoxLayout(); ir.setSpacing(4)
        ir.addWidget(QtWidgets.QLabel("avg")); ir.addWidget(self._start)
        ir.addWidget(QtWidgets.QLabel("–")); ir.addWidget(self._end)
        ir.addWidget(self._nfr_lbl)
        if with_mode:
            self._mode = _NoScrollComboBox()
            self._mode.addItems(["Flat-field divide", "Subtract"])
            self._mode.setFixedWidth(104)
            ir.addStretch(1); ir.addWidget(self._mode)
        else:
            self._mode = None
            ir.addStretch(1)
        v.addLayout(ir)

        # Compute + status
        self._compute_btn = QtWidgets.QPushButton("Compute field")
        self._compute_btn.clicked.connect(self._compute)
        v.addWidget(self._compute_btn)
        self._status = QtWidgets.QLabel("Not computed.")
        self._status.setStyleSheet("color:#9a9a9a;font-size:10px"); self._status.setWordWrap(True)
        v.addWidget(self._status)

    # ── helpers ───────────────────────────────────────────────────
    def _raw_source(self):
        """The current source: an explicit ``list[str]`` from a Browse…
        "Multiple files"/"Files sharing a stem" pick, else the plain path
        text (file / folder / glob)."""
        return self._explicit_paths if self._explicit_paths else self._path_ed.text().strip()

    def _kind(self) -> str:
        return source_kind(self._raw_source())

    def _dataset(self) -> str:
        return self._ds_combo.currentText().split("   ")[0].strip() or "exchange/data"

    def _set_explicit_paths(self, paths):
        """``paths`` is a resolved ``list[str]`` (Multiple files / stem
        match) or None to fall back to the plain path text. Uses
        blockSignals so the summary text it writes doesn't itself clear
        ``_explicit_paths`` via ``_on_path_changed``."""
        self._explicit_paths = paths
        self._path_ed.blockSignals(True)
        if paths:
            self._path_ed.setText(display_text_for_paths(paths))
            self._path_ed.setToolTip("\n".join(paths))
        else:
            self._path_ed.setToolTip("")
        self._path_ed.blockSignals(False)

    def _open_browse_dialog(self):
        dlg = BrowseFilesDialog(self, title=f"Select {self.title()}",
                                start_dir=self._path_ed.text().strip())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode = dlg.mode()
        if mode == "file":
            paths = dlg.paths()
            if not paths:
                return
            self._set_explicit_paths(None)
            self._path_ed.setText(paths[0])
        elif mode == "folder":
            self._set_explicit_paths(None)
            self._path_ed.setText(dlg.folder())
        else:  # "files" or "stem" — both resolve to an explicit file list
            paths = dlg.paths()
            if not paths:
                return
            self._set_explicit_paths(paths)
        self._update_frame_limit()
        self._compute()

    # ── cross-tab import (data_bridge.DataSourceRegistry) ───────────
    def _field_kind(self) -> str:
        """This selector's type ("dark"/"bright"/"background"), derived from
        its title — used to tag its own descriptor and to filter which
        registry entries its "Import from…" menu offers."""
        return self.title().strip().lower()

    def set_registry(self, registry, *, exclude_label=None):
        """Let this selector's "Import from…" menu offer the same-type field
        (e.g. Dark ↔ Dark) currently loaded in any *other* tab bound to
        `registry` — the same registry the Data card's own menu uses.
        `exclude_label` is the owning panel's own registry label, so a
        selector doesn't offer to import its own current value."""
        self._registry = registry
        self._exclude_label = exclude_label

    def describe_source(self, label: str):
        """Export this field as an importable source of type `_field_kind()`
        (if enabled and pointing at a path) — mirrors
        DataLoaderPanel.describe_source() so another tab's same-type selector
        can pull it via the registry. `path` may be a `list[str]` if this
        field's source is an explicit Multiple-files/stem pick."""
        if not self.isChecked():
            return None
        raw = self._raw_source()
        if not raw:
            return None
        return {"kind": "path", "path": raw,
                "dataset": self._dataset() if (isinstance(raw, str) and is_h5(raw)) else None,
                "field": self._field_kind(), "label": label}

    def _populate_import_menu(self):
        menu = self._import_menu
        menu.clear()
        sources = (self._registry.available(field=self._field_kind())
                   if self._registry is not None else [])
        sources = [d for d in sources if d.get("label") != self._exclude_label]
        if not sources:
            menu.addAction("(nothing loaded elsewhere)").setEnabled(False)
            return
        for desc in sources:
            menu.addAction(_fmt_source_desc(desc), lambda d=desc: self._apply_imported_source(d))

    def _apply_imported_source(self, desc: dict):
        if desc["kind"] == "buffer":
            self._import_buffer(desc["provider"])
            return
        path = desc["path"]
        if isinstance(path, list):
            self._set_explicit_paths(path)
        else:
            self._set_explicit_paths(None)
            self._path_ed.setText(path)
        if desc.get("dataset"):
            self._ds_combo.setEditText(desc["dataset"])
        self._update_frame_limit()
        self._compute()

    def _import_buffer(self, provider) -> None:
        """Snapshot another panel's frozen ring buffer to a temp HDF5 file
        and point at that, exactly like the Data card's buffer import."""
        with provider._buffer_lock:
            frames = list(provider._buffer) if (provider._buffer_frozen and provider._buffer) else None
        if not frames:
            QtWidgets.QMessageBox.warning(self, "No buffer", "Source buffer is empty.")
            return
        old = self._buffer_snapshot_file
        path = new_temp_h5_path()
        save_stack_h5(path, frames, dataset="buffer/data")
        self._buffer_snapshot_file = path
        self._path_ed.setText(path)
        self._ds_combo.setEditText("buffer/data")
        self._update_frame_limit()
        self._compute()
        if old is not None:
            import os
            try:
                os.unlink(old)
            except OSError:
                pass

    def _on_path_changed(self, p: str):
        from pathlib import Path
        # Fires only for a real text edit — a Browse… explicit-list pick sets
        # its own summary text via blockSignals, so this never races it.
        self._explicit_paths = None
        h5 = is_h5(p)
        self._ds_row.setVisible(h5)
        if h5 and Path(p).exists():
            try:
                items = list_h5_datasets(p)
            except Exception:
                items = []
            if items:
                keep = self._ds_combo.currentText().strip()
                self._ds_combo.blockSignals(True); self._ds_combo.clear()
                for name, shape in items:
                    self._ds_combo.addItem(f"{name}   {tuple(shape)}", name)
                idx = next((i for i in range(self._ds_combo.count())
                            if self._ds_combo.itemData(i) == keep), -1)
                if idx < 0:
                    idx = next((i for i, (n, s) in enumerate(items) if len(s) >= 3), 0)
                self._ds_combo.setCurrentIndex(idx)
                self._ds_combo.blockSignals(False)
        self._update_frame_limit()

    def _count_frames(self) -> int:
        """Number of frames available in the current source (0 if unknown)."""
        from pathlib import Path
        raw = self._raw_source()
        if not raw:
            return 0
        kind = self._kind()
        try:
            if kind == "hdf5":
                import h5py
                if not Path(raw).exists():
                    return 0
                with h5py.File(raw, "r") as f:
                    d = f[self._dataset()]
                    return int(d.shape[0]) if d.ndim >= 3 else 1
            if kind == "folder":
                from midas_gui.helpers import _collect_frame_paths
                return len(_collect_frame_paths(raw))
            if not Path(raw).exists():
                return 0
            if raw.lower().endswith((".tif", ".tiff")):
                import tifffile
                with tifffile.TiffFile(raw) as tf:
                    return len(tf.pages)
            return 1
        except Exception:
            return 0

    def _update_frame_limit(self):
        """Clamp the index spinboxes to the frames actually available."""
        n = self._count_frames()
        if n <= 0:
            self._nfr_lbl.setText("")
            return
        hi = n - 1
        self._nfr_lbl.setText(f"/ {hi}")
        for sp in (self._start, self._end):
            sp.blockSignals(True); sp.setMaximum(hi); sp.blockSignals(False)
        # default the end to the last frame (average the whole stack)
        if self._end.value() == 0 or self._end.value() > hi:
            self._end.blockSignals(True); self._end.setValue(hi); self._end.blockSignals(False)
        if self._start.value() > hi:
            self._start.setValue(hi)

    def _compute(self):
        raw = self._raw_source()
        if not raw:
            self._status.setText("Enter a path first."); return
        from midas_gui.workers import FieldAverageWorker
        self._compute_btn.setEnabled(False)
        self._status.setText("Computing…")
        self._worker = FieldAverageWorker(
            self._kind(), raw, self._dataset(),
            self._start.value(), self._end.value(), parent=self)
        self._worker.finished.connect(self._on_computed)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_computed(self, field):
        self._field = field
        self._compute_btn.setEnabled(True)
        self._status.setText(f"Computed — {field.shape}  "
                             f"[{float(field.min()):.4g}, {float(field.max()):.4g}]")
        self.fieldReady.emit()

    def _on_failed(self, msg):
        self._field = None
        self._compute_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg.strip().splitlines()[-1][:120]}")

    # ── public API ────────────────────────────────────────────────
    def is_enabled(self) -> bool:
        return self.isChecked()

    def has_pending(self) -> bool:
        return self.isChecked() and self._field is None

    def get_field(self):
        return self._field if self.isChecked() else None

    def get_mode(self) -> str:
        if self._mode is None:
            return "divide"
        return "divide" if self._mode.currentIndex() == 0 else "subtract"

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def get_state(self) -> dict:
        st = {
            "checked": self.isChecked(),
            "path": self._path_ed.text(),
            "dataset": self._ds_combo.currentText(),
            "start": self._start.value(),
            "end": self._end.value(),
        }
        if self._explicit_paths:
            st["explicit_paths"] = list(self._explicit_paths)
        if self._mode is not None:
            st["mode"] = self._mode.currentIndex()
        return st

    def set_state(self, state: dict):
        """Restore path/range/mode and re-compute the field if it was enabled —
        the "auto re-trigger this tab's own load pipeline" behavior applied to
        dark/bright/background fields specifically."""
        if not state:
            return
        explicit = state.get("explicit_paths")
        path = state.get("path", "")
        if explicit:
            self._set_explicit_paths(list(explicit))
        elif path:
            self._set_explicit_paths(None)
            self._path_ed.setText(path)
        ds = state.get("dataset")
        if ds:
            self._ds_combo.setEditText(ds)
        self._update_frame_limit()
        # Widen the range if needed so a saved value isn't silently clamped to 0
        # when the file couldn't be probed yet (e.g. path not found at restore time).
        hi = max(self._start.maximum(), int(state.get("start", 0)), int(state.get("end", 0)))
        self._start.setMaximum(hi); self._end.setMaximum(hi)
        if "start" in state:
            self._start.setValue(int(state["start"]))
        if "end" in state:
            self._end.setValue(int(state["end"]))
        if self._mode is not None and "mode" in state:
            self._mode.setCurrentIndex(int(state["mode"]))
        self.setChecked(bool(state.get("checked", False)))
        if self.isChecked() and (explicit or path):
            self._compute()


class MaskSelector(QtWidgets.QGroupBox):
    """Multiple mask sources unioned into one composite mask.

    Rows are mask files/folders plus an optional auto-managed "Tab 1 mask" row
    (set via :meth:`set_tab1_mask`), always kept at the top of the list.  Each
    row has a checkbox to include/exclude it from the union without deleting
    it.  ``composite_mask()`` OR's every *checked* source (each loaded → ``!=
    0``; a folder OR's all its frames).  Masked pixels are the ones a caller
    should zero / ignore.
    """
    maskChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Mask", parent)
        self._sources: list = []          # dicts: {kind, path, mask(cache), row(QWidget)}
        self._tab1_mask = None
        self._mask_warning = None         # set by composite_mask() when a source is dropped
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 4, 6, 6); v.setSpacing(4)
        self._list = QtWidgets.QVBoxLayout(); self._list.setSpacing(2)
        v.addLayout(self._list)
        row = QtWidgets.QHBoxLayout(); row.setSpacing(4)
        bf = QtWidgets.QPushButton("Add file…"); bf.clicked.connect(self._add_file)
        bd = QtWidgets.QPushButton("Add folder…"); bd.clicked.connect(self._add_folder)
        row.addWidget(bf); row.addWidget(bd)
        v.addLayout(row)
        self._status = QtWidgets.QLabel("No mask.")
        self._status.setStyleSheet("color:#9a9a9a;font-size:10px"); self._status.setWordWrap(True)
        v.addWidget(self._status)

    def _add_row(self, entry, label, at_top=False):
        rw = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(rw); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        chk = QtWidgets.QCheckBox()
        chk.setChecked(entry.get("enabled", True))
        chk.setToolTip("Include this mask in the composite (unchecking keeps it in the "
                        "list without deleting it).")
        chk.toggled.connect(lambda checked, e=entry: self._on_enabled_toggled(e, checked))
        lbl = QtWidgets.QLabel(label); lbl.setStyleSheet("font-size:10px")
        x = QtWidgets.QToolButton(); x.setText("✕"); x.setFixedWidth(22)
        x.clicked.connect(lambda: self._remove(entry))
        h.addWidget(chk); h.addWidget(lbl, 1); h.addWidget(x)
        entry["row"] = rw
        entry["checkbox"] = chk
        if at_top:
            self._list.insertWidget(0, rw)
        else:
            self._list.addWidget(rw)

    def _on_enabled_toggled(self, entry, checked):
        entry["enabled"] = checked
        self._mask_warning = None  # stale vs. the change just made; composite_mask() recomputes it
        self._refresh(); self.maskChanged.emit()

    def _remove(self, entry):
        if entry in self._sources:
            self._list.removeWidget(entry["row"]); entry["row"].deleteLater()
            self._sources.remove(entry)
            if entry["kind"] == "tab1":
                self._tab1_mask = None
            self._mask_warning = None
            self._refresh(); self.maskChanged.emit()

    def _add_source(self, kind, path, enabled=True):
        from pathlib import Path
        entry = {"kind": kind, "path": path, "mask": None, "row": None, "enabled": enabled}
        self._add_row(entry, f"{kind}: {Path(path).name}")
        self._sources.append(entry)
        self._mask_warning = None
        self._refresh(); self.maskChanged.emit()

    def add_file_source(self, path):
        """Programmatically add a mask *file* row (e.g. a tab's default mask).

        No-op on a blank path or one already present, so callers can wire a default
        idempotently without duplicating the row."""
        path = str(path or "").strip()
        if not path:
            return
        if any(e["kind"] == "file" and e["path"] == path for e in self._sources):
            return
        self._add_source("file", path)

    def _add_file(self):
        p = _browse(self, "Add mask file", "Images (*.tif *.tiff *.h5 *.hdf5);;All (*)")
        if p:
            self._add_source("file", p)

    def _add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Add mask folder")
        if d:
            self._add_source("folder", d)

    def set_tab1_mask(self, mask):
        """Add / update / remove the auto-managed Tab-1 mask row."""
        self._tab1_mask = None if mask is None else (np.asarray(mask) != 0)
        existing = next((e for e in self._sources if e["kind"] == "tab1"), None)
        if self._tab1_mask is None:
            if existing:
                self._remove(existing)
            return
        if existing is None:
            entry = {"kind": "tab1", "path": None, "mask": None, "row": None, "enabled": True}
            n = int(self._tab1_mask.sum())
            self._add_row(entry, f"Tab 1 mask ({n:,} px)", at_top=True)
            self._sources.insert(0, entry)
        self._mask_warning = None
        self._refresh(); self.maskChanged.emit()

    def _load_source(self, entry):
        if entry["kind"] == "tab1":
            return self._tab1_mask
        if entry["mask"] is not None:
            return entry["mask"]
        try:
            if entry["kind"] == "folder":
                acc = None
                for p in _collect_frame_paths(entry["path"]):
                    a = _load_image(p); a = a[0] if a.ndim == 3 else a
                    b = (a != 0)
                    acc = b if acc is None else (acc | b)
                m = acc
            else:
                a = _load_image(entry["path"]); a = a[0] if a.ndim == 3 else a
                m = (a != 0)
            entry["mask"] = m
            return m
        except Exception:
            return None

    @staticmethod
    def _source_label(entry):
        if entry["kind"] == "tab1":
            return "Tab 1 mask"
        from pathlib import Path
        return f"{entry['kind']}: {Path(entry['path']).name}"

    def composite_mask(self):
        """uint8 union of all *enabled* sources (1 = masked), or None if empty.

        A source is dropped (and flagged in the status label via _refresh())
        if it fails to load, or if its shape doesn't match the first
        successfully-loaded source's shape — previously this happened
        silently, which could make analysis quietly ignore a mask the user
        believed was active.
        """
        out = None
        dropped = []
        for e in self._sources:
            if not e.get("enabled", True):
                continue
            m = self._load_source(e)
            if m is None:
                dropped.append(self._source_label(e))
                continue
            m = np.asarray(m) != 0
            if out is None:
                out = m
            elif out.shape == m.shape:
                out = out | m
            else:
                dropped.append(f"{self._source_label(e)} (shape {m.shape} != {out.shape})")
        self._mask_warning = (
            f"Skipped {len(dropped)} source(s): " + "; ".join(dropped)
        ) if dropped else None
        self._refresh()
        return out.astype(np.uint8) if out is not None else None

    def has_live_mask_source(self) -> bool:
        """True iff the composite mask currently includes an enabled "tab1"
        (Mask-tab-drawn/computed, live) source — i.e. it cannot be fully
        reconstructed from file paths alone, so a caller (e.g. logging a
        Project attempt) should embed the raw array rather than relying on
        the file/folder sources' own path+hash provenance."""
        return any(e["kind"] == "tab1" and e.get("enabled", True) for e in self._sources)

    def _refresh(self):
        n = len(self._sources)
        if self._mask_warning:
            self._status.setText(
                (f"{n} mask source(s). " if n else "") + self._mask_warning)
            self._status.setStyleSheet("color:#e0a030;font-size:10px")
        else:
            self._status.setText(f"{n} mask source(s)" if n else "No mask.")
            self._status.setStyleSheet("color:#9a9a9a;font-size:10px")

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def get_state(self) -> dict:
        """Serializes file/folder sources only — the "tab1" source is owned
        and re-supplied at runtime by the tab via :meth:`set_tab1_mask`."""
        return {"sources": [{"kind": e["kind"], "path": e["path"],
                              "enabled": e.get("enabled", True)}
                             for e in self._sources if e["kind"] != "tab1"]}

    def set_state(self, state: dict):
        if not state:
            return
        for e in [e for e in self._sources if e["kind"] != "tab1"]:
            self._remove(e)
        for s in state.get("sources", []):
            kind, path = s.get("kind"), s.get("path")
            if kind and path:
                self._add_source(kind, path, enabled=s.get("enabled", True))


class IntensityStatsPanel(QtWidgets.QGroupBox):
    """Compact intensity-distribution readout + histogram for the Data Viewer.

    Display-only: the owning tab feeds it the (already corrected + masked) pixel
    values via :meth:`set_data`.  A scope selector (Current frame / All frames)
    emits :data:`scopeChanged` so the tab can recompute the right pixel set.
    """
    scopeChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Intensity statistics", parent)
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(6, 3, 6, 4); v.setSpacing(3)

        top = QtWidgets.QHBoxLayout(); top.setSpacing(6)
        self._scope = _NoScrollComboBox()
        self._scope.addItem("Current frame", "current")
        self._scope.addItem("All frames", "all")
        self._scope.setToolTip("Statistics for the selected frame, or combined over "
                               "all frames in the stack/folder.")
        self._scope.currentIndexChanged.connect(lambda _=0: self.scopeChanged.emit())
        top.addWidget(self._scope, 1)
        self._logchk = QtWidgets.QCheckBox("log y"); self._logchk.setChecked(True)
        self._logchk.toggled.connect(self._redraw_hist)
        top.addWidget(self._logchk)
        v.addLayout(top)

        self._plot = pg.PlotWidget(background="#2b2e35")
        # Min height only (no max) so the splitter above can grow the histogram.
        self._plot.setMinimumHeight(90)
        self._plot.setLabel("bottom", "intensity", **{"color": "#d0d0d0", "font-size": "12pt"})
        self._plot.setLabel("left", "log(count+1)", **{"color": "#d0d0d0", "font-size": "12pt"})
        for ax in ("bottom", "left"):
            self._plot.getAxis(ax).setTextPen("#c8c8c8")
            self._plot.getAxis(ax).setPen("#8a8a8a")
        self._curve = self._plot.plot(
            [], [], stepMode="center", fillLevel=0,
            brush=(90, 140, 220, 150), pen=pg.mkPen("#6ea8ff"))
        v.addWidget(self._plot)

        # Manual axis limits reuse the axis's own native right-click "Manual"
        # min/max fields rather than a separate row of spin boxes.
        self._manual_mode = False
        self._manual_range: Optional[tuple] = None   # (xmin, xmax, ymin, ymax)
        self._btn_auto, self._btn_manual = _add_auto_manual_buttons(
            self._plot, self._on_auto_clicked, self._on_manual_clicked)
        _install_manual_axis_capture(self._plot, self._on_manual_range_edited)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(_mono_font(8))
        # The readout is a fixed, short list of lines — show it in full (no inner
        # scrollbar) and let its height track the content. That leaves the plot as
        # the only flexible child, so the splitter above resizes the plot while the
        # whole panel (plot + text) moves as a unit.
        self._text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self._text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._text.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._text.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                 QtWidgets.QSizePolicy.Fixed)
        self._text.setStyleSheet(
            "QPlainTextEdit { background:#23252b; color:#d6d6d6; border:1px solid #444; }")
        v.addWidget(self._text)
        self._fit_text_height()
        self._hist = None

    def _fit_text_height(self):
        """Size the readout box to fit exactly its current line count."""
        fm = self._text.fontMetrics()
        n = max(1, self._text.document().blockCount())
        m = self._text.contentsMargins()
        doc_m = int(self._text.document().documentMargin()) * 2
        fr = self._text.frameWidth() * 2
        self._text.setFixedHeight(
            n * fm.lineSpacing() + doc_m + fr + m.top() + m.bottom() + 4)

    def scope(self) -> str:
        return self._scope.currentData()

    def set_scope_enabled(self, on: bool):
        self._scope.setEnabled(on)

    def _on_manual_range_edited(self, xmin, xmax, ymin, ymax):
        """The user set exact limits via an axis's native right-click
        "Manual" min/max fields (pyqtgraph already applied it to the view) —
        remember it as the held manual range: what "M" reapplies on a
        reclick, and what every live-acquisition redraw holds to while
        Manual is active."""
        self._manual_range = (xmin, xmax, ymin, ymax)
        if self._manual_mode:
            self._apply_manual_range()

    def _on_manual_clicked(self):
        """"M" clicked — switch to Manual, or (on a reclick) snap back to the
        exact limits held from the axes' native "Manual" min/max fields."""
        if self._manual_range is None:
            (xmin, xmax), (ymin, ymax) = self._plot.getViewBox().viewRange()
            self._manual_range = (xmin, xmax, ymin, ymax)
        self._manual_mode = True
        self._apply_manual_range()

    def _on_auto_clicked(self):
        """"A" clicked — switch to Auto, or (on a reclick) force an
        immediate re-fit of the histogram."""
        self._manual_mode = False
        self._redraw_hist()

    def _apply_manual_range(self):
        """Force the exact held manual limits, unclamped by any pan/zoom bound."""
        if self._manual_range is None:
            return
        xmin, xmax, ymin, ymax = self._manual_range
        vb = self._plot.getViewBox()
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None,
                     maxXRange=None, maxYRange=None)
        vb.setXRange(xmin, xmax, padding=0)
        vb.setYRange(ymin, ymax, padding=0)

    def set_data(self, values, scope: str = ""):
        vals = np.asarray(values, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self._text.setPlainText(f"{scope}\n(no pixels)")
            self._fit_text_height()
            self._hist = None; self._curve.setData([], [])
            return
        n = vals.size
        p70, p90, p99, p999, p9999 = np.percentile(vals, [70, 90, 99, 99.9, 99.99])

        def g(x):
            return f"{x:.6g}"

        def cnt(p):
            return int(np.count_nonzero(vals > p))
        lines = [
            scope,
            f"N      = {n:,}",
            f"p70    = {g(p70):<10} (>: {cnt(p70):,})",
            f"p90    = {g(p90):<10} (>: {cnt(p90):,})",
            f"p99    = {g(p99):<10} (>: {cnt(p99):,})",
            f"p99.9  = {g(p999):<10} (>: {cnt(p999):,})",
            f"p99.99 = {g(p9999):<10} (>: {cnt(p9999):,})",
        ]
        self._text.setPlainText("\n".join(lines))
        self._fit_text_height()

        # Histogram over the FULL intensity range so high-intensity pixels appear.
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        v_hist = vals
        if vals.size > 50_000_000:      # bound time only on very large stacks
            v_hist = vals[np.random.default_rng(0).integers(0, vals.size, 50_000_000)]
        counts, edges = np.histogram(v_hist, bins=256, range=(vmin, vmax))
        self._hist = (counts, edges)
        self._redraw_hist()

    def _redraw_hist(self, *_):
        if self._hist is None:
            self._curve.setData([], []); return
        counts, edges = self._hist
        y = counts.astype(float)
        log = self._logchk.isChecked()
        if log:
            y = np.log10(y + 1.0)
        self._curve.setData(edges, y)
        self._plot.setLabel("left", "log(count+1)" if log else "count",
                            **{"color": "#d0d0d0", "font-size": "12pt"})
        if self._manual_mode:
            # Manual mode is authoritative — hold the user's limits regardless
            # of how the histogram data changed (e.g. a new live frame).
            self._apply_manual_range()
            return
        # Fixed lower-left corner (x=0, y=-2); rescale to (0..xmax, -2..ymax) on refresh.
        xmax = float(edges[-1]) if edges.size else 1.0
        ymax = float(y.max()) if y.size else 1.0
        ymax = ymax * 1.08 if ymax > 0 else 1.0
        vb = self._plot.getViewBox()
        # xMax/maxXRange/maxYRange cap zoom-out at the natural full-data
        # view (same intent as ProfileViewer._apply_view_limits above) so
        # the histogram can't be zoomed out until it's lost in empty space.
        vb.setLimits(xMin=0.0, xMax=max(xmax, 1.0), yMin=-2.0, yMax=ymax,
                     maxXRange=max(xmax, 1.0), maxYRange=ymax + 2.0)
        vb.setXRange(0.0, max(xmax, 1.0), padding=0)
        vb.setYRange(-2.0, ymax, padding=0)


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


class DataLoaderPanel(QtWidgets.QWidget):
    """Left-hand data-loading panel shared by Tabs 0/2/3/4.

    Selects the five inputs — Data, Dark, Bright, Background, Mask — each a single
    file / a folder / an HDF5 dataset (container dropdown).  ``mode`` tailors the
    Data card:
      - ``"stack"``  — frame navigator (Data Viewer);
      - ``"single"`` — a frame-index spin (Calibrate / Refinement);
      - ``"stream"`` — frame range + stride, no in-memory load (Batch).
    Dark/Bright/Background reuse :class:`FieldSelector`; Mask uses
    :class:`MaskSelector`.  ``corrected(frame)`` applies dark/bright/background.
    """
    dataChanged = QtCore.pyqtSignal()     # data loaded, or current frame changed
    fieldsChanged = QtCore.pyqtSignal()   # dark/bright/background/mask changed
    monitorToggled = QtCore.pyqtSignal(bool)  # MONITOR button toggled (stream mode)
    bufferInvalidated = QtCore.pyqtSignal()   # this panel's own buffer was reset
    metadataDetected = QtCore.pyqtSignal(dict)  # auto-detected pxY/wavelength_A from a new Data load

    def __init__(self, parent=None, *, mode="single", data_dataset="exchange/data",
                 dark_dataset="exchange/data_dark", allow_live=False):
        super().__init__(parent)
        from midas_gui import style as S
        self._mode = mode
        self._stack = self._paths = self._h5 = None
        self._nframes = 0
        self._cur = None
        self._stream_preview_dirty = True   # "stream" mode only — see current_frame()
        self._preview_sum_n = 1             # "stream" mode only — see set_preview_sum
        self._live_src: Optional[PvaLiveSource] = None
        self._registry = None          # DataSourceRegistry, set by bind_registry()
        self._registry_label = ""      # this panel's own label in the registry
        self._explicit_paths = None    # list[str], set by a Browse… "Multiple files"/"stem" pick
        self._stem_filter = None       # str, "stream" mode's live filestem filter (see _raw_source)
        self._external = None          # another DataLoaderPanel this one delegates to, or None
        self._buffer_snapshot_file = None  # temp .h5 path from importing an external buffer (stream mode)
        self._buffer = None            # deque(maxlen=N) once "Use Buffer" is on
        # Guards all reads/writes of self._buffer: it's appended to from the GUI
        # thread (_on_live_frame) but read from background QThreads (e.g.
        # ProjectionWorker.run() via full_stack()) while streaming may still be
        # live, so a plain deque is not safe to share across threads here.
        self._buffer_lock = threading.Lock()
        self._buffer_active = False    # "Use Buffer" checked
        self._buffer_frozen = False    # True once frame arrival has paused
        # True only for the dataChanged emitted from a live PVA frame arriving —
        # lets viewers keep a manual color-scale window fixed across live frames
        # while still auto-resetting it for an actual new load/frame-navigation.
        self._live_frame_update = False
        self._buffer_stall_timer = QtCore.QTimer(self)
        self._buffer_stall_timer.setSingleShot(True)
        self._buffer_stall_timer.setInterval(2000)
        self._buffer_stall_timer.timeout.connect(self._on_buffer_stalled)

        # The cards live inside a scroll area; the stats panel (stack mode) sits
        # below it in a draggable vertical splitter (see the end of __init__).
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget(); self._scroll.setWidget(inner)
        lv = QtWidgets.QVBoxLayout(inner); lv.setContentsMargins(4, 4, 4, 4); lv.setSpacing(8)

        # Distinct background + accent right border so the data-loader panel stands
        # out from the middle parameters panel.
        self.setObjectName("dataLoaderPanel")
        inner.setObjectName("dataLoaderInner")
        self._scroll.setObjectName("dataLoaderScroll")
        self._scroll.viewport().setObjectName("dataLoaderViewport")
        self.setStyleSheet(
            f"#dataLoaderPanel {{ border: none; border-right: 2px solid {S.ACCENT}; }}"
            f"#dataLoaderScroll {{ border: none; }}"
            f"#dataLoaderViewport, #dataLoaderInner {{ background: #2b2e35; }}")

        # ── Live Data card (collapsible via its own checkbox; above Data) ──
        if allow_live:
            live_card = S.make_card("Live Data")
            live_card.setCheckable(True)
            live_card.setChecked(False)
            self._live_card = live_card
            self._live_content = QtWidgets.QWidget()
            lvbox = QtWidgets.QVBoxLayout(self._live_content)
            lvbox.setContentsMargins(0, 4, 0, 0); lvbox.setSpacing(4)
            pv_row = QtWidgets.QHBoxLayout(); pv_row.setSpacing(4)
            pv_row.addWidget(QtWidgets.QLabel("Live PV:"))
            self._pv_ed = _NoScrollComboBox()
            self._pv_ed.setEditable(True)
            self._pv_ed.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            self._pv_ed.lineEdit().setPlaceholderText("e.g. 20IDFF:Pva1:Image")
            for d in DEVICES:
                full_pv = f"{d.get('prefix', '')}{d.get('pva_suffix', '')}"
                self._pv_ed.addItem(d.get("name", ""), full_pv)
            self._pv_ed.setCurrentIndex(-1)
            self._pv_ed.setEditText("")
            self._pv_ed.activated.connect(self._on_pv_device_picked)
            pv_row.addWidget(self._pv_ed, 1)
            lvbox.addLayout(pv_row)
            btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(4)
            self._live_start_btn = QtWidgets.QPushButton("Start")
            self._live_stop_btn = QtWidgets.QPushButton("Stop")
            self._live_stop_btn.setEnabled(False)
            self._live_start_btn.clicked.connect(self._start_live)
            self._live_stop_btn.clicked.connect(self.stop_live)
            btn_row.addWidget(self._live_start_btn); btn_row.addWidget(self._live_stop_btn)
            lvbox.addLayout(btn_row)
            buf_row = QtWidgets.QHBoxLayout(); buf_row.setSpacing(4)
            buf_row.addWidget(QtWidgets.QLabel("N:"))
            self._buffer_n_spin = _NoScrollSpinBox()
            self._buffer_n_spin.setRange(2, 100)
            self._buffer_n_spin.setValue(20)
            self._buffer_n_spin.setFixedWidth(64)
            self._buffer_n_spin.setToolTip(
                "Number of most-recent live frames to keep in the ring buffer "
                "(max 100, to bound memory use for large-format detectors).")
            buf_row.addWidget(self._buffer_n_spin)
            self._buffer_btn = QtWidgets.QPushButton("Use Buffer")
            self._buffer_btn.setCheckable(True)
            self._buffer_btn.setToolTip(
                "Capture the last N live frames into memory. Turns yellow while "
                "filling, green once frame arrival pauses — the buffered frames "
                "then act as a normal stack (scrubbable, projectable).")
            self._buffer_btn.toggled.connect(self._on_buffer_toggled)
            buf_row.addWidget(self._buffer_btn, 1)
            self._buffer_save_btn = QtWidgets.QToolButton()
            self._buffer_save_btn.setText("\U0001F4BE")  # 💾
            self._buffer_save_btn.setFixedWidth(28)
            self._buffer_save_btn.setToolTip(
                "Save the buffered frames to an HDF5 file (dataset 'buffer/data').")
            self._buffer_save_btn.setEnabled(False)
            self._buffer_save_btn.clicked.connect(self._on_save_buffer)
            buf_row.addWidget(self._buffer_save_btn)
            lvbox.addLayout(buf_row)
            self._apply_buffer_style("off")
            self._live_status_lbl = QtWidgets.QLabel("Stopped.")
            self._live_status_lbl.setWordWrap(True)
            self._live_status_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
            lvbox.addWidget(self._live_status_lbl)
            self._live_content.setVisible(False)
            live_card.body.addWidget(self._live_content)
            live_card.toggled.connect(self._on_live_card_toggled)
            lv.addWidget(live_card)

        # ── Data card ──
        card = S.make_card("Data")
        self._path_ed = QtWidgets.QLineEdit()
        self._path_ed.setPlaceholderText("file / folder / .h5")
        self._path_ed.textChanged.connect(self._on_path_changed)
        self._path_ed.returnPressed.connect(self._load)
        browse = QtWidgets.QToolButton(); browse.setText("⋯"); browse.setFixedWidth(28)
        browse.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(browse)
        menu.addAction("Browse…", self._open_browse_dialog)
        menu.addSeparator()
        self._import_menu = menu.addMenu("Import from…")
        self._import_menu.aboutToShow.connect(self._populate_import_menu)
        browse.setMenu(menu)
        pr = QtWidgets.QHBoxLayout(); pr.setSpacing(4)
        pr.addWidget(self._path_ed); pr.addWidget(browse)
        if allow_live:
            reload_btn = QtWidgets.QToolButton(); reload_btn.setText("⟳"); reload_btn.setFixedWidth(28)
            reload_btn.setToolTip(
                "Reload the current Data path/folder/dataset — use this after "
                "stopping a live PV stream to restore the static data.")
            reload_btn.clicked.connect(self._load)
            pr.addWidget(reload_btn)
        card.body.addLayout(pr)

        self._ds_row = QtWidgets.QWidget()
        dr = QtWidgets.QHBoxLayout(self._ds_row); dr.setContentsMargins(0, 0, 0, 0); dr.setSpacing(4)
        self._ds_combo = _NoScrollComboBox(); self._ds_combo.setEditable(True)
        self._ds_combo.setEditText(data_dataset)
        self._ds_combo.currentIndexChanged.connect(lambda _=0: self._load() if self._nframes else None)
        dr.addWidget(QtWidgets.QLabel("Dataset:")); dr.addWidget(self._ds_combo, 1)
        self._ds_row.setVisible(False)
        card.body.addWidget(self._ds_row)

        # Mode-specific frame controls
        self._frame_spin = _NoScrollSpinBox(); self._frame_spin.setRange(0, 0)
        if mode == "stack":
            self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self._prev_btn = QtWidgets.QPushButton("◀"); self._prev_btn.setFixedWidth(30)
            self._next_btn = QtWidgets.QPushButton("▶"); self._next_btn.setFixedWidth(30)
            self._nframes_lbl = QtWidgets.QLabel("/ 0")
            self._frame_spin.setFixedWidth(64)
            self._slider.valueChanged.connect(self._set_frame)
            self._frame_spin.valueChanged.connect(self._set_frame)
            self._prev_btn.clicked.connect(lambda: self._set_frame(self._frame_spin.value() - 1))
            self._next_btn.clicked.connect(lambda: self._set_frame(self._frame_spin.value() + 1))
            nav = QtWidgets.QHBoxLayout(); nav.setSpacing(4)
            nav.addWidget(self._prev_btn); nav.addWidget(self._slider, 1)
            nav.addWidget(self._frame_spin); nav.addWidget(self._nframes_lbl); nav.addWidget(self._next_btn)
            self._nav_row = QtWidgets.QWidget(); self._nav_row.setLayout(nav)
            self._nav_row.setEnabled(False)
            card.body.addWidget(self._nav_row)
        elif mode == "single":
            self._frame_spin.valueChanged.connect(self._set_frame)
            fr = QtWidgets.QHBoxLayout(); fr.setSpacing(4)
            fr.addWidget(QtWidgets.QLabel("Frame:")); fr.addWidget(self._frame_spin); fr.addStretch(1)
            card.body.addLayout(fr)
        else:  # stream
            # start/end are always FILE (scan) numbers, never an index into
            # sub-frames or "Combine sub-frames" chunks — that setting is
            # completely orthogonal, controlling only how each file's own
            # internal sub-frames combine into output frames. _autofill_
            # frame_range keeps the live label below in sync with whichever
            # case applies:
            # • Several files (multi-select / folder / stem-recursive) →
            #   start/end are the smallest/largest scan number among them.
            # • A single file → start/end lock to its own scan number (equal),
            #   since there's exactly one file and nothing to range over.
            _fr_tip = (
                "The scan/file NUMBER to start/end at (parsed from filenames "
                "like ..._009243.vrx.h5 → 9243), not a frame index — end is "
                "inclusive.\n"
                "• Several files (folder / multi-select / stem search) → set "
                "these to pick a sub-range of scan numbers; both default to "
                "the full min…max found.\n"
                "• A single file → start/end always equal that file's own "
                "scan number and are disabled, since there's only one file "
                "to process. Use 'Combine sub-frames' below to control how "
                "its internal sub-frames combine into output frames — that's "
                "unrelated to start/end.\n"
                "• Other frame-indexed sources (e.g. a single multi-frame "
                "stack) fall back to a plain 0-based index, end exclusive, "
                "0 = all.")
            self._fr_start = _NoScrollSpinBox(); self._fr_start.setRange(0, 999999); self._fr_start.setFixedWidth(64)
            self._fr_start.setToolTip("First frame (inclusive).\n\n" + _fr_tip)
            self._fr_end = _NoScrollSpinBox(); self._fr_end.setRange(0, 999999); self._fr_end.setFixedWidth(64)
            self._fr_end.setToolTip("Last frame (exclusive). 0 = all frames.\n\n" + _fr_tip)
            self._fr_stride = _NoScrollSpinBox(); self._fr_stride.setRange(1, 100000); self._fr_stride.setValue(1); self._fr_stride.setFixedWidth(64)
            self._fr_stride.setToolTip("Take every Nth frame (1 = every frame).\n\n" + _fr_tip)
            sf = S.Form()
            sf.row(("start:", self._fr_start), ("end(0=all):", self._fr_end))
            sf.row(("stride:", self._fr_stride))
            card.body.addLayout(sf)
            self._fr_hint = QtWidgets.QLabel("")
            self._fr_hint.setWordWrap(True)
            self._fr_hint.setStyleSheet(f"color:{S.MUTED};font-size:10px;")
            card.body.addWidget(self._fr_hint)

            # Shown only when the resolved source is several separate HDF5
            # files (e.g. one VAREX *.vrx.h5 per scan point, each itself a
            # multi-frame stack) — see source_cfg()'s "hdf5_stack_glob" type.
            # A single-file HDF5 stack ("hdf5" type) streams its frames
            # as-is and never shows this row.
            self._combine_row = QtWidgets.QWidget()
            cr = QtWidgets.QHBoxLayout(self._combine_row)
            cr.setContentsMargins(0, 0, 0, 0); cr.setSpacing(4)
            cr.addWidget(QtWidgets.QLabel("Combine sub-frames:"))
            self._combine_chunk = _NoScrollSpinBox()
            self._combine_chunk.setRange(0, 999999); self._combine_chunk.setFixedWidth(64)
            self._combine_chunk.setToolTip(
                "How many consecutive raw sub-frames in each file to combine "
                "into one integrated frame (mpe_wf's OME_SUM). 0 = combine "
                "every sub-frame in the file into one (the usual case for a "
                "detector that writes several raw exposures per scan point).")
            self._combine_op_combo = _NoScrollComboBox()
            self._combine_op_combo.addItem("Mean", "mean")
            self._combine_op_combo.addItem("Sum", "sum")
            self._combine_op_combo.addItem("Max", "max")
            self._combine_op_combo.addItem("Median", "median")
            cr.addWidget(self._combine_chunk)
            cr.addWidget(QtWidgets.QLabel("op:")); cr.addWidget(self._combine_op_combo)
            cr.addStretch(1)
            self._combine_row.setVisible(False)
            card.body.addWidget(self._combine_row)
            # Changing how sub-frames combine changes what the cached
            # preview frame (_refresh_stream_preview) actually shows.
            self._combine_chunk.valueChanged.connect(self._on_combine_changed)
            self._combine_op_combo.currentIndexChanged.connect(self._on_combine_changed)

        self._info = QtWidgets.QLabel("No data loaded.")
        self._info.setStyleSheet("color:#9a9a9a;font-size:10px"); self._info.setWordWrap(True)
        card.body.addWidget(self._info)
        lv.addWidget(card)

        # ── Dark / Bright / Background ──
        fld = S.make_card("Dark / Bright / Background")
        self._dark_sel = FieldSelector("Dark", default_dataset=dark_dataset)
        self._bright_sel = FieldSelector("Bright", with_mode=True)
        self._bg_sel = FieldSelector("Background")
        for w in (self._dark_sel, self._bright_sel, self._bg_sel):
            w.fieldReady.connect(self.fieldsChanged)
            fld.body.addWidget(w)
        lv.addWidget(fld)

        # ── Mask ──
        self._mask_sel = MaskSelector()
        self._mask_sel.maskChanged.connect(self.fieldsChanged)
        lv.addWidget(self._mask_sel)

        # "stream" mode's cached preview frame (_peek_stream_frame) now
        # bakes dark/bright/background correction in — a field changing
        # after a preview was already cached must invalidate it too, or a
        # caller like Batch Integrate's Detector view would keep showing
        # the pre-change (or wrongly-corrected) frame despite fieldsChanged
        # asking it to refresh.
        self.fieldsChanged.connect(self._on_fields_changed_stream)

        # Intensity statistics + histogram (Data Viewer only). Created here but
        # placed in a draggable splitter below, not in the scrolling card column.
        self.stats_panel = None
        lv.addStretch(1)
        if mode == "stack":
            self.stats_panel = IntensityStatsPanel()

        # ── MONITOR button (stream mode only) — pinned to the bottom ──
        self._monitor_btn = None
        if mode == "stream":
            self._monitor_btn = QtWidgets.QPushButton("●  MONITOR")
            self._monitor_btn.setCheckable(True)
            self._monitor_btn.setMinimumHeight(30)
            self._monitor_btn.setToolTip(
                "Watch the data folder for new frames and integrate them "
                "automatically as they appear, reusing the detector map (no full "
                "re-run). Turns green while active.")
            self._monitor_btn.toggled.connect(self._on_monitor_toggled)
            self._apply_monitor_style(False)
            lv.addWidget(self._monitor_btn)

        # ── Outer layout: scroll on top, optional draggable stats panel below ──
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        if self.stats_panel is not None:
            self._left_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self._left_split.setChildrenCollapsible(False)
            self._left_split.setHandleWidth(6)
            self._left_split.addWidget(self._scroll)
            self._left_split.addWidget(self.stats_panel)
            self._left_split.setStretchFactor(0, 3)
            self._left_split.setStretchFactor(1, 1)
            self._left_split.setSizes([560, 320])
            outer.addWidget(self._left_split)
        else:
            outer.addWidget(self._scroll)

    # ── data source (path / dataset / loading) ────────────────────
    def _dataset(self) -> str:
        return self._ds_combo.currentText().split("   ")[0].strip() or "exchange/data"

    def _raw_source(self):
        """The current source: an explicit ``list[str]`` from a Browse…
        "Multiple files"/"Files sharing a stem" pick, a ``<folder>/**/<stem>*``
        glob pattern when "stream" mode's live filestem filter is set (see
        ``_set_stem_filter`` — this one substitution is what makes
        ``source_cfg()``, ``_load()`` and cross-tab "Import from…" all
        filestem-aware for free), else the plain path text (file / folder /
        glob). The ``**`` searches the entire folder tree below the selected
        folder, not just its direct children — matching files sharing a stem
        is meant to find every scan-point file regardless of which
        subfolder it landed in, unlike "Full folder" (one flat directory).
        ``helpers._collect_frame_paths`` globs this with ``recursive=True``,
        under which ``**`` also matches zero directories, so files directly
        in the selected folder still match too."""
        if self._explicit_paths:
            return self._explicit_paths
        text = self._path_ed.text().strip()
        if self._stem_filter and text:
            from pathlib import Path
            return str(Path(text) / "**" / (self._stem_filter + "*"))
        return text

    def _set_explicit_paths(self, paths):
        """``paths`` is a resolved ``list[str]`` (Multiple files / stem
        match) or None to fall back to the plain path text. Uses
        blockSignals so the summary text it writes doesn't itself clear
        ``_explicit_paths`` via ``_on_path_changed``."""
        self._explicit_paths = paths
        self._path_ed.blockSignals(True)
        if paths:
            self._path_ed.setText(display_text_for_paths(paths))
            self._path_ed.setToolTip("\n".join(paths))
        else:
            self._path_ed.setToolTip("")
        self._path_ed.blockSignals(False)

    def _set_stem_filter(self, folder, stem):
        """"stream" mode (Batch Integrate) only: keep a filestem pick as a
        live ``(folder, prefix)`` filter rather than resolving it to a frozen
        file list, so MONITOR picks up newly-arriving matching files too (see
        ``_raw_source``). ``stem`` falsy clears it. Uses blockSignals like
        ``_set_explicit_paths`` so writing ``folder`` into the path field
        doesn't itself clear ``_stem_filter`` via ``_on_path_changed``."""
        self._stem_filter = stem or None
        self._path_ed.blockSignals(True)
        if stem:
            self._path_ed.setText(folder)
            self._path_ed.setToolTip(f"Filestem filter: {stem}*")
        else:
            self._path_ed.setToolTip("")
        self._path_ed.blockSignals(False)

    def _open_browse_dialog(self):
        dlg = BrowseFilesDialog(self, title="Select data",
                                modes=("file", "files", "folder", "stem"),
                                start_dir=self._path_ed.text().strip())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode = dlg.mode()
        if mode == "file":
            paths = dlg.paths()
            if not paths:
                return
            self._set_explicit_paths(None)
            self._set_stem_filter(None, None)
            self._path_ed.setText(paths[0])
        elif mode == "folder":
            self._set_explicit_paths(None)
            self._set_stem_filter(None, None)
            self._path_ed.setText(dlg.folder())
        elif mode == "stem" and self._mode == "stream":
            # Batch Integrate keeps the filestem as a live filter instead of
            # a frozen file list — see _set_stem_filter.
            folder, stem = dlg.stem()
            if not stem:
                return
            self._set_explicit_paths(None)
            self._set_stem_filter(folder, stem)
        else:  # "files", or "stem" outside Batch Integrate's stream mode
            paths = dlg.paths()
            if not paths:
                return
            self._set_stem_filter(None, None)
            self._set_explicit_paths(paths)
        self._update_combine_visibility()
        self._load()

    def _update_combine_visibility(self):
        """Show the "Combine sub-frames" row (stream mode only) whenever the
        resolved source is HDF5 — a single bare file (``"hdf5"``) or several
        separate files (``source_cfg``'s ``"hdf5_stack_glob"``) both combine
        their own internal frame stack the same way (see ``source_cfg``) —
        and not for TIFF-family sources, which have no such stack to combine."""
        if not hasattr(self, "_combine_row"):
            return
        from pathlib import Path
        raw = self._raw_source()
        if isinstance(raw, str) and is_h5(raw) and Path(raw).is_file():
            self._combine_row.setVisible(True)
            return
        paths = raw if isinstance(raw, list) else (_collect_frame_paths(raw) if raw else [])
        self._combine_row.setVisible(bool(paths) and any(is_h5(p) for p in paths))

    def _on_path_changed(self, p: str):
        from pathlib import Path
        # Fires only for a real text edit — a Browse… explicit-list/stem pick
        # sets its own summary text via blockSignals, so this never races it.
        self._explicit_paths = None
        self._stem_filter = None
        h5 = is_h5(p)
        self._ds_row.setVisible(h5)
        self._update_combine_visibility()
        if h5 and Path(p).exists():
            try:
                items = list_h5_datasets(p)
            except Exception:
                items = []
            if items:
                keep = self._ds_combo.currentText().strip()
                self._ds_combo.blockSignals(True); self._ds_combo.clear()
                for name, shape in items:
                    self._ds_combo.addItem(f"{name}   {tuple(shape)}", name)
                idx = next((i for i in range(self._ds_combo.count())
                            if self._ds_combo.itemData(i) == keep), -1)
                if idx < 0:
                    idx = next((i for i, (n, s) in enumerate(items) if len(s) >= 3), 0)
                self._ds_combo.setCurrentIndex(idx)
                self._ds_combo.blockSignals(False)

    def _collect_paths(self, raw) -> list:
        return _collect_frame_paths(raw)

    # ── cross-tab data sharing (data_bridge.DataSourceRegistry) ─────
    def bind_registry(self, registry, label: str):
        """Register this panel as an importable data source under `label`
        (e.g. "Data Viewer"), and gain an "Import from…" menu listing every
        other bound panel's currently-loaded data."""
        self._registry = registry
        self._registry_label = label
        registry.register(label, self)
        for sel in (self._dark_sel, self._bright_sel, self._bg_sel):
            sel.set_registry(registry, exclude_label=label)

    def _describe_data_field(self):
        """Live snapshot of this panel's own Data (field="data"), or None if
        nothing is loaded. `path` may be a `list[str]` if this panel's own
        source is an explicit Multiple-files/stem pick."""
        if self._buffer_frozen and self._buffer:
            return {"kind": "buffer", "provider": self, "field": "data",
                    "label": self._registry_label}
        raw = self._raw_source()
        if not raw:
            return None
        return {"kind": "path", "path": raw,
                "dataset": self._dataset() if (isinstance(raw, str) and is_h5(raw)) else None,
                "field": "data", "label": self._registry_label}

    def describe_source(self):
        """Live snapshot of what this panel currently offers other panels —
        its main Data (if loaded) plus any of its own Dark/Bright/Background
        fields that are enabled and point at a path — called by the registry
        right before an "Import from…" menu is shown. Each entry is tagged
        with a `field` so importers only see sources of their own type."""
        out = []
        data = self._describe_data_field()
        if data is not None:
            out.append(data)
        for sel in (self._dark_sel, self._bright_sel, self._bg_sel):
            d = sel.describe_source(self._registry_label)
            if d is not None:
                out.append(d)
        return out

    def _populate_import_menu(self):
        menu = self._import_menu
        menu.clear()
        if self._registry is None:
            menu.addAction("(no other tabs loaded)").setEnabled(False)
            return
        sources = self._registry.available(exclude=self, field="data")
        if not sources:
            menu.addAction("(nothing loaded elsewhere)").setEnabled(False)
            return
        for desc in sources:
            menu.addAction(_fmt_source_desc(desc), lambda d=desc: self._apply_imported_source(d))

    def _apply_imported_source(self, desc: dict):
        if desc["kind"] == "path":
            self._clear_external()
            path = desc["path"]
            if isinstance(path, list):
                self._set_explicit_paths(path)
                self._load()
            else:
                self.set_path(path, dataset=desc.get("dataset"))
            return
        if self._mode == "stream":
            self._import_buffer_via_tempfile(desc["provider"])
        else:
            self.use_external_buffer(desc["provider"])

    def use_external_buffer(self, provider) -> None:
        """Delegate this panel's data entirely to `provider` (another
        DataLoaderPanel with a frozen, non-empty buffer) with no copying —
        `_get_frame`/`full_stack` forward to it as long as `self._external`
        is set."""
        self._clear_external()
        self._stack = self._paths = self._h5 = None
        self._reset_buffer()
        self._external = provider
        provider.bufferInvalidated.connect(self._on_external_invalidated)
        self._nframes = provider.n_frames()
        self._setup_navigator()
        self._info.setText(f"Imported: {provider._registry_label} buffer ({self._nframes} frames)")
        self._set_frame(self._nframes - 1)

    def _clear_external(self):
        if self._external is not None:
            try:
                self._external.bufferInvalidated.disconnect(self._on_external_invalidated)
            except Exception:
                pass
            self._external = None

    def _on_external_invalidated(self):
        self._clear_external()
        self._nframes = 0
        self._cur = None
        self._setup_navigator()
        self._info.setText("Source buffer was reset.")
        self.dataChanged.emit()

    def _import_buffer_via_tempfile(self, provider) -> None:
        """Stream-mode panels never hold frames in memory (`source_cfg()` just
        reports a path/dataset for BatchWorker to stream itself), so importing
        another panel's live buffer means snapshotting it to a real HDF5 file
        once and pointing at that, exactly like any other HDF5 source."""
        with provider._buffer_lock:
            frames = list(provider._buffer) if (provider._buffer_frozen and provider._buffer) else None
        if not frames:
            QtWidgets.QMessageBox.warning(self, "No buffer", "Source buffer is empty.")
            return
        old = self._buffer_snapshot_file
        path = new_temp_h5_path()
        save_stack_h5(path, frames, dataset="buffer/data")
        self._buffer_snapshot_file = path
        self.set_path(path, dataset="buffer/data")
        if old is not None:
            import os
            try:
                os.unlink(old)
            except OSError:
                pass

    def _load(self):
        from pathlib import Path
        self._clear_external()
        raw = self._raw_source()
        if not raw:
            return
        if self._mode == "stream":
            # No in-memory load of the whole dataset (that's the point of
            # stream mode for large scans) — just mark the cached preview
            # frame stale. current_frame() does the actual (cheap-if-
            # unneeded, since Pump Probe's "stream" loader never calls it)
            # one-frame "peek" lazily, on first ask — see current_frame /
            # _peek_stream_frame.
            if isinstance(raw, list):
                text = f"Source: {len(raw)} file(s) — {display_text_for_paths(raw)}"
            elif self._stem_filter:
                text = f"Source: {self._path_ed.text().strip()}  (filestem: {self._stem_filter}*)"
            else:
                text = f"Source: {raw}"
            self._info.setText(text)
            self._live_frame_update = False
            self._stream_preview_dirty = True
            self._autofill_frame_range()
            self.dataChanged.emit()
            return
        try:
            self._stack = self._paths = self._h5 = None; self._nframes = 0
            self._reset_buffer()
            if isinstance(raw, list):
                paths = self._collect_paths(raw)
                if not paths:
                    QtWidgets.QMessageBox.warning(self, "Empty", "No frames found."); return
                self._paths = paths; self._nframes = len(paths)
                kind = f"{self._nframes} file(s) selected"
            elif Path(raw).is_dir() or any(ch in raw for ch in "*?"):
                paths = self._collect_paths(raw)
                if not paths:
                    QtWidgets.QMessageBox.warning(self, "Empty", "No frames found."); return
                self._paths = paths; self._nframes = len(paths)
                kind = f"folder/glob ({self._nframes} files)"
            elif is_h5(raw):
                import h5py
                dset = self._dataset()
                with h5py.File(raw, "r") as f:
                    if dset not in f:
                        raise KeyError(f"dataset '{dset}' not in file")
                    shape = f[dset].shape
                n = shape[0] if len(shape) >= 3 else 1
                self._h5 = (raw, dset, n); self._nframes = n
                kind = f"HDF5 [{dset}] {shape}"
            else:
                import tifffile
                arr = np.asarray(tifffile.imread(raw))
                if arr.ndim >= 3:
                    self._stack = arr; self._nframes = arr.shape[0]
                    kind = f"TIFF stack {arr.shape}"
                else:
                    self._stack = arr[None, ...]; self._nframes = 1
                    kind = f"TIFF {arr.shape}"
            self._info.setText(f"Loaded: {kind}")
            self._setup_navigator()
            self._set_frame(0)
            detect_path = self._paths[0] if self._paths else raw
            detected = detect_geometry_from_path(detect_path)
            if detected:
                self.metadataDetected.emit(detected)
        except Exception:
            import traceback
            show_error(self, "Load error", traceback.format_exc())

    def _on_combine_changed(self, *_args):
        """A "Combine sub-frames" control (chunk size/op) changed — the
        cached preview frame (if any caller has asked for one via
        current_frame()) no longer reflects the current settings, and for a
        single hdf5 source, start/end's displayed range (combined-frame
        count) depends on chunk_size too — see _autofill_frame_range's
        "combined-frame index" branch — so it needs recomputing here, not
        just on source-path change."""
        if self._raw_source():
            self._stream_preview_dirty = True
            self._autofill_frame_range()
            self.dataChanged.emit()

    def _on_fields_changed_stream(self) -> None:
        if self._mode == "stream":
            self._stream_preview_dirty = True

    def set_preview_sum(self, n: int) -> None:
        """"stream" mode only: how many of the source's leading frames
        ``current_frame()``'s preview sums together (a display aid only —
        never affects the real batch run). A single weak/noisy frame can be
        dominated by detector artifacts (e.g. VAREX per-column gain
        non-uniformity) that hide the actual diffraction pattern; summing
        a handful together boosts the real signal enough to see it, at no
        cost to callers that never ask for a preview (Pump Probe)."""
        n = max(1, int(n))
        if n != self._preview_sum_n:
            self._preview_sum_n = n
            self._stream_preview_dirty = True

    def _peek_stream_frame(self):
        """Fetch, dark/bright/background-correct, and sum the first
        ``self._preview_sum_n`` frames the current "stream"-mode source
        would yield, for a caller's preview (e.g. Batch Integrate's
        Detector view overlay) — without eagerly loading the whole dataset
        the way "stack"/"single" mode does. Opens the exact same source the
        real run will use (``workers._open_source_cfg``), so e.g. a VAREX
        multi-file source previews its actual combined-per-file frames, not
        a raw sub-frame. Correction is applied to each constituent frame
        BEFORE summing (matching how the real batch run corrects every
        frame independently) — correcting only the final sum once would
        subtract just one dark frame's worth from an N-times-larger signal,
        making it look like dark subtraction barely did anything for N>1.
        Returns None if no source is set or it can't be opened (e.g. an
        incomplete pick, or a transient read error)."""
        cfg = self.source_cfg()
        if not (cfg.get("path") or cfg.get("paths")):
            return None
        try:
            from midas_gui.workers import _open_source_cfg
            source = _open_source_cfg(cfg)
            total = getattr(source, "n_frames", 0)
            if total == 0:
                return None
            n = max(1, min(self._preview_sum_n, total))
            acc = None
            for i in range(n):
                _fid, img = source.get(i)
                img = self.corrected(np.asarray(img, dtype=np.float64))
                acc = img if acc is None else acc + img
            return acc.astype(np.float32)
        except Exception:
            return None

    def _setup_navigator(self):
        hi = max(0, self._nframes - 1)
        self._frame_spin.blockSignals(True); self._frame_spin.setRange(0, hi); self._frame_spin.blockSignals(False)
        if self._mode == "stack":
            self._nav_row.setEnabled(self._nframes > 1)
            self._slider.blockSignals(True); self._slider.setRange(0, hi); self._slider.blockSignals(False)
            self._nframes_lbl.setText(f"/ {hi}")

    def _set_frame(self, i):
        if self._nframes == 0:
            return
        i = max(0, min(int(i), self._nframes - 1))
        widgets = [self._frame_spin] + ([self._slider] if self._mode == "stack" else [])
        for w in widgets:
            w.blockSignals(True); w.setValue(i); w.blockSignals(False)
        self._cur = self._get_frame(i)
        self._live_frame_update = False
        self.dataChanged.emit()

    def _get_frame(self, i: int) -> np.ndarray:
        if self._external is not None:
            return self._external._get_frame(i)
        i = max(0, min(i, self._nframes - 1))
        with self._buffer_lock:
            frame = list(self._buffer)[i] if (self._buffer_frozen and self._buffer) else None
        if frame is not None:
            return np.asarray(frame, dtype=np.float32)
        if self._stack is not None:
            return np.asarray(self._stack[i], dtype=np.float32)
        if self._paths is not None:
            arr = _load_image(self._paths[i]).astype(np.float32)
            return arr[0] if arr.ndim == 3 else arr
        if self._h5 is not None:
            path, dset, _ = self._h5
            return _load_image(path, data_loc=dset, frame=i).astype(np.float32)
        raise RuntimeError("No data loaded")

    # ── live PV stream (Live Data card, allow_live=True only) ───────
    def _on_live_card_toggled(self, checked):
        """The Live Data card's own checkbox collapses/expands its controls.
        Unchecking also stops any active stream — a hidden stream with no
        visible Stop button would otherwise be un-turn-offable."""
        self._live_content.setVisible(checked)
        if not checked:
            self.stop_live()

    def refresh_devices(self):
        """Repopulate the Live PV dropdown from the (possibly just-switched)
        active profile's ``constants.DEVICES`` list, preserving whatever the
        user currently has selected/typed. No-op if this panel has no Live
        Data card (``allow_live=False``)."""
        if not hasattr(self, "_pv_ed"):
            return
        prev_text = self._pv_ed.currentText()
        self._pv_ed.blockSignals(True)
        self._pv_ed.clear()
        for d in DEVICES:
            full_pv = f"{d.get('prefix', '')}{d.get('pva_suffix', '')}"
            self._pv_ed.addItem(d.get("name", ""), full_pv)
        idx = self._pv_ed.findText(prev_text)
        if idx >= 0:
            self._pv_ed.setCurrentIndex(idx)
        else:
            self._pv_ed.setCurrentIndex(-1)
            self._pv_ed.setEditText(prev_text)
        self._pv_ed.blockSignals(False)

    def _on_pv_device_picked(self, index):
        """Selecting a known device by name fills in its full live PV
        (prefix + PVA suffix); typing a PV by hand is untouched (this only
        fires on an explicit dropdown pick, not on text edits)."""
        pv = self._pv_ed.itemData(index)
        if pv:
            self._pv_ed.setEditText(pv)

    def _start_live(self):
        try:
            import pvapy  # noqa: F401
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, "pvapy not installed",
                "pvapy is a required dependency but isn't importable in this "
                "environment.\nReinstall it with:  pip install pvapy==5.4.1")
            return
        pv = self._pv_ed.currentText().strip()
        if not pv:
            QtWidgets.QMessageBox.warning(self, "No PV", "Enter a PV name first.")
            return
        if pv == _SIM_CHANNEL_NAME:
            # Fake Eiger-500K-shaped stream, no beamline hardware needed — see
            # midas_gui.sim_detector. Started lazily on first connect and left
            # running (harmless in-process thread) until app shutdown.
            from midas_gui import sim_detector
            try:
                sim_detector.ensure_running(pv)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Sim Detector failed to start", str(e))
                return
        if self._live_src is None:
            self._live_src = PvaLiveSource(self)
            self._live_src.frameReady.connect(self._on_live_frame)
            self._live_src.connectionChanged.connect(self._on_live_connection)
            self._live_src.error.connect(self._on_live_error)
        self._stack = self._paths = self._h5 = None
        self._nframes = 0
        if self._buffer_active:
            n = self._buffer_n_spin.value()
            with self._buffer_lock:
                self._buffer = deque(maxlen=n)
                self._buffer_frozen = False
            self._apply_buffer_style("filling")
            self._buffer_stall_timer.start()
        else:
            self._reset_buffer()
        if not self._live_src.start(pv):
            return
        self._live_start_btn.setEnabled(False)
        self._live_stop_btn.setEnabled(True)
        self._pv_ed.setEnabled(False)
        self._live_status_lbl.setText("Waiting for PV…")

    def start_live_pv(self, pv: str) -> bool:
        """Programmatic equivalent of picking `pv` in the Live PV combo and
        clicking Start — used by the MIDAS-bridge QLocalServer (app.py) so
        another app can trigger Live Data with no clicks in this GUI."""
        if getattr(self, "_pv_ed", None) is None:
            return False  # panel built without allow_live
        if self._live_src is not None and self._live_src.is_active() \
                and self._pv_ed.currentText().strip() == pv:
            return True  # already streaming this exact PV
        if self._live_src is not None and self._live_src.is_active():
            self.stop_live()
        live_card = getattr(self, "_live_card", None)
        if live_card is not None:
            live_card.setChecked(True)  # expand if collapsed
        self._pv_ed.setEditText(pv)
        self._start_live()
        return self._live_src is not None and self._live_src.is_active()

    def _on_live_frame(self, image, image_id):
        self._nframes = 1
        self._cur = image
        self._setup_navigator()
        status = f"Streaming — frame id {image_id}."
        if self._buffer_active:
            with self._buffer_lock:
                self._buffer.append(image)
                n, cap = len(self._buffer), self._buffer.maxlen
            self._buffer_frozen = False
            self._apply_buffer_style("filling")
            self._buffer_stall_timer.start()
            status += f"  Buffering ({n}/{cap})."
        self._live_status_lbl.setText(status)
        self._info.setText(
            f"Live: {self._pv_ed.currentText().strip()}  (id {image_id}, shape {image.shape})")
        self._live_frame_update = True
        self.dataChanged.emit()

    def _on_live_connection(self, is_connected):
        if self._live_src is not None and self._live_src.is_active():
            self._live_status_lbl.setText(
                "Connected — waiting for first frame…" if is_connected
                else "PV not connected.")

    def _on_live_error(self, msg):
        self._live_status_lbl.setText(f"Error: {msg}")
        self.stop_live()

    def stop_live(self):
        """Stop any active live PV stream. No-op on a panel built without
        allow_live, or if never started; safe to call from app shutdown."""
        live_src = getattr(self, "_live_src", None)
        if live_src is not None:
            live_src.stop()
        start_btn = getattr(self, "_live_start_btn", None)
        if start_btn is not None:
            start_btn.setEnabled(True)
            self._live_stop_btn.setEnabled(False)
            self._pv_ed.setEnabled(True)
            self._live_status_lbl.setText("Stopped.")
        if self._buffer_active and not self._buffer_frozen:
            self._on_buffer_stalled()

    # ── live buffer ("Use Buffer" — last-N-frames ring buffer) ─────
    def _on_buffer_toggled(self, checked):
        if checked:
            n = self._buffer_n_spin.value()
            with self._buffer_lock:
                self._buffer = deque(maxlen=n)
                self._buffer_frozen = False
            self._buffer_active = True
            self._buffer_n_spin.setEnabled(False)
            self._apply_buffer_style("filling")
            self._buffer_stall_timer.start()
        else:
            self._reset_buffer()

    def _on_buffer_stalled(self):
        """No new live frame for the stall interval — freeze the ring buffer
        into a usable stack (same shape of API as a loaded HDF5/folder stack)."""
        if not self._buffer_active or not self._buffer:
            return
        with self._buffer_lock:
            self._buffer_frozen = True
            n = len(self._buffer)
        self._nframes = n
        self._setup_navigator()
        self._apply_buffer_style("ready")
        self._live_status_lbl.setText(f"Streaming paused — buffer ready ({self._nframes} frames).")
        self._set_frame(self._nframes - 1)

    def _reset_buffer(self):
        """Turn buffering off and discard it — used by the toggle-off click and
        by starting a new stream / loading static data (unrelated data)."""
        self._buffer_stall_timer.stop()
        was_stack = self._buffer_frozen
        with self._buffer_lock:
            self._buffer = None
            self._buffer_frozen = False
        self._buffer_active = False
        btn = getattr(self, "_buffer_btn", None)
        if btn is not None:
            self._buffer_n_spin.setEnabled(True)
            btn.blockSignals(True); btn.setChecked(False); btn.blockSignals(False)
            self._apply_buffer_style("off")
        if was_stack:
            self._nframes = 1 if self._cur is not None else 0
            self._setup_navigator()
            self._live_frame_update = False
            self.dataChanged.emit()
            self.bufferInvalidated.emit()

    def _apply_buffer_style(self, state: str):
        btn = getattr(self, "_buffer_btn", None)
        if btn is None:
            return
        if state == "filling":
            n = len(self._buffer) if self._buffer is not None else 0
            cap = self._buffer.maxlen if self._buffer is not None else 0
            btn.setText(f"Buffering… ({n}/{cap})")
            btn.setStyleSheet(
                "QPushButton { background:#f9a825; color:#000; font-weight:bold; "
                "border:1px solid #c17900; border-radius:4px; padding:4px; }")
        elif state == "ready":
            n = len(self._buffer) if self._buffer is not None else 0
            btn.setText(f"Buffer Ready ({n})")
            btn.setStyleSheet(
                "QPushButton { background:#2e7d32; color:white; font-weight:bold; "
                "border:1px solid #1b5e20; border-radius:4px; padding:4px; }")
        else:
            btn.setText("Use Buffer")
            btn.setStyleSheet(
                "QPushButton { background:#3a3d44; color:#ddd; font-weight:bold; "
                f"border:1px solid {S.ACCENT}; border-radius:4px; padding:4px; }}")
        save_btn = getattr(self, "_buffer_save_btn", None)
        if save_btn is not None:
            save_btn.setEnabled(state == "ready")

    def _on_save_buffer(self):
        with self._buffer_lock:
            frames = list(self._buffer) if (self._buffer_frozen and self._buffer) else None
        if not frames:
            QtWidgets.QMessageBox.warning(self, "No buffer", "No frozen buffer to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save buffer to HDF5", "", "HDF5 (*.h5 *.hdf5)")
        if not path:
            return
        try:
            save_stack_h5(path, frames, dataset="buffer/data")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

    def full_stack(self) -> np.ndarray:
        if self._external is not None:
            return self._external.full_stack()
        with self._buffer_lock:
            frames = list(self._buffer) if (self._buffer_frozen and self._buffer) else None
        if frames is not None:
            return np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
        if self._stack is not None:
            return np.asarray(self._stack)
        if self._h5 is not None:
            import h5py
            path, dset, _ = self._h5
            with h5py.File(path, "r") as f:
                return np.asarray(f[dset][()])
        if self._paths is not None:
            frames = [self._get_frame(i) for i in range(self._nframes)]
            return np.stack(frames, axis=0)
        raise RuntimeError("No data loaded")

    def average_frames(self, start=0, end=None, step=1):
        """Mean of frames ``start:end:step`` (end None/<=0 = all), streamed one
        frame at a time so large folders / HDF5 stacks stay memory-safe.

        Returns a float32 2-D array, or None if no data / no frames selected.
        """
        if self._nframes == 0:
            return None
        end = self._nframes if (end is None or end <= 0) else min(int(end), self._nframes)
        start = max(0, int(start))
        step = max(1, int(step))
        acc, n = None, 0
        for i in range(start, end, step):
            a = self._get_frame(i).astype(np.float64)
            acc = a if acc is None else acc + a
            n += 1
        if n == 0:
            return None
        return (acc / n).astype(np.float32)

    # ── public API ────────────────────────────────────────────────
    def set_path(self, path, dataset=None, *, load=True):
        """Preset the data path (and HDF5 dataset); optionally load immediately."""
        self._path_ed.setText(str(path))
        if dataset is not None and is_h5(str(path)):
            self._ds_combo.setEditText(dataset)
        if load:
            from pathlib import Path
            if self._mode == "stream" or Path(str(path)).exists():
                self._load()

    def n_frames(self) -> int:
        return self._nframes

    def frame_index(self) -> int:
        return self._frame_spin.value()

    def set_frame(self, i, **_):
        self._set_frame(i)

    def current_frame(self):
        """Raw (uncorrected) current 2-D frame, or None.

        "stream" mode fetches this lazily, on first ask, from
        ``_peek_stream_frame`` — cached until ``_stream_preview_dirty`` is
        set again (on a source or "Combine sub-frames" change). Pump Probe
        also uses "stream" mode but never calls this, so the (sometimes
        multi-second, for a large multi-frame HDF5) peek only ever happens
        for a caller that actually wants a preview (Batch Integrate's
        Detector view)."""
        if self._mode == "stream":
            if getattr(self, "_stream_preview_dirty", True):
                self._cur = self._peek_stream_frame()
                self._nframes = 1 if self._cur is not None else 0
                self._stream_preview_dirty = False
            return self._cur
        if self._nframes == 0:
            return None
        if self._cur is None:
            self._cur = self._get_frame(self.frame_index())
        return self._cur

    def is_live_frame_update(self) -> bool:
        """True if the most recent dataChanged came from a live PVA frame
        arriving (as opposed to a new load, frame-navigation, or buffer
        reset) — callers use this to keep a manual color-scale window fixed
        across live frames while still auto-resetting it on genuinely new data."""
        return self._live_frame_update

    def source_cfg(self) -> dict:
        """Streaming source descriptor for BatchWorker (stream mode).

        An explicit "Multiple files" pick (an arbitrary list — see
        ``_raw_source``) becomes ``"tiff_list"``: not watchable by MONITOR
        (no single glob pattern describes it). A filestem pick is already
        folded into a ``<folder>/<stem>*`` glob by ``_raw_source``, so it
        comes out as an ordinary ``"tiff_glob"`` here and MONITOR re-globs it
        on every poll like any other glob path (see ``FolderMonitorWorker``).

        A single bare HDF5 file is ``"hdf5"`` — its internal frame stack is
        combined down to one (or a few, via the "Combine sub-frames"
        chunk-size) frame first, same as each file in a multi-file pick
        below (see ``workers._open_source_cfg``, which routes both through
        ``_HDF5StackGlobSource``) — since a single VAREX ``*.vrx.h5``
        holding several raw exposures for one scan point is exactly the
        same shape as one file out of a multi-file scan, just picked on
        its own (e.g. to preview/test one scan point). Several separate
        HDF5 files instead (an explicit "Multiple files" pick, or a "Full
        folder"/"Files sharing a name stem" selection that resolves to
        HDF5 files — e.g. one VAREX ``*.vrx.h5`` per scan point) become
        ``"hdf5_stack_glob"`` — see ``workers._HDF5StackGlobSource``/
        ``helpers.read_hdf5_stack_combined``.
        """
        from pathlib import Path
        raw = self._raw_source()
        if isinstance(raw, str) and is_h5(raw) and Path(raw).is_file():
            return {"type": "hdf5", "path": raw, "dataset": self._dataset(),
                    "chunk_size": (self._combine_chunk.value() or None)
                                 if hasattr(self, "_combine_chunk") else None,
                    "combine_op": (self._combine_op_combo.currentData()
                                  if hasattr(self, "_combine_op_combo") else "mean")}
        paths = raw if isinstance(raw, list) else (_collect_frame_paths(raw) if raw else [])
        # Drop dark acquisitions swept in by a folder/stem selection (the
        # beamline stores them alongside the scan they bracket) — see
        # helpers.is_dark_like_name. An explicit multi-file pick is left
        # alone: naming a dark file by hand is a deliberate choice.
        if not isinstance(raw, list):
            kept = [p for p in paths if not is_dark_like_name(p)]
            self._n_dark_skipped = len(paths) - len(kept)
            paths = kept
        else:
            self._n_dark_skipped = 0
        h5_paths = sorted(p for p in paths if is_h5(p))
        if h5_paths:
            return {"type": "hdf5_stack_glob", "paths": h5_paths,
                    "dataset": self._dataset(),
                    "chunk_size": (self._combine_chunk.value() or None)
                                 if hasattr(self, "_combine_chunk") else None,
                    "combine_op": (self._combine_op_combo.currentData()
                                  if hasattr(self, "_combine_op_combo") else "mean")}
        if isinstance(raw, list):
            return {"type": "tiff_list", "paths": list(raw)}
        return {"type": "tiff_glob", "path": raw}

    def _file_numbers(self) -> list:
        """Scan-point numbers parsed from a MULTI-file source's filenames
        (``C611_017Fe_1_load3_009243.vrx.h5`` → 9243), in the same sorted
        order ``source_cfg()`` hands the paths to the worker. Empty for a
        single-file source (whose start/end lock to its own scan number
        instead — see ``_autofill_frame_range``) or when the names carry no
        number.

        A "tiff_glob" cfg (plain folder pick, or a stem-filter's
        ``<folder>/**/<stem>*`` pattern from ``_raw_source``) carries only
        ``path`` — ``source_cfg()`` expands it to a path list internally
        just to check for HDF5 files, then discards that list — so it's
        re-expanded here via the same ``_collect_frame_paths`` to reach the
        folder/stem-recursive cases, not just the explicit-list/hdf5-stack
        ones that already have a ``paths`` key."""
        from pathlib import Path
        from midas_gui.workers import froot_and_frame_num
        cfg = self.source_cfg()
        paths = cfg.get("paths")
        if paths is None and cfg.get("type") == "tiff_glob" and cfg.get("path"):
            paths = _collect_frame_paths(cfg["path"])
        paths = paths or []
        if len(paths) < 2:
            return []
        nums, roots = [], set()
        for i, p in enumerate(paths):
            froot, num, _tag = froot_and_frame_num(Path(p).stem, -1)
            if num < 0:
                return []
            nums.append(num); roots.add(froot)
        self._file_prefix = roots.pop() if len(roots) == 1 else ""
        return nums

    def frame_range(self):
        """(start, end_exclusive_or_None, stride) as 0-based indices for
        ``workers.resolve_frame_indices`` — indices into the flattened
        COMBINED-frame space ``source.n_frames`` counts over (one entry per
        "Combine sub-frames" chunk, not one per file).

        For a multi-file source the start/end spinboxes hold FILE NUMBERS
        (auto-populated by ``_autofill_frame_range`` — what the user reads
        off the filenames), so translate them to indices here rather than
        changing the worker's/project's long-standing 0-based contract.
        ``end`` is inclusive in file-number terms (9243..9250 processes
        both ends, as a beamline scan range reads), and a gap in the
        numbering can't produce a bogus range because the bounds are found
        by scanning the sorted numbers rather than by arithmetic.

        A multi-file HDF5 pick combined with "Combine sub-frames" > splits
        each file into several combined-frame chunks, so a FILE-index range
        is not the same as a COMBINED-FRAME-index range — file index 3 of
        10 might be combined-frame index 12, not 3. Expand the file-index
        bounds through each matched file's own chunk count (cheap header
        reads, see ``_hdf5_multi_file_counts``) so the resulting range
        lands on the right combined-frame indices; without this, a
        multi-file range silently ran short (stopping partway through an
        early file while later files in the range were never touched at
        all) as soon as any file split into more than one combined frame."""
        stride = max(1, self._fr_stride.value())
        nums = self._file_numbers()
        if nums:
            lo, hi = self._fr_start.value(), self._fr_end.value()
            i0 = next((i for i, n in enumerate(nums) if n >= lo), len(nums))
            i1 = next((i for i, n in enumerate(nums) if n > hi), len(nums)) if hi > 0 else len(nums)
            cfg = self.source_cfg()
            paths = cfg.get("paths")
            if cfg.get("type") == "hdf5_stack_glob" and paths and len(paths) == len(nums):
                counts = self._hdf5_multi_file_counts(
                    paths, cfg.get("dataset"), cfg.get("chunk_size"))
                if counts is not None and len(counts) == len(paths):
                    cum = np.cumsum([0] + list(counts))
                    i0, i1 = int(cum[i0]), int(cum[i1])
            return (i0, i1, stride)
        if not self._fr_start.isEnabled():
            # Single-file hdf5 source (see _autofill_frame_range): start/end
            # show the file's own scan number for readability, not an index
            # — there is exactly one file to process regardless of how many
            # output frames "Combine sub-frames" splits it into.
            return (0, None, 1)
        end = self._fr_end.value() if self._fr_end.value() > 0 else None
        return (self._fr_start.value(), end, stride)

    @staticmethod
    def _hdf5_multi_file_counts(paths, dataset, chunk_size):
        """Per-file combined-frame count for a multi-file HDF5 pick — header
        reads only (see ``workers._HDF5StackGlobSource._stat``), no pixel
        decode — used by ``frame_range`` to translate a FILE-index range
        into the COMBINED-FRAME-index range ``resolve_frame_indices``
        expects. Returns ``None`` if the source can't be inspected (e.g.
        ``midas_integrate_v2``/h5py unavailable), so callers fall back to
        treating file index == frame index unchanged."""
        try:
            from midas_gui.workers import _HDF5StackGlobSource
            src = _HDF5StackGlobSource(paths, dataset, chunk_size=chunk_size)
            src._ensure_stats()
            return src._counts
        except Exception:
            return None

    def _autofill_frame_range(self):
        """Fill start/end with the full valid range for the current source
        and describe what they mean, so the range can't silently select
        nothing. start/end are always FILE (scan) numbers, never a sub-frame
        or "Combine sub-frames"-chunk index — that setting is orthogonal.
        Multi-file → min/max scan number among the files; single file →
        locked equal to that one file's own scan number (parsed via
        ``workers.froot_and_frame_num``), since there's only one file and
        nothing to range over regardless of its internal chunk count."""
        if self._mode != "stream" or not hasattr(self, "_fr_start"):
            return
        from pathlib import Path
        from midas_gui.workers import froot_and_frame_num
        self._file_prefix = ""
        try:
            cfg = self.source_cfg()
        except Exception:
            return
        nums = self._file_numbers()
        for w in (self._fr_start, self._fr_end):
            w.blockSignals(True)
        try:
            # Reset from any previous single-combined-frame lock (below) so
            # switching to a source that DOES have a real range to pick
            # doesn't leave the controls stuck disabled.
            self._fr_start.setEnabled(True)
            self._fr_end.setEnabled(True)
            self._fr_stride.setEnabled(True)
            if nums:
                self._fr_start.setValue(min(nums)); self._fr_end.setValue(max(nums))
                pfx = f"{self._file_prefix}_" if self._file_prefix else ""
                extra = ""
                n_dark = getattr(self, "_n_dark_skipped", 0)
                if n_dark:
                    extra += f"  Skipped {n_dark} dark file(s)."
                # Gaps matter enough to surface: the range reads as a single
                # contiguous scan but isn't, and a missing point mid-range is
                # usually an aborted/failed acquisition worth knowing about.
                missing = (max(nums) - min(nums) + 1) - len(nums)
                if missing > 0:
                    extra += f"  ⚠ {missing} number(s) missing in this range (gaps)."
                self._fr_hint.setText(
                    f"start/end = file numbers ({len(nums)} files: {pfx}"
                    f"{min(nums):06d} … {pfx}{max(nums):06d}), end inclusive.{extra}")
            elif cfg.get("type") == "hdf5" and cfg.get("path"):
                # start/end are always FILE numbers (see the "nums" branch
                # above for the multi-file case) — never an index into this
                # file's own sub-frames or "Combine sub-frames" chunks, which
                # is an orthogonal setting. A single file is one file, so
                # start=end=its own scan number and there's nothing to range
                # over regardless of how many combined output frames
                # "Combine sub-frames" below turns it into — frame_range()
                # special-cases this via these widgets' disabled state
                # rather than a literal index lookup.
                froot, num, _tag = froot_and_frame_num(Path(cfg["path"]).stem, -1)
                shown = num if num >= 0 else 0
                self._fr_start.setValue(shown); self._fr_end.setValue(shown)
                self._fr_stride.setValue(1)
                self._fr_start.setEnabled(False)
                self._fr_end.setEnabled(False)
                self._fr_stride.setEnabled(False)
                n = 0
                try:
                    from midas_gui.workers import _open_source_cfg
                    n = int(getattr(_open_source_cfg(cfg), "n_frames", 0) or 0)
                except Exception:
                    n = 0
                produces = (f" (produces {n} combined output frames via "
                            f"'Combine sub-frames' below)" if n > 1 else "")
                self._fr_hint.setText(
                    f"Single file (scan point {shown:06d}){produces} — the "
                    "whole file is always processed as one source; nothing "
                    "to select here.")
            else:
                self._fr_hint.setText("")
        finally:
            for w in (self._fr_start, self._fr_end):
                w.blockSignals(False)

    def dark(self):
        return self._dark_sel.get_field()

    def bright(self):
        return self._bright_sel.get_field()

    def background(self):
        return self._bg_sel.get_field()

    def bright_mode(self) -> str:
        return self._bright_sel.get_mode()

    def has_pending_fields(self):
        return [s for s in (self._dark_sel, self._bright_sel, self._bg_sel) if s.has_pending()]

    def composite_mask(self):
        return self._mask_sel.composite_mask()

    def has_live_mask_source(self) -> bool:
        return self._mask_sel.has_live_mask_source()

    def set_tab1_mask(self, mask):
        self._mask_sel.set_tab1_mask(mask)

    def add_mask_file(self, path):
        """Add a mask file to the mask selector (idempotent) — for wiring a tab default."""
        self._mask_sel.add_file_source(path)

    def corrected(self, frame):
        """Apply dark/bright/background to a raw frame (mask handled separately)."""
        if frame is None:
            return None
        d, b, g = self.dark(), self.bright(), self.background()
        if d is None and b is None and g is None:
            return np.asarray(frame, dtype=np.float32)
        return apply_field_corrections(frame, dark=d, bright=b,
                                       bright_mode=self.bright_mode(),
                                       background=g).astype(np.float32)

    # ── MONITOR button (stream mode) ───────────────────────────────
    def _apply_monitor_style(self, active: bool):
        from midas_gui import style as S
        if active:
            self._monitor_btn.setText("●  MONITORING")
            self._monitor_btn.setStyleSheet(
                "QPushButton { background:#2e7d32; color:white; font-weight:bold; "
                "border:1px solid #1b5e20; border-radius:4px; padding:4px; }")
        else:
            self._monitor_btn.setText("●  MONITOR")
            self._monitor_btn.setStyleSheet(
                "QPushButton { background:#3a3d44; color:#ddd; font-weight:bold; "
                f"border:1px solid {S.ACCENT}; border-radius:4px; padding:4px; }}")

    def _on_monitor_toggled(self, on: bool):
        self._apply_monitor_style(on)
        self.monitorToggled.emit(on)

    def set_monitor_active(self, on: bool):
        """Force the MONITOR button state without re-emitting (tab-driven revert)."""
        if self._monitor_btn is None:
            return
        self._monitor_btn.blockSignals(True)
        self._monitor_btn.setChecked(on)
        self._monitor_btn.blockSignals(False)
        self._apply_monitor_style(on)

    def is_monitoring(self) -> bool:
        return bool(self._monitor_btn and self._monitor_btn.isChecked())

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def get_state(self) -> dict:
        st = {
            "path": self._path_ed.text(),
            "dataset": self._ds_combo.currentText(),
            "dark": self._dark_sel.get_state(),
            "bright": self._bright_sel.get_state(),
            "background": self._bg_sel.get_state(),
            "mask": self._mask_sel.get_state(),
        }
        if self._explicit_paths:
            st["explicit_paths"] = list(self._explicit_paths)
        if self._mode == "stream" and self._stem_filter:
            st["stem_filter"] = self._stem_filter
        if self._mode == "stream":
            st["fr_start"] = self._fr_start.value()
            st["fr_end"] = self._fr_end.value()
            st["fr_stride"] = self._fr_stride.value()
            st["combine_chunk"] = self._combine_chunk.value()
            st["combine_op"] = self._combine_op_combo.currentData()
        else:
            st["frame_index"] = self._frame_spin.value()
        if getattr(self, "_pv_ed", None) is not None:
            st["live_pv"] = self._pv_ed.currentText()
        return st

    def set_state(self, state: dict):
        """Restores path/frame/field/mask sub-state, re-triggering the panel's own
        load pipeline via :meth:`set_path` — the one central place that implements
        "auto re-load path-backed data" for every tab that embeds this panel. A
        saved live-PV name is restored into the field but a live connection is
        never auto-started (that's a stateful, non-idempotent action)."""
        if not state:
            return
        explicit = state.get("explicit_paths")
        stem_filter = state.get("stem_filter") if self._mode == "stream" else None
        path = state.get("path", "")
        if explicit:
            self._set_explicit_paths(list(explicit))
            if state.get("dataset"):
                self._ds_combo.setEditText(state["dataset"])
            self._load()
        elif stem_filter and path:
            self._set_stem_filter(path, stem_filter)
            self._load()
        elif path:
            self.set_path(path, dataset=state.get("dataset"), load=True)
        if self._mode == "stream":
            if "fr_start" in state:
                self._fr_start.setValue(int(state["fr_start"]))
            if "fr_end" in state:
                self._fr_end.setValue(int(state["fr_end"]))
            if "fr_stride" in state:
                self._fr_stride.setValue(int(state["fr_stride"]))
            if "combine_chunk" in state:
                self._combine_chunk.setValue(int(state["combine_chunk"]))
            if state.get("combine_op"):
                idx = self._combine_op_combo.findData(state["combine_op"])
                if idx >= 0:
                    self._combine_op_combo.setCurrentIndex(idx)
            self._update_combine_visibility()
        elif "frame_index" in state and self._nframes:
            self.set_frame(int(state["frame_index"]))
        self._dark_sel.set_state(state.get("dark") or {})
        self._bright_sel.set_state(state.get("bright") or {})
        self._bg_sel.set_state(state.get("background") or {})
        self._mask_sel.set_state(state.get("mask") or {})
        if getattr(self, "_pv_ed", None) is not None and state.get("live_pv"):
            self._pv_ed.setEditText(state["live_pv"])


class LossCurveViewer(QtWidgets.QWidget):
    """Live loss-vs-iteration plot for optimisation tabs (refinement, learnable, PDF)."""

    def __init__(self, parent=None, ylabel="loss"):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(2)
        self._plot = pg.PlotWidget(background="k")
        self._plot.setLabel("left", ylabel)
        self._plot.setLabel("bottom", "iteration")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._curve = self._plot.plot([], [], pen=pg.mkPen("#f0a030", width=2),
                                      symbol="o", symbolSize=4, symbolBrush="#f0a030")
        layout.addWidget(self._plot)
        self._xs: list = []
        self._ys: list = []

    def reset(self):
        self._xs.clear(); self._ys.clear()
        self._curve.setData([], [])

    def add_point(self, it: int, loss: float):
        if loss != loss:  # NaN guard
            return
        self._xs.append(it); self._ys.append(loss)
        self._curve.setData(self._xs, self._ys)


def _convert_radial(x, lsd, px, wl, native, target):
    """Convert a radial axis between R (px), 2θ (deg) and Q (Å⁻¹).

    ``native`` is the unit ``x`` is already in ("R" or "Q"); returns ``x`` unchanged
    if the target matches or the geometry (lsd/px/wl) is missing.
    """
    x = np.asarray(x, dtype=float)
    if target == native or None in (lsd, px, wl):
        return x
    lsd, px, wl = float(lsd), float(px), float(wl)
    if native == "Q":
        tth = 2.0 * np.degrees(np.arcsin(np.clip(x * wl / (4 * math.pi), -1, 1)))
    else:  # R px
        tth = np.degrees(np.arctan(x * px / lsd))
    if target == "2th":
        return tth
    if target == "Q":
        return 4 * math.pi * np.sin(np.radians(tth) / 2) / wl
    return lsd * np.tan(np.radians(tth)) / px   # target == "R"


_XUNIT_LABEL = {"R": "R (px)", "2th": "2θ (°)", "Q": "Q (Å⁻¹)"}


class _UnitAxis(pg.AxisItem):
    """Bottom axis that relabels R-pixel tick positions in a chosen radial unit.

    The image/curves stay in their native coordinates; only the tick *labels* are
    converted, so the axis is exact (no resampling) even for a nonlinear unit."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._convert = None

    def set_convert(self, fn):
        self._convert = fn
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if self._convert is None or not len(values):
            return super().tickStrings(values, scale, spacing)
        conv = self._convert(np.asarray(values, dtype=float))
        return [f"{v:.4g}" for v in conv]


class WaterfallViewer(QtWidgets.QWidget):
    """2-D waterfall of 1-D profiles: x = R (px), y = frame index, colour = intensity.

    Rows are appended incrementally as frames are integrated, so the user watches
    every frame's radial integration stack up live.  The x-axis can be shown in
    R / 2θ / Q (tick labels converted from the run's calibration).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(2)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Waterfall (all frames)"))
        bar.addWidget(QtWidgets.QLabel("  cmap:"))
        self._cmap = _NoScrollComboBox(); self._cmap.addItems(COLORMAPS); self._cmap.setFixedWidth(90)
        self._cmap.setCurrentText(_DEFAULT_CMAP)
        self._cmap.currentTextChanged.connect(self._apply_cmap)
        bar.addWidget(self._cmap)
        self._log = QtWidgets.QCheckBox("Log"); self._log.setChecked(True)
        self._log.toggled.connect(self._redraw)
        bar.addWidget(self._log)
        bar.addWidget(QtWidgets.QLabel("  x:"))
        self._xunit_combo = _NoScrollComboBox()
        self._xunit_combo.addItem("R (px)", "R")
        self._xunit_combo.addItem("2θ (°)", "2th")
        self._xunit_combo.addItem("Q (Å⁻¹)", "Q")
        self._xunit_combo.setToolTip("Label the x-axis in R (px), 2θ (deg) or Q (Å⁻¹). "
                                     "Needs the run's calibration for the conversion.")
        self._xunit_combo.currentIndexChanged.connect(self._on_xunit_changed)
        bar.addWidget(self._xunit_combo)
        bar.addStretch(1)
        self._stat = QtWidgets.QLabel(""); self._stat.setStyleSheet("color:#aaa;font-size:10px")
        bar.addWidget(self._stat)
        layout.addLayout(bar)

        self._xaxis = _UnitAxis(orientation="bottom")
        self._plot = pg.PlotWidget(background="k", axisItems={"bottom": self._xaxis})
        self._plot.setLabel("left", "frame #")
        self._plot.setLabel("bottom", "R (px)")
        self._img = pg.ImageItem()
        self._plot.addItem(self._img)

        # Color-scale sidebar, same role as pg.ImageView's built-in histogram
        # widget on the single-detector/Hydra image viewers — this viewer uses
        # a bare PlotWidget+ImageItem (not ImageView) so the histogram must be
        # wired in by hand.
        self._hist = pg.HistogramLUTWidget()
        self._hist.setImageItem(self._img)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0); row.setSpacing(2)
        row.addWidget(self._plot, stretch=1)
        row.addWidget(self._hist)
        layout.addLayout(row, stretch=1)

        # Rows are written into a growing pre-allocated buffer (no per-frame vstack),
        # and redraws are throttled so a fast/large scan stays O(N) not O(N²).
        self._buf = None            # (capacity, n_r) float64
        self._nrows = 0
        self._r_axis = None
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True); self._redraw_timer.setInterval(100)
        self._redraw_timer.timeout.connect(self._redraw)
        # Axis conversion context (from the run's calibration).
        self._lsd = self._px = self._wl = None
        self._native_unit = "R"
        self._apply_cmap(_DEFAULT_CMAP)

    # ── x-axis units (R / 2θ / Q) ─────────────────────────────────────

    def set_axis_context(self, lsd_um, px_um, wavelength_A, native_unit="R"):
        """Provide the run's geometry so the x-axis can be labelled in R / 2θ / Q."""
        self._lsd, self._px, self._wl = lsd_um, px_um, wavelength_A
        self._native_unit = native_unit if native_unit in ("R", "Q") else "R"
        self._refresh_xaxis()

    def _refresh_xaxis(self):
        target = self._xunit_combo.currentData()
        self._xaxis.set_convert(
            lambda vals: _convert_radial(vals, self._lsd, self._px, self._wl,
                                         self._native_unit, target))
        self._plot.setLabel("bottom", _XUNIT_LABEL[target])

    def _on_xunit_changed(self, _=0):
        self._refresh_xaxis()

    def reset(self, r_axis=None):
        """Start a new scan (r_axis = radial bin-centre array in px), or clear
        the view entirely when called with no axis."""
        self._buf = None
        self._nrows = 0
        self._r_axis = None if r_axis is None else np.asarray(r_axis, dtype=float)
        self._img.clear()
        self._stat.setText("")

    def add_profile(self, profile):
        """Append one frame's 1-D profile as the next waterfall row (buffered;
        the image redraw is coalesced on a timer)."""
        p = np.asarray(profile, dtype=np.float64)
        if self._buf is None:
            self._buf = np.empty((16, p.size), dtype=np.float64)
        elif self._nrows >= self._buf.shape[0]:
            self._buf = np.vstack([self._buf, np.empty_like(self._buf)])  # grow ×2
        if p.size != self._buf.shape[1]:                 # profile length changed → reset buffer
            self._buf = np.empty((max(16, self._nrows + 1), p.size), dtype=np.float64)
            self._nrows = 0
        self._buf[self._nrows] = p
        self._nrows += 1
        self._stat.setText(f"{self._nrows} frames")
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _redraw(self):
        if self._buf is None or self._nrows == 0 or self._r_axis is None:
            return
        arr = self._buf[:self._nrows]                     # (n_frames, n_r) view — no copy
        disp = np.log10(np.clip(arr, 1e-6, None)) if self._log.isChecked() else arr
        # Exclude exact-zero pixels from the level calc (masked pre-log, like
        # ImageViewer) — unfilled/zero-padded rows would otherwise skew the
        # auto-level window toward zero.
        nonzero = arr != 0
        candidates = disp[np.isfinite(disp) & nonzero]
        fin = candidates if candidates.size else disp[np.isfinite(disp)]
        # Level from a strided sample (fast + stable) rather than sorting every pixel.
        if fin.size > 200_000:
            fin = fin[:: fin.size // 200_000]
        lo, hi = (float(np.percentile(fin, 30)), float(np.percentile(fin, 99))) if fin.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        # ImageItem (col-major): pass (n_r, n_frames) so x=R, y=frame
        self._img.setImage(disp.T, autoLevels=False, levels=(lo, hi))
        r0, r1 = float(self._r_axis[0]), float(self._r_axis[-1])
        self._img.setRect(QtCore.QRectF(r0, 0.0, r1 - r0, self._nrows))
        self._apply_view_limits(r0, r1, self._nrows)

    def _apply_view_limits(self, r0: float, r1: float, nrows: int):
        """Bound pan/zoom to the waterfall's (R, frame#) extent (+ margin), same
        intent as ``ImageViewer._apply_view_limits`` — stops the user
        scrolling/zooming out into an empty void or losing the image off-screen."""
        if not math.isfinite(r0) or not math.isfinite(r1) or nrows <= 0:
            return
        rmin, rmax = min(r0, r1), max(r0, r1)
        if rmax <= rmin:
            rmax = rmin + 1.0
        rpad = 0.5 * (rmax - rmin)
        npad = 0.5 * nrows
        vb = self._plot.getPlotItem().getViewBox()
        vb.setLimits(
            xMin=max(0.0, rmin - rpad), xMax=rmax + rpad,
            yMin=-npad, yMax=nrows + npad,
            minXRange=max((rmax - rmin) * 0.01, 1e-6),
            minYRange=max(nrows * 0.01, 1.0),
            maxXRange=(rmax - rmin) + 2 * rpad,
            maxYRange=nrows + 2 * npad,
        )

    def _apply_cmap(self, name: str):
        self._hist.gradient.setColorMap(_resolve_cmap(name))

    def display_state(self) -> dict:
        """Same contract/reasoning as ``ImageViewer.display_state`` —
        WaterfallViewer has no vmin%/vmax% (its color window is a plain
        pyqtgraph HistogramLUTWidget, not a percentile pair), just cmap/log."""
        return {"cmap": self._cmap.currentText(), "log": self._log.isChecked()}

    def set_display_state(self, state: Optional[dict]) -> None:
        if not state:
            return
        cmap = state.get("cmap")
        if cmap and self._cmap.findText(str(cmap)) >= 0:
            self._cmap.blockSignals(True)
            self._cmap.setCurrentText(str(cmap))
            self._cmap.blockSignals(False)
            self._apply_cmap(str(cmap))
        if "log" in state:
            self._log.blockSignals(True)
            self._log.setChecked(bool(state["log"]))
            self._log.blockSignals(False)
        self._redraw()


def _frame_color(i: int) -> tuple:
    """Map a frame index to an RGB colour using the golden-angle hue sequence.

    Consecutive frames get maximally separated hues so individual profiles
    remain distinguishable even in a dense stack.
    """
    hue = (i * 137.508) % 360.0   # golden angle → maximum hue separation
    # HSV → RGB (saturation=0.75, value=1.0)
    h = hue / 60.0; s = 0.75; v = 1.0
    hi = int(h) % 6; f = h - int(h)
    p_ = v * (1 - s); q_ = v * (1 - s * f); t_ = v * (1 - s * (1 - f))
    r, g, b = [(v, t_, p_), (q_, v, p_), (p_, v, t_),
                (p_, q_, v), (t_, p_, v), (v, p_, q_)][hi]
    return (int(r * 255), int(g * 255), int(b * 255))


class StackedProfileViewer(QtWidgets.QWidget):
    """All batch-integration profiles drawn with a vertical Y offset.

    Each frame gets a distinct colour from the golden-angle hue sequence so
    they remain identifiable in a dense stack.  The spacing spinbox (default
    500 counts) shifts each successive frame upward; set it to 0 to overlay
    all frames for a direct comparison.
    """

    # Publication-quality categorical palette (matplotlib tab10 order) — dark,
    # print-friendly colours that read well on a white background.
    _PUB_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

    # Two saved plot configurations. "White (publication)" is the default; "Dark"
    # preserves the original on-screen look.
    _THEMES = {
        "White (publication)": dict(
            bg="white", fg="#111111", grid_alpha=0.20, symbols=True,
            symbol_size=5, line_width=1.5, box=True, palette="pub"),
        "Dark": dict(
            bg="#111111", fg="#c8c8c8", grid_alpha=0.15, symbols=False,
            symbol_size=5, line_width=1.0, box=False, palette="hue"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(2)

        # Toolbar
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("spacing:"))
        self._spacing = _NoScrollDoubleSpinBox()
        self._spacing.setRange(0.0, 1e9)
        self._spacing.setValue(500.0)
        self._spacing.setDecimals(0)
        self._spacing.setSingleStep(100.0)
        self._spacing.setSuffix("  cts")
        self._spacing.setFixedWidth(90)
        self._spacing.valueChanged.connect(self._restack)
        bar.addWidget(self._spacing)
        bar.addWidget(QtWidgets.QLabel("x:"))
        self._xunit_combo = _NoScrollComboBox()
        self._xunit_combo.addItem("R (px)", "R")
        self._xunit_combo.addItem("2θ (°)", "2th")
        self._xunit_combo.addItem("Q (Å⁻¹)", "Q")
        self._xunit_combo.setToolTip("Plot the x-axis in R (px), 2θ (deg) or Q (Å⁻¹). "
                                     "Needs the run's calibration for the conversion.")
        self._xunit_combo.currentIndexChanged.connect(self._on_xunit_changed)
        bar.addWidget(self._xunit_combo)
        bar.addWidget(QtWidgets.QLabel("theme:"))
        self._theme_combo = _NoScrollComboBox()
        self._theme_combo.addItems(list(self._THEMES.keys()))
        self._theme_combo.setToolTip("Plot appearance preset "
                                     "(White = publication point+line, Dark = classic).")
        self._theme_combo.currentTextChanged.connect(self._apply_theme)
        bar.addWidget(self._theme_combo)
        self._labels_chk = QtWidgets.QCheckBox("Labels")
        self._labels_chk.setChecked(True)
        self._labels_chk.setToolTip("Show each file's name just below its curve (left edge).")
        self._labels_chk.toggled.connect(self._toggle_labels)
        bar.addWidget(self._labels_chk)
        self._legend_chk = QtWidgets.QCheckBox("Legend")
        self._legend_chk.setChecked(False)
        self._legend_chk.setToolTip("Show a corner legend mapping each curve to its source file.")
        self._legend_chk.toggled.connect(self._toggle_legend)
        bar.addWidget(self._legend_chk)
        self._grid_chk = QtWidgets.QCheckBox("Grid")
        self._grid_chk.setChecked(False)
        self._grid_chk.setToolTip("Show the horizontal + vertical grid.")
        self._grid_chk.toggled.connect(self._toggle_grid)
        bar.addWidget(self._grid_chk)
        bar.addStretch(1)

        # Top-right controls: line width / symbol size / label font, each as
        # [−] label [+], groups separated by a vertical bar.
        def _tbtn(txt, tip, slot):
            b = QtWidgets.QToolButton(); b.setText(txt); b.setToolTip(tip)
            b.setAutoRaise(True); b.clicked.connect(slot); return b

        def _sep():
            s = QtWidgets.QLabel("|"); s.setStyleSheet("color:#888;")
            return s

        def _group(label, tip_minus, on_minus, tip_plus, on_plus):
            bar.addWidget(_tbtn("−", tip_minus, on_minus))
            bar.addWidget(QtWidgets.QLabel(label))
            bar.addWidget(_tbtn("+", tip_plus, on_plus))

        _group("line", "Thinner lines", lambda: self._adjust_linewidth(-0.5),
               "Thicker lines", lambda: self._adjust_linewidth(0.5))
        bar.addWidget(_sep())
        _group("sym", "Smaller symbols", lambda: self._adjust_symbolsize(-1),
               "Larger symbols", lambda: self._adjust_symbolsize(1))
        bar.addWidget(_sep())
        _group("font", "Smaller labels", lambda: self._adjust_fontsize(-1),
               "Larger labels", lambda: self._adjust_fontsize(1))
        self._stat = QtWidgets.QLabel("")
        self._stat.setStyleSheet("color:#aaa;font-size:10px")
        bar.addWidget(self._stat)
        layout.addLayout(bar)

        self._r_axes: list = []          # native x per curve (R px, or Q if Q-uniform)
        self._profiles: list = []
        self._curves: list = []
        self._labels: list = []          # inline pg.TextItem per curve
        self._fontsize = 9
        self._data_bounds = None         # running (xmin, xmax, ymin, ymax) across all curves
        # Axis conversion context (from the run's calibration).
        self._lsd = self._px = self._wl = None
        self._native_unit = "R"          # unit the stored x arrays are already in

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "R (px)")
        self._plot.setLabel("left", "Intensity + offset")
        self._legend = self._plot.addLegend(offset=(10, 10), labelTextSize="8pt")
        self._legend.setVisible(False)
        layout.addWidget(self._plot, stretch=1)

        # Default theme = white publication.
        self._theme = "White (publication)"
        self._theme_cfg = self._THEMES[self._theme]
        self._linewidth = self._theme_cfg["line_width"]
        self._symbol_size = self._theme_cfg["symbol_size"]
        self._apply_theme(self._theme)

    # ── public API ───────────────────────────────────────────────────

    def reset(self, r_axis=None):
        """Clear all stored profiles, curves and inline labels."""
        self._r_axes.clear()
        self._profiles.clear()
        for c in self._curves:
            self._plot.removeItem(c)
        self._curves.clear()
        for ti in self._labels:
            self._plot.removeItem(ti)
        self._labels.clear()
        if self._legend is not None:
            self._legend.clear()
        self._stat.setText("")
        self._data_bounds = None

    def add_profile(self, r_axis, profile, label=None):
        """Append one frame's 1-D profile; draws it at its stacked offset.

        ``label`` (the source file id) is shown in the corner legend and as an
        inline tag just below the curve's left-most point.
        """
        r = np.asarray(r_axis, dtype=float)
        p = np.asarray(profile, dtype=float)
        i = len(self._profiles)
        self._r_axes.append(r)
        self._profiles.append(p)
        color = self._curve_color(i)
        offset = i * float(self._spacing.value())
        xd = self._x_display(r)
        curve = self._plot.plot(xd, p + offset, name=(str(label) if label else None))
        self._style_curve(curve, i)
        self._curves.append(curve)
        # Inline label just below the line's left-most point.
        ti = pg.TextItem(str(label) if label else f"#{i}", color=color, anchor=(0, 0))
        f = QtGui.QFont(); f.setPointSize(self._fontsize); ti.setFont(f)
        if xd.size:
            ti.setPos(float(xd[0]), float(p[0] + offset))
        ti.setVisible(self._labels_chk.isChecked())
        self._plot.addItem(ti)
        self._labels.append(ti)
        n = len(self._profiles)
        self._stat.setText(f"{n} frame{'s' if n != 1 else ''}")
        yd = p + offset
        fin = np.isfinite(xd) & np.isfinite(yd)
        if fin.any():
            xmin, xmax = float(xd[fin].min()), float(xd[fin].max())
            ymin, ymax = float(yd[fin].min()), float(yd[fin].max())
            if self._data_bounds is not None:
                xmin = min(xmin, self._data_bounds[0]); xmax = max(xmax, self._data_bounds[1])
                ymin = min(ymin, self._data_bounds[2]); ymax = max(ymax, self._data_bounds[3])
            self._data_bounds = (xmin, xmax, ymin, ymax)
            self._apply_view_limits(*self._data_bounds)

    # ── x-axis units (R / 2θ / Q) ─────────────────────────────────────

    def set_axis_context(self, lsd_um, px_um, wavelength_A, native_unit="R"):
        """Provide the run's geometry so the x-axis can be shown in R / 2θ / Q.

        ``native_unit`` is the unit the profiles arrive in ("R" px, or "Q" when
        Q-uniform binning is active)."""
        self._lsd, self._px, self._wl = lsd_um, px_um, wavelength_A
        self._native_unit = native_unit if native_unit in ("R", "Q") else "R"
        self._restack()
        self._plot.setLabel("bottom", self._xlabel(),
                            **{"color": self._theme_cfg["fg"], "font-size": "11pt"})

    def _x_display(self, x_native):
        """Convert a native x array to the currently-selected unit."""
        return _convert_radial(x_native, self._lsd, self._px, self._wl,
                               self._native_unit, self._xunit_combo.currentData())

    def _xlabel(self) -> str:
        return _XUNIT_LABEL[self._xunit_combo.currentData()]

    def _on_xunit_changed(self, _=0):
        self._restack()
        self._plot.setLabel("bottom", self._xlabel(),
                            **{"color": self._theme_cfg["fg"], "font-size": "11pt"})

    def _toggle_grid(self, on: bool):
        self._plot.showGrid(x=on, y=on, alpha=self._theme_cfg["grid_alpha"])

    # ── theme + styling ───────────────────────────────────────────────

    def _curve_color(self, i: int):
        """Per-curve colour for the active theme."""
        if self._theme_cfg["palette"] == "pub":
            return self._PUB_PALETTE[i % len(self._PUB_PALETTE)]
        return _frame_color(i)

    def _style_curve(self, curve, i: int):
        """Apply the active theme's pen + point/line style to one curve."""
        col = self._curve_color(i)
        curve.setPen(pg.mkPen(col, width=self._linewidth))
        if self._theme_cfg["symbols"]:
            curve.setSymbol("o")
            curve.setSymbolSize(self._symbol_size)
            curve.setSymbolBrush(pg.mkBrush(col))
            curve.setSymbolPen(pg.mkPen(col))
        else:
            curve.setSymbol(None)

    def _apply_theme(self, name: str):
        cfg = self._THEMES.get(name)
        if cfg is None:
            return
        self._theme = name
        self._theme_cfg = cfg
        self._linewidth = cfg["line_width"]
        self._symbol_size = cfg["symbol_size"]
        self._plot.setBackground(cfg["bg"])
        pen = pg.mkPen(cfg["fg"], width=1)
        for ax_name in ("bottom", "left", "top", "right"):
            ax = self._plot.getAxis(ax_name)
            ax.setPen(pen); ax.setTextPen(pen)
        # Box frame (all four spines) for the publication theme.
        self._plot.showAxis("top", cfg["box"]); self._plot.showAxis("right", cfg["box"])
        if cfg["box"]:
            for ax_name in ("top", "right"):
                self._plot.getAxis(ax_name).setStyle(showValues=False)
        grid_on = self._grid_chk.isChecked()
        self._plot.showGrid(x=grid_on, y=grid_on, alpha=cfg["grid_alpha"])
        lbl = {"color": cfg["fg"], "font-size": "11pt"}
        self._plot.setLabel("bottom", self._xlabel(), **lbl)
        self._plot.setLabel("left", "Intensity + offset", **lbl)
        try:
            self._legend.setLabelTextColor(cfg["fg"])
        except Exception:
            pass
        # Restyle existing curves + inline labels.
        for i, curve in enumerate(self._curves):
            self._style_curve(curve, i)
        for i, ti in enumerate(self._labels):
            ti.setColor(self._curve_color(i))
        if self._theme_combo.currentText() != name:
            self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentText(name)
            self._theme_combo.blockSignals(False)

    def _toggle_legend(self, on: bool):
        if self._legend is not None:
            self._legend.setVisible(on)

    def _toggle_labels(self, on: bool):
        for ti in self._labels:
            ti.setVisible(on)

    def _adjust_linewidth(self, delta: float):
        self._linewidth = min(8.0, max(0.5, self._linewidth + delta))
        for i, curve in enumerate(self._curves):
            self._style_curve(curve, i)

    def _adjust_symbolsize(self, delta: int):
        """Grow/shrink the point markers (point+line themes). Also turns markers
        on if the current theme was line-only, so the control always has effect."""
        self._symbol_size = min(20, max(1, self._symbol_size + delta))
        if not self._theme_cfg.get("symbols"):
            self._theme_cfg = dict(self._theme_cfg, symbols=True)
        for i, curve in enumerate(self._curves):
            self._style_curve(curve, i)

    def _adjust_fontsize(self, delta: int):
        self._fontsize = min(24, max(5, self._fontsize + delta))
        f = QtGui.QFont(); f.setPointSize(self._fontsize)
        for ti in self._labels:
            ti.setFont(f)

    # ── internal ─────────────────────────────────────────────────────

    def _restack(self, _=None):
        """Redraw all curves + inline labels (offsets, spacing, x-unit)."""
        spacing = float(self._spacing.value())
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        for i, (curve, r, p) in enumerate(
                zip(self._curves, self._r_axes, self._profiles)):
            xd = self._x_display(r)
            yd = p + i * spacing
            curve.setData(xd, yd)
            if i < len(self._labels) and xd.size:
                self._labels[i].setPos(float(xd[0]), float(p[0] + i * spacing))
            fin = np.isfinite(xd) & np.isfinite(yd)
            if fin.any():
                xmins.append(float(xd[fin].min())); xmaxs.append(float(xd[fin].max()))
                ymins.append(float(yd[fin].min())); ymaxs.append(float(yd[fin].max()))
        if self._curves:
            self._plot.autoRange()
        if xmins:
            self._data_bounds = (min(xmins), max(xmaxs), min(ymins), max(ymaxs))
            self._apply_view_limits(*self._data_bounds)

    def _apply_view_limits(self, xmin, xmax, ymin, ymax):
        """Bound pan/zoom to the stacked profiles' combined data (+ margin),
        same intent as ``ProfileViewer._apply_view_limits`` — stops the user
        scrolling/zooming arbitrarily far from where the data actually is."""
        if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
            return
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        xpad = 0.15 * (xmax - xmin)
        ypad = 0.15 * (ymax - ymin)
        vb = self._plot.getPlotItem().getViewBox()
        vb.setLimits(xMin=max(0.0, xmin - xpad), xMax=xmax + xpad,
                     yMin=ymin - ypad, yMax=ymax + ypad,
                     maxXRange=(xmax - xmin) + 2 * xpad,
                     maxYRange=(ymax - ymin) + 2 * ypad)


class LogPanel(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(1000)
        self.setFont(_mono_font(9))
        self.setMaximumHeight(120)

    def append(self, line: str):
        self.appendPlainText(line)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
