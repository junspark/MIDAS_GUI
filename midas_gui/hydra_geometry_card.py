"""Reusable ring-simulation + calibration-load/save + radial-integration
widget, extracted from the Data Viewer tab (``tab_view.py``) so the same
implementation can be shared by the single-detector view and each of the
Hydra (4-panel GE detector) view's per-panel geometry cards.

A ``DetectorGeometryCard`` owns its own materials list, geometry fields
(wavelength/Lsd/pixel/BC/tilt), ring overlay, and calibration file
load/save — but it does *not* own an image, a viewer, or a profile plot.
Those are bound in via ``set_image_source``/``set_viewer``/``set_profile_view``
(and, for the shared radial-integration toolbar controls, ``set_radial_controls``)
so the same card class works whether there's one detector (bound once, for
the tab's lifetime) or several (rebound each time the active panel changes).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from midas_gui.constants import (MATERIALS, DEFAULT_WAVELENGTH, DEFAULT_PIXEL_UM,
                           DEFAULT_LSD_UM, DEFAULT_BC_Y, DEFAULT_BC_Z, DEFAULT_RING_WIDTH,
                           DEFAULT_STEP_WAVELENGTH, DEFAULT_STEP_TWO_THETA,
                           DEFAULT_STEP_LSD_MM, DEFAULT_STEP_PIXEL, DEFAULT_STEP_BC,
                           DEFAULT_STEP_TILT)
from midas_gui.dialogs import show_error
from midas_gui.helpers import (_fspin, _NoScrollSpinBox, _browse,
                         simulate_rings, simulate_rings_from_dspacings,
                         read_geometry, geometry_fields_from_file,
                         _spec_from_result_ns, _NoScrollComboBox,
                         make_kedge_label, make_pixel_label, tilted_ring_xy,
                         write_poni, write_standalone_paramstest,
                         im_trans_codes_from_checkboxes, _apply_im_trans,
                         parse_dspacing_text)
from midas_gui.workers import build_integration_context, integrate_frame
from midas_gui import style as S

# Default ring colors assigned to new materials, cycled by row count. First
# entry matches the single hardcoded ring color the old single-material UI used.
_MATERIAL_COLORS = ("#f0c060", "#4fc3f7", "#ab47bc", "#66bb6a", "#ef5350",
                     "#ffca28", "#26a69a", "#ec407a", "#7e57c2", "#8d6e63")


_CUSTOM_DSPACING = "Custom (d-spacings)"

# Lattice defaults used to seed the (hidden) lattice widgets when a material
# has no a/b/c/.../sg of its own yet — a "dspacing"-kind material (e.g.
# AgBH) or a brand-new dialog opened straight into d-spacing mode.
_FALLBACK_LATTICE = dict(a=5.4116, b=5.4116, c=5.4116, alpha=90.0, beta=90.0, gamma=90.0, sg=225)


class MaterialDialog(QtWidgets.QDialog):
    """Edit one ring-simulation material: name, preset, and either a
    lattice + space group (crystalline) or an explicit list of d-spacings
    (non-crystalline standards, e.g. silver behenate)."""

    def __init__(self, material: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Material")
        v = QtWidgets.QVBoxLayout(self)

        self._name = QtWidgets.QLineEdit(material["name"])
        v.addLayout(S.Form().row(("Name:", self._name)))

        self._preset = _NoScrollComboBox()
        for name in MATERIALS:
            self._preset.addItem(name)
        self._preset.addItem("Custom")
        self._preset.addItem(_CUSTOM_DSPACING)
        default_preset = _CUSTOM_DSPACING if material.get("kind") == "dspacing" else "Custom"
        idx = self._preset.findText(material.get("preset", default_preset))
        self._preset.setCurrentIndex(idx if idx >= 0 else self._preset.findText(default_preset))
        self._preset.currentTextChanged.connect(self._on_preset)
        v.addLayout(S.Form().row(("Preset:", self._preset)))

        latt0 = _FALLBACK_LATTICE if material.get("kind") == "dspacing" else material
        _LW, _AW = 78, 66     # compact lattice / angle-SG cell widths
        self._a = _fspin(0.1, 100.0, 3, latt0["a"]); self._a.setFixedWidth(_LW)
        self._b = _fspin(0.1, 100.0, 3, latt0["b"]); self._b.setFixedWidth(_LW)
        self._c = _fspin(0.1, 100.0, 3, latt0["c"]); self._c.setFixedWidth(_LW)
        self._al = _fspin(1.0, 179.0, 2, latt0["alpha"]); self._al.setFixedWidth(_AW)
        self._be = _fspin(1.0, 179.0, 2, latt0["beta"]); self._be.setFixedWidth(_AW)
        self._ga = _fspin(1.0, 179.0, 2, latt0["gamma"]); self._ga.setFixedWidth(_AW)
        self._sg = _NoScrollSpinBox(); self._sg.setRange(1, 230); self._sg.setValue(latt0["sg"])
        self._sg.setFixedWidth(_AW)
        self._cubic = QtWidgets.QCheckBox("Cubic (a=b=c, α=β=γ=90°)")
        self._cubic.setToolTip("Enter only a — b and c mirror it and all angles are fixed at 90°.")
        self._cubic.setChecked(bool(material.get("cubic", False)))
        self._cubic.toggled.connect(self._apply_mode)
        self._a.valueChanged.connect(self._on_a_changed)
        self._latt_widget = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(self._latt_widget)
        lv.setContentsMargins(0, 0, 0, 0)
        latt = S.Form()
        latt.row(("a:", self._a), ("b:", self._b), ("c:", self._c))
        latt.row(("α:", self._al), ("β:", self._be), ("γ:", self._ga))
        latt.row(("SG #:", self._sg))
        lv.addLayout(latt)
        lv.addWidget(self._cubic)
        v.addWidget(self._latt_widget)

        self._dsp_widget = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(self._dsp_widget)
        dv.setContentsMargins(0, 0, 0, 0)
        self._dsp_ed = QtWidgets.QLineEdit(self._format_d_list(material.get("d_list", [])))
        self._dsp_ed.setToolTip(
            "Space- or comma-separated d-spacings in Angstrom, largest first "
            "(e.g. a lamellar standard's harmonic series). No space group or "
            "hkl — rings are labelled by order (n1, n2, ...) instead.")
        dv.addLayout(S.Form().row(("d-spacings (Å):", self._dsp_ed)))
        v.addWidget(self._dsp_widget)

        self._apply_mode()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    @staticmethod
    def _format_d_list(d_list) -> str:
        return " ".join(f"{d:.4f}" for d in d_list)

    def _current_mode(self) -> str:
        name = self._preset.currentText()
        if name == _CUSTOM_DSPACING:
            return "dspacing"
        if name in MATERIALS and MATERIALS[name].get("kind") == "dspacing":
            return "dspacing"
        return "lattice"

    def _on_preset(self, name: str):
        if name not in ("Custom", _CUSTOM_DSPACING) and name in MATERIALS:
            m = MATERIALS[name]
            self._name.setText(name)
            if m.get("kind") == "dspacing":
                self._dsp_ed.setText(self._format_d_list(m["d_list"]))
            else:
                for w, k in ((self._a, "a"), (self._b, "b"), (self._c, "c"),
                             (self._al, "alpha"), (self._be, "beta"), (self._ga, "gamma")):
                    w.blockSignals(True); w.setValue(m[k]); w.blockSignals(False)
                self._sg.setValue(m["sg"])
        self._apply_mode()

    def _apply_mode(self, *_):
        """Show/enable the lattice grid or the d-spacing field depending on
        the selected preset; within lattice mode, enable editing only for a
        custom material ('Cubic' further locks b, c and the angles)."""
        dspacing = self._current_mode() == "dspacing"
        self._latt_widget.setVisible(not dspacing)
        self._dsp_widget.setVisible(dspacing)

        custom = self._preset.currentText() == "Custom"
        self._cubic.setEnabled(custom)
        cubic = custom and self._cubic.isChecked()
        self._a.setEnabled(custom); self._sg.setEnabled(custom)
        for w in (self._b, self._c, self._al, self._be, self._ga):
            w.setEnabled(custom and not cubic)
        if cubic:
            self._sync_cubic()

        self._dsp_ed.setEnabled(self._preset.currentText() == _CUSTOM_DSPACING)

    def _sync_cubic(self):
        v = self._a.value()
        for w in (self._b, self._c):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        for w in (self._al, self._be, self._ga):
            w.blockSignals(True); w.setValue(90.0); w.blockSignals(False)

    def _on_a_changed(self, *_):
        if self._cubic.isEnabled() and self._cubic.isChecked():
            self._sync_cubic()

    @staticmethod
    def _parse_d_list(text: str) -> list:
        return parse_dspacing_text(text)

    def apply_to(self, material: dict):
        """Write the dialog's current values back into ``material``."""
        material["name"] = self._name.text().strip() or material["name"]
        material["preset"] = self._preset.currentText()
        material["kind"] = self._current_mode()
        if material["kind"] == "dspacing":
            material["d_list"] = self._parse_d_list(self._dsp_ed.text())
        else:
            material["a"] = self._a.value(); material["b"] = self._b.value(); material["c"] = self._c.value()
            material["alpha"] = self._al.value(); material["beta"] = self._be.value()
            material["gamma"] = self._ga.value()
            material["sg"] = self._sg.value()
            material["cubic"] = self._cubic.isChecked()


class DetectorGeometryCard(QtWidgets.QWidget):
    """Ring simulation + calibration load/save + radial integration for one
    detector's geometry. Bind it to the widgets it needs to act on:

    - ``set_image_source(image_provider, mask_provider=None)`` — callables
      returning the current 2-D frame (or None) and an optional bad-pixel
      mask for that frame.
    - ``set_viewer(viewer)`` — a ``PickableImageViewer``/``ROIImageViewer``
      to draw rings on and receive BC-pick/ring-fit signals from. Safe to
      rebind (e.g. when switching the active Hydra panel): the card clears
      its overlays off the old viewer first.
    - ``set_profile_view(profile_view)`` — a ``ProfileViewer``-compatible
      sink (``set_profile``/``set_ring_markers``) for the radial plot.
    - ``set_radial_controls(r_bin_spin, auto_checkbox)`` — externally-owned
      widgets (shown in the profile view's own toolbar), since these may be
      shared across several cards (e.g. one R-bin/Auto setting for all 4
      Hydra panels) rather than duplicated per card.
    """

    pushGeometry = QtCore.pyqtSignal(dict)    # "-> Send geometry to Calibrate" clicked
    imTransChanged = QtCore.pyqtSignal()      # a Flip Y/Flip Z/Transpose checkbox toggled
    geometryChanged = QtCore.pyqtSignal()     # BC/tilt/calibration changed (edit, pick, or file load)

    def __init__(self, parent=None, *, show_rotate: bool = False):
        super().__init__(parent)
        self._show_rotate = show_rotate
        self._materials: list = []
        self._ring_items: list = []
        self._label_items: list = []
        self._pick_ring_item = None
        self._picked_r: Optional[float] = None
        self._calib_geom: Optional[dict] = None
        self._calib_ctx = None
        self._calib_ctx_sig = None
        self._rad_grid_cache = None

        self._viewer = None
        self._profile_view = None
        self._cake_view = None
        self._image_provider: Callable[[], Optional[np.ndarray]] = lambda: None
        self._mask_provider: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None
        self._rad_r_bin: Optional[QtWidgets.QDoubleSpinBox] = None
        self._rad_auto: Optional[QtWidgets.QCheckBox] = None

        self._build_ui()

    # ── Wiring ───────────────────────────────────────────────────

    def set_image_source(self, image_provider: Callable[[], Optional[np.ndarray]],
                          mask_provider: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None):
        self._image_provider = image_provider
        self._mask_provider = mask_provider

    def set_profile_view(self, profile_view):
        self._profile_view = profile_view

    def set_cake_view(self, cake_view):
        self._cake_view = cake_view

    def set_radial_controls(self, r_bin_spin: QtWidgets.QDoubleSpinBox,
                             auto_checkbox: QtWidgets.QCheckBox):
        self._rad_r_bin = r_bin_spin
        self._rad_auto = auto_checkbox
        r_bin_spin.valueChanged.connect(self._on_rad_param_changed)

    def set_viewer(self, viewer):
        """Bind (or rebind) the image viewer this card draws rings on and
        receives beam-centre picks from. Clears this card's overlays off the
        previous viewer first, so switching the active panel in Hydra mode
        never leaves stale rings behind on a viewer another card now owns.
        Also clears the viewer's own in-progress Pick BC/Pick Ring click
        state (it's a single viewer shared by all panels) so points picked
        while this panel was active can never leak into another panel's
        circle fit."""
        old = self._viewer
        if old is not None:
            try:
                old.bcPicked.disconnect(self._on_bc_picked)
            except TypeError:
                pass
            try:
                old.ringFitBC.disconnect(self._on_ring_fit_bc)
            except TypeError:
                pass
            for it in self._ring_items + self._label_items:
                old._iv.removeItem(it)
            if self._pick_ring_item is not None:
                old._iv.removeItem(self._pick_ring_item)
            old._clear_ring_points()
        self._ring_items = []; self._label_items = []; self._pick_ring_item = None
        self._viewer = viewer
        if viewer is not None:
            viewer.bcPicked.connect(self._on_bc_picked)
            viewer.ringFitBC.connect(self._on_ring_fit_bc)
            self._redraw_rings()
            self._redraw_picked_ring()

    def bc_auto_enabled(self) -> bool:
        return self._bc_auto.isChecked()

    def center_beam_on(self, ny: float, nz: float):
        """Set the beam centre to the image centre without retriggering
        ``_on_bc_changed`` — used when 'Beam centre = image centre' is
        checked and a fresh frame/projection arrives."""
        for w, v in ((self._bcy, ny), (self._bcz, nz)):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)

    def im_trans_codes(self) -> list:
        """Ordered MIDAS ImTransOpt codes from the Transforms checkboxes."""
        return im_trans_codes_from_checkboxes(self._flip_y, self._flip_z, self._transp)

    def rotate_deg(self) -> float:
        """Per-panel-only clockwise display rotation (degrees) — deliberately
        NOT part of im_trans_codes()/get_geometry(), so it never reaches
        DetectorState/composite geometry or calibration-file export."""
        return self._rotate.value() if self._rotate is not None else 0.0

    def any_material_rings(self) -> bool:
        return self._any_material_rings()

    def refresh_rings_and_radial(self):
        """Call after the bound image changes (new frame/file) — redraws any
        existing ring overlay and the picked-radius ring, then reintegrates
        if Auto is on."""
        if self._ring_items or self._label_items:
            self._redraw_rings()
        self._redraw_picked_ring()
        self._maybe_auto_radial()

    def refresh_after_projection(self):
        """Call after a stack projection completes — mirrors
        ``refresh_rings_and_radial`` but keys the ring redraw off whether any
        material has simulated rings (matches the pre-extraction behavior)."""
        if self._any_material_rings():
            self._redraw_rings()
        self._maybe_auto_radial()

    def maybe_auto_radial(self):
        """Call after something changed that only affects the profile, not
        ring placement (e.g. dark/bright/mask correction) — reintegrates if
        Auto is on, without touching the ring overlay."""
        self._maybe_auto_radial()

    def _maybe_auto_radial(self):
        if self._rad_auto is not None and self._rad_auto.isChecked():
            self.radial_integrate()

    def refresh_geometry(self):
        """Public alias for the post-geometry-change refresh (redraw rings,
        reintegrate if Auto is on) — used by an owner restoring saved state."""
        self._after_geometry_change()

    def _after_geometry_change(self):
        """Common tail of every action that changes the effective geometry
        (simulate, load calibration, apply an external geometry dict, or a
        saved-state restore): redraw rings, then reintegrate if Auto is on,
        otherwise just refresh the ring markers already on the plot."""
        self._redraw_rings()
        if self._rad_auto is not None and self._rad_auto.isChecked():
            self.radial_integrate()
        else:
            self._refresh_profile_markers()
        self.geometryChanged.emit()

    # ── Public geometry API (mirrors the pre-extraction DataViewerTab API) ──

    def get_geometry(self) -> dict:
        """Current manual geometry — λ (Å), pixel (µm), Lsd (µm), beam centre (px).

        Lsd is entered in mm (display) but always returned/used in µm."""
        return {
            "wavelength_A": self._wl.value(),
            "pxY": self._px.value(),
            "Lsd": self._lsd_um(),
            "BC_y": self._bcy.value(),
            "BC_z": self._bcz.value(),
            "ty": self._ty.value(),
            "tz": self._tz.value(),
            "im_trans": self.im_trans_codes(),
        }

    def _lsd_um(self) -> float:
        """Lsd in µm (internal unit) from the mm display field."""
        return self._lsd.value() * 1000.0

    # ── Shared-field sync (Hydra: λ / max 2θ / px mirrored across panels) ──

    def get_shared_fields(self) -> dict:
        """λ, max 2θ, and pixel size — the 3 fields an owner (the Hydra page)
        may mirror across several cards, since the same X-ray beam and GE
        detector model make them physically identical for every panel."""
        return {
            "wavelength_A": self._wl.value(),
            "max2theta": self._max2t.value(),
            "pxY": self._px.value(),
        }

    def apply_shared_fields(self, fields: dict):
        """Apply a shared-field dict from another card without re-emitting
        this card's own change signals (the owner already knows it's
        propagating a sync) — then refresh exactly as ``_on_sim_param_changed``
        would for a direct edit: resimulate if live ring simulation is on
        (ring radii themselves depend on λ/px/max2θ, not just their on-image
        position), otherwise redraw/reintegrate with the new values."""
        for w, key in ((self._wl, "wavelength_A"), (self._max2t, "max2theta"),
                       (self._px, "pxY")):
            v = fields.get(key)
            if v is not None:
                w.blockSignals(True); w.setValue(float(v)); w.blockSignals(False)
        if self._sim_btn.isChecked() and self._image_provider() is not None:
            self._simulate()
        else:
            self._after_geometry_change()

    def set_geometry(self, g: dict):
        """Replace the manual-geometry fields from a geometry dict (e.g. the Calibrate
        tab's result). Values are µm/Å/px; the Lsd field displays mm."""
        if not g:
            return
        for w, key, scale in ((self._wl, "wavelength_A", 1.0), (self._px, "pxY", 1.0),
                              (self._lsd, "Lsd", 0.001), (self._bcy, "BC_y", 1.0),
                              (self._bcz, "BC_z", 1.0), (self._ty, "ty", 1.0),
                              (self._tz, "tz", 1.0)):
            v = g.get(key)
            if v is not None:
                w.blockSignals(True); w.setValue(float(v) * scale); w.blockSignals(False)
        if g.get("BC_y") is not None or g.get("BC_z") is not None:
            self._bc_auto.setChecked(False)   # use the supplied beam centre
        if g.get("im_trans") is not None:
            im_trans = g["im_trans"] or []
            self._flip_y.setChecked(1 in im_trans)
            self._flip_z.setChecked(2 in im_trans)
            self._transp.setChecked(3 in im_trans)
        has_full = any(g.get(k) not in (None, 0, 0.0) for k in ("tx", "ty", "tz")) \
            or bool(g.get("distortion")) \
            or (g.get("NrPixelsY") and g.get("NrPixelsZ"))
        if has_full:
            self._apply_full_geometry_dict(g)
        self._after_geometry_change()

    def _apply_full_geometry_dict(self, g: dict):
        """Build ``self._calib_geom`` from a full geometry dict (tilts/distortion/
        detector size) so the tilt/distortion-aware radial engine is used."""
        px = float(g.get("pxY") or 0.0)
        geom = {
            "wavelength_A": g.get("wavelength_A"),
            "Lsd": g.get("Lsd"), "BC_y": g.get("BC_y"), "BC_z": g.get("BC_z"),
            "tx": float(g.get("tx", 0.0) or 0.0),
            "ty": float(g.get("ty", 0.0) or 0.0),
            "tz": float(g.get("tz", 0.0) or 0.0),
            "pxY": px, "pxZ": float(g.get("pxZ") or px),
            "NrPixelsY": g.get("NrPixelsY"), "NrPixelsZ": g.get("NrPixelsZ"),
            "distortion": dict(g.get("distortion") or {}),
            "im_trans": list(g.get("im_trans") or []),
        }
        img = self._image_provider()
        if not (geom["NrPixelsY"] and geom["NrPixelsZ"]) and img is not None:
            nz, ny = img.shape
            geom["NrPixelsY"], geom["NrPixelsZ"] = ny, nz
        required = ("wavelength_A", "Lsd", "BC_y", "BC_z", "pxY",
                    "NrPixelsY", "NrPixelsZ")
        if any(not geom.get(k) for k in required):
            return   # incomplete — keep circle binning
        self._calib_geom = geom
        self._calib_ctx = self._calib_ctx_sig = None
        tilt = any(abs(geom[k]) > 1e-9 for k in ("tx", "ty", "tz"))
        mode = ("full integration: tilts"
                + ("+distortion" if geom["distortion"] else "")
                if (tilt or geom["distortion"]) else "full integration")
        self._calib_lbl.setText(f"Geometry from Calibrate tab  ·  {mode}")

    # ── GUI state ────────────────────────────────────────────────

    def state_widgets(self) -> dict:
        d = {
            "calib_ed": self._calib_ed,
            "wl": self._wl,
            "lsd": self._lsd,
            "px": self._px,
            "max2t": self._max2t,
            "bc_auto": self._bc_auto,
            "bcy": self._bcy,
            "bcz": self._bcz,
            "ty": self._ty,
            "tz": self._tz,
            "flip_y": self._flip_y,
            "flip_z": self._flip_z,
            "transp": self._transp,
            "show_rings": self._show_rings,
            "show_labels": self._show_labels,
            "ring_width": self._ring_width,
        }
        if self._rotate is not None:
            d["rotate"] = self._rotate
        return d

    def materials_state(self) -> list:
        return [{k: v for k, v in m.items() if not k.startswith("_")}
                for m in self._materials]

    def set_calib_path(self, path: str):
        """Set the calibration-file path field and load it if it exists —
        used when restoring saved GUI state (a saved value should always win
        over whatever loading re-triggers as a default)."""
        self._calib_ed.setText(path)
        if path and Path(path).exists():
            self._load_calibration()

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        lv = QtWidgets.QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)

        def _br(w=30):
            b = QtWidgets.QPushButton("…"); b.setFixedWidth(w); return b

        def _frow(ed, slot):
            r = QtWidgets.QHBoxLayout(); r.setSpacing(4)
            r.addWidget(ed); b = _br(); b.clicked.connect(slot); r.addWidget(b); return r

        # ── Ring simulation card ──
        ring = S.make_card("Ring simulation")
        self._materials_box = QtWidgets.QVBoxLayout()
        self._materials_box.setSpacing(3)
        ring.body.addLayout(self._materials_box)
        self._add_material("Ni (FCC)")
        add_mat_btn = QtWidgets.QPushButton("+ Add material")
        add_mat_btn.setToolTip("Overlay rings from another material simultaneously.")
        add_mat_btn.clicked.connect(lambda: self._add_material())
        ring.body.addWidget(add_mat_btn)
        ring.body.addWidget(S.hline())

        self._wl = _fspin(0.0001, 1e6, 4, DEFAULT_WAVELENGTH, "Å", step=DEFAULT_STEP_WAVELENGTH)
        # Lsd is shown/entered in mm (calculations & files still use µm).
        self._lsd = _fspin(0.001, 1e6, 3, DEFAULT_LSD_UM / 1000.0, " mm", step=DEFAULT_STEP_LSD_MM)
        self._lsd.setFixedWidth(120)
        self._px = _fspin(0.1, 1e6, 2, DEFAULT_PIXEL_UM, "µm", step=DEFAULT_STEP_PIXEL)
        self._max2t = _fspin(0.001, 180.0, 1, 25.0, "°", step=DEFAULT_STEP_TWO_THETA)
        geo = S.Form()
        geo.row((make_kedge_label(self._wl, "λ:"), self._wl), ("max 2θ:", self._max2t))
        geo.row(("Lsd:", self._lsd), (make_pixel_label(self._px, "px:"), self._px))
        ring.body.addLayout(geo)

        self._flip_y = QtWidgets.QCheckBox("Flip Y"); self._flip_z = QtWidgets.QCheckBox("Flip Z")
        self._transp = QtWidgets.QCheckBox("Transpose")
        self._flip_y.setToolTip(
            "MIDAS ImTransOpt image transform, applied to the raw detector\n"
            "image before display/integration and saved into calibration files.")
        tb_trans = QtWidgets.QHBoxLayout(); tb_trans.setSpacing(8)
        tb_trans.addWidget(self._flip_y); tb_trans.addWidget(self._flip_z)
        tb_trans.addWidget(self._transp)
        if self._show_rotate:
            self._rotate = _fspin(-360.0, 360.0, 2, 0.0, "°")
            self._rotate.setFixedWidth(76)
            self._rotate.setToolTip(
                "Clockwise rotation applied to this panel's own raw display/\n"
                "radial-integration image only — NOT applied to the Composite view.")
            tb_trans.addWidget(QtWidgets.QLabel("Rotate:"))
            tb_trans.addWidget(self._rotate)
        else:
            self._rotate = None
        tb_trans.addStretch(1)
        trans_card = S.make_card("Transforms")
        trans_card.body.addLayout(tb_trans)
        for cb in (self._flip_y, self._flip_z, self._transp):
            cb.toggled.connect(self.imTransChanged.emit)
        if self._rotate is not None:
            self._rotate.valueChanged.connect(self.imTransChanged.emit)

        self._bc_auto = QtWidgets.QCheckBox("Beam centre = image centre"); self._bc_auto.setChecked(True)
        ring.body.addWidget(self._bc_auto)
        self._bcy = _fspin(-1e5, 1e5, 1, DEFAULT_BC_Y, "px", step=DEFAULT_STEP_BC)
        self._bcz = _fspin(-1e5, 1e5, 1, DEFAULT_BC_Z, "px", step=DEFAULT_STEP_BC)
        self._bcy.setEnabled(False); self._bcz.setEnabled(False)
        self._bc_auto.toggled.connect(lambda c: (self._bcy.setEnabled(not c), self._bcz.setEnabled(not c)))
        ring.body.addLayout(S.Form().row(("BC_y:", self._bcy), ("BC_z:", self._bcz)))

        self._ty = _fspin(-180.0, 180.0, 2, 0.0, "°", step=DEFAULT_STEP_TILT)
        self._tz = _fspin(-180.0, 180.0, 2, 0.0, "°", step=DEFAULT_STEP_TILT)
        self._ty.setToolTip("Detector tilt about the Y axis — bends the simulated rings.")
        self._tz.setToolTip("Detector tilt about the Z axis — bends the simulated rings.")
        ring.body.addLayout(S.Form().row(("ty:", self._ty), ("tz:", self._tz)))

        # Send λ / pixel / Lsd / beam-centre to the Calibrate tab (seed values).
        self._to_calib_btn = QtWidgets.QPushButton("→ Send geometry to Calibrate")
        self._to_calib_btn.setToolTip(
            "Copy λ, pixel size, Lsd and beam centre from here into the Calibrate "
            "tab's detector + seed fields.")
        self._to_calib_btn.clicked.connect(
            lambda: self.pushGeometry.emit(self.get_geometry()))
        ring.body.addWidget(self._to_calib_btn)

        ctl = QtWidgets.QHBoxLayout()
        self._show_rings = QtWidgets.QCheckBox("Rings"); self._show_rings.setChecked(True)
        self._show_rings.toggled.connect(self._set_rings_visible)
        self._show_labels = QtWidgets.QCheckBox("Labels"); self._show_labels.setChecked(True)
        self._show_labels.toggled.connect(self._set_rings_visible)
        self._ring_width = _fspin(0.5, 10.0, 1, DEFAULT_RING_WIDTH, "px")
        self._ring_width.setToolTip("Line thickness of the simulated rings on the image.")
        self._ring_width.setMaximumWidth(80)
        self._ring_width.valueChanged.connect(self._redraw_rings)
        ctl.addWidget(self._show_rings); ctl.addWidget(self._show_labels)
        ctl.addSpacing(8)
        ctl.addWidget(QtWidgets.QLabel("thickness:"))
        ctl.addWidget(self._ring_width)
        ctl.addStretch(1)
        ring.body.addLayout(ctl)
        self._sim_btn = S.primary_btn("Simulate rings")
        self._sim_btn.setCheckable(True)
        self._sim_btn.setToolTip(
            "Toggle live ring simulation — while on, rings recompute automatically "
            "whenever material, lattice, or geometry parameters change.")
        self._sim_btn.toggled.connect(self._on_sim_toggled)
        ring.body.addWidget(self._sim_btn)
        for w in (self._wl, self._lsd, self._px, self._max2t):
            w.valueChanged.connect(self._on_sim_param_changed)
        self._ring_info = QtWidgets.QPlainTextEdit(); self._ring_info.setReadOnly(True)
        self._ring_info.setMaximumHeight(140)
        self._ring_info.setStyleSheet(f"font-family:{S.MONO_CSS};font-size:10px")
        ring.body.addWidget(self._ring_info)
        lv.addWidget(ring)
        lv.addWidget(trans_card)

        # ── Calibration card ──
        calc = S.make_card("Load/save calibration (optional)")
        self._calib_ed = QtWidgets.QLineEdit()
        self._calib_ed.setPlaceholderText("calibration.json / paramstest.txt / .poni…")
        calc.body.addLayout(_frow(self._calib_ed, self._browse_calib))
        self._calib_ed.returnPressed.connect(self._load_calibration)
        self._calib_lbl = QtWidgets.QLabel("No calibration loaded — using manual geometry / BC.")
        self._calib_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._calib_lbl.setWordWrap(True)
        calc.body.addWidget(self._calib_lbl)
        save_row = QtWidgets.QHBoxLayout(); save_row.setSpacing(4)
        self._save_json_btn = QtWidgets.QPushButton("Save JSON")
        self._save_json_btn.setToolTip(
            "Save the current geometry (manual fields, or the loaded calibration's\n"
            "full geometry) as a calibration.json.")
        self._save_json_btn.clicked.connect(lambda: self._save_calibration("json"))
        self._save_params_btn = QtWidgets.QPushButton("Save params (.txt)")
        self._save_params_btn.setToolTip(
            "Save the current geometry as a MIDAS parameter file (paramstest.txt).")
        self._save_params_btn.clicked.connect(lambda: self._save_calibration("paramstest"))
        self._save_poni_btn = QtWidgets.QPushButton("Save PONI")
        self._save_poni_btn.setToolTip(
            "Save the current geometry as a pyFAI .poni file.\n"
            "Note: ty/tz tilts have no PONI equivalent and are not exported.")
        self._save_poni_btn.clicked.connect(lambda: self._save_calibration("poni"))
        save_row.addWidget(self._save_json_btn)
        save_row.addWidget(self._save_params_btn)
        save_row.addWidget(self._save_poni_btn)
        calc.body.addLayout(save_row)
        lv.addWidget(calc)

        # Recompute rings / radial profile when the beam centre is edited manually.
        self._bcy.valueChanged.connect(self._on_bc_changed)
        self._bcz.valueChanged.connect(self._on_bc_changed)
        self._ty.valueChanged.connect(self._on_bc_changed)
        self._tz.valueChanged.connect(self._on_bc_changed)

    # ── Materials list ───────────────────────────────────────────

    @staticmethod
    def _swatch_style(color: str) -> str:
        return f"background-color:{color}; border:1px solid #555; border-radius:2px;"

    def _new_material_defaults(self, name: Optional[str] = None) -> dict:
        if name is None:
            name = f"Material {len(self._materials) + 1}"
        base = MATERIALS.get(name)
        if base is not None and base.get("kind") == "dspacing":
            m = dict(kind="dspacing", d_list=list(base["d_list"]))
            preset = name
        elif base is not None:
            m = dict(a=base["a"], b=base["b"], c=base["c"],
                      alpha=base["alpha"], beta=base["beta"], gamma=base["gamma"],
                      sg=base["sg"])
            preset = name
        else:
            m = dict(a=5.4116, b=5.4116, c=5.4116, alpha=90.0, beta=90.0, gamma=90.0, sg=225)
            preset = "Custom"
        m.update(name=name, preset=preset, enabled=True, cubic=False,
                  color=_MATERIAL_COLORS[len(self._materials) % len(_MATERIAL_COLORS)])
        return m

    def _build_material_row(self, material: dict) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
        chk = QtWidgets.QCheckBox()
        chk.setChecked(material["enabled"])
        chk.setToolTip("Show this material's rings")
        chk.toggled.connect(lambda checked, m=material: self._on_material_enabled(m, checked))
        swatch = QtWidgets.QPushButton()
        swatch.setFixedSize(18, 18)
        swatch.setToolTip("Ring color for this material (image + integration plot)")
        swatch.setStyleSheet(self._swatch_style(material["color"]))
        swatch.clicked.connect(lambda _, m=material, sw=swatch: self._pick_material_color(m, sw))
        name_btn = QtWidgets.QPushButton(material["name"])
        name_btn.setFlat(True)
        name_btn.setCursor(QtCore.Qt.PointingHandCursor)
        name_btn.setStyleSheet(
            "QPushButton{text-align:left; color:#8ecdf7; text-decoration:underline; "
            "border:none; padding:0;}")
        name_btn.setToolTip("Edit this material's lattice, space group, and name")
        name_btn.clicked.connect(lambda _, m=material, nb=name_btn: self._edit_material(m, nb))
        del_btn = QtWidgets.QPushButton("✕")
        del_btn.setFixedSize(20, 20)
        del_btn.setToolTip("Remove this material")
        del_btn.clicked.connect(lambda _, m=material, r=row: self._delete_material(m, r))
        h.addWidget(chk); h.addWidget(swatch); h.addWidget(name_btn, 1); h.addWidget(del_btn)
        row._del_btn = del_btn
        return row

    def _update_material_delete_buttons(self):
        many = len(self._materials) > 1
        for i in range(self._materials_box.count()):
            item = self._materials_box.itemAt(i)
            row = item.widget() if item is not None else None
            if row is not None:
                row._del_btn.setEnabled(many)

    def _add_material(self, name: Optional[str] = None):
        material = self._new_material_defaults(name)
        self._materials.append(material)
        self._materials_box.addWidget(self._build_material_row(material))
        self._update_material_delete_buttons()
        self._on_sim_param_changed()

    def set_materials(self, materials: list):
        """Replace the whole materials list (e.g. from a loaded GUI state)."""
        for i in reversed(range(self._materials_box.count())):
            item = self._materials_box.takeAt(i)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self._materials = []
        for md in materials:
            m = dict(md)
            m.setdefault("preset", "Custom")
            m.setdefault("cubic", False)
            m.setdefault("enabled", True)
            m.setdefault("color", _MATERIAL_COLORS[len(self._materials) % len(_MATERIAL_COLORS)])
            self._materials.append(m)
            self._materials_box.addWidget(self._build_material_row(m))
        if not self._materials:
            self._add_material("Ni (FCC)")
        self._update_material_delete_buttons()

    def _on_material_enabled(self, material: dict, checked: bool):
        material["enabled"] = checked
        self._on_sim_param_changed()

    def _pick_material_color(self, material: dict, swatch_btn: QtWidgets.QPushButton):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(material["color"]), self, "Ring color")
        if not col.isValid():
            return
        material["color"] = col.name()
        swatch_btn.setStyleSheet(self._swatch_style(material["color"]))
        self._redraw_rings()
        self._refresh_profile_markers()

    def _edit_material(self, material: dict, name_btn: QtWidgets.QPushButton):
        dlg = MaterialDialog(material, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            dlg.apply_to(material)
            name_btn.setText(material["name"])
            self._on_sim_param_changed()

    def _delete_material(self, material: dict, row: QtWidgets.QWidget):
        if len(self._materials) <= 1:
            return
        self._materials.remove(material)
        self._materials_box.removeWidget(row)
        row.deleteLater()
        self._update_material_delete_buttons()
        self._on_sim_param_changed()

    def _any_material_rings(self) -> bool:
        return any(m.get("_rings") for m in self._materials)

    def _primary_material_name(self) -> str:
        for m in self._materials:
            if m["enabled"]:
                return m["name"]
        return self._materials[0]["name"] if self._materials else "Custom"

    # ── Ring simulation ───────────────────────────────────────────

    def _on_sim_toggled(self, checked: bool):
        """"Simulate rings" is now a live mode, not a one-shot action."""
        self._sim_btn.setText("Simulate rings (live)" if checked else "Simulate rings")
        if checked:
            self._simulate()

    def _on_sim_param_changed(self, *_):
        """Material/lattice/geometry field edited — resimulate while live mode is on.

        Guarded with ``getattr`` because materials are seeded (via
        ``_add_material``) before ``self._sim_btn`` exists during ``_build_ui``.

        λ/Lsd/px/max2θ feed the radial-integration geometry
        (``_effective_calib_geom``) and any already-simulated rings even
        when live ring simulation is off — without this else-branch, editing
        them silently left the profile/rings stale until something else
        (e.g. a beam-centre edit) happened to refresh them. Mirrors
        ``_on_bc_changed``'s tail."""
        sim_btn = getattr(self, "_sim_btn", None)
        if sim_btn is not None and sim_btn.isChecked() and self._image_provider() is not None:
            self._simulate()
        else:
            if self._any_material_rings():
                self._redraw_rings()
            self._maybe_auto_radial()
            self.geometryChanged.emit()

    def _simulate(self):
        img = self._image_provider()
        if img is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load data first."); return
        lines, errors, any_rings = [], [], False
        for m in self._materials:
            m["_rings"] = []
            if not m["enabled"]:
                continue
            try:
                if m.get("kind") == "dspacing":
                    rings = simulate_rings_from_dspacings(
                        m["d_list"], self._wl.value(), self._lsd_um(),
                        self._px.value(), self._max2t.value())
                else:
                    lattice = dict(a=m["a"], b=m["b"], c=m["c"],
                                   alpha=m["alpha"], beta=m["beta"], gamma=m["gamma"])
                    rings = simulate_rings(lattice, m["sg"], self._wl.value(),
                                           self._lsd_um(), self._px.value(), self._max2t.value())
            except Exception:
                import traceback
                errors.append(f"{m['name']}: {traceback.format_exc().splitlines()[-1]}")
                continue
            m["_rings"] = rings
            any_rings = True
            lines.append(f"{m['name']}: {len(rings)} rings")
            lines.append(f"{'hkl':>10}  {'2θ(°)':>7}  {'d(Å)':>7}  {'R(px)':>8}")
            for r in rings:
                label = f"n{r['order']}" if r["hkl"] is None else str(tuple(r["hkl"]))
                lines.append(f"{label:>10}  {r['two_theta_deg']:7.3f}  "
                             f"{r['d_spacing']:7.4f}  {r['radius_px']:8.1f}")
            lines.append("")
        self._after_geometry_change()
        if errors:
            lines.append("Errors:"); lines.extend(errors)
        if not any_rings and not errors:
            lines = ["No enabled materials."]
        self._ring_info.setPlainText("\n".join(lines).rstrip())
        if errors and not any_rings:
            show_error(self, "Simulation error", "\n".join(errors))

    def _clear_rings(self):
        if self._viewer is None:
            self._ring_items.clear(); self._label_items.clear()
            return
        for it in self._ring_items + self._label_items:
            self._viewer._iv.removeItem(it)
        self._ring_items.clear(); self._label_items.clear()

    def _redraw_rings(self):
        self._clear_rings()
        img = self._image_provider()
        if self._viewer is None or img is None or not self._any_material_rings():
            return
        bc_y, bc_z = self._bcy.value(), self._bcz.value()
        ty, tz = self._ty.value(), self._tz.value()
        tilted = abs(ty) > 1e-9 or abs(tz) > 1e-9
        px = self._px.value()
        th = np.linspace(0, 2 * math.pi, 400)
        vis_r = self._show_rings.isChecked()
        vis_l = self._show_labels.isChecked() and vis_r
        for m in self._materials:
            rings = m.get("_rings")
            if not m["enabled"] or not rings:
                continue
            pen = pg.mkPen(m["color"], width=self._ring_width.value(), style=QtCore.Qt.DotLine)
            for r in rings:
                rad = r["radius_px"]
                if not (rad > 0 and math.isfinite(rad)):
                    continue
                if tilted:
                    ys, zs = tilted_ring_xy(r["two_theta_deg"], 0.0, ty, tz,
                                             self._lsd_um(), bc_y, bc_z, px, px)
                    label_y, label_z = ys[len(ys) // 2], zs[len(zs) // 2]
                else:
                    ys = bc_y + rad * np.cos(th); zs = bc_z + rad * np.sin(th)
                    label_y, label_z = bc_y, bc_z - rad
                item = pg.PlotDataItem(ys, zs, pen=pen)
                item.setVisible(vis_r)
                self._viewer._iv.addItem(item); self._ring_items.append(item)
                label = f"n{r['order']}" if r["hkl"] is None else "".join(str(x) for x in r["hkl"])
                txt = pg.TextItem(label, color=m["color"], anchor=(0.5, 1.0))
                txt.setPos(label_y, label_z)
                txt.setVisible(vis_l)
                self._viewer._iv.addItem(txt); self._label_items.append(txt)
        # beam-centre marker
        bc = pg.ScatterPlotItem([bc_y], [bc_z], symbol="+", size=16,
                                pen=pg.mkPen("#00cfff", width=2), brush=pg.mkBrush(0, 0, 0, 0))
        bc.setVisible(vis_r)
        self._viewer._iv.addItem(bc); self._ring_items.append(bc)

    def _set_rings_visible(self, *_):
        vis_r = self._show_rings.isChecked()
        vis_l = self._show_labels.isChecked() and vis_r
        for it in self._ring_items:
            it.setVisible(vis_r)
        for it in self._label_items:
            it.setVisible(vis_l)

    # ── Beam-centre picking / radial integration ──────────────────

    def _on_bc_picked(self, bc_y, bc_z):
        """Single-click BC pick from the image (PickableImageViewer)."""
        self._bc_auto.setChecked(False)
        self._bcy.setValue(bc_y); self._bcz.setValue(bc_z)   # triggers _on_bc_changed

    def _on_ring_fit_bc(self, bc_y, bc_z, r_px):
        """BC from a 3+ point circle fit on a ring (PickableImageViewer)."""
        self._bc_auto.setChecked(False)
        self._bcy.setValue(bc_y); self._bcz.setValue(bc_z)   # triggers _on_bc_changed

    def _on_bc_changed(self, *_):
        """Beam centre edited (manually or by a pick) — refresh overlays/plot."""
        if self._any_material_rings():
            self._redraw_rings()
        self._redraw_picked_ring()
        self._maybe_auto_radial()
        self.geometryChanged.emit()

    def on_radius_clicked(self, r_px: float):
        """A radius was clicked on the profile — draw its ring on the image.
        Returns the message to show on the caller's info label."""
        self._picked_r = float(r_px)
        self._redraw_picked_ring()
        return f"Picked radius: {r_px:.1f} px  (magenta ring)"

    def _redraw_picked_ring(self):
        """(Re)draw the click-picked ring (magenta) about the current beam centre."""
        if self._pick_ring_item is not None and self._viewer is not None:
            self._viewer._iv.removeItem(self._pick_ring_item)
            self._pick_ring_item = None
        r = self._picked_r
        img = self._image_provider()
        if r is None or img is None or self._viewer is None:
            return
        bc_y, bc_z = self._bcy.value(), self._bcz.value()
        th = np.linspace(0, 2 * math.pi, 512)
        self._pick_ring_item = pg.PlotDataItem(
            bc_y + r * np.cos(th), bc_z + r * np.sin(th),
            pen=pg.mkPen("#ff30ff", width=1.8))
        self._viewer._iv.addItem(self._pick_ring_item)

    def _on_rad_param_changed(self, *_):
        self._maybe_auto_radial()

    def _refresh_profile_markers(self):
        if self._profile_view is None:
            return
        groups = [{"radii": [r["radius_px"] for r in m["_rings"]], "color": m["color"]}
                  for m in self._materials if m["enabled"] and m.get("_rings")]
        self._profile_view.set_ring_markers(
            groups, self._lsd_um(), self._px.value(), self._wl.value())

    def _effective_calib_geom(self, img: np.ndarray) -> Optional[dict]:
        """Geometry used for radial integration: the loaded calibration's full
        geometry if present, otherwise one synthesized from the Ring-simulation
        widgets when a tilt is set — so the profile stays tilt-consistent with
        the on-image ring overlay even without a loaded calibration file.

        When a calibration is loaded, `self._calib_geom` is a snapshot frozen
        at load time (see `_apply_full_geometry_dict`) — it must NOT be
        returned verbatim, or a later edit to the live wavelength/Lsd/BC/tilt
        widgets (typed, picked, or shared-field-synced) would move the rings
        (which always read the live widgets) without moving the radial
        profile. `tx`/`distortion`/`NrPixelsY`/`NrPixelsZ` have no live
        widget equivalent, so those alone come from the frozen snapshot."""
        if self._calib_geom is not None:
            geom = dict(self._calib_geom)
            px = self._px.value()
            geom.update({
                "wavelength_A": self._wl.value(), "Lsd": self._lsd_um(),
                "BC_y": self._bcy.value(), "BC_z": self._bcz.value(),
                "ty": self._ty.value(), "tz": self._tz.value(),
                "pxY": px, "pxZ": px,
            })
            return geom
        ty, tz = self._ty.value(), self._tz.value()
        if abs(ty) < 1e-9 and abs(tz) < 1e-9:
            return None
        if img is None:
            return None
        nz, ny = img.shape
        px = self._px.value()
        return {
            "wavelength_A": self._wl.value(), "Lsd": self._lsd_um(),
            "BC_y": self._bcy.value(), "BC_z": self._bcz.value(),
            "tx": 0.0, "ty": ty, "tz": tz,
            "pxY": px, "pxZ": px,
            "NrPixelsY": ny, "NrPixelsZ": nz, "distortion": {},
            "im_trans": list(self.im_trans_codes()),
        }

    def show_radial_help(self):
        """Explain how the radial-integration plot's profile is computed."""
        QtWidgets.QMessageBox.information(
            self, "Radial integration — how it's calculated",
            "The plot shows intensity vs. radius: the azimuthal (angular) average "
            "of the image about the beam centre, grouped into rings of width "
            "\"R bin\".\n\n"
            "• Calibration loaded, or a tilt (ty/tz) set on the Ring-simulation "
            "card: the full MIDAS geometry engine is used. Pixels are binned into "
            "(η, R) cells honouring detector tilt and distortion, and each R-bin's "
            "value is a pixel-count-weighted mean across η — "
            "Σ(cell_mean·count) / Σ(count) — robust to partial or uneven azimuthal "
            "coverage.\n\n"
            "• Otherwise: a fast circle-binning fallback is used. Pixels are "
            "grouped purely by distance from the beam centre (BC_y, BC_z) into "
            "R-bins, and each bin's value is Σintensity / Σpixels — a plain "
            "per-bin mean, with no tilt correction.\n\n"
            "If full-geometry integration fails, the plot automatically falls "
            "back to circle binning and a warning is shown above the "
            "calibration card.")

    def radial_integrate(self):
        """Azimuthal average of the current frame.

        With a loaded calibration file, or a tilt dialled into the Ring-simulation
        card, the full geometry (tilts + distortion) is used via the MIDAS
        integration engine; otherwise a fast circle-binning about the beam centre
        is used."""
        img = self._image_provider()
        if img is None or self._rad_r_bin is None or self._profile_view is None:
            return
        mask = self._mask_provider(img) if self._mask_provider is not None else None
        geom = self._effective_calib_geom(img)
        if geom is not None:
            try:
                r_axis, prof = self._midas_radial(img, geom, mask)
            except Exception:
                import traceback
                self._calib_lbl.setText(
                    "Full-geometry integration failed — using circle binning. "
                    "See error log.")
                self._log_error(traceback.format_exc())
                r_axis, prof = self._radial_profile(
                    img, self._bcy.value(), self._bcz.value(),
                    self._rad_r_bin.value(), mask=mask)
        else:
            r_axis, prof = self._radial_profile(
                img, self._bcy.value(), self._bcz.value(), self._rad_r_bin.value(),
                mask=mask)
        self._profile_view.set_profile(
            r_axis, prof, wavelength_A=self._wl.value(),
            lsd_um=self._lsd_um(), px_um=self._px.value())
        self._refresh_profile_markers()

    def _midas_radial(self, img, g, mask):
        """Radial profile via the MIDAS engine, honouring the given geometry's
        tilts + distortion (not just concentric circles). ``g`` is either the
        loaded calibration's geometry or one synthesized from the live Ring-sim
        widgets (see ``_effective_calib_geom``). The binning geometry is built
        once per (geometry, R-bin, image shape, mask) and reused across frames;
        only the per-frame integration runs on a frame change.

        ``img`` is the display-oriented frame (``image_provider`` already
        applied this card's Transforms checkboxes, for on-screen viewing) —
        it is un-transformed back to raw here so the backend can apply
        ``g["im_trans"]`` itself via ``spec.TransOpt`` exactly once, the same
        as every other integration call site."""
        import json
        import torch
        r_bin = max(float(self._rad_r_bin.value()), 0.1)
        eta_bin = 5.0
        im_trans = tuple(g.get("im_trans") or ())
        if im_trans:
            img = _apply_im_trans(img, tuple(reversed(im_trans)))   # display → raw
        nz, ny = img.shape
        mask = None if mask is None else np.ascontiguousarray(mask, dtype=bool)
        mask_fp = None if mask is None else hash(mask.tobytes())
        sig = (round(float(g["Lsd"]), 3), round(float(g["BC_y"]), 3),
               round(float(g["BC_z"]), 3), round(float(g.get("tx") or 0.0), 4),
               round(float(g.get("ty") or 0.0), 4), round(float(g.get("tz") or 0.0), 4),
               round(float(g["pxY"]), 4), round(float(g.get("pxZ") or g["pxY"]), 4),
               round(float(g["wavelength_A"]), 6), g.get("NrPixelsY"), g.get("NrPixelsZ"),
               round(r_bin, 4), (nz, ny), mask_fp, im_trans,
               json.dumps(g.get("distortion") or {}, sort_keys=True))
        if self._calib_ctx is None or self._calib_ctx_sig != sig:
            spec = _spec_from_result_ns(
                r_bin, eta_bin, NrPixelsY=ny, NrPixelsZ=nz,
                pxY=g["pxY"], pxZ=g.get("pxZ") or g["pxY"], Lsd=g["Lsd"],
                BC_y=g["BC_y"], BC_z=g["BC_z"], tx=g.get("tx") or 0.0,
                ty=g.get("ty") or 0.0, tz=g.get("tz") or 0.0,
                wavelength_A=g["wavelength_A"], distortion=g.get("distortion") or {},
                im_trans=im_trans)
            ctx = build_integration_context(spec, "hard", mask, (None, None), weighted=True)
            self._calib_ctx = (spec, ctx); self._calib_ctx_sig = sig
        spec, ctx = self._calib_ctx
        img_t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float64))
        prof, _, cake_2d = integrate_frame(
            img_t, spec, ctx["geom"], "hard", (None, None), None, False,
            corr_counts=ctx["corr_counts"], return_cake=True,
            weighted=True, cnt_cake=ctx["cnt"])
        if self._cake_view is not None:
            n_eta = spec.n_eta_bins
            eta_ax = float(spec.EtaMin) + float(spec.EtaBinSize) * (np.arange(n_eta) + 0.5)
            self._cake_view.set_cake(cake_2d, ctx["r_ax"], eta_ax)
        return ctx["r_ax"], prof

    def _log_error(self, text):
        """Append a traceback to the crash log (no LogPanel on this tab)."""
        try:
            from midas_gui.app import _log
            _log(text)
        except Exception:
            pass

    def _radial_grid(self, shape, bc_y, bc_z, r_bin):
        """Cached per-pixel radial bin index + axis, keyed on (shape, BC, r_bin).

        The pixel→radius grid only changes when the shape / beam centre / bin size
        change — not frame-to-frame — so scrubbing frames reuses it (one bincount
        instead of rebuilding indices+hypot each tick)."""
        r_bin = max(float(r_bin), 1e-6)
        key = (tuple(shape), round(float(bc_y), 4), round(float(bc_z), 4), round(r_bin, 6))
        cache = self._rad_grid_cache
        if cache is not None and cache[0] == key:
            return cache[1], cache[2], cache[3]
        NZ, NY = shape
        zz, yy = np.indices((NZ, NY))
        r = np.hypot(yy - bc_y, zz - bc_z)
        nbins = max(1, int(r.max() / r_bin) + 1)
        which = np.minimum((r / r_bin).astype(np.int64), nbins - 1).ravel()
        r_axis = (np.arange(nbins) + 0.5) * r_bin
        self._rad_grid_cache = (key, which, nbins, r_axis)
        return which, nbins, r_axis

    def _radial_profile(self, img: np.ndarray, bc_y: float, bc_z: float,
                        r_bin: float = 1.0, mask: Optional[np.ndarray] = None):
        """Mean intensity vs radius (px) about (bc_y, bc_z), using the cached grid.

        bc_y is the column (Y/x) and bc_z the row (Z/y); image shape is (NZ, NY).
        ``mask`` (bool, True = exclude) drops pixels; non-finite pixels are ignored.
        Returns (r_axis_px, profile), NaN in empty bins.
        """
        which, nbins, r_axis = self._radial_grid(img.shape, bc_y, bc_z, r_bin)
        vals = img.ravel()
        good = np.isfinite(vals)
        if mask is not None:
            good &= ~mask.ravel()
        sums = np.bincount(which[good], weights=vals[good], minlength=nbins)
        counts = np.bincount(which[good], minlength=nbins)
        prof = np.full(nbins, np.nan, dtype=np.float64)
        nz = counts > 0
        prof[nz] = sums[nz] / counts[nz]
        return r_axis, prof

    # ── Calibration file ────────────────────────────────────────────

    def _browse_calib(self):
        p = _browse(self, "Open calibration file",
                    "Calibration (*.json *.poni *.txt);;All (*)")
        if p:
            self._calib_ed.setText(p)
            self._load_calibration()

    def _load_calibration(self):
        """Read geometry (BC, Lsd, pixel size, wavelength) from a MIDAS paramstest,
        pyFAI .poni, or calibration.json file and apply it to the ring overlay and
        the radial integration."""
        path = self._calib_ed.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "No file", "Select a calibration file first.")
            return
        try:
            geo = read_geometry(path)
            wl, lsd, px = geo["wavelength_A"], geo["Lsd_um"], geo["px_um"]
            bcy, bcz = geo["BC_y"], geo["BC_z"]
            if all(v is None for k, v in geo.items() if k != "im_trans"):
                self._calib_lbl.setText("No recognised geometry in file.")
                return
            self._bc_auto.setChecked(False)   # geometry now comes from the file
            lsd_mm = (lsd / 1000.0) if lsd is not None else None
            for w, v in ((self._wl, wl), (self._lsd, lsd_mm), (self._px, px),
                         (self._bcy, bcy), (self._bcz, bcz)):
                if v is not None:
                    w.blockSignals(True); w.setValue(float(v)); w.blockSignals(False)
            im_trans = geo.get("im_trans") or []
            self._flip_y.setChecked(1 in im_trans)
            self._flip_z.setChecked(2 in im_trans)
            self._transp.setChecked(3 in im_trans)
            parts = []
            if lsd is not None: parts.append(f"Lsd={float(lsd)/1000:.2f} mm")
            if bcy is not None and bcz is not None:
                parts.append(f"BC=({float(bcy):.1f}, {float(bcz):.1f})")
            if wl is not None: parts.append(f"λ={float(wl):.5g} Å")
            if px is not None: parts.append(f"px={float(px):.4g} µm")
            self._calib_ctx = self._calib_ctx_sig = None
            try:
                self._calib_geom = geometry_fields_from_file(path)
                d = self._calib_geom
                for w, key in ((self._ty, "ty"), (self._tz, "tz")):
                    v = d.get(key)
                    if v is not None:
                        w.blockSignals(True); w.setValue(float(v)); w.blockSignals(False)
                tilt = any(abs(float(d.get(k) or 0.0)) > 1e-9 for k in ("tx", "ty", "tz"))
                mode = ("full integration: tilts"
                        + ("+distortion" if d.get("distortion") else "")
                        if (tilt or d.get("distortion"))
                        else "full integration (geometry-correct)")
            except Exception:
                self._calib_geom = None
                mode = "scalar geometry (circle binning)"
            self._calib_lbl.setText(
                f"Loaded {Path(path).suffix or 'file'} — " + "  ".join(parts)
                + f"  ·  {mode}")
            self._after_geometry_change()
        except Exception:
            import traceback
            show_error(self, "Calibration load error", traceback.format_exc())

    def get_full_geometry(self) -> Optional[dict]:
        """Public alias for ``_export_geom`` — the best-available full
        geometry (NrPixelsY/Z, pxY/Z, Lsd, BC, tx/ty/tz, distortion,
        im_trans), in the same dict shape ``helpers.geometry_fields_from_file``
        returns. Used by an owner (e.g. the Hydra page) to sync this card's
        geometry into a ``hydra.DetectorState``."""
        return self._export_geom()

    def _export_geom(self) -> Optional[dict]:
        """Full geometry dict for calibration export: the loaded calibration's
        geometry if present (carries tilts/distortion/detector size from the
        file), otherwise one built from the current manual / Ring-simulation
        fields (tx=0, distortion empty)."""
        if self._calib_geom is not None:
            geom = dict(self._calib_geom)
            geom["im_trans"] = self.im_trans_codes()   # current checkboxes win
            return geom
        img = self._image_provider()
        if img is None:
            return None
        nz, ny = img.shape
        px = self._px.value()
        return {
            "wavelength_A": self._wl.value(), "Lsd": self._lsd_um(),
            "BC_y": self._bcy.value(), "BC_z": self._bcz.value(),
            "tx": 0.0, "ty": self._ty.value(), "tz": self._tz.value(),
            "pxY": px, "pxZ": px, "NrPixelsY": ny, "NrPixelsZ": nz,
            "distortion": {}, "im_trans": self.im_trans_codes(),
        }

    def _save_calibration(self, kind: str):
        """Save the current geometry (see ``_export_geom``) as a calibration
        file — JSON (GUI bare-key format), MIDAS paramstest.txt, or pyFAI .poni."""
        geom = self._export_geom()
        if geom is None:
            QtWidgets.QMessageBox.warning(self, "No geometry", "Load data first.")
            return
        specs = {
            "json": ("Save calibration.json", "calibration.json", "JSON (*.json)"),
            "paramstest": ("Save MIDAS parameter file", "paramstest.txt", "Text (*.txt)"),
            "poni": ("Save calibration.poni", "calibration.poni", "PONI (*.poni)"),
        }
        title, default_name, filt = specs[kind]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, default_name, filt)
        if not path:
            return
        try:
            if kind == "json":
                import json
                Path(path).write_text(json.dumps(geom, indent=2, default=str))
            elif kind == "paramstest":
                from types import SimpleNamespace
                ns = SimpleNamespace(
                    NrPixelsY=int(geom["NrPixelsY"]), NrPixelsZ=int(geom["NrPixelsZ"]),
                    pxY=float(geom["pxY"]), pxZ=float(geom.get("pxZ") or geom["pxY"]),
                    Lsd=float(geom["Lsd"]), BC_y=float(geom["BC_y"]), BC_z=float(geom["BC_z"]),
                    tx=float(geom.get("tx") or 0.0), ty=float(geom.get("ty") or 0.0),
                    tz=float(geom.get("tz") or 0.0), wavelength_A=float(geom["wavelength_A"]),
                    distortion=geom.get("distortion") or {},
                    im_trans=list(geom.get("im_trans") or []),
                    _calibrant_name=self._primary_material_name())
                write_standalone_paramstest(ns, path)
            elif kind == "poni":
                write_poni(geom, path)
        except Exception:
            import traceback
            show_error(self, "Save failed", traceback.format_exc())
            return
        QtWidgets.QMessageBox.information(self, "Saved", f"Calibration saved to:\n{path}")
