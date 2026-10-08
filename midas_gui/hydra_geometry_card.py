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
                         parse_dspacing_text, ring_on_image_mask as _ring_on_image_mask,
                         browse_start_dir)
from midas_gui.workers import build_integration_context, integrate_frame
from midas_gui import style as S

# Default ring colors assigned to new materials, cycled by row count. First
# entry matches the single hardcoded ring color the old single-material UI used.
_MATERIAL_COLORS = ("#f0c060", "#4fc3f7", "#ab47bc", "#66bb6a", "#ef5350",
                     "#ffca28", "#26a69a", "#ec407a", "#7e57c2", "#8d6e63")

# η bin size (deg) used for the (η, R) cake, by both the full-geometry engine
# path and the circle-binning fallback, so the two are directly comparable.
CAKE_ETA_BIN_DEG = 5.0

# Integration kernels (the same vocabulary Batch Integrate's kernel selector
# uses, see constants.KERNELS). "hard" drops each pixel wholly into one (η, R)
# cell — cheap enough to keep up with a live stream; "subpixel2" splits every
# pixel over a 2x2 subgrid, which is what Batch Integrate defaults to and what
# the Data Viewer's "Accurate" tick and the cake both use.
FAST_KERNEL = "hard"
ACCURATE_KERNEL = "subpixel2"

#: Max engine binning contexts kept alive at once (see ``_midas_radial``).
_CTX_CACHE_MAX = 6


_CUSTOM_DSPACING = "Custom (d-spacings)"

# Lattice defaults used to seed the (hidden) lattice widgets when a material
# has no a/b/c/.../sg of its own yet — a "dspacing"-kind material (e.g.
# AgBH) or a brand-new dialog opened straight into d-spacing mode.
_FALLBACK_LATTICE = dict(a=5.4116, b=5.4116, c=5.4116, alpha=90.0, beta=90.0, gamma=90.0, sg=225)


def _ring_label_pos(ys, zs, img_shape, box_w: float = 0.0, box_h: float = 0.0,
                    y_up: bool = True):
    """Where to anchor a ring's ``hkl``/order label, and with which
    ``pg.TextItem`` anchor, given the ring's plotted points, the image's
    ``(rows, cols)`` shape, and the label box's size in *data* units.

    Returns ``(y, z, (anchor_x, anchor_y))``, or ``None`` when no part of the
    ring crosses the image — a label out in the empty space beside the
    detector describes nothing the user can see, so it is not drawn at all
    (neither is the ring; see ``_redraw_rings``).

    The anchor point is the highest on-image point of the arc: clear of the
    data below it, and stable as the geometry is nudged. The obvious choice —
    the ring's twelve o'clock point — is wrong whenever the beam centre is
    near an edge: on a wide, short SAXS strip with the centre at the left,
    every ring's top lies hundreds of pixels below the frame and all the
    labels pile up off-screen. Anchoring on the visible arc means the beam
    centre's position relative to the frame decides where each label lands,
    which is what makes the labels follow the rings the user can actually see.

    The *anchor* then keeps the box itself on the image. A text box is drawn
    at a fixed screen size, so ``box_w``/``box_h`` are the caller's font
    metrics converted to data units: the box grows away from the arc when
    there is room above it and back over the arc when there is not, and slides
    its horizontal anchor to the box edge near a left/right border. ``y_up``
    says whether increasing ``z`` renders upward (false when the display
    origin is top-left, which inverts the view) — anchors are resolved in
    screen space, so that flip has to be accounted for here.
    """
    nz, ny = img_shape[:2]
    ys = np.asarray(ys, dtype=float); zs = np.asarray(zs, dtype=float)
    on = _ring_on_image_mask(ys, zs, img_shape) & np.isfinite(ys) & np.isfinite(zs)
    if not on.any():
        return None
    idx = np.flatnonzero(on)
    top = idx[np.argmax(zs[idx])]
    ly, lz = float(ys[top]), float(zs[top])

    # Grow the box off the top of the arc when it fits between there and the
    # top edge; otherwise fold it back down over the arc, which is inside the
    # image by construction.
    grow_up_in_data = (lz + box_h) <= (nz - 1)
    anchor_y = 1.0 if (grow_up_in_data == bool(y_up)) else 0.0
    if box_w > 0 and ly - 0.5 * box_w < 0:
        anchor_x = 0.0                      # box grows right, off the left edge
    elif box_w > 0 and ly + 0.5 * box_w > ny - 1:
        anchor_x = 1.0                      # box grows left, off the right edge
    else:
        anchor_x = 0.5
    return ly, lz, (anchor_x, anchor_y)


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
        # 5 decimals on the cell edges, not 3. Several presets are published to
        # that precision (LaB6 4.15692, Si 5.43102, W 3.16525) and a 3-decimal
        # spinbox does not merely display them short — setValue() rounds, so
        # opening this dialog and pressing OK wrote the rounded value back.
        # 5 covers every entry in MATERIALS exactly; angles get the matching
        # widening even though the presets are all whole degrees.
        _LW, _AW = 104, 92    # compact lattice / angle-SG cell widths
        self._a = _fspin(0.1, 100.0, 5, latt0["a"]); self._a.setFixedWidth(_LW)
        self._b = _fspin(0.1, 100.0, 5, latt0["b"]); self._b.setFixedWidth(_LW)
        self._c = _fspin(0.1, 100.0, 5, latt0["c"]); self._c.setFixedWidth(_LW)
        self._al = _fspin(1.0, 179.0, 4, latt0["alpha"]); self._al.setFixedWidth(_AW)
        self._be = _fspin(1.0, 179.0, 4, latt0["beta"]); self._be.setFixedWidth(_AW)
        self._ga = _fspin(1.0, 179.0, 4, latt0["gamma"]); self._ga.setFixedWidth(_AW)
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
        # 6 decimals, not 4, for the same reason as the cell edges above: the
        # text here is re-parsed on OK, so the displayed precision *is* the
        # stored precision. A harmonic series divides into repeating decimals
        # (AgBH's 58.380/9 = 6.486666…), which 4 decimals rounded away at
        # 3e-5 A per round-trip.
        return " ".join(f"{d:.6f}" for d in d_list)

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

    pushGeometry = QtCore.pyqtSignal(dict)    # "Send →" (geometry → Calibrate) clicked
    pullGeometry = QtCore.pyqtSignal()        # "← Get" (geometry ← Calibrate) clicked
    imTransChanged = QtCore.pyqtSignal()      # a Flip Y/Flip Z/Transpose checkbox toggled
    geometryChanged = QtCore.pyqtSignal()     # BC/tilt/calibration changed (edit, pick, or file load)

    def __init__(self, parent=None, *, show_rotate: bool = False):
        super().__init__(parent)
        self._show_rotate = show_rotate
        self._materials: list = []
        #: What a calibration load wrote into the geometry boxes; see
        #: _snapshot_calib_baseline.
        self._calib_widget_baseline: Optional[dict] = None
        self._ring_items: list = []
        self._label_items: list = []
        self._pick_ring_item = None
        self._picked_r: Optional[float] = None
        self._calib_geom: Optional[dict] = None
        #: Engine binning contexts, keyed by their (geometry, kernel, bins,
        #: shape, mask) signature — the radial profile and the cake may run
        #: with different bin sizes/kernels, so one slot is not enough.
        self._calib_ctx_cache: dict = {}
        self._rad_grid_cache = None
        self._eta_grid_cache = None
        #: Live ring simulation armed (the "live" tick *and* a Simulate click).
        self._sim_live_on = False
        #: Geometry the currently-drawn rings were simulated with (frozen, so
        #: a one-shot simulation does not follow later parameter edits).
        self._ring_draw_geom: Optional[dict] = None

        self._viewer = None
        self._profile_view = None
        self._cake_view = None
        self._image_provider: Callable[[], Optional[np.ndarray]] = lambda: None
        #: ``fn(suffix) -> full path`` for the Save dialogs; see
        #: set_save_name_provider.
        self._save_name_provider: Optional[Callable[[str], Optional[str]]] = None
        self._mask_provider: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None
        self._rad_r_bin: Optional[QtWidgets.QDoubleSpinBox] = None
        self._rad_auto: Optional[QtWidgets.QCheckBox] = None
        self._rad_accurate: Optional[QtWidgets.QCheckBox] = None
        self._cake_r_bin: Optional[QtWidgets.QDoubleSpinBox] = None
        self._cake_eta_bin: Optional[QtWidgets.QDoubleSpinBox] = None

        self._build_ui()

    # ── Wiring ───────────────────────────────────────────────────

    def set_image_source(self, image_provider: Callable[[], Optional[np.ndarray]],
                          mask_provider: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None):
        self._image_provider = image_provider
        self._mask_provider = mask_provider

    def set_save_name_provider(self, fn: Optional[Callable[[str], Optional[str]]]):
        """Supply ``fn(suffix) -> full default path`` for the Save dialogs.

        Without it the dialogs open on bare names (``paramstest.txt``) in
        whatever directory the app was launched from, so every save is a
        navigate-and-type. The Calibrate tab has named its files
        ``<expid>_<data stem><suffix>`` beside the data for a while
        (``CalibrationTab._default_save_path``); this lets the Data Viewer and
        the Hydra panels do the same instead of each inventing one.

        Optional, and failures fall back to the bare name, so a card with no
        provider behaves exactly as before.
        """
        self._save_name_provider = fn

    def set_profile_view(self, profile_view):
        self._profile_view = profile_view

    def set_cake_view(self, cake_view):
        self._cake_view = cake_view

    def set_radial_controls(self, r_bin_spin: QtWidgets.QDoubleSpinBox,
                             auto_checkbox: QtWidgets.QCheckBox,
                             accurate_checkbox: Optional[QtWidgets.QCheckBox] = None):
        """Bind the profile toolbar's R-bin / Auto controls, and optionally the
        "Accurate" tick that switches the profile onto the full Batch-Integrate
        pipeline (see ``radial_integrate``). Hydra passes no accurate tick —
        its 4 panels integrate on every frame, so the fast path is the only
        sensible one there."""
        self._rad_r_bin = r_bin_spin
        self._rad_auto = auto_checkbox
        self._rad_accurate = accurate_checkbox
        r_bin_spin.valueChanged.connect(self._on_rad_param_changed)
        if accurate_checkbox is not None:
            accurate_checkbox.toggled.connect(self._on_rad_param_changed)

    def set_cake_controls(self, r_bin_spin: QtWidgets.QDoubleSpinBox,
                           eta_bin_spin: QtWidgets.QDoubleSpinBox):
        """Bind the cake's own R-bin / η-bin spin boxes (shown above the cake
        plot). Once bound, ``radial_integrate`` stops writing the cake as a
        by-product — the two now use different binning, so only the cake's own
        Calculate may fill it."""
        self._cake_r_bin = r_bin_spin
        self._cake_eta_bin = eta_bin_spin

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
            self._sync_dspacing_picking()
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

    def has_calibration(self) -> bool:
        """Whether a calibration file is currently loaded (as opposed to only
        the manual Ring-simulation widgets holding whatever values they were
        left at) — see ``set_calib_path``/``_load_calibration``."""
        return self._calib_geom is not None

    def get_geometry(self) -> dict:
        """Current geometry — λ (Å), pixel (µm), Lsd (µm), beam centre (px).

        Lsd is entered in mm (display) but always returned/used in µm.

        ``tx`` and the distortion coefficients have no field on this card, so
        they come from the loaded calibration file if there is one. They used
        to be dropped here, which meant "Send →" quietly delivered a geometry
        missing the panel's installation azimuth: load a Hydra paramstest at
        tx=300 with 13 refined coefficients, press Send, and the Calibrate
        tab received neither. The values were on screen in the summary line
        the whole time, which is what made it read as Send having done
        nothing.

        A key is included only when it is actually known. The receiver
        (``tab_calibrate.apply_geometry``) treats a present key as a value to
        seed and tick, so sending a fabricated 0 would pin the fit to the
        wrong azimuth -- the same failure that erased tx on result feedback.
        Absent means unknown, and the auto-seeder handles it.
        """
        g = {
            "wavelength_A": self._wl.value(),
            "pxY": self._px.value(),
            "Lsd": self._lsd_um(),
            "BC_y": self._bcy.value(),
            "BC_z": self._bcz.value(),
            "ty": self._ty.value(),
            "tz": self._tz.value(),
            "im_trans": self.im_trans_codes(),
        }
        g["tx"] = self._tx.value()
        cal = self._calib_geom or {}
        if cal.get("distortion"):
            g["distortion"] = dict(cal["distortion"])
        return g

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
                              (self._bcz, "BC_z", 1.0), (self._tx, "tx", 1.0),
                              (self._ty, "ty", 1.0), (self._tz, "tz", 1.0)):
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
            self._snapshot_calib_baseline()
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
        self._calib_ctx_cache.clear()
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

        # tx is the panel's installation azimuth ABOUT THE BEAM, not a small
        # alignment tilt like ty/tz -- hence the full circle, and hence it
        # does not bend the rings at all: rotating about the beam maps a ring
        # (a circle centred on the beam) onto itself. What it does set is
        # where the panel sits in the lab frame, so the image is drawn
        # rotated by it and the eta readout agrees with the integration.
        # Without it the display was a detector-plane view labelled with lab
        # axes, and every eta was out by exactly tx.
        self._tx = _fspin(-360.0, 360.0, 2, 0.0, "°", step=DEFAULT_STEP_TILT)
        self._tx.setToolTip(
            "Installation azimuth of the panel about the beam (0-360°). "
            "Rotates the displayed image into the lab frame; the ring "
            "radii are unchanged by it.")
        self._ty = _fspin(-180.0, 180.0, 2, 0.0, "°", step=DEFAULT_STEP_TILT)
        self._tz = _fspin(-180.0, 180.0, 2, 0.0, "°", step=DEFAULT_STEP_TILT)
        self._ty.setToolTip("Detector tilt about the Y axis — bends the simulated rings.")
        self._tz.setToolTip("Detector tilt about the Z axis — bends the simulated rings.")
        ring.body.addLayout(S.Form().row(("tx:", self._tx)))
        ring.body.addLayout(S.Form().row(("ty:", self._ty), ("tz:", self._tz)))

        # Two-way geometry hand-off with the Calibrate tab.
        calib_row = QtWidgets.QHBoxLayout(); calib_row.setSpacing(4)
        calib_row.addWidget(S.LabelRight("Geometry:"))
        self._to_calib_btn = QtWidgets.QPushButton("Send →")
        self._to_calib_btn.setToolTip(
            "Copy λ, pixel size, Lsd and beam centre from here into the Calibrate "
            "tab's detector + seed fields.")
        self._to_calib_btn.clicked.connect(
            lambda: self.pushGeometry.emit(self.get_geometry()))
        self._from_calib_btn = QtWidgets.QPushButton("← Get")
        self._from_calib_btn.setToolTip(
            "Pull the Calibrate tab's latest calibrated geometry (λ, pixel size, "
            "Lsd, beam centre, tilts and distortion) into these fields.")
        self._from_calib_btn.clicked.connect(self.pullGeometry.emit)
        calib_row.addWidget(self._to_calib_btn, 1)
        calib_row.addWidget(self._from_calib_btn, 1)
        ring.body.addLayout(calib_row)

        ctl = QtWidgets.QHBoxLayout()
        self._show_rings = QtWidgets.QCheckBox("Rings"); self._show_rings.setChecked(True)
        self._show_rings.toggled.connect(self._set_rings_visible)
        self._show_labels = QtWidgets.QCheckBox("Labels"); self._show_labels.setChecked(True)
        self._show_labels.toggled.connect(self._set_rings_visible)
        self._ring_width = _fspin(0.5, 10.0, 1, DEFAULT_RING_WIDTH, "px")
        self._ring_width.setToolTip("Line thickness of the simulated rings on the image.")
        self._ring_width.setMaximumWidth(80)
        self._ring_width.valueChanged.connect(self._redraw_rings)
        self._ring_width.valueChanged.connect(self._refresh_profile_markers)
        ctl.addWidget(self._show_rings); ctl.addWidget(self._show_labels)
        ctl.addSpacing(8)
        ctl.addWidget(QtWidgets.QLabel("thickness:"))
        ctl.addWidget(self._ring_width)
        ctl.addStretch(1)
        ring.body.addLayout(ctl)
        sim_row = QtWidgets.QHBoxLayout(); sim_row.setSpacing(4)
        self._sim_btn = S.primary_btn("Simulate rings")
        self._sim_btn.setToolTip(
            "Simulate the enabled materials' rings once, from the current "
            "parameters. With \"live\" ticked, clicking this instead arms live "
            "mode (button turns green): rings then track every material, "
            "lattice, and geometry edit until live is switched off.")
        self._sim_btn.clicked.connect(self._on_sim_clicked)
        self._sim_live = QtWidgets.QCheckBox("live")
        self._sim_live.setToolTip(
            "Tick before clicking Simulate rings to keep the overlay in sync "
            "with the parameters. Unticked, a click is a one-shot simulation "
            "and later parameter edits leave the drawn rings untouched.")
        self._sim_live.toggled.connect(self._on_sim_live_toggled)
        self._sim_clear_btn = QtWidgets.QPushButton("✕")
        self._sim_clear_btn.setFixedSize(26, 26)
        self._sim_clear_btn.setToolTip("Remove all simulated rings from the image.")
        self._sim_clear_btn.clicked.connect(self._clear_simulated_rings)
        sim_row.addWidget(self._sim_btn, 1)
        sim_row.addWidget(self._sim_live)
        sim_row.addWidget(self._sim_clear_btn)
        ring.body.addLayout(sim_row)
        for w in (self._wl, self._lsd, self._px, self._max2t):
            w.valueChanged.connect(self._on_sim_param_changed)
        self._ring_info = QtWidgets.QPlainTextEdit(); self._ring_info.setReadOnly(True)
        self._ring_info.setMaximumHeight(140)
        self._ring_info.setStyleSheet(f"font-family:{S.MONO_CSS};font-size:10px")
        ring.body.addWidget(self._ring_info)
        # Transforms sits above Ring simulation: in the Data Viewer the card
        # column is Projection → this card, so this puts Transforms between
        # Projection and Ring simulation, where it belongs (it describes the
        # image, not the ring model).
        lv.addWidget(trans_card)
        lv.addWidget(ring)

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
        # tx was added to this card without this line, so it fed
        # get_geometry() while notifying nobody: editing the roll left the
        # pixel readout's η, the lab-frame compass, the radial profile and
        # (once the viewer gained one) the lab-frame image rotation all
        # sitting on the previous value until some *other* field was touched.
        self._tx.valueChanged.connect(self._on_bc_changed)

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
        self._sync_dspacing_picking()

    def _on_material_enabled(self, material: dict, checked: bool):
        """A material's enabled tick only changes which already-simulated
        rings are visible — not their geometry — so the on-image overlay must
        be redrawn here even when live mode is off (the case
        ``_on_sim_param_changed`` alone leaves untouched, see its docstring).
        Mirrors ``refresh_after_projection``'s same guard."""
        material["enabled"] = checked
        if self._any_material_rings():
            self._redraw_rings()
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

    def _any_dspacing_material(self) -> bool:
        """Is an enabled material a d-spacing-list (non-crystalline) one —
        AgBH, or a hand-entered d-spacing list?"""
        return any(m.get("enabled") and m.get("kind") == "dspacing"
                   for m in self._materials)

    def _sync_dspacing_picking(self):
        """Manual d-spacing ring picking is only meaningful for a SAXS-style
        d-spacing calibrant (AgBH), so its viewer controls follow the selected
        materials rather than sitting there permanently."""
        viewer = self._viewer
        if viewer is None or not hasattr(viewer, "set_dspacing_picking_visible"):
            return
        viewer.set_dspacing_picking_visible(self._any_dspacing_material())

    def _primary_material_name(self) -> str:
        for m in self._materials:
            if m["enabled"]:
                return m["name"]
        return self._materials[0]["name"] if self._materials else "Custom"

    # ── Ring simulation ───────────────────────────────────────────

    def _sim_is_live(self) -> bool:
        """Live ring simulation armed? Requires both the "live" tick and a
        click on Simulate rings (ticking the box alone changes nothing)."""
        return bool(getattr(self, "_sim_live_on", False))

    def _set_sim_live(self, on: bool):
        self._sim_live_on = bool(on)
        # Green while live, otherwise back to the plain accent "primary" look.
        self._sim_btn.setStyleSheet(S.SUCCESS_BTN_QSS if on else "")
        self._sim_btn.setText("Simulate rings (live)" if on else "Simulate rings")

    def _on_sim_clicked(self):
        """"Simulate rings" clicked — one-shot, or arm live mode if "live" is
        ticked. Either way the rings are (re)simulated right now."""
        if self._sim_live.isChecked():
            self._set_sim_live(True)
        self._simulate()

    def _on_sim_live_toggled(self, checked: bool):
        """Unticking "live" disarms live mode immediately (and drops the green);
        ticking it only takes effect on the next Simulate rings click."""
        if not checked:
            self._set_sim_live(False)

    def _clear_simulated_rings(self):
        """"✕" — drop every simulated ring from the image and from the
        profile's ring markers, and disarm live mode (otherwise the next
        parameter edit would immediately draw them again). Deliberately
        leaves the click-picked radius ring alone: that one is a manual
        marker, not a simulation."""
        self._set_sim_live(False)
        self._sim_live.setChecked(False)
        for m in self._materials:
            m["_rings"] = []
        self._ring_draw_geom = None
        self._clear_rings()
        self._refresh_profile_markers()
        self._ring_info.setPlainText("")

    def _on_sim_param_changed(self, *_):
        """Material/lattice/geometry field edited — resimulate while live mode is on.

        Guarded with ``getattr`` because materials are seeded (via
        ``_add_material``) before ``self._sim_btn`` exists during ``_build_ui``.

        λ/Lsd/px/max2θ feed the radial-integration geometry
        (``_effective_calib_geom``) even when live ring simulation is off —
        without this else-branch, editing them silently left the profile
        stale until something else (e.g. a beam-centre edit) happened to
        refresh it. Mirrors ``_on_bc_changed``'s tail. The ring overlay
        itself is *not* touched when live is off: a one-shot simulation is
        frozen at the parameters it was run with (see ``_ring_draw_geom``)."""
        self._sync_dspacing_picking()
        sim_btn = getattr(self, "_sim_btn", None)
        if sim_btn is not None and self._sim_is_live() and self._image_provider() is not None:
            self._simulate()
        else:
            self._maybe_auto_radial()
            self.geometryChanged.emit()

    def _compute_material_rings(self):
        """(Re)compute every enabled material's ring positions (``_rings``)
        from the current wavelength/Lsd/pixel-size/max-2θ — pure geometry
        (d-spacing + wavelength + detector distance/pixel size), no image
        needed. Returns ``(errors, any_rings)``. Split out of ``_simulate``
        so ``simulate_rings_without_image`` can reuse the same computation
        without that method's image requirement (which exists only for the
        on-image overlay ``_redraw_rings`` draws afterwards)."""
        errors, any_rings = [], False
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
        return errors, any_rings

    def simulate_rings_without_image(self):
        """Compute every enabled material's rings and refresh the profile-view
        markers, without requiring an image — used by the Data Viewer's Radial
        Profile "Load profile file" mode, where there is no image to overlay
        rings on but the profile markers (which need only the geometry, not
        detector pixels) should still reflect the current
        materials/wavelength/Lsd/pixel size."""
        self._compute_material_rings()
        self._refresh_profile_markers()

    def _simulate(self):
        img = self._image_provider()
        if img is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load data first."); return
        errors, any_rings = self._compute_material_rings()
        lines = []
        for m in self._materials:
            if not m["enabled"] or not m.get("_rings"):
                continue
            rings = m["_rings"]
            lines.append(f"{m['name']}: {len(rings)} rings")
            lines.append(f"{'hkl':>10}  {'2θ(°)':>7}  {'d(Å)':>7}  {'R(px)':>8}")
            for r in rings:
                label = f"n{r['order']}" if r["hkl"] is None else str(tuple(r["hkl"]))
                lines.append(f"{label:>10}  {r['two_theta_deg']:7.3f}  "
                             f"{r['d_spacing']:7.4f}  {r['radius_px']:8.1f}")
            lines.append("")
        # Freeze the placement parameters this simulation was run with, so a
        # later BC/tilt edit cannot silently move rings that are no longer
        # being recomputed (only live mode re-runs _simulate).
        self._ring_draw_geom = self._current_ring_geom() if any_rings else None
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

    def _current_ring_geom(self) -> dict:
        """The placement parameters the ring overlay is drawn with."""
        return {"bc_y": self._bcy.value(), "bc_z": self._bcz.value(),
                "ty": self._ty.value(), "tz": self._tz.value(),
                "px": self._px.value(), "lsd": self._lsd_um()}

    def _redraw_rings(self):
        self._clear_rings()
        img = self._image_provider()
        if self._viewer is None or img is None or not self._any_material_rings():
            return
        # Rings are placed with the geometry their radii were computed from,
        # not the live widgets — a one-shot simulation stays put when the
        # beam centre or a tilt is nudged afterwards. Live mode instead
        # tracks the widgets: a BC/tilt edit only moves rings (radii are
        # unchanged), so it never reaches _simulate and would otherwise
        # leave the overlay pinned to the snapshot.
        g = self._current_ring_geom() if self._sim_is_live() \
            else (self._ring_draw_geom or self._current_ring_geom())
        bc_y, bc_z = g["bc_y"], g["bc_z"]
        ty, tz = g["ty"], g["tz"]
        tilted = abs(ty) > 1e-9 or abs(tz) > 1e-9
        px = g["px"]
        lsd_um = g["lsd"]
        th = np.linspace(0, 2 * math.pi, 400)
        vis_r = self._show_rings.isChecked()
        vis_l = self._show_labels.isChecked() and vis_r
        # Ring arcs and their labels are confined to the detector image: a
        # simulated ring drawn across the empty canvas beside the frame is not
        # a prediction about anything the user can check, and the labels out
        # there were the noisiest part of it. Points off the image become NaN
        # and `connect="finite"` breaks the polyline there, so an arc that
        # leaves and re-enters the frame is drawn as the two visible segments
        # rather than one chord across the gap. (The beam-centre marker is
        # deliberately exempt — it is the one thing worth seeing off-image.)
        vb = self._viewer._iv.getView().getViewBox()
        try:
            px_w, px_h = vb.viewPixelSize()
            y_up = not vb.yInverted()
        except Exception:
            px_w = px_h = 1.0
            y_up = True
        px_w = px_w if px_w and px_w > 0 else 1.0
        px_h = px_h if px_h and px_h > 0 else 1.0
        label_font = S.font_px()
        fm = QtGui.QFontMetrics(label_font)
        box_h = fm.height() * px_h
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
                                             lsd_um, bc_y, bc_z, px, px)
                else:
                    ys = bc_y + rad * np.cos(th); zs = bc_z + rad * np.sin(th)
                label = f"n{r['order']}" if r["hkl"] is None else "".join(str(x) for x in r["hkl"])
                placement = _ring_label_pos(
                    ys, zs, img.shape,
                    box_w=fm.horizontalAdvance(label) * px_w, box_h=box_h, y_up=y_up)
                if placement is None:
                    continue          # ring misses the detector entirely
                on = _ring_on_image_mask(ys, zs, img.shape)
                ys_clip = np.where(on, ys, np.nan)
                zs_clip = np.where(on, zs, np.nan)
                item = pg.PlotDataItem(ys_clip, zs_clip, pen=pen, connect="finite")
                item.setVisible(vis_r)
                self._viewer._iv.addItem(item); self._ring_items.append(item)
                label_y, label_z, anchor = placement
                txt = pg.TextItem(label, color=m["color"], anchor=anchor)
                txt.setFont(label_font)
                txt.setPos(label_y, label_z)
                txt.setVisible(vis_l)
                self._viewer._iv.addItem(txt); self._label_items.append(txt)
        # Beam-centre marker — always the *live* beam centre, not the frozen
        # one the rings were drawn with, so a BC edit is visibly reflected.
        bc = pg.ScatterPlotItem([self._bcy.value()], [self._bcz.value()],
                                symbol="+", size=16,
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
        self._refresh_profile_markers()

    # ── Beam-centre picking / radial integration ──────────────────

    def _set_bc(self, bc_y, bc_z):
        """Move the beam centre to (bc_y, bc_z) as ONE change.

        Setting the two spins separately fires valueChanged twice, so every
        pick cost two full passes of _on_bc_changed -- and downstream, in
        Hydra, two rebuilds of a 44-megapixel four-panel composite on the GUI
        thread. The first of those ran at a geometry nobody asked for: the new
        bc_y paired with the OLD bc_z. So this is not only half the work, it
        drops a rebuild at a beam centre that never existed.
        """
        self._bcy.blockSignals(True); self._bcz.blockSignals(True)
        try:
            self._bcy.setValue(bc_y)
            self._bcz.setValue(bc_z)
        finally:
            self._bcy.blockSignals(False); self._bcz.blockSignals(False)
        self._on_bc_changed()

    def _on_bc_picked(self, bc_y, bc_z):
        """Single-click BC pick from the image (PickableImageViewer)."""
        self._bc_auto.setChecked(False)
        self._set_bc(bc_y, bc_z)

    def _on_ring_fit_bc(self, bc_y, bc_z, r_px):
        """BC from a 3+ point circle fit on a ring (PickableImageViewer)."""
        self._bc_auto.setChecked(False)
        self._set_bc(bc_y, bc_z)

    def _on_bc_changed(self, *_):
        """Beam centre (or a tilt) edited manually or by a pick — refresh
        overlays/plot. In live mode the rings move with it; otherwise only the
        beam-centre marker does (the rings stay frozen at the parameters they
        were simulated with)."""
        if self._sim_is_live():
            self._ring_draw_geom = self._current_ring_geom()
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
        """(Re)draw the click-picked ring (magenta) about the current beam
        centre. ``r`` came from the profile's ``_x_to_r`` (a flat-panel 2θ
        radius), so on a tilted geometry it must go back through the same
        tilt projection ``_redraw_rings`` uses for material rings — a plain
        circle at that radius is the wrong curve once ty/tz != 0 and drifts
        away from where the ring (and the peak it marks) actually sits."""
        if self._pick_ring_item is not None and self._viewer is not None:
            self._viewer._iv.removeItem(self._pick_ring_item)
            self._pick_ring_item = None
        r = self._picked_r
        img = self._image_provider()
        if r is None or img is None or self._viewer is None:
            return
        bc_y, bc_z = self._bcy.value(), self._bcz.value()
        ty, tz = self._ty.value(), self._tz.value()
        if abs(ty) > 1e-9 or abs(tz) > 1e-9:
            lsd_um, px = self._lsd_um(), self._px.value()
            two_theta_deg = math.degrees(math.atan2(r * px, lsd_um))
            ys, zs = tilted_ring_xy(two_theta_deg, 0.0, ty, tz,
                                     lsd_um, bc_y, bc_z, px, px)
        else:
            th = np.linspace(0, 2 * math.pi, 512)
            ys, zs = bc_y + r * np.cos(th), bc_z + r * np.sin(th)
        self._pick_ring_item = pg.PlotDataItem(
            ys, zs, pen=pg.mkPen("#ff30ff", width=1.8))
        self._viewer._iv.addItem(self._pick_ring_item)

    def _on_rad_param_changed(self, *_):
        self._maybe_auto_radial()

    def _refresh_profile_markers(self):
        if self._profile_view is None:
            return
        # Same hkl/order text as the on-image ring labels, so a peak in the
        # profile and the arc it came from carry the identical name. Hidden
        # together with the image overlay: unchecking "Rings" should clear
        # the profile markers too, not just the on-image arcs.
        groups = [{"radii": [r["radius_px"] for r in m["_rings"]],
                   "labels": [f"n{r['order']}" if r["hkl"] is None
                              else "".join(str(x) for x in r["hkl"])
                              for r in m["_rings"]],
                   "color": m["color"]}
                  for m in self._materials if m["enabled"] and m.get("_rings")
                  ] if self._show_rings.isChecked() else []
        self._profile_view.set_ring_markers(
            groups, self._lsd_um(), self._px.value(), self._wl.value(),
            width=self._ring_width.value())

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
            for w, key, scale in ((self._wl, "wavelength_A", 1.0),
                                  (self._lsd, "Lsd", 0.001),
                                  (self._bcy, "BC_y", 1.0), (self._bcz, "BC_z", 1.0),
                                  (self._ty, "ty", 1.0), (self._tz, "tz", 1.0)):
                if not self._widget_edited(w, geom.get(key), scale):
                    continue
                geom[key] = w.value() / scale
            if self._widget_edited(self._px, geom.get("pxY")):
                # One px widget for both axes, so an edit can only set them equal.
                geom["pxY"] = geom["pxZ"] = self._px.value()
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
            "tx": self._tx.value(), "ty": ty, "tz": tz,
            "pxY": px, "pxZ": px,
            "NrPixelsY": ny, "NrPixelsZ": nz, "distortion": {},
            "im_trans": list(self.im_trans_codes()),
        }

    @staticmethod
    def _widget_edited(w: QtWidgets.QDoubleSpinBox, frozen, scale: float = 1.0) -> bool:
        """Has ``w`` been moved off the calibration's own ``frozen`` value?

        The spinboxes display 1–4 decimals, so reading the geometry back out of
        them quantises it: at this file's 14° tilt, ty/tz alone round to 0.01°
        and move every ring by ~0.5 px — precisely the accuracy the "Accurate"
        path exists to deliver. So a widget still showing the frozen value (to
        within its own display resolution, which is as finely as a user can
        type into it) is treated as untouched, and the full-precision number
        from the calibration file is kept. Only a real edit overrides it."""
        if frozen is None:
            return True
        return abs(w.value() - float(frozen) * scale) > 0.5 * 10.0 ** -w.decimals()

    def _engine_geom(self, img: np.ndarray) -> Optional[dict]:
        """Full geometry for the accurate (engine) path — like
        ``_effective_calib_geom`` but never returns ``None`` just because the
        tilts happen to be zero: the engine path is explicitly requested
        there, so a flat-detector geometry is still built from the live
        widgets. ``None`` only when there is no image to size it from."""
        geom = self._effective_calib_geom(img)
        if geom is not None:
            return geom
        if img is None:
            return None
        nz, ny = img.shape
        px = self._px.value()
        return {
            "wavelength_A": self._wl.value(), "Lsd": self._lsd_um(),
            "BC_y": self._bcy.value(), "BC_z": self._bcz.value(),
            "tx": self._tx.value(), "ty": self._ty.value(), "tz": self._tz.value(),
            "pxY": px, "pxZ": px,
            "NrPixelsY": ny, "NrPixelsZ": nz, "distortion": {},
            "im_trans": list(self.im_trans_codes()),
        }

    def show_radial_help(self):
        """Explain how the radial-integration plot's profile is computed."""
        QtWidgets.QMessageBox.information(
            self, "Radial integration — how it's calculated",
            "The plot shows intensity vs. radius: the azimuthal (angular) mean "
            "of the image about the beam centre, grouped into rings of width "
            "\"R bin\".\n\n"
            "\"Accurate\" ticked: the full Batch-Integrate pipeline runs — the "
            "same MIDAS engine, geometry and subpixel K=2 kernel Batch Integrate "
            "uses — so detector tilt (tx/ty/tz), pixel size and distortion are all "
            "honoured, and each R-bin's value is a pixel-count-weighted mean "
            "across η: Σ(cell_mean·count) / Σ(count). This is the trustworthy "
            "profile, but it is far slower and will not keep up with a live "
            "stream.\n\n"
            "\"Accurate\" unticked (the default, fast):\n"
            "• Calibration loaded, or a tilt (ty/tz) set on the Ring-simulation "
            "card: the MIDAS engine is used with the hard-binning kernel (each "
            "pixel lands wholly in one cell — no subpixel splitting).\n"
            "• Otherwise: a fast circle-binning fallback. Pixels are grouped "
            "purely by distance from the beam centre (BC_y, BC_z) into R-bins, "
            "and each bin's value is Σintensity / Σpixels — a plain per-bin "
            "mean, with no tilt correction.\n\n"
            "If full-geometry integration fails, the plot automatically falls "
            "back to circle binning and a warning is shown above the "
            "calibration card.")

    def radial_integrate(self):
        """Azimuthal mean of the current frame.

        Fast path (the default): the MIDAS engine with the hard-binning kernel
        when a calibration file is loaded or a tilt is dialled into the
        Ring-simulation card, and a plain circle binning about the beam centre
        otherwise. This is cheap enough to keep up with a live stream, but it
        does not split pixels across bins.

        Accurate path (the "Accurate" tick above the plot): always the full
        Batch-Integrate pipeline — same geometry build, same subpixel K=2
        kernel — so tilts, pixel size and distortion are fully accounted for.
        """
        img = self._image_provider()
        if img is None or self._rad_r_bin is None or self._profile_view is None:
            return
        mask = self._mask_provider(img) if self._mask_provider is not None else None
        accurate = self._rad_accurate is not None and self._rad_accurate.isChecked()
        geom = self._engine_geom(img) if accurate else self._effective_calib_geom(img)
        r_axis = prof = None
        if geom is not None:
            try:
                r_axis, prof = self._midas_radial(
                    img, geom, mask,
                    kernel=(ACCURATE_KERNEL if accurate else FAST_KERNEL),
                    # The cake has its own bin-size controls and its own
                    # Calculate button once they are bound, so this run must
                    # not overwrite it with a differently-binned by-product.
                    set_cake=(self._cake_r_bin is None))
            except Exception:
                import traceback
                self._calib_lbl.setText(
                    "Full-geometry integration failed — using circle binning. "
                    "See error log.")
                self._log_error(traceback.format_exc())
        if prof is None:
            r_axis, prof = self._radial_profile(
                img, self._bcy.value(), self._bcz.value(),
                self._rad_r_bin.value(), mask=mask)
        self._profile_view.set_profile(
            r_axis, prof, wavelength_A=self._wl.value(),
            lsd_um=self._lsd_um(), px_um=self._px.value())
        self._refresh_profile_markers()

    def cake_integrate(self):
        """Compute + display the (η, R) cake for the current frame.

        Always takes the accurate route: the full Batch-Integrate pipeline
        (MIDAS engine, subpixel K=2) through the loaded calibration's geometry
        or one synthesized from the live Ring-simulation widgets, at the R and
        η bin sizes set above the plot. Unlike the radial profile this is an
        explicit, on-demand Calculate, so there is no live-view budget to keep
        to. A plain polar binning about the beam centre is the fallback if the
        engine path fails outright.
        """
        img = self._image_provider()
        if img is None or self._cake_view is None:
            return
        mask = self._mask_provider(img) if self._mask_provider is not None else None
        geom = self._engine_geom(img)
        if geom is not None:
            try:
                self._midas_radial(img, geom, mask, kernel=ACCURATE_KERNEL,
                                   r_bin=self._cake_r_bin_value(),
                                   eta_bin=self._cake_eta_bin_value(),
                                   set_cake=True)
                return
            except Exception:
                import traceback
                self._calib_lbl.setText(
                    "Full-geometry cake integration failed — using circle "
                    "binning. See error log.")
                self._log_error(traceback.format_exc())
        cake, r_axis, eta_axis = self._cake_bin(
            img, self._bcy.value(), self._bcz.value(),
            self._cake_r_bin_value(), eta_bin=self._cake_eta_bin_value(),
            mask=mask)
        self._set_cake_axis_context()
        self._cake_view.set_cake(cake, r_axis, eta_axis)

    def _set_cake_axis_context(self):
        """Hand the cake view the live geometry so its x-axis can be labelled
        in 2θ / d / Q as well as R (px). Cheap and idempotent — called on every
        cake refresh, since the user may have dialled the fields since the last
        one."""
        if self._cake_view is None or not hasattr(self._cake_view, "set_axis_context"):
            return
        self._cake_view.set_axis_context(
            self._lsd_um(), self._px.value(), self._wl.value())

    def _cake_r_bin_value(self) -> float:
        """Cake R bin size — its own control if bound, else the profile's."""
        if self._cake_r_bin is not None:
            return max(float(self._cake_r_bin.value()), 0.1)
        if self._rad_r_bin is not None:
            return max(float(self._rad_r_bin.value()), 0.1)
        return 1.0

    def _cake_eta_bin_value(self) -> float:
        if self._cake_eta_bin is not None:
            return max(float(self._cake_eta_bin.value()), 0.05)
        return CAKE_ETA_BIN_DEG

    def _midas_radial(self, img, g, mask, *, kernel: str = FAST_KERNEL,
                      r_bin: Optional[float] = None,
                      eta_bin: float = CAKE_ETA_BIN_DEG,
                      set_cake: bool = True):
        """Radial profile via the MIDAS engine, honouring the given geometry's
        tilts + distortion (not just concentric circles). ``g`` is either the
        loaded calibration's geometry or one synthesized from the live Ring-sim
        widgets (see ``_effective_calib_geom``/``_engine_geom``). ``kernel`` is
        the same integration kernel Batch Integrate exposes — ``"hard"`` for
        the fast live-view path, ``"subpixel2"`` for the accurate one.

        The binning geometry is built once per (geometry, kernel, bins, image
        shape, mask) and cached, so alternating between the fast profile, the
        accurate profile and the cake does not rebuild a context each time;
        only the per-frame integration runs on a frame change.

        ``img`` is the display-oriented frame (``image_provider`` already
        applied this card's Transforms checkboxes, for on-screen viewing) —
        it is un-transformed back to raw here so the backend can apply
        ``g["im_trans"]`` itself via ``spec.TransOpt`` exactly once, the same
        as every other integration call site."""
        import json
        import torch
        if r_bin is None:
            r_bin = self._rad_r_bin.value() if self._rad_r_bin is not None else 1.0
        r_bin = max(float(r_bin), 0.1)
        eta_bin = max(float(eta_bin), 0.05)
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
               round(r_bin, 4), round(eta_bin, 4), kernel, (nz, ny), mask_fp, im_trans,
               json.dumps(g.get("distortion") or {}, sort_keys=True))
        entry = self._calib_ctx_cache.get(sig)
        if entry is None:
            spec = _spec_from_result_ns(
                r_bin, eta_bin, NrPixelsY=ny, NrPixelsZ=nz,
                pxY=g["pxY"], pxZ=g.get("pxZ") or g["pxY"], Lsd=g["Lsd"],
                BC_y=g["BC_y"], BC_z=g["BC_z"], tx=g.get("tx") or 0.0,
                ty=g.get("ty") or 0.0, tz=g.get("tz") or 0.0,
                wavelength_A=g["wavelength_A"], distortion=g.get("distortion") or {},
                im_trans=im_trans)
            ctx = build_integration_context(spec, kernel, mask, (None, None), weighted=True)
            entry = (spec, ctx)
            # Bounded: the handful of live combinations (fast/accurate profile,
            # cake) churn otherwise as the user drags a bin-size spinbox.
            if len(self._calib_ctx_cache) >= _CTX_CACHE_MAX:
                self._calib_ctx_cache.pop(next(iter(self._calib_ctx_cache)))
            self._calib_ctx_cache[sig] = entry
        spec, ctx = entry
        img_t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float64))
        # return_cake=True → a 4-tuple (prof, sigma, cake, cake_sigma). Unpacking
        # it into three raised ValueError on *every* call, so this whole engine
        # path silently fell back to circle binning (see radial_integrate /
        # cake_integrate) — which drops the tilt correction and splits every
        # peak on a tilted geometry.
        prof, _, cake_2d, _ = integrate_frame(
            img_t, spec, ctx["geom"], kernel, (None, None), None, False,
            corr_counts=ctx["corr_counts"], return_cake=True,
            weighted=True, cnt_cake=ctx["cnt"])
        if set_cake and self._cake_view is not None:
            n_eta = spec.n_eta_bins
            eta_ax = float(spec.EtaMin) + float(spec.EtaBinSize) * (np.arange(n_eta) + 0.5)
            self._set_cake_axis_context()
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

    def _eta_grid(self, shape, bc_y, bc_z, eta_bin):
        """Cached per-pixel η bin index + axis, keyed on (shape, BC, η-bin) —
        the azimuthal counterpart of ``_radial_grid``.

        η follows the app's own display-space convention, η = atan2(Y − BC_y,
        Z − BC_z) with 0° straight up (+Z) — the same one
        ``helpers.draw_polar_bin_overlay`` draws its η spokes with — so a
        feature in the cake sits at the η the on-image overlay marks."""
        eta_bin = max(float(eta_bin), 1e-3)
        key = (tuple(shape), round(float(bc_y), 4), round(float(bc_z), 4),
               round(eta_bin, 6))
        cache = self._eta_grid_cache
        if cache is not None and cache[0] == key:
            return cache[1], cache[2], cache[3]
        NZ, NY = shape
        zz, yy = np.indices((NZ, NY))
        eta = np.degrees(np.arctan2(yy - bc_y, zz - bc_z))       # (-180, 180]
        nbins = max(1, int(math.ceil(360.0 / eta_bin)))
        which = np.clip(((eta + 180.0) / eta_bin).astype(np.int64),
                        0, nbins - 1).ravel()
        eta_axis = -180.0 + (np.arange(nbins) + 0.5) * eta_bin
        self._eta_grid_cache = (key, which, nbins, eta_axis)
        return which, nbins, eta_axis

    def _cake_bin(self, img: np.ndarray, bc_y: float, bc_z: float,
                  r_bin: float = 1.0, eta_bin: float = CAKE_ETA_BIN_DEG,
                  mask: Optional[np.ndarray] = None):
        """Mean intensity per (η, R) bin about (bc_y, bc_z), using the cached
        grids — the 2-D counterpart of ``_radial_profile``, and equally
        tilt-blind (pixels are grouped purely by distance and azimuth about
        the beam centre).

        ``mask`` (bool, True = exclude) drops pixels; non-finite pixels are
        ignored. Returns (cake (n_eta, n_r), r_axis_px, eta_axis_deg). Empty
        bins are 0.0 rather than NaN, matching the engine cake — CakeViewer
        keeps exact zeros out of its auto-level window already.
        """
        which_r, n_r, r_axis = self._radial_grid(img.shape, bc_y, bc_z, r_bin)
        which_eta, n_eta, eta_axis = self._eta_grid(img.shape, bc_y, bc_z, eta_bin)
        vals = img.ravel()
        good = np.isfinite(vals)
        if mask is not None:
            good &= ~mask.ravel()
        flat = which_eta[good] * n_r + which_r[good]
        n = n_eta * n_r
        sums = np.bincount(flat, weights=vals[good], minlength=n)
        counts = np.bincount(flat, minlength=n)
        cake = np.zeros(n, dtype=np.float64)
        nz = counts > 0
        cake[nz] = sums[nz] / counts[nz]
        return cake.reshape(n_eta, n_r), r_axis, eta_axis

    # ── Calibration file ────────────────────────────────────────────

    def _browse_calib(self):
        p = _browse(self, "Open calibration file",
                    "Calibration (*.json *.poni *.txt);;All (*)",
                    start_dir=browse_start_dir(self._calib_ed.text()))
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
            self._calib_ctx_cache.clear()
            try:
                self._calib_geom = geometry_fields_from_file(path)
                d = self._calib_geom
                for w, key in ((self._tx, "tx"), (self._ty, "ty"), (self._tz, "tz")):
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
            self._snapshot_calib_baseline()
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
        """Full geometry dict for calibration export and for whoever owns this
        card (the Hydra page syncs it into a ``hydra.DetectorState``).

        A loaded calibration supplies what no widget can -- distortion
        coefficients and the detector size -- and the **live fields win over
        it**, because the fields are what the user can see and edit.

        This used to return ``self._calib_geom`` wholesale, which made every
        geometry box on the card inert the moment a calibration was loaded.
        On the Hydra page that is always: each panel loads a bundled
        ``ps_ge{n}.txt`` at startup, so editing tx/ty/tz or the beam centre
        there changed nothing at all and the boxes displayed values the
        composite was not using. Measured before the fix: panel 1's box read
        0.00 while the composite placed it at 296.885 deg, and typing 138 into
        the box left both numbers untouched.

        Loading writes the file's values into those same widgets (see
        :meth:`set_geometry` and :meth:`_load_calibration`), so straight after
        a load the override is a no-op and only a deliberate edit diverges.
        """
        base = dict(self._calib_geom) if self._calib_geom is not None else None
        img = self._image_provider()
        if base is None and img is None:
            return None
        px = self._px.value()
        live = {
            "wavelength_A": self._wl.value(), "Lsd": self._lsd_um(),
            "BC_y": self._bcy.value(), "BC_z": self._bcz.value(),
            "tx": self._tx.value(), "ty": self._ty.value(), "tz": self._tz.value(),
            "pxY": px, "im_trans": self.im_trans_codes(),
        }
        if base is None:
            nz, ny = img.shape
            return {**live, "pxZ": px, "NrPixelsY": ny, "NrPixelsZ": nz,
                    "distortion": {}}
        # Override only the fields the user has actually MOVED since the
        # calibration was loaded, compared against what the load itself wrote
        # into each box -- not against the file's value. The boxes are
        # fixed-decimal (tx has 2), so a file's 296.885 displays as 296.88 and
        # "the widget always wins" would quietly round a calibrated roll by
        # 0.005 deg, about 0.15 px out at the edge of a Hydra panel. Comparing
        # against the baseline makes an untouched box mean exactly nothing.
        baseline = self._calib_widget_baseline or {}
        geom = dict(base)
        for key, val in live.items():
            if key not in baseline or baseline[key] != val:
                geom[key] = val
        # One px box against a file that may carry a non-square pair: only a
        # changed box means "square pixels, this size". An untouched one must
        # not flatten pxZ onto pxY behind the user's back.
        if "pxY" in geom and abs(float(base.get("pxY") or px) - px) > 1e-9:
            geom["pxZ"] = px
        return geom

    def _snapshot_calib_baseline(self) -> None:
        """Remember what a calibration load just put into each geometry box,
        so :meth:`_export_geom` can tell an untouched box from an edited one
        without fighting the boxes' display precision."""
        self._calib_widget_baseline = {
            "wavelength_A": self._wl.value(), "Lsd": self._lsd_um(),
            "BC_y": self._bcy.value(), "BC_z": self._bcz.value(),
            "tx": self._tx.value(), "ty": self._ty.value(), "tz": self._tz.value(),
            "pxY": self._px.value(), "im_trans": self.im_trans_codes(),
        }

    def _save_calibration(self, kind: str):
        """Save the current geometry (see ``_export_geom``) as a calibration
        file — JSON (GUI bare-key format), MIDAS paramstest.txt, or pyFAI .poni."""
        geom = self._export_geom()
        if geom is None:
            QtWidgets.QMessageBox.warning(self, "No geometry", "Load data first.")
            return
        specs = {
            "json": ("Save calibration.json", "calibration.json",
                     ".instr.json", "JSON (*.json)"),
            "paramstest": ("Save MIDAS parameter file", "paramstest.txt",
                           ".instr.txt", "Text (*.txt)"),
            "poni": ("Save calibration.poni", "calibration.poni",
                     ".poni", "PONI (*.poni)"),
        }
        title, default_name, suffix, filt = specs[kind]
        if self._save_name_provider is not None:
            try:
                default_name = self._save_name_provider(suffix) or default_name
            except Exception:
                pass      # naming is a convenience; never block a save
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
