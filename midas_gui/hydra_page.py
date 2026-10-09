"""Hydra (4-panel GE detector) page for the Data Viewer tab.

``HydraViewerPage`` wires together the pieces built in earlier phases:
``HydraLoaderPanel`` (sibling auto-discovery + frame navigation),
``HydraDetectorToolbar`` (ge1/ge2/ge3/ge4/composite selector),
``midas_gui.hydra`` (the windmill-compositing engine), and one
``DetectorGeometryCard`` per panel + one for the composite — each panel's
beam-centre/ring calibration is independent, matching the physical reality
that the 4 GE panels are separate detectors.

Scope for this first version (see the approved implementation plan):
- No dark/bright/background correction or intensity-range masking for Hydra
  frames (single-detector mode already has this; Hydra reuses raw frames).
- The R-bin/Auto/Integrate radial-toolbar controls are shared across all 5
  panel cards (one setting, not duplicated 5x) — same simplification the
  plan calls out for v1.

The radial-integration plot (``HydraProfileViewer``) shows all 4 panels'
profiles at once (each computed independently, from its own beam-centre/
geometry) plus a toggleable "Composite" curve — the NaN-aware sum of the
four, each resampled onto a shared 2theta axis first (not a radial
integration of the already-composited image, which would double-count any
panel overlap and mix registration error into the profile).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui import hydra
from midas_gui.helpers import (_load_image, _apply_im_trans, _fspin, apply_field_corrections,
                         geometry_fields_from_file, widgets_to_dict, apply_dict_to_widgets)
from midas_gui.hydra_geometry_card import DetectorGeometryCard
from midas_gui.hydra_widgets import HydraLoaderPanel, HydraDetectorToolbar, HydraProfileViewer
from midas_gui.roi_tools import ROIImageViewer, ROIRibbon
from midas_gui.helpers import browse_start_dir
from midas_gui.widgets import (_convert_radial, OriginToolButton,
                               build_lab_frame_axes_items)
from midas_gui import style as S

#: Decimation factor for the composite built for the Data Viewer.
#:
#: The windmill canvas for a real four-panel GE array is 6656x6656 -- 44.3
#: megapixels, several times any screen, rebuilt synchronously on the GUI
#: thread. Building it every second pixel costs a quarter of the work
#: (remap 2.09 s -> ~0.5 s, composite max 1.61 s -> ~0.4 s, autolevel
#: 0.49 s -> ~0.12 s) for an image still larger than the widget showing it.
#:
#: This is display resolution only, and it is NOT a shortcut taken behind a
#: measurement's back: the "Composite" radial curve sums the per-panel
#: profiles rather than integrating this image (see _refresh_composite_curve),
#: and Calibrate imports per-panel geometries, not this canvas (see
#: export_for_calibration). What does read it -- the composite card's own
#: overlays and radial integration -- is kept correct by scaling that card's
#: pixel size, beam centre and NrPixels to match, in
#: _reseed_composite_card_if_needed. Set to 1 for a full-resolution canvas.
COMPOSITE_DISPLAY_STEP = 2


class _ProfileSinkAdapter:
    """Adapts a DetectorGeometryCard's ProfileViewer-shaped calls
    (``set_profile``/``set_ring_markers``) into one named curve on a shared
    ``HydraProfileViewer`` — each card thinks it owns a normal profile
    plot, but all 5 draw onto the same multi-curve widget."""

    def __init__(self, viewer: HydraProfileViewer, key: str):
        self._viewer = viewer
        self._key = key

    def set_profile(self, r_px, profile, *, sigma=None, wavelength_A=None,
                    lsd_um=None, px_um=None):
        self._viewer.set_curve(self._key, r_px, profile,
                               lsd_um=lsd_um, px_um=px_um, wavelength_A=wavelength_A)

    def set_ring_markers(self, groups, lsd_um=None, px_um=None, wl=None, width=1.5):
        # Ring markers (vertical lines at each material's ring 2theta) aren't
        # well-defined on a plot overlaying 4 independently-calibrated
        # panels at once — skipped for this shared multi-curve plot.
        pass


class HydraViewerPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._states: dict = {}                 # panel number -> hydra.DetectorState
        self._raw_frames: dict = {}              # panel number -> last-loaded frame (post ImTrans)
        self._composite_img: Optional[np.ndarray] = None
        self._big_det_size: Optional[int] = None
        #: (canvas size, mean Lsd, px, per-panel Lsd) the composite card was
        #: last seeded for -- see _reseed_composite_card_if_needed.
        self._composite_seeded_key: Optional[tuple] = None
        self._disp_key = None                    # (shape, active key) — fresh-display detection
        self._active_card: Optional[DetectorGeometryCard] = None
        self._syncing_shared = False             # re-entrancy guard for _sync_shared_fields
        self._build_ui()
        self._on_panel_changed(self._toolbar.current())

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(0)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False); split.setHandleWidth(6)
        root.addWidget(split)

        # ── LEFT: Hydra data loader ──
        self._loader = HydraLoaderPanel()
        self._loader.setMinimumWidth(200)
        self._loader.siblingsChanged.connect(self._on_siblings_changed)
        self._loader.frameChanged.connect(self._on_frame_changed)
        self._loader.fieldsChanged.connect(self._on_fields_changed)
        self._loader.projectionChanged.connect(self._on_fields_changed)
        split.addWidget(self._loader)

        # ── MIDDLE: Projection card (top) + one geometry card per panel +
        #    one for the composite (below, same as the single-detector tab's
        #    Projection-card-then-geometry-card middle-panel layout) ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setMinimumWidth(260)
        inner = QtWidgets.QWidget()
        inner_lv = QtWidgets.QVBoxLayout(inner)
        inner_lv.setContentsMargins(0, 0, 0, 0); inner_lv.setSpacing(8)
        inner_lv.addWidget(self._loader.projection_card())
        # Per-panel calibration is the whole point of the composite, and the
        # only way to set it was to switch panel, browse, load, four times
        # over -- with the card showing one panel at a time, so you could not
        # see what the other three were on. One pick, matched by filename.
        self._load4_btn = QtWidgets.QPushButton("Load 4 panel calibrations…")
        self._load4_btn.setToolTip(
            "Pick the calibration files for ge1-ge4 in one go. Each file is "
            "matched to a panel by 'ge<n>' in its name; anything unmatched is "
            "reported rather than guessed at.")
        self._load4_btn.clicked.connect(self._load_four_calibrations)
        inner_lv.addWidget(self._load4_btn)
        self._load4_lbl = QtWidgets.QLabel("")
        self._load4_lbl.setStyleSheet("color:#9a9a9a;font-size:10px")
        self._load4_lbl.setWordWrap(True)
        inner_lv.addWidget(self._load4_lbl)
        self._card_stack = QtWidgets.QStackedWidget()
        inner_lv.addWidget(self._card_stack, 1)
        scroll.setWidget(inner)
        # The multi-curve radial plot and shared R-bin/Auto controls are
        # built below (right side) before this loop's set_profile_view/
        # set_radial_controls calls, so build them first and reorder the
        # widget insertion into `right`/`split` afterward.
        self._profile_view = HydraProfileViewer()
        self._rad_r_bin = _fspin(0.1, 20.0, 2, 1.0, "px"); self._rad_r_bin.setFixedWidth(56)
        self._rad_r_bin.setToolTip(
            "Radial bin size for the azimuthal mean — shared across all "
            "Hydra panels and the composite.")
        self._rad_auto = QtWidgets.QCheckBox("Auto"); self._rad_auto.setChecked(True)
        self._rad_auto.setToolTip("Recompute each panel's radial integration when "
                                  "its beam centre, tilt, or the frame changes.")

        self._cards: dict = {}
        for key in ("ge1", "ge2", "ge3", "ge4", "composite"):
            card = DetectorGeometryCard(show_rotate=(key != "composite"))
            card.set_image_source(self._make_image_provider(key), None)
            # Four panels plus a composite all saving "paramstest.txt" into
            # the launch directory is four chances to overwrite the wrong
            # one, so each proposes a name carrying which panel it is.
            card.set_save_name_provider(
                lambda suffix, k=key: self._default_save_path(k, suffix))
            if key == "composite":
                # The "Composite" curve on the shared plot is the DERIVED
                # resample-and-sum of ge1-4's own curves (see
                # _refresh_composite_curve) — deliberately NOT this card's
                # own radial_integrate() (which would instead integrate the
                # already-composited image and reintroduce the
                # double-counting/registration-error problem that design
                # explicitly avoids). Leaving its profile/radial-control
                # bindings unset makes its own radial_integrate() a no-op;
                # its ring-simulation/BC fields are still fully functional
                # for the on-image ring overlay. λ/max2θ/px are still kept
                # in sync with the ge1-4 cards (see _sync_shared_fields) —
                # same beam/detector model, so its ring radii should match.
                card.geometryChanged.connect(self._on_composite_geometry_changed)
            else:
                # Every ge-card's own curve is bound once, permanently —
                # unlike the viewer (only one image shown at a time), all 4
                # profiles should stay visible on the shared plot regardless
                # of which panel is the currently *active* one for display.
                card.set_profile_view(_ProfileSinkAdapter(self._profile_view, key))
                card.set_radial_controls(self._rad_r_bin, self._rad_auto)
                n = int(key[2])
                card.geometryChanged.connect(lambda n=n: self._on_card_geometry_changed(n))
                card.imTransChanged.connect(lambda n=n: self._on_card_geometry_changed(n))
            self._cards[key] = card
            self._card_stack.addWidget(card)
        split.addWidget(scroll)

        # ── RIGHT: image viewer (with its own toolbar row) + radial plot ──
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.setHandleWidth(8)
        self._viewer = ROIImageViewer()
        # Same display-origin selector as the single-detector Data Viewer.
        self._origin_btn = OriginToolButton(self._viewer)
        self._viewer._toolbar_layout.addWidget(self._origin_btn)
        self._toolbar = HydraDetectorToolbar()
        self._toolbar.panelChanged.connect(self._on_panel_changed)
        self._viewer._toolbar_layout.addWidget(self._toolbar)
        # Same overlay and the same control as the single-detector Data
        # Viewer. It matters more here: the composite is built by rotating
        # four panels about the beam, so "which way is the hutch" is the one
        # thing a windmill picture makes hard to answer by eye.
        self._axis_items: list = []
        self._lab_axes_on = QtWidgets.QCheckBox("Lab-frame axes")
        self._lab_axes_on.setToolTip(
            "Overlay MIDAS lab-frame axes (X_Lab/Y_Lab), the beam-direction ⊗ "
            "glyph, and an η sweep arc, anchored at the beam centre of "
            "whichever panel (or the composite) is shown.")
        self._lab_axes_on.toggled.connect(self._on_lab_axes_toggled)
        self._viewer._toolbar_layout.addWidget(self._lab_axes_on)
        self._viewer.originChanged.connect(self._redraw_lab_axes_if_on)
        self._roi_ribbon = ROIRibbon()
        self._viewer.set_ribbon(self._roi_ribbon)
        viewer_container = QtWidgets.QWidget()
        vc_layout = QtWidgets.QHBoxLayout(viewer_container)
        vc_layout.setContentsMargins(0, 0, 0, 0); vc_layout.setSpacing(0)
        vc_layout.addWidget(self._roi_ribbon)
        vc_layout.addWidget(self._viewer, 1)
        right.addWidget(viewer_container)

        ptb = self._profile_view._toolbar_layout
        self._rad_btn = QtWidgets.QPushButton("Integrate")
        self._rad_btn.setToolTip("Recompute all 4 panels' (and the composite's) "
                                 "radial integration now, regardless of Auto.")
        self._rad_btn.clicked.connect(lambda: self._refresh_profile_curves(force=True))
        # Insert right before the toolbar's trailing stretch, whatever index
        # that currently is (robust to HydraProfileViewer's own toolbar
        # layout, rather than a hardcoded position).
        insert_at = ptb.count() - 1
        ptb.insertWidget(insert_at, self._rad_btn)
        ptb.insertWidget(insert_at, self._rad_auto)
        ptb.insertWidget(insert_at, self._rad_r_bin)
        ptb.insertWidget(insert_at, QtWidgets.QLabel("  R bin:"))
        # R-bin edits already trigger each of the 5 cards' own auto-radial
        # (wired inside set_radial_controls); refresh the derived Composite
        # curve afterward since it depends on all 4 panels' fresh curves.
        self._rad_r_bin.valueChanged.connect(lambda *_: self._refresh_composite_curve())
        self._profile_view.compositeVisibilityChanged.connect(
            lambda *_: self._refresh_composite_curve())
        right.addWidget(self._profile_view)
        right.setStretchFactor(0, 3); right.setStretchFactor(1, 1)
        right.setMinimumWidth(320)
        split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 0); split.setStretchFactor(2, 1)
        split.setSizes([286, 361, 950])

    # ── Per-panel / composite frame sourcing ────────────────────────

    def _make_image_provider(self, key: str):
        """Each card's image_provider is bound once, at construction, to
        whichever panel/composite it represents — independent of which card
        is currently the *active* one (that only controls the viewer/plot
        binding, via set_viewer/set_profile_view in _on_panel_changed)."""
        if key == "composite":
            return lambda: self._composite_img
        n = int(key[2])
        return lambda: self._raw_frames.get(n)

    #: A panel number in a filename, e.g. "…_ge3.instr.txt" or "ge3/…".
    _PANEL_IN_NAME = re.compile(r"ge\s*([1-4])\b", re.IGNORECASE)

    @classmethod
    def _panel_for_file(cls, path) -> Optional[int]:
        """Which panel a calibration file belongs to, or None.

        Read from the *last* ``ge<n>`` in the full path, so a per-panel
        folder (``…/ge3/calib.txt``) and a per-panel filename
        (``…_ge3.instr.txt``) both work, and a run folder that happens to
        contain "ge1" does not out-vote the file's own name.
        """
        hits = cls._PANEL_IN_NAME.findall(str(path))
        return int(hits[-1]) if hits else None

    def _load_four_calibrations(self):
        """Pick up to four panel calibrations at once and apply each to its
        own card.

        Matched by filename rather than by pick order: a file dialog returns
        its selection sorted, not in the order they were clicked, so trusting
        order would silently cross-assign panels -- and a calibration on the
        wrong panel is a composite that looks plausible and is wrong.
        Anything unmatched is reported, never guessed at.
        """
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select calibration files for ge1–ge4",
            browse_start_dir(self._loader.data_path()
                             if hasattr(self._loader, "data_path") else ""),
            "Calibration (*.json *.poni *.txt);;All (*)")
        if not paths:
            return
        chosen: dict = {}
        unmatched, clashes = [], []
        for p in paths:
            n = self._panel_for_file(p)
            if n is None:
                unmatched.append(Path(p).name)
            elif n in chosen:
                clashes.append(f"ge{n}")
            else:
                chosen[n] = p
        found_by_name = self._fill_in_sibling_calibrations(chosen)
        for n, p in sorted(chosen.items()):
            card = self._cards.get(f"ge{n}")
            if card is not None:
                card.set_calib_path(p)
        bits = []
        if found_by_name:
            bits.append("matched by name: "
                        + ", ".join(f"ge{n}" for n in sorted(found_by_name)))
        if chosen:
            bits.append("loaded " + ", ".join(f"ge{n}" for n in sorted(chosen)))
        if unmatched:
            bits.append("no ge1–4 in: " + ", ".join(unmatched))
        if clashes:
            bits.append("more than one file for " + ", ".join(sorted(set(clashes)))
                        + " — kept the first")
        missing = [f"ge{n}" for n in (1, 2, 3, 4) if n not in chosen]
        if missing and chosen:
            bits.append("unchanged: " + ", ".join(missing))
        self._load4_lbl.setText("  ·  ".join(bits))

    def _fill_in_sibling_calibrations(self, chosen: dict) -> set:
        """Complete a partial pick from the files sitting next to it.

        The button promises four calibrations and then hands over a
        multi-select dialog, so picking one file and pressing Open loads one
        panel -- reported as "I loaded 4x calibration files and I still see
        only 1x set". Panel calibrations are written as a set and named
        alike (``..._ge1.instr.txt`` beside ``..._ge2.instr.txt``), so the
        other three are usually right there.

        Only ever fills panels the pick did not cover, and only from a file
        that actually exists on disk, so an explicit choice is never
        overridden and nothing is invented. Returns the panels filled in, to
        be named in the summary -- a calibration that arrived without being
        clicked should say so.
        """
        if not chosen:
            return set()
        filled = set()
        for n in (1, 2, 3, 4):
            if n in chosen:
                continue
            for have_n, have_p in sorted(chosen.items()):
                cand = self._PANEL_IN_NAME.sub(
                    lambda m, _n=n: m.group(0)[:-len(m.group(1))] + str(_n),
                    str(have_p))
                if cand != str(have_p) and Path(cand).exists():
                    chosen[n] = cand
                    filled.add(n)
                    break
        return filled

    def _default_save_path(self, key: str, suffix: str) -> str:
        """``<expid>_<data stem>_<panel><suffix>``, beside the data.

        Same rule as ``CalibrationTab._default_save_path`` and the Data
        Viewer's, with the panel key appended because these five cards
        describe five different detectors and would otherwise all propose
        the same filename.
        """
        from midas_gui.helpers import browse_start_dir
        parts = []
        try:
            provider = getattr(self, "_expid_provider", None)
            expid = (provider() or "").strip() if provider else ""
        except Exception:
            expid = ""
        if expid:
            parts.append(expid)
        data_path = self._loader.data_path() if hasattr(self._loader, "data_path") else ""
        if data_path:
            stem = Path(data_path).name.rsplit(".", 1)[0]
            if stem:
                parts.append(stem)
        parts.append(key)
        name = "_".join(parts) + suffix
        start = browse_start_dir(data_path) if data_path else ""
        return str(Path(start) / name) if start else name

    def _ensure_states_for_siblings(self, siblings: dict):
        for n in siblings:
            if n in self._states:
                continue
            st = hydra.DetectorState()
            st.load_default(n)
            self._states[n] = st
            fields = geometry_fields_from_file(str(hydra.default_param_file(n)))
            self._cards[f"ge{n}"].set_geometry(fields)

    def _apply_field_selectors(self, n: int, st: hydra.DetectorState):
        """Pull this panel's dark/bright/background arrays (if the
        corresponding HydraFieldSelector is checked and has computed one)
        out of the loader and onto its DetectorState."""
        st.dark = self._loader.dark(n)
        st.bright = self._loader.bright(n)
        st.bright_mode = self._loader.bright_mode()
        st.background = self._loader.background(n)
        st.proj_raw = self._loader.projected(n)

    def _load_panel_frame(self, n: int) -> Optional[np.ndarray]:
        path = self._loader.siblings().get(n)
        st = self._states.get(n)
        if st is None:
            return None
        if path is not None:
            st.data_file = path
        self._apply_field_selectors(n, st)
        try:
            if st.proj_raw is not None:
                # Already corrected + projected by ProjectionWorker.
                img = st.proj_raw
            else:
                if path is None:
                    return None
                img = _load_image(path, self._loader.dataset(), self._loader.frame_index())
                if st.dark is not None or st.bright is not None or st.background is not None:
                    # Same order as the single-detector tab: correction on the
                    # raw frame, before ImTransOpt (tab_view.py's _on_loader_data).
                    img = apply_field_corrections(img, dark=st.dark, bright=st.bright,
                                                  bright_mode=st.bright_mode,
                                                  background=st.background)
            img = _apply_im_trans(img, tuple(st.im_trans_opts))
            img = hydra.apply_panel_rotation(img, self._cards[f"ge{n}"].rotate_deg())
        except Exception:
            return None
        self._raw_frames[n] = img
        return img

    def _build_composite_if_needed(self):
        if self._composite_img is not None:
            return
        siblings = self._loader.siblings()
        active = {n: p for n, p in siblings.items() if n in self._states}
        if len(active) < 2:
            self._composite_img = None
            return
        for n, p in active.items():
            self._states[n].data_file = p
            self._apply_field_selectors(n, self._states[n])
        try:
            comp, big_det_size = hydra.build_windmill_composite(
                active, self._loader.frame_index(), self._loader.dataset(),
                self._states, op="max", step=COMPOSITE_DISPLAY_STEP)
        except Exception:
            self._composite_img = None
            return
        self._composite_img = comp
        self._big_det_size = big_det_size
        self._reseed_composite_card_if_needed(big_det_size, active)

    def _reseed_composite_card_if_needed(self, big_det_size: int, active_panels: dict):
        """Seed the composite card's beam centre at the canvas centre (the
        composite is registered so its own geometric centre IS BigDetSize/2)
        the first time this canvas size is seen.

        ``big_det_size`` is the canvas's FULL-RESOLUTION extent; the image
        actually built is decimated by ``COMPOSITE_DISPLAY_STEP``. The
        geometry seeded here describes the image, not the full canvas, so
        beam centre and NrPixels are in built pixels and the pixel size is
        scaled by the same factor -- the field of view is identical, which
        is what keeps this card's rings and radial integration correct at
        any step.

        Lsd is the mean across the
        contributing panels (they're all roughly the same sample-to-detector
        distance on a real Hydra rig); wavelength comes from whichever ge
        card was loaded first, since DetectorState itself has no wavelength
        (it's a ring-simulation-only parameter, not part of the compositing
        math). A user can still hand-edit the composite card afterward —
        this only fires once per distinct canvas size."""
        step = max(1, int(COMPOSITE_DISPLAY_STEP))
        n_out = (int(big_det_size) + step - 1) // step
        if not active_panels:
            return
        states = [self._states[n] for n in active_panels]
        # Keyed on the panels' actual geometry, not just the canvas size --
        # the same rule hydra._big_det_size_cache already uses, and for the
        # same reason it names: "loading a different calibration file for
        # one panel".
        #
        # Canvas size alone was wrong, and silently. Loading four panel
        # calibrations changes every panel's Lsd but usually not the canvas
        # extent, so the composite kept whatever it was seeded with partway
        # through that load. Reported from 1-ID-E WAXS: the composite card
        # read Lsd 3071.610 mm, which is exactly the mean of ge1's real
        # 2392.223 and three bundled ~3298 defaults -- it had been seeded
        # after ge1 landed and never revisited. Every simulated ring then
        # sat at the wrong radius and the integrated 2-theta axis was scaled
        # by 28%, with nothing on screen saying so.
        seed_key = (n_out, round(sum(s.lsd for s in states) / len(states), 3),
                    round(states[0].px, 4),
                    tuple(sorted((n, round(self._states[n].lsd, 3))
                                 for n in active_panels)))
        if self._composite_seeded_key == seed_key:
            return
        self._composite_seeded_key = seed_key
        lsd = sum(s.lsd for s in states) / len(states)
        px = states[0].px * step          # a built pixel spans `step` detector pixels
        wl = 0.172973
        for n in active_panels:
            g = self._cards[f"ge{n}"].get_geometry()
            if g.get("wavelength_A"):
                wl = g["wavelength_A"]
                break
        half = n_out / 2.0
        self._cards["composite"].set_geometry({
            "wavelength_A": wl, "pxY": px, "Lsd": lsd,
            "BC_y": half, "BC_z": half, "tx": 0.0, "ty": 0.0, "tz": 0.0,
            "NrPixelsY": n_out, "NrPixelsZ": n_out,
            "distortion": {}, "im_trans": [],
        })

    # ── Signal handlers ──────────────────────────────────────────

    def _on_siblings_changed(self, siblings: dict):
        self._ensure_states_for_siblings(siblings)
        if siblings:
            source_key = f"ge{min(siblings)}"
            detected = self._loader.detected_geometry()
            if detected:
                self._cards[source_key].apply_shared_fields(detected)
            self._sync_shared_fields(source_key)
        self._toolbar.set_available(siblings.keys())
        self._composite_img = None
        self._composite_seeded_key = None
        self._refresh_display()
        self._refresh_profile_curves()

    def _on_frame_changed(self, _idx: int):
        self._composite_img = None
        self._refresh_display()
        self._refresh_profile_curves()

    def _on_fields_changed(self):
        """Dark/bright/background changed (any panel) — recompute the
        corrected frames/composite and refresh, same as a frame change."""
        self._composite_img = None
        self._refresh_display()
        self._refresh_profile_curves()

    def _on_panel_changed(self, key: str):
        """Only the image viewer (one image at a time) rebinds when the
        active panel changes — the profile-plot/radial-controls bindings
        are permanent (set once per card in _build_ui) so all 5 curves stay
        live regardless of which panel is currently displayed."""
        if self._active_card is not None:
            self._active_card.set_viewer(None)
        self._card_stack.setCurrentWidget(self._cards[key])
        self._active_card = self._cards[key]
        self._active_card.set_viewer(self._viewer)
        self._refresh_display()

    def _on_card_geometry_changed(self, n: int):
        """A ge1-4 card's geometry changed (BC edit/pick, calibration file
        load, or a shared-field edit) — mirror λ/max2θ/px onto the other
        panels + composite, sync the full geometry into the matching
        DetectorState, force the composite (which depends on every panel's
        geometry) to rebuild, and refresh the derived Composite curve (that
        panel's own curve already refreshed itself via the card's normal
        geometry-change handling)."""
        card = self._cards.get(f"ge{n}")
        if card is None:
            return
        if not self._syncing_shared:
            self._sync_shared_fields(f"ge{n}")
        fields = card.get_full_geometry()
        if fields is None:
            return
        st = self._states.setdefault(n, hydra.DetectorState())
        st.load_from_geometry_dict(fields)
        self._composite_img = None
        if self._toolbar.current() in ("composite", f"ge{n}"):
            self._refresh_display()
        self._refresh_composite_curve()

    def _on_composite_geometry_changed(self):
        """The Composite card's own geometry changed — only λ/max2θ/px are
        mirrored back out to ge1-4 (its Lsd/BC/tx stay independent, seeded
        once from the panels — see _reseed_composite_card_if_needed)."""
        if not self._syncing_shared:
            self._sync_shared_fields("composite")

    _SHARED_FIELD_KEYS = ("ge1", "ge2", "ge3", "ge4", "composite")

    def _sync_shared_fields(self, source_key: str):
        """Mirror λ/max2θ/px from the card at `source_key` onto every other
        card (ge1-4 + composite) — see DetectorGeometryCard.get_shared_fields/
        apply_shared_fields. Guarded by `_syncing_shared` so applying the
        value to a sibling (which itself emits geometryChanged) doesn't
        recurse into another sync."""
        src = self._cards.get(source_key)
        if src is None:
            return
        shared = src.get_shared_fields()
        # Pixel size is NOT shared with the composite in either direction.
        # The composite canvas is decimated by COMPOSITE_DISPLAY_STEP and its
        # card's px is deliberately panel_px * step so that NrPixels * px --
        # the physical extent -- stays the same (see
        # _reseed_composite_card_if_needed). Mirroring that number onto the
        # panels tells each one its pixels are `step` times larger than they
        # are; mirroring a panel's back over it undoes the scaling the
        # decimated canvas depends on. Wavelength and max 2theta are genuinely
        # common to every panel, so those still propagate.
        #
        # This was latent until the panel cards' own geometry fields started
        # being read (DetectorGeometryCard._export_geom no longer returns the
        # loaded calibration wholesale), at which point the inflated px
        # reached each panel's DetectorState and moved the composite.
        shared_no_px = {k: v for k, v in shared.items() if k != "pxY"}
        self._syncing_shared = True
        try:
            for key in self._SHARED_FIELD_KEYS:
                if key == source_key:
                    continue
                card = self._cards.get(key)
                if card is not None:
                    card.apply_shared_fields(
                        shared_no_px
                        if "composite" in (key, source_key) else shared)
        finally:
            self._syncing_shared = False

    def _refresh_profile_curves(self, force: bool = False):
        """(Re)load every available panel's frame and — if Auto is on (or
        `force`, from the Integrate button) — reintegrate it, independent
        of which panel is currently the *active* one for image display.
        Then refresh the derived Composite curve."""
        siblings = self._loader.siblings()
        for n in (1, 2, 3, 4):
            if n not in siblings:
                self._profile_view.clear_curve(f"ge{n}")
                continue
            img = self._load_panel_frame(n)
            card = self._cards[f"ge{n}"]
            if img is not None and (force or self._rad_auto.isChecked()):
                card.radial_integrate()
        self._refresh_composite_curve()

    def _refresh_composite_curve(self):
        """Recompute the toggleable "Composite" curve: each available
        panel's own (already-computed) profile, converted to a shared
        2theta axis, resampled onto one common grid, and NaN-aware summed —
        not a radial integration of the composited image (that would
        double-count any panel overlap and mix registration error into the
        profile). Deliberately reads back the per-panel curves already
        pushed into HydraProfileViewer rather than recomputing them here."""
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
        # Push through the normal set_curve(r_px, ...) API using the first
        # contributing panel's geometry as an arbitrary-but-consistent
        # reference — round-trips correctly for any X-axis unit selection
        # since it's the exact inverse of the R -> 2theta conversion above.
        ref_lsd, ref_px, ref_wl = natives[0][2], natives[0][3], natives[0][4]
        r_ref = ref_lsd * np.tan(np.radians(common)) / ref_px
        self._profile_view.set_curve("composite", r_ref, summed,
                                     lsd_um=ref_lsd, px_um=ref_px, wavelength_A=ref_wl)

    def _refresh_display(self):
        key = self._toolbar.current()
        if key == "composite":
            self._build_composite_if_needed()
            img = self._composite_img
        else:
            img = self._load_panel_frame(int(key[2]))
        if img is None:
            return
        disp_key = (img.shape, key)
        fresh = (self._disp_key != disp_key)
        self._disp_key = disp_key
        self._viewer.set_image(img, autorange=fresh)
        if self._active_card is not None:
            self._active_card.refresh_rings_and_radial()
        # The compass is anchored to the shown detector's beam centre, so a
        # panel switch moves it even when nothing about the geometry changed.
        self._redraw_lab_axes_if_on()

    # ── Lab-frame axes overlay ───────────────────────────
    # Port of the Data Viewer's overlay (tab_view._draw_lab_axes). The one
    # difference is the anchor: this page shows a different detector
    # depending on the toolbar, so the compass follows whichever card is
    # active rather than a single geometry card.

    def _on_lab_axes_toggled(self, checked: bool):
        if checked:
            self._draw_lab_axes()
        else:
            self._clear_lab_axes()

    def _redraw_lab_axes_if_on(self, *_args):
        if getattr(self, "_lab_axes_on", None) is not None \
                and self._lab_axes_on.isChecked():
            self._draw_lab_axes()

    def _clear_lab_axes(self):
        for it in self._axis_items:
            self._viewer._iv.removeItem(it)
        self._axis_items.clear()

    def _draw_lab_axes(self):
        self._clear_lab_axes()
        img = getattr(self._viewer, "_data", None)
        card = self._cards.get(self._toolbar.current())
        if img is None or card is None:
            return
        geo = card.get_geometry() or {}
        bcy, bcz = geo.get("BC_y"), geo.get("BC_z")
        if bcy is None or bcz is None:
            return
        items = build_lab_frame_axes_items(
            self._viewer._iv, img.shape, float(bcy), float(bcz))
        for it in items:
            self._viewer._iv.addItem(it)
        self._axis_items.extend(items)

    # ── Export (Calibrate tab's Hydra "← Data Viewer" import) ───────

    def export_for_calibration(self) -> dict:
        """Anchor path + each present panel's full geometry (BC/Lsd/tilts/
        transforms) — consumed by ``HydraCalibrationPage.import_from_viewer``."""
        siblings = self._loader.siblings()
        return {
            "anchor_path": self._loader.current_path(),
            "geometries": {n: self._cards[f"ge{n}"].get_full_geometry()
                          for n in siblings},
        }

    # ── GUI state ────────────────────────────────────────────────

    def get_state(self) -> dict:
        cards = {}
        for key, card in self._cards.items():
            fields = widgets_to_dict(card.state_widgets())
            fields["materials"] = card.materials_state()
            cards[key] = fields
        return {
            "anchor_path": self._loader.current_path(),
            "active_panel": self._toolbar.current(),
            "rad_r_bin": self._rad_r_bin.value(),
            "rad_auto": self._rad_auto.isChecked(),
            "viewer": self._viewer.display_state(),
            "cards": cards,
        }

    def set_state(self, state: dict):
        if not state:
            return
        self._viewer.set_display_state(state.get("viewer"))
        self._origin_btn.sync()
        for key, fields in (state.get("cards") or {}).items():
            card = self._cards.get(key)
            if card is None:
                continue
            calib_path = fields.get("calib_ed")
            if calib_path:
                card.set_calib_path(calib_path)
            materials = fields.get("materials")
            if materials:
                card.set_materials(materials)
            apply_dict_to_widgets(card.state_widgets(), fields)
            card.refresh_geometry()
        self._rad_r_bin.setValue(state.get("rad_r_bin", self._rad_r_bin.value()))
        self._rad_auto.setChecked(state.get("rad_auto", self._rad_auto.isChecked()))
        anchor = state.get("anchor_path")
        if anchor and Path(anchor).exists():
            self._loader.set_path(anchor)
        panel = state.get("active_panel")
        if panel:
            self._toolbar.set_current(panel)
