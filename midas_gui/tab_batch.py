"""Tab 3 — Batch Integrate.

Ports the v3 batch tab and adds Phase-1 features:
  - kernel selector (hard / subpixel K=2/4 / polygon)
  - physics corrections (polarization + solid angle)
  - per-bin variance / σ output (error model selectable)
  - native Q-uniform binning
  - output formats CSV / XYE / FXYE / DAT / HDF5
"""
from __future__ import annotations

import os
import sys
import datetime as _dt
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui.constants import (KERNELS, ERROR_MODELS,
                           DEFAULT_NICKEL_DIR, DEFAULT_KERNEL,
                           DEFAULT_ERROR_MODEL)
from midas_gui.helpers import (_fspin, _browse, _build_spec, spec_from_geometry_file,
                               geometry_fields_from_file,
                               resolve_calibration_fields, full_calibration_snapshot,
                               pixel_readout_text,
                               collapse_cake_eta,
                               make_calib_values_button,
                               rmax_corner_px, rmax_edge_px, draw_polar_bin_overlay,
                               _NoScrollSpinBox, _NoScrollComboBox,
                               widgets_to_dict, apply_dict_to_widgets,
                               check_output_dir_writable,
                               suggest_integration_output_dir,
                               suggest_working_dir, bc_path_parts,
                               browse_start_dir, warn_if_path_missing,
                               is_h5, list_h5_1d_datasets)
from midas_gui.widgets import (LogPanel, CorrectionFlagsWidget, WaterfallViewer,
                               StackedProfileViewer, DataLoaderPanel, OutputFormatSelector,
                               ImageViewer, OriginToolButton, build_lab_frame_axes_items,
                               CakeStackViewer)
from midas_gui.workers import (BatchWorker, BatchRunCoordinator, apply_q_uniform,
                               DriftWorker, FolderMonitorWorker, write_all_profiles)
from midas_gui.dialogs import show_error
from midas_gui.hydra_widgets import HydraModeRibbon
from midas_gui.hydra_batch_page import HydraBatchPage
from midas_gui.job_queue import JobQueuePanel
from midas_gui.cake_params import (parse_cake_csv, write_cake_csv,
                                   omega_for_window)
from midas_gui import project
from midas_gui import run_log
from midas_gui import settings
from midas_gui import style as S


class _RadialBinsDialog(QtWidgets.QDialog):
    """Δ/min/max for one radial-axis mode (R in px, or Q in Å⁻¹), opened
    from Batch Integrate's mode-dependent "R bins…"/"Q bins…" button next to
    the Bin type dropdown. Hosts the tab's own spinboxes directly (not
    copies) — this is a relocated view of the same widgets ``_build_spec``
    and GUI-state save/restore already use, not a separate value store.
    Only the R variant takes Corner/Edge presets; Q has no detector-geometry
    equivalent."""

    def __init__(self, mode: str, bin_spin, min_spin, max_spin,
                 corner_btn=None, edge_btn=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Radial (R) bins" if mode == "R" else "Q bins")
        v = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "R bin/Rmin/Rmax define the underlying integration grid, in "
            "detector pixels. Rmax 0 = auto (farthest detector corner from "
            "the beam centre)." if mode == "R" else
            "Bin uniformly in Q (Å⁻¹) instead of R, for OUTPUT only — the "
            "same radial axis, alternate units. The Radial (R) bins still "
            "set the underlying integration grid; this rebins that result "
            "into Q.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{S.MUTED};font-size:10px;padding-bottom:4px")
        v.addWidget(note)
        form = S.Form()
        if corner_btn is not None:
            preset_row = QtWidgets.QHBoxLayout(); preset_row.setSpacing(4)
            preset_row.addWidget(QtWidgets.QLabel("Rmax presets:"))
            preset_row.addWidget(corner_btn); preset_row.addWidget(edge_btn)
            preset_row.addStretch(1)
            form.full(preset_row)
        form.row((f"{mode} bin:", bin_spin))
        form.row((f"{mode}min:", min_spin))
        form.row((f"{mode}max:", max_spin))
        v.addLayout(form)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.accept); btns.accepted.connect(self.accept)
        v.addWidget(btns)


class _AzimuthalBinsDialog(QtWidgets.QDialog):
    """Δ/min/max for the azimuthal (η) integration axis, opened from Batch
    Integrate's "Azimuthal bins…" button. Hosts the tab's own η-bin/min/max
    spinboxes directly (not copies) — see ``_RadialBinsDialog``."""

    def __init__(self, bin_spin, min_spin, max_spin, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Azimuthal (η) bins")
        v = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "η bin controls how the 2-D (η, R) cake is collapsed to a 1-D "
            "profile (see Azim. mean) and, when Multi-azimuth output is on, "
            "defines the output sectors themselves.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{S.MUTED};font-size:10px;padding-bottom:4px")
        v.addWidget(note)
        form = S.Form()
        form.row(("η bin:", bin_spin))
        form.row(("η min:", min_spin))
        form.row(("η max:", max_spin))
        v.addLayout(form)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.accept); btns.accepted.connect(self.accept)
        v.addWidget(btns)


class _CakeParamsDialog(QtWidgets.QDialog):
    """Every column of an mpe_wf_saxs_waxs ``cake_parameters`` CSV in one
    place, with Load and Save — the counterpart of that project's own
    cake-parameter window (``archive/gui_config_cake_params.py``), whose file
    format ``cake_params.parse_cake_csv``/``write_cake_csv`` read and write.

    The same values are also reachable from "R bins…", "Azimuthal bins…" and
    the loader's Combine sub-frames, and those stay: mid-run you want the one
    axis you are adjusting, not all nine. This is the other view — the file as
    a file.

    Unlike those two dialogs, this one keeps its OWN spinboxes and copies
    values in and out. They can host the tab's real widgets because each such
    widget appears in exactly one of them; a widget has one parent, so adding
    ``_r_min`` here as well would silently reparent it out of
    ``_RadialBinsDialog``. Copying also gives Apply something to mean: open
    this, try numbers, close, and the next run is untouched.
    """

    # (CSV key, row label, BatchTab attribute). OME_SUM is the exception —
    # it lives on the loader, not the tab, so _target() resolves it by hand.
    SPEC = (
        ("R_MIN",     "R_MIN  (R min)",                "_r_min"),
        ("R_MAX",     "R_MAX  (R max)",                "_r_max"),
        ("R_STEP",    "R_STEP  (R bin)",               "_r_bin"),
        ("ETA_MIN",   "ETA_MIN  (η min)",              "_eta_min"),
        ("ETA_MAX",   "ETA_MAX  (η max)",              "_eta_max"),
        ("ETA_STEP",  "ETA_STEP  (η bin)",             "_e_bin"),
        ("OME_SUM",   "OME_SUM  (Combine sub-frames)", None),
        ("OME_START", "OME_START  (ω of raw frame 0)", "_ome_start"),
        ("OME_STEP",  "OME_STEP  (ω per raw sub-frame)", "_ome_step"),
    )

    # Two controls that are NOT CSV columns and must stay out of SPEC — the
    # file mpe_wf reads has exactly nine, and
    # test_the_dialog_covers_every_csv_column_in_order is what keeps that
    # true. (BatchTab attribute, row label, tooltip.)
    EXTRA = (
        ("_ome_channel", "Omega channel:",
         "A 1-D dataset in the source HDF5 holding the measured rotation, "
         "read per frame instead of the OME_START/OME_STEP ramp. Blank "
         "falls back to OME_START/OME_STEP."),
        ("_ome_collapse", "Averaged / summed:",
         "These images were averaged or summed outside this app, so the "
         "per-frame ramp does not describe them — give every frame the one "
         "mean ω of the whole run instead."),
    )

    def __init__(self, tab, parent=None):
        super().__init__(parent or tab)
        self._tab = tab
        self.setWindowTitle("Cake parameters")
        v = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "The nine columns of an mpe_wf cake_parameters CSV. The first "
            "seven are this tab's own R/η binning plus the loader's Combine "
            "sub-frames, gathered here — editing them here or in "
            "'R bins…'/'Azimuthal bins…' is the same setting either way. "
            "OME_START/OME_STEP are the rotation: ω of raw sub-frame 0, and "
            "the increment per raw sub-frame. Each integrated frame is "
            "stamped with the ω at the middle of the raw frames it was built "
            "from, and that angle is written to the zarr and the combined "
            "HDF5. Both zero means every frame is recorded at ω = 0°.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{S.MUTED};font-size:10px;padding-bottom:4px")
        v.addWidget(note)

        self._spins: dict = {}
        form = S.Form()
        for key, label, _attr in self.SPEC:
            spin = self._mirror(self._target(key))
            self._spins[key] = spin
            form.row((label + ":", spin))
        # The two non-CSV omega controls, below a separator so the nine
        # columns above still read as "the file".
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet(f"color:{S.MUTED}")
        form.full(line)
        self._ome_channel = QtWidgets.QComboBox()
        self._ome_channel.setEditable(True)
        # Without this the popup is only as wide as the closed combo and
        # Qt elides the MIDDLE of every entry — "instrum...R/samRy" — which
        # is exactly the part that distinguishes one container from
        # another (.../D/HR/ vs .../E/HL/). See _widen_popup.
        self._ome_channel.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._ome_collapse = QtWidgets.QCheckBox("one ω for all frames")
        self._extras = {"_ome_channel": self._ome_channel,
                        "_ome_collapse": self._ome_collapse}
        for attr, label, tip in self.EXTRA:
            w = self._extras[attr]
            w.setToolTip(tip)
            form.row((label, w))
        v.addLayout(form)

        btns = QtWidgets.QDialogButtonBox()
        load_btn = btns.addButton("Load CSV…", QtWidgets.QDialogButtonBox.ActionRole)
        save_btn = btns.addButton("Save CSV…", QtWidgets.QDialogButtonBox.ActionRole)
        apply_btn = btns.addButton(QtWidgets.QDialogButtonBox.Apply)
        close_btn = btns.addButton(QtWidgets.QDialogButtonBox.Close)
        load_btn.setToolTip("Read a cake_parameters CSV into these fields "
                            "(header row + last data row).")
        save_btn.setToolTip("Write these nine values as a cake_parameters CSV, "
                            "in the layout mpe_wf's own tools read.")
        load_btn.clicked.connect(self._on_load)
        save_btn.clicked.connect(self._on_save)
        apply_btn.clicked.connect(self.apply_to_tab)
        close_btn.clicked.connect(self.accept)
        v.addWidget(btns)

    @staticmethod
    def _widen_popup(combo, cap: int = 900) -> None:
        """Size the drop-down list to its longest entry.

        A QComboBox popup inherits the width of the closed box, and Qt
        elides anything longer — in the middle, which for an HDF5 path is
        the only part that differs between entries
        (``instrument/SMS/D/HR/samRy`` against
        ``instrument/SMS/E/HL/samRy`` both render as
        ``instrum...R/samRy``). Widen the VIEW rather than the combo so the
        form layout is unaffected, and cap it so a pathological name can't
        push the popup off-screen; entries longer than the cap keep their
        full text as a tooltip.
        """
        fm = combo.fontMetrics()
        widest = max((fm.boundingRect(combo.itemText(i)).width()
                      for i in range(combo.count())), default=0)
        if widest:
            combo.view().setMinimumWidth(min(widest + 40, cap))

    # ── which widget each key really lives in ──────────────────────────
    def _target(self, key):
        """The tab widget this key is a view of, or None when there isn't one
        — OME_SUM needs a loader with a Combine sub-frames spin, which not
        every source mode has."""
        if key == "OME_SUM":
            return getattr(self._tab._loader, "_combine_chunk", None)
        attr = next((a for k, _l, a in self.SPEC if k == key), None)
        return getattr(self._tab, attr, None) if attr else None

    @staticmethod
    def _mirror(target):
        """A spinbox matching the range and precision of the field it stands
        in for, so this dialog can't accept a value the tab would silently
        clamp on the way back."""
        if isinstance(target, QtWidgets.QSpinBox):
            s = _NoScrollSpinBox()
            s.setRange(target.minimum(), target.maximum())
            s.setSingleStep(target.singleStep())
            s.setSuffix(target.suffix())
            return s
        s = _fspin(-1e9, 1e9, 4, 0.0)
        if isinstance(target, QtWidgets.QDoubleSpinBox):
            s.setRange(target.minimum(), target.maximum())
            s.setDecimals(target.decimals())
            s.setSuffix(target.suffix())
        return s

    # ── copy in / copy out ─────────────────────────────────────────────
    def load_from_tab(self) -> None:
        for key, spin in self._spins.items():
            target = self._target(key)
            if target is not None:
                spin.setValue(target.value())
            # A key with nowhere to go is shown greyed rather than hidden —
            # it is still a column this dialog will write.
            spin.setEnabled(target is not None)
        self._refresh_channels()
        self._ome_channel.setEditText(self._tab._ome_channel.currentText())
        self._ome_collapse.setChecked(self._tab._ome_collapse.isChecked())

    def _refresh_channels(self) -> None:
        """Offer the 1-D datasets of the loaded HDF5, omega-ish names first.

        Rebuilt on every open because the loaded source changes underneath
        this dialog. The combo stays editable and the current text is
        preserved across the rebuild, so a path typed by hand — or one from
        a file that isn't loaded right now — survives; the list is a
        convenience, not the set of legal answers."""
        typed = self._ome_channel.currentText()
        self._ome_channel.blockSignals(True)
        try:
            self._ome_channel.clear()
            self._ome_channel.addItem("")   # blank = use OME_START/OME_STEP
            cfg = {}
            try:
                cfg = self._tab._loader.source_cfg() or {}
            except Exception:
                pass
            path = cfg.get("path")
            if path and is_h5(path):
                try:
                    for name, _n in list_h5_1d_datasets(path):
                        self._ome_channel.addItem(name)
                        self._ome_channel.setItemData(
                            self._ome_channel.count() - 1, name,
                            QtCore.Qt.ToolTipRole)
                except Exception:
                    pass
            self._widen_popup(self._ome_channel)
            self._ome_channel.setEditText(typed)
        finally:
            self._ome_channel.blockSignals(False)

    def apply_to_tab(self) -> None:
        for key, spin in self._spins.items():
            target = self._target(key)
            if target is None:
                continue
            if isinstance(target, QtWidgets.QSpinBox):
                target.setValue(int(round(spin.value())))
            else:
                target.setValue(float(spin.value()))
        self._tab._ome_channel.setEditText(self._ome_channel.currentText().strip())
        self._tab._ome_collapse.setChecked(self._ome_collapse.isChecked())
        self._tab._refresh_cake_summary()

    def values(self) -> dict:
        return {k: float(s.value()) for k, s in self._spins.items()}

    def showEvent(self, ev):
        # Re-read every time rather than once at construction: the fields move
        # from the two bins dialogs, a project restore, or an Rmax auto-fill.
        self.load_from_tab()
        super().showEvent(ev)

    # ── the two file buttons ───────────────────────────────────────────
    def _on_load(self) -> None:
        # Deliberately the tab's own loader, so the dialog's Load and the
        # pre-existing code path are one function with one set of rules.
        self._tab._load_cake_csv()
        self.load_from_tab()

    def _on_save(self) -> None:
        # Apply first: a file that disagrees with the tab it was saved from is
        # worse than no file at all.
        self.apply_to_tab()
        default = self._tab._suggest_cake_csv_path()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save cake parameters CSV", str(default), "CSV (*.csv);;All (*)")
        if not path:
            return
        try:
            write_cake_csv(path, self.values())
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Cake CSV", f"Could not write:\n{path}\n\n{e}")
            return
        self._tab._log.append(f"[batch] Saved cake parameters to {path}")


class BatchTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._orphans: list = []       # aborted workers kept alive until they wind down
        self._drift_worker = None
        self._drift_traj = None
        self._calib_result = None
        # Geometry behind the Detector-view pixel readout, resolved lazily and
        # dropped by _refresh_detector_preview; see _radial_readout.
        self._readout_geom = None
        self._wf_started = False
        self._monitor_worker = None
        self._integrated_fids: set = set()   # frame ids already displayed (batch + monitor)
        self._geom_cache = None              # cached integration context (detector map)
        self._geom_sig = None                # signature the cached context was built for
        self._bin_overlay_items: list = []   # Rmin/Rmax + bin-grid overlay on _det_view
        self._axis_items: list = []          # Lab-frame axes overlay on _det_view
        self._last_shape_mismatch_logged: Optional[tuple] = None
        self._last_calib_fail_note: Optional[str] = None
        # (shape, im_trans) the Detector view is currently framed for — see
        # _refresh_detector_preview for why the view is only re-framed when
        # this changes.
        self._det_view_framed_for: Optional[tuple] = None
        # Built lazily on first switch to Hydra mode: it owns 8 pyqtgraph
        # widgets (4 WaterfallViewer + 4 StackedProfileViewer), and most
        # sessions never touch Hydra Batch Integrate — see .context/DECISIONS.md's
        # pyqtgraph interpreter-teardown / widget-count crash-risk entry.
        self._hydra_page: Optional[HydraBatchPage] = None
        self._hydra_registry = None          # DataSourceRegistry, set by bind_hydra_registry()
        self._hydra_registry_label = ""
        self._last_run_inputs: dict = {}
        self._last_run_fields: dict = {}
        self._last_results: Optional[dict] = None   # for the Save button — see _on_done
        self._last_axis_ctx: Optional[tuple] = None  # (lsd, px, wl) for the Save button
        self._last_spec = None       # IntegrationSpec for the Save button — see _save_results
        self._last_kernel = None
        self._last_weighted = None
        self._last_run_out_dir: Optional[str] = None  # set in _run() — see _on_done
        self._expid_provider = None  # () -> str, wired by app.py's MainWindow — see set_expid_provider
        self._cake_dialog = None  # built on first use — see _open_cake_params_dialog
        # Raw sub-frame windows of the loaded source, for the ω readouts —
        # see _recompute_omega_span (refreshed on source change only).
        self._omega_span: Optional[dict] = None
        # The source_cfg the cached span was built from, so a repeat signal
        # for an unchanged source costs nothing (see _recompute_omega_span).
        self._omega_span_key: Optional[str] = None
        # Coalesces bursts: a folder pick walks every file's HDF5 header, and
        # dataChanged/valueChanged can fire several times for one user
        # action (and once per step while a spin box is held down). Without
        # this the GUI thread would repeat that walk for each of them.
        self._omega_span_timer = QtCore.QTimer(self)
        self._omega_span_timer.setSingleShot(True)
        self._omega_span_timer.setInterval(150)
        self._omega_span_timer.timeout.connect(self._recompute_omega_span)
        self._project_ctx: Optional[project.ProjectContext] = None
        self._build_ui()
        self._loader.monitorToggled.connect(self._toggle_monitor)
        self._loader.dataChanged.connect(self._refresh_detector_preview)
        self._loader.dataChanged.connect(self._maybe_autofill_output_dir)
        # The ω readouts follow the data: a new source changes which raw
        # sub-frames exist, and therefore which angles they map to.
        self._loader.dataChanged.connect(self._schedule_omega_span)
        # …and the loader's own range hint gets the angular half of its
        # range from us — it counts sub-frames and knows nothing about ω.
        self._loader.set_omega_hint_fn(self._omega_hint_tail)
        self._loader.fieldsChanged.connect(self._refresh_detector_preview)
        # "stream" mode's preview frame is fetched off the GUI thread (see
        # DataLoaderPanel._start_preview_worker) — dataChanged/fieldsChanged
        # above just kick that background read off; this is what actually
        # re-draws the Detector view once the real frame lands.
        self._loader.previewFrameReady.connect(self._refresh_detector_preview)
        self._use_tab2_btn.toggled.connect(self._refresh_detector_preview)
        self._json_ed.textChanged.connect(lambda *_: self._refresh_detector_preview())
        self._use_tab2_btn.toggled.connect(self._update_calib_src_enabled)
        self._update_calib_src_enabled()
        self._loader.set_path(DEFAULT_NICKEL_DIR)

    def set_project_context(self, ctx: "project.ProjectContext"):
        self._project_ctx = ctx
        if self._hydra_page is not None:
            self._hydra_page.set_project_context(ctx)

    def set_calibration(self, result):
        self._apply_calib_result(result)
        self._use_tab2_btn.setChecked(True)
        self._refresh_detector_preview()

    def _apply_calib_result(self, result):
        self._calib_result = result
        self._calib_src_lbl.setText(
            f"From Tab 2: Lsd={result.Lsd/1000:.3f} mm  "
            f"λ={result.wavelength_A:.5f} Å  {result.NrPixelsY}×{result.NrPixelsZ} px")

    def _update_calib_src_enabled(self):
        """Grey out the calibration-file field/browse button while "From Tab
        2" is selected — they only apply to the "From file" source."""
        from_file = not self._use_tab2_btn.isChecked()
        self._json_ed.setEnabled(from_file)
        self._json_browse_btn.setEnabled(from_file)

    def _calib_fields_in_use(self):
        """Resolve the geometry currently selected (Tab-2 result or file), as a
        dict of display fields — or (None, note) if unavailable. Also backs
        the "View calibration" popup (see helpers.make_calib_values_button),
        called fresh each time it's opened."""
        return resolve_calibration_fields(
            self._calib_result, self._use_json_btn.isChecked(), self._json_ed.text())

    def _apply_rmax_preset(self, formula) -> None:
        """Corner/Edge button handler — ``formula`` is ``rmax_corner_px`` or
        ``rmax_edge_px``; resolves the active calibration and fills Rmax."""
        fields, note = self._calib_fields_in_use()
        if not fields or fields.get("BC_y") is None or fields.get("NrPixelsY") is None:
            self._log.append(f"[batch] Can't set Rmax preset: {note}")
            return
        value = formula(fields["BC_y"], fields["BC_z"], fields["NrPixelsY"], fields["NrPixelsZ"])
        self._r_max.setValue(value)

    def _q_mode_active(self) -> bool:
        """True when the Bin type dropdown selects Q (rebin the OUTPUT
        uniformly in Q, not the underlying R-uniform integration grid)."""
        return self._bin_type.currentData() == "Q"

    def _on_bin_type_changed(self, *_args) -> None:
        self._radial_bins_btn.setText("Q bins…" if self._q_mode_active() else "R bins…")

    def _open_radial_bins_dialog(self) -> None:
        (self._q_bins_dialog if self._q_mode_active() else self._r_bins_dialog).exec_()

    def _load_cake_csv(self) -> None:
        """"Load cake parameters CSV…" button — applies R_MIN/R_MAX/R_STEP/
        ETA_MIN/ETA_MAX/ETA_STEP/OME_SUM from an mpe_wf_saxs_waxs-style
        cake_parameters CSV (see cake_params.parse_cake_csv) to the matching
        fields. OME_START/OME_STEP land in the two invisible spins, from
        where they become each frame's recorded rotation angle at run time
        (see ``_omega_cfg`` and ``cake_params.omega_for_window``).

        Reached both from ``_CakeParamsDialog``'s Load button and, before that
        dialog existed, directly from the calibration card — one function, so
        both routes obey the same rules."""
        path = _browse(self, "Open cake parameters CSV", "CSV (*.csv);;All (*)")
        if not path:
            return
        values = parse_cake_csv(path)
        if not values:
            QtWidgets.QMessageBox.warning(
                self, "Cake CSV", f"Could not parse any values from:\n{path}")
            return
        applied = []
        if "R_MIN" in values:
            self._r_min.setValue(values["R_MIN"])
            applied.append(f"R_MIN={values['R_MIN']:g}")
        if "R_MAX" in values:
            self._r_max.setValue(values["R_MAX"])
            applied.append(f"R_MAX={values['R_MAX']:g}")
        if "R_STEP" in values:
            self._r_bin.setValue(values["R_STEP"])
            applied.append(f"R_STEP={values['R_STEP']:g}")
        if "ETA_MIN" in values:
            self._eta_min.setValue(values["ETA_MIN"])
            applied.append(f"ETA_MIN={values['ETA_MIN']:g}")
        if "ETA_MAX" in values:
            self._eta_max.setValue(values["ETA_MAX"])
            applied.append(f"ETA_MAX={values['ETA_MAX']:g}")
        if "ETA_STEP" in values:
            self._e_bin.setValue(values["ETA_STEP"])
            applied.append(f"ETA_STEP={values['ETA_STEP']:g}")
        if "OME_SUM" in values and hasattr(self._loader, "_combine_chunk"):
            self._loader._combine_chunk.setValue(int(values["OME_SUM"]))
            applied.append(f"OME_SUM={int(values['OME_SUM'])} -> Combine sub-frames")
        if "OME_START" in values:
            self._ome_start.setValue(values["OME_START"])
            applied.append(f"OME_START={values['OME_START']:g}°")
        if "OME_STEP" in values:
            self._ome_step.setValue(values["OME_STEP"])
            applied.append(f"OME_STEP={values['OME_STEP']:g}°/sub-frame")
        self._log.append(f"[batch] Loaded cake parameters from {path}: {', '.join(applied) or '(nothing recognized)'}")
        self._refresh_cake_summary()

    def _open_cake_params_dialog(self) -> None:
        """"Cake parameters…" — built on first use and kept, like the two bins
        dialogs, so it holds its position on screen between openings. It
        re-reads the tab in ``showEvent``, so a stale instance can't show
        stale numbers."""
        if self._cake_dialog is None:
            self._cake_dialog = _CakeParamsDialog(self, parent=self)
        self._cake_dialog.show()
        self._cake_dialog.raise_()
        self._cake_dialog.activateWindow()

    def _suggest_cake_csv_path(self) -> Path:
        """Where a cake_parameters CSV would go if the mpe_wf convention were
        followed — ``<expid>_bc/cake_parameters.<beamline>.<detector>.csv``.

        A *suggestion* for the Save-As dialog, nothing more. mpe_wf's own
        editor writes this path silently, but it runs as ``S20IDUSER`` and we
        do not; putting the same path in a file dialog shows the user where
        their file is about to land in a shared beamline tree before it lands
        there. Pure path arithmetic — nothing is created here.

        ``suggest_working_dir`` rather than ``suggest_integration_output_dir``
        because the CSV belongs to the analysis root, not to one scan's
        per-detector output folder, and because it recognises a ``_bc``
        directory the data already sits in (see its docstring). Falls back
        through the Output directory field and the source folder to the home
        directory, so this always returns somewhere the dialog can open.
        """
        src_cfg = self._loader.source_cfg()
        rep = src_cfg.get("path")
        if not rep:
            paths = src_cfg.get("paths") or []
            rep = paths[0] if paths else None
        expid = self._expid_provider().strip() if self._expid_provider else ""
        parts = bc_path_parts(rep, expid_fallback=expid) if rep else None
        detector = (parts.detector if parts and parts.detector else "detector")
        # mpe_wf spells the beamline with neither dashes nor case: 20-ID-E is
        # "20ide" in its DETECTORS_BY_BEAMLINE and in every filename it writes.
        beamline = settings.active_profile().lower().replace("-", "")
        # Every candidate is checked for existence, and the ladder ends at
        # home. The positional derivation inside bc_path_parts is correct for
        # the full mpe_wf layout and invents a plausible-looking sibling for
        # anything shallower, and the Output directory field is a suggestion
        # that may not have been created yet either; opening the save dialog
        # on a path nobody has made, inside a shared beamline tree, invites
        # creating it by accident.
        candidates = [
            suggest_working_dir(rep, expid_fallback=expid) if rep else None,
            browse_start_dir(self._out_ed.text(), fallback="") or None,
            Path(rep).parent if rep else None,
        ]
        directory = next((Path(c) for c in candidates
                          if c and Path(c).is_dir()), Path.home())
        return directory / f"cake_parameters.{beamline}.{detector}.csv"

    def _cake_summary_text(self) -> str:
        """One line carrying every value a cake_parameters CSV can set, read
        back off the widgets that actually hold them.

        Those fields live behind the "R bins…"/"Azimuthal bins…" popups and,
        for OME_SUM, over in the loader card, so after a CSV load there was
        nowhere to see at a glance what had landed. This is that glance. It
        reads the widgets rather than the CSV on purpose: it is then equally
        true for values typed by hand, restored from a project, or auto-filled
        (Rmax) once a calibration resolves — "currently loaded", not "last
        imported"."""
        rmax = self._r_max.value()
        parts = [
            f"R {self._r_min.value():g}–{'auto' if rmax <= 0 else format(rmax, 'g')} px"
            f"  ΔR {self._r_bin.value():g} px",
            f"η {self._eta_min.value():g}…{self._eta_max.value():g}°"
            f"  Δη {self._e_bin.value():g}°",
        ]
        if self._q_mode_active():
            parts.append(f"Q out {self._q_min.value():g}–{self._q_max.value():g}"
                         f"  ΔQ {self._q_bin.value():g} Å⁻¹")
        chunk = getattr(self._loader, "_combine_chunk", None)
        n_sum = int(chunk.value()) if chunk is not None else 1
        if n_sum > 1:
            # Named as OME_SUM here because it IS OME_SUM: the cake dialog's
            # row for it edits this very spin box rather than a copy of it
            # (see _open_cake_params_dialog), and a user who has only seen
            # the CSV has no other way to find that out.
            parts.append(f"sum {n_sum} sub-frames (OME_SUM)")
        # Rotation is shown only when there is one. A run at a fixed angle is
        # the common case and "ω 0–0°" would be noise on every line.
        chan = self._ome_channel.currentText().strip()
        step = self._ome_step.value()
        if chan:
            parts.append(f"ω from {chan}")
        elif step or self._ome_start.value():
            part = f"ω {self._ome_start.value():g}°  Δω {step:g}°/sub-frame"
            # The multiplication that is easy to get wrong: combining N
            # sub-frames leaves consecutive OUTPUT frames N·OME_STEP apart,
            # not OME_STEP apart. Both rates, side by side, whenever they
            # differ.
            if n_sum > 1 and step:
                part += f" = {step * n_sum:g}°/frame"
            parts.append(part)
        if self._ome_collapse.isChecked():
            parts.append("ω averaged")
        line = "   ".join(parts)
        mapping = self._omega_map_text()
        return line + ("\n   " + mapping if mapping else "")

    def _omega_map_text(self) -> str:
        """Second summary line: which raw sub-frames become which angles.

        This is the whole point of the pair of readouts. ``start``/``end``
        and "Combine sub-frames" in the loader card count raw sub-frames;
        OME_START/OME_STEP/OME_SUM assign degrees to those same sub-frames.
        They are one axis in two coordinate systems, and nothing in the GUI
        used to say so — so state it for the data actually loaded: the raw
        range on the left, the angles it produces on the right, the frame
        count in between.

        Pure arithmetic on ``self._omega_span``, which is refreshed only
        when the SOURCE changes (see ``_recompute_omega_span``), so typing
        in OME_START re-renders this without touching a file. Returns ``""``
        when there is nothing to say — no source loaded, or one this app
        cannot ask for a sub-frame window."""
        span = getattr(self, "_omega_span", None)
        if not span or span.get("n", 0) < 1:
            return ""
        n = span["n"]
        lo, hi = span["first"][0], span["last"][1]
        # On a multi-file pick each file restarts at OME_START (one HDF5
        # sub-frame stack is one rotation — see
        # workers._HDF5StackGlobSource.omega_channel_window), so the range
        # is the range WITHIN a file and the angles repeat per file. Saying
        # "0…1441" flat there would claim a single continuous ramp.
        each = span.get("multi_file")
        where = " of each file" if each else ""
        chan = self._ome_channel.currentText().strip()
        if chan:
            return (f"sub-frames {lo}…{hi}{where} → ω read per frame from "
                    f"'{chan}'   {n} frame(s)")
        start, step = self._ome_start.value(), self._ome_step.value()
        if not step and not start and n > 1:
            # The state that produced the all-zero /Omegas in a real run: a
            # rotation loaded, no angles configured. omega_for_window
            # returns a genuine 0.0 for it and the output looks fine, so
            # this is the only place it can be noticed.
            return (f"sub-frames {lo}…{hi}{where} → ω 0° on all {n} frames "
                    "(OME_START/OME_STEP not set)")
        if self._ome_collapse.isChecked():
            one = omega_for_window(start, step, lo, hi)
            return (f"sub-frames {lo}…{hi}{where} → one ω {one:g}° for all "
                    f"{n} frame(s) (ω averaged)")
        a = omega_for_window(start, step, *span["first"])
        b = omega_for_window(start, step, *span["last"])
        rep = " in every file" if each else ""
        return (f"sub-frames {lo}…{hi}{where} → ω {a:g}°…{b:g}°{rep}   "
                f"{n} frame(s)"
                + ("   (each file restarts at OME_START)" if each else ""))

    def _omega_hint_tail(self) -> str:
        """The one-clause ω tail appended to the loader's start/end hint —
        see ``widgets.DataLoaderPanel.set_omega_hint_fn``.

        The loader card counts sub-frames and deliberately knows nothing
        about angles; this puts the angular reading of its own range next to
        it, in the place where the range is set. Deliberately terse — the
        cake summary carries the detail — and empty when no source is
        loaded, which is what keeps the hint byte-identical in every tab
        that never sets a hint function."""
        span = getattr(self, "_omega_span", None)
        if not span or span.get("n", 0) < 1:
            return ""
        chan = self._ome_channel.currentText().strip()
        if chan:
            return f"→ ω read from '{chan}'."
        start, step = self._ome_start.value(), self._ome_step.value()
        if not step and not start:
            return "→ ω 0° (OME_START/OME_STEP not set in the cake parameters)."
        lo, hi = span["first"][0], span["last"][1]
        if self._ome_collapse.isChecked():
            return f"→ one ω {omega_for_window(start, step, lo, hi):g}° (averaged)."
        a = omega_for_window(start, step, *span["first"])
        b = omega_for_window(start, step, *span["last"])
        per = " in each file" if span.get("multi_file") else ""
        return f"→ ω {a:g}°…{b:g}°{per}."

    def _schedule_omega_span(self, *_args) -> None:
        """Ask for a span refresh soon, coalescing a burst into one walk.

        Everything this is wired to can fire repeatedly for a single user
        action, and on a folder pick each firing would re-open every file's
        header on the GUI thread. Best-effort: if the timer is gone (teardown)
        just do the work inline."""
        t = getattr(self, "_omega_span_timer", None)
        if t is None:
            self._recompute_omega_span()
            return
        t.start()

    def _recompute_omega_span(self, *_args) -> None:
        """Re-read the loaded source's raw sub-frame windows, then re-render
        both ω readouts.

        Header reads only (``_open_source_cfg`` → ``_ensure_stats`` never
        decodes a pixel), and only on a change of SOURCE or of "Combine
        sub-frames" — both of which already pay for exactly this walk in the
        loader itself. Editing OME_START/OME_STEP does not come through
        here: the readouts recompute their angles from the cached windows,
        so a spin box can be dragged without touching the filesystem.

        The windows come from the source rather than from arithmetic on the
        raw count so that the readout cannot disagree with the run: a short
        final chunk, a start/end filter and the per-file restart are all
        decided in ``raw_window_for_index``, which is what the run uses
        too."""
        span = key = None
        try:
            from midas_gui.workers import _open_source_cfg
            cfg = self._loader.source_cfg()
            # Same source as last time — the windows cannot have moved, and
            # on a many-file pick re-deriving them is the expensive part.
            key = repr(sorted(cfg.items(), key=lambda kv: kv[0]))
            if key == self._omega_span_key:
                return
            src = _open_source_cfg(cfg)
            n = int(getattr(src, "n_frames", 0) or 0)
            win = getattr(src, "raw_window_for_index", None)
            if n > 0 and win is not None:
                span = {"n": n,
                        "first": tuple(int(v) for v in win(0)),
                        "last": tuple(int(v) for v in win(n - 1)),
                        "multi_file": len(cfg.get("paths") or []) > 1}
        except Exception:
            # A live/PVA source, nothing loaded yet, an unreadable file — all
            # ordinary, and all mean the same thing here: no mapping to show.
            span = None
            key = None
        self._omega_span = span
        self._omega_span_key = key if span is not None else None
        self._refresh_cake_summary()
        try:
            self._loader.refresh_omega_hint()
        except Exception:
            pass

    def _omega_cfg(self) -> dict:
        """The rotation half of a run's configuration, in the shape
        ``BatchWorker``/``BatchRunCoordinator`` take it — assembled here so
        both run sites (and the recorded attempt inputs) agree by
        construction, the same way ``q_cfg`` is."""
        return {
            "start": float(self._ome_start.value()),
            "step": float(self._ome_step.value()),
            "channel": self._ome_channel.currentText().strip(),
            "collapse": bool(self._ome_collapse.isChecked()),
        }

    def _refresh_cake_summary(self, *_args) -> None:
        """Keep the cake-parameter summary label in step with the fields.

        Best-effort: this is wired to a lot of signals, some of which can fire
        while the tab is still being built, and a summary label is never worth
        taking the tab down for."""
        lbl = getattr(self, "_cake_lbl", None)
        if lbl is None:
            return
        try:
            lbl.setText(self._cake_summary_text())
        except Exception:
            pass

    def _refresh_omega_hint(self, *_args) -> None:
        """Re-render the loader's range hint alone, without re-reading the
        source — the angles change, the sub-frame windows behind them do
        not. Best-effort, like ``_refresh_cake_summary``: a hint is never
        worth taking the tab down for."""
        try:
            self._loader.refresh_omega_hint()
        except Exception:
            pass

    def _radial_readout(self, col, row) -> str:
        """2θ / Q / d / η under the cursor — see
        ``widgets.ImageViewer.set_radial_readout_fn``.

        Reads the cached geometry rather than resolving the calibration on
        every hover: the "From file" source parses a file, which this must
        not do at cursor-move rate. ``_refresh_detector_preview`` drops the
        cache whenever the calibration source changes.

        The preview is drawn with ``set_raw_frame`` under this same
        calibration's ``im_trans``, so the hovered pixel is already in the
        geometry's frame.
        """
        if self._readout_geom is None:
            fields, _note = self._calib_fields_in_use()
            self._readout_geom = fields or {}
        return pixel_readout_text(col, row, self._readout_geom)

    def _refresh_detector_preview(self, *_args) -> None:
        """Refresh the Detector-view tab's frame + Rmin/Rmax/bin-grid overlay —
        called on new/changed data, a calibration-source change, or any of
        the Rmin/Rmax/R-bin/η-bin/Show-bin-grid controls changing. The Rmax
        auto-fill and overlay must not depend on a frame already being
        loaded — calibration commonly arrives before data does."""
        # The calibration may have changed under us (this is the hub every
        # source change routes through), so the readout's cached geometry is
        # no longer trustworthy. Dropping it here keeps _radial_readout free
        # of file IO on the hover path.
        self._readout_geom = None
        self._det_view._refresh_coord_bar()
        # current_frame() already applies dark/bright/background correction
        # to each constituent frame before any "Preview: sum first N"
        # summing (see DataLoaderPanel._start_preview_worker) — correcting
        # again here would double-apply it. In "stream" mode this may return
        # a stale (or None) frame immediately while a fresh one is fetched
        # off the GUI thread in the background — previewFrameReady re-calls
        # this method once that lands, so the view still ends up current.
        frame = self._loader.current_frame()
        fields, note = self._calib_fields_in_use()
        if frame is not None:
            # BC_y/BC_z (and the overlay drawn from them) are defined in the
            # *flipped/transposed* frame the calibration was fit against —
            # see widgets.ImageViewer.set_raw_frame for why every such
            # display goes through that one function instead of each call
            # site flip-then-set_image-ing on its own.
            # Re-frame the view (and re-level) only when the displayed image
            # is genuinely different. Every Rmin/Rmax/R-bin/η-bin/Show-bin-grid
            # control routes through this same refresh, and autoRange() throws
            # away whatever pan/zoom the user had set — ticking "Show bin grid"
            # on a zoomed-out view snapped it back to a tight fit, which reads
            # as the overlay having zoomed the image in. reset_levels would
            # likewise discard a manual colour-scale window. The Data Viewer
            # and Calibrate tab already pass autorange=False for refreshes
            # that aren't new data; this is the same rule, keyed on what
            # actually changes the picture's extent.
            codes = tuple((fields or {}).get("im_trans") or ())
            framed_for = (tuple(frame.shape), codes)
            fresh = framed_for != self._det_view_framed_for
            self._det_view.set_raw_frame(frame, (fields or {}).get("im_trans"),
                                          autorange=fresh, reset_levels=fresh)
            self._det_view_framed_for = framed_for
        if not fields or fields.get("BC_y") is None or fields.get("NrPixelsY") is None:
            # No visible sign otherwise that the overlay silently isn't being
            # drawn (e.g. "From file" pointing at a saved *project* .json
            # instead of an actual calibration geometry file/paramstest.txt/
            # .poni — geometry_fields_from_file rejects it, but every other
            # caller of _calib_fields_in_use() throws the note away). Log it
            # once per distinct note so re-checking a box or nudging a
            # spinbox doesn't spam the log on every refresh.
            if note != self._last_calib_fail_note:
                self._log.append(
                    f"[batch] Detector-view overlay (Rmin/Rmax, bin grid, "
                    f"lab-frame axes) not drawn: {note}")
                self._last_calib_fail_note = note
            draw_polar_bin_overlay(
                self._det_view, self._bin_overlay_items,
                bc_y=0.0, bc_z=0.0, r_min=0.0, r_max=0.0, r_bin=1.0, e_bin=5.0)
            self._clear_lab_axes()
            return
        self._last_calib_fail_note = None
        if self._r_max.value() == 0.0:
            self._r_max.blockSignals(True)
            self._r_max.setValue(rmax_corner_px(
                fields["BC_y"], fields["BC_z"], fields["NrPixelsY"], fields["NrPixelsZ"]))
            self._r_max.blockSignals(False)
        if frame is not None:
            expected = (int(fields["NrPixelsZ"]), int(fields["NrPixelsY"]))
            if tuple(frame.shape) != expected and expected != self._last_shape_mismatch_logged:
                self._log.append(
                    f"[batch] Warning: calibration is for a {expected[1]}x{expected[0]} "
                    f"detector but the loaded frame is {frame.shape[1]}x{frame.shape[0]} — "
                    "the beam-centre/Rmin/Rmax overlay and lab-frame axes will be drawn in "
                    "the WRONG place (or effectively invisible) until the calibration "
                    "matches this data's actual detector size.")
                self._last_shape_mismatch_logged = expected
        draw_polar_bin_overlay(
            self._det_view, self._bin_overlay_items,
            bc_y=fields["BC_y"], bc_z=fields["BC_z"],
            r_min=self._r_min.value(), r_max=self._r_max.value(),
            r_bin=self._r_bin.value(), e_bin=self._e_bin.value(),
            eta_min=self._eta_min.value(), eta_max=self._eta_max.value(),
            show_grid=self._grid_chk.isChecked(),
            tx=fields.get("tx") or 0.0, ty=fields.get("ty") or 0.0,
            tz=fields.get("tz") or 0.0, lsd_um=fields.get("Lsd"),
            pxY_um=fields.get("pxY"), pxZ_um=fields.get("pxZ"))
        self._redraw_lab_axes_if_on(fields["BC_y"], fields["BC_z"])

    def _on_preview_sum_changed(self, n: int) -> None:
        self._loader.set_preview_sum(n)
        self._refresh_detector_preview()

    # ── Lab-frame axes overlay (same as Data Viewer/Calibrate — see
    # widgets.build_lab_frame_axes_items) ───────────────────────────
    def _on_lab_axes_toggled(self, checked: bool) -> None:
        if checked:
            self._refresh_detector_preview()
        else:
            self._clear_lab_axes()

    def _on_origin_changed(self, *_args) -> None:
        """Display origin flipped — the compass is drawn in screen terms, so
        it has to be re-derived rather than carried along by the ViewBox's
        now-inverted Y axis."""
        if not self._lab_axes_chk.isChecked():
            return
        fields, _ = self._calib_fields_in_use()
        if not fields or fields.get("BC_y") is None:
            return
        self._redraw_lab_axes_if_on(fields["BC_y"], fields["BC_z"])

    def _redraw_lab_axes_if_on(self, bc_y: float, bc_z: float) -> None:
        if not self._lab_axes_chk.isChecked():
            return
        self._clear_lab_axes()
        img = self._det_view._data
        if img is None:
            return
        items = build_lab_frame_axes_items(self._det_view._iv, img.shape, bc_y, bc_z)
        for it in items:
            self._det_view._iv.addItem(it)
        self._axis_items.extend(items)

    def _clear_lab_axes(self) -> None:
        for it in self._axis_items:
            self._det_view._iv.removeItem(it)
        self._axis_items.clear()

    def _resolved_im_trans(self) -> tuple:
        """ImTransOpt codes from the active calibration source — the same
        flip/transpose the geometry (BC/tilts) was fit in, which every raw
        frame streamed into BatchWorker/FolderMonitorWorker must also get."""
        fields, _ = self._calib_fields_in_use()
        return tuple(fields.get("im_trans") or []) if fields else ()

    def set_mask_from_tab1(self, mask):
        self._loader.set_tab1_mask(mask)

    def bind_hydra_registry(self, registry, label: str):
        """Same role as `widgets.DataLoaderPanel.bind_registry`, for this
        tab's Hydra loader — deferred like `set_project_context` since the
        Hydra page (and its loader) is built lazily on first use."""
        self._hydra_registry = registry
        self._hydra_registry_label = label
        if self._hydra_page is not None:
            self._hydra_page._loader.bind_registry(registry, label)

    def _ensure_hydra_page(self) -> HydraBatchPage:
        if self._hydra_page is None:
            self._hydra_page = HydraBatchPage()
            self._mode_stack.addWidget(self._hydra_page)
            if self._project_ctx is not None:
                self._hydra_page.set_project_context(self._project_ctx)
            if self._hydra_registry is not None:
                self._hydra_page._loader.bind_registry(
                    self._hydra_registry, self._hydra_registry_label)
        return self._hydra_page

    def set_hydra_panel_calibration(self, n: int, result):
        """A Hydra panel's fit finished on the Calibrate tab — hand it to
        this tab's own Hydra Batch Integrate page (building it on first use)."""
        self._ensure_hydra_page().set_panel_calibration(n, result)

    def _sync_zarr_grouping_enabled(self):
        """Grey the grouping combo out while "zarr" is unchecked — it is the
        only format it affects."""
        on = "zarr" in self._fmt.checked_keys()
        self._zarr_grouping.setEnabled(on)
        self._zarr_grouping.setToolTip(
            self._zarr_grouping_tip if on else
            "Only applies to Zarr output — tick Zarr in Output format to "
            "choose how many frames share one archive.")

    def _zarr_grouping_key(self) -> str:
        """"frame" | "file" | "run" — what ``BatchWorker``'s
        ``zarr_grouping`` expects, from the combo's item data."""
        return self._zarr_grouping.currentData() or "frame"

    def _ion_csv_extras(self) -> tuple:
        """Optional beam-monitor CSV column groups, as the tuple
        ``ion_csv.write_ion_csv`` expects. Collapsed into ONE value rather
        than a kwarg per checkbox: a new batch option has to be threaded
        through four independent run constructors (in-process, as-job argv,
        CLI, queue), so each extra kwarg is paid for four times over."""
        return tuple(k for k, chk in (("env", self._ion_env_chk),
                                      ("motors", self._ion_motors_chk))
                     if chk.isChecked())

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def _state_widgets(self) -> dict:
        return {
            "use_tab2_btn": self._use_tab2_btn,
            "use_json_btn": self._use_json_btn,
            "json_ed": self._json_ed,
            "kernel": self._kernel,
            # Single-value widget, so widgets_to_dict persists it for free —
            # unlike OutputFormatSelector, which is excluded from this map.
            "zarr_grouping": self._zarr_grouping,
            "r_bin": self._r_bin,
            "e_bin": self._e_bin,
            "r_min": self._r_min,
            "r_max": self._r_max,
            "eta_min": self._eta_min,
            "eta_max": self._eta_max,
            # Invisible; the run's rotation — see where they're built.
            "ome_start": self._ome_start,
            "ome_step": self._ome_step,
            "ome_channel": self._ome_channel,
            "ome_collapse": self._ome_collapse,
            "grid_chk": self._grid_chk,
            "lab_axes_chk": self._lab_axes_chk,
            "preview_sum_n": self._preview_sum_n,
            "azim": self._azim,
            "multi_azimuth": self._multi_azimuth_chk,
            "ion_env_chk": self._ion_env_chk,
            "ion_motors_chk": self._ion_motors_chk,
            "var_check": self._var_check,
            "err_model": self._err_model,
            "bin_type": self._bin_type,
            "q_min": self._q_min,
            "q_max": self._q_max,
            "q_bin": self._q_bin,
            "mon_ed": self._mon_ed,
            "drift_chk": self._drift_chk,
            "drift_anchor_ed": self._drift_anchor_ed,
            "drift_param": self._drift_param,
            "drift_knots": self._drift_knots,
            "drift_bayesian": self._drift_bayesian,
            "out_ed": self._out_ed,
            "run_mode": self._run_mode,
            "n_workers": self._n_workers,
        }

    def get_state(self) -> dict:
        return {
            "fields": widgets_to_dict(self._state_widgets()),
            "corr": self._corr_widget.get_state(),
            "fmt": self._fmt.get_state(),
            "loader": self._loader.get_state(),
            "det_view": self._det_view.display_state(),
            "waterfall": self._waterfall.display_state(),
            "cake_stack": self._cake_stack_view.display_state(),
            "hydra": {"active_mode": self._mode_ribbon.mode(),
                      "page": self._hydra_page.get_state() if self._hydra_page else {}},
            "calib_result": project.sanitize_result_dict(self._calib_result),
        }

    def set_state(self, state: dict):
        self._loader.set_state(state.get("loader") or {})
        fields = dict(state.get("fields", {}))
        # A project-attempt's "fmt_keys" (see project.integrate_attempt_gui_fields)
        # rides along in "fields" but isn't a plain widget — pull it out before
        # the generic apply_dict_to_widgets pass (which would just ignore it).
        fmt_keys = fields.pop("fmt_keys", None)
        # Likewise "q_check": the boolean the Bin type dropdown replaced —
        # still emitted by project.integrate_attempt_gui_fields (shared with
        # HydraBatchPage, which still uses the checkbox) when restoring a
        # saved integration attempt's provenance.
        if fields.pop("q_check", None):
            fields.setdefault("bin_type", "Q")
        apply_dict_to_widgets(self._state_widgets(), fields)
        self._det_view.set_display_state(state.get("det_view"))
        self._origin_btn.sync()
        self._waterfall.set_display_state(state.get("waterfall"))
        self._cake_stack_view.set_display_state(state.get("cake_stack"))
        self._corr_widget.set_state(state.get("corr") or {})
        self._fmt.set_state(fmt_keys if fmt_keys is not None else state.get("fmt"))
        hydra_state = state.get("hydra") or {}
        page_state = hydra_state.get("page") or {}
        if page_state:
            self._ensure_hydra_page().set_state(page_state)
        self._mode_ribbon.set_mode(hydra_state.get("active_mode", "single"))
        calib_state = state.get("calib_result")
        if calib_state:
            self._apply_calib_result(project.calibration_namespace(calib_state))
            self._refresh_detector_preview()

    # ── File > Open Project… ─────────────────────────────────────────
    def apply_project_integration(self, attempts: dict) -> None:
        """``attempts`` maps panel key (``"single"`` or ``"ge1"``..``"ge4"``)
        to that panel's integration-attempt metadata (``project.read_attempt``)
        — called after File > Open Project… when the user opts to populate
        this tab. Unlike plain GUI-state load, this also installs the
        recorded ``calibration_snapshot`` as a live, usable calibration (via
        ``set_calibration``/``set_panel_calibration``) so Run works
        immediately, without needing Tab 2 re-run first."""
        if not attempts:
            return
        single_meta = attempts.get("single")
        hydra_metas = {k: v for k, v in attempts.items() if k != "single"}
        state = {}
        if single_meta is not None:
            state["fields"] = project.integrate_attempt_gui_fields(single_meta)
            state["loader"] = project.integrate_attempt_loader_state(single_meta)
        if hydra_metas:
            shared_fields, loader_state, anchor_path = {}, {}, None
            for panel_key, meta in sorted(hydra_metas.items()):
                shared_fields = project.integrate_attempt_gui_fields(meta)
                loader_state = project.integrate_attempt_loader_state(meta)
                if anchor_path is None:
                    anchor_path = loader_state.get("path")
            state["hydra"] = {"active_mode": "hydra",
                               "page": {"fields": shared_fields, "loader": loader_state,
                                        "anchor_path": anchor_path}}
        elif single_meta is not None:
            state["hydra"] = {"active_mode": "single"}
        self.set_state(state)

        if single_meta is not None:
            snap = single_meta.get("calibration_snapshot")
            if snap:
                self.set_calibration(project.calibration_namespace(snap))
            self._populate_plots_from_attempt(single_meta)
        for panel_key, meta in hydra_metas.items():
            snap = meta.get("calibration_snapshot")
            n = int(panel_key[2:])
            if snap:
                self._ensure_hydra_page().set_panel_calibration(
                    n, project.calibration_namespace(snap))
            self._ensure_hydra_page().populate_panel_plots(n, meta)

    def _populate_plots_from_attempt(self, meta: dict) -> None:
        """Fill the Waterfall/Stacked-profiles views from an integration
        attempt's embedded ``profiles``/``r_axis_px``/``frame_ids`` arrays
        (``project.read_attempt_results``, stashed onto ``meta`` under
        ``_results_arrays`` by the Open Project dialog) — same widget calls
        ``_on_frame`` makes per-frame during a live run, just replayed in one
        shot instead of streamed. Best-effort: a missing/incompatible
        calibration (for the x-axis re-labelling) never blocks the plots.

        A multi-azimuth attempt stores ``(n_frames, n_eta, n_r)`` cakes rather
        than 1-D profiles. Those go to the "Eta-R cakes" tab whole, and the two
        1-D views get an η-collapse of them (see :meth:`_collapse_cakes`) —
        feeding the raw cakes to a 1-D viewer used to raise on the first
        frame, which took the whole restore with it.
        """
        arrays = meta.get("_results_arrays") or {}
        r_axis = arrays.get("r_axis_px")
        profiles = arrays.get("profiles")
        if r_axis is None or profiles is None or len(profiles) == 0:
            return
        profiles = np.asarray(profiles)
        # Clear the views before re-deriving their axis context from this
        # attempt's calibration — otherwise set_axis_context()'s _restack()
        # re-plots whatever attempt/run was displayed previously under the
        # new geometry (see _run()'s identical fix).
        self._waterfall.reset(r_axis)
        self._stack_view.reset(r_axis)
        self._cake_stack_view.clear()
        try:
            spec = self._build_spec()
            axctx = (float(spec.Lsd), float(spec.pxY), float(spec.Wavelength))
            self._waterfall.set_axis_context(*axctx)
            self._stack_view.set_axis_context(*axctx)
            self._cake_stack_view.set_axis_context(*axctx)
        except Exception:
            pass
        frame_ids = arrays.get("frame_ids") or list(range(len(profiles)))
        if profiles.ndim == 3:
            self._restore_cake_stack(meta, r_axis, profiles, frame_ids)
            profiles = self._collapse_cakes(profiles)
        self._integrated_fids = set()
        for fid, prof in zip(frame_ids, profiles):
            self._waterfall.add_profile(prof)
            self._stack_view.add_profile(r_axis, prof, label=fid)
            self._integrated_fids.add(str(fid))
        self._wf_started = True
        self._view_tabs.setCurrentWidget(self._waterfall)

    def _restore_cake_stack(self, meta: dict, r_axis, cakes, frame_ids) -> None:
        """Replay a multi-azimuth attempt's stored cakes into the "Eta-R
        cakes" tab. The η axis isn't a ``results`` array — ``_log_to_project``
        records it in the attempt's ``extra`` as ``eta_axis_deg`` — so an
        attempt written before that existed falls back to evenly spaced bins
        over a full turn, which is what every multi-azimuth run this tab can
        produce actually uses."""
        eta_axis = meta.get("eta_axis_deg")
        n_eta = int(np.asarray(cakes).shape[1])
        if eta_axis is None or len(eta_axis) != n_eta:
            step = 360.0 / n_eta
            eta_axis = -180.0 + step * (np.arange(n_eta) + 0.5)
        self._cake_stack_view.set_cakes(cakes, r_axis, eta_axis,
                                        frame_ids=frame_ids)

    @staticmethod
    def _collapse_cakes(cakes):
        """``(n_frames, n_eta, n_r)`` → ``(n_frames, n_r)``, averaging each
        frame's filled η bins — see ``helpers.collapse_cake_eta`` (shared
        with ``cake_hdf5.write_cake_h5``'s own fallback).

        The run's *own* collapsed profile is not stored (multi-azimuth mode
        keeps the cake instead), so this reconstructs one for the Waterfall /
        Stacked-profiles views. It is an approximation of the engine's
        count-weighted collapse, not a reproduction of it."""
        return collapse_cake_eta(cakes)

    def _on_job_done(self, job) -> None:
        """``JobQueuePanel``'s ``on_job_done`` callback: a background
        ``screen`` job runs in a separate process, so its results only ever
        reach this tab via the ``_bg_job_results.npz`` sidecar
        ``batch_cli._write_results_sidecar`` leaves in ``job.out_dir`` —
        reuses ``_populate_plots_from_attempt``'s replay logic, same as
        restoring a project's saved attempt."""
        self._archive_job_log(job)
        if not job.out_dir:
            self._log.append(f"[batch] Job {job.session} has no recorded "
                             f"output folder — can't load its results.")
            return
        sidecar = Path(job.out_dir) / "_bg_job_results.npz"
        if not sidecar.is_file():
            self._log.append(f"[batch] No results file found for job {job.session} "
                             f"(expected {sidecar}).")
            return
        try:
            with np.load(sidecar) as npz:
                arrays = {
                    "r_axis_px": npz["r_axis_px"],
                    "profiles": npz["profiles"],
                    "frame_ids": list(npz["frame_ids"]),
                }
                eta_axis_deg = (npz["eta_axis_deg"].tolist()
                               if "eta_axis_deg" in npz.files else None)
        except Exception as e:
            self._log.append(f"[batch] Could not load results for job {job.session}: {e}")
            return
        self._populate_plots_from_attempt({"_results_arrays": arrays,
                                          "eta_axis_deg": eta_axis_deg})
        self._log.append(f"[batch] Loaded results from background job: {job.session}")

    def shutdown(self):
        """Interrupt + bounded-wait every Hydra-page worker on app close —
        this tab's own workers are already covered by MainWindow's generic
        QThread sweep, but those nested inside ``self._hydra_page`` are not
        (the sweep only inspects this tab's own ``vars()``, not recursively)."""
        if self._hydra_page is not None:
            self._hydra_page.shutdown()
        self._job_queue.shutdown()

    def _on_mode_changed(self, mode: str):
        """Leftmost ribbon switched between "single" and "hydra" — swap the
        visible page (building the Hydra page on first use), mirroring
        CalibrationTab's/DataViewerTab's identical split."""
        self._mode_stack.setCurrentWidget(
            self._ensure_hydra_page() if mode == "hydra" else self._hsplit)

    def set_hydra_available(self, enabled: bool) -> None:
        """Show/hide the Hydra option on the mode ribbon (only meaningful at
        the 1-ID-E beamline profile — see MainWindow.apply_hydra_visibility)."""
        self._mode_ribbon.set_hydra_enabled(enabled)

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(0)

        # Leftmost mode ribbon: "Single detector" (this tab's existing view)
        # vs. "Hydra" (4-panel GE detector batch integration) — same pattern
        # as the Data Viewer / Calibrate tabs' splits.
        self._mode_ribbon = HydraModeRibbon()
        self._mode_ribbon.modeChanged.connect(self._on_mode_changed)
        root.addWidget(self._mode_ribbon)

        self._mode_stack = QtWidgets.QStackedWidget()
        root.addWidget(self._mode_stack, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False); split.setHandleWidth(6)
        self._mode_stack.addWidget(split); self._hsplit = split

        # ── LEFT: data loader (streaming source + dark/bright/bg + mask) ──
        # unify_combine=True: start/end + "Combine sub-frames" apply the same
        # way to HDF5 and TIFF-family sources here, replacing "stride" — see
        # widgets.DataLoaderPanel's unify_combine docs. Pump Probe embeds the
        # same panel WITHOUT this flag and keeps the old stride behavior.
        self._loader = DataLoaderPanel(mode="stream", unify_combine=True)
        self._loader.setMinimumWidth(200)
        split.addWidget(self._loader)

        # ── MIDDLE: parameters ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setMinimumWidth(260)
        inner = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(inner); lv.setContentsMargins(2, 2, 2, 2); lv.setSpacing(8)
        scroll.setWidget(inner)

        def _br(w=30):
            b = QtWidgets.QPushButton("…"); b.setFixedWidth(w); return b

        # ── Calibration source ──
        cal = S.make_card("Calibration source")
        src_row = QtWidgets.QHBoxLayout(); src_row.setSpacing(10)
        self._use_tab2_btn = QtWidgets.QRadioButton("From Tab 2")
        self._use_json_btn = QtWidgets.QRadioButton("From file")
        self._use_tab2_btn.setChecked(True)
        src_row.addWidget(self._use_tab2_btn); src_row.addWidget(self._use_json_btn); src_row.addStretch(1)
        cal.body.addLayout(src_row)
        self._calib_src_lbl = QtWidgets.QLabel("(run Tab 2 first)")
        self._calib_src_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        cal.body.addWidget(self._calib_src_lbl)
        self._json_ed = QtWidgets.QLineEdit()
        self._json_ed.setPlaceholderText("calibration.json / paramstest.txt / .poni…")
        jr = QtWidgets.QHBoxLayout(); jr.setSpacing(4); jr.addWidget(self._json_ed, 1)
        self._json_browse_btn = _br(); self._json_browse_btn.clicked.connect(lambda: self._json_ed.setText(
            _browse(self, "Open calibration file",
                    "Calibration (*.json *.txt *.poni);;All (*)") or "")); jr.addWidget(self._json_browse_btn)
        self._json_ed.textChanged.connect(
            lambda t: self._use_json_btn.setChecked(True) if t.strip() else None)
        cal.body.addLayout(jr)
        # "Calibration values" used to be an always-visible grid here (took a
        # lot of vertical space); it's now a popup, opened on click, showing
        # the same fields — see helpers.make_calib_values_button.
        calib_view_btn = make_calib_values_button(self._calib_fields_in_use)
        cal.body.addWidget(calib_view_btn, 0, QtCore.Qt.AlignLeft)
        cake_csv_btn = QtWidgets.QPushButton("Cake parameters…")
        cake_csv_btn.setToolTip(
            "Show all nine cake_parameters columns "
            "(R_MIN/R_MAX/R_STEP/ETA_MIN/ETA_MAX/ETA_STEP/OME_SUM/OME_START/"
            "OME_STEP) in one editor, and load or save them as an "
            "mpe_wf_saxs_waxs-style CSV. OME_SUM is the loader's 'Combine "
            "sub-frames' chunk size (only meaningful for a multi-file HDF5 "
            "source). OME_START/OME_STEP are the rotation — every integrated "
            "frame is stamped with the ω of the raw frames it was built "
            "from, and that angle goes into the zarr and the combined HDF5. "
            "Also here: a measured omega channel, and the averaged/summed "
            "override.")
        cake_csv_btn.clicked.connect(self._open_cake_params_dialog)
        cal.body.addWidget(cake_csv_btn, 0, QtCore.Qt.AlignLeft)
        # The cake parameters themselves are behind two popups and the loader
        # card; this is the only place all of them are visible at once, and
        # the only feedback that a CSV load actually changed anything.
        self._cake_lbl = QtWidgets.QLabel()
        # Two lines since the sub-frame↔ω mapping joined it, and the
        # second one is long — wrap rather than clip it in a narrow panel.
        self._cake_lbl.setWordWrap(True)
        self._cake_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._cake_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._cake_lbl.setToolTip(
            "The cake parameters currently in force, wherever they came from "
            "(CSV, typed by hand, restored with a project, or auto-filled). "
            "Edit them all together in 'Cake parameters…' above, or one axis "
            "at a time in 'R bins…' / 'Azimuthal bins…' below.")
        cal.body.addWidget(self._cake_lbl)
        lv.addWidget(cal)

        # ── Integration ──
        integ = S.make_card("Integration")
        self._kernel = _NoScrollComboBox()
        for label, key in KERNELS.items():
            self._kernel.addItem(label, key)
        _ki = self._kernel.findData(DEFAULT_KERNEL)
        if _ki >= 0:
            self._kernel.setCurrentIndex(_ki)
        def _section_label(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px;font-weight:bold;")
            return lbl

        # One Form for every label:value row in this card (Kernel through
        # Azim. mean) — a QGridLayout sizes its label column to the widest
        # label added to THAT SAME instance, so splitting R/Q/Eta into
        # separate Form()s (as an earlier pass did) left each section's
        # entry cells starting at a slightly different x depending on its
        # own longest label ("η min:" vs "R bin:" vs "ΔQ:"). One shared
        # Form makes every entry cell line up regardless of section.
        # Section headers and checkboxes go in via .full() (spans the
        # whole row) so they interleave with the label:value rows without
        # starting a new grid.
        pf = S.Form()
        pf.row(("Kernel:", self._kernel))

        # Grouped per axis (bin size + range together, plus anything else
        # that's really about that axis) so the relationships are visible
        # instead of scattered across the card. The actual Δ/min/max fields
        # for each axis live in a popup dialog (opened by the button next to
        # it) rather than inline — see _RadialBinsDialog/_AzimuthalBinsDialog
        # below — freeing this card down to the choices that matter at a
        # glance:
        #   RADIAL:    Bin type selects which axis the adjacent button edits.
        #              R bin/Rmin/Rmax always define the underlying
        #              integration grid; Q only rebins that result for
        #              OUTPUT (see _run()'s "Always R-uniform..." comment) —
        #              not an independent axis.
        #   AZIMUTHAL: η bin/η min/η max live behind "Azimuthal bins…".
        #              Azim. mean (how η is collapsed to 1-D) and
        #              Multi-azimuth output (whether it's collapsed at all)
        #              stay inline — both are about the same η axis. Maps to
        #              a cake_parameters CSV's
        #              R_MIN/R_MAX/R_STEP/ETA_MIN/ETA_MAX/ETA_STEP
        #              one-for-one (see _load_cake_csv). Rmax 0.0 is the
        #              "auto" sentinel: left untouched, it's passed through
        #              as RMax=None so the backend's own farthest-corner
        #              default stays authoritative (see helpers._build_spec)
        #              — auto-filled to that same corner value once
        #              calibration resolves. η min/max default -180/180
        #              (full circle), matching the backend's own default
        #              (same None-means-"leave the backend default"
        #              contract as Rmin/Rmax).
        self._r_bin = _fspin(0.1, 20.0, 2, 1.0, "px")
        self._r_min = _fspin(0.0, 1_000_000.0, 2, 0.0, "px")
        self._r_max = _fspin(0.0, 1_000_000.0, 2, 0.0, "px")
        self._r_max.setToolTip(
            "0 = auto (farthest detector corner from the beam centre).\n"
            "Use the Corner/Edge presets in the R bins dialog, or type your own.")
        self._rmax_corner_btn = QtWidgets.QPushButton("Corner")
        self._rmax_corner_btn.setToolTip("Set Rmax to the farthest detector CORNER from the beam centre.")
        self._rmax_corner_btn.clicked.connect(lambda: self._apply_rmax_preset(rmax_corner_px))
        self._rmax_edge_btn = QtWidgets.QPushButton("Edge")
        self._rmax_edge_btn.setToolTip("Set Rmax to the farthest detector EDGE from the beam centre.")
        self._rmax_edge_btn.clicked.connect(lambda: self._apply_rmax_preset(rmax_edge_px))
        for w in (self._r_min, self._r_max, self._r_bin):
            w.valueChanged.connect(self._refresh_detector_preview)
        self._q_min = _fspin(0.0, 100.0, 3, 0.5, "Å⁻¹")
        self._q_max = _fspin(0.0, 100.0, 3, 8.0, "Å⁻¹")
        self._q_bin = _fspin(0.0001, 1.0, 4, 0.01, "Å⁻¹")
        self._r_bins_dialog = _RadialBinsDialog(
            "R", self._r_bin, self._r_min, self._r_max,
            self._rmax_corner_btn, self._rmax_edge_btn, parent=self)
        self._q_bins_dialog = _RadialBinsDialog(
            "Q", self._q_bin, self._q_min, self._q_max, parent=self)

        pf.full(_section_label("RADIAL"))
        self._bin_type = _NoScrollComboBox()
        self._bin_type.addItem("Radial", "R")
        self._bin_type.addItem("Q", "Q")
        self._bin_type.setToolTip(
            "Radial — bin uniformly in R (detector pixels).\n"
            "Q — additionally rebin the OUTPUT uniformly in Q (Å⁻¹); the "
            "underlying integration grid is still R-uniform.")
        self._bin_type.currentIndexChanged.connect(self._on_bin_type_changed)
        self._radial_bins_btn = QtWidgets.QPushButton("R bins…")
        self._radial_bins_btn.clicked.connect(self._open_radial_bins_dialog)
        bt_row = QtWidgets.QHBoxLayout(); bt_row.setSpacing(4)
        bt_row.addWidget(self._bin_type, 1); bt_row.addWidget(self._radial_bins_btn)
        pf.row(("Bin type:", bt_row))

        self._e_bin = _fspin(0.5, 360.0, 1, 5.0, "°")
        self._eta_min = _fspin(-180.0, 180.0, 1, -180.0, "°")
        self._eta_max = _fspin(-180.0, 180.0, 1, 180.0, "°")
        for w in (self._eta_min, self._eta_max, self._e_bin):
            w.valueChanged.connect(self._refresh_detector_preview)
        self._azim_bins_dialog = _AzimuthalBinsDialog(
            self._e_bin, self._eta_min, self._eta_max, parent=self)
        # Not dead widgets, and deliberately in no layout: these four are
        # the run's rotation, and only _CakeParamsDialog shows them.
        # OME_START/OME_STEP are two of the nine columns an mpe_wf
        # cake_parameters CSV must carry (its reader rejects a missing or
        # empty one); the channel and the averaged/summed flag are this
        # app's own and are not written to the CSV. Holding all four as
        # widgets rather than plain attributes is what lets
        # _state_widgets() round-trip them through Save/Load GUI State for
        # free, and gives the dialog a real range to mirror. They are read
        # at run time into BatchWorker's omega_cfg — see _omega_cfg().
        self._ome_start = _fspin(-1e6, 1e6, 4, 0.0)
        self._ome_step = _fspin(-1e6, 1e6, 4, 0.0)
        self._ome_channel = QtWidgets.QComboBox()
        self._ome_channel.setEditable(True)
        self._ome_collapse = QtWidgets.QCheckBox()
        for _w in (self._r_min, self._r_max, self._r_bin, self._eta_min,
                   self._eta_max, self._e_bin, self._q_min, self._q_max,
                   self._q_bin, self._ome_start, self._ome_step):
            _w.valueChanged.connect(self._refresh_cake_summary)
        self._ome_channel.currentTextChanged.connect(self._refresh_cake_summary)
        self._ome_collapse.toggled.connect(self._refresh_cake_summary)
        # The loader's hint carries the same angles, so it follows the same
        # four widgets. No file is touched — see _omega_hint_tail.
        for _w in (self._ome_start, self._ome_step):
            _w.valueChanged.connect(self._refresh_omega_hint)
        self._ome_channel.currentTextChanged.connect(self._refresh_omega_hint)
        self._ome_collapse.toggled.connect(self._refresh_omega_hint)
        self._bin_type.currentIndexChanged.connect(self._refresh_cake_summary)
        _chunk = getattr(self._loader, "_combine_chunk", None)
        if _chunk is not None:
            # OME_SUM: changes the number of output frames and every
            # frame's window, so the whole span is restated, not just the
            # summary text (_recompute_omega_span refreshes both).
            _chunk.valueChanged.connect(self._schedule_omega_span)
        self._refresh_cake_summary()

        pf.full(_section_label("AZIMUTHAL"))
        self._azim_bins_btn = QtWidgets.QPushButton("Azimuthal bins…")
        self._azim_bins_btn.clicked.connect(self._azim_bins_dialog.exec_)
        azim_btn_row = QtWidgets.QHBoxLayout(); azim_btn_row.setSpacing(4)
        azim_btn_row.addWidget(self._azim_bins_btn); azim_btn_row.addStretch(1)
        pf.full(azim_btn_row)

        self._azim = _NoScrollComboBox()
        self._azim.addItem("Pixel-weighted", True)
        self._azim.addItem("η-bin mean (legacy)", False)
        self._azim.setToolTip(
            "How the 2-D (η, R) cake is collapsed to a 1-D profile:\n"
            "• Pixel-weighted — Σ(mean·count)/Σ(count); robust to partial azimuthal\n"
            "  coverage / off-detector beam centres and independent of η-bin size.\n"
            "• η-bin mean — unweighted mean of the per-η-bin means (can distort the\n"
            "  profile with a coarse η bin when the beam centre is off the detector).")
        pf.row(("Azim. mean:", self._azim))
        integ.body.addLayout(pf)

        self._multi_azimuth_chk = QtWidgets.QCheckBox("Multi-azimuth output (cake)")
        self._multi_azimuth_chk.setToolTip(
            "Keep every azimuthal (η) sector as a SEPARATE output profile "
            "instead of collapsing to one full-circle mean profile per "
            "frame. Reuses the η bin/η range above to define the sectors.\n\n"
            "Off by default — η bin already defaults to 5° over the full "
            "360° (72 internal bins) purely to control collapse-weighting "
            "resolution; turning this on repurposes that same setting to "
            "also define real output granularity, so every existing run's "
            "result size is unaffected unless you opt in.\n\n"
            "Needed for per-azimuth GSAS-II/texture analysis. Not yet "
            "supported together with Q-uniform bins.\n\n"
            "Setting η bin to 360° does NOT make this equivalent to leaving "
            "it unchecked — that just collapses the sectors to one trivial "
            "360°-wide sector, still written as a (1-row) cake instead of a "
            "plain profile, and HDF5 output is still skipped. For a real "
            "azimuthal-mean profile, leave this box unchecked.")
        integ.body.addWidget(self._multi_azimuth_chk)

        self._var_check = QtWidgets.QCheckBox("Per-bin variance (σ)")
        self._var_check.setToolTip(
            "Compute per-bin σ via the chosen error model.\n"
            "Mutually exclusive with corrections (corrections win; σ→√I).")
        self._err_model = _NoScrollComboBox(); self._err_model.addItems(ERROR_MODELS); self._err_model.setEnabled(False)
        if DEFAULT_ERROR_MODEL in ERROR_MODELS:
            self._err_model.setCurrentText(DEFAULT_ERROR_MODEL)
        self._var_check.toggled.connect(self._err_model.setEnabled)
        vrow = QtWidgets.QHBoxLayout(); vrow.setSpacing(6)
        vrow.addWidget(self._var_check); vrow.addWidget(self._err_model, 1)
        integ.body.addLayout(vrow)
        lv.addWidget(integ)

        # ── Corrections ──
        self._corr_widget = CorrectionFlagsWidget()
        lv.addWidget(self._corr_widget)

        # ── Monitor normalisation ──
        mon = S.make_card("Monitor normalisation (optional)")
        self._mon_ed = QtWidgets.QLineEdit()
        self._mon_ed.setPlaceholderText("monitor.txt  (one value per line)")
        monr = QtWidgets.QHBoxLayout(); monr.setSpacing(4); monr.addWidget(self._mon_ed, 1)
        bmon = _br(); bmon.clicked.connect(lambda: self._mon_ed.setText(
            _browse(self, "Open monitor file", "Text (*.txt *.dat *.csv);;All (*)") or ""))
        monr.addWidget(bmon)
        mon.body.addLayout(monr)
        mon_note = QtWidgets.QLabel(
            "Each profile is divided by the corresponding monitor value.\n"
            "File: one floating-point number per line, one per processed frame.")
        mon_note.setWordWrap(True)
        mon_note.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        mon.body.addWidget(mon_note)
        lv.addWidget(mon)

        # ── Drift correction ──
        drift = S.make_card("Drift correction (long scans)")
        self._drift_chk = QtWidgets.QCheckBox("Enable per-frame geometry drift correction")
        drift.body.addWidget(self._drift_chk)
        self._drift_anchor_ed = QtWidgets.QLineEdit()
        self._drift_anchor_ed.setPlaceholderText("anchors.json  ({frame_idx: {Lsd, BC_y, BC_z}})")
        drow = QtWidgets.QHBoxLayout(); drow.setSpacing(4); drow.addWidget(self._drift_anchor_ed, 1)
        bdrift = QtWidgets.QPushButton("…"); bdrift.setFixedWidth(30)
        bdrift.clicked.connect(lambda: self._drift_anchor_ed.setText(
            _browse(self, "Open anchor JSON", "JSON (*.json);;All (*)") or ""))
        drow.addWidget(bdrift); drift.body.addLayout(drow)
        df = S.Form()
        self._drift_param = _NoScrollComboBox()
        self._drift_param.addItems(["spline", "linear", "constant"])
        self._drift_knots = _NoScrollSpinBox(); self._drift_knots.setRange(2, 1_000_000); self._drift_knots.setValue(5)
        df.row(("Parametrization:", self._drift_param), ("n_knots:", self._drift_knots))
        drift.body.addLayout(df)
        self._drift_bayesian = QtWidgets.QCheckBox("Bayesian σ estimate"); self._drift_bayesian.setChecked(True)
        drift.body.addWidget(self._drift_bayesian)
        self._drift_fit_btn = QtWidgets.QPushButton("Fit trajectory")
        self._drift_fit_btn.clicked.connect(self._fit_drift)
        drift.body.addWidget(self._drift_fit_btn)
        self._drift_status_lbl = QtWidgets.QLabel("No trajectory fitted")
        self._drift_status_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        drift.body.addWidget(self._drift_status_lbl)
        lv.addWidget(drift)
        # Not used in production yet — hidden from the GUI but left fully wired
        # (DriftWorker/_fit_drift/state save-restore all still work) so it can
        # be shown again by removing this one line.
        drift.setVisible(False)

        # ── Output ──
        out = S.make_card("Output")
        self._out_ed = QtWidgets.QLineEdit(); self._out_ed.setPlaceholderText("Output directory…")
        warn_if_path_missing(self._out_ed, self, is_output_dir=True)
        orow = QtWidgets.QHBoxLayout(); orow.setSpacing(4); orow.addWidget(self._out_ed, 1)
        bou = _br(); bou.clicked.connect(lambda: self._out_ed.setText(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Output directory", browse_start_dir(self._out_ed.text())) or "")); orow.addWidget(bou)
        self._suggest_out_btn = QtWidgets.QPushButton("Suggest")
        self._suggest_out_btn.setToolTip(
            "Fill in <outroot>/<expid>_bc/<file-root>/<detector>/, matching "
            "mpe_wf_saxs_waxs's own outroot/<expid>_bc/<froot>/<detector>/ "
            "output-folder convention. Read positionally off the loaded "
            "source's own folder depth (<outroot>/<expid>/<detector>/"
            "<froot>/<files>) — no need to type the Exp ID first; falls "
            "back to <source folder>/<file-root>/, no detector segment, "
            "only when the source path isn't that deep. "
            "Output within is further split into per-format subfolders "
            "(csv/, xye/, fxye/, dat/, 2d_csv/, h5/, zarr/).")
        self._suggest_out_btn.clicked.connect(self._apply_suggested_output_dir)
        orow.addWidget(self._suggest_out_btn)
        out.body.addLayout(S.Form().row(("Folder:", orow)))
        self._fmt = OutputFormatSelector()
        out.body.addWidget(self._fmt)
        # How many output frames share one .zarr.zip. A rotation's worth of
        # frames in one archive is the useful default for a folder of
        # multi-sub-frame files — three 1442-sub-frame files at OME_SUM 10
        # otherwise produce 435 single-cake archives — but per-frame stays
        # the default because it is what every existing project recorded.
        self._zarr_grouping = _NoScrollComboBox()
        for _label, _key in (("One zarr per output frame", "frame"),
                             ("One zarr per source file", "file"),
                             ("One zarr for the whole run", "run")):
            self._zarr_grouping.addItem(_label, _key)
        self._zarr_grouping_tip = (
            "How many integrated frames share one .zarr.zip.\n"
            "Per source file groups by ROTATION — the same unit omega is "
            "measured from, so one HDF5 sub-frame stack (or, for one-frame-"
            "per-file data, the whole selection) becomes one archive.\n"
            "Batch Parallel cannot split a group across workers, so grouping "
            "caps the worker count at the number of groups.")
        self._zarr_grouping_row = S.Form().row(("Zarr grouping:", self._zarr_grouping))
        out.body.addLayout(self._zarr_grouping_row)
        # Disabled rather than hidden while zarr is off: a control that
        # silently does nothing is worse than one that says why it can't.
        self._fmt.changed.connect(self._sync_zarr_grouping_enabled)
        self._sync_zarr_grouping_enabled()

        # Per-frame beam-monitor CSV. Always written (it is small, and I0 is
        # what SAXS normalises against); these only add OPTIONAL columns, so
        # there is no on/off box to leave in the wrong state. See
        # midas_gui.ion_csv for why a column with no live data anywhere is
        # dropped rather than written blank.
        ion_lbl = QtWidgets.QLabel(
            "Ion-chamber CSV — written automatically, one row per frame "
            "(frame, I0, I, transmission). Also include:")
        ion_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        ion_lbl.setWordWrap(True)
        out.body.addWidget(ion_lbl)
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

        # ── Run mode ──
        run_mode_card = S.make_card("Run mode")
        mode_row = QtWidgets.QHBoxLayout(); mode_row.setSpacing(6)
        _mode_tip = (
            "Sequential: frames integrated one at a time.\n"
            "Batch Parallel: splits this run's frames across N workers sharing "
            "one detector map (built once). The worker count auto-shrinks so "
            f"each worker gets ≥{BatchRunCoordinator.MIN_FRAMES_PER_WORKER} frames.")
        _mode_lbl = S.LabelRight("Mode:")
        _mode_lbl.setToolTip(_mode_tip)
        mode_row.addWidget(_mode_lbl)
        self._run_mode = _NoScrollComboBox()
        self._run_mode.addItem("Sequential", "sequential")
        self._run_mode.addItem("Batch Parallel", "batch_parallel")
        self._run_mode.setToolTip(_mode_tip)
        mode_row.addWidget(self._run_mode)
        mode_row.addWidget(S.LabelRight("Workers:"))
        _max_workers = os.cpu_count() or 8
        self._n_workers = _NoScrollSpinBox()
        self._n_workers.setRange(1, _max_workers)
        self._n_workers.setValue(min(4, _max_workers))
        self._n_workers.setEnabled(False)
        mode_row.addWidget(self._n_workers)
        self._run_mode.currentIndexChanged.connect(
            lambda *_: self._n_workers.setEnabled(self._run_mode.currentData() == "batch_parallel"))
        run_mode_card.body.addLayout(mode_row)
        lv.addWidget(run_mode_card)

        # ── Run ──
        self._run_btn = S.primary_btn("Start Integration")
        self._run_btn.clicked.connect(self._run)
        self._abort_btn = QtWidgets.QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.setToolTip("Stop after the current frame, keeping frames already integrated.")
        self._abort_btn.clicked.connect(self._abort)
        self._abort_btn.setStyleSheet(S.DANGER_BTN_QSS)
        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip(
            "Write the lineouts already computed this run to disk, in the "
            "checked format(s) above. Only enabled after a run that had no "
            "Output folder set — when an Output folder was set, the run "
            "already wrote everything there as it went, so Save would just "
            "duplicate it.")
        self._save_btn.clicked.connect(self._save_results)
        self._save_btn.setStyleSheet(S.SUCCESS_BTN_QSS)
        run_row = QtWidgets.QHBoxLayout(); run_row.setSpacing(6)
        run_row.addWidget(self._run_btn, 1)
        run_row.addWidget(self._abort_btn, 1)
        run_row.addWidget(self._save_btn, 1)
        lv.addLayout(run_row)
        self._run_job_btn = QtWidgets.QPushButton("Run as background job")
        self._run_job_btn.setToolTip(
            "Linux only. Runs this same integration in a detached `screen` "
            "session (python -m midas_gui.batch_cli) instead of in-process — "
            "survives closing this GUI. Needs an Output folder (results and "
            "a calibration snapshot are written there for the job to read).\n"
            "Progress/log/cancel are tracked in the 'Background jobs' panel below.")
        self._run_job_btn.clicked.connect(self._run_as_job)
        lv.addWidget(self._run_job_btn)
        self._clear_btn = QtWidgets.QPushButton("Clear results")
        self._clear_btn.setToolTip(
            "Remove the integrated profiles/plots computed this session for the "
            "current data so a fresh integration can start. Does NOT delete raw "
            "data or any files on disk.")
        self._clear_btn.clicked.connect(self._clear_results)
        lv.addWidget(self._clear_btn)
        self._prog = QtWidgets.QProgressBar(); self._prog.setRange(0, 100); self._prog.setVisible(False)
        lv.addWidget(self._prog)
        self._prog_lbl = QtWidgets.QLabel(""); self._prog_lbl.setStyleSheet(f"font-size:10px;color:{S.MUTED}")
        lv.addWidget(self._prog_lbl)
        lv.addStretch(1)
        split.addWidget(scroll)

        # Right: waterfall / stacked-profiles / detector-view / logs tabs
        # A view-only option (not an integration parameter — see
        # tooltip), so it sits on the Detector-view toolbar next to Origin
        # rather than in the left Integration card.
        self._grid_chk = QtWidgets.QCheckBox("Show bin grid")
        self._grid_chk.setToolTip(
            "Overlay the (R, η) integration bin grid on the Detector view "
            "tab: the Rmin/Rmax boundaries, an arc at each R-bin edge and "
            "a spoke at each η-bin edge.\n\n"
            "Everything is bounded to η min/max, so the overlay covers the "
            "region actually being integrated — a limited azimuth draws an "
            "annulus sector, not the whole ring.\n\n"
            "Thinned to at most ~50 arcs / ~72 spokes for legibility with "
            "fine bin sizes. Unchecking this hides the overlay entirely, "
            "including Rmin/Rmax.\n\n"
            "View-only — has no effect on the integration itself.")
        self._grid_chk.toggled.connect(self._refresh_detector_preview)
        self._view_tabs = QtWidgets.QTabWidget()
        self._waterfall = WaterfallViewer()
        self._stack_view = StackedProfileViewer()
        # One (η, R) cake per frame — only filled by a "Multi-azimuth output"
        # run, which is what produces a per-frame cake instead of the
        # η-collapsed profile the other two views show.
        self._cake_stack_view = CakeStackViewer()
        self._det_view = ImageViewer()
        self._det_view.set_radial_readout_fn(self._radial_readout)
        self._origin_btn = OriginToolButton(self._det_view)
        self._det_view._toolbar_layout.addWidget(self._origin_btn)
        self._det_view._toolbar_layout.addWidget(self._grid_chk)
        self._lab_axes_chk = QtWidgets.QCheckBox("Lab-frame axes")
        self._lab_axes_chk.setToolTip(
            "Overlay MIDAS lab-frame axes (X_Lab/Y_Lab), the beam-direction "
            "⊗ glyph, and an η sweep arc, anchored at the active "
            "calibration's beam centre — same overlay as the Data "
            "Viewer/Calibrate tabs.")
        self._lab_axes_chk.toggled.connect(self._on_lab_axes_toggled)
        # Flipping the display origin inverts the ViewBox's Y axis; the compass
        # points at the hutch, not the pixel grid, so it is re-derived rather
        # than carried along (widgets.build_lab_frame_axes_items).
        self._det_view.originChanged.connect(self._on_origin_changed)
        self._det_view._toolbar_layout.addWidget(self._lab_axes_chk)
        self._det_view._toolbar_layout.addWidget(QtWidgets.QLabel("Preview: sum first"))
        self._preview_sum_n = _NoScrollSpinBox()
        self._preview_sum_n.setRange(1, 999); self._preview_sum_n.setValue(1)
        self._preview_sum_n.setFixedWidth(50)
        self._preview_sum_n.setToolTip(
            "Detector-view preview only — never affects the real batch run. "
            "Sums this many of the source's leading frames together before "
            "display, to boost signal enough to see the actual diffraction "
            "pattern beneath detector-readout artifacts (e.g. VAREX "
            "per-column gain non-uniformity) that dominate a single frame.")
        self._preview_sum_n.valueChanged.connect(self._on_preview_sum_changed)
        self._det_view._toolbar_layout.addWidget(self._preview_sum_n)
        self._view_tabs.addTab(self._det_view, "Detector view")
        self._view_tabs.addTab(self._waterfall, "Waterfall")
        self._view_tabs.addTab(self._stack_view, "Stacked profiles")
        self._view_tabs.addTab(self._cake_stack_view, "Eta-R cakes")
        # Logs tab: background-job rows (JobQueuePanel) above one shared log
        # surface, used by both the in-process run and the currently-focused
        # background job's tailed output.
        self._log = LogPanel()
        self._log.setMaximumHeight(16_777_215)   # let the layout size it
        self._job_queue = JobQueuePanel(self._log, on_job_done=self._on_job_done)
        self._logs_tab = QtWidgets.QWidget()
        logs_tab_layout = QtWidgets.QVBoxLayout(self._logs_tab)
        logs_tab_layout.setContentsMargins(4, 4, 4, 4)
        logs_tab_layout.setSpacing(4)
        logs_tab_layout.addWidget(self._job_queue)
        logs_tab_layout.addWidget(self._log, 1)
        self._view_tabs.addTab(self._logs_tab, "Logs")
        self._view_tabs.setMinimumWidth(320)
        split.addWidget(self._view_tabs)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 0); split.setStretchFactor(2, 1)
        split.setSizes([286, 361, 950])

    # ── Run ────────────────────────────────────────────────────────

    def _build_spec(self):
        # Always R-uniform; Q-uniform is handled by rebinning in the worker because the
        # kernels do not implement Q-mode binning (see analyze_workflows/workflow_analysis.md).
        r_bin = self._r_bin.value(); e_bin = self._e_bin.value()
        r_min = self._r_min.value(); r_max = self._r_max.value() or None
        eta_min = self._eta_min.value(); eta_max = self._eta_max.value()
        if self._use_tab2_btn.isChecked():
            if self._calib_result is None:
                raise RuntimeError("No calibration from Tab 2. Run Tab 2 first.")
            return _build_spec(self._calib_result, r_bin, e_bin, r_min=r_min, r_max=r_max,
                               eta_min=eta_min, eta_max=eta_max)
        path = self._json_ed.text().strip()
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        return spec_from_geometry_file(path, r_bin, e_bin, r_min=r_min, r_max=r_max,
                                       eta_min=eta_min, eta_max=eta_max)

    def integration_settings(self) -> dict:
        """This tab's current binning / kernel / format choices, in the shape
        the Batch Queue tab's "Copy from Batch Integrate" button consumes.

        Read-only and widget-shaped rather than a spec: the queue applies these
        to several different calibrations, so it needs the *settings*, not one
        resolved ``IntegrationSpec``."""
        return {"kernel": self._kernel.currentData(),
                "r_bin": self._r_bin.value(), "e_bin": self._e_bin.value(),
                "r_min": self._r_min.value(), "r_max": self._r_max.value(),
                "eta_min": self._eta_min.value(), "eta_max": self._eta_max.value(),
                "fmt": self._fmt.checked_keys(),
                "weighted": bool(self._azim.currentData()),
                "chunk_size": self._loader.source_cfg().get("chunk_size") or 0,
                "combine_op": self._loader.source_cfg().get("combine_op") or "mean"}

    def set_expid_provider(self, provider) -> None:
        """Wired by app.py's MainWindow: ``provider()`` returns the header's
        current Exp ID (shared app-wide, not owned by this tab) for
        ``_suggest_output_dir`` to consume."""
        self._expid_provider = provider

    def _suggest_output_dir(self) -> Optional[Path]:
        """Best-effort ``<outroot>/<expid>_bc/<froot>/<detector>/`` off the
        loaded source path.

        The layout parse itself lives in ``helpers.bc_path_parts`` (shared with
        the Calibrate tab's working-directory suggestion, which composes a
        different tail from the same parse — see
        ``helpers.suggest_working_dir``). The header's Exp ID field feeds other
        things (see ``set_expid_provider``) but is intentionally not required
        here: the whole point of "Suggest" is to read the convention off the
        data that's actually loaded. Returns None when no source is loaded.
        """
        src_cfg = self._loader.source_cfg()
        rep = src_cfg.get("path")
        if not rep:
            paths = src_cfg.get("paths") or []
            rep = paths[0] if paths else None
        if not rep:
            return None
        expid = self._expid_provider().strip() if self._expid_provider else ""
        return suggest_integration_output_dir(rep, expid_fallback=expid)

    def _apply_suggested_output_dir(self):
        suggested = self._suggest_output_dir()
        if suggested is not None:
            self._out_ed.setText(str(suggested))
            reason = check_output_dir_writable(suggested)
            if reason:
                self._log.append(f"[batch] Warning: {reason}")
        else:
            self._log.append("[batch] No data source loaded yet — nothing to suggest.")

    def _maybe_autofill_output_dir(self):
        """Auto-fill the Output folder once a source loads, unless the user
        already typed/picked one — unlike mpe_wf's own GUIs (which only ever
        hint via placeholder text and never auto-fill), MIDAS_GUI fills this
        in live since there's no separate confirm-and-launch step to catch
        a wrong guess."""
        if self._out_ed.text().strip():
            return
        suggested = self._suggest_output_dir()
        if suggested is not None:
            self._out_ed.setText(str(suggested))
            reason = check_output_dir_writable(suggested)
            if reason:
                self._log.append(f"[batch] Warning: {reason}")

    def _run(self):
        if self._worker and self._worker.isRunning():
            return
        # A fresh batch run resets the display; stop any active monitor first.
        if self._loader.is_monitoring():
            self._stop_monitor()
            self._loader.set_monitor_active(False)
        self._orphans = [o for o in self._orphans if o.isRunning()]   # drop finished ones
        try:
            spec = self._build_spec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Calibration error", str(e)); return

        src_cfg = self._loader.source_cfg()
        if not (src_cfg.get("path") or src_cfg.get("paths")):
            QtWidgets.QMessageBox.warning(
                self, "No data",
                "Select a data folder/glob, HDF5 file, or file selection."); return

        kernel = self._kernel.currentData()
        corrections = self._corr_widget.build_corrections()
        variance_cfg = ({"error_model": self._err_model.currentText()}
                        if self._var_check.isChecked() else None)
        if variance_cfg and self._corr_widget.any_enabled():
            self._log.append("[batch] Note: corrections enabled → variance ignored (σ=√I).")
            variance_cfg = None

        out_dir = self._out_ed.text().strip() or None
        if out_dir:
            reason = check_output_dir_writable(out_dir)
            if reason:
                QtWidgets.QMessageBox.critical(self, "Output folder not writable", reason)
                return
        self._last_run_out_dir = out_dir
        fmts = self._fmt.checked_keys()
        q_cfg = ({"QMin": self._q_min.value(), "QMax": self._q_max.value(),
                  "QBinSize": self._q_bin.value()} if self._q_mode_active() else None)
        omega_cfg = self._omega_cfg()
        multi_azimuth = self._multi_azimuth_chk.isChecked()
        if multi_azimuth and q_cfg:
            QtWidgets.QMessageBox.warning(
                self, "Incompatible options",
                "Multi-azimuth output isn't supported together with "
                "Q-uniform bins yet. Uncheck Multi-azimuth output, or set "
                "Bin type back to Radial."); return
        lsd, px, wl = float(spec.Lsd), float(spec.pxY), float(spec.Wavelength)
        _axctx = (lsd, px, wl, "Q" if q_cfg else "R")

        # Dark / bright / background fields (from the loader)
        for sel in self._loader.has_pending_fields():
            QtWidgets.QMessageBox.warning(
                self, "Field not computed",
                f"'{sel.title()}' is enabled but not computed. "
                "Click 'Compute field' in that box first."); return
        dark = self._loader.dark()
        bright = self._loader.bright()
        background = self._loader.background()
        bright_mode = self._loader.bright_mode()

        self._run_btn.setEnabled(False); self._abort_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._last_results = None
        self._prog.setVisible(True); self._prog.setValue(0)
        self._wf_started = False
        self._integrated_fids = set()
        # Clear any previous run's views before re-deriving their axis context
        # from this run's geometry — otherwise set_axis_context()'s _restack()
        # re-plots stale curves under the new context, and a leftover curve
        # with no finite data (e.g. an empty/fully-masked profile) sends
        # pyqtgraph's autoRange() a [nan, nan] range and crashes the run.
        self._stack_view.reset()
        self._waterfall.reset()
        self._cake_stack_view.clear()
        self._stack_view.set_axis_context(*_axctx)
        self._waterfall.set_axis_context(*_axctx)
        # The cake's R axis is never Q-rebinned (multi-azimuth and Q-uniform
        # are mutually exclusive, rejected above), so it takes no native unit.
        self._cake_stack_view.set_axis_context(lsd, px, wl)
        self._view_tabs.setCurrentWidget(self._waterfall)
        self._log.append("─" * 40 + "\nStarting batch integration…")
        self._open_screen_log("integrate", {
            "Formats": ", ".join(self._fmt.checked_keys()) or "(none)",
            "Kernel": self._kernel.currentData(),
            "Output": self._out_ed.text().strip() or "(none)",
            "Zarr group": self._zarr_grouping_key(),
            "Run mode": self._run_mode.currentData(),
        })

        # Frame range (from the loader)
        frame_range = self._loader.frame_range()

        # Monitor normalisation file
        monitor_file = self._mon_ed.text().strip() or None

        # Drift trajectory (optional)
        drift_traj = None
        if self._drift_chk.isChecked():
            if self._drift_traj is None:
                QtWidgets.QMessageBox.warning(
                    self, "No trajectory",
                    "Drift correction enabled but no trajectory fitted.\n"
                    "Click 'Fit trajectory' first."); return
            drift_traj = self._drift_traj

        weighted = bool(self._azim.currentData())
        sig = self._integration_signature(src_cfg, kernel, corrections, weighted)
        context = self._geom_cache if (sig == self._geom_sig and
                                       self._geom_cache is not None) else None

        self._last_run_inputs = {
            "src_cfg": src_cfg, "kernel": kernel, "fmt": fmts,
            "frame_range": frame_range, "monitor_file": monitor_file,
            "q_cfg": q_cfg, "omega_cfg": omega_cfg,
            "weighted": weighted, "bright_mode": bright_mode,
            "mask_sources": self._loader.get_state().get("mask"),
            "r_bin": self._r_bin.value(), "e_bin": self._e_bin.value(),
            "multi_azimuth": multi_azimuth,
            "zarr_grouping": self._zarr_grouping_key(),
            "ion_csv_extras": list(self._ion_csv_extras()),
        }
        self._last_axis_ctx = (lsd, px, wl)
        # Stashed for the Save button, which runs long after this method
        # returns and has no other way to recover the geometry/kernel this
        # run used (needed for cake_hdf5.write_cake_h5 on a multi-azimuth
        # save — see _save_results).
        self._last_spec = spec
        self._last_kernel = kernel
        self._last_weighted = weighted
        mask = self._loader.composite_mask()
        self._last_run_fields = {
            "mask": mask,
            "mask_is_file_backed": mask is not None and not self._loader.has_live_mask_source(),
        }
        # The whole calibration, not just the display subset — an integration
        # attempt's provenance already embeds this (see _log_to_project); the
        # cake HDF5 writer embeds the same snapshot directly in the file.
        calib_snapshot, _calib_note = full_calibration_snapshot(
            self._calib_result, self._use_json_btn.isChecked(), self._json_ed.text())

        self._worker = BatchRunCoordinator(
            spec, src_cfg, self._loader.composite_mask(), out_dir, fmts, kernel,
            corrections, variance_cfg, q_cfg=q_cfg, omega_cfg=omega_cfg,
            frame_range=frame_range, monitor_file=monitor_file,
            drift_traj=drift_traj, parent=self,
            dark=dark, bright=bright, background=background, bright_mode=bright_mode,
            weighted=weighted, context=context, im_trans=self._resolved_im_trans(),
            multi_azimuth=multi_azimuth, calibration_snapshot=calib_snapshot,
            zarr_grouping=self._zarr_grouping_key(),
            ion_csv_extras=self._ion_csv_extras(),
            run_mode=self._run_mode.currentData(), n_workers=self._n_workers.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_done.connect(self._on_frame)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.log_line.connect(self._emit)
        self._worker.geom_ready.connect(lambda ctx, s=sig: self._cache_geom(s, ctx))
        self._worker.start()

    def _run_as_job(self):
        """Launch this same integration as a detached background `screen`
        job (see job_queue.JobQueuePanel) instead of running in-process —
        survives closing this GUI. Everything the CLI needs is written to
        disk first: the background process (a fresh `python -m
        midas_gui.batch_cli`) has no access to this GUI's live state."""
        try:
            # Called for validation only — the background process rebuilds its
            # own spec from the snapshot on disk, so the result is discarded.
            # Keep the call: it is what reports a bad calibration up front,
            # rather than letting the detached job fail where nobody sees it.
            self._build_spec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Calibration error", str(e)); return

        src_cfg = self._loader.source_cfg()
        if not (src_cfg.get("path") or src_cfg.get("paths")):
            QtWidgets.QMessageBox.warning(
                self, "No data",
                "Select a data folder/glob, HDF5 file, or file selection."); return

        out_dir = self._out_ed.text().strip()
        if not out_dir:
            QtWidgets.QMessageBox.warning(
                self, "Output folder required",
                "Background jobs need an Output folder — the job writes its "
                "calibration snapshot and results there."); return
        reason = check_output_dir_writable(out_dir)
        if reason:
            QtWidgets.QMessageBox.critical(self, "Output folder not writable", reason)
            return
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for sel in self._loader.has_pending_fields():
            QtWidgets.QMessageBox.warning(
                self, "Field not computed",
                f"'{sel.title()}' is enabled but not computed. "
                "Click 'Compute field' in that box first."); return

        fmts = self._fmt.checked_keys()
        if not fmts:
            QtWidgets.QMessageBox.warning(
                self, "No format", "Check at least one output format first."); return
        multi_azimuth = self._multi_azimuth_chk.isChecked()
        if multi_azimuth and self._q_mode_active():
            QtWidgets.QMessageBox.warning(
                self, "Incompatible options",
                "Multi-azimuth output isn't supported together with "
                "Q-uniform bins yet. Uncheck Multi-azimuth output, or set "
                "Bin type back to Radial."); return
        if self._q_mode_active():
            QtWidgets.QMessageBox.warning(
                self, "Not supported in background jobs yet",
                "Q-uniform bins aren't wired into background jobs yet.\n"
                "Set Bin type back to Radial, or use 'Start Integration' "
                "for an in-process run."); return

        import tifffile
        from midas_gui.helpers import write_standalone_paramstest
        # Materialize the calibration this run uses to a standalone file so
        # the background process never needs Tab 2's live in-memory result.
        if self._use_tab2_btn.isChecked():
            if self._calib_result is None:
                QtWidgets.QMessageBox.critical(
                    self, "Calibration error", "No calibration from Tab 2. Run Tab 2 first."); return
            calib_path = out_path / "_bg_job_calibration.txt"
            write_standalone_paramstest(self._calib_result, calib_path)
        else:
            calib_path = self._json_ed.text().strip()
            if not calib_path or not Path(calib_path).exists():
                QtWidgets.QMessageBox.critical(
                    self, "Calibration error",
                    f"Calibration file not found: {calib_path}"); return

        argv = [sys.executable, "-m", "midas_gui.batch_cli",
               "--calib-file", str(calib_path),
               "--r-bin", str(self._r_bin.value()), "--eta-bin", str(self._e_bin.value())]
        if self._r_min.value():
            argv += ["--r-min", str(self._r_min.value())]
        if self._r_max.value():
            argv += ["--r-max", str(self._r_max.value())]

        if src_cfg["type"] == "tiff_glob":
            argv += ["--source-type", "tiff_glob", "--source-path", src_cfg["path"]]
        elif src_cfg["type"] == "hdf5":
            argv += ["--source-type", "hdf5", "--source-path", src_cfg["path"],
                     "--dataset", src_cfg.get("dataset", "frames")]
        elif src_cfg["type"] == "hdf5_stack_glob":
            argv += ["--source-type", "hdf5_stack_glob", "--source-paths", *src_cfg["paths"],
                     "--dataset", src_cfg.get("dataset", "exchange/data")]
        else:
            argv += ["--source-type", "tiff_list", "--source-paths", *src_cfg["paths"]]

        argv += ["--out-dir", str(out_path), "--fmts", ",".join(fmts),
                 "--kernel", self._kernel.currentData()]

        # start/end are FILE/SCAN NUMBERS (batch_cli.py's --frame-start/
        # --frame-end match this meaning now); chunk_size/combine_op are
        # "Combine sub-frames" — both baked into src_cfg by a unify_combine
        # panel (see widgets.DataLoaderPanel.source_cfg), so just forward
        # them verbatim rather than going through frame_range() (which is
        # always (0, None, 1) for this panel — see its docstring).
        chunk_size = src_cfg.get("chunk_size")
        argv += ["--chunk-size", str(chunk_size if chunk_size is not None else 0),
                 "--combine-op", src_cfg.get("combine_op") or "mean"]
        if src_cfg.get("frame_start") is not None:
            argv += ["--frame-start", str(src_cfg["frame_start"])]
        if src_cfg.get("frame_end") is not None:
            argv += ["--frame-end", str(src_cfg["frame_end"])]

        if multi_azimuth:
            argv += ["--multi-azimuth"]
        argv += ["--weighted"] if bool(self._azim.currentData()) else ["--no-weighted"]

        # Rotation angle. Emitted from the same _omega_cfg() the in-process
        # run uses, so both paths write identical omegas for identical
        # settings — this argv is the only channel a background job has to
        # learn them, and omitting it silently wrote ω=0 on every frame.
        # start/step go out unconditionally (even at 0/0, which is a real
        # angle, not "unset") so the launched command line in the Logs tab
        # always states the angles the job will record.
        ome = self._omega_cfg()
        argv += ["--ome-start", str(ome["start"]), "--ome-step", str(ome["step"])]
        if ome["channel"]:
            argv += ["--ome-channel", ome["channel"]]
        if ome["collapse"]:
            argv += ["--ome-collapse"]
        # Same reasoning as the omega block above: the job cannot see the
        # combo, so an omitted flag would silently fall back to per-frame
        # archives and the two run paths would disagree.
        argv += ["--zarr-grouping", self._zarr_grouping_key()]
        # Same reasoning again: the job cannot see the checkboxes, so an
        # omitted flag would quietly drop the optional columns on this path
        # only, and the two run modes would disagree about the same run.
        extras = self._ion_csv_extras()
        if extras:
            argv += ["--ion-csv-extras", ",".join(extras)]

        if self._corr_widget.polar_check.isChecked():
            argv += ["--polarization",
                     "--pol-fraction", str(self._corr_widget.pol_fraction.value()),
                     "--pol-plane", str(self._corr_widget.pol_plane.value())]
        if self._corr_widget.solid_check.isChecked():
            argv += ["--solid-angle"]
        if self._var_check.isChecked() and not self._corr_widget.any_enabled():
            argv += ["--variance", "--error-model", self._err_model.currentText()]

        mask = self._loader.composite_mask()
        if mask is not None:
            mask_path = out_path / "_bg_job_mask.tif"
            tifffile.imwrite(str(mask_path), mask.astype("uint8"))
            argv += ["--mask", str(mask_path)]
        for name, arr in (("dark", self._loader.dark()), ("bright", self._loader.bright()),
                         ("background", self._loader.background())):
            if arr is not None:
                p = out_path / f"_bg_job_{name}.tif"
                tifffile.imwrite(str(p), np.asarray(arr, dtype=np.float32))
                argv += [f"--{name}", str(p)]
        if self._loader.dark() is not None or self._loader.bright() is not None:
            argv += ["--bright-mode", self._loader.bright_mode()]

        monitor_file = self._mon_ed.text().strip()
        if monitor_file:
            argv += ["--monitor-file", monitor_file]

        # Progress-bar estimate only — reopen the (already start/end-filtered,
        # chunk-combined) source the same way the job itself will, rather
        # than frame_range() (always (0, None, 1) for this panel).
        try:
            from midas_gui.workers import _open_source_cfg
            total = int(_open_source_cfg(src_cfg).n_frames) or 1
        except Exception:
            total = self._loader.n_frames() or 1

        job = self._job_queue.launch(argv, name=out_path.name or "batch", total_frames=total,
                                     out_dir=str(out_path))
        if job is not None:
            self._log.append(f"[batch] Launched background job: {job.session} "
                             f"(see the Logs tab)")
            self._view_tabs.setCurrentWidget(self._logs_tab)

    def _abort(self):
        """Stop the run. First ask the worker to stop cooperatively (clean finish
        with a summary); if it does not stop quickly, detach + terminate it and free
        the slot so a new run can start immediately (the orphaned thread winds down
        on its own). Frames already integrated were written to disk as the run went."""
        w = self._worker
        if not (w and w.isRunning()):
            return
        self._abort_btn.setEnabled(False)
        self._abort_btn.setText("Aborting…")
        self._log.append("[batch] aborting after the current frame…")
        w.requestInterruption()
        if w.wait(3000):
            return   # stopped cooperatively → _on_done fires and resets the UI
        # Still inside a long frame — detach and let it wind down on its own.
        # (No terminate(): killing a thread inside torch/numpy can crash the app.)
        for sig in (w.progress, w.frame_done, w.finished, w.failed, w.log_line):
            try:
                sig.disconnect()
            except Exception:
                pass
        self._orphans.append(w)
        self._worker = None
        self._reset_run_buttons(); self._prog.setVisible(False)
        self._log.append("[batch] aborted (a background thread is finishing the "
                         "current frame). Completed frames were saved.")

    def _on_progress(self, done, total):
        self._prog.setValue(int(100 * done / total) if total else 0)
        self._prog_lbl.setText(f"Integrated {done} / {total} frames")

    def _on_frame(self, fid, r_ax, prof, sigma):
        if not getattr(self, "_wf_started", False):
            self._waterfall.reset(r_ax)
            self._stack_view.reset(r_ax)
            self._wf_started = True
        self._waterfall.add_profile(prof)
        self._stack_view.add_profile(r_ax, prof, label=fid)
        self._integrated_fids.add(str(fid))
        self._log.append(f"  frame {fid}: peak={prof.max():.1f}")

    def _reset_run_buttons(self):
        self._run_btn.setEnabled(True)
        self._abort_btn.setEnabled(False); self._abort_btn.setText("Abort")

    # ── persistent run log (see midas_gui.run_log) ──────────────────

    def _emit(self, msg: str) -> None:
        """One line to the Log tab and to this run's on-disk screen log."""
        self._log.append(msg)
        if getattr(self, "_screen_log", None) is not None:
            self._screen_log.write(msg)

    def _open_screen_log(self, kind: str, fields: dict):
        """Start a run log under ~/midas_runs/midas_screen_logs/<beamline>/
        <expid>/. Best-effort — a log that cannot be opened costs one line in
        the Log tab, never the run."""
        from midas_gui import settings
        profile = settings.active_profile()
        expid = self._expid_provider().strip() if self._expid_provider else ""
        try:
            src = (self._loader.source_cfg() or {}).get("path")
        except Exception:
            src = None
        self._screen_log = run_log.open_log(
            kind, profile=profile, expid=expid,
            stem=Path(src).stem if src else "")
        if self._screen_log.path is None:
            self._log.append("[batch] Note: no run log written — "
                             + (self._screen_log.error or "could not open it"))
            return
        if not expid:
            self._log.append(f"[batch] Note: Exp ID is blank, so this run's "
                             f"log is filed under {run_log.NO_EXPID}/.")
        self._screen_log.header("MIDAS GUI — Batch Integrate",
                                {"Beamline": profile,
                                 "Experiment": expid or "(blank)",
                                 "Started": _dt.datetime.now().isoformat(
                                     timespec="seconds"),
                                 **fields})
        self._log.append(f"[batch] Run log: {self._screen_log.path}")

    def _close_screen_log(self) -> None:
        if getattr(self, "_screen_log", None) is not None:
            self._screen_log.close()
            self._screen_log = None

    def _archive_job_log(self, job) -> None:
        """Copy a finished background job's screenlog into the same tree the
        in-process run writes to.

        The live file has to stay in JOBS_DIR — job adoption after a GUI
        restart scans that one fixed location (see job_queue.JOBS_DIR) — so
        this copies rather than moves. Without it, which run mode you happened
        to use would decide whether your log was findable.
        """
        try:
            src = Path(getattr(job, "logfile", "") or "")
            if not src.is_file():
                return
            from midas_gui import settings
            expid = self._expid_provider().strip() if self._expid_provider else ""
            dest = run_log.log_path("integrate_job",
                                    profile=settings.active_profile(),
                                    expid=expid, stem=job.session)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            self._log.append(f"[batch] Run log: {dest}")
        except Exception as exc:
            self._log.append(f"[batch] Could not archive the job log: {exc}")

    def _on_done(self, data):
        self._reset_run_buttons(); self._prog.setVisible(False)
        self._close_screen_log()
        n = data["n"]; out = data.get("out_paths", [])
        aborted = data.get("aborted", False)
        verb = "aborted after" if aborted else "Done —"
        msg = f"{verb} {n} frames integrated"
        if out:
            # Output is now split into per-format subfolders (csv/xye/.../h5/zarr)
            # under the chosen Output dir — report that dir, not one file's
            # own parent, so the message doesn't just name whichever format
            # happened to run last.
            msg += f"\nSaved to: {self._out_ed.text().strip() or Path(out[0]).parent}"
        if (self._last_run_inputs.get("multi_azimuth")
                and "h5" in (self._last_run_inputs.get("fmt") or [])):
            msg += ("\nNote: HDF5 wasn't written — it isn't supported in "
                    "multi-azimuth mode (use the text formats or the Zarr/"
                    "GSAS-II export instead).")
        self._log.append(msg)
        self._prog_lbl.setText(f"{'Aborted' if aborted else 'Complete'}: {n} frames")
        if n and data.get("r_axis_px") is not None and self._last_axis_ctx is not None:
            profiles_arr = data.get("profiles")
            n_eta_bins = (int(profiles_arr.shape[1])
                         if profiles_arr is not None and profiles_arr.ndim == 3 else 1)
            self._last_results = {
                "r_axis_px": data["r_axis_px"], "profiles": profiles_arr,
                "sigmas": data.get("sigmas"), "frame_ids": data.get("frame_ids"),
                "n_eta_bins": n_eta_bins, "eta_axis": data.get("eta_axis"),
            }
            if self._last_run_out_dir:
                # The run itself already wrote everything (subfoldered, per
                # format) to this dir as it went — enabling Save here would
                # just dump a second, flat, unsubfoldered copy of the text
                # formats on top (the exact duplication reported against
                # this dir). Save's job is only to persist results that
                # were never written anywhere, i.e. a no-Output-folder run.
                self._save_btn.setEnabled(False)
                msg += ("\n(Save is disabled — results are already saved above; "
                        "Save is only for runs with no Output folder set.)")
                self._log.append(msg.rsplit('\n', 1)[-1])
            else:
                self._save_btn.setEnabled(True)
        self._update_cake_stack(data)
        self._log_to_project(data)
        QtWidgets.QMessageBox.information(self, "Aborted" if aborted else "Done", msg)

    def _update_cake_stack(self, data) -> None:
        """Fill (or clear) the "Eta-R cakes" tab from a finished run.

        Only a "Multi-azimuth output" run has cakes to show: that mode makes
        the worker accumulate each frame's ``cake_2d`` rather than the
        η-collapsed profile, so ``profiles`` comes back ``(n_frames, n_eta,
        n_r)`` alongside an ``eta_axis``. Anything else — including an
        η-collapsed run whose profiles are 2-D — clears the tab rather than
        leaving the previous run's cakes sitting there looking current.
        Both ``BatchWorker`` and ``BatchRunCoordinator`` (parallel) return the
        same payload shape, so this covers either run mode."""
        profiles = data.get("profiles")
        eta_axis = data.get("eta_axis")
        r_axis = data.get("r_axis_px")
        if (profiles is None or np.asarray(profiles).ndim != 3
                or eta_axis is None or r_axis is None):
            self._cake_stack_view.clear()
            return
        self._cake_stack_view.set_cakes(profiles, r_axis, eta_axis,
                                        frame_ids=data.get("frame_ids"))

    def _log_to_project(self, data):
        if not self._project_ctx or not self._project_ctx.path:
            return
        # The whole calibration, not the display subset _calib_fields_in_use
        # returns — an attempt has to be able to reconstruct the geometry it
        # ran under (helpers.full_calibration_snapshot).
        calib_fields, _note = full_calibration_snapshot(
            self._calib_result, self._use_json_btn.isChecked(), self._json_ed.text())
        calib_ref = None
        if self._use_tab2_btn.isChecked() and self._calib_result is not None:
            calib_ref = getattr(self._calib_result, "_project_attempt_ref", None)
        profiles_arr = data.get("profiles")
        n_eta_bins = (int(profiles_arr.shape[1])
                     if profiles_arr is not None and profiles_arr.ndim == 3 else 1)
        eta_axis = data.get("eta_axis")
        try:
            ref = project.append_integration_attempt(
                self._project_ctx.path, "single",
                inputs=self._last_run_inputs, finished_payload=data,
                calibration_snapshot=calib_fields, calib_attempt_ref=calib_ref,
                extra={"active_profile": settings.active_profile(),
                       "n_eta_bins": n_eta_bins,
                       "eta_axis_deg": (eta_axis.tolist() if eta_axis is not None else None)},
                **self._last_run_fields)
            self._log.append(f"Logged to project: {ref}")
        except Exception:
            import traceback as _tb
            self._log.append("Could not log to project file:\n" + _tb.format_exc())

    def _on_fail(self, msg):
        self._reset_run_buttons(); self._prog.setVisible(False)
        if getattr(self, "_screen_log", None) is not None:
            self._screen_log.write("ERROR:\n" + msg)
        self._close_screen_log()
        show_error(self, "Integration failed", msg, log=self._log, log_prefix="\nERROR:\n")

    def _clear_results(self):
        """Clear this session's computed profiles/plots for the current data so a
        fresh integration can start. Only in-session results are cleared — no raw
        data or files on disk are touched."""
        if self._worker and self._worker.isRunning():
            QtWidgets.QMessageBox.warning(
                self, "Run in progress",
                "Abort the running integration before clearing results.")
            return
        if self._loader.is_monitoring():
            self._stop_monitor()
            self._loader.set_monitor_active(False)
        self._waterfall.reset()
        self._stack_view.reset()
        self._cake_stack_view.clear()
        self._integrated_fids = set()
        self._wf_started = False
        self._last_results = None
        self._save_btn.setEnabled(False)
        self._prog.setVisible(False); self._prog.setValue(0)
        self._prog_lbl.setText("")
        self._log.append("Cleared session results — raw data untouched.")

    def _save_results(self):
        """Write the lineouts already computed this run to disk — independent
        of whether an Output folder was set before running (that only wired
        up incremental per-frame writes; this writes everything now, in the
        currently-checked format(s))."""
        if not self._last_results or self._last_axis_ctx is None:
            return
        fmts = self._fmt.checked_keys()
        if not fmts:
            QtWidgets.QMessageBox.warning(
                self, "No format", "Check at least one output format first."); return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Save lineouts to…", self._out_ed.text().strip())
        if not out_dir:
            return
        if "2d_csv" in fmts and self._last_results.get("n_eta_bins", 1) <= 1:
            self._log.append(
                "[batch] Note: 2D CSV (cake) isn't saved by Save for this run — "
                "per-frame cakes are only kept when 'Multi-azimuth output' was "
                "checked; re-run with that (or an Output folder + 2D CSV "
                "checked) to get that format.")
        lsd, px, wl = self._last_axis_ctx
        # Available whenever a run actually produced these results (see
        # _start_batch's stashing) — geometry for cake_hdf5.write_cake_h5 on
        # a multi-azimuth save. BinArea is left unpopulated here (no cheap
        # geometry rebuild post-run); everything else is full-fidelity.
        last_spec = self._last_spec
        cake_params = None
        if last_spec is not None:
            cake_params = {
                'RMin': float(last_spec.RMin), 'RMax': float(last_spec.RMax),
                'RBinSize': float(last_spec.RBinSize),
                'EtaMin': float(last_spec.EtaMin), 'EtaMax': float(last_spec.EtaMax),
                'EtaBinSize': float(last_spec.EtaBinSize),
            }
        calib_snapshot, _calib_note = full_calibration_snapshot(
            self._calib_result, self._use_json_btn.isChecked(), self._json_ed.text())
        try:
            paths = write_all_profiles(
                out_dir, fmts, self._last_results["r_axis_px"],
                self._last_results["profiles"], self._last_results["sigmas"],
                self._last_results["frame_ids"], lsd, px, wl,
                eta_axis=self._last_results.get("eta_axis"),
                spec=last_spec, calibration_snapshot=calib_snapshot,
                kernel=self._last_kernel, weighted=self._last_weighted,
                cake_params=cake_params,
                # Same angles the run's own HDF5 got — Save must not produce
                # a quietly omega-less copy of the same results.
                omegas=self._last_results.get("omegas"))
        except Exception as e:
            show_error(self, "Save failed", str(e), log=self._log, log_prefix="\nERROR:\n")
            return
        self._log.append(f"Saved {len(paths)} file(s) to {out_dir}")

    # ── Folder monitoring (live new-file integration) ──────────────

    def _integration_signature(self, src_cfg, kernel, corrections, weighted):
        """Signature identifying a reusable detector map for the current settings.

        Deliberately does NOT include the data source. ``build_integration_context``
        takes only (spec, kernel, mask, corrections, weighted) — the detector map
        is a property of the geometry, not of the frames fed through it — so
        keying on the source path only threw the map away every time the user
        pointed the tab at a different file under the same calibration. It does
        include Eta min/max, which the source terms used to sit next to and
        which genuinely do change the spec (``_build_spec`` passes them through,
        and the context's eta axis is derived from them)."""
        if self._use_tab2_btn.isChecked():
            calib = ("tab2", id(self._calib_result))
        else:
            calib = ("file", self._json_ed.text().strip())
        mask = self._loader.composite_mask()
        mask_id = None if mask is None else (tuple(mask.shape), int(np.count_nonzero(mask)))
        pol, sa = corrections
        return (calib, kernel, round(self._r_bin.value(), 4), round(self._e_bin.value(), 4),
                round(self._r_min.value(), 4), round(self._r_max.value(), 4),
                round(self._eta_min.value(), 4), round(self._eta_max.value(), 4),
                bool(weighted), pol is not None, sa is not None, mask_id)

    def _cache_geom(self, sig, ctx):
        self._geom_cache = ctx
        self._geom_sig = sig

    def _toggle_monitor(self, on):
        if on:
            self._start_monitor()
        else:
            self._stop_monitor()

    def _start_monitor(self):
        if self._monitor_worker and self._monitor_worker.isRunning():
            return
        if self._worker and self._worker.isRunning():
            QtWidgets.QMessageBox.warning(
                self, "Busy", "Wait for the batch run to finish (or Abort) before monitoring.")
            self._loader.set_monitor_active(False); return
        src_cfg = self._loader.source_cfg()
        if src_cfg.get("type") != "tiff_glob" or not src_cfg.get("path"):
            QtWidgets.QMessageBox.warning(
                self, "Folder needed",
                "MONITOR watches a folder (optionally filtered by a filestem) "
                "for new TIFF frames — select a folder or a filestem pick as "
                "the data source (HDF5 sources and an explicit multi-file "
                "pick can't be monitored — there's no folder to watch).")
            self._loader.set_monitor_active(False); return
        if (src_cfg.get("chunk_size") not in (None, 1)
                or src_cfg.get("frame_start") is not None
                or src_cfg.get("frame_end") is not None):
            QtWidgets.QMessageBox.warning(
                self, "Not supported with MONITOR",
                "MONITOR processes newly-arrived frames live, one at a time, "
                "so it can't combine them or restrict them to a file-number "
                "range. Reset 'Combine sub-frames' to 1 and start/end to the "
                "full range to use MONITOR.")
            self._loader.set_monitor_active(False); return
        try:
            spec = self._build_spec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Calibration error", str(e))
            self._loader.set_monitor_active(False); return
        for sel in self._loader.has_pending_fields():
            QtWidgets.QMessageBox.warning(
                self, "Field not computed",
                f"'{sel.title()}' is enabled but not computed. Click 'Compute field' first.")
            self._loader.set_monitor_active(False); return

        kernel = self._kernel.currentData()
        corrections = self._corr_widget.build_corrections()
        variance_cfg = ({"error_model": self._err_model.currentText()}
                        if self._var_check.isChecked() else None)
        if variance_cfg and self._corr_widget.any_enabled():
            variance_cfg = None
        q_cfg = ({"QMin": self._q_min.value(), "QMax": self._q_max.value(),
                  "QBinSize": self._q_bin.value()} if self._q_mode_active() else None)
        _axctx = (float(spec.Lsd), float(spec.pxY), float(spec.Wavelength),
                  "Q" if q_cfg else "R")
        self._stack_view.set_axis_context(*_axctx)
        self._waterfall.set_axis_context(*_axctx)
        weighted = bool(self._azim.currentData())
        dark = self._loader.dark(); bright = self._loader.bright()
        background = self._loader.background(); bmode = self._loader.bright_mode()
        out_dir = self._out_ed.text().strip() or None
        fmts = self._fmt.checked_keys()

        sig = self._integration_signature(src_cfg, kernel, corrections, weighted)
        context = self._geom_cache if (sig == self._geom_sig and
                                       self._geom_cache is not None) else None

        self._monitor_worker = FolderMonitorWorker(
            spec, src_cfg["path"], self._loader.composite_mask(), kernel, corrections,
            variance_cfg, q_cfg=q_cfg, dark=dark, bright=bright, background=background,
            bright_mode=bmode, weighted=weighted, seen=set(self._integrated_fids),
            context=context, out_dir=out_dir, fmts=fmts, parent=self,
            im_trans=self._resolved_im_trans())
        self._monitor_worker.frame_done.connect(self._on_frame)
        self._monitor_worker.new_count.connect(self._on_monitor_count)
        self._monitor_worker.log_line.connect(self._log.append)
        self._monitor_worker.failed.connect(self._on_monitor_fail)
        self._monitor_worker.geom_ready.connect(lambda ctx, s=sig: self._cache_geom(s, ctx))
        self._view_tabs.setCurrentWidget(self._stack_view)
        self._log.append("─" * 40 + "\nMONITOR active — watching the folder for new frames…")
        self._monitor_worker.start()

    def _stop_monitor(self):
        w = self._monitor_worker
        if w and w.isRunning():
            w.requestInterruption()
            if not w.wait(3000):
                # Detach + orphan rather than terminate() (which can crash the app
                # if the thread is inside a native integration call).
                for s in (w.frame_done, w.new_count, w.status, w.log_line,
                          w.failed, w.geom_ready):
                    try:
                        s.disconnect()
                    except Exception:
                        pass
                self._orphans.append(w)
            self._log.append("[monitor] stopped.")
        self._monitor_worker = None

    def _on_monitor_count(self, n):
        self._prog_lbl.setText(f"Monitoring — {n} new frame(s) integrated")

    def _on_monitor_fail(self, msg):
        self._loader.set_monitor_active(False)
        self._monitor_worker = None
        show_error(self, "Monitor failed", msg, log=self._log, log_prefix="\n[monitor] ERROR:\n")

    # ── Drift correction ───────────────────────────────────────────

    def _fit_drift(self):
        """Parse the anchors JSON and fit the drift trajectory."""
        if self._drift_worker and self._drift_worker.isRunning():
            return
        anchor_path = self._drift_anchor_ed.text().strip()
        if not anchor_path:
            QtWidgets.QMessageBox.warning(self, "Missing", "Specify an anchor JSON file."); return
        from pathlib import Path as _Path
        import json as _json
        try:
            raw = _json.loads(_Path(anchor_path).read_text())
            # JSON keys are strings; convert to int
            anchors = {int(k): v for k, v in raw.items()}
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "JSON error", str(e)); return
        if len(anchors) < 2:
            QtWidgets.QMessageBox.warning(self, "Too few anchors",
                                           "Need at least 2 anchor frames."); return
        try:
            calib_result = self._calib_result
            if calib_result is None:
                raise RuntimeError("Run Tab 2 calibration first.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Calibration missing", str(e)); return

        # Sample indices: span the anchor range
        idx_min = min(anchors); idx_max = max(anchors)
        sample_indices = list(range(idx_min, idx_max + 1))

        cfg = {
            "parametrization": self._drift_param.currentText(),
            "n_knots": self._drift_knots.value(),
            "bayesian_sigma": self._drift_bayesian.isChecked(),
        }
        self._drift_status_lbl.setText("Fitting…")
        self._drift_fit_btn.setEnabled(False)
        self._log.append("─" * 40 + f"\nFitting drift trajectory ({len(anchors)} anchors)…")
        self._drift_worker = DriftWorker(
            calib_result, anchors, sample_indices, cfg, parent=self)
        self._drift_worker.log_line.connect(self._log.append)
        self._drift_worker.finished.connect(self._on_drift_done)
        self._drift_worker.failed.connect(self._on_drift_fail)
        self._drift_worker.start()

    def _on_drift_done(self, traj):
        self._drift_traj = traj
        self._drift_fit_btn.setEnabled(True)
        Lsd_range = f"{traj.Lsd_t.min()/1000:.3f}–{traj.Lsd_t.max()/1000:.3f} mm"
        self._drift_status_lbl.setText(f"Trajectory ready: Lsd {Lsd_range}  ({len(traj.frame_indices)} knots)")
        self._log.append(f"[drift] trajectory fitted  Lsd {Lsd_range}")

    def _on_drift_fail(self, msg):
        self._drift_fit_btn.setEnabled(True)
        self._drift_status_lbl.setText("Fitting failed")
        show_error(self, "Drift fitting failed", msg, log=self._log, log_prefix="\n[drift] ERROR:\n")
