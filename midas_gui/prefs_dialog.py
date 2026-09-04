"""Preferences dialog — edit local defaults, materials, calibrants, menus and
algorithms; save them as your local defaults, load/save a JSON config, or reset to
the shipped defaults.

Tables are pre-filled with the current *effective* values (shipped defaults plus
whatever your config already sets), so you add / remove / modify from a complete
list. Saving writes the per-user config file (:mod:`midas_gui.settings`); because
widgets bake values at construction, changes apply on the next launch.
"""
from __future__ import annotations

import json

from PyQt5 import QtWidgets

from midas_gui import settings
from midas_gui import constants as C

_PATH_ROWS = [
    ("calibrant_tif", "Calibrant TIFF:", "file"),
    ("calibrant_h5", "Calibrant HDF5:", "file"),
    ("nickel_h5", "Sample HDF5:", "file"),
    ("nickel_dir", "Sample folder:", "dir"),
    ("nickel_frame0", "Sample frame:", "file"),
    ("calib_file", "Calibration file:", "file"),
    ("pdf_iq_file", "PDF I(Q) file:", "file"),
    ("pdf_calib", "PDF calibration:", "file"),
]
_MAT_HEADERS = ["name", "a", "b", "c", "α", "β", "γ", "SG"]
_DEV_HEADERS = ["name", "prefix", "PVA suffix"]


def _effective_cfg() -> dict:
    """Snapshot the current effective defaults from ``constants`` into the schema."""
    mats = {n: dict(m) for n, m in C.MATERIALS.items()}
    cals = {}
    for n in C.CALIBRANTS:
        m = C._LATT.get(n)
        if m is None and n in C._LC:
            a, b, c, al, be, ga = C._LC[n]
            m = {"a": a, "b": b, "c": c, "alpha": al, "beta": be, "gamma": ga,
                 "sg": C._SG.get(n, 225)}
        if m:
            cals[n] = dict(m)
    return {
        "geometry": {
            "wavelength_A": C.DEFAULT_WAVELENGTH, "pixel_um": C.DEFAULT_PIXEL_UM,
            "lsd_um": C.DEFAULT_LSD_UM, "bc_y": C.DEFAULT_BC_Y, "bc_z": C.DEFAULT_BC_Z,
            "pixel_presets": [list(p) for p in C.PIXEL_PRESETS],
            "k_edge_foils": [list(k) for k in C.K_EDGE_FOILS],
        },
        "viewer_steps": {
            "wavelength": C.DEFAULT_STEP_WAVELENGTH, "two_theta": C.DEFAULT_STEP_TWO_THETA,
            "lsd_mm": C.DEFAULT_STEP_LSD_MM, "pixel": C.DEFAULT_STEP_PIXEL,
            "bc": C.DEFAULT_STEP_BC, "tilt": C.DEFAULT_STEP_TILT,
        },
        "materials": mats,
        "calibrants": cals,
        "devices": [dict(d) for d in C.DEVICES],
        "paths": {
            "calibrant_tif": C.DEFAULT_CALIBRANT_TIF, "calibrant_h5": C.DEFAULT_CALIBRANT_H5,
            "nickel_h5": C.DEFAULT_NICKEL_H5, "nickel_dir": C.DEFAULT_NICKEL_DIR,
            "nickel_frame0": C.DEFAULT_NICKEL_FRAME0, "calib_file": C.DEFAULT_CALIB_FILE,
            "pdf_iq_file": C.DEFAULT_PDF_IQ_FILE, "pdf_calib": C.DEFAULT_PDF_CALIB,
        },
        "ui": {
            "integration_kernel": C.DEFAULT_KERNEL, "calibration_pipeline": C.DEFAULT_PIPELINE,
            "output_format": C.DEFAULT_OUTPUT_FORMAT, "azimuthal_method": C.DEFAULT_ERROR_MODEL,
            "plot_theme": C.DEFAULT_COLORMAP, "visible_tabs": list(C.DEFAULT_VISIBLE_TABS),
            "ui_scale": C.DEFAULT_UI_SCALE,
        },
    }


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent or main_window)
        self._mw = main_window
        self.setWindowTitle("MIDAS GUI — Preferences")
        self.setMinimumSize(620, 560)

        root = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "Your local defaults. Lists below start from the shipped defaults — "
            "add / remove / modify as you like. Changes apply on the next launch.")
        info.setWordWrap(True); info.setStyleSheet("color:#aaa;font-size:11px")
        root.addWidget(info)

        # ── profile selector ─────────────────────────────────────────────
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(QtWidgets.QLabel("Profile:"))
        self._profile_combo = QtWidgets.QComboBox()
        self._profile_combo.activated.connect(self._on_profile_switch)
        prow.addWidget(self._profile_combo, 1)
        for label, slot in (
            ("New…", self._new_profile), ("Duplicate…", self._duplicate_profile),
            ("Rename…", self._rename_profile), ("Delete", self._delete_profile),
        ):
            b = QtWidgets.QPushButton(label); b.clicked.connect(slot)
            prow.addWidget(b)
        root.addLayout(prow)
        self._refresh_profile_combo()

        self._tabs = QtWidgets.QTabWidget(); root.addWidget(self._tabs, 1)
        self._build_geometry_tab()
        self._build_viewer_steps_tab()
        self._build_paths_tab()
        self._mat_table = self._build_table_tab("Materials", _MAT_HEADERS)
        self._cal_table = self._build_table_tab("Calibrants", _MAT_HEADERS)
        self._build_devices_tab()
        self._build_menus_tab()
        self._build_algorithms_tab()
        self._build_tabs_tab()
        self._build_display_tab()

        # action row
        arow = QtWidgets.QHBoxLayout()
        for label, tip, slot in (
            ("Copy from Data Viewer", "Copy the Data Viewer's live λ / pixel / Lsd / "
             "beam centre into the geometry fields.", self._capture_state),
            ("Load config (JSON)…", "Load a JSON config file into this form.",
             self._load_json),
            ("Save config to JSON…", "Export the current form to a JSON file to share.",
             self._save_json),
            ("Reset to shipped defaults", "Discard your local config and return to the "
             "shipped defaults.", self._reset),
        ):
            b = QtWidgets.QPushButton(label); b.setToolTip(tip); b.clicked.connect(slot)
            arow.addWidget(b)
        arow.addStretch(1)
        root.addLayout(arow)

        self._loc_label = QtWidgets.QLabel()
        self._loc_label.setStyleSheet("color:#888;font-size:10px"); self._loc_label.setWordWrap(True)
        root.addWidget(self._loc_label)
        self._update_profile_label()

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Save).setText("Save as my defaults")
        bb.accepted.connect(self._save); bb.rejected.connect(self.reject)
        root.addWidget(bb)

        self._populate(_effective_cfg())

    # ── tab builders ───────────────────────────────────────────────────
    def _build_geometry_tab(self):
        w = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(w)
        self._g_wl = QtWidgets.QLineEdit(); self._g_px = QtWidgets.QLineEdit()
        self._g_lsd = QtWidgets.QLineEdit(); self._g_bcy = QtWidgets.QLineEdit()
        self._g_bcz = QtWidgets.QLineEdit()
        f.addRow("Wavelength λ (Å):", self._g_wl)
        f.addRow("Pixel size (µm):", self._g_px)
        f.addRow("Lsd (mm):", self._g_lsd)
        f.addRow("Beam centre y (px):", self._g_bcy)
        f.addRow("Beam centre z (px):", self._g_bcz)
        self._tabs.addTab(w, "Geometry")

    def _build_viewer_steps_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            "Amount each field changes per click of its up/down arrows, in the "
            "Data Viewer's Ring simulation card."))
        f = QtWidgets.QFormLayout()

        def step_spin(dec, lo=1e-6, hi=1e6):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi); s.setDecimals(dec); s.setFixedWidth(110)
            return s

        self._st_wl   = step_spin(5)
        self._st_2t   = step_spin(3)
        self._st_lsd  = step_spin(4)
        self._st_px   = step_spin(3)
        self._st_bc   = step_spin(3)
        self._st_tilt = step_spin(4)
        f.addRow("λ step (Å):", self._st_wl)
        f.addRow("max 2θ step (°):", self._st_2t)
        f.addRow("Lsd step (mm):", self._st_lsd)
        f.addRow("Pixel size step (µm):", self._st_px)
        f.addRow("BC_y / BC_z step (px):", self._st_bc)
        f.addRow("ty / tz step (°):", self._st_tilt)
        v.addLayout(f)
        v.addStretch(1)
        self._tabs.addTab(w, "Data Viewer")

    def _build_paths_tab(self):
        w = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(w)
        self._paths = {}
        for key, label, kind in _PATH_ROWS:
            ed = QtWidgets.QLineEdit(); self._paths[key] = ed
            b = QtWidgets.QPushButton("…"); b.setFixedWidth(28)
            b.clicked.connect(lambda _=0, e=ed, k=kind: self._browse_path(e, k))
            r = QtWidgets.QHBoxLayout(); r.setSpacing(4); r.addWidget(ed); r.addWidget(b)
            f.addRow(label, r)
        self._tabs.addTab(w, "Paths")

    def _build_table_tab(self, title, headers):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setColumnWidth(0, 150)
        v.addWidget(table, 1)
        br = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add")
        add.clicked.connect(lambda: self._add_row(table, headers))
        rem = QtWidgets.QPushButton("Remove selected")
        rem.clicked.connect(lambda: self._remove_rows(table))
        br.addWidget(add); br.addWidget(rem); br.addStretch(1)
        v.addLayout(br)
        self._tabs.addTab(w, title)
        return table

    def _build_devices_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            "Devices offered in the Data Viewer's Live Data PV dropdown. The live "
            "PV is built as prefix + PVA suffix, e.g. 20IDFF: + Pva1:Image → "
            "20IDFF:Pva1:Image. You can still type any other PV by hand there."))
        table = QtWidgets.QTableWidget(0, len(_DEV_HEADERS))
        table.setHorizontalHeaderLabels(_DEV_HEADERS)
        table.setColumnWidth(0, 150)
        v.addWidget(table, 1)
        br = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add")
        add.clicked.connect(lambda: self._add_row(table, _DEV_HEADERS))
        rem = QtWidgets.QPushButton("Remove selected")
        rem.clicked.connect(lambda: self._remove_rows(table))
        br.addWidget(add); br.addWidget(rem); br.addStretch(1)
        v.addLayout(br)
        self._tabs.addTab(w, "Devices")
        self._dev_table = table

    def _build_menus_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel("Pixel-size presets (clickable 'px' menu)"))
        self._px_table = QtWidgets.QTableWidget(0, 2)
        self._px_table.setHorizontalHeaderLabels(["label", "µm"])
        v.addWidget(self._px_table, 1)
        r1 = QtWidgets.QHBoxLayout()
        a1 = QtWidgets.QPushButton("Add"); a1.clicked.connect(
            lambda: self._add_row(self._px_table, ["label", "µm"]))
        d1 = QtWidgets.QPushButton("Remove selected"); d1.clicked.connect(
            lambda: self._remove_rows(self._px_table))
        r1.addWidget(a1); r1.addWidget(d1); r1.addStretch(1); v.addLayout(r1)
        v.addWidget(QtWidgets.QLabel("K-edge foils (clickable 'λ' menu)"))
        self._ke_table = QtWidgets.QTableWidget(0, 2)
        self._ke_table.setHorizontalHeaderLabels(["element", "keV"])
        v.addWidget(self._ke_table, 1)
        r2 = QtWidgets.QHBoxLayout()
        a2 = QtWidgets.QPushButton("Add"); a2.clicked.connect(
            lambda: self._add_row(self._ke_table, ["element", "keV"]))
        d2 = QtWidgets.QPushButton("Remove selected"); d2.clicked.connect(
            lambda: self._remove_rows(self._ke_table))
        r2.addWidget(a2); r2.addWidget(d2); r2.addStretch(1); v.addLayout(r2)
        self._tabs.addTab(w, "Menus")

    def _build_algorithms_tab(self):
        w = QtWidgets.QWidget(); f = QtWidgets.QFormLayout(w)
        self._ui_kernel = QtWidgets.QComboBox()
        for label, key in C.KERNELS.items():
            self._ui_kernel.addItem(label, key)
        self._ui_pipe = QtWidgets.QComboBox()
        for label, key, enabled in C.PIPELINES:
            self._ui_pipe.addItem(label, key)
            if not enabled:
                self._ui_pipe.model().item(self._ui_pipe.count() - 1).setEnabled(False)
        self._ui_fmt = QtWidgets.QComboBox()
        for label in C.OUTPUT_FORMATS:
            self._ui_fmt.addItem(label, C.OUTPUT_FORMATS[label])
        self._ui_err = QtWidgets.QComboBox(); self._ui_err.addItems(C.ERROR_MODELS)
        self._ui_cmap = QtWidgets.QComboBox(); self._ui_cmap.addItems(C.COLORMAPS)
        f.addRow("Calibration pipeline:", self._ui_pipe)
        f.addRow("Integration kernel:", self._ui_kernel)
        f.addRow("Output format:", self._ui_fmt)
        f.addRow("Error model:", self._ui_err)
        f.addRow("Colormap / theme:", self._ui_cmap)
        self._tabs.addTab(w, "Algorithms")

    def _build_tabs_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            "Choose which tabs are visible. Data Viewer, Mask, Calibrate and Batch "
            "Integrate are always shown. Changes apply immediately."))
        self._tab_checks = {}
        for name in C.ALWAYS_TABS:
            cb = QtWidgets.QCheckBox(name); cb.setChecked(True); cb.setEnabled(False)
            v.addWidget(cb); self._tab_checks[name] = cb
        for name in C.OPTIONAL_TABS:
            cb = QtWidgets.QCheckBox(name)
            v.addWidget(cb); self._tab_checks[name] = cb
        v.addStretch(1)
        self._tabs.addTab(w, "Tabs")

    def _build_display_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            "Interface scale — a whole-application zoom for the layout AND fonts, for "
            "use on HiDPI / 4K monitors where the default looks too small."))
        form = QtWidgets.QFormLayout()
        self._ui_scale = QtWidgets.QDoubleSpinBox()
        self._ui_scale.setRange(0.5, 4.0); self._ui_scale.setSingleStep(0.05)
        self._ui_scale.setDecimals(2); self._ui_scale.setSuffix("  ×")
        self._ui_scale.setFixedWidth(110)
        form.addRow("Interface scale:", self._ui_scale)
        v.addLayout(form)
        presets = QtWidgets.QHBoxLayout(); presets.setSpacing(6)
        presets.addWidget(QtWidgets.QLabel("Presets:"))
        for label, val in (("100% (1080p)", 1.0), ("125%", 1.25),
                           ("150% (1440p)", 1.5), ("200% (4K)", 2.0)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=0, x=val: self._ui_scale.setValue(x))
            presets.addWidget(b)
        presets.addStretch(1)
        v.addLayout(presets)
        note = QtWidgets.QLabel(
            "Applied via Qt's QT_SCALE_FACTOR, so the whole layout scales uniformly. "
            "Takes effect after a restart — use “Save as my defaults”, then relaunch "
            "(you'll be offered a restart if the scale changed).")
        note.setWordWrap(True); note.setStyleSheet("color:#aaa;font-size:10px")
        v.addWidget(note)
        v.addStretch(1)
        self._tabs.addTab(w, "Display")

    # ── table helpers ──────────────────────────────────────────────────
    def _add_row(self, table, headers, values=None):
        r = table.rowCount(); table.insertRow(r)
        values = values or [""] * table.columnCount()
        for c in range(table.columnCount()):
            val = values[c] if c < len(values) else ""
            table.setItem(r, c, QtWidgets.QTableWidgetItem("" if val == "" else str(val)))

    def _remove_rows(self, table):
        for r in sorted({i.row() for i in table.selectedIndexes()}, reverse=True):
            table.removeRow(r)

    def _mat_dict(self, table) -> dict:
        out = {}
        for r in range(table.rowCount()):
            def cell(c):
                it = table.item(r, c); return it.text().strip() if it else ""
            name = cell(0)
            if not name:
                continue
            try:
                out[name] = {"a": float(cell(1)), "b": float(cell(2)), "c": float(cell(3)),
                             "alpha": float(cell(4)), "beta": float(cell(5)),
                             "gamma": float(cell(6)), "sg": int(float(cell(7)))}
            except Exception:
                pass
        return out

    def _device_list(self, table):
        out = []
        for r in range(table.rowCount()):
            def cell(c):
                it = table.item(r, c); return it.text().strip() if it else ""
            name = cell(0)
            if not name:
                continue
            out.append({"name": name, "prefix": cell(1), "pva_suffix": cell(2)})
        return out

    def _pairs(self, table):
        out = []
        for r in range(table.rowCount()):
            a = table.item(r, 0); b = table.item(r, 1)
            an = a.text().strip() if a else ""
            try:
                if an:
                    out.append([an, float(b.text())])
            except Exception:
                pass
        return out

    # ── populate / assemble ────────────────────────────────────────────
    def _populate(self, cfg):
        geo = cfg.get("geometry", {})
        self._g_wl.setText(str(geo.get("wavelength_A", "")))
        self._g_px.setText(str(geo.get("pixel_um", "")))
        self._g_lsd.setText(self._um_to_mm_text(geo.get("lsd_um", "")))   # store µm, show mm
        self._g_bcy.setText(str(geo.get("bc_y", "")))
        self._g_bcz.setText(str(geo.get("bc_z", "")))
        steps = cfg.get("viewer_steps", {}) or {}
        self._st_wl.setValue(float(steps.get("wavelength", C.DEFAULT_STEP_WAVELENGTH) or C.DEFAULT_STEP_WAVELENGTH))
        self._st_2t.setValue(float(steps.get("two_theta", C.DEFAULT_STEP_TWO_THETA) or C.DEFAULT_STEP_TWO_THETA))
        self._st_lsd.setValue(float(steps.get("lsd_mm", C.DEFAULT_STEP_LSD_MM) or C.DEFAULT_STEP_LSD_MM))
        self._st_px.setValue(float(steps.get("pixel", C.DEFAULT_STEP_PIXEL) or C.DEFAULT_STEP_PIXEL))
        self._st_bc.setValue(float(steps.get("bc", C.DEFAULT_STEP_BC) or C.DEFAULT_STEP_BC))
        self._st_tilt.setValue(float(steps.get("tilt", C.DEFAULT_STEP_TILT) or C.DEFAULT_STEP_TILT))
        paths = cfg.get("paths", {})
        for key, ed in self._paths.items():
            ed.setText(str(paths.get(key, "") or ""))
        # "dspacing"-kind materials (e.g. AgBH) have no a/b/c/SG to show in
        # this lattice-only table — they're only editable from the Ring
        # Simulation "+ Add material" dialog for now, so skip them here
        # rather than rendering broken "None" cells.
        self._mat_table.setRowCount(0)
        for name, m in (cfg.get("materials", {}) or {}).items():
            if m.get("kind") == "dspacing":
                continue
            self._add_row(self._mat_table, _MAT_HEADERS,
                          [name, m.get("a"), m.get("b"), m.get("c"),
                           m.get("alpha"), m.get("beta"), m.get("gamma"), m.get("sg")])
        self._cal_table.setRowCount(0)
        for name, m in (cfg.get("calibrants", {}) or {}).items():
            if m.get("kind") == "dspacing":
                continue
            self._add_row(self._cal_table, _MAT_HEADERS,
                          [name, m.get("a"), m.get("b"), m.get("c"),
                           m.get("alpha"), m.get("beta"), m.get("gamma"), m.get("sg")])
        self._dev_table.setRowCount(0)
        for d in cfg.get("devices", []) or []:
            self._add_row(self._dev_table, _DEV_HEADERS,
                          [d.get("name", ""), d.get("prefix", ""), d.get("pva_suffix", "")])
        self._px_table.setRowCount(0)
        for p in geo.get("pixel_presets", []) or []:
            self._add_row(self._px_table, ["label", "µm"], list(p))
        self._ke_table.setRowCount(0)
        for k in geo.get("k_edge_foils", []) or []:
            self._add_row(self._ke_table, ["element", "keV"], list(k))
        ui = cfg.get("ui", {})
        self._select(self._ui_kernel, ui.get("integration_kernel"), by_data=True)
        self._select(self._ui_pipe, ui.get("calibration_pipeline"), by_data=True)
        self._select(self._ui_fmt, ui.get("output_format"), by_data=True)
        self._select(self._ui_err, ui.get("azimuthal_method"))
        self._select(self._ui_cmap, ui.get("plot_theme"))
        visible = ui.get("visible_tabs")
        if isinstance(visible, list):
            vis = set(visible)
            for name, cb in self._tab_checks.items():
                if cb.isEnabled():           # skip the always-on (disabled) boxes
                    cb.setChecked(name in vis)
        try:
            self._ui_scale.setValue(float(ui.get("ui_scale", 1.0) or 1.0))
        except Exception:
            self._ui_scale.setValue(1.0)

    @staticmethod
    def _um_to_mm_text(um) -> str:
        """Format a µm value as a mm string for display (blank passes through)."""
        if um in (None, ""):
            return ""
        try:
            return f"{float(um) / 1000.0:g}"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _select(combo, value, by_data=False):
        if value is None:
            return
        i = combo.findData(value) if by_data else combo.findText(str(value))
        if i >= 0:
            combo.setCurrentIndex(i)

    def _assemble(self) -> dict:
        def num(ed):
            t = ed.text().strip()
            return float(t) if t else None
        geo = {}
        for key, ed in (("wavelength_A", self._g_wl), ("pixel_um", self._g_px),
                        ("lsd_um", self._g_lsd), ("bc_y", self._g_bcy), ("bc_z", self._g_bcz)):
            v = num(ed)
            if v is not None:
                geo[key] = v
        if geo.get("lsd_um") is not None:
            geo["lsd_um"] *= 1000.0          # mm display → µm stored
        geo["pixel_presets"] = self._pairs(self._px_table)
        geo["k_edge_foils"] = self._pairs(self._ke_table)
        viewer_steps = {
            "wavelength": self._st_wl.value(), "two_theta": self._st_2t.value(),
            "lsd_mm": self._st_lsd.value(), "pixel": self._st_px.value(),
            "bc": self._st_bc.value(), "tilt": self._st_tilt.value(),
        }
        paths = {k: ed.text().strip() for k, ed in self._paths.items() if ed.text().strip()}
        # "materials" here REPLACES the whole live MATERIALS dict (see
        # constants._apply). The table can't represent "dspacing"-kind
        # entries (e.g. AgBH — see _populate's skip above), so carry those
        # forward from the current live dict or a Save would silently drop
        # them from the user's profile.
        materials = self._mat_dict(self._mat_table)
        for name, m in C.MATERIALS.items():
            if m.get("kind") == "dspacing" and name not in materials:
                materials[name] = dict(m)
        return {
            "geometry": geo,
            "viewer_steps": viewer_steps,
            "materials": materials,
            "calibrants": self._mat_dict(self._cal_table),
            "devices": self._device_list(self._dev_table),
            "paths": paths,
            "ui": {
                "calibration_pipeline": self._ui_pipe.currentData(),
                "integration_kernel": self._ui_kernel.currentData(),
                "output_format": self._ui_fmt.currentData(),
                "azimuthal_method": self._ui_err.currentText(),
                "plot_theme": self._ui_cmap.currentText(),
                "visible_tabs": [name for name, cb in self._tab_checks.items()
                                 if cb.isChecked()],
                "ui_scale": round(self._ui_scale.value(), 3),
            },
        }

    # ── actions ────────────────────────────────────────────────────────
    def _browse_path(self, edit, kind):
        if kind == "dir":
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file")
        if p:
            edit.setText(p)

    # ── profile management ────────────────────────────────────────────
    def _update_profile_label(self):
        self._loc_label.setText(
            f"Profile '{settings.active_profile()}': {settings.user_config_path()}")

    def _refresh_profile_combo(self):
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItems(settings.list_profiles())
        idx = self._profile_combo.findText(settings.active_profile())
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.blockSignals(False)

    def _reload_after_profile_change(self):
        """Common tail for every profile-management action: refresh live
        constants, repopulate the form, sync the combo/label, and nudge tab
        visibility live (other settings still need a restart)."""
        C.reload_from_config()
        self._populate(_effective_cfg())
        self._refresh_profile_combo()
        self._update_profile_label()
        try:
            if self._mw is not None and hasattr(self._mw, "on_profile_changed"):
                self._mw.on_profile_changed()
        except Exception:
            pass

    def _on_profile_switch(self, idx: int):
        name = self._profile_combo.itemText(idx)
        if name == settings.active_profile():
            return
        if QtWidgets.QMessageBox.question(
                self, "Switch profile",
                f"Switch to profile '{name}'? Any unsaved edits in this dialog "
                "will be discarded.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes) != QtWidgets.QMessageBox.Yes:
            self._refresh_profile_combo()
            return
        try:
            settings.set_active_profile(name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Switch failed", str(e))
            self._refresh_profile_combo()
            return
        self._reload_after_profile_change()
        QtWidgets.QMessageBox.information(
            self, "Profile switched",
            f"Now using profile '{name}'. Most values apply immediately; a few "
            "(e.g. spin-box step sizes) only take effect after a restart.")

    def _new_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New profile", "Profile name:")
        if not ok or not name.strip():
            return
        try:
            settings.create_profile(name.strip(), seed_cfg=C.shipped_defaults())
            settings.set_active_profile(name.strip())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Create failed", str(e)); return
        self._reload_after_profile_change()

    def _duplicate_profile(self):
        base = self._profile_combo.currentText()
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Duplicate profile",
            f"New profile name (copies the current form, based on '{base}'):")
        if not ok or not name.strip():
            return
        try:
            settings.create_profile(name.strip(), seed_cfg=self._assemble())
            settings.set_active_profile(name.strip())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Duplicate failed", str(e)); return
        self._reload_after_profile_change()

    def _rename_profile(self):
        old = self._profile_combo.currentText()
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename profile", "New name:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        try:
            settings.rename_profile(old, new.strip())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Rename failed", str(e)); return
        self._refresh_profile_combo()
        self._update_profile_label()

    def _delete_profile(self):
        name = self._profile_combo.currentText()
        if QtWidgets.QMessageBox.question(
                self, "Delete profile", f"Delete profile '{name}'? This cannot be undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        try:
            settings.delete_profile(name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Delete failed", str(e)); return
        self._reload_after_profile_change()

    def _capture_state(self):
        try:
            g = self._mw._view_tab.get_geometry()
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Unavailable",
                                          "Could not read the Data Viewer geometry.")
            return
        self._g_wl.setText(str(g.get("wavelength_A", "")))
        self._g_px.setText(str(g.get("pxY", "")))
        self._g_lsd.setText(self._um_to_mm_text(g.get("Lsd", "")))   # µm → mm display
        self._g_bcy.setText(str(g.get("BC_y", "")))
        self._g_bcz.setText(str(g.get("BC_z", "")))

    def _load_json(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load config (JSON)", "", "JSON (*.json);;All (*)")
        if not p:
            return
        try:
            cfg = json.loads(open(p).read())
            assert isinstance(cfg, dict)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e)); return
        # merge over the effective snapshot so missing sections keep sensible values
        base = _effective_cfg()
        for k in ("geometry", "viewer_steps", "materials", "calibrants", "devices", "paths", "ui"):
            if k in cfg:
                base[k] = cfg[k]
        self._populate(base)
        QtWidgets.QMessageBox.information(
            self, "Loaded", "Config loaded into the form. Click 'Save as my defaults' to keep it.")

    def _save_json(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save config to JSON", "midas_gui_config.json", "JSON (*.json)")
        if not p:
            return
        try:
            with open(p, "w") as fh:
                json.dump(self._assemble(), fh, indent=2)
            QtWidgets.QMessageBox.information(self, "Saved", f"Config written to:\n{p}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _reset(self):
        if QtWidgets.QMessageBox.question(
                self, "Reset to shipped defaults",
                "Discard your local config and return to the shipped defaults?\n"
                "(Takes effect on the next launch.)") != QtWidgets.QMessageBox.Yes:
            return
        try:
            settings.reset_user_config()
            QtWidgets.QMessageBox.information(self, "Reset",
                                             "Local config removed. Restart to apply.")
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Reset failed", str(e))

    def _save(self):
        cfg = self._assemble()
        try:
            path = settings.save_user_config(cfg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e)); return
        # Tab visibility can apply live (all tabs already exist); other settings
        # (baked into widgets at construction) still need a restart.
        applied_live = False
        try:
            if self._mw is not None and hasattr(self._mw, "apply_tab_visibility"):
                self._mw.apply_tab_visibility(cfg["ui"]["visible_tabs"])
                applied_live = True
        except Exception:
            pass
        note = ("Tab visibility applied now; restart the GUI for other changes to apply."
                if applied_live else "Restart the GUI to apply.")
        # If the interface scale changed, offer an immediate relaunch (it's a
        # startup-only setting — QT_SCALE_FACTOR is read before the QApplication).
        scale_changed = abs(float(cfg["ui"].get("ui_scale", 1.0))
                            - float(getattr(C, "DEFAULT_UI_SCALE", 1.0) or 1.0)) > 1e-6
        if scale_changed and self._mw is not None and hasattr(self._mw, "restart_app"):
            self.accept()
            self._mw._offer_restart(
                f"Saved. Interface scale set to {cfg['ui']['ui_scale']:g}×.")
            return
        QtWidgets.QMessageBox.information(
            self, "Saved", f"Saved as your defaults:\n{path}\n\n{note}")
        self.accept()
