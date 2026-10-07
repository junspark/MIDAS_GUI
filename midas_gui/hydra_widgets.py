"""UI building blocks for the Data Viewer tab's Hydra (4-panel GE) mode.

``HydraModeRibbon`` is the leftmost strip of the whole Data Viewer tab —
it switches the tab between the existing single-detector view and the new
Hydra view. ``HydraLoaderPanel`` and ``HydraDetectorToolbar`` are the
Hydra page's own loader and image-toolbar widgets. A multi-curve profile
viewer is added to this module in a later phase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from midas_gui.helpers import (_NoScrollSpinBox, _NoScrollComboBox, hydra_siblings,
                         hydra_panel_index, is_h5, list_h5_datasets, source_kind,
                         detect_geometry_from_path, warn_if_path_missing)
from midas_gui.workers import FieldAverageWorker, ProjectionWorker
from midas_gui import hydra
from midas_gui import style as S
from midas_gui.dialogs import BrowseFilesDialog
from midas_gui.widgets import _convert_radial, _XUNIT_LABEL, MaskSelector, _fmt_source_desc


class _VerticalToggleButton(QtWidgets.QAbstractButton):
    """Checkable button whose label is painted rotated 90° (reads
    bottom-to-top), for a narrow vertical mode-switch ribbon. Custom-painted
    (rather than a styled QToolButton) so rotated text stays legible under
    the app's global stylesheet — mirrors ``roi_tools._VerticalLabel``."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = text
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedWidth(32)
        self.setAttribute(QtCore.Qt.WA_Hover, True)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(32, 120)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self.isChecked():
            bg = "#2e7d32"
        elif self.underMouse():
            bg = "#333333"
        else:
            bg = "#1c1c1c"
        painter.fillRect(self.rect(), QtGui.QColor(bg))
        painter.setPen(QtGui.QColor("#f5f5f5"))
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(12)
        painter.setFont(font)
        painter.translate(0, self.height())
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(0, 0, self.height(), self.width()),
                          QtCore.Qt.AlignCenter, self._text)
        painter.end()

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)


class HydraModeRibbon(QtWidgets.QWidget):
    """Leftmost vertical strip of the Data Viewer tab. Two exclusive modes:
    "Single detector" (today's existing view, unchanged) and "Hydra" (the
    new 4-panel GE detector view)."""

    modeChanged = QtCore.pyqtSignal(str)   # "single" | "hydra"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(32)
        self.setObjectName("hydraModeRibbon")
        self.setStyleSheet(
            "QWidget#hydraModeRibbon { background-color: #1c1c1c; "
            "border-right: 1px solid #444; }")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)
        self._single_btn = _VerticalToggleButton("Single detector")
        self._single_btn.setToolTip("Single-detector data viewer")
        self._hydra_btn = _VerticalToggleButton("Hydra")
        self._hydra_btn.setToolTip("Hydra 4-panel GE detector viewer")
        self._single_btn.setChecked(True)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.addButton(self._single_btn)
        self._group.addButton(self._hydra_btn)
        layout.addWidget(self._single_btn)
        layout.addWidget(self._hydra_btn)
        layout.addStretch(1)
        self._single_btn.toggled.connect(self._on_toggled)
        self._hydra_btn.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool):
        if not checked:
            return
        self.modeChanged.emit("hydra" if self.sender() is self._hydra_btn else "single")

    def mode(self) -> str:
        return "hydra" if self._hydra_btn.isChecked() else "single"

    def set_mode(self, mode: str):
        (self._hydra_btn if mode == "hydra" else self._single_btn).setChecked(True)

    def set_hydra_enabled(self, enabled: bool):
        """Show/hide the Hydra button (only meaningful at the 1-ID-E
        beamline profile). Falls back to Single detector first if Hydra
        mode was active when it's disabled."""
        if not enabled and self._hydra_btn.isChecked():
            self._single_btn.setChecked(True)
        self._hydra_btn.setVisible(enabled)


class HydraFieldSelector(QtWidgets.QGroupBox):
    """Sibling-aware dark / bright / background field picker for Hydra mode.

    Same role as ``widgets.FieldSelector`` (single-detector tab), but one
    path is entered for any *one* ge panel's field file and the other 3 are
    auto-discovered via ``helpers.hydra_siblings`` — exactly like
    ``HydraLoaderPanel``'s own main data path — then each panel's field is
    reduced to a mean independently (``workers.FieldAverageWorker``, one per
    panel).
    """
    #: emitted whenever any panel's field finishes computing, or the
    #: checkbox is toggled (turning correction on/off is itself a change).
    fieldsReady = QtCore.pyqtSignal()

    def __init__(self, title, parent=None, *, with_mode=False,
                 default_dataset="exchange/data"):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(False)
        self._default_dataset = default_dataset
        self._fields: dict = {}          # panel -> np.ndarray
        self._sibling_paths: dict = {}   # panel -> path
        self._workers: dict = {}         # panel -> FieldAverageWorker (kept alive)
        self._pending: set = set()
        self._registry = None            # DataSourceRegistry, set by set_registry()
        self._exclude_label = None       # owning panel's registry label — skip its own entry
        self._data_path_provider = None  # callable → owning panel's current Data path

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 4); outer.setSpacing(2)
        self._body = QtWidgets.QWidget()
        self._body.setVisible(False)
        self.toggled.connect(self._body.setVisible)
        self.toggled.connect(lambda *_: self.fieldsReady.emit())
        self.toggled.connect(self._on_toggled)
        outer.addWidget(self._body)
        v = QtWidgets.QVBoxLayout(self._body)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)

        self._path_ed = QtWidgets.QLineEdit()
        self._path_ed.setPlaceholderText("Any one ge1-ge4 panel file…")
        self._path_ed.editingFinished.connect(lambda: self._set_path(self._path_ed.text().strip()))
        warn_if_path_missing(self._path_ed, self)
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

        self._status_lbls = {}
        status_row = QtWidgets.QHBoxLayout(); status_row.setSpacing(6)
        for n in (1, 2, 3, 4):
            lbl = QtWidgets.QLabel(f"ge{n}")
            lbl.setStyleSheet(self._status_style("none"))
            status_row.addWidget(lbl)
            self._status_lbls[n] = lbl
        status_row.addStretch(1)
        v.addLayout(status_row)

        self._ds_row = QtWidgets.QWidget()
        dr = QtWidgets.QHBoxLayout(self._ds_row); dr.setContentsMargins(0, 0, 0, 0); dr.setSpacing(4)
        self._ds_combo = _NoScrollComboBox()
        self._ds_combo.setEditable(True); self._ds_combo.setEditText(default_dataset)
        self._ds_combo.currentIndexChanged.connect(self._update_frame_limit)
        dr.addWidget(QtWidgets.QLabel("Dataset:")); dr.addWidget(self._ds_combo, 1)
        self._ds_row.setVisible(False)
        v.addWidget(self._ds_row)

        self._start = _NoScrollSpinBox(); self._start.setRange(0, 0); self._start.setFixedWidth(50)
        self._end = _NoScrollSpinBox(); self._end.setRange(0, 0); self._end.setFixedWidth(50)
        self._end.setToolTip("Last frame index in the mean (inclusive), applied to every panel.")
        self._nfr_lbl = QtWidgets.QLabel("")
        self._nfr_lbl.setStyleSheet("color:#9a9a9a;font-size:10px")
        ir = QtWidgets.QHBoxLayout(); ir.setSpacing(4)
        ir.addWidget(QtWidgets.QLabel("mean")); ir.addWidget(self._start)
        ir.addWidget(QtWidgets.QLabel("–")); ir.addWidget(self._end)
        ir.addWidget(self._nfr_lbl)
        if with_mode:
            self._mode_combo = _NoScrollComboBox()
            self._mode_combo.addItems(["Flat-field divide", "Subtract"])
            self._mode_combo.setFixedWidth(104)
            ir.addStretch(1); ir.addWidget(self._mode_combo)
        else:
            self._mode_combo = None
            ir.addStretch(1)
        v.addLayout(ir)

        self._compute_btn = QtWidgets.QPushButton("Compute field")
        self._compute_btn.clicked.connect(self._compute_all)
        v.addWidget(self._compute_btn)
        self._status = QtWidgets.QLabel("Not computed.")
        self._status.setStyleSheet("color:#9a9a9a;font-size:10px"); self._status.setWordWrap(True)
        v.addWidget(self._status)

    @staticmethod
    def _status_style(state: str) -> str:
        color = {"found": "#8899aa", "done": "#66bb6a", "error": "#ef5350", "none": "#555"}[state]
        weight = "bold" if state in ("done", "error") else "normal"
        return f"color:{color}; font-weight:{weight};"

    def _kind_of(self, path: str) -> str:
        return source_kind(path)

    def _dataset(self) -> str:
        return self._ds_combo.currentText().split("   ")[0].strip() or self._default_dataset

    def set_data_path_provider(self, fn):
        """`fn()` returns the owning panel's current Data path (str). Used to
        prefill this field the first time it's checked, and as the Browse…
        dialog's starting folder while this field has no path of its own —
        mirrors ``widgets.FieldSelector.set_data_path_provider``."""
        self._data_path_provider = fn

    def _prefill_from_data(self, checked: bool):
        """On first check, default this field to the same file/folder as
        the Hydra page's Data source. A no-op if a path is already set."""
        if not checked or self._path_ed.text().strip() or self._data_path_provider is None:
            return
        src = self._data_path_provider()
        if src:
            self._set_path(src)

    def _on_toggled(self, checked: bool):
        """Mirrors ``widgets.FieldSelector._on_toggled``: checking prefills
        from the Data source; unchecking resets the path (and every
        discovered sibling panel) entirely, so a later re-check prefills
        fresh instead of keeping whatever was picked/computed before."""
        if checked:
            self._prefill_from_data(checked)
        else:
            self._reset_path()

    def _reset_path(self):
        """Clear this field back to its empty, uncomputed startup state."""
        self._workers = {}
        self._pending = set()
        self._fields = {}
        self._sibling_paths = {}
        for lbl in self._status_lbls.values():
            lbl.setStyleSheet(self._status_style("none"))
        self._path_ed.setText("")
        self._ds_row.setVisible(False)
        self._ds_combo.setEditText(self._default_dataset)
        self._status.setText("Not computed.")

    def _open_browse_dialog(self):
        # "Multiple files" isn't offered here — the other 3 panels are
        # auto-discovered from one anchor path (helpers.hydra_siblings),
        # which has no way to generalize to an arbitrary per-file pick list.
        start = self._path_ed.text().strip()
        if not start and self._data_path_provider is not None:
            start = self._data_path_provider() or ""
        dlg = BrowseFilesDialog(self, title=f"Select {self.title()}",
                                modes=("file", "folder", "stem"), start_dir=start)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode = dlg.mode()
        if mode == "file":
            paths = dlg.paths()
            if paths:
                self._set_path(paths[0])
        elif mode == "folder":
            self._set_path(dlg.folder())
        else:  # "stem"
            folder, stem = dlg.stem()
            self._set_stem(folder, stem)

    def _set_stem(self, folder: str, stem: str):
        """Every sibling panel's field = every TIFF-family file in that
        panel's own folder starting with `stem` — sibling *folders* are
        discovered via `hydra_siblings` (needs a real existing path), then
        `folder*` is appended per panel; the resulting glob strings flow
        through the existing folder-kind frame mean unchanged."""
        if not folder or not stem:
            return
        self._path_ed.setText(str(Path(folder) / (stem + "*")))
        if hydra_panel_index(folder) is None:
            self._sibling_paths = {}
        else:
            self._sibling_paths = {n: str(Path(p) / (stem + "*"))
                                   for n, p in hydra_siblings(folder).items()}
        for n, lbl in self._status_lbls.items():
            lbl.setStyleSheet(self._status_style("found" if n in self._sibling_paths else "none"))
        self._fields = {}
        self._ds_row.setVisible(False)   # a TIFF-family glob, never HDF5
        self._update_frame_limit()
        if not self._sibling_paths:
            self._status.setText("No sibling panels found for this folder.")
        else:
            self._status.setText(f"Found {len(self._sibling_paths)}/4 panel(s) — not computed.")

    def _set_path(self, path: str):
        if not path:
            return
        self._path_ed.setText(path)
        self._sibling_paths = hydra_siblings(path)
        for n, lbl in self._status_lbls.items():
            lbl.setStyleSheet(self._status_style("found" if n in self._sibling_paths else "none"))
        self._fields = {}
        h5 = is_h5(path)
        self._ds_row.setVisible(h5)
        if h5 and Path(path).exists():
            try:
                items = list_h5_datasets(path)
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
        if not self._sibling_paths:
            self._status.setText("File not found on disk.")
        else:
            self._status.setText(f"Found {len(self._sibling_paths)}/4 panel(s) — not computed.")

    def _count_frames(self, path: str) -> int:
        if not path:
            return 0
        try:
            kind = self._kind_of(path)
            if kind == "hdf5":
                import h5py
                if not Path(path).exists():
                    return 0
                with h5py.File(path, "r") as f:
                    d = f[self._dataset()]
                    return int(d.shape[0]) if d.ndim >= 3 else 1
            if kind == "folder":
                from midas_gui.helpers import _collect_frame_paths
                return len(_collect_frame_paths(path))
            if not Path(path).exists():
                return 0
            if path.lower().endswith((".tif", ".tiff")):
                import tifffile
                with tifffile.TiffFile(path) as tf:
                    return len(tf.pages)
            return 1
        except Exception:
            return 0

    def _update_frame_limit(self, *_):
        """Clamp the mean-range spinboxes to the (first found panel's) frame
        count — every panel of a synchronized Hydra scan has the same
        number of frames, so one reference path is enough."""
        ref = next(iter(self._sibling_paths.values()), self._path_ed.text().strip())
        n = self._count_frames(ref)
        if n <= 0:
            self._nfr_lbl.setText("")
            return
        hi = n - 1
        self._nfr_lbl.setText(f"/ {hi}")
        for sp in (self._start, self._end):
            sp.blockSignals(True); sp.setMaximum(hi); sp.blockSignals(False)
        if self._end.value() == 0 or self._end.value() > hi:
            self._end.blockSignals(True); self._end.setValue(hi); self._end.blockSignals(False)
        if self._start.value() > hi:
            self._start.setValue(hi)

    def _compute_all(self):
        if not self._sibling_paths:
            self._status.setText("Enter a path first."); return
        self._fields = {}
        self._pending = set(self._sibling_paths)
        self._compute_btn.setEnabled(False)
        self._status.setText("Computing…")
        self._workers = {}
        for n, path in self._sibling_paths.items():
            w = FieldAverageWorker(self._kind_of(path), path, self._dataset(),
                                   self._start.value(), self._end.value(), parent=self)
            w.finished.connect(lambda field, n=n: self._on_one_computed(n, field))
            w.failed.connect(lambda err, n=n: self._on_one_failed(n, err))
            self._workers[n] = w
            w.start()

    def _on_one_computed(self, n: int, field):
        self._fields[n] = field
        self._pending.discard(n)
        self._status_lbls[n].setStyleSheet(self._status_style("done"))
        if not self._pending:
            self._compute_btn.setEnabled(True)
            self._status.setText(f"Computed {len(self._fields)}/{len(self._sibling_paths)} panel(s) "
                                 f"— {field.shape} [{float(field.min()):.4g}, {float(field.max()):.4g}]")
        self.fieldsReady.emit()

    def _on_one_failed(self, n: int, err: str):
        self._pending.discard(n)
        self._status_lbls[n].setStyleSheet(self._status_style("error"))
        if not self._pending:
            self._compute_btn.setEnabled(True)
        last = err.strip().splitlines()[-1] if err and err.strip() else "error"
        self._status.setText(f"ge{n} failed: {last}")

    # ── Public accessors ─────────────────────────────────────────

    def field(self, n: int) -> Optional[np.ndarray]:
        return self._fields.get(n) if self.isChecked() else None

    def mode(self) -> str:
        if self._mode_combo is None:
            return "divide"
        return "divide" if self._mode_combo.currentIndex() == 0 else "subtract"

    def set_path(self, path: str):
        """Programmatic equivalent of typing a path and pressing Enter —
        used by Save/Load GUI State restore."""
        self._set_path(path)

    def current_path(self) -> str:
        return self._path_ed.text().strip()

    # ── cross-tab import (data_bridge.DataSourceRegistry) ───────────
    def _field_kind(self) -> str:
        """This selector's type ("dark"/"bright"/"background"), derived from
        its title — mirrors ``widgets.FieldSelector._field_kind``."""
        return self.title().strip().lower()

    def set_registry(self, registry, *, exclude_label=None):
        """Let this selector's "Import from…" menu offer the same-type field
        currently loaded in any *other* tab bound to `registry` — including
        single-detector tabs (the registry doesn't distinguish Hydra from
        single-detector sources, only by `field`)."""
        self._registry = registry
        self._exclude_label = exclude_label

    def describe_source(self, label: str):
        """Export this field's anchor path (if enabled and set) — mirrors
        ``widgets.FieldSelector.describe_source``. Never a `list[str]`:
        Hydra fields have no "Multiple files" mode."""
        if not self.isChecked():
            return None
        raw = self._path_ed.text().strip()
        if not raw:
            return None
        return {"kind": "path", "path": raw,
                "dataset": self._dataset() if is_h5(raw) else None,
                "field": self._field_kind(), "label": label}

    def _populate_import_menu(self):
        menu = self._import_menu
        menu.clear()
        sources = (self._registry.available(field=self._field_kind())
                   if self._registry is not None else [])
        sources = [d for d in sources if d.get("label") != self._exclude_label
                   and not isinstance(d.get("path"), list)]
        if not sources:
            menu.addAction("(nothing loaded elsewhere)").setEnabled(False)
            return
        for desc in sources:
            menu.addAction(_fmt_source_desc(desc), lambda d=desc: self._apply_imported_source(d))

    def _apply_imported_source(self, desc: dict):
        self._set_path(desc["path"])
        if desc.get("dataset"):
            self._ds_combo.setEditText(desc["dataset"])
            self._update_frame_limit()


class HydraLoaderPanel(QtWidgets.QWidget):
    """Left-hand loader for the Hydra page. One path field — point it at any
    single GE panel's file — auto-discovers the other panels via
    ``helpers.hydra_siblings``.

    ``mode`` tailors the Data card, mirroring ``widgets.DataLoaderPanel``'s
    own ``mode`` switch:
      - ``"nav"``    — a frame navigator shared across all panels (they are
        synchronized frames of the same scan) — Data Viewer / Calibrate.
      - ``"stream"`` — a shared frame-range + stride row, no in-memory frame
        navigation, plus one independent :class:`~midas_gui.widgets.MaskSelector`
        per panel (masks are physically panel-specific bad-pixel/beamstop
        maps, unlike the shared data path) — Batch Integrate.
    """

    siblingsChanged = QtCore.pyqtSignal(dict)   # {panel_num: path}, may be {}
    frameChanged = QtCore.pyqtSignal(int)
    fieldsChanged = QtCore.pyqtSignal()         # dark/bright/background changed (any panel)
    projectionChanged = QtCore.pyqtSignal()     # projection turned on/off or (re)computed

    #: Dataset key fixed for v1 — every bundled/real Hydra HDF5 file used so
    #: far shares this convention (see hydra_default_geometry's callers).
    DATASET = "exchange/data"

    def __init__(self, parent=None, *, mode="nav"):
        super().__init__(parent)
        self._mode = mode
        self._siblings: dict = {}
        self._detected: dict = {}        # last auto-detected pxY/wavelength_A (nav mode only)
        self._n_frames = 1
        self._frame = 0
        self._is_projection = False
        self._proj_raw: dict = {}        # panel -> corrected+projected 2-D array
        self._proj_workers: dict = {}    # panel -> ProjectionWorker (kept alive)
        self._proj_pending: set = set()
        self._proj_errors: dict = {}
        self._mask_sels: dict = {}       # panel -> MaskSelector (stream mode only)
        self._registry = None            # DataSourceRegistry, set by bind_registry()
        self._registry_label = ""        # this panel's own label in the registry
        self._build_ui()

    def _build_ui(self):
        lv = QtWidgets.QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)
        card = S.make_card("Hydra data")

        self._path_ed = QtWidgets.QLineEdit()
        self._path_ed.setPlaceholderText("Any one ge1-ge4 panel file…")
        self._path_ed.returnPressed.connect(
            lambda: self._set_path(self._path_ed.text().strip()))
        warn_if_path_missing(self._path_ed, self)
        row = QtWidgets.QHBoxLayout(); row.setSpacing(4)
        row.addWidget(self._path_ed)
        browse = QtWidgets.QToolButton(); browse.setText("⋯"); browse.setFixedWidth(28)
        browse.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(browse)
        menu.addAction("Browse…", self._open_browse_dialog)
        menu.addSeparator()
        self._import_menu = menu.addMenu("Import from…")
        self._import_menu.aboutToShow.connect(self._populate_import_menu)
        browse.setMenu(menu)
        row.addWidget(browse)
        card.body.addLayout(row)

        # One tick per panel: which GE files make up this Hydra array.
        # Checkboxes rather than a dropdown because the four choices are
        # independent and all four should be readable at once. They keep
        # the old labels' colouring, so "found on disk" still reads at a
        # glance -- the tick is a separate question from the colour.
        # True while _set_path/set_state ticks these programmatically, so
        # the per-tick re-emit does not fire once per box mid-rebuild.
        self._loading_panel_ticks = False
        self._status_lbls = {}
        status_row = QtWidgets.QHBoxLayout(); status_row.setSpacing(6)
        for n in (1, 2, 3, 4):
            lbl = QtWidgets.QCheckBox(f"ge{n}")
            lbl.setStyleSheet(self._status_style(False))
            lbl.setEnabled(False)       # until a path says this panel exists
            lbl.setToolTip(
                f"Include ge{n} in the Hydra array. Enabled only when a ge{n} "
                f"file sits beside the one above; untick to leave that panel "
                f"out of the composite, the caking and its per-panel card.")
            lbl.toggled.connect(self._on_panel_selection_changed)
            status_row.addWidget(lbl)
            self._status_lbls[n] = lbl
        status_row.addStretch(1)
        card.body.addLayout(status_row)

        if self._mode == "nav":
            nav_row = QtWidgets.QHBoxLayout(); nav_row.setSpacing(4)
            self._prev_btn = QtWidgets.QPushButton("◀"); self._prev_btn.setFixedWidth(28)
            self._prev_btn.setObjectName("frameNavBtn")
            self._next_btn = QtWidgets.QPushButton("▶"); self._next_btn.setFixedWidth(28)
            self._next_btn.setObjectName("frameNavBtn")
            self._frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self._frame_slider.setObjectName("frameNavSlider")
            self._frame_spin = _NoScrollSpinBox(); self._frame_spin.setFixedWidth(60)
            self._prev_btn.clicked.connect(lambda: self._set_frame(self._frame - 1))
            self._next_btn.clicked.connect(lambda: self._set_frame(self._frame + 1))
            self._frame_slider.valueChanged.connect(
                lambda v: self._set_frame(v, from_widget="slider"))
            self._frame_spin.valueChanged.connect(
                lambda v: self._set_frame(v, from_widget="spin"))
            nav_row.addWidget(self._prev_btn)
            nav_row.addWidget(self._frame_slider, 1)
            nav_row.addWidget(self._frame_spin)
            nav_row.addWidget(self._next_btn)
            card.body.addLayout(nav_row)
            self._set_nav_enabled(False)
        else:   # "stream" — shared frame-range + stride, no navigator
            self._fr_start = _NoScrollSpinBox(); self._fr_start.setRange(0, 999999)
            self._fr_start.setFixedWidth(64)
            self._fr_end = _NoScrollSpinBox(); self._fr_end.setRange(0, 999999)
            self._fr_end.setFixedWidth(64)
            self._fr_end.setToolTip("Last frame (exclusive). 0 = all frames.")
            self._fr_stride = _NoScrollSpinBox(); self._fr_stride.setRange(1, 100000)
            self._fr_stride.setValue(1); self._fr_stride.setFixedWidth(64)
            sf = S.Form()
            sf.row(("start:", self._fr_start), ("end(0=all):", self._fr_end))
            sf.row(("stride:", self._fr_stride))
            card.body.addLayout(sf)

        self._info_lbl = QtWidgets.QLabel("")
        self._info_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._info_lbl.setWordWrap(True)
        card.body.addWidget(self._info_lbl)

        lv.addWidget(card)

        # ── Dark / Bright / Background (sibling-aware, one pick per field) ──
        fld = S.make_card("Dark / Bright / Background")
        self._dark_sel = HydraFieldSelector("Dark", default_dataset="exchange/data_dark")
        self._bright_sel = HydraFieldSelector("Bright", with_mode=True)
        self._bg_sel = HydraFieldSelector("Background")
        for w in (self._dark_sel, self._bright_sel, self._bg_sel):
            w.fieldsReady.connect(self.fieldsChanged)
            w.set_data_path_provider(lambda: self._path_ed.text().strip())
            fld.body.addWidget(w)
        lv.addWidget(fld)

        # ── Mask (stream mode only — independent per panel, no sibling
        #    auto-discovery: bad-pixel/beamstop masks are physically
        #    panel-specific and don't follow a shared naming convention) ──
        if self._mode == "stream":
            mask_card = S.make_card("Mask  (independent per panel)")
            for n in (1, 2, 3, 4):
                sel = MaskSelector()
                sel.setTitle(f"ge{n} mask")
                sel.maskChanged.connect(self.fieldsChanged)
                self._mask_sels[n] = sel
                mask_card.body.addWidget(sel)
            lv.addWidget(mask_card)

        # ── Projection (stack max/sum/mean, all panels + composite) ──
        # Built here (owns the projection logic/state) but NOT added to this
        # panel's own layout — HydraViewerPage places this card at the top
        # of the middle panel instead (see projection_card()), mirroring
        # where the single-detector tab's Projection card sits.
        proj = S.make_card("Projection")
        m_row = QtWidgets.QHBoxLayout(); m_row.setSpacing(8)
        m_row.addWidget(S.LabelRight("Method:"))
        self._proj_method = {}
        for meth in ("max", "sum", "mean"):
            rb = QtWidgets.QRadioButton(meth.capitalize()); m_row.addWidget(rb)
            self._proj_method[meth] = rb
        self._proj_method["max"].setChecked(True)
        m_row.addStretch(1)
        proj.body.addLayout(m_row)
        self._proj_skip = _NoScrollSpinBox(); self._proj_skip.setRange(0, 1000000)
        self._proj_skip.setValue(0); self._proj_skip.setFixedWidth(72)
        self._proj_skip.setToolTip(
            "Ignore this many leading frames (of every panel) before projecting.")
        self._proj_nframes = _NoScrollSpinBox(); self._proj_nframes.setRange(0, 1000000)
        self._proj_nframes.setValue(0); self._proj_nframes.setFixedWidth(72)
        self._proj_nframes.setToolTip(
            "Number of frames to project after Skip frames\n"
            "(0 = use all remaining frames).")
        skip_row = S.Form()
        skip_row.row(("Skip frames:", self._proj_skip), ("N frames:", self._proj_nframes))
        proj.body.addLayout(skip_row)
        self._proj_btn = QtWidgets.QPushButton("Project stack")
        self._proj_btn.setToolTip(
            "Projects every currently-found panel's own frame stack and shows\n"
            "the result in place of the current frame, for that panel AND the\n"
            "Composite view, until 'Back to frames' is clicked.")
        self._proj_btn.clicked.connect(self._project_all)
        self._proj_frame_btn = QtWidgets.QPushButton("Back to frames")
        self._proj_frame_btn.clicked.connect(self._back_to_frames)
        proj.body.addLayout(S.button_grid([self._proj_btn, self._proj_frame_btn], 2))
        self._proj_info = QtWidgets.QLabel("")
        self._proj_info.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._proj_info.setWordWrap(True)
        proj.body.addWidget(self._proj_info)
        proj.setEnabled(False)
        self._proj_grp = proj

        lv.addStretch(1)

    def _selected_siblings(self) -> dict:
        """The panels this array actually uses: found on disk AND ticked.

        ``self._siblings`` stays the full found set so the info line can
        keep reporting how many exist -- excluding a panel must never look
        like a failed detection."""
        return {n: p for n, p in (self._siblings or {}).items()
                if self._status_lbls[n].isChecked()}

    def _on_panel_selection_changed(self, *_args):
        """A ge tick moved. Re-emit the active set: every consumer treats
        siblingsChanged as authoritative (hydra_page._on_siblings_changed,
        _toolbar.set_available), so filtering at this one emit point is the
        whole mechanism -- no consumer needs to know selection exists."""
        if self._loading_panel_ticks or not self._siblings:
            return
        self._refresh_panel_info()
        self.siblingsChanged.emit(self._selected_siblings())

    def _refresh_panel_info(self):
        """The found/selected/frames line. Reports both counts, so a panel
        left out is never mistaken for one that could not be found."""
        found = len(self._siblings or {})
        sel = len(self._selected_siblings())
        if found < 2:
            self._info_lbl.setText(
                "Fewer than 2 Hydra panels found next to this file \u2014 "
                "check the path.")
        elif sel < 2:
            self._info_lbl.setText(
                f"Only {sel} of {found} panels selected \u2014 tick at least 2 "
                f"to form a Hydra array.")
        else:
            extra = "" if sel == found else f"  \u00b7  {sel} selected"
            self._info_lbl.setText(
                f"Found {found}/4 panels{extra}  \u00b7  {self._n_frames} frame(s)")

    @staticmethod
    def _status_style(found: bool) -> str:
        color = "#66bb6a" if found else "#666"
        weight = "bold" if found else "normal"
        return f"color:{color}; font-weight:{weight};"

    def _set_nav_enabled(self, enabled: bool):
        for w in (self._prev_btn, self._next_btn, self._frame_slider, self._frame_spin):
            w.setEnabled(enabled)

    def _open_browse_dialog(self):
        # Single file only — this panel's frame index comes from one anchor
        # file's own internal frame count (hydra.n_frames_in), not separate
        # per-frame files, so folder/multi/stem selection doesn't apply.
        dlg = BrowseFilesDialog(self, title="Select Hydra data", modes=("file",),
                                start_dir=self._path_ed.text().strip())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        paths = dlg.paths()
        if paths:
            self._set_path(paths[0])

    # ── cross-tab data sharing (data_bridge.DataSourceRegistry) ─────
    def bind_registry(self, registry, label: str):
        """Register this panel as an importable data source under `label`
        (e.g. "Data Viewer (Hydra)"), and gain an "Import from…" menu
        listing every other bound panel's currently-loaded data — mirrors
        ``widgets.DataLoaderPanel.bind_registry``."""
        self._registry = registry
        self._registry_label = label
        registry.register(label, self)
        for sel in (self._dark_sel, self._bright_sel, self._bg_sel):
            sel.set_registry(registry, exclude_label=label)

    def describe_source(self):
        """Live snapshot of what this panel offers other panels — its main
        Data anchor path (if found) plus any of its own Dark/Bright/
        Background fields that are enabled and point at a path."""
        out = []
        raw = self._path_ed.text().strip()
        if raw:
            out.append({"kind": "path", "path": raw,
                       "dataset": self.DATASET if is_h5(raw) else None,
                       "field": "data", "label": self._registry_label})
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
        # A single-detector Data field can be an arbitrary multi-file pick,
        # which this panel's one-anchor-path sibling-discovery can't use.
        sources = [d for d in sources if not isinstance(d.get("path"), list)]
        if not sources:
            menu.addAction("(nothing loaded elsewhere)").setEnabled(False)
            return
        for desc in sources:
            menu.addAction(_fmt_source_desc(desc), lambda d=desc: self._apply_imported_source(d))

    def _apply_imported_source(self, desc: dict):
        self._set_path(desc["path"])

    def detected_geometry(self) -> dict:
        """Best-effort pxY/wavelength_A auto-detected from the current anchor
        path (see helpers.detect_geometry_from_path) — always {} in "stream"
        mode (Batch Integrate gets its geometry from a calibration, not the
        raw file)."""
        return dict(self._detected)

    def _set_path(self, path: str):
        if not path:
            return
        self._clear_projection(emit=False)
        self._path_ed.setText(path)
        self._detected = detect_geometry_from_path(path) if self._mode == "nav" else {}
        siblings = hydra_siblings(path)
        self._siblings = siblings
        # A new path re-enables and re-ticks whatever it found, so the
        # default stays exactly the pre-selection behaviour: every panel on
        # disk is used. A tick left over for a panel this path lacks is
        # cleared rather than carried across.
        self._loading_panel_ticks = True
        try:
            for n, lbl in self._status_lbls.items():
                found = n in siblings
                lbl.setStyleSheet(self._status_style(found))
                lbl.setEnabled(found)
                lbl.setChecked(found)
        finally:
            self._loading_panel_ticks = False
        if len(siblings) < 2:
            self._refresh_panel_info()
            # Nav widgets exist in "nav" mode only -- the success path below
            # already guards this call, but this branch never did, so
            # pointing Batch Integrate's Hydra loader at a file with no
            # siblings raised AttributeError instead of showing the
            # message. Pre-existing; found by the selection tests.
            if self._mode == "nav":
                self._set_nav_enabled(False)
            self._n_frames = 1
            self._proj_grp.setEnabled(False)
            self.siblingsChanged.emit({})
            return
        try:
            first_path = next(iter(siblings.values()))
            self._n_frames = hydra.n_frames_in(first_path, self.DATASET)
        except Exception as exc:
            self._info_lbl.setText(f"Could not read frame count: {exc}")
            self._n_frames = 1
        if self._mode == "nav":
            self._frame_slider.blockSignals(True)
            self._frame_slider.setRange(0, max(0, self._n_frames - 1))
            self._frame_slider.blockSignals(False)
            self._frame_spin.blockSignals(True)
            self._frame_spin.setRange(0, max(0, self._n_frames - 1))
            self._frame_spin.blockSignals(False)
            self._set_nav_enabled(self._n_frames > 1)
        self._frame = 0
        self._proj_grp.setEnabled(self._n_frames > 1)
        self._refresh_panel_info()
        self.siblingsChanged.emit(self._selected_siblings())
        self.frameChanged.emit(0)

    def _set_frame(self, i: int, from_widget: Optional[str] = None):
        if self._is_projection:
            self._clear_projection()
        i = max(0, min(int(i), max(0, self._n_frames - 1)))
        changed = (i != self._frame)
        self._frame = i
        if from_widget != "slider":
            self._frame_slider.blockSignals(True); self._frame_slider.setValue(i)
            self._frame_slider.blockSignals(False)
        if from_widget != "spin":
            self._frame_spin.blockSignals(True); self._frame_spin.setValue(i)
            self._frame_spin.blockSignals(False)
        if changed:
            self.frameChanged.emit(i)

    # ── Projection (stack max/sum/mean, all panels + composite) ─

    def _apply_project_style(self, active: bool):
        """Green highlight on "Project stack" while a projection is being
        displayed — mirrors the single-detector tab's
        DataViewerTab._apply_project_style."""
        if active:
            self._proj_btn.setStyleSheet(
                "QPushButton { background:#2e7d32; color:white; font-weight:bold; "
                "border:1px solid #1b5e20; border-radius:4px; padding:4px; }")
        else:
            self._proj_btn.setStyleSheet("")

    def _project_all(self):
        if not self._siblings or self._n_frames <= 1:
            QtWidgets.QMessageBox.warning(self, "No stack", "Projection needs a stack.")
            return
        if self._proj_pending:
            return
        method = next(m for m, b in self._proj_method.items() if b.isChecked())
        skip = self._proj_skip.value()
        nframes = self._proj_nframes.value()
        self._proj_raw = {}
        self._proj_errors = {}
        self._proj_pending = set(self._siblings)
        self._proj_btn.setEnabled(False)
        self._proj_info.setText("Projecting stacks…")
        self._proj_workers = {}
        for n, path in self._siblings.items():
            w = ProjectionWorker(
                lambda p=path: hydra.load_full_stack(p, self.DATASET), method, 0, skip, nframes,
                dark=self.dark(n), bright=self.bright(n), background=self.background(n),
                bright_mode=self.bright_mode(), parent=self)
            w.finished.connect(lambda img, info, n=n: self._on_one_projected(n, img, info))
            w.failed.connect(lambda err, n=n: self._on_one_projection_failed(n, err))
            self._proj_workers[n] = w
            w.start()

    def _on_one_projected(self, n: int, img, info: str):
        self._proj_raw[n] = img
        self._proj_pending.discard(n)
        if not self._proj_pending:
            self._finish_projection()

    def _on_one_projection_failed(self, n: int, err: str):
        self._proj_errors[n] = err.strip().splitlines()[-1] if err and err.strip() else "error"
        self._proj_pending.discard(n)
        if not self._proj_pending:
            self._finish_projection()

    def _finish_projection(self):
        self._proj_btn.setEnabled(True)
        self._is_projection = bool(self._proj_raw)
        self._apply_project_style(self._is_projection)
        ok = ", ".join(f"ge{n}" for n in sorted(self._proj_raw))
        msg = f"Projected {len(self._proj_raw)}/{len(self._siblings)} panel(s)"
        if ok:
            msg += f" ({ok})"
        if self._proj_errors:
            errs = "; ".join(f"ge{n}: {e}" for n, e in sorted(self._proj_errors.items()))
            msg += f"  — failed: {errs}"
        self._proj_info.setText(msg)
        self.projectionChanged.emit()

    def _back_to_frames(self):
        self._clear_projection()

    def _clear_projection(self, emit: bool = True):
        was_on = self._is_projection or bool(self._proj_raw)
        self._is_projection = False
        self._proj_raw = {}
        self._proj_errors = {}
        self._apply_project_style(False)
        if emit and was_on:
            self.projectionChanged.emit()

    def is_projection(self) -> bool:
        return self._is_projection

    def projected(self, n: int) -> Optional[np.ndarray]:
        return self._proj_raw.get(n) if self._is_projection else None

    def projection_card(self) -> QtWidgets.QGroupBox:
        """The "Projection" card — owned/built here, but placed by
        ``HydraViewerPage`` at the top of the middle panel instead of this
        loader panel's own (left-side) layout."""
        return self._proj_grp

    # ── Public accessors ─────────────────────────────────────────
    def siblings(self) -> dict:
        """The panels that make up the array: found on disk AND ticked.

        This is the SELECTED set, matching what ``siblingsChanged`` emits.
        It returned the full found set until a run with one panel ticked
        calibrated all four anyway -- the selection filtered the signal (so
        the toolbar updated and it looked applied) while every page read
        its panel list back through this accessor and got all four. Both
        routes have to mean the same thing or the selection is decorative.

        ``found_siblings`` is the unfiltered set, for reporting only.
        """
        return self._selected_siblings()

    def found_siblings(self) -> dict:
        """Every panel discovered beside the anchor path, ticked or not --
        for saying "3 of 4 selected", never for deciding what to process."""
        return dict(self._siblings)

    def n_frames(self) -> int:
        return self._n_frames

    def frame_index(self) -> int:
        return self._frame

    def dataset(self) -> str:
        return self.DATASET

    def set_path(self, path: str):
        """Programmatic equivalent of typing a path and pressing Enter —
        used by Save/Load GUI State restore."""
        self._set_path(path)

    def current_path(self) -> str:
        return self._path_ed.text().strip()

    def dark(self, n: int) -> Optional[np.ndarray]:
        return self._dark_sel.field(n)

    def bright(self, n: int) -> Optional[np.ndarray]:
        return self._bright_sel.field(n)

    def background(self, n: int) -> Optional[np.ndarray]:
        return self._bg_sel.field(n)

    def bright_mode(self) -> str:
        return self._bright_sel.mode()

    # ── Stream-mode accessors (Batch Integrate) ─────────────────────

    def source_cfg(self, n: int) -> dict:
        """Streaming source descriptor for ``BatchWorker``, for panel ``n`` —
        mirrors ``widgets.DataLoaderPanel.source_cfg()``."""
        path = self._siblings.get(n, "")
        if source_kind(path) == "hdf5":
            return {"type": "hdf5", "path": path, "dataset": self.DATASET}
        return {"type": "tiff_glob", "path": path}

    def frame_range(self) -> tuple:
        """Shared (start, end_or_None, stride) — Hydra panels are
        synchronized frames of one scan, so one range applies to all."""
        end = self._fr_end.value() if self._fr_end.value() > 0 else None
        return (self._fr_start.value(), end, max(1, self._fr_stride.value()))

    def composite_mask(self, n: int) -> Optional[np.ndarray]:
        sel = self._mask_sels.get(n)
        return sel.composite_mask() if sel is not None else None

    def has_live_mask_source(self, n: int) -> bool:
        sel = self._mask_sels.get(n)
        return sel.has_live_mask_source() if sel is not None else False

    # ── GUI state (Save/Load GUI State, stream mode only) ───────────

    def get_state(self) -> dict:
        if self._mode != "stream":
            return {}
        return {
            "fr_start": self._fr_start.value(), "fr_end": self._fr_end.value(),
            "fr_stride": self._fr_stride.value(),
            "masks": {n: sel.get_state() for n, sel in self._mask_sels.items()},
            "panels": sorted(n for n, cb in self._status_lbls.items()
                             if cb.isChecked()),
        }

    def set_state(self, state: dict):
        if self._mode != "stream" or not state:
            return
        if "fr_start" in state:
            self._fr_start.setValue(int(state["fr_start"]))
        if "fr_end" in state:
            self._fr_end.setValue(int(state["fr_end"]))
        if "fr_stride" in state:
            self._fr_stride.setValue(int(state["fr_stride"]))
        for n_key, mstate in (state.get("masks") or {}).items():
            sel = self._mask_sels.get(int(n_key))
            if sel is not None:
                sel.set_state(mstate)
        # Panel selection. This can only ever NARROW what the current path
        # actually has: a saved tick for a panel not on disk must not
        # re-enable a box no file can back, so the found set still gates it.
        if state.get("panels") is not None:
            want = {int(n) for n in state["panels"]}
            self._loading_panel_ticks = True
            try:
                for n, cb in self._status_lbls.items():
                    if cb.isEnabled():
                        cb.setChecked(n in want)
            finally:
                self._loading_panel_ticks = False
            if self._siblings:
                self._refresh_panel_info()
                self.siblingsChanged.emit(self._selected_siblings())


class HydraDetectorToolbar(QtWidgets.QWidget):
    """Row of 5 exclusive buttons above the Hydra image viewer: ge1-ge4 show
    that panel's own raw frame; composite shows the geometry-based windmill
    composite of every currently-available panel."""

    panelChanged = QtCore.pyqtSignal(str)   # "ge1".."ge4" | "composite"

    _KEYS = ("ge1", "ge2", "ge3", "ge4", "composite")

    def __init__(self, parent=None, *, include_composite: bool = True):
        super().__init__(parent)
        self._keys = self._KEYS if include_composite else self._KEYS[:-1]
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        self._buttons = {}
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        for key in self._keys:
            label = "Composite" if key == "composite" else key.upper()
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.toggled.connect(lambda checked, k=key: self._on_toggled(k, checked))
            layout.addWidget(btn)
            self._group.addButton(btn)
            self._buttons[key] = btn
        layout.addStretch(1)
        self._buttons["ge1"].setChecked(True)

    def _on_toggled(self, key: str, checked: bool):
        if checked:
            self.panelChanged.emit(key)

    def set_available(self, panel_numbers):
        """Enable only the ge-buttons for panels actually found, and
        Composite iff at least 2 panels are available. If the currently
        selected button just became disabled, falls back to the first
        enabled one (emitting panelChanged)."""
        panel_numbers = set(panel_numbers)
        for n in (1, 2, 3, 4):
            self._buttons[f"ge{n}"].setEnabled(n in panel_numbers)
        if "composite" in self._buttons:
            self._buttons["composite"].setEnabled(len(panel_numbers) >= 2)
        cur = self.current()
        if not self._buttons[cur].isEnabled():
            for key in self._keys:
                if self._buttons[key].isEnabled():
                    self._buttons[key].setChecked(True)
                    break

    def current(self) -> str:
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return "ge1"

    def set_current(self, key: str):
        if key in self._buttons and self._buttons[key].isEnabled():
            self._buttons[key].setChecked(True)


# Fixed per-curve colors — semantic, not index-based, so ge1 is always the
# same color regardless of which panels happen to be loaded/visible.
_HYDRA_CURVE_COLORS = {
    "ge1": "#4fc3f7", "ge2": "#ef5350", "ge3": "#66bb6a", "ge4": "#ab47bc",
    "composite": "#f5f5f5",
}
_HYDRA_CURVE_KEYS = ("ge1", "ge2", "ge3", "ge4", "composite")


class HydraProfileViewer(QtWidgets.QWidget):
    """Radial-integration plot for Hydra mode: one independently-computed
    curve per panel (ge1-4), each converted from its own native R-pixel
    axis to a shared display unit (R/2θ/Q) via ``widgets._convert_radial``,
    plus a toggleable "Composite" curve. This widget only displays curves —
    the composite curve's data (a resampled, NaN-aware sum of the four
    panels' own profiles) is computed by the owner (``HydraViewerPage``)
    and pushed in via ``set_curve`` like any other curve."""

    #: emitted when the "Composite" checkbox is toggled, so the owner knows
    #: whether it's worth computing/pushing that curve at all.
    compositeVisibilityChanged = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None, *, composite_as_button: bool = False):
        super().__init__(parent)
        self._native: dict = {}   # key -> (r_px, profile, lsd_um, px_um, wavelength_A)
        self._curves: dict = {}
        self._checks: dict = {}
        self._composite_as_button = composite_as_button
        self._overall_btn: Optional[QtWidgets.QPushButton] = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(2)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("X:"))
        self._xaxis = _NoScrollComboBox()
        self._xaxis.addItems(["R (px)", "2θ (°)", "Q (Å⁻¹)"])
        self._xaxis.setCurrentIndex(1)   # 2θ — the meaningful shared axis across panels
        self._xaxis.currentIndexChanged.connect(self._replot)
        bar.addWidget(self._xaxis)
        self._logy = QtWidgets.QCheckBox("Log Y")
        self._logy.toggled.connect(self._replot)
        bar.addWidget(self._logy)
        bar.addSpacing(12)
        for key in _HYDRA_CURVE_KEYS:
            if key == "composite" and self._composite_as_button:
                btn = QtWidgets.QPushButton("Overall")
                btn.setCheckable(True)
                btn.toggled.connect(self._on_overall_toggled)
                bar.addWidget(btn)
                self._overall_btn = btn
                self._apply_overall_style(False)
                continue
            label = "Composite" if key == "composite" else key.upper()
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(True)
            chk.setStyleSheet(f"QCheckBox {{ color: {_HYDRA_CURVE_COLORS[key]}; }}")
            if key == "composite":
                chk.toggled.connect(self.compositeVisibilityChanged.emit)
            chk.toggled.connect(self._replot)
            bar.addWidget(chk)
            self._checks[key] = chk
        bar.addStretch(1)
        self._toolbar_layout = bar
        layout.addLayout(bar)

        self._plot = pg.PlotWidget(background="k")
        self._plot.setLabel("left", "Mean intensity")
        self._plot.setLabel("bottom", _XUNIT_LABEL["2th"])
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.addLegend(offset=(-10, 10))
        for key in _HYDRA_CURVE_KEYS:
            label = "Composite" if key == "composite" else key.upper()
            pen = pg.mkPen(_HYDRA_CURVE_COLORS[key], width=2,
                           style=QtCore.Qt.DashLine if key == "composite" else QtCore.Qt.SolidLine)
            self._curves[key] = self._plot.plot([], [], pen=pen, name=label)
        layout.addWidget(self._plot, stretch=1)

    def _unit_key(self) -> str:
        return ("R", "2th", "Q")[self._xaxis.currentIndex()]

    def set_curve(self, key: str, r_px, profile, *, lsd_um=None, px_um=None,
                 wavelength_A=None):
        """(Re)place one curve's data, in its own native R-pixel axis plus
        the geometry needed to convert it — independent per curve, since
        each Hydra panel generally has its own Lsd/pixel size/beam centre."""
        if key not in self._curves:
            return
        self._native[key] = (np.asarray(r_px), np.asarray(profile), lsd_um, px_um, wavelength_A)
        self._replot()

    def clear_curve(self, key: str):
        self._native.pop(key, None)
        if key in self._curves:
            self._curves[key].setData([], [])

    def get_native(self, key: str) -> Optional[tuple]:
        """The last data pushed via ``set_curve`` for ``key`` — its own
        native (r_px, profile, lsd_um, px_um, wavelength_A), or None. Used
        by an owner (e.g. deriving the "Composite" curve from ge1-4's own
        curves) to read back what's already been computed."""
        return self._native.get(key)

    def _apply_overall_style(self, active: bool):
        if active:
            self._overall_btn.setStyleSheet(
                "QPushButton { background: #2e7d32; color: white; font-weight: bold; "
                "border: 1px solid #1b5e20; border-radius: 4px; padding: 3px 10px; }")
        else:
            self._overall_btn.setStyleSheet(
                "QPushButton { border: 1px solid #666; border-radius: 4px; padding: 3px 10px; }")

    def _on_overall_toggled(self, active: bool):
        self._apply_overall_style(active)
        self.compositeVisibilityChanged.emit(active)
        self._replot()
        if active:
            self._plot.getPlotItem().getViewBox().autoRange()

    def _replot(self, *_):
        target = self._unit_key()
        log = self._logy.isChecked()
        overall_active = self._composite_as_button and self._overall_btn.isChecked()
        self._plot.setLabel("bottom", _XUNIT_LABEL[target])
        self._plot.setLabel("left", "log₁₀(intensity)" if log else "Mean intensity")
        xs, ys = [], []
        for key, curve in self._curves.items():
            if key == "composite":
                visible = overall_active if self._composite_as_button else self._checks[key].isChecked()
            else:
                visible = self._checks[key].isChecked() and not overall_active
            data = self._native.get(key)
            if not visible or data is None:
                curve.setData([], [])
                continue
            r_px, profile, lsd, px, wl = data
            x = _convert_radial(r_px, lsd, px, wl, "R", target)
            y = profile
            if log:
                y = np.where(y > 0, np.log10(np.maximum(y, 1e-30)), np.nan)
            curve.setData(x, y)
            xs.append(np.asarray(x)); ys.append(np.asarray(y))
        if xs:
            all_x = np.concatenate(xs); all_y = np.concatenate(ys)
            fin = np.isfinite(all_x) & np.isfinite(all_y)
            if fin.any():
                self._apply_view_limits(float(all_x[fin].min()), float(all_x[fin].max()),
                                        float(all_y[fin].min()), float(all_y[fin].max()))

    def _apply_view_limits(self, xmin, xmax, ymin, ymax):
        """Bound pan/zoom to the currently-visible curves' combined data
        (+ margin), same intent as the single-detector radial plot's
        ``ProfileViewer._apply_view_limits`` — stops the user scrolling/
        zooming arbitrarily far from where the data actually is."""
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

    def composite_visible(self) -> bool:
        if self._composite_as_button:
            return self._overall_btn.isChecked()
        return self._checks["composite"].isChecked()
