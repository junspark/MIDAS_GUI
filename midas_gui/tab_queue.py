"""Batch Queue tab — integrate many samples in one go.

A beamtime produces stages: a run of samples sharing a calibration and a set of
dark/bright/background frames, then the calibrant changes and the next stage
begins. Batch Integrate handles one sample at a time; this tab holds the whole
plan as a tree

    Calibration            (geometry + detector mask)
      └─ Corrections       (dark / bright / background)
           └─ Samples      (one HDF5 file, or one folder of frames)

and runs it, several samples at a time, writing each sample into its own
directory in an output tree that mirrors the input hierarchy — its integrated
``csv/``/``zarr/``/``h5/`` output *and* its own project file with results and
provenance, so a sample stays self-describing long after the session.

The pieces are deliberately split: :mod:`batch_queue` is the model and the path
mirroring (no Qt, unit-tested), :mod:`queue_policy` is the scheduling decision
(no Qt, unit-tested), :mod:`queue_runner` owns the workers, and this module is
the widgets.
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from midas_gui import project, settings
from midas_gui import style as S
from midas_gui.batch_queue import (
    BatchQueue, CalibrationNode, CorrectionsNode, KIND_FOLDER, KIND_HDF5,
    SOURCE_FILE, SOURCE_TAB2, detect_dataset, infer_data_root, is_unrooted,
    plan_output_dirs, project_path_for, sample_source_cfg,
)
from midas_gui.constants import DEFAULT_KERNEL, H5_EXTS, KERNELS
from midas_gui.dialogs import AddSamplesDialog, show_error
from midas_gui.helpers import (_NoScrollComboBox, _NoScrollSpinBox, _fspin,
                               _build_spec, check_output_dir_writable,
                               list_h5_datasets, resolve_calibration_fields,
                               rmax_corner_px, rmax_edge_px, spec_from_geometry_file,
                               browse_start_dir, warn_if_path_missing)
from midas_gui.queue_runner import RunItem, SampleRunScheduler, default_max_concurrent
from midas_gui.widgets import LogPanel, OutputFormatSelector

#: Single-frame file suffixes that make a folder sample's children worth
#: listing — same rule ``dialogs.find_samples_below`` uses to decide a
#: directory holds frames (TIFF-family + ``.ge*`` + HDF5).
_FRAME_EXTS = {".tif", ".tiff", ".cbf", ".edf"}

#: How many children to list under an expanded folder sample. A frame
#: folder can hold tens of thousands of files; this keeps the tree usable.
_FOLDER_EXPAND_LIMIT = 500


def _folder_children(folder: str, limit: int = _FOLDER_EXPAND_LIMIT) -> list:
    """Frame file names sitting directly in ``folder``, sorted, capped at
    ``limit``. Read-only preview for the queue tree's lazy expansion — never
    raises on an unreadable or vanished folder."""
    names = []
    try:
        with os.scandir(str(folder)) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                suf = Path(entry.name).suffix.lower()
                if suf in _FRAME_EXTS or suf in H5_EXTS or suf.startswith(".ge"):
                    names.append(entry.name)
    except OSError:
        pass
    names.sort()
    return names[:limit], len(names)

# Tree column layout.
COL_NAME, COL_DETAIL, COL_STATUS = 0, 1, 2

#: Roles on a tree item pointing back at the model object it renders.
_ROLE_KIND = QtCore.Qt.UserRole        # "cal" | "corr" | "sample"
_ROLE_OBJ = QtCore.Qt.UserRole + 1


class CorrectionsDialog(QtWidgets.QDialog):
    """Edit one corrections node — dark / bright / background.

    Follows the ``MaterialDialog`` precedent in ``hydra_geometry_card``: a small
    modal whose :meth:`apply_to` writes back into the model object, so the tree
    never holds half-edited state."""

    def __init__(self, node: CorrectionsNode, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Corrections")
        self.resize(560, 200)
        form = S.Form()
        self._name = QtWidgets.QLineEdit(node.name)
        self._fields = {}
        form.row(("Name:", self._name))
        for key, label in (("dark", "Dark:"), ("bright", "Bright:"),
                           ("background", "Background:")):
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
            ed = QtWidgets.QLineEdit(getattr(node, key) or "")
            ed.setPlaceholderText("file or folder — mean taken before use")
            warn_if_path_missing(ed, self)
            btn = QtWidgets.QToolButton(); btn.setText("…")
            btn.clicked.connect(lambda _c=False, e=ed: self._browse(e))
            h.addWidget(ed, 1); h.addWidget(btn)
            self._fields[key] = ed
            if key == "background":
                self._bg_scale = _fspin(0.0, 1e9, 3, node.background_scale)
                self._bg_scale.setToolTip(
                    "Multiply the background frame by this factor before it is subtracted.")
                h.addWidget(QtWidgets.QLabel("scale:")); h.addWidget(self._bg_scale)
            form.row((label, row))
        self._mode = _NoScrollComboBox()
        self._mode.addItem("divide", "divide"); self._mode.addItem("subtract", "subtract")
        idx = self._mode.findData(node.bright_mode)
        if idx >= 0:
            self._mode.setCurrentIndex(idx)
        form.row(("Bright mode:", self._mode))

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        hint = QtWidgets.QLabel(
            "Samples nested under this node are corrected with these frames. "
            "Add another corrections node under the same calibration when the "
            "darks are retaken mid-stage.")
        hint.setWordWrap(True); hint.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        v.addWidget(hint)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def _browse(self, edit):
        start = browse_start_dir(edit.text())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select frame file", start)
        if not path:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if path:
            edit.setText(path)

    def apply_to(self, node: CorrectionsNode):
        node.name = self._name.text().strip() or "Corrections"
        for key, ed in self._fields.items():
            setattr(node, key, ed.text().strip() or None)
        node.background_scale = self._bg_scale.value()
        node.bright_mode = self._mode.currentData() or "divide"


class CalibrationDialog(QtWidgets.QDialog):
    """Edit one calibration node — where its geometry comes from, and its mask.

    The mask sits here rather than on the corrections nodes because it is a
    property of the detector (beamstop, dead pixels, panel gaps), and because
    that is what lets every sample under this calibration share one built
    detector map."""

    def __init__(self, node: CalibrationNode, live_result=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.resize(620, 260)
        self._live_result = live_result
        self._snapshot = node.calib_snapshot

        v = QtWidgets.QVBoxLayout(self)
        form = S.Form()
        self._name = QtWidgets.QLineEdit(node.name)
        form.row(("Name:", self._name))
        v.addLayout(form)

        self._from_tab2 = QtWidgets.QRadioButton("From the Calibrate tab")
        self._from_file = QtWidgets.QRadioButton("From a geometry file")
        self._from_tab2.setChecked(node.source == SOURCE_TAB2)
        self._from_file.setChecked(node.source == SOURCE_FILE)
        v.addWidget(self._from_tab2)
        note = QtWidgets.QLabel(
            "Snapshotted when you press OK, so re-running Calibrate later "
            "cannot silently change an already-queued stage."
            if live_result is not None or node.calib_snapshot else
            "No calibration available from the Calibrate tab yet — run one, or "
            "use a geometry file.")
        note.setWordWrap(True); note.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        note.setContentsMargins(22, 0, 0, 4)
        v.addWidget(note)
        v.addWidget(self._from_file)

        file_row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(file_row)
        h.setContentsMargins(22, 0, 0, 0); h.setSpacing(4)
        self._file_ed = QtWidgets.QLineEdit(node.file_path or "")
        self._file_ed.setPlaceholderText("paramstest .txt / .poni / .json")
        self._file_ed.textEdited.connect(lambda *_: self._from_file.setChecked(True))
        warn_if_path_missing(self._file_ed, self)
        b = QtWidgets.QToolButton(); b.setText("…")
        b.clicked.connect(self._browse_calib)
        h.addWidget(self._file_ed, 1); h.addWidget(b)
        v.addWidget(file_row)

        mask_row = QtWidgets.QWidget()
        mh = QtWidgets.QHBoxLayout(mask_row)
        mh.setContentsMargins(0, 0, 0, 0); mh.setSpacing(4)
        mh.addWidget(QtWidgets.QLabel("Mask:"))
        self._mask_ed = QtWidgets.QLineEdit(self._mask_path_of(node) or "")
        self._mask_ed.setPlaceholderText("optional — a mask file for this detector")
        warn_if_path_missing(self._mask_ed, self)
        mb = QtWidgets.QToolButton(); mb.setText("…")
        mb.clicked.connect(self._browse_mask)
        mh.addWidget(self._mask_ed, 1); mh.addWidget(mb)
        v.addWidget(mask_row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        v.addStretch(1)
        v.addWidget(btns)

    @staticmethod
    def _mask_path_of(node) -> Optional[str]:
        for src in node.mask_sources or []:
            if src.get("path"):
                return src["path"]
        return None

    def _browse_calib(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select calibration", browse_start_dir(self._file_ed.text()),
            "Geometry (*.json *.txt *.poni);;All files (*)")
        if path:
            self._file_ed.setText(path)
            self._from_file.setChecked(True)

    def _browse_mask(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select mask", browse_start_dir(self._mask_ed.text()))
        if path:
            self._mask_ed.setText(path)

    def apply_to(self, node: CalibrationNode):
        node.name = self._name.text().strip() or "Calibration"
        if self._from_file.isChecked():
            node.source = SOURCE_FILE
            node.file_path = self._file_ed.text().strip() or None
        else:
            node.source = SOURCE_TAB2
            if self._live_result is not None:
                node.calib_snapshot = project.sanitize_result_dict(self._live_result)
        mask = self._mask_ed.text().strip()
        node.mask_sources = [{"kind": "file", "path": mask, "enabled": True}] if mask else None


class SampleDialog(QtWidgets.QDialog):
    """Edit one sample — its label, and (HDF5 only) which internal dataset
    holds the frames.

    The dataset combo is populated the same way the Data Viewer's loader
    panel populates its own (``helpers.list_h5_datasets``), and the
    selection already showing is whatever ``batch_queue.detect_dataset``
    picked when the sample was added — first ≥3-D dataset, else the first
    one at all — so opening this dialog on an untouched sample shows the
    same default the Data Viewer would."""

    def __init__(self, sample, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sample")
        self.resize(520, 160)
        self._sample = sample
        v = QtWidgets.QVBoxLayout(self)
        form = S.Form()
        self._label = QtWidgets.QLineEdit(sample.label)
        form.row(("Label:", self._label))

        self._ds_combo = None
        if sample.kind == KIND_HDF5:
            self._ds_combo = _NoScrollComboBox()
            self._ds_combo.setEditable(True)
            try:
                items = list_h5_datasets(sample.path)
            except Exception:
                items = []
            for name, shape in items:
                self._ds_combo.addItem(f"{name}   {tuple(shape)}", name)
            current = sample.dataset or detect_dataset(sample.path)
            idx = next((i for i in range(self._ds_combo.count())
                        if self._ds_combo.itemData(i) == current), -1)
            if idx >= 0:
                self._ds_combo.setCurrentIndex(idx)
            else:
                self._ds_combo.setEditText(current)
            form.row(("Dataset:", self._ds_combo))
        v.addLayout(form)

        if sample.kind == KIND_HDF5:
            hint = QtWidgets.QLabel(
                "Which HDF5 dataset holds the frames. Defaults to what the "
                "Data Viewer would pick for this file — change it only when "
                "that guess is wrong.")
            hint.setWordWrap(True); hint.setStyleSheet(f"color:{S.MUTED};font-size:10px")
            v.addWidget(hint)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        v.addStretch(1)
        v.addWidget(btns)

    def apply_to(self, sample):
        sample.label = self._label.text().strip() or sample.label
        if self._ds_combo is not None:
            text = self._ds_combo.currentText().split("   ")[0].strip()
            sample.dataset = text or sample.dataset


class BatchQueueTab(QtWidgets.QWidget):
    """The tab: the tree, the queue-wide settings, and the run bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = BatchQueue()
        self._calib_result = None          # live result from the Calibrate tab
        self._project_ctx = None
        self._scheduler: Optional[SampleRunScheduler] = None
        self._settings_provider = None     # set by app.py → Batch Integrate's settings
        self._items: dict = {}             # sample key -> QTreeWidgetItem
        self._run_rows: dict = {}          # sample key -> (cal, corr, sample, out_dir)
        self._build_ui()
        self._rebuild_tree()

    # ── wiring from app.py ───────────────────────────────────────

    def set_calibration(self, result):
        """A calibration result is available from the Calibrate tab — either
        a Fit that just finished, or one restored from an opened project —
        offer it to new calibration nodes.

        Existing nodes keep their snapshot: a queue built this morning must not
        change because the calibrant was re-fit this afternoon."""
        self._calib_result = result
        self._refresh_calib_hint()

    def set_project_context(self, ctx):
        self._project_ctx = ctx

    def set_settings_provider(self, provider):
        """``provider()`` returns Batch Integrate's current integration
        settings dict, for the "Copy from Batch Integrate" button."""
        self._settings_provider = provider

    # ── UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ── left: the tree ──
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Queue", "Details", "Status"])
        self._tree.setColumnWidth(COL_NAME, 240)
        self._tree.setColumnWidth(COL_DETAIL, 320)
        self._tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._tree.itemDoubleClicked.connect(lambda *_: self._edit_selected())
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemExpanded.connect(self._on_item_expanded)
        lv.addWidget(self._tree, 1)

        row = QtWidgets.QHBoxLayout(); row.setSpacing(4)
        for text, slot, tip in (
            ("+ Calibration", self._add_calibration,
             "Add a calibration stage. Its geometry is snapshotted now."),
            ("+ Corrections", self._add_corrections,
             "Add a dark/bright/background set under the selected calibration."),
            ("+ Samples…", self._add_samples,
             "Pick HDF5 files and frame folders to run under the selected "
             "corrections node."),
            ("Edit…", self._edit_selected, "Edit the selected node."),
            ("Remove", self._remove_selected, "Remove the selected nodes."),
        ):
            b = QtWidgets.QPushButton(text); b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        lv.addLayout(row)
        split.addWidget(left)

        # ── right: settings ──
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        roots = S.make_card("Output layout")
        self._data_root = QtWidgets.QLineEdit()
        self._data_root.setPlaceholderText("auto — the folder every sample sits under")
        self._data_root.setToolTip(
            "Samples are mirrored relative to this folder. Left blank, it is "
            "the deepest folder every queued sample shares.")
        self._data_root.textChanged.connect(lambda *_: self._refresh_preview())
        warn_if_path_missing(self._data_root, self)
        self._out_root = QtWidgets.QLineEdit()
        self._out_root.setPlaceholderText("where the results tree is written")
        self._out_root.textChanged.connect(lambda *_: self._refresh_preview())
        warn_if_path_missing(self._out_root, self, is_output_dir=True)
        for label, ed, browse in (("Data root:", self._data_root, True),
                                  ("Output root:", self._out_root, True)):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
            h.addWidget(ed, 1)
            if browse:
                b = QtWidgets.QToolButton(); b.setText("…")
                b.clicked.connect(lambda _c=False, e=ed: self._browse_dir(e))
                h.addWidget(b)
            f = S.Form(); f.row((label, w))
            roots.body.addLayout(f)
        self._preview = QtWidgets.QLabel("")
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        roots.body.addWidget(self._preview)
        rv.addWidget(roots)

        integ = S.make_card("Integration settings (whole queue)")
        copy_btn = QtWidgets.QPushButton("Copy from Batch Integrate")
        copy_btn.setToolTip("Take the R/η binning, formats and kernel currently "
                             "set on the Batch Integrate tab.")
        copy_btn.clicked.connect(self._copy_from_batch)
        integ.body.addWidget(copy_btn)
        self._kernel = _NoScrollComboBox()
        for label, key in KERNELS.items():
            self._kernel.addItem(label, key)
        ki = self._kernel.findData(DEFAULT_KERNEL)
        if ki >= 0:
            self._kernel.setCurrentIndex(ki)
        self._r_bin = _fspin(0.1, 20.0, 2, 1.0, "px")
        self._e_bin = _fspin(0.5, 360.0, 1, 5.0, "°")
        self._r_min = _fspin(0.0, 1_000_000.0, 2, 0.0, "px")
        self._r_max = _fspin(0.0, 1_000_000.0, 2, 0.0, "px")
        self._eta_min = _fspin(-180.0, 180.0, 1, -180.0, "°")
        self._eta_max = _fspin(-180.0, 180.0, 1, 180.0, "°")
        self._r_max.setToolTip(
            "Manual mode only. 0 = auto (the backend's farthest-corner default).")
        self._r_max_mode = _NoScrollComboBox()
        for label, key in (("Manual", "manual"), ("Corner", "corner"), ("Edge", "edge")):
            self._r_max_mode.addItem(label, key)
        self._r_max_mode.setToolTip(
            "Manual: use the value at left for every stage.\n"
            "Corner/Edge: computed per calibration node when the queue runs, "
            "from that stage's own beam centre and detector size — the same "
            "formulas as Batch Integrate's Corner/Edge Rmax presets. Right "
            "for a queue whose stages don't share one detector geometry.")
        self._r_max_mode.currentIndexChanged.connect(self._on_r_max_mode_changed)
        r_max_row = QtWidgets.QHBoxLayout(); r_max_row.setSpacing(4)
        r_max_row.addWidget(self._r_max, 1); r_max_row.addWidget(self._r_max_mode)
        form = S.Form()
        form.row(("Kernel:", self._kernel))
        form.row(("R bin:", self._r_bin), ("η bin:", self._e_bin))
        form.row(("R min:", self._r_min), ("R max:", r_max_row))
        form.row(("η min:", self._eta_min), ("η max:", self._eta_max))
        integ.body.addLayout(form)
        # Combine sub-frames — mirrors the Batch Integrate control, whose own
        # default is 1 (one integrated frame per raw frame — right for an
        # HDF5 holding a scan of distinct points). 0 (combine the whole file
        # into one) is for a detector writing several raw exposures per scan
        # point, and must be chosen deliberately: left at 0 by default, an
        # HDF5 holding N distinct scan points was silently reduced to ONE mean
        # profile — ten frames in, one CSV out, no error.
        self._combine_chunk = _NoScrollSpinBox()
        self._combine_chunk.setRange(0, 999999)
        self._combine_chunk.setFixedWidth(70)
        self._combine_chunk.setValue(1)
        self._combine_chunk.setToolTip(
            "HDF5 samples only: how many consecutive raw sub-frames in each "
            "file to combine into one integrated frame.\n"
            "1 = one integrated frame per raw frame (an HDF5 holding a scan).\n"
            "0 = combine the whole file into one (a detector writing several "
            "exposures per scan point).")
        self._combine_op = _NoScrollComboBox()
        for label, key in (("Mean", "mean"), ("Sum", "sum"),
                           ("Max", "max"), ("Median", "median")):
            self._combine_op.addItem(label, key)
        form.row(("Combine sub-frames:", self._combine_chunk), ("op:", self._combine_op))
        self._fmt = OutputFormatSelector()
        integ.body.addWidget(self._fmt)
        self._weighted = QtWidgets.QCheckBox("Azimuthal mean (weighted)")
        self._weighted.setChecked(True)
        integ.body.addWidget(self._weighted)
        rv.addWidget(integ)

        run = S.make_card("Run")
        rr = QtWidgets.QHBoxLayout(); rr.setSpacing(6)
        rr.addWidget(QtWidgets.QLabel("Samples at once:"))
        self._concurrency = _NoScrollSpinBox()
        self._concurrency.setRange(1, 32)
        self._concurrency.setValue(default_max_concurrent())
        self._concurrency.setToolTip(
            "How many samples integrate simultaneously. Each one holds its own "
            "frames in memory, so the useful ceiling is RAM, not cores.")
        self._concurrency.setFixedWidth(60)
        rr.addWidget(self._concurrency)
        rr.addStretch(1)
        self._run_btn = S.primary_btn("Run queue")
        self._run_btn.clicked.connect(self._run)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(S.DANGER_BTN_QSS)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        rr.addWidget(self._run_btn); rr.addWidget(self._cancel_btn)
        run.body.addLayout(rr)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        run.body.addWidget(self._progress)
        rv.addWidget(run)
        rv.addStretch(1)
        split.addWidget(right)
        split.setSizes([720, 460])
        outer.addWidget(split, 1)

        self._log = LogPanel()
        outer.addWidget(self._log)
        self._refresh_calib_hint()
        self._on_r_max_mode_changed()

    def _on_r_max_mode_changed(self, *_args):
        self._r_max.setEnabled(self._r_max_mode.currentData() == "manual")

    def _browse_dir(self, edit):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder", browse_start_dir(edit.text()))
        if path:
            edit.setText(path)

    def _refresh_calib_hint(self):
        pass   # the hint lives in CalibrationDialog, rebuilt each time it opens

    # ── tree ─────────────────────────────────────────────────────

    def _rebuild_tree(self):
        self._tree.blockSignals(True)
        self._tree.clear()
        self._items.clear()
        for ci, cal in enumerate(self._queue.calibrations):
            cal_item = QtWidgets.QTreeWidgetItem(self._tree)
            cal_item.setText(COL_NAME, f"⚙  {cal.name}")
            cal_item.setText(COL_DETAIL, self._describe_calibration(cal))
            cal_item.setData(COL_NAME, _ROLE_KIND, "cal")
            cal_item.setData(COL_NAME, _ROLE_OBJ, ci)
            cal_item.setExpanded(True)
            for ki, corr in enumerate(cal.corrections):
                corr_item = QtWidgets.QTreeWidgetItem(cal_item)
                corr_item.setText(COL_NAME, f"▤  {corr.name}")
                corr_item.setText(COL_DETAIL, corr.describe())
                corr_item.setData(COL_NAME, _ROLE_KIND, "corr")
                corr_item.setData(COL_NAME, _ROLE_OBJ, (ci, ki))
                corr_item.setExpanded(True)
                for si, sample in enumerate(corr.samples):
                    s_item = QtWidgets.QTreeWidgetItem(corr_item)
                    icon = "📄" if sample.kind == KIND_HDF5 else "📁"
                    s_item.setText(COL_NAME, f"{icon}  {sample.label}")
                    detail = sample.path
                    if sample.kind == KIND_HDF5 and sample.dataset:
                        detail += f"   [{sample.dataset}]"
                    s_item.setText(COL_DETAIL, detail)
                    s_item.setToolTip(COL_DETAIL, detail)
                    s_item.setFlags(s_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    s_item.setCheckState(
                        COL_NAME,
                        QtCore.Qt.Checked if sample.enabled else QtCore.Qt.Unchecked)
                    s_item.setData(COL_NAME, _ROLE_KIND, "sample")
                    s_item.setData(COL_NAME, _ROLE_OBJ, (ci, ki, si))
                    self._items[self._sample_key(ci, ki, si)] = s_item
                    if sample.kind == KIND_FOLDER:
                        placeholder = QtWidgets.QTreeWidgetItem(s_item)
                        placeholder.setText(COL_NAME, "…")
                        placeholder.setData(COL_NAME, _ROLE_KIND, "placeholder")
        self._tree.blockSignals(False)
        self._refresh_preview()

    def _on_item_expanded(self, item):
        """Lazily list a folder sample's frame files the first time it is
        expanded — walking every queued folder up front would stall the UI
        on a large beamtime tree for no benefit until someone looks."""
        if item.data(COL_NAME, _ROLE_KIND) != "sample" or item.childCount() != 1:
            return
        child = item.child(0)
        if child.data(COL_NAME, _ROLE_KIND) != "placeholder":
            return
        ci, ki, si = item.data(COL_NAME, _ROLE_OBJ)
        try:
            sample = self._queue.calibrations[ci].corrections[ki].samples[si]
        except (IndexError, TypeError):
            return
        item.removeChild(child)
        names, total = _folder_children(sample.path)
        if not names:
            empty = QtWidgets.QTreeWidgetItem(item)
            empty.setText(COL_NAME, "(no frame files found)")
            empty.setData(COL_NAME, _ROLE_KIND, "placeholder")
            return
        for name in names:
            c = QtWidgets.QTreeWidgetItem(item)
            c.setText(COL_NAME, name)
            c.setData(COL_NAME, _ROLE_KIND, "file")
        if total > len(names):
            more = QtWidgets.QTreeWidgetItem(item)
            more.setText(COL_NAME, f"… {total - len(names)} more not shown")
            more.setData(COL_NAME, _ROLE_KIND, "placeholder")

    @staticmethod
    def _sample_key(ci, ki, si) -> str:
        return f"{ci}/{ki}/{si}"

    @staticmethod
    def _describe_calibration(cal: CalibrationNode) -> str:
        if cal.using_file():
            src = f"file: {Path(cal.file_path).name}" if cal.file_path else "no file set"
        else:
            src = "from Calibrate tab" if cal.calib_snapshot else "no snapshot yet"
        mask = ""
        for m in cal.mask_sources or []:
            if m.get("path"):
                mask = f"   mask={Path(m['path']).name}"
                break
        return src + mask

    def _on_item_changed(self, item, column):
        """Only the sample checkboxes are user-editable."""
        if item.data(COL_NAME, _ROLE_KIND) != "sample":
            return
        ci, ki, si = item.data(COL_NAME, _ROLE_OBJ)
        enabled = item.checkState(COL_NAME) == QtCore.Qt.Checked
        self._queue.calibrations[ci].corrections[ki].samples[si].enabled = enabled
        self._refresh_preview()

    def _selected(self):
        items = self._tree.selectedItems()
        return items[0] if items else None

    def _selected_calibration_index(self) -> Optional[int]:
        item = self._selected()
        while item is not None:
            kind = item.data(COL_NAME, _ROLE_KIND)
            obj = item.data(COL_NAME, _ROLE_OBJ)
            if kind == "cal":
                return obj
            if kind in ("corr", "sample"):
                return obj[0]
            item = item.parent()
        return None

    def _selected_corrections_index(self):
        item = self._selected()
        while item is not None:
            kind = item.data(COL_NAME, _ROLE_KIND)
            obj = item.data(COL_NAME, _ROLE_OBJ)
            if kind == "corr":
                return obj
            if kind == "sample":
                return obj[0], obj[1]
            item = item.parent()
        return None

    # ── editing ──────────────────────────────────────────────────

    def _add_calibration(self):
        node = CalibrationNode(name=f"Calibration {len(self._queue.calibrations) + 1}")
        dlg = CalibrationDialog(node, live_result=self._calib_result, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        dlg.apply_to(node)
        node.corrections.append(CorrectionsNode(name="No corrections"))
        self._queue.calibrations.append(node)
        self._rebuild_tree()

    def _add_corrections(self):
        ci = self._selected_calibration_index()
        if ci is None:
            QtWidgets.QMessageBox.information(
                self, "Select a calibration",
                "Select the calibration to add a corrections set under.")
            return
        cal = self._queue.calibrations[ci]
        node = CorrectionsNode(name=f"Corrections {len(cal.corrections) + 1}")
        dlg = CorrectionsDialog(node, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        dlg.apply_to(node)
        cal.corrections.append(node)
        self._rebuild_tree()

    def _add_samples(self):
        target = self._selected_corrections_index()
        if target is None:
            QtWidgets.QMessageBox.information(
                self, "Select a corrections node",
                "Samples hang off a corrections node — select one (or add a "
                "calibration first).")
            return
        ci, ki = target
        dlg = AddSamplesDialog(parent=self, start_dir=self._start_dir())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        samples = dlg.samples()
        if not samples:
            return
        self._queue.calibrations[ci].corrections[ki].samples.extend(samples)
        self._rebuild_tree()
        self._log.append(f"Added {len(samples)} sample(s).")

    def _start_dir(self) -> str:
        paths = self._queue.sample_paths(enabled_only=False)
        return str(Path(paths[-1]).parent) if paths else self._data_root.text().strip()

    def _edit_selected(self):
        item = self._selected()
        if item is None:
            return
        kind = item.data(COL_NAME, _ROLE_KIND)
        obj = item.data(COL_NAME, _ROLE_OBJ)
        if kind == "cal":
            node = self._queue.calibrations[obj]
            dlg = CalibrationDialog(node, live_result=self._calib_result, parent=self)
        elif kind == "corr":
            node = self._queue.calibrations[obj[0]].corrections[obj[1]]
            dlg = CorrectionsDialog(node, parent=self)
        elif kind == "sample":
            ci, ki, si = obj
            node = self._queue.calibrations[ci].corrections[ki].samples[si]
            dlg = SampleDialog(node, parent=self)
        else:
            return
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            dlg.apply_to(node)
            self._rebuild_tree()

    def _remove_selected(self):
        """Removes bottom-up so earlier deletions can't shift later indices."""
        targets = []
        for item in self._tree.selectedItems():
            targets.append((item.data(COL_NAME, _ROLE_KIND),
                            item.data(COL_NAME, _ROLE_OBJ)))
        if not targets:
            return
        for kind, obj in sorted(targets, key=lambda t: (
                len(t[1]) if isinstance(t[1], tuple) else 0,
                t[1] if isinstance(t[1], tuple) else (t[1],)), reverse=True):
            try:
                if kind == "sample":
                    ci, ki, si = obj
                    del self._queue.calibrations[ci].corrections[ki].samples[si]
                elif kind == "corr":
                    ci, ki = obj
                    del self._queue.calibrations[ci].corrections[ki]
                elif kind == "cal":
                    del self._queue.calibrations[obj]
            except (IndexError, TypeError):
                pass
        self._rebuild_tree()

    # ── settings ─────────────────────────────────────────────────

    def _copy_from_batch(self):
        if self._settings_provider is None:
            self._log.append("Batch Integrate is not available to copy from.")
            return
        st = self._settings_provider() or {}
        for key, widget in (("r_bin", self._r_bin), ("e_bin", self._e_bin),
                            ("r_min", self._r_min), ("r_max", self._r_max),
                            ("eta_min", self._eta_min), ("eta_max", self._eta_max)):
            if st.get(key) is not None:
                widget.setValue(float(st[key]))
        if st.get("r_max") is not None:
            # Batch Integrate has no persistent Corner/Edge mode of its own —
            # its Corner/Edge buttons are one-shot presets that write a plain
            # number into the same field we just copied — so a copy always
            # means "use this number as-is".
            self._r_max_mode.setCurrentIndex(self._r_max_mode.findData("manual"))
        if st.get("kernel"):
            i = self._kernel.findData(st["kernel"])
            if i >= 0:
                self._kernel.setCurrentIndex(i)
        if st.get("fmt"):
            self._fmt.set_state(st["fmt"])
        if st.get("weighted") is not None:
            self._weighted.setChecked(bool(st["weighted"]))
        if st.get("chunk_size") is not None:
            self._combine_chunk.setValue(int(st["chunk_size"]))
        if st.get("combine_op"):
            i = self._combine_op.findData(st["combine_op"])
            if i >= 0:
                self._combine_op.setCurrentIndex(i)
        self._log.append("Copied integration settings from Batch Integrate.")

    def _settings_dict(self) -> dict:
        return {"kernel": self._kernel.currentData(),
                "r_bin": self._r_bin.value(), "e_bin": self._e_bin.value(),
                "r_min": self._r_min.value(), "r_max": self._r_max.value(),
                "r_max_mode": self._r_max_mode.currentData(),
                "eta_min": self._eta_min.value(), "eta_max": self._eta_max.value(),
                "fmt": self._fmt.checked_keys(),
                "weighted": self._weighted.isChecked(),
                "chunk_size": self._combine_chunk.value(),
                "combine_op": self._combine_op.currentData(),
                "max_concurrent": self._concurrency.value()}

    def _apply_settings_dict(self, st: dict):
        if not st:
            return
        for key, widget in (("r_bin", self._r_bin), ("e_bin", self._e_bin),
                            ("r_min", self._r_min), ("r_max", self._r_max),
                            ("eta_min", self._eta_min), ("eta_max", self._eta_max)):
            if st.get(key) is not None:
                widget.setValue(float(st[key]))
        if st.get("kernel"):
            i = self._kernel.findData(st["kernel"])
            if i >= 0:
                self._kernel.setCurrentIndex(i)
        if st.get("r_max_mode"):
            i = self._r_max_mode.findData(st["r_max_mode"])
            if i >= 0:
                self._r_max_mode.setCurrentIndex(i)
        if st.get("fmt"):
            self._fmt.set_state(st["fmt"])
        if st.get("weighted") is not None:
            self._weighted.setChecked(bool(st["weighted"]))
        if st.get("chunk_size") is not None:
            self._combine_chunk.setValue(int(st["chunk_size"]))
        if st.get("combine_op"):
            i = self._combine_op.findData(st["combine_op"])
            if i >= 0:
                self._combine_op.setCurrentIndex(i)
        if st.get("max_concurrent"):
            self._concurrency.setValue(int(st["max_concurrent"]))

    # ── the mapping preview ──────────────────────────────────────

    def _sync_roots(self):
        self._queue.data_root = self._data_root.text().strip() or None
        self._queue.out_root = self._out_root.text().strip() or None

    def _refresh_preview(self):
        self._sync_roots()
        rows, problems = plan_output_dirs(self._queue)
        if problems:
            self._preview.setText("⚠  " + "  ".join(problems))
            return
        inferred = self._queue.data_root or infer_data_root(self._queue.sample_paths())
        head = rows[0]
        unrooted = sum(1 for r in rows if is_unrooted(r[3], self._queue.out_root))
        text = (f"{len(rows)} sample(s) · data root {inferred}\n"
                f"{head[2].path}\n    →  {head[3]}")
        if unrooted:
            text += (f"\n⚠  {unrooted} sample(s) sit outside the data root and go "
                     "to _unrooted/.")
        self._preview.setText(text)

    # ── running ──────────────────────────────────────────────────

    def _resolve_spec(self, cal: CalibrationNode):
        """One ``IntegrationSpec`` per calibration node, from its snapshot or
        its geometry file — the same two producers Batch Integrate uses."""
        st = self._settings_dict()
        kw = dict(r_min=st["r_min"], r_max=self._resolve_r_max(cal, st),
                  eta_min=st["eta_min"], eta_max=st["eta_max"])
        if cal.using_file():
            if not cal.file_path or not Path(cal.file_path).exists():
                raise FileNotFoundError(
                    f"'{cal.name}': calibration file not found: {cal.file_path}")
            return spec_from_geometry_file(cal.file_path, st["r_bin"], st["e_bin"], **kw)
        if not cal.calib_snapshot:
            raise RuntimeError(
                f"'{cal.name}' has no calibration snapshot — edit it and pick a "
                "source.")
        return _build_spec(project.calibration_namespace(cal.calib_snapshot),
                           st["r_bin"], st["e_bin"], **kw)

    def _resolve_r_max(self, cal: CalibrationNode, st: dict) -> Optional[float]:
        """The Rmax to integrate this calibration's samples to.

        Manual mode uses the queue-wide spinbox value as-is (0 → auto,
        matching Batch Integrate). Corner/Edge is computed per calibration
        node, from *this* node's own beam centre and detector size — a queue
        commonly spans stages with different detectors, so a single
        precomputed number (as Batch Integrate's one-shot preset buttons
        produce) would be wrong for every stage but the one it was taken
        from."""
        mode = st.get("r_max_mode", "manual")
        if mode == "manual":
            return st["r_max"] or None
        fields, note = self._calib_fields_with_note(cal)
        if not fields or fields.get("BC_y") is None or fields.get("NrPixelsY") is None:
            self._log.append(
                f"[queue] '{cal.name}': can't compute {mode} Rmax ({note}) — "
                "leaving Rmax at auto (backend's farthest-corner default).")
            return None
        formula = rmax_corner_px if mode == "corner" else rmax_edge_px
        return formula(fields["BC_y"], fields["BC_z"],
                       fields["NrPixelsY"], fields["NrPixelsZ"])

    def _calib_fields_with_note(self, cal: CalibrationNode):
        result = (project.calibration_namespace(cal.calib_snapshot)
                  if cal.calib_snapshot else None)
        return resolve_calibration_fields(
            result, cal.using_file(), cal.file_path or "",
            source_label=f"Batch Queue '{cal.name}'")

    def _calib_fields(self, cal: CalibrationNode):
        fields, _note = self._calib_fields_with_note(cal)
        return fields

    def _build_run_items(self, rows) -> list:
        from midas_gui.helpers import average_field, source_kind
        st = self._settings_dict()
        specs, items = {}, []
        for ci, (cal, corr, sample, out_dir) in enumerate(rows):
            cal_key = f"cal{id(cal)}"
            if cal_key not in specs:
                specs[cal_key] = self._resolve_spec(cal)
            spec = specs[cal_key]

            def _field(path):
                if not path:
                    return None
                try:
                    return average_field(source_kind(path), path)
                except Exception as e:
                    self._log.append(f"[{sample.label}] could not read '{path}': {e}")
                    return None

            background = _field(corr.background)
            if background is not None:
                background = background * corr.background_scale

            key = self._key_for(cal, corr, sample)
            items.append(RunItem(
                key=key, group=cal_key, label=sample.label, spec=spec,
                source_cfg=sample_source_cfg(
                    sample, chunk_size=st["chunk_size"] or None,
                    combine_op=st["combine_op"] or "mean"),
                out_dir=str(out_dir),
                mask=None, mask_is_file_backed=True,
                fmts=tuple(st["fmt"]) or ("csv",), kernel=st["kernel"],
                corrections=(None, None), variance_cfg=None, q_cfg=None,
                dark=_field(corr.dark), bright=_field(corr.bright),
                background=background, bright_mode=corr.bright_mode,
                weighted=st["weighted"],
                im_trans=tuple((self._calib_fields(cal) or {}).get("im_trans") or ()),
                calibration_snapshot=self._calib_fields(cal),
                extra={"corrections_node": corr.name, "calibration_node": cal.name}))
        return items

    def _key_for(self, cal, corr, sample) -> str:
        for ci, c in enumerate(self._queue.calibrations):
            if c is cal:
                for ki, k in enumerate(c.corrections):
                    if k is corr:
                        for si, s in enumerate(k.samples):
                            if s is sample:
                                return self._sample_key(ci, ki, si)
        return sample.path

    def _run(self):
        if self._scheduler is not None and self._scheduler.is_running():
            return
        self._sync_roots()
        rows, problems = plan_output_dirs(self._queue)
        if problems:
            QtWidgets.QMessageBox.warning(self, "Cannot run", "\n\n".join(problems))
            return
        reason = check_output_dir_writable(self._queue.out_root)
        if reason:
            QtWidgets.QMessageBox.warning(self, "Output root not writable", reason)
            return
        try:
            items = self._build_run_items(rows)
        except Exception:
            show_error(self, "Cannot run", traceback.format_exc(), log=self._log)
            return
        self._run_rows = {it.key: rows[i] for i, it in enumerate(items)}
        self._env = project.environment_snapshot()   # once, not once per sample

        for key, item in self._items.items():
            item.setText(COL_STATUS, "queued" if key in self._run_rows else "")
        self._scheduler = SampleRunScheduler(
            items, max_concurrent=self._concurrency.value(), parent=self)
        self._scheduler.sampleStarted.connect(
            lambda k: self._set_status(k, "running…"))
        self._scheduler.sampleProgress.connect(
            lambda k, d, t: self._set_status(k, f"{d}/{t}"))
        self._scheduler.sampleFinished.connect(self._on_sample_finished)
        self._scheduler.sampleFailed.connect(
            lambda k, _m: self._set_status(k, "failed"))
        self._scheduler.logLine.connect(self._log.append)
        self._scheduler.runFinished.connect(self._on_run_finished)

        self._run_btn.setEnabled(False); self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(items)); self._progress.setValue(0)
        self._scheduler.start()

    def _cancel(self):
        if self._scheduler is not None:
            self._scheduler.cancel()

    def _set_status(self, key: str, text: str):
        item = self._items.get(key)
        if item is not None:
            item.setText(COL_STATUS, text)

    def _on_sample_finished(self, key: str, data: dict):
        self._set_status(key, f"done ({data.get('n', 0)})")
        self._write_sample_project(key, data)
        if self._scheduler is not None:
            c = self._scheduler.counts()
            self._progress.setValue(c["done"] + c["failed"] + c["cancelled"])

    def _on_run_finished(self, counts: dict):
        self._run_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)

    # ── per-sample project ───────────────────────────────────────

    def _write_sample_project(self, key: str, data: dict):
        """One project per sample, beside its integrated data.

        ``create_project`` only when the file is missing: ``overwrite=True`` is
        destructive, and ``/analysis`` is append-only, so re-running a sample
        adds ``attempt_0002`` rather than discarding the first run.
        ``write_gui_workspace`` runs *before* the attempt because it copies the
        whole file to ``.bak`` on every call — cheap while the project is still
        near-empty, expensive once the results arrays are in it."""
        row = self._run_rows.get(key)
        if row is None:
            return
        cal, corr, sample, out_dir = row
        proj_path = project_path_for(out_dir, sample.label)
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            if not proj_path.exists():
                project.create_project(str(proj_path), name=sample.label)
            project.write_gui_workspace(
                str(proj_path),
                tabs={"Batch Queue": {"sample": sample.to_json(),
                                       "corrections": corr.to_json(),
                                       "calibration": cal.to_json(),
                                       "settings": self._settings_dict()}},
                meta={"expid": "", "active_profile": settings.active_profile()})
            ref = project.append_integration_attempt(
                str(proj_path), "single",
                inputs={"src_cfg": sample_source_cfg(
                            sample, chunk_size=self._combine_chunk.value() or None,
                            combine_op=self._combine_op.currentData() or "mean"),
                        "kernel": self._kernel.currentData(),
                        "fmt": self._fmt.checked_keys(),
                        "dark": corr.dark, "bright": corr.bright,
                        "background": corr.background,
                        "background_scale": corr.background_scale,
                        "r_bin": self._r_bin.value(), "e_bin": self._e_bin.value()},
                finished_payload=data,
                calibration_snapshot=self._calib_fields(cal),
                # An intra-file HDF5 path that would not resolve in this
                # per-sample project; the origin goes in `extra` instead.
                calib_attempt_ref=None,
                environment=getattr(self, "_env", None),
                extra={"active_profile": settings.active_profile(),
                       "queue": {"calibration_node": cal.name,
                                  "corrections_node": corr.name,
                                  "data_root": self._queue.data_root,
                                  "out_root": self._queue.out_root},
                       "source_calib_attempt": self._source_calib_attempt()})
            self._log.append(f"[{sample.label}] project: {proj_path}  ({ref})")
        except Exception:
            self._log.append(f"[{sample.label}] could not write its project file:\n"
                             + traceback.format_exc())

    def _source_calib_attempt(self) -> Optional[dict]:
        ctx = self._project_ctx
        ref = getattr(self._calib_result, "_project_attempt_ref", None)
        if ctx is None or not getattr(ctx, "path", None) or not ref:
            return None
        return {"project": ctx.path, "ref": ref}

    # ── project state ────────────────────────────────────────────

    def get_state(self) -> dict:
        self._sync_roots()
        self._queue.settings = self._settings_dict()
        return {"queue": self._queue.to_json()}

    def set_state(self, state: dict):
        if not state:
            return
        self._queue = BatchQueue.from_json((state or {}).get("queue"))
        self._data_root.blockSignals(True)
        self._out_root.blockSignals(True)
        self._data_root.setText(self._queue.data_root or "")
        self._out_root.setText(self._queue.out_root or "")
        self._data_root.blockSignals(False)
        self._out_root.blockSignals(False)
        self._apply_settings_dict(self._queue.settings)
        self._rebuild_tree()

    def shutdown(self):
        if self._scheduler is not None and self._scheduler.is_running():
            self._scheduler.cancel()
