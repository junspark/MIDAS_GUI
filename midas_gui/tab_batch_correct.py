"""Tab — Batch Correction.

Reduce each selected HDF5 file's sub-frame stack to one frame per group of
N consecutive sub-frames (mean / median / sum / max), correcting dark /
bright / background along the way, and write the result out as HDF5 — one
output file per input file.

What this tab is NOT: an integration. Nothing here needs a calibration. It
produces corrected detector frames, which Batch Integrate (or anything else)
can then consume.

Almost all of the machinery is borrowed rather than new:

* the source picker, the Dark/Bright/Background fields, the frame range and
  the "Combine sub-frames: N / op:" row all come from the shared
  ``DataLoaderPanel(mode="stream", unify_combine=True)`` — the same panel
  Batch Integrate uses, so N and the op mean exactly what they mean there;
* the chunk arithmetic is ``helpers._stack_chunk_bounds`` via
  ``frame_correct.chunk_ranges``, so a reduced frame covers exactly the
  exposures Batch Integrate would have integrated together;
* the per-chunk metadata averaging is ``h5_metadata.align``.

The genuinely new parts live in ``midas_gui.frame_correct``: the order in
which corrections are applied (see that module's docstring — it is the one
thing ``sum`` gets wrong if done the obvious way) and the dark-file ladder.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui import frame_correct as FC
from midas_gui import run_log
from midas_gui import style as S
from midas_gui.dialogs import show_error
from midas_gui.helpers import (CORRECTION_EXT, CORRECTION_SUFFIX,
                               correction_subdir,
                               _NoScrollComboBox, _NoScrollSpinBox,
                               apply_dict_to_widgets, browse_start_dir,
                               check_output_dir_writable,
                               suggest_correction_output_dir, widgets_to_dict,
                               warn_if_path_missing)
from midas_gui.widgets import DataLoaderPanel, ImageViewer, LogPanel
from midas_gui.job_queue import JobQueuePanel
from midas_gui.workers import BatchCorrectionCoordinator, BatchCorrectionWorker


class BatchCorrectionTab(QtWidgets.QWidget):
    """Chunked frame reduction with field correction, HDF5 in → HDF5 out."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[BatchCorrectionWorker] = None
        # Explicit, rather than asking the worker whether it isRunning():
        # `finished` is emitted from INSIDE QThread.run(), so at the moment
        # the handler fires the thread is still alive and isRunning() is
        # True. Deciding button state from that left Run/Preview disabled
        # and Cancel enabled after a completed run, with nothing scheduled
        # to correct it.
        self._running = False
        self._outputs: list = []
        self._im_trans: list = []
        self._expid_provider = None   # () -> str, wired by app.py
        # Open only while a run is in flight — see _emit / _run / _close_log.
        self._screen_log: Optional[run_log.ScreenLog] = None
        self._build_ui()

    def set_expid_provider(self, provider) -> None:
        """Header Exp ID field, read live for the output-path suggestion —
        same callback Batch Integrate and Calibrate take."""
        self._expid_provider = provider

    # ── cross-tab wiring ────────────────────────────────────────────────
    def set_display_transform(self, codes):
        """Follow the Data Viewer's flips in the preview, the same way the
        Mask Builder does — display only; nothing written is transformed."""
        self._im_trans = list(codes or [])
        if self._preview_raw is not None:
            self._viewer.set_raw_frame(self._preview_raw, self._im_trans,
                                       autorange=False)

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False); split.setHandleWidth(6)
        root.addWidget(split)

        # ── LEFT: the shared loader. unify_combine=True is what surfaces
        # "Combine sub-frames: N / op:" for every source type — this tab's
        # entire reduction control, reused rather than rebuilt.
        self._loader = DataLoaderPanel(mode="stream", unify_combine=True)
        self._loader.setMinimumWidth(200)
        # The method is chosen by the checkboxes in the Correction card (one
        # run can produce several), so the panel's single-choice "op:"
        # dropdown would be a dead control sitting beside the live one.
        self._loader.set_combine_op_visible(False)
        self._loader.dataChanged.connect(self._on_data_changed)
        # The range/combine spins don't emit dataChanged, so the planned
        # frame count would go stale exactly when it matters most.
        for attr in ("_fr_start", "_fr_end", "_combine_chunk"):
            spin = getattr(self._loader, attr, None)
            if spin is not None:
                spin.valueChanged.connect(self._refresh_name_preview)
        split.addWidget(self._loader)

        # ── MIDDLE: output + options ──
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setMinimumWidth(280)
        inner = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(inner)
        lv.setContentsMargins(6, 6, 6, 6); lv.setSpacing(8)
        scroll.setWidget(inner)

        lv.addWidget(self._build_output_card())
        lv.addWidget(self._build_options_card())
        lv.addWidget(self._build_run_card())
        lv.addStretch(1)
        split.addWidget(scroll)

        # ── RIGHT: preview + log ──
        self._preview_raw: Optional[np.ndarray] = None
        right = QtWidgets.QTabWidget()
        self._viewer = ImageViewer()
        right.addTab(self._viewer, "Preview")
        self._log = LogPanel()
        # This tab's own queue. BatchTab has a separate JobQueuePanel in its
        # own Logs area (tab_batch.py); one instance per tab is what keeps
        # the two tabs' jobs independent.
        self._job_queue = JobQueuePanel(self._log, on_job_done=self._on_job_done)
        right.addTab(self._log, "Log")
        split.addWidget(right)
        split.setStretchFactor(2, 1)
        for w in (self._out_ed, self._suffix_ed, self._ext_ed, self._out_ds_ed,
                  self._dark_ds_ed):
            w.setCursorPosition(0)
        self._refresh_name_preview()
        self._on_ops_changed()
        self._refresh_enabled()

    def _build_output_card(self):
        card = S.make_card("Output")
        self._out_ed = QtWidgets.QLineEdit()
        self._out_ed.setPlaceholderText("Output directory…")
        warn_if_path_missing(self._out_ed, self, is_output_dir=True)
        row = QtWidgets.QHBoxLayout(); row.setSpacing(4)
        row.addWidget(self._out_ed, 1)
        browse = QtWidgets.QPushButton("…"); browse.setFixedWidth(30)
        browse.clicked.connect(lambda: self._out_ed.setText(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Output directory",
                browse_start_dir(self._out_ed.text())) or ""))
        row.addWidget(browse)
        self._suggest_btn = QtWidgets.QPushButton("Suggest")
        self._suggest_btn.setToolTip(
            "Fill in <outroot>/<expid>_bc/<froot>/<detector>/dark_subtracted/ "
            "— Batch Integrate's own output convention with one more segment "
            "naming what the files are, so a froot's reduced frames and its "
            "cakes sit side by side.\n\n"
            "Read positionally off the loaded source's folder depth, the same "
            "way Batch Integrate's Suggest is, so it needs no Exp ID typed in.")
        self._suggest_btn.clicked.connect(self._apply_suggested_output_dir)
        row.addWidget(self._suggest_btn)
        card.body.addLayout(row)

        form = S.Form()
        self._suffix_ed = QtWidgets.QLineEdit(CORRECTION_SUFFIX)
        # Placeholder, not just a default value: these two fields are saved
        # into a project, so an old project keeps pinning whatever tail it
        # was saved with long after the shipped default moves. Showing the
        # default in an emptied field is what makes that recoverable
        # without guessing.
        self._suffix_ed.setPlaceholderText(CORRECTION_SUFFIX)
        self._suffix_ed.setToolTip(
            "Appended to the source name, before the extension. Only the\n"
            "final extension is replaced, so a detector tag is kept:\n\n"
            "  AgBeH_10s_000021.h5   →  AgBeH_10s_000021_cor.h5\n"
            "  CeO2_030319.vrx.h5    →  CeO2_030319.vrx_cor.h5\n\n"
            "Saved with the project, so an older project keeps its own\n"
            "tail — clear the field to fall back to the default shown.")
        self._ext_ed = QtWidgets.QLineEdit(CORRECTION_EXT)
        self._ext_ed.setPlaceholderText(CORRECTION_EXT)
        self._ext_ed.setFixedWidth(64)
        self._ext_ed.setToolTip(
            "Output file extension. .hdf, .h5, .hdf5 and .nxs are all\n"
            "recognised as HDF5 by this GUI's own loaders.")
        self._out_ds_ed = QtWidgets.QLineEdit("exchange/data")
        self._out_ds_ed.setMinimumWidth(150)
        self._out_ds_ed.setToolTip(
            "HDF5 path the reduced (M, H, W) stack is written to.\n"
            "Leaving this at exchange/data means the output can be loaded\n"
            "straight back into any tab of this GUI.")
        form.row(("Suffix:", self._suffix_ed), ("ext:", self._ext_ed))
        form.row(("Dataset:", self._out_ds_ed))

        self._dtype_combo = _NoScrollComboBox()
        for name in FC.OUTPUT_DTYPES:
            self._dtype_combo.addItem(name, name)
        self._dtype_combo.setToolTip(
            "On-disk dtype for the corrected stack.\n\n"
            "float32 — the default, and what you want in almost every case.\n"
            "          GSAS-II reads float HDF5 images fine: its HDF5 reader\n"
            "          has no dtype check at all and its integration casts to\n"
            "          float anyway.\n"
            "uint32  — for workflows that hand the data to something\n"
            "          integer-only. GSAS-II's *TIFF* reader is the usual\n"
            "          reason: it truncates float32 to int32 on load.\n\n"
            "The cast happens last, after every correction, so it never\n"
            "affects the arithmetic. Unsigned output cannot hold negatives,\n"
            "so values are rounded, then clipped to 0, and the number of\n"
            "clipped pixels is reported in the progress log — background\n"
            "subtraction routinely goes negative, and an unreported clip\n"
            "would be invisible data loss.")
        self._comp_combo = _NoScrollComboBox()
        self._comp_combo.addItem("None", "none")
        self._comp_combo.addItem("gzip", "gzip")
        self._comp_combo.addItem("lzf", "lzf")
        self._comp_combo.setToolTip(
            "HDF5 compression for the output stack.\n\n"
            "None  — fastest to write and to read back; the right choice for\n"
            "        a scratch reduction something else consumes immediately.\n"
            "gzip  — smallest, slowest. Pair it with Shuffle.\n"
            "lzf   — a middle option: much faster than gzip, less effective.\n\n"
            "A compressed dataset is chunked one frame per chunk, so reading\n"
            "a single frame never decompresses the whole stack.")
        self._comp_level = _NoScrollSpinBox()
        self._comp_level.setRange(0, 9); self._comp_level.setValue(4)
        self._comp_level.setToolTip("gzip level. Higher is smaller and slower.")
        self._shuffle_chk = QtWidgets.QCheckBox("Shuffle")
        self._shuffle_chk.setChecked(True)
        self._shuffle_chk.setToolTip(
            "Byte-transpose each plane before compressing. On float32\n"
            "detector data this groups the near-constant exponent bytes\n"
            "together and typically buys more than raising the gzip level.\n"
            "(Less of a win on uint32 output, which has no exponent byte to\n"
            "group — but still usually worth leaving on.)")
        form.row(("Output dtype:", self._dtype_combo))
        form.row(("Compression:", self._comp_combo),
                 ("level:", self._comp_level))
        form.full(self._shuffle_chk)
        card.body.addLayout(form)

        # Per-frame beam-monitor CSV, one per SOURCE FILE in the output root.
        # Always written (the ion chambers are a property of the exposure, so
        # one copy per combine op would be the same table four times); these
        # only add optional columns. Same control set as Batch Integrate — see
        # midas_gui.ion_csv.
        ion_lbl = QtWidgets.QLabel(
            "Ion-chamber CSV — written automatically, one row per output "
            "frame (frame, I0, I, transmission). Also include:")
        ion_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        ion_lbl.setWordWrap(True)
        card.body.addWidget(ion_lbl)
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
        card.body.addLayout(ion_row)
        self._name_lbl = QtWidgets.QLabel("")
        self._name_lbl.setWordWrap(True)
        self._name_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        card.body.addWidget(self._name_lbl)
        self._comp_combo.currentIndexChanged.connect(self._refresh_compression_row)
        self._suffix_ed.textChanged.connect(self._refresh_name_preview)
        self._ext_ed.textChanged.connect(self._refresh_name_preview)
        self._refresh_compression_row()
        return card

    def _build_options_card(self):
        card = S.make_card("Correction")

        # Checkboxes, not the loader's single-choice "op:" dropdown: the
        # expensive part of a run is reading and correcting each sub-frame,
        # which is identical whichever op consumes it, so computing all four
        # costs one pass instead of four (frame_correct.reduce_chunk_multi).
        # Each lands in its own dark_subtracted_<op> folder.
        card.body.addWidget(QtWidgets.QLabel("Methods (one output set each):"))
        self._op_chks = {}
        grid = QtWidgets.QGridLayout(); grid.setSpacing(4)
        for i, op in enumerate(FC.OPS):
            chk = QtWidgets.QCheckBox(op.capitalize())
            chk.setToolTip(
                f"Write a {op} reduction into "
                f"<output>/{correction_subdir(op)}/.\n\n"
                "Several methods cost one pass over the data, not one each — "
                "reading and\ncorrecting a sub-frame is the same work "
                "whichever method consumes it.")
            chk.toggled.connect(self._on_ops_changed)
            self._op_chks[op] = chk
            grid.addWidget(chk, i // 2, i % 2)
        self._op_chks["mean"].setChecked(True)
        card.body.addLayout(grid)
        self._op_note = QtWidgets.QLabel("")
        self._op_note.setWordWrap(True)
        self._op_note.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        card.body.addWidget(self._op_note)
        card.body.addWidget(S.hline())

        self._auto_dark_chk = QtWidgets.QCheckBox("Find each file's own dark")
        self._auto_dark_chk.setChecked(True)
        self._auto_dark_chk.setToolTip(
            "Resolve a dark per input file instead of using one dark for the\n"
            "whole run, preferring the most recently measured one:\n\n"
            "  1. the nearest *_dark_before_* sibling BEFORE this file's\n"
            "     number (darks are re-measured mid-scan, so this is\n"
            "     usually not the scan's first dark)\n"
            "  2. the nearest *_dark_after_* sibling after it\n"
            "  3. the file's own exchange/data_dark\n"
            "  4. the Dark field selected on the left\n\n"
            "Whichever is used is named in the Log, per file.\n"
            "Untick to use the Dark field on the left for everything.")
        card.body.addWidget(self._auto_dark_chk)

        form = S.Form()
        self._dark_ds_ed = QtWidgets.QLineEdit("exchange/data_dark")
        self._dark_ds_ed.setToolTip(
            "Dataset the dark pixels are read from inside whichever file\n"
            "the search above lands on.")
        form.row(("Dark dataset:", self._dark_ds_ed))
        card.body.addLayout(form)

        self._clip_chk = QtWidgets.QCheckBox("Clip negatives to zero")
        self._clip_chk.setChecked(True)
        self._clip_chk.setToolTip(
            "Applied ONCE, to the combined frame — not to each sub-frame.\n"
            "Clipping before combining would discard the negative half of\n"
            "the read noise and bias a Sum upward.\n\n"
            "Untick to keep negative pixels, e.g. to check that a dark is\n"
            "centred rather than over-subtracting.")
        card.body.addWidget(self._clip_chk)

        # The three fields on the left do NOT behave alike across a
        # multi-file run, and nothing on screen said so — asked directly:
        # "it is unclear the role of these dark / bright / background in the
        # case when we are doing multiple files."
        scope = QtWidgets.QLabel(
            "Across several files: <b>Bright</b> and <b>Background</b> are "
            "applied unchanged to every file. <b>Dark</b> is resolved per "
            "file when the box above is ticked — the Dark field on the left "
            "is then only the fallback for a file with none of its own. "
            "Untick it and that one Dark is used for every file.")
        scope.setWordWrap(True)
        scope.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        card.body.addWidget(scope)
        card.body.addWidget(S.hline())

        note = QtWidgets.QLabel(
            "Each sub-frame is corrected before the frames are combined, so a "
            "Sum of N frames loses N darks rather than one. Set N in "
            "“Combine sub-frames” on the left; the method is chosen here.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        card.body.addWidget(note)
        return card

    def _build_run_card(self):
        card = S.make_card("Run")
        # Run mode, worded and shaped as Batch Integrate's (tab_batch.py) --
        # the same choice should not read differently in two tabs. Files are
        # the unit of parallelism here: chunks never cross a file boundary
        # and each input yields its own output, so workers share nothing
        # (BatchCorrectionCoordinator's docstring has the rest).
        import os
        _max_workers = max(1, (os.cpu_count() or 4))
        mode_row = QtWidgets.QHBoxLayout(); mode_row.setSpacing(4)
        mode_row.addWidget(S.LabelRight("Mode:"))
        self._run_mode = _NoScrollComboBox()
        self._run_mode.addItem("Sequential", "sequential")
        self._run_mode.addItem("Parallel", "parallel")
        self._run_mode.setToolTip(
            "Sequential \u2014 one file at a time.\n"
            "Parallel \u2014 several files at once, on worker threads. Each "
            "output is independent, so the result is identical either way; "
            "this only trades memory and disk bandwidth for wall-clock.")
        mode_row.addWidget(self._run_mode, 1)
        mode_row.addWidget(S.LabelRight("Workers:"))
        self._n_workers = _NoScrollSpinBox()
        self._n_workers.setRange(1, _max_workers)
        self._n_workers.setValue(min(4, _max_workers))
        self._n_workers.setEnabled(False)
        self._run_mode.currentIndexChanged.connect(
            lambda: self._n_workers.setEnabled(
                self._run_mode.currentData() == "parallel"))
        mode_row.addWidget(self._n_workers)
        card.body.addLayout(mode_row)
        self._preview_btn = QtWidgets.QPushButton("Preview first chunk")
        self._preview_btn.setToolTip(
            "Reduce only the first chunk of the first file and show it, so a "
            "long run isn't launched blind.")
        self._preview_btn.clicked.connect(self._preview)
        self._run_btn = QtWidgets.QPushButton("Run correction")
        self._run_btn.clicked.connect(self._run)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel)
        self._cancel_btn.setEnabled(False)
        card.body.addLayout(S.button_grid(
            [self._preview_btn, self._run_btn, self._cancel_btn], 2))
        self._job_btn = QtWidgets.QPushButton("Run as background job")
        self._job_btn.setToolTip(
            "Run this correction in a detached `screen` session instead of "
            "in this process, so it survives closing the GUI. Needs an "
            "output folder: the job writes its field images and its results "
            "list there.")
        self._job_btn.clicked.connect(self._run_as_job)
        card.body.addWidget(self._job_btn)
        self._prog = QtWidgets.QProgressBar()
        self._prog.setRange(0, 100); self._prog.setVisible(False)
        card.body.addWidget(self._prog)
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        card.body.addWidget(self._status)
        return card

    def _refresh_name_preview(self):
        """Show what the first input file will be called on disk, and how
        many output frames it will produce.

        The frame count is not decoration. start/end are a raw SUB-FRAME
        window here, and leaving end at 0 clamps that window to sub-frame 0
        alone — one output frame, with "Combine sub-frames" silently having
        nothing to combine. Printing "20 raw → 1 frame" makes that visible
        before a run rather than after one.
        """
        paths = self._h5_paths()
        if not paths:
            self._name_lbl.setText("")
            return
        src = Path(paths[0])
        text = f"{src.name}  →  {self._out_name(src)}"
        plan = self._plan_for(src)
        if plan is not None:
            n_raw, n_out = plan
            text += f"\n{n_raw} raw sub-frame(s) → {n_out} output frame(s)"
            if n_out == 1 and n_raw > 1 and (self._chunk_settings()[0] or 0) != 0:
                text += "  ⚠ check start/end"
        self._name_lbl.setText(text)

    def _plan_for(self, src: Path):
        """``(raw sub-frames in file, output frames)`` for ``src`` under the
        current settings, from the dataset SHAPE alone — one h5py header
        read, no pixels."""
        try:
            import h5py
            chunk, fr_start, fr_end = self._chunk_settings()
            with h5py.File(str(src), "r") as f:
                dset = f[self._loader._dataset()]
                if dset.ndim == 2:
                    return 1, 1
                n_raw = int(dset.shape[0])
            return n_raw, len(FC.chunk_ranges(n_raw, chunk_size=chunk,
                                              raw_start=fr_start, raw_end=fr_end))
        except Exception:
            return None

    def _out_name(self, src: Path) -> str:
        """Mirror of ``BatchCorrectionWorker._out_path``'s naming, for the
        preview label. Kept trivial on purpose; the worker remains the one
        that actually names the file."""
        base = src.stem          # keeps any detector tag — see _out_path
        ext = self._ext_ed.text().strip() or CORRECTION_EXT
        if not ext.startswith("."):
            ext = "." + ext
        suffix = self._suffix_ed.text().strip() or CORRECTION_SUFFIX
        return f"{base}{suffix}{ext}"

    def _apply_suggested_output_dir(self):
        suggested = self._suggest_output_dir()
        if suggested is None:
            self._log.append("No data source loaded yet — nothing to suggest.")
            return
        self._out_ed.setText(str(suggested))
        reason = check_output_dir_writable(suggested)
        if reason:
            self._log.append(f"Warning: {reason}")

    def _suggest_output_dir(self) -> Optional[Path]:
        cfg = self._loader.source_cfg()
        rep = cfg.get("path")
        if not rep:
            paths = cfg.get("paths") or []
            rep = paths[0] if paths else None
        if not rep:
            return None
        expid = self._expid_provider().strip() if self._expid_provider else ""
        return suggest_correction_output_dir(rep, expid_fallback=expid)

    def _maybe_autofill_output_dir(self):
        """Fill the Output folder once a source loads, unless something is
        already there — same live auto-fill as Batch Integrate, which has no
        separate confirm-and-launch step to catch a wrong guess either."""
        if self._out_ed.text().strip():
            return
        suggested = self._suggest_output_dir()
        if suggested is not None:
            self._out_ed.setText(str(suggested))

    def _selected_ops(self) -> list:
        return [op for op in FC.OPS if self._op_chks[op].isChecked()]

    def _on_ops_changed(self):
        # Reachable mid-build: setting the default method fires `toggled`
        # before the note label and the Run button exist.
        if not hasattr(self, "_op_note"):
            return
        ops = self._selected_ops()
        if not ops:
            self._op_note.setText("⚠ pick at least one method.")
        else:
            self._op_note.setText("→ " + ",  ".join(
                correction_subdir(op) + "/" for op in ops))
        if hasattr(self, "_run_btn"):
            self._refresh_enabled()

    def _refresh_compression_row(self):
        gzip = self._comp_combo.currentData() == "gzip"
        self._comp_level.setEnabled(gzip)
        self._shuffle_chk.setEnabled(self._comp_combo.currentData() != "none")

    # ── source resolution ───────────────────────────────────────────────
    def _h5_paths(self) -> list:
        """The selected HDF5 files, already start/end filtered, or ``[]``.

        Only HDF5 sources are accepted. "Chunks restart at every file" has
        no meaning for TIFF/GE, where a file holds exactly one frame and
        there is no sub-frame stack to group — see :meth:`_run`, which says
        so rather than silently applying a different rule.

        For a MULTI-file pick, ``frame_start``/``frame_end`` are FILE (scan)
        NUMBERS and belong here, narrowing the list — the same filter
        ``workers._open_source_cfg`` applies for Batch Integrate. They are
        emphatically not sub-frame bounds; see :meth:`_chunk_settings`.
        """
        cfg = self._loader.source_cfg()
        if cfg.get("type") == "hdf5":
            return [cfg["path"]]
        if cfg.get("type") == "hdf5_stack_glob":
            from midas_gui.workers import _filter_paths_by_frame_number
            return list(_filter_paths_by_frame_number(
                cfg.get("paths") or [], cfg.get("frame_start"),
                cfg.get("frame_end")))
        return []

    def _chunk_settings(self) -> tuple:
        """``(chunk_size, raw_start, raw_end)`` — bounds in RAW SUB-FRAME
        space, which is the only thing the chunker can use.

        ``source_cfg()``'s ``frame_start``/``frame_end`` carry two different
        meanings under one pair of keys, and getting them mixed up is not a
        small error:

        * single ``"hdf5"`` file → 0-based inclusive RAW SUB-FRAME bounds
          within that file. Pass them through.
        * multi-file ``"hdf5_stack_glob"`` → FILE (scan) NUMBERS. They have
          already narrowed the list in :meth:`_h5_paths`, and feeding them
          on as sub-frame bounds is nonsense: a start of 1382 against a
          12-sub-frame file leaves an empty window, so every file produces
          zero chunks and the run dies with "the selected frame range
          leaves no sub-frames". Return None here instead — each file is
          chunked over its whole stack.

        Deliberately does NOT return the panel's ``combine_op``: the method
        is this tab's own (the Correction card's checkboxes), and returning
        a value nobody uses is how the dropdown came to look meaningful.
        """
        cfg = self._loader.source_cfg()
        if cfg.get("type") == "hdf5":
            return (cfg.get("chunk_size"), cfg.get("frame_start"),
                    cfg.get("frame_end"))
        return cfg.get("chunk_size"), None, None

    def _on_data_changed(self):
        self._maybe_autofill_output_dir()
        self._refresh_name_preview()
        self._refresh_enabled()

    def _refresh_enabled(self):
        has = bool(self._h5_paths()) and bool(self._selected_ops())
        self._run_btn.setEnabled(has and not self._running)
        self._preview_btn.setEnabled(has and not self._running)
        self._cancel_btn.setEnabled(self._running)

    def _worker_kwargs(self, paths) -> dict:
        chunk, fr_start, fr_end = self._chunk_settings()
        return dict(
            dataset=self._loader._dataset(), chunk_size=chunk,
            op=self._selected_ops(),
            out_dataset=self._out_ds_ed.text().strip() or "exchange/data",
            suffix=(self._suffix_ed.text().strip() or CORRECTION_SUFFIX),
            out_ext=self._ext_ed.text().strip() or CORRECTION_EXT,
            dark=self._loader.dark(), bright=self._loader.bright(),
            background=self._loader.background(),
            bright_mode=self._loader.bright_mode(),
            auto_dark=self._auto_dark_chk.isChecked(),
            dark_dataset=self._dark_ds_ed.text().strip() or "exchange/data_dark",
            clip_negatives=self._clip_chk.isChecked(),
            compression=self._comp_combo.currentData(),
            level=self._comp_level.value(),
            shuffle=self._shuffle_chk.isChecked(),
            out_dtype=self._dtype_combo.currentData() or "float32",
            ion_csv_extras=tuple(
                k for k, chk in (("env", self._ion_env_chk),
                                 ("motors", self._ion_motors_chk))
                if chk.isChecked()),
            raw_start=fr_start, raw_end=fr_end)

    # ── preview ─────────────────────────────────────────────────────────
    def _preview(self):
        paths = self._h5_paths()
        if not paths:
            return
        try:
            import h5py
            chunk, fr_start, fr_end = self._chunk_settings()
            path = Path(paths[0])
            with h5py.File(str(path), "r") as f:
                dset = f[self._loader._dataset()]
                if dset.ndim == 2:
                    raw = np.asarray(dset[...], np.float32)[None]
                else:
                    ranges = FC.chunk_ranges(int(dset.shape[0]), chunk_size=chunk,
                                             raw_start=fr_start, raw_end=fr_end)
                    if not ranges:
                        show_error(self, "Nothing to preview",
                                   "The selected frame range leaves no sub-frames.")
                        return
                    lo, hi = ranges[0]
                    raw = np.asarray(dset[lo:hi + 1], np.float32)
                dark, why = FC.resolve_dark(
                    path, dark_dataset=self._dark_ds_ed.text().strip()
                    or "exchange/data_dark",
                    shape=tuple(raw.shape[-2:]), fallback=self._loader.dark(),
                    auto=self._auto_dark_chk.isChecked())
            op = (self._selected_ops() or ["mean"])[0]
            img = FC.reduce_chunk(
                raw, op, dark=dark, bright=self._loader.bright(),
                bright_mode=self._loader.bright_mode(),
                background=self._loader.background(),
                clip_negatives=self._clip_chk.isChecked())
        except Exception as exc:
            show_error(self, "Preview failed", str(exc))
            return
        self._preview_raw = img
        self._viewer.set_raw_frame(img, self._im_trans)
        self._status.setText(
            f"{path.name}: {op} over {raw.shape[0]} frame(s) → {img.shape}")
        self._log.append(f"Preview — {path.name}: {op} over {raw.shape[0]} "
                         f"frame(s); dark = {why}")

    # ── run ─────────────────────────────────────────────────────────────
    def _run(self):
        if self._running:
            return
        paths = self._h5_paths()
        if not paths:
            cfg_type = self._loader.source_cfg().get("type")
            if cfg_type in ("tiff_glob", "tiff_list"):
                show_error(
                    self, "HDF5 sources only",
                    "Batch Correction reduces each file's internal sub-frame "
                    "stack, and chunks never cross a file boundary.\n\n"
                    "A TIFF/GE file holds exactly one frame, so there is no "
                    "stack within a file to combine. Select HDF5 files "
                    "instead.")
            else:
                show_error(self, "No data", "Select an HDF5 file or files.")
            return
        out_dir = self._out_ed.text().strip()
        if not out_dir:
            show_error(self, "No output folder",
                       "Choose a folder for the corrected files.")
            return
        reason = check_output_dir_writable(out_dir)
        if reason:
            show_error(self, "Output folder not writable", reason)
            return

        chunk, _s, _e = self._chunk_settings()
        ops = self._selected_ops()
        if not ops:
            show_error(self, "No method selected",
                       "Tick at least one of Mean / Median / Sum / Max.")
            return
        self._outputs = []
        self._open_screen_log(paths, ops, chunk, out_dir)
        self._emit(
            f"Batch Correction — {len(paths)} file(s) × {len(ops)} method(s) "
            f"({', '.join(ops)}), chunk={chunk or 'whole file'}, out={out_dir}")
        for p in paths:
            self._emit(f"  source: {p}", panel=False)
        self._prog.setVisible(True); self._prog.setValue(0)
        n_workers = (self._n_workers.value()
                     if self._run_mode.currentData() == "parallel" else 1)
        # Always the coordinator, even for one worker: one code path means
        # Sequential cannot drift from Parallel, and at n=1 it is a single
        # BatchCorrectionWorker with a pass-through for the monitor CSVs.
        self._worker = BatchCorrectionCoordinator(
            paths, out_dir=out_dir, n_workers=n_workers, parent=self,
            **self._worker_kwargs(paths))
        if n_workers > 1:
            self._emit(f"  run mode: parallel, {n_workers} worker(s) over "
                       f"{len(paths)} file(s)", panel=False)
        self._worker.progress.connect(self._on_progress)
        self._worker.fileDone.connect(self._on_file_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._running = True
        self._worker.start()
        self._refresh_enabled()

    def _run_as_job(self):
        """Launch this same correction as a detached `screen` job (see
        job_queue.JobQueuePanel) instead of running in-process -- survives
        closing this GUI. Everything the CLI needs goes to disk first: the
        background process (a fresh `python -m midas_gui.correct_cli`) has
        no access to this GUI\u2019s live state. Mirrors
        BatchTab._run_as_job."""
        import sys
        paths = self._h5_paths()
        if not paths:
            cfg_type = self._loader.source_cfg().get("type")
            if cfg_type in ("tiff_glob", "tiff_list"):
                show_error(
                    self, "HDF5 sources only",
                    "Batch Correction reduces each file\u2019s internal sub-frame "
                    "stack, and chunks never cross a file boundary.\n\n"
                    "A TIFF/GE file holds exactly one frame, so there is no "
                    "stack within a file to combine. Select HDF5 files "
                    "instead.")
            else:
                show_error(self, "No data", "Select an HDF5 file or files.")
            return
        ops = self._selected_ops()
        if not ops:
            show_error(self, "No method selected",
                       "Tick at least one of Mean / Median / Sum / Max.")
            return
        out_dir = self._out_ed.text().strip()
        if not out_dir:
            show_error(self, "Output folder required",
                       "Background jobs need an output folder \u2014 the job "
                       "writes its field images and results list there.")
            return
        reason = check_output_dir_writable(out_dir)
        if reason:
            show_error(self, "Output folder not writable", reason)
            return
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        kw = self._worker_kwargs(paths)
        argv = [sys.executable, "-m", "midas_gui.correct_cli",
                "--paths", *[str(p) for p in paths],
                "--dataset", str(kw["dataset"] or "exchange/data"),
                "--out-dir", str(out_path),
                "--ops", ",".join(ops),
                "--chunk-size", str(kw["chunk_size"] or 0),
                "--suffix", kw["suffix"], "--ext", kw["out_ext"],
                "--out-dataset", kw["out_dataset"],
                "--out-dtype", kw["out_dtype"],
                "--compression", str(kw["compression"] or "none"),
                "--level", str(kw["level"]),
                "--dark-dataset", kw["dark_dataset"],
                "--bright-mode", kw["bright_mode"]]
        argv += ["--shuffle"] if kw["shuffle"] else []
        argv += ["--auto-dark"] if kw["auto_dark"] else ["--no-auto-dark"]
        argv += (["--clip-negatives"] if kw["clip_negatives"]
                 else ["--no-clip-negatives"])
        if kw["ion_csv_extras"]:
            argv += ["--ion-csv-extras", ",".join(sorted(kw["ion_csv_extras"]))]
        if kw["raw_start"] is not None:
            argv += ["--raw-start", str(kw["raw_start"])]
        if kw["raw_end"] is not None:
            argv += ["--raw-end", str(kw["raw_end"])]
        n_workers = (self._n_workers.value()
                     if self._run_mode.currentData() == "parallel" else 1)
        argv += ["--n-workers", str(n_workers)]

        # Field images can\u2019t travel on a command line. Same approach, and
        # the same _bg_job_ naming, as BatchTab._run_as_job.
        import tifffile
        for name in ("dark", "bright", "background"):
            arr = kw.get(name)
            if arr is None:
                continue
            f = out_path / f"_bg_job_{name}.tif"
            tifffile.imwrite(str(f), np.asarray(arr, dtype=np.float32))
            argv += [f"--{name}", str(f)]

        total = sum(1 for _ in paths)
        job = self._job_queue.launch(argv, name=out_path.name or "correct",
                                     total_frames=max(1, total),
                                     out_dir=str(out_path))
        if job is not None:
            self._log.append(
                f"[correct] Launched background job: {job.session} "
                f"({len(paths)} file(s) \u00d7 {len(ops)} method(s))")

    def _on_job_done(self, job) -> None:
        """A detached correction finished \u2014 report what it wrote.

        The job has no Qt signals reaching this process, so its own
        ``_bg_job_correction.json`` is the only thing to read back."""
        import json
        try:
            data = json.loads(
                (Path(job.out_dir) / "_bg_job_correction.json").read_text())
        except Exception:
            self._log.append(f"[correct] Job {job.session} finished.")
            return
        self._log.append(
            f"[correct] Job {job.session} finished \u2014 "
            f"{data.get('n', 0)} file(s) written.")

    def _emit(self, msg: str, *, screen: bool = True, panel: bool = True) -> None:
        """One line to the on-screen panel and to this run's screen log.

        The two are deliberately separable: per-chunk progress belongs in the
        file (that is what makes a screenlog worth reading afterwards) but
        would drown the panel, which already shows the same thing in the
        status bar and progress bar.
        """
        if panel:
            self._log.append(msg)
        if screen and self._screen_log is not None:
            self._screen_log.write(msg)

    def _open_screen_log(self, paths, ops, chunk, out_dir) -> None:
        """Start this run's on-disk log, beside the caking logs the dark
        ladder was derived from. Best-effort: a log that cannot be opened
        costs one line in the panel, never the run."""
        from midas_gui import settings
        profile = settings.active_profile()
        expid = self._expid_provider().strip() if self._expid_provider else ""
        stem = Path(paths[0]).stem if len(paths) == 1 else ""
        self._screen_log = run_log.open_log(
            "correct", profile=profile, expid=expid, stem=stem)
        if self._screen_log.path is None:
            self._log.append(
                "Note: no run log written — "
                + (self._screen_log.error or "could not open the log file"))
            return
        if not expid:
            self._log.append(
                f"Note: Exp ID is blank, so this run's log is filed under "
                f"{run_log.NO_EXPID}/.")
        self._screen_log.header("MIDAS GUI — Batch Correction", {
            "Beamline": profile,
            "Experiment": expid or "(blank)",
            "Started": _dt.datetime.now().isoformat(timespec="seconds"),
            "Files": len(paths),
            "Methods": ", ".join(ops),
            "Chunk": chunk or "whole file",
            "Output": out_dir,
            "Out dtype": self._dtype_combo.currentData() or "float32",
            "Compression": self._comp_combo.currentData() or "none",
        })
        self._log.append(f"Run log: {self._screen_log.path}")

    def _close_screen_log(self) -> None:
        if self._screen_log is not None:
            self._screen_log.close()
            self._screen_log = None

    def shutdown(self):
        """App close — MainWindow calls this on any tab that has it
        (app.py's close sweep). The generic sweep covers this tab's own
        QThreads; the job queue's poll timer and its tracked `screen`
        sessions are not QThreads, so they need this. Detached jobs are
        deliberately left running: outliving the GUI is the point of them."""
        self._job_queue.shutdown()

    def _cancel(self):
        if self._running and self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._status.setText("Cancelling after the current chunk…")

    def _on_progress(self, done, total, msg):
        self._prog.setMaximum(max(1, total)); self._prog.setValue(done)
        self._status.setText(msg)
        # Every progress line reaches the file; only the dark choice — the
        # one decision worth auditing at a glance — reaches the panel.
        self._emit(msg, panel="dark = " in msg)

    def _on_file_done(self, path):
        self._outputs.append(path)
        self._emit(f"  wrote {path}")

    def _on_finished(self, outputs):
        self._running = False
        self._prog.setVisible(False)
        self._emit(f"Done — {len(outputs)} file(s) written.")
        self._close_screen_log()
        self._status.setText(f"Done — {len(outputs)} file(s) written.")
        self._refresh_enabled()

    def _on_failed(self, err):
        self._running = False
        self._prog.setVisible(False)
        self._emit(err)
        self._close_screen_log()
        self._status.setText("Failed — see the Log tab.")
        self._refresh_enabled()
        show_error(self, "Batch Correction failed", err.strip().splitlines()[-1])

    # ── GUI state (Save/Load GUI State, project save/restore) ───────────
    def _state_widgets(self) -> dict:
        return {
            "out_ed": self._out_ed, "suffix_ed": self._suffix_ed,
            "ext_ed": self._ext_ed,
            "out_ds_ed": self._out_ds_ed, "comp_combo": self._comp_combo,
            "comp_level": self._comp_level, "shuffle_chk": self._shuffle_chk,
            "dtype_combo": self._dtype_combo,
            "ion_env_chk": self._ion_env_chk,
            "ion_motors_chk": self._ion_motors_chk,
            "auto_dark_chk": self._auto_dark_chk, "dark_ds_ed": self._dark_ds_ed,
            "clip_chk": self._clip_chk,
            "run_mode": self._run_mode, "n_workers": self._n_workers,
            **{f"op_{op}": chk for op, chk in self._op_chks.items()},
        }

    def get_state(self) -> dict:
        state = {"widgets": widgets_to_dict(self._state_widgets())}
        try:
            state["loader"] = self._loader.get_state()
        except Exception:
            pass
        return state

    def set_state(self, state: dict):
        if not isinstance(state, dict):
            return
        apply_dict_to_widgets(self._state_widgets(), state.get("widgets", {}))
        self._refresh_compression_row()
        self._on_ops_changed()
        # A QLineEdit restored (or constructed) with text longer than its
        # width shows its TAIL — "exchange/data" reading as "hange/data".
        # Park every caret at 0 so each field reads from the start.
        for w in (self._out_ed, self._suffix_ed, self._ext_ed, self._out_ds_ed,
                  self._dark_ds_ed):
            w.setCursorPosition(0)
        if state.get("loader"):
            try:
                self._loader.set_state(state["loader"])
            except Exception:
                pass
        self._refresh_enabled()
