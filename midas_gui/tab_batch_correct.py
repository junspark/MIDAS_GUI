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

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui import frame_correct as FC
from midas_gui import style as S
from midas_gui.dialogs import show_error
from midas_gui.helpers import (_NoScrollComboBox, _NoScrollSpinBox,
                               apply_dict_to_widgets, browse_start_dir,
                               check_output_dir_writable, widgets_to_dict,
                               warn_if_path_missing)
from midas_gui.widgets import DataLoaderPanel, ImageViewer, LogPanel
from midas_gui.workers import BatchCorrectionWorker


class BatchCorrectionTab(QtWidgets.QWidget):
    """Chunked frame reduction with field correction, HDF5 in → HDF5 out."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[BatchCorrectionWorker] = None
        self._outputs: list = []
        self._im_trans: list = []
        self._build_ui()

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
        self._loader.dataChanged.connect(self._on_data_changed)
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
        right.addTab(self._log, "Log")
        split.addWidget(right)
        split.setStretchFactor(2, 1)
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
        card.body.addLayout(row)

        form = S.Form()
        self._suffix_ed = QtWidgets.QLineEdit("_corr")
        self._suffix_ed.setToolTip(
            "Appended to each source file's name stem.\n"
            "run_009243.vrx.h5  →  run_009243_corr.h5")
        self._out_ds_ed = QtWidgets.QLineEdit("exchange/data")
        self._out_ds_ed.setToolTip(
            "HDF5 path the reduced (M, H, W) stack is written to.\n"
            "Leaving this at exchange/data means the output can be loaded\n"
            "straight back into any tab of this GUI.")
        form.row(("Suffix:", self._suffix_ed))
        form.row(("Dataset:", self._out_ds_ed))

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
            "together and typically buys more than raising the gzip level.")
        form.row(("Compression:", self._comp_combo),
                 ("level:", self._comp_level))
        form.full(self._shuffle_chk)
        card.body.addLayout(form)
        self._comp_combo.currentIndexChanged.connect(self._refresh_compression_row)
        self._refresh_compression_row()
        return card

    def _build_options_card(self):
        card = S.make_card("Correction")
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

        note = QtWidgets.QLabel(
            "Each sub-frame is corrected before the frames are combined, so a "
            "Sum of N frames loses N darks rather than one. Set N and the "
            "method in “Combine sub-frames” on the left.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        card.body.addWidget(note)
        return card

    def _build_run_card(self):
        card = S.make_card("Run")
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
        self._prog = QtWidgets.QProgressBar()
        self._prog.setRange(0, 100); self._prog.setVisible(False)
        card.body.addWidget(self._prog)
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        card.body.addWidget(self._status)
        return card

    def _refresh_compression_row(self):
        gzip = self._comp_combo.currentData() == "gzip"
        self._comp_level.setEnabled(gzip)
        self._shuffle_chk.setEnabled(self._comp_combo.currentData() != "none")

    # ── source resolution ───────────────────────────────────────────────
    def _h5_paths(self) -> list:
        """The selected HDF5 files, or ``[]``.

        Only HDF5 sources are accepted. "Chunks restart at every file" has
        no meaning for TIFF/GE, where a file holds exactly one frame and
        there is no sub-frame stack to group — see :meth:`_run`, which says
        so rather than silently applying a different rule.
        """
        cfg = self._loader.source_cfg()
        if cfg.get("type") == "hdf5":
            return [cfg["path"]]
        if cfg.get("type") == "hdf5_stack_glob":
            return list(cfg.get("paths") or [])
        return []

    def _chunk_settings(self) -> tuple:
        cfg = self._loader.source_cfg()
        return (cfg.get("chunk_size"), cfg.get("combine_op", "mean"),
                cfg.get("frame_start"), cfg.get("frame_end"))

    def _on_data_changed(self):
        self._refresh_enabled()

    def _refresh_enabled(self):
        running = bool(self._worker and self._worker.isRunning())
        has = bool(self._h5_paths())
        self._run_btn.setEnabled(has and not running)
        self._preview_btn.setEnabled(has and not running)
        self._cancel_btn.setEnabled(running)

    def _worker_kwargs(self, paths) -> dict:
        chunk, op, fr_start, fr_end = self._chunk_settings()
        return dict(
            dataset=self._loader._dataset(), chunk_size=chunk, op=op,
            out_dataset=self._out_ds_ed.text().strip() or "exchange/data",
            suffix=self._suffix_ed.text().strip(),
            dark=self._loader.dark(), bright=self._loader.bright(),
            background=self._loader.background(),
            bright_mode=self._loader.bright_mode(),
            auto_dark=self._auto_dark_chk.isChecked(),
            dark_dataset=self._dark_ds_ed.text().strip() or "exchange/data_dark",
            clip_negatives=self._clip_chk.isChecked(),
            compression=self._comp_combo.currentData(),
            level=self._comp_level.value(),
            shuffle=self._shuffle_chk.isChecked(),
            raw_start=fr_start, raw_end=fr_end)

    # ── preview ─────────────────────────────────────────────────────────
    def _preview(self):
        paths = self._h5_paths()
        if not paths:
            return
        try:
            import h5py
            chunk, op, fr_start, fr_end = self._chunk_settings()
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
        if self._worker and self._worker.isRunning():
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

        chunk, op, _s, _e = self._chunk_settings()
        self._outputs = []
        self._log.append(
            f"Batch Correction — {len(paths)} file(s), op={op}, "
            f"chunk={chunk or 'whole file'}, out={out_dir}")
        self._prog.setVisible(True); self._prog.setValue(0)
        self._worker = BatchCorrectionWorker(
            paths, out_dir=out_dir, parent=self, **self._worker_kwargs(paths))
        self._worker.progress.connect(self._on_progress)
        self._worker.fileDone.connect(self._on_file_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()
        self._refresh_enabled()

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._status.setText("Cancelling after the current chunk…")

    def _on_progress(self, done, total, msg):
        self._prog.setMaximum(max(1, total)); self._prog.setValue(done)
        self._status.setText(msg)
        if "dark = " in msg:
            self._log.append(msg)

    def _on_file_done(self, path):
        self._outputs.append(path)
        self._log.append(f"  wrote {path}")

    def _on_finished(self, outputs):
        self._prog.setVisible(False)
        self._log.append(f"Done — {len(outputs)} file(s) written.")
        self._status.setText(f"Done — {len(outputs)} file(s) written.")
        self._refresh_enabled()

    def _on_failed(self, err):
        self._prog.setVisible(False)
        self._log.append(err)
        self._status.setText("Failed — see the Log tab.")
        self._refresh_enabled()
        show_error(self, "Batch Correction failed", err.strip().splitlines()[-1])

    # ── GUI state (Save/Load GUI State, project save/restore) ───────────
    def _state_widgets(self) -> dict:
        return {
            "out_ed": self._out_ed, "suffix_ed": self._suffix_ed,
            "out_ds_ed": self._out_ds_ed, "comp_combo": self._comp_combo,
            "comp_level": self._comp_level, "shuffle_chk": self._shuffle_chk,
            "auto_dark_chk": self._auto_dark_chk, "dark_ds_ed": self._dark_ds_ed,
            "clip_chk": self._clip_chk,
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
        if state.get("loader"):
            try:
                self._loader.set_state(state["loader"])
            except Exception:
                pass
        self._refresh_enabled()
