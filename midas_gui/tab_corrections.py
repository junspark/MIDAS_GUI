"""Tab 5 — Corrections & Physics.

Preview the effect of each correction on a single frame before a batch run.
Pixel-domain: polarization, solid angle, empty subtraction.
Profile-domain: cylindrical absorption, Compton subtraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from midas_gui.helpers import (_load_image, _fspin, _twocol, _browse, is_h5,
                               widgets_to_dict, apply_dict_to_widgets,
                               browse_start_dir, warn_if_path_missing)
from midas_gui.constants import (DEFAULT_NICKEL_FRAME0,
                                 POL_PLANE_HORIZONTAL_ETA_DEG)
from midas_gui.widgets import LogPanel
from midas_gui.dialogs import show_error
from midas_gui.workers import CorrectionPreviewWorker, LearnableGainWorker
from midas_gui import style as S


class CorrectionsTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._image: Optional[np.ndarray] = None
        self._dark: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._worker = None
        self._gain_ref_image: Optional[np.ndarray] = None
        self._gain_drift_image: Optional[np.ndarray] = None
        self._gain_map: Optional[np.ndarray] = None
        self._gain_worker = None
        self._build_ui()
        if Path(self._img_ed.text().strip() or "x").exists():
            self._load_img()

    def set_calibration(self, result):
        self._result = result
        self._src_lbl.setText(f"From Tab 2: Lsd={result.Lsd/1000:.3f} mm  λ={result.wavelength_A:.5f} Å")
        self._run_btn.setEnabled(self._image is not None)
        self._gain_train_btn.setEnabled(
            self._gain_ref_image is not None and self._gain_drift_image is not None)

    def set_mask_from_tab1(self, mask):
        self._mask = mask

    # ── GUI state (Save/Load GUI State) ─────────────────────────────
    def _state_widgets(self) -> dict:
        return {
            "img_ed": self._img_ed,
            "img_h5_ed": self._img_h5_ed,
            "rbin": self._rbin, "ebin": self._ebin,
            "pol_chk": self._pol_chk,
            "pol_frac": self._pol_frac, "pol_plane": self._pol_plane,
            "sa_chk": self._sa_chk,
            "empty_chk": self._empty_chk,
            "empty_ed": self._empty_ed,
            "empty_scale": self._empty_scale,
            "abs_chk": self._abs_chk,
            "abs_mu": self._abs_mu,
            "comp_chk": self._comp_chk,
            "comp_comp": self._comp_comp,
            "comp_scale": self._comp_scale,
            "gain_ref_ed": self._gain_ref_ed,
            "gain_drift_ed": self._gain_drift_ed,
            "gain_nsteps": self._gain_nsteps,
            "gain_lr": self._gain_lr,
            "gain_unity_w": self._gain_unity_w,
            "gain_smooth_w": self._gain_smooth_w,
            "out_ed": self._out_ed,
        }

    def get_state(self) -> dict:
        return {"fields": widgets_to_dict(self._state_widgets())}

    def set_state(self, state: dict):
        fields = state.get("fields", {})
        apply_dict_to_widgets(self._state_widgets(), fields)
        img_path = fields.get("img_ed")
        if img_path and Path(img_path).exists():
            self._load_img()
        ref_path = fields.get("gain_ref_ed")
        if ref_path and Path(ref_path).exists():
            self._load_gain_ref()
        drift_path = fields.get("gain_drift_ed")
        if drift_path and Path(drift_path).exists():
            self._load_gain_drift()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(8)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setFixedWidth(559)
        inner = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(inner); lv.setSpacing(6)
        scroll.setWidget(inner)

        def _br(w=28):
            b = QtWidgets.QPushButton("…"); b.setFixedWidth(w); return b

        def _frow(ed, slot):
            r = QtWidgets.QHBoxLayout(); r.setSpacing(3)
            r.addWidget(ed); b = _br(); b.clicked.connect(slot); r.addWidget(b); return r

        grp_src = QtWidgets.QGroupBox("Calibration source")
        sv = QtWidgets.QVBoxLayout(grp_src)
        self._src_lbl = QtWidgets.QLabel("(run Tab 2 calibration first)")
        self._src_lbl.setStyleSheet("color:#aaa;font-size:10px"); self._src_lbl.setWordWrap(True)
        sv.addWidget(self._src_lbl)
        lv.addWidget(grp_src)

        grp_img = QtWidgets.QGroupBox("Frame to preview")
        gf = QtWidgets.QFormLayout(grp_img); gf.setSpacing(4)
        self._img_ed = QtWidgets.QLineEdit(DEFAULT_NICKEL_FRAME0); self._img_ed.setPlaceholderText("Sample frame…")
        gf.addRow("Image:", _frow(self._img_ed, self._browse_img))
        self._img_h5_lbl = QtWidgets.QLabel("  Dataset:"); self._img_h5_ed = QtWidgets.QLineEdit("exchange/data")
        self._img_h5_lbl.setVisible(False); self._img_h5_ed.setVisible(False)
        gf.addRow(self._img_h5_lbl, self._img_h5_ed)
        self._img_ed.textChanged.connect(lambda p: (
            self._img_h5_lbl.setVisible(is_h5(p)), self._img_h5_ed.setVisible(is_h5(p))))
        self._img_ed.returnPressed.connect(self._load_img)
        self._img_h5_ed.editingFinished.connect(
            lambda: self._image is not None and self._load_img())
        self._rbin = _fspin(0.1, 20.0, 2, 1.0, "px"); self._ebin = _fspin(0.5, 360.0, 1, 5.0, "°")
        gf.addRow("Bins:", _twocol("R:", self._rbin, "η:", self._ebin))
        lv.addWidget(grp_img)

        # Pixel-domain corrections
        grp_px = QtWidgets.QGroupBox("Pixel-domain corrections")
        pf = QtWidgets.QFormLayout(grp_px); pf.setSpacing(4)
        self._pol_chk = QtWidgets.QCheckBox("Polarization")
        self._pol_frac = _fspin(0.0, 1.0, 3, 0.99); self._pol_plane = _fspin(-180, 180, 1,
                                                  POL_PLANE_HORIZONTAL_ETA_DEG, "°")
        pf.addRow(self._pol_chk)
        pf.addRow(_twocol("frac:", self._pol_frac, "plane:", self._pol_plane))
        self._sa_chk = QtWidgets.QCheckBox("Solid angle (tilt-aware)")
        pf.addRow(self._sa_chk)
        self._empty_chk = QtWidgets.QCheckBox("Empty subtraction")
        pf.addRow(self._empty_chk)
        self._empty_ed = QtWidgets.QLineEdit(); self._empty_ed.setPlaceholderText("empty-cell frame…")
        warn_if_path_missing(self._empty_ed, self)
        pf.addRow("file:", _frow(self._empty_ed, self._browse_empty))
        self._empty_scale = _fspin(0.0, 1e9, 3, 1.0)
        pf.addRow("scale:", self._empty_scale)
        lv.addWidget(grp_px)

        # Profile-domain corrections
        grp_pr = QtWidgets.QGroupBox("Profile-domain corrections")
        prf = QtWidgets.QFormLayout(grp_pr); prf.setSpacing(4)
        self._abs_chk = QtWidgets.QCheckBox("Cylindrical absorption")
        self._abs_mu = _fspin(0.0, 1e9, 3, 0.5)
        prf.addRow(self._abs_chk, self._abs_mu)
        prf.addRow(QtWidgets.QLabel("μR (absorption × capillary radius)"))
        self._comp_chk = QtWidgets.QCheckBox("Compton subtraction")
        prf.addRow(self._comp_chk)
        self._comp_comp = QtWidgets.QLineEdit("Ce:1,O:2")
        self._comp_comp.setToolTip("Composition as element:fraction pairs, e.g. Ce:1,O:2")
        prf.addRow("composition:", self._comp_comp)
        self._comp_scale = _fspin(0.0, 1e9, 3, 1.0)
        prf.addRow("scale:", self._comp_scale)
        lv.addWidget(grp_pr)

        self._run_btn = S.primary_btn("Compute corrected profile"); self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        lv.addWidget(self._run_btn)

        # ── Per-pixel gain training ──
        grp_gain = QtWidgets.QGroupBox("Per-pixel gain training (LearnableGain)")
        gv = QtWidgets.QVBoxLayout(grp_gain); gv.setSpacing(4)
        gf2 = QtWidgets.QFormLayout(); gf2.setSpacing(4)
        self._gain_ref_ed = QtWidgets.QLineEdit(); self._gain_ref_ed.setPlaceholderText("Reference (clean) frame…")
        warn_if_path_missing(self._gain_ref_ed, self)
        gf2.addRow("Ref frame:", _frow(self._gain_ref_ed, self._browse_gain_ref))
        self._gain_drift_ed = QtWidgets.QLineEdit(); self._gain_drift_ed.setPlaceholderText("Drifted frame…")
        warn_if_path_missing(self._gain_drift_ed, self)
        gf2.addRow("Drift frame:", _frow(self._gain_drift_ed, self._browse_gain_drift))
        self._gain_nsteps = _fspin(1, 1_000_000, 0, 100); self._gain_nsteps.setSingleStep(10)
        self._gain_lr = _fspin(0.0, 1e6, 4, 0.02)
        gf2.addRow(_twocol("Steps:", self._gain_nsteps, "lr:", self._gain_lr))
        self._gain_unity_w = _fspin(0.0, 1e6, 5, 1e-4); self._gain_smooth_w = _fspin(0.0, 1e6, 5, 1e-3)
        gf2.addRow(_twocol("unity_w:", self._gain_unity_w, "smooth_w:", self._gain_smooth_w))
        gv.addLayout(gf2)
        # Output folder. Without one the Save dialog below opens on a bare
        # filename, i.e. wherever the app happened to be launched from —
        # the same problem the Calibrate tab's save dialogs already fixed.
        self._out_ed = QtWidgets.QLineEdit()
        self._out_ed.setPlaceholderText("Output folder…")
        self._out_ed.setToolTip("Folder the gain map's Save dialog opens on.")
        warn_if_path_missing(self._out_ed, self, is_output_dir=True)
        bou = QtWidgets.QPushButton("…"); bou.setFixedWidth(30)
        bou.clicked.connect(lambda: self._out_ed.setText(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Output folder", browse_start_dir(self._out_ed.text())) or ""))
        outr = QtWidgets.QHBoxLayout(); outr.setSpacing(4)
        outr.addWidget(QtWidgets.QLabel("Output:"))
        outr.addWidget(self._out_ed, 1); outr.addWidget(bou)
        gv.addLayout(outr)
        self._gain_train_btn = QtWidgets.QPushButton("Train Gain")
        self._gain_train_btn.setEnabled(False)
        self._gain_train_btn.clicked.connect(self._train_gain)
        gv.addWidget(self._gain_train_btn)
        self._gain_prog = QtWidgets.QProgressBar(); self._gain_prog.setRange(0, 100)
        self._gain_prog.setVisible(False); gv.addWidget(self._gain_prog)
        self._gain_stats_lbl = QtWidgets.QLabel("—")
        self._gain_stats_lbl.setStyleSheet("color:#aaa;font-size:10px"); gv.addWidget(self._gain_stats_lbl)
        self._gain_save_btn = QtWidgets.QPushButton("Save gain map (NPZ)…")
        self._gain_save_btn.setEnabled(False); self._gain_save_btn.clicked.connect(self._save_gain_map)
        gv.addWidget(self._gain_save_btn)
        lv.addWidget(grp_gain)

        lv.addStretch(1)
        root.addWidget(scroll)

        # Right: two plots + log
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plots = QtWidgets.QWidget(); pv = QtWidgets.QVBoxLayout(plots); pv.setContentsMargins(0, 0, 0, 0)
        self._top = pg.PlotWidget(background="k")
        self._top.setLabel("left", "intensity"); self._top.setLabel("bottom", "2θ (°)")
        self._top.showGrid(x=True, y=True, alpha=0.2); self._top.addLegend()
        self._c_unc = self._top.plot([], [], pen=pg.mkPen("#888", width=2), name="uncorrected")
        self._c_cor = self._top.plot([], [], pen=pg.mkPen("#88ccff", width=2), name="corrected")
        pv.addWidget(self._top, stretch=2)
        self._bot = pg.PlotWidget(background="k")
        self._bot.setLabel("left", "correction factor"); self._bot.setLabel("bottom", "2θ (°)")
        self._bot.showGrid(x=True, y=True, alpha=0.2)
        self._c_fac = self._bot.plot([], [], pen=pg.mkPen("#f0a030", width=2))
        pv.addWidget(self._bot, stretch=1)
        right.addWidget(plots)
        self._log = LogPanel()
        right.addWidget(self._log)
        right.setStretchFactor(0, 4); right.setStretchFactor(1, 1)
        root.addWidget(right, stretch=1)

    def _browse_gain_ref(self):
        p = _browse(self, "Open reference (clean) frame", "Images (*.tif *.tiff *.h5 *.hdf5);;All (*)",
                    start_dir=browse_start_dir(self._gain_ref_ed.text()))
        if p: self._gain_ref_ed.setText(p); self._load_gain_ref()

    def _browse_gain_drift(self):
        p = _browse(self, "Open drifted frame", "Images (*.tif *.tiff *.h5 *.hdf5);;All (*)",
                    start_dir=browse_start_dir(self._gain_drift_ed.text()))
        if p: self._gain_drift_ed.setText(p); self._load_gain_drift()

    def _load_gain_ref(self):
        path = self._gain_ref_ed.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Reference frame not found."); return
        try:
            self._gain_ref_image = _load_image(path)
            self._log.append(f"[gain] ref loaded: {Path(path).name} {self._gain_ref_image.shape}")
            self._gain_train_btn.setEnabled(
                self._gain_ref_image is not None and self._gain_drift_image is not None
                and self._result is not None)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def _load_gain_drift(self):
        path = self._gain_drift_ed.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Drifted frame not found."); return
        try:
            self._gain_drift_image = _load_image(path)
            self._log.append(f"[gain] drift loaded: {Path(path).name} {self._gain_drift_image.shape}")
            self._gain_train_btn.setEnabled(
                self._gain_ref_image is not None and self._gain_drift_image is not None
                and self._result is not None)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def _train_gain(self):
        if self._result is None or self._gain_ref_image is None or self._gain_drift_image is None:
            QtWidgets.QMessageBox.warning(self, "Missing",
                                           "Need calibration + reference frame + drifted frame."); return
        if self._gain_worker and self._gain_worker.isRunning():
            return
        cfg = {
            "n_steps": int(self._gain_nsteps.value()),
            "lr": self._gain_lr.value(),
            "unity_weight": self._gain_unity_w.value(),
            "smoothness_weight": self._gain_smooth_w.value(),
            "gain_scale": 0.1,
            "drift_threshold": 0.01,
            "r_bin": self._rbin.value(),
            "eta_bin": self._ebin.value(),
        }
        self._gain_train_btn.setEnabled(False)
        self._gain_prog.setVisible(True); self._gain_prog.setValue(0)
        self._gain_stats_lbl.setText("Training…")
        self._log.append("─" * 40 + "\nTraining LearnableGain…")
        self._gain_worker = LearnableGainWorker(
            self._result, self._gain_ref_image, self._gain_drift_image,
            self._mask, cfg, parent=self)
        self._gain_worker.progress.connect(self._on_gain_progress)
        self._gain_worker.log_line.connect(self._log.append)
        self._gain_worker.finished.connect(self._on_gain_done)
        self._gain_worker.failed.connect(self._on_gain_fail)
        self._gain_worker.start()

    def _on_gain_progress(self, step, total, loss):
        self._gain_prog.setValue(int(100 * step / total) if total else 0)

    def _on_gain_done(self, d):
        self._gain_train_btn.setEnabled(True)
        self._gain_prog.setVisible(False)
        self._gain_map = d["gain_map"]
        self._gain_save_btn.setEnabled(True)
        stats = (f"min={d['gain_min']:.4f}  max={d['gain_max']:.4f}  "
                 f"mean={d['gain_mean']:.4f}  drifted: {d['n_drifted']:,} px")
        self._gain_stats_lbl.setText(stats)
        self._log.append(f"[gain] {stats}")

    def _on_gain_fail(self, msg):
        self._gain_train_btn.setEnabled(True)
        self._gain_prog.setVisible(False)
        self._gain_stats_lbl.setText("Training failed")
        show_error(self, "Gain training failed", msg, log=self._log, log_prefix="\n[gain] ERROR:\n")

    def _save_gain_map(self):
        if self._gain_map is None: return
        from pathlib import Path
        out = self._out_ed.text().strip()
        start = str(Path(browse_start_dir(out) or ".") / "gain_map.npz")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save gain map", start, "NumPy (*.npz)")
        if not path: return
        np.savez_compressed(path, gain_map=self._gain_map)
        self._log.append(f"[gain] saved: {path}")
        QtWidgets.QMessageBox.information(self, "Saved", f"Gain map saved:\n{path}")

    def _browse_img(self):
        p = _browse(self, "Open frame", "Images (*.tif *.tiff *.h5 *.hdf5 *.ge*);;All (*)",
                    start_dir=browse_start_dir(self._img_ed.text()))
        if p: self._img_ed.setText(p); self._load_img()

    def _browse_empty(self):
        p = _browse(self, "Open empty-cell frame", "Images (*.tif *.tiff *.h5 *.hdf5);;All (*)",
                    start_dir=browse_start_dir(self._empty_ed.text()))
        if p: self._empty_ed.setText(p)

    def _load_img(self):
        path = self._img_ed.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Image not found."); return
        try:
            self._image = _load_image(path, data_loc=self._img_h5_ed.text().strip() or "exchange/data")
            self._log.append(f"Image loaded: {Path(path).name} {self._image.shape}")
            self._run_btn.setEnabled(self._result is not None)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def _run(self):
        if self._result is None or self._image is None:
            QtWidgets.QMessageBox.warning(self, "Missing", "Need calibration + image."); return
        if self._worker and self._worker.isRunning():
            return
        cfg = {"r_bin": self._rbin.value(), "eta_bin": self._ebin.value()}
        if self._pol_chk.isChecked():
            cfg["polarization"] = {"frac": self._pol_frac.value(), "plane": self._pol_plane.value()}
        if self._sa_chk.isChecked():
            cfg["solid_angle"] = True
        if self._empty_chk.isChecked() and self._empty_ed.text().strip():
            cfg["empty"] = {"path": self._empty_ed.text().strip(), "scale": self._empty_scale.value()}
        if self._abs_chk.isChecked():
            cfg["absorption"] = {"mu_R": self._abs_mu.value()}
        if self._comp_chk.isChecked():
            cfg["compton"] = {"composition": self._comp_comp.text().strip(),
                              "scale": self._comp_scale.value()}
        self._run_btn.setEnabled(False)
        self._log.append("─" * 40 + "\nComputing corrected profile…")
        self._worker = CorrectionPreviewWorker(
            self._result, self._image, self._dark, self._mask, cfg, parent=self)
        self._worker.log_line.connect(self._log.append)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, d):
        self._run_btn.setEnabled(True)
        tt = d["two_theta"]
        self._c_unc.setData(tt, d["profile_unc"])
        self._c_cor.setData(tt, d["profile_corr"])
        fac = d["factor"]; finite = np.isfinite(fac)
        self._c_fac.setData(tt[finite], fac[finite])
        self._log.append("Done — corrected profile ready.")

    def _on_fail(self, msg):
        self._run_btn.setEnabled(True)
        show_error(self, "Correction failed", msg, log=self._log, log_prefix="\nERROR:\n")
