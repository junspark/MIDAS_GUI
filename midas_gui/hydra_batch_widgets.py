"""Per-panel calibration-source widget for the Batch Integrate tab's Hydra
mode.

``HydraBatchPanelCard`` holds everything specific to integrating ONE GE
panel: which geometry to use (the fit that just completed on the Calibrate
tab's Hydra page, or a browsed geometry file), the read-only display of that
geometry, **its caking range**, and this panel's own run progress.

**Caking is per panel, not shared.** It used to be shared, by analogy with
R bin / eta bin and with no recorded reasoning — while the page's own
Corner/Edge presets computed Rmax "from whichever panel is currently
selected" and then wrote it into the one shared field, which only makes
sense if Rmax were per panel. It is: each GE module is a physically
separate detector with its own beam centre and its own corner distance,
the same side of the boundary Calibrate already draws
(DECISIONS 2026-08-24: shared wavelength/pixel/calibrant, independent
transforms and seed). mpe_wf_saxs_waxs settles it too — it keeps one
``cake_parameters.<beamline>.<detector>.csv`` per detector, and for Hydra
each panel IS a detector.

The six fields here are exactly the caking half of that CSV: R_MIN, R_MAX,
R_STEP, ETA_MIN, ETA_MAX, ETA_STEP. (eta min/max had no widget on this page
at all before.) What stays shared on ``HydraBatchPage`` is the processing
recipe rather than the geometry — kernel, azimuthal-mean weighting,
per-bin variance, Q-uniform bins — none of which mpe_wf models per
detector either.
"""
from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from midas_gui.helpers import (
    _browse, _build_spec, spec_from_geometry_file, resolve_calibration_fields,
    full_calibration_snapshot, make_calib_values_button, _fspin,
    rmax_corner_px, rmax_edge_px)
from midas_gui.hydra_widgets import panel_color
from midas_gui import style as S


class HydraBatchPanelCard(QtWidgets.QWidget):
    """One GE panel's calibration source + values + caking + run progress."""

    #: Any of this panel's six caking values changed. The page redraws that
    #: panel's overlay from it, so the picture tracks the numbers.
    cakingChanged = QtCore.pyqtSignal(int)
    #: Load / Save this panel's mpe_wf cake_parameters CSV. The PAGE does the
    #: file dialog and the path arithmetic -- the beamline token and the
    #: analysis root come from the loader and the Exp ID header, neither of
    #: which a card can see.
    cakeLoadRequested = QtCore.pyqtSignal(int)
    cakeSaveRequested = QtCore.pyqtSignal(int)

    def __init__(self, panel_number: int, parent=None):
        super().__init__(parent)
        self.panel_number = panel_number
        self.result = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        lv = QtWidgets.QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)

        # ── Calibration source ──
        cal = S.make_card(f"ge{self.panel_number} — Calibration source")
        src_row = QtWidgets.QHBoxLayout(); src_row.setSpacing(10)
        self._use_calib_btn = QtWidgets.QRadioButton("From Calibrate tab")
        self._use_file_btn = QtWidgets.QRadioButton("From file")
        self._use_calib_btn.setChecked(True)
        src_row.addWidget(self._use_calib_btn); src_row.addWidget(self._use_file_btn)
        src_row.addStretch(1)
        cal.body.addLayout(src_row)
        self._calib_src_lbl = QtWidgets.QLabel(
            f"(run Calibrate tab's Hydra fit for ge{self.panel_number} first)")
        self._calib_src_lbl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._calib_src_lbl.setWordWrap(True)
        cal.body.addWidget(self._calib_src_lbl)
        self._json_ed = QtWidgets.QLineEdit()
        self._json_ed.setPlaceholderText("calibration.json / paramstest.txt / .poni…")
        jr = QtWidgets.QHBoxLayout(); jr.setSpacing(4); jr.addWidget(self._json_ed, 1)
        bj = QtWidgets.QPushButton("…"); bj.setFixedWidth(30)
        bj.clicked.connect(lambda: self._json_ed.setText(
            _browse(self, f"Open ge{self.panel_number} calibration file",
                    "Calibration (*.json *.txt *.poni);;All (*)") or ""))
        jr.addWidget(bj)
        self._json_ed.textChanged.connect(
            lambda t: self._use_file_btn.setChecked(True) if t.strip() else None)
        cal.body.addLayout(jr)
        # "Calibration values" used to be an always-visible grid here — with
        # 4 of these panels stacked it cost even more space than the
        # single-detector tab's version; now a popup, opened on click,
        # showing the same fields — see helpers.make_calib_values_button.
        calib_view_btn = make_calib_values_button(self._calib_fields_in_use)
        cal.body.addWidget(calib_view_btn, 0, QtCore.Qt.AlignLeft)
        lv.addWidget(cal)

        # ── Caking (this panel only) ──
        n = self.panel_number
        colour = panel_color(n)
        cake = S.make_card(f"ge{n} — caking")
        # The colour is the same one this panel's radial curve and its
        # overlay use, so the card, the plot and the picture agree about
        # which panel is which without reading a label.
        cake.body.addWidget(self._colour_key_label(colour, n))
        self._r_min = _fspin(0.0, 1_000_000.0, 2, 0.0, "px")
        self._r_max = _fspin(0.0, 1_000_000.0, 2, 0.0, "px")
        self._r_bin = _fspin(0.1, 20.0, 2, 1.0, "px")
        self._eta_min = _fspin(-360.0, 360.0, 1, -180.0, "°")
        self._eta_max = _fspin(-360.0, 360.0, 1, 180.0, "°")
        self._eta_bin = _fspin(0.5, 360.0, 1, 5.0, "°")
        self._r_max.setToolTip(
            "0 = auto (farthest detector corner from THIS panel's beam "
            "centre).\nUse Corner/Edge, or type your own.")
        self._rmax_corner_btn = QtWidgets.QPushButton("Corner")
        self._rmax_corner_btn.setToolTip(
            f"Set Rmax to the farthest detector CORNER from ge{n}'s beam centre.")
        self._rmax_corner_btn.clicked.connect(
            lambda: self._apply_rmax_preset(rmax_corner_px))
        self._rmax_edge_btn = QtWidgets.QPushButton("Edge")
        self._rmax_edge_btn.setToolTip(
            f"Set Rmax to the farthest detector EDGE from ge{n}'s beam centre.")
        self._rmax_edge_btn.clicked.connect(
            lambda: self._apply_rmax_preset(rmax_edge_px))
        rmax_row = QtWidgets.QHBoxLayout(); rmax_row.setSpacing(4)
        rmax_row.addWidget(self._r_max)
        rmax_row.addWidget(self._rmax_corner_btn)
        rmax_row.addWidget(self._rmax_edge_btn)
        cf = S.Form()
        cf.row(("Rmin:", self._r_min))
        cf.row(("Rmax:", rmax_row))
        cf.row(("R bin:", self._r_bin))
        cf.row(("ηmin:", self._eta_min), ("ηmax:", self._eta_max))
        cf.row(("η bin:", self._eta_bin))
        cake.body.addLayout(cf)
        csv_row = QtWidgets.QHBoxLayout(); csv_row.setSpacing(4)
        load_btn = QtWidgets.QPushButton("Load cake CSV…")
        load_btn.setToolTip(
            f"Read an mpe_wf cake_parameters CSV into ge{n}'s six fields.\n\n"
            "The file's OME_SUM/OME_START/OME_STEP describe the rotation, "
            "not the caking, and this page has no rotation controls — they "
            "are reported rather than silently dropped.")
        load_btn.clicked.connect(lambda: self.cakeLoadRequested.emit(self.panel_number))
        save_btn = QtWidgets.QPushButton("Save cake CSV…")
        save_btn.setToolTip(
            f"Write ge{n}'s six fields as cake_parameters.<beamline>.ge{n}.csv "
            "— the exact name run_midas_for_cakes_gui.sh looks up under "
            "<expid>_bc/, so the workflow and this GUI read the same file.\n\n"
            "All nine columns are written (OME_* as 0): mpe_wf's reader turns "
            "a blank cell into a ValueError.")
        save_btn.clicked.connect(lambda: self.cakeSaveRequested.emit(self.panel_number))
        csv_row.addWidget(load_btn); csv_row.addWidget(save_btn); csv_row.addStretch(1)
        cake.body.addLayout(csv_row)
        for w in (self._r_min, self._r_max, self._r_bin,
                  self._eta_min, self._eta_max, self._eta_bin):
            w.valueChanged.connect(lambda *_: self.cakingChanged.emit(self.panel_number))
        lv.addWidget(cake)

        # ── Run progress (this panel only) ──
        prog_card = S.make_card(f"ge{self.panel_number} — progress")
        self._prog = QtWidgets.QProgressBar(); self._prog.setRange(0, 100)
        prog_card.body.addWidget(self._prog)
        self._status_lbl = QtWidgets.QLabel("Idle")
        self._status_lbl.setStyleSheet(f"font-size:10px;color:{S.MUTED}")
        prog_card.body.addWidget(self._status_lbl)
        lv.addWidget(prog_card)

        lv.addStretch(1)

    @staticmethod
    def _colour_key_label(colour: str, n: int) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(f"■ ge{n}")
        lbl.setStyleSheet(f"color:{colour};font-size:10px")
        lbl.setToolTip("This panel's colour in the detector-view overlay "
                       "and in the radial plot.")
        return lbl

    def _apply_rmax_preset(self, formula) -> None:
        """Corner/Edge, resolved from THIS panel's own calibration — which
        is what the shared field could never honour."""
        fields, _note = self._calib_fields_in_use()
        if not fields or fields.get("BC_y") is None or fields.get("NrPixelsY") is None:
            return
        self._r_max.setValue(formula(fields["BC_y"], fields["BC_z"],
                                     fields["NrPixelsY"], fields["NrPixelsZ"]))

    def caking(self) -> dict:
        """This panel's six caking values, in the CAKE_KEYS vocabulary so a
        cake_parameters CSV round-trips without a second translation."""
        return {"R_MIN": self._r_min.value(), "R_MAX": self._r_max.value(),
                "R_STEP": self._r_bin.value(),
                "ETA_MIN": self._eta_min.value(), "ETA_MAX": self._eta_max.value(),
                "ETA_STEP": self._eta_bin.value()}

    def set_caking(self, values: dict) -> None:
        """Apply whichever of the six a dict carries, leaving the rest. One
        signal at the end, not six -- each would redraw the overlay."""
        pairs = (("R_MIN", self._r_min), ("R_MAX", self._r_max),
                 ("R_STEP", self._r_bin), ("ETA_MIN", self._eta_min),
                 ("ETA_MAX", self._eta_max), ("ETA_STEP", self._eta_bin))
        touched = False
        for key, widget in pairs:
            v = (values or {}).get(key)
            if v is None:
                continue
            widget.blockSignals(True)
            widget.setValue(float(v))
            widget.blockSignals(False)
            touched = True
        if touched:
            self.cakingChanged.emit(self.panel_number)

    # ── Calibration source ──────────────────────────────────────────

    def set_calibration(self, result):
        """A Hydra panel fit finished on the Calibrate tab — adopt it as
        this panel's geometry (mirrors ``BatchTab.set_calibration``)."""
        self.result = result
        self._calib_src_lbl.setText(
            f"From Calibrate tab: Lsd={result.Lsd/1000:.3f} mm  "
            f"λ={result.wavelength_A:.5f} Å  {result.NrPixelsY}×{result.NrPixelsZ} px")
        self._use_calib_btn.setChecked(True)

    def _calib_fields_in_use(self):
        """Resolve this panel's active geometry as a dict of display fields —
        or ``(None, note)`` if unavailable. Also backs the "View calibration"
        popup (see helpers.make_calib_values_button), called fresh each time
        it's opened."""
        return resolve_calibration_fields(
            self.result, self._use_file_btn.isChecked(), self._json_ed.text(),
            source_label=f"Calibrate tab (ge{self.panel_number})")

    def full_calib_snapshot(self):
        """This panel's active geometry as the *whole* calibration rather than
        the display subset — what gets recorded as an integration attempt's
        ``calibration_snapshot`` (see
        ``helpers.full_calibration_snapshot``)."""
        return full_calibration_snapshot(
            self.result, self._use_file_btn.isChecked(), self._json_ed.text(),
            source_label=f"Calibrate tab (ge{self.panel_number})")

    def resolved_im_trans(self) -> tuple:
        """ImTransOpt codes from this panel's active calibration source — see
        ``BatchTab._resolved_im_trans`` (single-detector counterpart)."""
        fields, _ = self._calib_fields_in_use()
        return tuple(fields.get("im_trans") or []) if fields else ()

    def using_file(self) -> bool:
        return self._use_file_btn.isChecked()

    def file_path(self) -> str:
        return self._json_ed.text().strip()

    def resolved_spec(self):
        """Build an ``IntegrationSpec`` from whichever calibration source is
        active, caked by THIS panel's own six values — raises if neither a
        Calibrate-tab result nor a valid file is set.

        No longer takes r_bin/e_bin/r_min/r_max from the caller: they were
        the page's shared fields, and caking is per panel now (see the
        module docstring). eta_min/eta_max reach a spec from this page for
        the first time — both builders have always accepted them, the page
        just never had the widgets to pass.

        Rmax 0.0 is the "auto" sentinel the spinbox documents, passed as
        None so ``spec_from_calibration_result`` picks the farthest corner
        itself rather than integrating out to a literal zero radius.
        """
        c = self.caking()
        r_max = c["R_MAX"] or None            # 0.0 = auto, not a radius
        kw = dict(r_min=c["R_MIN"], r_max=r_max,
                  eta_min=c["ETA_MIN"], eta_max=c["ETA_MAX"])
        if not self._use_file_btn.isChecked():
            if self.result is None:
                raise RuntimeError(
                    f"ge{self.panel_number}: no calibration from the Calibrate tab. "
                    "Run its Hydra fit first.")
            return _build_spec(self.result, c["R_STEP"], c["ETA_STEP"], **kw)
        path = self._json_ed.text().strip()
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"ge{self.panel_number}: calibration file not found: {path}")
        return spec_from_geometry_file(path, c["R_STEP"], c["ETA_STEP"], **kw)

    # ── Progress ─────────────────────────────────────────────────────

    def set_progress(self, done: int, total: int):
        self._prog.setValue(int(100 * done / total) if total else 0)
        self._status_lbl.setText(f"Integrated {done} / {total} frames")

    def set_status(self, text: str):
        self._status_lbl.setText(text)

    def reset_progress(self):
        self._prog.setValue(0)
        self._status_lbl.setText("Idle")

    # ── GUI state ────────────────────────────────────────────────────

    def state_widgets(self) -> dict:
        # The six caking widgets ride the page's existing per-card state
        # round-trip (widgets_to_dict / apply_dict_to_widgets), so a project
        # keeps each panel's caking without any new plumbing.
        return {
            "use_calib_btn": self._use_calib_btn, "use_file_btn": self._use_file_btn,
            "json_ed": self._json_ed,
            "r_min": self._r_min, "r_max": self._r_max, "r_bin": self._r_bin,
            "eta_min": self._eta_min, "eta_max": self._eta_max,
            "eta_bin": self._eta_bin,
        }
