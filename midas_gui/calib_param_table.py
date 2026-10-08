"""One table for everything the fit is told about a parameter.

A calibration parameter carries three independent decisions and one
consequence:

* **seed** — do you supply a starting value, and what is it?
* **refine** — is the fit then free to move it?
* **window** — if it moves, how far?
* and, falling out of the first two, **what the fit will actually do**.

Those used to live in two places that could not be seen at once: the seed
value and its "include in seed" tick behind a non-modal "Manual seed…"
dialog, the refine tick and the ± window in a card two cards further down.
The split was pure packaging -- the dialog never owned its widgets, it
reparented the caller's -- but it made the pairing unreadable, and the
pairing is the part that is not obvious. Seeding something you are NOT
refining is how you pin a parameter to a measured value, which is exactly
what ``tx`` is for: a GE panel's installation azimuth is an input the
powder fit must not touch.

So the row is the unit here, not the panel. Each parameter owns a control
row -- seed tick, value, refine tick, ± window -- and a muted sub-row
underneath saying what the fit will do with it. The sub-row exists because
the left column is a ~260 px scroll area: height is free and width is not,
and the outcome strings ("refined, from this value") are long.

Row order is :data:`dialogs.PARAMETER_LIMIT_ROWS`, which is also the order
the refine card and ``_REFINE_BOXES`` use -- reading the same parameter at
a different height in two panels is what made them look like they
disagreed.

The widgets are supplied by the caller and merely placed here, exactly as
the dialog did. That keeps every ``state_widgets()`` key pointing at the
same object, so saved projects keep loading across this change.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

from PyQt5 import QtWidgets

from midas_gui.dialogs import PARAMETER_LIMIT_ROWS
from midas_gui.helpers import _fspin, _NoScrollComboBox
from midas_gui import style as S


#: (refined?, seeded?) -> what the fit does with the parameter. Phrased as an
#: outcome rather than echoing the ticks, because the pairing is the part that
#: is not obvious. Carried over verbatim from the dialog this replaces.
OUTCOME = {
    (True, True): "refined, from this value",
    (True, False): "refined, auto-seeded",
    (False, True): "held at this value",
    (False, False): "held at default",
}

#: limit-row slot -> the seed tick that governs it. ``BC_y``/``BC_z`` share
#: one tick (the backend takes the pair or neither -- see
#: ``calib._resolve_seed``), so the tick spans both rows and ``BC_z`` is
#: marked as continuing the span rather than owning one.
_SEED_TICK_FOR = {
    "Lsd": "Lsd", "BC_y": "BC", "BC_z": None,
    "tx": "tx", "ty": "ty", "tz": "tz",
    "wavelength_A": None, "distortion": "Distortion",
}

#: limit-row slot -> the seed value box, by the caller's key. Two rows have
#: no seed box at all: the wavelength is seeded from the Detector & Calibrant
#: card, and the 15 distortion coefficients live behind their own "..."
#: button. Their seed cells stay empty rather than growing a control that
#: would do nothing.
_SEED_VALUE_FOR = {
    "Lsd": "Lsd", "BC_y": "BC_y", "BC_z": "BC_z",
    "tx": "tx", "ty": "ty", "tz": "tz",
}


class CalibrationParameterTable(QtWidgets.QWidget):
    """The merged seed/refine/limits grid.

    Every widget passed in is *placed*, never copied: the caller keeps
    ownership and keeps its own references, so ``state_widgets()`` and the
    project round-trip are unaffected by the move.

    :param seed_ticks: key -> "include in seed" checkbox. Keys are the seed
        vocabulary (``BC`` for the pair), not the limit-row vocabulary.
    :param seed_values: ``BC_y``/``BC_z``/``Lsd``/``tx``/``ty``/``tz`` ->
        value spin box.
    :param refine_ticks: key -> "refine this" checkbox, same vocabulary as
        ``seed_ticks`` plus ``Wavelength``.
    :param dist_seed_btn: the "..." button holding the distortion seed
        coefficients, placed in the distortion row's value cell.
    :param on_limits_changed: called whenever any ± control changes.
    """

    def __init__(self, *, seed_ticks: Mapping, seed_values: Mapping,
                 refine_ticks: Mapping, dist_seed_btn=None,
                 dist_refine_row=None,
                 on_limits_changed: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self._on_limits_changed = on_limits_changed

        #: seed key -> the label saying what the fit will do with that row.
        self._status_lbls: dict = {}
        #: seed key -> its "include in seed" tick (what _sync_status reads).
        self._seed_boxes: dict = dict(seed_ticks)
        #: seed key -> its Refine tick. Filled by set_refine_boxes, which the
        #: caller can only call once both cards exist.
        self._refine_boxes: dict = {}

        # Exposed under the names the tab already uses for these, so the
        # existing limits machinery (_sync_limits_mode, _sync_seed_steps,
        # _update_limits_label) keeps working against the same dict objects.
        self.limit_widgets: dict = {}
        self.limit_name_lbls: dict = {}
        self.limit_row_sub: dict = {}
        self.limit_row_cells: dict = {}
        self.limit_row_index: dict = {}
        self.limit_cells: list = []
        #: slot -> the QLabel naming it in column 0
        self.row_name_lbls: dict = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        hint = QtWidgets.QLabel(
            "Seed = where the fit starts.  Refine = free to move.  "
            "Seeded but not refined pins a measured value.")
        hint.setWordWrap(True)
        hint.setToolTip(
            "Seeding and refining are separate choices. Ticking Seed only "
            "says where the fit STARTS; whether it is then free to move is "
            "Refine. Seeding something you are not refining is how you pin "
            "it to a measured value -- which on a Hydra panel is exactly "
            "what tx is for.")
        hint.setStyleSheet(f"color:{S.MUTED};font-size:10px;padding-bottom:2px;")
        outer.addWidget(hint)
        self._hint = hint

        #: The header says which column is which, so neither tick needs to
        #: repeat the parameter name and the row stays inside the column.
        self._limits_hdr = QtWidgets.QLabel("")
        self._limits_hdr.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._limits_hdr.setWordWrap(True)
        outer.addWidget(self._limits_hdr)

        #: The backend re-centres every window on each E-M iterate, so the ±
        #: values below bound one iteration, not the run. Ticking this makes
        #: them bound the run instead -- the owning tab divides by the E-M
        #: iteration count before handing them over. Needed wherever the
        #: geometry is mechanically constrained (Hydra panels share one frame,
        #: so a panel's Lsd cannot really wander 5 mm), and off by default
        #: because it tightens every per-iteration trust region.
        self._hard_cap = QtWidgets.QCheckBox(
            "± values are a hard cap on the whole run")
        self._hard_cap.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        outer.addWidget(self._hard_cap)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(1)
        self.grid = grid
        outer.addLayout(grid)

        for text, col in (("Seed", 1), ("Refine", 2), ("start value", 3),
                          ("\u00b1 range", 4), ("units", 5)):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"color:{S.MUTED};font-size:9px")
            grid.addWidget(lbl, 0, col)
        grid.setColumnStretch(3, 1)

        rows = {r[0]: (r[2], r[3], r[4], r[5]) for r in PARAMETER_LIMIT_ROWS}
        order = tuple(row[0] for row in PARAMETER_LIMIT_ROWS)

        #: the name column, so no tick has to repeat it
        _NAME = {"Lsd": "Lsd", "BC_y": "Beam centre", "BC_z": "BC_z",
                 "tx": "tx", "ty": "ty", "tz": "tz",
                 "wavelength_A": "Wavelength", "distortion": "Distortion"}

        gr = 1
        for name in order:
            unit0, win0, abs_unit, dec = rows[name]
            tick_key = _SEED_TICK_FOR.get(name)

            lbl0 = QtWidgets.QLabel(_NAME[name])
            if tick_key is None:
                lbl0.setStyleSheet(f"color:{S.MUTED};font-size:10px")
            grid.addWidget(lbl0, gr, 0)
            self.row_name_lbls[name] = lbl0

            # col 1 / col 2: the two decisions, as bare ticks under their
            # headers. Their captions are blanked because column 0 names the
            # row -- repeating it is what pushed "Beam centre" and
            # "Distortion" past the edge of the column.
            if tick_key is not None and tick_key in self._seed_boxes:
                tick = self._seed_boxes[tick_key]
                tick.setText("")
                grid.addWidget(tick, gr, 1)
            ref_key = "Wavelength" if name == "wavelength_A" else tick_key
            ref_box = refine_ticks.get(ref_key or "")
            if ref_box is not None:
                ref_box.setText("")
                grid.addWidget(ref_box, gr, 2)
            # Distortion is one row with two decisions, not two rows that
            # look like duplicates of each other: its refine SELECTION
            # (which coefficients may move) sits with the refine tick, and
            # its seed VALUES sit in the start-value cell. dist_refine_row
            # already carries the tick, so it replaces the plain one.
            if name == "distortion" and dist_refine_row is not None:
                grid.addWidget(dist_refine_row, gr, 2)

            # col 3: where the fit starts
            val_key = _SEED_VALUE_FOR.get(name)
            if val_key is not None and val_key in seed_values:
                seed_values[val_key].setMaximumWidth(100)
                grid.addWidget(seed_values[val_key], gr, 3)
            elif name == "distortion" and dist_seed_btn is not None:
                grid.addWidget(dist_seed_btn, gr, 3)

            # col 4 / col 5: the window and its unit. The opt-in box sits
            # with the window it switches on.
            cb = QtWidgets.QCheckBox("")
            lbl = QtWidgets.QLabel("")
            spin = _fspin(0.0, 1e6, dec, win0, "")
            spin.setMaximumWidth(64)
            combo = _NoScrollComboBox()
            combo.setMaximumWidth(48)
            combo.addItems([u for u in ("%", abs_unit) if u])
            combo.setCurrentText(unit0 or abs_unit)
            spin.setEnabled(False); combo.setEnabled(False)
            cb.toggled.connect(spin.setEnabled)
            cb.toggled.connect(combo.setEnabled)
            if on_limits_changed is not None:
                cb.toggled.connect(on_limits_changed)
                spin.valueChanged.connect(on_limits_changed)
                combo.currentTextChanged.connect(on_limits_changed)
            inner = QtWidgets.QWidget()
            crow = QtWidgets.QHBoxLayout(inner)
            crow.setContentsMargins(0, 0, 0, 0); crow.setSpacing(2)
            crow.addWidget(cb); crow.addWidget(lbl)
            crow.addWidget(QtWidgets.QLabel("\u00b1")); crow.addWidget(spin)
            grid.addWidget(inner, gr, 4)
            grid.addWidget(combo, gr, 5)

            self.limit_widgets[name] = (cb, spin, combo)
            self.limit_name_lbls[name] = lbl
            self.limit_row_sub[name] = ""
            self.limit_row_cells[name] = [inner, combo]
            self.limit_row_index[name] = gr
            self.limit_cells.extend([inner, combo])

            # what the fit will do with this row, spanning underneath
            if tick_key is not None and tick_key in self._seed_boxes:
                status = QtWidgets.QLabel("")
                status.setWordWrap(True)
                grid.addWidget(status, gr + 1, 0, 1, 6)
                self._status_lbls[tick_key] = status
                self._seed_boxes[tick_key].toggled.connect(self._sync_status)
                gr += 1
            gr += 1

        self._limits_note = QtWidgets.QLabel("")
        self._limits_note.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        self._limits_note.setWordWrap(True)
        grid.addWidget(self._limits_note, gr, 0, 1, 6)
        self.limit_cells.append(self._limits_note)
        self._sync_status()

    # -- grid detail the owning tab used to reach in and do itself --------

    def set_limit_row_visible(self, name: str, visible: bool):
        """Show or hide one parameter's ± controls, leaving its seed and
        refine controls alone -- the crystalline windows are coarser than
        the manual fit's, so some rows have no window of their own."""
        for w in self.limit_row_cells[name]:
            w.setVisible(visible)

    def covers_following_row(self, name: str, covers: bool):
        """Say that one parameter's ± window also bounds the next parameter.

        The crystalline tilt window is a single value for ty and tz
        (``tolTilts``). The old one-line-per-parameter grid said so by
        spanning the cell down a row. That cannot work here: a parameter
        owns three lines, so spanning reaches across the *next* parameter's
        value box and draws on top of it. The window is captioned instead,
        which survives the row being hidden and does not depend on
        neighbouring rows staying put.
        """
        lbl = self.limit_name_lbls[name]
        lbl.setText("ty + tz" if covers else self.limit_row_sub[name])
        lbl.setVisible(bool(lbl.text()))

    # ── what the fit will do with each row ──────────────────────────

    def set_refine_boxes(self, boxes: Mapping):
        """Attach the Refine ticks so each row can say what the fit will do.

        Deliberately one-way: the two tick columns are independent choices
        and all four pairings mean something different. They are reported
        together, never wired together.
        """
        self._refine_boxes = {k: v for k, v in boxes.items()
                              if k in self._status_lbls}
        for box in self._refine_boxes.values():
            box.toggled.connect(self._sync_status)
        self._sync_status()

    def sync_status(self):
        """Recompute every row's outcome.

        Needed because the labels are driven by ``toggled``, and a signal
        only fires when something *changes*: ``apply_dict_to_widgets``
        restores every tick with signals BLOCKED, so a reopened project
        showed whatever the labels happened to say before. The same trap
        ``_sync_seed_enabled`` already exists to work around.
        """
        self._sync_status()

    def _sync_status(self, *_args):
        for key, lbl in self._status_lbls.items():
            box = self._refine_boxes.get(key)
            if box is None:
                continue
            refined = box.isChecked()
            seeded = self._seed_boxes[key].isChecked()
            lbl.setText(OUTCOME[(refined, seeded)])
            lbl.setStyleSheet(
                "font-size:10px;color:" + ("#d7861f" if refined else "#8a8a8a"))
