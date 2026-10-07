"""The Calibrate tab's left column must tell the truth about the seed.

The seed summary must show the seed *values*, not just which parameters are
seeded, and the Limits note must re-centre when one of those values moves.

The seed spin boxes live inside ``ManualSeedDialog``, which is non-modal and
normally closed, so from the tab itself the starting Lsd / BC / tilts were
invisible — while the Refine card right below promised "± a window around its
seed value" and the crystalline footer "centred on the seed", neither of which
showed the value they meant. Reported from the beamline on an AgBH fit.

``forked`` per .context/DECISIONS.md — builds a CalibrationTab (pyqtgraph).
"""
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(app):
    from midas_gui.tab_calibrate import CalibrationTab
    return CalibrationTab()


def test_unseeded_summary_says_what_the_fit_will_do_instead(tab):
    for cb in tab._seed_enables:
        cb.setChecked(False)
    txt = tab._seed_summary_lbl.text()
    assert "Fully automatic" in txt
    # The point of the line: say where the numbers come from when they are
    # not the user's.
    assert "auto-seeded from the image" in txt and "0°" in txt


def test_seeded_parameters_show_their_values(tab):
    tab._seed_bcy.setValue(1024.5)
    tab._seed_bcz.setValue(1030.25)
    tab._seed_lsd.setValue(1382.4)
    tab._seed_ty.setValue(0.37)
    tab._seed_en_bc.setChecked(True)
    tab._seed_en_lsd.setChecked(True)
    tab._seed_en_ty.setChecked(True)

    txt = tab._seed_summary_lbl.text()
    assert "BC 1024.5, 1030.25 px" in txt      # full 3-dp precision kept
    assert "Lsd 1382.4 mm" in txt              # trailing zeros trimmed
    assert "ty 0.37°" in txt
    # Unseeded ones are named, not silently dropped.
    assert "auto: tx, tz, Distortion" in txt


def test_summary_follows_a_value_edited_after_the_tick(tab):
    """Pick BC, the arrow keys and result feedback all move the spin boxes
    without touching the enable ticks the summary used to hang off."""
    tab._seed_en_lsd.setChecked(True)
    tab._seed_lsd.setValue(500.0)
    assert "Lsd 500 mm" in tab._seed_summary_lbl.text()
    tab._seed_lsd.setValue(612.125)
    assert "Lsd 612.125 mm" in tab._seed_summary_lbl.text()


def test_distortion_seed_count_reaches_the_summary(tab):
    tab._seed_dist = {"iso_R2": 1e-6, "iso_R4": -2e-9}
    tab._seed_en_dist.setChecked(True)
    tab._update_seed_dist_label()
    assert "Distortion 2 coeff" in tab._seed_summary_lbl.text()


# ── The ± windows are centred on the seed, so the note moves with it ──

def test_limits_note_recentres_when_the_seed_moves(tab):
    """Reported from the beamline: the Data Viewer's "Send →" pushed
    Lsd 13900 mm into the seed, but the note went on printing the window
    around the previous 13868 — which reads as Send having been dropped.

    Only the ± widgets refreshed the note, so it sat at whichever seed was
    current when a ± box was last touched.
    """
    agbh = next(tab._cal.itemText(i) for i in range(tab._cal.count())
                if "AgBH" in tab._cal.itemText(i))
    tab._cal.setCurrentText(agbh)
    tab._seed_en_bc.setChecked(True)
    tab._seed_en_lsd.setChecked(True)
    for slot in ("Lsd", "BC_y"):
        tab._limit_widgets[slot][0].setChecked(True)
    tab._limit_widgets["Lsd"][1].setValue(50.0)        # ±50 %
    tab._seed_lsd.setValue(13868.0)
    tab._seed_bcy.setValue(3815.3)
    assert "Lsd ∈ [6934, 20802]" in tab._limits_note.text()

    # ...the "Send →", which touches no ± widget.
    tab._seed_lsd.setValue(13900.0)
    tab._seed_bcy.setValue(3816.0)
    assert "Lsd ∈ [6950, 20850]" in tab._limits_note.text()
    assert "BC_y ∈ [3766, 3866]" in tab._limits_note.text()


def test_crystalline_limits_note_recentres_too(tab):
    """Same wiring, the other note — "Always applied, centred on the seed"
    is just as wrong when it is centred on a stale one."""
    tab._cal.setCurrentText("CeO2")
    tab._limit_widgets["Lsd"][2].setCurrentText("%")
    tab._limit_widgets["Lsd"][1].setValue(50.0)
    tab._seed_lsd.setValue(1000.0)
    assert "Lsd ±500 mm" in tab._limits_note.text()
    tab._seed_lsd.setValue(2000.0)
    assert "Lsd ±1000 mm" in tab._limits_note.text()


# ── a ticked seed must be editable ───────────────────────────────────────────

def test_ticking_a_seed_enables_its_value_box(tab):
    assert not tab._seed_lsd.isEnabled()
    tab._seed_en_lsd.setChecked(True)
    assert tab._seed_lsd.isEnabled()
    tab._seed_en_lsd.setChecked(False)
    assert not tab._seed_lsd.isEnabled()


def test_bc_gates_both_coordinates(tab):
    tab._seed_en_bc.setChecked(True)
    assert tab._seed_bcy.isEnabled() and tab._seed_bcz.isEnabled()


def test_a_reopened_project_does_not_come_back_ticked_but_greyed_out(tab, app):
    """Reported from the beamline: every box in the Manual seed dialog was
    ticked and every value box greyed out, so the seed could be seen but not
    edited.

    The enabled state used to be wired as checkbox.toggled -> setEnabled, and
    a signal only fires on a CHANGE — apply_dict_to_widgets restores each
    tick with signals blocked, so nothing ever enabled the boxes again.
    """
    from midas_gui.tab_calibrate import CalibrationTab
    tab._seed_en_bc.setChecked(True)
    tab._seed_en_lsd.setChecked(True)
    tab._seed_en_ty.setChecked(True)
    tab._seed_lsd.setValue(13900.0)
    state = tab.get_state()

    fresh = CalibrationTab()
    fresh.set_state(state)
    assert fresh._seed_en_lsd.isChecked() and fresh._seed_lsd.isEnabled()
    assert fresh._seed_en_bc.isChecked() and fresh._seed_bcy.isEnabled()
    assert fresh._seed_en_ty.isChecked() and fresh._seed_ty.isEnabled()
    # ...and an unticked one stays locked, rather than everything turning on.
    assert not fresh._seed_en_tz.isChecked() and not fresh._seed_tz.isEnabled()


def test_the_three_tilt_boxes_are_the_same_width(tab):
    """tx is the same two-decimal degree field as ty and tz; only ty/tz were
    being narrowed, so tx rendered wider than its own siblings."""
    assert len({w.maximumWidth() for w in tab._seed_tilts}) == 1


# ── the seed dialog must reflect the Refine selection ────────────────────────
#
# Reported as "manual seed does not reflect the actual refine parameters
# selection below". The two are NOT wired together on purpose: seeding a
# parameter you are not refining is how you pin it to a measured value. So
# each row states the OUTCOME of the pairing instead.

def _status(tab):
    return {k: l.text() for k, l in tab._seed_dialog._status_lbls.items()}


def test_every_seed_row_says_what_the_fit_will_do(tab):
    assert set(_status(tab)) == {"BC", "Lsd", "tx", "ty", "tz", "Distortion"}


def test_refined_but_unseeded_reads_as_auto_seeded(tab):
    tab._ref_lsd.setChecked(True)
    tab._seed_en_lsd.setChecked(False)
    assert _status(tab)["Lsd"] == "refined, auto-seeded"


def test_refined_and_seeded_reads_as_starting_from_the_value(tab):
    tab._ref_lsd.setChecked(True)
    tab._seed_en_lsd.setChecked(True)
    assert _status(tab)["Lsd"] == "refined, from this value"


def test_seeded_but_not_refined_reads_as_pinned(tab):
    """The combination that looked like a contradiction: the seed dialog
    shows tx ticked while the Refine card calls tx fixed. Both are right —
    it means hold tx at this value."""
    tab._ref_tx.setChecked(False)
    tab._seed_en_tx.setChecked(True)
    assert _status(tab)["tx"] == "held at this value"


def test_neither_reads_as_held_at_default(tab):
    tab._ref_tx.setChecked(False)
    tab._seed_en_tx.setChecked(False)
    assert _status(tab)["tx"] == "held at default"


def test_the_labels_follow_the_refine_card_live(tab):
    """The Refine card stays editable while this non-modal dialog is open."""
    tab._seed_en_ty.setChecked(True)
    tab._ref_ty.setChecked(True)
    assert _status(tab)["ty"] == "refined, from this value"
    tab._ref_ty.setChecked(False)
    assert _status(tab)["ty"] == "held at this value"


# ── the two parameter lists read in the same order ──────────────────────
#
# Reported from the beamline: "we should also make the two list of
# parameters appear in the same order." The Refine card ran Lsd, BC, ty, tz,
# tx while the Manual seed dialog led with BC — so the same parameter sat at
# a different height in each. The two panels are deliberately independent
# choices (seeding says where the fit starts, refining whether it may move),
# which is exactly why they have to be read side by side.

def _grid_order(layout, QtWidgets):
    """Row labels of a QGridLayout, top to bottom — the checkbox text in
    column 0 where there is one, else the "name:" label in column 1."""
    rows = {}
    for i in range(layout.count()):
        r, c, *_ = layout.getItemPosition(i)
        w = layout.itemAt(i).widget()
        if w is None:
            continue
        if c == 0 and isinstance(w, QtWidgets.QCheckBox) and w.text():
            rows[r] = w.text()
        elif c == 1 and r not in rows and isinstance(w, QtWidgets.QLabel) \
                and w.text().endswith(":"):
            rows[r] = w.text().rstrip(":")
    return [rows[k] for k in sorted(rows)]


def test_the_refine_card_follows_the_canonical_parameter_order(tab):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    assert _grid_order(tab._refine_grid, QtWidgets) == \
        ["Lsd", "BC", "tx", "ty", "tz", "Wavelength"]


def test_the_seed_dialog_follows_the_same_order(tab):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    grid = tab._seed_dialog.findChild(QtWidgets.QGridLayout)
    # "Beam centre" is the seed dialog's label for the BC pair (the backend
    # takes BC_y/BC_z together), and Distortion is seedable but has no
    # Wavelength counterpart — the shared parameters must still line up.
    assert _grid_order(grid, QtWidgets) == \
        ["Lsd", "Beam centre", "tx", "ty", "tz", "Distortion"]


def test_both_panels_order_their_shared_parameters_identically(tab):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    refine = _grid_order(tab._refine_grid, QtWidgets)
    seed = [("BC" if n == "Beam centre" else n)
            for n in _grid_order(tab._seed_dialog.findChild(QtWidgets.QGridLayout),
                                 QtWidgets)]
    shared = set(refine) & set(seed)
    assert len(shared) >= 5
    assert [n for n in refine if n in shared] == [n for n in seed if n in shared]


def test_the_order_comes_from_one_place(tab):
    """Three hand-written sequences is how they drifted apart; the Refine
    card now derives its order from PARAMETER_LIMIT_ROWS."""
    from midas_gui.dialogs import PARAMETER_LIMIT_ROWS
    canonical = [r[0] for r in PARAMETER_LIMIT_ROWS]
    assert canonical.index("Lsd") < canonical.index("BC_y") \
        < canonical.index("tx") < canonical.index("ty") < canonical.index("tz")
    # ...and the refine flags' own logical order already agreed with it.
    from midas_gui.tab_calibrate import CalibrationTab
    assert CalibrationTab._REFINE_BOXES == ("Lsd", "BC", "tx", "ty", "tz",
                                            "Wavelength")
