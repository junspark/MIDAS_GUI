"""The Calibrate tab's left column must tell the truth about the seed.

The seed summary must show the seed *values*, not just which parameters are
seeded, and the Limits note must re-centre when one of those values moves.

The seed spin boxes live inside ``ManualSeedDialog``, which is non-modal and
normally closed, so from the tab itself the starting Lsd / BC / tilts were
invisible — while the Refine card right below promised "± a window around its
seed value" and the crystalline footer its window "centred on the seed", neither of which
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
    """Same wiring, the other note. A %-width window is a fraction of the
    seed, so the crystalline footer has to recompute when the seed moves —
    otherwise it quotes a width the run will not use."""
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
    return {k: l.text() for k, l in tab._param_table._status_lbls.items()}


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

def _table_order(tab):
    """Row names top to bottom, read from the table's own column-0 labels.

    Column 0 names every row now. It used to be read off whichever
    checkbox happened to carry the caption, which is exactly the coupling
    that let the two panels drift into different orders."""
    t = tab._param_table
    return [t.row_name_lbls[n].text()
            for n in sorted(t.row_name_lbls, key=lambda n: t.limit_row_index[n])]


def test_the_table_follows_the_canonical_parameter_order(tab):
    """Every parameter, in PARAMETER_LIMIT_ROWS order, named once in
    column 0. "Beam centre" captions the pair's first row because one seed
    tick gates both (the backend takes BC_y/BC_z together)."""
    assert _table_order(tab) == ["Lsd", "Beam centre", "BC_z", "tx", "ty",
                                 "tz", "Wavelength", "Distortion"]


def test_seeding_and_refining_are_read_on_one_row(tab):
    """They used to be two grids that had to be kept in the same order, in
    two panels that could not be seen at once. One grid is what retires that
    whole class of drift -- so assert the pairing is per row, not that two
    orders happen to agree."""
    table = tab._param_table
    for key, seed_box, value in (("Lsd", tab._seed_en_lsd, tab._seed_lsd),
                                 ("tx", tab._seed_en_tx, tab._seed_tx)):
        line = table.limit_row_index[key]
        for w in (seed_box, table._refine_boxes[key], value):
            assert table.grid.getItemPosition(
                _index_of(table.grid, w))[0] == line, \
                f"{key}: {w} is not on its parameter's line"


def _index_of(layout, widget):
    for i in range(layout.count()):
        if layout.itemAt(i).widget() is widget:
            return i
    raise AssertionError(f"{widget} is not in the grid")


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
