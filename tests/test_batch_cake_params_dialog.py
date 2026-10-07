"""``BatchTab``'s Cake parameters editor — the nine-column mpe_wf view.

Covers the two things that are easy to break silently: that the dialog's own
spinboxes really are a two-way mirror of the tab's widgets (it copies rather
than hosting them — see ``_CakeParamsDialog``'s docstring for why it has to),
and that ``OME_START``/``OME_STEP`` survive a full load → edit → save cycle.

The dialog also hosts the two omega controls that are NOT CSV columns — the
measured-channel combo and the averaged/summed override — and keeping those
out of ``SPEC`` while still round-tripping them is its own trap, so both
halves of that are pinned below.

Qt import discipline per STATE.md: import inside fixtures, never at module
scope.
"""
import pytest
from pathlib import Path

pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(qapp):
    from midas_gui.tab_batch import BatchTab
    return BatchTab()


@pytest.fixture
def dialog(tab):
    from midas_gui.tab_batch import _CakeParamsDialog
    return _CakeParamsDialog(tab)


def _write(path, **over):
    from midas_gui.cake_params import CAKE_KEYS, write_cake_csv
    values = {k: 0 for k in CAKE_KEYS}
    values.update(over)
    write_cake_csv(str(path), values)
    return values


# ── the mirror ───────────────────────────────────────────────────────────

def test_the_dialog_reads_the_tab_on_open(tab, dialog):
    tab._r_min.setValue(12.0)
    tab._eta_max.setValue(90.0)
    tab._ome_start.setValue(3.25)
    dialog.load_from_tab()
    assert dialog._spins["R_MIN"].value() == 12.0
    assert dialog._spins["ETA_MAX"].value() == 90.0
    assert dialog._spins["OME_START"].value() == 3.25


def test_apply_pushes_every_mapped_key_back(tab, dialog):
    dialog.load_from_tab()
    dialog._spins["R_STEP"].setValue(4.0)
    dialog._spins["ETA_MIN"].setValue(-90.0)
    dialog._spins["OME_SUM"].setValue(7)
    dialog._spins["OME_STEP"].setValue(0.5)
    dialog.apply_to_tab()
    assert tab._r_bin.value() == 4.0
    assert tab._eta_min.value() == -90.0
    assert tab._loader._combine_chunk.value() == 7
    assert tab._ome_step.value() == 0.5


def test_apply_refreshes_the_summary_line(tab, dialog):
    """The summary label under the button is the only always-visible readout
    of these values; an Apply it didn't follow would look like a no-op."""
    dialog.load_from_tab()
    dialog._spins["R_STEP"].setValue(3.0)
    dialog.apply_to_tab()
    assert "ΔR 3 px" in tab._cake_lbl.text()


def test_editing_the_dialog_without_apply_leaves_the_tab_alone(tab, dialog):
    """The reason it mirrors instead of hosting: you can try numbers here and
    close without having changed the next run."""
    before = tab._r_bin.value()
    dialog.load_from_tab()
    dialog._spins["R_STEP"].setValue(before + 1.0)
    assert tab._r_bin.value() == before


def test_mirror_spins_inherit_the_range_of_what_they_stand_in_for(tab, dialog):
    """So the dialog can't accept a value the tab would silently clamp."""
    assert dialog._spins["R_STEP"].minimum() == tab._r_bin.minimum()
    assert dialog._spins["R_STEP"].maximum() == tab._r_bin.maximum()
    assert dialog._spins["ETA_MIN"].minimum() == tab._eta_min.minimum()


# ── the inert pair ───────────────────────────────────────────────────────

def test_ome_start_and_step_survive_a_full_round_trip(tab, dialog, tmp_path, monkeypatch):
    """Loaded from a CSV, carried through the dialog, written back out
    unchanged — the two values Batch Integrate turns into every frame's ω."""
    from midas_gui import cake_params, tab_batch
    src = tmp_path / "in.csv"
    _write(src, R_MIN=10, R_STEP=2, OME_START=1.75, OME_STEP=0.25)
    monkeypatch.setattr(tab_batch, "_browse", lambda *a, **k: str(src))
    tab._load_cake_csv()
    assert tab._ome_start.value() == 1.75
    assert tab._ome_step.value() == 0.25

    dialog.load_from_tab()
    out = tmp_path / "out.csv"
    cake_params.write_cake_csv(str(out), dialog.values())
    back = cake_params.parse_cake_csv(str(out))
    assert back["OME_START"] == 1.75
    assert back["OME_STEP"] == 0.25
    assert back["R_MIN"] == 10.0 and back["R_STEP"] == 2.0


def test_ome_start_and_step_round_trip_through_gui_state(tab):
    """They are widgets, not floats, precisely so Save/Load GUI State carries
    them with no extra code — assert that actually holds."""
    tab._ome_start.setValue(2.5)
    tab._ome_step.setValue(0.75)
    state = tab.get_state()
    other = type(tab)()
    other.set_state(state)
    assert other._ome_start.value() == 2.5
    assert other._ome_step.value() == 0.75


# ── the suggested save path ──────────────────────────────────────────────

def _with_source(tab, path, expid=""):
    tab._loader.source_cfg = lambda: {"path": str(path)}
    tab._expid_provider = (lambda: expid) if expid else None
    return tab


def test_suggested_path_lands_in_an_existing_bc_directory(tab, tmp_path, monkeypatch):
    from midas_gui import tab_batch
    bc = tmp_path / "park_may26_bc"
    (bc / "sam1").mkdir(parents=True)
    _with_source(tab, bc / "sam1" / "sam1_000001.tif")
    monkeypatch.setattr(tab_batch.settings, "active_profile", lambda: "20-ID-E")
    out = tab._suggest_cake_csv_path()
    assert out.parent == bc
    assert out.name.startswith("cake_parameters.20ide.")
    assert out.suffix == ".csv"


def test_suggested_path_carries_the_detector_token(tab, tmp_path, monkeypatch):
    from midas_gui import tab_batch
    bc = tmp_path / "export" / "park_may26_bc"
    src = tmp_path / "export" / "park_may26" / "s20varex2" / "sam1"
    src.mkdir(parents=True)
    bc.mkdir(parents=True)
    _with_source(tab, src / "sam1_000001.tif")
    monkeypatch.setattr(tab_batch.settings, "active_profile", lambda: "20-ID-E")
    assert tab._suggest_cake_csv_path() == bc / "cake_parameters.20ide.s20varex2.csv"


def test_a_flat_source_folder_still_yields_a_usable_path(tab, tmp_path):
    """No ``_bc`` anywhere and a shallow layout: it must degrade to somewhere
    real rather than raising or naming a directory nobody created."""
    flat = tmp_path / "flat"
    flat.mkdir()
    _with_source(tab, flat / "img_000001.tif")
    out = tab._suggest_cake_csv_path()
    assert out.name.endswith(".csv")
    assert out.parent.is_dir()


def test_no_source_loaded_still_yields_a_usable_path(tab):
    _with_source(tab, "")
    out = tab._suggest_cake_csv_path()
    assert out.name.endswith(".csv")
    assert out.parent.is_dir()


def test_the_dialog_covers_every_csv_column_in_order(dialog):
    """``SPEC`` is hand-written next to the rows it builds; if a column is
    ever added to ``CAKE_KEYS`` the dialog must grow a row for it, or it will
    quietly write a default for something the user was never shown."""
    from midas_gui.cake_params import CAKE_KEYS
    assert tuple(k for k, _l, _a in type(dialog).SPEC) == CAKE_KEYS
    assert tuple(dialog._spins) == CAKE_KEYS


# ── the two omega controls that are not CSV columns ──────────────────────

def test_the_omega_channel_and_override_mirror_both_ways(tab, dialog):
    """They live on the tab and the dialog copies them, exactly like the nine
    spinboxes — so an edit only lands when the user hits apply."""
    tab._ome_channel.setEditText("measurement/omegas")
    tab._ome_collapse.setChecked(True)
    dialog.load_from_tab()
    assert dialog._extras["_ome_channel"].currentText() == "measurement/omegas"
    assert dialog._extras["_ome_collapse"].isChecked() is True

    dialog._extras["_ome_channel"].setEditText("other/ome")
    dialog._extras["_ome_collapse"].setChecked(False)
    assert tab._ome_channel.currentText() == "measurement/omegas", "not yet"
    dialog.apply_to_tab()
    assert tab._ome_channel.currentText() == "other/ome"
    assert tab._ome_collapse.isChecked() is False


def test_the_omega_controls_stay_out_of_the_csv(tab, dialog):
    """``write_cake_csv`` takes ``dialog.values()``; a channel name is not a
    number and mpe_wf's reader would choke on a tenth column, so these two
    must never reach it. ``SPEC`` being the only source of ``values()`` is
    what enforces that — this asserts the consequence."""
    from midas_gui.cake_params import CAKE_KEYS
    tab._ome_channel.setEditText("measurement/omegas")
    tab._ome_collapse.setChecked(True)
    dialog.load_from_tab()
    assert set(dialog.values()) == set(CAKE_KEYS)


def test_the_omega_controls_round_trip_through_gui_state(tab):
    """A combo and a checkbox, not floats — ``widgets_to_dict`` handles both
    already, but only if they are registered in ``_state_widgets``."""
    tab._ome_channel.setEditText("measurement/omegas")
    tab._ome_collapse.setChecked(True)
    state = tab.get_state()
    other = type(tab)()
    other.set_state(state)
    assert other._ome_channel.currentText() == "measurement/omegas"
    assert other._ome_collapse.isChecked() is True


def test_the_run_config_reports_what_the_widgets_say(tab):
    """``_omega_cfg`` is the one place the UI meets ``BatchWorker``; a typo
    here would silently give every run the defaults."""
    tab._ome_start.setValue(5.0)
    tab._ome_step.setValue(0.25)
    tab._ome_channel.setEditText("  measurement/omegas  ")
    tab._ome_collapse.setChecked(True)
    assert tab._omega_cfg() == {"start": 5.0, "step": 0.25,
                                "channel": "measurement/omegas",
                                "collapse": True}


def test_an_unconfigured_tab_asks_for_zero_degrees(tab):
    """Not ``None``, not a sentinel — the defaults are a stationary sample."""
    assert tab._omega_cfg() == {"start": 0.0, "step": 0.0,
                                "channel": "", "collapse": False}


# ── η binning: one lineout over a limited azimuth ───────────────────────

def test_eta_bin_accepts_a_single_bin_spanning_the_whole_range(tab):
    """Reported: "I also want to range 110 for eta step to get 1 lineout but
    the gui will not let me."

    η bin was capped at 30°, which is not a physical limit — one bin
    covering the entire selected azimuth is a normal thing to ask for, and
    is how you get a single lineout out of a wedge. η can span at most a
    full turn, so that is the cap.
    """
    assert tab._e_bin.maximum() >= 360.0
    tab._eta_min.setValue(-100.0)
    tab._eta_max.setValue(10.0)
    tab._e_bin.setValue(110.0)
    assert tab._e_bin.value() == pytest.approx(110.0), \
        "the η bin was silently clamped"


def test_the_cake_dialog_mirrors_the_widened_eta_range(dialog):
    """The dialog builds its spins from the tab's own ranges, so a clamp
    would reappear here if the two ever drifted."""
    dialog._spins["ETA_STEP"].setValue(110.0)
    assert dialog._spins["ETA_STEP"].value() == pytest.approx(110.0)


def test_the_omega_channel_popup_is_wide_enough_for_full_paths(dialog):
    """HDF5 dataset paths differ only in the middle — instrument/SMS/D/HR/
    against instrument/SMS/E/HL/ — and that is exactly what Qt elides when
    the popup inherits the closed combo's width."""
    combo = dialog._ome_channel
    combo.clear()
    for name in ("", "instrument/SMS/D/HR/samRy", "instrument/SMS/E/HL/samRy"):
        combo.addItem(name)
    dialog._widen_popup(combo)
    longest = max(combo.fontMetrics().boundingRect(combo.itemText(i)).width()
                  for i in range(combo.count()))
    assert combo.view().minimumWidth() >= longest, \
        "the popup is narrower than its longest entry"
