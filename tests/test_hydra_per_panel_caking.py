"""Caking is per panel on the Batch Integrate Hydra page, not shared.

Asked for at 1-ID-E: "why is this shared? I want to be able to set individual
caking parameters for each panel, just like the mpe_wf_saxs_waxs project."

It was shared for no recorded reason. .context/DECISIONS.md documents the
Hydra split for this page -- hand-off, per-panel masks, deferred drift/MONITOR,
lazy construction -- and says nothing about integration parameters; the only
justification in the code was a comment reading "Rmin/Rmax -- shared across
panels like R bin/eta bin". Meanwhile the page's own Corner/Edge presets
computed Rmax "from whichever panel is currently selected" and wrote it into
the one shared field, which only makes sense if Rmax were per panel.

It is. Each GE module is a physically separate detector with its own beam
centre and its own corner distance -- the same side of the boundary Calibrate
already draws (DECISIONS 2026-08-24: shared wavelength/pixel/calibrant,
independent transforms and seed). mpe_wf_saxs_waxs settles it: one
cake_parameters.<beamline>.<detector>.csv per detector, and for Hydra each
panel IS a detector.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets. Most
cases here need only a card, not the whole (heaviest-in-repo) page.
"""
import pytest

pytestmark = pytest.mark.forked

#: The caking half of an mpe_wf cake_parameters CSV, as read off a real
#: 1-ID-E file (cake_parameters.1ide.pixirad.csv).
REAL_CSV = {"R_MIN": 20.0, "R_MAX": 3236.77, "R_STEP": 1.0,
            "ETA_MIN": -180.0, "ETA_MAX": 180.0, "ETA_STEP": 360.0}


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def card(app):
    from midas_gui.hydra_batch_widgets import HydraBatchPanelCard
    return HydraBatchPanelCard(3)


def test_the_six_caking_fields_are_the_cake_csv_vocabulary(card):
    """Named in CAKE_KEYS terms so an mpe_wf CSV round-trips without a
    second translation layer in between."""
    from midas_gui.cake_params import CAKE_KEYS
    assert set(card.caking()) == set(CAKE_KEYS) - {"OME_SUM", "OME_START", "OME_STEP"}


def test_a_real_cake_csv_round_trips(card):
    card.set_caking(REAL_CSV)
    assert card.caking() == REAL_CSV


def test_setting_caking_emits_once_not_once_per_field(card):
    """Each emission redraws that panel's overlay; six for one apply would
    be five redraws of a picture nobody saw."""
    seen = []
    card.cakingChanged.connect(seen.append)
    card.set_caking(REAL_CSV)
    assert seen == [3]


def test_a_partial_dict_leaves_the_other_fields_alone(card):
    card.set_caking(REAL_CSV)
    card.set_caking({"R_MAX": 900.0})
    got = card.caking()
    assert got["R_MAX"] == 900.0
    assert got["R_MIN"] == 20.0 and got["ETA_STEP"] == 360.0


def test_panels_are_independent(app):
    from midas_gui.hydra_batch_widgets import HydraBatchPanelCard
    a, b = HydraBatchPanelCard(1), HydraBatchPanelCard(2)
    a.set_caking({"R_MAX": 111.0})
    assert b.caking()["R_MAX"] == 0.0, "one panel's caking reached another"


def test_each_panel_has_its_own_colour_and_it_is_the_app_wide_one(app):
    """The card, the radial curve and the overlay must agree about which
    colour means ge3, or four overlays on one canvas are unreadable."""
    from midas_gui.hydra_widgets import panel_color, _HYDRA_CURVE_COLORS
    seen = {panel_color(n) for n in (1, 2, 3, 4)}
    assert len(seen) == 4, "two panels share a colour"
    for n in (1, 2, 3, 4):
        assert panel_color(n) == _HYDRA_CURVE_COLORS[f"ge{n}"]


def test_rmax_auto_sentinel_reaches_the_spec_as_none(card):
    """0.0 means "farthest corner", which the spec builder resolves. Passing
    a literal 0.0 would integrate out to zero radius."""
    captured = {}
    from midas_gui import hydra_batch_widgets as mod
    card.result = type("R", (), {"Lsd": 2e5, "pxY": 200.0, "wavelength_A": 0.1729})()
    orig = mod._build_spec
    mod._build_spec = lambda result, r_bin, e_bin, **kw: captured.update(kw) or "spec"
    try:
        card.set_caking({"R_MAX": 0.0})
        card.resolved_spec()
        assert captured["r_max"] is None
        card.set_caking({"R_MAX": 900.0})
        card.resolved_spec()
        assert captured["r_max"] == 900.0
    finally:
        mod._build_spec = orig


def test_eta_range_now_reaches_the_spec_at_all(card):
    """eta_min/eta_max had no widget on this page before, so no Hydra run
    could ever limit its azimuth — both spec builders always accepted them."""
    captured = {}
    from midas_gui import hydra_batch_widgets as mod
    card.result = type("R", (), {"Lsd": 2e5, "pxY": 200.0, "wavelength_A": 0.1729})()
    orig = mod._build_spec
    mod._build_spec = lambda result, r_bin, e_bin, **kw: captured.update(kw) or "spec"
    try:
        card.set_caking({"ETA_MIN": -30.0, "ETA_MAX": 45.0})
        card.resolved_spec()
    finally:
        mod._build_spec = orig
    assert captured["eta_min"] == -30.0 and captured["eta_max"] == 45.0


# ── mpe_wf cake_parameters CSV, one per panel ────────────────────────────
# Asked for: each panel loads/saves its own, "just like the
# mpe_wf_saxs_waxs project". Its convention is
# cake_parameters.<beamline>.<detector>.csv under <expid>_bc/
# (run_midas_for_cakes_gui.sh:1208), and on Hydra the panel IS the detector.

def test_the_suggested_name_is_the_one_mpe_wf_looks_up(app, monkeypatch):
    from midas_gui import settings
    from midas_gui.hydra_batch_page import HydraBatchPage
    monkeypatch.setattr(settings, "active_profile", lambda: "1-ID-E")
    page = HydraBatchPage()
    page.set_expid_provider(lambda: "connolly_oct26")
    for n in (1, 4):
        assert page._cake_csv_path(n).name == f"cake_parameters.1ide.ge{n}.csv"


def test_a_saved_file_is_byte_compatible_with_mpe_wf(app, tmp_path):
    """All nine columns, %g formatted. mpe_wf's reader turns a blank cell
    into a ValueError, so the three OME_* this page cannot supply are
    written as 0 rather than left empty."""
    from midas_gui.cake_params import write_cake_csv
    from midas_gui.hydra_batch_widgets import HydraBatchPanelCard
    card = HydraBatchPanelCard(2)
    card.set_caking(REAL_CSV)
    out = tmp_path / "c.csv"
    write_cake_csv(str(out), card.caking())
    header, row = out.read_text().strip().splitlines()
    assert header == ("R_MIN,R_MAX,R_STEP,ETA_MIN,ETA_MAX,ETA_STEP,"
                      "OME_SUM,OME_START,OME_STEP")
    assert row == "20,3236.77,1,-180,180,360,0,0,0"


def test_a_real_1ide_file_round_trips_into_a_panel(app, tmp_path):
    """Verbatim content of cake_parameters.1ide.pixirad.csv as found in a
    2026-10 experiment — including a LIMITED azimuth, which this page had
    no widget to represent before."""
    from midas_gui.cake_params import parse_cake_csv
    from midas_gui.hydra_batch_widgets import HydraBatchPanelCard
    src = tmp_path / "cake_parameters.1ide.pixirad.csv"
    src.write_text("R_MIN,R_MAX,R_STEP,ETA_MIN,ETA_MAX,ETA_STEP,"
                   "OME_SUM,OME_START,OME_STEP\n"
                   "20,3236.77,1,-25,100,360,10,0,0\n")
    card = HydraBatchPanelCard(1)
    card.set_caking(parse_cake_csv(str(src)))
    got = card.caking()
    assert got["ETA_MIN"] == -25.0 and got["ETA_MAX"] == 100.0
    assert got["R_MIN"] == 20.0 and got["R_MAX"] == 3236.77


def test_loading_one_panels_file_does_not_touch_the_others(app, fixture_tmp=None):
    from midas_gui.hydra_batch_widgets import HydraBatchPanelCard
    a, b = HydraBatchPanelCard(1), HydraBatchPanelCard(2)
    a.set_caking(REAL_CSV)
    assert b.caking()["R_MAX"] == 0.0


# ── which file these numbers came from ───────────────────────────────────
# Reported at 1-ID-E: "There is no way to tell which caking parameter file is
# being used." Six spin boxes look identical whether they were loaded from a
# workflow CSV, typed, or left at defaults -- and the run uses the boxes.
def test_a_card_starts_saying_it_has_no_file(card):
    assert "No cake file" in card._cake_src_lbl.text()
    assert card.cake_source() == ""


def test_loading_names_the_file(card):
    card.set_cake_source("cake_parameters.1ide.ge1.csv")
    assert "cake_parameters.1ide.ge1.csv" in card._cake_src_lbl.text()
    assert card._cake_src_lbl.text().startswith("from ")


def test_saving_says_so_rather_than_claiming_the_values_came_from_there(card):
    card.set_cake_source("cake_parameters.1ide.ge1.csv", verb="saved to")
    assert card._cake_src_lbl.text().startswith("saved to ")


def test_a_hand_edit_is_flagged_against_the_file(card):
    """The file no longer describes what will run."""
    card.set_cake_source("cake_parameters.1ide.ge1.csv")
    card._r_min.setValue(card._r_min.value() + 10.0)
    assert "edited since" in card._cake_src_lbl.text()


def test_the_flag_does_not_pile_up_on_repeated_edits(card):
    card.set_cake_source("cake_parameters.1ide.ge1.csv")
    for d in (1.0, 2.0, 3.0):
        card._r_min.setValue(card._r_min.value() + d)
    assert card._cake_src_lbl.text().count("edited since") == 1


def test_setting_values_then_the_source_does_not_read_as_edited(card):
    """set_caking fires valueChanged, so the source must be set after it --
    otherwise every freshly loaded file immediately claims to be edited."""
    card.set_caking({"R_MIN": 40.0, "R_MAX": 900.0, "R_STEP": 0.5,
                     "ETA_MIN": -180.0, "ETA_MAX": 180.0, "ETA_STEP": 5.0})
    card.set_cake_source("cake_parameters.1ide.ge1.csv")
    assert "edited since" not in card._cake_src_lbl.text()
