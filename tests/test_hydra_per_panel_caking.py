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
