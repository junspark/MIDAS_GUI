"""The Hydra viewer gets the same lab-frame compass as the Data Viewer.

Asked for directly: "can you add lab frame toggle like the single panel
case in the hydra case?"

It matters more here than on the single-detector tab. The composite is
built by rotating four panels about the beam into one windmill canvas, so
"which way is the hutch" is exactly the question the picture makes hard to
answer by eye -- and the toolbar had no way to ask it.

The one difference from tab_view's version is the anchor: this page shows a
different detector depending on the toolbar, so the compass follows
whichever card is active instead of a single geometry card.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def page(app):
    from midas_gui.hydra_page import HydraViewerPage
    p = HydraViewerPage()
    p._viewer.set_image(np.zeros((256, 256), dtype=float))
    return p


def test_the_overlay_is_off_until_asked_for(page):
    assert not page._lab_axes_on.isChecked()
    assert page._axis_items == []


def test_toggling_draws_and_clears_the_compass(page):
    page._lab_axes_on.setChecked(True)
    assert page._axis_items, "the compass should be on the viewer"
    page._lab_axes_on.setChecked(False)
    assert page._axis_items == [], "and fully removed again, not just hidden"


def test_the_compass_follows_the_panel_being_shown(page):
    """Each Hydra panel has its own beam centre, so an overlay anchored to
    one card would point from the wrong place on every other panel."""
    page._cards["ge1"]._bcy.setValue(60.0); page._cards["ge1"]._bcz.setValue(70.0)
    page._cards["ge2"]._bcy.setValue(190.0); page._cards["ge2"]._bcz.setValue(200.0)

    page._toolbar.set_current("ge1")
    page._lab_axes_on.setChecked(True)
    first = [it for it in page._axis_items]
    assert first

    page._toolbar.set_current("ge2")
    page._draw_lab_axes()
    assert page._axis_items, "still drawn after a panel switch"
    # Rebuilt, not reused -- a stale item set would be anchored to ge1's BC.
    assert all(it not in first for it in page._axis_items)


def test_a_geometry_with_no_beam_centre_draws_nothing(page):
    """Better an absent compass than one anchored at a guessed origin."""
    page._lab_axes_on.setChecked(True)
    page._clear_lab_axes()
    card = page._cards[page._toolbar.current()]
    orig = card.get_geometry
    card.get_geometry = lambda: {}
    try:
        page._draw_lab_axes()
        assert page._axis_items == []
    finally:
        card.get_geometry = orig


# ── loading four panel calibrations at once ──────────────────────────────

def test_a_panel_is_identified_by_the_last_ge_in_the_path():
    """Asked for: "this card should show option to load 4x calibration
    files, one for each panel."

    Matched by name rather than by pick order, because a file dialog
    returns its selection sorted rather than in click order -- trusting
    order would silently cross-assign panels, and a calibration on the
    wrong panel gives a composite that looks plausible and is wrong.
    """
    from midas_gui.hydra_page import HydraViewerPage as H
    assert H._panel_for_file("/d/s_00001_ge1.instr.txt") == 1
    assert H._panel_for_file("/d/ge2/calibration.json") == 2
    assert H._panel_for_file("/d/s_00001_GE4.instr.txt") == 4    # any case
    # The file's own name wins over a run folder that happens to say ge1.
    assert H._panel_for_file("/d/ge1_run/s_00001_ge3.poni") == 3
    # Not a panel: guessing here would put a calibration on the wrong one.
    assert H._panel_for_file("/d/calibration.json") is None
    assert H._panel_for_file("/d/ge12/x.txt") is None


def test_each_matched_file_goes_to_its_own_card(page, monkeypatch):
    seen = {}
    for n in (1, 2, 3, 4):
        card = page._cards[f"ge{n}"]
        monkeypatch.setattr(card, "set_calib_path",
                            lambda p, k=n: seen.__setitem__(k, p))
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileNames",
        staticmethod(lambda *a, **k: (
            ["/d/s_ge3.instr.txt", "/d/s_ge1.instr.txt"], "")))
    page._load_four_calibrations()
    assert seen == {1: "/d/s_ge1.instr.txt", 3: "/d/s_ge3.instr.txt"}
    msg = page._load4_lbl.text()
    assert "loaded ge1, ge3" in msg
    # Says what it did NOT touch, so a partial pick is not mistaken for a
    # full one.
    assert "unchanged: ge2, ge4" in msg


def test_unmatched_and_duplicate_picks_are_reported_not_guessed(page, monkeypatch):
    for n in (1, 2, 3, 4):
        monkeypatch.setattr(page._cards[f"ge{n}"], "set_calib_path",
                            lambda p: None)
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileNames",
        staticmethod(lambda *a, **k: (
            ["/d/a_ge1.txt", "/d/b_ge1.txt", "/d/plain.json"], "")))
    page._load_four_calibrations()
    msg = page._load4_lbl.text()
    assert "no ge1–4 in: plain.json" in msg
    assert "more than one file for ge1" in msg and "kept the first" in msg


def test_cancelling_the_dialog_changes_nothing(page, monkeypatch):
    touched = []
    for n in (1, 2, 3, 4):
        monkeypatch.setattr(page._cards[f"ge{n}"], "set_calib_path",
                            lambda p: touched.append(p))
    monkeypatch.setattr("PyQt5.QtWidgets.QFileDialog.getOpenFileNames",
                        staticmethod(lambda *a, **k: ([], "")))
    page._load_four_calibrations()
    assert touched == [] and page._load4_lbl.text() == ""


def test_one_pick_fills_in_its_siblings(page, tmp_path, monkeypatch):
    """Reported: "I loaded 4x calibration files ... and I still see only 1x
    set of calibration." The button promises four and then hands over a
    multi-select dialog, so picking one file and pressing Open loaded one
    panel. Panel calibrations are written as a set and named alike, so the
    other three are usually sitting right next to the one that was clicked.
    """
    for n in (1, 2, 3, 4):
        (tmp_path / f"s_00001.ge{n}.instr.txt").write_text("Lsd 1\n")
    seen = {}
    for n in (1, 2, 3, 4):
        monkeypatch.setattr(page._cards[f"ge{n}"], "set_calib_path",
                            lambda p, k=n: seen.__setitem__(k, p))
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileNames",
        staticmethod(lambda *a, **k: (
            [str(tmp_path / "s_00001.ge1.instr.txt")], "")))
    page._load_four_calibrations()
    assert sorted(seen) == [1, 2, 3, 4]
    assert seen[3].endswith("ge3.instr.txt")
    # Says which ones arrived without being clicked.
    assert "matched by name: ge2, ge3, ge4" in page._load4_lbl.text()


def test_an_explicit_pick_is_never_overridden(page, tmp_path, monkeypatch):
    """Filling in gaps must not second-guess a file the user chose."""
    for n in (1, 2):
        (tmp_path / f"s_00001.ge{n}.instr.txt").write_text("Lsd 1\n")
    (tmp_path / "other.ge2.instr.txt").write_text("Lsd 2\n")
    seen = {}
    for n in (1, 2, 3, 4):
        monkeypatch.setattr(page._cards[f"ge{n}"], "set_calib_path",
                            lambda p, k=n: seen.__setitem__(k, p))
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileNames",
        staticmethod(lambda *a, **k: (
            [str(tmp_path / "s_00001.ge1.instr.txt"),
             str(tmp_path / "other.ge2.instr.txt")], "")))
    page._load_four_calibrations()
    assert seen[2].endswith("other.ge2.instr.txt"), "the chosen file wins"


def test_nothing_is_invented_when_there_are_no_siblings(page, tmp_path, monkeypatch):
    (tmp_path / "lonely.ge1.instr.txt").write_text("Lsd 1\n")
    seen = {}
    for n in (1, 2, 3, 4):
        monkeypatch.setattr(page._cards[f"ge{n}"], "set_calib_path",
                            lambda p, k=n: seen.__setitem__(k, p))
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getOpenFileNames",
        staticmethod(lambda *a, **k: (
            [str(tmp_path / "lonely.ge1.instr.txt")], "")))
    page._load_four_calibrations()
    assert sorted(seen) == [1]
    assert "unchanged: ge2, ge3, ge4" in page._load4_lbl.text()
