"""Choosing which GE panels make up the Hydra array.

Asked for at the beamline: "in the hydra case, we should have an option to
choose which ge to include as hydra array." Before this, the array was
whatever ``helpers.hydra_siblings`` found on disk, with no way to leave a
panel out — a mispositioned or badly-saved panel had to be worked around
downstream.

All three Hydra pages treat ``siblingsChanged`` as the authoritative active
set (``_ensure_states_for_siblings``, ``_toolbar.set_available``), so the
selection is applied at that one emit point and no consumer knows it
exists. These tests pin that chokepoint, and the two things that are easy
to get wrong around it: that the default is still "every panel found", and
that a saved selection can never re-enable a panel the current path has no
file for.

``forked`` per .context/DECISIONS.md — builds Qt widgets.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_panels(tmp_path, panels=(1, 2, 3, 4), n_frames=5):
    """A Hydra-shaped set of sibling files, one per requested panel."""
    h5py = pytest.importorskip("h5py")
    for n in panels:
        with h5py.File(str(tmp_path / f"scan_1s_00001.ge{n}.h5"), "w") as f:
            f.create_dataset("exchange/data",
                             data=np.zeros((n_frames, 4, 3), np.float32))
    anchor = min(panels)
    return str(tmp_path / f"scan_1s_00001.ge{anchor}.h5")


@pytest.fixture
def loader(app):
    from midas_gui.hydra_widgets import HydraLoaderPanel
    return HydraLoaderPanel(mode="stream")


def _emissions(loader):
    seen = []
    loader.siblingsChanged.connect(lambda s: seen.append(sorted(s)))
    return seen


# ── the default must not change ─────────────────────────────────────────

def test_every_panel_found_is_selected_by_default(loader, tmp_path):
    """The pre-selection behaviour is the default: load a path, get every
    panel that exists beside it."""
    seen = _emissions(loader)
    loader._set_path(_make_panels(tmp_path))
    assert seen[-1] == [1, 2, 3, 4]
    assert all(cb.isChecked() and cb.isEnabled()
               for cb in loader._status_lbls.values())


def test_a_panel_with_no_file_is_disabled_and_unticked(loader, tmp_path):
    loader._set_path(_make_panels(tmp_path, panels=(1, 2, 4)))
    assert loader._status_lbls[3].isEnabled() is False
    assert loader._status_lbls[3].isChecked() is False
    for n in (1, 2, 4):
        assert loader._status_lbls[n].isEnabled() is True


def test_a_disabled_panel_cannot_be_forced_into_the_array(loader, tmp_path):
    """setChecked on a disabled box still sets the state in Qt, so the
    filter must gate on the found set, not only on the tick."""
    seen = _emissions(loader)
    loader._set_path(_make_panels(tmp_path, panels=(1, 2, 4)))
    loader._status_lbls[3].setChecked(True)
    assert 3 not in loader._selected_siblings()
    assert seen[-1] == [1, 2, 4]


# ── the selection itself ────────────────────────────────────────────────

def test_unticking_a_panel_drops_it_from_the_emitted_set(loader, tmp_path):
    seen = _emissions(loader)
    loader._set_path(_make_panels(tmp_path))
    loader._status_lbls[3].setChecked(False)
    assert seen[-1] == [1, 2, 4]


def test_the_other_panels_keep_their_own_paths(loader, tmp_path):
    loader._set_path(_make_panels(tmp_path))
    loader._status_lbls[3].setChecked(False)
    sel = loader._selected_siblings()
    assert set(sel) == {1, 2, 4}
    for n, path in sel.items():
        assert path.endswith(f".ge{n}.h5")


def test_reticking_restores_it(loader, tmp_path):
    seen = _emissions(loader)
    loader._set_path(_make_panels(tmp_path))
    loader._status_lbls[2].setChecked(False)
    loader._status_lbls[2].setChecked(True)
    assert seen[-1] == [1, 2, 3, 4]


def test_the_full_found_set_is_kept_for_reporting(loader, tmp_path):
    """Excluding a panel must never look like a failed detection, so the
    found set survives the filtering."""
    loader._set_path(_make_panels(tmp_path))
    loader._status_lbls[3].setChecked(False)
    assert sorted(loader._siblings) == [1, 2, 3, 4]
    assert sorted(loader._selected_siblings()) == [1, 2, 4]


def test_a_new_path_reselects_everything_it_finds(loader, tmp_path):
    """A stale tick must not carry across to an unrelated scan."""
    seen = _emissions(loader)
    loader._set_path(_make_panels(tmp_path))
    loader._status_lbls[3].setChecked(False)
    other = tmp_path / "second"
    other.mkdir()
    loader._set_path(_make_panels(other))
    assert seen[-1] == [1, 2, 3, 4]


# ── the info line ───────────────────────────────────────────────────────

def test_the_info_line_reports_found_and_selected(loader, tmp_path):
    loader._set_path(_make_panels(tmp_path))
    assert "4/4 panels" in loader._info_lbl.text()
    assert "selected" not in loader._info_lbl.text(), \
        "with nothing excluded the line should read as it always did"
    loader._status_lbls[3].setChecked(False)
    text = loader._info_lbl.text()
    assert "4/4 panels" in text and "3 selected" in text


def test_too_few_selected_says_so_rather_than_blaming_the_path(loader, tmp_path):
    """Unticking down to one panel is a different problem from a path with
    one panel beside it, and the two must not share a message."""
    loader._set_path(_make_panels(tmp_path))
    for n in (2, 3, 4):
        loader._status_lbls[n].setChecked(False)
    text = loader._info_lbl.text()
    assert "selected" in text and "check the path" not in text


def test_too_few_found_still_blames_the_path(loader, tmp_path):
    seen = _emissions(loader)
    loader._set_path(_make_panels(tmp_path, panels=(1,)))
    assert "check the path" in loader._info_lbl.text()
    assert seen[-1] == [], "a one-panel path is not a Hydra array"


# ── state round-trip ────────────────────────────────────────────────────

def test_the_selection_round_trips_through_state(loader, tmp_path, app):
    from midas_gui.hydra_widgets import HydraLoaderPanel
    path = _make_panels(tmp_path)
    loader._set_path(path)
    loader._status_lbls[3].setChecked(False)
    state = loader.get_state()
    assert state["panels"] == [1, 2, 4]

    other = HydraLoaderPanel(mode="stream")
    other._set_path(path)
    other.set_state(state)
    assert sorted(other._selected_siblings()) == [1, 2, 4]


def test_restoring_cannot_re_enable_a_panel_this_path_lacks(loader, tmp_path):
    """A project saved against a 4-panel scan, reopened on a 3-panel one,
    must not tick a box with no file behind it."""
    loader._set_path(_make_panels(tmp_path, panels=(1, 2, 4)))
    loader.set_state({"panels": [1, 2, 3, 4]})
    assert loader._status_lbls[3].isChecked() is False
    assert sorted(loader._selected_siblings()) == [1, 2, 4]


def test_state_without_panels_leaves_the_selection_alone(loader, tmp_path):
    """Every project saved before this existed has no "panels" key, and
    must keep restoring as "use every panel found"."""
    loader._set_path(_make_panels(tmp_path))
    loader.set_state({"fr_start": 0})
    assert sorted(loader._selected_siblings()) == [1, 2, 3, 4]
