""""Full folder" must be able to reach frames in subfolders.

At 1-ID-E the pixirad data root is nothing but per-load-step subfolders
(``APS_4130_C_load_step_0`` …), so picking it in "Full folder" reported
"0 frame file(s) found in this folder" and there was no way to process the
tree. Recursion turns the pick into an explicit file list, so it is offered
only where the caller can consume one.
"""
import pytest
from PyQt5 import QtWidgets

from midas_gui.dialogs import BrowseFilesDialog, _find_frame_files

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tree(tmp_path):
    """A root holding only subfolders, like the real pixirad root."""
    (tmp_path / "loose_note.txt").write_text("not a frame")
    for step in range(3):
        d = tmp_path / f"load_step_{step}"
        d.mkdir()
        for i in range(2):
            (d / f"scan_{step}_{i:04d}.tif").write_bytes(b"")
    (tmp_path / "load_step_0" / "deeper").mkdir()
    (tmp_path / "load_step_0" / "deeper" / "scan_x_0001.h5").write_bytes(b"")
    return tmp_path


def _dlg(app, start, modes=("file", "files", "folder", "stem")):
    d = BrowseFilesDialog(None, start_dir=str(start), modes=modes)
    d._mode_btns["folder"].setChecked(True) if len(modes) > 1 else None
    return d


# ── the finder ───────────────────────────────────────────────────────────
def test_non_recursive_finds_nothing_in_a_folder_of_subfolders(tree):
    assert _find_frame_files(str(tree), recursive=False) == []


def test_recursive_finds_every_frame_at_any_depth(tree):
    found = _find_frame_files(str(tree), recursive=True)
    assert len(found) == 7                      # 3 steps x 2 tif, + 1 nested h5
    assert all(not f.endswith(".txt") for f in found)
    assert any(f.endswith("scan_x_0001.h5") for f in found)


def test_the_list_is_sorted_and_deduplicated(tree):
    found = _find_frame_files(str(tree), recursive=True)
    assert found == sorted(found)
    assert len(found) == len(set(found))


# ── where the checkbox appears ───────────────────────────────────────────
def test_offered_in_folder_mode(app, tree):
    d = _dlg(app, tree)
    assert d._recursive_chk.isVisibleTo(d)


def test_hidden_in_every_other_mode(app, tree):
    d = _dlg(app, tree)
    for mode in ("file", "files", "stem"):
        d._mode_btns[mode].setChecked(True)
        assert not d._recursive_chk.isVisibleTo(d), mode


def test_never_offered_when_the_caller_cannot_take_a_file_list(app, tree):
    """The Hydra panel cards pass modes without "files" — they set a single
    path or a stem, so a recursive sweep has nowhere to go."""
    d = _dlg(app, tree, modes=("file", "folder", "stem"))
    assert not d._recursive_chk.isVisibleTo(d)


# ── what it reports ──────────────────────────────────────────────────────
def test_the_dead_end_points_at_the_subfolders(app, tree):
    d = _dlg(app, tree)
    assert "0 frame file(s)" in d._info.text()
    assert "subfolder" in d._info.text(), "should say where the frames are"


def test_ticking_it_reports_the_recursive_count(app, tree):
    d = _dlg(app, tree)
    d._recursive_chk.setChecked(True)
    assert "7 frame file(s)" in d._info.text()


# ── OK gating ────────────────────────────────────────────────────────────
def test_an_empty_recursive_sweep_cannot_be_accepted(app, tmp_path):
    (tmp_path / "empty_sub").mkdir()
    d = _dlg(app, tmp_path)
    assert d._ok_btn.isEnabled(), "a plain folder pick may be watched as it fills"
    d._recursive_chk.setChecked(True)
    assert not d._ok_btn.isEnabled(), "a frozen list of nothing processes nothing"


def test_a_non_empty_recursive_sweep_can_be_accepted(app, tree):
    d = _dlg(app, tree)
    d._recursive_chk.setChecked(True)
    assert d._ok_btn.isEnabled()


# ── the result handed to the caller ──────────────────────────────────────
def test_accepting_yields_the_file_list_not_just_the_folder(app, tree):
    d = _dlg(app, tree)
    d._recursive_chk.setChecked(True)
    d._on_accept()
    assert d.mode() == "folder"          # the user did pick a folder
    assert d.recursive() is True
    assert len(d.paths()) == 7           # but this is what the caller uses
    assert d.folder() == str(tree)


def test_a_plain_folder_pick_is_unchanged(app, tree):
    d = _dlg(app, tree)
    d._on_accept()
    assert d.mode() == "folder"
    assert d.recursive() is False
    assert d.paths() == []
    assert d.folder() == str(tree)
