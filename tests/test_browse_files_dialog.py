"""``dialogs.BrowseFilesDialog`` — the four source-selection modes.

Reported from the beamline: "Multiple files ... the menu did not let me",
and "This option to choose multiple files with the same froot also having
issues." What those turned out to be is pinned here.

Builds Qt widgets, hence forked, with every Qt / midas_gui import deferred
into the fixtures — STATE.md's rule for new Qt test files.
"""
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def scan(tmp_path):
    """A froot's worth of files, plus the dark pair the beamline leaves
    alongside them."""
    folder = tmp_path / "Cu_tensile_cracked"
    folder.mkdir()
    for n in range(1382, 1395):
        (folder / f"Cu_tensile_cracked_{n:06d}.h5").write_bytes(b"x")
    for tag in ("before", "after"):
        (folder / f"Cu_tensile_cracked_dark_{tag}_001381.h5").write_bytes(b"x")
    return folder


# ── the froot parse behind "Files sharing a name stem" ──────────────────

@pytest.mark.parametrize("name,froot", [
    ("Cu_tensile_cracked_001382.h5", "Cu_tensile_cracked"),
    ("CeO2_65keV_1040mm_0pt2x0pt2mm2_5s_030319.vrx.h5",
     "CeO2_65keV_1040mm_0pt2x0pt2mm2_5s"),
    ("Fe9Cr_KGT6038_load1_waxs_029339.vrx.h5", "Fe9Cr_KGT6038_load1_waxs"),
    ("AgBeH_10s_000021.h5", "AgBeH_10s"),
    ("oddname.h5", "oddname"),          # no trailing number — plain stem
])
def test_froot_strips_the_file_number_and_detector_tag(app, name, froot):
    from midas_gui.dialogs import _froot_of
    assert _froot_of(name) == froot


def test_stem_autofill_matches_the_whole_froot_not_one_file(app, scan):
    """The defect: auto-filling with ``Path(...).stem`` put the file NUMBER
    in the stem, so the pattern matched exactly the file that was clicked
    and "Files sharing a name stem" quietly degenerated into "Single
    file"."""
    from midas_gui.dialogs import BrowseFilesDialog, _froot_of
    dlg = BrowseFilesDialog(None, title="t", modes=("file", "files", "folder",
                                                    "stem"),
                            start_dir=str(scan))
    dlg._mode_btns["stem"].setChecked(True)
    dlg._stem_ed.setText(_froot_of("Cu_tensile_cracked_001382.h5"))
    matches = dlg._stem_matches()
    assert len(matches) == 15, "the froot should reach every sibling"

    dlg._stem_ed.setText("Cu_tensile_cracked_001382")   # the old behaviour
    assert len(dlg._stem_matches()) == 1


def test_stem_mode_ok_needs_a_stem_that_matches_something(app, scan):
    from midas_gui.dialogs import BrowseFilesDialog
    dlg = BrowseFilesDialog(None, title="t", modes=("file", "files", "folder",
                                                    "stem"),
                            start_dir=str(scan))
    dlg._mode_btns["stem"].setChecked(True)
    assert not dlg._ok_btn.isEnabled(), "empty stem matches nothing"
    dlg._stem_ed.setText("Cu_tensile_cracked")
    dlg._update_ok_enabled()
    assert dlg._ok_btn.isEnabled()


# ── "Multiple files" ─────────────────────────────────────────────────────

def test_multiple_files_mode_allows_an_extended_selection(app, scan):
    """Reported as "the menu did not let me". The mode does switch the view
    to ExtendedSelection — Ctrl/Shift-click — which the info line now says
    out loud, because nothing else on screen did."""
    from PyQt5 import QtWidgets
    from midas_gui.dialogs import BrowseFilesDialog
    dlg = BrowseFilesDialog(None, title="t", modes=("file", "files", "folder",
                                                    "stem"),
                            start_dir=str(scan))
    assert dlg._tree.selectionMode() == \
        QtWidgets.QAbstractItemView.SingleSelection
    dlg._mode_btns["files"].setChecked(True)
    assert dlg._tree.selectionMode() == \
        QtWidgets.QAbstractItemView.ExtendedSelection
    dlg._update_info()
    assert "Ctrl-click" in dlg._info.text()


def test_the_dialog_is_not_taller_than_it_claims(app, scan):
    """Its bottom row carries the stem field and the OK button; if the
    layout ever demanded more height than the window, those would be the
    first things to go missing."""
    from midas_gui.dialogs import BrowseFilesDialog
    dlg = BrowseFilesDialog(None, title="t", modes=("file", "files", "folder",
                                                    "stem"),
                            start_dir=str(scan))
    assert dlg.minimumSizeHint().height() <= dlg.size().height()
    dlg._mode_btns["stem"].setChecked(True)
    assert dlg._stem_row.isVisibleTo(dlg), "the stem field must be reachable"
