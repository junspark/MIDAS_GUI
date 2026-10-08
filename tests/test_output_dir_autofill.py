"""The Output folder must follow the loaded data, but never eat a user's pick.

Both batch tabs auto-filled the Output folder only while the field was empty.
On Batch Integrate that meant *never*: the loader opens the bundled sample
data during construction, so by the time real data was loaded the field
already held the sample's path and the guard returned early. Measured on a
fresh BatchTab before the fix::

    out before            : .../MIDAS_GUI_bc/nickel_tifs/test_data
    _suggest_output_dir() : .../connolly_oct26_bc/AgBeh_80p725keV_3s/pixirad
    out AFTER loading real data : .../MIDAS_GUI_bc/nickel_tifs/test_data

i.e. a run would write into whatever folder the sample data (or the previous
run) had named. The test is now provenance -- is this string still the one we
put there? -- rather than emptiness.

``_suggest_output_dir`` is pure path arithmetic and never stats anything, so
stubbing ``source_cfg`` exercises the real code path without needing the
beamline mounts. Qt imports inside fixtures per STATE.md.
"""
import pytest
from pathlib import Path

pytest.importorskip("PyQt5.QtWidgets")

# The real 1-ID-E layout: <outroot>/<expid>/<detector>/<froot>/<file>
PIXI = "/mnt/s1c/connolly_oct26/pixirad/AgBeh_80p725keV_3s/AgBeh_80p725keV_3s_003510.pixi.h5"
PIXI_OUT = Path("/mnt/s1c/connolly_oct26_bc/AgBeh_80p725keV_3s/pixirad")
# A second scan, to stand for "the user loaded different data". Path
# derivation itself is characterized in test_batch_output_dir.py; these
# cases are only about which of the two values wins.
SCAN2 = "/mnt/s1c/connolly_oct26/pixirad/gC_1cm_80p725keV_5s/gC_1cm_80p725keV_5s_003509.pixi.h5"
SCAN2_OUT = Path("/mnt/s1c/connolly_oct26_bc/gC_1cm_80p725keV_5s/pixirad")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(params=["integrate", "correct"])
def tab(qapp, request):
    """Both tabs carry the same rule, so every case below runs against both."""
    if request.param == "integrate":
        from midas_gui.tab_batch import BatchTab
        t = BatchTab()
    else:
        from midas_gui.tab_batch_correct import BatchCorrectionTab
        t = BatchCorrectionTab()
    t._expid_provider = None
    return t


def _load(tab, path):
    """Point the tab at `path` and fire the same handler dataChanged does."""
    tab._loader.source_cfg = lambda: {"path": str(path)}
    tab._maybe_autofill_output_dir()
    return tab._out_ed.text().strip()


def test_autofill_fills_from_the_loaded_source(tab):
    tab._out_ed.setText("")
    assert _load(tab, PIXI) == str(PIXI_OUT)


def test_a_stale_autofill_is_replaced_when_the_data_changes(tab):
    """The actual bug: the field already held a path, so nothing updated."""
    tab._out_ed.setText("")
    _load(tab, PIXI)
    assert _load(tab, SCAN2) == str(SCAN2_OUT)


def test_a_startup_seeded_path_does_not_wedge_the_field(tab):
    """A fresh tab may already show a path derived from the bundled sample
    data. That is ours, so real data must still win."""
    tab._out_ed.setText("")
    _load(tab, "/repo/MIDAS_GUI/test_data/nickel_tifs/x_000001.tif")
    seeded = tab._out_ed.text().strip()
    assert seeded                                  # the trap's precondition
    assert _load(tab, PIXI) == str(PIXI_OUT)


def test_a_folder_the_user_typed_is_never_overwritten(tab):
    tab._out_ed.setText("")
    _load(tab, PIXI)
    tab._out_ed.setText("/data/my_own_folder")     # as the … browse button does
    assert _load(tab, SCAN2) == "/data/my_own_folder"


def test_clearing_the_field_asks_for_a_fresh_suggestion(tab):
    tab._out_ed.setText("")
    _load(tab, PIXI)
    tab._out_ed.setText("")
    assert _load(tab, SCAN2) == str(SCAN2_OUT)


def test_suggest_button_result_is_ours_and_keeps_following_the_data(tab):
    """Suggest asks for the convention, so later data changes keep applying
    it — unlike a hand-picked folder."""
    tab._out_ed.setText("/data/my_own_folder")
    tab._loader.source_cfg = lambda: {"path": PIXI}
    tab._apply_suggested_output_dir()
    assert tab._out_ed.text().strip() == str(PIXI_OUT)
    assert _load(tab, SCAN2) == str(SCAN2_OUT)


def test_an_underivable_path_leaves_the_field_alone(tab):
    tab._out_ed.setText("")
    _load(tab, PIXI)
    tab._loader.source_cfg = lambda: {}            # nothing loaded
    tab._maybe_autofill_output_dir()
    assert tab._out_ed.text().strip() == str(PIXI_OUT)
