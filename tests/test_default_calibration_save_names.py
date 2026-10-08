"""Saving a calibration should propose a name, not a blank dialog.

The Data Viewer's and Hydra's "Load/save calibration" cards offered bare
names -- "paramstest.txt", "calibration.json" -- in whatever directory the
app happened to be launched from, so every save was a navigate-and-type and
five Hydra cards all proposed the same filename.

CalibrationTab has named its files <expid>_<data stem><suffix> beside the
data for a while (_default_save_stem / _default_save_path). These are the
same rule, so a geometry saved in one tab and the same geometry saved in
another are not named by different conventions.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import pytest

pytestmark = pytest.mark.forked


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def view(app):
    from midas_gui.tab_view import DataViewerTab
    return DataViewerTab()


def test_the_name_is_built_from_the_loaded_data(view):
    from pathlib import Path
    p = view._default_save_path(".instr.txt")
    assert p.endswith(".instr.txt")
    # Named after the data file, and sitting beside it rather than in the
    # launch directory -- the half of this that actually saves the typing.
    assert Path(p).name.startswith(Path(view._loader.data_path()).name.rsplit(".", 1)[0])
    assert Path(p).parent == Path(view._loader.data_path()).parent


def test_the_exp_id_is_prepended_when_there_is_one(view):
    from pathlib import Path
    without = Path(view._default_save_path(".poni")).name
    view.set_expid_provider(lambda: "connolly_oct26")
    with_id = Path(view._default_save_path(".poni")).name
    assert with_id == f"connolly_oct26_{without}"


def test_a_blank_exp_id_does_not_leave_a_dangling_underscore(view):
    from pathlib import Path
    view.set_expid_provider(lambda: "   ")
    name = Path(view._default_save_path(".instr.json")).name
    assert not name.startswith("_")
    # A provider that raises must not take the save down either.
    view.set_expid_provider(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert Path(view._default_save_path(".instr.json")).name == name


def test_the_card_actually_asks_for_the_name(view):
    """The provider is only useful if _save_calibration consults it."""
    assert view._geom_card._save_name_provider is not None
    assert view._geom_card._save_name_provider(".instr.txt") == \
        view._default_save_path(".instr.txt")


def test_each_hydra_card_names_its_own_panel(app):
    """Five cards describing five different detectors must not all propose
    the same filename -- that is five chances to overwrite the wrong one."""
    from midas_gui.hydra_page import HydraViewerPage
    page = HydraViewerPage()
    page._expid_provider = lambda: "connolly_oct26"
    names = {k: page._cards[k]._save_name_provider(".instr.txt")
             for k in ("ge1", "ge2", "ge3", "ge4", "composite")}
    assert len(set(names.values())) == 5, names
    for key, name in names.items():
        assert key in name, (key, name)


def test_a_card_without_a_provider_keeps_the_old_bare_name(app):
    """Cards nobody wired must behave exactly as before."""
    from midas_gui.hydra_geometry_card import DetectorGeometryCard
    assert DetectorGeometryCard()._save_name_provider is None
