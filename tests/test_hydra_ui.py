"""Offscreen UI test for the Data Viewer tab's Hydra page: mode switching,
sibling loading, per-panel/composite display, per-panel calibration wiring,
and GUI-state round-trip. Uses the small synthetic fixture at
test_data/gui_synthetic/hydra/ (see make_hydra_test_data.py).
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "test_data" / "gui_synthetic" / "hydra"

# Run each test in this file in its own forked subprocess (pytest-forked):
# this file's pyqtgraph-teardown segfault, documented in .context/STATE.md /
# DECISIONS.md, then aborts only that subprocess, reported as a normal
# FAILED with signal info, instead of crashing the whole pytest run.
pytestmark = pytest.mark.forked


_QT_LOADED = False

# Bound by _load_qt() at fixture time, declared here so static analysis
# (and the pyflakes diff in the review recipe) can still resolve them.
QtWidgets = None
geometry_fields_from_file = HydraFieldSelector = DataViewerTab = None


def _load_qt():
    """Import Qt (and the GUI modules under test, which import it
    transitively) and publish them as module globals.

    Deliberately NOT done at module level. pytest imports this module during
    collection, in the *parent* process, while pytest-forked runs each test
    in a forked child. Importing PyQt5 in the parent initialises macOS
    CoreFoundation, and CoreFoundation may not be used in a forked child —
    every test then dies with SIGSEGV ("The process has forked and you
    cannot use this CoreFoundation functionality safely") before its body
    runs. Importing from inside the ``app`` fixture happens in the child, so
    each child initialises CoreFoundation itself, which is legal.

    See .context/STATE.md; ``tests/test_set_raw_frame.py`` is the file this
    pattern was taken from."""
    global _QT_LOADED
    if _QT_LOADED:
        return
    from PyQt5 import QtWidgets
    from midas_gui.helpers import geometry_fields_from_file
    from midas_gui.hydra_widgets import HydraFieldSelector
    from midas_gui.tab_view import DataViewerTab
    _QT_LOADED = True
    globals().update(
        QtWidgets=QtWidgets,
        geometry_fields_from_file=geometry_fields_from_file,
        HydraFieldSelector=HydraFieldSelector,
        DataViewerTab=DataViewerTab,
    )


@pytest.fixture(scope="module")
def app():
    _load_qt()
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _qt_teardown(app):
    """Each test builds a full DataViewerTab (Hydra page + 5 geometry cards
    + several pyqtgraph ViewBoxes). Left to accumulate across many tests in
    one process, pyqtgraph's global ViewBox registry has caused a hard
    segfault (reproduced when running this whole file, not any single test
    in isolation) — a known category of pyqtgraph teardown fragility around
    its ViewBox-list bookkeeping. An explicit ``gc.collect()`` here made it
    *worse* (crashed sooner) by forcing collection mid-teardown, so this
    only pumps the event loop to let Qt's own deferred deletion run instead
    of forcing Python-level GC."""
    yield
    app.processEvents()


@pytest.fixture()
def fixture_available():
    if not FIXTURE_DIR.exists():
        pytest.skip("test_data/gui_synthetic/hydra/ fixture not present — "
                    "run make_hydra_test_data.py")
    return FIXTURE_DIR


def test_hydra_field_selector_sibling_discovery_and_compute(app, tmp_path):
    """HydraFieldSelector (Dark/Bright/Background) auto-discovers sibling
    panel files the same way the main Hydra data path does, and computes
    each panel's field independently. A standalone QGroupBox (no
    pg.ImageView/ViewBox), so it doesn't add to the pyqtgraph-teardown
    -crash risk the other, heavier tests in this file are mindful of."""
    import h5py

    for n in (1, 2, 3, 4):
        d = tmp_path / f"ge{n}"
        d.mkdir()
        data = np.full((1, 8, 8), float(n) * 10.0, dtype=np.float32)
        with h5py.File(d / f"dark.ge{n}.h5", "w") as f:
            f.create_dataset("exchange/data", data=data)

    sel = HydraFieldSelector("Dark", default_dataset="exchange/data")
    sel.setChecked(True)
    sel._set_path(str(tmp_path / "ge1" / "dark.ge1.h5"))
    assert sorted(sel._sibling_paths.keys()) == [1, 2, 3, 4]
    assert sel.field(1) is None   # not computed yet

    sel._compute_all()
    for w in list(sel._workers.values()):
        w.wait()
    for _ in range(20):
        app.processEvents()

    for n in (1, 2, 3, 4):
        field = sel.field(n)
        assert field is not None
        assert np.allclose(field, float(n) * 10.0)

    sel.setChecked(False)
    assert sel.field(1) is None   # unchecked -> no correction, even though computed
    # Uncheck also resets the path/siblings/computed fields outright (mirrors
    # widgets.FieldSelector) — a stale pick must not survive to the next check.
    assert sel._path_ed.text().strip() == ""
    assert sel._sibling_paths == {}
    assert sel._fields == {}

    sel.setChecked(True)
    assert sel._path_ed.text().strip() == ""  # no Data-path provider set here — nothing to prefill


def test_hydra_field_selector_enter_on_a_missing_path_warns(app, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: calls.append(a))

    sel = HydraFieldSelector("Dark", default_dataset="exchange/data")
    missing = tmp_path / "ge1" / "dark.ge1.h5"
    sel._path_ed.setText(str(missing))
    sel._path_ed.returnPressed.emit()
    assert len(calls) == 1
    assert str(missing) in calls[0][-1]


def test_hydra_loader_panel_enter_on_a_missing_path_warns(app, monkeypatch, tmp_path):
    from midas_gui.hydra_widgets import HydraLoaderPanel

    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: calls.append(a))

    panel = HydraLoaderPanel()
    missing = tmp_path / "ge1" / "data.ge1.h5"
    panel._path_ed.setText(str(missing))
    panel._path_ed.returnPressed.emit()
    assert len(calls) == 1
    assert str(missing) in calls[0][-1]


def test_hydra_loader_panel_browse_starts_at_the_typed_directory(app, monkeypatch, tmp_path):
    import midas_gui.hydra_widgets as HW

    seen = {}

    class _FakeDialog:
        def __init__(self, parent=None, *, title="", modes=(), start_dir=""):
            seen["start_dir"] = start_dir
        def exec_(self):
            return QtWidgets.QDialog.Rejected
    monkeypatch.setattr(HW, "BrowseFilesDialog", _FakeDialog)

    panel = HW.HydraLoaderPanel()
    panel._path_ed.setText(str(tmp_path))
    panel._open_browse_dialog()
    assert seen["start_dir"] == str(tmp_path)


def test_mode_ribbon_switches_pages(app):
    tab = DataViewerTab()
    assert tab._mode_ribbon.mode() == "single"
    assert tab._mode_stack.currentWidget() is tab._hsplit

    tab._mode_ribbon.set_mode("hydra")
    assert tab._mode_stack.currentWidget() is tab._hydra_page

    tab._mode_ribbon.set_mode("single")
    assert tab._mode_stack.currentWidget() is tab._hsplit


def test_hydra_page_loads_siblings_and_shows_panels(app, fixture_available):
    tab = DataViewerTab()
    tab._mode_ribbon.set_mode("hydra")
    hp = tab._hydra_page

    hp._loader.set_path(str(fixture_available / "ge1" / "panel.ge1.h5"))
    app.processEvents()
    assert sorted(hp._loader.siblings().keys()) == [1, 2, 3, 4]
    assert hp._toolbar.current() == "ge1"
    for n in (1, 2, 3, 4):
        assert hp._toolbar._buttons[f"ge{n}"].isEnabled()
    assert hp._toolbar._buttons["composite"].isEnabled()

    for key in ("ge2", "ge3", "ge4", "composite", "ge1"):
        hp._toolbar.set_current(key)
        app.processEvents()
        assert hp._toolbar.current() == key
        assert hp._active_card is hp._cards[key]


def test_hydra_pick_ring_points_dont_leak_across_panels(app, fixture_available):
    """Pick BC/Pick Ring state (PickableImageViewer._ring_pts etc.) lives on
    the single ROIImageViewer shared by all 5 panel cards, not per-panel.
    Switching the active panel must clear any in-progress picks so points
    clicked while ge1 was active can never end up mixed into ge2's circle
    fit (regression for cross-panel point leakage)."""
    tab = DataViewerTab()
    tab._mode_ribbon.set_mode("hydra")
    hp = tab._hydra_page
    hp._loader.set_path(str(fixture_available / "ge1" / "panel.ge1.h5"))
    app.processEvents()

    viewer = hp._viewer
    assert hp._toolbar.current() == "ge1"
    viewer._add_ring_point(10.0, 10.0)
    viewer._add_ring_point(20.0, 10.0)
    assert len(viewer._ring_pts) == 2

    hp._toolbar.set_current("ge2")
    app.processEvents()
    assert viewer._ring_pts == []   # ge1's partial pick doesn't survive the switch

    viewer._add_ring_point(5.0, 5.0)
    viewer._add_ring_point(5.0, 15.0)
    viewer._add_ring_point(15.0, 5.0)
    assert len(viewer._ring_pts) == 3   # only ge2's own points, none from ge1


def test_hydra_composite_builds_with_matched_calibration(app, fixture_available):
    """Also covers two geometry-edit regression guards on this same
    DataViewerTab/Hydra page, rather than building a fresh one each,  to
    avoid piling up extra pyqtgraph ViewBox/ImageView instances in this
    process (see the 2026-08-23 pyqtgraph-teardown-crash DECISIONS.md
    entry on keeping the count of Hydra-page instances built across this
    file down):

    1. A beam-centre edit on a panel that already has a full geometry
       loaded (every bundled default does, since tx != 0) must move that
       panel's radial profile, not just its ring overlay — see the
       _effective_calib_geom live-value-override fix in
       hydra_geometry_card.py.
    2. λ/max2θ/px edited on one ge card mirror onto ge1-4 and the Composite
       card — see DetectorGeometryCard.get_shared_fields/apply_shared_fields
       and HydraViewerPage._sync_shared_fields.
    """
    tab = DataViewerTab()
    tab._mode_ribbon.set_mode("hydra")
    hp = tab._hydra_page
    hp._loader.set_path(str(fixture_available / "ge1" / "panel.ge1.h5"))
    app.processEvents()

    for n in (1, 2, 3, 4):
        fields = geometry_fields_from_file(str(fixture_available / f"ge{n}" / f"ps_ge{n}.txt"))
        hp._cards[f"ge{n}"].set_geometry(fields)
    app.processEvents()

    hp._toolbar.set_current("composite")
    app.processEvents()
    assert hp._composite_img is not None
    # The canvas is 512 px across; the image is built every
    # COMPOSITE_DISPLAY_STEP-th pixel of it, so assert the relationship
    # rather than a literal that silently stops meaning anything if the
    # display step changes.
    from midas_gui.hydra_page import COMPOSITE_DISPLAY_STEP as STEP
    built = (512 + STEP - 1) // STEP
    assert hp._composite_img.shape == (built, built)
    # composite card auto-seeded at the centre of the image actually built
    assert hp._cards["composite"]._bcy.value() == pytest.approx(built / 2.0)
    assert hp._cards["composite"]._bcz.value() == pytest.approx(built / 2.0)

    # (1) beam-centre edit -> radial profile
    card = hp._cards["ge1"]
    assert card._calib_geom is not None   # full geometry (tx != 0) — the bug's precondition
    before = hp._profile_view.get_native("ge1")
    assert before is not None
    r_before, prof_before = np.asarray(before[0]), np.asarray(before[1])

    card._bc_auto.setChecked(False)
    card._bcy.setValue(card._bcy.value() + 50.0)
    app.processEvents()

    after = hp._profile_view.get_native("ge1")
    r_after, prof_after = np.asarray(after[0]), np.asarray(after[1])
    assert r_before.shape != r_after.shape or not np.allclose(
        np.nan_to_num(prof_before), np.nan_to_num(prof_after))

    # (2) shared λ/max2θ/px sync across ge1-4 + composite
    src = hp._cards["ge2"]
    new_wl = src._wl.value() + 0.01
    new_px = src._px.value() + 5.0
    new_max2t = src._max2t.value() + 2.0
    src._wl.setValue(new_wl)
    src._px.setValue(new_px)
    src._max2t.setValue(new_max2t)
    app.processEvents()

    for key in ("ge1", "ge3", "ge4", "composite"):
        c = hp._cards[key]
        assert c._wl.value() == pytest.approx(new_wl)
        assert c._px.value() == pytest.approx(new_px)
        assert c._max2t.value() == pytest.approx(new_max2t)


def test_hydra_profile_plot_shows_all_four_curves_plus_composite(app, fixture_available):
    tab = DataViewerTab()
    tab._mode_ribbon.set_mode("hydra")
    hp = tab._hydra_page
    hp._loader.set_path(str(fixture_available / "ge1" / "panel.ge1.h5"))
    app.processEvents()
    for n in (1, 2, 3, 4):
        fields = geometry_fields_from_file(str(fixture_available / f"ge{n}" / f"ps_ge{n}.txt"))
        hp._cards[f"ge{n}"].set_geometry(fields)
    app.processEvents()

    pv = hp._profile_view
    for n in (1, 2, 3, 4):
        assert pv.get_native(f"ge{n}") is not None, f"ge{n} curve missing"
    composite = pv.get_native("composite")
    assert composite is not None
    r_ref, summed, lsd, px, wl = composite
    assert len(r_ref) == len(summed) == 500
    finite = summed[~__import__("numpy").isnan(summed)]
    assert finite.size > 0 and finite.max() > 0

    # hiding Composite clears it; re-showing recomputes it
    pv._checks["composite"].setChecked(False)
    app.processEvents()
    assert pv.get_native("composite") is None
    pv._checks["composite"].setChecked(True)
    app.processEvents()
    assert pv.get_native("composite") is not None


def test_hydra_integrate_button_refreshes_all_curves(app, fixture_available):
    tab = DataViewerTab()
    tab._mode_ribbon.set_mode("hydra")
    hp = tab._hydra_page
    hp._loader.set_path(str(fixture_available / "ge1" / "panel.ge1.h5"))
    app.processEvents()
    for n in (1, 2, 3, 4):
        fields = geometry_fields_from_file(str(fixture_available / f"ge{n}" / f"ps_ge{n}.txt"))
        hp._cards[f"ge{n}"].set_geometry(fields)
    app.processEvents()

    hp._rad_auto.setChecked(False)
    for n in (1, 2, 3, 4):
        hp._profile_view.clear_curve(f"ge{n}")
    hp._profile_view.clear_curve("composite")
    assert hp._profile_view.get_native("ge1") is None

    hp._rad_btn.click()
    app.processEvents()
    for n in (1, 2, 3, 4):
        assert hp._profile_view.get_native(f"ge{n}") is not None
    assert hp._profile_view.get_native("composite") is not None


def test_hydra_state_round_trips(app, fixture_available):
    tab = DataViewerTab()
    tab._mode_ribbon.set_mode("hydra")
    hp = tab._hydra_page
    hp._loader.set_path(str(fixture_available / "ge1" / "panel.ge1.h5"))
    app.processEvents()
    hp._toolbar.set_current("ge3")
    app.processEvents()

    state = tab.get_state()
    assert state["hydra"]["active_mode"] == "hydra"
    assert state["hydra"]["page"]["active_panel"] == "ge3"

    tab2 = DataViewerTab()
    tab2.set_state(state)
    app.processEvents()
    assert tab2._mode_ribbon.mode() == "hydra"
    assert tab2._hydra_page._toolbar.current() == "ge3"


