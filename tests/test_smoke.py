"""Smoke tests for the midas-gui package.

The version-consistency checks always run (no GUI / MIDAS backend needed). The
build test constructs the full window offscreen and is skipped gracefully when
PyQt5 or the MIDAS analysis backends are not installed in the environment.
"""
import pathlib
import re

import pytest

import midas_gui

# Several tests below build a full offscreen MainWindow (pyqtgraph-heavy);
# running them in one process lets teardown corruption accumulate across
# tests and crash the interpreter (see .context/STATE.md's
# interpreter-teardown crash-risk note). pytest-forked runs each test in
# this file in its own subprocess, so a crash is reported as a normal
# FAILED with signal info instead of aborting the whole pytest run.
pytestmark = pytest.mark.forked


def test_version_is_nonempty_string():
    assert isinstance(midas_gui.__version__, str) and midas_gui.__version__


def test_version_matches_pyproject():
    """__version__ in the package must match pyproject.toml (release.sh keeps them in sync)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text()
    m = re.search(r'^version = "([^"]+)"', text, re.M)
    assert m is not None, "version not found in pyproject.toml"
    assert m.group(1) == midas_gui.__version__


def test_app_builds_offscreen():
    """MainWindow constructs headless with the always-on + default-visible tabs."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    try:
        import midas_gui.app as app_mod
        import midas_gui.constants as C
    except Exception as exc:  # MIDAS backends absent → nothing to test here
        pytest.skip(f"midas_gui.app needs the full MIDAS stack: {exc}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = app_mod.MainWindow()
    # apply_tab_visibility shows a tab when "always or name in visible", so
    # the count is the UNION of the pinned tabs and the visible optional
    # ones -- not the sum. Summing double-counted, because
    # DEFAULT_VISIBLE_TABS is not reliably "optional tabs only": a profile
    # that has ever saved Preferences > Tabs writes its whole visible set,
    # pinned tabs included, and constants.reload_from_config() replaces the
    # global with that list wholesale. On this machine it holds all 9, so
    # the old assertion read 9 == 4 + 9 and failed for a reason that had
    # nothing to do with the window.
    #
    # Intersecting with OPTIONAL_TABS makes the arithmetic right either way
    # -- shipped default (4 optional) or a saved config that repeats the
    # pinned ones -- so this no longer depends on whose machine runs it.
    expected = set(C.ALWAYS_TABS) | (set(C.OPTIONAL_TABS) & set(C.DEFAULT_VISIBLE_TABS))
    assert win.centralWidget().count() == len(expected)
    labels = {win.centralWidget().tabText(i).split("  ", 1)[-1]
              for i in range(win.centralWidget().count())}
    assert labels == expected


def test_tab_visibility_toggle():
    """Hiding every optional tab leaves exactly the four always-on tabs; showing all
    optional tabs brings the full set. All tabs stay constructed (live apply)."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    try:
        import midas_gui.app as app_mod
        import midas_gui.constants as C
    except Exception as exc:
        pytest.skip(f"midas_gui.app needs the full MIDAS stack: {exc}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = app_mod.MainWindow()
    win.apply_tab_visibility([])
    assert win.centralWidget().count() == len(C.ALWAYS_TABS)
    win.apply_tab_visibility(C.OPTIONAL_TABS)
    assert win.centralWidget().count() == len(C.ALWAYS_TABS) + len(C.OPTIONAL_TABS)

    # Header project-name indicator (top-right corner of the tab bar,
    # separate from the muted status-bar "Project: none" label) — reused
    # from this same window rather than constructing another MainWindow,
    # since each one adds several pyqtgraph widgets to the same process
    # (see .context/STATE.md's interpreter-teardown crash-risk note).
    assert win._header_project_lbl.text() == ""
    win._set_project_path("/tmp/my_experiment.h5")
    assert "my_experiment.h5" in win._header_project_lbl.text()
    assert win._project_lbl.text() == "Project: my_experiment.h5"
    win._set_project_path(None)
    assert win._header_project_lbl.text() == ""
    assert win._project_lbl.text() == "Project: none"


def test_on_profile_changed_refreshes_devices_and_calibrants(monkeypatch):
    """A profile switch (MainWindow.on_profile_changed) must repopulate the
    Data Viewer's Live PV dropdown and the Calibrate tab's Calibrant dropdown
    (single-detector + Hydra) live, not only at tab-construction time — the
    reported "Live View devices don't update on profile change" bug, plus
    the same fix applied to the other profile-scoped option list."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    try:
        import midas_gui.app as app_mod
        import midas_gui.constants as C
    except Exception as exc:
        pytest.skip(f"midas_gui.app needs the full MIDAS stack: {exc}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = app_mod.MainWindow()

    import midas_gui.widgets as W
    new_devices = [{"name": "newDevice", "prefix": "new:", "pva_suffix": "Pva1:Image"}]
    monkeypatch.setattr(W, "DEVICES", new_devices)
    monkeypatch.setattr(C, "CALIBRANTS", ["OnlyThisCalibrant"])
    import midas_gui.tab_calibrate as tab_calibrate_mod
    import midas_gui.hydra_calib_page as hydra_calib_page_mod
    monkeypatch.setattr(tab_calibrate_mod, "CALIBRANTS", C.CALIBRANTS)
    monkeypatch.setattr(hydra_calib_page_mod, "CALIBRANTS", C.CALIBRANTS)

    win.on_profile_changed()

    pv_combo = win._view_tab._loader._pv_ed
    assert [pv_combo.itemText(i) for i in range(pv_combo.count())] == ["newDevice"]
    # The single-detector dropdown is the unified list built by
    # constants.calibrant_combo_items(): the profile's crystalline CALIBRANTS
    # first, then the non-crystalline d-spacing entries. Those come from
    # MATERIALS, not CALIBRANTS, so patching CALIBRANTS alone no longer empties
    # the combo — the profile still fully controls the crystalline half, which
    # is what this test is about.
    assert [win._cal_tab._cal.itemText(i) for i in range(win._cal_tab._cal.count())] == \
        ["OnlyThisCalibrant", "AgBH (silver behenate)", "Custom d-spacings…"]
    # Hydra has no d-spacing support yet (see calibration_unification_plan.md),
    # so its combo is still exactly the profile's crystalline list.
    hydra_cal = win._cal_tab._hydra_page._cal
    assert [hydra_cal.itemText(i) for i in range(hydra_cal.count())] == ["OnlyThisCalibrant"]


def _make_ge_h5(path, energy_kev=80.61):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("exchange/data", data=np.zeros((1, 4, 4), dtype="uint16"))
        f.create_dataset("instrument/HEM/Energy", data=[energy_kev])


def test_data_viewer_autodetects_geometry_from_ge_file(tmp_path, monkeypatch):
    """Loading a .ge1-tagged HDF5 file into the Data Viewer under a beamline
    profile auto-populates the geometry card's pixel size + wavelength (see
    helpers.detect_geometry_from_path) — the wired-up end of the feature, not
    just the pure detection logic already covered in test_helpers.py."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    try:
        import midas_gui.app as app_mod
        import midas_gui.constants as C
    except Exception as exc:
        pytest.skip(f"midas_gui.app needs the full MIDAS stack: {exc}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = app_mod.MainWindow()

    import midas_gui.settings as settings_mod
    monkeypatch.setattr(settings_mod, "active_profile", lambda: "1-ID-E")

    path = tmp_path / "dark_scan_002030.ge1.h5"
    _make_ge_h5(path, energy_kev=80.61)
    win._view_tab._loader.set_path(str(path))

    g = win._view_tab.get_geometry()
    assert g["pxY"] == 200.0
    # The wavelength spin box only keeps 4 decimals — compare with matching tolerance.
    assert g["wavelength_A"] == pytest.approx(C.HC_KEV_A / 80.61, abs=1e-4)


def test_calibrate_autodetects_geometry_from_ge_file(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    try:
        import midas_gui.app as app_mod
        import midas_gui.constants as C
    except Exception as exc:
        pytest.skip(f"midas_gui.app needs the full MIDAS stack: {exc}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = app_mod.MainWindow()

    import midas_gui.settings as settings_mod
    monkeypatch.setattr(settings_mod, "active_profile", lambda: "20-ID-D")

    path = tmp_path / "scan_001.vrx.h5"
    _make_ge_h5(path, energy_kev=50.0)
    win._cal_tab._loader.set_path(str(path))

    assert win._cal_tab._pxY.value() == 150.0
    assert win._cal_tab._wl.value() == pytest.approx(C.HC_KEV_A / 50.0, abs=1e-5)


def test_trr_filename_parser():
    """TRR filenames parse to (fshw, delay, id) with the delay sign flipped."""
    from midas_gui.tab_pumpprobe import parse_trr_filename
    fshw, delay, fid = parse_trr_filename(
        "Ex01_Sa01_Sc17-28.0fshw-2e-09delay211898.tif", "Ex01_Sa01_Sc17")
    assert fshw == -28.0 and delay == 2e-09 and fid == 211898
    assert parse_trr_filename("not_a_trr_frame.tif", "Ex01") is None


def test_colormap_resolves_without_matplotlib():
    """`_resolve_cmap` must never return None even when matplotlib (which supplies
    'hot'/'viridis'/… to pyqtgraph) is unavailable — passing None into pyqtgraph is
    what crashed viewer construction on fresh Linux/Windows envs. A valid ColorMap
    here is exactly what keeps ImageViewer/WaterfallViewer building."""
    pytest.importorskip("PyQt5.QtWidgets")
    try:
        import pyqtgraph as pg
        import midas_gui.widgets as W
    except Exception as exc:
        pytest.skip(f"GUI stack unavailable: {exc}")
    orig_get, orig_mpl = pg.colormap.get, pg.colormap.getFromMatplotlib
    try:
        # Simulate no matplotlib: only pyqtgraph-native (CET-*) names resolve.
        pg.colormap.get = lambda name, source=None: (
            orig_get(name) if (source is None and str(name).startswith("CET-")) else None)
        def _no_mpl(name):
            raise ImportError("no matplotlib")
        pg.colormap.getFromMatplotlib = _no_mpl
        cm = W._resolve_cmap("hot")            # 'hot' unavailable without matplotlib
        assert cm is not None
        assert cm.getLookupTable(0.0, 1.0, 8) is not None   # usable ColorMap
    finally:
        pg.colormap.get, pg.colormap.getFromMatplotlib = orig_get, orig_mpl


def test_pumpprobe_grouping():
    """Repeats average per delay and the reference (negative delays) subtracts to ΔI."""
    import numpy as np
    from midas_gui.workers import PumpProbeWorker
    profiles = np.array([[1., 1., 1.], [1.1, 1.1, 1.1],       # delay -1 (reference)
                         [2., 3., 4.], [2.2, 3.2, 4.2]])       # delay +1
    delays = np.array([-1., -1., 1., 1.])
    res = PumpProbeWorker._group_and_difference(profiles, delays, None)
    assert res["delays"] == [-1.0, 1.0]
    assert np.allclose(res["reference"], [1.05, 1.05, 1.05])
    assert np.allclose(res["dI"][1], [1.05, 2.05, 3.05])


def test_optional_tabs_are_closable_and_pinned_ones_are_not():
    """Every optional tab carries an ✕; the four ALWAYS_TABS have theirs
    stripped, so the set of tabs you can close is exactly the set Preferences ▸
    Tabs lets you uncheck."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    try:
        import midas_gui.app as app_mod
        import midas_gui.constants as C
    except Exception as exc:
        pytest.skip(f"midas_gui.app needs the full MIDAS stack: {exc}")
    # Held in a local: a bare expression lets the QApplication be collected
    # mid-test, and MainWindow() then aborts the interpreter.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    win = app_mod.MainWindow()
    win.apply_tab_visibility(C.OPTIONAL_TABS)

    tabs = win.centralWidget()
    bar = tabs.tabBar()
    assert tabs.tabsClosable()
    pinned_names = set(C.ALWAYS_TABS)
    for i in range(tabs.count()):
        name = tabs.tabText(i).split("  ", 1)[-1]
        has_x = any(bar.tabButton(i, side) is not None
                    for side in (QtWidgets.QTabBar.RightSide,
                                 QtWidgets.QTabBar.LeftSide))
        assert has_x is (name not in pinned_names), name


def test_closing_an_optional_tab_hides_it_and_persists_the_rest(monkeypatch):
    """Clicking ✕ drops that one tab, keeps every other visible tab, and writes
    the reduced set to ui.visible_tabs — the same key Preferences writes, so the
    two can't disagree. The save is intercepted: this must never touch the
    machine's real ~/.config/midas_gui profile."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    try:
        import midas_gui.app as app_mod
        import midas_gui.constants as C
        from midas_gui import settings
    except Exception as exc:
        pytest.skip(f"midas_gui.app needs the full MIDAS stack: {exc}")
    # Held in a local: a bare expression lets the QApplication be collected
    # mid-test, and MainWindow() then aborts the interpreter.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    saved = {}
    monkeypatch.setattr(settings, "load_config",
                        lambda **kw: {"paths": {"outroot": "/tmp/keepme"}})
    monkeypatch.setattr(settings, "save_user_config", lambda cfg: saved.update(cfg))
    monkeypatch.setattr(C, "reload_from_config", lambda: None)

    win = app_mod.MainWindow()
    win.apply_tab_visibility(C.OPTIONAL_TABS)
    tabs = win.centralWidget()
    victim = "Zarr Viewer"
    idx = next(i for i in range(tabs.count()) if tabs.tabText(i).endswith(victim))
    win._close_optional_tab(idx)

    names = [tabs.tabText(i).split("  ", 1)[-1] for i in range(tabs.count())]
    assert victim not in names
    assert set(C.ALWAYS_TABS) <= set(names)
    assert set(names) == set(C.ALWAYS_TABS) | (set(C.OPTIONAL_TABS) - {victim})

    assert saved["ui"]["visible_tabs"] == [n for n in C.OPTIONAL_TABS if n != victim]
    # Read-modify-write, not replace: an unrelated key already in the overlay
    # must survive being written back.
    assert saved["paths"]["outroot"] == "/tmp/keepme"


def test_a_pinned_tab_cannot_be_closed_even_if_the_signal_reaches_it(monkeypatch):
    """The stripped ✕ is cosmetic and per-index; the guard in
    _close_optional_tab is what actually protects the four pinned tabs."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    try:
        import midas_gui.app as app_mod
        import midas_gui.constants as C
        from midas_gui import settings
    except Exception as exc:
        pytest.skip(f"midas_gui.app needs the full MIDAS stack: {exc}")
    # Held in a local: a bare expression lets the QApplication be collected
    # mid-test, and MainWindow() then aborts the interpreter.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    monkeypatch.setattr(settings, "save_user_config",
                        lambda cfg: pytest.fail("a pinned tab must not persist anything"))

    win = app_mod.MainWindow()
    win.apply_tab_visibility(C.OPTIONAL_TABS)
    tabs = win.centralWidget()
    before = tabs.count()
    idx = next(i for i in range(tabs.count())
               if tabs.tabText(i).endswith("Batch Integrate"))
    win._close_optional_tab(idx)
    assert tabs.count() == before
