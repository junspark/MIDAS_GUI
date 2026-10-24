"""Tests for ``tab_zarrviewer.ZarrViewerTab`` — the ported mpe_wf_saxs_waxs
zarr browser.

Rather than hand-rolling a fake zarr store, these drive a real ``.ave.zarr.zip``
produced by ``BatchWorker`` (same fixture-building pattern as
``test_batch_zarr_output.py``), so the tests exercise the actual production
schema (real ``REtaMap``, real group layout) the tab is built to read.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from PyQt5 import QtWidgets

from midas_gui import constants as C
from midas_gui.tab_zarrviewer import ZarrViewerTab


# ── Fixture: a real .ave.zarr.zip, built once per module ────────────────────

def _tiny_calib_result(**overrides):
    fields = dict(
        Lsd=200000.0, BC_y=32.0, BC_z=32.0, tx=0.0, ty=0.0, tz=0.0,
        distortion={}, pxY=200.0, pxZ=200.0,
        NrPixelsY=64, NrPixelsZ=64, wavelength_A=0.1729,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _make_tiff_frames(tmp_path, n=2, size=64):
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i:04d}.tif"
        tifffile.imwrite(str(p),
                         (rng.random((size, size)) * 100 + 10).astype(np.float32))
        paths.append(str(p))
    return paths


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def zarr_fixture(app, tmp_path_factory):
    """One real .ave.zarr.zip, written by the same BatchWorker/write_gsas_zarr_zip
    path Batch Integrate's "zarr" output format uses."""
    pytest.importorskip("torch")
    pytest.importorskip("midas_integrate_v2")
    pytest.importorskip("zarr")
    import midas_gui.workers as wk
    from midas_gui.helpers import _build_spec

    tmp = tmp_path_factory.mktemp("zarrviewer")
    (tmp / "in").mkdir()
    paths = _make_tiff_frames(tmp / "in", n=1)
    spec = _build_spec(_tiny_calib_result(), r_bin=2.0, eta_bin=45.0)
    out = tmp / "out"
    worker = wk.BatchWorker(
        spec, {"type": "tiff_list", "paths": paths}, None, out, ["zarr"],
        "subpixel2", (None, None), None)
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures, failures[0]

    zips = sorted((out / "zarr").glob("*.zarr.zip"))
    assert len(zips) == 1
    return zips[0]


@pytest.fixture
def tab(app):
    return ZarrViewerTab()


# ── Tab registration ─────────────────────────────────────────────────────

def _shipped_visible_tabs():
    """The code-shipped DEFAULT_VISIBLE_TABS, independent of whatever the
    machine's own active profile has saved to ~/.config/midas_gui/ — a real
    profile's own ``ui.visible_tabs`` (once saved, e.g. from an older
    Preferences ▸ Tabs session that predates a newly-added default-visible
    tab) overrides the live ``constants.DEFAULT_VISIBLE_TABS`` global at
    import time (see ``constants.reload_from_config()``), so asserting
    against that global directly would make this test's outcome depend on
    whichever machine/profile happens to run it. ``shipped_defaults()`` is
    the escape hatch constants.py itself provides for exactly this."""
    return C.shipped_defaults()["ui"]["visible_tabs"]


def test_zarr_viewer_is_visible_by_default_but_still_an_optional_tab():
    """Shown out of the box (constants.DEFAULT_VISIBLE_TABS), same tier as
    Calib. Refinement/Batch Queue/Pump Probe — but still a toggleable
    OPTIONAL_TAB, not one of the four hard-pinned ALWAYS_TABS, so it can
    still be hidden from Preferences ▸ Tabs like any of those."""
    assert "Zarr Viewer" in C.OPTIONAL_TABS
    assert "Zarr Viewer" not in C.ALWAYS_TABS
    assert "Zarr Viewer" in _shipped_visible_tabs()


def test_zarr_viewer_shows_up_with_default_visibility_after_batch_queue(app):
    """Last in the shipped run of tabs, right after Batch Queue — you queue
    the integrations, then you open what they wrote. It sat after Batch
    Integrate until b9fba11 moved it one place later."""
    import midas_gui.app as app_mod
    win = app_mod.MainWindow()
    win.apply_tab_visibility(_shipped_visible_tabs())
    names = [win.centralWidget().tabText(i) for i in range(win.centralWidget().count())]
    assert any(n.endswith("Zarr Viewer") for n in names)
    queue_idx = next(i for i, n in enumerate(names) if n.endswith("Batch Queue"))
    assert names[queue_idx + 1].endswith("Zarr Viewer")


def test_zarr_viewer_can_still_be_hidden_like_any_optional_tab(app):
    import midas_gui.app as app_mod
    win = app_mod.MainWindow()
    win.apply_tab_visibility([t for t in _shipped_visible_tabs() if t != "Zarr Viewer"])
    names = [win.centralWidget().tabText(i) for i in range(win.centralWidget().count())]
    assert not any(n.endswith("Zarr Viewer") for n in names)


# ── Empty state ──────────────────────────────────────────────────────────

def test_tab_builds_with_no_file_loaded(tab):
    assert tab._tree.topLevelItemCount() == 0
    assert tab._cur_data is None
    # The slice spinner is only disabled once an array is selected and turns
    # out not to be 3-D (see _on_tree_click) — matching the source, it's not
    # pre-disabled before anything has been picked.
    assert tab._slice_spin.maximum() == 1
    assert tab._lbl_path.text() == "No file loaded"


# ── Loading a real fixture ────────────────────────────────────────────────

def test_load_file_populates_the_tree_and_caches_retamap(tab, zarr_fixture):
    tab._load_file(str(zarr_fixture))
    assert tab._tree.topLevelItemCount() == 1  # single root item
    assert tab._root is not None
    assert tab._retamap is not None
    assert tab._retamap.shape[0] == 5  # Radius,2Theta,Eta,BinArea,Q
    assert tab._lam is not None and tab._lam > 0

    root_item = tab._tree.topLevelItem(0)
    child_names = {root_item.child(i).text(0).split("  ")[0]
                  for i in range(root_item.childCount())}
    # Real write_gsas_zarr_zip output — see the fixture's own group layout.
    assert {"REtaMap", "InstrumentParameters", "Omegas"} <= child_names


def test_selecting_retamap_populates_slice_selector(tab, zarr_fixture):
    tab._load_file(str(zarr_fixture))
    arr = tab._root["REtaMap"]
    tab._cur_path = "REtaMap"
    tab._cur_data = arr[...]
    # Mirror what _on_tree_click does for a 3-D array, without needing a
    # real QTreeWidgetItem click.
    n = tab._cur_data.shape[0]
    tab._slice_spin.setMaximum(max(n, 1))
    tab._slice_spin.setEnabled(True)
    tab._refresh_plot()
    assert tab._slice_spin.isEnabled()
    assert tab._slice_spin.maximum() == 5


def test_get_integration_axes_all_four_units_against_real_retamap(tab, zarr_fixture):
    tab._load_file(str(zarr_fixture))
    sum_frames = tab._root["SumFrames"]
    shape = sum_frames.shape
    assert len(shape) == 2

    for label in ("R bin", "2θ (deg)", "Q (Å⁻¹)", "d (Å)"):
        tab._xaxis_cb.setCurrentText(label)
        x_coords, xlabel, eta_coords = tab._get_integration_axes(shape)
        assert eta_coords is not None and len(eta_coords) == shape[1]
        if label == "R bin":
            assert x_coords is None
        else:
            assert x_coords is not None and len(x_coords) == shape[0]
            assert xlabel == label


def test_dispatch_plot_handles_a_real_2d_integration_array(tab, zarr_fixture):
    """Smoke test: selecting SumFrames and dispatching must not raise, for
    every X-axis unit and both 2-D-map / 1-D-lines plot modes."""
    tab._load_file(str(zarr_fixture))
    tab._cur_path = "SumFrames"
    tab._cur_data = tab._root["SumFrames"][...]
    for mode in ("2-D map", "1-D lines"):
        tab._plot_mode_cb.setCurrentText(mode)
        for label in ("R bin", "2θ (deg)", "Q (Å⁻¹)", "d (Å)"):
            tab._xaxis_cb.setCurrentText(label)
            tab._refresh_plot()  # must not raise


# ── Metadata / attributes panel ──────────────────────────────────────────

def test_metadata_panel_populates_for_root_group_and_array(tab, zarr_fixture):
    tab._load_file(str(zarr_fixture))

    tab._show_metadata(path="/", node=tab._root, role="root")
    assert "Root group" in tab._attr_header.text()
    # provenance_history is a root attr — see midas_gui/provenance.py.
    assert tab._attr_tree.topLevelItemCount() >= 1

    grp = tab._root["InstrumentParameters"]
    tab._show_metadata(path="InstrumentParameters", node=grp, role="group")
    assert "Group" in tab._attr_header.text()

    arr = tab._root["REtaMap"]
    tab._show_metadata(path="REtaMap", node=arr, role="array")
    assert "Array" in tab._attr_header.text()
    assert "REtaMap" in tab._attr_header.text()
    # REtaMap's own attrs (Header/Units/nRBins/nEtaBins) should show up.
    assert "no attributes" not in tab._attr_text.toPlainText()


# ── Azimuth-bin parsing ───────────────────────────────────────────────────

@pytest.mark.parametrize("text,n_eta,expected", [
    ("", 8, list(range(8))),
    ("all", 8, list(range(8))),
    ("ALL", 8, list(range(8))),
    ("0,2,5", 8, [0, 2, 5]),
    ("0, 2, 5", 8, [0, 2, 5]),
    ("5,0,2", 8, [0, 2, 5]),          # sorted
    ("0,0,2", 8, [0, 2]),             # de-duplicated
    ("7,8,9", 8, [7]),                # out-of-range indices dropped
    ("banana", 8, list(range(8))),    # unparseable -> falls back to all
])
def test_parse_eta_bins(tab, text, n_eta, expected):
    tab._eta_edit.setText(text)
    assert tab._parse_eta_bins(n_eta) == expected


# ── Clim-lock / zoom-lock toggles ─────────────────────────────────────────

def test_clim_lock_prefills_percentiles_from_current_array(tab, zarr_fixture):
    tab._load_file(str(zarr_fixture))
    tab._cur_path = "SumFrames"
    tab._cur_data = tab._root["SumFrames"][...]
    assert tab._clim_lo.text() == "" and tab._clim_hi.text() == ""
    tab._clim_lock.setChecked(True)
    assert tab._clim_lo.text() != ""
    assert tab._clim_hi.text() != ""


def test_zoom_lock_captures_and_clears_the_view(tab, zarr_fixture):
    tab._load_file(str(zarr_fixture))
    tab._cur_path = "SumFrames"
    tab._cur_data = tab._root["SumFrames"][...]
    tab._refresh_plot()
    tab._zoom_lock.setChecked(True)
    assert tab._zoom_lock_xlim is not None
    tab._zoom_lock.setChecked(False)
    assert tab._zoom_lock_xlim is None
    assert tab._zoom_lock_ylim is None


# ── eta rows are drawn at their real height ─────────────────────────────

def test_sparse_eta_rows_are_not_stretched_to_fill_the_gaps():
    """Found on a real four-panel run at 1-ID-E.

    pcolormesh treats coordinates as cell CENTRES and infers edges from
    the spacing. Four panels caked 10 deg wide but centred 90 deg apart
    were therefore drawn as four 90-deg bands: the eta axis ran -225 to
    +135, and the picture claimed nine times the azimuthal coverage
    actually collected. An axis reading -220 deg on a detector spanning
    +-180 is how it was noticed.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from midas_gui.tab_zarrviewer import _draw_eta_bands

    eta = np.array([-180.0, -90.0, 0.0, 90.0])
    x = np.linspace(2.0, 10.0, 64)
    z = np.random.default_rng(0).random((4, 64))
    norm = mcolors.Normalize(0, 1)

    fig = plt.figure(); ax = fig.add_subplot(111)
    _draw_eta_bands(ax, x, eta, z, None, cmap="viridis", norm=norm)
    lo, hi = ax.get_ylim()
    assert lo < -200, "the old behaviour should still infer 90-deg cells"
    plt.close(fig)

    fig = plt.figure(); ax = fig.add_subplot(111)
    _draw_eta_bands(ax, x, eta, z, 10.0, cmap="viridis", norm=norm)
    lo, hi = ax.get_ylim()
    assert lo > -200 and hi < 120, f"eta axis {lo}..{hi} still stretched"
    assert lo < -185 and hi > 95, "the bands themselves must stay visible"
    plt.close(fig)


def test_a_contiguous_caking_still_takes_the_single_quadmesh_path():
    """35 rows 10 deg apart and 10 deg wide ARE contiguous; drawing them
    one at a time would be correct but much slower, so the fast path has
    to survive."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from midas_gui.tab_zarrviewer import _draw_eta_bands

    eta = np.arange(-175.0, 176.0, 10.0)
    x = np.linspace(2.0, 10.0, 32)
    z = np.zeros((eta.size, 32))
    fig = plt.figure(); ax = fig.add_subplot(111)
    _draw_eta_bands(ax, x, eta, z, 10.0, cmap="viridis",
                    norm=mcolors.Normalize(0, 1))
    assert len(ax.collections) == 1, "contiguous rows were drawn one by one"
    plt.close(fig)
