"""Characterization of ``BatchTab._suggest_output_dir`` as it behaves today.

Written *before* the shared-helper refactor (``helpers.suggest_expid_bc_dir``)
so that refactor is provably behavior-preserving. Every assertion here records
what the positional derivation already produced, including the case it gets
wrong — see ``test_a_file_directly_inside_a_bc_dir_misreads_the_expid``. That
one is not aspirational; it is the bug that motivated the Calibrate tab's
different resolution order, pinned here so Batch's own behavior can't drift
without someone noticing.

Qt import discipline per STATE.md: import inside fixtures, never at module
scope.
"""
import pytest
from pathlib import Path

pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(qapp):
    from midas_gui.tab_batch import BatchTab
    return BatchTab()


def _with_source(tab, path, expid=""):
    """Point the tab at `path` without touching the filesystem.

    ``_suggest_output_dir`` is pure path arithmetic — it never stats anything —
    so a source that doesn't exist exercises it exactly as a real one would,
    and lets these tests assert against the real beamline paths below without
    needing /net mounted.
    """
    tab._loader.source_cfg = lambda: {"path": str(path)}
    tab._expid_provider = (lambda: expid) if expid else None
    return tab


# ── The canonical mpe_wf layout: <outroot>/<expid>/<detector>/<froot>/<files> ──

def test_canonical_four_deep_layout(tab):
    _with_source(tab, "/net/s20iddata/export/park_may26/ge3/sam1/sam1_000001.tif")
    assert tab._suggest_output_dir() == Path(
        "/net/s20iddata/export/park_may26_bc/sam1/ge3")


def test_frame_number_is_stripped_to_the_froot(tab):
    """The suggestion keys off ``froot``, not the individual frame, so every
    frame of a scan lands in one folder."""
    a = _with_source(tab, "/r/exp/ge3/sam1/sam1_000001.tif")._suggest_output_dir()
    b = _with_source(tab, "/r/exp/ge3/sam1/sam1_009243.tif")._suggest_output_dir()
    assert a == b == Path("/r/exp_bc/sam1/ge3")


def test_a_glob_source_uses_the_literal_prefix(tab):
    _with_source(tab, "/net/data/export/park_may26/ge3/sam1/sam1_*.tif")
    assert tab._suggest_output_dir() == Path(
        "/net/data/export/park_may26_bc/sam1/ge3")


def test_a_glob_with_no_stem_left_after_trimming(tab):
    """``rstrip("_-.")`` can eat the whole prefix; the code falls back to the
    untrimmed name rather than producing an empty path segment."""
    _with_source(tab, "/a/b/c/d/*.tif")
    out = tab._suggest_output_dir()
    assert out is not None and "" not in out.parts


def test_paths_list_is_used_when_there_is_no_single_path(tab):
    """Several files in ONE folder still derive that folder's dir."""
    tab._loader.source_cfg = lambda: {
        "paths": ["/net/d/export/e/ge2/s1/s1_000001.tif",
                  "/net/d/export/e/ge2/s1/s1_000002.tif"]}
    tab._expid_provider = None
    assert tab._suggest_output_dir() == Path("/net/d/export/e_bc/s1/ge2")


def test_a_selection_spanning_folders_suggests_the_root_they_share(tab):
    """Changed deliberately. This used to return the FIRST path's dir, which
    is the behaviour that sent 23 load steps' output into load_step_0 at
    1-ID-E: the run fans out to one dir per source folder, so naming one of
    them in the Output field is a lie. The shared root is the honest answer,
    and ``group_paths_by_output_dir`` rebases onto it to reproduce exactly
    the same per-folder directories."""
    tab._loader.source_cfg = lambda: {
        "paths": ["/net/d/export/e/ge2/s1/s1_000001.tif",
                  "/net/d/export/e/ge2/s2/s2_000001.tif"]}
    tab._expid_provider = None
    out = tab._suggest_output_dir()
    assert out == Path("/net/d/export/e_bc")
    # and it is the root the per-folder dirs actually sit under
    for d in tab._output_group_dirs():
        assert str(d).startswith(str(out) + "/")


def test_the_fan_out_hint_appears_only_for_a_multi_folder_selection(tab):
    tab._expid_provider = None
    tab._loader.source_cfg = lambda: {
        "paths": ["/net/d/export/e/ge2/s1/s1_000001.tif",
                  "/net/d/export/e/ge2/s1/s1_000002.tif"]}
    tab._update_output_multi_hint()
    assert not tab._out_multi_lbl.isVisible()

    tab._loader.source_cfg = lambda: {
        "paths": ["/net/d/export/e/ge2/s1/s1_000001.tif",
                  "/net/d/export/e/ge2/s2/s2_000001.tif"]}
    tab._update_output_multi_hint()
    assert tab._out_multi_lbl.isVisibleTo(tab)
    assert "2 source folders" in tab._out_multi_lbl.text()
    assert "does NOT" in tab._out_multi_lbl.text()


# ── Too shallow to read expid/detector positionally ──────────────────────────

def test_shallow_source_without_expid_falls_back_to_the_froot_folder(tab):
    """Two levels deep: no detector or expid to find, and the containing
    folder is already named after the froot, so it is used as-is rather than
    nesting ``sam1/sam1``."""
    _with_source(tab, "/data/sam1/sam1_000001.tif")
    assert tab._suggest_output_dir() == Path("/data/sam1")


def test_shallow_source_with_expid_uses_the_header_field(tab):
    _with_source(tab, "/data/sam1/sam1_000001.tif", expid="park_may26")
    assert tab._suggest_output_dir() == Path("/data/sam1/park_may26_bc/sam1")


def test_a_blank_expid_field_counts_as_no_expid(tab):
    _with_source(tab, "/data/sam1/sam1_000001.tif", expid="   ")
    assert tab._suggest_output_dir() == Path("/data/sam1")


def test_no_source_loaded_suggests_nothing(tab):
    tab._loader.source_cfg = lambda: {}
    assert tab._suggest_output_dir() is None


# ── The case the positional read gets wrong ──────────────────────────────────

def test_a_file_directly_inside_a_bc_dir_misreads_the_expid(tab):
    """The real session path that motivated the Calibrate tab's own ladder.

    The .h5 sits *directly inside* an already-``_bc`` directory, only three
    levels below the mount, so the positional read labels ``export`` as the
    expid and proposes ``/net/s20iddata/export_bc`` — a sibling of the mount
    root, which parkjs cannot create. Batch keeps this behavior for now (it is
    correct for the layout Batch is actually pointed at); Calibrate recognizes
    the ``_bc`` directory instead. Pinned so the shared-helper refactor doesn't
    quietly change Batch while fixing Calibrate.
    """
    _with_source(tab, "/net/s20iddata/export/s20a/PUP_AML_stubbins_sep26_bc/"
                      "PUP_AML_stubbins_sep26_beniwal_gui.h5")
    assert tab._suggest_output_dir() == Path(
        "/net/s20iddata/export_bc/PUP_AML_stubbins_sep26_beniwal_gui/s20a")
