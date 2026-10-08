"""Tests for Batch Integrate's Browse… parity: DataLoaderPanel(mode="stream")'s
filestem-filter/explicit-multi-file source_cfg(), the workers.py source that
backs an explicit file pick (`_ExplicitTIFFSource`), and Batch Parallel's
frame-index resolution / chunk-splitting / worker-count math and the
write_all_profiles() Save-button helper (workers.BatchRunCoordinator).

Mostly dependency-free (PyQt5 + numpy + tifffile); the two write_all_profiles
tests additionally need midas_integrate_v2 and skip if it's unavailable.
"""
import numpy as np
import pytest


def _make_app_and_module():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    import midas_gui.widgets as W
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return W, app


def test_stem_filter_becomes_glob_pattern_in_source_cfg():
    # "**" makes the glob recursive — "Files sharing a stem" is meant to
    # find every scan-point file below the selected folder, in any
    # subfolder, not just its direct children.
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="stream")
    panel._set_stem_filter("/tmp/x", "scan_")
    assert panel.source_cfg() == {"type": "tiff_glob", "path": "/tmp/x/**/scan_*"}


def test_explicit_multi_file_pick_becomes_tiff_list():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="stream")
    paths = ["/tmp/x/a.tif", "/tmp/x/b.tif"]
    panel._set_explicit_paths(paths)
    assert panel.source_cfg() == {"type": "tiff_list", "paths": paths}


def test_plain_folder_is_still_tiff_glob():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="stream")
    panel._path_ed.setText("/tmp/x")
    assert panel.source_cfg() == {"type": "tiff_glob", "path": "/tmp/x"}


def test_checking_dark_prefills_from_data_path():
    # Checking Dark/Bright/Background should default to the same file as
    # Data, since that's the common case — a dark/bright/background frame
    # living alongside the data itself.
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._path_ed.setText("/tmp/x/data.h5")
    panel._dark_sel.setChecked(True)
    assert panel._dark_sel._path_ed.text().strip() == "/tmp/x/data.h5"


def test_checking_dark_does_not_override_an_existing_path():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._path_ed.setText("/tmp/x/data.h5")
    panel._dark_sel._path_ed.setText("/tmp/x/dark.h5")
    panel._dark_sel.setChecked(True)
    assert panel._dark_sel._path_ed.text().strip() == "/tmp/x/dark.h5"


def test_unchecking_dark_resets_the_path():
    # Unchecking must clear the path outright (not just hide it) so a later
    # re-check prefills fresh from Data rather than keeping a stale pick.
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._path_ed.setText("/tmp/x/data.h5")
    panel._dark_sel.setChecked(True)
    assert panel._dark_sel._path_ed.text().strip() == "/tmp/x/data.h5"
    panel._dark_sel.setChecked(False)
    assert panel._dark_sel._path_ed.text().strip() == ""


def test_rechecking_dark_reprefills_from_the_current_data_path():
    # Re-checking after an uncheck must reflect whatever Data currently
    # points at, even if Data changed while Dark was off — the stale path
    # from before the uncheck must not persist.
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._path_ed.setText("/tmp/x/data.h5")
    panel._dark_sel.setChecked(True)
    panel._dark_sel.setChecked(False)
    panel._path_ed.setText("/tmp/y/other.h5")  # Data changed while Dark was off
    panel._dark_sel.setChecked(True)
    assert panel._dark_sel._path_ed.text().strip() == "/tmp/y/other.h5"


def test_checked_dark_ignores_a_later_data_path_change():
    # While Dark stays checked, editing Data must not silently retarget
    # it — only an uncheck/recheck cycle re-syncs (see the two tests above).
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._path_ed.setText("/tmp/x/data.h5")
    panel._dark_sel.setChecked(True)
    panel._path_ed.setText("/tmp/y/other.h5")
    assert panel._dark_sel._path_ed.text().strip() == "/tmp/x/data.h5"


def test_dark_browse_starts_from_data_folder_when_dark_path_is_empty():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._path_ed.setText("/tmp/x/data.h5")
    seen = {}

    class _FakeDialog:
        def __init__(self, parent, *, title, start_dir=""):
            seen["start_dir"] = start_dir

        def exec_(self):
            return 0  # QDialog.Rejected

    orig = W.BrowseFilesDialog
    W.BrowseFilesDialog = _FakeDialog
    try:
        panel._dark_sel._open_browse_dialog()
    finally:
        W.BrowseFilesDialog = orig
    assert seen["start_dir"] == "/tmp/x/data.h5"


def test_unchecking_dark_emits_fields_changed():
    # Turning a correction off is itself a change the preview must react to
    # (previously only a *completed compute* emitted fieldReady, so
    # unchecking Dark/Bright/Background left the corrected preview stale).
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._dark_sel.setChecked(True)
    seen = []
    panel.fieldsChanged.connect(lambda: seen.append(True))
    panel._dark_sel.setChecked(False)
    assert seen
    assert panel._dark_sel.get_field() is None


def test_unchecking_dark_clears_a_previously_computed_field():
    # Uncheck resets the field along with the path — recheck must not
    # silently resurrect a stale computed field; the user must Compute again.
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._dark_sel.setChecked(True)
    panel._dark_sel._field = np.zeros((2, 2))  # pretend Compute already ran
    panel._dark_sel.setChecked(False)
    seen = []
    panel.fieldsChanged.connect(lambda: seen.append(True))
    panel._dark_sel.setChecked(True)
    assert seen
    assert panel._dark_sel.get_field() is None


def test_background_scale_defaults_to_one_and_multiplies_the_field():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    assert panel._bg_sel._scale.value() == 1.0
    panel._bg_sel.setChecked(True)
    panel._bg_sel._field = np.full((2, 2), 3.0)
    assert np.allclose(panel._bg_sel.get_field(), 3.0)   # scale=1 -> unchanged

    panel._bg_sel._scale.setValue(2.5)
    assert np.allclose(panel._bg_sel.get_field(), 7.5)
    assert panel.background() is not None
    assert np.allclose(panel.background(), 7.5)


def test_dark_and_bright_have_no_scale_field():
    # Only Background gets the scale knob — Dark/Bright are unchanged.
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    assert panel._dark_sel._scale is None
    assert panel._bright_sel._scale is None


def test_background_scale_change_emits_fields_changed():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    seen = []
    panel.fieldsChanged.connect(lambda: seen.append(True))
    panel._bg_sel._scale.setValue(2.0)
    assert seen


def test_background_scale_roundtrips_through_get_set_state():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="single")
    panel._bg_sel._scale.setValue(0.5)
    state = panel.get_state()
    assert state["background"]["scale"] == 0.5

    restored = W.DataLoaderPanel(mode="single")
    restored.set_state(state)
    assert restored._bg_sel._scale.value() == 0.5


def test_manual_edit_clears_stem_filter_and_explicit_paths():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="stream")
    panel._set_stem_filter("/tmp/x", "scan_")
    panel._path_ed.setText("/tmp/y")   # a real user edit, not a Browse… pick
    assert panel._stem_filter is None
    assert panel.source_cfg() == {"type": "tiff_glob", "path": "/tmp/y"}

    panel._set_explicit_paths(["/tmp/x/a.tif"])
    panel._path_ed.setText("/tmp/z")
    assert panel._explicit_paths is None
    assert panel.source_cfg() == {"type": "tiff_glob", "path": "/tmp/z"}


def test_info_label_reports_filestem_and_file_count(tmp_path):
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="stream")
    panel._set_stem_filter(str(tmp_path), "scan_")
    panel._load()
    assert "filestem: scan_*" in panel._info.text()

    panel2 = W.DataLoaderPanel(mode="stream")
    panel2._set_explicit_paths([str(tmp_path / "a.tif"), str(tmp_path / "b.tif")])
    panel2._load()
    assert "2 file(s)" in panel2._info.text()


def test_stem_filter_roundtrips_through_get_set_state():
    W, _app = _make_app_and_module()
    panel = W.DataLoaderPanel(mode="stream")
    panel._set_stem_filter("/tmp/x", "scan_")
    state = panel.get_state()
    assert state["stem_filter"] == "scan_"
    assert state["path"] == "/tmp/x"

    restored = W.DataLoaderPanel(mode="stream")
    restored.set_state(state)
    assert restored._stem_filter == "scan_"
    assert restored.source_cfg() == {"type": "tiff_glob", "path": "/tmp/x/**/scan_*"}


def test_explicit_tiff_source_reads_paths_in_order(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    import midas_gui.workers as wk

    paths = []
    for i in range(3):
        p = tmp_path / f"scan_{i:04d}.tif"
        tifffile.imwrite(str(p), np.full((4, 4), i, dtype=np.float32))
        paths.append(str(p))

    src = wk._ExplicitTIFFSource(paths)
    assert src.n_frames == 3
    frames = list(src)
    assert [fid for fid, _ in frames] == ["scan_0000", "scan_0001", "scan_0002"]
    assert frames[2][1].mean() == 2.0


def test_batch_worker_open_source_dispatches_tiff_list(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    torch = pytest.importorskip("torch")
    import midas_gui.workers as wk

    p = tmp_path / "a.tif"
    tifffile.imwrite(str(p), np.zeros((4, 4), dtype=np.float32))

    worker = wk.BatchWorker.__new__(wk.BatchWorker)
    worker._src = {"type": "tiff_list", "paths": [str(p)]}
    source = worker._open_source()
    assert isinstance(source, wk._ExplicitTIFFSource)
    assert source.n_frames == 1


def test_explicit_tiff_source_random_access_matches_iteration(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    import midas_gui.workers as wk

    paths = []
    for i in range(3):
        p = tmp_path / f"scan_{i:04d}.tif"
        tifffile.imwrite(str(p), np.full((4, 4), i, dtype=np.float32))
        paths.append(str(p))

    src = wk._ExplicitTIFFSource(paths)
    for i in range(3):
        fid, img = src.get(i)
        assert fid == f"scan_{i:04d}"
        assert img.mean() == i


# ── Batch Parallel: frame-index resolution / chunk-splitting / worker count ──

def test_resolve_frame_indices_full_range():
    import midas_gui.workers as wk
    assert wk.resolve_frame_indices(25, None) == list(range(25))


def test_resolve_frame_indices_with_stride_and_end():
    import midas_gui.workers as wk
    assert wk.resolve_frame_indices(25, (2, 20, 3)) == list(range(2, 20, 3))


def test_resolve_frame_indices_end_clamped_to_n_frames():
    import midas_gui.workers as wk
    assert wk.resolve_frame_indices(10, (5, 1000, 1)) == list(range(5, 10))


def test_split_into_chunks_covers_all_indices_in_order():
    import midas_gui.workers as wk
    indices = list(range(23))
    chunks = wk._split_into_chunks(indices, 4)
    assert len(chunks) == 4
    assert sum(chunks, []) == indices
    # near-equal: no chunk more than 1 larger than the smallest
    assert max(len(c) for c in chunks) - min(len(c) for c in chunks) <= 1


def test_split_into_chunks_more_chunks_than_items():
    import midas_gui.workers as wk
    indices = [0, 1, 2]
    chunks = wk._split_into_chunks(indices, 10)
    assert len(chunks) == 3
    assert all(len(c) == 1 for c in chunks)
    assert sum(chunks, []) == indices


def test_batch_parallel_frame_done_reorders_out_of_completion_order():
    """Batch Parallel's chunks run concurrently, so a faster chunk's
    frame_done signals can arrive before a slower, earlier chunk's — the
    waterfall/stacked-profile views must still see frames in overall sorted
    frame-index order, not wall-clock completion order."""
    import midas_gui.workers as wk

    coord = wk.BatchRunCoordinator(
        spec=None, source_cfg=None, mask=None, out_dir=None, fmts=[],
        kernel=None, corrections=None, variance_cfg=None)

    class _DummyWorker:
        pass

    chunk_a, chunk_b = [0, 1, 2], [3, 4, 5]
    w_a, w_b = _DummyWorker(), _DummyWorker()
    coord._chunks = [chunk_a, chunk_b]
    coord._live_order = [i for c in coord._chunks for i in c]
    coord._chunk_frame_counter = {id(w_a): 0, id(w_b): 0}

    emitted = []
    coord.frame_done.connect(lambda fid, r_ax, prof, sigma: emitted.append(fid))

    # Chunk b (fast) completes fully before chunk a (slow) reports anything.
    for fid in (3, 4, 5):
        coord._on_chunk_frame(w_b, chunk_b, str(fid), None, None, None)
    assert emitted == []   # buffered — frames 0-2 haven't arrived yet
    for fid in (0, 1, 2):
        coord._on_chunk_frame(w_a, chunk_a, str(fid), None, None, None)

    assert emitted == ["0", "1", "2", "3", "4", "5"]


def _make_h5_stack(path, n_raw=10, size=4):
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w") as f:
        f.create_dataset("exchange/data",
                          data=np.zeros((n_raw, size, size), dtype=np.float32))


def test_frame_range_multi_file_hdf5_spans_all_files_with_combine_chunk(tmp_path):
    """Regression: a multi-file HDF5 pick with "Combine sub-frames" set
    used to return FILE-index bounds (0..len(paths)) even though the
    worker consumes COMBINED-FRAME indices — so any chunk_size that split
    a file into more than one combined frame made the run stop partway
    through the first file(s) and never reach later files at all."""
    pytest.importorskip("h5py")
    W, _app = _make_app_and_module()

    paths = []
    for num in (9251, 9253, 9255):
        p = tmp_path / f"C611_017Fe_1_load3_{num:06d}.vrx.h5"
        _make_h5_stack(p, n_raw=10)
        paths.append(str(p))

    panel = W.DataLoaderPanel(mode="stream")
    panel._set_explicit_paths(paths)
    panel._combine_chunk.setValue(3)   # 10 raw frames / chunk 3 -> 4 combined frames/file
    panel._autofill_frame_range()

    assert panel.source_cfg()["type"] == "hdf5_stack_glob"
    # 3 files x 4 combined frames each = 12, not the file count (3).
    assert panel.frame_range() == (0, 12, 1)


# ── unify_combine (Batch Integrate): stride replaced by "Combine sub-frames" ──

def _make_tiff_files(tmp_path, nums):
    tifffile = pytest.importorskip("tifffile")
    paths = []
    for n in nums:
        p = tmp_path / f"scan_{n:06d}.tif"
        tifffile.imwrite(str(p), np.zeros((3, 3), dtype=np.float32))
        paths.append(str(p))
    return paths


def test_unify_combine_off_by_default_matches_plain_panel():
    """Every existing "stream" consumer (Pump Probe, and a bare-constructed
    panel like the other tests in this file) must see byte-identical
    source_cfg()/frame_range() behavior — unify_combine defaults to False."""
    W, _app = _make_app_and_module()
    plain = W.DataLoaderPanel(mode="stream")
    default = W.DataLoaderPanel(mode="stream", unify_combine=False)
    for panel in (plain, default):
        panel._path_ed.setText("/tmp/x")
        assert panel.source_cfg() == {"type": "tiff_glob", "path": "/tmp/x"}
        assert not panel._unify_combine


def test_unify_combine_hides_stride_shows_combine_for_tiff_folder(tmp_path):
    W, _app = _make_app_and_module()
    _make_tiff_files(tmp_path, [9241, 9242, 9243])
    panel = W.DataLoaderPanel(mode="stream", unify_combine=True)
    # The stride spinbox still exists (frame_range()'s shape relies on it)
    # but is never added to a layout for a unify_combine panel.
    assert panel._fr_stride.parent() is None
    panel._path_ed.setText(str(tmp_path))
    panel._update_combine_visibility()
    assert not panel._combine_row.isHidden()   # visible even for a TIFF folder


def test_unify_combine_source_cfg_gains_chunk_and_frame_bounds_for_tiff(tmp_path):
    W, _app = _make_app_and_module()
    _make_tiff_files(tmp_path, [9241, 9242, 9243, 9244, 9245])
    panel = W.DataLoaderPanel(mode="stream", unify_combine=True)
    panel._path_ed.setText(str(tmp_path))
    panel._autofill_frame_range()

    cfg = panel.source_cfg()
    assert cfg["type"] == "tiff_glob"
    assert cfg["chunk_size"] == 1 and cfg["combine_op"] == "mean"
    assert (cfg["frame_start"], cfg["frame_end"]) == (9241, 9245)
    # unify_combine always short-circuits frame_range() — filtering/chunking
    # is baked into source_cfg() instead (see workers._open_source_cfg).
    assert panel.frame_range() == (0, None, 1)

    panel._combine_chunk.setValue(2)
    panel._fr_start.setValue(9242)
    panel._fr_end.setValue(9244)
    cfg2 = panel.source_cfg()
    assert (cfg2["frame_start"], cfg2["frame_end"], cfg2["chunk_size"]) == (9242, 9244, 2)

    import midas_gui.workers as wk
    src = wk._open_source_cfg(cfg2)
    assert src.n_frames == 2
    assert [fid for fid, _ in src] == ["scan_009242.frame_0_1", "scan_009244"]

    # Regression: changing "Combine sub-frames" (chunk size or op) AFTER
    # start/end were narrowed must NOT snap them back to the full range —
    # that pick is the user's own data-range selection, orthogonal to how
    # sub-frames combine (DataLoaderPanel._on_combine_changed).
    panel._combine_op_combo.setCurrentIndex(
        panel._combine_op_combo.findData("sum"))
    assert (panel._fr_start.value(), panel._fr_end.value()) == (9242, 9244)
    panel._combine_chunk.setValue(3)
    assert (panel._fr_start.value(), panel._fr_end.value()) == (9242, 9244)


def test_unify_combine_explicit_multi_file_pick_gains_frame_bounds(tmp_path):
    W, _app = _make_app_and_module()
    paths = _make_tiff_files(tmp_path, [10, 11, 12])
    panel = W.DataLoaderPanel(mode="stream", unify_combine=True)
    panel._set_explicit_paths(paths)
    panel._autofill_frame_range()
    cfg = panel.source_cfg()
    assert cfg["type"] == "tiff_list"
    assert cfg["paths"] == paths
    assert (cfg["frame_start"], cfg["frame_end"]) == (10, 12)


def test_unify_combine_hdf5_stack_glob_also_gains_frame_bounds(tmp_path):
    pytest.importorskip("h5py")
    W, _app = _make_app_and_module()
    paths = []
    for num in (9251, 9253, 9255):
        p = tmp_path / f"C611_017Fe_1_load3_{num:06d}.vrx.h5"
        _make_h5_stack(p, n_raw=10)
        paths.append(str(p))

    panel = W.DataLoaderPanel(mode="stream", unify_combine=True)
    panel._set_explicit_paths(paths)
    panel._combine_chunk.setValue(3)
    panel._autofill_frame_range()

    cfg = panel.source_cfg()
    assert cfg["type"] == "hdf5_stack_glob"
    assert (cfg["frame_start"], cfg["frame_end"]) == (9251, 9255)
    assert panel.frame_range() == (0, None, 1)

    panel._fr_start.setValue(9253)
    panel._fr_end.setValue(9253)
    cfg2 = panel.source_cfg()

    import midas_gui.workers as wk
    src = wk._open_source_cfg(cfg2)
    # Only the middle file survives the filter -> its own 4 combined chunks.
    assert src.n_frames == 4


# ── unify_combine: single HDF5 file's start/end filter its own raw sub-frames ──

def test_unify_combine_single_hdf5_file_start_end_span_full_raw_count(tmp_path):
    """A single bare HDF5 file has no scan-NUMBER range (there's only one
    file) — a unify_combine panel instead bounds start/end to the file's own
    raw sub-frame count and defaults to the full range."""
    pytest.importorskip("h5py")
    W, _app = _make_app_and_module()
    p = tmp_path / "scan_009243.h5"
    _make_h5_stack(p, n_raw=10)

    panel = W.DataLoaderPanel(mode="stream", unify_combine=True)
    panel._path_ed.setText(str(p))
    panel._autofill_frame_range()

    assert panel.source_cfg()["type"] == "hdf5"
    assert panel._fr_start.minimum() == 0 and panel._fr_start.maximum() == 9
    assert panel._fr_end.minimum() == 0 and panel._fr_end.maximum() == 9
    assert panel._fr_start.isEnabled() and panel._fr_end.isEnabled()
    assert (panel._fr_start.value(), panel._fr_end.value()) == (0, 9)

    cfg = panel.source_cfg()
    assert (cfg["frame_start"], cfg["frame_end"]) == (0, 9)
    assert panel.frame_range() == (0, None, 1)

    import midas_gui.workers as wk
    # chunk_size defaults to 1 (no combining) -> one output frame per raw
    # sub-frame, all 10 survive the unfiltered (0, 9) range.
    assert wk._open_source_cfg(cfg).n_frames == 10


def test_unify_combine_single_hdf5_file_start_end_filter_raw_subframes(tmp_path):
    """Narrowing start/end filters the file's raw sub-frames BEFORE
    "Combine sub-frames" chunks whatever survives — a 10-raw-frame file
    restricted to [2, 7] with chunk_size=3 combines only 2..4 and 5..7."""
    pytest.importorskip("h5py")
    W, _app = _make_app_and_module()
    p = tmp_path / "scan_009243.h5"
    _make_h5_stack(p, n_raw=10)

    panel = W.DataLoaderPanel(mode="stream", unify_combine=True)
    panel._path_ed.setText(str(p))
    panel._combine_chunk.setValue(3)
    panel._autofill_frame_range()
    panel._fr_start.setValue(2)
    panel._fr_end.setValue(7)

    cfg = panel.source_cfg()
    assert (cfg["frame_start"], cfg["frame_end"], cfg["chunk_size"]) == (2, 7, 3)

    import midas_gui.workers as wk
    src = wk._open_source_cfg(cfg)
    assert src.n_frames == 2
    assert [fid for fid, _ in src] == ["scan_009243.frame_2_4", "scan_009243.frame_5_7"]


def test_unify_combine_single_hdf5_file_single_raw_frame_disables_start_end(tmp_path):
    """A single raw sub-frame (nothing to range over) disables start/end,
    same spirit as the pre-existing scan-number lock for a non-unify_combine
    panel — just gated on the raw count instead of the file count."""
    pytest.importorskip("h5py")
    W, _app = _make_app_and_module()
    p = tmp_path / "scan_009243.h5"
    _make_h5_stack(p, n_raw=1)

    panel = W.DataLoaderPanel(mode="stream", unify_combine=True)
    panel._path_ed.setText(str(p))
    panel._autofill_frame_range()

    assert not panel._fr_start.isEnabled()
    assert not panel._fr_end.isEnabled()
    assert (panel._fr_start.value(), panel._fr_end.value()) == (0, 0)


def test_unify_combine_off_single_hdf5_file_keeps_legacy_scan_number_lock(tmp_path):
    """Every non-unify_combine consumer (Pump Probe, a bare-constructed
    panel) must keep today's behavior byte-identical: start/end lock to the
    file's own scan number and are disabled, regardless of raw frame count."""
    pytest.importorskip("h5py")
    W, _app = _make_app_and_module()
    p = tmp_path / "scan_009243.h5"
    _make_h5_stack(p, n_raw=10)

    panel = W.DataLoaderPanel(mode="stream")
    panel._path_ed.setText(str(p))
    panel._autofill_frame_range()

    assert not panel._fr_start.isEnabled()
    assert not panel._fr_end.isEnabled()
    assert (panel._fr_start.value(), panel._fr_end.value()) == (9243, 9243)
    assert "frame_start" not in panel.source_cfg()


def test_resolve_worker_count_shrinks_to_minimum_ten_per_worker():
    import midas_gui.workers as wk
    # plenty of frames — full requested count survives
    assert wk.resolve_worker_count(100, 8, 10) == 8
    # too few frames for 8 workers at 10/worker — shrinks to 1 (15 // 10 == 1)
    assert wk.resolve_worker_count(15, 8, 10) == 1
    # exactly enough for 2 workers
    assert wk.resolve_worker_count(20, 8, 10) == 2
    # never below 1, even with very few frames
    assert wk.resolve_worker_count(3, 8, 10) == 1


# ── write_all_profiles (backs the batch tabs' Save button) ──────────────────

def test_write_all_profiles_writes_every_frame_and_format(tmp_path):
    pytest.importorskip("midas_integrate_v2")
    import midas_gui.workers as wk

    n = 3
    r_axis = np.linspace(1.0, 10.0, 20)
    profiles = np.random.rand(n, 20)
    sigmas = np.sqrt(profiles)
    frame_ids = [f"frame_{i:03d}" for i in range(n)]

    paths = wk.write_all_profiles(
        tmp_path, ["csv", "dat", "h5"], r_axis, profiles, sigmas, frame_ids,
        lsd=200000.0, px=200.0, wl=0.2)

    # Output names follow the <froot>_<NNNNNN> convention (workers.
    # frame_output_base), so "frame_000" is written as "frame_000000".
    for i in range(n):
        assert (tmp_path / f"frame_{i:06d}.csv").exists()
        assert (tmp_path / f"frame_{i:06d}.dat").exists()
    assert (tmp_path / "integrated.h5").exists()
    assert len(paths) == 2 * n + 1


def test_write_all_profiles_skips_2d_csv(tmp_path):
    pytest.importorskip("midas_integrate_v2")
    import midas_gui.workers as wk

    r_axis = np.linspace(1.0, 10.0, 5)
    profiles = np.random.rand(1, 5)
    frame_ids = ["f0"]
    paths = wk.write_all_profiles(
        tmp_path, ["2d_csv"], r_axis, profiles, None, frame_ids,
        lsd=200000.0, px=200.0, wl=0.2)
    assert paths == []
    assert not any(tmp_path.iterdir())


# ── Enter-to-validate on the path fields ─────────────────────────────────

def test_data_field_enter_on_a_missing_path_warns_instead_of_loading(monkeypatch, tmp_path):
    # Regression: pressing Enter on a typo'd path used to fall straight
    # into _load()'s tifffile/h5py attempt and surface a raw traceback
    # dialog. It should now show one friendly "Not found" warning and never
    # reach _load() at all.
    from PyQt5 import QtWidgets
    W, _app = _make_app_and_module()

    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: calls.append(a))

    panel = W.DataLoaderPanel(mode="single")
    missing = tmp_path / "nope.h5"
    panel._path_ed.setText(str(missing))
    panel._path_ed.returnPressed.emit()
    assert len(calls) == 1
    assert str(missing) in calls[0][-1]
    assert panel._nframes == 0   # _load() never ran


def test_data_field_enter_on_an_existing_path_loads_normally(tmp_path):
    import tifffile
    W, _app = _make_app_and_module()

    path = tmp_path / "frame.tif"
    tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.float32))

    panel = W.DataLoaderPanel(mode="single")
    panel._path_ed.setText(str(path))
    panel._path_ed.returnPressed.emit()
    assert panel._nframes == 1


def test_dark_field_enter_on_a_missing_path_warns(monkeypatch, tmp_path):
    from PyQt5 import QtWidgets
    W, _app = _make_app_and_module()

    calls = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                         lambda *a, **k: calls.append(a))

    panel = W.DataLoaderPanel(mode="single")
    panel._dark_sel.setChecked(True)
    panel._dark_sel._path_ed.setText(str(tmp_path / "nope.h5"))
    panel._dark_sel._path_ed.returnPressed.emit()
    assert len(calls) == 1


# ── Browse dialog starts at the typed path ───────────────────────────────

def test_data_field_browse_starts_at_the_typed_directory(monkeypatch, tmp_path):
    W, _app = _make_app_and_module()

    seen = {}

    class _FakeDialog:
        def __init__(self, parent=None, *, title="", modes=(), start_dir=""):
            seen["start_dir"] = start_dir
        def exec_(self):
            from PyQt5 import QtWidgets
            return QtWidgets.QDialog.Rejected
    monkeypatch.setattr(W, "BrowseFilesDialog", _FakeDialog)

    panel = W.DataLoaderPanel(mode="single")
    panel._path_ed.setText(str(tmp_path))
    panel._open_browse_dialog()
    assert seen["start_dir"] == str(tmp_path)
