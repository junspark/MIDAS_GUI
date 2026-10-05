"""Batch Correction, end to end — tab → ``BatchCorrectionWorker`` → HDF5.

The numeric guarantees live in ``test_frame_correct.py`` and the dark ladder
in ``test_dark_autodetect.py``; what's pinned here is the whole path working
together against real files on disk: one output per input, chunks that never
cross a file boundary, metadata averaged per chunk, and the compression
choice actually reaching h5py.

Builds Qt widgets, hence forked, with every Qt / midas_gui GUI import
deferred into the fixtures — STATE.md's rule for new Qt test files.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked

SHAPE = (6, 7)
N_RAW = 12          # raw sub-frames per input file
DARK_LEVEL = 10.0


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_file(path, *, n=N_RAW, base=0.0, with_dark=True, meta=True):
    """A VAREX-shaped input: ``exchange/data`` ramped so each sub-frame is
    distinguishable, an internal dark, and a per-acquisition metadata array
    twice the frame count (lights then darks — the real DAQ layout)."""
    import h5py
    data = np.stack([np.full(SHAPE, base + i, np.float32) for i in range(n)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("exchange/data", data=data)
        if with_dark:
            f.create_dataset("exchange/data_dark",
                             data=np.full((2,) + SHAPE, DARK_LEVEL, np.float32))
        if meta:
            # One sample per acquisition: n lights then n darks, as the real
            # files do — the light/dark split is what `n_aligned` recovers.
            f.create_dataset("instrument/Scalers/E/US_IC",
                             data=np.arange(2 * n, dtype=np.float64))
    return path


@pytest.fixture
def inputs(tmp_path):
    src = tmp_path / "scan"
    return [_make_file(src / f"scan_{n:06d}.vrx.h5", base=100.0 * i)
            for i, n in enumerate((10, 11))]


def _read(path, dataset="exchange/data"):
    import h5py
    with h5py.File(str(path), "r") as f:
        return np.asarray(f[dataset])


def _run_worker(inputs, out_dir, app, **kwargs):
    """Drive the worker synchronously — ``run()`` directly, no event loop."""
    from midas_gui.workers import BatchCorrectionWorker
    opts = dict(dataset="exchange/data", chunk_size=4, op="mean",
                out_dir=str(out_dir), compression=None)
    opts.update(kwargs)
    captured = {"outputs": None, "error": None, "log": []}
    w = BatchCorrectionWorker([str(p) for p in inputs], **opts)
    w.finished.connect(lambda o: captured.__setitem__("outputs", o))
    w.failed.connect(lambda e: captured.__setitem__("error", e))
    w.progress.connect(lambda d, t, m: captured["log"].append(m))
    w.run()
    assert captured["error"] is None, captured["error"]
    return captured


# ── worker: output shape and naming ─────────────────────────────────────

def test_one_output_file_per_input_file(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app)
    assert len(got["outputs"]) == 2


def test_output_name_keeps_the_stem_and_drops_the_detector_tag(app, inputs, tmp_path):
    """``Path.stem`` strips only the LAST suffix, so a naive stem would give
    ``scan_000010.vrx_corr.h5``."""
    got = _run_worker(inputs, tmp_path / "out", app)
    names = sorted(__import__("pathlib").Path(p).name for p in got["outputs"])
    assert names == ["scan_000010_corr.h5", "scan_000011_corr.h5"]


def test_custom_suffix_is_honoured(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app, suffix="_avg10")
    assert all(p.endswith("_avg10.h5") for p in got["outputs"])


def test_chunking_splits_each_file_independently(app, inputs, tmp_path):
    """12 raw sub-frames at chunk 5 → 3 chunks (5, 5, 2) PER FILE, not a
    single run of chunks spanning both files."""
    got = _run_worker(inputs, tmp_path / "out", app, chunk_size=5)
    for path in got["outputs"]:
        assert _read(path).shape == (3,) + SHAPE
        assert _read(path, "frame_ranges").tolist() == [[0, 4], [5, 9], [10, 11]]


def test_whole_file_chunking_gives_one_frame_each(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app, chunk_size=None)
    for path in got["outputs"]:
        assert _read(path).shape == (1,) + SHAPE


def test_output_dataset_name_is_configurable(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app, out_dataset="data/corrected")
    assert _read(got["outputs"][0], "data/corrected").shape[0] == 3


# ── worker: the numbers ─────────────────────────────────────────────────

def test_mean_values_are_the_chunk_mean_minus_the_dark(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app, chunk_size=4, op="mean")
    # File 0, chunk 0 = sub-frames 0..3 = values 0,1,2,3 → mean 1.5, −10 dark,
    # clipped at 0.
    assert np.allclose(_read(got["outputs"][0])[0], 0.0)
    # Chunk 2 = 8,9,10,11 → mean 9.5, −10 → −0.5 → clipped to 0.
    assert np.allclose(_read(got["outputs"][0])[2], 0.0)


def test_sum_loses_one_dark_per_sub_frame(app, inputs, tmp_path):
    """The whole point, end to end: a 4-frame sum must lose 4 darks."""
    got = _run_worker(inputs, tmp_path / "out", app, chunk_size=4, op="sum",
                      clip_negatives=False)
    # File 1 is based at 100, so chunk 0 = 100,101,102,103 → 406, −4×10 = 366.
    assert np.allclose(_read(got["outputs"][1])[0], 406.0 - 4 * DARK_LEVEL)


def test_unclipped_output_keeps_negative_pixels(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app, chunk_size=4, op="mean",
                      clip_negatives=False)
    assert _read(got["outputs"][0])[0].max() < 0   # mean 1.5 − dark 10


@pytest.mark.parametrize("op", ["mean", "median", "sum", "max"])
def test_every_op_runs_end_to_end(app, inputs, tmp_path, op):
    got = _run_worker(inputs, tmp_path / f"out_{op}", app, op=op)
    assert _read(got["outputs"][0]).shape == (3,) + SHAPE


# ── worker: dark resolution ─────────────────────────────────────────────

def test_sibling_dark_before_is_preferred_and_logged(app, tmp_path):
    folder = tmp_path / "scan"
    import h5py
    folder.mkdir(parents=True)
    with h5py.File(str(folder / "scan_dark_before_000009.vrx.h5"), "w") as f:
        f.create_dataset("exchange/data_dark",
                         data=np.full((2,) + SHAPE, 40.0, np.float32))
    data = _make_file(folder / "scan_000010.vrx.h5", base=1000.0)
    got = _run_worker([data], tmp_path / "out", app, chunk_size=None, op="mean",
                      clip_negatives=False)
    # Mean of 1000..1011 is 1005.5; the sibling dark (40) must win over the
    # file's own internal dark (10).
    assert np.allclose(_read(got["outputs"][0])[0], 1005.5 - 40.0)
    assert any("dark_before_000009" in m for m in got["log"])


def test_auto_dark_off_uses_the_loader_field(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app, chunk_size=None, op="mean",
                      auto_dark=False, clip_negatives=False,
                      dark=np.full(SHAPE, 2.0, np.float32))
    assert np.allclose(_read(got["outputs"][0])[0], 5.5 - 2.0)
    assert any("loader" in m.lower() for m in got["log"])


def test_a_file_with_no_dark_anywhere_still_runs(app, tmp_path):
    data = _make_file(tmp_path / "scan" / "scan_000010.vrx.h5", with_dark=False)
    got = _run_worker([data], tmp_path / "out", app, chunk_size=None, op="mean")
    assert np.allclose(_read(got["outputs"][0])[0], 5.5)
    assert any("no dark" in m.lower() for m in got["log"])


def test_the_dark_used_is_recorded_in_the_output_file(app, inputs, tmp_path):
    import h5py
    got = _run_worker(inputs, tmp_path / "out", app)
    with h5py.File(got["outputs"][0], "r") as f:
        assert "data_dark" in f.attrs["midas_gui_dark"]
        assert f.attrs["midas_gui_combine_op"] == "mean"


# ── worker: metadata ────────────────────────────────────────────────────

def test_metadata_is_averaged_over_each_chunk(app, inputs, tmp_path):
    """The per-acquisition array is length 2N (lights then darks). Only the
    N light entries belong to the frames, and each output frame gets the
    mean of its own chunk's entries."""
    got = _run_worker(inputs, tmp_path / "out", app, chunk_size=4)
    ic = _read(got["outputs"][0], "instrument/Scalers/E/US_IC")
    # Lights are 0..11; chunks (0-3), (4-7), (8-11) → means 1.5, 5.5, 9.5.
    assert np.allclose(ic, [1.5, 5.5, 9.5])


def test_metadata_absent_in_the_source_is_simply_absent(app, tmp_path):
    import h5py
    data = _make_file(tmp_path / "scan" / "scan_000010.vrx.h5", meta=False)
    got = _run_worker([data], tmp_path / "out", app)
    with h5py.File(got["outputs"][0], "r") as f:
        assert "instrument" not in f


# ── worker: compression ─────────────────────────────────────────────────

@pytest.mark.parametrize("comp", [None, "none", "gzip", "lzf"])
def test_each_compression_choice_round_trips(app, inputs, tmp_path, comp):
    got = _run_worker(inputs, tmp_path / f"out_{comp}", app, compression=comp)
    assert _read(got["outputs"][0]).shape == (3,) + SHAPE


def test_the_compression_filter_is_recorded_on_the_output_dataset(app, inputs,
                                                                  tmp_path):
    """The size check below needs a realistically-sized stack to mean
    anything, so assert the filter itself here — this is the part that
    actually proves the setting reached h5py."""
    import h5py
    plain = _run_worker(inputs, tmp_path / "plain", app, compression=None)
    gz = _run_worker(inputs, tmp_path / "gz", app, compression="gzip",
                     level=6, shuffle=True)
    with h5py.File(plain["outputs"][0], "r") as f:
        assert f["exchange/data"].compression is None
    with h5py.File(gz["outputs"][0], "r") as f:
        dset = f["exchange/data"]
        assert dset.compression == "gzip"
        assert dset.compression_opts == 6
        assert dset.shuffle is True
        assert dset.chunks == (1,) + SHAPE   # one frame per chunk


def test_gzip_shrinks_a_realistically_sized_stack(app, tmp_path):
    """On the tiny fixture above, HDF5's own filter/chunk metadata is larger
    than 504 bytes of pixels and the compressed file comes out BIGGER — a
    true property of small datasets, not a bug. Size is only a meaningful
    check at a realistic frame size."""
    import pathlib
    import h5py
    big = tmp_path / "scan" / "scan_000010.vrx.h5"
    big.parent.mkdir(parents=True)
    with h5py.File(str(big), "w") as f:
        f.create_dataset("exchange/data",
                         data=np.zeros((8, 256, 256), np.float32))
    plain = _run_worker([big], tmp_path / "plain", app, chunk_size=2,
                        compression=None)
    gz = _run_worker([big], tmp_path / "gz", app, chunk_size=2,
                     compression="gzip", shuffle=True)
    assert (pathlib.Path(gz["outputs"][0]).stat().st_size
            < pathlib.Path(plain["outputs"][0]).stat().st_size)


# ── worker: cancellation ────────────────────────────────────────────────

def test_cancelling_before_the_run_writes_nothing(app, inputs, tmp_path):
    from midas_gui.workers import BatchCorrectionWorker
    out = tmp_path / "out"
    w = BatchCorrectionWorker([str(p) for p in inputs], dataset="exchange/data",
                              chunk_size=4, op="mean", out_dir=str(out))
    got = {}
    w.finished.connect(lambda o: got.setdefault("o", o))
    w.cancel()
    w.run()
    assert got["o"] == []
    assert not out.exists() or list(out.glob("*.h5")) == []


# ── tab ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tab(app):
    from midas_gui.tab_batch_correct import BatchCorrectionTab
    return BatchCorrectionTab()


def test_tab_builds_and_round_trips_its_state(tab):
    tab._suffix_ed.setText("_reduced")
    tab._comp_combo.setCurrentIndex(1)
    tab._clip_chk.setChecked(False)
    state = tab.get_state()

    from midas_gui.tab_batch_correct import BatchCorrectionTab
    other = BatchCorrectionTab()
    other.set_state(state)
    assert other._suffix_ed.text() == "_reduced"
    assert other._comp_combo.currentData() == "gzip"
    assert other._clip_chk.isChecked() is False


def test_gzip_level_is_only_enabled_for_gzip(tab):
    tab._comp_combo.setCurrentIndex(0)          # None
    assert not tab._comp_level.isEnabled() and not tab._shuffle_chk.isEnabled()
    tab._comp_combo.setCurrentIndex(1)          # gzip
    assert tab._comp_level.isEnabled() and tab._shuffle_chk.isEnabled()
    tab._comp_combo.setCurrentIndex(2)          # lzf — shuffle applies, level doesn't
    assert not tab._comp_level.isEnabled() and tab._shuffle_chk.isEnabled()


def test_run_is_disabled_until_an_hdf5_source_is_selected(tab):
    assert not tab._run_btn.isEnabled()
    assert tab._h5_paths() == []


def test_tiff_selection_is_refused_with_an_explanation(tab, tmp_path, monkeypatch):
    for name in ("a_000001.tif", "a_000002.tif"):
        (tmp_path / name).write_bytes(b"")
    tab._loader._set_explicit_paths([str(tmp_path / "a_000001.tif"),
                                     str(tmp_path / "a_000002.tif")])
    assert tab._h5_paths() == []
    shown = {}
    monkeypatch.setattr("midas_gui.tab_batch_correct.show_error",
                        lambda parent, title, msg: shown.update(t=title, m=msg))
    tab._run()
    assert "HDF5" in shown["t"]
    assert "one frame" in shown["m"]


def test_hdf5_selection_enables_the_run(tab, inputs):
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    assert len(tab._h5_paths()) == 2


def test_run_without_an_output_folder_says_so(tab, inputs, monkeypatch):
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    shown = {}
    monkeypatch.setattr("midas_gui.tab_batch_correct.show_error",
                        lambda parent, title, msg: shown.update(t=title))
    tab._run()
    assert shown["t"] == "No output folder"


def test_preview_reduces_only_the_first_chunk(tab, inputs):
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    tab._loader._combine_chunk.setValue(4)
    tab._clip_chk.setChecked(False)
    tab._preview()
    assert tab._preview_raw is not None
    assert tab._preview_raw.shape == SHAPE
    # Sub-frames 0..3 → mean 1.5, internal dark 10.
    assert np.allclose(tab._preview_raw, 1.5 - DARK_LEVEL)
