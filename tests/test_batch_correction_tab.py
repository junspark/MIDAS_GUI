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
    ``scan_000010.vrx.dark_subtracted.hdf`` — carrying a detector tag that no
    longer describes the file."""
    import pathlib
    got = _run_worker(inputs, tmp_path / "out", app)
    names = sorted(pathlib.Path(p).name for p in got["outputs"])
    assert names == ["scan_000010.dark_subtracted.hdf",
                     "scan_000011.dark_subtracted.hdf"]
    assert all(pathlib.Path(p).parent.name == "dark_subtracted_mean"
               for p in got["outputs"])


def test_custom_suffix_and_extension_are_honoured(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app, suffix="_avg10",
                      out_ext=".h5")
    assert all(p.endswith("_avg10.h5") for p in got["outputs"])


def test_a_bare_extension_gets_its_dot(app, inputs, tmp_path):
    got = _run_worker(inputs, tmp_path / "out", app, out_ext="hdf5")
    assert all(p.endswith(".dark_subtracted.hdf5") for p in got["outputs"])


def test_the_default_output_extension_loads_back_as_hdf5(app, inputs, tmp_path):
    """`.hdf` has to be in H5_EXTS or the reduced frames can't be reopened in
    this GUI, which would make the whole output a dead end."""
    from midas_gui.helpers import is_h5
    got = _run_worker(inputs, tmp_path / "out", app)
    assert all(is_h5(p) for p in got["outputs"])


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
    import pathlib
    got = _run_worker(inputs, tmp_path / f"out_{op}", app, op=op)
    assert _read(got["outputs"][0]).shape == (3,) + SHAPE
    assert pathlib.Path(got["outputs"][0]).parent.name == f"dark_subtracted_{op}"


def test_several_ops_each_get_their_own_folder(app, inputs, tmp_path):
    """One run, four methods: 2 files × 4 ops = 8 outputs, one folder each."""
    import pathlib
    got = _run_worker(inputs, tmp_path / "out", app,
                      op=["mean", "median", "sum", "max"])
    assert len(got["outputs"]) == 8
    folders = {pathlib.Path(p).parent.name for p in got["outputs"]}
    assert folders == {"dark_subtracted_mean", "dark_subtracted_median",
                       "dark_subtracted_sum", "dark_subtracted_max"}


def test_multi_op_results_match_running_each_op_alone(app, inputs, tmp_path):
    """The single-pass optimisation must not change any answer."""
    import pathlib
    together = _run_worker(inputs, tmp_path / "multi", app,
                           op=["mean", "sum", "max", "median"],
                           clip_negatives=False)
    by_folder = {pathlib.Path(p).parent.name: p for p in together["outputs"]
                 if pathlib.Path(p).name.startswith("scan_000010")}
    for op in ("mean", "sum", "max", "median"):
        alone = _run_worker([inputs[0]], tmp_path / f"solo_{op}", app, op=op,
                            clip_negatives=False)
        assert np.allclose(_read(by_folder[f"dark_subtracted_{op}"]),
                           _read(alone["outputs"][0])), op


def test_each_output_records_the_op_that_made_it(app, inputs, tmp_path):
    import h5py, pathlib
    got = _run_worker(inputs, tmp_path / "out", app, op=["mean", "max"])
    for path in got["outputs"]:
        with h5py.File(path, "r") as f:
            assert (f.attrs["midas_gui_combine_op"]
                    == pathlib.Path(path).parent.name.rsplit("_", 1)[-1])


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


def test_suggested_output_dir_is_the_integration_folder(tab, inputs):
    """Batch Correction writes its per-op folders INTO Batch Integrate's own
    output folder, so a froot's reduced frames and its cakes sit side by
    side rather than one burying the other."""
    from midas_gui.helpers import suggest_integration_output_dir
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    assert tab._suggest_output_dir() == suggest_integration_output_dir(
        str(inputs[0]))


def test_suggest_button_fills_the_field(tab, inputs):
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    tab._out_ed.clear()
    tab._apply_suggested_output_dir()
    assert tab._out_ed.text()


def test_suggest_with_no_source_says_so_instead_of_raising(tab):
    tab._apply_suggested_output_dir()
    assert tab._out_ed.text() == ""


def test_the_name_preview_matches_what_the_worker_writes(tab, inputs, tmp_path,
                                                         app):
    """The preview label is a second implementation of the naming rule, so
    pin the two together rather than trusting them to stay in step."""
    import pathlib
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    tab._refresh_name_preview()
    # First line is "<source>  →  <output>"; the second is the frame plan.
    previewed = tab._name_lbl.text().splitlines()[0].split("→")[-1].strip()
    got = _run_worker([inputs[0]], tmp_path / "out", app)
    assert previewed == pathlib.Path(got["outputs"][0]).name


def test_the_preview_reports_the_planned_output_frame_count(tab, inputs,
                                                            monkeypatch):
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    # Drive _chunk_settings rather than the loader's spin boxes: setting
    # those re-triggers the panel's own autofill, which opens the source to
    # recount and leaves a worker thread behind that never joins under
    # --forked.
    monkeypatch.setattr(tab, "_chunk_settings", lambda: (4, "mean", None, None))
    tab._refresh_name_preview()
    assert f"{N_RAW} raw sub-frame(s) → 3 output frame(s)" in tab._name_lbl.text()


def test_an_end_of_zero_is_flagged_rather_than_silently_keeping_one_frame(
        tab, inputs, monkeypatch):
    """start/end are a raw SUB-FRAME window for a single HDF5 file, so end=0
    clamps it to sub-frame 0 alone and "Combine sub-frames" has nothing to
    combine. The field was labelled "end(0=all)", which is only true for a
    NON-unify panel — see widgets.DataLoaderPanel's start/end row, where
    frame_range() special-cases `hi > 0`. A unify_combine panel bakes the
    bounds straight into source_cfg(), where 0 is taken literally."""
    tab._loader._set_explicit_paths([str(inputs[0])])
    monkeypatch.setattr(tab, "_chunk_settings", lambda: (4, "mean", 0, 0))
    tab._refresh_name_preview()
    text = tab._name_lbl.text()
    assert "→ 1 output frame(s)" in text
    assert "check start/end" in text, "the no-op window was not flagged"


def test_a_full_window_is_not_flagged(tab, inputs, monkeypatch):
    tab._loader._set_explicit_paths([str(inputs[0])])
    monkeypatch.setattr(tab, "_chunk_settings",
                        lambda: (4, "mean", 0, N_RAW - 1))
    tab._refresh_name_preview()
    assert "check start/end" not in tab._name_lbl.text()


def test_the_end_label_drops_its_0_equals_all_claim_for_this_panel(tab):
    """The label is shared with Batch Integrate, and "0 = all" holds only
    for a non-unify panel. Pin the corrected text so it can't drift back."""
    from PyQt5 import QtWidgets
    labels = [w.text() for w in tab._loader.findChildren(QtWidgets.QLabel)]
    assert not any("0=all" in t for t in labels), \
        "end(0=all) is wrong for a unify_combine panel"


def test_long_field_text_reads_from_the_start_not_the_tail(tab):
    """A QLineEdit narrower than its text shows the TAIL — "exchange/data"
    rendering as "hange/data", which looks like a corrupted default."""
    for field in (tab._out_ds_ed, tab._dark_ds_ed):
        assert field.cursorPosition() == 0


def test_tab_builds_and_round_trips_its_state(tab):
    tab._suffix_ed.setText("_reduced")
    tab._ext_ed.setText(".h5")
    tab._comp_combo.setCurrentIndex(1)
    tab._clip_chk.setChecked(False)
    state = tab.get_state()

    from midas_gui.tab_batch_correct import BatchCorrectionTab
    other = BatchCorrectionTab()
    other.set_state(state)
    assert other._suffix_ed.text() == "_reduced"
    assert other._ext_ed.text() == ".h5"
    assert other._comp_combo.currentData() == "gzip"
    assert other._clip_chk.isChecked() is False


def test_gzip_level_is_only_enabled_for_gzip(tab):
    tab._comp_combo.setCurrentIndex(0)          # None
    assert not tab._comp_level.isEnabled() and not tab._shuffle_chk.isEnabled()
    tab._comp_combo.setCurrentIndex(1)          # gzip
    assert tab._comp_level.isEnabled() and tab._shuffle_chk.isEnabled()
    tab._comp_combo.setCurrentIndex(2)          # lzf — shuffle applies, level doesn't
    assert not tab._comp_level.isEnabled() and tab._shuffle_chk.isEnabled()


def test_buttons_come_back_after_a_completed_run(tab, inputs):
    """Regression: `finished` is emitted from INSIDE QThread.run(), so the
    thread is still alive when the handler fires. Deciding button state from
    isRunning() left Run and Preview dead and only Cancel alive after a
    successful run, with nothing scheduled to put it right."""
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    tab._refresh_enabled()
    assert tab._run_btn.isEnabled() and not tab._cancel_btn.isEnabled()

    tab._running = True
    tab._refresh_enabled()
    assert not tab._run_btn.isEnabled() and tab._cancel_btn.isEnabled()

    tab._on_finished(["/tmp/one.hdf"])
    assert tab._run_btn.isEnabled(), "Run stayed disabled after the run ended"
    assert tab._preview_btn.isEnabled(), "Preview stayed disabled"
    assert not tab._cancel_btn.isEnabled(), "Cancel stayed enabled at rest"


def test_buttons_come_back_after_a_failed_run(tab, inputs, monkeypatch):
    # _on_failed raises a modal dialog, which blocks forever offscreen.
    monkeypatch.setattr("midas_gui.tab_batch_correct.show_error",
                        lambda *a, **k: None)
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    tab._running = True
    tab._on_failed("Traceback…\nboom")
    assert tab._run_btn.isEnabled() and not tab._cancel_btn.isEnabled()


def test_cancel_is_dead_before_anything_runs(tab):
    assert not tab._cancel_btn.isEnabled()


def test_methods_are_checkboxes_and_default_to_mean(tab):
    assert tab._selected_ops() == ["mean"]
    assert set(tab._op_chks) == set(("mean", "median", "sum", "max"))


def test_unticking_every_method_disables_the_run(tab, inputs):
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    tab._refresh_enabled()
    assert tab._run_btn.isEnabled()
    for chk in tab._op_chks.values():
        chk.setChecked(False)
    assert not tab._run_btn.isEnabled()
    assert "at least one" in tab._op_note.text()


def test_the_note_names_every_folder_that_will_be_written(tab):
    tab._op_chks["max"].setChecked(True)
    note = tab._op_note.text()
    assert "dark_subtracted_mean/" in note and "dark_subtracted_max/" in note


def test_selected_methods_reach_the_worker_kwargs(tab, inputs):
    tab._loader._set_explicit_paths([str(p) for p in inputs])
    tab._op_chks["median"].setChecked(True)
    assert tab._worker_kwargs(inputs)["op"] == ["mean", "median"]


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
