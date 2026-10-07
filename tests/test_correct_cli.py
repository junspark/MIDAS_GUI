"""Batch Correction survives the background-job path, not just the in-process one.

Batch Correction has two run paths that share no code: ``_run`` builds a
``BatchCorrectionCoordinator`` in this process, while ``_run_as_job``
serialises everything to an argv and hands it to a detached
``python -m midas_gui.correct_cli``. The failure mode this file exists to
prevent is the one ``test_batch_cli_omega`` documents for the integration
side: a setting wired into one path and silently absent from the other, so
a background job writes different files than the button beside it, and
nothing can report the difference.

So the important test is
``test_the_launched_argv_reproduces_the_tabs_own_worker_kwargs``: it drives
the real argv builder and feeds what comes out to the real CLI parser,
closing the loop rather than asserting each half in isolation.

Qt import discipline per STATE.md: import inside fixtures, never at module
scope.
"""
import numpy as np
import pytest

SHAPE = (6, 5)


# ── the CLI half: argv → worker kwargs (no Qt) ──────────────────────────

def _parse(extra):
    from midas_gui import correct_cli
    base = ["--paths", "a.h5", "--out-dir", "o"]
    return correct_cli._build_arg_parser().parse_args(base + extra)


def test_defaults_match_the_tabs_defaults():
    from midas_gui import correct_cli
    from midas_gui.helpers import CORRECTION_EXT, CORRECTION_SUFFIX
    kw = correct_cli._worker_kwargs(_parse([]))
    assert kw["dataset"] == "exchange/data"
    assert kw["out_dataset"] == "exchange/data"
    assert kw["suffix"] == CORRECTION_SUFFIX
    assert kw["out_ext"] == CORRECTION_EXT
    assert kw["dark_dataset"] == "exchange/data_dark"
    assert kw["out_dtype"] == "float32"
    assert kw["auto_dark"] is True and kw["clip_negatives"] is True


def test_chunk_size_zero_means_the_whole_file():
    """0 is the "no chunking" sentinel on the command line; the worker
    expects None for it, the same translation the tab makes."""
    from midas_gui import correct_cli
    assert correct_cli._worker_kwargs(_parse(["--chunk-size", "0"]))["chunk_size"] is None
    assert correct_cli._worker_kwargs(_parse(["--chunk-size", "25"]))["chunk_size"] == 25


def test_every_method_can_be_requested_at_once():
    from midas_gui import correct_cli
    from midas_gui.frame_correct import OPS
    kw = correct_cli._worker_kwargs(_parse(["--ops", ",".join(OPS)]))
    assert kw["op"] == list(OPS)


def test_an_unknown_method_is_refused_by_name():
    """Silently dropping it would produce a run that looks fine and is
    missing an output set."""
    from midas_gui import correct_cli
    with pytest.raises(SystemExit, match="banana"):
        correct_cli._worker_kwargs(_parse(["--ops", "mean,banana"]))


def test_an_empty_method_list_is_refused():
    from midas_gui import correct_cli
    with pytest.raises(SystemExit, match="at least one method"):
        correct_cli._worker_kwargs(_parse(["--ops", ","]))


@pytest.mark.parametrize("flag,key,value", [
    ("--no-auto-dark", "auto_dark", False),
    ("--no-clip-negatives", "clip_negatives", False),
    ("--shuffle", "shuffle", True),
])
def test_the_boolean_flags_reach_the_worker(flag, key, value):
    from midas_gui import correct_cli
    assert correct_cli._worker_kwargs(_parse([flag]))[key] is value


def test_ion_csv_extras_round_trip():
    from midas_gui import correct_cli
    kw = correct_cli._worker_kwargs(_parse(["--ion-csv-extras", "env,motors"]))
    assert set(kw["ion_csv_extras"]) == {"env", "motors"}
    assert correct_cli._worker_kwargs(_parse([]))["ion_csv_extras"] == ()


def test_a_field_image_is_loaded_from_its_path(tmp_path):
    """Field images cannot travel on a command line, so the tab writes them
    beside the output and passes paths. This is the reading half."""
    tifffile = pytest.importorskip("tifffile")
    from midas_gui import correct_cli
    f = tmp_path / "_bg_job_dark.tif"
    tifffile.imwrite(str(f), np.full(SHAPE, 7.0, np.float32))
    kw = correct_cli._worker_kwargs(_parse(["--dark", str(f)]))
    assert kw["dark"] is not None
    assert np.allclose(np.asarray(kw["dark"]), 7.0)
    assert correct_cli._worker_kwargs(_parse([]))["dark"] is None


# ── the GUI half: the tab's settings → argv → the parser ────────────────

@pytest.fixture
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(qapp):
    from midas_gui.tab_batch_correct import BatchCorrectionTab
    return BatchCorrectionTab()


def _h5(path):
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("exchange/data",
                         data=np.zeros((8,) + SHAPE, np.float32))
    return path


def _launched_argv(tab, tmp_path, monkeypatch, paths=None):
    """Drive the real ``_run_as_job`` far enough to capture the argv it
    would hand to ``screen``, stubbing only the subprocess."""
    paths = paths or [_h5(tmp_path / "scan" / "scan_000010.vrx.h5")]
    tab._out_ed.setText(str(tmp_path / "out"))
    monkeypatch.setattr(tab, "_h5_paths", lambda: [str(p) for p in paths])
    seen = {}

    def _capture(argv, **_kw):
        seen["argv"] = argv
        return None                 # no job handle: nothing else to drive
    monkeypatch.setattr(tab._job_queue, "launch", _capture)
    tab._run_as_job()
    return seen.get("argv")


def test_the_launched_argv_reproduces_the_tabs_own_worker_kwargs(
        tab, tmp_path, monkeypatch):
    """The whole point in one assertion: what the tab would run in-process
    and what it sends to the background job must be the same configuration."""
    from midas_gui import correct_cli
    for op, chk in tab._op_chks.items():
        chk.setChecked(op in ("mean", "max"))
    tab._clip_chk.setChecked(False)
    tab._auto_dark_chk.setChecked(True)
    tab._shuffle_chk.setChecked(True)

    paths = [_h5(tmp_path / "scan" / "scan_000010.vrx.h5")]
    argv = _launched_argv(tab, tmp_path, monkeypatch, paths)
    assert argv is not None, "_run_as_job bailed out before launching"
    parsed = correct_cli._build_arg_parser().parse_known_args(argv[3:])[0]
    cli = correct_cli._worker_kwargs(parsed)
    gui = tab._worker_kwargs([str(p) for p in paths])

    # Field images are compared by content, not identity -- the CLI reloads
    # them from the TIFFs the tab just wrote.
    for key in ("dataset", "chunk_size", "op", "out_dataset", "suffix",
                "out_ext", "bright_mode", "auto_dark", "dark_dataset",
                "clip_negatives", "shuffle", "out_dtype", "level",
                "raw_start", "raw_end"):
        assert cli[key] == gui[key], f"{key}: {cli[key]!r} != {gui[key]!r}"
    assert set(cli["ion_csv_extras"]) == set(gui["ion_csv_extras"])
    assert (cli["compression"] or "none") == (gui["compression"] or "none")


def test_the_argv_carries_the_run_mode_worker_count(tab, tmp_path, monkeypatch):
    tab._run_mode.setCurrentIndex(tab._run_mode.findData("parallel"))
    tab._n_workers.setValue(3)
    argv = _launched_argv(tab, tmp_path, monkeypatch)
    assert argv is not None
    assert argv[argv.index("--n-workers") + 1] == "3"


def test_sequential_sends_one_worker(tab, tmp_path, monkeypatch):
    tab._run_mode.setCurrentIndex(tab._run_mode.findData("sequential"))
    tab._n_workers.setValue(8)       # ignored unless the mode is parallel
    argv = _launched_argv(tab, tmp_path, monkeypatch)
    assert argv[argv.index("--n-workers") + 1] == "1"


def test_the_field_images_are_written_beside_the_output(tab, tmp_path, monkeypatch):
    pytest.importorskip("tifffile")
    monkeypatch.setattr(tab._loader, "dark", lambda: np.full(SHAPE, 3.0, np.float32))
    argv = _launched_argv(tab, tmp_path, monkeypatch)
    assert argv is not None
    f = tmp_path / "out" / "_bg_job_dark.tif"
    assert f.is_file(), "the dark was not materialised for the detached job"
    assert argv[argv.index("--dark") + 1] == str(f)


def test_no_output_folder_refuses_before_launching(tab, tmp_path, monkeypatch):
    """A detached job has nowhere to put its field images or results list."""
    from midas_gui import tab_batch_correct as mod
    errs = []
    monkeypatch.setattr(mod, "show_error", lambda *a, **k: errs.append(a[1:]))
    monkeypatch.setattr(tab, "_h5_paths",
                        lambda: [str(_h5(tmp_path / "s" / "s_000010.vrx.h5"))])
    monkeypatch.setattr(tab._job_queue, "launch",
                        lambda *a, **k: pytest.fail("launched with no out dir"))
    tab._out_ed.setText("")
    tab._run_as_job()
    assert errs, "no error was reported"


def test_no_method_selected_refuses_before_launching(tab, tmp_path, monkeypatch):
    from midas_gui import tab_batch_correct as mod
    errs = []
    monkeypatch.setattr(mod, "show_error", lambda *a, **k: errs.append(a[1:]))
    for chk in tab._op_chks.values():
        chk.setChecked(False)
    monkeypatch.setattr(tab._job_queue, "launch",
                        lambda *a, **k: pytest.fail("launched with no method"))
    _launched_argv(tab, tmp_path, monkeypatch)
    assert errs, "no error was reported"


def test_a_tiff_source_is_refused_with_an_explanation(tab, tmp_path, monkeypatch):
    """Chunks never cross a file boundary, and a TIFF holds one frame — so
    there is nothing within a file to combine. Same refusal the in-process
    run gives."""
    from midas_gui import tab_batch_correct as mod
    errs = []
    monkeypatch.setattr(mod, "show_error", lambda *a, **k: errs.append(a[1:]))
    monkeypatch.setattr(tab, "_h5_paths", lambda: [])
    monkeypatch.setattr(tab._loader, "source_cfg", lambda: {"type": "tiff_glob"})
    monkeypatch.setattr(tab._job_queue, "launch",
                        lambda *a, **k: pytest.fail("launched for a TIFF source"))
    tab._out_ed.setText(str(tmp_path / "out"))
    tab._run_as_job()
    assert errs and "HDF5" in errs[0][0]


# ── the job queue has to recognise this job's output ────────────────────

def test_the_queue_parses_this_clis_progress_lines():
    """correct_cli tags its lines [correct]; the queue's regex was written
    for [batch] only, so a correction job would have shown no progress."""
    from midas_gui.job_queue import _PROGRESS_RE
    m = _PROGRESS_RE.search("[correct] PROGRESS 7/320")
    assert m and m.groups() == ("7", "320")
    assert _PROGRESS_RE.search("[batch] PROGRESS 3/10"), "integration jobs broke"
