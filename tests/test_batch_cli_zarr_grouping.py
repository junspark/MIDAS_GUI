"""Zarr grouping survives the background-job path, not just the in-process one.

Same two-run-path hazard the omega tests exist for (see
``tests/test_batch_cli_omega.py``): ``BatchTab._run`` constructs a
``BatchRunCoordinator`` in-process, while ``BatchTab._run_as_job`` serialises
everything to an argv for a detached ``python -m midas_gui.batch_cli``. A
setting wired into only the first is not an error anywhere — the job just
quietly does something else, and here "something else" is 435 single-cake
archives instead of the 3 that were asked for.

So the important test is
``test_the_launched_argv_carries_the_tabs_own_grouping``: it drives the real
argv builder and feeds the result to the real CLI parser, closing the loop
rather than asserting each half separately.

Qt import discipline per STATE.md: import inside fixtures, never at module
scope.
"""
import pytest


# ── the CLI half: argv -> grouping (pure logic, no Qt) ──────────────────

def _parse(extra):
    from midas_gui import batch_cli
    base = ["--calib-file", "c.json", "--out-dir", "o",
            "--source-type", "hdf5", "--source-path", "scan.h5"]
    return batch_cli._build_arg_parser().parse_args(base + extra)


def test_the_cli_default_is_the_historical_per_frame_behaviour():
    """Every project and script written before grouping existed ran one
    archive per output frame; the default must keep doing that."""
    assert _parse([]).zarr_grouping == "frame"


@pytest.mark.parametrize("mode", ["frame", "file", "run"])
def test_each_grouping_mode_is_accepted(mode):
    assert _parse(["--zarr-grouping", mode]).zarr_grouping == mode


def test_an_unknown_grouping_is_rejected_rather_than_silently_ignored():
    """argparse choices, not a free string: a typo that fell through to the
    worker's own fallback would silently write per-frame archives again."""
    with pytest.raises(SystemExit):
        _parse(["--zarr-grouping", "per-file"])


def test_the_cli_hands_the_grouping_to_the_worker():
    """The flag existing is not the point — reaching ``BatchWorker`` is.
    ``main`` is too heavy to call here (it opens the source and integrates),
    so this pins the construction site by reading it."""
    import inspect
    from midas_gui import batch_cli
    assert "zarr_grouping=args.zarr_grouping" in inspect.getsource(batch_cli.main)


def test_the_worker_falls_back_to_frame_for_a_value_it_does_not_know():
    """Defence in depth for the non-CLI callers (Hydra, tests, a project
    restored from a newer version): an unrecognised mode must degrade to the
    documented default, never crash a run that has already read 70 GB."""
    pytest.importorskip("PyQt5.QtWidgets")
    from midas_gui.workers import BatchWorker
    w = BatchWorker.__new__(BatchWorker)
    BatchWorker.__init__(
        w, None, {"type": "tiff_glob", "path": "x"}, None, None, ["zarr"],
        "subpixel2", (None, None), None, zarr_grouping="per-file")
    assert w._zarr_grouping == "frame"


# ── the GUI half: the tab's combo -> argv -> back to the same key ───────

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt5.QtWidgets")
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(qapp):
    from midas_gui.tab_batch import BatchTab
    return BatchTab()


def _launched_argv(tab, tmp_path, monkeypatch):
    """Drive the real ``_run_as_job`` far enough to capture the argv it would
    hand to ``screen``, stubbing only what needs a real calibration, real
    frames or a real subprocess."""
    calib = tmp_path / "calib.json"
    calib.write_text("{}")
    tab._use_tab2_btn.setChecked(False)
    tab._json_ed.setText(str(calib))
    tab._out_ed.setText(str(tmp_path / "out"))

    class _Spec:
        TransOpt = ()
    monkeypatch.setattr(tab, "_build_spec", lambda: _Spec())
    monkeypatch.setattr(tab._loader, "source_cfg",
                        lambda: {"type": "hdf5_stack_glob",
                                 "paths": [str(tmp_path / "a.h5")],
                                 "dataset": "exchange/data", "chunk_size": 10,
                                 "combine_op": "mean"})
    monkeypatch.setattr(tab._loader, "has_pending_fields", lambda: [])
    monkeypatch.setattr(tab._fmt, "checked_keys", lambda: ["zarr"])

    seen = {}

    def _capture(argv, **_kw):
        seen["argv"] = argv
        return None
    monkeypatch.setattr(tab._job_queue, "launch", _capture)
    tab._run_as_job()
    assert "argv" in seen, "_run_as_job bailed out before launching"
    return seen["argv"]


@pytest.mark.parametrize("mode", ["frame", "file", "run"])
def test_the_launched_argv_carries_the_tabs_own_grouping(tab, tmp_path,
                                                         monkeypatch, mode):
    """The whole hazard in one assertion: what the tab would run in-process
    and what it sends to the background job must agree."""
    from midas_gui import batch_cli
    idx = tab._zarr_grouping.findData(mode)
    assert idx >= 0, f"no combo entry for {mode!r}"
    tab._zarr_grouping.setCurrentIndex(idx)

    argv = _launched_argv(tab, tmp_path, monkeypatch)
    parsed = batch_cli._build_arg_parser().parse_known_args(argv[3:])[0]
    assert parsed.zarr_grouping == tab._zarr_grouping_key() == mode


def test_the_grouping_is_always_on_the_command_line(tab, tmp_path, monkeypatch):
    """Even at the default. The launched command line is echoed into the Logs
    tab and is the only place a user can see what a detached job will write."""
    assert "--zarr-grouping" in _launched_argv(tab, tmp_path, monkeypatch)


def test_the_tab_defaults_to_per_frame(tab):
    assert tab._zarr_grouping_key() == "frame"


def test_the_combo_is_disabled_until_zarr_is_a_chosen_format(tab):
    """A control that silently does nothing is worse than one that says why
    it cannot act, so the combo follows the Zarr checkbox."""
    tab._fmt.set_state(["csv"])
    assert not tab._zarr_grouping.isEnabled()
    tab._fmt.set_state(["csv", "zarr"])
    assert tab._zarr_grouping.isEnabled()
