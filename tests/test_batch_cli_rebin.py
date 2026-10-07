"""Q- and 2θ-uniform output binning survive the background-job path.

Batch Integrate has two run paths that share no code: ``BatchTab._run``
builds a ``BatchRunCoordinator`` in-process, while ``BatchTab._run_as_job``
serialises everything to an argv for a detached
``python -m midas_gui.batch_cli``. Q-uniform was wired into the first only,
and the second simply refused to start ("Q-uniform bins aren't wired into
background jobs yet"). This closes that gap and adds 2θ alongside it, so
these tests drive the real argv builder and feed the result to the real CLI
parser rather than asserting each half in isolation.

Qt import discipline per STATE.md: import inside fixtures, never at module
scope.
"""
import pytest


# ── the CLI half: argv → rebin cfg (pure logic, no Qt) ──────────────────

def _parse(extra):
    from midas_gui import batch_cli
    base = ["--calib-file", "c.json", "--out-dir", "o",
            "--source-type", "hdf5", "--source-path", "scan.h5"]
    return batch_cli._build_arg_parser().parse_args(base + extra)


def test_no_rebin_flags_means_plain_radial():
    """Absent flags are "Bin type = Radial" — no rebin at all, not a rebin
    with defaults."""
    from midas_gui import batch_cli
    assert batch_cli._rebin_cfg(_parse([])) is None


@pytest.mark.parametrize("unit", ["Q", "2th"])
def test_the_flags_become_the_cfg_the_workers_read(unit):
    from midas_gui import batch_cli
    from midas_gui.workers import rebin_cfg_parts
    cfg = batch_cli._rebin_cfg(_parse([
        "--rebin-unit", unit, "--rebin-min", "0.5",
        "--rebin-max", "8.0", "--rebin-step", "0.01"]))
    assert rebin_cfg_parts(cfg) == (unit, 0.5, 8.0, 0.01)


def test_a_unit_without_its_bounds_is_refused():
    """Silently defaulting the grid would produce a plausible-looking
    profile over the wrong range."""
    from midas_gui import batch_cli
    with pytest.raises(SystemExit, match="--rebin-min"):
        batch_cli._rebin_cfg(_parse(["--rebin-unit", "Q"]))


@pytest.mark.parametrize("bad", [
    ["--rebin-step", "0"],          # zero-width bins
    ["--rebin-max", "0.1"],         # max below min
])
def test_a_degenerate_grid_is_refused(bad):
    from midas_gui import batch_cli
    argv = ["--rebin-unit", "Q", "--rebin-min", "0.5",
            "--rebin-max", "8.0", "--rebin-step", "0.01"]
    for i in range(0, len(bad), 2):
        argv[argv.index(bad[i]) + 1] = bad[i + 1]
    with pytest.raises(SystemExit):
        batch_cli._rebin_cfg(_parse(argv))


def test_an_unknown_unit_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        _parse(["--rebin-unit", "d"])


# ── the GUI half: the tab's own settings → argv ─────────────────────────

@pytest.fixture
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(qapp):
    from midas_gui.tab_batch import BatchTab
    return BatchTab()


def _launched_argv(tab, tmp_path, monkeypatch):
    """Drive the real ``_run_as_job`` far enough to capture the argv it would
    hand to ``screen``, stubbing only what needs a real calibration, real
    frames or a real subprocess. Mirrors ``test_batch_cli_omega``."""
    calib = tmp_path / "calib.json"
    calib.write_text("{}")
    tab._use_tab2_btn.setChecked(False)
    tab._json_ed.setText(str(calib))
    tab._out_ed.setText(str(tmp_path / "out"))

    class _Spec:
        TransOpt = ()
    monkeypatch.setattr(tab, "_build_spec", lambda: _Spec())
    monkeypatch.setattr(tab._loader, "source_cfg",
                        lambda: {"type": "hdf5", "path": str(tmp_path / "scan.h5"),
                                 "dataset": "exchange/data", "chunk_size": 25,
                                 "combine_op": "mean",
                                 "frame_start": 0, "frame_end": 1441})
    monkeypatch.setattr(tab._loader, "has_pending_fields", lambda: [])
    monkeypatch.setattr(tab._fmt, "checked_keys", lambda: ["zarr"])

    seen = {}

    def _capture(argv, **_kw):
        seen["argv"] = argv
        return None                 # no job handle: nothing else to drive
    monkeypatch.setattr(tab._job_queue, "launch", _capture)
    tab._run_as_job()
    return seen.get("argv")


def test_radial_sends_no_rebin_flags(tab, tmp_path, monkeypatch):
    tab._bin_type.setCurrentIndex(tab._bin_type.findData("R"))
    argv = _launched_argv(tab, tmp_path, monkeypatch)
    assert argv is not None, "_run_as_job bailed out before launching"
    assert "--rebin-unit" not in argv


@pytest.mark.parametrize("unit,lo,hi,step", [("Q", 0.75, 6.5, 0.004),
                                             ("2th", 1.25, 7.5, 0.005)])
def test_the_launched_argv_carries_the_tabs_own_rebin_settings(
        tab, tmp_path, monkeypatch, unit, lo, hi, step):
    """The whole point in one assertion: what the tab would run in-process
    and what it sends to the background job must be the same config."""
    from midas_gui import batch_cli
    tab._bin_type.setCurrentIndex(tab._bin_type.findData(unit))
    lo_spin, hi_spin, step_spin = tab._REBIN_SPINS[unit](tab)
    lo_spin.setValue(lo); hi_spin.setValue(hi); step_spin.setValue(step)

    argv = _launched_argv(tab, tmp_path, monkeypatch)
    assert argv is not None, "_run_as_job bailed out before launching"
    parsed = batch_cli._build_arg_parser().parse_known_args(argv[3:])[0]
    assert batch_cli._rebin_cfg(parsed) == tab._rebin_cfg()


def test_q_uniform_no_longer_refuses_to_launch(tab, tmp_path, monkeypatch):
    """The reported error: pressing "Run as background job" with Bin type=Q
    popped "Q-uniform bins aren't wired into background jobs yet" and
    launched nothing."""
    from PyQt5 import QtWidgets
    boxes = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: boxes.append(a[1:3])))
    tab._bin_type.setCurrentIndex(tab._bin_type.findData("Q"))
    argv = _launched_argv(tab, tmp_path, monkeypatch)
    assert boxes == [], f"a dialog still blocks the launch: {boxes}"
    assert argv is not None and "--rebin-unit" in argv


@pytest.mark.parametrize("unit", ["Q", "2th"])
def test_multi_azimuth_is_still_refused_with_either_unit(
        tab, tmp_path, monkeypatch, unit):
    """Still a real restriction — the rebin only handles a 1-D profile — so
    it must survive, and name whichever unit is selected."""
    from PyQt5 import QtWidgets
    boxes = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: boxes.append(a[2])))
    tab._bin_type.setCurrentIndex(tab._bin_type.findData(unit))
    tab._multi_azimuth_chk.setChecked(True)
    argv = _launched_argv(tab, tmp_path, monkeypatch)
    assert argv is None, "multi-azimuth + a rebin should not launch"
    assert len(boxes) == 1 and tab._rebin_label() in boxes[0]
