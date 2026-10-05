"""The calibration working directory: where it is derived from, and where
intermediate files land inside it.

Calibration scratch used to go either next to the raw data or into a
``mkstemp`` file nobody could find again. It now goes in one deletable
``.midas_scratch/`` folder inside a directory the user named — so these tests
pin both halves: the ladder that proposes that directory, and ``scratch_dir``
refusing to write where it can't.
"""
import os
import stat
import tempfile
from pathlib import Path

import pytest

from conftest import force_rmtree

from midas_gui.helpers import (SCRATCH_DIRNAME, check_output_dir_writable,
                               scratch_dir, session_scratch_dir,
                               correction_subdir,
                               suggest_correction_output_dir,
                               suggest_integration_output_dir,
                               suggest_working_dir)

_SCRATCH: list = []


@pytest.fixture(autouse=True)
def _clean_scratch():
    """Own tempdir rather than ``tmp_path``: pyproject pins
    ``--basetemp=.scratch``, which races with ``--forked`` and errors these
    out in setup. Owning it means owning the cleanup — chmod the read-only
    cases back first, or rmtree can't remove them."""
    yield
    while _SCRATCH:
        force_rmtree(_SCRATCH.pop())
    # session_scratch_dir() cleans itself with atexit, which pytest-forked's
    # os._exit() never reaches — remove it here so the tests that exercise the
    # no-working-directory fallback don't leave a /tmp folder per run.
    from midas_gui import helpers
    if helpers._SESSION_SCRATCH:
        force_rmtree(helpers._SESSION_SCRATCH)
        helpers._SESSION_SCRATCH = None


def _tmpdir() -> Path:
    d = tempfile.mkdtemp(prefix="mg_workdir_")
    _SCRATCH.append(d)
    return Path(d)


# ── The resolution ladder ────────────────────────────────────────────────────

def test_the_canonical_mpe_wf_layout_gives_the_bare_bc_root():
    assert suggest_working_dir(
        "/net/s20iddata/export/park_may26/ge3/sam1/sam1_000001.tif") \
        == Path("/net/s20iddata/export/park_may26_bc")


def test_data_already_inside_a_bc_dir_stays_in_it():
    """The real session path. Read positionally this is only three levels
    below the mount, so the positional rung would call ``export`` the expid
    and propose ``/net/s20iddata/export_bc`` — a sibling of the mount root
    nobody can create. Recognising the ``_bc`` folder the data sits in gets
    the directory that demonstrably *is* writable."""
    p = ("/net/s20iddata/export/s20a/PUP_AML_stubbins_sep26_bc/"
         "PUP_AML_stubbins_sep26_beniwal_gui.h5")
    assert suggest_working_dir(p) == Path(
        "/net/s20iddata/export/s20a/PUP_AML_stubbins_sep26_bc")
    # ...and that is emphatically not what the positional read alone says.
    assert suggest_working_dir(p) != Path("/net/s20iddata/export_bc")


def test_a_bc_ancestor_further_up_is_still_found():
    assert suggest_working_dir("/data/beam_sep26_bc/ge3/sam1/sam1_000001.tif") \
        == Path("/data/beam_sep26_bc")


def test_a_shallow_path_falls_back_to_the_header_expid():
    assert suggest_working_dir("/data/flat/sam1_000001.tif",
                               expid_fallback="park_may26") \
        == Path("/data/flat/park_may26_bc")


def test_a_shallow_path_with_no_expid_proposes_nothing():
    """Better an empty field than a working directory inside the raw-data
    tree — that is exactly the littering this feature exists to stop."""
    assert suggest_working_dir("/data/flat/sam1_000001.tif") is None


def test_no_data_path_proposes_nothing():
    assert suggest_working_dir("") is None


def test_batch_keeps_its_own_froot_detector_tail():
    """The two derivations share a parse but not an answer — Batch still
    wants the full subpath, and still reads it positionally."""
    src = "/net/s20iddata/export/park_may26/ge3/sam1/sam1_000001.tif"
    assert suggest_integration_output_dir(src) == Path(
        "/net/s20iddata/export/park_may26_bc/sam1/ge3")
    assert suggest_working_dir(src) == Path(
        "/net/s20iddata/export/park_may26_bc")


def test_batch_correction_writes_into_batchs_own_output_folder():
    """Batch Correction's per-op folders are leaves of Batch Integrate's own
    output folder, not a parse of their own — so the two tabs can never
    disagree about where a froot's output lives, and the reduced frames sit
    beside the cakes rather than burying them."""
    src = "/net/s20iddata/export/park_may26/ge3/sam1/sam1_000001.tif"
    base = suggest_correction_output_dir(src)
    assert base == Path("/net/s20iddata/export/park_may26_bc/sam1/ge3")
    assert base / correction_subdir("mean") == Path(
        "/net/s20iddata/export/park_may26_bc/sam1/ge3/dark_subtracted_mean")


def test_each_method_gets_its_own_leaf_folder():
    """One run can produce several reductions; a mean and a max of the same
    scan must not land on top of each other."""
    names = {correction_subdir(op) for op in ("mean", "median", "sum", "max")}
    assert names == {"dark_subtracted_mean", "dark_subtracted_median",
                     "dark_subtracted_sum", "dark_subtracted_max"}


def test_batch_correction_keeps_the_detector_segment():
    """Two detectors in one scan must not write into the same folder — the
    output filenames alone would not distinguish them."""
    a = suggest_correction_output_dir(
        "/net/x/export/expt/varexD/sam1/sam1_000001.h5")
    b = suggest_correction_output_dir(
        "/net/x/export/expt/varexE/sam1/sam1_000001.h5")
    assert a != b and a.name == "varexD" and b.name == "varexE"


def test_batch_correction_follows_the_shallow_fallback_too():
    """A flat folder has no positional layout to read; it must still land
    somewhere sensible rather than returning None. It inherits Batch's own
    fallback wholesale, froot segment included."""
    src = "/data/sam1/sam1_000001.h5"
    assert (suggest_correction_output_dir(src, expid_fallback="junetest")
            == suggest_integration_output_dir(src, expid_fallback="junetest")
            == Path("/data/sam1/junetest_bc/sam1"))


def test_batch_correction_proposes_nothing_without_a_data_path():
    assert suggest_correction_output_dir("") is None


# ── Writability ──────────────────────────────────────────────────────────────

def test_an_unwritable_candidate_reports_a_reason_naming_the_blocker():
    d = _tmpdir()
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)      # r-x: can't create inside
    reason = check_output_dir_writable(d / "nested")
    assert reason and str(d) in reason
    assert "writable" in reason


def test_a_writable_candidate_reports_nothing():
    assert check_output_dir_writable(_tmpdir() / "does" / "not" / "exist") is None


# ── scratch_dir ──────────────────────────────────────────────────────────────

def test_scratch_lands_in_one_deletable_folder_inside_the_working_dir():
    work = _tmpdir()
    d = scratch_dir(work, "calib_20260924-120000_sam1")
    assert d.is_dir()
    assert d.parent == work / SCRATCH_DIRNAME
    # Everything the GUI generates is under the one folder the user can bin.
    assert SCRATCH_DIRNAME in d.relative_to(work).parts


def test_two_runs_in_one_working_dir_do_not_share_a_scratch_folder():
    """Fit-time intermediates are generically named (``residual_corr.bin``,
    ``panel_shifts.txt``), so without a per-run leaf the second fit would
    overwrite the first's."""
    work = _tmpdir()
    a = scratch_dir(work, "calib_20260924-120000_sam1")
    b = scratch_dir(work, "calib_20260924-120500_sam1")
    assert a != b and a.is_dir() and b.is_dir()


def test_hydra_panels_in_one_run_do_not_share_a_scratch_folder():
    """Run All starts all four panel workers at once against one cfg."""
    work = _tmpdir()
    run = "calib_20260924-120000"
    dirs = {scratch_dir(work, run, f"ge{n}") for n in (1, 2, 3, 4)}
    assert len(dirs) == 4


def test_an_unwritable_working_dir_raises_rather_than_writing_elsewhere():
    """Loudly, not by silently falling back: a fit that runs and then fails to
    record its residual map is the failure this whole change is about."""
    work = _tmpdir()
    os.chmod(work, stat.S_IRUSR | stat.S_IXUSR)
    with pytest.raises(OSError) as e:
        scratch_dir(work, "calib_run")
    assert str(work) in str(e.value)
    assert not (work / SCRATCH_DIRNAME).exists()


def test_no_working_dir_falls_back_to_a_self_cleaning_session_dir():
    d = scratch_dir(None, "calib_run")
    assert d.is_dir()
    assert str(d).startswith(session_scratch_dir())


def test_the_session_scratch_dir_is_one_per_process():
    assert session_scratch_dir() is session_scratch_dir()


def test_create_false_names_the_folder_without_making_it():
    work = _tmpdir()
    d = scratch_dir(work, "calib_run", create=False)
    assert d == work / SCRATCH_DIRNAME / "calib_run"
    assert not d.exists()


# ── panel_shifts.txt naming ──────────────────────────────────────────────────

def _panel_unpacked():
    torch = pytest.importorskip("torch")
    return {"panel_delta_yz": torch.tensor([[0.1, -0.2], [0.3, 0.4]]),
            "panel_delta_theta": torch.tensor([0.001, -0.002]),
            "panel_delta_lsd": torch.tensor([1.5, -1.5]),
            "panel_delta_p2": torch.tensor([0.0, 0.0])}


_LAYOUT = {"n_y": 1, "n_z": 2, "sy": 100, "sz": 100}


def test_panel_shifts_are_named_after_the_result_not_generically():
    from types import SimpleNamespace
    from midas_gui import calib
    d = _tmpdir()
    r = SimpleNamespace()
    calib._attach_panel_result(r, _panel_unpacked(), _LAYOUT, str(d), stem="ceo2")
    assert Path(r.panel_shifts_path).name == "ceo2_panelshifts.txt"
    assert Path(r.panel_shifts_path).is_file()


def test_two_fits_sharing_a_folder_keep_both_sets_of_shifts():
    """The bug the stem fixes: a generic ``panel_shifts.txt`` meant the second
    fit silently overwrote the first's, and four Hydra panels raced."""
    from types import SimpleNamespace
    from midas_gui import calib
    d = _tmpdir()
    a, b = SimpleNamespace(), SimpleNamespace()
    calib._attach_panel_result(a, _panel_unpacked(), _LAYOUT, str(d), stem="ge1")
    calib._attach_panel_result(b, _panel_unpacked(), _LAYOUT, str(d), stem="ge2")
    assert a.panel_shifts_path != b.panel_shifts_path
    assert sorted(p.name for p in Path(d).glob("*_panelshifts.txt")) == \
        ["ge1_panelshifts.txt", "ge2_panelshifts.txt"]


def test_a_free_form_exp_id_cannot_escape_the_scratch_folder():
    """The stem comes from the Exp ID header field, which is free text."""
    from types import SimpleNamespace
    from midas_gui import calib
    d = _tmpdir()
    r = SimpleNamespace()
    calib._attach_panel_result(r, _panel_unpacked(), _LAYOUT, str(d),
                               stem="../../etc/pwn")
    assert Path(r.panel_shifts_path).parent == d


def test_no_scratch_still_writes_shifts_somewhere_valid():
    """A blank working directory must not make a multi-panel fit fatal —
    missing panel shifts are wrong geometry, not just a coarser one."""
    from types import SimpleNamespace
    from midas_gui import calib
    r = SimpleNamespace()
    calib._attach_panel_result(r, _panel_unpacked(), _LAYOUT, None, stem="ceo2")
    assert Path(r.panel_shifts_path).is_file()
    assert str(r.panel_shifts_path).startswith(session_scratch_dir())
