"""Persistent per-run screen logs — ``midas_gui.run_log``.

Two things matter here and neither is the happy path: the file must land in
the SAME tree shape the caking workflow already uses (``<beamline>/<expid>/``,
lowercased — ``tests/test_dark_autodetect`` replays that corpus, and a second
incompatible layout would fragment it), and a log that cannot be written must
never take the run down with it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from midas_gui import run_log


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Redirect LOG_ROOT so no test ever writes into the real corpus."""
    r = tmp_path / "midas_screen_logs"
    monkeypatch.setattr(run_log, "LOG_ROOT", r)
    return r


# ── tree shape ───────────────────────────────────────────────────────────────

def test_directory_matches_the_existing_caking_layout(root):
    """The real tree already holds 20-id-e/pan_jul26/*.screenlog."""
    assert run_log.log_dir("20-ID-E", "pan_jul26") == root / "20-id-e" / "pan_jul26"


def test_profile_is_lowercased_to_match_the_folders_on_disk(root):
    assert run_log.log_dir("20-ID-D", "x").parent.name == "20-id-d"
    assert run_log.log_dir("1-ID-E", "x").parent.name == "1-id-e"


def test_blank_fields_get_obvious_placeholders_not_a_guess(root):
    d = run_log.log_dir("", "")
    assert d == root / run_log.NO_BEAMLINE / run_log.NO_EXPID


@pytest.mark.parametrize("hostile", [
    "../../../etc", "a/b", "20-ID-E/../..", "  ", "..",
])
def test_a_header_field_cannot_steer_the_write_out_of_the_tree(root, hostile):
    """Exp ID is a free-text field in the window header, and this path is a
    shared directory — a slug that escaped it would write somewhere nobody
    is looking, at best."""
    d = run_log.log_dir("20-ID-E", hostile).resolve()
    assert root.resolve() in d.parents


def test_filename_carries_expid_kind_stem_and_a_timestamp(root):
    p = run_log.log_path("correct", profile="20-ID-E", expid="brown_sep26",
                         stem="scan_000021")
    assert p.suffix == ".screenlog"
    assert p.name.startswith("brown_sep26_correct_scan_000021_")
    assert p.parent == root / "20-id-e" / "brown_sep26"


def test_two_runs_of_the_same_files_do_not_collide(root):
    """A timestamp rather than a frame range, because re-running the same
    files to compare settings is a normal thing to do."""
    import time
    a = run_log.log_path("correct", profile="p", expid="e", stem="s")
    time.sleep(1.05)
    b = run_log.log_path("correct", profile="p", expid="e", stem="s")
    assert a != b


# ── writing ──────────────────────────────────────────────────────────────────

def test_lines_are_flushed_as_they_are_written(root):
    """The reason to want this file is usually a run that died, and a
    buffered tail is exactly the part that would be missing."""
    log = run_log.open_log("correct", profile="20-ID-E", expid="e")
    log.write("first")
    assert Path(log.path).read_text() == "first\n"      # readable before close
    log.write("second")
    assert Path(log.path).read_text() == "first\nsecond\n"
    log.close()


def test_header_is_written_in_the_caking_logs_shape(root):
    log = run_log.open_log("correct", profile="20-ID-E", expid="e")
    log.header("MIDAS GUI — Batch Correction", {"Beamline": "20-ID-E"})
    log.close()
    lines = Path(log.path).read_text().splitlines()
    assert lines[0] == "=" * 40
    assert lines[1] == "MIDAS GUI — Batch Correction"
    assert "Beamline     : 20-ID-E" in lines


def test_the_directory_is_created_on_demand(root):
    assert not root.exists()
    log = run_log.open_log("correct", profile="20-ID-E", expid="brand_new")
    log.write("x"); log.close()
    assert (root / "20-id-e" / "brand_new").is_dir()


# ── a log must never break a run ─────────────────────────────────────────────

def test_an_unopenable_log_is_inert_rather_than_an_exception(root, monkeypatch):
    """A log is a record of work, not the work."""
    def boom(*_a, **_k):
        raise PermissionError("read-only share")
    monkeypatch.setattr(Path, "mkdir", boom)
    log = run_log.open_log("correct", profile="20-ID-E", expid="e")
    assert log.path is None
    assert "PermissionError" in (log.error or "")
    log.write("swallowed")        # must not raise
    log.header("t", {"a": 1})     # must not raise
    log.close()


def test_a_write_that_fails_midway_stops_quietly(root):
    log = run_log.open_log("correct", profile="20-ID-E", expid="e")
    log.write("before")
    log._fh.close()               # the share drops out from under us
    log.write("after")            # must not raise
    assert Path(log.path).read_text() == "before\n"


def test_it_works_as_a_context_manager(root):
    with run_log.open_log("correct", profile="20-ID-E", expid="e") as log:
        log.write("x")
        path = log.path
    assert Path(path).read_text() == "x\n"
