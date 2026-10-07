"""Dark auto-detection for Batch Correction — ``frame_correct.resolve_dark``.

Three layers, deliberately:

1. Synthetic unit tests for each rung of the ladder. Self-contained; these
   are the ones that must always run.
2. A replay of the 160 real VAREX scan folders on the beamline share, which
   is where the per-file rule was derived from (and where the folder-wide
   "one bracketing dark per scan" assumption was caught being wrong).
3. A replay of the beamline's own 198 caking screenlogs, which record what
   dark the legacy pipeline actually used for 192 historical runs.

2 and 3 skip rather than fail when their data isn't reachable, so the suite
still runs off this machine — but when the share IS mounted they are the
real proof, because they are the beamline's own choices rather than mine.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from midas_gui.frame_correct import (NO_DARK, dark_candidates, resolve_dark,
                                     split_scan_name)
from midas_gui.helpers import is_dark_like_name

BEAMLINE_DATA = Path("/home/beams/S20IDUSER/mnt/s20a")
SCREEN_LOGS = Path.home() / "midas_runs" / "midas_screen_logs"
DARK_DS = "exchange/data_dark"
DATA_DS = "exchange/data"


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_vrx(path: Path, *, data=None, dark=None, shape=(4, 5)):
    """A minimal VAREX-shaped file: ``exchange/data`` and optionally
    ``exchange/data_dark``, both ``(n, H, W)``."""
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        if data is not None:
            f.create_dataset(DATA_DS, data=np.asarray(data, dtype=np.float32))
        if dark is not None:
            f.create_dataset(DARK_DS, data=np.asarray(dark, dtype=np.float32))
    return path


def _const(value, n=2, shape=(4, 5)):
    return np.full((n,) + shape, float(value), dtype=np.float32)


# ── 1. the ladder, rung by rung ──────────────────────────────────────────────

def test_name_split_recovers_froot_number_and_suffix():
    p = split_scan_name("C611_017Fe_1_load0_008802.vrx.h5")
    assert (p.froot, p.num, p.width, p.suffix) == \
        ("C611_017Fe_1_load0", 8802, 6, ".vrx.h5")


def test_name_split_tolerates_a_name_with_no_number():
    p = split_scan_name("weird.h5")
    assert p.num is None and p.froot == "weird"


def test_nearest_preceding_dark_before_wins(tmp_path):
    """The rule the on-disk survey forced: darks are re-measured mid-scan, so
    the right dark for file N is the most recent one before it — NOT the
    scan's first."""
    for n in (10, 30):
        _write_vrx(tmp_path / f"scan_{n:06d}_dark_before.vrx.h5", dark=_const(n))
    # Name them the way the beamline does: <froot>_dark_before_<N>.
    for p in tmp_path.glob("*"):
        p.unlink()
    _write_vrx(tmp_path / "scan_dark_before_000010.vrx.h5", dark=_const(1.0))
    _write_vrx(tmp_path / "scan_dark_before_000030.vrx.h5", dark=_const(7.0))
    data = _write_vrx(tmp_path / "scan_000035.vrx.h5", data=_const(100.0))

    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert np.allclose(arr, 7.0), "took the first dark of the scan, not the nearest"
    assert "dark_before_000030" in why


def test_dark_after_used_when_no_dark_precedes(tmp_path):
    _write_vrx(tmp_path / "scan_dark_after_000050.vrx.h5", dark=_const(3.0))
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert np.allclose(arr, 3.0) and "dark_after" in why


def test_own_data_dark_when_no_sibling(tmp_path):
    """The self-dark convention — 102 of the 160 folders on disk use it."""
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5",
                      data=_const(100.0), dark=_const(5.0))
    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert np.allclose(arr, 5.0) and "own" in why


def test_sibling_outranks_own_data_dark(tmp_path):
    """Both present: the explicitly-acquired bracketing dark wins, which is
    what the legacy pipeline did in every log that had one."""
    _write_vrx(tmp_path / "scan_dark_before_000019.vrx.h5", dark=_const(2.0))
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5",
                      data=_const(100.0), dark=_const(5.0))
    arr, _why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert np.allclose(arr, 2.0)


def test_prefixed_dark_outranks_folder_level_dark(tmp_path):
    _write_vrx(tmp_path / "dark_before_000019.vrx.h5", dark=_const(9.0))
    _write_vrx(tmp_path / "scan_dark_before_000018.vrx.h5", dark=_const(2.0))
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert np.allclose(arr, 2.0), why


def test_bare_folder_level_dark_is_still_found(tmp_path):
    """The one ``Dark root : dark_before`` outlier in the log corpus."""
    _write_vrx(tmp_path / "dark_before_000019.vrx.h5", dark=_const(9.0))
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    arr, _why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert np.allclose(arr, 9.0)


@pytest.mark.parametrize("odd", ["scan_real_dark_before_000019.vrx.h5",
                                 "scan_repeat_dark_before_000019.vrx.h5"])
def test_real_and_repeat_dark_variants_are_matched(tmp_path, odd):
    """``_real_dark_before`` / ``_repeat_dark_before`` — the other two log
    outliers. They rank as a plain dark (no ``dark_before`` direction match
    on the froot split), but they must still be FOUND."""
    _write_vrx(tmp_path / odd, dark=_const(4.0))
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    arr, _why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert np.allclose(arr, 4.0)


def test_wrong_shape_sibling_falls_through_to_the_next_rung(tmp_path):
    _write_vrx(tmp_path / "scan_dark_before_000019.vrx.h5",
               dark=np.zeros((2, 9, 9), np.float32))
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5",
                      data=_const(100.0), dark=_const(5.0))
    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert np.allclose(arr, 5.0), why


def test_loader_fallback_used_when_nothing_else_resolves(tmp_path):
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5),
                            fallback=np.full((4, 5), 6.0, np.float32))
    assert np.allclose(arr, 6.0) and "loader" in why.lower()


def test_mismatched_loader_fallback_is_skipped_and_said_so(tmp_path):
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5),
                            fallback=np.zeros((9, 9), np.float32))
    assert arr is None and "skipped" in why


def test_no_dark_anywhere_is_reported_not_raised(tmp_path):
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5))
    assert arr is None and why == NO_DARK


def test_auto_off_ignores_siblings_entirely(tmp_path):
    _write_vrx(tmp_path / "scan_dark_before_000019.vrx.h5", dark=_const(2.0))
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    arr, why = resolve_dark(data, dark_dataset=DARK_DS, shape=(4, 5), auto=False,
                            fallback=np.full((4, 5), 6.0, np.float32))
    assert np.allclose(arr, 6.0) and "loader" in why.lower()


def test_candidate_order_is_stable_regardless_of_directory_order(tmp_path):
    for n in (5, 12, 19):
        _write_vrx(tmp_path / f"scan_dark_before_{n:06d}.vrx.h5", dark=_const(n))
    data = _write_vrx(tmp_path / "scan_000020.vrx.h5", data=_const(100.0))
    first = [p.name for p, _ in dark_candidates(data)]
    assert first == sorted(first, key=lambda n: -split_scan_name(n).num)
    assert first[0].endswith("000019.vrx.h5")


# ── 2. replay: the real scan folders on the beamline share ───────────────────

def _scan_folders():
    try:
        return sorted(BEAMLINE_DATA.glob("*/varex*/*/"))
    except OSError:
        return []


@pytest.mark.skipif(not BEAMLINE_DATA.is_dir(),
                    reason="beamline share not mounted")
def test_every_real_data_file_gets_its_nearest_preceding_dark():
    """Replay all 160 VAREX scan folders currently on the share.

    Name-level only — no pixels are read, so this is cheap and stays honest
    about what it checks: WHICH FILE the ladder picks. At the time of
    writing that is 936 data files across 56 dark-bearing folders, and the
    expected answer is the nearest preceding ``dark_before`` for every
    single one.
    """
    checked = mismatches = 0
    for folder in _scan_folders():
        try:
            files = [p for p in folder.iterdir()
                     if p.is_file() and p.name.endswith(".h5")]
        except OSError:
            continue
        befores = sorted(
            (split_scan_name(p.name).num, p) for p in files
            if is_dark_like_name(p) and "dark_before" in p.name.lower()
            and split_scan_name(p.name).num is not None)
        if not befores:
            continue
        for p in files:
            if is_dark_like_name(p):
                continue
            num = split_scan_name(p.name).num
            if num is None:
                continue
            preceding = [q for n, q in befores if n < num]
            if not preceding:
                continue
            cands = dark_candidates(p)
            checked += 1
            if not cands or cands[0][0] != preceding[-1]:
                mismatches += 1
    # Mismatches are a code defect and always fail. The SIZE of the corpus
    # is not ours to control: the share is live and experiments are archived
    # off it (PUP_AML_stubbins_sep26 alone was 936 of these files and is
    # gone), so a shrunken corpus is environmental churn, not a regression.
    # Report it and skip rather than failing the suite for it — the
    # synthetic tests above carry the actual guarantee; this replay is
    # corroboration from real data while real data is there.
    assert mismatches == 0, f"{mismatches} of {checked} real files got the wrong dark"
    if checked < 25:
        pytest.skip(f"only {checked} dark-bearing files on the share — "
                    f"too few for this replay to corroborate much")


@pytest.mark.skipif(not BEAMLINE_DATA.is_dir(),
                    reason="beamline share not mounted")
def test_real_dark_less_folders_fall_through_to_the_self_dark():
    """The other half of the share: folders with no dark sibling at all must
    produce no candidate, so the ladder reaches the file's own data_dark."""
    seen = 0
    for folder in _scan_folders():
        try:
            files = [p for p in folder.iterdir()
                     if p.is_file() and p.name.endswith(".h5")]
        except OSError:
            continue
        if not files or any(is_dark_like_name(p) for p in files):
            continue
        seen += 1
        assert dark_candidates(files[0]) == []
    if seen == 0:
        pytest.skip("no dark-less scan folders on the share right now")


# ── 3. replay: the beamline's own screenlogs ─────────────────────────────────

_ESC = re.compile(r"\x1b\[[0-9;]*m")
_DATA_RE = re.compile(r"^Data file\s*:\s*(\S+)", re.M)
_DARK_RE = re.compile(r"^\*\*\* dark file\s*:\s*(\S+)", re.M)
_LOC_RE = re.compile(r"^Dark loc\s*:\s*(\S+)", re.M)

#: The one historical run whose dark is NOT inferable from filenames, listed
#: by name so a second one can never slip in unnoticed. Its operator passed an
#: explicit ``-P`` override pointing at a DIFFERENT SCAN's folder
#: (``test_JAsample_ysweep/``) and at a file carrying no dark marker in its
#: name at all (``test_JAsample_ysweep_021957.vrx.h5``). No filesystem rule
#: can recover that choice, and none should try to guess it: a deliberate
#: cross-scan dark is exactly what the loader's own Dark field — and
#: ``resolve_dark(auto=False)`` — exist for.
MANUAL_OVERRIDE_LOGS = {
    "pan_jul26_caking_s20varex2_test_JAsample_114C_ysweep_21960-21960.screenlog",
}


def _log_pairs():
    """``[(log name, data path, dark path, dark loc)]`` from every screenlog."""
    out = []
    for log in sorted(SCREEN_LOGS.glob("*.screenlog")):
        text = _ESC.sub("", log.read_bytes().decode("utf-8", "replace")).replace("\r", "\n")
        data, dark = _DATA_RE.search(text), _DARK_RE.search(text)
        if not (data and dark):
            continue
        loc = _LOC_RE.search(text)
        out.append((log.name, Path(data.group(1)), Path(dark.group(1)),
                    loc.group(1) if loc else None))
    return out


@pytest.mark.skipif(not SCREEN_LOGS.is_dir(), reason="screenlog corpus not present")
def test_logged_runs_all_read_the_dark_from_exchange_data_dark():
    """The assumption the whole ladder rests on: there is no separate dark
    image format to support, only the question of which FILE holds
    ``exchange/data_dark``."""
    pairs = _log_pairs()
    if not pairs:
        pytest.skip("no resolvable runs in the screenlog corpus")
    locs = {loc for _n, _d, _k, loc in pairs}
    assert locs == {"/" + DARK_DS}, f"unexpected dark locations in the logs: {locs}"


@pytest.mark.skipif(not SCREEN_LOGS.is_dir(), reason="screenlog corpus not present")
def test_ladder_reproduces_each_logged_dark_choice(tmp_path):
    """Replay every historical run: rebuild its folder from the names the log
    records, and check the ladder picks the dark the beamline picked.

    The source data for these runs is long archived, so the folder is
    reconstructed rather than read — which bounds the claim to the ranking,
    the part that can actually be wrong. A run whose dark IS its data file
    (the self-dark half of the corpus) must produce no sibling candidate, so
    that the ladder falls through to rung 2.

    :data:`MANUAL_OVERRIDE_LOGS` is excluded by name, not by a loosened
    assertion — see that constant for why one run is genuinely out of reach.
    """
    pairs = _log_pairs()
    if not pairs:
        pytest.skip("no resolvable runs in the screenlog corpus")
    wrong = []
    inferable = [row for row in pairs if row[0] not in MANUAL_OVERRIDE_LOGS]
    assert len(pairs) - len(inferable) == len(MANUAL_OVERRIDE_LOGS), \
        "MANUAL_OVERRIDE_LOGS names a log that is no longer in the corpus"
    for i, (name, data, dark, _loc) in enumerate(inferable):
        folder = tmp_path / f"run{i:03d}"
        folder.mkdir()
        (folder / data.name).touch()
        if dark.name != data.name:
            (folder / dark.name).touch()
        cands = dark_candidates(folder / data.name)
        if dark.name == data.name:
            if cands:
                wrong.append((name, "expected self-dark", cands[0][0].name))
        elif not cands or cands[0][0].name != dark.name:
            wrong.append((name, dark.name, cands[0][0].name if cands else None))
    assert not wrong, \
        f"{len(wrong)} of {len(inferable)} logged runs disagreed: {wrong[:5]}"
    assert len(inferable) > 150, \
        f"only {len(inferable)} runs replayed — log corpus looks truncated"


@pytest.mark.skipif(not SCREEN_LOGS.is_dir(), reason="screenlog corpus not present")
def test_the_manual_override_run_is_genuinely_not_inferable(tmp_path):
    """Guard the exemption itself: confirm the excluded run's dark really is
    unreachable from filenames, so MANUAL_OVERRIDE_LOGS can't quietly become
    a dumping ground for cases the ladder merely gets wrong."""
    pairs = {name: (data, dark) for name, data, dark, _loc in _log_pairs()}
    for name in MANUAL_OVERRIDE_LOGS:
        if name not in pairs:
            pytest.skip(f"{name} not in the corpus")
        data, dark = pairs[name]
        assert dark.parent.name != data.parent.name, \
            "dark is in the same folder after all — this one IS inferable"
        assert not is_dark_like_name(dark), \
            "dark carries a dark marker after all — this one IS inferable"
