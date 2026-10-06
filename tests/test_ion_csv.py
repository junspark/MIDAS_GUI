"""Per-frame beam-monitor CSV — ``midas_gui.ion_csv``.

The interesting behaviour is all at the edges, because "no reading" has three
genuinely different meanings at 20-ID (channel not declared for the hutch,
declared but absent from the file, present but not live) and conflating them
would either hide a real monitor or invent one. See the module docstring.

Pure numerics and file I/O — no Qt, so no ``forked`` marker needed here.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_gui import ion_csv as IC


def _write(tmp_path, rows, extras=()):
    """``(header list, data rows as lists)``, or None when nothing was written."""
    out = IC.write_ion_csv(tmp_path / "m.csv", rows, extras=extras)
    if out is None:
        return None
    lines = [ln for ln in open(out).read().splitlines() if ln]
    return lines[0].split(","), [ln.split(",") for ln in lines[1:]]


# ── hutch resolution ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,want", [
    ("/data/s20a/varexE/run_001.h5", "E"),
    ("/data/s20a/VAREXD/run_001.h5", "D"),
    ("/data/s20a/varexd_nested/x/run.h5", "D"),
    ("/data/s20a/something_else/run.h5", None),
    ("", None),
    (None, None),
])
def test_hutch_is_read_from_the_path(path, want):
    """The file's own active_instrument is empty upstream, so the path is the
    only signal there is."""
    assert IC.resolve_hutch(path) == want


def test_station_a_has_no_monitor_mapping():
    """A's IC4_foil_I0/IC5_foil_I1 names look like a pair but no sample sits
    in its beam path, so claiming them would be worse than claiming nothing."""
    assert "A" not in IC.ION_CHAMBER_H5_PATHS
    assert IC.metadata_h5_paths("A") == IC.METADATA_H5_PATHS


def test_d_hutch_declares_no_transmission_channel():
    assert "ion_chamber_i" not in IC.ION_CHAMBER_H5_PATHS["D"]
    assert "ion_chamber_i0" in IC.ION_CHAMBER_H5_PATHS["D"]


# ── the three absence modes ──────────────────────────────────────────────────

def test_absent_key_drops_the_column_entirely(tmp_path):
    """D hutch: the transmitted channel does not exist. A column of blanks
    would read as a broken detector rather than a station without one."""
    header, _rows = _write(tmp_path, IC.rows_from_metas(
        [{"ion_chamber_i0": 900.0}, {"ion_chamber_i0": 905.0}]))
    assert header == ["frame", "I0"]


def test_none_valued_key_also_drops_the_column(tmp_path):
    """E hutch with no D2PD in the file — the documented auto-detect."""
    header, _rows = _write(tmp_path, IC.rows_from_metas(
        [{"ion_chamber_i0": 900.0, "ion_chamber_i": None}] * 2))
    assert header == ["frame", "I0"]


def test_an_all_nan_channel_is_dropped(tmp_path):
    """E hutch's inactive SMS sub-config reads NaN on every frame; the data
    itself is what says which of HL/HR was live."""
    rows = IC.rows_from_metas([
        {"ion_chamber_i0": 1.0, "motor:HL/samX": 1.5, "motor:HR/samX": np.nan},
        {"ion_chamber_i0": 1.0, "motor:HL/samX": 1.6, "motor:HR/samX": np.nan}])
    header, _ = _write(tmp_path, rows, extras=("motors",))
    assert "HL/samX" in header and "HR/samX" not in header


def test_a_partly_nan_channel_keeps_its_column_and_says_nan(tmp_path):
    """Dropping a column because SOME frames are NaN would lose the frames
    that did read — and a blank there would be indistinguishable from None."""
    rows = IC.rows_from_metas([{"ion_chamber_i0": 1.0, "motor:HL/samX": np.nan},
                               {"ion_chamber_i0": 1.0, "motor:HL/samX": 2.0}])
    header, data = _write(tmp_path, rows, extras=("motors",))
    col = header.index("HL/samX")
    assert [r[col] for r in data] == ["nan", "2"]


# ── transmission ─────────────────────────────────────────────────────────────

def test_transmission_is_computed_from_both_chambers(tmp_path):
    header, data = _write(tmp_path, IC.rows_from_metas(
        [{"ion_chamber_i0": 1000.0, "ion_chamber_i": 250.0}]))
    assert header == ["frame", "I0", "I", "transmission"]
    assert data[0] == ["0", "1000", "250", "0.25"]


def test_a_dropped_beam_is_blank_not_a_division_error(tmp_path):
    """I0 == 0 is a dropped beam, not infinite transmission."""
    header, data = _write(tmp_path, IC.rows_from_metas(
        [{"ion_chamber_i0": 0.0, "ion_chamber_i": 5.0},
         {"ion_chamber_i0": 100.0, "ion_chamber_i": 5.0}]))
    col = header.index("transmission")
    assert [r[col] for r in data] == ["", "0.05"]


# ── optional column groups ───────────────────────────────────────────────────

def test_extras_are_opt_in(tmp_path):
    meta = {"ion_chamber_i0": 1.0, "current": 102.0, "motor:HR/samX": 3.0}
    base, _ = _write(tmp_path, IC.rows_from_metas([meta]))
    assert base == ["frame", "I0"]
    env, _ = _write(tmp_path, IC.rows_from_metas([meta]), extras=("env",))
    assert env == ["frame", "I0", "ring_current_mA"]
    both, _ = _write(tmp_path, IC.rows_from_metas([meta]),
                     extras=("env", "motors"))
    assert both == ["frame", "I0", "ring_current_mA", "HR/samX"]


def test_ring_current_is_not_confused_with_a_beam_monitor(tmp_path):
    """It squatted in the writer's "I" slot once already; that bug is what
    produced files reading I = 200.025 mA. It is an env column, nothing more."""
    header, _ = _write(tmp_path, IC.rows_from_metas([{"current": 102.0}]),
                       extras=("env",))
    assert "I" not in header and "ring_current_mA" in header


# ── nothing worth writing ────────────────────────────────────────────────────

def test_no_file_when_there_is_no_real_data(tmp_path):
    """An unrecognised-path run must not litter the output folder with a file
    containing only a frame counter."""
    assert _write(tmp_path, IC.rows_from_metas([{}, {}])) is None
    assert not list(tmp_path.iterdir())


def test_no_file_for_no_frames(tmp_path):
    assert _write(tmp_path, []) is None


# ── the Batch Correction adapter ─────────────────────────────────────────────

def test_rows_from_aligned_reads_the_hutch_channels(tmp_path):
    aligned = {"instrument/Scalers/E/US_IC": np.array([100.0, 110.0]),
               "instrument/Scalers/E/D2PD": np.array([25.0, 27.5]),
               "instrument/StorageRing/SRCurrent": np.array([102.0, 102.0]),
               "instrument/SMS/E/HL/samX": np.array([1.0, 2.0])}
    rows = IC.rows_from_aligned(aligned, "E", 2)
    header, data = _write(tmp_path, rows, extras=("env", "motors"))
    assert header == ["frame", "I0", "I", "transmission",
                      "ring_current_mA", "HL/samX"]
    assert data[1] == ["1", "110", "27.5", "0.25", "102", "2"]


def test_rows_from_aligned_ignores_a_series_too_short_to_index(tmp_path):
    """h5_metadata.align leaves arrays it could not reduce alone, so they may
    not be one-value-per-frame — indexing them anyway would silently pair the
    wrong reading with the wrong frame."""
    aligned = {"instrument/Scalers/E/US_IC": np.array([100.0])}   # 1 < 3 frames
    rows = IC.rows_from_aligned(aligned, "E", 3)
    assert all(r["I0"] is None for r in rows)
    assert _write(tmp_path, rows) is None


def test_rows_from_aligned_on_an_unknown_hutch_has_no_chambers():
    rows = IC.rows_from_aligned(
        {"instrument/Scalers/E/US_IC": np.array([1.0, 2.0])}, None, 2)
    assert all(r["I0"] is None and r["I"] is None for r in rows)


def test_both_adapters_agree_on_the_same_readings(tmp_path):
    """The two tabs reach the data by different routes; the CSV they produce
    for the same exposures must not differ."""
    via_meta = IC.rows_from_metas(
        [{"ion_chamber_i0": 100.0, "ion_chamber_i": 25.0},
         {"ion_chamber_i0": 110.0, "ion_chamber_i": 27.5}])
    via_tree = IC.rows_from_aligned(
        {"instrument/Scalers/E/US_IC": np.array([100.0, 110.0]),
         "instrument/Scalers/E/D2PD": np.array([25.0, 27.5])}, "E", 2)
    assert _write(tmp_path, via_meta) == _write(tmp_path, via_tree)
