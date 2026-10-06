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


@pytest.mark.parametrize("profile,want", [
    ("20-ID-E", "E"), ("20-id-e", "E"), ("20-ID-D", "D"),
    ("1-ID-E", None),       # a different beamline that merely ends in E
    ("17-BM", None), ("Default", None), ("", None), (None, None),
])
def test_the_profile_is_the_fallback_when_the_path_is_silent(profile, want):
    """Reported from the beamline: an Eiger run lives under eiger2/, which
    matches neither varexE nor varexD, so a file carrying perfectly good
    Scalers/E/US_IC data silently produced no CSV. The header already says
    which station the user is on."""
    assert IC.resolve_hutch("/mnt/s20a/brown_sep26/eiger2/x.h5", profile) == want


def test_the_path_outranks_the_profile():
    """A varexD folder names the station outright; a stale profile must not
    override it."""
    assert IC.resolve_hutch("/mnt/s20a/x/varexD/y.h5", "20-ID-E") == "D"


def test_station_a_has_no_monitor_mapping():
    """A's IC4_foil_I0/IC5_foil_I1 names look like a pair but no sample sits
    in its beam path, so claiming them would be worse than claiming nothing."""
    assert "A" not in IC.ION_CHAMBER_H5_PATHS
    assert IC.metadata_h5_paths("A") == IC.METADATA_H5_PATHS


def test_d_hutch_declares_no_transmission_channel():
    assert "ion_chamber_i" not in IC.ION_CHAMBER_H5_PATHS["D"]
    assert "ion_chamber_i0" in IC.ION_CHAMBER_H5_PATHS["D"]


# ── which acquisitions were the lights ───────────────────────────────────────
#
# A VAREX/Eiger stack records one metadata sample per detector acquisition,
# light AND dark, in one flat array — 20 light + 20 dark frames give 40
# scaler entries. Something has to say which half is which, and getting it
# wrong reports the shutter-closed baseline as the beam.

def _tree(light, dark, *, chan="instrument/Scalers/E/US_IC", ts=None):
    """A per-acquisition tree with `light` then `dark` (or vice versa)."""
    t = {chan: np.asarray(list(light) + list(dark), dtype=float)}
    if ts is not None:
        t["NDArray/NDArrayTimeStamp"] = np.asarray(ts, dtype=float)
    return t


def test_lights_are_the_brighter_block_wherever_it_sits():
    """Order-agnostic on purpose: VAREX trails its darks, a 2026-10 Eiger
    file puts them first. The monitor itself says which is which."""
    lights, darks = [1000.0] * 4, [100.0] * 4
    off, dark_off, note = IC.split_light_dark(
        _tree(lights, darks), n_light=4, n_dark=4, hutch="E")
    assert (off, dark_off) == (0, 4) and "lights first" in note

    off, dark_off, note = IC.split_light_dark(
        _tree(darks, lights), n_light=4, n_dark=4, hutch="E")
    assert (off, dark_off) == (4, 0) and "lights second" in note


def test_the_timestamp_gap_is_a_cross_check_not_the_decision():
    """Steady cadence with one long gap at the boundary — it should agree,
    and say so."""
    ts = [0, 10, 20, 30, 45, 55, 65, 75]          # long gap between 3 and 4
    _off, _d, note = IC.split_light_dark(
        _tree([100.0] * 4, [1000.0] * 4, ts=ts), n_light=4, n_dark=4, hutch="E")
    assert "lights second" in note and "gap at 3 agrees" in note


def test_a_gap_that_contradicts_the_intensity_is_reported(tmp_path):
    """Better to say the two signals disagree than to quietly pick one."""
    ts = [0, 10, 25, 35, 45, 55, 65, 75]          # gap in the wrong place
    _off, _d, note = IC.split_light_dark(
        _tree([100.0] * 4, [1000.0] * 4, ts=ts), n_light=4, n_dark=4, hutch="E")
    assert "DISAGREES" in note


def test_no_dark_block_means_a_single_block():
    off, dark_off, note = IC.split_light_dark(
        _tree([1.0] * 4, []), n_light=4, n_dark=0, hutch="E")
    assert (off, dark_off) == (0, None) and "single block" in note


# ── the rows themselves ──────────────────────────────────────────────────────

def test_every_hutch_channel_is_written_under_its_own_name():
    """"Write out all IC numbers faithfully" — no channel is renamed to I or
    I0, because naming one of them that would assert which is the
    transmitted monitor, and that is setup-dependent."""
    tree = {"instrument/Scalers/E/US_IC": np.array([10.0, 12.0, 1.0, 1.0]),
            "instrument/Scalers/E/DS_IC": np.array([8.0, 9.0, 0.5, 0.5]),
            "instrument/Scalers/E/D2PD": np.array([70.0, 72.0, 2.0, 2.0])}
    rows, _note = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 1)],
                                    n_light=2, n_dark=2)
    assert rows[0]["US_IC"] == pytest.approx(11.0)
    assert rows[0]["DS_IC"] == pytest.approx(8.5)
    assert rows[0]["D2PD"] == pytest.approx(71.0)
    assert "transmission" not in rows[0] and "I0" not in rows[0]


def test_each_channel_gets_a_dark_companion():
    """The dark IMAGES are not written out, but their monitor readings are a
    real measurement of the shutter-closed baseline."""
    tree = {"instrument/Scalers/E/US_IC": np.array([10.0, 12.0, 1.0, 3.0])}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 1)],
                                n_light=2, n_dark=2)
    assert rows[0]["US_IC"] == pytest.approx(11.0)
    assert rows[0]["US_IC" + IC.DARK_SUFFIX] == pytest.approx(2.0)


def test_values_are_averaged_over_the_same_chunks_as_the_images():
    tree = {"instrument/Scalers/E/US_IC": np.array([10.0, 20.0, 30.0, 40.0])}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 1), (2, 3)],
                                n_light=4, n_dark=0)
    assert [r["US_IC"] for r in rows] == [pytest.approx(15.0), pytest.approx(35.0)]


def test_sensitivity_companions_are_not_per_frame_columns():
    tree = {"instrument/Scalers/E/US_IC": np.array([1.0, 2.0]),
            "instrument/Scalers/E/US_IC_sensitivity": np.array([0])}
    assert IC.scaler_channels(tree, "E") == ["instrument/Scalers/E/US_IC"]


def test_an_unplaceable_source_writes_nothing(tmp_path):
    rows, _ = IC.rows_from_tree({"instrument/Scalers/E/US_IC": np.array([1.0, 2.0])},
                                None, frame_ranges=[(0, 1)], n_light=2)
    assert IC.write_ion_csv(tmp_path / "m.csv", rows) is None
    assert not list(tmp_path.iterdir())


def test_rows_carry_their_source_file_so_a_froot_csv_stays_traceable():
    rows, _ = IC.rows_from_tree({"instrument/Scalers/E/US_IC": np.array([1.0, 2.0])},
                                "E", frame_ranges=[(0, 1)], n_light=2,
                                source="AgBeH_10s_000021.h5")
    assert rows[0]["source_file"] == "AgBeH_10s_000021.h5"


def test_header_leads_with_the_index_columns(tmp_path):
    rows, _ = IC.rows_from_tree({"instrument/Scalers/E/US_IC": np.array([1.0, 2.0])},
                                "E", frame_ranges=[(0, 1)], n_light=2, source="s.h5")
    out = IC.write_ion_csv(tmp_path / "m.csv", rows)
    assert open(out).readline().strip() == "frame,source_file,US_IC"
