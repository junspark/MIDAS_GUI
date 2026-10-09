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
    # 1-ID-E was asserted to be None here, on the reasoning that it is "a
    # different beamline that merely ends in E". That was a guess about a
    # station nobody had looked at, and it was wrong: 2026-10 1-ID files
    # carry instrument/Scalers/{B,C,E}, with IC1-IC3 under E. The cost of
    # the guess was total and silent -- no hutch meant no channels, no rows,
    # and no beam-monitor CSV written at all for anyone on that profile,
    # while the run still reported success.
    ("1-ID-E", "E"), ("1-id-e", "E"),
    ("17-BM", None), ("Default", None), ("", None), (None, None),
])
def test_the_profile_is_the_fallback_when_the_path_is_silent(profile, want):
    """Reported from the beamline: an Eiger run lives under eiger2/, which
    matches neither varexE nor varexD, so a file carrying perfectly good
    Scalers/E/US_IC data silently produced no CSV. The header already says
    which station the user is on.

    17-BM stays None deliberately: its scaler group has not been seen in a
    real file, and a CSV of the wrong channels is worse than no CSV.
    """
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
    chunk = next(r for r in rows if r["kind"] == "chunk")
    assert chunk["E:US_IC"] == pytest.approx(11.0)
    assert chunk["E:DS_IC"] == pytest.approx(8.5)
    assert chunk["E:D2PD"] == pytest.approx(71.0)
    assert "transmission" not in chunk and "I0" not in chunk


def test_the_dark_gets_one_row_not_a_column_on_every_chunk():
    """The dark IMAGES are not written out, but their monitor readings are a
    real measurement of the shutter-closed baseline — one measurement, so
    one row, under the same column names the lights use."""
    tree = {"instrument/Scalers/E/US_IC": np.array([10.0, 12.0, 1.0, 3.0])}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 1)],
                                n_light=2, n_dark=2)
    assert [r["kind"] for r in rows] == ["dark", "chunk"]
    assert rows[0]["E:US_IC"] == pytest.approx(2.0)     # mean of 1.0, 3.0
    assert rows[1]["E:US_IC"] == pytest.approx(11.0)    # mean of 10.0, 12.0
    # The companion column is gone: one reading must not be repeated per
    # chunk as though it varied.
    assert not any(k.endswith("_dark") for r in rows for k in r)


def test_the_dark_is_averaged_over_its_whole_block_not_the_light_chunks():
    """Chunk boundaries are a property of the light frames; the dark block
    is acquired once and has no meaningful split along them."""
    tree = {"instrument/Scalers/E/US_IC":
            np.array([10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0])}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 1), (2, 3)],
                                n_light=4, n_dark=4)
    assert rows[0]["kind"] == "dark"
    assert rows[0]["E:US_IC"] == pytest.approx(2.5)     # mean of all four
    assert [r["E:US_IC"] for r in rows[1:]] == [pytest.approx(15.0),
                                              pytest.approx(35.0)]


def test_no_dark_block_means_no_dark_row(tmp_path):
    """A source whose metadata holds only light frames must not grow an
    empty dark row, nor a kind column that distinguishes nothing."""
    tree = {"instrument/Scalers/E/US_IC": np.array([10.0, 20.0])}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 1)],
                                n_light=2, n_dark=0)
    assert [r.get("kind") for r in rows] == ["chunk"]
    out = IC.write_ion_csv(tmp_path / "m.csv", rows)
    assert open(out).readline().strip() == \
        "kind,source_file,frame_start,frame_end,E:US_IC"


def test_the_dark_row_leads_the_file(tmp_path):
    tree = {"instrument/Scalers/E/US_IC": np.array([10.0, 12.0, 1.0, 3.0])}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 1)],
                                n_light=2, n_dark=2, source="s.h5")
    out = IC.write_ion_csv(tmp_path / "m.csv", rows)
    lines = [ln.strip() for ln in open(out)]
    assert lines[0] == "kind,source_file,frame_start,frame_end,E:US_IC"
    assert lines[1].startswith("dark,s.h5,0,1,")
    assert lines[2].startswith("chunk,s.h5,0,1,")


def test_values_are_averaged_over_the_same_chunks_as_the_images():
    tree = {"instrument/Scalers/E/US_IC": np.array([10.0, 20.0, 30.0, 40.0])}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 1), (2, 3)],
                                n_light=4, n_dark=0)
    assert [r["E:US_IC"] for r in rows] == [pytest.approx(15.0), pytest.approx(35.0)]


def test_sensitivity_companions_are_not_per_frame_columns():
    tree = {"instrument/Scalers/E/US_IC": np.array([1.0, 2.0]),
            "instrument/Scalers/E/US_IC_sensitivity": np.array([0])}
    assert IC.scaler_channels(tree, "E") == ["instrument/Scalers/E/US_IC"]


def test_an_unplaceable_source_still_writes_the_scalers_it_can_read(tmp_path):
    """Changed deliberately. Scaler groups are now identified from the FILE
    (instrument/Scalers/<G>) rather than from a hutch guessed off the path or
    the profile, so an unresolved hutch no longer suppresses them -- it only
    costs the sample-motor columns and the I0/I naming, which genuinely do
    need to know the station.

    The old behaviour wrote no CSV at all here, which is the same silent
    omission that left 1-ID-E with no sidecar for weeks. Real monitor
    readings in the file should reach the CSV whether or not the GUI can name
    the hutch they belong to."""
    rows, _ = IC.rows_from_tree({"instrument/Scalers/E/US_IC": np.array([1.0, 2.0])},
                                None, frame_ranges=[(0, 1)], n_light=2)
    out = IC.write_ion_csv(tmp_path / "m.csv", rows)
    assert out is not None
    assert "E:US_IC" in open(out).readline()


def test_a_source_with_no_scalers_at_all_still_writes_nothing(tmp_path):
    """The litter guard that mattered is intact: nothing to report, no file."""
    rows, _ = IC.rows_from_tree({}, None, frame_ranges=[(0, 1)], n_light=2)
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
    assert open(out).readline().strip() == \
        "kind,source_file,frame_start,frame_end,E:US_IC"


def test_a_row_is_identified_by_its_file_and_raw_frame_range():
    """One CSV holds rows from several source files (they accumulate per
    froot), so an ordinal would repeat across files. The range is also the
    same one the output HDF5 records in `frame_ranges`."""
    tree = {"instrument/Scalers/E/US_IC": np.arange(8.0)}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 3), (4, 7)],
                                n_light=8, n_dark=0, source="scan_000021.h5")
    assert [(r["source_file"], r["frame_start"], r["frame_end"]) for r in rows] \
        == [("scan_000021.h5", 0, 3), ("scan_000021.h5", 4, 7)]
    assert not any("frame" in r for r in rows)   # the bare ordinal is gone


def test_the_dark_rows_range_is_its_own_block():
    """The dark images are not written out, so there is no output frame for
    the dark row to index against — its range is over the dark acquisition
    block's own sub-frames."""
    tree = {"instrument/Scalers/E/US_IC":
            np.array([10.0, 12.0, 14.0, 1.0, 2.0])}
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 2)],
                                n_light=3, n_dark=2)
    assert (rows[0]["kind"], rows[0]["frame_start"], rows[0]["frame_end"]) \
        == ("dark", 0, 1)


def test_both_tabs_sidecars_are_the_same_file_but_for_the_name(tmp_path):
    """Asked for: "batch integrate should output identical metadata csv file
    as when background correction runs. The only difference is the file name
    extension (like bi or bc)."

    This replaces a test that pinned the OPPOSITE -- that Batch Integrate's
    CSV had no `kind` column, because that path built rows from
    metadata_for_index dicts and had no dark row. That divergence is the
    bug: the same builder now serves both, so the only way they can differ
    is the filename.
    """
    tree = _three_groups()
    args = dict(frame_ranges=[(0, 1), (2, 3)], n_light=4,
                source="scan_000012.vrx.h5")
    bc, _ = IC.rows_from_tree(tree, "E", **args)
    bi, _ = IC.rows_from_tree(tree, "E", **args)      # Batch Integrate's route
    a = IC.write_ion_csv(tmp_path / ("x" + IC.SUFFIX_BATCH_CORRECTION), bc)
    b = IC.write_ion_csv(tmp_path / ("x" + IC.SUFFIX_BATCH_INTEGRATE), bi)
    assert open(a).read() == open(b).read()
    assert str(a).endswith("_bc.csv") and str(b).endswith("_bi.csv")


# ── which dark was actually subtracted ───────────────────────────────────

def test_the_dark_row_names_the_dark_that_was_subtracted():
    """Left open when the dark became a row (DECISIONS 2026-10-06):

        "The CSV's dark row is the dark block *inside the same file*. Batch
        Correction's SUBTRACTED dark may be a different acquisition
        entirely... Reconciling the two is a real follow-up."

    Both things are true and they are different measurements, so the file
    has to state both rather than pick one. The monitor readings are this
    file's own shutter-closed block; the images were corrected with
    whatever ``frame_correct.resolve_dark``'s ladder chose, usually the
    nearest ``_dark_before`` sibling. A reader comparing a chunk against
    the baseline row was otherwise comparing it against a dark that may
    never have been subtracted.
    """
    tree = _tree([1000.0] * 4, [100.0] * 4)
    rows, _ = IC.rows_from_tree(
        tree, "E", frame_ranges=[(0, 3)], n_light=4, n_dark=4,
        source="s_00001.ge3.h5",
        subtracted_dark="s_dark_before_00000.ge3.h5")
    dark = [r for r in rows if r.get("kind") == "dark"]
    assert len(dark) == 1
    assert dark[0]["subtracted_dark"] == "s_dark_before_00000.ge3.h5"
    # The monitor readings themselves are still this file's own dark block,
    # which is what makes the two columns worth distinguishing.
    assert dark[0]["source_file"] == "s_00001.ge3.h5"


def test_only_the_dark_row_carries_it():
    """It is a statement about the baseline, not about each chunk. Putting
    it on every row is the width-doubling the companion columns were
    removed for."""
    tree = _tree([1000.0] * 4, [100.0] * 4)
    rows, _ = IC.rows_from_tree(
        tree, "E", frame_ranges=[(0, 1), (2, 3)], n_light=4, n_dark=4,
        source="s.h5", subtracted_dark="elsewhere.h5")
    for r in rows:
        if r.get("kind") == "chunk":
            assert "subtracted_dark" not in r


def test_the_column_is_absent_when_nothing_reported_a_dark():
    """A blank column reads as a missing value rather than an absent
    concept -- the same reason `kind` only appears when some row has one."""
    tree = _tree([1000.0] * 4, [100.0] * 4)
    rows, _ = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 3)],
                                n_light=4, n_dark=4, source="s.h5")
    assert all("subtracted_dark" not in r for r in rows)
    headers = [h for _k, h in IC._columns(rows, ())]
    assert "subtracted_dark" not in headers


def test_the_column_is_written_out_when_present():
    tree = _tree([1000.0] * 4, [100.0] * 4)
    rows, _ = IC.rows_from_tree(
        tree, "E", frame_ranges=[(0, 3)], n_light=4, n_dark=4,
        source="s.h5", subtracted_dark="d.h5")
    headers = [h for _k, h in IC._columns(rows, ())]
    assert "subtracted_dark" in headers
    # Index columns keep their fixed order; this one trails them.
    assert headers.index("subtracted_dark") < headers.index("E:US_IC")


# ── which station a profile means ────────────────────────────────────────

def test_the_beamlines_in_use_resolve_to_a_scaler_group():
    """A profile that maps to nothing writes NO CSV at all, silently: the
    run reports success and simply has no sidecar.

    1-ID-E was missing for exactly that reason and nobody noticed, because
    nothing fails -- resolve_hutch returns None, scaler_channels returns [],
    rows_from_tree builds nothing, and the writer has nothing to write.
    Confirmed against 2026-10 1-ID files, which carry
    instrument/Scalers/{B,C,E} with IC1-IC3 under E.
    """
    for profile in ("1-ID-E", "20-ID-E"):
        assert IC.resolve_hutch("", profile) == "E", profile
    assert IC.resolve_hutch("", "20-ID-D") == "D"
    # Case and surrounding whitespace are how it arrives from the header.
    assert IC.resolve_hutch("", "  1-id-e  ") == "E"


def test_an_unknown_profile_still_yields_nothing_rather_than_a_guess():
    """Writing a CSV of the wrong channels would be worse than writing
    none, so a station nobody has checked gets no entry."""
    assert IC.resolve_hutch("", "17-BM") is None
    assert IC.resolve_hutch("", "") is None
    assert IC.resolve_hutch("", None) is None


def test_the_path_still_outranks_the_profile():
    """A varexD folder names the station outright; the profile is only the
    fallback for layouts that say nothing."""
    assert IC.resolve_hutch("/data/varexD/x.h5", "1-ID-E") == "D"


def test_channels_come_from_the_file_not_from_a_mapping():
    """Which is why one letter per profile is enough: 1-ID-E records
    IC1/IC2/IC3 where 20-ID-E records US_IC, and neither needs its own
    channel list."""
    tree = {"instrument/Scalers/E/IC1": np.arange(8.0),
            "instrument/Scalers/E/IC2": np.arange(8.0),
            "instrument/Scalers/E/IC1_sensitivity": np.array([1.0])}
    got = [p.rsplit("/", 1)[-1] for p in IC.scaler_channels(tree, "E")]
    assert got == ["IC1", "IC2"], "sensitivities are scalars, not columns"


# ── every hutch in the beam path, not just the one you are sitting in ────────
# Asked for at 1-ID-E: "we write out all the ion chamber PVs instead of just
# the E hutch ones." A 2026-10 .pixi.h5 carries three scaler groups -- B
# (IC1-IC9, S1, S2, T), C (IC1-IC6) and E (IC1-IC8, S1, S2, T), all ten
# samples long for a ten-frame scan -- and the CSV held only E's eleven of
# those twenty-nine. Every hutch monitors the same beam, and you normalise
# against whichever was live.

def _three_groups(n=4):
    tree = {}
    for group, chans in (("B", 3), ("C", 2), ("E", 2)):
        for i in range(1, chans + 1):
            tree[f"instrument/Scalers/{group}/IC{i}"] = np.full(n, float(i))
            tree[f"instrument/Scalers/{group}/IC{i}_sensitivity"] = np.array([float(i) * 10])
    return tree


def test_all_known_groups_are_written_not_just_the_resolved_hutch():
    rows, _ = IC.rows_from_tree(_three_groups(), "E", frame_ranges=[(0, 3)], n_light=4)
    chunk = next(r for r in rows if r["kind"] == "chunk")
    for group, chans in (("B", 3), ("C", 2), ("E", 2)):
        for i in range(1, chans + 1):
            assert chunk[f"{group}:IC{i}"] == pytest.approx(float(i))


def test_the_group_prefix_is_what_keeps_IC1_unambiguous():
    """IC1 exists in B, C and E and is a different ion chamber in each, so a
    bare name would collide and silently keep only one."""
    assert IC.column_name("instrument/Scalers/B/IC1") == "B:IC1"
    assert IC.column_name("instrument/Scalers/E/IC1") == "E:IC1"
    chans = IC.scaler_channels(_three_groups())
    names = [IC.column_name(p) for p in chans]
    assert len(names) == len(set(names)) == 7


def test_groups_come_out_in_declared_order():
    names = [IC.column_name(p) for p in IC.scaler_channels(_three_groups())]
    assert names == ["B:IC1", "B:IC2", "B:IC3", "C:IC1", "C:IC2", "E:IC1", "E:IC2"]


def test_sensitivities_are_written_as_a_constant_down_the_column():
    """One value per file, not per acquisition -- but the gain the counts
    were taken at, so without it the counts cannot become a current."""
    rows, _ = IC.rows_from_tree(_three_groups(), "E", frame_ranges=[(0, 1), (2, 3)],
                                n_light=4)
    chunks = [r for r in rows if r["kind"] == "chunk"]
    assert len(chunks) == 2
    for r in chunks:
        assert r["B:IC1_sensitivity"] == pytest.approx(10.0)
        assert r["E:IC2_sensitivity"] == pytest.approx(20.0)


def test_a_sensitivity_is_not_mistaken_for_a_channel():
    chans = [IC.column_name(p) for p in IC.scaler_channels(_three_groups())]
    assert not any(n.endswith("_sensitivity") for n in chans)


def test_station_A_stays_out():
    """No sample sits in A's beam path, so however I0-suggestive its channel
    names are, it is not a per-sample monitor."""
    tree = {"instrument/Scalers/A/IC1": np.arange(4.0),
            "instrument/Scalers/E/IC1": np.arange(4.0)}
    assert [IC.column_name(p) for p in IC.scaler_channels(tree)] == ["E:IC1"]


def test_an_undeclared_group_is_reported_rather_than_dropped_in_silence():
    """SCALER_GROUPS is an explicit allow-list by choice, so a new station is
    invisible until someone adds it -- which is exactly how 1-ID-E came to
    have no CSV at all. It has to announce itself."""
    tree = {"instrument/Scalers/E/IC1": np.arange(4.0),
            "instrument/Scalers/Z/IC1": np.arange(4.0)}
    assert IC.unknown_scaler_groups(tree) == ["Z"]
    _rows, note = IC.rows_from_tree(tree, "E", frame_ranges=[(0, 3)], n_light=4)
    assert "Z" in note and "not written" in note
