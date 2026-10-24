"""The dark decision: which dark a given input file gets, and why.

Plain pytest — ``dark_policy`` imports no Qt, and keeping it that way is
the point of the split. No forked marker, no ``QApplication``.

The behaviour under test is the one that was wrong: Batch Correction and
Batch Integrate answered this differently, so the same selection produced
different corrections depending on which tab ran it.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_gui import dark_policy as D


def _req(**kw):
    return D.DarkRequest(**kw)


# ── which mode a pick implies ───────────────────────────────────────────

@pytest.mark.parametrize("origin,auto", [
    ("file", False),
    ("files", False),
    ("folder", False),
    ("folder_recursive", True),
    ("stem", True),
])
def test_the_pick_decides_the_mode(origin, auto):
    """The whole rule, in one table. A traversal resolves per file; an
    explicit pick uses the Dark field."""
    assert D.wants_auto(_req(origin=origin)) is auto


def test_a_stem_pick_autodetects_even_though_it_is_a_flat_list():
    """It globs <folder>/**/<stem>* and crosses subfolder boundaries
    whatever the "Include subfolders" box says, so it is a traversal
    however flat the resulting list looks."""
    assert D.wants_auto(_req(origin="stem"))
    assert "stem" in D.AUTO_ORIGINS


def test_every_origin_the_loader_can_report_is_known_here():
    """Guards the seam: a new pick mode added to the loader must be
    classified deliberately, not default to explicit by omission."""
    assert D.ORIGINS == {"file", "files", "folder", "folder_recursive", "stem"}


def test_the_override_beats_the_pick_both_ways():
    assert D.wants_auto(_req(origin="file", override=D.ALWAYS_AUTO))
    assert not D.wants_auto(
        _req(origin="folder_recursive", override=D.ALWAYS_EXPLICIT))


def test_the_mode_is_describable_without_running_anything():
    """The UI must be able to state the mode before a run, or the user
    finds out which dark was used by reading the log afterwards."""
    assert "per file" in D.describe(_req(origin="folder_recursive"))
    assert "every file" in D.describe(_req(origin="file", path="/d/dk.h5"))
    assert "dk.h5" in D.describe(_req(origin="file", path="/d/dk.h5"))


# ── resolution ──────────────────────────────────────────────────────────

def _vrx(path, *, data=None, dark=None, shape=(4, 5)):
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    ramp = np.linspace(-0.5, 0.5, shape[0] * shape[1],
                       dtype=np.float32).reshape(shape)
    ramp -= ramp.mean()
    with h5py.File(str(path), "w") as f:
        if data is not None:
            f.create_dataset("exchange/data",
                             data=np.full((2,) + shape, float(data),
                                          np.float32) + ramp[None])
        if dark is not None:
            f.create_dataset("exchange/data_dark",
                             data=np.full((2,) + shape, float(dark),
                                          np.float32) + ramp[None])
    return path


def test_auto_mode_with_nothing_resolvable_skips_rather_than_falling_back(
        tmp_path):
    """The load-bearing one.

    resolve_dark returns at its rung 3 whenever a fallback is passed and so
    never reaches "no dark found" — so handing it the GUI dark as a safety
    net would silently convert "I could not find this file's dark" into "I
    used some other scan's dark". In a traversal that is exactly the error
    that must not be allowed to pass unnoticed.
    """
    data = _vrx(tmp_path / "scan_000020.vrx.h5", data=100.0)
    got = D.resolve(data, _req(origin="folder_recursive",
                               array=np.full((4, 5), 6.0, np.float32)),
                    shape=(4, 5))
    assert got.skipped
    assert got.array is None
    assert "scan_000020.vrx.h5" in got.why


def test_auto_mode_finds_the_sibling_dark(tmp_path):
    _vrx(tmp_path / "scan_dark_before_000019.vrx.h5", dark=3.0)
    data = _vrx(tmp_path / "scan_000020.vrx.h5", data=100.0)
    got = D.resolve(data, _req(origin="folder_recursive"), shape=(4, 5))
    assert not got.skipped
    assert np.allclose(got.array.mean(), 3.0)
    assert "dark_before_000019" in got.why


def test_explicit_mode_ignores_a_sibling_dark_entirely(tmp_path):
    """The user named a dark; a file that happens to sit next to a
    dark_before must not quietly override it."""
    _vrx(tmp_path / "scan_dark_before_000019.vrx.h5", dark=3.0)
    data = _vrx(tmp_path / "scan_000020.vrx.h5", data=100.0)
    got = D.resolve(data, _req(origin="files",
                               array=np.full((4, 5), 6.0, np.float32)),
                    shape=(4, 5))
    assert np.allclose(got.array, 6.0)
    assert not got.skipped


def test_explicit_mode_with_no_dark_set_runs_without_one(tmp_path):
    """Correcting without a dark is legitimate — it just has to be said."""
    data = _vrx(tmp_path / "scan_000020.vrx.h5", data=100.0)
    got = D.resolve(data, _req(origin="file"), shape=(4, 5))
    assert got.array is None and not got.skipped
    assert "without one" in got.why


def test_a_zero_placeholder_does_not_satisfy_auto_mode(tmp_path):
    """The white_oct26 shape: the file's own data_dark is all zeros, so
    there is genuinely no dark and the file must be skipped rather than
    corrected with zeros."""
    data = _vrx(tmp_path / "scan_000020.vrx.h5", data=100.0)
    import h5py
    with h5py.File(str(data), "a") as f:
        if "exchange/data_dark" in f:
            del f["exchange/data_dark"]
        f.create_dataset("exchange/data_dark",
                         data=np.zeros((2, 4, 5), np.float32))
    got = D.resolve(data, _req(origin="folder_recursive"), shape=(4, 5))
    assert got.skipped, got.why


# ── Hydra: one panel's choice, every panel's path ───────────────────────

@pytest.mark.parametrize("src,want", [
    # the token appears TWICE in a real path — directory and extension
    ("/d/white/ge1/SCAN/SCAN_005826.ge1.h5", "/d/white/ge2/SCAN/SCAN_005826.ge2.h5"),
    # case preserved: helpers._HYDRA_PANEL_RE lowercases, producing a path
    # that does not exist on a case-sensitive filesystem
    ("/d/GE1/x.GE1.h5", "/d/GE2/x.GE2.h5"),
    # an underscore-prefixed token: helpers._HYDRA_PANEL_RE's leading \b
    # misses this, so it rewrote only the directory half
    ("/d/ge1/scan_ge1.instr.txt", "/d/ge2/scan_ge2.instr.txt"),
])
def test_panel_substitution_rewrites_every_token(src, want):
    assert D.substitute_panel(src, 2) == want


@pytest.mark.parametrize("src", [
    "/d/average1/scan.h5",      # hydra.PANEL_IN_NAME rewrites this one
    "/d/edge2/scan.h5",
    "/d/image3/scan.h5",
    "/d/ge12/scan.h5",          # ge12 is not panel 1
])
def test_panel_substitution_leaves_lookalikes_alone(src):
    out = D.substitute_panel(src, 4)
    assert out is None or "ge4" not in out.replace("/d/", "", 1).split("/")[0]


def test_a_path_with_no_panel_token_cannot_be_derived():
    assert D.substitute_panel("/d/darks/dark_mean.tif", 3) is None


def test_a_missing_sibling_panel_is_a_skip_naming_the_path(tmp_path):
    """Never another panel's dark. Applying ge1's dark to ge2 data is the
    mistake this exists to prevent, so it fails rather than falls back."""
    (tmp_path / "ge1").mkdir()
    anchor = tmp_path / "ge1" / "dark.ge1.h5"
    anchor.write_bytes(b"")
    path, why = D.panel_dark_path(str(anchor), 2)
    assert path is None
    assert "ge2" in why and "dark.ge2.h5" in why


def test_a_present_sibling_panel_resolves(tmp_path):
    for n in (1, 2):
        (tmp_path / f"ge{n}").mkdir()
        (tmp_path / f"ge{n}" / f"dark.ge{n}.h5").write_bytes(b"")
    path, why = D.panel_dark_path(str(tmp_path / "ge1" / "dark.ge1.h5"), 2)
    assert path == str(tmp_path / "ge2" / "dark.ge2.h5"), why


def test_an_anchor_with_no_token_says_so_rather_than_guessing(tmp_path):
    """A dark in a shared folder with no panel token is a real setup; the
    message must distinguish it from a missing sibling."""
    p = tmp_path / "dark_mean.tif"
    p.write_bytes(b"")
    path, why = D.panel_dark_path(str(p), 2)
    assert path is None
    assert "carries no ge1-ge4 token" in why


# ── the recursive search covers the CHOSEN TREE ─────────────────────────

def _h5(root, sub, name, *, exp, dark=None, data=None, rings=False,
        shape=(32, 32)):
    import h5py
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    ramp = np.linspace(-3, 3, shape[0] * shape[1],
                       dtype=np.float32).reshape(shape)
    ramp -= ramp.mean()
    p = d / name
    with h5py.File(str(p), "w") as f:
        f.create_dataset("instrument/Detector/DetAcqTime",
                         data=np.array([float(exp)]))
        if dark is not None:
            a = np.full((3,) + shape, float(dark), np.float32) + ramp[None]
            if rings:
                a[:, 8:10, :] += 5000      # a dark that is not one
            f.create_dataset("exchange/data_dark", data=a)
        if data is not None:
            f.create_dataset("exchange/data",
                             data=np.full((3,) + shape, float(data),
                                          np.float32) + ramp[None])
    return p


def test_the_search_reaches_darks_in_sibling_folders(tmp_path):
    """The rule as specified: choosing expid/ge1 with subfolders searches
    the WHOLE tree. At 1-ID-E the darks live in their own sibling folders
    (AlCaBa_S1_RT_Darks/) one level up from the scans they serve, so the
    per-directory search this replaces found nothing at all.
    """
    _h5(tmp_path, "MySample_Darks", "MySample_Darks_000100.ge1.h5",
        exp=10.0, dark=1778.0)
    scan = _h5(tmp_path, "MySample_scan1", "MySample_scan1_000120.ge1.h5",
               exp=10.0, data=2100.0)
    got = D.resolve(scan, D.DarkRequest(origin="folder_recursive",
                                        search_root=str(tmp_path),
                                        exposure=10.0), shape=(32, 32))
    assert not got.skipped, got.why
    assert np.allclose(got.array.mean(), 1778.0)


def test_a_wrong_exposure_dark_is_rejected_not_merely_ranked_low(tmp_path):
    """The white_oct26 failure. Every dark there was taken at 0.1 s while
    the data is at 1 s and 10 s, and a GE panel's offset grows with
    exposure -- the 0.1 s dark is ~19 counts ABOVE the 1 s data, so
    subtracting it drove the whole background negative and clipped it to
    zero. Ranking it low is not enough: a wrong dark that is merely
    unpreferred still gets used when it is the only one.
    """
    _h5(tmp_path, "Darks_short", "Darks_short_000100.ge1.h5",
        exp=0.1, dark=1778.0)
    scan = _h5(tmp_path, "scan1", "scan1_000120.ge1.h5", exp=10.0,
               data=2100.0)
    got = D.resolve(scan, D.DarkRequest(origin="folder_recursive",
                                        search_root=str(tmp_path),
                                        exposure=10.0), shape=(32, 32))
    assert got.skipped
    assert "exposure" in got.why and "0.1" in got.why


def test_the_skip_message_names_what_it_rejected_and_why(tmp_path):
    """A skip that just says "no dark" sends someone to go and look. This
    one has to say which files it considered and what was wrong."""
    _h5(tmp_path, "Darks_a", "Darks_a_000100.ge1.h5", exp=0.1, dark=1778.0)
    _h5(tmp_path, "Darks_b", "Darks_b_000110.ge1.h5", exp=5.0, dark=1858.0)
    scan = _h5(tmp_path, "s", "s_000120.ge1.h5", exp=10.0, data=2100.0)
    got = D.resolve(scan, D.DarkRequest(origin="folder_recursive",
                                        search_root=str(tmp_path),
                                        exposure=10.0), shape=(32, 32))
    assert got.skipped
    assert "Darks_a_000100" in got.why and "Darks_b_000110" in got.why


def test_the_nearest_preceding_dark_wins_across_the_tree(tmp_path):
    """Scan numbers are global, so the rule the 192-log survey validated
    inside one folder means the same thing across a tree."""
    _h5(tmp_path, "d_far", "d_far_Darks_000100.ge1.h5", exp=10.0, dark=11.0)
    _h5(tmp_path, "d_near", "d_near_Darks_000118.ge1.h5", exp=10.0, dark=22.0)
    _h5(tmp_path, "d_after", "d_after_Darks_000130.ge1.h5", exp=10.0, dark=33.0)
    scan = _h5(tmp_path, "s", "s_000120.ge1.h5", exp=10.0, data=2100.0)
    got = D.resolve(scan, D.DarkRequest(origin="folder_recursive",
                                        search_root=str(tmp_path),
                                        exposure=10.0), shape=(32, 32))
    assert np.allclose(got.array.mean(), 22.0), got.why


def test_a_dark_whose_frames_show_diffraction_is_rejected(tmp_path):
    """Name is a clue; the frames settle it. A file called a dark whose
    dark plane carries rings is not a dark whatever it is called."""
    _h5(tmp_path, "Bad_Darks", "Bad_Darks_000118.ge1.h5", exp=10.0,
        dark=1778.0, rings=True)
    _h5(tmp_path, "Good_Darks", "Good_Darks_000100.ge1.h5", exp=10.0,
        dark=1778.0)
    scan = _h5(tmp_path, "s", "s_000120.ge1.h5", exp=10.0, data=2100.0)
    got = D.resolve(scan, D.DarkRequest(origin="folder_recursive",
                                        search_root=str(tmp_path),
                                        exposure=10.0), shape=(32, 32))
    assert not got.skipped, got.why
    assert "Good_Darks" in got.why, "the ringed candidate was not vetoed"


def test_the_frame_check_is_a_veto_not_a_classifier():
    """Stated because it bounds what the check may be used for. Across 15
    real darks and 15 real data planes the distributions OVERLAP at the
    low end -- a weak or blank data frame is indistinguishable from a
    dark -- so this may reject a named candidate, never promote an
    unnamed file into being one.
    """
    from midas_gui.frame_correct import has_diffraction_structure as H
    flat = np.full((64, 64), 1778.0, np.float32)
    flat += np.linspace(-3, 3, 64 * 64, dtype=np.float32).reshape(64, 64)
    assert not H(flat)
    ringed = flat.copy()
    ringed[20:24, :] += 5000
    assert H(ringed)
    # a weak measurement reads like a dark, and that is why it is a veto
    weak = flat.copy()
    weak[20, :] += 2.0
    assert not H(weak)


# ── Hydra: one panel's pick, every panel's own dark ─────────────────────

def test_a_panel_request_is_scoped_to_that_panel_s_own_tree():
    """The one thing Hydra must never do.

    Picking expid/ge1 recursively must search expid/ge2 for ge2's dark.
    Handing every panel the same search root is precisely how ge1's dark
    ends up subtracted from ge2 data, and the panels are different
    detectors with different offsets.
    """
    from midas_gui.dark_policy import substitute_panel
    root = "/data/white_oct26/ge1"
    assert substitute_panel(root, 2) == "/data/white_oct26/ge2"
    assert substitute_panel(root, 4) == "/data/white_oct26/ge4"
    anchor = "/data/white_oct26/ge1/Darks/Darks_005787.ge1.h5"
    assert substitute_panel(anchor, 3) == \
        "/data/white_oct26/ge3/Darks/Darks_005787.ge3.h5"


def test_a_root_with_no_panel_token_scopes_to_nothing_rather_than_sharing():
    """A pick that is not panel-specific (expid/, say) gives no per-panel
    root. Returning the shared root instead would silently let every panel
    search every other panel's folders."""
    from midas_gui.dark_policy import substitute_panel
    assert substitute_panel("/data/white_oct26", 2) is None


# ── background-job --dark arguments ────────────────────────────────────

def _scan(root, sub, name, *, exp, dark=None, data=None, shape=(16, 16)):
    import h5py
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    ramp = np.linspace(-3, 3, shape[0] * shape[1],
                       dtype=np.float32).reshape(shape)
    ramp -= ramp.mean()
    p = d / name
    with h5py.File(str(p), "w") as f:
        f.create_dataset("instrument/Detector/DetAcqTime",
                         data=np.array([float(exp)]))
        if dark is not None:
            f.create_dataset("exchange/data_dark",
                             data=np.full((2,) + shape, float(dark),
                                          np.float32) + ramp[None])
        if data is not None:
            f.create_dataset("exchange/data",
                             data=np.full((2,) + shape, float(data),
                                          np.float32) + ramp[None])
    return p


def test_autodetect_passes_the_source_path_and_copies_nothing(tmp_path):
    """The fix for what a real run exposed.

    The first version wrote each resolved dark out as a pre-averaged
    TIFF: 17 MB per (scan, panel), 816 MB and a visibly blocked GUI
    before a 48-step job had launched -- to duplicate a file already on
    disk. The job reads the original instead.
    """
    tree = tmp_path / "tree"
    dark = _scan(tree, "Darks", "Darks_000100.ge1.h5", exp=10.0, dark=1778.0)
    scans = [_scan(tree, f"s{i}", f"s{i}_0001{i:02d}.ge1.h5", exp=10.0,
                   data=2100.0) for i in range(8)]
    out = tmp_path / "out"; out.mkdir()
    b = D.DarkArgBuilder(out)
    req = D.DarkRequest(origin="folder_recursive", search_root=str(tree))
    args = [b.arg_for(s, req) for s in scans]

    assert all(a[:2] == ["--dark", str(dark)] for a in args), args
    assert all("--dark-dataset" in a for a in args)
    assert b.written() == 0, "a TIFF was written for a dark already on disk"
    assert list(out.glob("*.tif")) == []


def test_each_scan_gets_the_dark_that_resolved_for_it(tmp_path):
    """Passing paths must not blur which dark belongs to which scan."""
    tree = tmp_path / "tree"
    d1 = _scan(tree, "d1", "d1_Darks_000100.ge1.h5", exp=10.0, dark=11.0)
    d2 = _scan(tree, "d2", "d2_Darks_000200.ge1.h5", exp=10.0, dark=22.0)
    early = _scan(tree, "a", "a_000150.ge1.h5", exp=10.0, data=2100.0)
    late = _scan(tree, "b", "b_000250.ge1.h5", exp=10.0, data=2100.0)
    out = tmp_path / "out"; out.mkdir()
    b = D.DarkArgBuilder(out)
    req = D.DarkRequest(origin="folder_recursive", search_root=str(tree))
    assert b.arg_for(early, req)[1] == str(d1)
    assert b.arg_for(late, req)[1] == str(d2)


def test_an_explicit_dark_field_is_still_written_once_per_panel(tmp_path):
    """The one case with no source file behind it: the loader hands us an
    array, which may already honour a frame range the user set on it, so
    re-reading the path would quietly ignore that. One TIFF per panel,
    not per step."""
    pytest.importorskip("tifffile")
    scan = _scan(tmp_path / "t", "s", "s_000150.ge1.h5", exp=10.0, data=2100.0)
    out = tmp_path / "out"; out.mkdir()
    b = D.DarkArgBuilder(out)
    req = D.DarkRequest(origin="files", array=np.full((16, 16), 7.0, np.float32))
    a1 = b.arg_for(scan, req, tag="ge1")
    a2 = b.arg_for(scan, req, tag="ge1")
    a3 = b.arg_for(scan, req, tag="ge2")
    assert a1 == a2 and a1[1] != a3[1]
    assert b.written() == 2
    assert "--dark-dataset" not in a1, "an explicit array has no dataset"


def test_a_step_with_no_resolvable_dark_returns_none_and_is_recorded(tmp_path):
    """None, not an empty list: the caller must DROP the step. Returning
    [] would silently integrate it with no dark at all, and returning
    some other scan's dark is the error the policy exists to prevent."""
    tree = tmp_path / "tree"
    _scan(tree, "Darks", "Darks_000100.ge1.h5", exp=0.1, dark=1778.0)
    scan = _scan(tree, "s", "s_000150.ge1.h5", exp=10.0, data=2100.0)
    out = tmp_path / "out"; out.mkdir()
    b = D.DarkArgBuilder(out)
    req = D.DarkRequest(origin="folder_recursive", search_root=str(tree))
    assert b.arg_for(scan, req) is None
    assert len(b.skipped) == 1 and "exposure" in b.skipped[0][1]
    assert b.written() == 0


def test_explicit_mode_with_no_dark_set_is_an_empty_arg_not_a_skip(tmp_path):
    """Running without a dark is legitimate when the user chose none; it
    must not be confused with failing to find one."""
    scan = _scan(tmp_path / "t", "s", "s_000150.ge1.h5", exp=10.0, data=2100.0)
    out = tmp_path / "out"; out.mkdir()
    b = D.DarkArgBuilder(out)
    assert b.arg_for(scan, D.DarkRequest(origin="files")) == []
    assert b.skipped == []
