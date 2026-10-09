"""A recursive pick must not flatten many source folders into one output dir.

Reported at 1-ID-E: a recursive "Full folder" Batch Integrate over
``connolly_oct26/pixirad/APS_4130_C_load_step_*`` wrote all 23 load steps'
output into ``connolly_oct26_bc/APS_4130_C_load_step_0/pixirad/`` -- 1,790
files in one folder. The ``<expid>_bc/<froot>/<detector>`` convention reads
``froot`` positionally off each file's own path, so the selection implies 23
output dirs; the run applied it once, to the first file.
"""
from pathlib import Path

import pytest

from midas_gui.helpers import (group_paths_by_output_dir,
                               suggest_integration_output_dir)

ROOT = "/data/s1c/connolly_oct26/pixirad"


def _files(*steps, per_step=1):
    return [f"{ROOT}/APS_4130_C_load_step_{n}/APS_4130_C_load_step_{n}_{4158 + i:06d}.pixi.h5"
            for n in steps for i in range(per_step)]


def test_the_untouched_single_folder_case_is_still_one_group():
    """Every non-recursive pick must keep taking the existing one-run path."""
    files = _files(0, per_step=3)
    groups = group_paths_by_output_dir(files, out_dir="/tmp/out")
    assert len(groups) == 1
    assert groups[0][1] == files


def test_a_recursive_sweep_splits_one_group_per_source_folder():
    groups = group_paths_by_output_dir(_files(0, 1, 5), out_dir=None)
    assert len(groups) == 3
    assert [len(f) for _d, f in groups] == [1, 1, 1]


def test_each_group_lands_where_picking_that_folder_by_hand_would():
    """The whole point: identical layout to 23 manual single-folder runs."""
    for d, files in group_paths_by_output_dir(_files(0, 1, 5), out_dir=None):
        assert d == suggest_integration_output_dir(files[0])
        assert d.name == "pixirad"
        assert "load_step_" in d.parent.name


def test_a_stale_autofilled_output_does_not_capture_every_group():
    """The field auto-fills from the first file. Honouring it would reproduce
    exactly the reported bug, so it is recognised and discarded."""
    files = _files(0, 1, 5)
    stale = str(suggest_integration_output_dir(files[0]))
    dirs = [str(d) for d, _ in group_paths_by_output_dir(files, out_dir=stale)]
    assert len(set(dirs)) == 3
    assert sum(d == stale for d in dirs) == 1, "only step 0 belongs there"


def test_a_custom_output_root_fans_out_under_itself():
    groups = group_paths_by_output_dir(_files(0, 1, 5), out_dir="/tmp/myout")
    dirs = [d for d, _ in groups]
    assert all(str(d).startswith("/tmp/myout/") for d in dirs)
    assert len(set(dirs)) == 3
    assert {d.parent.name for d in dirs} == {
        "APS_4130_C_load_step_0", "APS_4130_C_load_step_1", "APS_4130_C_load_step_5"}


def test_groups_come_back_in_path_order():
    dirs = [str(d) for d, _ in group_paths_by_output_dir(_files(5, 0, 1), out_dir=None)]
    assert dirs == sorted(dirs)


def test_every_selected_file_is_placed_exactly_once():
    files = _files(0, 1, 5, per_step=4)
    placed = [f for _d, fs in group_paths_by_output_dir(files, out_dir=None) for f in fs]
    assert sorted(placed) == sorted(files)
    assert len(placed) == len(files)


def test_an_empty_selection_is_no_groups():
    assert group_paths_by_output_dir([], out_dir="/tmp/out") == []


# ── darks must not split away from the load step they belong to ──────────
def _folder_with_darks(step):
    d = f"{ROOT}/Faber_A_restart4_load_step_{step}"
    stem = f"Faber_A_restart4_load_step_{step}"
    return ([f"{d}/{stem}_{3933 + i:06d}.pixi.h5" for i in range(13)]
            + [f"{d}/{stem}_dark_before_003932.pixi.h5",
               f"{d}/{stem}_dark_after_003946.pixi.h5"])


def test_a_folders_darks_stay_with_its_scan():
    """suggest_integration_output_dir reads the froot off the FILE name, so a
    dark derives ``..._dark_before/`` of its own. Picking the folder by hand
    puts everything in one place; grouping by folder must match."""
    files = _folder_with_darks(0)
    groups = group_paths_by_output_dir(files, out_dir=None)
    assert len(groups) == 1, [str(d) for d, _ in groups]
    d, got = groups[0]
    assert len(got) == 15, "13 frames + 2 darks"
    assert "dark" not in d.parent.name, f"darks captured the output dir: {d}"
    assert d.parent.name == "Faber_A_restart4_load_step_0"


def test_many_folders_each_keep_their_own_darks():
    files = _folder_with_darks(0) + _folder_with_darks(1)
    groups = group_paths_by_output_dir(files, out_dir=None)
    assert len(groups) == 2
    assert all(len(f) == 15 for _d, f in groups)
    assert {d.parent.name for d, _ in groups} == {
        "Faber_A_restart4_load_step_0", "Faber_A_restart4_load_step_1"}


def test_a_folder_holding_only_darks_still_resolves():
    only_darks = _folder_with_darks(2)[13:]
    groups = group_paths_by_output_dir(only_darks, out_dir=None)
    assert len(groups) == 1
    assert groups[0][0] is not None
