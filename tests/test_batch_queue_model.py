"""Batch Queue data model — the three-level calibration→corrections→samples tree.

Pure logic, no Qt: ``midas_gui.batch_queue`` deliberately imports no PyQt5 (only
``constants``, which is Qt-free), which is what lets this file run unforked and
keeps the model testable without a QApplication. If an import of this module
ever starts pulling Qt in, ``test_model_module_imports_no_qt`` below fails —
that is the guard, not a comment.

The contracts worth pinning here are the ones a saved project depends on: the
JSON round-trip must survive a reload, and ``sample_source_cfg`` must keep
emitting exactly what ``workers._open_source_cfg`` dispatches on.
"""
import subprocess
import sys

import pytest

from midas_gui.batch_queue import (
    BatchQueue, CalibrationNode, CorrectionsNode, Sample,
    DEFAULT_DATASET, KIND_FOLDER, KIND_HDF5, SOURCE_FILE, SOURCE_TAB2,
    default_label, detect_dataset, kind_for_path, sample_source_cfg,
)


def _queue():
    """Two calibrations; the first with two corrections nodes, mixed kinds."""
    return BatchQueue(
        calibrations=[
            CalibrationNode(
                name="AgBH", source=SOURCE_TAB2,
                calib_snapshot={"Lsd": 1000.0, "BC_y": 1.5},
                mask_sources=[{"kind": "file", "path": "/m/beamstop.tif",
                               "enabled": True}],
                corrections=[
                    CorrectionsNode(name="Corr 1", dark="/d/dark_001.tif",
                                    samples=[Sample("/d/s1/scan_001.h5"),
                                             Sample("/d/s1/scan_002.h5")]),
                    CorrectionsNode(name="Corr 2", dark="/d/dark_099.tif",
                                    bright="/d/bright.tif",
                                    samples=[Sample("/d/tiffs/sampleA")]),
                ]),
            CalibrationNode(
                name="CeO2", source=SOURCE_FILE, file_path="/c/ceria.txt",
                corrections=[CorrectionsNode(samples=[Sample("/d2/scan_010.h5")])]),
        ],
        data_root="/d", out_root="/out", settings={"r_bin": 1.0, "fmt": ["csv"]})


# ── the Qt-free guarantee ────────────────────────────────────────────

def test_model_module_imports_no_qt():
    """The whole point of the module split — if this breaks, the pure tests
    start dragging PyQt5 into the parent process and the forked Qt tests in
    this suite become unsafe (see STATE.md).

    Checked in a clean subprocess rather than against this process's
    ``sys.modules``: by the time pytest runs this, another test file in the
    same session may already have imported Qt, which would make an in-process
    check pass vacuously."""
    code = ("import sys, midas_gui.batch_queue;"
            "print([m for m in sys.modules if m.startswith('PyQt5')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"batch_queue pulled Qt in: {out.stdout}"


# ── kinds and labels ─────────────────────────────────────────────────

@pytest.mark.parametrize("path, kind", [
    ("/d/scan.h5", KIND_HDF5), ("/d/scan.hdf5", KIND_HDF5),
    ("/d/scan.nxs", KIND_HDF5), ("/d/SCAN.H5", KIND_HDF5),
    ("/d/scan.vrx.h5", KIND_HDF5),
    ("/d/sampleA", KIND_FOLDER), ("/d/sampleA/", KIND_FOLDER),
    ("/d/frame.tif", KIND_FOLDER),   # a lone frame file is not a sample kind
])
def test_kind_is_decided_by_suffix_alone(path, kind):
    """Suffix-only, so a queue restored from another machine still classifies."""
    assert kind_for_path(path) == kind


@pytest.mark.parametrize("path, label", [
    ("/d/s1/scan_001.h5", "scan_001"),
    ("/d/s1/scan_001.vrx.h5", "scan_001.vrx"),   # compound suffix keeps its inner part
    ("/d/tiffs/sampleA", "sampleA"),
    ("/d/tiffs/sampleA/", "sampleA"),
])
def test_default_label(path, label):
    assert default_label(path) == label


def test_sample_fills_in_kind_and_label():
    s = Sample("/d/s1/scan_001.h5")
    assert (s.kind, s.label, s.enabled, s.dataset) == (KIND_HDF5, "scan_001", True, None)
    f = Sample("/d/tiffs/sampleA")
    assert (f.kind, f.label) == (KIND_FOLDER, "sampleA")


def test_explicit_label_and_kind_are_not_overwritten():
    s = Sample("/d/s1/scan_001.h5", kind=KIND_HDF5, label="my sample")
    assert s.label == "my sample"


# ── traversal ────────────────────────────────────────────────────────

def test_iter_samples_walks_the_tree_in_order():
    q = _queue()
    got = [(c.name, k.name, s.label) for c, k, s in q.iter_samples()]
    assert got == [
        ("AgBH", "Corr 1", "scan_001"), ("AgBH", "Corr 1", "scan_002"),
        ("AgBH", "Corr 2", "sampleA"),
        ("CeO2", "Corrections", "scan_010"),
    ]
    assert q.sample_count() == 4


def test_disabled_samples_are_skipped_when_asked():
    q = _queue()
    q.calibrations[0].corrections[0].samples[1].enabled = False
    assert q.sample_count() == 4
    assert q.sample_count(enabled_only=True) == 3
    assert "scan_002" not in [s.label for _c, _k, s in q.iter_samples(enabled_only=True)]
    assert q.sample_paths() == ["/d/s1/scan_001.h5", "/d/tiffs/sampleA", "/d2/scan_010.h5"]


def test_empty_queue_is_well_behaved():
    q = BatchQueue()
    assert q.sample_count() == 0 and q.sample_paths() == []
    assert list(q.iter_samples()) == []


# ── serialisation ────────────────────────────────────────────────────

def test_queue_round_trips_through_json():
    """This is what a saved project stores — a lossy round trip would silently
    drop a user's queue on reopen."""
    q = _queue()
    back = BatchQueue.from_json(q.to_json())
    assert back.to_json() == q.to_json()
    assert back.data_root == "/d" and back.out_root == "/out"
    assert back.settings == {"r_bin": 1.0, "fmt": ["csv"]}
    cal = back.calibrations[0]
    assert cal.name == "AgBH" and cal.source == SOURCE_TAB2
    assert cal.calib_snapshot == {"Lsd": 1000.0, "BC_y": 1.5}
    assert cal.mask_sources[0]["path"] == "/m/beamstop.tif"
    assert cal.corrections[1].bright == "/d/bright.tif"
    assert back.calibrations[1].using_file() and back.calibrations[1].file_path == "/c/ceria.txt"
    assert isinstance(cal.corrections[0].samples[0], Sample)


def test_json_is_actually_json_serialisable():
    import json
    assert json.loads(json.dumps(_queue().to_json()))["calibrations"][0]["name"] == "AgBH"


@pytest.mark.parametrize("payload", [None, {}, {"calibrations": None}])
def test_from_json_tolerates_missing_and_empty(payload):
    q = BatchQueue.from_json(payload)
    assert q.calibrations == [] and q.settings == {} and q.data_root is None


def test_from_json_tolerates_a_queue_saved_before_a_field_existed():
    """Old projects must load with defaults rather than raising."""
    q = BatchQueue.from_json({"calibrations": [
        {"name": "old", "corrections": [{"samples": [{"path": "/d/a.h5"}]}]}]})
    cal = q.calibrations[0]
    assert cal.source == SOURCE_TAB2 and cal.mask_sources is None
    corr = cal.corrections[0]
    assert (corr.name, corr.dark, corr.bright_mode) == ("Corrections", None, "divide")
    assert corr.samples[0].label == "a" and corr.samples[0].kind == KIND_HDF5
    assert corr.background_scale == 1.0   # saved before the field existed


def test_background_scale_defaults_to_one_and_round_trips():
    assert CorrectionsNode().background_scale == 1.0
    corr = CorrectionsNode(background="/d/bg.tif", background_scale=2.5)
    back = CorrectionsNode.from_json(corr.to_json())
    assert back.background_scale == 2.5


def test_corrections_describe():
    assert CorrectionsNode().describe() == "no corrections"
    assert CorrectionsNode(dark="/d/dk.tif", background="/d/bg.tif").describe() == (
        "dark=dk.tif  background=bg.tif")


# ── source descriptors ───────────────────────────────────────────────

def test_hdf5_sample_source_cfg():
    """Must stay exactly what workers._open_source_cfg dispatches on."""
    cfg = sample_source_cfg(Sample("/d/scan.h5", dataset="exchange/data"))
    assert cfg == {"type": "hdf5", "path": "/d/scan.h5", "dataset": "exchange/data",
                   "chunk_size": None, "combine_op": "mean"}


def test_hdf5_sample_without_a_dataset_falls_back():
    assert sample_source_cfg(Sample("/d/scan.h5"))["dataset"] == DEFAULT_DATASET


def test_folder_sample_source_cfg():
    assert sample_source_cfg(Sample("/d/sampleA")) == {
        "type": "tiff_glob", "path": "/d/sampleA"}


def test_combine_options_are_passed_through_for_hdf5():
    cfg = sample_source_cfg(Sample("/d/scan.h5"), chunk_size=10, combine_op="sum")
    assert (cfg["chunk_size"], cfg["combine_op"]) == (10, "sum")


def test_source_cfg_types_are_the_ones_the_worker_knows():
    for s in (Sample("/d/scan.h5"), Sample("/d/folder")):
        assert sample_source_cfg(s)["type"] in ("hdf5", "tiff_glob",
                                                "tiff_list", "hdf5_stack_glob")


# ── dataset detection ────────────────────────────────────────────────

def test_detect_dataset_prefers_the_first_3d_dataset(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    p = tmp_path / "scan.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("meta/flat", data=np.zeros((4, 4)))        # 2-D, skipped
        f.create_dataset("exchange/data", data=np.zeros((3, 4, 4)))  # 3-D, wanted
    assert detect_dataset(p) == "exchange/data"


def test_detect_dataset_falls_back_to_2d_then_to_the_default(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    p = tmp_path / "flat.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("only/2d", data=np.zeros((4, 4)))
    assert detect_dataset(p) == "only/2d"

    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w"):
        pass
    assert detect_dataset(empty) == DEFAULT_DATASET


def test_detect_dataset_never_raises_on_a_bad_path(tmp_path):
    """A queue is routinely built before every file has landed — the run
    reports a broken file far better than tree-building can."""
    assert detect_dataset(tmp_path / "nope.h5") == DEFAULT_DATASET
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not an hdf5 file")
    assert detect_dataset(junk) == DEFAULT_DATASET
