"""Batch Correction across several worker threads — ``BatchCorrectionCoordinator``.

Asked for at the beamline: "there should be an option to run batch
correction as a background job in parallel like the caking functionality."
Files are the unit of work: chunks never cross a file boundary and each
input yields its own output per method, so nothing is shared — except the
per-froot beam-monitor CSV, which is the one thing a split would otherwise
write twice with half the rows each. That merge is the only genuinely new
logic here, so most of this file is about proving a parallel run and a
sequential one cannot differ.

``forked`` per .context/DECISIONS.md — constructs QThread-derived workers.
"""
import csv

import numpy as np
import pytest

pytestmark = pytest.mark.forked

SHAPE = (6, 5)
N_RAW = 12
DARK_LEVEL = 10.0


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_file(path, *, n=N_RAW, base=0.0):
    h5py = pytest.importorskip("h5py")
    data = np.stack([np.full(SHAPE, base + i, np.float32) for i in range(n)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("exchange/data", data=data)
        f.create_dataset("exchange/data_dark",
                         data=np.full((2,) + SHAPE, DARK_LEVEL, np.float32))
        f.create_dataset("instrument/Scalers/E/US_IC",
                         data=np.arange(2 * n, dtype=np.float64))
    return path


@pytest.fixture
def six_files(tmp_path):
    """Six numbered files of ONE froot — the case that matters, because it
    is what a real scan looks like and what forces the CSV merge."""
    src = tmp_path / "scan"
    return [_make_file(src / f"scan_{10 + i:06d}.vrx.h5", base=100.0 * i)
            for i in range(6)]


def _run(paths, out_dir, *, n_workers, ops=("mean",), chunk_size=4):
    """Drive a coordinator to completion, synchronously."""
    from midas_gui.workers import BatchCorrectionCoordinator
    coord = BatchCorrectionCoordinator(
        paths, out_dir=str(out_dir), n_workers=n_workers,
        dataset="exchange/data", chunk_size=chunk_size, op=list(ops),
        out_dataset="exchange/data", auto_dark=True,
        dark_dataset="exchange/data_dark", clip_negatives=True)
    got, fails = [], []
    coord.finished.connect(got.append)
    coord.failed.connect(fails.append)
    coord.start()
    coord.wait()
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance().processEvents()
    assert not fails, fails[0]
    assert got, "finished never fired"
    return got[0]


def _read(path, dataset="exchange/data"):
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "r") as f:
        return np.asarray(f[dataset])


def _csvs(root):
    return sorted(p for p in root.rglob("*_bc.csv"))


# ── the split itself ────────────────────────────────────────────────────

@pytest.mark.parametrize("n_paths,n,expected", [
    (6, 3, [2, 2, 2]),
    (7, 3, [3, 2, 2]),      # remainder goes to the earliest workers
    (2, 5, [1, 1]),         # never more workers than files
    (1, 4, [1]),
])
def test_slices_are_contiguous_and_never_empty(n_paths, n, expected):
    from midas_gui.workers import BatchCorrectionCoordinator as C
    paths = list(range(n_paths))
    groups = C._slices(paths, min(n, n_paths))
    assert [len(g) for g in groups] == expected
    assert [p for g in groups for p in g] == paths, "a file was lost or reordered"


def test_no_files_is_not_a_crash():
    from midas_gui.workers import BatchCorrectionCoordinator as C
    assert C._slices([], 3) == []


# ── equivalence: the whole point ────────────────────────────────────────

def test_parallel_writes_the_same_files_as_sequential(app, six_files, tmp_path):
    seq = _run(six_files, tmp_path / "seq", n_workers=1)
    par = _run(six_files, tmp_path / "par", n_workers=3)

    rel_seq = sorted(str(p).split("/seq/", 1)[1] for p in seq)
    rel_par = sorted(str(p).split("/par/", 1)[1] for p in par)
    assert rel_seq == rel_par
    assert len(rel_seq) == len(six_files)


def test_parallel_pixels_are_identical_to_sequential(app, six_files, tmp_path):
    """Correction is per-file and deterministic, so concurrency must not
    change a single value — not 'close', identical."""
    seq = sorted(_run(six_files, tmp_path / "seq", n_workers=1))
    par = sorted(_run(six_files, tmp_path / "par", n_workers=4))
    for a, b in zip(seq, par):
        assert np.array_equal(_read(a), _read(b)), f"{a} != {b}"


def test_every_method_survives_the_split(app, six_files, tmp_path):
    seq = sorted(_run(six_files, tmp_path / "s", n_workers=1,
                      ops=("mean", "median", "sum", "max")))
    par = sorted(_run(six_files, tmp_path / "p", n_workers=3,
                      ops=("mean", "median", "sum", "max")))
    assert len(seq) == len(par) == 4 * len(six_files)
    for a, b in zip(seq, par):
        assert np.array_equal(_read(a), _read(b))


# ── the merge ───────────────────────────────────────────────────────────

def test_one_froot_split_across_workers_still_yields_one_csv(app, six_files, tmp_path):
    """The reason children do not write their own: six files of one scan
    spread over three workers must leave ONE csv holding every row, not
    three each holding a third."""
    out = tmp_path / "par"
    _run(six_files, out, n_workers=3)
    found = _csvs(tmp_path)
    assert len(found) == 1, f"expected one merged CSV, got {found}"


def test_the_merged_csv_matches_the_sequential_one(app, six_files, tmp_path):
    """A monitor CSV is written one level ABOVE the output folder (the
    output dir is per-detector, the readings are per-exposure), so the two
    runs need their own <froot>/<detector> trees or they would write the
    same file and this would compare a run with itself."""
    _run(six_files, tmp_path / "seq" / "eiger2", n_workers=1)
    _run(six_files, tmp_path / "par" / "eiger2", n_workers=3)
    (seq_csv,) = _csvs(tmp_path / "seq")
    (par_csv,) = _csvs(tmp_path / "par")

    def rows(p):
        with open(p, newline="") as fh:
            return list(csv.reader(fh))

    assert rows(seq_csv), "the sequential run wrote an empty CSV"
    assert rows(seq_csv) == rows(par_csv), \
        "the merged CSV differs from the one a sequential run writes"


def test_every_source_file_appears_in_the_merged_csv(app, six_files, tmp_path):
    out = tmp_path / "par"
    _run(six_files, out, n_workers=3)
    (path,) = _csvs(tmp_path)
    with open(path, newline="") as fh:
        text = fh.read()
    for p in six_files:
        assert p.name in text, f"{p.name} missing from the merged CSV"


# ── cancel and failure ──────────────────────────────────────────────────

def test_cancel_stops_every_worker_and_finishes_once(app, six_files, tmp_path):
    from midas_gui.workers import BatchCorrectionCoordinator
    coord = BatchCorrectionCoordinator(
        six_files, out_dir=str(tmp_path / "out"), n_workers=3,
        dataset="exchange/data", chunk_size=4, op=["mean"],
        out_dataset="exchange/data", auto_dark=True,
        dark_dataset="exchange/data_dark", clip_negatives=True)
    done = []
    coord.finished.connect(done.append)
    coord.start()
    coord.cancel()
    coord.wait()
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance().processEvents()
    assert len(done) == 1, f"finished fired {len(done)} times, want exactly 1"
    assert not coord.isRunning()


def test_a_bad_source_reports_once_not_once_per_worker(app, tmp_path):
    """N workers hitting the same unreadable share would otherwise raise N
    error dialogs on the tab."""
    from midas_gui.workers import BatchCorrectionCoordinator
    missing = [tmp_path / f"nope_{i}.h5" for i in range(4)]
    coord = BatchCorrectionCoordinator(
        missing, out_dir=str(tmp_path / "out"), n_workers=4,
        dataset="exchange/data", chunk_size=2, op=["mean"],
        out_dataset="exchange/data", auto_dark=False,
        dark_dataset="exchange/data_dark", clip_negatives=True)
    fails, done = [], []
    coord.failed.connect(fails.append)
    coord.finished.connect(done.append)
    coord.start()
    coord.wait()
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance().processEvents()
    assert len(fails) == 1, f"failed fired {len(fails)} times, want 1"


# ── thread lifetime ─────────────────────────────────────────────────────

def test_the_children_are_not_qt_children_of_the_coordinator(app, six_files, tmp_path):
    """A QThread that is a Qt child dies with its parent via the C++
    parent-owns-children cascade — while still running, that is a fatal
    "QThread: Destroyed while thread is still running" abort rather than an
    exception, and PyQt's keep-alive for running threads does not prevent
    it. Reported from the beamline as a crash on aborting a run.
    """
    from midas_gui.workers import BatchCorrectionCoordinator
    coord = BatchCorrectionCoordinator(
        six_files, out_dir=str(tmp_path / "out"), n_workers=2,
        dataset="exchange/data", chunk_size=4, op=["mean"],
        out_dataset="exchange/data", auto_dark=True,
        dark_dataset="exchange/data_dark", clip_negatives=True)
    coord.start()
    try:
        assert coord._workers, "no workers were created"
        for w in coord._workers:
            assert w.parent() is None, \
                "a worker is parented to the coordinator — it would be " \
                "destroyed mid-run if the coordinator were released"
    finally:
        coord.cancel()
        coord.wait()
