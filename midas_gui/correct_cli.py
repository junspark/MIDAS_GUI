"""Headless Batch Correction runner — ``python -m midas_gui.correct_cli``.

The Batch Correction counterpart of ``batch_cli``: runs exactly one
correction job with no GUI, so it can be supervised by an external
``screen`` session (see ``job_queue.JobQueuePanel``) and outlive the GUI
process that launched it. Builds the same ``BatchCorrectionCoordinator``
the Batch Correction tab builds and calls its workers' ``run()``
synchronously — no Qt event loop is needed; a ``QApplication`` instance is
only required so PyQt will let us construct the QThread-derived workers.

Prints structured lines to stdout, unbuffered:
  ``[correct] PROGRESS <done>/<total>``  — parsed by JobQueuePanel, whose
                                           regex accepts this tag as well as
                                           ``[batch]``
  ``[correct] FINISHED n=<count>``

On success also writes ``_bg_job_correction.json`` into ``--out-dir``,
listing the output paths — this process has no Qt signals reaching the GUI
that launched it, so that file is the one thing the tab can read afterwards
to say what the job produced.

Every option here mirrors one field of ``BatchCorrectionTab._worker_kwargs``.
The two must stay the same shape, or a background job and a "Run correction"
of the same settings write different files — the bug the equivalent
docstring in ``batch_cli._omega_cfg`` exists to prevent, and what
``tests/test_correct_cli.py`` pins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import midas_gui._paths  # noqa: F401  (KMP_DUPLICATE_LIB_OK / HDF5_USE_FILE_LOCKING —
# this entry point never goes through app.py's import chain, and a long
# background correction reading HDF5 over an NFS beamline share is exactly
# the file-locking hang _paths.py guards against. Same reason batch_cli
# imports it.)

from midas_gui.frame_correct import COMPRESSIONS, OPS, OUTPUT_DTYPES
from midas_gui.helpers import CORRECTION_EXT, CORRECTION_SUFFIX


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m midas_gui.correct_cli",
        description="Run one Batch Correction job headlessly.")
    p.add_argument("--paths", nargs="+", required=True,
                   help="Source HDF5 files (chunks never cross a file)")
    p.add_argument("--dataset", default="exchange/data",
                   help="Image stack dataset inside each source file")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ops", default="mean",
                   help=f"Comma-separated methods, one output set each: "
                        f"{','.join(OPS)}")
    p.add_argument("--chunk-size", type=int, default=0,
                   help="Raw sub-frames combined per output frame; 0 = whole file")
    p.add_argument("--raw-start", type=int, default=None,
                   help="First raw sub-frame to use (single-file sources)")
    p.add_argument("--raw-end", type=int, default=None,
                   help="Last raw sub-frame to use, inclusive")

    p.add_argument("--suffix", default=CORRECTION_SUFFIX)
    p.add_argument("--ext", dest="out_ext", default=CORRECTION_EXT)
    p.add_argument("--out-dataset", default="exchange/data")
    p.add_argument("--out-dtype", default="float32", choices=list(OUTPUT_DTYPES))
    # "none" is a real member of COMPRESSIONS, not a missing value --
    # frame_correct.compression_kwargs treats it and None alike.
    p.add_argument("--compression", default="none", choices=list(COMPRESSIONS))
    p.add_argument("--level", type=int, default=4, help="gzip level")
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--auto-dark", dest="auto_dark", action="store_true",
                   default=True, help="Resolve each file's own dark (default)")
    p.add_argument("--no-auto-dark", dest="auto_dark", action="store_false",
                   help="Use only the --dark image, for every file")
    p.add_argument("--dark-dataset", default="exchange/data_dark")
    p.add_argument("--clip-negatives", dest="clip_negatives",
                   action="store_true", default=True)
    p.add_argument("--no-clip-negatives", dest="clip_negatives",
                   action="store_false")

    # Field images cannot travel on a command line, so the tab writes them
    # beside the output as _bg_job_*.tif and passes the paths -- the same
    # thing BatchTab._run_as_job does for its mask/dark/bright/background.
    p.add_argument("--dark", default=None, help="Dark field image (tif/h5)")
    p.add_argument("--bright", default=None, help="Bright field image (tif/h5)")
    p.add_argument("--background", default=None, help="Background image (tif/h5)")
    p.add_argument("--bright-mode", default="divide",
                   choices=["divide", "subtract"])

    p.add_argument("--ion-csv-extras", default="",
                   help="Comma-separated optional monitor-CSV column groups: "
                        "env,motors")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Files are processed by this many concurrent workers")
    return p


def _load_field(path):
    """Same loader batch_cli uses, so a field image means the same thing to
    both entry points."""
    if not path:
        return None
    from midas_gui.helpers import _load_image
    return _load_image(path)


def _worker_kwargs(args) -> dict:
    """The kwargs ``BatchCorrectionWorker`` takes, from parsed argv.

    Mirrors ``BatchCorrectionTab._worker_kwargs`` exactly — see this
    module's docstring for why that matters.
    """
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    bad = [o for o in ops if o not in OPS]
    if bad:
        raise SystemExit(f"[correct] ERROR: unknown method(s) {', '.join(bad)}; "
                         f"expected any of {', '.join(OPS)}")
    if not ops:
        raise SystemExit("[correct] ERROR: --ops needs at least one method")
    return dict(
        dataset=args.dataset,
        chunk_size=args.chunk_size or None,
        op=ops,
        out_dataset=args.out_dataset,
        suffix=args.suffix,
        out_ext=args.out_ext,
        dark=_load_field(args.dark), bright=_load_field(args.bright),
        background=_load_field(args.background),
        bright_mode=args.bright_mode,
        auto_dark=bool(args.auto_dark),
        dark_dataset=args.dark_dataset,
        clip_negatives=bool(args.clip_negatives),
        compression=args.compression,
        level=args.level,
        shuffle=bool(args.shuffle),
        out_dtype=args.out_dtype,
        ion_csv_extras=tuple(g for g in args.ion_csv_extras.split(",") if g),
        raw_start=args.raw_start, raw_end=args.raw_end)


def _write_results_sidecar(out_dir: str, outputs: list) -> None:
    """What the job produced, for the tab to read when the job lands."""
    path = Path(out_dir) / "_bg_job_correction.json"
    path.write_text(json.dumps({"n": len(outputs), "outputs": outputs},
                               indent=2))


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    from midas_gui.helpers import check_output_dir_writable
    reason = check_output_dir_writable(args.out_dir)
    if reason:
        print(f"[correct] ERROR: {reason}", flush=True)
        return 1
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # A QApplication is required to construct QThread/QObject subclasses even
    # though no event loop ever runs -- the workers' run() is invoked
    # directly. Same reason batch_cli.main builds one.
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])

    from midas_gui.workers import BatchCorrectionCoordinator

    coord = BatchCorrectionCoordinator(
        args.paths, out_dir=args.out_dir,
        n_workers=max(1, args.n_workers), **_worker_kwargs(args))

    exit_code = [0]
    outputs: list = []

    def _on_progress(done, total, msg):
        print(f"[correct] PROGRESS {done}/{total}", flush=True)
        if msg:
            print(f"[correct] {msg}", flush=True)

    def _on_failed(msg):
        print(f"[correct] ERROR: {msg}", flush=True)
        exit_code[0] = 1

    def _on_finished(paths):
        outputs.extend(paths)
        print(f"[correct] FINISHED n={len(paths)}", flush=True)

    coord.progress.connect(_on_progress)
    coord.failed.connect(_on_failed)
    coord.finished.connect(_on_finished)

    # start() spawns the workers as real QThreads; with no event loop
    # running, wait() is what joins them. The children's queued signals are
    # delivered by the direct connections above (same thread affinity as
    # batch_cli's synchronous run()).
    coord.start()
    coord.wait()
    app.processEvents()

    if exit_code[0] == 0:
        try:
            _write_results_sidecar(args.out_dir, outputs)
        except Exception as e:
            print(f"[correct] WARNING: could not write results sidecar: {e}",
                  flush=True)

    del app
    return exit_code[0]


if __name__ == "__main__":
    sys.exit(main())
