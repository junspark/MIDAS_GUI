"""Headless Batch Integrate runner — ``python -m midas_gui.batch_cli``.

Runs exactly one Batch Integrate job with no GUI, so it can be supervised by
an external ``screen`` session (see ``job_queue.JobQueuePanel``) and outlive
the GUI process that launched it. Builds the same ``IntegrationSpec`` /
``BatchWorker`` the Batch Integrate tab uses, calling ``BatchWorker.run()``
synchronously (no Qt event loop needed — a ``QApplication`` instance is only
required so PyQt will let us construct the QThread-derived worker object).

Prints two kinds of structured lines to stdout, unbuffered:
  ``[batch] PROGRESS <done>/<total>``   — one per frame, parsed by JobQueuePanel
  ``[launcher] DONE exit=<code>``       — printed by the launching shell
                                          command after this process exits,
                                          not by this module itself.

On success, also writes ``_bg_job_results.npz`` into ``--out-dir`` — the
arrays ``BatchTab._populate_plots_from_attempt`` needs to fill the
Waterfall/Stacked-profiles/Eta-R cakes tabs. This process has no Qt signals
reaching the GUI that launched it (it's a detached `screen` session), and
the user's chosen ``--fmts`` don't necessarily round-trip that shape (plain
csv doesn't), so this sidecar is the one thing ``JobQueuePanel``'s
``on_job_done`` callback can always rely on.
"""
from __future__ import annotations

import argparse
import sys

import midas_gui._paths  # noqa: F401  (KMP_DUPLICATE_LIB_OK / HDF5_USE_FILE_LOCKING env vars —
# this entry point never goes through app.py's import chain, so without this
# import it got neither: a long-running background job reading HDF5 over an
# NFS-mounted beamline share was just as exposed to the file-locking hang
# _paths.py now guards against as the interactive GUI is.)

from midas_gui.constants import POL_PLANE_HORIZONTAL_ETA_DEG


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m midas_gui.batch_cli",
        description="Run one Batch Integrate job with no GUI.")
    p.add_argument("--calib-file", required=True,
                   help="Calibration source: paramstest.txt / calibration.json / .poni")
    p.add_argument("--r-bin", type=float, default=1.0, help="R bin size (px)")
    p.add_argument("--eta-bin", type=float, default=5.0, help="eta bin size (deg)")
    p.add_argument("--r-min", type=float, default=None, help="Rmin (px); default backend auto")
    p.add_argument("--r-max", type=float, default=None, help="Rmax (px); default backend auto")

    p.add_argument("--source-type", required=True,
                   choices=["tiff_glob", "hdf5", "tiff_list", "hdf5_stack_glob"])
    p.add_argument("--source-path", help="Folder/glob (tiff_glob) or file path (hdf5)")
    p.add_argument("--source-paths", nargs="+",
                   help="Explicit file list (tiff_list / hdf5_stack_glob)")
    p.add_argument("--dataset", default="frames",
                   help="HDF5 dataset path (hdf5 / hdf5_stack_glob sources only)")

    p.add_argument("--out-dir", required=True)
    p.add_argument("--fmts", default="csv",
                   help="Comma-separated output format keys, e.g. csv,h5,zarr "
                        "(see constants.OUTPUT_FORMATS values)")
    p.add_argument("--kernel", default="subpixel2",
                   choices=["hard", "subpixel2", "subpixel4", "polygon"])

    # frame-start/frame-end are FILE/SCAN NUMBERS (parsed from filenames) for
    # every multi-file source-type, not frame indices — None (the default,
    # unset) means unbounded on that side. For a single "hdf5" source they
    # instead mean a 0-based inclusive RAW SUB-FRAME range within that one
    # file (see widgets.DataLoaderPanel.source_cfg's "hdf5" branch), applied
    # before chunk-size combines whatever survives. chunk-size/combine-op
    # mirror the GUI's "Combine sub-frames" control: 0 = combine everything
    # selected into one frame, 1 (default) = no combining.
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=1)
    p.add_argument("--combine-op", default="mean",
                   choices=["mean", "sum", "max", "median"])

    # Rotation angle per output frame. These mirror the Batch Integrate tab's
    # own omega widgets one-for-one (see tab_batch.BatchTab._omega_cfg), so a
    # background job records the same angles an in-process run of the same
    # settings would. Defaulting start/step to 0.0 is not a sentinel for
    # "unset": it is a genuine 0 deg on every frame, which is the right answer
    # for a stationary sample (see cake_params.omega_for_window).
    p.add_argument("--ome-start", type=float, default=0.0,
                   help="OME_START: omega (deg) of raw sub-frame 0")
    p.add_argument("--ome-step", type=float, default=0.0,
                   help="OME_STEP: omega increment (deg) per raw sub-frame")
    p.add_argument("--ome-channel", default="",
                   help="1-D HDF5 dataset holding a measured omega per raw "
                        "sub-frame; blank falls back to --ome-start/--ome-step")
    p.add_argument("--ome-collapse", action="store_true",
                   help="Averaged/summed data: give every output frame the one "
                        "run-wide mean omega instead of a per-frame value")

    p.add_argument("--zarr-grouping", default="frame",
                   choices=["frame", "file", "run"],
                   help="How many integrated frames share one .zarr.zip: one "
                        "each (default), one per source file (i.e. per "
                        "rotation), or one for the whole run")

    p.add_argument("--multi-azimuth", action="store_true")
    p.add_argument("--weighted", dest="weighted", action="store_true", default=True)
    p.add_argument("--no-weighted", dest="weighted", action="store_false")

    p.add_argument("--polarization", action="store_true")
    p.add_argument("--pol-fraction", type=float, default=0.99)
    p.add_argument("--pol-plane", type=float,
                   default=POL_PLANE_HORIZONTAL_ETA_DEG,
                   help="Azimuth of the polarization plane in MIDAS eta (deg). "
                        "MIDAS eta is measured from vertical, so the default 90 "
                        "is the horizontal ring plane; 0 is vertical.")
    p.add_argument("--solid-angle", action="store_true")

    p.add_argument("--variance", action="store_true", help="Compute per-bin sigma")
    p.add_argument("--error-model", default="poisson",
                   choices=["poisson", "azimuthal", "hybrid"])

    p.add_argument("--mask", default=None, help="Mask image (tif/h5); nonzero = masked")
    p.add_argument("--dark", default=None, help="Dark field image (tif/h5)")
    p.add_argument("--bright", default=None, help="Bright field image (tif/h5)")
    p.add_argument("--background", default=None, help="Background image (tif/h5)")
    p.add_argument("--bright-mode", default="divide", choices=["divide", "subtract"])

    p.add_argument("--monitor-file", default=None,
                   help="Text file, one monitor value per line")
    return p


def _source_cfg(args) -> dict:
    combine = {"chunk_size": args.chunk_size or None, "combine_op": args.combine_op}
    if args.source_type == "tiff_glob":
        if not args.source_path:
            raise SystemExit("--source-path is required for --source-type tiff_glob")
        return {"type": "tiff_glob", "path": args.source_path,
                "frame_start": args.frame_start, "frame_end": args.frame_end, **combine}
    if args.source_type == "hdf5":
        if not args.source_path:
            raise SystemExit("--source-path is required for --source-type hdf5")
        return {"type": "hdf5", "path": args.source_path, "dataset": args.dataset,
                "frame_start": args.frame_start, "frame_end": args.frame_end, **combine}
    if args.source_type == "tiff_list":
        if not args.source_paths:
            raise SystemExit("--source-paths is required for --source-type tiff_list")
        return {"type": "tiff_list", "paths": list(args.source_paths),
                "frame_start": args.frame_start, "frame_end": args.frame_end, **combine}
    if args.source_type == "hdf5_stack_glob":
        if not args.source_paths:
            raise SystemExit("--source-paths is required for --source-type hdf5_stack_glob")
        return {"type": "hdf5_stack_glob", "paths": list(args.source_paths),
                "dataset": args.dataset,
                "frame_start": args.frame_start, "frame_end": args.frame_end, **combine}
    raise SystemExit(f"Unknown --source-type: {args.source_type}")


def _omega_cfg(args) -> dict:
    """The rotation half of the run configuration, in the shape
    ``BatchWorker`` takes it. Mirrors ``BatchTab._omega_cfg`` exactly — the two
    must stay the same shape or a background job and an in-process run of the
    same settings write different angles, which is the bug this function
    exists to prevent."""
    return {"start": float(args.ome_start), "step": float(args.ome_step),
            "channel": (args.ome_channel or "").strip(),
            "collapse": bool(args.ome_collapse)}


def _load_field(path):
    if not path:
        return None
    from midas_gui.helpers import _load_image
    return _load_image(path)


def _build_corrections(args):
    pol = sa = None
    if args.polarization:
        from midas_integrate_v2 import PolarizationCorrection
        pol = PolarizationCorrection(pol_fraction=args.pol_fraction,
                                     pol_plane_eta_deg=args.pol_plane)
    if args.solid_angle:
        from midas_integrate_v2 import SolidAngleCorrection
        sa = SolidAngleCorrection()
    return (pol, sa)


def _write_results_sidecar(out_dir: str, data: dict) -> None:
    """Persist the ``r_axis_px``/``profiles``/``frame_ids``/``eta_axis`` arrays
    a finished ``BatchWorker`` run carries as ``_bg_job_results.npz`` in
    ``out_dir`` — see the module docstring for why. No-op if the run produced
    nothing to plot (``n == 0``)."""
    import numpy as np
    from pathlib import Path

    r_axis = data.get("r_axis_px")
    profiles = data.get("profiles")
    if r_axis is None or profiles is None or len(profiles) == 0:
        return
    frame_ids = data.get("frame_ids") or list(range(len(profiles)))
    payload = {
        "r_axis_px": np.asarray(r_axis),
        "profiles": np.asarray(profiles),
        "frame_ids": np.array([str(f) for f in frame_ids]),
    }
    eta_axis = data.get("eta_axis")
    if eta_axis is not None:
        payload["eta_axis_deg"] = np.asarray(eta_axis)
    np.savez(Path(out_dir) / "_bg_job_results.npz", **payload)


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    from midas_gui.helpers import check_output_dir_writable
    reason = check_output_dir_writable(args.out_dir)
    if reason:
        print(f"[batch] ERROR: {reason}", flush=True)
        return 1

    # A QApplication instance is required to construct QThread/QObject
    # subclasses (BatchWorker) even though we never call .start() or run an
    # event loop — .run() is invoked directly, synchronously, on this thread.
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])

    from midas_gui.helpers import spec_from_geometry_file, full_calibration_snapshot
    from midas_gui.workers import BatchWorker

    spec = spec_from_geometry_file(args.calib_file, args.r_bin, args.eta_bin,
                                   r_min=args.r_min, r_max=args.r_max)
    # (None, True, path) is the "file-only, no live CalibrationResult" call
    # pattern full_calibration_snapshot already supports (it builds a
    # duck-typed result via helpers.result_ns_from_geometry_file itself).
    calib_snapshot, _calib_note = full_calibration_snapshot(None, True, args.calib_file)
    src_cfg = _source_cfg(args)
    mask = _load_field(args.mask)
    if mask is not None:
        mask = (mask != 0)
    dark = _load_field(args.dark)
    bright = _load_field(args.bright)
    background = _load_field(args.background)
    fmts = [f.strip() for f in args.fmts.split(",") if f.strip()]
    corrections = _build_corrections(args)
    variance_cfg = {"error_model": args.error_model} if args.variance else None
    # No frame_range kwarg: start/end/chunk_size filtering is baked into
    # src_cfg itself (see _source_cfg) and applied by workers._open_source_cfg
    # before the source is opened — BatchWorker defaults frame_range to
    # (0, None, 1), i.e. "process everything the source yields."

    worker = BatchWorker(
        spec, src_cfg, mask, args.out_dir, fmts, args.kernel,
        corrections, variance_cfg,
        monitor_file=args.monitor_file,
        dark=dark, bright=bright, background=background, bright_mode=args.bright_mode,
        weighted=args.weighted, multi_azimuth=args.multi_azimuth,
        omega_cfg=_omega_cfg(args),
        zarr_grouping=args.zarr_grouping,
        im_trans=tuple(spec.TransOpt or ()), calibration_snapshot=calib_snapshot)

    exit_code = [0]

    def _on_progress(done, total):
        print(f"[batch] PROGRESS {done}/{total}", flush=True)

    def _on_log(line):
        print(line, flush=True)

    def _on_failed(msg):
        print(f"[batch] ERROR: {msg}", flush=True)
        exit_code[0] = 1

    def _on_finished(data):
        n = data.get("n", 0)
        out = data.get("out_paths") or []
        print(f"[batch] FINISHED n={n} out_paths={len(out)}", flush=True)
        if not data.get("aborted") and exit_code[0] == 0:
            try:
                _write_results_sidecar(args.out_dir, data)
            except Exception as e:
                print(f"[batch] WARNING: could not write GUI results sidecar: {e}",
                     flush=True)

    worker.progress.connect(_on_progress)
    worker.log_line.connect(_on_log)
    worker.failed.connect(_on_failed)
    worker.finished.connect(_on_finished)
    worker.run()

    del app
    return exit_code[0]


if __name__ == "__main__":
    sys.exit(main())
