"""Headless single-calibration runner — ``python -m midas_gui.calib_cli``.

Runs exactly one calibration (one single-detector run, or one Hydra panel) in
its own OS process, invoked by ``workers.CalibrationWorker`` via ``QProcess``.
This is the calibration counterpart of ``batch_cli.py``'s "Run as background
job" pattern, but simpler: no ``screen`` session, no detachment from the GUI's
lifetime — just a plain child process the GUI waits on and streams the log
from, the same way it used to wait on an in-process ``QThread``.

Why a subprocess instead of a ``QThread``: a calibration pipeline call is one
uninterruptible native torch/scipy call that can run for minutes.
``terminate()`` on a ``QThread`` stuck inside one risks corrupting the whole
GUI process (the old ``CalibrationWorker._abort()`` could therefore only
*detach*, not actually stop it). Hydra's Parallel mode also had to disable
per-panel log capture (``capture_stdout=False``) to avoid every concurrent
QThread racing on the same process-global ``sys.stdout``. Running each
calibration as its own child process fixes both: the OS can really kill one
without touching the GUI, and every child owns its own real stdout, so full
per-line log capture is always safe, sequential or parallel, single-detector
or all four Hydra panels at once.

No PyQt import here at all — unlike ``batch_cli.py``'s ``BatchWorker``,
``calib.run_pipeline``/``calib.normalize_result`` are plain functions, not a
QThread subclass, so no ``QApplication`` instance is needed to construct
anything.

Input: ``--job-dir <dir>`` containing
  ``calib_job.json`` — ``{"mode": str, "bright_mode": str, "cfg": {...}}``,
                       the same ``cfg`` dict ``calib.run_pipeline`` has always
                       taken, minus ``"mask"`` (moved to the npz below — a
                       mask array has no JSON representation) and with any
                       ``set`` values (``cfg["refine"]["distortion_coeffs"]``)
                       turned into lists (restored on the way back in, since
                       ``calib._distortion_coeffs`` already accepts either).
  ``calib_job.npz``  — ``image`` (required), and whichever of ``dark``,
                       ``bright``, ``background``, ``mask`` the run uses.
                       ``image``/``dark`` are exactly what used to be passed
                       straight into ``CalibrationWorker``'s constructor:
                       raw, uncorrected, untransformed frames.

Output: prints every log line to stdout, unbuffered (the backend's own
``print()`` calls land here directly — no ``sys.stdout`` redirection needed,
since this process's stdout is real and private to this one job). On success,
writes ``calib_result.pkl`` (the normalized result, pickled) into
``--job-dir`` and exits 0. On failure, prints the traceback and exits 1 — the
launching ``CalibrationWorker`` uses the captured log text as the failure
message, matching the old ``traceback.format_exc()`` convention.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

import midas_gui._paths  # noqa: F401  (KMP_DUPLICATE_LIB_OK / HDF5_USE_FILE_LOCKING env vars —
# this entry point never goes through app.py's import chain, so without this
# import a calibration run reading HDF5 over an NFS-mounted beamline share
# would be exposed to the same file-locking hang _paths.py otherwise guards
# the interactive GUI against, and a calibration backend already known to be
# sensitive to the OpenMP-duplicate-runtime issue would lose that guard too.

JOB_JSON = "calib_job.json"
JOB_NPZ = "calib_job.npz"
RESULT_PKL = "calib_result.pkl"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m midas_gui.calib_cli",
        description="Run one midas-gui calibration (single-detector or one "
                    "Hydra panel) in a separate process.")
    p.add_argument("--job-dir", required=True,
                   help=f"Directory holding {JOB_JSON}/{JOB_NPZ}; "
                        f"{RESULT_PKL} is written here on success.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    job_dir = Path(args.job_dir)

    try:
        job = json.loads((job_dir / JOB_JSON).read_text())
        arrays = np.load(job_dir / JOB_NPZ)
    except Exception:
        print(traceback.format_exc(), flush=True)
        return 1

    try:
        # Imported inside the try block, same as every other heavy-import
        # site in this codebase (calib.py itself, workers.py): a backend
        # import failure should read as a calibration failure, not a Python
        # traceback with no "[calibrate]" context.
        from midas_gui import calib
        from midas_gui.helpers import apply_field_corrections

        mode = job["mode"]
        cfg = dict(job["cfg"])
        bright_mode = job.get("bright_mode", "divide")

        def _field(name):
            return arrays[name] if name in arrays.files else None

        image = np.asarray(arrays["image"], dtype=np.float32)
        dark = _field("dark")
        bright = _field("bright")
        background = _field("background")
        mask = _field("mask")

        # Mirrors workers.CalibrationWorker.run()'s old body exactly: bright/
        # background correction and mask zeroing happen here, on the raw
        # frame, before the pipeline ever sees it. dark is NOT pre-applied —
        # calib.run_pipeline hands it to the backend, which subtracts it
        # itself (same as before).
        if bright is not None or background is not None:
            image = apply_field_corrections(
                image, dark=None, bright=bright,
                bright_mode=bright_mode, background=background,
            ).astype(np.float32)
            print(
                f"[calibrate] applied "
                f"{'bright(' + bright_mode + ') ' if bright is not None else ''}"
                f"{'background ' if background is not None else ''}correction",
                flush=True)
        if mask is not None:
            image = image.copy()
            image[mask.astype(bool)] = 0.0   # zero sentinels before calibration

        # refine.distortion_coeffs travels as a list over JSON (sets aren't
        # JSON-serializable); calib._distortion_coeffs() already accepts
        # either a set or any other iterable via set(coeffs), so no
        # conversion is needed here beyond what json.loads already gave us.

        raw = calib.run_pipeline(mode, image, dark, cfg)
        NY, NZ = calib.effective_pixel_counts(image, cfg.get("im_trans", ()))
        result = calib.normalize_result(
            raw, mode, NY=NY, NZ=NZ, pxY=cfg["pxY"], pxZ=cfg.get("pxZ"),
            wavelength=cfg["wavelength"], panel_layout=cfg.get("panel_layout"),
            scratch=cfg.get("scratch_dir"), stem=cfg.get("save_stem", ""))
        result._calibrant_name = cfg["calibrant"]

        # A torch tensor has to leave this process CPU-resident and detached
        # from autograd: the GUI process may not share a CUDA context (or
        # have a GPU at all), and a tensor still wired into a graph can't be
        # pickled.
        rc = getattr(result, "residual_corr_map", None)
        if rc is not None and hasattr(rc, "detach"):
            result.residual_corr_map = rc.detach().cpu()

        import pickle
        with open(job_dir / RESULT_PKL, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return 0
    except Exception:
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
