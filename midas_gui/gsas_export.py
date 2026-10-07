"""Export one "final" Batch-Integrate attempt as a native GSAS-II zarr.

Writes ``<out_path>`` (a ``.zarr.zip``) using
``midas_integrate_v2.io.zarr_gsas.write_gsas_zarr_zip`` directly — that
function already reproduces the exact layout GSAS-II's built-in
``G2pwd_MIDAS.py`` "MIDAS zarr" importer (and MIDAS's own
``gsas_ii_refine.py``) expect, bit-for-bit. This module's job is only to
rebuild the inputs that writer needs (a spec, a per-frame cake, a bin-area
array) from ONE attempt already logged in a midas-gui project file — the
project's append-only attempt history is never touched or exported wholesale.

Layout parity with Batch Integrate is a requirement, not a coincidence: a
``.zarr.zip`` must read the same way whichever path wrote it. Both call the
same backend writer, so the arrays and groups match by construction; the
``provenance_history`` entry in the root attrs — including the
``instrument_params`` geometry snapshot — is stamped here to match too. See
``tests/test_zarr_layout_parity.py``, which writes one file by each path and
diffs their structure.

A ``<out_path>.provenance.json`` sidecar carries, in addition, the attempt's
full metadata (params, hashed input paths, environment snapshot, calibration
snapshot) verbatim plus a few export-specific fields — attempt-level history
the Batch Integrate path has no equivalent for.

That sidecar is for us, not for GSAS-II, and is deliberately invisible to it.
GSAS-II does read two sidecars, but they are plain text at
``os.path.splitext(filename)[0]`` — i.e. ``<stem>.zarr.samprm`` and
``<stem>.zarr.instprm`` — and anything in them OVERRIDES the zip. We write
neither, so nothing here can perturb a GSAS-II import. Worth knowing if that
changes: ``.samprm`` is the only route by which sample metadata reaches a
GSAS-II histogram at all. ``readMidas`` does read ``Temperature``/``Pressure``
off each ``OmegaSumFrame`` dataset's attrs (which we write), but then assigns
them into ``sampleprmList``, a list of tuples — a ``TypeError`` swallowed by a
bare ``except``, so those values are dropped upstream. Its other default worth
noting: an un-annotated histogram gets ``InstrName = 'APS 1-ID'``.

Scope (v1): single-detector Batch Integrate attempts only, R-uniform binning
only (a Q-uniform attempt's stored ``r_axis_px`` is Q-rebinned, not a simple
function of ``spec``, so the geometry/bin-area can't be reconstructed from it
here). Works whether the attempt's stored ``profiles`` is 2-D (today's
default, single full-circle profile per frame — degenerates to one azimuth)
or 3-D (multi-azimuth "cake" mode, see ``tab_batch.py``'s checkbox).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from midas_gui import h5_metadata
from midas_gui import project
from midas_gui import provenance as prov
from midas_gui.helpers import _build_spec, _apply_im_trans
from midas_gui.workers import build_integration_context


def _read_embedded_mask(project_path, ref: str) -> Optional[np.ndarray]:
    with h5py.File(str(project_path), "r") as f:
        grp = f.get(ref.lstrip("/"))
        if grp is None or "mask" not in grp:
            return None
        return grp["mask"][()]


def _source_metadata_snapshot(meta: dict, n_frames: int):
    """The source HDF5's ``instrument/`` tree for an already-logged attempt,
    aligned to the ``n_frames`` this export is about to write.

    ``None`` whenever there is nothing to copy or no confident way to copy it:
    a TIFF-backed attempt, an attempt whose recorded source files have since
    moved, or one whose frames span several HDF5 files (a single snapshot
    can't honestly represent two files' PVs, and picking one would silently
    mislabel the rest). Reopening the recorded source is reconstruction after
    the fact, so it fails quietly and the export goes out without the tree
    rather than not going out at all.

    Frame alignment is delegated to ``_HDF5StackGlobSource`` rather than
    recomputed here: rebuilding the source from the same ``src_cfg`` the run
    used is the only way to be sure this export's notion of "which raw
    sub-frames are behind output frame i" matches the one Batch Integrate
    already wrote into the per-frame stores.
    """
    src_cfg = ((meta.get("inputs") or {}).get("src_cfg")) or {}
    if src_cfg.get("type") not in ("hdf5", "hdf5_stack_glob"):
        return None
    try:
        from midas_gui.workers import _open_source_cfg
        source = _open_source_cfg(dict(src_cfg))
        get_ctx = getattr(source, "h5_context_for_index", None)
        if get_ctx is None:
            return None
        contexts = [get_ctx(i) for i in range(int(n_frames))]
    except Exception:
        return None
    if not contexts:
        return None
    paths = {c["path"] for c in contexts}
    if len(paths) != 1:
        return None
    path = contexts[0]["path"]
    ranges = [c["frame_ranges"][0] for c in contexts]
    datasets = h5_metadata.snapshot(path, frame_ranges=ranges,
                                    n_aligned=contexts[0]["n_aligned"])
    if not datasets:
        return None
    return {"path": path, "datasets": datasets}


def _omegas_for_attempt(results: dict, inputs: dict, n_frames: int):
    """``(omegas, source_label)`` for a logged attempt's frames.

    Three cases, in descending order of trust:

    1. The attempt recorded ``results/omegas`` — the angles the run actually
       stamped into its own output. Used verbatim, which is the only way a
       *measured* omega channel can survive into this path at all: the
       channel is read off the raw frames, and nothing here reopens them.
    2. No stored angles, but the attempt recorded an ``omega_cfg``: rebuild
       the computed OME_START/OME_STEP ramp. Frame ``i`` here is an OUTPUT
       frame, so this is only exact when no sub-frame combining was in play
       — hence the label, so a reader can tell.
    3. Neither (an attempt logged before omega was recorded at all): zeros.
       NOT ``range(n_frames)``, which is what this function replaced — that
       wrote frame indices into an axis the backend labels "Degrees", and
       reproducing it here would keep the wrong number alive in files
       written from now on. An unrecorded angle is better reported as 0 and
       named as unavailable in the provenance entry.
    """
    stored = results.get("omegas")
    if stored is not None and len(stored) == n_frames:
        return [float(v) for v in stored], "recorded"

    cfg = (inputs or {}).get("omega_cfg") or {}
    start = float(cfg.get("start") or 0.0)
    step = float(cfg.get("step") or 0.0)
    if cfg:
        from midas_gui.cake_params import omega_series
        windows = [(i, i) for i in range(n_frames)]
        return (omega_series(start, step, windows,
                             collapse=bool(cfg.get("collapse"))),
                "recomputed from omega_cfg")

    return [0.0] * n_frames, "unavailable"


def export_gsas_zarr(project_path, panel_key: str, attempt_ref: str, out_path) -> Path:
    """Write a MIDAS-native GSAS-II ``.zarr.zip`` (+ provenance sidecar) from
    one integration attempt. Returns the zarr path actually written.

    Raises ``ValueError`` for conditions this v1 doesn't support (no results,
    no calibration snapshot, Q-uniform binning, a file-backed mask that isn't
    embedded in the project) rather than silently producing a wrong export.
    """
    meta = project.read_attempt(project_path, attempt_ref)
    results = project.read_attempt_results(project_path, attempt_ref)
    if not results or "profiles" not in results or "r_axis_px" not in results:
        raise ValueError(f"Attempt {attempt_ref} has no integration results to export.")

    calib_snapshot = meta.get("calibration_snapshot")
    if not calib_snapshot:
        raise ValueError(
            f"Attempt {attempt_ref} has no calibration snapshot recorded — "
            "cannot rebuild the geometry needed for a GSAS-II export.")

    inputs = meta.get("inputs") or {}
    if inputs.get("q_cfg"):
        from midas_gui.workers import rebin_cfg_parts
        unit = rebin_cfg_parts(inputs["q_cfg"])[0]
        label = {"Q": "Q", "2th": "2θ"}[unit]
        raise ValueError(
            f"Attempt {attempt_ref} was run with {label}-uniform binning — "
            f"its stored r_axis_px is {label}-rebinned, not the backend's "
            "own grid, so the per-bin areas GSAS-II needs can't be "
            "reconstructed from it. Re-run this attempt with Bin type set "
            "to Radial to make it exportable.")

    if meta.get("mask_present") and not meta.get("mask_embedded"):
        raise ValueError(
            f"Attempt {attempt_ref}'s mask was file-backed, not embedded "
            "in the project — GSAS-II export currently only supports "
            "attempts whose mask was embedded (drawn live / not loaded "
            "from a file). Re-run Batch Integrate with an embedded mask "
            "to make this attempt exportable.")

    profiles = np.asarray(results["profiles"])
    r_axis_px = np.asarray(results["r_axis_px"])
    frame_ids = results.get("frame_ids") or []
    multi_azimuth = profiles.ndim == 3

    result_ns = project.calibration_namespace(calib_snapshot)
    r_bin = float(inputs.get("r_bin", 1.0))
    if multi_azimuth:
        e_bin = float(inputs.get("e_bin", 5.0))
    else:
        # The original run's η bin (default 5° × 72 internal bins) was only ever
        # used for internal collapse-weighting, not output shape — force a
        # single azimuth here to match the single profile stored per frame.
        e_bin = 360.0

    spec = _build_spec(result_ns, r_bin, e_bin)
    spec.validate()
    if int(spec.n_r_bins) != r_axis_px.size:
        raise ValueError(
            "Rebuilt geometry's radial binning doesn't match this attempt's "
            f"stored r_axis_px ({spec.n_r_bins} vs {r_axis_px.size} bins) — "
            "its recorded R bin size may be stale or from an older schema.")

    mask = None
    if meta.get("mask_present"):
        mask = _read_embedded_mask(project_path, attempt_ref)
        im_trans = tuple(getattr(spec, "TransOpt", None) or ())
        if mask is not None and im_trans:
            mask = _apply_im_trans(mask.astype(np.float32), im_trans)

    # Bin area is a property of geometry + mask alone (polarization/solid-angle
    # corrections reweight intensity within a bin, not which pixels fall in
    # it), so it's reconstructed the same way regardless of what corrections
    # the original run used.
    kernel = inputs.get("kernel", "subpixel2")
    ctx = build_integration_context(spec, kernel, mask, (None, None), weighted=True)
    bin_area = ctx["cnt"]
    if bin_area is None:
        raise ValueError(
            f"Could not derive a bin-area array for attempt {attempt_ref}'s geometry.")

    from midas_integrate_v2.io.zarr_gsas import write_gsas_zarr_zip

    out_path = Path(out_path)
    if not str(out_path).endswith(".zarr.zip"):
        out_path = Path(str(out_path) + ".zarr.zip")

    n_frames = profiles.shape[0]
    cakes = (profiles[i] if multi_azimuth else profiles[i][None, :]
             for i in range(n_frames))
    omegas, omega_source = _omegas_for_attempt(results, inputs, n_frames)
    write_gsas_zarr_zip(out_path, cakes, spec=spec,
                        omegas=omegas, bin_area=bin_area)

    # provenance_history inside the zip, in the same shape Batch Integrate
    # writes it. Both paths run the same backend writer, so the arrays and
    # groups already matched; until this, the provenance did not — Batch
    # Integrate stamped a build_entry() into the root attrs while this path
    # wrote only the sidecar below, so which path produced a file changed
    # where (and whether) you could read its geometry back.
    try:
        entry = prov.build_entry(
            'midas_gui.gsas_export',
            inputs=[str(project_path)],
            cake_params={
                'RMin': float(spec.RMin), 'RMax': float(spec.RMax),
                'RBinSize': float(spec.RBinSize), 'EtaMin': float(spec.EtaMin),
                'EtaMax': float(spec.EtaMax), 'EtaBinSize': float(spec.EtaBinSize),
            },
            instrument_params=prov.instrument_params_from_spec(spec),
            extra={
                'kernel': kernel, 'weighted': True,
                'multi_azimuth': bool(multi_azimuth),
                'n_frames': int(n_frames),
                'source_project': str(Path(project_path).resolve()),
                'panel_key': panel_key, 'attempt_ref': attempt_ref,
                'omega_source': omega_source,
            },
        )
        # Batch Integrate copies the source HDF5's instrument/ PV snapshot into
        # every store it writes; this path copies the same tree from the same
        # file, so a reader can't tell which writer produced a given archive.
        # The attempt already records where the frames came from (its
        # inputs.src_cfg), which is the only reason this is reconstructible
        # after the fact at all.
        snap = _source_metadata_snapshot(meta, n_frames)
        if snap:
            entry.setdefault('extra', {})['source_h5'] = snap['path']

        def _mutate(extracted, _snap=snap, _e=entry):
            if _snap:
                h5_metadata.write_into_extracted(extracted, _snap['datasets'])
            prov.stamp_extracted(extracted, _e)

        prov.rewrite_zip(out_path, _mutate)
    except Exception:
        # Best-effort, exactly as in Batch Integrate: a failed stamp must not
        # cost the user an export that otherwise succeeded. The sidecar below
        # is written either way.
        pass

    # The sidecar stays: it carries the attempt's own metadata (stored params,
    # calibration snapshot, frame ids) that has no equivalent on the Batch
    # Integrate path. It is an addition to the in-zip entry, not the place the
    # geometry lives any more — and it is ours, not GSAS-II's: the sidecars
    # G2pwd_MIDAS.py reads are <stem>.zarr.samprm/.instprm, which we do not
    # write. See the module docstring.
    provenance = dict(meta)
    provenance["source_project"] = str(Path(project_path).resolve())
    provenance["panel_key"] = panel_key
    provenance["attempt_ref"] = attempt_ref
    provenance["export_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["frame_ids"] = list(frame_ids)
    prov_path = Path(str(out_path) + ".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2, default=str))

    return out_path
