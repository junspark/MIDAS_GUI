"""Carry the source HDF5's instrument metadata forward into the output zarr.

A VAREX/areaDetector HDF5 arrives with a few hundred EPICS PVs snapshotted
under ``instrument/`` at acquisition time — motor positions, slit gaps,
monochromator angles, insertion-device gap, scaler channels, the lot. MIDAS's
own pipeline keeps them: ``ffGenerateZipRefactor`` copies the tree into an
intermediate "input zarr" and ``integrator.py:_enrich_zarr_with_metadata``
re-copies it into the final output. MIDAS_GUI has no intermediate — it reads
frames and six named scalars straight into memory — so everything else used to
stop at the HDF5 and the ``.zarr.zip`` came out with only what the backend
writer has slots for.

This module closes that gap by the one route available after the fact: reopen
the source file and copy the tree into the finished archive. Two consequences
worth knowing:

* **HDF5 sources only.** A TIFF stack has no such tree, so nothing is copied
  and nothing is missing — there was never anything to carry.
* **Nothing here is read by GSAS-II.** Its importer (``G2pwd_MIDAS.py``) looks
  at exactly ``InstrumentParameters``/``REtaMap``/``OmegaSumFrame`` and ignores
  every other group, so this is pure provenance: it cannot perturb an import,
  and it cannot substitute for one either. See ``.context/DECISIONS.md``,
  2026-09-28, for the full contract.

The per-frame averaging rule is MIDAS's, kept deliberately: a 1-D array with
one entry per raw frame is *averaged* over each output frame's chunk of raw
sub-frames rather than copied at raw length, so its length matches the image
stack it describes. What is NOT MIDAS's is how the raw frame count is found.
``_enrich_zarr_with_metadata`` tests ``len(arr) == total_frames``; on real
20-ID files that matches nothing at all, because the DAQ writes one metadata
sample per *acquisition* — lights and darks together in one flat array — so a
10-frame scan carries length-20 metadata. The caller therefore passes
``n_aligned``, the light-block length that
``workers._HDF5StackGlobSource._metadata_frame_count`` recovers from the
timestamp gap, and arrays are matched against that instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Copied recursively, in this order. ``instrument`` is the EPICS PV snapshot;
# ``active_instrument`` is meant to name the station that produced the file
# (empty in every real file checked so far — a known upstream gap — but copied
# anyway so it starts carrying meaning for free once the DAQ fills it in).
# Deliberately NOT copied: ``exchange``/``NDArray`` (the image data itself,
# already in the zarr as cakes) and ``misc`` (empty in practice, and its
# ``NDArrayTimeStamp`` is consumed rather than forwarded).
COPY_GROUPS = ("instrument", "active_instrument")

_MAX_TOTAL_BYTES = 8 * 1024 * 1024  # a guard, not a budget: see snapshot()


def _decode(value):
    """h5py hands back ``bytes`` for its string types; zarr wants ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _to_plain(arr: np.ndarray) -> np.ndarray:
    """A numpy array zarr 2 can store: object/bytes string arrays flattened to
    a fixed-width unicode dtype, everything else left alone."""
    if arr.dtype.kind in ("O", "S"):
        return np.array([_decode(v) for v in arr.ravel()],
                        dtype=object).astype(str).reshape(arr.shape)
    return arr


def _chunk_mean(arr: np.ndarray, ranges: Sequence[tuple[int, int]]) -> np.ndarray:
    """One value per output frame: the mean of the raw entries it combines.

    Strings can't be averaged, so they take the first entry of the range —
    the same "tile, don't average" fallback ``_coerce_metadata_array`` uses
    upstream. A range that falls off the end of the array is clipped rather
    than dropped, so a short array still yields one entry per output frame.
    """
    is_numeric = arr.dtype.kind in "fiub"
    out = []
    for start, end in ranges:
        hi = min(int(end), arr.size - 1)
        lo = min(max(int(start), 0), hi)
        window = arr[lo:hi + 1]
        if window.size == 0:
            out.append(arr[0] if arr.size else 0.0)
        elif is_numeric:
            out.append(float(np.mean(window.astype(np.float64))))
        else:
            out.append(window[0])
    return np.asarray(out)


def read_tree(h5_path, *, groups: Iterable[str] = COPY_GROUPS) -> dict:
    """``{zarr-relative path: numpy array}`` for every dataset under ``groups``,
    exactly as stored — no per-frame alignment.

    Split from :func:`snapshot` because the raw tree is the same for every
    output frame of a given source file while the alignment is not: a batch
    run reads this once per file and calls :func:`align` per frame, rather
    than reopening a few hundred datasets for each frame it writes.

    Never raises for a source that simply doesn't have this metadata: an
    unreadable file, a missing group or an individually unreadable dataset all
    come back as "that much less in the dict". The ``_MAX_TOTAL_BYTES`` guard
    is there so a file that puts something enormous under ``instrument/``
    (nothing seen does — a real tree is ~24 KiB) can't silently balloon every
    output frame's archive; hitting it stops the walk rather than truncating an
    array, so what is written is always complete as far as it goes.
    """
    out: dict = {}
    budget = [0]          # a list, so the nested visitor can add to it
    try:
        import h5py
    except Exception:
        return out

    def visit(prefix, name, obj):
        if not isinstance(obj, h5py.Dataset):
            return None
        if budget[0] >= _MAX_TOTAL_BYTES:
            return True          # truthy → h5py stops the visit
        try:
            arr = _to_plain(np.atleast_1d(np.asarray(obj[()])))
        except Exception:
            return None
        if arr.size == 0:
            return None
        budget[0] += int(arr.nbytes)
        out[f"{prefix}/{name}"] = arr
        return None

    try:
        with h5py.File(str(h5_path), "r") as f:
            for group_name in groups:
                node = f.get(group_name)
                if node is None or not hasattr(node, "visititems"):
                    continue
                node.visititems(lambda n, o, _p=group_name: visit(_p, n, o))
    except Exception:
        return out
    return out


def align(tree: dict, frame_ranges: Sequence[tuple[int, int]] | None,
          n_aligned: int | None) -> dict:
    """Reduce a :func:`read_tree` result to one entry per output frame.

    ``frame_ranges`` is one ``(first_raw, last_raw)`` inclusive pair per output
    frame in the zarr being written; ``n_aligned`` is how many leading entries
    of a per-acquisition array correspond to real light frames. Pass either as
    ``None`` and the tree comes back untouched — honest, just not aligned to
    the image stack.
    """
    if not (frame_ranges and n_aligned):
        return dict(tree)
    out = {}
    for path, arr in tree.items():
        if arr.ndim == 1 and arr.size >= n_aligned:
            arr = _chunk_mean(arr[:n_aligned], frame_ranges)
        out[path] = arr
    return out


def snapshot(h5_path, *, frame_ranges: Sequence[tuple[int, int]] | None = None,
             n_aligned: int | None = None,
             groups: Iterable[str] = COPY_GROUPS) -> dict:
    """:func:`read_tree` then :func:`align` — the one-shot form, for a caller
    writing a single store. A caller writing many stores from one source file
    should read the tree once and align it per frame instead."""
    return align(read_tree(h5_path, groups=groups), frame_ranges, n_aligned)


def write_into_extracted(extracted_dir, snap: dict) -> int:
    """Materialise ``snap`` as zarr arrays inside an already-extracted store.

    Takes the extracted directory rather than the ``.zarr.zip`` so this can
    share one extract → edit → repack pass with the provenance stamp (see
    ``provenance.rewrite_zip``) instead of repacking the archive twice.
    Returns how many datasets were written; a dataset that zarr refuses (an
    exotic dtype, say) is skipped rather than failing the whole copy.
    """
    if not snap:
        return 0
    import zarr
    root = zarr.open_group(zarr.DirectoryStore(str(extracted_dir)), mode="a")
    written = 0
    for path, arr in snap.items():
        try:
            root.create_dataset(path, data=arr, overwrite=True)
            written += 1
        except Exception:
            continue
    return written


def write_into_hdf5(h5_file, snap: dict) -> int:
    """Materialise ``snap`` as datasets inside an already-open HDF5 file.

    The HDF5 counterpart of :func:`write_into_extracted`, for Batch
    Correction's reduced-frame output (``frame_correct.write_corrected_h5``),
    which writes a plain ``.h5`` rather than a zarr store. Same contract:
    paths are written exactly as :func:`read_tree` recorded them, so the
    output's ``instrument/`` tree mirrors the source file's, and a dataset
    HDF5 refuses (an exotic dtype, a name colliding with one already
    written) is skipped rather than failing the whole copy.

    Variable-length unicode is written through ``h5py.string_dtype()``;
    numpy's fixed-width ``U`` dtype has no HDF5 equivalent and would
    otherwise be one of those silent skips.

    Returns how many datasets were written.
    """
    if not snap:
        return 0
    import h5py
    written = 0
    for path, arr in snap.items():
        try:
            arr = np.asarray(arr)
            if arr.dtype.kind == "U":
                h5_file.create_dataset(path, data=arr.astype(object),
                                       dtype=h5py.string_dtype())
            else:
                h5_file.create_dataset(path, data=arr)
            written += 1
        except Exception:
            continue
    return written


def copy_into_zip(zarr_zip_path, h5_path, **kwargs) -> int:
    """``snapshot`` + ``write_into_extracted`` against a closed ``.zarr.zip``.

    The standalone form, for a caller with nothing else to change in the
    archive. A caller that is also stamping provenance should build the
    snapshot itself and do both inside one ``provenance.rewrite_zip``.
    """
    from . import provenance
    snap = snapshot(h5_path, **kwargs)
    if not snap:
        return 0
    count = [0]

    def mutate(extracted: Path) -> None:
        count[0] = write_into_extracted(extracted, snap)

    provenance.rewrite_zip(zarr_zip_path, mutate)
    return count[0]
