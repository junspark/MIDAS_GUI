"""Chunked frame reduction with field correction — the engine behind the
Batch Correction tab (``tab_batch_correct.py``).

Reduces a multi-frame HDF5 stack to one output frame per group of
``chunk_size`` consecutive raw sub-frames (mean / median / sum / max),
correcting dark / bright / background along the way, and writes the result
back out as HDF5.

Nothing here touches Qt widgets or the GUI thread — it is the numeric half,
so it can be tested directly. (It does import ``helpers``, which pulls PyQt5
in transitively; that is true of every module in this package.)

**Order of operations.** Every raw sub-frame is corrected BEFORE it is
combined, and the clip-at-zero is deferred to the very end:

    corrected_i = apply_field_corrections(raw_i, …, clip_negative=False)
    combined    = op(corrected_0 … corrected_{n-1})
    out         = clip(combined, 0, None)        # once, if requested

This matters, and only one of the four ops makes it obvious:

* ``sum`` is only correct this way. Correcting the COMBINED frame would
  subtract a single dark from an n-times-larger signal — the exact bug
  ``workers.StreamPreviewWorker`` documents avoiding for its preview sum.
  Done per sub-frame, an n-frame sum correctly loses ``n × dark``.
* ``mean``/``max``/``median`` come out identical either way, so one code
  path is right for all four: dark subtraction and flat-field divide are
  monotone per-pixel maps, and ``max``/``median`` commute with any monotone
  per-pixel map.
* The clip is deferred because it is NOT linear. Clipping each sub-frame at
  zero before summing throws away the negative half of the read noise and
  biases the sum upward; clipping once at the end is both unbiased for
  ``sum``/``mean`` and (clip being monotone) identical for ``max``/``median``.

So the single rule — correct each sub-frame, combine, clip once — is the
correct one for every op, which is why there is no per-op special-casing
below.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import numpy as np

from midas_gui.helpers import (_COMBINE_OPS, _stack_chunk_bounds, average_field,
                               apply_field_corrections, is_dark_like_name, is_h5)

#: Output ops offered by the Batch Correction tab, in UI order. A subset of
#: ``helpers._COMBINE_OPS`` keys (that table also carries the "average" alias
#: kept for older callers), so the two can never drift apart silently —
#: ``reduce_chunk`` validates against ``_COMBINE_OPS`` itself.
OPS = ("mean", "median", "sum", "max")

#: ``<froot>_<NNNNNN><suffix>`` — the beamline's own file-naming convention
#: (``C611_017Fe_1_load0_008802.vrx.h5``). The same split
#: ``workers.froot_and_frame_num`` does for frame IDs, but over a FILE NAME:
#: there is no chunk suffix to strip here, and the dotted tail
#: (``.vrx.h5``) is kept whole rather than reduced by ``Path.stem``, because
#: :func:`resolve_dark` reconstructs sibling filenames from it.
_NAME_NUM_RE = re.compile(r"^(?P<froot>.+)_(?P<num>\d+)(?P<suffix>\..*)$")


class NameParts(NamedTuple):
    """``C611_017Fe_1_load0_008802.vrx.h5`` split for sibling-name rebuilding."""
    froot: str
    num: Optional[int]
    width: int
    suffix: str


def split_scan_name(name) -> NameParts:
    """Split a detector filename into ``(froot, number, zero-pad width, suffix)``.

    ``number`` is ``None`` for a name that carries no ``_<digits>`` run, in
    which case ``froot`` is everything before the first dot and the caller
    simply has no number to count from.
    """
    base = Path(str(name)).name
    m = _NAME_NUM_RE.match(base)
    if not m:
        head, dot, tail = base.partition(".")
        return NameParts(head, None, 0, (dot + tail) if dot else "")
    digits = m.group("num")
    return NameParts(m.group("froot"), int(digits), len(digits), m.group("suffix"))


# ═════════════════════════════════════════════════════════════════════════════
#  Chunking
# ═════════════════════════════════════════════════════════════════════════════

def chunk_ranges(n_raw: int, *, chunk_size: Optional[int] = None,
                 raw_start: Optional[int] = None,
                 raw_end: Optional[int] = None) -> list:
    """Every chunk's inclusive 0-based ``(lo, hi)`` raw sub-frame range.

    A deliberately thin loop over ``helpers._stack_chunk_bounds`` rather than
    its own arithmetic: that function is documented as the single source of
    truth for where chunk *k* starts, and sharing it is what keeps Batch
    Correction's chunk boundaries identical to the ones Batch Integrate
    integrates over. A short final chunk is kept, not dropped.
    """
    out = []
    k = 0
    while True:
        bounds = _stack_chunk_bounds(int(n_raw), k, chunk_size=chunk_size,
                                     raw_start=raw_start, raw_end=raw_end)
        if bounds is None:
            return out
        out.append(bounds)
        k += 1


# ═════════════════════════════════════════════════════════════════════════════
#  Reduction
# ═════════════════════════════════════════════════════════════════════════════

def reduce_chunk(frames, op: str = "mean", *, dark=None, bright=None,
                 bright_mode: str = "divide", background=None,
                 clip_negatives: bool = True) -> np.ndarray:
    """Correct every frame, combine them with ``op``, clip once. Returns float32.

    See the module docstring for why that order, and why it is the same order
    for all four ops.

    ``mean``/``sum``/``max`` accumulate one frame at a time rather than
    materialising the corrected stack — a 10-frame chunk of 2880² float64 is
    660 MB, which a long run would pay for on every chunk. ``median`` has no
    streaming form and does materialise, in float32.
    """
    name = "mean" if str(op).lower() == "average" else str(op).lower()
    if name not in _COMBINE_OPS:
        raise ValueError(f"Unknown combine op {op!r}; expected one of "
                         f"{', '.join(sorted(_COMBINE_OPS))}.")
    has_fields = dark is not None or bright is not None or background is not None

    def corrected(frame) -> np.ndarray:
        arr = np.asarray(frame)
        if not has_fields:
            return arr.astype(np.float32, copy=False)
        # clip_negative=False — the clip happens once, below, on the combined
        # frame. See the module docstring.
        return apply_field_corrections(
            arr, dark=dark, bright=bright, bright_mode=bright_mode,
            background=background, clip_negative=False).astype(np.float32)

    if name == "median":
        # Materialise the list before stacking: np.stack([]) raises its own
        # "need at least one array" before any emptiness check downstream
        # could fire, and an empty chunk should report what is actually
        # wrong with it.
        planes = [corrected(f) for f in frames]
        if not planes:
            raise ValueError("reduce_chunk: no frames to combine.")
        out = np.median(np.stack(planes, axis=0), axis=0)
    else:
        acc = None
        n = 0
        for frame in frames:
            cur = corrected(frame)
            n += 1
            if acc is None:
                # float64 for the additive ops so a long sum doesn't lose
                # low-order bits; max stays in the frame's own float32.
                acc = cur.astype(np.float64) if name in ("sum", "mean") else cur.copy()
            elif name == "max":
                np.maximum(acc, cur, out=acc)
            else:
                acc += cur
        if acc is None:
            raise ValueError("reduce_chunk: no frames to combine.")
        out = (acc / n) if name == "mean" else acc

    out = np.asarray(out, dtype=np.float32)
    return np.clip(out, 0.0, None) if clip_negatives else out


# ═════════════════════════════════════════════════════════════════════════════
#  Dark resolution
# ═════════════════════════════════════════════════════════════════════════════
#
# Derived from two independent surveys of real 20-ID data, both replayed as
# tests in tests/test_dark_autodetect.py:
#
# (a) The beamline's own 198 caking screenlogs in
#     ~/midas_runs/midas_screen_logs. 192 of them resolved a dark, and in
#     192/192 the pixels came from an ``exchange/data_dark`` dataset — there
#     is no separate dark image format to handle, only the question of WHICH
#     FILE holds it. Those logs show two patterns at roughly 50/50: a
#     ``<froot>_dark_before_<first data number - 1>`` sibling, or the data
#     file's own ``data_dark``.
#
# (b) The 160 VAREX scan folders still on disk under
#     /home/beams/S20IDUSER/mnt/s20a/*/varex*/. These correct a wrong
#     assumption the logs alone would have left in place. A scan folder is
#     NOT one acquisition bracketed by one dark pair: darks are RE-MEASURED
#     THROUGHOUT, so a folder holds a series of segments, each of the form
#
#         <froot>_dark_before_<n> … data … <froot>_dark_after_<m>
#         <froot>_dark_before_<m+1> … data … <froot>_dark_after_<m'>  …
#
#     Fe9Cr_KGT6038_load1_waxs alone carries 11 ``dark_before`` and 8
#     ``dark_after`` files among its data. Of the 936 data files sitting in
#     the 56 dark-bearing folders, 936 — every one — have a preceding
#     ``dark_before``, at a median distance of 8 files and a maximum of 121.
#     The other 102 folders carry no dark sibling at all.
#
# Hence the rule below is PER DATA FILE, not per folder: the nearest
# PRECEDING ``dark_before``. A folder-wide "dark number = first data number
# - 1" rule (which is what the logs on their own suggest, and what the
# legacy script did) is right only for a scan's first segment and would hand
# a stale dark to the ~900 files after it. The per-file rule reproduces the
# logged choice exactly whenever the scan has a single segment, which is why
# it is compatible with (a) rather than a departure from it.
#
# ``dark_after`` never appears in the log corpus but is common on disk, so
# it is handled as the fallback direction; ``helpers.is_dark_like_name``
# already matches both spellings.

#: What ``resolve_dark`` returns when it found nothing — a real value, not an
#: error: bright/background-only correction is legitimate, it just has to be
#: said out loud in the log.
NO_DARK = "no dark found — correcting without one"


def _mean_dark(path, dataset: str, shape) -> Optional[np.ndarray]:
    """Mean of ``dataset`` in ``path``, or None if unreadable/absent/wrong shape.

    A shape mismatch drops to the next rung rather than raising — the same
    skip-and-warn contract ``apply_field_corrections`` applies to a field
    left over from a different detector.
    """
    try:
        arr = np.asarray(average_field("hdf5", str(path), dataset), dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    if shape is not None and tuple(arr.shape) != tuple(shape):
        return None
    return arr


def _has_dataset(path, dataset: str) -> bool:
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            dset = f.get(dataset)
            return dset is not None and getattr(dset, "size", 0) > 0
    except Exception:
        return False


def dark_candidates(data_path, *, number: Optional[int] = None) -> list:
    """Dark-like HDF5 siblings of ``data_path``, best first, for THIS file.

    Ranked on ``(shares this file's froot, direction, distance, name)``:

    * a sibling whose root extends this file's own wins over an unrelated
      one (the bare folder-level ``dark_before`` case);
    * the nearest ``dark_before`` BEFORE this file's number wins — that is
      the dark most recently measured for this exposure (see the survey
      above: darks are re-measured mid-scan, so "nearest" and "the scan's
      first dark" are usually different files);
    * then the nearest ``dark_after`` AFTER it, for a file that precedes
      every dark in its folder;
    * then any other dark-like sibling by absolute distance.

    Ties break on name, so the answer is stable across filesystems with
    different directory ordering. Returns ``[(path, why)]``; an unreadable
    folder yields ``[]``.
    """
    data_path = Path(str(data_path))
    parts = split_scan_name(data_path.name)
    num = number if number is not None else parts.num
    try:
        siblings = sorted(p for p in data_path.parent.iterdir()
                          if p.is_file() and is_h5(p) and is_dark_like_name(p))
    except Exception:
        return []

    ranked = []
    for p in siblings:
        if p == data_path:
            continue
        cand = split_scan_name(p.name)
        prefixed = bool(parts.froot) and cand.froot.startswith(parts.froot)
        low = p.name.lower()
        if cand.num is None or num is None:
            direction, dist, why = 2, 10 ** 9, "dark"
        elif "dark_before" in low and cand.num < num:
            direction, dist, why = 0, num - cand.num, "preceding dark_before"
        elif "dark_after" in low and cand.num > num:
            direction, dist, why = 1, cand.num - num, "following dark_after"
        else:
            direction, dist, why = 2, abs(cand.num - num), "nearest dark"
        ranked.append(((0 if prefixed else 1, direction, dist, p.name), p,
                       f"{why} {p.name}"))
    ranked.sort(key=lambda row: row[0])
    return [(p, why) for _key, p, why in ranked]


def resolve_dark(data_path, *, dark_dataset: str = "exchange/data_dark",
                 number: Optional[int] = None, fallback=None, shape=None,
                 auto: bool = True) -> tuple:
    """Pick the dark for ONE data file. Returns ``(array_or_None, why)``.

    The ladder, in order — see the block comment above for the data it was
    derived from. Every rung that yields an unreadable file or an array of
    the wrong shape falls through to the next rather than raising:

    1. the nearest dark-like sibling, ranked by :func:`dark_candidates`
       (in practice: the most recently measured ``dark_before``);
    2. the data file's OWN ``dark_dataset`` — the self-dark convention the
       102 dark-less folders on disk use;
    3. ``fallback`` — whatever the user chose in the loader's Dark field;
    4. nothing, reported as :data:`NO_DARK`.

    ``auto=False`` skips rungs 1-2 and goes straight to the fallback, for a
    user who wants one dark applied across the whole run.

    ``why`` always names the file the dark came from, so a run's log can be
    audited after the fact rather than taken on trust.
    """
    data_path = Path(str(data_path))
    if auto:
        for path, why in dark_candidates(data_path, number=number):
            arr = _mean_dark(path, dark_dataset, shape)
            if arr is not None:
                return arr, f"{why} [{dark_dataset}]"
        if _has_dataset(data_path, dark_dataset):
            arr = _mean_dark(data_path, dark_dataset, shape)
            if arr is not None:
                return arr, f"own {dark_dataset} in {data_path.name}"
    if fallback is not None:
        arr = np.asarray(fallback, dtype=np.float32)
        if arr.ndim == 2 and (shape is None or tuple(arr.shape) == tuple(shape)):
            return arr, "Dark field from the loader"
        return None, (f"loader Dark field is {arr.shape}, not {tuple(shape)} — "
                      f"skipped; correcting without a dark")
    return None, NO_DARK


# ═════════════════════════════════════════════════════════════════════════════
#  Output
# ═════════════════════════════════════════════════════════════════════════════

#: ``compression`` values the tab offers, mapped to the h5py kwargs they mean.
#: ``None`` is stored uncompressed — fastest to write and to read back, and
#: the right default for a scratch reduction that something else consumes
#: immediately.
COMPRESSIONS = ("none", "gzip", "lzf")


def compression_kwargs(name: Optional[str], level: int = 4,
                       shuffle: bool = False) -> dict:
    """h5py ``create_dataset`` kwargs for a :data:`COMPRESSIONS` choice.

    ``shuffle`` is offered separately because it is what makes gzip worth
    using on detector data: byte-transposing a float32 plane groups the
    (nearly constant) exponent bytes together, and typically buys more than
    raising the gzip level does.
    """
    key = (name or "none").lower()
    if key in ("", "none"):
        return {}
    if key == "lzf":
        return {"compression": "lzf", "shuffle": bool(shuffle)}
    if key == "gzip":
        return {"compression": "gzip",
                "compression_opts": int(np.clip(int(level), 0, 9)),
                "shuffle": bool(shuffle)}
    raise ValueError(f"Unknown compression {name!r}; expected one of "
                     f"{', '.join(COMPRESSIONS)}.")


def write_corrected_h5(path, frames, *, dataset: str = "exchange/data",
                       metadata: Optional[dict] = None,
                       frame_ranges: Optional[Sequence] = None,
                       attrs: Optional[dict] = None,
                       provenance_entry: Optional[dict] = None,
                       compression: Optional[str] = None, level: int = 4,
                       shuffle: bool = False) -> str:
    """Write reduced ``frames`` as one ``(M, H, W)`` float32 HDF5 dataset.

    Also writes, when given: the per-chunk-averaged instrument metadata tree
    (``metadata``, from ``h5_metadata.align``) at its original paths; the
    ``(M, 2)`` inclusive raw sub-frame range each output frame was built from
    (``frame_ranges``, as ``frame_ranges`` beside the data — the audit trail
    for "which exposures is this frame?"); plain root attributes; and a
    ``provenance`` stamp via ``provenance.append_to_hdf5_attrs``.

    Chunked along the frame axis (one 2-D plane per HDF5 chunk) whenever
    compression is on, since a compressed dataset must be chunked and a
    whole-stack chunk would force a full decompress to read one frame.
    """
    import h5py
    from midas_gui import h5_metadata, provenance as _prov

    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    out = Path(str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = compression_kwargs(compression, level, shuffle)
    if kwargs:
        kwargs["chunks"] = (1,) + arr.shape[1:]
    with h5py.File(str(out), "w") as f:
        f.create_dataset(dataset, data=arr, **kwargs)
        if frame_ranges is not None:
            f.create_dataset("frame_ranges",
                             data=np.asarray(frame_ranges, dtype=np.int64))
        if metadata:
            h5_metadata.write_into_hdf5(f, metadata)
        for key, value in (attrs or {}).items():
            try:
                f.attrs[key] = value
            except Exception:
                f.attrs[key] = str(value)
        if provenance_entry:
            try:
                _prov.append_to_hdf5_attrs(f, provenance_entry)
            except Exception:
                pass
    return str(out)
