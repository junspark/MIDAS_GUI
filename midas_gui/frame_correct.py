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

def reduce_chunk_multi(frames, ops, *, dark=None, bright=None,
                       bright_mode: str = "divide", background=None,
                       clip_negatives: bool = True,
                       skip_blank: bool = False, stats: Optional[dict] = None) -> dict:
    """``{op: frame}`` for several ops over ONE pass of ``frames``.

    Correcting a sub-frame is the same work whichever op consumes it, and
    reading it off disk is far more expensive than either, so selecting
    mean + median + sum + max costs one pass rather than four. That is the
    whole reason this exists; :func:`reduce_chunk` is a thin wrapper for
    the single-op case so there is only one implementation of the ordering
    rule described in the module docstring.

    Only ``median`` needs the corrected stack kept — the others accumulate
    one frame at a time — so the memory cost is a single plane per op
    unless median is asked for.

    ``skip_blank`` drops raw sub-frames that are entirely zero before they
    reach the combine. A Pixirad arms one frame before it starts counting,
    so frame 1 of every ``.pixi.h5`` is all zeros -- and averaging it in
    scales every pixel by ``(n-1)/n``. Measured on 1-ID-E
    ``air_80p725keV_3s_003512.pixi.h5``: 10 raw frames, frame 0 blank, and
    the written output matched ``mean(frames 0..9)`` at 100% of pixels,
    i.e. exactly 10/9 low. Silent, and it scales with how few frames you
    combine -- 10% over ten frames, 33% over three.

    The test is the RAW frame, before correction: a corrected frame can be
    legitimately all-zero (a flat field that cancels, or a clip), and that
    is real data, not a missing exposure. A frame that was zero as it came
    off the detector never recorded anything, so it is not an observation
    of zero for any of the four ops -- it is the absence of an observation.

    ``sum`` and ``max`` are unaffected either way (adding or maximising
    against zero changes nothing), so this only moves ``mean`` and
    ``median``.

    **Off by default, and that is a retraction.** It shipped on by default
    and was wrong to: an all-zero raw frame is NOT reliably an arming frame.
    It broke three ``test_batch_correction_tab`` cases whose synthetic stack
    ramps ``base + i`` from 0, so sub-frame 0 is legitimately all-zero data
    and dropping it moved a mean from 1.5 to 2.0. Nothing in the pixel values
    distinguishes "the detector had not started counting" from "this exposure
    really was zero" -- position does not either, since the Pixirad's dud and
    that ramp's first frame are both frame 0. The caller has to say, from
    something outside the array (the detector, or the user). A chunk that is ENTIRELY blank keeps
    its frames: there is no good answer there, and raising "no frames to
    combine" for a run that legitimately contains a dead chunk would be
    worse than returning the zeros the detector actually produced.

    ``stats``, when given, is filled with ``{"blank_skipped": n}`` so a
    caller can report the drop instead of it being invisible -- the whole
    point being that the old behaviour was wrong *quietly*.
    """
    names = []
    for op in ops:
        name = "mean" if str(op).lower() == "average" else str(op).lower()
        if name not in _COMBINE_OPS:
            raise ValueError(f"Unknown combine op {op!r}; expected one of "
                             f"{', '.join(sorted(_COMBINE_OPS))}.")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("reduce_chunk_multi: no ops selected.")

    has_fields = dark is not None or bright is not None or background is not None
    need_stack = "median" in names
    need_total = "sum" in names or "mean" in names

    planes = []
    total = None
    running_max = None
    n = 0
    blank = 0
    blank_like = None
    for frame in frames:
        arr = np.asarray(frame)
        if skip_blank and not arr.any():
            blank += 1
            blank_like = arr            # all-zero, so one stands for all
            continue
        if has_fields:
            # clip_negative=False — clipped once per op at the end. See the
            # module docstring for why the clip cannot move earlier.
            cur = apply_field_corrections(
                arr, dark=dark, bright=bright, bright_mode=bright_mode,
                background=background, clip_negative=False).astype(np.float32)
        else:
            cur = arr.astype(np.float32, copy=False)
        n += 1
        if need_stack:
            planes.append(cur)
        if need_total:
            total = cur.astype(np.float64) if total is None else total + cur
        if "max" in names:
            running_max = cur.copy() if running_max is None else \
                np.maximum(running_max, cur, out=running_max)
    if n == 0 and blank:
        # Every frame in this chunk was blank. Returning the detector's own
        # zeros beats failing the whole run over a dead chunk, so redo the
        # pass without the skip rather than raise. Rebuilt from a stand-in
        # rather than re-iterating `frames`: that may be a one-shot
        # iterator, and every frame here is all-zero by definition, so
        # `blank` copies of one of them is the same input exactly.
        out = reduce_chunk_multi(
            [blank_like] * blank, ops, dark=dark, bright=bright,
            bright_mode=bright_mode, background=background,
            clip_negatives=clip_negatives, skip_blank=False)
        if stats is not None:
            stats["blank_skipped"] = 0
            stats["all_blank"] = True
        return out
    if n == 0:
        raise ValueError("reduce_chunk: no frames to combine.")
    if stats is not None:
        stats["blank_skipped"] = blank
        stats["all_blank"] = False

    raw_out = {}
    if "mean" in names:
        raw_out["mean"] = total / n
    if "sum" in names:
        raw_out["sum"] = total
    if "max" in names:
        raw_out["max"] = running_max
    if "median" in names:
        raw_out["median"] = np.median(np.stack(planes, axis=0), axis=0)

    out = {}
    for name in names:
        plane = np.asarray(raw_out[name], dtype=np.float32)
        out[name] = np.clip(plane, 0.0, None) if clip_negatives else plane
    return out


def reduce_chunk(frames, op: str = "mean", *, dark=None, bright=None,
                 bright_mode: str = "divide", background=None,
                 clip_negatives: bool = True,
                 skip_blank: bool = False, stats: Optional[dict] = None) -> np.ndarray:
    """Correct every frame, combine them with ``op``, clip once. Returns float32.

    See the module docstring for why that order, and why it is the same order
    for all four ops.

    ``mean``/``sum``/``max`` accumulate one frame at a time rather than
    materialising the corrected stack — a 10-frame chunk of 2880² float64 is
    660 MB, which a long run would pay for on every chunk. ``median`` has no
    streaming form and does materialise, in float32.
    """
    name = "mean" if str(op).lower() == "average" else str(op).lower()
    return reduce_chunk_multi(  # skip_blank/stats: see reduce_chunk_multi
        frames, [name], dark=dark, bright=bright, bright_mode=bright_mode,
        background=background, clip_negatives=clip_negatives,
        skip_blank=skip_blank, stats=stats)[name]


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


#: Output dtypes offered for the corrected stack, in UI order.
#:
#: ``float32`` is the default and the honest one: GSAS-II's HDF5 image reader
#: has no dtype gate at all (it dispatches on array *shape* and hands the
#: array over uncast), and its integration casts to float anyway. ``uint32``
#: exists only for workflows that pass the data to something integer-only —
#: notably GSAS-II's *TIFF* reader, which truncates float32 to int32 on load.
OUTPUT_DTYPES = ("float32", "uint32")


def cast_for_output(arr, dtype: str = "float32"):
    """``(array, n_clipped_low, n_clipped_high)`` — the on-disk copy.

    Called on the fully corrected stack and nowhere else, so correction
    arithmetic stays in float and only the serialised copy is narrowed.

    Unsigned output cannot represent the negatives that background
    subtraction routinely produces, and a bare ``astype`` would silently
    WRAP them (-1 becomes 4294967295) rather than fail — so round first,
    then clip, then report what was lost. Rounding is ``rint``, not the
    truncation ``astype`` would do on its own: a 0.6 count belongs in bin 1.
    NaN has no unsigned representation either and would convert to a
    platform-defined value, so it is folded into the low clip rather than
    left to chance.

    A run that quietly destroyed half the detector must not be
    indistinguishable from a clean one, which is why the counts come back
    instead of being swallowed.
    """
    if dtype == "float32":
        return np.asarray(arr, dtype=np.float32), 0, 0
    if dtype != "uint32":
        raise ValueError(f"Unknown output dtype {dtype!r}; expected one of "
                         f"{', '.join(OUTPUT_DTYPES)}.")
    hi = float(np.iinfo(np.uint32).max)
    rounded = np.rint(np.asarray(arr, dtype=np.float64))
    bad = np.isnan(rounded)
    n_low = int(np.count_nonzero((rounded < 0.0) | bad))
    n_high = int(np.count_nonzero(rounded > hi))
    rounded = np.where(bad, 0.0, rounded)
    return np.clip(rounded, 0.0, hi).astype(np.uint32), n_low, n_high


def write_corrected_h5(path, frames, *, dataset: str = "exchange/data",
                       metadata: Optional[dict] = None,
                       frame_ranges: Optional[Sequence] = None,
                       attrs: Optional[dict] = None,
                       provenance_entry: Optional[dict] = None,
                       compression: Optional[str] = None, level: int = 4,
                       shuffle: bool = False, dtype: str = "float32",
                       log=None) -> str:
    """Write reduced ``frames`` as one ``(M, H, W)`` HDF5 dataset.

    ``dtype`` is one of :data:`OUTPUT_DTYPES` and is applied by
    :func:`cast_for_output` at the last possible moment — immediately before
    ``create_dataset``, after every correction has been done in float. When
    a narrowing cast clips anything, ``log`` (a one-argument callable, if
    given) is told how much.

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
    # The very last thing that happens to the pixels before they are written.
    arr, n_low, n_high = cast_for_output(arr, dtype)
    if log is not None and (n_low or n_high):
        log(f"[correct] {dtype}: clipped {n_low:,} px below 0 and "
            f"{n_high:,} px above {np.iinfo(np.uint32).max:,} "
            f"({Path(str(path)).name})")
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
