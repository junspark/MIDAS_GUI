"""Per-frame beam-monitor CSV, shared by Batch Integrate and Batch Correction.

The MATLAB SAXS workflow this GUI replaces emitted a CSV of ion-chamber
readings alongside its outputs. The values already exist here — see
``.context/DECISIONS.md`` 2026-09-25 — but only inside the ``.ave.zarr.zip``,
so the number you normalise against was not reachable without cracking the
archive open. This is the export, not new acquisition.

**Why this module exists rather than living in one of the two tabs.** The two
batch paths reach the same numbers by different routes:

* Batch Integrate already has them resolved per frame, via
  ``workers._HDF5StackGlobSource.metadata_for_index`` — see :func:`rows_from_metas`.
* Batch Correction never calls that. It has the aligned instrument tree from
  ``h5_metadata.align``, keyed by raw HDF5 path — see :func:`rows_from_aligned`.

What they genuinely share is the *per-hutch monitor mapping*, which used to be
private to ``_HDF5StackGlobSource``. It lives here now and ``workers`` imports
it back, so there is one table rather than two that can drift.

**Three distinct ways a value can be unavailable**, all real and all different
(pinned by ``tests/test_batch_zarr_gsas.py``):

1. the key is missing — that channel is not declared for this hutch at all
   (D has no transmission monitor; an unrecognised path has neither);
2. the value is ``None`` — declared, but absent or unreadable in this file.
   This is the documented auto-detect for E hutch's optional ``D2PD``;
3. the value is ``NaN`` — hardware present but not live. Real behaviour for
   E hutch's inactive SMS sub-config.

A column that is unavailable *for every frame* is dropped from the CSV rather
than written as a column of blanks, which would read as a bug. A column that
is merely missing *some* frames keeps its place, with empty cells for ``None``
and ``nan`` for ``NaN`` — those two are not the same thing and are not
flattened together.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

from midas_gui import h5_metadata

#: Fixed per-frame scalars, present for any source regardless of hutch.
#: ``current`` is storage-ring current in mA — NOT a beam monitor; the real
#: ones are in :data:`ION_CHAMBER_H5_PATHS`.
METADATA_H5_PATHS = {
    "temperature": "instrument/GSAS2_PVS/Temperature",
    "pressure": "instrument/GSAS2_PVS/Pressure",
    "current": "instrument/StorageRing/SRCurrent",
}

#: Real beam-monitor ion chambers, by hutch — stopgap mapping, from the
#: beamline's own confirmation (cross-checked against
#: ~/mnt/s1b/bluesky_dev/mpe_xml/20ide_instr_attributes_trans.xml) rather than
#: the HDF5 file's own ``active_instrument`` (documented upstream as always
#: empty, so it can't be used to pick a hutch). D hutch has no transmission
#: monitor yet, hence no "i" entry. E hutch's "i" is setup-dependent (a pin
#: diode, "D2PD", when present) — the "declared but absent in this file"
#: handling degrades that to None gracefully, which doubles as the auto-detect.
#: Station A is deliberately excluded: no sample sits in its beam path, so its
#: scalers (however I0/I1-suggestive their names) aren't a per-sample pair.
ION_CHAMBER_H5_PATHS = {
    "D": {"ion_chamber_i0": "instrument/Scalers/D/IC2"},
    "E": {"ion_chamber_i0": "instrument/Scalers/E/US_IC",
          "ion_chamber_i": "instrument/Scalers/E/D2PD"},
}

#: Sample-motion-system motor groups, by hutch. E has two coexisting
#: sub-configs (HL/HR) with no reliable way to tell which is physically in use
#: for a given file — both are captured rather than guessed, and the inactive
#: one shows up as all-NaN, i.e. the data says which was live.
SAMPLE_MOTOR_H5_GROUPS = {
    "D": ["instrument/SMS/D/HR"],
    "E": ["instrument/SMS/E/HL", "instrument/SMS/E/HR"],
}

#: Optional column groups, selectable per run. The ion chambers themselves are
#: not in here: they are the point of the file and are always written.
EXTRA_GROUPS = ("env", "motors")

#: Column order and header spelling. ``transmission`` is derived, not read.
#: Fixed per-frame scalars offered as the "env" extra group.
_ENV_COLUMNS = (("current", "ring_current_mA"), ("temperature", "temperature"),
                ("pressure", "pressure"))

MOTOR_PREFIX = "motor:"

#: Suffix on a dark-block column. The dark images themselves are not written
#: out, but their monitor readings are a real measurement of the shutter-
#: closed baseline and are the thing you subtract before taking any ratio.
#: Filename tail for each tab's beam-monitor CSV, appended to
#: ``<froot>_<detector>``. Both tabs write the same kind of sidecar about
#: the same scan and put it in the same place (beside the detector folder),
#: so only this two-letter tag says which produced it:
#: ``gC_1s_ICtweak_eiger2_bc.csv`` next to ``gC_1s_ICtweak_eiger2_bi.csv``.
SUFFIX_BATCH_CORRECTION = "_bc.csv"
SUFFIX_BATCH_INTEGRATE = "_bi.csv"


#: Retired. The dark is a ROW now (``kind="dark"``), not a companion column
#: per channel — see :func:`rows_from_tree`. Kept only so an external caller
#: importing the name still resolves; nothing in this module uses it.
DARK_SUFFIX = "_dark"

_SCALERS_PREFIX = "instrument/Scalers/"


#: Profiles that name a 20-ID station, and the hutch each one means. Used
#: only as a fallback — and deliberately NOT a prefix match: "1-ID-E" ends in
#: E but is a different beamline entirely, and reading its scalers as 20-ID's
#: would invent a monitor rather than report none.
#: Profile name -> the ``instrument/Scalers/<letter>`` group that profile's
#: station writes into. Only the hutch LETTER is needed: ``scaler_channels``
#: then reads whichever channels the file actually carries, so a station with
#: different hardware (1-ID-E records IC1/IC2/IC3 where 20-ID-E records
#: US_IC/DS_IC) needs no entry of its own beyond this.
#:
#: 1-ID-E was missing, and the cost was silent: ``resolve_hutch`` returned
#: None, ``scaler_channels`` returned nothing, no rows were built and **no
#: beam-monitor CSV was written at all** for anyone on that profile. The run
#: reported success and simply had no sidecar. Confirmed against 2026-10
#: 1-ID files, which carry instrument/Scalers/{B,C,E} with IC1-IC3 under E.
#:
#: Stations are added here only once their scaler group has been seen in a
#: real file -- 17-BM has no entry because nobody has checked, and guessing
#: a letter would write a CSV of the wrong channels rather than none.
_PROFILE_HUTCH = {"20-id-e": "E", "20-id-d": "D", "1-id-e": "E"}


def resolve_hutch(path, profile: Optional[str] = None) -> Optional[str]:
    """Stopgap hutch detection, from the source path or the active profile.

    The HDF5 file's own ``active_instrument`` is documented as always empty
    upstream (confirmed again on a real 2026-10 file, where it reads ``b""``),
    so there is no per-file signal to use.

    The path is tried first, since it is the more specific statement: a
    ``varexE``/``varexD`` folder names the station outright. But that only
    ever matched VAREX layouts — an Eiger run lives under ``eiger2/`` and
    matched nothing, so a file carrying perfectly good ``Scalers/E/US_IC``
    data silently produced no CSV. The profile is the fallback: the user has
    already told the GUI which station they are on, in the header.

    ``None`` (neither says) means every ion-chamber and sample-motor field is
    skipped, the same as any other "not available for this source".
    """
    text = str(path or "").lower()
    if "varexe" in text:
        return "E"
    if "varexd" in text:
        return "D"
    return _PROFILE_HUTCH.get(str(profile or "").strip().lower())


def metadata_h5_paths(hutch: Optional[str]) -> dict:
    """Flat ``{key: h5_path}`` for a hutch: the fixed scalars plus whichever
    ion-chamber entries apply (none, for an unrecognised hutch)."""
    paths = dict(METADATA_H5_PATHS)
    paths.update(ION_CHAMBER_H5_PATHS.get(hutch, {}))
    return paths


def _is_real(v) -> bool:
    """A value that carries information — not missing, not NaN."""
    if v is None:
        return False
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _fmt(v) -> str:
    """Empty for missing, ``nan`` for NaN, ``%.6g`` otherwise. Deliberately
    NOT ``cake_params``'s "0"-for-missing rule: that exists because its
    downstream reader rejects blank cells, whereas here a zero ion-chamber
    reading is a meaningful and wrong value."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if math.isnan(f) else f"{f:.6g}"


# ── which acquisitions were the light frames ─────────────────────────────────

def scaler_channels(tree: dict, hutch: Optional[str]) -> list:
    """Every per-acquisition scaler channel for ``hutch``, in file order.

    Read from the file rather than from a mapping, because no mapping would
    stay honest: the E group alone carries US/DS ion chambers, four blade
    readings each, a pin diode and a TetrAMM, and which of them is wired to
    what changes between setups. The ``*_sensitivity`` companions are
    scalars, not per-acquisition, so they are not columns here.
    """
    if not hutch:
        return []
    prefix = f"{_SCALERS_PREFIX}{hutch}/"
    out = []
    for path, arr in tree.items():
        if not path.startswith(prefix) or path.endswith("_sensitivity"):
            continue
        arr = np.atleast_1d(np.asarray(arr))
        if arr.ndim == 1 and arr.dtype.kind in "fiub" and arr.size > 1:
            out.append(path)
    return sorted(out)


def split_light_dark(tree: dict, *, n_light: int, n_dark: int,
                     hutch: Optional[str]) -> tuple:
    """``(light_offset, dark_offset, note)`` into the per-acquisition arrays.

    A VAREX/Eiger HDF5 stack records one metadata sample per detector
    acquisition — light *and* dark — in a single flat chronological array, so
    a file with 20 light and 20 dark frames has 40 scaler entries and
    something has to say which half is which.

    The monitor itself is the signal: an ion chamber reads roughly ten times
    higher with the shutter open, so the block with the larger I0 is the
    light block. That is order-agnostic, which matters because the two
    detectors disagree — VAREX trails its darks, and a 2026-10 Eiger file
    puts them FIRST. The previous rule assumed the leading block was always
    the lights, and so reported shutter-closed readings for every Eiger run.

    The acquisition timestamps and the frame counts are used as a CHECK, not
    as the decision: a clean split shows one anomalously long gap at the
    boundary (confirmed on real data: 39 gaps of ~10.01 s and one of 15.39 s,
    exactly at index 19). When the gap disagrees with the intensity, the note
    says so rather than quietly preferring one.

    ``dark_offset`` is None when the arrays hold only the light frames.
    """
    total = 0
    for path in scaler_channels(tree, hutch) or list(tree):
        arr = np.atleast_1d(np.asarray(tree[path]))
        if arr.ndim == 1:
            total = max(total, int(arr.size))
    n_light = int(n_light or 0)
    n_dark = int(n_dark or 0)
    if n_light <= 0 or total < n_light + n_dark or n_dark <= 0:
        return 0, None, "single block (no dark acquisitions in the metadata)"

    # Candidate split: the two blocks are [0, n_light) and [n_light, ...).
    chans = scaler_channels(tree, hutch)
    i0_path = (ION_CHAMBER_H5_PATHS.get(hutch, {}) or {}).get("ion_chamber_i0")
    probe = i0_path if i0_path in tree else (chans[0] if chans else None)
    if probe is None:
        return 0, n_light, "no scaler to compare; assumed lights first"
    arr = np.asarray(tree[probe], dtype=np.float64)
    first = float(np.nanmean(arr[:n_light]))
    second = float(np.nanmean(arr[n_light:n_light + n_dark]))
    lights_first = first >= second
    light_off, dark_off = (0, n_light) if lights_first else (n_dark, 0)

    note = (f"lights {'first' if lights_first else 'second'} by {Path(probe).name}"
            f" ({first:.5g} vs {second:.5g})")
    gap = _gap_index(tree)
    if gap is not None:
        expected = n_light - 1 if lights_first else n_dark - 1
        note += (f"; timestamp gap at {gap} "
                 + ("agrees" if gap == expected else
                    f"DISAGREES (expected {expected})"))
    return light_off, dark_off, note


def _gap_index(tree: dict) -> Optional[int]:
    """Index of the one anomalously long inter-acquisition gap, or None.

    The timestamps live at ``NDArray/NDArrayTimeStamp`` on the files checked;
    ``misc/NDArrayTimeStamp`` is also accepted because that is where
    ``workers._metadata_frame_count`` has always looked for them.
    """
    for key in ("NDArray/NDArrayTimeStamp", "misc/NDArrayTimeStamp"):
        arr = tree.get(key)
        if arr is None:
            continue
        ts = np.atleast_1d(np.asarray(arr, dtype=np.float64))
        if ts.ndim != 1 or ts.size < 3:
            continue
        diffs = np.diff(ts)
        typical = float(np.median(diffs))
        idx = int(np.argmax(diffs))
        if typical > 0 and diffs[idx] > 1.3 * typical:
            return idx
    return None


# ── adapters: two paths, same rows ───────────────────────────────────────────

def rows_from_metas(metas) -> list:
    """Rows from Batch Integrate's per-frame ``metadata_for_index`` dicts.

    Keys arrive already resolved (``ion_chamber_i0``/``ion_chamber_i``/
    ``current``/``temperature``/``pressure``/``motor:<leaf>/<channel>``), so
    this is a rename into the CSV's own column names and nothing more.
    """
    rows = []
    for n, meta in enumerate(metas):
        meta = meta or {}
        row = {"frame": n, "source_file": "",
               "I0": meta.get("ion_chamber_i0"),
               "I": meta.get("ion_chamber_i")}
        for key, _header in _ENV_COLUMNS:
            row[key] = meta.get(key)
        for key, value in meta.items():
            if key.startswith(MOTOR_PREFIX):
                row[key] = value
        rows.append(row)
    return rows


def rows_from_tree(tree: dict, hutch: Optional[str], *, frame_ranges,
                   n_light: int, n_dark: int = 0, source: str = "",
                   subtracted_dark: str = "") -> tuple:
    """``(rows, note)`` straight from an unaligned :func:`h5_metadata.read_tree`.

    Does its own slicing and chunk-averaging rather than taking
    ``h5_metadata.align``'s output, because that helper reduces the LEADING
    ``n_aligned`` entries and the light block is not always leading — see
    :func:`split_light_dark`.

    Every scaler channel the hutch has is written, under its own name,
    exactly as recorded. No ratio is computed: a transmission needs an
    air/empty-beam I/I0 reference that this file does not carry, so a column
    called "transmission" here would be a raw ratio wearing a name it has
    not earned.

    **The dark is a ROW, not a set of columns.** The dark images are not
    written out, but their monitor readings are a real measurement of the
    shutter-closed baseline, so they belong in the file. They used to ride
    along as a ``<channel>_dark`` companion column on every chunk row, which
    doubled the width and repeated one measurement on each line as though it
    varied per chunk. It does not: the dark block is acquired once. So it
    gets one row, ``kind="dark"``, carrying the mean over the whole dark
    block under the SAME column names the lights use — directly comparable,
    and the table stays half as wide.

    ``subtracted_dark`` closes the gap that entry left open. The dark row is
    a monitor measurement of the dark block **inside this file**; the dark
    actually subtracted from the images is picked by
    ``frame_correct.resolve_dark``, whose ladder prefers the nearest
    ``_dark_before`` sibling and so is often a different acquisition. The
    two were both called "the dark" and only one of them was in the CSV, so
    a reader comparing a chunk against the baseline row could be comparing
    it against a dark that was never subtracted. Passing the resolver's own
    ``why`` string writes it into a ``subtracted_dark`` column on the dark
    row, so the file states both facts instead of implying they are one.
    """
    light_off, dark_off, note = split_light_dark(
        tree, n_light=n_light, n_dark=n_dark, hutch=hutch)
    ranges = list(frame_ranges or [])
    n_rows = len(ranges)

    def chunked(h5_path, offset):
        """One value per output frame, averaged over the same raw window the
        image chunk used — shifted onto the requested block."""
        arr = tree.get(h5_path)
        if arr is None or offset is None:
            return None
        arr = np.atleast_1d(np.asarray(arr))
        if arr.ndim != 1 or arr.size < offset + n_light:
            return None
        block = arr[offset:offset + n_light]
        return h5_metadata._chunk_mean(block, ranges)

    def block_mean(h5_path, offset, count):
        """One value over a whole acquisition block — what the dark row
        carries. The dark is acquired once, so it is averaged once rather
        than split along the light frames' chunk boundaries, which have no
        meaning on the other side of the split."""
        arr = tree.get(h5_path)
        if arr is None or offset is None or not count:
            return None
        arr = np.atleast_1d(np.asarray(arr))
        if arr.ndim != 1 or arr.size < offset + count:
            return None
        return h5_metadata._chunk_mean(arr[offset:offset + count],
                                       [(0, count - 1)])[0]

    series: dict = {}        # per-chunk light values
    dark_values: dict = {}   # one value per column, over the whole dark block
    for path in scaler_channels(tree, hutch):
        name = path.rsplit("/", 1)[-1]
        series[name] = chunked(path, light_off)
        dark_values[name] = block_mean(path, dark_off, n_dark)
    for key, _header in _ENV_COLUMNS:
        series[key] = chunked(METADATA_H5_PATHS[key], light_off)
        dark_values[key] = block_mean(METADATA_H5_PATHS[key], dark_off, n_dark)
    for group in SAMPLE_MOTOR_H5_GROUPS.get(hutch, []):
        for path in tree:
            if path.startswith(group + "/"):
                leaf = "/".join(path.split("/")[-2:])
                arr = chunked(path, light_off)
                if arr is not None:
                    series[MOTOR_PREFIX + leaf] = arr
                    dark_values[MOTOR_PREFIX + leaf] = block_mean(
                        path, dark_off, n_dark)

    rows = []
    # The dark leads, so the shutter-closed baseline is the first thing read
    # rather than something to scroll right for. Its range is over the dark
    # acquisition block's own sub-frames: the dark images are not written
    # out, so there is no output frame for it to index against.
    if dark_off is not None and n_dark and any(
            v is not None for v in dark_values.values()):
        row = {"kind": "dark", "source_file": source,
               "frame_start": 0, "frame_end": n_dark - 1}
        if subtracted_dark:
            # Named on the dark row only: it is a statement about what the
            # baseline is, and repeating it on every chunk row would be the
            # same width-doubling the companion columns were removed for.
            row["subtracted_dark"] = subtracted_dark
        row.update({key: dark_values.get(key) for key in series})
        rows.append(row)
    # A row is identified by its file and the raw sub-frame range it was
    # built from — the same range the output HDF5 records in `frame_ranges`
    # — rather than by an ordinal. One CSV can hold rows from several source
    # files (they are accumulated per froot), where an ordinal repeats.
    for n, (lo, hi) in enumerate(ranges):
        row = {"kind": "chunk", "source_file": source,
               "frame_start": lo, "frame_end": hi}
        for key, arr in series.items():
            row[key] = None if arr is None else arr[n]
        rows.append(row)
    return rows, note


# ── writer ───────────────────────────────────────────────────────────────────

#: Columns that identify a row rather than measure anything, in header
#: order. Never counted when deciding whether the file is worth writing, and
#: each appears ONLY when some row carries it — the two producers identify a
#: row differently and neither should grow the other's blank columns:
#:
#: * :func:`rows_from_tree` (Batch Correction) → ``kind``, ``source_file``,
#:   ``frame_start``, ``frame_end``. One CSV can hold rows from several
#:   source files, so the file name plus the raw sub-frame range IS the
#:   identity; a bare ordinal would not be unique across files.
#: * :func:`rows_from_metas` (Batch Integrate) → ``frame``, ``source_file``.
#:   It has one row per output frame and no raw range to report.
_INDEX_COLUMNS = (("kind", "kind"), ("frame", "frame"),
                  ("source_file", "source_file"),
                  ("frame_start", "frame_start"), ("frame_end", "frame_end"),
                  ("subtracted_dark", "subtracted_dark"))


def _columns(rows, extras) -> list:
    """``[(row key, header)]`` for the columns worth writing.

    A column no row has a live value for is dropped rather than blanked: on
    D hutch, or for a channel that was not wired up, an empty column reads as
    a broken detector instead of an absent one. Channel columns keep their
    own names — ``US_IC``, ``DS_IC``, ``D2PD`` — because naming one of them
    "I" would be asserting which is the transmitted monitor, and that is
    setup-dependent.
    """
    extras = set(extras or ())
    index_keys = {k for k, _h in _INDEX_COLUMNS}
    cols = [(k, h) for k, h in _INDEX_COLUMNS if any(k in r for r in rows)]
    measured = set()
    env_keys = {k for k, _h in _ENV_COLUMNS}
    chan_keys = sorted({k for r in rows for k in r
                        if k not in env_keys
                        and not k.startswith(MOTOR_PREFIX)
                        and k not in index_keys})
    for key in chan_keys:
        if any(_is_real(r.get(key)) for r in rows):
            cols.append((key, key))
            measured.add(key)
    if "env" in extras:
        for key, header in _ENV_COLUMNS:
            if any(_is_real(r.get(key)) for r in rows):
                cols.append((key, header))
                measured.add(key)
    if "motors" in extras:
        names = sorted({k for r in rows for k in r if k.startswith(MOTOR_PREFIX)})
        for key in names:
            if any(_is_real(r.get(key)) for r in rows):
                cols.append((key, key[len(MOTOR_PREFIX):]))
                measured.add(key)
    return cols if measured else []


def write_ion_csv(path, rows, *, extras=()) -> Optional[str]:
    """Write the per-frame monitor CSV; return its path, or None if skipped.

    Nothing is written when no real value survives anywhere — an A-hutch or
    unplaceable run should not litter the output folder with a file holding
    only a frame counter.
    """
    rows = list(rows or ())
    cols = _columns(rows, extras)
    if not rows or not cols:
        return None
    out = Path(str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [h for _k, h in cols]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: _fmt(row.get(k)) for k, h in cols})
    return str(out)
