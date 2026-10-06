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
_BASE_COLUMNS = (("frame", "frame"), ("I0", "I0"), ("I", "I"),
                 ("transmission", "transmission"))
_ENV_COLUMNS = (("current", "ring_current_mA"), ("temperature", "temperature"),
                ("pressure", "pressure"))

MOTOR_PREFIX = "motor:"


#: Profiles that name a 20-ID station, and the hutch each one means. Used
#: only as a fallback — and deliberately NOT a prefix match: "1-ID-E" ends in
#: E but is a different beamline entirely, and reading its scalers as 20-ID's
#: would invent a monitor rather than report none.
_PROFILE_HUTCH = {"20-id-e": "E", "20-id-d": "D"}


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


def _transmission(row: dict):
    """``I / I0``, or None when it is not defined. A zero I0 is a dropped
    beam, not an infinite transmission."""
    i, i0 = row.get("I"), row.get("I0")
    if not (_is_real(i) and _is_real(i0)) or float(i0) == 0.0:
        return None
    return float(i) / float(i0)


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
        row = {"frame": n,
               "I0": meta.get("ion_chamber_i0"),
               "I": meta.get("ion_chamber_i")}
        for key, _header in _ENV_COLUMNS:
            row[key] = meta.get(key)
        for key, value in meta.items():
            if key.startswith(MOTOR_PREFIX):
                row[key] = value
        rows.append(row)
    return rows


def rows_from_aligned(aligned: dict, hutch: Optional[str], n_frames: int) -> list:
    """Rows from Batch Correction's ``h5_metadata.align`` result.

    ``aligned`` is ``{h5 path: one value per output frame}``. Arrays that
    ``align`` could not reduce (2-D, or shorter than the frame count) are
    left alone by it, so they are skipped here rather than mis-indexed.
    """
    def series(h5_path):
        arr = aligned.get(h5_path)
        if arr is None:
            return None
        arr = np.atleast_1d(np.asarray(arr))
        if arr.ndim != 1 or arr.size < n_frames:
            return None
        return arr

    chambers = ION_CHAMBER_H5_PATHS.get(hutch, {})
    named = {"I0": series(chambers.get("ion_chamber_i0")),
             "I": series(chambers.get("ion_chamber_i"))}
    for key, _header in _ENV_COLUMNS:
        named[key] = series(METADATA_H5_PATHS[key])

    # Motor leaves live under the hutch's SMS group(s); their channel names
    # differ per station and are discovered rather than hardcoded.
    motors = {}
    for group in SAMPLE_MOTOR_H5_GROUPS.get(hutch, []):
        for h5_path in aligned:
            if h5_path.startswith(group + "/"):
                leaf = "/".join(h5_path.split("/")[-2:])
                arr = series(h5_path)
                if arr is not None:
                    motors[MOTOR_PREFIX + leaf] = arr

    rows = []
    for n in range(n_frames):
        row = {"frame": n}
        for key, arr in named.items():
            row[key] = None if arr is None else arr[n]
        for key, arr in motors.items():
            row[key] = arr[n]
        rows.append(row)
    return rows


# ── writer ───────────────────────────────────────────────────────────────────

def _columns(rows, extras) -> list:
    """``[(row key, header)]`` for the columns worth writing.

    A column every row leaves unavailable is dropped, not blanked: on D hutch
    that means no ``I`` and no ``transmission`` at all, which says "this
    station has no transmission monitor" instead of showing an empty column
    that reads as a failure.
    """
    extras = set(extras or ())
    cols = []
    for key, header in _BASE_COLUMNS:
        if key == "frame" or any(_is_real(r.get(key)) for r in rows):
            cols.append((key, header))
    if "env" in extras:
        cols += [(k, h) for k, h in _ENV_COLUMNS
                 if any(_is_real(r.get(k)) for r in rows)]
    if "motors" in extras:
        names = sorted({k for r in rows for k in r if k.startswith(MOTOR_PREFIX)})
        cols += [(k, k[len(MOTOR_PREFIX):]) for k in names
                 if any(_is_real(r.get(k)) for r in rows)]
    return cols


def write_ion_csv(path, rows, *, extras=()) -> Optional[str]:
    """Write the per-frame monitor CSV; return its path, or None if skipped.

    Nothing is written when no real value survives anywhere — an A-hutch or
    unrecognised-path run should not litter the output folder with a file
    containing only a frame counter.
    """
    rows = list(rows or ())
    for row in rows:
        row["transmission"] = _transmission(row)
    cols = _columns(rows, extras)
    if not rows or len(cols) <= 1:          # frame counter alone is not data
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
