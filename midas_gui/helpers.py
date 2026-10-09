"""Module-level helpers: image IO, transforms, ring prediction, spec building,
and the no-scroll spinbox / two-column layout widgets used everywhere.

These are ported verbatim from midas_workflow_gui_v3.py (the frozen template) so
the established conventions in context/design_rules.md are preserved exactly.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui.constants import _SENTINELS, _LATT, H5_EXTS, _V2_TO_V1, HC_KEV_A
from midas_gui import style as S

# checkmark SVG written to a temp file so the QSS image: property can use it
import tempfile as _tf
import atexit as _atexit
import os as _os


def _make_checkmark_svg() -> str:
    """White tick SVG → temp file.  Returns forward-slash path for Qt QSS."""
    _svg = (
        b"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'>"
        b"<polyline points='2,7 5.5,11 12,3' stroke='white' stroke-width='2.2'"
        b" fill='none' stroke-linecap='round' stroke-linejoin='round'/>"
        b"</svg>"
    )
    f = _tf.NamedTemporaryFile(suffix=".svg", delete=False)
    f.write(_svg); f.close()
    _atexit.register(_os.unlink, f.name)
    return f.name.replace("\\", "/")   # Qt QSS needs forward slashes on Windows


def _make_arrow_svg(direction: str = "down", color: str = "#333333") -> str:
    """Small filled triangle arrow → temp file, for spinbox/combo sub-controls.

    direction: 'up' or 'down'. Returns a forward-slash path for Qt QSS.
    """
    pts = "2,7 8,7 5,2" if direction == "up" else "2,3 8,3 5,8"
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'>"
        f"<polygon points='{pts}' fill='{color}'/></svg>"
    ).encode()
    f = _tf.NamedTemporaryFile(suffix=".svg", delete=False)
    f.write(svg); f.close()
    _atexit.register(_os.unlink, f.name)
    return f.name.replace("\\", "/")


def apply_ui_scale() -> float:
    """Set QT_SCALE_FACTOR from the configured ``ui.ui_scale`` and enable crisp
    HiDPI pixmaps. Must run before any QApplication instance exists (Qt only
    reads QT_SCALE_FACTOR / the AA_UseHighDpiPixmaps attribute at that point).

    Shared by ``app.main()`` and ``auto_attenuation.app.main()`` so a
    standalone window scales identically to the main GUI at any interface
    scale. Returns the clamped scale actually applied.
    """
    from midas_gui import constants as C
    try:
        scale = float(getattr(C, "DEFAULT_UI_SCALE", 1.0) or 1.0)
    except Exception:
        scale = 1.0
    scale = min(4.0, max(0.5, scale))
    _os.environ["QT_SCALE_FACTOR"] = f"{scale:.4g}"
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    return scale


def browse_start_dir(path_text: str, fallback: str = "") -> str:
    """Directory to seed a Browse dialog at, given a path field's current text.

    Returns the typed path itself if it's an existing directory, its parent
    if that exists, else *fallback*.
    """
    text = (path_text or "").strip()
    if text:
        p = Path(text)
        if p.is_dir():
            return str(p)
        if p.parent.exists():
            return str(p.parent)
    return fallback


def path_is_missing(line_edit: QtWidgets.QLineEdit, parent: QtWidgets.QWidget, *,
                     is_output_dir: bool = False) -> bool:
    """Check the field's current text and pop up a dialog if it's a
    non-empty path that doesn't exist on disk; returns whether it popped one.

    Informational ("will be created when you run") for ``is_output_dir=True``,
    a "Not found" warning otherwise. Empty text never triggers a popup.
    Use this directly (as an early-return guard) inside a field's own
    ``returnPressed``/``editingFinished`` handler when that handler already
    acts on the path — e.g. loading it — so a missing path shows this one
    friendly message instead of *also* whatever error the load raises. For a
    field with no such handler, connect ``warn_if_path_missing`` instead.
    """
    text = line_edit.text().strip()
    if not text or Path(text).exists():
        return False
    if is_output_dir:
        QtWidgets.QMessageBox.information(
            parent, "Output folder",
            f"This folder does not exist yet:\n\n{text}\n\n"
            "It will be created when you run.")
    else:
        QtWidgets.QMessageBox.warning(
            parent, "Not found", f"Path not found:\n\n{text}")
    return True


def warn_if_path_missing(line_edit: QtWidgets.QLineEdit,
                          parent: QtWidgets.QWidget, *,
                          is_output_dir: bool = False) -> None:
    """Connect ``line_edit.returnPressed`` to a missing-path check
    (``path_is_missing``). Purely additive — safe to call alongside any
    handler the field already has, since Qt fires every connected slot; use
    this for a field whose existing handler (if any) doesn't itself act on
    the path (e.g. it only clamps a spinbox range), so there's no risk of a
    second, less friendly error dialog stacking on top of this one.
    """
    line_edit.returnPressed.connect(
        lambda: path_is_missing(line_edit, parent, is_output_dir=is_output_dir))


# ── Image IO ──────────────────────────────────────────────────────────────────

def _load_image(path: str | Path, data_loc: str = "exchange/data",
                frame: int = 0) -> np.ndarray:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        return np.asarray(tifffile.imread(str(p)), dtype=np.float32)
    if ext in H5_EXTS:
        import h5py
        with h5py.File(str(p), "r") as f:
            dset = f[data_loc]
            data = dset[frame] if dset.ndim >= 3 else dset[...]
        return np.asarray(data, dtype=np.float32)
    if ".ge" in p.name.lower():
        arr = np.fromfile(str(p), dtype=np.uint16, offset=8192)
        for side in (2048, 4096, 1024, 512):
            if arr.size >= side * side and arr.size % (side * side) == 0:
                return arr.reshape(-1, side, side)[frame].astype(np.float32)
        raise ValueError(f"Cannot reshape GE file {p}")
    raise ValueError(f"Unsupported format: {p.suffix}")


_COMBINE_OPS = {
    "mean": lambda s: np.mean(s, axis=0, dtype=np.float64),
    "sum": lambda s: np.sum(s, axis=0, dtype=np.float64),
    "max": lambda s: np.max(s, axis=0),
    "median": lambda s: np.median(s, axis=0),
}


def _stack_chunk_bounds(n: int, k: int, *, chunk_size: Optional[int],
                        raw_start: Optional[int], raw_end: Optional[int]):
    """Inclusive 0-based ``(lo, hi)`` raw sub-frame bounds of combined chunk
    ``k`` in an ``n``-sub-frame stack, or ``None`` when ``k`` is past the end
    (or the ``raw_start``/``raw_end`` filter leaves nothing).

    The single source of truth for this arithmetic: both
    ``read_hdf5_stack_combined`` (all chunks) and ``read_hdf5_stack_chunk``
    (exactly one) go through it, so a whole-file read and a single-chunk read
    of the same file can never disagree about where chunk ``k`` starts — and
    ``workers._HDF5StackGlobSource._stat`` can keep deriving the chunk COUNT
    from the dataset shape alone, which only stays honest while the chunking
    rule has one definition.
    """
    lo = max(0, raw_start) if raw_start is not None else 0
    hi = min(n - 1, raw_end) if raw_end is not None else n - 1
    n_eff = hi - lo + 1
    if n_eff <= 0 or k < 0:
        return None
    size = chunk_size if chunk_size else n_eff
    start = lo + k * size
    if start > hi:
        return None
    return start, min(start + size - 1, hi)


def read_hdf5_stack_chunk(path, dataset: str, k: int, *,
                          chunk_size: Optional[int] = None, op: str = "mean",
                          raw_start: Optional[int] = None,
                          raw_end: Optional[int] = None):
    """One combined frame — chunk ``k`` — out of an HDF5 sub-frame stack,
    reading ONLY that chunk's raw sub-frames.

    Same chunking, filtering and ``op`` semantics as
    ``read_hdf5_stack_combined``; the difference is cost. That function
    decodes every chunk and returns the list, so asking it for one frame of a
    1442-sub-frame VAREX file reads all 23.9 GB (~230 s over NFS) and holds
    ~1.9 GB of combined frames, to hand back 33 MB. Random access into a
    large stack — a Detector-view preview, a parallel ``BatchWorker`` chunk
    starting mid-file — must not pay for the frames it never asked for.

    Returns a 2-D ``float32`` array, or ``None`` when ``k`` is past the last
    chunk. A plain 2-D dataset is the single chunk 0 and is passed through
    unchanged (``chunk_size``/``op`` ignored), matching
    ``read_hdf5_stack_combined``'s ``[dataset]``.
    """
    import h5py
    combine = _COMBINE_OPS.get(op, _COMBINE_OPS["mean"])
    with h5py.File(str(path), "r") as f:
        dset = f[dataset]
        if dset.ndim == 2:
            return np.asarray(dset[...], dtype=np.float32) if k == 0 else None
        bounds = _stack_chunk_bounds(int(dset.shape[0]), k, chunk_size=chunk_size,
                                     raw_start=raw_start, raw_end=raw_end)
        if bounds is None:
            return None
        start, end = bounds
        stack = np.asarray(dset[start:end + 1], dtype=np.float32)
        return combine(stack).astype(np.float32)


def read_hdf5_stack_combined(path, dataset: str, *, chunk_size: Optional[int] = None,
                             op: str = "mean", raw_start: Optional[int] = None,
                             raw_end: Optional[int] = None) -> list:
    """Read an HDF5 ``(N, H, W)`` (or plain ``(H, W)``) dataset and combine
    consecutive raw sub-frames into one or more 2-D frames.

    For a VAREX-style file where every raw sub-frame belongs to the same
    scan point (e.g. ``exchange/data`` shape ``(10, 2880, 2880)``),
    ``chunk_size=None`` (or ``0``) combines ALL frames into one — mirrors
    ``mpe_wf_saxs_waxs``'s ``--avg-full-stack`` mode, which sets its
    equivalent (``OmegaSumFrames``) to the file's own frame count for
    exactly this case. A positive ``chunk_size`` instead splits the N
    frames into ``ceil(N / chunk_size)`` contiguous chunks, each combined
    independently (mirrors ``run_background_correction.py``'s
    ``_chunked``/``_aggregate``).

    ``op`` is one of "mean" (default) / "sum" / "max" / "median".

    ``raw_start``/``raw_end`` (0-based, inclusive, both optional) restrict
    which raw sub-frames are read/combined at all — e.g. ``raw_start=2,
    raw_end=7`` on a 10-frame dataset combines only frames 2..7, ignoring
    0, 1, 8 and 9 entirely, applied BEFORE chunking. Ignored for a plain
    2-D dataset (nothing to sub-select). An empty effective range (e.g.
    ``raw_start`` past the dataset's last frame) returns ``[]`` rather than
    raising.

    Returns a list of 2-D ``float32`` arrays (length 1 for the common
    whole-file case). A plain 2-D dataset returns ``[dataset]``
    unchanged, ``chunk_size``/``op`` ignored.
    """
    import h5py
    combine = _COMBINE_OPS.get(op, _COMBINE_OPS["mean"])
    with h5py.File(str(path), "r") as f:
        dset = f[dataset]
        if dset.ndim == 2:
            return [np.asarray(dset[...], dtype=np.float32)]
        n = int(dset.shape[0])
        out = []
        k = 0
        while True:
            bounds = _stack_chunk_bounds(n, k, chunk_size=chunk_size,
                                         raw_start=raw_start, raw_end=raw_end)
            if bounds is None:
                break
            start, end = bounds
            stack = np.asarray(dset[start:end + 1], dtype=np.float32)
            out.append(combine(stack).astype(np.float32))
            k += 1
        return out


def _apply_im_trans(image: np.ndarray, codes: tuple) -> np.ndarray:
    """Apply MIDAS image transform codes: 1=flipY, 2=flipZ, 3=transpose."""
    for c in codes:
        if c == 1:
            image = image[:, ::-1]
        elif c == 2:
            image = image[::-1, :]
        elif c == 3:
            image = image.T
    return np.ascontiguousarray(image)


# ── Hydra (4-panel GE detector) sibling-file auto-discovery ────────────────────

_HYDRA_PANEL_RE = re.compile(r"\bge([1-4])\b", re.IGNORECASE)


def hydra_panel_index(path: str) -> Optional[int]:
    """Return the Hydra panel number (1-4) encoded in `path`, or None.

    Matches a `geN` token as a whole word, e.g. the `ge1` folder or the
    `.ge1.` filename infix used by this beamline's naming convention (both
    can appear in the same path, e.g. `.../ge1/scan_002020.ge1.h5`)."""
    m = _HYDRA_PANEL_RE.search(Path(path).as_posix())
    return int(m.group(1)) if m else None


def hydra_siblings(path: str) -> dict:
    """Find the sibling ge1-ge4 files for `path` by substituting every
    `geN` token with `geM`. Returns only paths that exist on disk (may be
    fewer than 4 — e.g. a shot that wasn't saved for every panel)."""
    n = hydra_panel_index(path)
    if n is None:
        return {}
    out = {}
    for m in range(1, 5):
        cand = _HYDRA_PANEL_RE.sub(f"ge{m}", path)
        if Path(cand).exists():
            out[m] = cand
    return out


def im_trans_codes_from_checkboxes(flip_y, flip_z, transp) -> list:
    """Ordered MIDAS ``ImTransOpt`` codes from the 3 Transforms checkboxes.

    Fixed composition order (flips, then transpose) — matches
    ``_apply_im_trans`` and is what every tab's "Transforms:" row emits.
    """
    codes = []
    if flip_y.isChecked(): codes.append(1)
    if flip_z.isChecked(): codes.append(2)
    if transp.isChecked(): codes.append(3)
    return codes


def parse_im_trans(text: str) -> list:
    """Ordered ``ImTransOpt`` codes from a MIDAS paramstest's text.

    ``ImTransOpt`` is a repeatable key — one line per transform op, applied
    in file order. A lone ``ImTransOpt 0`` is MIDAS's explicit no-op and is
    dropped.
    """
    codes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "ImTransOpt":
            try:
                c = int(float(parts[1]))
            except ValueError:
                continue
            if c != 0:
                codes.append(c)
    return codes


def is_h5(path: str) -> bool:
    return Path(path).suffix.lower() in H5_EXTS


# ── Auto-detect geometry from a loaded file (Data Viewer / Calibrate) ──────────
# Filename conventions and the HDF5 energy-metadata location below are specific
# to the APS 1-ID-E / 20-ID-D / 20-ID-E beamlines, so detection only fires for
# those profiles (see settings.active_profile()) — elsewhere the tags/dataset
# path may not mean the same thing (or may not exist at all).
_AUTO_DETECT_PROFILES = {"1-ID-E", "20-ID-D", "20-ID-E"}
_DETECTOR_FILENAME_TAGS = (
    (".ge1", "ge"), (".ge2", "ge"), (".ge3", "ge"), (".ge4", "ge"), (".ge5", "ge"),
    (".vrx", "vrx"),
    (".pxrd", "pxrd"), (".pixi", "pxrd"),
    (".pmg", "pimega"),
)
# Pixel size (µm) for each recognized detector tag.
#
# "pxrd" was long listed without a size, so a Pixirad was identified but its
# pixel size never auto-populated. Both halves of that bit the 1-ID-E SAXS
# setup on 2026-10-08: files there are written ".pixi", which this table did
# not match at all, so loading one left the px box holding whatever the
# previously-loaded calibration had put there (200 µm, from a GE) — and a
# geometry saved from that card described the GE's pixel, not the Pixirad's.
# 62 µm is the PIXIRAD CdTe pitch and is already what constants.PIXEL_PRESETS
# offers under "Pixirad".
_DETECTOR_PIXEL_UM = {"ge": 200.0, "vrx": 150.0, "pimega": 55.0, "pxrd": 62.0}
# HDF5 dataset holding the beam energy (keV) used to derive wavelength, on the
# beamlines above — confirmed as the authoritative source over the other
# energy-like datasets present in these files (HRM/IDEnergy readbacks, which
# may reflect a different or inactive monochromator/undulator setpoint).
_ENERGY_DATASET = "instrument/HEM/Energy"


def detect_detector_from_filename(path: str) -> Optional[str]:
    """Detector tag ("ge"/"vrx"/"pxrd") from a known filename convention, or
    None if the filename doesn't match any of them."""
    name = Path(path).name.lower()
    for tag, det in _DETECTOR_FILENAME_TAGS:
        if tag in name:
            return det
    return None


def detect_wavelength_from_h5(path: str) -> Optional[float]:
    """Wavelength (Å) derived from `_ENERGY_DATASET` (keV) in an HDF5 frame
    file, or None if the file isn't HDF5, the dataset is absent, or anything
    else goes wrong reading it (best-effort, like project.py's provenance
    logging — a metadata read must never block loading the file)."""
    if not is_h5(path):
        return None
    try:
        import h5py
        with h5py.File(path, "r") as f:
            ds = f.get(_ENERGY_DATASET)
            if ds is None:
                return None
            energy_kev = float(np.atleast_1d(ds[()])[0])
        if energy_kev <= 0:
            return None
        return HC_KEV_A / energy_kev
    except Exception:
        return None


def detect_geometry_from_path(path: str, *, profile: Optional[str] = None) -> dict:
    """Best-effort auto-detected geometry hints from a just-loaded file's name
    ('pxY' from a known detector-file naming convention) and, for an HDF5
    frame file, its beam-energy metadata ('wavelength_A') — gated to the
    beamline profiles that use these conventions. Returns a dict using the
    same key vocabulary as DetectorGeometryCard.get_geometry/set_geometry
    (e.g. {"pxY": 200.0, "wavelength_A": 0.1653}); a field is omitted rather
    than guessed when it can't be detected, so callers should only override
    the fields actually present."""
    if profile is None:
        from midas_gui import settings
        profile = settings.active_profile()
    if profile not in _AUTO_DETECT_PROFILES:
        return {}
    out = {}
    px = _DETECTOR_PIXEL_UM.get(detect_detector_from_filename(path))
    if px is not None:
        out["pxY"] = px
    wl = detect_wavelength_from_h5(path)
    if wl is not None:
        out["wavelength_A"] = wl
    return out


class BcParts(NamedTuple):
    """The pieces of mpe_wf's output-folder convention, parsed off a data path.

    Kept as a decomposition rather than a finished string because Batch
    Integrate and Calibrate need different tails from the *same* parse — see
    :func:`suggest_integration_output_dir` and :func:`suggest_working_dir`.
    """
    root: Path        # <outroot>/<expid>_bc, or the shallow fallback root
    froot: str
    detector: str     # "" when the layout is too shallow to locate one
    positional: bool  # True when read off the 4-deep mpe_wf layout
    has_bc: bool      # True when `root` actually ends in "_bc"


def bc_path_parts(data_path, *, expid_fallback: str = "") -> Optional[BcParts]:
    """Parse ``<outroot>/<expid>/<detector>/<froot>/<files>`` off `data_path`.

    Mirrors mpe_wf_saxs_waxs's own ``outroot/<expid>_bc/<froot>/<detector>/``
    output-folder convention (``~/mnt/<station>/<expid>_bc/<froot>/<detector>/``
    — e.g. beamline home ``/home/beams/S20IDUSER`` for 20-ID,
    ``/home/beams/S1IDUSER`` for 1-ID; ``~`` itself is just whatever directory
    the source path happens to live under, this function never hardcodes it).

    Raw data is read from mpe_wf's fixed layout — four directories deep
    counting the file's own containing folder — so ``expid``/``detector``/
    ``outroot`` are read *positionally* rather than asked of the user: the
    whole point is to read this off the data that's actually loaded, not
    require someone to first type the Exp ID. When the source doesn't have
    that much depth (e.g. files sitting directly under a flat folder), falls
    back to `expid_fallback` if given, else to the source folder itself, and
    reports ``positional=False`` so callers can tell a located layout from a
    guess. Pure path arithmetic — nothing here touches the filesystem.

    Returns None when `data_path` is empty.
    """
    if not data_path:
        return None
    from midas_gui.workers import froot_and_frame_num  # helpers<-workers cycle
    p = Path(data_path)
    name = p.name
    if any(c in name for c in "*?["):
        # A glob pattern ("<folder>/<stem>*"), not a real file — take the
        # literal prefix before the first wildcard as the stem.
        name = re.split(r"[*?\[]", name, maxsplit=1)[0].rstrip("_-.") or name
    else:
        name = p.stem
    froot, _num, _tag = froot_and_frame_num(name, 0)

    # froot_dir = the folder actually holding the files (often froot-named
    # itself); its parent is <detector>, and <expid> is one level above that —
    # mpe_wf's layout puts them at this fixed depth regardless of what any of
    # these folders happen to be named.
    froot_dir = p.parent
    ancestors = froot_dir.parents
    if len(ancestors) >= 3:
        return BcParts(root=ancestors[2] / f"{ancestors[1].name}_bc",
                       froot=froot, detector=ancestors[0].name,
                       positional=True, has_bc=True)

    expid = (expid_fallback or "").strip()
    root = froot_dir / f"{expid}_bc" if expid else froot_dir
    return BcParts(root=root, froot=froot, detector="",
                   positional=False, has_bc=bool(expid))


def suggest_integration_output_dir(data_path, *, expid_fallback: str = "") -> Optional[Path]:
    """``<outroot>/<expid>_bc/<froot>/<detector>`` — Batch Integrate's convention.

    The shallow fallback deliberately guards against duplicating ``froot`` when
    the source folder is itself named after it (the common
    ``<froot>/<froot>_NNNNNN.tif`` layout).
    """
    parts = bc_path_parts(data_path, expid_fallback=expid_fallback)
    if parts is None:
        return None
    if parts.positional:
        return parts.root / parts.froot / parts.detector
    return (parts.root if parts.root.name == parts.froot
            else parts.root / parts.froot)


def _folder_output_dir(files, expid_fallback: str = ""):
    """The one output dir a SOURCE FOLDER's files imply.

    ``suggest_integration_output_dir`` reads the froot off each file's own
    NAME, so one folder yields several: a load step's darks are
    ``..._dark_before``/``..._dark_after`` and derive directories of their
    own. Picking a folder by hand puts all of them in one place, so grouping
    must too. Cannot be answered by passing the folder itself -- the
    positional parse assumes the last component is a file, and a directory
    path comes back scrambled (``s1c_bc/Faber_A_restart4_load_step/
    connolly_oct26``).

    Chosen by agreement: the directory most of the folder's files imply,
    breaking a tie on the shortest path. Scan frames outnumber the two
    darks, and a dark's froot is the scan's froot plus a suffix, so both
    rules point the same way -- and a 1-scan-plus-2-darks folder, where the
    count is a three-way tie, still lands on the scan.
    """
    counts = {}
    for f in files:
        d = suggest_integration_output_dir(f, expid_fallback=expid_fallback)
        if d is not None:
            counts[str(d)] = counts.get(str(d), 0) + 1
    if not counts:
        return None
    return Path(min(counts, key=lambda k: (-counts[k], len(k), k)))


def group_paths_by_output_dir(paths, *, out_dir=None, expid_fallback: str = "") -> list:
    """Split a file selection into ``[(out_dir, [paths]), ...]`` -- one entry
    per SOURCE FOLDER, each with the output directory that folder implies.

    A recursive "Full folder" pick spans many source folders, and the
    ``<expid>_bc/<froot>/<detector>`` convention derives the output dir from
    each file's own path, so the selection maps to many dirs rather than
    one. Applying it once writes every subfolder's output into whichever
    folder the first file implied -- reported from 1-ID-E as every load step
    landing in one directory.

    Grouping is by containing folder, not by derived directory, so one run
    covers exactly what picking that folder by hand covers (its darks
    included -- see :func:`_folder_output_dir`).

    ``out_dir`` is the Output field as it stands:

    * unset, or equal to a directory one of the groups already derives (i.e.
      the value the field auto-filled to from one file), it is a stale
      single-folder guess and each group uses its own derived directory.
    * anything else is a deliberate choice of root, and the groups are placed
      under it at the same relative positions they have to each other, so a
      custom root fans out rather than collapsing.

    A selection inside ONE folder returns a single entry -- every
    non-recursive pick -- which is the caller's existing one-run path.
    """
    paths = [str(x) for x in paths or []]
    if not paths:
        return []

    by_folder = {}
    for f in paths:
        by_folder.setdefault(str(Path(f).parent), []).append(f)
    if len(by_folder) == 1:
        only = next(iter(by_folder.values()))
        return [(_folder_output_dir(only, expid_fallback)
                 or (Path(out_dir) if out_dir else None), only)]

    resolved = []          # (derived_dir_or_None, files)
    for folder in sorted(by_folder):
        files = by_folder[folder]
        resolved.append((_folder_output_dir(files, expid_fallback), files))

    keys = [str(d) for d, _ in resolved if d is not None]
    custom = bool(out_dir) and str(out_dir) not in keys
    if custom and keys:
        import os
        common = Path(os.path.commonpath(keys)) if len(keys) > 1 else Path(keys[0]).parent
        rebase = lambda d: Path(out_dir) / Path(str(d)).relative_to(common)
    else:
        rebase = lambda d: Path(str(d))

    groups = [((rebase(d) if d is not None
                else (Path(out_dir) if out_dir else None)), files)
              for d, files in resolved]
    groups.sort(key=lambda g: str(g[0]))
    return groups


#: Batch Correction's per-op subfolder under the integration output dir —
#: ``dark_subtracted_mean``, ``dark_subtracted_max``, … Named for what the
#: files in it are, the same way Batch Integrate's output is split into
#: per-format subfolders, so a froot's reduced frames and its cakes sit side
#: by side instead of one burying the other. The op is in the folder name
#: because one run can produce several: a mean and a max of the same scan
#: must not land on top of each other.
CORRECTION_SUBDIR_PREFIX = "dark_subtracted"


def correction_subdir(op: str) -> str:
    """``dark_subtracted_<op>`` — one output folder per combine method."""
    return f"{CORRECTION_SUBDIR_PREFIX}_{str(op).lower()}"

#: Tail of a Batch Correction output filename: ``<stem>_cor.hdf5``, i.e. the
#: source name with its whole dotted extension replaced (see
#: ``BatchCorrectionWorker._out_path``, which strips a detector tag like
#: ``.vrx`` along with the extension — it no longer describes the file).
#: ``.hdf5`` is in ``H5_EXTS``, so the result loads straight back into any tab.
#: Both are editable per run; these are only the defaults.
CORRECTION_SUFFIX = "_cor"
CORRECTION_EXT = ".h5"


def suggest_panel_base_output_dir(data_path, *, expid_fallback: str = "") -> Optional[Path]:
    """``<outroot>/<expid>_bc/<froot>`` — the SHARED base for a Hydra run.

    Deliberately :func:`suggest_integration_output_dir` minus its
    ``<detector>`` tail. The Hydra page's Output field is one directory for
    all four panels and each panel appends its own ``ge{n}/`` (see
    ``HydraBatchPage._run_panel``), so including the detector segment here
    would produce ``…/<froot>/ge1/ge1``.

    The representative path is any one panel's file: every panel of a Hydra
    set shares a froot and an outroot, and differs only in the detector.

    The detector tag is also stripped off the END of the froot when it is
    there. ``bc_path_parts`` reads the froot off the file name, which at
    1-ID carries it -- ``test_chamber_003516.ge1`` -- and a *shared* base
    called ``…/test_chamber_003516.ge1/`` holding ge2, ge3 and ge4 names the
    whole set after one panel. Single-detector output keeps the tag, which
    is right there: it has exactly one detector and the tag says which.
    """
    parts = bc_path_parts(data_path, expid_fallback=expid_fallback)
    if parts is None:
        return None
    froot = parts.froot
    if parts.detector and froot.lower().endswith("." + parts.detector.lower()):
        froot = froot[: -(len(parts.detector) + 1)]
    if parts.positional:
        return parts.root / froot
    return (parts.root if parts.root.name == froot
            else parts.root / froot)


def suggest_correction_output_dir(data_path, *, expid_fallback: str = "") -> Optional[Path]:
    """``<outroot>/<expid>_bc/<froot>/<detector>`` — the PARENT Batch
    Correction writes its per-op folders into, giving e.g.
    ``…/<detector>/dark_subtracted_mean/``.

    Deliberately :func:`suggest_integration_output_dir` itself rather than a
    parse of its own, so the two tabs can never disagree about where a
    froot's analysis output lives — Batch Correction only adds a leaf (see
    :func:`correction_subdir`). The ``<detector>`` segment is kept (the
    shorthand for this convention usually omits it): a scan recorded on two
    detectors would otherwise write both reductions into one folder, where
    the filenames alone do not distinguish them.
    """
    return suggest_integration_output_dir(data_path, expid_fallback=expid_fallback)


def suggest_working_dir(data_path, *, expid_fallback: str = "") -> Optional[Path]:
    """The bare ``<expid>_bc`` analysis root to use as a working directory.

    Deliberately *not* :func:`suggest_integration_output_dir` minus its tail,
    for two reasons:

    **An existing ``_bc`` directory wins over the positional read.** The
    positional derivation assumes the full 4-deep mpe_wf layout, and quietly
    mislabels shallower ones. A real case: ``.../export/s20a/
    PUP_AML_stubbins_sep26_bc/run.h5`` sits *directly inside* an already-``_bc``
    directory, only three levels below the mount, so the positional read calls
    ``export`` the expid and proposes ``/net/s20iddata/export_bc`` — a sibling
    of the mount root that nobody can create. Recognising the ``_bc`` directory
    the data already lives in gets the right answer, and it is the directory
    that demonstrably *is* writable, since the data is sitting in it. Batch
    keeps the positional read (it is correct for the layout Batch is pointed
    at); only the working-directory ladder puts ``_bc`` recognition first.

    **No fallback into the data tree.** Where Batch falls back to
    ``<source folder>/<froot>``, this returns None. A working directory inside
    the raw-data tree is precisely the littering the working directory exists
    to stop; an empty field that makes the user choose is the better answer.
    """
    if not data_path:
        return None
    parent = Path(data_path).parent
    # 1. the data's own folder, 2. the nearest ancestor above it.
    for cand in (parent, *parent.parents):
        if cand.name.endswith("_bc"):
            return cand
    parts = bc_path_parts(data_path, expid_fallback=expid_fallback)
    if parts is None or not parts.has_bc:
        return None
    return parts.root


def check_output_dir_writable(path: str | Path) -> Optional[str]:
    """None if `path` can be written to (created if missing), else a
    human-readable reason it can't.

    Batch Integrate's suggested output folder (``<expid>_bc/<froot>/
    <detector>``) is not something this GUI creates ahead of time in
    production — it may already exist, made by someone else, with no
    write access for whoever is actually running the batch (this differs
    from `park_may26_bc`, which was deliberately made world-writable for
    testing and is not representative). `mkdir(parents=True)` only
    succeeds if the nearest *existing* ancestor is writable, so that's
    what gets checked when `path` itself doesn't exist yet."""
    p = Path(path)
    check = p
    while not check.exists():
        parent = check.parent
        if parent == check:
            break
        check = parent
    if not _os.access(check, _os.W_OK):
        user = _os.environ.get("USER") or _os.environ.get("LOGNAME") or "you"
        if check == p:
            return (f"'{p}' exists but isn't writable by {user}. Pick a "
                     "different output folder, or ask whoever owns it to "
                     "grant write access.")
        return (f"Can't create '{p}' — '{check}' isn't writable by {user}. "
                 "Pick a different output folder, or ask whoever owns it "
                 "to grant write access.")
    return None


SCRATCH_DIRNAME = ".midas_scratch"

_SESSION_SCRATCH: Optional[str] = None


def session_scratch_dir() -> str:
    """One ``mkdtemp`` per process, removed at exit.

    Where scratch goes when no working directory is set. Same contract as
    :func:`new_temp_h5_path`: the caller gets somewhere valid to write without
    having to special-case "nowhere", and the process cleans up after itself
    rather than leaving files in /tmp for someone else to find.
    """
    global _SESSION_SCRATCH
    if _SESSION_SCRATCH is None:
        import shutil
        _SESSION_SCRATCH = _tf.mkdtemp(prefix="midas_gui_scratch_")
        _atexit.register(shutil.rmtree, _SESSION_SCRATCH, ignore_errors=True)
    return _SESSION_SCRATCH


def scratch_dir(work_dir, *parts: str, create: bool = True) -> Path:
    """``<work_dir>/.midas_scratch/<parts…>`` — where intermediate files go.

    Everything under ``.midas_scratch`` is machine-generated and re-derivable:
    the user can delete the whole folder at any time without losing anything
    they asked for, because every deliberate save on the Calibrate tab goes
    through a file dialog elsewhere. The GUI never deletes it on their behalf
    — an analysis directory that empties itself between sessions is its own
    kind of surprise.

    Keeping it to one subfolder, under a directory the user named, is the
    whole point: the alternative that shipped before this was scratch landing
    either next to the raw data or in an unfindable ``mkstemp`` file.

    `parts` exist because the files written here (``residual_corr.bin``,
    the backend's ``calibration.json``, ``panel_shifts.txt``) are all
    *generically named*, so two fits sharing a working directory would
    overwrite each other and four concurrent Hydra panels would race. Callers
    pass a per-run (and per-panel) leaf to keep them apart.

    With no `work_dir`, falls back to :func:`session_scratch_dir` so callers
    never have to handle "nowhere to write" themselves.

    Raises OSError carrying :func:`check_output_dir_writable`'s human-readable
    reason rather than writing somewhere the user can't — failing loudly beats
    a fit that runs and then silently records a file it never managed to save.
    """
    base = Path(session_scratch_dir()) if not work_dir \
        else Path(work_dir) / SCRATCH_DIRNAME
    d = base.joinpath(*parts) if parts else base
    if create:
        reason = check_output_dir_writable(d)
        if reason:
            raise OSError(reason)
        d.mkdir(parents=True, exist_ok=True)
    return d


def new_temp_h5_path(prefix: str = "midas_buffer_") -> str:
    """Fresh temp .h5 path, auto-deleted at process exit."""
    f = _tf.NamedTemporaryFile(prefix=prefix, suffix=".h5", delete=False)
    f.close()
    _atexit.register(lambda p=f.name: _os.path.exists(p) and _os.unlink(p))
    return f.name


def save_stack_h5(path: str, frames, dataset: str = "buffer/data") -> None:
    """Write a sequence of 2-D frames as one (N,H,W) float32 dataset in `path`."""
    import h5py
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=arr)


# ── Dark / bright / background field building ───────────────────────────────────

def list_h5_datasets(path: str | Path) -> list:
    """Return [(name, shape), …] for every ≥2-D dataset in an HDF5 file."""
    import h5py
    items: list = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            items.append((name, tuple(obj.shape)))

    with h5py.File(str(path), "r") as f:
        f.visititems(_visit)
    return items


# Name fragments that mark a 1-D dataset as the measured rotation angle,
# most specific first. Taken verbatim from mpe_wf_saxs_waxs
# (``gui_data_explorer.py``'s ``_OMEGA_HINTS``), which uses the same list to
# pre-select an omega stream, so the two GUIs rank the same file the same way.
_OMEGA_HINTS = ("/omegas", "samry", "omega")


def list_h5_1d_datasets(path: str | Path) -> list:
    """Return ``[(name, length), …]`` for every 1-D dataset in an HDF5 file,
    with omega-looking names first (see ``_OMEGA_HINTS``).

    The 1-D sibling of :func:`list_h5_datasets`, which lists the ≥2-D
    (image) datasets. This one populates the omega-channel picker in Batch
    Integrate's cake-parameters dialog: a rotation stage writes one angle per
    raw sub-frame into a flat array alongside the frames.

    The hint only ORDERS the list — nothing is auto-selected. A wrong channel
    silently relabels every frame's angle, so picking one stays the user's
    deliberate act; mpe_wf's own picker is likewise ``allow_none=True`` with a
    blank default.
    """
    import h5py
    items: list = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
            items.append((name, int(obj.shape[0])))

    with h5py.File(str(path), "r") as f:
        f.visititems(_visit)

    def _rank(item):
        low = "/" + item[0].lower()
        for i, hint in enumerate(_OMEGA_HINTS):
            if hint in low:
                return (i, item[0])
        return (len(_OMEGA_HINTS), item[0])

    return sorted(items, key=_rank)


_DARK_NAME_RE = re.compile(r'(^|[_.])dark([_.]|$)|_dark_(before|after)\b', re.IGNORECASE)


def is_dark_like_name(path) -> bool:
    """True when a filename looks like a DARK/background acquisition rather
    than sample data (``..._dark_before_009242.vrx.h5``,
    ``..._dark_after_009388.vrx.h5``).

    Beamline convention puts the dark frames in the SAME folder as the scan
    they bracket, so a "Full folder"/stem selection sweeps them up as if
    they were data — silently integrating two bogus frames and (worse)
    skewing an auto-detected file-number range outward at both ends. Ported
    from mpe_wf_saxs_waxs, which guards the same way when auto-populating
    its froot/file-number fields (see gui_bc_launcher.py's dark-file check
    and run_cakes.discover_all_froots's ``_dark_before``/``_dark_after``
    skip)."""
    return bool(_DARK_NAME_RE.search(Path(str(path)).name))


# Known image-file formats a folder load will pick up, in the fixed order
# their paths are concatenated in (see _collect_frame_paths) — grouped under
# one label per format so widgets.DataLoaderPanel's optional "Format:" filter
# can offer only the formats actually present in a given folder.
_FOLDER_FORMAT_GROUPS = {
    "TIFF": ("*.tif", "*.tiff"),
    "HDF5": ("*.h5", "*.hdf5"),
    "GE": ("*.ge*",),
    "CBF": ("*.cbf",),
    "EDF": ("*.edf",),
}


def _folder_format_groups(folder) -> dict:
    """Which of ``_FOLDER_FORMAT_GROUPS`` are present in ``folder``, each
    mapped to its sorted matching paths (as ``str``) — empty groups omitted.
    Used both by ``_collect_frame_paths`` (to filter) and by
    ``widgets.DataLoaderPanel`` (to populate its "Format:" combo with only
    the formats a given folder actually has)."""
    p = Path(folder)
    out = {}
    for label, patterns in _FOLDER_FORMAT_GROUPS.items():
        paths = []
        for ext in patterns:
            paths.extend(sorted(p.glob(ext)))
        if paths:
            out[label] = [str(x) for x in paths]
    return out


def _collect_frame_paths(raw, ext_group: Optional[str] = None) -> list:
    """Frames from a folder or a *.tif glob (sorted).  Mirrors tab_view logic.

    ``raw`` may also be an already-resolved ``list[str]`` of explicit paths
    (an arbitrary multi-file selection from ``dialogs.BrowseFilesDialog``,
    which has no single string/glob representation) — returned as-is.

    ``ext_group`` restricts a folder listing to one label from
    ``_FOLDER_FORMAT_GROUPS`` (e.g. ``"TIFF"``); ``None`` (the default) keeps
    the original behavior of merging every known format, in the same fixed
    per-format order as before."""
    if isinstance(raw, list):
        return raw
    import glob as _glob
    p = Path(raw)
    if p.is_dir():
        groups = _folder_format_groups(p)
        if ext_group and ext_group in groups:
            return groups[ext_group]
        out = []
        for paths in groups.values():
            out.extend(paths)
        return out
    # recursive=True only changes behavior when the pattern contains "**"
    # (the filestem-filter case, widgets.DataLoaderPanel._raw_source) — a
    # plain glob without it is unaffected.
    return sorted(_glob.glob(raw, recursive=True))


# ── Pre-integrated 1-D profile files (Batch Integrate's own output formats) ──

# Which axis a given extension's first column holds, matching workers.py's
# write_profile dispatch (write_csv → R_px, write_xye/write_fxye → 2θ in
# degrees, write_dat → Q in inverse angstrom).
_PROFILE_EXT_KIND = {
    ".csv": "r_px",
    ".dat": "q_invA",
    ".xye": "two_theta_deg",
    ".fxye": "two_theta_deg",
}

PROFILE_FILE_FILTER = "Profile files (*.csv *.xye *.dat *.fxye);;All files (*)"


def profile_file_axis_kind(path) -> str:
    """Which physical axis ``path``'s first column holds, from its
    extension — ``"r_px"``, ``"two_theta_deg"``, or ``"q_invA"``. Unknown
    extensions default to ``"r_px"`` (the plain 2/3-column case)."""
    return _PROFILE_EXT_KIND.get(Path(path).suffix.lower(), "r_px")


def load_profile_file(path: str):
    """Load a pre-integrated 1-D profile file → ``(x, y, sigma_or_None)``.

    Tolerates comma- or whitespace-separated 2- or 3-column data with a
    leading comment/header line (``#`` comments skipped; a non-numeric first
    row is too) — round-trips ``workers.write_csv``'s comma format,
    ``workers.write_dat``'s whitespace format, and ``workers.write_xye``'s
    space-delimited 3-column format unchanged. What ``x`` physically means
    (R_px / 2θ / Q) is not this function's concern — see
    ``profile_file_axis_kind``."""
    with open(path, "r") as fh:
        first = ""
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                first = s
                break
    delim = "," if "," in first else None
    try:
        float(first.split(delim)[0] if delim else first.split()[0])
        skip = 0
    except ValueError:
        skip = 1
    arr = np.loadtxt(path, delimiter=delim, comments="#", skiprows=skip)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise ValueError(f"Profile file needs >=2 columns (x, y); got {arr.shape[1]}.")
    x = arr[:, 0].astype(np.float64)
    y = arr[:, 1].astype(np.float64)
    sigma = arr[:, 2].astype(np.float64) if arr.shape[1] >= 3 else None
    return x, y, sigma


def native_axis_to_r_px(x, x_kind: str, lsd_um: float, px_um: float,
                        wavelength_A: Optional[float] = None):
    """Convert a profile file's native x-axis back into detector-pixel
    radius, the algebraic inverse of ``widgets.ProfileViewer._r_to_x`` — lets
    a loaded 2θ/Q-native file be plotted through the normal ``set_profile``
    path (full R/2θ/Q toggle, ring overlay) once a calibration supplies the
    geometry needed to place it there."""
    x = np.asarray(x, dtype=np.float64)
    if x_kind == "r_px":
        return x
    if x_kind == "two_theta_deg":
        two_theta = np.radians(x)
    elif x_kind == "q_invA":
        if not wavelength_A:
            raise ValueError("q_invA -> r_px conversion needs a wavelength.")
        two_theta = 2.0 * np.arcsin(np.clip(x * wavelength_A / (4 * math.pi), -1.0, 1.0))
    else:
        raise ValueError(f"Unknown profile axis kind: {x_kind!r}")
    return (lsd_um / px_um) * np.tan(two_theta)


def radial_axes_from_r_px(r_px, lsd_um: float, px_um: float,
                          wavelength_A: Optional[float] = None) -> dict:
    """Forward-convert a detector-pixel radial axis into 2θ/d/Q — the
    algebraic inverse of :func:`native_axis_to_r_px`, and the same formulas
    as ``widgets._convert_radial``'s R-native branch, so the numbers agree
    with what the rest of the GUI already calls "2θ"/"d"/"Q" for the same
    geometry. Used by ``cake_hdf5.write_cake_h5`` to store companion axes
    alongside ``r_px``.

    Always returns ``"two_theta_deg"``; ``"d_angstrom"``/``"q_invA"`` are
    only added when ``wavelength_A`` is given (both undefined without one).
    ``d_angstrom`` is ``+inf`` at ``r_px == 0`` (2θ = 0), matching
    ``widgets._UnitAxis``'s convention for the beam axis."""
    r_px = np.asarray(r_px, dtype=np.float64)
    two_theta = np.arctan(r_px * px_um / lsd_um)
    out = {"two_theta_deg": np.degrees(two_theta)}
    if wavelength_A:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["d_angstrom"] = wavelength_A / (2.0 * np.sin(two_theta / 2.0))
        out["q_invA"] = 4.0 * math.pi * np.sin(two_theta / 2.0) / wavelength_A
    return out


def display_text_for_paths(paths: list) -> str:
    """Text a Data/Dark/Bright/Background/Stack field should show after an
    explicit multi-file Browse… pick (``dialogs.BrowseFilesDialog`` "Multiple
    files"/"Files sharing a name stem" modes): the one file's own path if
    exactly one was picked, else the shared parent folder of every picked
    file — never a bare "N files selected" count, so the field always shows
    a real filesystem path like every other selection mode does."""
    if len(paths) == 1:
        return paths[0]
    import os as _os
    parents = {str(Path(p).parent) for p in paths}
    return next(iter(parents)) if len(parents) == 1 else _os.path.commonpath(list(parents))


def source_kind(path) -> str:
    """Classify a data-source path as "folder" (dir or glob), "hdf5", or
    "file" — the ``kind`` argument ``average_field``/``FieldAverageWorker``
    need to know how to read it. An explicit ``list[str]`` of paths (an
    arbitrary multi-file selection) is treated as "folder" too — both are
    "a set of single-frame files to take the mean of / stack over"."""
    if isinstance(path, list):
        return "folder"
    if Path(path).is_dir() or any(c in path for c in "*?"):
        return "folder"
    if is_h5(path):
        return "hdf5"
    return "file"


def average_field(kind: str, path: str, dataset: str = "exchange/data",
                  idx_start: int = 0, idx_end: int = -1) -> np.ndarray:
    """Build a single 2-D field as the mean over an index range.

    kind:
      "file"   — a single image file; if it holds a 3-D stack, mean [start..end].
      "folder" — a folder or *.tif glob; mean frames [start..end] across files.
      "hdf5"   — mean dataset[start..end+1] if 3-D, else the 2-D dataset.

    idx_end = -1 means "through the last frame" (inclusive).
    """
    def _slice(n: int) -> tuple:
        s = max(0, int(idx_start))
        e = n - 1 if idx_end is None or int(idx_end) < 0 else min(int(idx_end), n - 1)
        return s, e

    if kind == "hdf5":
        import h5py
        with h5py.File(str(path), "r") as f:
            dset = f[dataset]
            if dset.ndim >= 3:
                s, e = _slice(dset.shape[0])
                return np.asarray(dset[s:e + 1], dtype=np.float64).mean(axis=0)
            return np.asarray(dset[...], dtype=np.float64)

    if kind == "folder":
        paths = _collect_frame_paths(path)
        if not paths:
            raise ValueError(f"No frames found for '{path}'")
        s, e = _slice(len(paths))
        acc, n = None, 0
        for p in paths[s:e + 1]:
            a = _load_image(p).astype(np.float64)
            a = a[0] if a.ndim == 3 else a       # guard multi-page file in a folder
            acc = a if acc is None else acc + a
            n += 1
        return acc / max(n, 1)

    # single file
    arr = _load_image(path).astype(np.float64)
    if arr.ndim >= 3:
        s, e = _slice(arr.shape[0])
        return arr[s:e + 1].mean(axis=0)
    return arr


def read_frame_range(kind: str, path: str, dataset: str = "exchange/data",
                     idx_start: int = 0, idx_end: int = -1) -> np.ndarray:
    """Read the raw (N, Y, X) frame stack over an index range, unaveraged.

    Same ``kind``/index-range semantics as :func:`average_field` (which this
    mirrors), for callers that need the individual frames rather than their
    mean — e.g. a dead/hot-pixel mask built from per-pixel variance across a
    dark stack.
    """
    def _slice(n: int) -> tuple:
        s = max(0, int(idx_start))
        e = n - 1 if idx_end is None or int(idx_end) < 0 else min(int(idx_end), n - 1)
        return s, e

    if kind == "hdf5":
        import h5py
        with h5py.File(str(path), "r") as f:
            dset = f[dataset]
            if dset.ndim >= 3:
                s, e = _slice(dset.shape[0])
                return np.asarray(dset[s:e + 1], dtype=np.float32)
            return np.asarray(dset[...], dtype=np.float32)[None, ...]

    if kind == "folder":
        paths = _collect_frame_paths(path)
        if not paths:
            raise ValueError(f"No frames found for '{path}'")
        s, e = _slice(len(paths))
        frames = []
        for p in paths[s:e + 1]:
            a = _load_image(p).astype(np.float32)
            a = a[0] if a.ndim == 3 else a       # guard multi-page file in a folder
            frames.append(a)
        return np.stack(frames, axis=0)

    # single file
    arr = _load_image(path).astype(np.float32)
    if arr.ndim >= 3:
        s, e = _slice(arr.shape[0])
        return arr[s:e + 1]
    return arr[None, ...]


def apply_field_corrections(img: np.ndarray, *, dark=None, bright=None,
                            bright_mode: str = "divide", background=None,
                            clip_negative: bool = True) -> np.ndarray:
    """Apply dark subtraction, bright (flat-field divide OR subtract) and background.

    Order: (img − dark) → bright → (− background) → clip≥0.  For divide mode the
    flat field is dark-corrected too: out / (bright − dark) × mean(bright − dark).
    Returns float64.  Any field may be None.

    A field whose shape doesn't match ``img`` (typically a dark/bright/background
    left over from reusing a session saved against a different detector) is
    skipped rather than raising — mirrors ``MaskSelector.composite_mask()``'s
    "skip + warn" handling of a mismatched mask source.
    """
    import warnings
    out = np.asarray(img, dtype=np.float64)

    def _checked(field, label):
        if field is None:
            return None
        arr = np.asarray(field, dtype=np.float64)
        if arr.shape != out.shape:
            warnings.warn(
                f"apply_field_corrections: {label} shape {arr.shape} != "
                f"image shape {out.shape} — skipped", RuntimeWarning, stacklevel=2)
            return None
        return arr

    d = _checked(dark, "dark")
    if d is not None:
        out = out - d
    b = _checked(bright, "bright")
    if b is not None:
        if d is not None:
            b = b - d
        if bright_mode == "subtract":
            out = out - b
        else:  # flat-field divide, rescaled to preserve counts
            b = np.clip(b, 1e-9, None)
            out = out / b * float(np.mean(b))
    g = _checked(background, "background")
    if g is not None:
        out = out - g
    if clip_negative:
        out = np.clip(out, 0.0, None)
    return out


# ── Ring prediction (calibrant → ring radii in px) ──────────────────────────────

def _predict_ring_radii(result) -> list:
    """Predicted ring radii (px) for the result's calibrant geometry, out to
    the farthest corner the detector frame actually reaches — not an
    arbitrary wavelength-side cutoff. Falls back to 30° only when the
    detector's pixel dimensions aren't known (e.g. a result built before an
    image was loaded)."""
    max_2theta = 30.0
    ny = getattr(result, "NrPixelsY", 0) or 0
    nz = getattr(result, "NrPixelsZ", 0) or 0
    if ny > 0 and nz > 0:
        try:
            max_2theta = max_two_theta_deg(
                result.BC_y, result.BC_z, ny, nz, result.Lsd,
                result.pxY, getattr(result, "pxZ", None))
        except Exception:
            max_2theta = 30.0
    d_list = getattr(result, "_d_list", None)
    if d_list:
        try:
            rings = simulate_rings_from_dspacings(
                d_list, result.wavelength_A, result.Lsd, result.pxY,
                max_2theta_deg=max_2theta)
            return sorted({round(r["radius_px"], 3) for r in rings})
        except Exception:
            return []
    try:
        from midas_hkls import SpaceGroup, Lattice, generate_hkls
        cal = getattr(result, "_calibrant_name", "CeO2")
        lp  = _LATT.get(cal, _LATT["CeO2"])
        lat = Lattice(a=lp["a"], b=lp["b"], c=lp["c"],
                      alpha=lp["alpha"], beta=lp["beta"], gamma=lp["gamma"])
        refs = generate_hkls(SpaceGroup.from_number(lp["sg"]), lat,
                             wavelength_A=result.wavelength_A, two_theta_max_deg=max_2theta)
        return sorted({round(result.Lsd * math.tan(math.radians(r.two_theta_deg))
                             / result.pxY, 3) for r in refs})
    except Exception:
        return []


# ── Spec building (always via spec_from_calibration_result — RhoD in µm) ─────────

def simulate_rings(lattice: dict, sg: int, wavelength_A: float, lsd_um: float,
                   px_um: float, max_2theta_deg: float = 30.0) -> list:
    """Simulate Debye-Scherrer ring radii (px) for an arbitrary lattice.

    lattice: dict with a,b,c,alpha,beta,gamma.  Returns a list of dicts
    {radius_px, two_theta_deg, hkl, d_spacing} — one entry per distinct ring,
    labelled by the lowest-index reflection contributing to it.
    """
    from midas_hkls import SpaceGroup, Lattice, generate_hkls
    lat = Lattice(a=lattice["a"], b=lattice["b"], c=lattice["c"],
                  alpha=lattice["alpha"], beta=lattice["beta"], gamma=lattice["gamma"])
    refs = generate_hkls(SpaceGroup.from_number(int(sg)), lat,
                         wavelength_A=wavelength_A, two_theta_max_deg=max_2theta_deg)
    by_ring = {}
    for r in refs:
        rn = getattr(r, "ring_nr", None)
        key = rn if rn is not None else round(r.two_theta_deg, 4)
        if key not in by_ring:
            by_ring[key] = r
    out = []
    for r in by_ring.values():
        radius_px = lsd_um * math.tan(math.radians(r.two_theta_deg)) / px_um
        out.append({
            "radius_px": radius_px,
            "two_theta_deg": float(r.two_theta_deg),
            "hkl": (int(r.h), int(r.k), int(r.l)),
            "d_spacing": float(r.d_spacing),
        })
    out.sort(key=lambda d: d["radius_px"])
    return out


def simulate_rings_from_dspacings(d_list, wavelength_A: float, lsd_um: float,
                                  px_um: float, max_2theta_deg: float = 30.0) -> list:
    """Simulate Debye-Scherrer ring radii (px) for an explicit list of
    d-spacings (Angstrom) — for non-crystalline standards (e.g. silver
    behenate) that have no space group to derive rings from.

    Returns the same per-ring dict shape as :func:`simulate_rings`
    (radius_px, two_theta_deg, hkl, d_spacing) plus ``order`` (the 1-based
    index into the sorted, largest-d-first list); ``hkl`` is always None.
    """
    out = []
    for i, d in enumerate(sorted(d_list, reverse=True), start=1):
        if d <= 0:
            continue
        s = wavelength_A / (2.0 * d)
        if s > 1.0:
            continue  # this order isn't reachable at this wavelength
        two_theta_deg = 2.0 * math.degrees(math.asin(s))
        if two_theta_deg > max_2theta_deg:
            continue
        radius_px = lsd_um * math.tan(math.radians(two_theta_deg)) / px_um
        out.append({
            "radius_px": radius_px,
            "two_theta_deg": two_theta_deg,
            "hkl": None,
            "order": i,
            "d_spacing": float(d),
        })
    out.sort(key=lambda r: r["radius_px"])
    return out


def parse_dspacing_text(text: str) -> list:
    """Parse a comma/whitespace-separated d-spacing list (Angstrom) from a
    text field, dropping blank/unparsable/non-positive tokens. Shared by the
    Ring Simulation material dialog and the Calibrate tab's manual
    d-spacing ring-picking fit."""
    out = []
    for tok in re.split(r"[,\s]+", text.strip()):
        if not tok:
            continue
        try:
            d = float(tok)
        except ValueError:
            continue
        if d > 0:
            out.append(d)
    return out


def fit_circle_algebraic(pts: list) -> Optional[tuple]:
    """Algebraic least-squares circle fit through ``pts`` (x, y). Returns
    ``(cx, cy, r)`` or ``None`` if the points are too few/collinear."""
    arr = np.array(pts, dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = -(x ** 2 + y ** 2)
    try:
        res, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    D, E, F = res
    cx, cy = -D / 2, -E / 2
    r2 = cx ** 2 + cy ** 2 - F
    return (cx, cy, math.sqrt(r2)) if r2 > 0 else None


def _auto_seed_from_picks(picks, wavelength_A: float, pxY_um: float, pxZ_um: float):
    """Rough (Lsd, BC_y, BC_z) seed from picked (Y_px, Z_px, d_spacing) points,
    used when the caller doesn't supply one for :func:`fit_geometry_from_ring_picks`.
    Groups points by exact d-spacing (one group per picked ring), algebraically
    circle-fits each group, and combines the per-ring centers/radii into a
    single seed. Falls back to the image center / a generic 1 m Lsd if no
    group has enough points (>=3) to circle-fit."""
    from collections import defaultdict
    groups = defaultdict(list)
    for y, z, d in picks:
        groups[d].append((y, z))
    centers, lsds = [], []
    for d, pts in groups.items():
        if len(pts) < 3:
            continue
        fit = fit_circle_algebraic(pts)
        if fit is None:
            continue
        cy, cz, r = fit
        centers.append((cy, cz))
        s = wavelength_A / (2.0 * d)
        if 0 < s <= 1.0:
            two_theta_deg = 2.0 * math.degrees(math.asin(s))
            if two_theta_deg > 0:
                lsds.append(r * pxY_um / math.tan(math.radians(two_theta_deg)))
    if not centers:
        ys = [p[0] for p in picks]; zs = [p[1] for p in picks]
        bc_y = sum(ys) / len(ys) if ys else 0.0
        bc_z = sum(zs) / len(zs) if zs else 0.0
        return (1.0e6, bc_y, bc_z, "fallback")
    bc_y = float(np.median([c[0] for c in centers]))
    bc_z = float(np.median([c[1] for c in centers]))
    lsd = float(np.median(lsds)) if lsds else 1.0e6
    return (lsd, bc_y, bc_z, "ok")


def fit_geometry_from_ring_picks(picks, wavelength_A: float, pxY_um: float,
                                 pxZ_um: float, seed=None,
                                 tilt_seed=(0.0, 0.0, 0.0), refine=None,
                                 bounds=None) -> dict:
    """Fit detector geometry from user-picked ring points and their known
    d-spacings, bypassing any crystallographic calibrant backend entirely.
    ``picks`` is an iterable of (Y_px, Z_px, d_spacing_A) triples; points on
    the same ring share the same d-spacing value.

    ``refine`` is a dict of booleans selecting which parameters float —
    the same shape ``CalibrationTab._refine_flags()`` produces (keys
    ``"Lsd"``, ``"BC"`` (jointly gates BC_y/BC_z), ``"tx"``, ``"ty"``,
    ``"tz"``, ``"Wavelength"``; any other keys, e.g. ``"Distortion"``, are
    ignored). ``refine=None`` reproduces the historical behavior: only
    Lsd/BC free, tilt fixed at ``tilt_seed`` (default 0) and wavelength
    fixed at ``wavelength_A``.

    ``bounds`` optionally restricts free parameters to a box: a dict keyed by
    *slot* name (``"Lsd"``, ``"BC_y"``, ``"BC_z"``, ``"tx"``, ``"ty"``,
    ``"tz"``, ``"wavelength_A"``) mapping to a ``(lo, hi)`` pair in fit units
    (µm / px / deg / Å); a missing or ``None`` entry means ±inf. Because
    scipy's Levenberg-Marquardt implementation rejects bounds outright, the
    solver is chosen accordingly: ``"lm"`` when every bound is infinite (so
    the unbounded path stays exactly as it has always been) and ``"trf"``
    as soon as any bound is finite. A seed outside its box is clamped into
    it rather than raising, and reported in ``clamped``.

    For each picked point, the *observed* 2theta implied by a trial geometry
    is computed via :func:`_pixel_to_two_theta_deg` — the closed-form
    inverse of the tilted forward-projection used by
    :func:`tilted_ring_xy`/``_draw_corrected_rings`` (reduces exactly to
    ``atan2(r_px, Lsd)`` when tilt is 0). The residual against the
    *expected* 2theta from Bragg's law (``2*asin(wavelength/(2*d))``) is
    minimized over whichever parameters ``refine`` marks free.

    Returns a dict with ``Lsd``, ``BC_y``, ``BC_z`` (µm/px), ``tx``, ``ty``,
    ``tz`` (deg), ``wavelength_A`` (Å), ``residual_deg_rms``, ``success``,
    ``message``, ``seed_quality`` ("ok"/"fallback"/"given"), ``n_free``,
    plus the identifiability report described in
    :func:`_fit_parameter_sigma`: ``sigma`` (per-slot 1-sigma estimate; 0.0
    for held-fixed parameters, ``inf`` when the data cannot constrain one),
    ``at_limit`` (names resting on an active bound, whose ``sigma`` is not
    meaningful), ``clamped``, and ``method``.

    The ``sigma`` report is the point of this function for small-2theta
    calibrants: with only a short ring arc on the detector, Lsd/BC are
    strongly correlated and tilt is effectively unconstrained, so the fit
    can converge happily onto noise. ``sigma`` is what makes that visible
    instead of silent — see ``CalibrationTab`` and ``ManualDspacingCalibWorker``.
    """
    from scipy.optimize import least_squares
    picks = list(picks)
    pts = np.array([(p[0], p[1]) for p in picks], dtype=np.float64)
    d = np.array([p[2] for p in picks], dtype=np.float64)

    if seed is None:
        lsd0, bcy0, bcz0, seed_quality = _auto_seed_from_picks(
            picks, wavelength_A, pxY_um, pxZ_um)
    else:
        lsd0, bcy0, bcz0 = seed
        seed_quality = "given"
    tx0, ty0, tz0 = tilt_seed

    refine = refine or {}
    ref_lsd = refine.get("Lsd", True)
    ref_bc = refine.get("BC", True)
    ref_tx = refine.get("tx", False)
    ref_ty = refine.get("ty", False)
    ref_tz = refine.get("tz", False)
    ref_wl = refine.get("Wavelength", False)

    # Ordered parameter slots: (name, seed value, is free to refine).
    slots = [("Lsd", lsd0, ref_lsd), ("BC_y", bcy0, ref_bc), ("BC_z", bcz0, ref_bc),
             ("tx", tx0, ref_tx), ("ty", ty0, ref_ty), ("tz", tz0, ref_tz),
             ("wavelength_A", wavelength_A, ref_wl)]
    free_idx = [i for i, s in enumerate(slots) if s[2]]
    fixed = {name: val for name, val, is_free in slots if not is_free}

    def unpack(p):
        vals = dict(fixed)
        for i, v in zip(free_idx, p):
            vals[slots[i][0]] = v
        return vals

    def resid(p):
        v = unpack(p)
        s = np.clip(v["wavelength_A"] / (2.0 * d), -1.0, 1.0)
        two_theta_calc = 2.0 * np.degrees(np.arcsin(s))
        two_theta_obs = _pixel_to_two_theta_deg(
            pts[:, 0], pts[:, 1], v["Lsd"], v["BC_y"], v["BC_z"],
            v["tx"], v["ty"], v["tz"], pxY_um, pxZ_um)
        return two_theta_obs - two_theta_calc

    p0 = [slots[i][1] for i in free_idx]
    free_names = [slots[i][0] for i in free_idx]
    bounds = bounds or {}
    lo = np.array([(bounds.get(n) or (-np.inf, np.inf))[0] for n in free_names], dtype=float)
    hi = np.array([(bounds.get(n) or (-np.inf, np.inf))[1] for n in free_names], dtype=float)
    bounded = bool(len(p0)) and bool(np.isfinite(lo).any() or np.isfinite(hi).any())

    sigma = {name: 0.0 for name, _, _ in slots}
    at_limit: set = set()
    clamped: set = set()
    method = "lm"

    if not p0:
        # Nothing selected to refine — report the seed geometry's own residual.
        vals, success, message, fun = unpack([]), True, "nothing to refine (all parameters fixed)", resid([])
        jac = None
    else:
        if bounded:
            # trf raises if x0 is outside the box; nudge strictly inside instead.
            method = "trf"
            x0 = np.clip(np.asarray(p0, dtype=float), lo, hi)
            clamped = {n for n, a, b in zip(free_names, p0, x0) if a != b}
            sol = least_squares(resid, x0, method="trf", bounds=(lo, hi))
        else:
            sol = least_squares(resid, p0, method="lm")
        vals = unpack(list(sol.x))
        success, message, fun = bool(sol.success), str(sol.message), sol.fun
        jac = sol.jac
        if bounded:
            # A parameter that ran to the edge of its box isn't determined by
            # the data, so its covariance-based sigma would be meaningless.
            # scipy's own active_mask is the authoritative signal but is
            # xtol-relative to the *parameter*, which is far too strict at
            # Lsd's magnitude (~1e7 µm): trf routinely halts a few µm short of
            # a bound on ftol and reports it inactive. The scale that actually
            # matters to the user is the width of the window they specified,
            # so also treat "within 0.1% of the box width of an edge" as at
            # the limit.
            for name, x, a, b, act in zip(free_names, sol.x, lo, hi, sol.active_mask):
                span = (b - a) if (np.isfinite(a) and np.isfinite(b)) else np.inf
                tol = 1e-3 * span if np.isfinite(span) else 1e-6 * max(abs(x), 1.0)
                if act != 0 or (np.isfinite(a) and x - a <= tol) \
                        or (np.isfinite(b) and b - x <= tol):
                    at_limit.add(name)

    if jac is not None:
        sigma.update(_fit_parameter_sigma(jac, np.asarray(fun), free_names))
    for name in at_limit:
        sigma[name] = 0.0

    return {
        "Lsd": float(vals["Lsd"]), "BC_y": float(vals["BC_y"]), "BC_z": float(vals["BC_z"]),
        "tx": float(vals["tx"]), "ty": float(vals["ty"]), "tz": float(vals["tz"]),
        "wavelength_A": float(vals["wavelength_A"]),
        "residual_deg_rms": float(np.sqrt(np.mean(np.asarray(fun) ** 2))) if len(fun) else 0.0,
        "success": success, "message": message,
        "seed_quality": seed_quality, "n_free": len(p0),
        "sigma": sigma, "at_limit": at_limit, "clamped": clamped, "method": method,
    }


def result_refined_tx(result) -> bool:
    """Whether this fit actually refined ``tx``, and so whether ``result.tx``
    carries information worth feeding back into the seed.

    Almost always ``False``. ``tx`` is a panel's installation azimuth about
    the beam, not an alignment tilt, and every powder pipeline freezes it on
    purpose: it reaches a ring radius only through the azimuthal distortion
    harmonics, so ``(tx, phi_k) -> (tx + d, phi_k + k*d)`` is an exact gauge
    orbit and refining it corrupts all six phases with no residual signature
    (``midas_calibrate_v2.forward.geometry``; enforced at
    ``compat/from_v1.py``, ``_add(s, "tx", v1.tx, refined=False)``).

    The one exception is the manual d-spacing fit, which refines ``tx`` as an
    ordinary free parameter when its Refine box is ticked
    (:func:`fit_geometry_from_ring_picks`). That fit reports a 1-sigma for
    each free parameter and only for those, so the presence of a ``tx`` sigma
    is the fit's own statement that it refined ``tx`` -- which beats asking
    the UI what was ticked, since a ticked box the solver dropped would lie.

    Written as a question about the result rather than about the pipeline so
    that a surface which later grows a d-spacing route gets the right answer
    without a second copy of this reasoning.
    """
    return "tx" in (getattr(result, "fit_sigma", None) or {})


def _fit_parameter_sigma(jac, resid_vec, free_names) -> dict:
    """1-sigma estimates for the free parameters of a ``least_squares`` solve,
    from the usual linearised covariance ``inv(J.T @ J) * s^2`` with
    ``s^2 = sum(r^2) / (m - n)``.

    Returns ``inf`` for every parameter when ``J.T @ J`` is singular — that is
    the honest answer for a genuinely degenerate fit (e.g. tilt at very small
    2theta, where the residual barely responds to it at all), and it is what
    callers threshold on to warn the user rather than reporting a converged
    value that is really just fitted noise.
    """
    m, n = len(resid_vec), len(free_names)
    if n == 0:
        return {}
    s2 = float(np.sum(np.asarray(resid_vec) ** 2)) / max(m - n, 1)
    try:
        cov = np.linalg.inv(np.asarray(jac).T @ np.asarray(jac)) * s2
        sig = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sig = np.full(n, np.inf)
    return {name: float(v) for name, v in zip(free_names, sig)}


def _tilt_matrix_np(tx_deg: float, ty_deg: float, tz_deg: float) -> np.ndarray:
    """Pure-numpy ``Rx(tx) @ Ry(ty) @ Rz(tz)`` (degrees), matching the rotation
    convention of ``midas_calibrate_v2.forward.geometry.build_tilt_matrix``."""
    tx, ty, tz = math.radians(tx_deg), math.radians(ty_deg), math.radians(tz_deg)
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rx @ Ry @ Rz


def _tilt_project_YZ(two_theta_deg: np.ndarray, eta_deg: np.ndarray,
                      tx: float, ty: float, tz: float, Lsd_um: float,
                      bc_y: float, bc_z: float, pxY_um: float, pxZ_um: float):
    """Shared core of :func:`tilted_ring_xy` (fixed 2θ, η sweep — a ring) and
    :func:`tilted_spoke_xy` (fixed η, 2θ sweep — a radial spoke).
    ``two_theta_deg``/``eta_deg`` broadcast against each other (one of them
    is typically a scalar-shaped array of the other's length)."""
    tt = np.radians(np.asarray(two_theta_deg, dtype=float))
    eta = np.radians(np.asarray(eta_deg, dtype=float))
    tt, eta = np.broadcast_arrays(tt, eta)
    u = np.stack([
        np.cos(tt),
        -np.sin(tt) * np.sin(eta),
        np.sin(tt) * np.cos(eta),
    ], axis=-1)                                      # (n, 3) unit ray directions
    TRs = _tilt_matrix_np(tx, ty, tz)
    n_hat = TRs[:, 0]
    denom = u @ n_hat
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    t = Lsd_um * TRs[0, 0] / denom
    off = t[..., None] * u - np.array([Lsd_um, 0.0, 0.0])
    Yc = off @ TRs[:, 1]
    Zc = off @ TRs[:, 2]
    Y_px = bc_y - Yc / pxY_um
    Z_px = bc_z + Zc / pxZ_um
    return Y_px, Z_px


def tilted_ring_xy(two_theta_deg: float, tx: float, ty: float, tz: float,
                    Lsd_um: float, bc_y: float, bc_z: float,
                    pxY_um: float, pxZ_um: float, n: int = 400,
                    eta_min: float = 0.0, eta_max: float = 360.0):
    """Forward-project a diffraction ring at ``two_theta_deg`` through the tilt
    geometry (tx/ty/tz, degrees) onto detector pixel coordinates.

    Returns ``(Y_px, Z_px)`` arrays of length ``n`` tracing the ring, with the
    last point equal to the first so the polyline a caller feeds straight
    into ``pg.PlotDataItem`` closes with no seam (``pyqtgraph`` never
    auto-closes a plain polyline back to its start). Reduces exactly to the
    plain circle ``bc + r*(sin η, cos η)`` when tx=ty=tz=0, since it inverts
    the same ray/tilt-plane geometry as
    ``midas_integrate_v2.forward.pixels.pixel_to_REta_from_spec``.

    ``eta_min``/``eta_max`` draw an ARC instead of the whole ring, for a
    caller depicting an integration region that does not span the full
    azimuth. The default is the full turn, so every existing caller — the
    calibration-ring overlays, which show where the rings ARE rather than
    which part is integrated — is unchanged.
    """
    eta = np.linspace(float(eta_min), float(eta_max), n, endpoint=True)
    return _tilt_project_YZ(np.full(n, two_theta_deg), eta, tx, ty, tz,
                             Lsd_um, bc_y, bc_z, pxY_um, pxZ_um)


def distortion_rho_d_um(NrPixelsY, NrPixelsZ, bc_y: float, bc_z: float,
                         pxY_um: float, pxZ_um: float) -> Optional[float]:
    """The distortion normalisation radius ρ_d, in µm.

    Reproduces ``spec_from_calibration_result``'s own definition exactly —
    the beam-centre-to-farthest-corner distance in px (measured to ``N-1``,
    the last pixel index) times the mean pixel pitch. It has to match: the
    harmonic basis is evaluated at ρ = R_µm / ρ_d, so a ρ_d off by even the
    pixel pitch rescales every term and the polynomial no longer describes
    the detector it was fitted on.

    ``None`` when the detector size is unknown (a result that never carried
    ``NrPixelsY``/``NrPixelsZ``), which callers read as "cannot evaluate
    distortion here" rather than substituting a guess.
    """
    try:
        NY, NZ = int(NrPixelsY or 0), int(NrPixelsZ or 0)
    except (TypeError, ValueError):
        return None
    if NY <= 0 or NZ <= 0:
        return None
    px_mean = 0.5 * (float(pxY_um) + float(pxZ_um or pxY_um))
    corner_px = math.hypot(max(bc_y, NY - 1 - bc_y), max(bc_z, NZ - 1 - bc_z))
    return corner_px * px_mean if corner_px > 0 else None


def ring_xy_corrected(two_theta_deg: float, tx: float, ty: float, tz: float,
                       Lsd_um: float, bc_y: float, bc_z: float,
                       pxY_um: float, pxZ_um: float, *,
                       distortion: Optional[dict] = None,
                       rho_d_um: Optional[float] = None, n: int = 400):
    """Where a ring at ``two_theta_deg`` actually lands on the detector, through
    the *full* forward model — tilt (:func:`tilted_ring_xy`) **and** the refined
    distortion harmonics.

    :func:`tilted_ring_xy` answers "where does the undistorted ray hit?". That
    is not where the ring is drawn on a detector whose calibration refined
    distortion coefficients, because the backend reports a pixel's radius as
    ``R_corrected = D(ρ, η) · R_projected`` (``midas_calibrate_v2.forward.
    geometry.pixel_to_REta``). A ring is the locus of pixels whose *corrected*
    radius equals the Bragg radius, so this inverts that relation instead of
    ignoring it.

    Per η, solve ``D(rad/ρ_d, η) · rad = Lsd·tan(2θ)`` for the projected radius
    ``rad`` by fixed-point iteration — D is within a few percent of 1 for any
    physical calibration, so the map is a strong contraction and this converges
    in a handful of passes. The recovered ``rad`` becomes a per-point effective
    2θ, which :func:`_tilt_project_YZ` then projects exactly as for a ring.

    ``distortion`` is a v2-named coefficient dict (``iso_R2``, ``a1``,
    ``phi1``, …) as carried by an ``AutoCalibrationResult``; the model itself
    comes from :mod:`midas_distortion`, the shared leaf ``midas_calibrate_v2``
    and ``midas_integrate_v2`` both evaluate, so there is one definition of it
    rather than a second copy here.

    With no coefficients — or no ``rho_d_um`` to normalise them against — the
    solve is skipped entirely and the result is bit-identical to
    :func:`tilted_ring_xy`, which is what every untilted/undistorted caller
    still gets.

    Note: the empirical ``residual_corr_map`` (a smooth per-pixel ΔR the
    backend adds *after* the harmonics, present only when residual-map
    refinement was run) is NOT applied here — it is a sub-pixel term and would
    need the map tensor in a redraw path. Callers that care should say so.
    """
    eta = np.linspace(0.0, 360.0, n, endpoint=True)
    p = _distortion_coeff_vector(distortion)
    R_target_um = float(Lsd_um) * math.tan(math.radians(float(two_theta_deg)))
    if p is None or not rho_d_um or rho_d_um <= 0 or R_target_um == 0.0:
        # Nothing to invert. Pass the requested 2θ straight through rather
        # than round-tripping it through tan/arctan, so this stays *exactly*
        # tilted_ring_xy — the reduction property its docstring promises.
        tt_eff = np.full(n, float(two_theta_deg))
    else:
        from midas_distortion import distortion_factor
        rad = np.full(n, R_target_um, dtype=float)
        for _ in range(_DISTORTION_SOLVE_ITERS):
            D = distortion_factor(rad / float(rho_d_um), eta, p)
            # A non-positive factor is not a physical distortion — bail out
            # and draw the undistorted ring rather than a folded-over curve.
            if not np.all(np.isfinite(D)) or np.any(D <= 0.0):
                rad = np.full(n, R_target_um, dtype=float)
                break
            nxt = R_target_um / D
            converged = np.max(np.abs(nxt - rad)) <= 1e-13 * abs(R_target_um)
            rad = nxt
            if converged:
                break
        tt_eff = np.degrees(np.arctan(rad / float(Lsd_um)))
    return _tilt_project_YZ(tt_eff, eta, tx, ty, tz,
                             Lsd_um, bc_y, bc_z, pxY_um, pxZ_um)


def ring_on_image_mask(ys, zs, img_shape):
    """Boolean mask of the ``(ys, zs)`` ring points that fall on the detector
    image whose array shape is ``img_shape`` (``(rows, cols, ...)``, i.e.
    ``(NrPixelsZ, NrPixelsY)``).

    Shared by every ring overlay (Data Viewer, Calibrate) so a predicted ring
    that swings outside the frame is confined to the pixels a caller can
    actually check it against, rather than drawn across the empty canvas
    beside the detector.
    """
    nz, ny = img_shape[:2]
    return (ys >= 0) & (ys <= ny - 1) & (zs >= 0) & (zs <= nz - 1)


#: Fixed-point passes in :func:`ring_xy_corrected`. D is a near-unity
#: multiplier, so this converges to float64 noise in ~5; 20 is headroom for a
#: badly-scaled coefficient set, and costs nothing on a 400-point ring.
_DISTORTION_SOLVE_ITERS = 20


def _distortion_coeff_vector(distortion: Optional[dict]):
    """A v2-ordered 15-vector for ``distortion``, or ``None`` when it holds
    nothing to apply (missing, empty, or every coefficient zero — the common
    case of a calibration that never refined distortion)."""
    if not distortion:
        return None
    try:
        from midas_distortion import v2_coeffs_from_named
        p = v2_coeffs_from_named(distortion)
    except Exception:
        return None
    return p if np.any(p != 0.0) else None


def tilted_spoke_xy(two_theta_lo_deg: float, two_theta_hi_deg: float, eta_deg: float,
                     tx: float, ty: float, tz: float, Lsd_um: float,
                     bc_y: float, bc_z: float, pxY_um: float, pxZ_um: float,
                     n: int = 24):
    """Forward-project the radial line at fixed ``eta_deg`` from
    ``two_theta_lo_deg`` to ``two_theta_hi_deg`` through the tilt geometry —
    the spoke counterpart of :func:`tilted_ring_xy`'s ring. A tilted
    detector's η-bin edges are not exactly straight lines through the beam
    centre (same reason its R-bin edges are not exactly circles), so this
    is sampled at ``n`` points rather than drawn as a single 2-point line."""
    tt = np.linspace(two_theta_lo_deg, two_theta_hi_deg, n)
    return _tilt_project_YZ(tt, np.full(n, eta_deg), tx, ty, tz,
                             Lsd_um, bc_y, bc_z, pxY_um, pxZ_um)


def _pixel_to_two_theta_deg(Y_px, Z_px, Lsd_um: float, bc_y: float, bc_z: float,
                             tx: float, ty: float, tz: float,
                             pxY_um: float, pxZ_um: float):
    """Closed-form inverse of :func:`_tilt_project_YZ`: the observed 2theta
    (degrees) for picked pixel(s) given a trial tilted geometry. Exact
    because the in-plane offset ``_tilt_project_YZ`` builds has zero
    component along the tilt plane's normal (``TRs[:,0]``) by construction,
    so it is fully recoverable from its two in-plane basis components
    (``TRs[:,1]``, ``TRs[:,2]``) alone — no iteration needed per point.
    Reduces exactly to ``degrees(atan2(r_px, Lsd))`` when tx=ty=tz=0."""
    Yc = (bc_y - np.asarray(Y_px, dtype=float)) * pxY_um
    Zc = (np.asarray(Z_px, dtype=float) - bc_z) * pxZ_um
    TRs = _tilt_matrix_np(tx, ty, tz)
    y_hat, z_hat = TRs[:, 1], TRs[:, 2]
    Px = Lsd_um + Yc * y_hat[0] + Zc * z_hat[0]
    Py = Yc * y_hat[1] + Zc * z_hat[1]
    Pz = Yc * y_hat[2] + Zc * z_hat[2]
    norm = np.sqrt(Px ** 2 + Py ** 2 + Pz ** 2)
    return np.degrees(np.arccos(np.clip(Px / norm, -1.0, 1.0)))


def pixel_eta_deg(Y_px, Z_px, bc_y: float, bc_z: float,
                  pxY_um: float, pxZ_um: float, tx: float = 0.0):
    """Azimuth η (degrees) of pixel(s) about the beam centre.

    Matches the backend's ``pixel_to_REta`` — ``atan2(-Yc, Zc)`` with
    ``Yc = (bc_y - Y) * pxY`` and ``Zc = (Z - bc_z) * pxZ`` — so η = 0 is
    straight **up** (+Z), not along +Y, and η increases towards +Y. The same
    convention the η spokes are drawn in (see the note in ``draw_bin_grid``);
    swapping the arguments puts every value 90° out.

    ``tx`` is the panel's installation roll about the beam. It is a pure
    offset here: MEASURED against ``midas_calibrate_v2.forward.geometry.
    pixel_to_REta`` at five pixels and three rolls, the backend's η is
    exactly this detector-plane angle plus tx, wrapped to (-180, 180].
    It defaults to 0 because η was flat for a long time and most detectors
    are unrolled, but a rolled panel binned without it is reported a full tx
    away from the cake axis it actually lands in — at 1-ID with tx=120° the
    status bar read η = -120.28° for a pixel the integration puts at
    -0.28°.

    Still deliberately flat in ty/tz: η names which spoke of the cake a pixel
    falls in, and the cake's η binning is this in-plane angle. 2θ is the
    quantity that must be tilt-corrected, and
    :func:`_pixel_to_two_theta_deg` is where that happens.
    """
    Yc = (bc_y - np.asarray(Y_px, dtype=float)) * pxY_um
    Zc = (np.asarray(Z_px, dtype=float) - bc_z) * pxZ_um
    eta = np.degrees(np.arctan2(-Yc, Zc))
    if tx:
        eta = (eta + float(tx) + 180.0) % 360.0 - 180.0
    return eta


def im_trans_map_point(col, row, shape, codes):
    """Map an image *point* through MIDAS transform codes — the point-wise
    counterpart of :func:`_apply_im_trans`, which only transforms arrays.

    ``shape`` is the ``(rows, cols)`` of the image the point is currently
    in; a transpose changes it, so it is tracked across the sequence.
    Returns ``(col, row)`` in the transformed frame.

    Exists for the Mask Builder, which displays the **raw** detector image
    on purpose while a calibration's beam centre lives in the transformed
    frame — so a hovered point has to be carried into that frame before any
    geometry is applied to it.
    """
    n_rows, n_cols = int(shape[0]), int(shape[1])
    col, row = int(col), int(row)
    for c in codes or ():
        if c == 1:                       # flipY — image[:, ::-1]
            col = n_cols - 1 - col
        elif c == 2:                     # flipZ — image[::-1, :]
            row = n_rows - 1 - row
        elif c == 3:                     # transpose — image.T
            col, row = row, col
            n_rows, n_cols = n_cols, n_rows
    return col, row


def map_point_xy(x: float, y: float, shape, codes):
    """Map a *continuous* (x=col, y=row) position through MIDAS transform
    codes, returning ``(x, y, n_cols, n_rows)`` — the position plus the frame
    extent it now lives in.

    The continuous sibling of :func:`im_trans_map_point`, which works in
    whole-pixel indices. Geometry drawn over an image (ROI corners, polygon
    vertices) is continuous, and a flip there is ``x -> W - x``, not
    ``W - 1 - x``: index ``c`` covers ``[c, c+1]``, and reversing the axis
    sends that span to ``[W-c-1, W-c]``. Using the index form on an edge
    coordinate shifts every shape half a pixel.
    """
    n_rows, n_cols = float(shape[0]), float(shape[1])
    for c in codes or ():
        if c == 1:                       # flipY — columns reverse
            x = n_cols - x
        elif c == 2:                     # flipZ — rows reverse
            y = n_rows - y
        elif c == 3:                     # transpose
            x, y = y, x
            n_rows, n_cols = n_cols, n_rows
    return x, y, n_cols, n_rows


def map_roi_state(pos, size, angle_deg: float, shape, codes):
    """Re-place a rotatable rectangular ROI so it covers the same detector
    pixels after ``codes`` are applied to the image under it.

    Takes and returns pyqtgraph's own ``(pos, size, angle)`` description: an
    origin corner, a local ``(w, h)`` extent, and a rotation in degrees
    measured as ``atan2`` of the local +x axis in parent coordinates (CCW,
    verified against ``RectROI.mapToParent``). Covers ``RectROI``,
    ``EllipseROI`` and ``CircleROI``; a ``PolyLineROI`` maps its vertices
    with :func:`map_point_xy` instead.

    Every MIDAS transform is a reflection (flipY, flipZ and transpose each
    have determinant −1), so an odd number of them flips the handedness of
    the ROI's local frame. Anchoring the result at the mapped origin corner
    would then put the rectangle on the wrong side of its own origin — the
    shape would look right only for an even number of codes. So the anchor
    moves to the mapped ``(0, h)`` corner in that case, which restores a
    right-handed frame with the same ``(w, h)``.

    Lengths are preserved (the maps are orthogonal), so ``size`` is returned
    unchanged even across a transpose.
    """
    w, h = float(size[0]), float(size[1])
    a = math.radians(float(angle_deg or 0.0))
    ca, sa = math.cos(a), math.sin(a)
    x0, y0 = float(pos[0]), float(pos[1])
    # Local +x is (cos a, sin a); local +y is that turned +90°.
    corners = {
        "c00": (x0, y0),
        "c10": (x0 + w * ca, y0 + w * sa),
        "c01": (x0 - h * sa, y0 + h * ca),
    }
    mapped = {k: map_point_xy(x, y, shape, codes)[:2]
              for k, (x, y) in corners.items()}
    e1 = (mapped["c10"][0] - mapped["c00"][0],
          mapped["c10"][1] - mapped["c00"][1])
    n_reflections = sum(1 for c in (codes or ()) if c in (1, 2, 3))
    anchor = mapped["c00"] if n_reflections % 2 == 0 else mapped["c01"]
    return (list(anchor), [w, h],
            math.degrees(math.atan2(e1[1], e1[0])))


def _fmt_g(v: float, nd: int) -> str:
    """Fixed-point with trailing zeros trimmed, or an em dash when the value
    is not finite (d-spacing diverges on the beam axis)."""
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") or "0"


def pixel_readout_text(col, row, geom: dict) -> str:
    """One status-bar clause naming 2θ / Q / d / η at pixel ``(col, row)``.

    ``geom`` is the geometry dict the tabs already build (``Lsd`` in µm,
    ``BC_y``/``BC_z`` in px, ``pxY``/optional ``pxZ`` in µm, optional
    ``tx``/``ty``/``tz`` in degrees, optional ``wavelength_A``). Returns
    ``""`` when the geometry cannot place the pixel at all (no Lsd, no beam
    centre or no pixel size), which is what keeps a viewer with no
    calibration rendering byte-identically to before this existed.

    Degrades per quantity rather than all-or-nothing: without a wavelength
    there is no Q or d, but 2θ and η are still well defined, so those are
    shown alone. d is reported as an em dash on the beam axis, where it
    diverges.

    Pure and cheap — no Qt, no IO. It is called on every hover (rate-limited
    to 60 Hz by the viewer's ``SignalProxy``), so it must stay that way.
    """
    if not geom:
        return ""
    lsd = geom.get("Lsd")
    bc_y, bc_z = geom.get("BC_y"), geom.get("BC_z")
    pxY = geom.get("pxY") or geom.get("px")
    if None in (lsd, bc_y, bc_z, pxY) or float(lsd) <= 0 or float(pxY) <= 0:
        return ""
    pxZ = geom.get("pxZ") or pxY
    tt = float(_pixel_to_two_theta_deg(
        col, row, float(lsd), float(bc_y), float(bc_z),
        float(geom.get("tx") or 0.0), float(geom.get("ty") or 0.0),
        float(geom.get("tz") or 0.0), float(pxY), float(pxZ)))
    eta = float(pixel_eta_deg(col, row, float(bc_y), float(bc_z),
                              float(pxY), float(pxZ),
                              float(geom.get("tx") or 0.0)))
    parts = [f"2θ = {_fmt_g(tt, 4)}°"]
    wl = geom.get("wavelength_A")
    if wl and float(wl) > 0:
        wl = float(wl)
        sin_th = math.sin(math.radians(tt) / 2.0)
        q = 4.0 * math.pi * sin_th / wl
        d = wl / (2.0 * sin_th) if sin_th > 0 else math.inf
        parts.append(f"Q = {_fmt_g(q, 4)} Å⁻¹")
        parts.append(f"d = {_fmt_g(d, 4)} Å")
    parts.append(f"η = {_fmt_g(eta, 2)}°")
    return "    ".join(parts)


def read_geometry(path: str | Path) -> dict:
    """Parse beam-centre / distance / pixel / wavelength from a calibration file.

    Supports three formats, auto-detected by extension then content:
      - MIDAS ``paramstest`` text  (``Lsd``, ``BC y z``, ``Wavelength``, ``px`` — µm/Å)
      - pyFAI ``.poni``            (SI units: Distance/Poni1/Poni2 in m, Wavelength in m)
      - calibration ``.json``      (as saved by the Calibrate tab)

    Returns a dict with keys ``wavelength_A``, ``Lsd_um``, ``px_um``, ``BC_y``,
    ``BC_z``, ``im_trans`` — any of which may be ``None``/``[]`` if the file
    does not carry it. ``im_trans`` is the ordered list of MIDAS
    ``ImTransOpt`` codes (1=flipY, 2=flipZ, 3=transpose).
    Note: PONI tilts (Rot1/2/3) are ignored — only the beam-centre projection is used.
    """
    p = Path(path)
    text = p.read_text()
    suf = p.suffix.lower()
    out = {"wavelength_A": None, "Lsd_um": None, "px_um": None,
           "BC_y": None, "BC_z": None, "im_trans": []}

    # ── calibration.json ──
    if suf == ".json" or text.lstrip().startswith("{"):
        import json
        d = json.loads(text)
        out["wavelength_A"] = d.get("wavelength_A")
        out["Lsd_um"] = d.get("Lsd")
        out["px_um"] = d.get("pxY") if d.get("pxY") is not None else d.get("px")
        out["BC_y"] = d.get("BC_y")
        out["BC_z"] = d.get("BC_z")
        out["im_trans"] = list(d.get("im_trans") or [])
        return out

    # ── pyFAI .poni ──
    if suf == ".poni" or "poni_version" in text or "Poni1" in text:
        vals, det_cfg = {}, {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if key == "detector_config":
                try:
                    import json
                    det_cfg = json.loads(val)
                except Exception:
                    det_cfg = {}
            else:
                vals[key] = val

        def _f(k):
            try:
                return float(vals[k])
            except (KeyError, ValueError):
                return None

        dist, poni1, poni2, wl_m = _f("distance"), _f("poni1"), _f("poni2"), _f("wavelength")
        px1 = det_cfg.get("pixel1"); px2 = det_cfg.get("pixel2")
        px1 = float(px1) if px1 is not None else None
        px2 = float(px2) if px2 is not None else px1
        out["Lsd_um"] = dist * 1e6 if dist is not None else None
        out["px_um"] = px1 * 1e6 if px1 is not None else None
        out["wavelength_A"] = wl_m * 1e10 if wl_m is not None else None
        # MIDAS convention (matches midas_integrate_v2.poni_to_bc):
        # BC_y = Poni1/pxY, BC_z = Poni2/pxZ.
        if poni1 is not None and px1:
            out["BC_y"] = poni1 / px1
        if poni2 is not None and px2:
            out["BC_z"] = poni2 / px2
        return out

    # ── MIDAS paramstest key-value text ──
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        try:
            if key == "Lsd" and len(parts) >= 2:
                out["Lsd_um"] = float(parts[1])
            elif key == "BC" and len(parts) >= 3:
                out["BC_y"], out["BC_z"] = float(parts[1]), float(parts[2])
            elif key == "Wavelength" and len(parts) >= 2:
                out["wavelength_A"] = float(parts[1])
            elif key in ("px", "pxY") and len(parts) >= 2:
                out["px_um"] = float(parts[1])
        except ValueError:
            continue
    out["im_trans"] = parse_im_trans(text)
    return out


def write_poni(geom: dict, path: str | Path) -> None:
    """Write a pyFAI ``.poni`` file from a normalized geometry dict (the same
    shape ``geometry_fields_from_file``/``get_geometry`` produce: ``Lsd``,
    ``BC_y``, ``BC_z``, ``pxY``, ``pxZ`` in µm/px, ``wavelength_A`` in Å).

    Inverts the MIDAS convention used by ``read_geometry``/
    ``geometry_fields_from_file`` (``BC_y = Poni1/pxY``, ``BC_z = Poni2/pxZ``).
    MIDAS ``tx``/``ty``/``tz`` tilts have no equivalent in PONI's Rot1-3
    convention and are **not** exported (Rot1/2/3 are written as 0.0) —
    matching the existing reader's documented limitation.
    """
    px_y_um = float(geom["pxY"])
    px_z_um = float(geom.get("pxZ") or geom["pxY"])
    px1_m, px2_m = px_y_um * 1e-6, px_z_um * 1e-6
    distance_m = float(geom["Lsd"]) * 1e-6
    poni1 = float(geom["BC_y"]) * px1_m
    poni2 = float(geom["BC_z"]) * px2_m
    wavelength_m = float(geom["wavelength_A"]) * 1e-10
    ny, nz = geom.get("NrPixelsY"), geom.get("NrPixelsZ")
    max_shape = f"[{int(nz)}, {int(ny)}]" if (ny and nz) else "null"
    lines = [
        "# MIDAS GUI — Data Viewer calibration export",
        "poni_version: 2.1",
        "Detector: Detector",
        f'Detector_config: {{"pixel1": {px1_m!r}, "pixel2": {px2_m!r}, "max_shape": {max_shape}}}',
        f"Distance: {distance_m!r}",
        f"Poni1: {poni1!r}",
        f"Poni2: {poni2!r}",
        "Rot1: 0.0",
        "Rot2: 0.0",
        "Rot3: 0.0",
        f"Wavelength: {wavelength_m!r}",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def _apply_panel_fields(spec, panel_layout: dict | None, panel_shifts_path: str | None):
    """Set the 7 panel fields ``midas_integrate_v2``'s ``IntegrationSpec`` needs
    (``NPanelsY/NPanelsZ/PanelSizeY/PanelSizeZ/PanelGapsY/PanelGapsZ/
    PanelShiftsFile``) from a ``panel_layout`` dict — no-op when absent.

    ``spec_from_calibration_result``/``spec_from_v1_params`` don't know about
    panels at all (same gap as ``TransOpt``, patched the same way just below),
    so every spec builder that wants panel corrections applied must call this
    itself. ``gap_y``/``gap_z`` may be a single int (uniform gap, the only
    shape the Calibrate tab's UI produces) or an explicit per-gap list (as
    parsed back from a saved paramstest's ``PanelGapsY``/``PanelGapsZ``) —
    mirrors ``PanelLayout.regular``'s own uniform-gap expansion.
    """
    if not panel_layout:
        return
    n_y, n_z = int(panel_layout["n_y"]), int(panel_layout["n_z"])

    def _gaps(g, n):
        if isinstance(g, (list, tuple)):
            return list(g)
        return [int(g)] * max(n - 1, 0)

    spec.NPanelsY = n_y
    spec.NPanelsZ = n_z
    spec.PanelSizeY = int(panel_layout["sy"])
    spec.PanelSizeZ = int(panel_layout["sz"])
    spec.PanelGapsY = _gaps(panel_layout.get("gap_y", 0), n_y)
    spec.PanelGapsZ = _gaps(panel_layout.get("gap_z", 0), n_z)
    spec.PanelShiftsFile = str(panel_shifts_path or "")


# ── R-range presets + polar (R, η) bin-grid overlay ─────────────────────────
# Used by the Integration tabs' Rmin/Rmax fields (Corner/Edge preset buttons)
# and their "Detector view" preview (Rmin/Rmax boundary + optional bin grid).

def rmax_corner_px(bc_y: float, bc_z: float, ny: int, nz: int) -> float:
    """Distance from the beam centre to the farthest detector CORNER — the
    same formula ``spec_from_calibration_result`` uses internally when
    ``RMax=None`` (auto), so leaving Rmax at this value always matches what
    a run actually integrates to."""
    return float(math.hypot(max(bc_y, ny - 1 - bc_y), max(bc_z, nz - 1 - bc_z)))


def rmax_edge_px(bc_y: float, bc_z: float, ny: int, nz: int) -> float:
    """Distance from the beam centre to the farthest detector EDGE (the
    largest perpendicular distance to any one of the 4 straight sides) —
    smaller than :func:`rmax_corner_px`, excludes the corner regions beyond
    that edge."""
    return float(max(bc_y, ny - 1 - bc_y, bc_z, nz - 1 - bc_z))


def radial_spline_values(r, radii, values):
    """Evaluate the free-form monotone-cubic (PCHIP) spline through the
    ``(radii, values)`` control points at ``r`` (scalar or array), flat
    beyond the first/last knot (clip then interpolate — no extrapolation
    overshoot). ``radii``/``values`` need not be pre-sorted; requires at
    least 2 knots after sorting (``PchipInterpolator``'s own minimum)."""
    from scipy.interpolate import PchipInterpolator
    radii = np.asarray(radii, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(radii)
    radii_s, values_s = radii[order], values[order]
    r_arr = np.asarray(r, dtype=float)
    interp = PchipInterpolator(radii_s, values_s, extrapolate=False)
    return interp(np.clip(r_arr, radii_s[0], radii_s[-1]))


def radial_spline_threshold_map(shape, bc_y: float, bc_z: float, radii, values):
    """Per-pixel threshold floor: the free-form spline through ``(radii,
    values)`` (see :func:`radial_spline_values`) evaluated at each pixel's
    radius from (bc_y, bc_z). ``shape`` is (NZ, NY), matching every raw
    detector frame in this codebase."""
    nz, ny = shape[:2]
    zz, yy = np.indices((nz, ny))
    r = np.hypot(yy - bc_y, zz - bc_z)
    return radial_spline_values(r, radii, values)


def median_intensity_near_bc(img, bc_y: float, bc_z: float, r_max: float = 10.0) -> float:
    """Median pixel value within ``r_max`` px of (bc_y, bc_z) — an outlier-
    resistant brightness reference near the beam centre (unlike the image's
    raw max, which can be a single saturated/hot pixel), used to scale the
    radial-threshold curve editor's default Y-axis view and default curve."""
    nz, ny = img.shape[:2]
    zz, yy = np.indices((nz, ny))
    r = np.hypot(yy - bc_y, zz - bc_z)
    mask = r <= r_max
    if not np.any(mask):
        return float(np.nanmax(img)) if img.size else 1.0
    return float(np.nanmedian(img[mask]))


def default_radial_threshold_points(hi: float, rmax: float):
    """Default knots for a freshly loaded image: a steep drop across the
    inner ~quarter of the corner radius down to zero, then flat. ``hi`` is
    normally :func:`median_intensity_near_bc` (an outlier-resistant
    brightness reference), ``rmax`` the corner radius from the current BC."""
    hi = max(0.0, float(hi))
    rmax = max(1.0, float(rmax))
    r_fracs = (0.0, 0.10, 0.25, 0.60)
    y_fracs = (1.0, 0.5, 0.1, 0.0)
    radii = np.array([f * rmax for f in r_fracs])
    values = np.array([f * hi for f in y_fracs])
    return radii, values


def apply_radial_threshold(img, bc_y: float, bc_z: float, radii, values):
    """Copy of ``img`` with pixels below the radial spline threshold
    (see :func:`radial_spline_threshold_map`) zeroed."""
    thr_map = radial_spline_threshold_map(img.shape, bc_y, bc_z, radii, values)
    out = img.copy()
    out[img < thr_map] = 0.0
    return out


def max_two_theta_deg(bc_y: float, bc_z: float, ny: int, nz: int,
                      lsd_um: float, pxY_um: float, pxZ_um: float = None) -> float:
    """True maximum 2theta reached anywhere on the detector frame — the
    farthest CORNER from the beam centre in physical units (handles an
    off-centre beam and non-square pixels), converted through Lsd. This is
    the bound a predicted ring list should use, instead of an arbitrary
    wavelength-side cutoff."""
    pxZ_um = pxZ_um or pxY_um
    dy_um = max(bc_y, ny - 1 - bc_y) * pxY_um
    dz_um = max(bc_z, nz - 1 - bc_z) * pxZ_um
    corner_um = math.hypot(dy_um, dz_um)
    return math.degrees(math.atan(corner_um / lsd_um))


def _thinned_bin_edges(lo: float, hi: float, step: float, max_count: int) -> np.ndarray:
    """Bin-edge positions between ``lo``/``hi`` spaced by ``step``, evenly
    strided down to at most ``max_count`` values so a fine bin size doesn't
    draw an unreadable number of overlay rings/spokes. Returns an empty
    array for a degenerate range/step."""
    if step <= 0 or hi <= lo:
        return np.empty(0, dtype=float)
    edges = np.arange(lo, hi + step / 2, step)
    edges = edges[edges <= hi + 1e-9]
    if edges.size > max_count:
        stride = int(math.ceil(edges.size / max_count))
        edges = edges[::stride]
    return edges


def draw_polar_bin_overlay(viewer, items: list, *, bc_y: float, bc_z: float,
                           r_min: float, r_max: float, r_bin: float, e_bin: float,
                           eta_min: float = -180.0, eta_max: float = 180.0,
                           show_grid: bool = False, max_rings: int = 50,
                           max_spokes: int = 72,
                           tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
                           lsd_um: Optional[float] = None,
                           pxY_um: Optional[float] = None,
                           pxZ_um: Optional[float] = None,
                           color: Optional[str] = None,
                           clear: bool = True) -> None:
    """If ``show_grid``, draw the Rmin/Rmax boundaries plus the polar
    (R, η) bin grid — arcs at each radial-bin edge plus spokes at each
    η-bin edge, both thinned to at most ``max_rings``/``max_spokes`` — onto
    ``viewer._iv``. Nothing is drawn when ``show_grid`` is false.

    **The arcs are bounded to ``eta_min``/``eta_max``, like the spokes.**
    They used to be full circles regardless, so integrating a limited
    azimuth drew a grid over the whole detector and the overlay claimed a
    region far larger than the one being binned. The grid depicts the
    integration region, so it has to stop where the region does; with the
    spokes at η min and η max already drawn, the result closes as the
    annulus sector it actually is.

    ``tx``/``ty``/``tz`` (deg) + ``lsd_um``/``pxY_um``/``pxZ_um`` are
    optional: when the detector has a non-trivial tilt AND all three
    lengths are supplied, rings/spokes are forward-projected through the
    real tilt geometry (:func:`tilted_ring_xy`/:func:`tilted_spoke_xy` —
    the same geometry ``midas_integrate_v2`` bins pixels with), instead of
    drawn as plain circles/straight lines around ``bc_y``/``bc_z``. A
    tilted detector's true R-bin boundaries are not circles centred on the
    beam centre, so without this the overlay visibly drifts off the actual
    diffraction rings as tilt grows — the plain-circle path is kept as the
    fallback for untilted geometries (and any caller that doesn't have
    Lsd/pixel size handy) since it's cheaper and exact in that case.

    ``items`` is the caller's own persistent list of previously-drawn
    items: cleared and rebuilt in place every call, mirroring the
    add/remove overlay lifecycle ``tab_calibrate.py``'s ``_draw_rings``
    uses for calibration rings.
    """
    import pyqtgraph as pg
    if clear:
        for it in items:
            viewer._iv.removeItem(it)
        items.clear()
    # ``color`` keys the whole overlay to one detector. Four panels' caking
    # regions on one composite canvas are only readable if each is drawn in
    # the colour that panel already has everywhere else (see
    # hydra_widgets.panel_color); ``clear=False`` is what lets them stack
    # into a single `items` list that one later call still tears down.
    # Default stays the original orange/blue so every existing caller is
    # pixel-identical.
    bound_color = color or "orange"
    grid_color = color or (120, 180, 255)
    if r_max <= 0:
        return
    tilt_aware = (bool(lsd_um) and bool(pxY_um) and bool(pxZ_um)
                  and (abs(tx) > 1e-9 or abs(ty) > 1e-9 or abs(tz) > 1e-9))
    # Clamp to a sane sweep: a non-positive or wrapped span means "no limit
    # given", and anything at or past a full turn is the whole ring.
    span = float(eta_max) - float(eta_min)
    if not (0.0 < span < 360.0 - 1e-9):
        arc_lo, arc_hi = 0.0, 360.0
    else:
        arc_lo, arc_hi = float(eta_min), float(eta_max)
    # Same parameterisation as the spokes below — bc + r*(sin η, cos η),
    # η = 0 straight up. The old full-circle path swept bc + r*(cos, sin),
    # which traces the SAME circle and so was harmless while the sweep was
    # a whole turn; as an arc it would be a quarter turn out of place.
    arc = np.radians(np.linspace(arc_lo, arc_hi, 256))

    def _ring_xy(r):
        if tilt_aware:
            two_theta = math.degrees(math.atan(r * pxY_um / lsd_um))
            return tilted_ring_xy(two_theta, tx, ty, tz, lsd_um, bc_y, bc_z,
                                   pxY_um, pxZ_um, n=256,
                                   eta_min=arc_lo, eta_max=arc_hi)
        return bc_y + r * np.sin(arc), bc_z + r * np.cos(arc)

    def _circle(r, pen):
        Y, Z = _ring_xy(r)
        item = pg.PlotDataItem(Y, Z, pen=pen)
        viewer._iv.addItem(item)
        items.append(item)

    if not show_grid:
        return

    if r_min > 0:
        _circle(r_min, pg.mkPen(bound_color, width=1.2, style=QtCore.Qt.DashLine))
    _circle(r_max, pg.mkPen(bound_color, width=1.5))
    def _spoke_xy(eta):
        if tilt_aware:
            tt_lo = math.degrees(math.atan(r_min * pxY_um / lsd_um))
            tt_hi = math.degrees(math.atan(r_max * pxY_um / lsd_um))
            return tilted_spoke_xy(tt_lo, tt_hi, float(eta), tx, ty, tz,
                                   lsd_um, bc_y, bc_z, pxY_um, pxZ_um)
        # bc + r*(sin η, cos η) — same convention tilted_spoke_xy reduces
        # to at zero tilt (and pixel_to_REta's eta=atan2(-Yc,Zc)): η=0 is
        # straight up (+Z), not along +Y. cos/sin here (not sin/cos)
        # would draw each η spoke 90° off from where it actually is.
        th_r = math.radians(float(eta))
        return ([bc_y + r_min * math.sin(th_r), bc_y + r_max * math.sin(th_r)],
                [bc_z + r_min * math.cos(th_r), bc_z + r_max * math.cos(th_r)])

    def _spoke(eta, pen):
        Y, Z = _spoke_xy(eta)
        item = pg.PlotDataItem(Y, Z, pen=pen)
        viewer._iv.addItem(item); items.append(item)

    # The two ends of the azimuth, told apart by line style. A caked region
    # is directional -- eta runs from start to end -- and with both edges
    # drawn identically there is nothing to say which way round it goes, or
    # which edge you just moved. Same grammar as the radial pair above:
    # the LOWER bound dashes, the UPPER is solid. Drawn in the boundary
    # colour at boundary width so they read as limits, not as grid lines.
    #
    # Only when the sweep is actually limited: at a full turn the two
    # coincide and a "start" marker would be an arbitrary ray across the
    # image.
    if 0.0 < (float(eta_max) - float(eta_min)) < 360.0 - 1e-9:
        _spoke(eta_min, pg.mkPen(bound_color, width=1.2, style=QtCore.Qt.DashLine))
        _spoke(eta_max, pg.mkPen(bound_color, width=1.5))

    grid_pen = pg.mkPen(grid_color, width=0.8)
    for r in _thinned_bin_edges(r_min, r_max, r_bin, max_rings):
        if r_min < r < r_max:
            _circle(r, grid_pen)
    eta_edges = _thinned_bin_edges(eta_min, eta_max, e_bin, max_spokes)
    if eta_max - eta_min >= 360.0 - 1e-6:
        eta_edges = eta_edges[eta_edges < eta_max - 1e-9]   # drop the wraparound duplicate
    for eta in eta_edges:
        _spoke(eta, grid_pen)


def _drop_missing_residual_map(spec):
    """Clear ``spec.ResidualCorrectionMap`` when it names a file that isn't
    readable, and say so.

    midas_integrate_v2 treats an unreadable map as fatal
    (``forward/pixels.py:_load_residual_map``), so one stale path takes down
    every integration, cake and pseudo-strain view built from the geometry —
    even though the map is a refinement on top of a geometry that is otherwise
    complete. A path can go stale in several ordinary ways: the map was never
    built (the Build-residual-map box was off, or too few non-outlier fits),
    the .bin was cleaned up alongside other scratch files, or a project is
    reopened on a host where that mount is absent. Degrading to "no map" keeps
    the rest of the result usable; failing shut does not.
    """
    import os
    rcm = getattr(spec, "ResidualCorrectionMap", None)
    if rcm and not os.path.isfile(str(rcm)):
        spec.ResidualCorrectionMap = None
        print(f"[spec] note: residual correction map {rcm!s} is not readable — "
              f"integrating without it. The geometry is unaffected; re-run the "
              f"calibration with 'Build residual map' on to regenerate it.",
              flush=True)
    return spec


def _build_spec(result, r_bin: float, eta_bin: float,
                r_min: Optional[float] = None, r_max: Optional[float] = None,
                eta_min: Optional[float] = None, eta_max: Optional[float] = None):
    """``r_min``/``r_max`` (px) are ``None`` by default, meaning "leave
    ``spec_from_calibration_result``'s own default" (``RMin=10.0``,
    ``RMax=None``→auto-corner) — only the Integration tabs' explicit
    Rmin/Rmax fields override them; every other caller (pump-probe,
    GSAS export, diagnostic cake previews) is unaffected. ``eta_min``/
    ``eta_max`` (deg) — same "None leaves the backend default" contract
    (``EtaMin=-180``/``EtaMax=180``, i.e. the full circle) — only
    Batch Integrate's explicit Eta min/max fields (populated directly or
    via a cake_parameters CSV's ETA_MIN/ETA_MAX, see ``cake_params.py``)
    override them."""
    from midas_calibrate_v2.compat.to_integrate import spec_from_calibration_result
    kwargs = dict(RBinSize=r_bin, EtaBinSize=eta_bin)
    if r_min is not None:
        kwargs["RMin"] = r_min
    if r_max is not None:
        kwargs["RMax"] = r_max
    if eta_min is not None:
        kwargs["EtaMin"] = eta_min
    if eta_max is not None:
        kwargs["EtaMax"] = eta_max
    spec = spec_from_calibration_result(result, **kwargs)
    # spec_from_calibration_result doesn't copy ImTransOpt — set it here so
    # every consumer of this spec can hand the RAW frame straight to
    # midas_integrate_v2 and let its own apply_trans_opt=True do the flip
    # (never flip the pixel array in midas_gui for a backend integration call).
    spec.TransOpt = list(getattr(result, "im_trans", []) or [])
    _apply_panel_fields(spec, getattr(result, "panel_layout", None),
                        getattr(result, "panel_shifts_path", None))
    return _drop_missing_residual_map(spec)


def _spec_from_json(path: str, r_bin: float, eta_bin: float):
    # compat.to_integrate copies the JSON's "residual_corr_bin" key straight
    # into the spec, so a calibration.json outlives the .bin it points at.
    from midas_calibrate_v2.compat.to_integrate import spec_from_calibration_json
    return _drop_missing_residual_map(
        spec_from_calibration_json(path, RBinSize=r_bin, EtaBinSize=eta_bin))


# v1 paramstest index (p#) → v2 harmonic name — the inverse of the single source
# of truth ``constants._V2_TO_V1`` (avoids a hand-maintained second copy).
_PARAMSTEST_DISTORTION = {v1: v2 for v2, v1 in _V2_TO_V1.items()}


def _crystallography(cal) -> dict:
    """``SpaceGroup``/``LatticeConstant`` kwargs for a calibrant, or ``{}``.

    A d-spacing calibrant (AgBH, ``Custom d-spacings…``) is not a crystal and
    has no cell to report. This used to be ``_SG.get(cal, 225)`` /
    ``_LC.get(cal, _LC["CeO2"])``, which filled both fields with *ceria's*
    values — so calibrating on silver behenate wrote ``SpaceGroup 225`` and
    ``LatticeConstant 5.4116 …`` into paramstest.txt, and the Results panel
    (which reads that same file back) displayed a ceria structure for a run
    that never saw any ceria. Returning ``{}`` leaves both at
    ``CalibrationParams``' own unset defaults instead of asserting a structure
    that does not exist; see ``_record_dspacing_calibrant`` for what is
    reported in their place.
    """
    from midas_gui.constants import _SG, _LC
    lattice = _LC.get(cal)
    if lattice is None:
        return {}
    return dict(SpaceGroup=_SG.get(cal, 225), LatticeConstant=lattice)


def _record_dspacing_calibrant(p, cal, result) -> None:
    """Name the calibrant and report the d-spacings the fit used, for a
    calibrant with no crystal structure. No-op for a crystalline one, whose
    SpaceGroup/LatticeConstant already say what it was."""
    from midas_gui.constants import _LC
    if _LC.get(cal) is not None:
        return
    p.extra["Calibrant"] = cal
    # Prefer the rings actually picked over the material's full table: a
    # 10-entry AgBH list fitted from 3 rings should report the 3.
    d_used = (getattr(result, "_d_used", None)
              or getattr(result, "_d_list", None) or [])
    if len(d_used):
        p.extra["DSpacings"] = " ".join(f"{float(d):.6f}" for d in d_used)


def write_standalone_paramstest(result, path, *, extra=None):
    """Write a v1 ``paramstest.txt`` from an AutoCalibrationResult (no geometry
    dependency on a live pipeline). Single implementation shared by the Calibrate
    and Export tabs. ``extra`` merges extra key/values into ``params.extra``."""
    import math
    from midas_calibrate.params import CalibrationParams
    from midas_gui.constants import _SG, _LC
    cal = getattr(result, "_calibrant_name", "CeO2")
    NY, NZ = result.NrPixelsY, result.NrPixelsZ
    pxY = float(result.pxY); pxZ = float(result.pxZ) if result.pxZ else pxY
    RhoD = math.sqrt(max(result.BC_y, NY - result.BC_y) ** 2 +
                     max(result.BC_z, NZ - result.BC_z) ** 2)
    p = CalibrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=pxY, pxZ=pxZ, Lsd=result.Lsd,
        BC_y=result.BC_y, BC_z=result.BC_z, tx=result.tx, ty=result.ty, tz=result.tz,
        Wavelength=result.wavelength_A, **_crystallography(cal),
        RhoD=RhoD, MaxRingRad=RhoD * 0.97)
    _record_dspacing_calibrant(p, cal, result)
    for v2n, v1n in _V2_TO_V1.items():
        val = (result.distortion or {}).get(v2n)
        if val is not None:
            setattr(p, v1n, float(val))
    for k, v in (extra or {}).items():
        p.extra[k] = v
    p.write(str(path))
    im_trans = getattr(result, "im_trans", None)
    if im_trans:
        with open(path, "a") as f:
            for code in im_trans:
                f.write(f"ImTransOpt {int(code)}\n")
    return p


def paramstest_pairs(result, selected=None) -> list:
    """(key, value) pairs exactly as written to paramstest.txt — generated by
    writing a temp file with the shared writer, so the readout matches the file.

    ``selected``, when given, is the set of v2 distortion-coefficient names
    actually chosen for the run that produced ``result`` — distortion rows
    outside that set are dropped from the *display* only. The underlying
    file/``result.distortion`` keep every p-slot's real value (including any
    legitimately-held-fixed nonzero value carried from a prior calibration),
    since that's what the geometry model actually used."""
    import os, tempfile
    fd, tmp = tempfile.mkstemp(suffix=".txt"); os.close(fd)
    try:
        write_standalone_paramstest(result, tmp)
        lines = Path(tmp).read_text().splitlines()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    pairs = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        key = parts[0]
        if selected is not None and key in _PARAMSTEST_DISTORTION \
                and _PARAMSTEST_DISTORTION[key] not in selected:
            continue
        pairs.append((key, " ".join(parts[1:])))
    return pairs


def _spec_from_result_ns(r_bin, eta_bin, r_min: Optional[float] = None,
                         r_max: Optional[float] = None,
                         eta_min: Optional[float] = None, eta_max: Optional[float] = None,
                         **fields):
    """Build an IntegrationSpec from geometry fields via a duck-typed result.

    Routes through ``spec_from_calibration_result`` so RhoD, RMax and the bin
    counts are derived exactly as for a live calibration result. ``r_min``/
    ``r_max``/``eta_min``/``eta_max`` — see ``_build_spec``.
    """
    from types import SimpleNamespace
    from midas_calibrate_v2.compat.to_integrate import spec_from_calibration_result
    ns = SimpleNamespace(
        NrPixelsY=int(fields["NrPixelsY"]), NrPixelsZ=int(fields["NrPixelsZ"]),
        pxY=float(fields["pxY"]), pxZ=float(fields.get("pxZ") or fields["pxY"]),
        Lsd=float(fields["Lsd"]), BC_y=float(fields["BC_y"]), BC_z=float(fields["BC_z"]),
        tx=float(fields.get("tx") or 0.0), ty=float(fields.get("ty") or 0.0),
        tz=float(fields.get("tz") or 0.0), wavelength_A=float(fields["wavelength_A"]),
        distortion=fields.get("distortion") or {}, residual_corr_bin_path=None)
    kwargs = dict(RBinSize=float(r_bin), EtaBinSize=float(eta_bin))
    if r_min is not None:
        kwargs["RMin"] = r_min
    if r_max is not None:
        kwargs["RMax"] = r_max
    if eta_min is not None:
        kwargs["EtaMin"] = eta_min
    if eta_max is not None:
        kwargs["EtaMax"] = eta_max
    spec = spec_from_calibration_result(ns, **kwargs)
    spec.TransOpt = list(fields.get("im_trans") or [])
    _apply_panel_fields(spec, fields.get("panel_layout"), fields.get("panel_shifts_path"))
    return spec


def geometry_fields_from_file(path: str) -> dict:
    """Parse a MIDAS paramstest, a pyFAI ``.poni``, or a calibration ``.json``
    into a normalized full-geometry dict (auto-detected by extension then content).

    Returns keys ``NrPixelsY, NrPixelsZ, pxY, pxZ, Lsd`` (µm), ``BC_y, BC_z`` (px),
    ``tx, ty, tz`` (deg), ``wavelength_A`` (Å), ``distortion`` (dict), ``im_trans``
    (ordered list of MIDAS ``ImTransOpt`` codes).  ``pxZ`` defaults to ``pxY``
    and tilts default to 0 when absent.  Raises ``ValueError`` if a required
    key is missing.

    PONI tilts (Rot1/2/3) are not mapped to MIDAS ty/tz/tx — only the beam-centre
    translation is used (consistent with MIDAS's own ``poni_to_bc``).
    """
    import json
    p = Path(path)
    text = p.read_text()
    suf = p.suffix.lower()

    def _norm(fields):
        fields["pxZ"] = fields.get("pxZ") or fields["pxY"]
        for t in ("tx", "ty", "tz"):
            fields[t] = fields.get(t) or 0.0
        fields["distortion"] = fields.get("distortion") or {}
        # Absent and empty are different answers. A paramstest with no
        # ImTransOpt lines genuinely means "no transform"; a backend-written
        # calibration.json (which records no im_trans at all) and a .poni
        # (which has no such concept) mean "this file does not say". Callers
        # that restore UI state must not read the second as the first and
        # silently untick the user's flips -- so record which one it was,
        # while still handing every existing caller a plain list.
        fields["im_trans_in_file"] = fields.get("im_trans") is not None
        fields["im_trans"] = list(fields.get("im_trans") or [])
        fields["panel_layout"] = fields.get("panel_layout") or None
        ps = fields.get("panel_shifts_path") or None
        if ps and not Path(ps).is_file():
            # A bare filename (the v1/C DetectorMapper convention — resolved
            # relative to the working directory) or a stale absolute path
            # from before this geometry file was copied/moved elsewhere.
            # Retry it next to the geometry file itself, since that's where
            # every midas-gui writer (write_panel_shifts_file's sidecar
            # convention) actually puts it.
            beside = p.parent / Path(ps).name
            if beside.is_file():
                ps = str(beside)
        fields["panel_shifts_path"] = ps
        return fields

    # ── calibration.json (GUI bare keys OR pipeline *_um/_px/_deg keys) ──
    if suf == ".json" or text.lstrip().startswith("{"):
        c = json.loads(text)

        def g(*keys):
            for k in keys:
                if k in c and c[k] is not None:
                    return c[k]
            return None

        fields = dict(
            NrPixelsY=g("NrPixelsY"), NrPixelsZ=g("NrPixelsZ"),
            pxY=g("pxY", "pxY_um"), pxZ=g("pxZ", "pxZ_um"),
            Lsd=g("Lsd", "Lsd_um"), BC_y=g("BC_y", "BC_y_px"), BC_z=g("BC_z", "BC_z_px"),
            tx=g("tx", "tx_deg"), ty=g("ty", "ty_deg"), tz=g("tz", "tz_deg"),
            wavelength_A=g("wavelength_A", "Wavelength"), distortion=c.get("distortion", {}),
            im_trans=c.get("im_trans"),   # None when absent -- see _norm
            panel_layout=c.get("panel_layout"), panel_shifts_path=c.get("panel_shifts_path"))
        missing = [k for k in ("NrPixelsY", "NrPixelsZ", "pxY", "Lsd", "BC_y", "BC_z",
                               "wavelength_A") if fields[k] is None]
        if missing:
            raise ValueError(f"calibration json missing keys: {', '.join(missing)}")
        return _norm(fields)

    # ── pyFAI .poni ──
    if suf == ".poni" or "poni_version" in text or "Poni1" in text:
        from midas_integrate_v2 import poni_to_bc
        vals, det_cfg = {}, {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k, v = k.strip().lower(), v.strip()
            if k == "detector_config":
                try:
                    det_cfg = json.loads(v)
                except Exception:
                    det_cfg = {}
            else:
                vals[k] = v

        def f(k):
            try:
                return float(vals[k])
            except (KeyError, ValueError):
                return None

        dist_m, poni1, poni2, wl_m = f("distance"), f("poni1"), f("poni2"), f("wavelength")
        px1, px2, shape = det_cfg.get("pixel1"), det_cfg.get("pixel2"), det_cfg.get("max_shape")
        if None in (dist_m, poni1, poni2, wl_m) or px1 is None or px2 is None or not shape:
            raise ValueError(
                "PONI missing Distance/Poni1/Poni2/Wavelength or Detector_config "
                "with pixel1/pixel2 + max_shape — cannot build an integration spec.")
        pxZ_um, pxY_um = float(px1) * 1e6, float(px2) * 1e6      # axis1=slow=Z, axis2=fast=Y
        NrPixelsZ, NrPixelsY = int(shape[0]), int(shape[1])
        bc_y, bc_z = poni_to_bc(float(poni1), float(poni2), pxY_um, pxZ_um)
        return _norm(dict(
            NrPixelsY=NrPixelsY, NrPixelsZ=NrPixelsZ, pxY=pxY_um, pxZ=pxZ_um,
            Lsd=float(dist_m) * 1e6, BC_y=bc_y, BC_z=bc_z,
            tx=0.0, ty=0.0, tz=0.0, wavelength_A=float(wl_m) * 1e10, distortion={}))

    # ── MIDAS paramstest ──
    kv, p_vals, panel_kv = {}, {}, {}
    NY = NZ = None
    panel_shifts_path = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        try:
            if key == "Lsd":
                kv["Lsd"] = float(parts[1])
            elif key == "BC":
                kv["BC_y"], kv["BC_z"] = float(parts[1]), float(parts[2])
            elif key in ("tx", "ty", "tz"):
                kv[key] = float(parts[1])
            elif key == "Wavelength":
                kv["wavelength_A"] = float(parts[1])
            elif key in ("px", "pxY"):
                kv["pxY"] = float(parts[1])
            elif key == "NrPixelsY":
                NY = int(float(parts[1]))
            elif key == "NrPixelsZ":
                NZ = int(float(parts[1]))
            elif key == "NPanelsY":
                panel_kv["n_y"] = int(float(parts[1]))
            elif key == "NPanelsZ":
                panel_kv["n_z"] = int(float(parts[1]))
            elif key == "PanelSizeY":
                panel_kv["sy"] = int(float(parts[1]))
            elif key == "PanelSizeZ":
                panel_kv["sz"] = int(float(parts[1]))
            elif key == "PanelGapsY":
                panel_kv["gap_y"] = [int(float(x)) for x in parts[1:]]
            elif key == "PanelGapsZ":
                panel_kv["gap_z"] = [int(float(x)) for x in parts[1:]]
            elif key == "PanelShiftsFile":
                panel_shifts_path = " ".join(parts[1:]) or None
            elif len(key) > 1 and key[0] == "p" and key[1:].isdigit():
                p_vals[key] = float(parts[1])
        except (ValueError, IndexError):
            continue
    missing = [n for n in ("Lsd", "BC_y", "pxY", "wavelength_A") if n not in kv]
    if NY is None or NZ is None:
        missing.append("NrPixelsY/NrPixelsZ")
    if missing:
        raise ValueError(f"paramstest missing keys: {', '.join(missing)}")
    dist = {v2: p_vals[p1] for p1, v2 in _PARAMSTEST_DISTORTION.items() if p1 in p_vals}
    panel_layout = None
    if all(k in panel_kv for k in ("n_y", "n_z", "sy", "sz")):
        panel_layout = {
            "n_y": panel_kv["n_y"], "n_z": panel_kv["n_z"],
            "sy": panel_kv["sy"], "sz": panel_kv["sz"],
            "gap_y": panel_kv.get("gap_y", 0), "gap_z": panel_kv.get("gap_z", 0),
        }
    return _norm(dict(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=kv["pxY"], pxZ=kv["pxY"],
        Lsd=kv["Lsd"], BC_y=kv["BC_y"], BC_z=kv["BC_z"], tx=kv.get("tx"), ty=kv.get("ty"),
        tz=kv.get("tz"), wavelength_A=kv["wavelength_A"], distortion=dist,
        im_trans=parse_im_trans(text),
        panel_layout=panel_layout, panel_shifts_path=panel_shifts_path))


def spec_from_geometry_file(path: str, r_bin: float, eta_bin: float,
                            r_min: Optional[float] = None, r_max: Optional[float] = None,
                            eta_min: Optional[float] = None, eta_max: Optional[float] = None):
    """Build an IntegrationSpec from a MIDAS paramstest, a pyFAI ``.poni``, or a
    calibration ``.json``.  Thin wrapper over :func:`geometry_fields_from_file`.
    ``r_min``/``r_max``/``eta_min``/``eta_max`` — see ``_build_spec``."""
    return _spec_from_result_ns(r_bin, eta_bin, r_min=r_min, r_max=r_max,
                                eta_min=eta_min, eta_max=eta_max,
                                **geometry_fields_from_file(path))


def result_ns_from_geometry_file(path: str):
    """Build a duck-typed calibration *result* (SimpleNamespace) from a paramstest,
    ``.poni`` or ``.json`` geometry file — carries the attributes
    ``spec_from_calibration_result`` / the tabs' ``set_calibration`` expect
    (``wavelength_A, Lsd, BC_y, BC_z, pxY, pxZ, NrPixelsY, NrPixelsZ, tx, ty, tz,
    distortion, im_trans``)."""
    from types import SimpleNamespace
    f = geometry_fields_from_file(path)
    return SimpleNamespace(
        NrPixelsY=int(f["NrPixelsY"]), NrPixelsZ=int(f["NrPixelsZ"]),
        pxY=float(f["pxY"]), pxZ=float(f["pxZ"]),
        Lsd=float(f["Lsd"]), BC_y=float(f["BC_y"]), BC_z=float(f["BC_z"]),
        tx=float(f["tx"]), ty=float(f["ty"]), tz=float(f["tz"]),
        wavelength_A=float(f["wavelength_A"]), distortion=f["distortion"],
        im_trans=list(f.get("im_trans") or []),
        residual_corr_bin_path=None,
        panel_layout=f.get("panel_layout"), panel_shifts_path=f.get("panel_shifts_path"))


def resolve_calibration_fields(calib_result, use_file: bool, file_path: str, *,
                               source_label: str = "Tab 2 calibration"):
    """Resolve the geometry currently selected (an in-memory calibration
    result, or a geometry file), as a dict of display fields — or
    ``(None, note)`` if unavailable. Shared by ``BatchTab`` (single-detector)
    and ``HydraBatchPanelCard`` (one panel's own calibration source)."""
    if use_file or calib_result is None:
        path = (file_path or "").strip()
        if not path:
            return None, "No calibration file selected."
        if not Path(path).exists():
            return None, "Calibration file not found."
        try:
            return geometry_fields_from_file(path), f"From file: {Path(path).name}"
        except Exception as e:
            return None, f"Unreadable calibration file: {e}"
    r = calib_result
    fields = {
        "wavelength_A": getattr(r, "wavelength_A", None),
        "Lsd": getattr(r, "Lsd", None),
        "BC_y": getattr(r, "BC_y", None), "BC_z": getattr(r, "BC_z", None),
        "tx": getattr(r, "tx", 0.0), "ty": getattr(r, "ty", 0.0),
        "tz": getattr(r, "tz", 0.0),
        "pxY": getattr(r, "pxY", None), "pxZ": getattr(r, "pxZ", None),
        "NrPixelsY": getattr(r, "NrPixelsY", None),
        "NrPixelsZ": getattr(r, "NrPixelsZ", None),
        "distortion": getattr(r, "distortion", {}) or {},
        "im_trans": list(getattr(r, "im_trans", []) or []),
    }
    return fields, f"From {source_label}."


def full_calibration_snapshot(calib_result, use_file: bool, file_path: str, *,
                              source_label: str = "Tab 2 calibration"):
    """Every field of the calibration currently selected — not just the
    display subset :func:`resolve_calibration_fields` returns.

    Same ``(dict | None, note)`` contract as that function, so it is a
    drop-in wherever the *whole* calibration matters rather than the handful
    of numbers a user reads off a panel. That is the provenance path: an
    integration attempt's ``calibration_snapshot`` has to be able to
    reconstruct the calibration the run actually used, which the 13 display
    fields cannot (they drop ``_calibrant_name``, ``_panel_unpacked``,
    ``panel_layout``, the refined-parameter σ / at-limit flags, ...).

    The display fields are overlaid *on top of* the raw result, not merged
    under it: they are read straight off the same object, so no value can
    drift, but they also supply defaults a bare result may not carry
    (``tx/ty/tz`` → 0.0, ``distortion`` → {}, ``im_trans`` → []). Keeping the
    output a strict superset of ``resolve_calibration_fields``' is what lets
    every existing reader — ``project.calibration_namespace``,
    ``render_calib_value_grid``, ``gsas_export`` — consume it unchanged.
    """
    fields, note = resolve_calibration_fields(calib_result, use_file, file_path,
                                              source_label=source_label)
    if fields is None:
        return None, note
    if use_file or calib_result is None:
        try:
            result = result_ns_from_geometry_file((file_path or "").strip())
        except Exception:
            # Unreadable on the re-parse (it parsed once, for `fields`) —
            # the display subset is still an honest record. Never block
            # logging an otherwise-good integration over this.
            return fields, note
    else:
        result = calib_result
    from midas_gui import project   # deferred: project doesn't import helpers
    full = project.sanitize_result_dict(result) or {}
    return {**full, **fields}, note


def collapse_cake_eta(cakes) -> np.ndarray:
    """``(n_frames, n_eta, n_r)`` → ``(n_frames, n_r)``, averaging each
    frame's filled eta bins.

    Shared by ``tab_batch.BatchTab._collapse_cakes`` (reconstructing a
    profile for the Waterfall/Stacked-profiles views, since multi-azimuth
    mode keeps the cake instead of the run's own collapsed profile) and
    ``cake_hdf5.write_cake_h5``'s fallback when no real engine-collapsed
    profile is available. Exact-zero bins are unfilled eta/R coverage rather
    than measured zeros — the same convention ``CakeViewer``'s auto-levelling
    uses — so they're excluded from the mean instead of dragging it toward
    zero. It is an approximation of the engine's count-weighted collapse,
    not a reproduction of it."""
    arr = np.asarray(cakes, dtype=np.float64)
    filled = (arr != 0).sum(axis=1)
    return arr.sum(axis=1) / np.maximum(filled, 1)


def render_calib_value_grid(grid: "QtWidgets.QGridLayout", note_label: "QtWidgets.QLabel",
                            fields: Optional[dict], note: str) -> None:
    """Populate a read-only 2-column key/value grid of calibration-geometry
    fields (as resolved by :func:`resolve_calibration_fields`). Shared by
    ``BatchTab`` and ``HydraBatchPanelCard`` so the "Calibration values"
    display looks identical in both places."""
    while grid.count():
        it = grid.takeAt(0)
        w = it.widget()
        if w is not None:
            w.deleteLater()
    note_label.setText(note)
    if not fields:
        return

    def _num(v, fmt):
        return "—" if v is None else format(float(v), fmt)

    lsd = fields.get("Lsd")
    pxY = fields.get("pxY"); pxZ = fields.get("pxZ") or pxY
    dist = fields.get("distortion") or {}
    n_dist = sum(1 for v in dist.values() if abs(float(v)) > 1e-12)
    im_trans = fields.get("im_trans") or []
    _IM_TRANS_NAMES = {1: "Flip Y", 2: "Flip Z", 3: "Transpose"}
    trans_txt = (", ".join(_IM_TRANS_NAMES.get(c, str(c)) for c in im_trans)
                 if im_trans else "None")
    rows = [
        ("λ (Å)", _num(fields.get("wavelength_A"), ".5f")),
        ("Lsd (mm)", "—" if lsd is None else format(float(lsd) / 1000.0, ".3f")),
        ("BC_y (px)", _num(fields.get("BC_y"), ".2f")),
        ("BC_z (px)", _num(fields.get("BC_z"), ".2f")),
        ("tx (°)", _num(fields.get("tx"), ".4f")),
        ("ty (°)", _num(fields.get("ty"), ".4f")),
        ("tz (°)", _num(fields.get("tz"), ".4f")),
        ("pxY (µm)", _num(pxY, ".3f")),
        ("pxZ (µm)", _num(pxZ, ".3f")),
        ("Detector", f"{fields.get('NrPixelsY') or '—'} × {fields.get('NrPixelsZ') or '—'}"),
        ("Distortion", f"{n_dist} non-zero coeff" + ("s" if n_dist != 1 else "")),
        ("ImTransOpt", trans_txt),
    ]
    ncols = 2
    per = (len(rows) + ncols - 1) // ncols
    for i, (k, v) in enumerate(rows):
        col, row = divmod(i, per)
        kl = QtWidgets.QLabel(k + ":")
        kl.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        vl = QtWidgets.QLabel(v)
        vl.setStyleSheet(f"font-family:{S.MONO_CSS};font-size:10px")
        vl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        grid.addWidget(kl, row, col * 2, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        grid.addWidget(vl, row, col * 2 + 1, QtCore.Qt.AlignVCenter)
    grid.setColumnStretch(ncols * 2 + 1, 1)


# ── GUI-state serialization (Save/Load GUI State) ────────────────────────────────

def widgets_to_dict(widgets: dict) -> dict:
    """Snapshot a ``{key: widget}`` map into a plain JSON-able dict, by widget type:
    spin boxes → ``.value()``, combo boxes → current text, line edits → ``.text()``,
    checkable buttons (or a checkable ``QGroupBox``) → ``.isChecked()``. Unrecognized
    widget types are skipped."""
    out = {}
    for key, w in widgets.items():
        if isinstance(w, QtWidgets.QAbstractSpinBox):
            out[key] = w.value()
        elif isinstance(w, QtWidgets.QComboBox):
            out[key] = w.currentText()
        elif isinstance(w, QtWidgets.QLineEdit):
            out[key] = w.text()
        elif isinstance(w, QtWidgets.QAbstractButton):
            out[key] = w.isChecked()
        elif isinstance(w, QtWidgets.QGroupBox) and w.isCheckable():
            out[key] = w.isChecked()
    return out


def apply_dict_to_widgets(widgets: dict, data: dict) -> None:
    """Inverse of :func:`widgets_to_dict`. Restores each field in its own
    try/except (a stale or missing key can't abort the rest of the restore) and
    blocks signals around each set, same convention as ``set_geometry``. Combo
    boxes are matched by text; a value no longer present in the list is left
    untouched rather than raising."""
    for key, w in widgets.items():
        if key not in data:
            continue
        val = data[key]
        w.blockSignals(True)
        try:
            if isinstance(w, QtWidgets.QAbstractSpinBox):
                w.setValue(val)
            elif isinstance(w, QtWidgets.QComboBox):
                idx = w.findText(str(val))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                elif w.isEditable():
                    w.setEditText(str(val))
            elif isinstance(w, QtWidgets.QLineEdit):
                w.setText(str(val))
            elif isinstance(w, QtWidgets.QAbstractButton):
                w.setChecked(bool(val))
            elif isinstance(w, QtWidgets.QGroupBox) and w.isCheckable():
                w.setChecked(bool(val))
        except Exception:
            pass
        finally:
            w.blockSignals(False)


# ── No-scroll spinboxes (prevent accidental wheel value changes) ────────────────

class _NoScrollSpinBox(QtWidgets.QSpinBox):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    def wheelEvent(self, e):
        e.ignore()


class _NoScrollDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    def wheelEvent(self, e):
        e.ignore()


class _NoScrollComboBox(QtWidgets.QComboBox):
    """QComboBox that ignores mouse-wheel scrolls so the selection never changes
    by accident; the event propagates to the parent (e.g. the scroll panel).
    The drop-down popup still scrolls normally when open."""
    def wheelEvent(self, e):
        e.ignore()


def refresh_combo_items(combo: QtWidgets.QComboBox, items) -> None:
    """Repopulate a plain-text combo box from ``items``, keeping the current
    selection if it still exists, else falling back to index 0. Used to bring
    profile-scoped dropdowns (e.g. the Calibrant combo) up to date after a
    profile switch without disturbing the user's current pick."""
    prev = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(list(items))
    idx = combo.findText(prev)
    combo.setCurrentIndex(idx if idx >= 0 else 0)
    combo.blockSignals(False)


def _fspin(lo, hi, dec, val, suf="", step=None):
    """``step`` (if given) fixes the up/down-arrow increment; omit it to keep the
    default adaptive-decimal stepping used everywhere else in the GUI."""
    s = _NoScrollDoubleSpinBox()
    s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val)
    if step is None:
        s.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    else:
        s.setStepType(QtWidgets.QAbstractSpinBox.DefaultStepType)
        s.setSingleStep(step)
    if suf:
        s.setSuffix(f"  {suf}")
    s.setMaximumWidth(104)   # keep numeric fields compact (don't stretch to fill forms)
    return s


def _clickable_menu_label(text, entries, parent=None):
    """A clickable, form-label-sized widget with a popup menu.

    Looks like a field label (underlined, accent colour) but occupies the same
    space and pops a menu on click. ``entries`` is either a list of
    ``(label, callback)`` pairs, or a zero-arg callable returning that list —
    the callable form is re-invoked right before the menu opens, so entries
    backed by a per-profile config (e.g. ``constants.PIXEL_PRESETS``) always
    reflect whichever profile is active, not whatever it was when this widget
    was constructed.
    """
    from PyQt5 import QtWidgets
    btn = QtWidgets.QToolButton(parent)
    btn.setText(text)
    btn.setAutoRaise(True)
    btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setStyleSheet(
        "QToolButton { border: none; padding: 0 2px; color: #4da3ff; }"
        "QToolButton::menu-indicator { image: none; }")
    f = btn.font(); f.setUnderline(True); btn.setFont(f)
    menu = QtWidgets.QMenu(btn)

    def _populate():
        menu.clear()
        items = entries() if callable(entries) else entries
        for label, cb in items:
            act = menu.addAction(label)
            act.triggered.connect(lambda _checked=False, c=cb: c())

    if callable(entries):
        menu.aboutToShow.connect(_populate)
    _populate()
    btn.setMenu(menu)
    return btn


def make_kedge_label(wl_spin, text="λ:", parent=None):
    """A clickable 'λ' label that pops a menu to either type a photon energy
    (keV, auto-converted to wavelength) or pick a common K-edge foil energy."""
    from midas_gui import constants as C

    btn = QtWidgets.QToolButton(parent)
    btn.setText(text)
    btn.setAutoRaise(True)
    btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setStyleSheet(
        "QToolButton { border: none; padding: 0 2px; color: #4da3ff; }"
        "QToolButton::menu-indicator { image: none; }")
    f = btn.font(); f.setUnderline(True); btn.setFont(f)

    menu = QtWidgets.QMenu(btn)

    energy_row = QtWidgets.QWidget(menu)
    row = QtWidgets.QHBoxLayout(energy_row)
    row.setContentsMargins(8, 4, 8, 4)
    row.addWidget(QtWidgets.QLabel("Energy:"))
    cur_wl = wl_spin.value()
    energy_spin = _fspin(0.1, 999.0, 3, C.HC_KEV_A / cur_wl if cur_wl > 0 else 10.0, "keV")
    row.addWidget(energy_spin)

    def _apply_energy():
        keV = energy_spin.value()
        if keV > 0:
            wl_spin.setValue(float(C.HC_KEV_A / keV))
        menu.close()

    energy_spin.lineEdit().returnPressed.connect(_apply_energy)
    apply_btn = QtWidgets.QToolButton()
    apply_btn.setText("↵")
    apply_btn.setToolTip("Apply energy → wavelength")
    apply_btn.clicked.connect(_apply_energy)
    row.addWidget(apply_btn)

    energy_action = QtWidgets.QWidgetAction(menu)
    energy_action.setDefaultWidget(energy_row)
    menu.addAction(energy_action)
    menu.addSeparator()

    # Foil entries are rebuilt right before the menu opens (not once here) so a
    # profile switch's updated constants.K_EDGE_FOILS shows up immediately.
    _foil_actions: list = []

    def _rebuild_foils():
        for act in _foil_actions:
            menu.removeAction(act)
        _foil_actions.clear()
        for sym, keV in C.K_EDGE_FOILS:
            label = f"{sym}   {keV:.2f} keV · {C.HC_KEV_A / keV:.5f} Å"
            act = menu.addAction(label)
            act.triggered.connect(
                lambda _checked=False, l=C.HC_KEV_A / keV: wl_spin.setValue(float(l)))
            _foil_actions.append(act)

    menu.aboutToShow.connect(_rebuild_foils)
    _rebuild_foils()

    btn.setMenu(menu)
    btn.setToolTip("Click to enter a photon energy (keV) or pick a common K-edge foil energy.")
    return btn


def make_pixel_label(px_spin, text="px:", also=None, parent=None):
    """A clickable pixel-size label that pops a common-detector menu; selecting an
    entry sets ``px_spin`` (and ``also``, if given) to that detector's pixel size."""
    from midas_gui import constants as C

    def _setter(um):
        def _apply():
            px_spin.setValue(float(um))
            if also is not None:
                also.setValue(float(um))
        return _apply

    # A callable, not a precomputed list: constants.PIXEL_PRESETS is rebound (not
    # mutated) by a profile switch, so re-reading it via module attribute access
    # each time the menu opens (_clickable_menu_label's aboutToShow rebuild) is
    # what keeps this current — a one-time import here would go stale.
    def _entries():
        return [(f"{name}  ({um:g} µm)", _setter(um)) for name, um in C.PIXEL_PRESETS]

    btn = _clickable_menu_label(text, _entries, parent)
    btn.setToolTip("Click to set the pixel size from a common detector.")
    return btn


def make_calib_values_button(fields_getter, text="View calibration ▾", parent=None):
    """A clickable button that pops a menu showing the calibration-geometry
    value grid (built by :func:`render_calib_value_grid`) instead of leaving
    it always visible in the layout — same click-to-see-options interaction
    as :func:`make_pixel_label`/:func:`make_kedge_label` above, just showing
    a read-only grid instead of a list of selectable presets.

    ``fields_getter`` is a zero-arg callable returning ``(fields, note)`` —
    the shape :func:`resolve_calibration_fields` already returns — invoked
    fresh each time the menu opens (via ``aboutToShow``, matching
    ``_clickable_menu_label``'s pattern) so the popup always reflects
    whichever calibration source is currently active."""
    btn = QtWidgets.QToolButton(parent)
    btn.setText(text)
    btn.setAutoRaise(True)
    btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setStyleSheet(
        "QToolButton { border: none; padding: 0 2px; color: #4da3ff; }"
        "QToolButton::menu-indicator { image: none; }")
    f = btn.font(); f.setUnderline(True); btn.setFont(f)
    btn.setToolTip("Click to view the calibration geometry currently in use.")

    menu = QtWidgets.QMenu(btn)

    def _populate():
        # Rebuild the whole popup body (not just the grid contents) on every
        # open: a QMenu computes its popup size from the QWidgetAction's
        # sizeHint at show time, and mutating a *persistent* grid layout in
        # place risks that size being stale (e.g. the very first open, before
        # any fields exist, would otherwise cache a near-zero size). A fresh
        # widget tree each time guarantees the sizeHint matches the content
        # about to be shown.
        menu.clear()
        host = QtWidgets.QWidget(menu)
        hv = QtWidgets.QVBoxLayout(host)
        hv.setContentsMargins(10, 8, 10, 8); hv.setSpacing(6)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(18); grid.setVerticalSpacing(4)
        grid_host = QtWidgets.QWidget(); grid_host.setLayout(grid)
        hv.addWidget(grid_host)
        note_label = QtWidgets.QLabel()
        note_label.setStyleSheet(f"color:{S.MUTED};font-size:10px")
        note_label.setWordWrap(True)
        hv.addWidget(note_label)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(host)
        menu.addAction(action)

        fields, note = fields_getter()
        render_calib_value_grid(grid, note_label, fields, note)

    menu.aboutToShow.connect(_populate)
    btn.setMenu(menu)
    return btn


# ── Layout helpers ──────────────────────────────────────────────────────────────

def _twocol(lbl1, w1, lbl2, w2):
    """Two label+widget pairs on one row: 4 px within a pair, 20 px between pairs.

    Labels passed as strings are auto-converted to right-aligned QLabels.
    """
    def _lbl(x):
        if isinstance(x, str):
            l = QtWidgets.QLabel(x)
            l.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return l
        return x
    h = QtWidgets.QHBoxLayout()
    h.setSpacing(4)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(_lbl(lbl1))
    h.addWidget(w1)
    h.addSpacing(20)          # clear visual gap between the two pairs
    h.addWidget(_lbl(lbl2))
    h.addWidget(w2)
    h.addStretch(1)
    return h


def _sep():
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.HLine)
    f.setFrameShadow(QtWidgets.QFrame.Sunken)
    return f


def _browse(parent, caption, filt, start_dir: str = "") -> str:
    p, _ = QtWidgets.QFileDialog.getOpenFileName(parent, caption, start_dir, filt)
    return p
