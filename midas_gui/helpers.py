"""Module-level helpers: image IO, transforms, ring prediction, spec building,
log stream, and the no-scroll spinbox / two-column layout widgets used everywhere.

These are ported verbatim from midas_workflow_gui_v3.py (the frozen template) so
the established conventions in context/design_rules.md are preserved exactly.
"""
from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Optional

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


def read_hdf5_stack_combined(path, dataset: str, *, chunk_size: Optional[int] = None,
                             op: str = "mean") -> list:
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
        n = dset.shape[0]
        size = chunk_size if chunk_size else n
        out = []
        for start in range(0, n, size):
            stack = np.asarray(dset[start:start + size], dtype=np.float32)
            out.append(combine(stack).astype(np.float32))
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
    (".pxrd", "pxrd"),
    (".pmg", "pimega"),
)
# Pixel size (µm) for each recognized detector tag. No entry for "pxrd" —
# Pixirad is identified but its pixel size isn't auto-populated (not given).
_DETECTOR_PIXEL_UM = {"ge": 200.0, "vrx": 150.0, "pimega": 55.0}
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


def _collect_frame_paths(raw) -> list:
    """Frames from a folder or a *.tif glob (sorted).  Mirrors tab_view logic.

    ``raw`` may also be an already-resolved ``list[str]`` of explicit paths
    (an arbitrary multi-file selection from ``dialogs.BrowseFilesDialog``,
    which has no single string/glob representation) — returned as-is."""
    if isinstance(raw, list):
        return raw
    import glob as _glob
    p = Path(raw)
    if p.is_dir():
        out = []
        for ext in ("*.tif", "*.tiff", "*.h5", "*.hdf5", "*.ge*", "*.cbf", "*.edf"):
            out.extend(sorted(p.glob(ext)))
        return [str(x) for x in out]
    # recursive=True only changes behavior when the pattern contains "**"
    # (the filestem-filter case, widgets.DataLoaderPanel._raw_source) — a
    # plain glob without it is unaffected.
    return sorted(_glob.glob(raw, recursive=True))


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
    "a set of single-frame files to average/stack over"."""
    if isinstance(path, list):
        return "folder"
    if Path(path).is_dir() or any(c in path for c in "*?"):
        return "folder"
    if is_h5(path):
        return "hdf5"
    return "file"


def average_field(kind: str, path: str, dataset: str = "exchange/data",
                  idx_start: int = 0, idx_end: int = -1) -> np.ndarray:
    """Build a single 2-D field by averaging over an index range.

    kind:
      "file"   — a single image file; if it holds a 3-D stack, average [start..end].
      "folder" — a folder or *.tif glob; average frames [start..end] across files.
      "hdf5"   — average dataset[start..end+1] if 3-D, else the 2-D dataset.

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


def apply_field_corrections(img: np.ndarray, *, dark=None, bright=None,
                            bright_mode: str = "divide", background=None,
                            clip_negative: bool = True) -> np.ndarray:
    """Apply dark subtraction, bright (flat-field divide OR subtract) and background.

    Order: (img − dark) → bright → (− background) → clip≥0.  For divide mode the
    flat field is dark-corrected too: out / (bright − dark) × mean(bright − dark).
    Returns float64.  Any field may be None.
    """
    out = np.asarray(img, dtype=np.float64)
    d = None if dark is None else np.asarray(dark, dtype=np.float64)
    if d is not None:
        out = out - d
    if bright is not None:
        b = np.asarray(bright, dtype=np.float64)
        if d is not None:
            b = b - d
        if bright_mode == "subtract":
            out = out - b
        else:  # flat-field divide, rescaled to preserve counts
            b = np.clip(b, 1e-9, None)
            out = out / b * float(np.mean(b))
    if background is not None:
        out = out - np.asarray(background, dtype=np.float64)
    if clip_negative:
        out = np.clip(out, 0.0, None)
    return out


# ── Ring prediction (calibrant → ring radii in px) ──────────────────────────────

def _predict_ring_radii(result) -> list:
    """Predicted ring radii (px) for the result's calibrant geometry."""
    d_list = getattr(result, "_d_list", None)
    if d_list:
        try:
            rings = simulate_rings_from_dspacings(
                d_list, result.wavelength_A, result.Lsd, result.pxY)
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
                             wavelength_A=result.wavelength_A, two_theta_max_deg=30.0)
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
                                 pxZ_um: float, seed=None) -> dict:
    """Fit detector Lsd + beam center (BC_y, BC_z) from user-picked ring points
    and their known d-spacings, bypassing any crystallographic calibrant
    backend entirely (tilt is fixed at 0 — a plain untilted Debye-Scherrer
    geometry). ``picks`` is an iterable of (Y_px, Z_px, d_spacing_A) triples;
    points on the same ring share the same d-spacing value.

    For each picked point, the *observed* 2theta implied by a trial geometry
    is ``atan2(r_px, Lsd)`` where ``r_px`` is the radial pixel distance from
    the trial beam center (this is the untilted case of the same forward
    model used by :func:`tilted_ring_xy`/``_draw_corrected_rings``). The
    residual against the *expected* 2theta from Bragg's law
    (``2*asin(wavelength/(2*d))``) is minimized over (Lsd, BC_y, BC_z).

    Returns a dict with ``Lsd``, ``BC_y``, ``BC_z`` (µm/px), ``residual_deg_rms``,
    ``success``, ``message``, and ``seed_quality`` ("ok"/"fallback"/"given").
    """
    from scipy.optimize import least_squares
    picks = list(picks)
    pts = np.array([(p[0], p[1]) for p in picks], dtype=np.float64)
    d = np.array([p[2] for p in picks], dtype=np.float64)
    s = np.clip(wavelength_A / (2.0 * d), -1.0, 1.0)
    two_theta_calc = 2.0 * np.degrees(np.arcsin(s))

    if seed is None:
        lsd0, bcy0, bcz0, seed_quality = _auto_seed_from_picks(
            picks, wavelength_A, pxY_um, pxZ_um)
    else:
        lsd0, bcy0, bcz0 = seed
        seed_quality = "given"

    def resid(p):
        Lsd, bc_y, bc_z = p
        Yc = (bc_y - pts[:, 0]) * pxY_um
        Zc = (pts[:, 1] - bc_z) * pxZ_um
        r = np.hypot(Yc, Zc)
        two_theta_obs = np.degrees(np.arctan2(r, Lsd))
        return two_theta_obs - two_theta_calc

    sol = least_squares(resid, [lsd0, bcy0, bcz0], method="lm")
    Lsd, bc_y, bc_z = (float(v) for v in sol.x)
    return {
        "Lsd": Lsd, "BC_y": bc_y, "BC_z": bc_z,
        "residual_deg_rms": float(np.sqrt(np.mean(sol.fun ** 2))),
        "success": bool(sol.success), "message": str(sol.message),
        "seed_quality": seed_quality,
    }


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
                    pxY_um: float, pxZ_um: float, n: int = 400):
    """Forward-project a diffraction ring at ``two_theta_deg`` through the tilt
    geometry (tx/ty/tz, degrees) onto detector pixel coordinates.

    Returns ``(Y_px, Z_px)`` arrays of length ``n`` tracing the ring, with the
    last point equal to the first so the polyline a caller feeds straight
    into ``pg.PlotDataItem`` closes with no seam (``pyqtgraph`` never
    auto-closes a plain polyline back to its start). Reduces exactly to the
    plain circle ``bc + r*(sin η, cos η)`` when tx=ty=tz=0, since it inverts
    the same ray/tilt-plane geometry as
    ``midas_integrate_v2.forward.pixels.pixel_to_REta_from_spec``.
    """
    eta = np.linspace(0.0, 360.0, n, endpoint=True)
    return _tilt_project_YZ(np.full(n, two_theta_deg), eta, tx, ty, tz,
                             Lsd_um, bc_y, bc_z, pxY_um, pxZ_um)


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
                           pxZ_um: Optional[float] = None) -> None:
    """If ``show_grid``, draw the Rmin/Rmax exclusion-boundary circles
    (skipping any non-positive radius) plus the full polar (R, η) bin
    grid — concentric circles at each radial-bin edge plus spokes at each
    η-bin edge, both thinned to at most ``max_rings``/``max_spokes`` — onto
    ``viewer._iv``. Nothing is drawn when ``show_grid`` is false.

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
    for it in items:
        viewer._iv.removeItem(it)
    items.clear()
    if r_max <= 0:
        return
    tilt_aware = (bool(lsd_um) and bool(pxY_um) and bool(pxZ_um)
                  and (abs(tx) > 1e-9 or abs(ty) > 1e-9 or abs(tz) > 1e-9))
    th = np.linspace(0, 2 * math.pi, 256)

    def _ring_xy(r):
        if tilt_aware:
            two_theta = math.degrees(math.atan(r * pxY_um / lsd_um))
            return tilted_ring_xy(two_theta, tx, ty, tz, lsd_um, bc_y, bc_z,
                                   pxY_um, pxZ_um, n=256)
        return bc_y + r * np.cos(th), bc_z + r * np.sin(th)

    def _circle(r, pen):
        Y, Z = _ring_xy(r)
        item = pg.PlotDataItem(Y, Z, pen=pen)
        viewer._iv.addItem(item)
        items.append(item)

    if not show_grid:
        return

    if r_min > 0:
        _circle(r_min, pg.mkPen("orange", width=1.2, style=QtCore.Qt.DashLine))
    _circle(r_max, pg.mkPen("orange", width=1.5))
    grid_pen = pg.mkPen((120, 180, 255), width=0.8)
    for r in _thinned_bin_edges(r_min, r_max, r_bin, max_rings):
        if r_min < r < r_max:
            _circle(r, grid_pen)
    eta_edges = _thinned_bin_edges(eta_min, eta_max, e_bin, max_spokes)
    if eta_max - eta_min >= 360.0 - 1e-6:
        eta_edges = eta_edges[eta_edges < eta_max - 1e-9]   # drop the wraparound duplicate
    for eta in eta_edges:
        if tilt_aware:
            tt_lo = math.degrees(math.atan(r_min * pxY_um / lsd_um))
            tt_hi = math.degrees(math.atan(r_max * pxY_um / lsd_um))
            Y, Z = tilted_spoke_xy(tt_lo, tt_hi, float(eta), tx, ty, tz,
                                    lsd_um, bc_y, bc_z, pxY_um, pxZ_um)
        else:
            # bc + r*(sin η, cos η) — same convention tilted_spoke_xy reduces
            # to at zero tilt (and pixel_to_REta's eta=atan2(-Yc,Zc)): η=0 is
            # straight up (+Z), not along +Y. cos/sin here (not sin/cos)
            # would draw each η spoke 90° off from where it actually is.
            th_r = math.radians(float(eta))
            Y = [bc_y + r_min * math.sin(th_r), bc_y + r_max * math.sin(th_r)]
            Z = [bc_z + r_min * math.cos(th_r), bc_z + r_max * math.cos(th_r)]
        item = pg.PlotDataItem(Y, Z, pen=grid_pen)
        viewer._iv.addItem(item); items.append(item)


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
    return spec


def _spec_from_json(path: str, r_bin: float, eta_bin: float):
    from midas_calibrate_v2.compat.to_integrate import spec_from_calibration_json
    return spec_from_calibration_json(path, RBinSize=r_bin, EtaBinSize=eta_bin)


# v1 paramstest index (p#) → v2 harmonic name — the inverse of the single source
# of truth ``constants._V2_TO_V1`` (avoids a hand-maintained second copy).
_PARAMSTEST_DISTORTION = {v1: v2 for v2, v1 in _V2_TO_V1.items()}


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
        Wavelength=result.wavelength_A, SpaceGroup=_SG.get(cal, 225),
        LatticeConstant=_LC.get(cal, _LC["CeO2"]), RhoD=RhoD, MaxRingRad=RhoD * 0.97)
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
            im_trans=c.get("im_trans", []),
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


# ── Log stream (redirect verbose stdout to a Qt signal) ─────────────────────────

class _LogStream(io.TextIOBase):
    def __init__(self, sig):
        super().__init__()
        self._sig = sig

    def write(self, s):
        if s.strip():
            self._sig.emit(s.rstrip())
        return len(s)

    def flush(self):
        pass


# ── GUI-state serialization (Save/Load GUI State) ────────────────────────────────

def widgets_to_dict(widgets: dict) -> dict:
    """Snapshot a ``{key: widget}`` map into a plain JSON-able dict, by widget type:
    spin boxes → ``.value()``, combo boxes → current text, line edits → ``.text()``,
    checkable buttons → ``.isChecked()``. Unrecognized widget types are skipped."""
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


def _browse(parent, caption, filt) -> str:
    p, _ = QtWidgets.QFileDialog.getOpenFileName(parent, caption, "", filt)
    return p
