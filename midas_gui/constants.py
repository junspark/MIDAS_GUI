"""Shared constants: calibrants, colormaps, dtype sentinels, distortion mapping,
lattice / space-group tables, and output-format lists.

Built-in values below are the shipped defaults; a per-user / per-group config
(JSON) is overlaid onto them at the bottom of this module — see
:mod:`midas_gui.settings`."""
from __future__ import annotations

import os

CALIBRANTS = ["CeO2", "LaB6", "Si", "Al2O3"]
COLORMAPS  = ["hot", "gray", "viridis", "inferno", "plasma", "turbo"]

# Output formats for the batch tab (label → short key)
OUTPUT_FORMATS = {
    "CSV  (R, I, σ)":          "csv",
    "XYE  (2θ, I, σ)":        "xye",
    "FXYE (centideg, I, σ)":   "fxye",
    "DAT  (Q, I, σ)":          "dat",
    "HDF5 (full stack)":        "h5",
    "2D CSV (cake, η×R)":       "2d_csv",
    "Zarr (cake, REtaMap)":     "zarr",
}

# Integration kernels (label → key)
KERNELS = {
    "Subpixel K=2 (default)": "subpixel2",
    "Subpixel K=4":           "subpixel4",
    "Hard bin (fastest)":     "hard",
    "Polygon (exact, slow)":  "polygon",
}

# Calibration pipelines (label → key, enabled?)
PIPELINES = [
    ("One-shot (default)",       "one_shot",   True),
    ("First-time (no prior)",    "first_time", True),
    ("Four-stage (patchy det.)", "four_stage", True),
    ("Bayesian MAP+Laplace",     "bayesian",   True),    # Phase 2
    ("Joint cake",               "joint",      True),    # Phase 3
    ("Multi-distance",           "multi",      False),   # multi-image, deferred
]

ERROR_MODELS = ["poisson", "azimuthal", "hybrid"]

# HDF5-like extensions (auto-show dataset field)
H5_EXTS = {".h5", ".hdf5", ".hdf", ".nxs"}

# Dtype → saturation sentinel (pixels at or above are considered dead/saturated)
_SENTINELS = {
    "uint8":  2**8  - 1,
    "uint16": 2**16 - 1,
    "uint32": 2**32 - 2,   # 2^32-1 is the Eiger dead-pixel value
    "int16":  2**15 - 1,
    "int32":  2**31 - 1,
}

# v2 distortion name → v1 paramstest p-slot
_V2_TO_V1 = {
    "iso_R2": "p2", "iso_R4": "p5", "iso_R6": "p4",
    "a1": "p7", "phi1": "p8", "a2": "p0", "phi2": "p6",
    "a3": "p9", "phi3": "p10", "a4": "p1", "phi4": "p3",
    "a5": "p11", "phi5": "p12", "a6": "p13", "phi6": "p14",
}

# Canonical ordering of the 15 distortion coefficients for display
DISTORTION_NAMES = [
    "iso_R2", "iso_R4", "iso_R6",
    "a1", "phi1", "a2", "phi2", "a3", "phi3",
    "a4", "phi4", "a5", "phi5", "a6", "phi6",
]

# Isotropic radial coefficients (η-fold 0) and the per-fold amp/phase pairs.
DISTORTION_ISO = ["iso_R2", "iso_R4", "iso_R6"]
def _fold(k):  # η-fold k (1..6) → its amplitude/phase pair
    return [f"a{k}", f"phi{k}"]

# Named distortion-refinement "modes": each selects a set of the 15 coefficients.
# Grouped by η-fold, matching how MIDAS names harmonics (bic_search ladder).
DISTORTION_PRESETS = {
    "None": [],
    "Isotropic only": list(DISTORTION_ISO),
    "Iso + up to 2-fold": DISTORTION_ISO + _fold(1) + _fold(2),
    "Iso + up to 4-fold": DISTORTION_ISO + _fold(1) + _fold(2) + _fold(3) + _fold(4),
    "All (15)": list(DISTORTION_NAMES),
}

# Lattice + space group per built-in calibrant.
# _LATT used for ring prediction; _SG / _LC used for paramstest export.
# Calibrant lattice constants — kept numerically identical to the matching
# MATERIALS entries below (single set of values per phase; do not let them drift).
_LATT = {
    "CeO2":  dict(a=5.4116,  b=5.4116,  c=5.4116,  alpha=90, beta=90, gamma=90,  sg=225),
    "LaB6":  dict(a=4.15692, b=4.15692, c=4.15692, alpha=90, beta=90, gamma=90,  sg=221),
    "Si":    dict(a=5.43102, b=5.43102, c=5.43102, alpha=90, beta=90, gamma=90,  sg=227),
    "Al2O3": dict(a=4.7589,  b=4.7589,  c=12.992,  alpha=90, beta=90, gamma=120, sg=167),
}

_SG = {"CeO2": 225, "LaB6": 221, "Si": 227, "Al2O3": 167}

_LC = {
    "CeO2":  (5.4116,  5.4116,  5.4116,  90.0, 90.0,  90.0),
    "LaB6":  (4.15692, 4.15692, 4.15692, 90.0, 90.0,  90.0),
    "Si":    (5.43102, 5.43102, 5.43102, 90.0, 90.0,  90.0),
    "Al2O3": (4.7589,  4.7589,  12.992,  90.0, 90.0, 120.0),
}

# Default detector parameters — set to the synthetic test_data geometry
# (Eiger2 500K: 75 µm pixels, λ=0.39 Å, Lsd=121 mm) for easy out-of-the-box testing.
DEFAULT_WAVELENGTH = 0.39      # Å
DEFAULT_PIXEL_UM   = 75.0      # µm
DEFAULT_LSD_UM     = 121_000.0 # µm
DEFAULT_BC_Y       = 10.0      # px (test data beam centre)
DEFAULT_BC_Z       = 10.0      # px
DEFAULT_RING_WIDTH = 2.0       # px (simulated-ring pen width on the image)

# Data Viewer ▸ Ring simulation card: spin-box step size (up/down-arrow increment)
# per field. Overridable via the per-user config key "viewer_steps"
# (Preferences ▸ Data Viewer).
DEFAULT_STEP_WAVELENGTH = 0.01   # Å      (λ)
DEFAULT_STEP_TWO_THETA  = 1.0    # deg    (max 2θ)
DEFAULT_STEP_LSD_MM     = 1.0    # mm     (Lsd)
DEFAULT_STEP_PIXEL      = 0.1    # µm     (pixel size)
DEFAULT_STEP_BC         = 1.0    # px     (BC_y / BC_z)
DEFAULT_STEP_TILT       = 0.1    # deg    (ty / tz)

# Photon energy ↔ wavelength: λ(Å) = HC_KEV_A / E(keV)
HC_KEV_A = 12.398420

# Common detector pixel sizes (label, pixel size in µm) for the clickable "px" menu.
PIXEL_PRESETS = [
    ("GE", 200.0), ("Varex", 150.0), ("Pilatus", 172.0), ("Eiger", 75.0),
]

# Common K-edge foil energies (element symbol, K absorption-edge energy in keV).
# Selecting one sets the wavelength to the edge (λ = HC_KEV_A / E). Ordered by
# increasing energy / decreasing wavelength.
K_EDGE_FOILS = [
    ("Pr", 41.991), ("Sm", 46.834), ("Yb", 61.332), ("Lu", 63.314),
    ("Hf", 65.351), ("Ta", 67.416), ("W", 69.525),  ("Re", 71.676),
    ("Pt", 78.395), ("Au", 80.725), ("Pb", 88.005), ("Bi", 90.526),
]

# ── Default test-data paths (repo-root test_data/, git-ignored) ─────────────────
from pathlib import Path as _Path
# The midas_gui project root (parent of the `midas_gui` package itself) —
# default starting folder for dialogs.BrowseFilesDialog.
PROJECT_ROOT = _Path(__file__).resolve().parent.parent
_TEST_DATA = PROJECT_ROOT / "test_data"
# Synthetic GUI fixtures live under test_data/gui_synthetic/ (moved there by the
# 2026-08-23 per-dataset reorg — these six defaults must track that subfolder).
_GUI_SYNTH = _TEST_DATA / "gui_synthetic"
DEFAULT_CALIBRANT_TIF = str(_GUI_SYNTH / "calibrant_ceria.tif")
DEFAULT_CALIBRANT_H5  = str(_GUI_SYNTH / "calibrant_ceria.h5")
DEFAULT_NICKEL_H5     = str(_GUI_SYNTH / "nickel_stack.h5")
DEFAULT_NICKEL_DIR    = str(_GUI_SYNTH / "nickel_tifs")
DEFAULT_NICKEL_FRAME0 = str(_GUI_SYNTH / "nickel_tifs" / "nickel_000.tif")
DEFAULT_CALIB_FILE    = str(_GUI_SYNTH / "calibration_synthetic.json")
# Real PDF workflow test data (test_data/test_pdf/, git-ignored — ~320 MB of
# raw beamline frames). Absent on a fresh checkout; the PDF tab just starts
# with nothing preloaded until a user points it elsewhere.
DEFAULT_PDF_DIR       = str(_TEST_DATA / "test_pdf")
DEFAULT_PDF_IQ_FILE   = str(_TEST_DATA / "test_pdf" / "iq" / "04_iq_Nickel.csv")
DEFAULT_PDF_CALIB     = str(_TEST_DATA / "test_pdf" / "calibration" / "03_ceo2_v1_params.txt")
DEFAULT_PDF_RAW_FRAME = str(_TEST_DATA / "test_pdf" / "raw" / "Nickel_004526.vrx.h5")
DEFAULT_PDF_MASK      = str(_TEST_DATA / "test_pdf" / "calibration" / "mask_pdf.tif")
DEFAULT_PDF_EMPTY_IQ  = str(_TEST_DATA / "test_pdf" / "iq" / "04_iq_airscatter.csv")
DEFAULT_PDF_CIF       = str(_TEST_DATA / "test_pdf" / "structures" / "Ni.cif")


# ── Materials database for ring simulation (Tab 0) ──────────────────────────────
# name → dict(a, b, c, alpha, beta, gamma [Å, deg], sg [space-group number]).
# Calibrants first, then common cubic metals / phases.
MATERIALS = {
    "CeO2 (calibrant)":  dict(a=5.4116, b=5.4116, c=5.4116,  alpha=90, beta=90, gamma=90,  sg=225),
    "LaB6 (calibrant)":  dict(a=4.15692, b=4.15692, c=4.15692, alpha=90, beta=90, gamma=90, sg=221),
    "Si (calibrant)":    dict(a=5.43102, b=5.43102, c=5.43102, alpha=90, beta=90, gamma=90, sg=227),
    "Al2O3 (corundum)":  dict(a=4.7589, b=4.7589, c=12.992,  alpha=90, beta=90, gamma=120, sg=167),
    "Cu (FCC)":          dict(a=3.6149, b=3.6149, c=3.6149,  alpha=90, beta=90, gamma=90,  sg=225),
    "Ni (FCC)":          dict(a=3.5238, b=3.5238, c=3.5238,  alpha=90, beta=90, gamma=90,  sg=225),
    "FCC steel (γ-Fe)":  dict(a=3.595,  b=3.595,  c=3.595,   alpha=90, beta=90, gamma=90,  sg=225),
    "BCC steel (α-Fe)":  dict(a=2.8665, b=2.8665, c=2.8665,  alpha=90, beta=90, gamma=90,  sg=229),
    "Au (FCC)":          dict(a=4.0782, b=4.0782, c=4.0782,  alpha=90, beta=90, gamma=90,  sg=225),
    "Ag (FCC)":          dict(a=4.0853, b=4.0853, c=4.0853,  alpha=90, beta=90, gamma=90,  sg=225),
    "Pt (FCC)":          dict(a=3.9242, b=3.9242, c=3.9242,  alpha=90, beta=90, gamma=90,  sg=225),
    "W (BCC)":           dict(a=3.16525, b=3.16525, c=3.16525, alpha=90, beta=90, gamma=90, sg=229),
    "Ti (HCP)":          dict(a=2.9508, b=2.9508, c=4.6855,  alpha=90, beta=90, gamma=120, sg=194),
    # Non-crystalline standard: a lamellar/smectic structure, not a 3-D
    # lattice, so it's defined directly by its d-spacings rather than
    # space-group + lattice parameters. d(001) = 58.380 A is the standard
    # literature/NIST-traceable long spacing for silver behenate; harmonics
    # 1-10 cover the practically observable SAXS/WAXS range.
    "AgBH (silver behenate)": dict(kind="dspacing", d_list=[58.380 / n for n in range(1, 11)]),
}


# ── UI / algorithm defaults (overridable) — first option of each list ───────────
DEFAULT_KERNEL        = "subpixel2"   # key in KERNELS (integration algorithm)
DEFAULT_PIPELINE      = "one_shot"    # key in PIPELINES (calibration algorithm)
DEFAULT_OUTPUT_FORMAT = "csv"         # key in OUTPUT_FORMATS
DEFAULT_ERROR_MODEL   = "poisson"     # value in ERROR_MODELS
DEFAULT_COLORMAP      = "hot"         # value in COLORMAPS

# Interface scale (whole-app zoom for HiDPI / 4K monitors). Applied at startup via
# Qt's QT_SCALE_FACTOR, so the entire layout + fonts scale uniformly. 1.0 suits a
# 1080p display; try ~1.5 for 1440p and ~2.0 for 4K. Overridable via ui.ui_scale.
DEFAULT_UI_SCALE      = 1.0           # multiplier, clamped to [0.5, 4.0] at startup

# ── Modular tabs ────────────────────────────────────────────────────────────────
# ALWAYS_TABS are pinned (cannot be hidden); OPTIONAL_TABS can be toggled from
# Preferences. Names must match the base tab labels used in app.py. DEFAULT_VISIBLE_TABS
# is the subset of OPTIONAL_TABS shown at startup (all optional tabs ship enabled);
# it is overridable via the per-user config key ``ui.visible_tabs``.
ALWAYS_TABS = ["Data Viewer", "Mask Builder", "Calibrate", "Batch Integrate"]
OPTIONAL_TABS = ["Calib. Refinement", "Corrections", "PDF Analysis", "Texture",
                 "Pump Probe", "Results & Export"]
# Optional tabs shown by default. Corrections / PDF Analysis / Texture / Results &
# Export ship hidden (turn them on in Settings ▸ Preferences ▸ Tabs).
DEFAULT_VISIBLE_TABS = ["Calib. Refinement", "Pump Probe"]

# ── Beamline devices (Data Viewer ▸ Live Data PV dropdown) ──────────────────────
# Default detector devices for 20-ID-D, extracted from the beamline's area-detector
# device-generation blueprint (20-ID-D `make_det(...)` calls). ``pva_suffix`` is the
# PVA image plugin every one of them exposes for live access; the full live PV is
# `prefix + pva_suffix`. Overridable via the per-user config key "devices"
# (Preferences ▸ Devices).
#
# "Sim Detector" is not a real beamline device: its PV (midasSim:Pva1:Image) is
# recognized by DataLoaderPanel._start_live (widgets.py), which lazily spins up
# an in-process midas_gui.sim_detector.SimDetectorServer on first connect — a
# fake Eiger2-500K-shaped PVA stream for exercising Live Data with no hardware.
DEFAULT_DEVICES = [
    {"name": "20iddNF",       "prefix": "20idOR1:",  "pva_suffix": "Pva1:Image"},
    {"name": "s20idPil",      "prefix": "20idPil:",  "pva_suffix": "Pva1:Image"},
    {"name": "pg4",           "prefix": "1idPG4:",   "pva_suffix": "Pva1:Image"},
    {"name": "20iddTomo",     "prefix": "20idGH1s:", "pva_suffix": "Pva1:Image"},
    {"name": "20iddFF",       "prefix": "20IDFF:",   "pva_suffix": "Pva1:Image"},
    {"name": "Sim Detector",  "prefix": "midasSim:", "pva_suffix": "Pva1:Image"},
]
DEVICES = [dict(d) for d in DEFAULT_DEVICES]

# ── Pump Probe (TR-XRD) defaults — TRR-group time-resolved test data ────────────
# The frames (~1.2 GB) live in a local-only, git-ignored folder under test_data/
# (test_data/trr_s7id/pump_probe_BTO/) so the large detector stack never enters
# git. See .gitignore and documentation/gui_documentation.md.
_TEST_DATA_PP        = _TEST_DATA / "trr_s7id" / "pump_probe_BTO"
DEFAULT_TRXRD_DIR    = str(_TEST_DATA_PP / "detimages")
DEFAULT_TRXRD_PREFIX = "Ex01_Sa01_Sc17"
# MIDAS calibration for the TRR Pilatus2M data (converted from its pyFAI/Fit2D
# geometry). Used as the default "From file" calibration in the Pump Probe tab.
DEFAULT_TRXRD_CALIB  = str(_TEST_DATA_PP / "Ex01_Sa01_Sc17_midas.txt")
# Default detector mask for the TRR data: 0 = valid pixel, 1 = bad pixel / module gap
# (MIDAS convention → non-zero is masked). invert_mask.tif matches this; its sibling
# mask_2021_dec.tif is the inverse (1 = valid) and must not be used as-is.
DEFAULT_TRXRD_MASK   = str(_TEST_DATA_PP / "invert_mask.tif")


# ── User / group config overlay ─────────────────────────────────────────────────
def _coerce_material(d: dict) -> dict:
    """Validate + normalise a material dict (raises on missing/invalid keys).

    Two kinds: "lattice" (default, a/b/c/alpha/beta/gamma/sg — ring radii
    computed via space-group crystallography) and "dspacing" (a bare list
    of d-spacings in Angstrom, for non-crystalline standards like silver
    behenate that have no space group to derive rings from)."""
    if d.get("kind") == "dspacing":
        d_list = [float(x) for x in d["d_list"]]
        if not d_list:
            raise ValueError("d_list must have at least one d-spacing")
        return {"kind": "dspacing", "d_list": d_list}
    m = {k: float(d[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}
    m["sg"] = int(d["sg"])
    return m


def _apply(cfg: dict) -> None:
    """Overlay a user/group config dict onto this module's defaults.

    Every domain is individually guarded so a single bad entry is skipped rather
    than breaking startup (this runs on every import of ``constants``).
    """
    if not cfg:
        return
    g = globals()

    def _set_num(section, key, target, cast=float):
        try:
            if section.get(key) is not None:
                g[target] = cast(section[key])
        except Exception:
            pass

    # geometry scalars
    geo = cfg.get("geometry", {}) or {}
    _set_num(geo, "wavelength_A", "DEFAULT_WAVELENGTH")
    _set_num(geo, "pixel_um", "DEFAULT_PIXEL_UM")
    _set_num(geo, "lsd_um", "DEFAULT_LSD_UM")
    _set_num(geo, "bc_y", "DEFAULT_BC_Y")
    _set_num(geo, "bc_z", "DEFAULT_BC_Z")

    # Data Viewer spin-box step sizes
    steps = cfg.get("viewer_steps", {}) or {}
    _set_num(steps, "wavelength", "DEFAULT_STEP_WAVELENGTH")
    _set_num(steps, "two_theta", "DEFAULT_STEP_TWO_THETA")
    _set_num(steps, "lsd_mm", "DEFAULT_STEP_LSD_MM")
    _set_num(steps, "pixel", "DEFAULT_STEP_PIXEL")
    _set_num(steps, "bc", "DEFAULT_STEP_BC")
    _set_num(steps, "tilt", "DEFAULT_STEP_TILT")

    # pixel presets / K-edge foils — replace wholesale if provided (per-entry guard)
    def _pairs(items):
        out = []
        for it in items or []:
            try:
                out.append((str(it[0]), float(it[1])))
            except Exception:
                pass
        return out
    if isinstance(geo.get("pixel_presets"), list):
        pp = _pairs(geo["pixel_presets"])
        if pp:
            g["PIXEL_PRESETS"] = pp
    if isinstance(geo.get("k_edge_foils"), list):
        ke = _pairs(geo["k_edge_foils"])
        if ke:
            g["K_EDGE_FOILS"] = ke

    # default paths (expand ~ and $VARS)
    _pmap = {
        "calibrant_tif": "DEFAULT_CALIBRANT_TIF", "calibrant_h5": "DEFAULT_CALIBRANT_H5",
        "nickel_h5": "DEFAULT_NICKEL_H5", "nickel_dir": "DEFAULT_NICKEL_DIR",
        "nickel_frame0": "DEFAULT_NICKEL_FRAME0", "calib_file": "DEFAULT_CALIB_FILE",
        "pdf_iq_file": "DEFAULT_PDF_IQ_FILE", "pdf_calib": "DEFAULT_PDF_CALIB",
        "pdf_raw_frame": "DEFAULT_PDF_RAW_FRAME", "pdf_mask": "DEFAULT_PDF_MASK",
        "pdf_empty_iq": "DEFAULT_PDF_EMPTY_IQ", "pdf_cif": "DEFAULT_PDF_CIF",
    }
    paths = cfg.get("paths", {}) or {}
    for key, target in _pmap.items():
        try:
            v = paths.get(key)
            if v:
                g[target] = os.path.expandvars(os.path.expanduser(str(v)))
        except Exception:
            pass

    # materials — the config's list REPLACES the built-in list when present
    # (so the Preferences dialog, pre-filled with the shipped list, can add /
    # remove / modify).  Guard: a non-empty but fully-unparseable list is ignored.
    if isinstance(cfg.get("materials"), dict):
        parsed = {}
        for name, d in cfg["materials"].items():
            try:
                parsed[str(name)] = _coerce_material(d)
            except Exception:
                pass
        if parsed or cfg["materials"] == {}:
            MATERIALS.clear(); MATERIALS.update(parsed)

    # calibrants — likewise replace the CALIBRANTS dropdown list when present;
    # _LATT/_SG/_LC are updated (not cleared) so lookups by name still resolve.
    if isinstance(cfg.get("calibrants"), dict):
        names = []
        for name, d in cfg["calibrants"].items():
            try:
                m = _coerce_material(d); nm = str(name)
                _LATT[nm] = dict(m)
                _SG[nm] = m["sg"]
                _LC[nm] = (m["a"], m["b"], m["c"], m["alpha"], m["beta"], m["gamma"])
                names.append(nm)
            except Exception:
                pass
        if names or cfg["calibrants"] == {}:
            CALIBRANTS[:] = names

    # devices — the config's list REPLACES the built-in list when present (same
    # pattern as materials/calibrants), so Preferences ▸ Devices, pre-filled with
    # the shipped list, can add / remove / modify.
    if isinstance(cfg.get("devices"), list):
        parsed = []
        for d in cfg["devices"]:
            try:
                name = str(d.get("name", "")).strip()
                if name:
                    parsed.append({"name": name,
                                    "prefix": str(d.get("prefix", "")).strip(),
                                    "pva_suffix": str(d.get("pva_suffix", "")).strip()})
            except Exception:
                pass
        if parsed or cfg["devices"] == []:
            DEVICES[:] = parsed

    # UI / algorithm defaults
    ui = cfg.get("ui", {}) or {}
    for key, target in (("integration_kernel", "DEFAULT_KERNEL"),
                        ("calibration_pipeline", "DEFAULT_PIPELINE"),
                        ("output_format", "DEFAULT_OUTPUT_FORMAT"),
                        ("azimuthal_method", "DEFAULT_ERROR_MODEL"),
                        ("plot_theme", "DEFAULT_COLORMAP")):
        try:
            if ui.get(key):
                g[target] = str(ui[key])
        except Exception:
            pass
    # visible optional tabs — replace wholesale if a list is provided
    try:
        if isinstance(ui.get("visible_tabs"), list):
            g["DEFAULT_VISIBLE_TABS"] = [str(x) for x in ui["visible_tabs"]]
    except Exception:
        pass
    # interface scale (float multiplier)
    try:
        if ui.get("ui_scale") is not None:
            g["DEFAULT_UI_SCALE"] = float(ui["ui_scale"])
    except Exception:
        pass


# Pristine snapshot of the shipped defaults (captured BEFORE any overlay), so the
# GUI can always offer / restore the built-in values regardless of the user config.
import copy as _copy

_SHIPPED = {
    "geometry": {
        "wavelength_A": DEFAULT_WAVELENGTH, "pixel_um": DEFAULT_PIXEL_UM,
        "lsd_um": DEFAULT_LSD_UM, "bc_y": DEFAULT_BC_Y, "bc_z": DEFAULT_BC_Z,
        "pixel_presets": [list(p) for p in PIXEL_PRESETS],
        "k_edge_foils": [list(k) for k in K_EDGE_FOILS],
    },
    "viewer_steps": {
        "wavelength": DEFAULT_STEP_WAVELENGTH, "two_theta": DEFAULT_STEP_TWO_THETA,
        "lsd_mm": DEFAULT_STEP_LSD_MM, "pixel": DEFAULT_STEP_PIXEL,
        "bc": DEFAULT_STEP_BC, "tilt": DEFAULT_STEP_TILT,
    },
    "materials": {n: dict(m) for n, m in MATERIALS.items()},
    "calibrants": {n: dict(_LATT[n]) for n in CALIBRANTS if n in _LATT},
    "devices": [dict(d) for d in DEFAULT_DEVICES],
    "paths": {
        "calibrant_tif": DEFAULT_CALIBRANT_TIF, "calibrant_h5": DEFAULT_CALIBRANT_H5,
        "nickel_h5": DEFAULT_NICKEL_H5, "nickel_dir": DEFAULT_NICKEL_DIR,
        "nickel_frame0": DEFAULT_NICKEL_FRAME0, "calib_file": DEFAULT_CALIB_FILE,
        "pdf_iq_file": DEFAULT_PDF_IQ_FILE, "pdf_calib": DEFAULT_PDF_CALIB,
        "pdf_raw_frame": DEFAULT_PDF_RAW_FRAME, "pdf_mask": DEFAULT_PDF_MASK,
        "pdf_empty_iq": DEFAULT_PDF_EMPTY_IQ, "pdf_cif": DEFAULT_PDF_CIF,
    },
    "ui": {
        "integration_kernel": DEFAULT_KERNEL, "calibration_pipeline": DEFAULT_PIPELINE,
        "output_format": DEFAULT_OUTPUT_FORMAT, "azimuthal_method": DEFAULT_ERROR_MODEL,
        "plot_theme": DEFAULT_COLORMAP, "visible_tabs": list(DEFAULT_VISIBLE_TABS),
        "ui_scale": DEFAULT_UI_SCALE,
    },
}


def shipped_defaults() -> dict:
    """Return a deep copy of the pristine shipped defaults (pre-overlay)."""
    return _copy.deepcopy(_SHIPPED)


def reload_from_config() -> None:
    """Reset every overlay-able global to its shipped default, then re-apply the
    (possibly just-switched-to) active profile's config on top.

    Plain re-running :func:`_apply` on a new config is not enough on its own:
    ``_apply`` only *overlays* keys that are present, so a value the previous
    profile overrode but the new profile doesn't mention would otherwise keep
    the old profile's value instead of reverting to the shipped default. Used
    by Preferences ▸ Profile switching (and by the "Reload config" menu item)
    to refresh live ``DEFAULT_*`` globals without requiring a restart.
    """
    _apply(shipped_defaults())
    try:
        from midas_gui import settings as _settings
        _apply(_settings.load_config(force=True))
    except Exception:
        pass


try:
    reload_from_config()
except Exception:
    pass
