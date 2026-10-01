"""Calibration pipeline layer.

The GUI offers several calibration pipelines, but only ``calibrate()`` and
``first_time_calibrate()`` accept a raw image.  The other pipelines
(``autocalibrate_four_stage`` etc.) take a pre-built ``V1Params`` object plus an
explicit seed, and each returns a *different* result type whose geometry lives in
a different attribute.

This module hides that heterogeneity behind two functions:

* :func:`run_pipeline` — dispatch on a mode string, returning whatever the
  underlying pipeline returns.
* :func:`normalize_result` — convert any pipeline output into a real
  :class:`AutoCalibrationResult`, so the rest of the GUI (spec building, ring
  drawing, paramstest export, results display) never needs to know which
  pipeline ran.

The unpacked-dict → AutoCalibrationResult mapping mirrors the canonical wrapping
in ``midas_calibrate_v2/pipelines/auto.py`` (the body of ``calibrate()``).
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

import midas_gui._paths  # noqa: F401  (sys.path setup must run before MIDAS imports)
from midas_gui.constants import _SG, _LC, _V2_TO_V1, DISTORTION_NAMES, DEFAULT_LSD_UM


def _supported_kwargs(fn, kwargs: dict) -> dict:
    """Drop kwargs the callable ``fn`` doesn't accept, so the GUI stays compatible
    across MIDAS-backend versions whose signatures differ (e.g. midas-calibrate-v2
    0.3.3 has no ``initial_BC_y`` / ``initial_BC_z`` beam-centre seed). If ``fn``
    takes ``**kwargs`` nothing is dropped. Dropped names are logged to stdout (which
    the calibration worker relays to the GUI log)."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    out = {k: v for k, v in kwargs.items() if k in params}
    dropped = [k for k in kwargs if k not in params]
    if dropped:
        print(f"[calib] note: {getattr(fn, '__name__', 'callable')}() does not accept "
              f"{', '.join(dropped)} in this backend version — ignoring.")
    return out


# ── Seed ────────────────────────────────────────────────────────────────────────

def _prep_transformed(image: np.ndarray, dark, im_trans: tuple):
    """Apply ``im_trans`` to image + dark once, together, before any seeding
    or solving happens — mirrors ``midas_calibrate_v2.calibrate()``'s own
    internal transform handling (it flips image AND dark, then derives
    NrPixelsY/Z, then seeds, all from the same transformed array).

    Needed only because ``autocalibrate_four_stage`` / ``_bayesian`` /
    ``_joint`` / ``pipelines.single.autocalibrate`` have no native
    ``im_trans`` parameter (see ROADMAP P3-1) — every caller of this helper
    must use its returned ``img``/``dark``/``NY``/``NZ`` for everything
    downstream (seed AND solve), never the original ``image``, or the seed
    and the solve run in two different frames.
    """
    img = image.astype(np.float32)
    dk = dark
    if im_trans:
        from midas_gui.helpers import _apply_im_trans
        img = _apply_im_trans(img, im_trans)
        if dk is not None:
            dk = _apply_im_trans(np.asarray(dk, dtype=np.float32), im_trans)
    NZ, NY = img.shape
    return img, dk, NY, NZ


def make_seed_safe(image: np.ndarray, wavelength: float, pxY: float,
                   calibrant: str):
    """Run the automatic seeder; return a Seed or None on failure.

    use_diplib=False is mandatory: diplib's median filter segfaults on macOS
    (the package's own one_shot pipeline disables it for the same reason, and a
    native segfault is not catchable by the try/except inside make_seed).
    """
    try:
        from midas_calibrate_v2.seed.auto_seed import make_seed
        return make_seed(image.astype(np.float32), wavelength_A=wavelength,
                         px_um=pxY, calibrant=calibrant, use_diplib=False)
    except Exception:
        return None


def _distortion_coeffs(refine: dict) -> set:
    """Resolve which distortion coefficients (v2 names) to refine.

    Prefers the per-coefficient ``distortion_coeffs`` set; falls back to the
    legacy single ``Distortion`` bool (all-or-nothing) for callers that predate
    the per-coefficient dialog.
    """
    coeffs = refine.get("distortion_coeffs")
    if coeffs is not None:
        return set(coeffs)
    return set(DISTORTION_NAMES) if bool(refine.get("Distortion", True)) else set()


def _resolve_seed(manual: Optional[dict], image: np.ndarray, wavelength: float,
                  pxY: float, calibrant: str) -> Optional[dict]:
    """Build a full geometry seed (BC_y, BC_z, Lsd, tx, ty, tz) for the
    v1-native pipelines (``build_v1_params`` needs concrete numbers to start
    from regardless of what's actually being *refined*), mixing whichever
    fields the GUI's per-parameter seed panel enabled with an automatic seed
    for the rest.

    ``manual`` is *sparse*: only keys the user ticked "include in seed" for
    are present (BC_y and BC_z always travel together — the backend has no
    way to seed one without the other). Returns ``None`` only when an
    automatic seed is needed to fill a gap and it fails; callers already
    raise their own "auto-seed failed, enable manual seed" error in that
    case, same as before this function existed.
    """
    manual = manual or {}
    have_bc = "BC_y" in manual and "BC_z" in manual
    have_lsd = "Lsd" in manual
    auto = None
    if not (have_bc and have_lsd):
        auto = make_seed_safe(image, wavelength, pxY, calibrant)
        if auto is None:
            return None
    seed = {
        "BC_y": manual["BC_y"] if have_bc else auto.BC_y,
        "BC_z": manual["BC_z"] if have_bc else auto.BC_z,
        "Lsd":  manual["Lsd"] if have_lsd else auto.Lsd_um,
    }
    for k in ("tx", "ty", "tz"):
        if manual.get(k) is not None:
            seed[k] = float(manual[k])
    if manual.get("distortion"):
        seed["distortion"] = dict(manual["distortion"])
    return seed


#: Parameter-window ("tolerance") fields on ``CalibrationParams``, in the units
#: the dataclass stores them in. ``midas_calibrate/param_vector.py:bounds()``
#: turns each into a hard ``(value - tol, value + tol)`` box constraint on the
#: LM solve, so these are real bounds, not hints — and they apply on *every*
#: crystalline route whether or not anyone sets them, at the defaults below.
TOL_FIELDS = ("tolLsd", "tolBC", "tolTilts", "tolDistortion", "tolWavelength")


def tol_defaults() -> dict:
    """The ``tol*`` defaults actually in force, read off the installed
    ``CalibrationParams`` rather than hardcoded — a backend release that
    retunes them must not leave the GUI displaying stale windows."""
    import dataclasses as dc
    from midas_calibrate.params import CalibrationParams
    out = {}
    for f in dc.fields(CalibrationParams):
        if f.name in TOL_FIELDS:
            out[f.name] = float(f.default)
    return out


def tols_are_default(tols: Optional[dict]) -> bool:
    """True when ``tols`` asks for nothing the backend would not already do."""
    if not tols:
        return True
    defaults = tol_defaults()
    return all(abs(float(v) - defaults[k]) <= 1e-12
               for k, v in tols.items() if k in defaults)


def _refine_dict(refine: dict) -> dict:
    """Translate the GUI refine flags into a v1 ``Refine`` dict.

    GUI flags: Lsd, BC, ty, tz, tx, Wavelength, plus distortion selection
    (``distortion_coeffs`` set of v2 harmonic names, or the legacy ``Distortion``
    bool).  Each selected v2 coefficient maps to its v1 ``p0..p14`` slot.
    """
    coeffs = _distortion_coeffs(refine)
    d = {
        "Lsd":        bool(refine.get("Lsd", True)),
        "BC":         bool(refine.get("BC", True)),
        "ty":         bool(refine.get("ty", True)),
        "tz":         bool(refine.get("tz", True)),
        "Wavelength": bool(refine.get("Wavelength", False)),
        "Parallax":   False,
    }
    for i in range(15):
        d[f"p{i}"] = False
    for name in coeffs:
        slot = _V2_TO_V1.get(name)
        if slot is not None:
            d[slot] = True
    return d


def build_v1_params(seed, *, wavelength, pxY, pxZ, calibrant, NY, NZ,
                    refine: dict, n_iter: int, device: str,
                    min_ring_px: float = 120.0, max_ring_px: Optional[float] = None,
                    tols: Optional[dict] = None):
    """Build a CalibrationParams (V1Params) from a seed.

    Mirrors the construction in ``pipelines/auto.py`` — RhoD is the BC-to-farthest
    -corner distance expressed in µm.

    ``tols`` optionally overrides the parameter windows (:data:`TOL_FIELDS`, in
    the dataclass's own units: µm / px / deg / Å). Keys left out keep the
    dataclass default, and ``tols=None`` reproduces the object this built
    before tolerances were plumbed through at all.
    """
    from midas_calibrate.params import CalibrationParams

    bc_y, bc_z = float(seed["BC_y"]), float(seed["BC_z"])
    lsd = float(seed["Lsd"])
    pxZ = pxZ or pxY
    rho_px = math.sqrt(max(bc_y, NY - bc_y) ** 2 + max(bc_z, NZ - bc_z) ** 2)
    if max_ring_px is None:
        max_ring_px = rho_px * 0.97
    a, b, c, alpha, beta, gamma = _LC.get(calibrant, _LC["CeO2"])

    # Optional seed tilts (default 0) and distortion coefficients (v2 harmonic
    # names → v1 p-slots) carried in from a prior calibration result.
    tx = float(seed.get("tx", 0.0)); ty = float(seed.get("ty", 0.0))
    tz = float(seed.get("tz", 0.0))
    p_seed = {f"p{i}": 0.0 for i in range(15)}
    for name, val in (seed.get("distortion") or {}).items():
        slot = _V2_TO_V1.get(name)
        if slot is not None:
            p_seed[slot] = float(val)

    v1 = CalibrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=pxY, pxZ=pxZ,
        Lsd=lsd, BC_y=bc_y, BC_z=bc_z, tx=tx, ty=ty, tz=tz,
        Wavelength=wavelength,
        SpaceGroup=_SG.get(calibrant, 225),
        LatticeConstant=(a, b, c, alpha, beta, gamma),
        RhoD=rho_px * pxY, MaxRingRad=max_ring_px, MinRingRad=min_ring_px,
        nIterations=n_iter, Refine=_refine_dict(refine),
        Device=device, Dtype="fp64", **p_seed,
    )
    for name, val in (tols or {}).items():
        if name in TOL_FIELDS and val is not None:
            setattr(v1, name, float(val))
    v1.validate()
    return v1


# ── Normalisation: any pipeline output → AutoCalibrationResult ───────────────────

_PANEL_KEYS = ("panel_delta_yz", "panel_delta_theta", "panel_delta_lsd", "panel_delta_p2")


def _extract_panel_unpacked(unpacked: dict) -> dict:
    """Pull panel delta tensors out of an unpacked dict (empty if none present)."""
    return {k: v for k, v in unpacked.items() if k in _PANEL_KEYS}


def _auto_result_from_unpacked(u: dict, *, NY, NZ, pxY, pxZ, wavelength,
                               strain=None, residual_map=None,
                               residual_bin_path=None):
    from midas_calibrate_v2 import AutoCalibrationResult
    pxZ = pxZ or pxY
    distortion = {n: float(u[n]) for n in DISTORTION_NAMES if n in u}
    return AutoCalibrationResult(
        Lsd=float(u["Lsd"]),
        BC_y=float(u["BC_y"]), BC_z=float(u["BC_z"]),
        tx=float(u.get("tx", 0.0)),
        ty=float(u["ty"]), tz=float(u["tz"]),
        distortion=distortion,
        pxY=pxY, pxZ=pxZ, NrPixelsY=NY, NrPixelsZ=NZ,
        wavelength_A=wavelength,
        post_residual_strain_uE=strain,
        residual_corr_map=residual_map,
        residual_corr_bin_path=residual_bin_path,
    )


def _attach_panel_result(result, panel_u: dict, panel_layout: Optional[dict],
                         scratch: Optional[str], stem: str = "") -> None:
    """Attach panel-layout results to ``result`` in a form downstream spec
    building / paramstest export can use directly.

    ``result._panel_unpacked`` (raw tensors, private) already lets the save
    dialog write a companion panel_shifts.txt on demand. That alone isn't
    enough for in-GUI integration (Results-tab preview, Batch Integrate's
    "Use Tab 2 calibration"): those build an IntegrationSpec straight from
    ``result`` with no save step, so the shifts need to already be on disk
    and the panel *grid* (rows/cols/size/gaps — not just the deltas) needs
    to be recorded somewhere too. Writes the file unconditionally (unlike
    residual_corr.bin, which stays in-memory-only when the fit didn't build
    one) since a missing panel correction silently produces the wrong
    geometry, not just a smaller residual. Sets two plain,
    JSON-serializable attributes (``panel_layout`` dict of ints,
    ``panel_shifts_path`` str) so both survive ``_save_json``'s
    underscore-attribute filter.

    Named ``<stem>_panelshifts.txt``, matching what the save dialog already
    does (``tab_calibrate._save_paramstest``) and for the same reason: the
    generic ``panel_shifts.txt`` this used to write meant two fits sharing a
    directory silently overwrote each other's shifts, and a Hydra run with
    four panels in flight raced. The per-run scratch leaf already separates
    them; the name also makes the file self-describing to anyone who goes
    looking.
    """
    if not panel_u or not panel_layout:
        return
    from midas_calibrate_v2.compat.to_v1 import write_panel_shifts_file
    from midas_gui.helpers import session_scratch_dir
    stem = re.sub(r"[^\w.-]", "_", stem or "calib")   # free-form Exp ID upstream
    # `scratch` arrives already resolved and created by helpers.scratch_dir at
    # the caller — do not re-resolve it here, that would nest a second
    # .midas_scratch inside it. Without one, fall back to the process temp dir
    # (cleaned at exit) so "no working directory set" still produces shifts.
    base = Path(scratch) if scratch else Path(session_scratch_dir())
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{stem}_panelshifts.txt"
    if not scratch:
        print(f"[calibrate] panel shifts written to session scratch ({path}) — "
              f"they won't survive this session. Save .json / Save paramstest, "
              f"or set a Working directory, to keep them.")
    write_panel_shifts_file(panel_u, path)
    result.panel_layout = dict(panel_layout)
    result.panel_shifts_path = str(path)


def _confirm_residual_bin(result):
    """Null out ``residual_corr_bin_path`` when it names a file nothing wrote.

    ``calibrate()`` mints the path from ``output_dir`` alone
    (``midas_calibrate_v2/pipelines/auto.py:680-683``) and returns it at
    ``:976`` without ever checking the file exists — but it only *builds* the
    map when ``build_residual_corr`` is on (``:702``), and it disables map
    building outright for a multi-panel fit (``:702-703``,
    ``build_residual_corr and panel_layout is None``). The phase-2 path at
    ``:773`` can also give up on too few surviving non-outlier fits. So an
    un-ticked "Build residual map", or any panel-layout run, hands back a
    result pointing at a ``residual_corr.bin`` that was never created.

    That matters because midas_integrate_v2 treats an unreadable
    ``ResidualCorrectionMap`` as fatal rather than ignoring it, so the phantom
    path takes down every integration, cake and pseudo-strain view built from
    an otherwise complete geometry. Worse, it does not stay in memory: it is
    persisted into the project snapshot and exported into paramstest, so one
    bad fit poisons files that outlive the session.

    ``helpers._drop_missing_residual_map`` defends the spec, but by then the
    bad path is already recorded. Clearing it here fixes it at the source, for
    every branch below, and is idempotent for the reroute branch that already
    confirms its own path.
    """
    p = getattr(result, "residual_corr_bin_path", None)
    if p and not os.path.isfile(str(p)):
        try:
            result.residual_corr_bin_path = None
        except AttributeError:      # frozen/slotted result — nothing to fix up
            pass
    return result


def normalize_result(raw, mode: str, *, NY, NZ, pxY, pxZ, wavelength,
                     panel_layout: Optional[dict] = None,
                     scratch: Optional[str] = None, stem: str = ""):
    """Return an AutoCalibrationResult regardless of which pipeline produced raw.

    Every branch's result goes out through :func:`_confirm_residual_bin`; see
    there for why a returned residual-map path cannot be trusted as written.

    When panel_layout was used, the refined panel shifts (panel_delta_yz /
    panel_delta_theta) are attached as ``result._panel_unpacked`` so the save
    dialog can write a companion panel_shifts.txt, and — via
    :func:`_attach_panel_result` — as ``result.panel_layout``/
    ``result.panel_shifts_path`` so in-GUI spec building
    (``helpers._build_spec``) can feed panel corrections to
    ``midas_integrate_v2`` without requiring an explicit save first. For
    one_shot + panel_layout, run_pipeline internally routes through
    autocalibrate_four_stage (which exposes stage2.unpacked); we detect this
    by checking for a ``.stage2`` attribute on the raw result.
    """
    return _confirm_residual_bin(_normalize_result_impl(
        raw, mode, NY=NY, NZ=NZ, pxY=pxY, pxZ=pxZ, wavelength=wavelength,
        panel_layout=panel_layout, scratch=scratch, stem=stem))


def _normalize_result_impl(raw, mode: str, *, NY, NZ, pxY, pxZ, wavelength,
                           panel_layout: Optional[dict] = None,
                           scratch: Optional[str] = None, stem: str = ""):
    """The per-mode conversion itself; see :func:`normalize_result`."""
    # one_shot+panel_layout was re-routed through four_stage to expose unpacked
    if mode == "one_shot" and hasattr(raw, "stage2"):
        effective_mode = "four_stage"
    else:
        effective_mode = mode

    if effective_mode == "one_shot":
        # Two different objects arrive here. calibrate() hands back a real
        # AutoCalibrationResult; the reroute through
        # pipelines.single.autocalibrate (run_pipeline takes it for non-default
        # tol*, Lsd/BC held fixed, or only one of ty/tz refined) hands back a v2
        # CalibrationResult — spec + unpacked, with no flat Lsd/BC_y/NrPixelsY
        # fields at all. Returning that unconverted crashed every downstream
        # reader of the geometry (paramstest export first). Detected the same
        # way the four_stage reroute is detected above: by the shape of raw.
        if hasattr(raw, "unpacked") and not hasattr(raw, "Lsd"):
            strain = getattr(raw, "post_residual_strain_uE", None)
            if strain is None and getattr(raw, "history", None):
                strain = raw.history[-1].mean_strain_uE
            result = _auto_result_from_unpacked(
                raw.unpacked, NY=NY, NZ=NZ, pxY=pxY, pxZ=pxZ,
                wavelength=wavelength, strain=strain,
                residual_map=getattr(raw, "residual_corr_map", None),
                residual_bin_path=getattr(raw, "_residual_bin_path", None))
            # run_pipeline pre-flipped the image for this branch, so the frame
            # the geometry was solved in is only recorded on raw._im_trans.
            # The non-rerouted one_shot path gets this natively from
            # calibrate(); without it a rerouted result would integrate in the
            # wrong frame.
            result.im_trans = tuple(getattr(raw, "_im_trans", ()) or ())
            panel_u = _extract_panel_unpacked(raw.unpacked)
            if panel_u:
                result._panel_unpacked = panel_u
                _attach_panel_result(result, panel_u, panel_layout, scratch, stem)
            return result
        return raw   # calibrate() already returns AutoCalibrationResult, no panel data

    if effective_mode == "first_time":
        pv = raw.result
        strain = (pv.history[-1].mean_strain_uE
                  if getattr(pv, "history", None) else None)
        result = _auto_result_from_unpacked(
            pv.unpacked, NY=NY, NZ=NZ, pxY=pxY, pxZ=pxZ,
            wavelength=wavelength, strain=strain)
        panel_u = _extract_panel_unpacked(pv.unpacked)
        if panel_u:
            result._panel_unpacked = panel_u
            _attach_panel_result(result, panel_u, panel_layout, scratch, stem)
        return result

    if effective_mode == "four_stage":
        pv = raw.stage2   # final geometry stage (PVCalibrationResult)
        strain = getattr(raw, "stage4_strain_uE", None)
        result = _auto_result_from_unpacked(
            pv.unpacked, NY=NY, NZ=NZ, pxY=pxY, pxZ=pxZ,
            wavelength=wavelength, strain=strain)
        panel_u = _extract_panel_unpacked(pv.unpacked)
        if panel_u:
            result._panel_unpacked = panel_u
            _attach_panel_result(result, panel_u, panel_layout, scratch, stem)
        return result

    if effective_mode == "bayesian":
        result = _auto_result_from_unpacked(
            raw.map_unpacked, NY=NY, NZ=NZ, pxY=pxY, pxZ=pxZ, wavelength=wavelength)
        panel_u = _extract_panel_unpacked(raw.map_unpacked)
        if panel_u:
            result._panel_unpacked = panel_u
            _attach_panel_result(result, panel_u, panel_layout, scratch, stem)
        lap = getattr(raw, "laplace", None)
        if lap is not None:
            names = list(getattr(lap, "refined_names", []) or [])
            sig = getattr(lap, "sigma_per_dim", None)
            if sig is not None:
                result._laplace_sigma = {n: float(s) for n, s in zip(names, sig)}
        return result

    if effective_mode == "joint":
        result = _auto_result_from_unpacked(
            raw.map_unpacked, NY=NY, NZ=NZ, pxY=pxY, pxZ=pxZ, wavelength=wavelength)
        panel_u = _extract_panel_unpacked(raw.map_unpacked)
        if panel_u:
            result._panel_unpacked = panel_u
            _attach_panel_result(result, panel_u, panel_layout, scratch, stem)
        return result

    if effective_mode == "frozen_point":
        pv = raw.res   # final PVCalibrationResult, same shape as first_time/four_stage
        strain = (pv.history[-1].mean_strain_uE
                  if getattr(pv, "history", None) else None)
        result = _auto_result_from_unpacked(
            pv.unpacked, NY=NY, NZ=NZ, pxY=pxY, pxZ=pxZ,
            wavelength=wavelength, strain=strain)
        result._frozen_point_converged = raw.converged
        result._frozen_point_n_iter = raw.n_iter
        return result

    raise ValueError(f"Unsupported pipeline mode for normalisation: {effective_mode}")


def effective_pixel_counts(image: np.ndarray, im_trans) -> tuple:
    """``(NrPixelsY, NrPixelsZ)`` in the frame the pipeline actually solved in.

    Every :func:`run_pipeline` branch fits in the *transformed* frame — either
    because it pre-transforms itself (:func:`_prep_transformed`) or because the
    backend applies ``im_trans`` internally — so the counts handed to
    :func:`normalize_result` have to come from the transformed shape too. Only
    opcode 3 (transpose) changes the shape, and it swaps Y and Z, so an odd
    number of them flips the pair; the mirrors (1, 2) leave it alone.

    Getting this wrong is silent and only bites on a **non-square** detector
    with a transpose active: the fit is fine, but the recorded
    ``NrPixelsY``/``NrPixelsZ`` describe a detector that was never fitted, and
    every downstream consumer of the result (integration spec, paramstest
    export, ring overlays) inherits the mismatch.
    """
    NZ, NY = np.asarray(image).shape
    if tuple(im_trans or ()).count(3) % 2:
        NY, NZ = NZ, NY
    return int(NY), int(NZ)


def tilt_seed_effective(mode: str, *, panel_layout=None, refine: Optional[dict] = None) -> bool:
    """Whether a manual tx/ty/tz seed will actually reach the underlying solver
    for this pipeline/config, mirroring :func:`run_pipeline`'s own branching —
    kept in sync with it deliberately rather than introspected generically,
    since the branch taken (not just the pipeline name) decides this.

    * ``four_stage`` / ``bayesian`` / ``joint`` always seed tilts via
      :func:`build_v1_params` (``CalibrationParams`` takes tx/ty/tz directly).
    * ``one_shot`` seeds tilts the same way when internally routed through
      ``autocalibrate_four_stage`` (``panel_layout`` set) — see the
      corresponding branch in :func:`run_pipeline`. Every other one_shot case
      (full, no, or partial distortion refinement) calls
      ``midas_calibrate_v2.calibrate()`` directly, whose ``initial_tx/ty/tz``
      kwargs are silently dropped by :func:`_supported_kwargs` unless the
      installed backend's signature actually exposes them — checked here at
      call time, not assumed, so this stays correct if a future backend
      release adds them (see QUESTIONS_FOR_COLLEAGUES.md item 1).
    * ``first_time`` never passes a tilt seed to ``first_time_calibrate()``
      at all, regardless of backend version.
    * ``frozen_point`` seeds tilts via :func:`build_v1_params` like the
      other advanced pipelines — ty/tz become the fit's starting point (and
      the iterative wrapper's re-centering anchor), and tx is used as a
      fixed value even though the pipeline never refines it. So a seed is
      genuinely used here, unlike ``first_time``, even though tx itself
      never moves.
    """
    if mode in ("four_stage", "bayesian", "joint", "frozen_point"):
        return True
    if mode == "first_time":
        return False
    if mode == "one_shot":
        if panel_layout:
            return True
        try:
            import inspect
            from midas_calibrate_v2 import calibrate
            params = inspect.signature(calibrate).parameters
            return all(f"initial_{k}" in params for k in ("tx", "ty", "tz"))
        except Exception:
            return False
    return False


# ── Dispatch ─────────────────────────────────────────────────────────────────────

def run_pipeline(mode: str, image: np.ndarray, dark, cfg: dict):
    """Run the requested calibration pipeline.

    cfg keys: wavelength, pxY, pxZ, calibrant, refine (dict), n_iter,
    lm_max_iter, device, build_residual_corr, im_trans, output_dir, and an
    optional manual seed {"BC_y","BC_z","Lsd"}.

    Returns the raw pipeline result; call :func:`normalize_result` on it.
    """
    wavelength = cfg["wavelength"]
    pxY        = cfg["pxY"]
    pxZ        = cfg.get("pxZ")
    calibrant  = cfg["calibrant"]
    refine     = cfg.get("refine", {})
    n_iter     = int(cfg.get("n_iter", 4))
    lm_iter    = int(cfg.get("lm_max_iter", 200))
    device     = cfg.get("device", "cpu")
    im_trans   = tuple(cfg.get("im_trans", ()))
    manual     = cfg.get("manual_seed")   # None or {"BC_y","BC_z","Lsd"}
    tols       = cfg.get("tols")          # None or a subset of TOL_FIELDS
    NZ, NY     = image.shape
    panel_layout = _build_panel_layout(cfg.get("panel_layout"))

    if mode == "one_shot":
        if panel_layout is not None:
            # calibrate() runs panel refinement internally but drops panel_delta_*
            # before returning AutoCalibrationResult.  Route through
            # autocalibrate_four_stage instead so stage2.unpacked retains the
            # refined panel shifts; normalize_result detects the FourStageResult
            # via its .stage2 attribute and handles it correctly.
            img, dk, pNY, pNZ = _prep_transformed(image, dark, im_trans)
            seed = _resolve_seed(manual, img, wavelength, pxY, calibrant)
            if seed is None:
                raise RuntimeError(
                    "Auto-seed failed for panel calibration (one_shot). "
                    "Enable manual seed (Pick BC / Pick Ring + Lsd) and retry.")
            v1 = build_v1_params(
                seed, wavelength=wavelength, pxY=pxY, pxZ=pxZ, calibrant=calibrant,
                NY=pNY, NZ=pNZ, refine=refine, n_iter=n_iter, device=device,
                tols=tols)
            from midas_calibrate_v2.pipelines import autocalibrate_four_stage
            return autocalibrate_four_stage(
                v1, img, dark=dk, device=device, panel_layout=panel_layout,
                spec=_panel_spec(v1, panel_layout), verbose=True)

        # Three things calibrate() structurally cannot express, all fixed the
        # same way: route through the lower-level single-pass routine that
        # four_stage / bayesian / joint already use, driven by a GUI-built
        # CalibrationParams.
        coeffs = _distortion_coeffs(refine)
        reroute = []
        # A distortion subset used to be rerouted too, on the grounds that
        # refine_distortion was an all-or-nothing bool. It is not: the kwarg is
        # Union[bool, str, Sequence[str]], so a partial selection reaches
        # calibrate() exactly as ticked (see the refine_distortion comment
        # below). Rerouting for it would cost STAGE-1 for nothing.
        if bool(refine.get("ty", True)) != bool(refine.get("tz", True)):
            # calibrate() takes one refine_tilts bool for both, which the GUI
            # has to compute as (ty or tz) — so refining exactly one of them is
            # not expressible and would silently refine both.
            reroute.append("only one of ty/tz refined")
        if not (refine.get("Lsd", True) and refine.get("BC", True)):
            # auto.py:619 hardcodes Refine={"Lsd": True, "BC": True, ...} and
            # there is no kwarg to change it, so on the plain path unchecking
            # either does nothing at all — the fit refines them regardless.
            held = [n for n in ("Lsd", "BC") if not refine.get(n, True)]
            reroute.append(f"{'/'.join(held)} held fixed")
        if not tols_are_default(tols):
            # calibrate() builds its own CalibrationParams, so its tol* windows
            # are always the dataclass defaults.
            reroute.append("non-default parameter limits")
        if reroute:
            print(f"[calib] note: routing one_shot through "
                  f"pipelines.single.autocalibrate — calibrate() cannot express "
                  f"{', '.join(reroute)}. This skips its STAGE-1 "
                  f"multi-hypothesis Lsd search, so the seed is used as given.")
            img, dk, pNY, pNZ = _prep_transformed(image, dark, im_trans)
            seed = _resolve_seed(manual, img, wavelength, pxY, calibrant)
            if seed is None:
                raise RuntimeError(
                    "Auto-seed failed for one_shot with "
                    + ", ".join(reroute) + ". Enable manual seed "
                    "(Pick BC / Pick Ring + Lsd) and retry.")
            v1 = build_v1_params(
                seed, wavelength=wavelength, pxY=pxY, pxZ=pxZ, calibrant=calibrant,
                NY=pNY, NZ=pNZ, refine=refine, n_iter=n_iter, device=device,
                tols=tols)
            build_rc = bool(cfg.get("build_residual_corr", True))
            bin_path = None
            if build_rc and cfg.get("scratch_dir"):
                # Already created and writability-checked by helpers.scratch_dir
                # at the caller; just name the file inside it.
                bin_path = str(Path(cfg["scratch_dir"]) / "residual_corr.bin")
            from midas_calibrate_v2.pipelines.single import autocalibrate
            raw = autocalibrate(
                v1, img, dark=dk, n_iter=n_iter, lm_max_iter=lm_iter,
                device=device, verbose=True,
                build_residual_corr=build_rc,
                residual_corr_path=bin_path)
            # bin_path is only where autocalibrate would *put* a residual map,
            # decided before it runs. It writes one only if it actually builds
            # one, and it gives up quietly on too few non-outlier fits
            # (single.py:322-345). Recording the path regardless hands
            # integration a ResidualCorrectionMap that isn't on disk, and
            # midas_integrate_v2 hard-fails on an unreadable map rather than
            # ignoring it — so confirm the file before claiming it.
            import os as _os
            raw._residual_bin_path = (        # no .bin field on CalibrationResult
                bin_path if bin_path and _os.path.isfile(bin_path) else None)
            raw._im_trans = im_trans             # ditto; see normalize_result
            return raw

        from midas_calibrate_v2 import calibrate
        kwargs = dict(
            wavelength=wavelength, pxY=pxY, dark=dark, calibrant=calibrant,
            # Scratch, not the working directory: calibrate() writes both
            # residual_corr.bin and a generically-named calibration.json here
            # (pipelines/auto.py:680-683, :921-955), neither of them asked for.
            output_dir=cfg.get("scratch_dir"),
            build_residual_corr=bool(cfg.get("build_residual_corr", True)),
            n_iter=n_iter, lm_max_iter=lm_iter, device=device, verbose=True,
            refine_tilts=bool(refine.get("ty", True) or refine.get("tz", True)),
            # calibrate() accepts a bool OR an explicit list of v2 coefficient
            # names (Union[bool, str, Sequence[str]] — see
            # forward.distortion.resolve_distortion_block, which it calls
            # internally): a partial selection reaches the fit exactly as
            # ticked, instead of being widened to "refine all 15" the way a
            # bare bool would. ``coeffs`` is already empty exactly when the
            # "Distortion" checkbox is off (see _distortion_coeffs).
            refine_distortion=(sorted(coeffs) if coeffs else False),
        )
        if pxZ:
            kwargs["pxZ"] = pxZ
        if im_trans:
            kwargs["im_trans"] = im_trans
        # BC + Lsd seed must be supplied together (see bugs_and_fixes Bug 5)
        if manual:
            have_bc = "BC_y" in manual and "BC_z" in manual
            if have_bc:
                kwargs["initial_BC_y"] = manual["BC_y"]
                kwargs["initial_BC_z"] = manual["BC_z"]
            if "Lsd" in manual:
                kwargs["initial_Lsd"] = manual["Lsd"]
            elif have_bc:
                # BC_guess (BC_y+BC_z) bypasses calibrate()'s own auto-seeder
                # entirely, Lsd included — so BC-only seeding would otherwise
                # silently start Lsd from the library's 1 m nominal. Auto-seed
                # once here just for a plausible Lsd instead.
                auto = make_seed_safe(image, wavelength, pxY, calibrant)
                if auto is not None and auto.Lsd_um:
                    kwargs["initial_Lsd"] = auto.Lsd_um
            # Seed tilts only if the installed calibrate() exposes them
            # (_supported_kwargs drops any it does not accept).
            for k in ("tx", "ty", "tz"):
                if manual.get(k) is not None:
                    kwargs[f"initial_{k}"] = float(manual[k])
        return calibrate(image, **_supported_kwargs(calibrate, kwargs))

    if mode == "first_time":
        from midas_calibrate_v2.pipelines import first_time_calibrate
        if not tols_are_default(tols):
            # first_time_calibrate() takes neither a CalibrationParams nor any
            # tol* kwarg — it has its own tilt_prior_deg/half_window_px knobs
            # on a different footing. Say so rather than accepting limits and
            # quietly ignoring them.
            print("[calib] WARNING: parameter limits are ignored by the "
                  "'First-time' pipeline — it takes no bounds arguments. Use "
                  "One-shot, Four-stage, Bayesian or Joint-cake for bounded "
                  "refinement.")
        a, b, c, alpha, beta, gamma = _LC.get(calibrant, _LC["CeO2"])
        kwargs = dict(
            lattice=(a, b, c, alpha, beta, gamma),
            space_group=_SG.get(calibrant, 225),
            wavelength_A=wavelength,
            pixel_size_um=pxY,
            n_pixels_y=NY, n_pixels_z=NZ,
            lsd_initial_guess_um=((manual or {}).get("Lsd", DEFAULT_LSD_UM)),
            bc_initial_guess=((manual["BC_y"], manual["BC_z"])
                              if manual and "BC_y" in manual and "BC_z" in manual
                              else None),
            dark=dark,
            # first_time_calibrate registers + refines panel shifts correctly
            # on its own (unlike four_stage/bayesian/joint below, which need
            # an explicit panel-aware spec) — it just needs the layout passed.
            panel_layout=panel_layout,
        )
        # Native im_trans since midas_calibrate_v2 0.15.0: the backend flips
        # image, dark and panel_mask together and re-derives n_pixels_y/z from
        # the transformed shape, so this branch hands over the RAW frame and
        # the codes — exactly like the calibrate() path above — and must NOT
        # pre-flip via _prep_transformed, which would apply the transform
        # twice (silently: a double flip looks like a valid image).
        #
        # Before 0.15.0 this branch passed no transform at all and no error
        # was raised, so a first_time calibration on a flipped detector simply
        # ran in the wrong frame and returned a confident wrong geometry.
        # Passed unguarded rather than through _supported_kwargs on purpose:
        # if the installed backend is too old to accept it, a loud TypeError
        # is the right outcome — silently dropping it is the exact bug above.
        if im_trans:
            kwargs["im_trans"] = im_trans
        return first_time_calibrate(image, **kwargs)

    if mode == "four_stage":
        from midas_calibrate_v2.pipelines import autocalibrate_four_stage
        img, dk, pNY, pNZ = _prep_transformed(image, dark, im_trans)
        seed = _resolve_seed(manual, img, wavelength, pxY, calibrant)
        if seed is None:
            raise RuntimeError(
                "Auto-seed failed for four-stage pipeline. "
                "Enable manual seed (Pick BC / Pick Ring + Lsd) and retry.")
        v1 = build_v1_params(
            seed, wavelength=wavelength, pxY=pxY, pxZ=pxZ, calibrant=calibrant,
            NY=pNY, NZ=pNZ, refine=refine, n_iter=n_iter, device=device,
            tols=tols)
        spec = _panel_spec(v1, panel_layout) if panel_layout is not None else None
        return autocalibrate_four_stage(v1, img, dark=dk, device=device,
                                        panel_layout=panel_layout, spec=spec,
                                        verbose=True)

    if mode in ("bayesian", "joint"):
        img, dk, pNY, pNZ = _prep_transformed(image, dark, im_trans)
        v1 = _seed_and_v1(img, wavelength, pxY, pxZ, calibrant, pNY, pNZ,
                          refine, n_iter, device, manual, tols=tols)
        spec = _panel_spec(v1, panel_layout) if panel_layout is not None else None
        if mode == "bayesian":
            from midas_calibrate_v2.pipelines import autocalibrate_bayesian
            return autocalibrate_bayesian(v1, img, mode="laplace", dark=dk,
                                          panel_layout=panel_layout, spec=spec)
        from midas_calibrate_v2.pipelines import autocalibrate_joint
        return autocalibrate_joint(v1, img, dark=dk, panel_layout=panel_layout,
                                   spec=spec)

    if mode == "frozen_point":
        # No native panel_layout support, unlike every other advanced
        # pipeline above: autocalibrate_frozen_point's residual_fn does not
        # pass panel_layout/panel_idx through to pseudo_strain_residual, so
        # a tiled detector would silently be fitted as a single panel.
        if panel_layout is not None:
            raise RuntimeError(
                "Frozen-point (high-tilt) does not support Multi-panel "
                "detectors yet. Uncheck 'Multi-panel' or choose a "
                "different pipeline.")
        img, dk, pNY, pNZ = _prep_transformed(image, dark, im_trans)
        if dk is not None:
            # point_pick() takes no dark argument (only a boolean mask), so
            # subtract it here — every other branch above hands dark to the
            # backend natively instead.
            img = np.clip(img - dk.astype(np.float32), 0, None)
        # v1.Refine (from build_v1_params, inside _seed_and_v1) already carries
        # the GUI's Distortion checkboxes for p0..p14 — iterate_frozen_point_
        # until_stable defers to it exactly like four_stage/bayesian/joint do,
        # so no separate refine_distortion override is passed here.
        #
        # tols is deliberately NOT forwarded here yet (unlike the identical
        # bayesian/joint call above) — tracked as a follow-up, not a backend
        # limitation: spec_from_v1_params() reads v1.tol* the same way for
        # every pipeline, so wiring it through is a one-line fix. Until then,
        # warn instead of silently running on CalibrationParams' hardcoded
        # defaults, mirroring first_time's disclosure above.
        if not tols_are_default(tols):
            print("[calib] WARNING: the parameter-window (±) settings are "
                  "not yet forwarded to the 'Frozen-point (high-tilt)' "
                  "pipeline — it runs with default bounds instead. Use "
                  "One-shot, Four-stage, Bayesian or Joint-cake for a "
                  "bounded refinement with your chosen window.")
        v1 = _seed_and_v1(img, wavelength, pxY, pxZ, calibrant, pNY, pNZ,
                          refine, n_iter, device, manual)
        if device != "cpu":
            print(f"[calib] note: Frozen-point (high-tilt) always runs on "
                  f"CPU — ignoring device={device!r}.")
        try:
            from midas_calibrate_v2.pipelines import iterate_frozen_point_until_stable
        except ImportError as e:      # backend older than the pipeline's release
            from importlib.metadata import PackageNotFoundError, version
            try:
                have = version("midas-calibrate-v2")
            except PackageNotFoundError:
                have = "unknown"
            raise RuntimeError(
                "Frozen-point (high-tilt) needs the frozen-point pipeline from "
                f"midas-calibrate-v2, which the installed {have} does not "
                "provide. Upgrade the backend, or choose a different pipeline."
            ) from e
        return iterate_frozen_point_until_stable(v1, img, lm_max_iter=lm_iter,
                                                 verbose=True)

    raise ValueError(f"Unknown pipeline mode: {mode}")


def _seed_and_v1(image, wavelength, pxY, pxZ, calibrant, NY, NZ,
                 refine, n_iter, device, manual, tols=None):
    """Seed (manual or auto) → build_v1_params. Shared by advanced pipelines.

    ``image`` must already be im_trans-transformed (via ``_prep_transformed``)
    — this seeds directly from whatever array is passed in, so the caller is
    responsible for making sure it's the same array that gets solved against.
    """
    seed = _resolve_seed(manual, image, wavelength, pxY, calibrant)
    if seed is None:
        raise RuntimeError(
            "Auto-seed failed. Enable manual seed (Pick BC / Pick Ring + Lsd).")
    return build_v1_params(
        seed, wavelength=wavelength, pxY=pxY, pxZ=pxZ, calibrant=calibrant,
        NY=NY, NZ=NZ, refine=refine, n_iter=n_iter, device=device, tols=tols)


def _build_panel_layout(cfg):
    """Build a PanelLayout.regular from a config dict, or None."""
    if not cfg:
        return None
    from midas_calibrate_v2.forward.panels import PanelLayout
    return PanelLayout.regular(
        int(cfg["n_y"]), int(cfg["n_z"]),
        int(cfg["sy"]), int(cfg["sz"]),
        gap_y=int(cfg.get("gap_y", 0)), gap_z=int(cfg.get("gap_z", 0)))


def _panel_spec(v1_params, panel_layout):
    """A CalibrationSpec with per-panel rigid-shift parameters
    (panel_delta_yz/panel_delta_theta) registered as refinable.

    autocalibrate_four_stage/_bayesian/_joint build their own spec via
    spec_from_v1_params() when none is passed, which never registers panel
    parameters — panel_layout alone only affects the FIXED forward
    projection (which panel a pixel belongs to), never what gets refined.
    Passing this spec in makes their existing freeze/thaw/refined-parameter
    logic (four_stage's own Stage 1/2 split in particular) actually include
    the panel shift. Tolerances mirror midas_calibrate_v2.calibrate()'s own
    panel_mode="shift" defaults (panel_tol_shift_px=3.0,
    panel_tol_rot_deg=1.0) — the one place in the installed package that
    already does per-panel rigid-shift refinement correctly.
    """
    from midas_calibrate_v2.compat.from_v1 import (
        spec_from_v1_params, add_panel_parameters)
    spec = spec_from_v1_params(v1_params)
    add_panel_parameters(spec, panel_layout.n_panels(),
                         tol_shift_px=3.0, tol_rot_deg=1.0)
    return spec
