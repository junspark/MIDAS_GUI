"""Background QThread workers — every heavy operation runs off the GUI thread.

Worker pattern (context/design_rules.md): redirect stdout to a log signal for
verbose pipelines, catch every exception and emit it, store the worker as an
instance variable on the caller so it is not GC'd mid-run.
"""
from __future__ import annotations

import math
import re
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore

import midas_gui._paths  # noqa: F401  (sys.path setup before MIDAS imports)
from midas_gui import calib
from midas_gui import provenance
from midas_gui.helpers import (_LogStream, _load_image, _apply_im_trans, _build_spec,
                               _spec_from_json, average_field, apply_field_corrections,
                               read_hdf5_stack_combined)


# ═════════════════════════════════════════════════════════════════════════════
#  Shared integration core (used by both single-frame and batch workers)
# ═════════════════════════════════════════════════════════════════════════════

def apply_q_uniform(spec, q_cfg: Optional[dict]):
    """Activate native Q-uniform binning on a spec when requested."""
    if q_cfg:
        spec.QMin     = float(q_cfg["QMin"])
        spec.QMax     = float(q_cfg["QMax"])
        spec.QBinSize = float(q_cfg["QBinSize"])
    return spec


def compute_r_axis(spec) -> np.ndarray:
    """Bin-centre radii in px, whether the spec is in R-uniform or Q-uniform mode."""
    n = spec.n_r_bins
    if spec.q_mode_active:
        wl  = float(spec.Wavelength)
        lsd = float(spec.Lsd); px = float(spec.pxY)
        q_c = spec.QMin + spec.QBinSize * (np.arange(n) + 0.5)
        two_theta = 2.0 * np.arcsin(np.clip(q_c * wl / (4 * math.pi), -1, 1))
        return lsd * np.tan(two_theta) / px
    return float(spec.RMin) + float(spec.RBinSize) * (np.arange(n) + 0.5)


def axis_conversions(r_px, lsd, px, wl):
    """Return (two_theta_deg, two_theta_centideg, Q_invA) from r in px."""
    r_px = np.asarray(r_px, dtype=float)
    two_theta = np.degrees(np.arctan(r_px * px / lsd))
    q = 4 * math.pi * np.sin(np.radians(two_theta) / 2) / wl
    return two_theta, two_theta * 100.0, q


_FID_NUM_RE = re.compile(r'^(?P<froot>.+?)_(?P<num>\d+)(?P<tag>[^_\d]*)$')


def froot_and_frame_num(fid, fallback_idx: int) -> tuple:
    """Parse ``(froot, frame_number, tag)`` out of a frame id, matching
    mpe_wf_saxs_waxs's own ``<froot>_<NNNNNN>`` output-naming convention when
    ``fid`` already follows it — the common case, since detector files are
    already named this way. Handles the frame number sitting mid-stem, not
    just at the end, since ``Path.stem`` only strips the *last* suffix —
    ``C611_017Fe_1_load3_009243.vrx.h5`` (this app's own test data) stems to
    ``..._009243.vrx``, with a non-numeric detector tag (``.vrx``) trailing
    the digits. ``tag`` preserves that (empty string when there isn't one).
    Falls back to ``(fid, fallback_idx, "")`` when ``fid`` has no
    underscore-digit run at all (e.g. a caller-supplied non-numeric id)."""
    m = _FID_NUM_RE.match(str(fid))
    if m:
        return m.group('froot'), int(m.group('num')), m.group('tag')
    return str(fid), int(fallback_idx), ''


def stamp_h5_provenance(h5_path, entry: dict) -> None:
    """Reopen ``h5_path`` (already written by ``midas_integrate_v2.write_h5``)
    and append a provenance entry to its root attrs. Best-effort: a failure
    here shouldn't take down an otherwise-successful batch run."""
    import h5py
    with h5py.File(str(h5_path), 'a') as h5:
        provenance.append_to_hdf5_attrs(h5, entry)


def build_geom(spec, kernel: str, mask):
    from midas_integrate_v2 import (
        SubpixelBinGeometry, HardBinGeometry, PolygonBinGeometry)
    if kernel == "hard":
        return HardBinGeometry.from_spec(spec, mask=mask)
    if kernel == "polygon":
        return PolygonBinGeometry.from_spec(spec, mask=mask, n_jobs=-1)
    K = 4 if kernel == "subpixel4" else 2
    return SubpixelBinGeometry.from_spec(spec, K=K, mask=mask)


def _profile_from_cake(cake_np: np.ndarray, count_cake: Optional[np.ndarray] = None) -> np.ndarray:
    """Collapse an (η, R) cake to a 1-D profile.

    ``count_cake`` None → unweighted mean over η bins (legacy "η-bin mean"): each η
    bin counts equally.  Fast, but with an off-detector beam centre the partially /
    unevenly filled η bins bias the result (worse with coarse η bins).

    ``count_cake`` given → pixel-count-weighted azimuthal mean:
    ``Σ_η(cell_mean · count) / Σ_η(count)``.  Independent of η-bin size and robust
    to partial azimuthal coverage.
    """
    if count_cake is not None:
        finite = np.isfinite(cake_np)
        w = np.where(finite, count_cake, 0.0)
        num = np.sum(np.where(finite, cake_np, 0.0) * w, axis=0)
        den = np.sum(w, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = np.where(den > 0, num / den, np.nan)
        return np.nan_to_num(prof, nan=0.0)
    prof = np.nanmean(cake_np, axis=0)
    return np.nan_to_num(prof, nan=0.0)


def count_cake(geom, kernel: str, NrPixelsZ: int, NrPixelsY: int) -> np.ndarray:
    """Per-(η,R)-cell pixel count for the plain-kernel geometry — integrate a
    ones-image without normalisation.  Used to pixel-weight the 1-D profile."""
    import torch
    import midas_integrate_v2 as m
    ones = torch.ones((NrPixelsZ, NrPixelsY), dtype=torch.float64)
    fn = {"hard": m.integrate_hard, "polygon": m.integrate_polygon}.get(
        kernel, m.integrate_subpixel)
    return fn(ones, geom, normalize=False).detach().cpu().numpy()


def q_grid_and_r(q_cfg, lsd, px, wl):
    """Uniform-Q bin centres + the matching R(px) for each Q (for axis/writers)."""
    n = max(1, int(round((q_cfg["QMax"] - q_cfg["QMin"]) / q_cfg["QBinSize"])))
    qgrid = q_cfg["QMin"] + q_cfg["QBinSize"] * (np.arange(n) + 0.5)
    two_theta = 2.0 * np.arcsin(np.clip(qgrid * wl / (4 * math.pi), -1, 1))
    r_of_q = lsd * np.tan(two_theta) / px
    return qgrid, r_of_q


def rebin_R_to_Q(r_ax, prof, sigma, qgrid, lsd, px, wl):
    """Rebin an R-uniform profile/σ onto a uniform-Q grid (the kernels don't do Q-mode).

    See analyze_workflows/workflow_analysis.md (P0-2): integrate R-uniform then interpolate
    onto uniform Q so rings land at the correct Q.
    """
    q_of_r = 4 * math.pi * np.sin(np.radians(np.degrees(np.arctan(r_ax * px / lsd))) / 2) / wl
    order = np.argsort(q_of_r)
    prof_q = np.interp(qgrid, q_of_r[order], prof[order])
    sig_q = np.interp(qgrid, q_of_r[order], sigma[order]) if sigma is not None else None
    return prof_q, sig_q


def corrections_counts(spec):
    """Per-(η,R)-bin pixel-count cake for normalising integrate_with_corrections.

    integrate_with_corrections returns SUMMED (unnormalised) counts per bin — a flat
    field integrates to a ramp rising with R.  Dividing by this counts cake (the same
    function applied to a ones-image) restores the per-pixel mean, matching the plain
    kernels.  See analyze_workflows/workflow_analysis.md (P0-1).
    """
    import torch
    import midas_integrate_v2 as m
    ones = torch.ones((spec.NrPixelsZ, spec.NrPixelsY), dtype=torch.float64)
    return m.integrate_with_corrections(ones, spec).detach().cpu().numpy()


def integrate_frame(img_t, spec, geom, kernel, corrections, variance_cfg,
                    need_sigma: bool, corr_counts=None, return_cake=False,
                    weighted: bool = False, cnt_cake=None):
    """Integrate one frame, returning (profile, sigma_or_None) or
    (profile, sigma, cake_2d, cake_sigma_2d).

    Routing:
      - corrections enabled  → integrate_with_corrections, NORMALISED by the pixel-count
        cake (pass corr_counts to avoid recomputing it per frame)
      - variance enabled     → integrate_<kernel>_with_variance (σ from error model)
      - otherwise            → plain kernel integration

    ``weighted`` collapses the (η, R) cake to 1-D with a pixel-count-weighted mean
    (robust to partial azimuthal coverage / off-detector beam centres) instead of the
    unweighted η-bin mean.  For the plain / variance paths pass ``cnt_cake`` (from
    :func:`count_cake`, computed once per geometry); the corrections path reuses its
    own ``corr_counts``.

    When return_cake=True, returns a 4-tuple (prof, sigma, cake_2d, cake_sigma) where
    cake_2d/cake_sigma are the (n_eta_bins, n_r_bins) normalised cake and its per-cell
    uncertainty — real per-cell σ for the variance-model path (computed before it's
    reduced to the 1-D ``sigma``), else √(cake) per cell (matching how ``sigma`` itself
    is derived for the plain/corrections paths).
    """
    import torch
    import midas_integrate_v2 as m

    pol, sa = corrections
    if pol is not None or sa is not None:
        int2d = m.integrate_with_corrections(
            img_t, spec, polarization=pol, solid_angle=sa).detach().cpu().numpy()
        counts = corr_counts if corr_counts is not None else corrections_counts(spec)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.where(counts > 0.5, int2d / counts, np.nan)
        prof = _profile_from_cake(norm, count_cake=counts if weighted else None)
        sigma = np.sqrt(np.maximum(prof, 0.0)) if need_sigma else None
        if return_cake:
            cake_sigma = np.sqrt(np.maximum(np.nan_to_num(norm, nan=0.0), 0.0))
            return prof, sigma, norm, cake_sigma
        return prof, sigma

    if variance_cfg is not None:
        em = variance_cfg.get("error_model", "poisson")
        fn = {
            "hard":      m.integrate_hard_with_variance,
            "polygon":   m.integrate_polygon_with_variance,
        }.get(kernel, m.integrate_subpixel_with_variance)
        mean2d, sig2d = fn(img_t, geom, error_model=em)
        mean_np = mean2d.detach().cpu().numpy()
        sig_np  = sig2d.detach().cpu().numpy()
        prof = _profile_from_cake(mean_np, count_cake=cnt_cake if weighted else None)
        # σ of the η-mean: sqrt(Σσ²)/N over valid η bins
        var = np.nansum(sig_np ** 2, axis=0)
        cnt = np.maximum(np.sum(np.isfinite(sig_np), axis=0), 1)
        sigma = np.nan_to_num(np.sqrt(var) / cnt, nan=0.0)
        if return_cake:
            # Real per-cell σ from the error model — strictly more correct than
            # the post-collapse √(cake) fallback used in the other two paths.
            cake_sigma = np.nan_to_num(sig_np, nan=0.0)
            return prof, sigma, mean_np, cake_sigma
        return prof, sigma

    fn = {
        "hard":    m.integrate_hard,
        "polygon": m.integrate_polygon,
    }.get(kernel, m.integrate_subpixel)
    int2d = fn(img_t, geom, normalize=True)
    cake_np = int2d.detach().cpu().numpy()
    prof = _profile_from_cake(cake_np, count_cake=cnt_cake if weighted else None)
    sigma = np.sqrt(np.maximum(prof, 0.0)) if need_sigma else None
    if return_cake:
        cake_sigma = np.sqrt(np.maximum(np.nan_to_num(cake_np, nan=0.0), 0.0))
        return prof, sigma, cake_np, cake_sigma
    return prof, sigma


def build_integration_context(spec, kernel: str, mask, corrections, weighted: bool) -> dict:
    """Build the one-time integration setup (the "detector map") for a spec.

    Returns everything the per-frame integration needs — the binning geometry,
    radial axis, pixel-count cakes, η axis and lengths — so it can be built once
    and reused across many frames (a batch run *or* live folder monitoring).
    """
    spec.validate()
    lsd, px, wl = float(spec.Lsd), float(spec.pxY), float(spec.Wavelength)
    pol, sa = corrections
    corr_on = pol is not None or sa is not None
    geom = None if corr_on else build_geom(spec, kernel, mask)
    r_ax = compute_r_axis(spec)
    corr_counts = corrections_counts(spec) if corr_on else None
    cnt = (count_cake(geom, kernel, spec.NrPixelsZ, spec.NrPixelsY)
           if (weighted and not corr_on) else None)
    n_eta = spec.n_eta_bins
    eta_ax = float(spec.EtaMin) + float(spec.EtaBinSize) * (np.arange(n_eta) + 0.5)
    return {"spec": spec, "geom": geom, "r_ax": r_ax, "corr_counts": corr_counts,
            "cnt": cnt, "eta_ax": eta_ax, "lsd": lsd, "px": px, "wl": wl,
            "corr_on": corr_on}


def write_profile(base, fmt, r_px, prof, sigma, lsd, px, wl,
                  cake_2d=None, eta_axis=None):
    """Write one integrated profile in the requested 1-D/2-D format."""
    import midas_integrate_v2 as m
    two_theta, two_theta_cd, q = axis_conversions(r_px, lsd, px, wl)
    sig = sigma if sigma is not None else np.sqrt(np.maximum(prof, 0.0))
    if fmt == "csv":
        m.write_csv(str(base) + ".csv", r_axis=r_px, intensity=prof, sigma=sig)
    elif fmt == "xye":
        m.write_xye(str(base) + ".xye", r_axis=two_theta, intensity=prof, sigma=sig)
    elif fmt == "fxye":
        m.write_fxye(str(base) + ".fxye", r_axis=two_theta_cd, intensity=prof, sigma=sig)
    elif fmt == "dat":
        m.write_dat(str(base) + ".dat", q_axis_invA=q, intensity=prof, sigma=sig)
    elif fmt == "2d_csv" and cake_2d is not None:
        out_path = str(base) + "_cake.csv"
        n_eta, n_r = cake_2d.shape
        eta_vals = eta_axis if eta_axis is not None else np.arange(n_eta, dtype=float)
        header = "eta\\R(px)," + ",".join(f"{r:.4f}" for r in r_px)
        rows = [f"{eta_vals[k]:.4f}," + ",".join(f"{v:.6g}" for v in cake_2d[k])
                for k in range(n_eta)]
        with open(out_path, "w") as fh:
            fh.write(header + "\n")
            fh.write("\n".join(rows) + "\n")


def write_frame_profiles(base, file_fmts, r_px, prof, sigma, lsd, px, wl,
                         cake_2d=None, cake_sigma=None, eta_axis=None) -> list:
    """Write one frame's 1-D output file(s) in every format in ``file_fmts``
    (``"2d_csv"``/``"h5"`` excluded — callers handle those separately).

    Without ``cake_2d``: writes ``prof``/``sigma`` once, as always.

    With ``cake_2d``/``cake_sigma`` (multi-azimuth/"cake" mode — see the
    "Multi-azimuth output (cake)" checkbox in Batch Integrate): writes one
    file *per azimuthal (η) bin* instead, named ``<base>_etaNNN.<fmt>``, each
    a genuine 1-D lineout for that sector — ``prof``/``sigma`` (the
    η-collapsed full-circle profile) are not written in this mode. ``"2d_csv"``
    in ``file_fmts`` is honored separately as the one whole-cake file, unaffected.
    Returns the list of paths written.
    """
    paths = []
    if cake_2d is not None and cake_sigma is not None:
        n_eta = cake_2d.shape[0]
        eta_ax = eta_axis if eta_axis is not None else np.arange(n_eta, dtype=float)
        for k in range(n_eta):
            eta_base = Path(str(base) + f"_eta{k:03d}")
            for f in file_fmts:
                if f == "2d_csv":
                    continue
                write_profile(eta_base, f, r_px, cake_2d[k], cake_sigma[k], lsd, px, wl)
                paths.append(str(eta_base) + "." + f)
        if "2d_csv" in file_fmts:
            write_profile(Path(base), "2d_csv", r_px, prof, sigma, lsd, px, wl,
                         cake_2d=cake_2d, eta_axis=eta_ax)
            paths.append(str(base) + "_cake.csv")
    else:
        for f in file_fmts:
            write_profile(Path(base), f, r_px, prof, sigma, lsd, px, wl)
            paths.append(str(base) + "." + f)
    return paths


# ═════════════════════════════════════════════════════════════════════════════
#  Dark / bright / background field averaging
# ═════════════════════════════════════════════════════════════════════════════

class FieldAverageWorker(QtCore.QThread):
    """Average a dark/bright/background field off the GUI thread.

    kind ∈ {"file","folder","hdf5"}; index range is inclusive (end=-1 → last).
    """
    finished = QtCore.pyqtSignal(object)   # 2-D np.ndarray
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, kind, path, dataset, idx_start, idx_end, parent=None):
        super().__init__(parent)
        self._kind, self._path, self._dataset = kind, path, dataset
        self._start, self._end = idx_start, idx_end

    def run(self):
        try:
            field = average_field(self._kind, self._path, self._dataset,
                                  self._start, self._end)
            self.finished.emit(np.asarray(field, dtype=np.float32))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Mask workers
# ═════════════════════════════════════════════════════════════════════════════

def spatial_outlier_mask(med, stackmax, k_sigma, hot_f, dead_f, overflow):
    """Spatial outlier mask (template_auto_mask_unconstrained.ipynb approach).

    5×5 median residual → 15×15 robust local MAD → Z-score → hot/dead/sat gates.
    Returns (mask_bool, breakdown_str).
    """
    from scipy.ndimage import median_filter
    med = med.astype(np.float64)
    mf = median_filter(med, size=5)
    resid = med - mf
    local_scale = median_filter(np.abs(resid), size=15) * 1.4826 + 1e-6
    z = resid / local_scale
    mf_safe = np.clip(mf, 1e-9, None)
    hot  = (z >  k_sigma) & (med > hot_f  * mf_safe)
    dead = (z < -k_sigma) & (med < dead_f * mf_safe)
    sat  = (stackmax >= overflow) if overflow is not None else np.zeros_like(med, dtype=bool)
    mask = hot | dead | sat
    info = (f"hot: {int(hot.sum()):,}  dead: {int(dead.sum()):,}  sat: {int(sat.sum()):,}")
    return mask, info


def temporal_constancy_mask(stack: np.ndarray, frozen_frac: float) -> tuple:
    """Flag pixels whose temporal std is far below the detector-wide typical variation.

    A detector module stuck at a constant value (dead, gap, stuck ADC) has temporal
    std ≈ 0.  The 75th-percentile of non-zero per-pixel std is used as reference so
    the threshold adapts to the overall signal level without being pulled by dead pixels.

    Returns (mask_bool, info_str).
    """
    temp_std = np.std(stack.astype(np.float64), axis=0)
    nonzero = temp_std[temp_std > 0]
    if len(nonzero) == 0:
        return np.zeros(temp_std.shape, dtype=bool), "frozen: 0 (no variation)"
    ref = np.percentile(nonzero, 75)
    frozen = temp_std < (frozen_frac * ref)
    return frozen, f"frozen: {int(frozen.sum()):,} (ref_std={ref:.2g})"


class MaskComputeWorker(QtCore.QThread):
    """Compute the combined bad-pixel mask: base threshold OR'd with any enabled
    advanced method (statistical outlier, spatial spike, azimuthal clip, learnable).

    Azimuthal-clip and learnable-mask require a calibration result (geometry).
    """
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)   # uint8 combined mask
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, image, base_mask, methods, *, stack_paths=None,
                 stack_hdf5=None, calib_result=None, parent=None):
        super().__init__(parent)
        self._image = image              # always raw detector-space
        self._base = base_mask
        self._methods = methods          # dict of {name: params or False}
        self._stack_paths = stack_paths
        self._stack_hdf5 = stack_hdf5    # (path, dataset, stride) for a 3-D HDF5 stack
        self._result = calib_result

    def run(self):
        try:
            import torch
            import midas_integrate_v2 as m
            combined = self._base.astype(bool).copy()
            parts = [f"threshold: {int(self._base.sum()):,}"]

            stat = self._methods.get("stat")
            cosmic_ray = self._methods.get("cosmic_ray")
            # Load the frame stack once if any temporal method needs it — either a
            # single HDF5 file with a 3-D (time, y, x) dataset, or a set of frame files.
            stack = None
            if stat or cosmic_ray:
                if self._stack_hdf5:
                    import h5py
                    path, dset, stride = self._stack_hdf5
                    stride = max(1, int(stride))
                    self.progress.emit(f"Loading HDF5 stack '{dset}' (stride {stride})…")
                    with h5py.File(path, "r") as f:
                        if dset not in f:
                            raise KeyError(f"dataset '{dset}' not in {path}")
                        ds = f[dset]
                        if ds.ndim == 3:
                            stack = np.asarray(ds[::stride], dtype=np.float32)
                        elif ds.ndim == 2:
                            stack = np.asarray(ds[()], dtype=np.float32)[None, ...]
                        else:
                            raise ValueError(
                                f"HDF5 dataset '{dset}' is {ds.ndim}-D; need a 2-D image "
                                "or a 3-D (time, y, x) sequence.")
                    self.progress.emit(f"HDF5 stack: {stack.shape[0]} frame(s) "
                                       f"of {stack.shape[1]}×{stack.shape[2]}")
                elif self._stack_paths:
                    self.progress.emit(f"Loading {len(self._stack_paths)} frames…")
                    frames = [_load_image(p).astype(np.float32) for p in self._stack_paths]
                    stack = np.stack(frames, axis=0)

            # Statistical auto-mask: spatial outlier and temporal constancy are
            # now independently selectable (either one, or both).
            if stat:
                do_spatial = bool(stat.get("spatial", True))
                do_temporal = bool(stat.get("temporal", False))
                # Temporal constancy: catches constant-value modules spatial methods miss
                if do_temporal:
                    if stack is not None and stack.shape[0] >= 2:
                        self.progress.emit("Temporal constancy check…")
                        fmask, finfo = temporal_constancy_mask(
                            stack, stat.get("frozen_frac", 0.05))
                        combined |= fmask
                        parts.append(finfo)
                    else:
                        self.progress.emit(
                            "[temporal] skipped — needs a stack of ≥2 frames")
                # Spatial outlier: temporal median if a stack is present, else the frame
                if do_spatial:
                    if stack is not None:
                        self.progress.emit("Computing temporal median…")
                        med = np.median(stack, axis=0); stackmax = stack.max(axis=0)
                    else:
                        med = self._image; stackmax = self._image
                    self.progress.emit("Statistical spatial-outlier detection…")
                    mask, info = spatial_outlier_mask(
                        med, stackmax, stat["k_sigma"], stat["hot_factor"],
                        stat["dead_factor"], stat.get("overflow"))
                    combined |= mask
                    parts.append(f"spatial({info})")

            # Cosmic-ray rejection (temporal σ-clip along the frame axis)
            if cosmic_ray:
                if stack is not None and stack.shape[0] >= 3:
                    self.progress.emit(
                        f"Cosmic-ray rejection (n_σ={cosmic_ray['n_sigma']}, "
                        f"{stack.shape[0]} frames)…")
                    from midas_integrate_v2.streaming import reject_cosmic_rays
                    _, cr_mask_3d = reject_cosmic_rays(
                        stack.astype(np.float64),
                        n_sigma=cosmic_ray["n_sigma"], mode="flag_only", use_mad=True)
                    cr_mask = cr_mask_3d.any(axis=0)
                    combined |= cr_mask
                    parts.append(f"cosmic-ray: {int(cr_mask.sum()):,}")
                elif stack is not None:
                    self.progress.emit(
                        "[cosmic-ray] skipped — need ≥3 frames "
                        f"(stack has {stack.shape[0]})")
                else:
                    self.progress.emit("[cosmic-ray] skipped — no stack folder specified")

            # Spatial spike rejection (geometry-free)
            spike = self._methods.get("spike")
            if spike:
                self.progress.emit("Spatial spike rejection…")
                _, sm = m.reject_spatial_spikes(
                    self._image.astype(np.float64), n_sigma=spike["n_sigma"],
                    method=spike.get("method", "laplacian"))
                combined |= sm.astype(bool)
                parts.append(f"spike: {int(sm.sum()):,}")

            # Azimuthal sigma-clip (needs geometry)
            azim = self._methods.get("azimuthal")
            if azim and self._result is not None:
                self.progress.emit("Azimuthal σ-clip…")
                from midas_gui.helpers import _build_spec, _apply_im_trans
                spec = _build_spec(self._result, 2.0, 5.0)
                geom = m.HardBinGeometry.from_spec(spec)   # needs per-pixel bins
                im_trans = tuple(getattr(self._result, "im_trans", ()) or ())
                # azimuthal_sigma_clip has no apply_trans_opt hook — it needs the
                # image in the geometry's transformed/world orientation. self._image
                # is raw, so transform it just for this call, then transform the
                # resulting mask back to raw before combining it with `combined`.
                img_xf = _apply_im_trans(self._image, im_trans) if im_trans else self._image
                _, am = m.azimuthal_sigma_clip(
                    img_xf.astype(np.float64), geom, n_sigma=azim["n_sigma"])
                am_raw = (_apply_im_trans(am.astype(np.uint8), tuple(reversed(im_trans))).astype(bool)
                          if im_trans else am.astype(bool))
                combined |= am_raw
                parts.append(f"azimuthal: {int(am_raw.sum()):,}")

            # Learnable mask (needs geometry; differentiable training)
            learn = self._methods.get("learnable")
            if learn and self._result is not None:
                self.progress.emit("Learnable mask training…")
                from midas_gui.helpers import _build_spec, _apply_im_trans
                spec = _build_spec(self._result, 2.0, 5.0)
                im_trans = tuple(getattr(self._result, "im_trans", ()) or ())
                # integrate_with_corrections flips internally (apply_trans_opt=True,
                # via spec.TransOpt) and applies the learnable-mask weights *after*
                # that flip, i.e. in transformed/world space — so self._image (raw)
                # is passed through untouched, but the mask's shape/static prior
                # must be built from a transformed copy of the running `combined`.
                combined_xf = (_apply_im_trans(combined.astype(np.uint8), im_trans).astype(bool)
                               if im_trans else combined)
                NZ, NY = combined_xf.shape
                static_t = torch.from_numpy(combined_xf)
                lm = m.LearnableMask(NZ, NY, init_weight=float(learn.get("init_weight", 0.9)),
                                     static_mask=static_t)
                img_t = torch.from_numpy(self._image.astype(np.float64))
                loss_fn = m.EtaUniformityLoss(intensity_floor=0.0)
                opt = torch.optim.Adam(lm.parameters(), lr=float(learn.get("lr", 0.5)))
                n_steps = int(learn.get("n_steps", 300))
                sp_wt = float(learn.get("sparsity_weight", 1e-4))
                for step in range(n_steps):
                    opt.zero_grad()
                    int2d = m.integrate_with_corrections(img_t, spec, learnable_mask=lm)
                    loss = loss_fn(int2d) + m.sparsity_prior(lm, weight=sp_wt, target=1.0)
                    loss.backward(); opt.step()
                    if step % 25 == 0 or step == n_steps - 1:
                        with torch.no_grad():
                            nlow = lm.n_low_weight_pixels(0.5)
                        self.progress.emit(f"Learnable step {step+1}/{n_steps}  "
                                           f"loss={float(loss.detach()):.4g}  masked≈{nlow:,}")
                hard = np.asarray(lm.extract_hard_mask(threshold=0.5)).astype(bool)
                hard_raw = (_apply_im_trans(hard.astype(np.uint8), tuple(reversed(im_trans))).astype(bool)
                            if im_trans else hard)
                combined |= hard_raw
                parts.append(f"learnable: {int(hard_raw.sum()):,}")

            out = combined.astype(np.uint8)
            n = int(out.sum())
            self.progress.emit(f"Done — {'  '.join(parts)}  →  combined: {n:,} "
                               f"({100*n/out.size:.3f}%)")
            self.finished.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Calibration worker (pipeline-aware)
# ═════════════════════════════════════════════════════════════════════════════

class ProjectionWorker(QtCore.QThread):
    """Load a frame stack and reduce it (max/sum/average) off the GUI thread.

    Loading a multi-GB stack + the reduction can take seconds-to-minutes; doing it
    here keeps the Data Viewer responsive. Field corrections (dark/bright/background)
    are captured on the GUI thread and applied to the projected image.
    """
    finished = QtCore.pyqtSignal(object, str)   # corrected 2-D image, info string
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, full_stack_fn, method, axis, skip, nframes=0, *, dark=None, bright=None,
                 background=None, bright_mode="divide", parent=None):
        super().__init__(parent)
        self._full_stack = full_stack_fn
        self._method, self._axis, self._skip, self._nframes = method, axis, skip, nframes
        self._dark, self._bright, self._background = dark, bright, background
        self._bright_mode = bright_mode

    def run(self):
        try:
            data = np.asarray(self._full_stack())
            if self._axis >= data.ndim:
                raise ValueError(f"Axis {self._axis} invalid for {data.ndim}-D data.")
            if self._skip > 0:
                if self._skip >= data.shape[0]:
                    raise ValueError(f"Skip frames ({self._skip}) ≥ stack size "
                                     f"({data.shape[0]}).")
                data = data[self._skip:]
            if self._nframes and self._nframes > 0:
                data = data[:self._nframes]
            n_used = data.shape[self._axis]
            fn = {"max": np.max, "sum": np.sum, "average": np.mean}[self._method]
            proj = np.squeeze(fn(data, axis=self._axis))
            if proj.ndim != 2:
                raise ValueError(f"Result is {proj.ndim}-D after projecting axis "
                                 f"{self._axis}; pick an axis that leaves a 2-D image.")
            if self._dark is not None or self._bright is not None or self._background is not None:
                out = apply_field_corrections(
                    proj, dark=self._dark, bright=self._bright,
                    bright_mode=self._bright_mode, background=self._background).astype(np.float32)
            else:
                out = proj.astype(np.float32)
            info = (f"{self._method.capitalize()} projection ({n_used} frames"
                    f"{f', skipped {self._skip}' if self._skip else ''}) → {proj.shape}  "
                    f"[{np.nanmin(proj):.3g}, {np.nanmax(proj):.3g}]")
            self.finished.emit(out, info)
        except Exception:
            self.failed.emit(traceback.format_exc())


class AllFrameStatsWorker(QtCore.QThread):
    """Compute unmasked pixel values across an entire stack off the GUI thread.

    The "All frames" intensity-stats scope needs to read + correct the whole
    stack (or live ring buffer), which can take seconds for a large stack —
    doing it synchronously on the GUI thread would freeze the UI. All
    GUI-derived inputs (dark/bright/background arrays, mask, intensity-range
    thresholds) are captured by the caller before construction so run() only
    touches numpy data, never Qt widgets.
    """
    finished = QtCore.pyqtSignal(object, int)   # unmasked values, n frames
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, full_stack_fn, *, dark=None, bright=None, background=None,
                 bright_mode="divide", composite_mask=None,
                 imask_on=False, imask_lo=0.0, imask_hi=0.0, parent=None):
        super().__init__(parent)
        self._full_stack = full_stack_fn
        self._dark, self._bright, self._background = dark, bright, background
        self._bright_mode = bright_mode
        self._composite_mask = composite_mask
        self._imask_on, self._imask_lo, self._imask_hi = imask_on, imask_lo, imask_hi

    def run(self):
        try:
            stack = np.asarray(self._full_stack())
            if stack.ndim == 2:
                stack = stack[None, ...]
            if self._dark is None and self._bright is None and self._background is None:
                corr = stack.astype(np.float32)
            else:
                corr = apply_field_corrections(
                    stack, dark=self._dark, bright=self._bright,
                    bright_mode=self._bright_mode, background=self._background).astype(np.float32)
            bad = ~np.isfinite(corr)
            if self._imask_on:
                if self._imask_lo > -1e9:
                    bad |= (corr <= self._imask_lo)
                if self._imask_hi > 0:
                    bad |= (corr > self._imask_hi)
            cm = self._composite_mask
            if cm is not None and cm.shape == corr.shape[1:]:
                bad |= (cm != 0)[None, :, :]
            self.finished.emit(corr[~bad], corr.shape[0])
        except Exception:
            self.failed.emit(traceback.format_exc())


class CalibrationWorker(QtCore.QThread):
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, mode, image, dark, cfg, parent=None,
                 bright=None, background=None, bright_mode="divide",
                 capture_stdout=True):
        super().__init__(parent)
        self._mode  = mode
        self._image = image
        self._dark  = dark
        self._cfg   = cfg
        self._bright = bright
        self._background = background
        self._bright_mode = bright_mode
        # sys.stdout/stderr are process-global, not per-thread — safe to
        # redirect only when a single CalibrationWorker runs at a time (the
        # single-detector tab, and Hydra's Sequential mode). Hydra's
        # Parallel mode runs several of these concurrently and must NOT
        # redirect (they'd race on the same global); it passes False here
        # and relies on the coarser finished/failed/log_line signals for
        # per-panel status instead of captured print() output.
        self._capture_stdout = capture_stdout

    def run(self):
        import sys
        old_out, old_err = sys.stdout, sys.stderr
        stream = _LogStream(self.log_line) if self._capture_stdout else None  # type: ignore
        if stream is not None:
            sys.stdout = sys.stderr = stream
        try:
            image = self._image.astype(np.float32)
            # Bright/background are applied here; dark stays passed to the pipeline.
            if self._bright is not None or self._background is not None:
                image = apply_field_corrections(
                    image, dark=None, bright=self._bright,
                    bright_mode=self._bright_mode, background=self._background
                ).astype(np.float32)
                self.log_line.emit(
                    f"[calibrate] applied "
                    f"{'bright(' + self._bright_mode + ') ' if self._bright is not None else ''}"
                    f"{'background ' if self._background is not None else ''}correction")
            mask = self._cfg.get("mask")
            if mask is not None:
                image = image.copy()
                image[mask.astype(bool)] = 0.0   # zero sentinels before calibration
            # Hand image/dark to the pipeline exactly as loaded, with the
            # Transforms checkboxes' codes intact in cfg["im_trans"] —
            # calib.run_pipeline applies them per-branch: the plain
            # midas_calibrate_v2.calibrate() path takes im_trans as a native
            # kwarg and flips internally; the other pipeline entry points
            # (four_stage/bayesian/joint/first_time/partial-distortion) have
            # no such parameter in the installed package, so run_pipeline
            # pre-flips for those itself. Either way, this worker never flips
            # the array — it just passes the raw data and codes through.
            raw = calib.run_pipeline(self._mode, image, self._dark, self._cfg)
            NZ, NY = image.shape
            result = calib.normalize_result(
                raw, self._mode, NY=NY, NZ=NZ,
                pxY=self._cfg["pxY"], pxZ=self._cfg.get("pxZ"),
                wavelength=self._cfg["wavelength"],
                panel_layout=self._cfg.get("panel_layout"),
                output_dir=self._cfg.get("output_dir"))
            result._calibrant_name = self._cfg["calibrant"]
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            # Only restore if we're still the active redirect — a newer run (after
            # an abort) may already have installed its own stream; don't clobber it.
            if sys.stdout is stream:
                sys.stdout = old_out
            if sys.stderr is stream:
                sys.stderr = old_err


class ManualDspacingCalibWorker(QtCore.QThread):
    """Fit Lsd + beam center from user-picked ring points and known d-spacings
    (Bragg's law), entirely bypassing ``calib.run_pipeline``/``midas_calibrate_v2``
    — used for non-crystalline calibrants (e.g. AgBH) that have no space group.
    Tilt is fixed at 0. Same signal names as ``CalibrationWorker`` so the tab's
    existing ``_on_done``/``_on_fail``/``_abort`` wiring works unchanged."""
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, picks, wavelength_A, pxY, pxZ, seed, NY, NZ,
                 material_name, d_list, parent=None):
        super().__init__(parent)
        self._picks = list(picks)
        self._wavelength_A = wavelength_A
        self._pxY = pxY
        self._pxZ = pxZ
        self._seed = seed
        self._NY = NY
        self._NZ = NZ
        self._material_name = material_name
        self._d_list = list(d_list)

    def run(self):
        from types import SimpleNamespace
        from midas_gui.helpers import fit_geometry_from_ring_picks
        try:
            self.log_line.emit(
                f"[manual fit] fitting Lsd + BC from {len(self._picks)} picked "
                f"points across {len(set(p[2] for p in self._picks))} ring(s), "
                f"calibrant='{self._material_name}' (tilt fixed at 0)…")
            fit = fit_geometry_from_ring_picks(
                self._picks, self._wavelength_A, self._pxY, self._pxZ,
                seed=self._seed)
            self.log_line.emit(
                f"[manual fit] seed={fit['seed_quality']}  success={fit['success']}  "
                f"residual RMS={fit['residual_deg_rms']:.4f}°  ({fit['message']})")
            if not fit["success"]:
                self.failed.emit(f"Manual fit did not converge: {fit['message']}")
                return
            result = SimpleNamespace(
                Lsd=fit["Lsd"], BC_y=fit["BC_y"], BC_z=fit["BC_z"],
                tx=0.0, ty=0.0, tz=0.0, distortion={},
                pxY=self._pxY, pxZ=self._pxZ or self._pxY,
                NrPixelsY=self._NY, NrPixelsZ=self._NZ,
                wavelength_A=self._wavelength_A, post_residual_strain_uE=None,
                _calibrant_name=self._material_name, _d_list=list(self._d_list),
            )
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Single-frame integration worker (Tab 2 post-calibration preview)
# ═════════════════════════════════════════════════════════════════════════════

class IntegrationWorker(QtCore.QThread):
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, result, image, dark, im_trans, r_bin, eta_bin,
                 mask=None, parent=None, bright=None, background=None,
                 bright_mode="divide", weighted=True):
        super().__init__(parent)
        self._result, self._image, self._dark = result, image, dark
        self._im_trans, self._r_bin, self._eta_bin = im_trans, r_bin, eta_bin
        self._mask = mask
        self._bright, self._background, self._bright_mode = bright, background, bright_mode
        self._weighted = weighted

    def run(self):
        try:
            import torch
            self.log_line.emit("[integrate] Building spec…")
            spec = _build_spec(self._result, self._r_bin, self._eta_bin)
            # spec.TransOpt carries self._result.im_trans (see helpers._build_spec);
            # midas_integrate_v2's apply_trans_opt=True (default) flips the image
            # internally, so image/dark/bright/background stay exactly as loaded.
            # Only the mask is pre-flipped here (no backend hook for it) — against
            # self._im_trans (the live Transforms-checkbox state this preview run
            # was requested with, which normally matches self._result.im_trans).
            img = self._image.astype(np.float32)
            if self._dark is not None or self._bright is not None or self._background is not None:
                img = apply_field_corrections(
                    img, dark=self._dark, bright=self._bright,
                    bright_mode=self._bright_mode, background=self._background).astype(np.float32)
            mask_t = None
            if self._mask is not None:
                mask_t = (_apply_im_trans(self._mask.astype(np.float32), self._im_trans)
                          if self._im_trans else self._mask.astype(np.float32))
            self.log_line.emit("[integrate] Running integration…")
            geom = build_geom(spec, "subpixel2", mask_t)
            # Needed for both the optional weighted profile and (below) masking
            # empty bins in the residual-strain cake, so compute unconditionally.
            cnt = count_cake(geom, "subpixel2", spec.NrPixelsZ, spec.NrPixelsY)
            img_t = torch.from_numpy(img.astype(np.float64))
            prof, _, cake_2d, _ = integrate_frame(img_t, spec, geom, "subpixel2",
                                               (None, None), None, need_sigma=False,
                                               return_cake=True,
                                               weighted=self._weighted,
                                               cnt_cake=cnt if self._weighted else None)
            r_ax = compute_r_axis(spec)
            n_eta = spec.n_eta_bins
            eta_ax = float(spec.EtaMin) + float(spec.EtaBinSize) * (np.arange(n_eta) + 0.5)

            # Rebin the per-pixel radial-correction map (already in the
            # im_trans-applied frame, same as `geom`) into the same (η, R) bins
            # as the cake — a ring × azimuth pseudo-strain map. apply_trans_opt
            # must be False: residual_corr_map is already transformed, unlike
            # the raw image above which needs the flip applied internally.
            resid_cake = None
            resid_map = getattr(self._result, "residual_corr_map", None)
            if resid_map is not None:
                import midas_integrate_v2 as m
                resid_t = resid_map.detach().to("cpu", torch.float64)
                resid_cake = m.integrate_subpixel(
                    resid_t, geom, apply_trans_opt=False, normalize=True
                ).detach().cpu().numpy()
                resid_cake = np.where(cnt > 0, resid_cake, np.nan)

            self.log_line.emit(f"[integrate] Done — {len(prof)} bins, peak={prof.max():.1f}")
            self.finished.emit({
                "r_axis_px": r_ax, "profile": prof,
                "wavelength_A": float(spec.Wavelength),
                "lsd_um": float(spec.Lsd), "px_um": float(spec.pxY),
                "cake_2d": cake_2d, "eta_axis_deg": eta_ax,
                "resid_cake": resid_cake,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Batch integration worker (Tab 3)
# ═════════════════════════════════════════════════════════════════════════════

def _open_source_cfg(cfg):
    """Open a ``DataLoaderPanel.source_cfg()`` descriptor as a frame source.
    Shared by ``BatchWorker._open_source`` and ``BatchRunCoordinator`` (which
    needs a frame count up front, before any ``BatchWorker`` exists, to split
    a Batch-Parallel run into chunks)."""
    from midas_integrate_v2.streaming import TIFFGlobSource, HDF5FrameSource
    if cfg["type"] == "tiff_glob":
        return TIFFGlobSource(cfg["path"])
    if cfg["type"] == "hdf5":
        # Route through _HDF5StackGlobSource (single-element path list) rather
        # than a plain HDF5FrameSource, so "Combine sub-frames" (chunk_size/
        # combine_op, now exposed for single-file HDF5 sources too) actually
        # takes effect instead of being silently ignored.
        return _HDF5StackGlobSource(
            [cfg["path"]], cfg.get("dataset", "frames"),
            chunk_size=cfg.get("chunk_size") or None, op=cfg.get("combine_op", "mean"))
    if cfg["type"] == "tiff_list":
        return _ExplicitTIFFSource(cfg["paths"])
    if cfg["type"] == "hdf5_stack_glob":
        return _HDF5StackGlobSource(
            cfg["paths"], cfg.get("dataset", "exchange/data"),
            chunk_size=cfg.get("chunk_size") or None, op=cfg.get("combine_op", "mean"))
    raise ValueError(f"Unknown source type: {cfg['type']}")


class BatchWorker(QtCore.QThread):
    progress   = QtCore.pyqtSignal(int, int)
    frame_done = QtCore.pyqtSignal(str, object, object, object)  # id, r_axis, prof, sigma
    finished   = QtCore.pyqtSignal(dict)
    failed     = QtCore.pyqtSignal(str)
    log_line   = QtCore.pyqtSignal(str)
    geom_ready = QtCore.pyqtSignal(object)   # integration context (for reuse/caching)

    def __init__(self, spec, source_cfg, mask, out_dir, fmts, kernel,
                 corrections, variance_cfg, q_cfg=None,
                 frame_range=None, frame_indices=None, monitor_file=None,
                 drift_traj=None, parent=None,
                 dark=None, bright=None, background=None, bright_mode="divide",
                 weighted=True, context=None, im_trans=(), multi_azimuth=False):
        super().__init__(parent)
        self._context = context              # prebuilt integration context or None
        self._spec = spec                    # always R-uniform (Q handled by rebinning)
        self._weighted = weighted            # pixel-weighted azimuthal mean (vs η-bin mean)
        # Off by default: R bin/η bin/η range already exist for a different
        # purpose (internal collapse-weighting resolution — η bin defaults to
        # 5° over the full 360°, i.e. 72 internal bins, for EVERY run). Turning
        # this on repurposes those same fields to also define real output
        # azimuthal sectors, keeping one profile per (frame, η bin) instead of
        # collapsing to one full-circle profile per frame.
        self._multi_azimuth = bool(multi_azimuth)
        self._src  = source_cfg
        self._mask = mask
        self._out_dir = Path(out_dir) if out_dir else None
        # Accept a single legacy format string too (older call sites / tests).
        self._fmts = [fmts] if isinstance(fmts, str) else list(fmts or [])
        self._kernel = kernel
        self._corrections = corrections      # (pol, sa)
        self._variance_cfg = variance_cfg    # dict or None
        self._q_cfg = q_cfg                  # {"QMin","QMax","QBinSize"} or None
        # frame_range: (start, end_exclusive_or_None, stride) — None means all frames.
        # Ignored when frame_indices is given (an explicit random-access chunk —
        # see BatchRunCoordinator, which splits one run across several BatchWorkers).
        self._frame_range = frame_range or (0, None, 1)
        self._frame_indices = list(frame_indices) if frame_indices is not None else None
        self._monitor_file = monitor_file    # path to text file, one value per line
        self._drift_traj = drift_traj        # DriftTrajectory or None
        # Dark / bright / background pre-processing (per-frame)
        self._dark, self._bright, self._background = dark, bright, background
        self._bright_mode = bright_mode
        # ImTransOpt codes the active geometry was fit in. spec.TransOpt (set
        # by helpers._build_spec) already carries this, so midas_integrate_v2
        # flips each streamed frame itself — this copy is used only to
        # pre-flip the mask, which has no such backend hook (see run()).
        self._im_trans = tuple(im_trans or ())

    def _open_source(self):
        return _open_source_cfg(self._src)

    def _iter_frames(self, source):
        """Yield ``(abs_i, fid, img)`` for the frames this worker should process.

        ``frame_indices`` (an explicit chunk assigned by ``BatchRunCoordinator``
        for Batch-Parallel mode) reads only those frames via ``source.get(i)`` —
        no wasted decode of frames outside the chunk. Otherwise streams the
        source sequentially, applying ``frame_range``'s start/end/stride (a
        leading skip still decodes-then-discards, matching prior behavior)."""
        if self._frame_indices is not None:
            for i in self._frame_indices:
                fid, img = source.get(i)
                yield i, fid, img
            return
        fr_start, fr_end, fr_stride = self._frame_range
        for abs_i, (fid, img) in enumerate(source):
            if abs_i < fr_start:
                continue
            if fr_end is not None and abs_i >= fr_end:
                break
            if (abs_i - fr_start) % fr_stride != 0:
                continue
            yield abs_i, fid, img

    def run(self):
        try:
            import torch
            import midas_integrate_v2 as m
            spec = self._spec
            spec.validate()
            lsd = float(spec.Lsd); px = float(spec.pxY); wl = float(spec.Wavelength)

            # spec.TransOpt already carries ImTransOpt (see helpers._build_spec) —
            # midas_integrate_v2's apply_trans_opt=True (default) flips the raw
            # streamed frame itself, so it is passed through untouched below.
            # The mask has no such hook (it's baked into the geometry map at
            # build time against the *transformed* pixel grid), so it alone is
            # pre-flipped here, once.
            dark, bright, background = self._dark, self._bright, self._background
            mask = (self._mask if not self._im_trans or self._mask is None
                    else _apply_im_trans(self._mask.astype(np.float32), self._im_trans))

            if self._context is not None:
                self.log_line.emit("[batch] Reusing existing detector map…")
                ctx = self._context
            else:
                self.log_line.emit("[batch] Building geometry (one-time)…")
                ctx = build_integration_context(spec, self._kernel, mask,
                                                self._corrections, self._weighted)
            self.geom_ready.emit(ctx)
            geom = ctx["geom"]; corr_on = ctx["corr_on"]
            corr_counts = ctx["corr_counts"]; cnt = ctx["cnt"]
            r_ax = ctx["r_ax"]; eta_ax = ctx["eta_ax"]
            want_zarr = "zarr" in self._fmts and self._out_dir is not None
            want_cake = ("2d_csv" in self._fmts) or self._multi_azimuth or want_zarr
            need_sigma = True   # xye/fxye require σ; always provide it
            if self._multi_azimuth and self._q_cfg:
                # Q-rebinning (rebin_R_to_Q) only handles a 1-D profile; combining
                # it with per-azimuth cake output isn't supported yet — the UI
                # already blocks this combination before starting the worker.
                raise RuntimeError(
                    "Multi-azimuth output isn't supported together with "
                    "Q-uniform bins yet.")
            # Q-uniform handled by rebinning the R-uniform profile (kernels lack Q-mode)
            if self._q_cfg:
                qgrid, r_ax = q_grid_and_r(self._q_cfg, lsd, px, wl)

            # Monitor normalisation: load per-frame scalars if a file was provided
            monitor_vals = None
            if self._monitor_file:
                try:
                    monitor_vals = [float(x) for x in
                                    Path(self._monitor_file).read_text().split()]
                    self.log_line.emit(
                        f"[batch] monitor file: {len(monitor_vals)} values loaded")
                except Exception as e:
                    self.log_line.emit(f"[batch] monitor file error: {e}")

            source = self._open_source()
            total = (len(self._frame_indices) if self._frame_indices is not None
                     else source.n_frames)
            range_desc = (f"chunk of {total} frame(s)" if self._frame_indices is not None
                          else f"frame_range={self._frame_range}")
            self.log_line.emit(
                f"[batch] {total} frames | kernel={self._kernel} | "
                f"corrections={'on' if corr_on else 'off'} | "
                f"variance={'on' if self._variance_cfg else 'off'} | "
                f"q_uniform={'on (rebinned)' if self._q_cfg else 'off'} | "
                f"{range_desc} | "
                f"monitor={'yes' if monitor_vals else 'no'} | "
                f"drift={'on' if self._drift_traj else 'off'}")
            if self._drift_traj is not None:
                self.log_line.emit(
                    f"[batch] drift trajectory: {len(self._drift_traj.frame_indices)} knots  "
                    f"Lsd [{self._drift_traj.Lsd_t.min():.0f}, {self._drift_traj.Lsd_t.max():.0f}] µm")

            fields_on = (dark is not None or bright is not None
                         or background is not None)
            if fields_on:
                self.log_line.emit(
                    f"[batch] field corrections: dark={'y' if dark is not None else 'n'} "
                    f"bright={self._bright_mode if bright is not None else 'n'} "
                    f"background={'y' if background is not None else 'n'}")

            aborted = False
            all_profiles, all_sigmas, frame_ids, out_paths = [], [], [], []
            all_omegas = []   # still needed below for h5's <lo>_<hi> combined-stem
            proc_idx = 0  # index into monitor_vals for processed frames only

            # Zarr is written ONE FILE PER COMBINED OUTPUT FRAME as it's
            # produced (below, inside the loop) rather than bundled into one
            # zarr for the whole run — mirrors mpe_wf's own one-zarr-per-
            # scan-point convention, extended so a file that "Combine
            # sub-frames" splits into several chunks gets one zarr per
            # chunk too (see `fid`'s naming in _HDF5StackGlobSource._fid).
            # Precompute what's shared across every per-frame write once,
            # up front, rather than repeating it per frame.
            zarr_dir = zarr_bin_area = zarr_prov_entry = write_gsas_zarr_zip = None
            if want_zarr:
                from midas_integrate_v2.io.zarr_gsas import write_gsas_zarr_zip
                zarr_dir = self._out_dir / "zarr"
                zarr_dir.mkdir(parents=True, exist_ok=True)
                zarr_bin_area = count_cake(geom, self._kernel, spec.NrPixelsZ, spec.NrPixelsY)
                zarr_prov_entry = provenance.build_entry(
                    'midas_gui.batch_integrate',
                    inputs=[self._src.get('path')] if self._src.get('path') else [],
                    cake_params={
                        'RMin': float(spec.RMin), 'RMax': float(spec.RMax),
                        'RBinSize': float(spec.RBinSize), 'EtaMin': float(spec.EtaMin),
                        'EtaMax': float(spec.EtaMax), 'EtaBinSize': float(spec.EtaBinSize),
                    },
                    extra={
                        'kernel': self._kernel, 'weighted': self._weighted,
                        'multi_azimuth': self._multi_azimuth,
                        'n_frames': 1, 'frame_range': list(self._frame_range),
                    },
                )

            for abs_i, fid, img in self._iter_frames(source):
                # Cooperative abort — stop cleanly, keeping frames already done.
                if self.isInterruptionRequested():
                    aborted = True
                    self.log_line.emit(f"[batch] aborted by user after {proc_idx} frame(s)")
                    break
                # img stays exactly as streamed — no im_trans applied to it in
                # Python. spec.TransOpt (see run()'s top) makes the backend
                # integrate_* call flip it internally further down.
                # Dark / bright / background pre-processing
                if fields_on:
                    img = apply_field_corrections(
                        img, dark=dark, bright=bright,
                        bright_mode=self._bright_mode, background=background)

                # Per-frame geometry when drift correction is active
                if self._drift_traj is not None:
                    cur_spec = _spec_from_trajectory(self._spec, self._drift_traj, abs_i)
                    cur_lsd  = float(cur_spec.Lsd)
                    cur_geom = None if corr_on else build_geom(cur_spec, self._kernel, mask)
                    cur_cc   = corrections_counts(cur_spec) if corr_on else None
                    cur_cnt  = (count_cake(cur_geom, self._kernel, cur_spec.NrPixelsZ,
                                           cur_spec.NrPixelsY)
                                if (self._weighted and not corr_on) else None)
                else:
                    cur_spec = spec
                    cur_lsd  = lsd
                    cur_geom = geom
                    cur_cc   = corr_counts
                    cur_cnt  = cnt

                img_t = torch.from_numpy(img.astype(np.float64))
                if want_cake:
                    prof, sigma, cake_2d, cake_sigma = integrate_frame(
                        img_t, cur_spec, cur_geom, self._kernel, self._corrections,
                        self._variance_cfg, need_sigma, corr_counts=cur_cc,
                        return_cake=True, weighted=self._weighted, cnt_cake=cur_cnt)
                else:
                    cake_2d = None; cake_sigma = None
                    prof, sigma = integrate_frame(
                        img_t, cur_spec, cur_geom, self._kernel, self._corrections,
                        self._variance_cfg, need_sigma, corr_counts=cur_cc,
                        weighted=self._weighted, cnt_cake=cur_cnt)

                if sigma is None:
                    sigma = np.sqrt(np.maximum(prof, 0.0))

                # Apply monitor normalisation
                if monitor_vals is not None and proc_idx < len(monitor_vals):
                    mon = float(monitor_vals[proc_idx])
                    if mon != 0.0:
                        prof = prof / mon
                        sigma = sigma / abs(mon)
                        if cake_2d is not None:
                            cake_2d = cake_2d / mon
                            cake_sigma = cake_sigma / abs(mon)

                if self._q_cfg:   # rebin R-uniform → uniform Q (not combined with cake mode)
                    prof, sigma = rebin_R_to_Q(compute_r_axis(spec), prof, sigma,
                                               qgrid, lsd, px, wl)
                # The live waterfall/stacked view always gets the η-collapsed profile,
                # in both modes — only the accumulated/stored result differs.
                if self._multi_azimuth and cake_2d is not None:
                    all_profiles.append(cake_2d)
                    all_sigmas.append(cake_sigma)
                else:
                    all_profiles.append(prof)
                    all_sigmas.append(sigma)
                if want_zarr and cake_2d is not None:
                    all_omegas.append(float(abs_i))
                    # One zarr per combined output frame, written immediately
                    # rather than accumulated — `fid` already carries the
                    # right per-chunk identity (a bare file stem when
                    # "Combine sub-frames" produces one output per file, or
                    # "<stem>.frame_<start>_<end>" per chunk — the actual raw
                    # sub-frame range it combines — when it splits a file
                    # into several, see _HDF5StackGlobSource._fid), so this
                    # naturally yields one zarr per image stack, and however
                    # many the chunking splits it into, uneven remainder
                    # chunk included, with the frames it covers visible in
                    # the filename.
                    zarr_path = zarr_dir / f"{fid}.ave.zarr.zip"
                    # Instrument metadata (temperature/pressure/storage-ring
                    # current), when the source can provide it (HDF5 stacks
                    # only — see _HDF5StackGlobSource.metadata_for_index) —
                    # always the mean across this chunk's raw sub-frames,
                    # regardless of the pixel combine op, and already
                    # aligned to the light-frame timestamps rather than a
                    # longer light+dark metadata array.
                    meta = None
                    get_meta = getattr(source, "metadata_for_index", None)
                    if get_meta is not None:
                        try:
                            meta = get_meta(abs_i)
                        except Exception:
                            meta = None
                    temps = pressures = currents = None
                    if meta:
                        if meta.get("temperature") is not None:
                            temps = [meta["temperature"]]
                        if meta.get("pressure") is not None:
                            pressures = [meta["pressure"]]
                        if meta.get("current") is not None:
                            currents = [meta["current"]]
                    try:
                        write_gsas_zarr_zip(
                            zarr_path, [cake_2d], spec=spec,
                            omegas=[float(abs_i)], bin_area=zarr_bin_area,
                            temperatures=temps, pressures=pressures,
                            currents=currents)
                        try:
                            provenance.append_to_zip(zarr_path, zarr_prov_entry)
                        except Exception:
                            self.log_line.emit(
                                f"[batch] note: provenance stamp on {zarr_path.name} "
                                "failed (non-fatal):\n" + traceback.format_exc())
                        out_paths.append(str(zarr_path))
                    except Exception:
                        self.log_line.emit(
                            f"[batch] zarr cake output for {fid!r} failed:\n"
                            + traceback.format_exc())
                frame_ids.append(fid)
                self.frame_done.emit(fid, r_ax, prof, sigma)
                self.progress.emit(proc_idx + 1, total)
                proc_idx += 1

                file_fmts = [f for f in self._fmts if f not in ("h5", "zarr")]
                if self._out_dir is not None and file_fmts:
                    froot, frame_num, tag = froot_and_frame_num(fid, abs_i)
                    stem = f"{froot}_{frame_num:06d}{tag}"
                    # One subfolder per lineout format (csv/, xye/,
                    # 2d_csv/, ...), at the same level as h5/ and zarr/
                    # below — rather than mixing every text format into
                    # one folder, or nesting them all under another
                    # "lineouts" layer.
                    for fmt in file_fmts:
                        fmt_dir = self._out_dir / fmt
                        fmt_dir.mkdir(parents=True, exist_ok=True)
                        out_paths.extend(write_frame_profiles(
                            fmt_dir / stem, [fmt], r_ax, prof, sigma, cur_lsd, px, wl,
                            cake_2d=(cake_2d if self._multi_azimuth else None),
                            cake_sigma=(cake_sigma if self._multi_azimuth else None),
                            eta_axis=eta_ax))

            # Combined h5 output name: <original-source-stem>.<start>_<end>
            # .cake — h5 remains one file for the whole run (unlike zarr,
            # written per-frame above), so it still needs an explicit
            # frame-index range. start/end are the actual processed 0-based
            # frame indices (all_omegas), matching what per-frame lineout
            # files already use via froot_and_frame_num(fid, abs_i) above.
            src_path = self._src.get('path')
            if src_path:
                out_stem = Path(src_path).stem
            else:
                src_paths = self._src.get('paths') or []
                out_stem = (froot_and_frame_num(Path(src_paths[0]).stem, -1)[0]
                            if src_paths else "integrated")
            lo = int(min(all_omegas)) if all_omegas else 0
            hi = int(max(all_omegas)) if all_omegas else 0
            combined_stem = f"{out_stem}.{lo:06d}_{hi:06d}.cake"

            prov_entry = provenance.build_entry(
                'midas_gui.batch_integrate',
                inputs=[self._src.get('path')] if self._src.get('path') else [],
                cake_params={
                    'RMin': float(spec.RMin), 'RMax': float(spec.RMax),
                    'RBinSize': float(spec.RBinSize), 'EtaMin': float(spec.EtaMin),
                    'EtaMax': float(spec.EtaMax), 'EtaBinSize': float(spec.EtaBinSize),
                },
                extra={
                    'kernel': self._kernel, 'weighted': self._weighted,
                    'multi_azimuth': self._multi_azimuth,
                    'n_frames': len(all_profiles), 'frame_range': list(self._frame_range),
                },
            )

            # HDF5: single file with the full stack — skipped in multi-azimuth mode,
            # midas_integrate_v2.write_h5 expects a 1-D profile per frame.
            if self._out_dir is not None and "h5" in self._fmts:
                if self._multi_azimuth:
                    self.log_line.emit(
                        "[batch] Note: HDF5 output isn't written in multi-azimuth "
                        "mode (write_h5 expects one profile per frame) — use the "
                        "text formats or GSAS-II zarr export instead.")
                else:
                    h5_dir = self._out_dir / "h5"
                    h5_dir.mkdir(parents=True, exist_ok=True)
                    h5_path = h5_dir / f"{combined_stem}.h5"
                    m.write_h5(str(h5_path),
                               profiles=np.array(all_profiles),
                               r_axis=r_ax,
                               frame_ids=frame_ids,
                               sigmas=np.array(all_sigmas))
                    try:
                        stamp_h5_provenance(h5_path, prov_entry)
                    except Exception:
                        self.log_line.emit(
                            f"[batch] note: provenance stamp on {h5_path.name} "
                            "failed (non-fatal):\n" + traceback.format_exc())
                    out_paths.append(str(h5_path))

            # Zarr cake output is written per-frame inside the loop above
            # (see the `want_zarr and cake_2d is not None` branch) — nothing
            # left to do here.

            n_proc = len(all_profiles)
            self.finished.emit({
                "n": n_proc, "r_axis_px": r_ax,
                "profiles": np.array(all_profiles) if all_profiles else np.array([]),
                "sigmas": np.array(all_sigmas) if all_sigmas else np.array([]),
                "frame_ids": frame_ids,
                "out_paths": out_paths,
                "aborted": aborted,
                "multi_azimuth": self._multi_azimuth,
                "eta_axis": eta_ax if self._multi_azimuth else None,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


class PumpProbeWorker(QtCore.QThread):
    """Integrate a pooled set of time-resolved (TR-XRD) frames with the MIDAS engine
    (identical primitives to BatchWorker), then group by pump-probe delay and
    reference-subtract to ΔI(q, delay).

    ``frames`` is a list of ``(path, delay, fshw)`` — one entry per raw detector
    image, with the delay already parsed (and sign-normalised) from the filename.
    """
    progress = QtCore.pyqtSignal(int, int)
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(dict)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, spec, frames, mask, kernel, corrections, weighted=True,
                 q_cfg=None, ref_delays=None, norm_range=None, context=None,
                 dark=None, bright=None, background=None, bright_mode="divide",
                 parent=None, im_trans=()):
        super().__init__(parent)
        self._spec = spec
        self._frames = frames                 # [(path, delay, fshw), …]
        self._mask = mask
        self._kernel = kernel
        self._corrections = corrections       # (pol, sa)
        self._weighted = weighted
        self._q_cfg = q_cfg                    # {"QMin","QMax","QBinSize"} or None
        self._ref_delays = ref_delays          # explicit reference-delay set or None
        self._norm_range = norm_range          # (qmin, qmax) per-pattern norm window or None
        self._context = context
        self._dark, self._bright, self._background = dark, bright, background
        self._bright_mode = bright_mode
        # spec.TransOpt already carries ImTransOpt (see helpers._build_spec) so
        # each frame loaded below is integrated exactly as read from disk; only
        # the mask needs a manual pre-flip (see BatchWorker for why).
        self._im_trans = tuple(im_trans or ())

    def run(self):
        try:
            import torch
            spec = self._spec
            spec.validate()
            lsd = float(spec.Lsd); px = float(spec.pxY); wl = float(spec.Wavelength)

            mask = (self._mask if not self._im_trans or self._mask is None
                    else _apply_im_trans(self._mask.astype(np.float32), self._im_trans))
            if self._context is not None:
                self.log_line.emit("[pump] Reusing existing detector map…")
                ctx = self._context
            else:
                self.log_line.emit("[pump] Building geometry (one-time)…")
                ctx = build_integration_context(spec, self._kernel, mask,
                                                self._corrections, self._weighted)
            geom = ctx["geom"]; corr_counts = ctx["corr_counts"]; cnt = ctx["cnt"]
            r_ax = ctx["r_ax"]

            if self._q_cfg:
                qgrid, r_ax = q_grid_and_r(self._q_cfg, lsd, px, wl)
            two_theta, _, q_ax = axis_conversions(r_ax, lsd, px, wl)

            fields_on = (self._dark is not None or self._bright is not None
                         or self._background is not None)
            total = len(self._frames)
            self.log_line.emit(
                f"[pump] {total} frames | kernel={self._kernel} | "
                f"corrections={'on' if ctx['corr_on'] else 'off'} | "
                f"q_uniform={'on' if self._q_cfg else 'off'} | "
                f"fields={'on' if fields_on else 'off'}")

            profiles, delays = [], []
            for i, (path, delay, _fshw) in enumerate(self._frames):
                if self.isInterruptionRequested():
                    self.log_line.emit(f"[pump] aborted after {i} frame(s)")
                    break
                img = _load_image(path)
                if fields_on:
                    img = apply_field_corrections(
                        img, dark=self._dark, bright=self._bright,
                        bright_mode=self._bright_mode, background=self._background)
                img_t = torch.from_numpy(img.astype(np.float64))
                prof, _ = integrate_frame(
                    img_t, spec, geom, self._kernel, self._corrections,
                    None, False, corr_counts=corr_counts,
                    weighted=self._weighted, cnt_cake=cnt)
                if self._q_cfg:
                    prof, _ = rebin_R_to_Q(compute_r_axis(spec), prof, None,
                                           qgrid, lsd, px, wl)
                if self._norm_range is not None:
                    prof = self._normalize(prof, q_ax, self._norm_range)
                profiles.append(prof); delays.append(float(delay))
                self.progress.emit(i + 1, total)

            if not profiles:
                raise RuntimeError("No frames were integrated.")

            result = self._group_and_difference(
                np.asarray(profiles), np.asarray(delays), self._ref_delays)
            result.update({"r_axis_px": r_ax, "q_axis": q_ax, "tth_axis": two_theta,
                           "lsd": lsd, "px": px, "wl": wl})
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    @staticmethod
    def _normalize(prof, q_ax, norm_range):
        """Divide a profile by its mean intensity in a q-window (per-pattern norm)."""
        lo, hi = float(norm_range[0]), float(norm_range[1])
        sel = (q_ax >= lo) & (q_ax <= hi)
        denom = float(np.mean(prof[sel])) if np.any(sel) else 0.0
        return prof / denom if denom else prof

    @staticmethod
    def _group_and_difference(profiles, delays, ref_delays):
        """Average repeats per delay → I_by_delay; subtract the reference (mean over
        ``ref_delays`` if given, else all negative delays, else the earliest delay)
        → ΔI(q, delay). Returns a dict of stacked arrays keyed by delay order."""
        uniq = sorted(set(delays.tolist()))
        I_by = np.array([profiles[delays == d].mean(axis=0) for d in uniq])
        if ref_delays:
            ref_set = [d for d in uniq if d in set(ref_delays)]
        else:
            ref_set = [d for d in uniq if d < 0]
        if not ref_set:
            ref_set = [uniq[0]]
        ref_idx = [uniq.index(d) for d in ref_set]
        reference = I_by[ref_idx].mean(axis=0)
        dI = I_by - reference
        n_per = [int(np.count_nonzero(delays == d)) for d in uniq]
        return {"delays": uniq, "I_by_delay": I_by, "reference": reference,
                "dI": dI, "ref_delays": ref_set, "n_per_delay": n_per,
                "n": int(profiles.shape[0])}


def _list_tiff_files(path: str) -> list:
    """Files a TIFFGlobSource would see for ``path`` (glob, folder, or file)."""
    p = Path(path)
    if any(ch in str(path) for ch in "*?"):
        from glob import glob as _glob
        return sorted(_glob(str(path)))
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*.tif")) + \
               sorted(str(x) for x in p.glob("*.tiff"))
    if p.is_file():
        return [str(p)]
    return []


class _ExplicitTIFFSource:
    """Iterate over an arbitrary, already-resolved ``list[str]`` of frame files
    (a Batch Integrate "Multiple files" Browse… pick — see
    ``widgets.DataLoaderPanel.source_cfg``'s ``"tiff_list"`` source type).

    Unlike ``TIFFGlobSource`` this can't be expressed as one glob pattern (the
    files may not share a name prefix / may span selections made in different
    moments), so it isn't watchable by MONITOR — see ``tab_batch.py``'s
    ``type != "tiff_glob"`` guard in ``_start_monitor``. Reads via
    ``helpers._load_image`` (not bare ``tifffile``) so it also covers ``.ge*``
    frames, matching every other multi-file picker in the app.
    """

    def __init__(self, paths):
        self._paths = [Path(p) for p in paths]

    @property
    def n_frames(self) -> int:
        return len(self._paths)

    def __iter__(self):
        for p in self._paths:
            img = _load_image(p).astype(np.float64)
            yield p.stem, (img[0] if img.ndim == 3 else img)

    def get(self, idx: int):
        p = self._paths[idx]
        img = _load_image(p).astype(np.float64)
        return p.stem, (img[0] if img.ndim == 3 else img)


class _HDF5StackGlobSource:
    """Iterate over an arbitrary, already-resolved ``list[str]`` of VAREX-style
    multi-frame HDF5 files (a Batch Integrate "Multiple files"/"Full folder"/
    "Files sharing a name stem" pick whose selection resolved to HDF5 files —
    see ``widgets.DataLoaderPanel.source_cfg``'s ``"hdf5_stack_glob"`` source
    type), each file's ``dataset`` combined per ``read_hdf5_stack_combined``
    (``chunk_size``/``op`` — see that function for the "whole file" default).

    Unlike ``_ExplicitTIFFSource`` a single file may yield more than one frame
    (when ``chunk_size`` splits its stack into several combined chunks), so
    ``n_frames``/``get(idx)`` are computed over a flattened index of how many
    combined frames each file yields, read from each file's dataset SHAPE
    (an h5py header read) rather than by decoding it. Counting by decoding
    is what this class used to do, and it made ``n_frames`` — which
    ``BatchWorker`` asks for before a run even starts — read and combine
    every selected file up front: for a 147-file VAREX scan that is tens of
    GB of I/O and RAM before the first frame is integrated, so the run
    looked hung (no progress, no output, no error). The pixel cache is
    likewise bounded to the most recently used file, since both ``__iter__``
    and a chunked ``BatchWorker`` walk the files in order and the old
    unbounded dict retained every decoded file for the life of the run."""

    #: HDF5 paths for the per-acquisition scalars mpe_wf/GSAS-II's zarr
    #: schema carries — fixed regardless of ``dataset`` (the cake-source
    #: dataset the user picked), since these live under the file's
    #: ``instrument/`` group, not under ``exchange/``.
    _METADATA_H5_PATHS = {
        "temperature": "instrument/GSAS2_PVS/Temperature",
        "pressure": "instrument/GSAS2_PVS/Pressure",
        "current": "instrument/StorageRing/SRCurrent",
    }

    def __init__(self, paths, dataset: str, *, chunk_size=None, op: str = "mean"):
        self._paths = [Path(p) for p in paths]
        self._dataset = dataset
        self._chunk_size = chunk_size
        self._op = op
        self._cache: dict = {}   # path index -> list[np.ndarray] (most-recent file only)
        self._counts: Optional[list] = None    # per-file combined-frame count
        self._raw_ns: Optional[list] = None    # per-file raw (pre-combine) sub-frame count
        self._metadata_cache: dict = {}   # path index -> {name: np.ndarray|None}

    def _stat(self, i: int) -> tuple:
        """``(n_chunks, raw_n)`` for file ``i`` — from its dataset shape
        alone (no pixel read), one h5py header open covering both. Mirrors
        ``read_hdf5_stack_combined``'s own chunking: a 2-D dataset is one
        frame; otherwise ``ceil(N / chunk_size)`` combined frames out of
        ``N`` raw ones, with a falsy chunk_size meaning "whole file" (one
        combined frame, still ``N`` raw)."""
        import h5py
        try:
            with h5py.File(str(self._paths[i]), "r") as f:
                dset = f[self._dataset]
                if dset.ndim == 2:
                    return 1, 1
                n = int(dset.shape[0])
        except Exception:
            return 1, 1   # unreadable/odd file — assume 1; get() surfaces the real error
        if not self._chunk_size:
            return 1, n
        size = int(self._chunk_size)
        return max(1, -(-n // size)), n   # ceil

    def _ensure_stats(self) -> None:
        if self._counts is None:
            stats = [self._stat(i) for i in range(len(self._paths))]
            self._counts = [s[0] for s in stats]
            self._raw_ns = [s[1] for s in stats]

    def _combined(self, i: int) -> list:
        cached = self._cache.get(i)
        if cached is None:
            cached = read_hdf5_stack_combined(
                self._paths[i], self._dataset,
                chunk_size=self._chunk_size, op=self._op)
            self._cache = {i: cached}   # keep only the current file
        return cached

    @property
    def n_frames(self) -> int:
        self._ensure_stats()
        return sum(self._counts)

    def _chunk_range(self, k: int, n_raw: int) -> tuple:
        """Inclusive ``(start, end)`` raw 0-based sub-frame range that
        combined-frame ``k`` was built from — the same range ``_fid`` embeds
        in the frame id, reused here to know exactly which raw metadata
        entries (see ``_read_metadata``) belong to this combined frame."""
        if not self._chunk_size:
            return 0, n_raw - 1
        size = int(self._chunk_size)
        start = k * size
        end = min(start + size, n_raw) - 1
        return start, end

    def _fid(self, p: Path, k: int, n_chunks: int, n_raw: int) -> str:
        """Frame id for chunk ``k`` of file ``p``. A single-chunk file (no
        "Combine sub-frames" split, or a 2-D dataset) is just its own stem.
        A multi-chunk file names each chunk by the actual raw 0-based
        sub-frame range it combines (``.frame_<start>_<end>``, no leading
        zeros — same convention as the whole-run zarr naming this replaced)
        rather than an opaque chunk index, so the frames a given output
        actually came from are visible directly in every name built from
        this id: zarr/h5 filenames, per-frame lineout files, and the
        waterfall/stacked-view labels."""
        if n_chunks == 1:
            return p.stem
        start, end = self._chunk_range(k, n_raw)
        return f"{p.stem}.frame_{start}_{end}"

    @staticmethod
    def _metadata_frame_count(f, n_data: int) -> int:
        """How many of a per-acquisition metadata array's *leading* entries
        are the ``n_data`` usable (light) frames actually being integrated,
        as opposed to trailing dark-frame acquisitions folded into the same
        flat, chronological array.

        A VAREX HDF5 stack records one metadata sample per detector
        acquisition — light *and* dark — in a single un-split array: a file
        with ``exchange/data`` shape ``(10, ...)`` and ``exchange/data_dark``
        shape ``(10, ...)`` has e.g. ``misc/NDArrayTimeStamp`` shape ``(20,)``,
        not ``(10,)``. The light->dark transition shows up as one
        anomalously large gap in those per-acquisition timestamps (confirmed
        on real data: 19 steady ~7.01s gaps matching ``Detector/DetAcqPeriod``
        plus exactly one ~9.47s gap, at the light/dark boundary) — use that
        gap, when present, to confirm which leading slice is the light
        block; fall back to ``n_data`` verbatim when there's no timestamp to
        check, or the array is already exactly ``n_data`` long."""
        try:
            ts = np.asarray(f["misc/NDArrayTimeStamp"][()], dtype=np.float64)
        except Exception:
            return n_data
        if ts.ndim != 1 or ts.size <= n_data:
            return n_data
        diffs = np.diff(ts)
        if diffs.size == 0:
            return n_data
        typical = np.median(diffs)
        gap_idx = int(np.argmax(diffs))
        if typical > 0 and diffs[gap_idx] > 1.3 * typical:
            return gap_idx + 1   # light block ends right after this gap
        return n_data

    def _read_metadata(self, i: int) -> dict:
        """Per-raw-acquisition metadata arrays for file ``i``, each already
        sliced down to the leading entries that correspond to the actual
        data frames (see ``_metadata_frame_count``) — cached per file like
        ``_stat``. Missing datasets/unreadable files come back as ``None``
        per key rather than raising, since not every source has this
        metadata (e.g. a non-VAREX HDF5 schema, or a differently-named
        instrument group)."""
        cached = self._metadata_cache.get(i)
        if cached is not None:
            return cached
        out = {k: None for k in self._METADATA_H5_PATHS}
        try:
            import h5py
            self._ensure_stats()
            n_data = self._raw_ns[i]
            with h5py.File(str(self._paths[i]), "r") as f:
                n_aligned = self._metadata_frame_count(f, n_data)
                for key, h5_path in self._METADATA_H5_PATHS.items():
                    if h5_path in f:
                        arr = np.asarray(f[h5_path][()], dtype=np.float64)
                        if arr.ndim == 1 and arr.size >= n_aligned:
                            out[key] = arr[:n_aligned]
        except Exception:
            pass
        self._metadata_cache[i] = out
        return out

    def metadata_for_index(self, idx: int) -> dict:
        """Chunk-averaged metadata (Temperature/Pressure/StorageRing current)
        for combined frame ``idx``, in the same flattened index space as
        ``get(idx)``/``__iter__``.

        Always the arithmetic mean across the chunk's raw sub-frames,
        independent of the pixel ``op`` (Mean/Sum/Max/Median) used to
        combine the cake data itself — metadata like temperature/pressure
        is a scalar sample of the environment during the exposure, not a
        detector count, so it is never summed/maxed the way pixel data can
        be. Returns ``None`` per key when that metadata isn't available for
        this source."""
        self._ensure_stats()
        remaining = idx
        for i, p in enumerate(self._paths):
            n_here = self._counts[i]
            if remaining < n_here:
                n_raw = self._raw_ns[i]
                start, end = self._chunk_range(remaining, n_raw)
                meta = self._read_metadata(i)
                out = {}
                for key, arr in meta.items():
                    if arr is None:
                        out[key] = None
                        continue
                    hi = min(end, arr.size - 1)
                    out[key] = float(np.mean(arr[start:hi + 1])) if start <= hi else None
                return out
            remaining -= n_here
        raise IndexError(idx)

    def __iter__(self):
        self._ensure_stats()
        for i, p in enumerate(self._paths):
            frames = self._combined(i)
            n_raw = self._raw_ns[i]
            for k, img in enumerate(frames):
                yield self._fid(p, k, len(frames), n_raw), img.astype(np.float64)

    def get(self, idx: int):
        # Locate the owning file from the header-only counts, so only THAT
        # file is decoded — walking _combined() over every preceding file
        # (as this used to) made a mid-scan random access decode the whole
        # scan up to that point, which is what BatchWorker's parallel
        # chunks do constantly (each starts at a different offset).
        self._ensure_stats()
        remaining = idx
        for i, p in enumerate(self._paths):
            n_here = self._counts[i]
            if remaining < n_here:
                frames = self._combined(i)
                if remaining >= len(frames):   # header count disagreed with the real read
                    raise IndexError(idx)
                return (self._fid(p, remaining, len(frames), self._raw_ns[i]),
                        frames[remaining].astype(np.float64))
            remaining -= n_here
        raise IndexError(idx)


class FolderMonitorWorker(QtCore.QThread):
    """Watch a folder for new TIFF frames and integrate only the new ones.

    Builds the integration context (the detector map) once — or reuses one passed
    in — then polls the folder; any file whose frame-id (filename stem) is not yet
    in ``seen`` is integrated with that same geometry and emitted via
    ``frame_done``.  Runs until the thread is interruption-requested.
    """
    frame_done = QtCore.pyqtSignal(str, object, object, object)  # fid, r_axis, prof, sigma
    new_count  = QtCore.pyqtSignal(int)     # cumulative new frames integrated
    status     = QtCore.pyqtSignal(str)
    geom_ready = QtCore.pyqtSignal(object)  # integration context (for caching)
    log_line   = QtCore.pyqtSignal(str)
    failed     = QtCore.pyqtSignal(str)

    def __init__(self, spec, folder, mask, kernel, corrections, variance_cfg,
                 q_cfg=None, dark=None, bright=None, background=None,
                 bright_mode="divide", weighted=True, seen=None, context=None,
                 out_dir=None, fmts=("csv",), poll_interval=1.0, parent=None,
                 im_trans=()):
        super().__init__(parent)
        self._spec = spec
        self._folder = folder
        self._mask = mask
        self._kernel = kernel
        self._corrections = corrections
        self._variance_cfg = variance_cfg
        self._q_cfg = q_cfg
        self._dark, self._bright, self._background = dark, bright, background
        self._bright_mode = bright_mode
        self._weighted = weighted
        self._seen = set(seen or [])
        self._context = context
        self._out_dir = Path(out_dir) if out_dir else None
        # Accept a single legacy format string too (older call sites / tests).
        self._fmts = [fmts] if isinstance(fmts, str) else list(fmts or [])
        self._poll_ms = int(max(0.2, poll_interval) * 1000)
        # See BatchWorker's __init__ note — same ImTransOpt discipline (applied
        # here in Python, never via spec.TransOpt).
        self._im_trans = tuple(im_trans or ())

    def run(self):
        try:
            import torch
            import tifffile
            spec = self._spec
            # See BatchWorker.run() — spec.TransOpt already carries ImTransOpt;
            # only the mask needs a manual pre-flip (no backend hook for it).
            dark, bright, background = self._dark, self._bright, self._background
            mask = (self._mask if not self._im_trans or self._mask is None
                    else _apply_im_trans(self._mask.astype(np.float32), self._im_trans))
            if self._context is not None:
                self.log_line.emit("[monitor] reusing existing detector map")
                ctx = self._context
            else:
                self.log_line.emit("[monitor] building detector map (one-time)…")
                ctx = build_integration_context(spec, self._kernel, mask,
                                                self._corrections, self._weighted)
            self.geom_ready.emit(ctx)
            geom = ctx["geom"]; corr_counts = ctx["corr_counts"]; cnt = ctx["cnt"]
            lsd, px, wl = ctx["lsd"], ctx["px"], ctx["wl"]
            r_ax = ctx["r_ax"]
            qgrid = None
            if self._q_cfg:
                qgrid, r_ax = q_grid_and_r(self._q_cfg, lsd, px, wl)
            fields_on = (dark is not None or bright is not None
                         or background is not None)
            save_fmts = [f for f in self._fmts if f in ("csv", "xye", "fxye", "dat")]
            can_save = self._out_dir is not None and bool(save_fmts)
            skipped_fmts = [f for f in self._fmts if f not in save_fmts]
            if self._out_dir is not None and skipped_fmts:
                self.log_line.emit(
                    f"[monitor] note: {', '.join(skipped_fmts)} not saved "
                    "incrementally; new frames are displayed only.")

            self.status.emit("monitoring")
            self.log_line.emit(f"[monitor] watching {self._folder}")
            count = 0
            while not self.isInterruptionRequested():
                try:
                    files = _list_tiff_files(self._folder)
                except Exception as e:
                    self.log_line.emit(f"[monitor] scan error: {e}")
                    files = []
                for f in files:
                    if self.isInterruptionRequested():
                        break
                    fid = Path(f).stem
                    if fid in self._seen:
                        continue
                    try:
                        img = np.asarray(tifffile.imread(f), dtype=np.float64)
                    except Exception as e:
                        # file may still be mid-write; retry on the next poll
                        self.log_line.emit(f"[monitor] skip {fid} (not ready: {e})")
                        continue
                    if fields_on:
                        img = apply_field_corrections(
                            img, dark=dark, bright=bright,
                            bright_mode=self._bright_mode, background=background)
                    img_t = torch.from_numpy(img)
                    prof, sigma = integrate_frame(
                        img_t, spec, geom, self._kernel, self._corrections,
                        self._variance_cfg, True, corr_counts=corr_counts,
                        weighted=self._weighted, cnt_cake=cnt)
                    if sigma is None:
                        sigma = np.sqrt(np.maximum(prof, 0.0))
                    if self._q_cfg:
                        prof, sigma = rebin_R_to_Q(compute_r_axis(spec), prof, sigma,
                                                   qgrid, lsd, px, wl)
                    self._seen.add(fid)
                    count += 1
                    self.frame_done.emit(fid, r_ax, prof, sigma)
                    self.new_count.emit(count)
                    self.log_line.emit(f"[monitor] +{fid}: peak={prof.max():.1f}")
                    if can_save:
                        self._out_dir.mkdir(parents=True, exist_ok=True)
                        froot, frame_num, tag = froot_and_frame_num(fid, count)
                        base = self._out_dir / f"{froot}_{frame_num:06d}{tag}"
                        for fmt in save_fmts:
                            write_profile(base, fmt, r_ax, prof,
                                          sigma, lsd, px, wl)
                # responsive sleep
                slept = 0
                while slept < self._poll_ms and not self.isInterruptionRequested():
                    self.msleep(100); slept += 100
            self.status.emit("stopped")
            self.log_line.emit(f"[monitor] stopped — {count} new frame(s) integrated")
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Batch Parallel: split one run's frames across several concurrent
#  BatchWorkers sharing one pre-built detector map
# ═════════════════════════════════════════════════════════════════════════════

def resolve_frame_indices(n_frames: int, frame_range) -> list:
    """Expand a ``(start, end_exclusive_or_None, stride)`` frame_range against
    a known frame count into an explicit, ordered list of absolute indices."""
    start, end, stride = frame_range or (0, None, 1)
    end = n_frames if end is None else min(int(end), n_frames)
    return list(range(int(start), end, max(1, int(stride))))


def frame_unit_for_cfg(source_cfg) -> str:
    """What one "frame" actually is for a given source — the thing
    ``frame_range``'s start/end/stride index. This differs by source type
    in a way that isn't obvious from the UI: a single multi-frame HDF5
    file indexes the RAW SUB-FRAMES inside it, while a multi-file source
    indexes the FILES (each already combined down to one frame by the
    "Combine sub-frames" setting — see ``_HDF5StackGlobSource``)."""
    kind = (source_cfg or {}).get("type")
    if kind == "hdf5":
        return "sub-frame in this HDF5 file"
    if kind == "hdf5_stack_glob":
        return "file (each combined to one frame)"
    if kind in ("tiff_list", "tiff_glob"):
        return "file"
    return "frame"


def describe_empty_frame_range(n_frames: int, frame_range, source_cfg=None) -> str:
    """A frame_range that selects nothing is nearly always an out-of-range
    ``start`` (e.g. a scan-point number like 9243 typed into a field that
    indexes 0..9 within one file), so say what the actual valid range is
    instead of only that the selection was empty."""
    start, end, stride = frame_range or (0, None, 1)
    unit = frame_unit_for_cfg(source_cfg)
    msg = [f"No frames match the selected frame range "
           f"(start={start}, end={'all' if end in (None, 0) else end}, stride={stride})."]
    if n_frames <= 0:
        msg.append("The selected source reports 0 frames — check the data path/dataset.")
    elif int(start) >= n_frames:
        msg.append(f"This source has {n_frames} frame(s), indexed 0..{n_frames - 1} "
                   f"— one per {unit} — so start={start} is past the end. "
                   f"Set start to 0 (or at most {n_frames - 1}).")
    else:
        msg.append(f"This source has {n_frames} frame(s), indexed 0..{n_frames - 1} "
                   f"— one per {unit}.")
    return "\n".join(msg)


def resolve_worker_count(n_items: int, requested: int, min_per_worker: int) -> int:
    """How many workers to actually use for ``n_items`` frames — the
    requested count, shrunk so every worker gets at least ``min_per_worker``
    items (never below 1)."""
    requested = max(1, int(requested))
    min_per_worker = max(1, int(min_per_worker))
    return min(requested, max(1, int(n_items) // min_per_worker))


def _split_into_chunks(indices: list, n_chunks: int) -> list:
    """Split a sorted list of frame indices into ``n_chunks`` contiguous,
    near-equal pieces (earlier chunks absorb the remainder). Concatenating
    the chunks back in order reproduces ``indices`` exactly."""
    n = len(indices)
    n_chunks = max(1, min(n_chunks, n))
    base, extra = divmod(n, n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        if size == 0:
            continue
        chunks.append(indices[start:start + size])
        start += size
    return chunks


def write_all_profiles(out_dir, fmts, r_axis, profiles, sigmas, frame_ids,
                       lsd, px, wl, eta_axis=None) -> list:
    """Write every frame's already-computed lineout to disk, in every format
    in ``fmts``. Backs the batch tabs' **Save** button — writing results that
    already exist in memory, independent of whether an output directory was
    set before the run — and ``BatchRunCoordinator``'s combined HDF5 write for
    Batch-Parallel mode (each chunk worker would otherwise write its own
    colliding ``integrated.h5``).

    ``profiles``/``sigmas`` may be ``(n_frames, n_r)`` (the usual case) or
    ``(n_frames, n_eta, n_r)`` (multi-azimuth/"cake" mode) — in the latter,
    one file per ``(frame, eta)`` is written per format, named
    ``<fid>_etaNNN.<fmt>``, via :func:`write_frame_profiles`.

    ``"2d_csv"`` (per-frame cake) is silently skipped when ``profiles`` is
    2-D — per-frame cake arrays aren't retained in memory after a run unless
    multi-azimuth mode was on; re-run with an output directory and 2D CSV
    checked to get that format. ``"h5"`` is silently skipped when ``profiles``
    is 3-D (``midas_integrate_v2.write_h5`` expects one profile per frame).
    Returns the list of paths written.
    """
    import midas_integrate_v2 as m
    out_dir = Path(out_dir)
    profiles = np.asarray(profiles)
    sigmas = (np.asarray(sigmas) if sigmas is not None and len(sigmas)
              else np.sqrt(np.maximum(profiles, 0.0)))
    frame_ids = list(frame_ids)
    multi = profiles.ndim == 3
    file_fmts = [f for f in fmts if f not in ("h5", "2d_csv", "zarr")]
    out_paths = []
    if file_fmts and len(frame_ids):
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fid in enumerate(frame_ids):
            froot, frame_num, tag = froot_and_frame_num(fid, i)
            base = out_dir / f"{froot}_{frame_num:06d}{tag}"
            if multi:
                out_paths.extend(write_frame_profiles(
                    base, file_fmts, r_axis, None, None, lsd, px, wl,
                    cake_2d=profiles[i], cake_sigma=sigmas[i], eta_axis=eta_axis))
            else:
                out_paths.extend(write_frame_profiles(
                    base, file_fmts, r_axis, profiles[i], sigmas[i], lsd, px, wl))
    if "h5" in fmts and len(frame_ids) and not multi:
        out_dir.mkdir(parents=True, exist_ok=True)
        h5_path = out_dir / "integrated.h5"
        m.write_h5(str(h5_path), profiles=profiles, r_axis=r_axis,
                   frame_ids=frame_ids, sigmas=sigmas)
        try:
            entry = provenance.build_entry(
                'midas_gui.batch_integrate.save',
                extra={'n_frames': len(frame_ids)})
            stamp_h5_provenance(h5_path, entry)
        except Exception:
            pass   # best-effort — a failed stamp shouldn't fail the save
        out_paths.append(str(h5_path))
    return out_paths


class _GeomBuildWorker(QtCore.QThread):
    """One-shot thread that builds the integration context ("detector map")
    for a spec/kernel/mask/corrections combo — the "detector mapping happens
    on one process first" step ``BatchRunCoordinator`` runs once, ahead of
    fanning per-frame integration out to N concurrent ``BatchWorker``s."""
    done   = QtCore.pyqtSignal(object)   # integration context dict
    failed = QtCore.pyqtSignal(str)

    def __init__(self, spec, kernel, mask, corrections, weighted, parent=None):
        super().__init__(parent)
        self._spec, self._kernel, self._mask = spec, kernel, mask
        self._corrections, self._weighted = corrections, weighted

    def run(self):
        try:
            ctx = build_integration_context(
                self._spec, self._kernel, self._mask, self._corrections, self._weighted)
            self.done.emit(ctx)
        except Exception:
            self.failed.emit(traceback.format_exc())


class BatchRunCoordinator(QtCore.QObject):
    """Runs one batch-integration job either as a single ``BatchWorker``
    ("sequential") or as several concurrent ``BatchWorker``s, each given a
    disjoint, contiguous slice of the frame list ("batch_parallel"), sharing
    one detector map built once up front by ``_GeomBuildWorker``.

    Exposes the same signal surface as ``BatchWorker`` (``progress``,
    ``frame_done``, ``finished``, ``failed``, ``log_line``, ``geom_ready``)
    plus ``isRunning()``/``start()``/``requestInterruption()``/``wait()``, so
    callers (``BatchTab``, ``HydraBatchPage``) can construct this in place of
    ``BatchWorker`` with no change to their signal wiring or abort logic.

    Threads, not OS processes, run the parallel chunks — numpy/torch release
    the GIL during their heavy compute, so this gets real multi-core
    throughput while staying in-process (no pickling geometry/spec objects
    across a process boundary, no cross-process progress plumbing) — the
    same approach ``hydra_batch_page.py``'s existing per-panel "Parallel"
    run mode already uses for its own, independent level of concurrency.
    """
    progress   = QtCore.pyqtSignal(int, int)
    frame_done = QtCore.pyqtSignal(str, object, object, object)
    finished   = QtCore.pyqtSignal(dict)
    failed     = QtCore.pyqtSignal(str)
    log_line   = QtCore.pyqtSignal(str)
    geom_ready = QtCore.pyqtSignal(object)

    MIN_FRAMES_PER_WORKER = 10

    def __init__(self, spec, source_cfg, mask, out_dir, fmts, kernel,
                 corrections, variance_cfg, q_cfg=None,
                 frame_range=None, monitor_file=None, drift_traj=None,
                 dark=None, bright=None, background=None, bright_mode="divide",
                 weighted=True, context=None, im_trans=(), multi_azimuth=False,
                 run_mode="sequential", n_workers=1, parent=None):
        super().__init__(parent)
        self._args = dict(
            spec=spec, source_cfg=source_cfg, mask=mask, out_dir=out_dir, fmts=fmts,
            kernel=kernel, corrections=corrections, variance_cfg=variance_cfg,
            q_cfg=q_cfg, frame_range=frame_range, monitor_file=monitor_file,
            drift_traj=drift_traj, dark=dark, bright=bright, background=background,
            bright_mode=bright_mode, weighted=weighted, im_trans=im_trans,
            multi_azimuth=multi_azimuth)
        self._context = context
        self._run_mode = run_mode if run_mode == "batch_parallel" else "sequential"
        self._n_workers_requested = max(1, int(n_workers))
        self._solo_worker: Optional[BatchWorker] = None
        self._geom_worker: Optional[_GeomBuildWorker] = None
        self._chunks: list = []
        self._chunk_workers: list = []
        self._chunk_results: dict = {}
        self._chunk_totals: dict = {}
        self._chunk_done: dict = {}
        self._interrupted = False
        # Live frame_done re-ordering (batch_parallel only) — chunks run
        # concurrently so their frame_done signals arrive in wall-clock
        # completion order, not frame order; buffer and re-emit in the
        # overall sorted-index order instead. See _on_chunk_frame.
        self._live_order: list = []
        self._live_ptr = 0
        self._live_pending: dict = {}
        self._chunk_frame_counter: dict = {}

    def isRunning(self) -> bool:
        if self._solo_worker is not None:
            return self._solo_worker.isRunning()
        if self._geom_worker is not None and self._geom_worker.isRunning():
            return True
        return any(w.isRunning() for w in self._chunk_workers)

    def start(self):
        if self._run_mode != "batch_parallel":
            self._start_sequential()
            return
        self._start_batch_parallel()

    def _start_sequential(self):
        w = BatchWorker(context=self._context, parent=self, **self._args)
        self._solo_worker = w
        w.progress.connect(self.progress)
        w.frame_done.connect(self.frame_done)
        w.finished.connect(self.finished)
        w.failed.connect(self.failed)
        w.log_line.connect(self.log_line)
        w.geom_ready.connect(self.geom_ready)
        w.start()

    def _start_batch_parallel(self):
        try:
            source = _open_source_cfg(self._args["source_cfg"])
            n_frames = source.n_frames
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        indices = resolve_frame_indices(n_frames, self._args["frame_range"])
        if not indices:
            self.failed.emit(describe_empty_frame_range(
                n_frames, self._args["frame_range"], self._args.get("source_cfg")))
            return
        n_workers = resolve_worker_count(
            len(indices), self._n_workers_requested, self.MIN_FRAMES_PER_WORKER)
        if n_workers < self._n_workers_requested:
            self.log_line.emit(
                f"[batch] {len(indices)} frame(s) too few for "
                f"{self._n_workers_requested} requested workers (minimum "
                f"{self.MIN_FRAMES_PER_WORKER} frames/worker) — using {n_workers}.")
        if n_workers <= 1:
            self._start_sequential()
            return
        self._chunks = _split_into_chunks(indices, n_workers)
        self.log_line.emit(
            f"[batch] Batch Parallel: {len(indices)} frames across "
            f"{len(self._chunks)} workers ({[len(c) for c in self._chunks]})")
        if self._context is not None:
            self._on_geom_ready(self._context)
            return
        self.log_line.emit("[batch] Building geometry (one-time, shared)…")
        gw = _GeomBuildWorker(self._args["spec"], self._args["kernel"],
                              self._args["mask"], self._args["corrections"],
                              self._args["weighted"], parent=self)
        self._geom_worker = gw
        gw.done.connect(self._on_geom_ready)
        gw.failed.connect(self.failed)
        gw.start()

    def _on_geom_ready(self, ctx):
        if self._interrupted:
            self.finished.emit({"n": 0, "r_axis_px": ctx.get("r_ax"),
                                "profiles": np.array([]), "sigmas": np.array([]),
                                "frame_ids": [], "out_paths": [], "aborted": True})
            return
        self.geom_ready.emit(ctx)
        self._live_order = [i for chunk in self._chunks for i in chunk]
        for chunk in self._chunks:
            w = BatchWorker(context=ctx, frame_indices=chunk, parent=self, **self._args)
            self._chunk_workers.append(w)
            self._chunk_totals[id(w)] = len(chunk)
            self._chunk_done[id(w)] = 0
            self._chunk_frame_counter[id(w)] = 0
            w.progress.connect(lambda done, total, w=w: self._on_chunk_progress(w, done))
            w.frame_done.connect(lambda fid, r_ax, prof, sigma, w=w, chunk=chunk:
                                 self._on_chunk_frame(w, chunk, fid, r_ax, prof, sigma))
            w.finished.connect(lambda data, w=w: self._on_chunk_finished(w, data))
            w.failed.connect(self.failed)
            w.log_line.connect(self.log_line)
            w.start()

    def _on_chunk_frame(self, w, chunk, fid, r_ax, prof, sigma):
        """Relay one chunk worker's frame_done, re-ordered to match the overall
        sorted frame-index order (``_live_order``) instead of wall-clock
        completion order — concurrent chunks would otherwise interleave
        arbitrarily, scrambling the waterfall/stacked-profile display."""
        k = self._chunk_frame_counter[id(w)]
        self._chunk_frame_counter[id(w)] = k + 1
        abs_i = chunk[k]   # BatchWorker._iter_frames processes `chunk` in order
        self._live_pending[abs_i] = (fid, r_ax, prof, sigma)
        while (self._live_ptr < len(self._live_order)
               and self._live_order[self._live_ptr] in self._live_pending):
            i = self._live_order[self._live_ptr]
            self.frame_done.emit(*self._live_pending.pop(i))
            self._live_ptr += 1

    def _on_chunk_progress(self, w, done):
        self._chunk_done[id(w)] = done
        self.progress.emit(sum(self._chunk_done.values()), sum(self._chunk_totals.values()))

    def _on_chunk_finished(self, w, data):
        self._chunk_results[id(w)] = data
        if len(self._chunk_results) < len(self._chunk_workers):
            return
        # All chunks reported — merge, preserving overall frame order (each
        # chunk is a contiguous slice of the sorted index list).
        merged_profiles, merged_sigmas, merged_ids, merged_out = [], [], [], []
        aborted = False
        r_axis = None
        eta_axis = None
        for cw in self._chunk_workers:
            d = self._chunk_results[id(cw)]
            if d.get("profiles") is not None and len(d["profiles"]):
                merged_profiles.extend(d["profiles"])
            if d.get("sigmas") is not None and len(d["sigmas"]):
                merged_sigmas.extend(d["sigmas"])
            merged_ids.extend(d.get("frame_ids") or [])
            merged_out.extend(d.get("out_paths") or [])
            aborted = aborted or d.get("aborted", False)
            if r_axis is None:
                r_axis = d.get("r_axis_px")
            if eta_axis is None:
                eta_axis = d.get("eta_axis")
        multi_azimuth = bool(self._args.get("multi_azimuth", False))
        out_dir = self._args["out_dir"]
        fmts = self._args["fmts"] or []
        if out_dir and "h5" in fmts and merged_profiles:
            if multi_azimuth:
                self.log_line.emit(
                    "[batch] Note: combined HDF5 output isn't written in "
                    "multi-azimuth mode — use the text formats or GSAS-II "
                    "zarr export instead.")
            else:
                try:
                    spec = self._args["spec"]
                    h5_paths = write_all_profiles(
                        out_dir, ["h5"], r_axis, merged_profiles, merged_sigmas, merged_ids,
                        float(spec.Lsd), float(spec.pxY), float(spec.Wavelength))
                    merged_out.extend(h5_paths)
                except Exception:
                    self.log_line.emit(
                        "[batch] combined HDF5 write failed:\n" + traceback.format_exc())
        self.finished.emit({
            "n": len(merged_profiles), "r_axis_px": r_axis,
            "profiles": np.array(merged_profiles) if merged_profiles else np.array([]),
            "sigmas": np.array(merged_sigmas) if merged_sigmas else np.array([]),
            "frame_ids": merged_ids, "out_paths": merged_out, "aborted": aborted,
            "multi_azimuth": multi_azimuth, "eta_axis": eta_axis,
        })

    def requestInterruption(self):
        self._interrupted = True
        if self._solo_worker is not None:
            self._solo_worker.requestInterruption()
        for w in self._chunk_workers:
            w.requestInterruption()

    def wait(self, ms=None) -> bool:
        import time
        deadline = None if ms is None else time.monotonic() + ms / 1000.0
        workers = ([self._geom_worker] if self._geom_worker is not None else []) + \
                  ([self._solo_worker] if self._solo_worker is not None else list(self._chunk_workers))
        ok = True
        for w in workers:
            if w is None or not w.isRunning():
                continue
            remaining_ms = 30000 if deadline is None else max(0, int((deadline - time.monotonic()) * 1000))
            if not w.wait(remaining_ms):
                ok = False
        return ok


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 4 — Calibration refinement (autograd against integrated profile)
# ═════════════════════════════════════════════════════════════════════════════

class RefinementWorker(QtCore.QThread):
    """Refine geometry by minimising an integrated-profile loss via autograd.

    Default loss is EtaUniformityLoss (rings should be flat in η).  The image is
    normalised to O(1) so the loss is well-conditioned; each refined parameter
    gets a unit-appropriate Adam learning rate; gradients are clipped and a NaN
    guard reverts to the last good geometry.
    """
    progress = QtCore.pyqtSignal(int, int, float, dict)   # step, total, loss, params
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)                   # updated AutoCalibrationResult
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, result, image, dark, mask, refine_names, *,
                 loss_kind="eta_uniformity", optimizer="adam", lr=0.5,
                 iters=100, r_bin=2.0, eta_bin=5.0, parent=None,
                 bright=None, background=None, bright_mode="divide"):
        super().__init__(parent)
        self._result, self._image, self._dark, self._mask = result, image, dark, mask
        self._names = refine_names
        self._loss_kind, self._optimizer = loss_kind, optimizer
        self._lr, self._iters = lr, iters
        self._r_bin, self._eta_bin = r_bin, eta_bin
        self._bright, self._background, self._bright_mode = bright, background, bright_mode

    # Typical step per parameter — used to scale the optimiser's search space so
    # every coordinate is O(1) (Nelder-Mead is scale-sensitive).
    # BC_y/BC_z use 0.5 px (not 2 px) for two reasons:
    #   1. η-uniformity is weakly sensitive to BC: shifting the beam centre
    #      translates rings but keeps them circular, producing near-zero
    #      azimuthal-variance signal. Tilts (ty/tz) deform rings into
    #      ellipses → large, unambiguous signal.
    #   2. Hard-bin (floor) assignment creates discrete steps in the loss.
    #      Large BC steps jump many pixels across bin boundaries, turning
    #      the objective into a noisy staircase that misleads Nelder-Mead.
    # Smaller step (0.5 px) + MAX_STEPS=3 → ±1.5 px exploration window.
    _STEP = {"Lsd": 500.0, "BC_y": 0.5, "BC_z": 0.5,
             "ty": 0.1, "tz": 0.1, "tx": 0.1, "Wavelength": 1e-4}
    # Map GUI/spec param name → AutoCalibrationResult attribute
    _ATTR = {"Lsd": "Lsd", "BC_y": "BC_y", "BC_z": "BC_z", "ty": "ty",
             "tz": "tz", "tx": "tx", "Wavelength": "wavelength_A"}

    def run(self):
        try:
            import copy
            import torch
            from scipy.optimize import minimize
            import midas_integrate_v2 as m

            spec = _build_spec(self._result, self._r_bin, self._eta_bin)
            # spec.TransOpt (set by _build_spec from self._result.im_trans) makes
            # every integrate_hard() call below flip img_t internally — so the
            # raw loader frame/dark/bright/background are used exactly as loaded.
            # Only the mask is pre-flipped: it's baked into build_geom()'s
            # geometry map against the *transformed* pixel grid, with no
            # backend-side apply_trans_opt hook of its own.
            dark, bright, background = self._dark, self._bright, self._background
            im_trans = tuple(getattr(self._result, "im_trans", ()) or ())
            mask = self._mask   # RAW space — matches img below, zeroed pointwise
            img = self._image.astype(np.float64)
            if dark is not None or bright is not None or background is not None:
                img = apply_field_corrections(
                    img, dark=dark, bright=bright,
                    bright_mode=self._bright_mode, background=background)
            # Prepare mask tensor once — passed to the geometry builder each iteration
            # so masked pixels are excluded from both intensity sums AND bin counts.
            # Zeroing in the image alone is not enough: HardBinGeometry still counts
            # those pixels in its normalisation, dragging down the mean and inflating
            # the η-variance, biasing the loss.
            if mask is not None:
                img = img.copy(); img[mask.astype(bool)] = 0.0   # img is still RAW here
                # mask_t excludes pixels at build_geom() time, which bins against
                # the *transformed* pixel grid — unlike img above, it needs the flip.
                mask_xf = (_apply_im_trans(mask.astype(np.float32), im_trans)
                           if im_trans else mask.astype(np.float32))
                mask_t = torch.from_numpy(mask_xf)
                n_bad = int(mask.astype(bool).sum())
                self.log_line.emit(
                    f"[refine] mask active: {n_bad:,} px excluded ({100*n_bad/mask.size:.2f}%)")
            else:
                mask_t = None
                self.log_line.emit("[refine] mask: none — all pixels included")
            scale = float(np.mean(img[img > 0])) or 1.0
            img_t = torch.from_numpy(img / scale)

            refined = [n for n in self._names if isinstance(getattr(spec, n, None), torch.Tensor)]
            if not refined:
                raise RuntimeError("No refinable parameters selected.")
            base = {n: float(getattr(spec, n).detach()) for n in refined}
            self.log_line.emit(f"[refine] refining {refined} (derivative-free Nelder-Mead)")
            self.log_line.emit(
                f"[refine] start: {', '.join(f'{k}={v:.5g}' for k, v in base.items())}")
            # BC_y/BC_z warning: η-uniformity barely changes when the beam
            # centre shifts (rings translate but stay circular), so the loss
            # landscape is nearly flat in BC.  A small L2 anchor prevents drift.
            bc_indices = [i for i, nm in enumerate(refined) if nm in ("BC_y", "BC_z")]
            if bc_indices:
                self.log_line.emit(
                    "[refine] BC note: η-uniformity has weak sensitivity to BC "
                    "(rings stay circular when centre shifts). "
                    "Step limited to ±0.5 px, L2 anchor active. "
                    "Use Tab 2 calibration for large BC corrections.")

            # MAX_STEPS: maximum search radius in normalised units.
            # Prevents the optimizer from exploring geometry where rings fall outside
            # the integration R-range, which produces an artificially uniform (empty)
            # cake that the objective misidentifies as a perfect minimum.
            # Physical limits: BC ±1.5 px, tilt ±0.3°, Lsd ±1500 µm, λ ±3e-4 Å.
            MAX_STEPS = 3.0

            def _set(x):
                with torch.no_grad():
                    for i, n in enumerate(refined):
                        getattr(spec, n).copy_(torch.tensor(base[n] + x[i] * self._STEP[n]))

            # Track the last real loss and the initial-loss reference for scaling
            # the BC regularisation weight (set after f0 is computed below).
            last_loss = [np.nan]
            f0_ref    = [1.0]   # updated after first evaluation; 0 bc_reg at x0

            def _objective(x):
                # Hard bounds: return a steep penalty without modifying the spec.
                # This keeps the optimizer inside the physically meaningful region.
                if np.any(np.abs(x) > MAX_STEPS):
                    return (np.nan_to_num(last_loss[0], nan=1.0)) * 100 + 1.0
                _set(x)
                geom = build_geom(spec, "hard", mask_t)   # mask excludes bad px from sums AND counts
                int2d = m.integrate_hard(img_t, geom, normalize=True).detach().cpu().numpy()
                int2d = np.nan_to_num(int2d, nan=0.0)
                m_e = int2d.mean(axis=0); v_e = int2d.var(axis=0)
                w = np.clip(m_e, 0, None)
                denom = float((w * w).sum())
                # Guard: if nearly all bins are empty the geometry collapsed rings
                # outside the R-range → this is a degenerate minimum, not a real one.
                if denom < 1e-4:
                    return (np.nan_to_num(last_loss[0], nan=1.0)) * 100 + 1.0
                eta_loss = float((v_e * w).sum() / denom)
                # L2 anchor for BC: 0.2 % of f0 per unit step.  Keeps BC from
                # drifting across the flat η-landscape; allows corrections where
                # the signal genuinely exceeds the regularisation cost (~1.8 % of
                # f0 at the 1.5 px hard limit).  Zero at x=0, so f0 is pure loss.
                bc_reg = (f0_ref[0] * 2e-3) * sum(x[i] ** 2 for i in bc_indices)
                loss = eta_loss + bc_reg
                last_loss[0] = loss
                return loss

            self._eval = 0
            n = len(refined)
            x0 = np.zeros(n)
            f0 = _objective(x0)   # bc_reg = 0 at x0 → f0 = pure η-loss
            f0_ref[0] = f0        # now regularisation weight is set
            self.log_line.emit(f"[refine] initial loss = {f0:.6g}")
            self.progress.emit(0, self._iters, f0, dict(base))

            def _cb(xk):
                self._eval += 1
                # Use the cached loss — do NOT re-call _objective here, that
                # would double the evaluation count and corrupt the spec state
                # between the optimizer's internal steps.
                params = {nm: base[nm] + xk[i] * self._STEP[nm]
                          for i, nm in enumerate(refined)}
                self.progress.emit(min(self._eval, self._iters), self._iters,
                                   np.nan_to_num(last_loss[0], nan=f0), params)

            # Symmetric simplex: explore both + and − directions so the optimizer
            # does not have to reflect past x0 before it can search all of them.
            rows = [x0]
            for i in range(n):
                v = x0.copy(); v[i] =  1.5; rows.append(v)
            # For n > 1, add a few negative-direction vertices within the n+1 limit.
            for i in range(min(n, n)):
                v = x0.copy(); v[i] = -0.75; rows.append(v)
            simplex = np.array(rows[:n + 1])

            # maxiter: each Nelder-Mead iteration is 1-4 evaluations; for n ≥ 4
            # convergence typically needs 200-500 iterations.  Use at least 400.
            res = minimize(_objective, x0, method="Nelder-Mead", callback=_cb,
                           options={"maxiter": max(self._iters, 400),
                                    "initial_simplex": simplex,
                                    "xatol": 1e-3, "fatol": 1e-5, "disp": False})

            # Safety: if the optimiser returned a worse geometry, revert.
            if res.fun > f0 * 1.05:
                self.log_line.emit(
                    f"[refine] WARN: optimised loss ({res.fun:.5g}) is worse than "
                    f"starting loss ({f0:.5g}). Reverting to original geometry.")
                self.finished.emit(copy.copy(self._result))
                return

            _set(res.x)
            final = {name: base[name] + res.x[i] * self._STEP[name]
                     for i, name in enumerate(refined)}
            self.log_line.emit(
                f"[refine] final loss={res.fun:.6g}  ({res.nfev} evals, {res.nit} iters)"
                f"  converged={res.success}")
            self.log_line.emit(
                f"[refine] Δ: {', '.join(f'{nm}={final[nm]-base[nm]:+.4g}' for nm in refined)}")

            new = copy.copy(self._result)
            for name, attr in self._ATTR.items():
                if name in refined:
                    setattr(new, attr, final[name])
            new._calibrant_name = getattr(self._result, "_calibrant_name", "CeO2")
            self.finished.emit(new)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 4b — Profile comparison (before/after refinement)
# ═════════════════════════════════════════════════════════════════════════════

class RefineCompareWorker(QtCore.QThread):
    """Integrate one frame with both the original and refined calibration results.

    Returns profiles on a common R axis so the tab can overlay them and show
    the difference curve.
    """
    finished = QtCore.pyqtSignal(object)   # dict: r_axis_px, profile_orig, profile_refined
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, orig_result, refined_result, image, mask=None,
                 r_bin=2.0, eta_bin=5.0, parent=None,
                 dark=None, bright=None, background=None, bright_mode="divide"):
        super().__init__(parent)
        self._orig = orig_result
        self._refined = refined_result
        self._image = image
        self._mask = mask
        self._r_bin = r_bin
        self._eta_bin = eta_bin
        self._dark, self._bright, self._background = dark, bright, background
        self._bright_mode = bright_mode

    def run(self):
        try:
            import torch
            # _build_spec (below, per result) sets spec.TransOpt from each
            # result's im_trans, so img_t below stays exactly as loaded — only
            # the mask is pre-flipped (see RefinementWorker for why). Both
            # results share the same im_trans lineage (refinement never
            # touches ImTransOpt).
            im_trans = tuple(getattr(self._orig, "im_trans", ()) or ())
            dark, bright, background, mask = (
                self._dark, self._bright, self._background, self._mask)
            if im_trans and mask is not None:
                mask = _apply_im_trans(mask.astype(np.float32), im_trans)
            mask_t = torch.from_numpy(mask.astype(np.float32)) if mask is not None else None
            img = self._image.astype(np.float64)
            if dark is not None or bright is not None or background is not None:
                img = apply_field_corrections(
                    img, dark=dark, bright=bright,
                    bright_mode=self._bright_mode, background=background)
            img_t = torch.from_numpy(img)
            profiles, r_axes = [], []
            for res in (self._orig, self._refined):
                spec = _build_spec(res, self._r_bin, self._eta_bin)
                geom = build_geom(spec, "subpixel2", mask_t)
                prof, _ = integrate_frame(img_t, spec, geom, "subpixel2",
                                          (None, None), None, need_sigma=False)
                profiles.append(prof)
                r_axes.append(compute_r_axis(spec))
            # Interpolate both onto the overlapping R range so subtraction is valid
            r0, r1 = r_axes
            r_min = max(float(r0.min()), float(r1.min()))
            r_max = min(float(r0.max()), float(r1.max()))
            n_bins = min(len(r0), len(r1))
            r_common = np.linspace(r_min, r_max, n_bins)
            p_orig = np.interp(r_common, r0, profiles[0])
            p_ref  = np.interp(r_common, r1, profiles[1])
            self.finished.emit({
                "r_axis_px": r_common,
                "profile_orig": p_orig,
                "profile_refined": p_ref,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 5 — Corrections & Physics preview (single frame)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_composition(text: str) -> dict:
    """Parse 'Ce:1,O:2' → {'Ce':1.0,'O':2.0}."""
    out = {}
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            el, frac = tok.split(":")
            out[el.strip()] = float(frac)
        else:
            out[tok] = 1.0
    return out


class CorrectionPreviewWorker(QtCore.QThread):
    """Integrate one frame with and without the selected corrections so the user
    can see the effect of each before committing to a batch run.

    Pixel-domain (via integrate_with_corrections): polarization, solid angle,
    empty subtraction.  Profile-domain: cylindrical absorption (÷ transmission),
    Compton subtraction (− incoherent intensity).
    """
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, result, image, dark, mask, cfg, parent=None):
        super().__init__(parent)
        self._result, self._image, self._dark, self._mask = result, image, dark, mask
        self._cfg = cfg

    def run(self):
        try:
            import torch
            import midas_integrate_v2 as m
            c = self._cfg
            spec = _build_spec(self._result, c.get("r_bin", 1.0), c.get("eta_bin", 5.0))
            img = self._image.astype(np.float64)
            if self._dark is not None:
                img = np.clip(img - self._dark.astype(np.float64), 0, None)
            if self._mask is not None:
                img = img.copy(); img[self._mask.astype(bool)] = 0.0
            img_t = torch.from_numpy(img)
            lsd, px, wl = float(spec.Lsd), float(spec.pxY), float(spec.Wavelength)

            # Uncorrected reference
            geom = build_geom(spec, "subpixel2", None)
            prof_unc, _ = integrate_frame(img_t, spec, geom, "subpixel2",
                                          (None, None), None, need_sigma=False)

            # Pixel-domain corrections
            pol = sa = empty = None
            if c.get("polarization"):
                pol = m.PolarizationCorrection(
                    pol_fraction=c["polarization"]["frac"],
                    pol_plane_eta_deg=c["polarization"]["plane"])
            if c.get("solid_angle"):
                sa = m.SolidAngleCorrection()
            if c.get("empty"):
                ev = c["empty"]
                empty_img = _load_image(ev["path"]).astype(np.float64)
                empty = m.EmptySubtraction(torch.from_numpy(empty_img),
                                           scale=ev.get("scale", 1.0))
            cake = m.integrate_with_corrections(
                img_t, spec, polarization=pol, solid_angle=sa,
                empty_subtraction=empty).detach().cpu().numpy()
            # integrate_with_corrections is unnormalised → divide by the pixel-count cake
            counts = corrections_counts(spec)
            with np.errstate(invalid="ignore", divide="ignore"):
                cake = np.where(counts > 0.5, cake / counts, np.nan)
            prof = _profile_from_cake(cake)

            r_ax = compute_r_axis(spec)
            two_theta = np.degrees(np.arctan(r_ax * px / lsd))
            q = 4 * math.pi * np.sin(np.radians(two_theta) / 2) / wl

            # Profile-domain corrections
            if c.get("absorption"):
                T = m.CylindricalAbsorption(mu_R=c["absorption"]["mu_R"]) \
                    (torch.from_numpy(np.radians(two_theta))).detach().cpu().numpy()
                T = np.clip(T, 1e-6, None)
                prof = prof / T
                self.log_line.emit(f"[corr] absorption μR={c['absorption']['mu_R']} "
                                   f"T range [{T.min():.3f},{T.max():.3f}]")
            if c.get("compton"):
                comp_cfg = c["compton"]
                comp = m.ComptonSubtraction(_parse_composition(comp_cfg["composition"]),
                                            wavelength_A=wl) \
                    (torch.from_numpy(q)).detach().cpu().numpy()
                comp = comp * comp_cfg.get("scale", 1.0)
                prof = prof - comp
                self.log_line.emit("[corr] Compton subtracted "
                                   f"(max {comp.max():.3g})")

            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(prof_unc > 0, prof / prof_unc, np.nan)

            self.finished.emit({
                "r_axis_px": r_ax, "profile_unc": prof_unc, "profile_corr": prof,
                "factor": factor, "two_theta": two_theta, "q": q,
                "wavelength_A": wl, "lsd_um": lsd, "px_um": px,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 6 — PDF analysis (image → I(Q) → G(r))
# ═════════════════════════════════════════════════════════════════════════════

def _to_np(x):
    """torch tensor or numpy → contiguous float numpy array."""
    if x is None:
        return None
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x,
                      dtype=np.float64)


def _load_iq_file(path: str):
    """Load a pre-integrated I(Q) file → (Q, I, sigma_or_None).

    Tolerates comma- or whitespace-separated 2- or 3-column data with a leading
    comment/header line (``#`` comments skipped; a non-numeric first row is too).
    """
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
        raise ValueError(f"I(Q) file needs ≥2 columns (Q, I); got {arr.shape[1]}.")
    q = arr[:, 0].astype(np.float64)
    intensity = arr[:, 1].astype(np.float64)
    sigma = arr[:, 2].astype(np.float64) if arr.shape[1] >= 3 else None
    return q, intensity, sigma


def _fit_subtraction_scale(I_meas, I_empty, q, comp, wavelength_A, q_min, q_max):
    """Least-squares empty-cell scale ``s`` (and offset ``c``).

    Away from Bragg peaks / at high Q the sample's own scattering tends to
    its self-scattering baseline ``⟨f²⟩+Compton`` (S(Q)→1), so
    ``I_meas - s·I_empty - c`` is fit to that baseline over
    ``[q_min, q_max]`` via ``np.linalg.lstsq``. Returns ``(s, c)``.
    """
    sel = (q >= q_min) & (q <= q_max)
    if sel.sum() < 4:
        raise ValueError(
            f"Too few points in fit window [{q_min}, {q_max}] to fit background scale.")
    f2, _ = comp.form_factor_averages(q[sel], wavelength_A=wavelength_A, anomalous=True)
    inc = comp.compton(q[sel], wavelength_A=wavelength_A)
    baseline = _to_np(f2) + _to_np(inc)
    target = I_meas[sel] - baseline
    A = np.column_stack([I_empty[sel], np.ones(int(sel.sum()))])
    (s, c), *_ = np.linalg.lstsq(A, target, rcond=None)
    return float(s), float(c)


def _absolute_normalize(I, q, comp, wavelength_A, q_window, anomalous=True):
    """Anchor the mean intensity over ``q_window`` to ``⟨f²⟩+⟨S_inc⟩``.

    Returns ``(I_normalized, K)`` with the scalar gain ``K`` so σ propagates
    as ``sigma*K`` (never an elementwise ratio, which blows up near I≈0).
    """
    q_lo, q_hi = q_window
    sel = (q >= q_lo) & (q <= q_hi)
    if sel.sum() < 4:
        raise ValueError(f"Too few points in normalization window [{q_lo}, {q_hi}].")
    f2, _ = comp.form_factor_averages(q[sel], wavelength_A=wavelength_A, anomalous=anomalous)
    inc = comp.compton(q[sel], wavelength_A=wavelength_A)
    baseline = _to_np(f2) + _to_np(inc)
    K = float(np.mean(baseline) / np.mean(I[sel]))
    return I * K, K


def _flatten_sq_tail(q, S, window, poly_deg=3, mad_k=3.0, n_iter=3):
    """PDFgetX3-style iterative MAD-clipped polynomial baseline flatten.

    Fits a degree-``poly_deg`` polynomial to ``S(Q)`` over ``window``,
    iteratively re-fitting after dropping points whose residual exceeds
    ``mad_k`` scaled MADs (Bragg peaks), then subtracts the fitted drift
    (recentring on 1) over the *full* Q range. Display-only — never fed back
    into G(r). Returns ``(S_flat, poly_coeffs)``.
    """
    q_lo, q_hi = window
    sel = (q >= q_lo) & (q <= q_hi)
    if sel.sum() < poly_deg + 2:
        raise ValueError("Too few points in tail-flatten window for the requested polynomial degree.")
    qs, Ss = q[sel], S[sel]
    poly = np.polyfit(qs, Ss, poly_deg)
    for _ in range(max(1, int(n_iter))):
        fit = np.polyval(poly, qs)
        resid = Ss - fit
        mad = np.median(np.abs(resid - np.median(resid))) or 1e-12
        keep = np.abs(resid - np.median(resid)) <= mad_k * 1.4826 * mad
        if keep.sum() < poly_deg + 2:
            break
        poly = np.polyfit(qs[keep], Ss[keep], poly_deg)
    baseline_full = np.polyval(poly, q)
    S_flat = S - baseline_full + 1.0
    return S_flat, poly


class PDFWorker(QtCore.QThread):
    """Polyatomic total-scattering PDF: I(Q) → Faber-Ziman S(Q) → G(r).

    Uses the ``midas_pdf`` backend (via :mod:`midas_gui.pdf_backend`): real
    composition-weighted normalization, Compton subtraction, end-to-end σ
    propagation, and optional differentiable scale/background refinement.
    I(Q) comes either from integrating a detector frame or from a pre-integrated
    ``Q,I,σ`` file.
    """
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, result, image, dark, mask, cfg, parent=None):
        super().__init__(parent)
        self._result, self._image, self._dark, self._mask = result, image, dark, mask
        self._cfg = cfg

    def _acquire_iq(self, c):
        """Return (q, I, sigma_I) either from the frame or a Q,I,σ file."""
        if c.get("iq_source", "image") == "file":
            path = c.get("iq_file", "")
            if not path or not Path(path).exists():
                raise FileNotFoundError(f"I(Q) file not found: {path!r}")
            self.log_line.emit(f"[pdf] loading I(Q) from {Path(path).name}")
            q, I, sig = _load_iq_file(path)
            return q, I, sig

        # image mode — integrate the frame (with Poisson variance for σ_I)
        import torch
        if self._result is None or self._image is None:
            raise ValueError("Image mode needs a calibration and a loaded frame.")
        spec = _build_spec(self._result, 1.0, c.get("eta_bin", 5.0))
        img = self._image.astype(np.float64)
        if self._dark is not None:
            img = np.clip(img - self._dark.astype(np.float64), 0, None)
        if self._mask is not None:
            img = img.copy(); img[self._mask.astype(bool)] = 0.0
        img_t = torch.from_numpy(img)
        lsd, px, wl = float(spec.Lsd), float(spec.pxY), float(spec.Wavelength)
        kernel = c.get("binning", "hard")
        self.log_line.emit(f"[pdf] integrating frame ({kernel} binning)…")
        geom = build_geom(spec, kernel, None)
        prof, sigma = integrate_frame(img_t, spec, geom, kernel, (None, None),
                                      {"error_model": "poisson"}, need_sigma=True)
        r_ax = compute_r_axis(spec)
        _, _, q = axis_conversions(r_ax, lsd, px, wl)
        return np.asarray(q, dtype=np.float64), prof.astype(np.float64), sigma.astype(np.float64)

    def run(self):
        try:
            import torch
            import midas_gui.pdf_backend as pdf
            c = self._cfg
            wl = float(c["wavelength"])
            rho0 = float(c.get("rho0") or 0.0)
            compton = bool(c.get("compton", True))
            window = c.get("window", "lorch")
            q_min, q_max = float(c["q_min"]), float(c["q_max"])

            q, I, sig = self._acquire_iq(c)

            # trim to [q_min, q_max]
            sel = (q >= q_min) & (q <= q_max) & np.isfinite(I)
            if sel.sum() < 8:
                raise ValueError(f"Too few I(Q) points in [{q_min}, {q_max}] Å⁻¹.")
            q, I = q[sel], I[sel]
            sig = sig[sel] if sig is not None else None

            comp = pdf.Composition(_parse_composition(c["composition"]),
                                   number_density=(rho0 or None))
            self.log_line.emit(
                f"[pdf] composition={comp.as_dict()}  ρ₀={rho0 or '—'}  "
                f"λ={wl:.5f} Å  compton={'on' if compton else 'off'}")

            r = np.arange(c["r_min"], c["r_max"], c["r_step"], dtype=np.float64)

            # ── Stage 2-3, step 1: empty-cell / background subtraction ──────────
            bg_scale_used = None
            if c.get("bg_enabled"):
                bg_cfg = c.get("bg") or {}
                bg_path = bg_cfg.get("iq_file", "")
                if not bg_path or not Path(bg_path).exists():
                    raise FileNotFoundError(f"Empty-cell I(Q) file not found: {bg_path!r}")
                q_bg, I_bg, sig_bg = _load_iq_file(bg_path)
                if q_bg.shape != q.shape or not np.allclose(q_bg, q):
                    sig_bg_interp = (np.interp(q, q_bg, sig_bg) if sig_bg is not None else None)
                    I_bg = np.interp(q, q_bg, I_bg)
                    sig_bg = sig_bg_interp

                # physical/fit transmission scale s (attenuator ratio, or a
                # least-squares high-Q fit) — this is independent of whether
                # the Q-dependent Paalman-Pings correction is layered on top.
                mode = bg_cfg.get("mode", "manual")
                if mode == "fit":
                    fit_q = bg_cfg.get("fit_q") or (q_max * 0.7, q_max)
                    s_manual, _off = _fit_subtraction_scale(
                        I, I_bg, q, comp, wl, float(fit_q[0]), float(fit_q[1]))
                else:
                    s_manual = float(bg_cfg.get("scale", 1.0))

                if bg_cfg.get("paalman_pings"):
                    pp = pdf.paalman_pings_cylinder_in_cylinder(
                        q, wavelength_A=wl,
                        mu_sample_um=float(bg_cfg["mu_sample_um"]),
                        mu_container_um=float(bg_cfg["mu_container_um"]),
                        R_sample_um=float(bg_cfg["r_sample_um"]),
                        R_container_um=float(bg_cfg["r_container_um"]))
                    # I_sample = [I_meas - (A_c_sc/A_c_c)·s·I_empty] / A_s_sc
                    scale_arr = (_to_np(pp["A_c_sc"]) / _to_np(pp["A_c_c"])) * s_manual
                    denom = _to_np(pp["A_s_sc"])
                    self.log_line.emit(
                        f"[pdf] Paalman-Pings empty-cell subtraction "
                        f"(s={s_manual:.4g}, median A_c_sc/A_c_c={np.median(scale_arr / s_manual):.4g})")
                else:
                    scale_arr = s_manual
                    denom = 1.0
                    self.log_line.emit(
                        f"[pdf] empty-cell subtraction (mode={mode}, s={s_manual:.4g})")

                I = (I - scale_arr * I_bg) / denom
                bg_scale_used = float(np.median(np.atleast_1d(scale_arr)))

                if sig is not None and sig_bg is not None:
                    sig = np.sqrt(sig ** 2 + (scale_arr * sig_bg) ** 2) / denom

            # ── Stage 2-3, step 2: detector efficiency ───────────────────────────
            if c.get("det_eff_enabled"):
                de_cfg = c.get("det_eff") or {}
                I_t, sig_t = pdf.apply_detector_efficiency(
                    torch.as_tensor(I, dtype=torch.float64),
                    torch.as_tensor(q, dtype=torch.float64),
                    wavelength_A=wl,
                    material=de_cfg.get("material", "Si"),
                    thickness_um=float(de_cfg.get("thickness_um", 500.0)),
                    density_g_cm3=de_cfg.get("density_g_cm3"),
                    sigma=(torch.as_tensor(sig, dtype=torch.float64) if sig is not None else None))
                I = _to_np(I_t)
                if sig_t is not None:
                    sig = _to_np(sig_t)
                self.log_line.emit(
                    f"[pdf] detector efficiency correction applied "
                    f"(material={de_cfg.get('material', 'Si')}, "
                    f"thickness={de_cfg.get('thickness_um', 500.0)} µm)")

            # ── Stage 2-3, step 3: absolute normalization ────────────────────────
            if c.get("absnorm_enabled"):
                an_cfg = c.get("absnorm") or {}
                q_win = an_cfg.get("q_window") or (q_max * 0.7, q_max)
                I, K = _absolute_normalize(
                    I, q, comp, wl, (float(q_win[0]), float(q_win[1])),
                    anomalous=bool(an_cfg.get("anomalous", True)))
                if sig is not None:
                    sig = sig * K
                self.log_line.emit(f"[pdf] absolute normalization K={K:.4g}")

            # ── Stage 2-3, step 4: differentiable multiple scattering ───────────
            ms_beta_median = None
            if c.get("ms_enabled"):
                ms_cfg = c.get("ms") or {}
                mu_um = ms_cfg.get("mu_um")
                if mu_um is None:
                    comp_dict = _parse_composition(c["composition"])
                    if len(comp_dict) == 1:
                        material = next(iter(comp_dict))
                        mu_um = pdf.linear_attenuation_um(
                            material, wl, density_g_cm3=ms_cfg.get("density_g_cm3"))
                    else:
                        density = ms_cfg.get("density_g_cm3")
                        if density is None:
                            raise ValueError(
                                "Multiple scattering: a compound sample needs an explicit "
                                "density (g/cm³) to auto-estimate μ, or set μ manually.")
                        mu_um = pdf.linear_attenuation_um(
                            comp_dict, wl, density_g_cm3=float(density))
                else:
                    mu_um = float(mu_um)
                R_um = float(ms_cfg.get("r_um", 500.0))
                tau = pdf.cylinder_effective_tau(mu_um, R_um)
                ms = pdf.slab_transport_ms(
                    comp, wavelength_A=wl, tau=tau,
                    albedo=float(ms_cfg.get("albedo", 0.9)),
                    q_max=float(ms_cfg.get("q_max", q_max)),
                    n_mu=int(ms_cfg.get("n_mu", 32)), n_tau=int(ms_cfg.get("n_tau", 100)))
                ms_bg = _to_np(pdf.ms_background_on_grid(q, I, ms))
                I = I - ms_bg
                ms_beta_median = float(np.median(_to_np(ms["beta"])))
                self.log_line.emit(
                    f"[pdf] multiple-scattering correction: τ={tau:.4g}, "
                    f"median β={ms_beta_median:.4g}")

            background = None
            scale = 1.0
            bg_coef = None
            refine_loss = None

            if c.get("refine"):
                if rho0 <= 0:
                    raise ValueError("Refinement needs a number density ρ₀ > 0.")
                steps = int(c.get("refine_steps", 120))
                self.log_line.emit(f"[pdf] refining normalization "
                                   f"(bg_order={c.get('bg_order', 0)}, steps={steps})…")
                res = pdf.refine_normalization(
                    q, I, comp, r, wavelength_A=wl, number_density=rho0,
                    sigma_intensity=sig, compton=compton, q_max=q_max, window=window,
                    r_min_phys=float(c.get("r_min_phys", 1.5)),
                    bg_order=int(c.get("bg_order", 0)), steps=steps)
                S = res.S; G = res.G; sigma_G = res.sigma_G
                background = _to_np(res.background)
                scale = float(res.scale)
                bg_coef = [float(x) for x in (res.bg_coef or [])]
                refine_loss = float(res.history[-1]) if res.history else None
                self.log_line.emit(
                    f"[pdf] refined: scale={scale:.4g}  loss={refine_loss:.4g}")
            else:
                G, sigma_G, S = pdf.i_of_q_to_Gr(
                    q, I, comp, r, wavelength_A=wl, sigma_intensity=sig,
                    compton=compton, q_max=q_max, window=window)

            # true reduced structure function F(Q) = Q·(S−1)
            Fq, _ = pdf.structure_function_F(q, S)

            S_np = _to_np(S); G_np = _to_np(G); sigG_np = _to_np(sigma_G)
            Fq_np = _to_np(Fq)

            # ── Stage 2-3, step 5: display-only S(Q) tail-flatten ────────────────
            S_flat_np = None
            if c.get("tail_flatten_enabled"):
                tf_cfg = c.get("tail_flatten") or {}
                window = tf_cfg.get("window") or (q_max * 0.6, q_max)
                S_flat_np, _poly = _flatten_sq_tail(
                    q, S_np, (float(window[0]), float(window[1])),
                    poly_deg=int(tf_cfg.get("poly_deg", 3)),
                    mad_k=float(tf_cfg.get("mad_k", 3.0)),
                    n_iter=int(tf_cfg.get("n_iter", 3)))
                self.log_line.emit("[pdf] tail-flattened S(Q) computed for display")

            # output-function family (needs ρ₀ for g/T/R)
            out_fn = c.get("output_fn", "G")
            fam_y, fam_sig = G_np, sigG_np
            if out_fn != "G" and rho0 > 0:
                fn = {"g": pdf.pair_distribution_g,
                      "T": pdf.total_correlation_T,
                      "R": pdf.radial_distribution_R}[out_fn]
                y, ys = fn(r, G, number_density=rho0, sigma_G=sigma_G)
                fam_y, fam_sig = _to_np(y), _to_np(ys)
            elif out_fn != "G":
                self.log_line.emit(f"[pdf] {out_fn}(r) needs ρ₀>0 — showing G(r).")
                out_fn = "G"

            self.finished.emit({
                "q": q, "Iq": I, "background": background,
                "S": S_np, "S_flat": S_flat_np, "Fq": Fq_np,
                "r": r, "Gr": G_np, "sigma_Gr": sigG_np,
                "Gr_family": {"name": out_fn, "y": fam_y, "sigma": fam_sig},
                "scale": scale, "bg_coef": bg_coef, "refine_loss": refine_loss,
                "bg_scale_used": bg_scale_used, "ms_beta_median": ms_beta_median,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


class PDFStructureFitWorker(QtCore.QThread):
    """CIF-driven small-box structure refinement (PDFfit-style) against a
    previously computed G(r) snapshot.

    Fits lattice scale ``a``, isotropic ADP ``u_iso``, and an overall
    ``scale`` by autograd (differentiable ``pdffit_gr``), with
    Hessian-derived uncertainties from ``midas_pdf.refine_structure``.
    """
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, r, G, sigma_G, cfg, parent=None):
        super().__init__(parent)
        self._r, self._G, self._sigma_G = r, G, sigma_G
        self._cfg = cfg

    def _build_crystal_tensor(self, pdf, c):
        if c.get("crystal_source", "cif") == "cif":
            path = c.get("cif_path", "")
            if not path or not Path(path).exists():
                raise FileNotFoundError(f"CIF file not found: {path!r}")
            crystal = pdf.read_cif_to_crystal(path)
        else:
            a, b, cc, alpha, beta, gamma = [float(x) for x in c["manual_lattice"]]
            lattice = pdf.Lattice(a=a, b=b, c=cc, alpha=alpha, beta=beta, gamma=gamma)
            sg = pdf.SpaceGroup.from_number(int(c["space_group_number"]))
            atoms = [
                pdf.Atom(element=at["element"],
                        fract=(float(at["x"]), float(at["y"]), float(at["z"])),
                        occupancy=float(at.get("occupancy", 1.0)),
                        B_iso=float(at.get("B_iso", 0.0)))
                for at in c["manual_atoms"]
            ]
            crystal = pdf.Crystal(lattice=lattice, space_group=sg, atoms=atoms)
        return crystal.to_torch()

    def run(self):
        try:
            import midas_gui.pdf_backend as pdf
            c = self._cfg

            crystal_t = self._build_crystal_tensor(pdf, c)
            r_max = float(c.get("r_max", 10.0))
            pairs = pdf.build_pair_list(crystal_t, r_max=r_max)

            fit_lo = float(c.get("fit_r_min", 1.5))
            fit_hi = float(c.get("fit_r_max", r_max))
            sel = (self._r >= fit_lo) & (self._r <= fit_hi)
            if sel.sum() < 8:
                raise ValueError(f"Too few G(r) points in fit range [{fit_lo}, {fit_hi}] Å.")
            r_fit = self._r[sel]
            G_obs = self._G[sel]
            sigma_inflate = float(c.get("sigma_inflate", 1.0))
            sig_fit = (self._sigma_G[sel] * sigma_inflate
                      if self._sigma_G is not None else None)

            init_a = c.get("init_a")
            init_a = float(init_a) if init_a is not None else None
            bg_order = c.get("bg_order")
            bg_order = int(bg_order) if bg_order is not None else None
            steps = int(c.get("steps", 120))

            self.log_line.emit(
                f"[pdf-fit] refining structure over r∈[{fit_lo},{fit_hi}] Å "
                f"({int(sel.sum())} pts, {steps} steps)…")

            res = pdf.refine_structure(
                crystal_t, r_fit, G_obs, pairs, sigma_obs=sig_fit,
                init_a=init_a, init_u_iso=float(c.get("init_u_iso", 0.005)),
                init_scale=float(c.get("init_scale", 1.0)),
                bg_order=bg_order, steps=steps, lr=float(c.get("lr", 0.05)),
                n_posterior_samples=int(c.get("n_posterior_samples", 0)))

            self.log_line.emit(
                f"[pdf-fit] done: fitted={res['fitted']}  "
                f"chi2_reduced={float(res['chi2_reduced']):.4g}")

            self.finished.emit({
                "fitted": res["fitted"], "uncertainty": res["uncertainty"],
                "chi2_reduced": float(res["chi2_reduced"]), "history": res["history"],
                "r_fit": r_fit, "G_obs": _to_np(G_obs), "G_calc": _to_np(res["G_calc"]),
                "sigma_fit": _to_np(sig_fit) if sig_fit is not None else None,
                "posterior": res["posterior"],
                "cov": _to_np(res["cov"]) if res.get("cov") is not None else None,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 7 — Texture / pole figure (per-ring azimuthal extraction)
# ═════════════════════════════════════════════════════════════════════════════

class PoleFigureWorker(QtCore.QThread):
    """Integrate one frame to an (η, R) cake, then map a selected ring to a
    stereographic pole figure and extract its azimuthal intensity I(η)."""
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, result, image, dark, mask, cfg, parent=None):
        super().__init__(parent)
        self._result, self._image, self._dark, self._mask = result, image, dark, mask
        self._cfg = cfg

    def run(self):
        try:
            import torch
            import midas_integrate_v2 as m
            c = self._cfg
            spec = _build_spec(self._result, c.get("r_bin", 2.0), c.get("eta_bin", 2.0))
            img = self._image.astype(np.float64)
            if self._dark is not None:
                img = np.clip(img - self._dark.astype(np.float64), 0, None)
            mask_t = self._mask.astype(np.float32) if self._mask is not None else None
            geom = build_geom(spec, "subpixel2", mask_t)
            cake = m.integrate_subpixel(torch.from_numpy(img), geom, normalize=True)
            cake = np.nan_to_num(cake.detach().cpu().numpy(), nan=0.0)
            n_eta = cake.shape[0]
            eta_axis = spec.EtaMin + spec.EtaBinSize * (np.arange(n_eta) + 0.5)
            r_axis = compute_r_axis(spec)

            ring = float(c["ring_px"]); cap = float(c.get("capture_px", 4.0))
            alpha, beta, inten = m.texture.cake_to_pole_figure(
                cake, eta_axis, r_axis, hkl_R_px=ring, capture_radius_px=cap,
                sample_rotation_chi_deg=c.get("chi", 0.0),
                sample_rotation_phi_deg=c.get("phi", 0.0))

            # I(η) at the ring: mean over R bins within the capture window
            sel = np.abs(r_axis - ring) <= cap
            i_eta = cake[:, sel].mean(axis=1) if sel.any() else cake.mean(axis=1)

            self.finished.emit({
                "alpha": np.asarray(alpha), "beta": np.asarray(beta),
                "intensity": np.asarray(inten), "eta_axis": eta_axis, "i_eta": i_eta,
                "ring_px": ring,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Learnable gain worker (Tab 5 — per-pixel spatial gain drift recovery)
# ═════════════════════════════════════════════════════════════════════════════

class LearnableGainWorker(QtCore.QThread):
    """Train a per-pixel LearnableGain module against a reference profile.

    Workflow (notebook 16):
      1. Integrate the reference (clean) frame → target cake.
      2. Each step: divide the drifted frame by the current gain estimate,
         integrate, measure MSE vs target, add priors, back-propagate.
      3. After convergence, extract the gain map and stats.

    The gain model is ``g_i = 1 + scale · r_i``.  With ``scale=0.1``,
    a raw parameter of ±1 corresponds to ±10 % gain drift — generous for
    typical detector behaviour.  ``gain_unity_prior`` anchors the mean
    close to 1 (removes the global-scale gauge ambiguity); ``gain_smoothness_prior``
    penalises high spatial frequencies (gain drift is physically smooth).
    """
    progress = QtCore.pyqtSignal(int, int, float)   # step, total, loss
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)             # dict with gain_map, stats
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, result, ref_image, drifted_image, mask, cfg, parent=None):
        super().__init__(parent)
        self._result   = result
        self._ref      = ref_image        # clean / reference frame (train target)
        self._drifted  = drifted_image    # frame with suspected gain drift
        self._mask     = mask
        self._cfg      = cfg

    def run(self):
        try:
            import copy, torch
            import midas_integrate_v2 as m

            c = self._cfg
            spec = _build_spec(self._result, c.get("r_bin", 1.0), c.get("eta_bin", 5.0))
            NZ, NY = spec.NrPixelsZ, spec.NrPixelsY

            ref  = self._ref.astype(np.float64)
            drif = self._drifted.astype(np.float64)
            if self._mask is not None:
                ref  = ref.copy();  ref[self._mask.astype(bool)]  = 0.0
                drif = drif.copy(); drif[self._mask.astype(bool)] = 0.0

            ref_t  = torch.from_numpy(ref)
            drif_t = torch.from_numpy(drif)

            # Integrate the reference frame once → training target
            self.log_line.emit("[gain] integrating reference frame → target…")
            target = m.integrate_with_corrections(ref_t, spec).detach()

            # Initialise learnable gain (centred on 1, scale = 10 %/unit)
            gain = m.LearnableGain(
                NrPixelsZ=int(NZ), NrPixelsY=int(NY),
                scale=float(c.get("gain_scale", 0.1)))
            unity_w    = float(c.get("unity_weight", 1e-4))
            smooth_w   = float(c.get("smoothness_weight", 1e-3))
            lr         = float(c.get("lr", 0.02))
            n_steps    = int(c.get("n_steps", 100))

            opt = torch.optim.Adam(gain.parameters(), lr=lr)
            self.log_line.emit(
                f"[gain] training {n_steps} steps  lr={lr}  "
                f"unity_w={unity_w}  smooth_w={smooth_w}")

            for step in range(n_steps):
                opt.zero_grad()
                g = gain().clamp(min=1e-6)          # current per-pixel gain map
                adjusted = drif_t / g               # remove the drift
                out = m.integrate_with_corrections(adjusted, spec)
                data_loss = (out - target).pow(2).mean()
                loss = (data_loss
                        + unity_w  * m.gain_unity_prior(gain)
                        + smooth_w * m.gain_smoothness_prior(gain))
                loss.backward()
                opt.step()

                loss_f = float(loss.detach())
                self.progress.emit(step + 1, n_steps, loss_f)
                if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
                    self.log_line.emit(
                        f"[gain] step {step+1:4d}/{n_steps}  "
                        f"loss={loss_f:.5g}  data={float(data_loss):.5g}")

            gain_map = gain.extract_gain_map()
            n_drifted = int(gain.n_drifted_pixels(threshold=float(c.get("drift_threshold", 0.01))))
            self.log_line.emit(
                f"[gain] done — gain range [{gain_map.min():.4f}, {gain_map.max():.4f}]  "
                f"drifted>{c.get('drift_threshold', 0.01)*100:.0f}%: {n_drifted:,} px")
            self.finished.emit({
                "gain_map": gain_map,
                "n_drifted": n_drifted,
                "gain_min": float(gain_map.min()),
                "gain_max": float(gain_map.max()),
                "gain_mean": float(gain_map.mean()),
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  Drift trajectory worker (Tab 3 — long-scan geometry drift correction)
# ═════════════════════════════════════════════════════════════════════════════

def _spec_from_trajectory(base_spec, traj, frame_abs_idx: int):
    """Return a deepcopy of base_spec with Lsd/BC_y/BC_z from the drift trajectory.

    Linear interpolation is used so frame indices between knots get smooth values.
    """
    import copy, torch
    Lsd_v = float(np.interp(frame_abs_idx, traj.frame_indices, traj.Lsd_t))
    BCy_v = float(np.interp(frame_abs_idx, traj.frame_indices, traj.BC_y_t))
    BCz_v = float(np.interp(frame_abs_idx, traj.frame_indices, traj.BC_z_t))
    s = copy.deepcopy(base_spec)
    with torch.no_grad():
        s.Lsd.copy_(torch.tensor(Lsd_v, dtype=s.Lsd.dtype))
        s.BC_y.copy_(torch.tensor(BCy_v, dtype=s.BC_y.dtype))
        s.BC_z.copy_(torch.tensor(BCz_v, dtype=s.BC_z.dtype))
    return s


class DriftWorker(QtCore.QThread):
    """Fit a per-frame geometry drift trajectory from calibrant anchor frames.

    The function ``fit_drift_trajectory`` (midas_integrate_v2.pipelines.drift)
    fits Lsd(t), BC_y(t), BC_z(t) as B-splines via L-BFGS with an optional
    Laplace-approx σ estimate.  The result is a ``DriftTrajectory`` dataclass.

    Inputs
    ------
    anchor_frames : dict  {frame_idx: {"Lsd": float, "BC_y": float, "BC_z": float}}
        Known-good geometry values at calibrant exposure indices.
    sample_indices : list of int
        Frame indices for sample data (geometry will be interpolated here).
    base_result : AutoCalibrationResult
        Used to build the base IntegrationSpec.
    cfg : dict
        parametrization ('spline'|'linear'|'constant'), n_knots, bayesian_sigma.
    """
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)    # DriftTrajectory
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, result, anchor_frames, sample_indices, cfg, parent=None):
        super().__init__(parent)
        self._result   = result
        self._anchors  = anchor_frames   # {int: {"Lsd":..., "BC_y":..., "BC_z":...}}
        self._samples  = sample_indices  # list of int
        self._cfg      = cfg

    def run(self):
        try:
            from midas_integrate_v2.pipelines.drift import fit_drift_trajectory
            c = self._cfg
            spec = _build_spec(self._result, 2.0, 5.0)   # geometry only; bins don't matter
            self.log_line.emit(
                f"[drift] fitting trajectory — {len(self._anchors)} anchors, "
                f"{len(self._samples)} sample frames  "
                f"param={c.get('parametrization','spline')}  "
                f"knots={c.get('n_knots', 5)}")
            traj = fit_drift_trajectory(
                self._anchors,
                self._samples,
                spec,
                parametrization=c.get("parametrization", "spline"),
                n_knots=int(c.get("n_knots", 5)),
                bayesian_sigma=bool(c.get("bayesian_sigma", True)),
            )
            self.log_line.emit(
                f"[drift] done — Lsd [{traj.Lsd_t.min():.1f}, {traj.Lsd_t.max():.1f}] µm  "
                f"BC_y [{traj.BC_y_t.min():.3f}, {traj.BC_y_t.max():.3f}]  "
                f"BC_z [{traj.BC_z_t.min():.3f}, {traj.BC_z_t.max():.3f}]")
            self.finished.emit(traj)
        except Exception:
            self.failed.emit(traceback.format_exc())
