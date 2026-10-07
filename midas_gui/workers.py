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
from midas_gui import h5_metadata
from midas_gui import ion_csv
from midas_gui import provenance
from midas_gui import settings
from midas_gui.helpers import (_LogStream, _load_image, _apply_im_trans, _build_spec,
                               _spec_from_json, average_field, apply_field_corrections,
                               read_hdf5_stack_chunk, load_profile_file,
                               CORRECTION_SUFFIX, CORRECTION_EXT)


# ═════════════════════════════════════════════════════════════════════════════
#  Shared integration core (used by both single-frame and batch workers)
# ═════════════════════════════════════════════════════════════════════════════

def apply_q_uniform(spec, q_cfg: Optional[dict]):
    """Activate the backend's NATIVE Q-uniform binning on a spec.

    Deliberately unused. ``IntegrationSpec`` really does carry
    QMin/QMax/QBinSize and a ``q_mode_active`` property that redefines
    ``n_r_bins``, but this GUI does not take that route: every Q (and 2θ)
    run integrates R-uniform and rebins the finished 1-D profile — see
    :func:`rebin_grid_and_r` / :func:`rebin_R_to_grid`.

    Kept because the native mode exists and may be worth revisiting, but do
    not wire it up casually: the two routes disagree about what the axis
    travelling with a profile *is*. The rebin route carries R in pixels
    (the native mode would carry Q), and every downstream consumer —
    writers, waterfall, stack view, the GSAS-II export guard — assumes
    pixels. Getting that wrong is what produced the 2θ = 180° plots fixed
    in 4cb7aaa."""
    if q_cfg:
        unit, lo, hi, step = rebin_cfg_parts(q_cfg)
        if unit != "Q":
            raise ValueError(
                f"native spec binning exists for Q only, not {unit!r}")
        spec.QMin, spec.QMax, spec.QBinSize = float(lo), float(hi), float(step)
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
# A trailing "_c<NN>" chunk suffix, as minted by _HDF5StackGlobSource when one
# multi-frame HDF5 file is split into several combined chunks.
_FID_CHUNK_RE = re.compile(r'^(?P<rest>.+)_c(?P<chunk>\d+)$')
# The current _HDF5StackGlobSource chunk suffix: ".frame_<start>_<end>", the
# raw 0-based sub-frame range a combined chunk was built from. Split off
# before the numeric parse for the same reason as "_c<NN>" — otherwise the
# trailing "_<end>" is mistaken for the stem's own frame number.
_FID_RANGE_RE = re.compile(r'^(?P<rest>.+)\.frame_(?P<start>\d+)_(?P<end>\d+)$')


def froot_and_frame_num(fid, fallback_idx: int) -> tuple:
    """Parse ``(froot, frame_number, tag)`` out of a frame id, matching
    mpe_wf_saxs_waxs's own ``<froot>_<NNNNNN>`` output-naming convention when
    ``fid`` already follows it — the common case, since detector files are
    already named this way. Handles the frame number sitting mid-stem, not
    just at the end, since ``Path.stem`` only strips the *last* suffix —
    ``C611_017Fe_1_load3_009243.vrx.h5`` (this app's own test data) stems to
    ``..._009243.vrx``, with a non-numeric detector tag (``.vrx``) trailing
    the digits. ``tag`` preserves that (empty string when there isn't one).

    A trailing chunk suffix minted by ``_HDF5StackGlobSource`` for the k-th
    combined chunk of one multi-frame file — currently
    ``.frame_<start>_<end>``, historically ``_c<NN>`` — is split off *before*
    the numeric parse and re-attached to ``tag``, so
    ``run_009243.vrx.frame_0_9`` keeps its stem frame number 9243 and stays
    distinct from its ``.frame_10_19`` sibling. When the stem carries no
    number of its own the chunk's own start index becomes the frame number and
    the suffix is dropped (``run.frame_0_9`` → ``('run', 0, '')``,
    ``run.frame_10_19`` → ``('run', 10, '')``) — still unique and correctly
    ordered across chunks, and matching mpe_wf's plain ``<froot>_<NNNNNN>``,
    rather than falling back to an unrelated loop counter.

    Falls back to ``(fid, fallback_idx, "")`` when ``fid`` has no
    underscore-digit run at all (e.g. a caller-supplied non-numeric id).

    NOTE: this is deliberately *not* injective — ``scan_1`` and ``scan_001``
    both parse to frame 1 — so callers that turn the result into a filename
    must go through :func:`frame_output_base`, which de-duplicates.
    """
    s = str(fid)
    chunk_suffix, chunk_num = '', None
    m = _FID_RANGE_RE.match(s)
    if m:
        s = m.group('rest')
        chunk_suffix = '.frame_%s_%s' % (m.group('start'), m.group('end'))
        chunk_num = int(m.group('start'))
    else:
        m = _FID_CHUNK_RE.match(s)
        if m:
            s = m.group('rest')
            chunk_suffix = '_c' + m.group('chunk')
            chunk_num = int(m.group('chunk'))
    m = _FID_NUM_RE.match(s)
    if m:
        return m.group('froot'), int(m.group('num')), m.group('tag') + chunk_suffix
    if chunk_num is not None:
        # Stem has no number of its own — the chunk index *is* the frame index.
        return s, chunk_num, ''
    return s, int(fallback_idx), ''


def frame_output_stem(fid, fallback_idx: int, used: set) -> str:
    """Return the filename stem to write frame ``fid``'s output under, in the
    ``<froot>_<NNNNNN><tag>`` convention, guaranteed unique within one run.

    ``used`` is a caller-owned set of already-issued names, mutated in place.
    Because :func:`froot_and_frame_num` normalises zero-padding, distinct
    frame ids can collapse onto one name — ``scan_1``, ``scan_01`` and
    ``scan_001`` all yield ``scan_000001`` — which would silently overwrite
    earlier frames. On a clash we fall back to the raw ``fid`` (which is
    unique by construction: it is a file stem, or a stem + chunk suffix), and
    only if *that* is somehow taken too do we append the loop index.

    Callers that write one frame into several per-format subfolders must
    allocate the stem ONCE and reuse it, rather than calling this per format
    — otherwise the second format sees its own stem as already used and
    needlessly falls back.
    """
    froot, frame_num, tag = froot_and_frame_num(fid, fallback_idx)
    name = f"{froot}_{frame_num:06d}{tag}"
    if name in used:
        name = str(fid)
        if name in used:
            name = f"{fid}_{int(fallback_idx):06d}"
    used.add(name)
    return name


def frame_output_base(out_dir, fid, fallback_idx: int, used: set):
    """:func:`frame_output_stem` joined onto ``out_dir`` — for callers that
    write every format of a frame into one flat directory."""
    return Path(out_dir) / frame_output_stem(fid, fallback_idx, used)


def stamp_h5_provenance(h5_path, entry: dict) -> None:
    """Reopen ``h5_path`` (already written by ``midas_integrate_v2.write_h5``)
    and append a provenance entry to its root attrs (``provenance_history``,
    the cross-tool JSON-in-attrs convention — see ``provenance.py``), AND
    mirror the resulting history into a root-level ``provenance_history``
    *dataset*, so it shows up in a plain ``h5ls``/tree view without needing
    to inspect attributes — the attrs form is easy to miss (confirmed: it
    was already there, just invisible to a quick look). Best-effort: a
    failure here shouldn't take down an otherwise-successful batch run."""
    import h5py
    with h5py.File(str(h5_path), 'a') as h5:
        provenance.append_to_hdf5_attrs(h5, entry)
        history_json = h5.attrs['provenance_history']
        if 'provenance_history' in h5:
            del h5['provenance_history']
        ds = h5.create_dataset('provenance_history', data=history_json)
        ds.attrs['format'] = 'JSON list of provenance entries (see midas_gui/provenance.py)'


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


#: Output radial units this app can rebin an R-uniform profile onto.
REBIN_UNITS = ("Q", "2th")

#: Display names for those units, for log lines and user-facing messages.
#: ``None`` (no rebin configured) reads as "off" so callers can index this
#: unconditionally.
_REBIN_LABEL = {None: "off", "Q": "Q", "2th": "2θ"}


def rebin_cfg_parts(cfg) -> tuple:
    """``(unit, lo, hi, step)`` from an output-rebin config.

    The canonical form is self-describing::

        {"unit": "Q" | "2th", "min": float, "max": float, "step": float}

    with ``min``/``max``/``step`` in that unit's own terms (Å⁻¹ for Q,
    degrees for 2θ). The legacy Q-only spelling — ``QMin``/``QMax``/
    ``QBinSize`` and no ``unit`` — is still accepted and read as Q, so
    projects saved before 2θ existed keep loading.

    This is the only place that knows the dict's shape; everything else goes
    through here.

    (The transport keyword is still named ``q_cfg`` throughout the workers
    and in saved projects' ``inputs``. It predates 2θ and renaming it would
    reach the on-disk project schema for naming alone — the dict itself says
    which unit it is.)"""
    if not cfg:
        return None, None, None, None
    unit = cfg.get("unit", "Q")
    if "min" in cfg:
        lo, hi, step = cfg["min"], cfg["max"], cfg["step"]
    else:                                   # legacy Q-only spelling
        lo, hi, step = cfg["QMin"], cfg["QMax"], cfg["QBinSize"]
    if unit not in REBIN_UNITS:
        raise ValueError(f"unknown output rebin unit {unit!r}; "
                         f"expected one of {REBIN_UNITS}")
    return unit, float(lo), float(hi), float(step)


def _r_px_of(values, unit, lsd, px, wl) -> np.ndarray:
    """R in detector pixels for an array of radial positions in ``unit``."""
    values = np.asarray(values, dtype=float)
    if unit == "Q":
        two_theta = 2.0 * np.arcsin(np.clip(values * wl / (4 * math.pi), -1, 1))
    else:                                   # "2th", already an angle
        two_theta = np.radians(values)
    return lsd * np.tan(two_theta) / px


def _unit_of_r_px(r_px, unit, lsd, px, wl) -> np.ndarray:
    """The inverse of :func:`_r_px_of`: pixels → ``unit``."""
    two_theta = np.arctan(np.asarray(r_px, dtype=float) * px / lsd)   # radians
    if unit == "Q":
        return 4 * math.pi * np.sin(two_theta / 2) / wl
    return np.degrees(two_theta)


def rebin_grid_and_r(cfg, lsd, px, wl):
    """Output bin centres + the matching R(px) for each — ``(grid, r_of_grid)``.

    ``grid`` is uniform in the config's unit (Q in Å⁻¹, or 2θ in degrees);
    ``r_of_grid`` is where each of those bins sits on the detector, in
    pixels. The second value is the axis that travels onward with the
    rebinned profile — writers, the waterfall and the stack view all expect
    pixels, so this is R, *not* the grid. See :func:`apply_q_uniform`.
    """
    unit, lo, hi, step = rebin_cfg_parts(cfg)
    n = max(1, int(round((hi - lo) / step)))
    grid = lo + step * (np.arange(n) + 0.5)
    return grid, _r_px_of(grid, unit, lsd, px, wl)


def rebin_R_to_grid(r_ax, prof, sigma, grid, lsd, px, wl, unit="Q"):
    """Rebin an R-uniform profile/σ onto a grid uniform in ``unit``.

    The integration kernels only bin uniformly in R (there is no 2θ mode at
    all, and the native Q mode is unused — see :func:`apply_q_uniform`), so
    uniform-Q and uniform-2θ output are produced here: integrate R-uniform,
    convert this run's R axis into the target unit, and interpolate onto the
    requested grid so rings land at the correct Q / 2θ.
    """
    x_of_r = _unit_of_r_px(r_ax, unit, lsd, px, wl)
    order = np.argsort(x_of_r)
    prof_x = np.interp(grid, x_of_r[order], prof[order])
    sig_x = np.interp(grid, x_of_r[order], sigma[order]) if sigma is not None else None
    return prof_x, sig_x


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
    elif fmt == "2d_csv":
        if cake_2d is None:
            # Was a silent no-op, and cost a real run its 2D CSVs: the caller
            # gated the cake on multi-azimuth mode, so with that off nothing
            # was written while the path was still reported as written. The
            # cake is always available here (see BatchWorker's want_cake,
            # which includes "2d_csv"), so reaching this is a bug in the
            # caller and says so rather than producing an empty output dir.
            raise ValueError("2d_csv output needs the (eta, R) cake; "
                             "none was passed to write_profile")
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
                         cake_2d=None, cake_sigma=None, eta_axis=None,
                         per_eta=None) -> list:
    """Write one frame's 1-D output file(s) in every format in ``file_fmts``
    (``"h5"`` excluded — callers handle that separately).

    ``per_eta`` (multi-azimuth/"cake" mode — see the "Multi-azimuth output
    (cake)" checkbox in Batch Integrate) writes one file *per azimuthal (η)
    bin*, named ``<base>_etaNNN.<fmt>``, each a genuine 1-D lineout for that
    sector; ``prof``/``sigma`` (the η-collapsed full-circle profile) are not
    written in that mode. Otherwise ``prof``/``sigma`` are written once per
    format, as always.

    ``"2d_csv"`` is the one whole-cake file, ``<base>_cake.csv``, and is
    written in BOTH modes whenever ``cake_2d`` is supplied — it is a picture
    of the cake, not a lineout, so per-η fan-out does not apply to it.

    ``per_eta`` defaults to ``cake_2d is not None and cake_sigma is not
    None``, which is what the presence of a cake used to mean on its own.
    That overload is why 2D CSV silently produced nothing: a caller with a
    cake but multi-azimuth off had no way to say "write the cake file, don't
    fan out", so it passed no cake at all and ``2d_csv`` fell through to
    nothing while still being counted as written. Pass ``per_eta``
    explicitly and the two decisions stay separate. Returns the list of paths
    actually written.
    """
    paths = []
    have_cake = cake_2d is not None
    if per_eta is None:
        per_eta = have_cake and cake_sigma is not None
    if per_eta and have_cake and cake_sigma is not None:
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
            if f == "2d_csv":
                # Skipped, not silently mis-reported, when there is genuinely
                # no cake to write (write_all_profiles' in-memory Save path
                # for a 1-D run — it excludes the format itself and logs).
                if not have_cake:
                    continue
                # eta_axis may be None; write_profile falls back to bin
                # indices for it, the same as in the fan-out branch.
                write_profile(Path(base), "2d_csv", r_px, prof, sigma,
                              lsd, px, wl, cake_2d=cake_2d, eta_axis=eta_axis)
                paths.append(str(base) + "_cake.csv")
                continue
            write_profile(Path(base), f, r_px, prof, sigma, lsd, px, wl)
            paths.append(str(base) + "." + f)
    return paths


# ═════════════════════════════════════════════════════════════════════════════
#  Dark / bright / background field mean
# ═════════════════════════════════════════════════════════════════════════════

class FieldAverageWorker(QtCore.QThread):
    """Reduce a dark/bright/background field to its mean off the GUI thread.

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


class StreamPreviewWorker(QtCore.QThread):
    """Fetch and dark/bright/background-correct the "stream"-mode preview sum
    off the GUI thread — the file-reading half of what
    ``widgets.DataLoaderPanel._peek_stream_frame`` used to do entirely on the
    main thread.

    Opens the exact same source the real batch run will use
    (``_open_source_cfg``), reads ``min(preview_sum_n, n_frames)`` frames,
    correcting each one BEFORE summing (matching the real run's per-frame
    correction — correcting only the final sum would subtract just one
    dark frame's worth from an N-times-larger signal) via the same
    ``apply_field_corrections`` used everywhere else, not
    ``DataLoaderPanel.corrected()`` itself — that method also updates each
    field selector's mismatch-warning label, which is a QWidget mutation and
    must stay on the GUI thread; the caller re-does that one check itself,
    once, against this result's shape, in its ``finished`` slot.

    Confirmed necessary against a real hang, not just theoretical: a
    multi-file VAREX HDF5 source over an NFS-mounted beamline share froze
    the whole app with no recovery when this ran synchronously (see
    .context/DECISIONS.md, 2026-09-25) — HDF5's file locking can hang
    indefinitely on such mounts, not just run slowly, and even with that
    hang separately fixed (HDF5_USE_FILE_LOCKING=FALSE — see midas_gui/
    _paths.py), a large multi-frame combine over real network storage can
    still take long enough that it belongs off the GUI thread regardless.
    """
    finished = QtCore.pyqtSignal(object)   # np.ndarray (float32) or None
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, cfg: dict, preview_sum_n: int,
                dark=None, bright=None, background=None,
                bright_mode: str = "divide", parent=None):
        super().__init__(parent)
        self._cfg = dict(cfg)
        self._preview_sum_n = max(1, int(preview_sum_n))
        self._dark, self._bright, self._background = dark, bright, background
        self._bright_mode = bright_mode

    def run(self):
        try:
            if not (self._cfg.get("path") or self._cfg.get("paths")):
                self.finished.emit(None)
                return
            source = _open_source_cfg(self._cfg)
            total = getattr(source, "n_frames", 0)
            if total == 0:
                self.finished.emit(None)
                return
            n = min(self._preview_sum_n, total)
            acc = None
            for i in range(n):
                _fid, img = source.get(i)
                img = np.asarray(img, dtype=np.float64)
                if self._dark is not None or self._bright is not None or self._background is not None:
                    img = apply_field_corrections(
                        img, dark=self._dark, bright=self._bright,
                        bright_mode=self._bright_mode, background=self._background)
                acc = img if acc is None else acc + img
            self.finished.emit(acc.astype(np.float32))
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
    """Load a frame stack and reduce it (max/sum/mean) off the GUI thread.

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
            # "average" kept as an alias so an older caller's method string
            # still resolves; the UI only ever sends "mean".
            fn = {"max": np.max, "sum": np.sum,
                  "mean": np.mean, "average": np.mean}[self._method]
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


class BatchCorrectionWorker(QtCore.QThread):
    """Reduce every selected HDF5 file's sub-frame stack and write the result
    back out as HDF5 — the engine of the Batch Correction tab.

    One output file per input file PER OP, holding that file's reduced
    chunks as a single ``(M, H, W)`` float32 dataset, in a
    ``dark_subtracted_<op>`` subfolder of the chosen output directory.
    Chunks never cross a file boundary, which is what makes the per-file
    output well defined in the first place.

    Several ops cost one pass, not several: reading a sub-frame off disk and
    correcting it is the same work whichever op consumes it, so mean, median,
    sum and max are all computed from one read (see
    ``frame_correct.reduce_chunk_multi``).

    Everything numeric lives in ``midas_gui.frame_correct``; this class is
    the Qt shell around it — threading, progress, cancellation, and the
    per-file bookkeeping (dark resolution, metadata alignment) that has to
    happen once per file rather than once per chunk.

    Memory is bounded by ONE chunk plus one output stack, never a whole input
    file: each chunk reads only its own raw sub-frames through an h5py slice.
    That is the same lesson ``_HDF5StackGlobSource`` records the hard way —
    decoding a whole 1442-sub-frame VAREX file to produce one frame cost
    23.9 GB and ~230 s over NFS.

    All GUI-derived inputs (the loader's dark/bright/background arrays, the
    output settings) are captured by the caller before construction, so
    ``run()`` touches no Qt widget.
    """
    progress = QtCore.pyqtSignal(int, int, str)   # done, total, message
    fileDone = QtCore.pyqtSignal(str)             # output path just written
    finished = QtCore.pyqtSignal(list)            # every output path
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, paths, dataset: str, *, chunk_size=None, op="mean",
                 out_dir: str, suffix: str = CORRECTION_SUFFIX,
                 out_ext: str = CORRECTION_EXT,
                 out_dataset: str = "exchange/data",
                 dark=None, bright=None, background=None,
                 bright_mode: str = "divide", auto_dark: bool = True,
                 dark_dataset: str = "exchange/data_dark",
                 clip_negatives: bool = True, compression=None,
                 level: int = 4, shuffle: bool = False,
                 out_dtype: str = "float32",
                 ion_csv_extras=(),
                 raw_start=None, raw_end=None, write_monitor_csvs: bool = True,
                 parent=None):
        super().__init__(parent)
        self._paths = [Path(p) for p in paths]
        self._dataset, self._out_dataset = dataset, out_dataset
        self._chunk_size = chunk_size
        # Accept a bare string or a list, so a single-op caller reads the
        # same as it always did.
        self._ops = [op] if isinstance(op, str) else list(op)
        self._out_dir, self._suffix = Path(out_dir), suffix
        self._out_ext = out_ext if out_ext.startswith(".") else "." + out_ext
        self._dark, self._bright, self._background = dark, bright, background
        self._bright_mode = bright_mode
        self._auto_dark, self._dark_dataset = auto_dark, dark_dataset
        self._clip = clip_negatives
        self._compression, self._level, self._shuffle = compression, level, shuffle
        # On-disk dtype for the corrected stack. Applied by
        # frame_correct.cast_for_output immediately before create_dataset,
        # never upstream — see its docstring for why unsigned needs care.
        # Imported here rather than at module scope, like every other
        # frame_correct use in this file.
        from midas_gui.frame_correct import OUTPUT_DTYPES
        self._out_dtype = out_dtype if out_dtype in OUTPUT_DTYPES else "float32"
        # Per-frame beam-monitor sidecar. Always written — it is small, and
        # the value it carries is the one SAXS normalises against. Only which
        # OPTIONAL columns it carries is a choice; write_ion_csv already
        # skips the file entirely when there is no real data behind it.
        self._ion_csv_extras = set(ion_csv_extras or ())
        self._raw_start, self._raw_end = raw_start, raw_end
        # False only when a BatchCorrectionCoordinator is driving several of
        # these over disjoint slices of ONE file list. A monitor CSV is per
        # froot -- one scan, many numbered files -- so two workers holding
        # different files of the same scan would each write the same
        # <froot>_<detector>_bc.csv with half the rows in it. The rows are
        # left on self._monitor_rows instead and the coordinator merges them
        # and writes once. See DECISIONS 2026-10-07.
        self._write_monitor_csvs_on_finish = bool(write_monitor_csvs)
        #: froot -> {"rows": [...], "notes": [...]}, readable after finished.
        self._monitor_rows: dict = {}
        self._cancel = False

    def _collect_monitor_rows(self, acc: dict, path, tree, ranges, *,
                              n_light: int, n_dark: int) -> None:
        """Add one file's per-frame monitor rows to its froot's bucket.

        Grouped by froot rather than per file because a froot is the unit a
        measurement is actually thought about in — one scan, many numbered
        files — and one CSV per file left the folder with as many sidecars
        as exposures.
        """
        from midas_gui.frame_correct import split_scan_name
        hutch = ion_csv.resolve_hutch(path, settings.active_profile())
        # The acquisition timestamps live outside the instrument/ tree that
        # gets copied into the output, so read them separately — they are
        # the cross-check on which block of scaler entries is the lights.
        csv_tree = dict(tree)
        try:
            csv_tree.update(h5_metadata.read_tree(path, groups=("NDArray",)))
        except Exception:
            pass
        rows, note = ion_csv.rows_from_tree(
            csv_tree, hutch, frame_ranges=ranges, n_light=n_light,
            n_dark=n_dark, source=Path(path).name)
        froot = split_scan_name(Path(path).name).froot or Path(path).stem
        bucket = acc.setdefault(froot, {"rows": [], "notes": []})
        bucket["rows"].extend(rows)
        if note:
            bucket["notes"].append(f"{Path(path).name}: {note}")

    def _write_monitor_csvs(self, acc: dict, done: int, total: int) -> None:
        """One ``<froot>_<detector>_bc.csv`` per froot.

        Written one level ABOVE the output directory. The output folder is
        per-detector (``…/<froot>/<detector>/``) while the monitor readings
        are a property of the exposure, shared by every detector that saw
        it — so the froot level is where one file serves them all.
        """
        if not acc or self._out_dir is None:
            return
        det = self._out_dir.name
        dest_dir = self._out_dir.parent or self._out_dir
        for froot, bucket in sorted(acc.items()):
            name = "_".join(p for p in (froot, det) if p) + ion_csv.SUFFIX_BATCH_CORRECTION
            try:
                written = ion_csv.write_ion_csv(
                    dest_dir / name, bucket["rows"],
                    extras=self._ion_csv_extras)
            except Exception:
                self.progress.emit(done, total,
                                   f"{name}: monitor CSV failed — "
                                   + traceback.format_exc(limit=1).strip())
                continue
            if written:
                for note in bucket["notes"]:
                    self.progress.emit(done, total, f"  light/dark — {note}")
                self.progress.emit(done, total, f"beam monitors: {written}")
            else:
                self.progress.emit(
                    done, total,
                    f"{froot}: no beam-monitor data to write "
                    "(unrecognised hutch, or no live channel)")

    def cancel(self):
        """Ask the run to stop. Checked between chunks, so the file being
        written finishes its current chunk rather than leaving a torn
        dataset — the partial file is simply never written."""
        self._cancel = True

    def _out_path(self, src: Path, op: str) -> Path:
        """``<out_dir>/dark_subtracted_<op>/<source stem><suffix><ext>`` —
        ``dark_subtracted_mean/AgBeH_10s_000021_cor.h5`` by default.

        The op is in the FOLDER rather than the filename so a mean and a max
        of the same scan can't land on top of each other, while each file's
        own name still matches its source. Which op produced a given file is
        also recorded inside it, as the ``midas_gui_combine_op`` attribute.

        The stem keeps EVERYTHING before the final extension — the detector
        tag included::

            AgBeH_10s_000021.h5        ->  AgBeH_10s_000021_cor.h5
            CeO2_030319.vrx.h5         ->  CeO2_030319.vrx_cor.h5
            testing100_000009.eiger2.h5 -> testing100_000009.eiger2_cor.h5

        An earlier rule split the whole dotted tail off, on the reasoning
        that a detector tag no longer describes a derived file. That was
        wrong in practice: the tag is how one froot's VAREX and Eiger
        reductions stay apart, and a beamline reading the output folder
        expects the source name back. Not every source carries a tag
        (``AgBeH_10s_000021.h5`` has none), which is exactly why the rule
        has to preserve whatever is there rather than assume a shape.
        """
        from midas_gui.helpers import correction_subdir
        base = src.stem
        return (self._out_dir / correction_subdir(op)
                / f"{base}{self._suffix}{self._out_ext}")

    def _plan(self) -> list:
        """``[(path, [(lo, hi), …])]`` — every file's chunk ranges, from
        dataset SHAPE alone (one h5py header read each, no pixels). Done up
        front so the progress bar has a real total instead of counting up to
        an unknown end."""
        import h5py
        from midas_gui.frame_correct import chunk_ranges
        plan = []
        for path in self._paths:
            with h5py.File(str(path), "r") as f:
                dset = f[self._dataset]
                if dset.ndim == 2:
                    plan.append((path, [(0, 0)]))
                    continue
                plan.append((path, chunk_ranges(
                    int(dset.shape[0]), chunk_size=self._chunk_size,
                    raw_start=self._raw_start, raw_end=self._raw_end)))
        return plan

    def run(self):
        try:
            import h5py
            from midas_gui.frame_correct import (reduce_chunk_multi, resolve_dark,
                                                 write_corrected_h5)
            plan = self._plan()
            total = sum(len(ranges) for _p, ranges in plan)
            if total == 0:
                self.failed.emit(
                    "Nothing to do: the selected frame range leaves no sub-frames.")
                return
            done = 0
            outputs = []
            #: froot -> {"rows": [...], "det": str, "dir": Path, "notes": []}
            metadata_rows = self._monitor_rows
            for path, ranges in plan:
                if self._cancel:
                    break
                tree = h5_metadata.read_tree(path)
                frames = {op: [] for op in self._ops}
                with h5py.File(str(path), "r") as f:
                    dset = f[self._dataset]
                    n_raw = int(dset.shape[0]) if dset.ndim == 3 else 1
                    # Dark acquisitions share the metadata axis with the
                    # lights (one sample per detector acquisition), so their
                    # count is what says the per-acquisition arrays are twice
                    # as long as the image stack. See ion_csv.split_light_dark.
                    _dk = f.get(self._dark_dataset)
                    n_dark_frames = (int(_dk.shape[0])
                                     if _dk is not None and _dk.ndim == 3 else 0)
                    shape = tuple(dset.shape[-2:])
                    # One dark per file, resolved once — see
                    # frame_correct.resolve_dark for the ladder and the real
                    # data it was derived from. `why` is logged so the choice
                    # can be audited afterwards instead of trusted.
                    dark, why = resolve_dark(
                        path, dark_dataset=self._dark_dataset, shape=shape,
                        fallback=self._dark, auto=self._auto_dark)
                    self.progress.emit(done, total, f"{path.name}: dark = {why}")
                    n_aligned = _HDF5StackGlobSource._metadata_frame_count(f, n_raw)
                    for lo, hi in ranges:
                        if self._cancel:
                            break
                        raw = (np.asarray(dset[lo:hi + 1], dtype=np.float32)
                               if dset.ndim == 3 else
                               np.asarray(dset[...], dtype=np.float32)[None])
                        combined = reduce_chunk_multi(
                            raw, self._ops, dark=dark, bright=self._bright,
                            bright_mode=self._bright_mode,
                            background=self._background,
                            clip_negatives=self._clip)
                        for op, plane in combined.items():
                            frames[op].append(plane)
                        done += 1
                        self.progress.emit(
                            done, total,
                            f"{path.name}  chunk "
                            f"{len(frames[self._ops[0]])}/{len(ranges)} "
                            f"(raw {lo}–{hi})")
                if self._cancel:
                    break
                aligned = h5_metadata.align(tree, ranges, n_aligned)
                # Accumulate this file's monitor rows under its froot; the
                # CSV is written once per froot after the loop, not per file.
                try:
                    self._collect_monitor_rows(
                        metadata_rows, path, tree, ranges, n_light=n_raw,
                        n_dark=n_dark_frames)
                except Exception:
                    # A sidecar must never cost the user the reduction that
                    # already succeeded.
                    self.progress.emit(
                        done, total,
                        f"{path.name}: monitor metadata failed — "
                        + traceback.format_exc(limit=1).strip())
                for op in self._ops:
                    out = write_corrected_h5(
                        self._out_path(path, op), frames[op],
                        dataset=self._out_dataset, metadata=aligned,
                        frame_ranges=ranges,
                        attrs={"midas_gui_source": str(path),
                               "midas_gui_combine_op": op,
                               "midas_gui_chunk_size": int(self._chunk_size or 0),
                               "midas_gui_dark": why},
                        provenance_entry=provenance.build_entry(
                            "batch_correction", inputs=[str(path)],
                            compute_checksums=False,
                            extra={"op": op,
                                   "chunk_size": int(self._chunk_size or 0),
                                   "dark": why,
                                   "clip_negatives": bool(self._clip)}),
                        compression=self._compression, level=self._level,
                        shuffle=self._shuffle, dtype=self._out_dtype,
                        log=lambda msg: self.progress.emit(done, total, msg))
                    outputs.append(out)
                    self.fileDone.emit(out)
            if self._write_monitor_csvs_on_finish:
                self._write_monitor_csvs(metadata_rows, done, total)
            if self._cancel:
                self.progress.emit(done, total,
                                   f"Cancelled — {len(outputs)} file(s) written.")
            self.finished.emit(outputs)
        except Exception:
            self.failed.emit(traceback.format_exc())


class BatchCorrectionCoordinator(QtCore.QObject):
    """Runs one Batch Correction job either as a single
    ``BatchCorrectionWorker`` ("sequential") or as several concurrent ones,
    each given a disjoint slice of the FILE list ("parallel").

    Exposes the same signal surface as ``BatchCorrectionWorker``
    (``progress``, ``fileDone``, ``finished``, ``failed``) plus
    ``start()``/``isRunning()``/``cancel()``/``wait()``, so
    ``BatchCorrectionTab`` can construct this in place of the worker with no
    change to its wiring -- the same contract ``BatchRunCoordinator`` has
    with ``BatchTab``.

    Files are the unit of work because they share nothing: chunks never
    cross a file boundary, and each input yields its own output per method,
    so no two workers can collide on a path. (Chunk-level splitting within
    one file would complicate the writer for no gain -- these runs are
    hundreds of files.)

    Threads, not OS processes, for the same reason ``BatchRunCoordinator``
    gives: h5py releases the GIL on reads and numpy on the reduction, so
    this gets real multi-core throughput while staying in-process, with no
    pickling and no cross-process progress plumbing.
    """
    progress = QtCore.pyqtSignal(int, int, str)
    fileDone = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(list)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, paths, *, out_dir, n_workers: int = 1, parent=None,
                 **worker_kwargs):
        super().__init__(parent)
        self._paths = [Path(p) for p in paths]
        self._out_dir = Path(out_dir)
        self._kwargs = worker_kwargs
        self._n = max(1, min(int(n_workers), len(self._paths) or 1))
        self._workers: list = []
        self._outputs: list = []
        self._done_counts: dict = {}
        self._totals: dict = {}
        self._failed_once = False
        self._finished_count = 0
        self._started = False

    # ── the same surface BatchCorrectionWorker offers ───────────────────
    def isRunning(self) -> bool:
        return any(w.isRunning() for w in self._workers)

    def cancel(self):
        for w in self._workers:
            w.cancel()

    def wait(self, *a):
        for w in self._workers:
            w.wait(*a)
        return True

    @staticmethod
    def _slices(paths, n) -> list:
        """``n`` contiguous, near-equal, non-empty slices of ``paths``.

        Contiguous rather than round-robin so one scan's numbered files
        mostly stay with one worker -- it does not affect correctness (the
        CSVs are merged either way) but it keeps each worker reading a
        locality-friendly run of files off the share.
        """
        out, lo = [], 0
        for i in range(n):
            hi = lo + len(paths) // n + (1 if i < len(paths) % n else 0)
            if hi > lo:
                out.append(paths[lo:hi])
            lo = hi
        return out

    def start(self):
        if self._started:
            return
        self._started = True
        groups = self._slices(self._paths, self._n)
        if not groups:
            self.finished.emit([])
            return
        for i, group in enumerate(groups):
            # Deliberately UNPARENTED. A QThread that is a Qt child is
            # destroyed by the C++ parent-owns-children cascade the instant
            # its parent is -- including while it is still running, which is
            # a fatal "QThread: Destroyed while thread is still running"
            # abort rather than an exception, and which PyQt's keep-alive
            # for running threads does not save you from. self._workers
            # below is the only reference these need, and it outlives them.
            # See DataLoaderPanel._start_preview_worker for the same note.
            w = BatchCorrectionWorker(
                group, out_dir=str(self._out_dir),
                # Every child collects rows; only this object writes them.
                write_monitor_csvs=False, **self._kwargs)
            w.progress.connect(lambda d, tt, m, k=i: self._on_progress(k, d, tt, m))
            w.fileDone.connect(self.fileDone.emit)
            w.finished.connect(lambda outs, k=i: self._on_child_finished(k, outs))
            w.failed.connect(self._on_child_failed)
            self._workers.append(w)
        for w in self._workers:
            w.start()

    # ── fan-in ──────────────────────────────────────────────────────────
    def _on_progress(self, key, done, total, msg):
        self._done_counts[key] = done
        self._totals[key] = total
        self.progress.emit(sum(self._done_counts.values()),
                           sum(self._totals.values()), msg)

    def _on_child_failed(self, msg):
        """First failure wins and stops the rest. Reported once: N workers
        hitting the same bad share would otherwise raise N dialogs."""
        if not self._failed_once:
            self._failed_once = True
            self.failed.emit(msg)
        self.cancel()

    def _on_child_finished(self, _key, outputs):
        self._outputs.extend(outputs)
        self._finished_count += 1
        if self._finished_count < len(self._workers):
            return
        if not self._failed_once:
            self._write_merged_monitor_csvs()
        self.finished.emit(self._outputs)

    def _write_merged_monitor_csvs(self):
        """One CSV per froot, from every worker's rows together.

        Rows carry source_file/frame_start/frame_end (DECISIONS 2026-10-06),
        so the merge is a concatenate and sort -- which also restores the
        file order the split broke.
        """
        merged: dict = {}
        for w in self._workers:
            for froot, bucket in (w._monitor_rows or {}).items():
                dest = merged.setdefault(froot, {"rows": [], "notes": []})
                dest["rows"].extend(bucket.get("rows") or [])
                dest["notes"].extend(bucket.get("notes") or [])
        for bucket in merged.values():
            bucket["rows"].sort(
                key=lambda r: (str(r.get("source_file") or ""),
                               r.get("kind") != "dark",
                               int(r.get("frame_start") or 0)))
            bucket["notes"].sort()
        if not merged or not self._workers:
            return
        total = sum(self._totals.values())
        # Any child can write them -- _write_monitor_csvs only reads
        # self._out_dir, which every child shares with this coordinator.
        self._workers[0]._write_monitor_csvs(merged, total, total)


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
            # midas_calibrate_v2.calibrate() and first_time_calibrate() paths
            # take im_trans as a native kwarg and flip internally; the other
            # entry points (four_stage/bayesian/joint/partial-distortion) still
            # have no such parameter, so run_pipeline pre-flips for those
            # itself. Either way, this worker never flips the array — it just
            # passes the raw data and codes through.
            raw = calib.run_pipeline(self._mode, image, self._dark, self._cfg)
            # Post-transform counts, not image.shape: every branch solves in the
            # transformed frame, and a transpose swaps Y/Z on a non-square
            # detector (see calib.effective_pixel_counts).
            NY, NZ = calib.effective_pixel_counts(
                image, self._cfg.get("im_trans", ()))
            result = calib.normalize_result(
                raw, self._mode, NY=NY, NZ=NZ,
                pxY=self._cfg["pxY"], pxZ=self._cfg.get("pxZ"),
                wavelength=self._cfg["wavelength"],
                panel_layout=self._cfg.get("panel_layout"),
                scratch=self._cfg.get("scratch_dir"),
                stem=self._cfg.get("save_stem", ""))
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
    """Fit detector geometry from user-picked ring points and known d-spacings
    (Bragg's law), entirely bypassing ``calib.run_pipeline``/``midas_calibrate_v2``
    — used for non-crystalline calibrants (e.g. AgBH) that have no space group.
    ``refine`` selects which of Lsd/BC/tx/ty/tz/Wavelength float (same dict
    shape as ``CalibrationTab._refine_flags()``; distortion is not supported
    here — no per-pixel intensity model). ``bounds`` optionally boxes any of
    them in (see :func:`~midas_gui.helpers.fit_geometry_from_ring_picks`).
    Same signal names as ``CalibrationWorker`` so the tab's existing
    ``_on_done``/``_on_fail``/``_abort`` wiring works unchanged."""
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    #: Per-parameter 1-sigma above which a refined value is reported as not
    #: actually constrained by the picked points. These are "coarser than any
    #: calibration worth keeping" scales, not statistical thresholds: a beam
    #: centre known to worse than 5 px, or an Lsd to worse than 1%, has not
    #: been measured by this fit. (Lsd/wavelength entries are fractions of the
    #: fitted value; the rest are absolute, in the parameter's own unit.)
    _SIGMA_WARN_FRAC = {"Lsd": 0.01, "wavelength_A": 0.01}
    _SIGMA_WARN_ABS  = {"BC_y": 5.0, "BC_z": 5.0, "tx": 0.5, "ty": 0.5, "tz": 0.5}
    #: (display unit, multiplier from fit units). Lsd is fit in µm but shown in
    #: mm, matching the seed card and every other Lsd readout in the GUI.
    _SIGMA_UNITS = {"Lsd": ("mm", 1e-3), "BC_y": ("px", 1.0), "BC_z": ("px", 1.0),
                    "tx": ("°", 1.0), "ty": ("°", 1.0), "tz": ("°", 1.0),
                    "wavelength_A": ("Å", 1.0)}

    def __init__(self, picks, wavelength_A, pxY, pxZ, seed, NY, NZ,
                 material_name, d_list, parent=None,
                 refine=None, tilt_seed=(0.0, 0.0, 0.0), bounds=None):
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
        self._refine = dict(refine) if refine else None
        self._tilt_seed = tuple(tilt_seed)
        self._bounds = dict(bounds) if bounds else None

    def run(self):
        from types import SimpleNamespace
        from midas_gui.helpers import fit_geometry_from_ring_picks
        try:
            refine = self._refine or {}
            free = [name for name, key in
                    (("Lsd", "Lsd"), ("BC", "BC"), ("tx", "tx"), ("ty", "ty"),
                     ("tz", "tz"), ("Wavelength", "Wavelength"))
                    if refine.get(key, key in ("Lsd", "BC"))]
            self.log_line.emit(
                f"[manual fit] fitting {', '.join(free)} from {len(self._picks)} picked "
                f"points across {len(set(p[2] for p in self._picks))} ring(s), "
                f"calibrant='{self._material_name}'…")
            fit = fit_geometry_from_ring_picks(
                self._picks, self._wavelength_A, self._pxY, self._pxZ,
                seed=self._seed, tilt_seed=self._tilt_seed, refine=self._refine,
                bounds=self._bounds)
            self.log_line.emit(
                f"[manual fit] seed={fit['seed_quality']}  success={fit['success']}  "
                f"solver={fit['method']}  residual RMS={fit['residual_deg_rms']:.4f}°  "
                f"({fit['message']})")
            if fit["clamped"]:
                self.log_line.emit(
                    f"[manual fit] seed value(s) outside the limits you set were moved "
                    f"onto the limit before fitting: {', '.join(sorted(fit['clamped']))}")
            for line in self._identifiability_lines(fit):
                self.log_line.emit(line)
            if not fit["success"]:
                self.failed.emit(f"Manual fit did not converge: {fit['message']}")
                return
            result = SimpleNamespace(
                Lsd=fit["Lsd"], BC_y=fit["BC_y"], BC_z=fit["BC_z"],
                tx=fit["tx"], ty=fit["ty"], tz=fit["tz"], distortion={},
                pxY=self._pxY, pxZ=self._pxZ or self._pxY,
                NrPixelsY=self._NY, NrPixelsZ=self._NZ,
                wavelength_A=fit["wavelength_A"], post_residual_strain_uE=None,
                _calibrant_name=self._material_name, _d_list=list(self._d_list),
            )
            result.fit_sigma = dict(fit["sigma"])
            result.fit_at_limit = set(fit["at_limit"])
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _identifiability_lines(self, fit) -> list:
        """``value ± sigma`` for each refined parameter, plus an explicit
        warning for any the picked points did not actually pin down.

        This is the readout that makes an ill-posed manual fit obvious. With
        only a short ring arc on the detector — the usual case for a
        large-Lsd SAXS geometry, where a low-order AgBH ring runs off the
        edge — Lsd and the beam centre trade off against each other almost
        freely, and tilt does essentially nothing to the residual at all. The
        fit still converges; the sigmas are what say whether to believe it.
        """
        refine = self._refine or {}
        free = [n for n in ("Lsd", "BC_y", "BC_z", "tx", "ty", "tz", "wavelength_A")
                if refine.get({"BC_y": "BC", "BC_z": "BC",
                               "wavelength_A": "Wavelength"}.get(n, n),
                              n in ("Lsd", "BC_y", "BC_z"))]
        if not free:
            return []
        sigma, at_limit, lines, suspect = fit["sigma"], fit["at_limit"], [], []
        for name in free:
            unit, mult = self._SIGMA_UNITS[name]
            val, sig = fit[name], sigma.get(name, 0.0)
            if name in at_limit:
                lines.append(f"[manual fit]   {name} = {val * mult:.5g} {unit}  (at limit "
                             f"— the fit ran to the edge of the range you allowed)")
                suspect.append(name)
                continue
            lines.append(f"[manual fit]   {name} = {val * mult:.5g} "
                         f"± {sig * mult:.3g} {unit}")
            frac = self._SIGMA_WARN_FRAC.get(name)
            limit = abs(val) * frac if frac is not None else self._SIGMA_WARN_ABS[name]
            if not math.isfinite(sig) or sig > limit:
                suspect.append(name)
        if suspect:
            lines.append(
                f"[manual fit] WARNING: {', '.join(suspect)} not constrained by these "
                f"picks — the value above is largely fitted noise. Pick points on more "
                f"rings or over a wider arc, hold the parameter fixed, or bound it "
                f"via Limits…")
        return lines


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
            # float64 the whole way, as BatchWorker does (_ExplicitTIFFSource and
            # every other source hand it float64, and it integrates float64). This
            # used to narrow to float32 and widen back at the torch call, which
            # cost ~7 significant digits and made the Calibrate tab's profile
            # differ from the Batch run it is meant to preview — small (4e-8
            # relative here) but a difference with no reason to exist, in the one
            # plot a user reads a peak position off.
            img = self._image.astype(np.float64)
            if self._dark is not None or self._bright is not None or self._background is not None:
                img = apply_field_corrections(
                    img, dark=self._dark, bright=self._bright,
                    bright_mode=self._bright_mode, background=self._background).astype(np.float64)
            mask_t = None
            if self._mask is not None:
                mask_t = (_apply_im_trans(self._mask.astype(np.float32), self._im_trans)
                          if self._im_trans else self._mask.astype(np.float32))
            self.log_line.emit("[integrate] Running integration…")
            geom = build_geom(spec, "subpixel2", mask_t)
            # Needed for both the optional weighted profile and (below) masking
            # empty bins in the residual-strain cake, so compute unconditionally.
            cnt = count_cake(geom, "subpixel2", spec.NrPixelsZ, spec.NrPixelsY)
            img_t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float64))
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

def _filter_paths_by_frame_number(paths, frame_start, frame_end):
    """Keep only the paths whose parsed scan/file number falls in
    ``[frame_start, frame_end]`` (inclusive; a ``None`` bound is unbounded on
    that side) — the ``workers``-side counterpart of
    ``widgets.DataLoaderPanel``'s start/end spinboxes for a Batch Integrate
    ``unify_combine`` panel (see ``source_cfg()``'s ``frame_start``/
    ``frame_end`` keys).

    A no-op (returns ``paths`` unchanged) when both bounds are ``None`` —
    which is always true for a config built by a non-``unify_combine`` panel,
    since it never sets these keys — or when any path's number can't be
    parsed, mirroring ``DataLoaderPanel._file_numbers()``'s own
    all-or-nothing rule (filtering only ever applies when EVERY file in the
    selection carries a parseable number)."""
    if frame_start is None and frame_end is None:
        return list(paths)
    nums = []
    for p in paths:
        _root, num, _tag = froot_and_frame_num(Path(p).stem, -1)
        if num < 0:
            return list(paths)
        nums.append(num)
    lo = frame_start if frame_start is not None else min(nums)
    hi = frame_end if frame_end is not None else max(nums)
    return [p for p, n in zip(paths, nums) if lo <= n <= hi]


def _open_source_cfg(cfg):
    """Open a ``DataLoaderPanel.source_cfg()`` descriptor as a frame source.
    Shared by ``BatchWorker._open_source`` and ``BatchRunCoordinator`` (which
    needs a frame count up front, before any ``BatchWorker`` exists, to split
    a Batch-Parallel run into chunks)."""
    from midas_integrate_v2.streaming import TIFFGlobSource, HDF5FrameSource
    # A "unify_combine" panel (Batch Integrate) embeds chunk_size/frame_start/
    # frame_end directly in a tiff_glob/tiff_list cfg — route those through
    # _ChunkCombinedFileSource instead of the plain sources below, which know
    # nothing about either. Absent for every other "stream" consumer (Pump
    # Probe, bare-constructed panels), so this is a no-op there.
    tiff_unify = cfg["type"] in ("tiff_glob", "tiff_list") and (
        "chunk_size" in cfg or "frame_start" in cfg or "frame_end" in cfg)
    if cfg["type"] == "tiff_glob":
        if tiff_unify:
            paths = _filter_paths_by_frame_number(
                _list_tiff_files(cfg["path"]), cfg.get("frame_start"), cfg.get("frame_end"))
            return _ChunkCombinedFileSource(
                paths, chunk_size=cfg.get("chunk_size") or None, op=cfg.get("combine_op", "mean"))
        return TIFFGlobSource(cfg["path"])
    if cfg["type"] == "hdf5":
        # Route through _HDF5StackGlobSource (single-element path list) rather
        # than a plain HDF5FrameSource, so "Combine sub-frames" (chunk_size/
        # combine_op, now exposed for single-file HDF5 sources too) actually
        # takes effect instead of being silently ignored. frame_start/
        # frame_end mean something different here than for the multi-file
        # types above: a single file has no scan-number range to filter by,
        # so a unify_combine panel instead uses these keys as a 0-based
        # inclusive RAW SUB-FRAME range within the one file (see
        # widgets.DataLoaderPanel.source_cfg's "hdf5" branch) — safe to
        # overload the same keys since this branch never goes through
        # _filter_paths_by_frame_number.
        return _HDF5StackGlobSource(
            [cfg["path"]], cfg.get("dataset", "frames"),
            chunk_size=cfg.get("chunk_size") or None, op=cfg.get("combine_op", "mean"),
            raw_start=cfg.get("frame_start"), raw_end=cfg.get("frame_end"))
    if cfg["type"] == "tiff_list":
        if tiff_unify:
            paths = _filter_paths_by_frame_number(
                cfg["paths"], cfg.get("frame_start"), cfg.get("frame_end"))
            return _ChunkCombinedFileSource(
                paths, chunk_size=cfg.get("chunk_size") or None, op=cfg.get("combine_op", "mean"))
        return _ExplicitTIFFSource(cfg["paths"])
    if cfg["type"] == "hdf5_stack_glob":
        # Filtering is a no-op (returns paths unchanged) unless frame_start/
        # frame_end are actually present — i.e. only for a unify_combine
        # panel's cfg — so this is safe for every existing caller too.
        paths = _filter_paths_by_frame_number(
            cfg["paths"], cfg.get("frame_start"), cfg.get("frame_end"))
        return _HDF5StackGlobSource(
            paths, cfg.get("dataset", "exchange/data"),
            chunk_size=cfg.get("chunk_size") or None, op=cfg.get("combine_op", "mean"))
    raise ValueError(f"Unknown source type: {cfg['type']}")


class _ZarrGroupWriter:
    """One open ``.zarr.zip`` plus the per-frame bookkeeping its closing
    needs — the unit of ``BatchWorker``'s ``zarr_grouping``.

    Frames are handed to the backend's ``GSASZarrWriter`` AS THEY ARE
    PRODUCED rather than collected and written at the end: that class
    "flushes the running OmegaSumFrame chunk every omega_sum_frames, so a
    long scan never has to be held in memory", and buffering a rotation's
    worth of ``(n_eta, n_r)`` cakes to hand ``write_gsas_zarr_zip`` a list
    would throw that away (a 1442-frame group at 72×1000 float64 is ~830 MB).
    Peak memory is therefore the same as the one-archive-per-frame path.

    ``omega_sum_frames`` is deliberately left at its default 1. It chunks
    ALREADY-INTEGRATED frames inside the archive and is not OME_SUM, which
    "Combine sub-frames" has applied upstream before integration — setting it
    from OME_SUM would collapse the same axis twice.

    Provenance and the source ``instrument/`` tree are applied once at
    :meth:`close`, in a single ``provenance.rewrite_zip`` pass per archive
    (strictly fewer extract/repack cycles than the per-frame path, which paid
    for one per frame).
    """

    def __init__(self, path: Path, *, spec, bin_area, prov_entry, log):
        from midas_integrate_v2.io.zarr_gsas import GSASZarrWriter
        self.path = Path(path)
        self._spec = spec
        self._prov_entry = prov_entry
        self._log = log
        self._w = GSASZarrWriter(self.path, spec=spec, bin_area=bin_area)
        self.n_frames = 0
        # Per-frame environment, accumulated for the one closing stamp.
        self._ring_currents: list = []
        self._sample_motors: list = []
        # Per-frame (first_raw, last_raw) windows into ONE source file, which
        # is what h5_metadata.align reduces "one entry per output frame"
        # from. Set to None the moment a second source file appears: a group
        # spanning several files has no single instrument/ tree to align.
        self._h5_path: Optional[str] = None
        self._frame_ranges: Optional[list] = []
        self._n_aligned: Optional[int] = None
        self._multi_source = False

    def add(self, cake, *, omega, meta, h5_ctx):
        """Stream one integrated cake in, recording what its closing stamp
        will need. ``meta`` is ``metadata_for_index``'s dict (or None) and
        ``h5_ctx`` is ``h5_context_for_index``'s (or None)."""
        temp = press = cur = cur_i0 = None
        ring = motors = None
        if meta:
            temp = meta.get("temperature")
            press = meta.get("pressure")
            # Real beam-monitor ion chambers go in the writer's own I/I0
            # slots; storage-ring current is a different quantity and rides
            # in the provenance entry instead (see the per-frame path this
            # replaced, and 77a0ea1).
            cur = meta.get("ion_chamber_i")
            cur_i0 = meta.get("ion_chamber_i0")
            ring = meta.get("current")
            motors = {k[len("motor:"):]: v for k, v in meta.items()
                      if k.startswith("motor:") and v is not None}
        self._w.add_frame(cake, omega=omega, temperature=temp, pressure=press,
                          current=cur, current_i0=cur_i0)
        self.n_frames += 1
        self._ring_currents.append(ring)
        self._sample_motors.append(motors or None)
        if h5_ctx:
            if self._h5_path is None:
                self._h5_path = h5_ctx["path"]
                self._n_aligned = h5_ctx["n_aligned"]
            if h5_ctx["path"] != self._h5_path:
                self._multi_source = True
                self._frame_ranges = None
            elif self._frame_ranges is not None:
                self._frame_ranges.extend(h5_ctx["frame_ranges"])

    def close(self, h5_trees: dict, *, rename_to: Optional[Path] = None) -> Path:
        """Finish the archive, stamp it, optionally rename it, return its
        final path. ``h5_trees`` is the caller's source-path -> tree cache:
        the tree is the same for every frame of a file and costs a few
        hundred dataset reads, so it is read once per file, not per group."""
        self._w.close()
        try:
            extra = dict(self._prov_entry.get("extra") or {})
            extra["n_frames"] = self.n_frames
            # One frame keeps the scalar shape the per-frame path wrote, so
            # existing readers of a "frame"-grouped archive see no change;
            # a real group reports one value per frame.
            if any(v is not None for v in self._ring_currents):
                extra["storage_ring_current_mA"] = (
                    self._ring_currents[0] if self.n_frames == 1
                    else list(self._ring_currents))
            if any(self._sample_motors):
                extra["sample_motors"] = (
                    self._sample_motors[0] if self.n_frames == 1
                    else list(self._sample_motors))
            snap = {}
            if self._h5_path:
                extra["source_h5"] = self._h5_path
                if self._multi_source:
                    self._log(
                        f"[batch] note: {self.path.name} groups frames from "
                        "several source files — instrument/ tree not copied "
                        "(no single file to align it to).")
                else:
                    tree = h5_trees.get(self._h5_path)
                    if tree is None:
                        tree = h5_metadata.read_tree(self._h5_path)
                        h5_trees[self._h5_path] = tree
                    snap = h5_metadata.align(tree, self._frame_ranges,
                                             self._n_aligned)
            entry = dict(self._prov_entry, extra=extra)

            def _mutate(extracted, _snap=snap, _e=entry):
                if _snap:
                    h5_metadata.write_into_extracted(extracted, _snap)
                provenance.stamp_extracted(extracted, _e)

            provenance.rewrite_zip(self.path, _mutate)
        except Exception:
            self._log(f"[batch] note: provenance stamp on {self.path.name} "
                      "failed (non-fatal):\n" + traceback.format_exc())
        if rename_to is not None and rename_to != self.path:
            self.path.replace(rename_to)
            self.path = Path(rename_to)
        return self.path


class BatchWorker(QtCore.QThread):
    progress   = QtCore.pyqtSignal(int, int)
    frame_done = QtCore.pyqtSignal(str, object, object, object)  # id, r_axis, prof, sigma
    finished   = QtCore.pyqtSignal(dict)
    failed     = QtCore.pyqtSignal(str)
    log_line   = QtCore.pyqtSignal(str)
    geom_ready = QtCore.pyqtSignal(object)   # integration context (for reuse/caching)

    def __init__(self, spec, source_cfg, mask, out_dir, fmts, kernel,
                 corrections, variance_cfg, q_cfg=None, omega_cfg=None,
                 frame_range=None, frame_indices=None, monitor_file=None,
                 drift_traj=None, parent=None,
                 dark=None, bright=None, background=None, bright_mode="divide",
                 weighted=True, context=None, im_trans=(), multi_azimuth=False,
                 calibration_snapshot=None, zarr_grouping="frame",
                 ion_csv_extras=()):
        super().__init__(parent)
        # Full calibration (helpers.full_calibration_snapshot), embedded
        # verbatim in cake-mode HDF5 output — see cake_hdf5.write_cake_h5.
        self._calibration_snapshot = calibration_snapshot
        # How many combined output frames share one .zarr.zip: "frame" (one
        # archive each, the original mpe_wf-parity behaviour), "file" (one
        # per source file — one per ROTATION, the same unit ω is measured
        # from, see the sources' zarr_group_key) or "run" (one for the lot).
        self._zarr_grouping = (zarr_grouping or "frame") if zarr_grouping in (
            "frame", "file", "run") else "frame"
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
        # Optional column groups for the per-frame beam-monitor CSV
        # ("env", "motors"). The CSV itself is always written; only its
        # optional columns are a choice. See midas_gui.ion_csv.
        self._ion_csv_extras = set(ion_csv_extras or ())
        self._src  = source_cfg
        self._mask = mask
        self._out_dir = Path(out_dir) if out_dir else None
        # Accept a single legacy format string too (older call sites / tests).
        self._fmts = [fmts] if isinstance(fmts, str) else list(fmts or [])
        self._kernel = kernel
        self._corrections = corrections      # (pol, sa)
        self._variance_cfg = variance_cfg    # dict or None
        self._q_cfg = q_cfg                  # {"QMin","QMax","QBinSize"} or None
        # {"start","step","channel","collapse"} or None — the rotation each
        # frame was collected at, written to the zarr's /Omegas and the
        # combined HDF5. None means start=step=0.0: a genuine 0° on every
        # frame (see cake_params.omega_for_window), which is the right answer
        # for a stationary sample and an honest one for an unconfigured run.
        self._omega_cfg = dict(omega_cfg or {})
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

    def _ion_csv_name(self) -> str:
        """``<froot>_<detector>_bi.csv`` — the same shape Batch Correction
        uses for its own monitor sidecar, differing only in the tag.

        The froot comes from the SOURCE file's name rather than from
        ``_run_out_stem()``, which carries the first frame's id
        (``gC_1s_ICtweak_000017.000000``) and would put a frame number in
        a filename that describes the whole run.
        """
        from midas_gui.frame_correct import split_scan_name
        src = self._src.get("path") or (self._src.get("paths") or [None])[0]
        froot = split_scan_name(Path(src).name).froot if src else ""
        det = self._out_dir.name if self._out_dir is not None else ""
        stem = "_".join(p for p in (froot, det) if p) or "run"
        return stem + ion_csv.SUFFIX_BATCH_INTEGRATE

    def _run_out_stem(self) -> str:
        """The run-level output stem: the source file's own stem, or the
        shared file root of a multi-file pick. Shared by the combined cake
        HDF5's ``<stem>.<lo>_<hi>.cake`` name and a "run"-grouped zarr, so
        the two name the same run the same way."""
        src_path = self._src.get('path')
        if src_path:
            return Path(src_path).stem
        src_paths = self._src.get('paths') or []
        return (froot_and_frame_num(Path(src_paths[0]).stem, -1)[0]
                if src_paths else "integrated")

    def _zarr_group_path(self, zarr_dir: Path, fid: str, group_key) -> tuple:
        """``(path_to_open, rename_pending)`` for a new zarr group.

        A group keyed on a real source file is named after it the moment it
        opens — ``<stem>.ave.zarr.zip``, the same name the per-frame path
        gave a file whose chunking produced exactly one output.

        A group that spans the run instead (``"run"`` grouping, or ``"file"``
        on one-frame-per-file data, where the selection is the rotation) has
        no such name, and takes the combined HDF5's ``<stem>.<lo>_<hi>``
        form so a second run over a different frame range cannot overwrite
        the first. That range is only known once the loop ends, so the
        archive is built under ``.part`` and renamed on close — which also
        leaves an obviously-incomplete file behind if a run dies mid-write,
        rather than a plausible-looking one."""
        if self._zarr_grouping == "frame" or group_key is None:
            return zarr_dir / f"{fid}.ave.zarr.zip", False
        key_path = Path(str(group_key))
        if self._zarr_grouping == "file" and key_path.is_file():
            return zarr_dir / f"{key_path.stem}.ave.zarr.zip", False
        return zarr_dir / f"{self._run_out_stem()}.ave.zarr.zip.part", True

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

    def _omega_resolver(self, source):
        """``(fn(abs_i) -> float, description)`` for this run's rotation angle.

        Built once per run rather than branched per frame, so the choice of
        source — computed ramp, measured channel, collapsed single angle —
        is made (and logged) in one place, and the inner loop just calls a
        function.

        The computed ramp is ``cake_params.omega_for_window`` over the raw
        sub-frame window each output frame was built from. Sources that
        cannot report a window (``_ExplicitTIFFSource``, and the backend's
        own ``TIFFGlobSource``/``HDF5FrameSource``) fall back to
        ``(abs_i, abs_i)``, which is exactly right for one-frame-per-file
        data and means neither backend class had to grow a method.
        """
        from midas_gui import cake_params

        cfg = self._omega_cfg
        start = float(cfg.get("start") or 0.0)
        step = float(cfg.get("step") or 0.0)
        channel = str(cfg.get("channel") or "").strip()
        collapse = bool(cfg.get("collapse"))

        window_fn = getattr(source, "raw_window_for_index", None)

        def _window(abs_i):
            if window_fn is None:
                return abs_i, abs_i
            try:
                return window_fn(abs_i)
            except Exception:
                return abs_i, abs_i

        if collapse:
            # One angle for every frame: the middle of the whole run. Taken
            # from the SOURCE's full extent (frame 0 … n_frames-1), not from
            # the frames this worker iterates — in Batch-Parallel mode each
            # worker sees only its own chunk, and two chunks that each
            # averaged over their own slice would report different angles for
            # what the user said was one averaged exposure.
            #
            # On a multi-file pick these are file-LOCAL indices (each file
            # restarts at OME_START — see omega_channel_window), so the
            # "run-wide" middle is the middle of the local index range. That
            # is the right answer for the single-file, single-exposure case
            # this override exists for, and there is no better one for a
            # selection whose files each own a separate rotation.
            try:
                n_total = int(source.n_frames)
            except Exception:
                n_total = 0
            lo = _window(0)[0] if n_total else 0
            hi = _window(n_total - 1)[1] if n_total else 0
            one = cake_params.omega_for_window(start, step, lo, hi)
            return (lambda abs_i, _v=one: _v), (
                f"averaged/summed override — one ω={one:g}° for every frame")

        if channel:
            local_fn = getattr(source, "omega_channel_window", None)
            if local_fn is not None:
                cache = {}
                failed = []

                def _measured(abs_i):
                    # The same file-local window the computed ramp uses
                    # (see omega_channel_window) — a measured omega channel
                    # is a 1-D dataset stored per file, indexed from 0 in
                    # each, and the ramp was aligned to that rather than the
                    # other way round. What this call adds is the file.
                    try:
                        path, lo, hi = local_fn(abs_i)
                        arr = cache.get(path)
                        if arr is None:
                            import h5py
                            with h5py.File(path, "r") as f:
                                arr = np.asarray(f[channel][()], dtype=np.float64).ravel()
                            cache.clear()   # one file at a time, like _combined
                            cache[path] = arr
                        hi = min(int(hi), arr.size - 1)
                        if int(lo) > hi:
                            raise IndexError(abs_i)
                        # Arithmetic mean over the window regardless of the
                        # pixel combine op, for the same reason
                        # metadata_for_index uses one: an angle is a sample of
                        # where the sample was, not a detector count.
                        return float(np.mean(arr[int(lo):hi + 1]))
                    except Exception:
                        if not failed:
                            failed.append(True)
                            self.log_line.emit(
                                f"[batch] omega channel '{channel}' unreadable — "
                                f"falling back to OME_START/OME_STEP "
                                f"({start:g}, {step:g}).")
                        lo, hi = _window(abs_i)
                        return cake_params.omega_for_window(start, step, lo, hi)

                return _measured, f"measured channel '{channel}'"
            self.log_line.emit(
                f"[batch] omega channel '{channel}' ignored — this source type "
                f"has no per-file metadata; using OME_START/OME_STEP.")

        def _computed(abs_i):
            lo, hi = _window(abs_i)
            return cake_params.omega_for_window(start, step, lo, hi)

        if start == 0.0 and step == 0.0:
            desc = "OME_START=0, OME_STEP=0 — every frame recorded at ω=0°"
        else:
            desc = f"computed from OME_START={start:g}, OME_STEP={step:g}"
        return _computed, desc

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

            # Say out loud which azimuth the polarization correction is being
            # applied on. MIDAS η is measured from vertical, so the ring plane
            # is 90 — a plane of 0 is the one mistake that silently makes a
            # ring's azimuthal modulation worse instead of removing it, and a
            # project saved before 2026-09-24 restores the old 0.0 default.
            pol_mod = (self._corrections or (None, None))[0]
            if pol_mod is not None:
                plane = float(getattr(pol_mod, "pol_plane_eta_deg", float("nan")))
                frac = float(getattr(pol_mod, "pol_fraction", float("nan")))
                off_plane = not abs(abs(plane) - 90.0) < 1e-6
                self.log_line.emit(
                    f"[batch] Polarization: plane η = {plane:g}°, fraction = {frac:g}"
                    + ("  ← NOT horizontal; η is measured from vertical, so the "
                       "storage-ring plane is 90°" if off_plane else ""))

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
            # Combined multi-azimuth HDF5 (cake_hdf5.write_cake_h5), written once
            # at the end of run() alongside/instead of the zarr-per-frame output.
            want_h5_cake = (self._multi_azimuth and "h5" in self._fmts
                            and self._out_dir is not None)
            need_sigma = True   # xye/fxye require σ; always provide it
            rebin_unit = rebin_cfg_parts(self._q_cfg)[0]
            if self._multi_azimuth and self._q_cfg:
                # The rebin (rebin_R_to_grid) only handles a 1-D profile;
                # combining it with per-azimuth cake output isn't supported
                # yet — the UI already blocks this before starting the worker.
                raise RuntimeError(
                    "Multi-azimuth output isn't supported together with "
                    f"{_REBIN_LABEL[rebin_unit]}-uniform bins yet.")
            # Uniform Q / 2theta come from rebinning the R-uniform profile
            # (the kernels only bin in R) -- see rebin_R_to_grid.
            if self._q_cfg:
                out_grid, r_ax = rebin_grid_and_r(self._q_cfg, lsd, px, wl)

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
            omega_of, omega_desc = self._omega_resolver(source)
            total = (len(self._frame_indices) if self._frame_indices is not None
                     else source.n_frames)
            range_desc = (f"chunk of {total} frame(s)" if self._frame_indices is not None
                          else f"frame_range={self._frame_range}")
            self.log_line.emit(
                f"[batch] {total} frames | kernel={self._kernel} | "
                f"corrections={'on' if corr_on else 'off'} | "
                f"variance={'on' if self._variance_cfg else 'off'} | "
                f"rebin={_REBIN_LABEL[rebin_unit] + ' (uniform)' if self._q_cfg else 'off'} | "
                f"{range_desc} | "
                f"monitor={'yes' if monitor_vals else 'no'} | "
                f"drift={'on' if self._drift_traj else 'off'}")
            # Worth a line of its own: every run now stamps a rotation angle
            # into its zarr and HDF5, including runs whose user never opened
            # the cake dialog, so what it used should not have to be inferred.
            self.log_line.emit(f"[batch] omega: {omega_desc}")
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
            used_names: set = set()   # frame_output_stem collision guard
            # Two parallel lists that used to be one. all_frame_idx holds
            # 0-based FRAME INDICES and feeds only the combined-HDF5 stem's
            # <lo>_<hi> token below; all_omegas holds real DEGREES and feeds
            # the zarr and both HDF5 writers. They were the same list back
            # when the zarr's /Omegas was filled with frame indices — which
            # is the bug this splits.
            all_frame_idx = []
            all_omegas = []   # degrees, one per processed frame (like frame_ids)
            # (abs frame index, metadata dict|None) per processed frame, for
            # the run-level beam-monitor CSV written after the loop.
            ion_meta: list = []
            # Real engine-collapsed 1-D lineout per frame, multi-azimuth mode
            # only — see cake_hdf5.write_cake_h5's collapsed_profiles/sigmas.
            all_cake_profiles, all_cake_sigmas = [], []
            proc_idx = 0  # index into monitor_vals for processed frames only

            # How many combined output frames share one .zarr.zip, written as
            # they are produced (below, inside the loop):
            #   "frame" — one archive each. The original behaviour, mirroring
            #             mpe_wf's one-zarr-per-scan-point convention and
            #             extended so a file that "Combine sub-frames" splits
            #             into chunks gets one per chunk (see `fid`'s naming
            #             in _HDF5StackGlobSource._fid).
            #   "file"  — one per ROTATION, via the sources' zarr_group_key:
            #             one HDF5 sub-frame stack, or a whole TIFF selection.
            #   "run"   — one for every frame processed.
            # A source with no zarr_group_key (plain TIFFGlobSource/
            # HDF5FrameSource, outside Batch Integrate) can only do "frame".
            # Precompute what's shared across every write once, up front.
            zarr_dir = zarr_bin_area = zarr_prov_entry = None
            zarr_group_key_fn = None
            zarr_writer: Optional[_ZarrGroupWriter] = None
            zarr_open_key = None
            # Shared by the zarr writer below and cake_hdf5.write_cake_h5 at
            # the end of run() — computed once, whichever wants it first.
            #
            # /REtaMap row 3 (and cake_hdf5's own BinArea) is documented as the
            # per-bin summed area weight, "a property of the geometry alone" —
            # so it has to be the plain-kernel pixel-area count even on the
            # corrections path, where ctx["geom"] is deliberately None.
            # corr_counts is not a substitute: it is normalised through the
            # soft-bin kernel and folds in the polarization / solid-angle
            # factors, neither of which belongs in an area. Build a geometry
            # here purely for the count, so either output written with
            # corrections on carries the same BinArea as one written with them
            # off — shared by BOTH consumers below, not just zarr, since
            # count_cake(None, ...) crashes identically for want_h5_cake.
            cake_bin_area = None
            if want_zarr or want_h5_cake:
                cake_geom = geom if geom is not None else build_geom(spec, self._kernel, mask)
                cake_bin_area = count_cake(cake_geom, self._kernel, spec.NrPixelsZ, spec.NrPixelsY)
            h5_trees: dict = {}   # source path -> its instrument/ tree, read once
            if want_zarr:
                zarr_dir = self._out_dir / "zarr"
                zarr_dir.mkdir(parents=True, exist_ok=True)
                zarr_bin_area = cake_bin_area
                if self._zarr_grouping == "file":
                    zarr_group_key_fn = getattr(source, "zarr_group_key", None)
                    if zarr_group_key_fn is None:
                        self.log_line.emit(
                            "[batch] note: this source cannot say which "
                            "rotation a frame belongs to — writing one zarr "
                            "per frame instead of one per source file.")
                elif self._zarr_grouping == "run":
                    zarr_group_key_fn = lambda _i: "<run>"   # noqa: E731
                zarr_prov_entry = provenance.build_entry(
                    'midas_gui.batch_integrate',
                    inputs=[self._src.get('path')] if self._src.get('path') else [],
                    cake_params={
                        'RMin': float(spec.RMin), 'RMax': float(spec.RMax),
                        'RBinSize': float(spec.RBinSize), 'EtaMin': float(spec.EtaMin),
                        'EtaMax': float(spec.EtaMax), 'EtaBinSize': float(spec.EtaBinSize),
                    },
                    instrument_params=provenance.instrument_params_from_spec(spec),
                    extra={
                        'kernel': self._kernel, 'weighted': self._weighted,
                        'multi_azimuth': self._multi_azimuth,
                        'n_frames': 1, 'frame_range': list(self._frame_range),
                        # Widen provenance to match what a project attempt
                        # already records (see .context/DECISIONS.md) — safe
                        # to embed here since GSAS-II's own reader
                        # (G2pwd_MIDAS.py) never inspects zarr/HDF5 attrs,
                        # only the REtaMap/OmegaSumFrame/InstrumentParameters
                        # sections this entry has nothing to do with.
                        'calibration_snapshot': self._calibration_snapshot,
                        'mask_present': self._mask is not None,
                        'bright_mode': self._bright_mode,
                        'monitor_file': self._monitor_file,
                        'q_cfg': self._q_cfg,
                        'src_cfg': self._src,
                        'active_profile': settings.active_profile(),
                    },
                )
            zarr_rename_pending = False

            def _close_zarr_group() -> Path:
                """Finish the open archive and clear the group state, giving
                a run-spanning group its final ``<lo>_<hi>`` name now that
                the processed range is known. Returns the path written."""
                nonlocal zarr_writer, zarr_open_key, zarr_rename_pending
                final = None
                if zarr_rename_pending:
                    lo = int(min(all_frame_idx)) if all_frame_idx else 0
                    hi = int(max(all_frame_idx)) if all_frame_idx else 0
                    final = zarr_dir / (f"{self._run_out_stem()}."
                                        f"{lo:06d}_{hi:06d}.ave.zarr.zip")
                path = zarr_writer.close(h5_trees, rename_to=final)
                zarr_writer = None
                zarr_open_key = None
                zarr_rename_pending = False
                return path

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

                if self._q_cfg:   # R-uniform → uniform Q/2θ (not combined with cake mode)
                    prof, sigma = rebin_R_to_grid(compute_r_axis(spec), prof, sigma,
                                                  out_grid, lsd, px, wl, rebin_unit)
                # The live waterfall/stacked view always gets the η-collapsed profile,
                # in both modes — only the accumulated/stored result differs.
                if self._multi_azimuth and cake_2d is not None:
                    all_profiles.append(cake_2d)
                    all_sigmas.append(cake_sigma)
                    # The real engine-collapsed 1-D lineout — discarded above
                    # in favour of the raw cake, but cake_hdf5.write_cake_h5
                    # wants it (a true collapse, not the masked-eta-mean
                    # approximation reconstructed later for the plot tabs).
                    all_cake_profiles.append(prof)
                    all_cake_sigmas.append(sigma)
                else:
                    all_profiles.append(prof)
                    all_sigmas.append(sigma)
                frame_omega = float(omega_of(abs_i))
                all_omegas.append(frame_omega)
                # Instrument metadata (temperature/pressure/ion-chamber/
                # sample-motor positions), when the source can provide it
                # (HDF5 stacks only — see
                # _HDF5StackGlobSource.metadata_for_index) — always the mean
                # across this chunk's raw sub-frames, regardless of the pixel
                # combine op, and already aligned to the light-frame
                # timestamps rather than a longer light+dark metadata array.
                #
                # Read here rather than inside the zarr branch below: the
                # beam-monitor CSV is worth having whichever output formats
                # are ticked, and zarr is not one of the defaults.
                meta = None
                get_meta = getattr(source, "metadata_for_index", None)
                if get_meta is not None:
                    try:
                        meta = get_meta(abs_i)
                    except Exception:
                        meta = None
                ion_meta.append((abs_i, meta))
                if (want_zarr or self._multi_azimuth) and cake_2d is not None:
                    # Feeds the <lo>_<hi> token in the combined HDF5 stem,
                    # which both the plain and the cake writer below use —
                    # so it is not zarr-only.
                    all_frame_idx.append(float(abs_i))
                if want_zarr and cake_2d is not None:
                    # Which archive this frame belongs to. Under "frame"
                    # grouping (and for any source that cannot name a
                    # rotation) there is no key function and `fid` is used
                    # instead, so every frame opens and closes its own
                    # archive and the behaviour is exactly the per-frame one
                    # this replaced — `fid` already carries the right
                    # per-chunk identity (a bare file stem when "Combine
                    # sub-frames" produces one output per file, or
                    # "<stem>.frame_<start>_<end>" per chunk, see
                    # _HDF5StackGlobSource._fid).
                    group_key = (zarr_group_key_fn(abs_i)
                                 if zarr_group_key_fn is not None else fid)
                    # Where this frame came from, so the source file's whole
                    # instrument/ PV snapshot can be copied forward into the
                    # archive (see h5_metadata). HDF5 stacks only — a TIFF
                    # source has no such tree and leaves this None.
                    h5_ctx = None
                    get_ctx = getattr(source, "h5_context_for_index", None)
                    if get_ctx is not None:
                        try:
                            h5_ctx = get_ctx(abs_i)
                        except Exception:
                            h5_ctx = None
                    try:
                        if zarr_writer is not None and group_key != zarr_open_key:
                            out_paths.append(str(_close_zarr_group()))
                        if zarr_writer is None:
                            open_path, pending = self._zarr_group_path(
                                zarr_dir, fid, group_key)
                            zarr_writer = _ZarrGroupWriter(
                                open_path, spec=spec, bin_area=zarr_bin_area,
                                prov_entry=zarr_prov_entry,
                                log=self.log_line.emit)
                            zarr_open_key = group_key
                            zarr_rename_pending = pending
                        zarr_writer.add(cake_2d, omega=frame_omega,
                                        meta=meta, h5_ctx=h5_ctx)
                    except Exception:
                        self.log_line.emit(
                            f"[batch] zarr cake output for {fid!r} failed:\n"
                            + traceback.format_exc())
                        # Drop the half-written group rather than keep adding
                        # to a writer that may be in an undefined state; the
                        # next frame starts a fresh one.
                        if zarr_writer is not None:
                            try:
                                zarr_writer.close(h5_trees)
                            except Exception:
                                pass
                            zarr_writer = None
                            zarr_open_key = None
                            zarr_rename_pending = False
                frame_ids.append(fid)
                self.frame_done.emit(fid, r_ax, prof, sigma)
                self.progress.emit(proc_idx + 1, total)
                proc_idx += 1

                file_fmts = [f for f in self._fmts if f not in ("h5", "zarr")]
                if self._out_dir is not None and file_fmts:
                    # Allocate the stem ONCE per frame (not per format) and
                    # through frame_output_stem, so differently-padded frame
                    # ids in one run can't collapse onto the same filename and
                    # silently overwrite each other. The same stem is reused
                    # across every format subfolder, which is safe: they are
                    # different directories.
                    stem = frame_output_stem(fid, abs_i, used_names)
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
                            # The cake goes over unconditionally — want_cake
                            # already computed one whenever 2d_csv is among
                            # the formats — and multi-azimuth decides only
                            # whether to fan out per η. Gating the cake
                            # itself on multi-azimuth is what made 2D CSV a
                            # no-op with that checkbox off.
                            cake_2d=cake_2d, cake_sigma=cake_sigma,
                            eta_axis=eta_ax, per_eta=self._multi_azimuth))

            # The last group has no following frame to close it — including
            # when the loop broke on an abort, where the frames already
            # integrated still belong in a readable archive.
            if zarr_writer is not None:
                try:
                    out_paths.append(str(_close_zarr_group()))
                except Exception:
                    self.log_line.emit(
                        "[batch] zarr cake output for the final group failed:\n"
                        + traceback.format_exc())

            # Combined h5 output name: <original-source-stem>.<start>_<end>
            # .cake — h5 remains one file for the whole run, so it still needs
            # an explicit frame-index range. start/end are the actual
            # processed 0-based frame indices (all_frame_idx), matching what
            # per-frame lineout files already use via
            # froot_and_frame_num(fid, abs_i) above. A "run"-grouped zarr
            # shares both the stem and the range — see _zarr_group_path.
            out_stem = self._run_out_stem()
            lo = int(min(all_frame_idx)) if all_frame_idx else 0
            hi = int(max(all_frame_idx)) if all_frame_idx else 0
            combined_stem = f"{out_stem}.{lo:06d}_{hi:06d}.cake"

            # Per-frame beam-monitor CSV. ONE file for the whole run, and
            # placed/named exactly as Batch Correction places its own (see
            # BatchCorrectionWorker._write_ion_csvs): beside the detector
            # folder rather than inside it, as
            # ``<froot>_<detector>_bi.csv``. The two tabs write the same
            # kind of sidecar about the same scan, so reading one should
            # not mean learning a second convention — it used to be
            # ``<stem>.<lo>_<hi>.ioncham.csv`` one level further down, and
            # a re-run over a different frame range left a second file
            # beside the first rather than replacing it.
            if self._out_dir is not None and ion_meta:
                try:
                    written = ion_csv.write_ion_csv(
                        self._out_dir.parent / self._ion_csv_name(),
                        ion_csv.rows_from_metas([m for _i, m in ion_meta]),
                        extras=self._ion_csv_extras)
                    if written:
                        out_paths.append(written)
                        self.log_line.emit(f"[batch] beam monitors: {written}")
                    else:
                        self.log_line.emit(
                            "[batch] no beam-monitor data for this source "
                            "(unrecognised hutch, or no live channel) — "
                            "no monitor CSV written")
                except Exception:
                    # Never cost the user a finished run over a sidecar.
                    self.log_line.emit("[batch] monitor CSV failed:\n"
                                       + traceback.format_exc())

            prov_entry = provenance.build_entry(
                'midas_gui.batch_integrate',
                inputs=[self._src.get('path')] if self._src.get('path') else [],
                cake_params={
                    'RMin': float(spec.RMin), 'RMax': float(spec.RMax),
                    'RBinSize': float(spec.RBinSize), 'EtaMin': float(spec.EtaMin),
                    'EtaMax': float(spec.EtaMax), 'EtaBinSize': float(spec.EtaBinSize),
                },
                instrument_params=provenance.instrument_params_from_spec(spec),
                extra={
                    'kernel': self._kernel, 'weighted': self._weighted,
                    'multi_azimuth': self._multi_azimuth,
                    'n_frames': len(all_profiles), 'frame_range': list(self._frame_range),
                    # Same widening as zarr_prov_entry above — see that
                    # comment for why this is safe.
                    'calibration_snapshot': self._calibration_snapshot,
                    'mask_present': self._mask is not None,
                    'bright_mode': self._bright_mode,
                    'monitor_file': self._monitor_file,
                    'q_cfg': self._q_cfg,
                    'src_cfg': self._src,
                    'active_profile': settings.active_profile(),
                    # Known only now the frame loop has finished — the project's
                    # own attempt record carries the same field.
                    'aborted': aborted,
                },
            )

            # HDF5: single file with the full stack. Multi-azimuth mode writes
            # the cake-capable layout (cake_hdf5.write_cake_h5); otherwise the
            # plain 1-D midas_integrate_v2.write_h5 path, unchanged.
            if self._out_dir is not None and "h5" in self._fmts:
                if self._multi_azimuth:
                    h5_dir = self._out_dir / "h5"
                    h5_dir.mkdir(parents=True, exist_ok=True)
                    h5_path = h5_dir / f"{combined_stem}.h5"
                    try:
                        from midas_gui.cake_hdf5 import write_cake_h5
                        write_cake_h5(
                            h5_path, cake=np.array(all_profiles),
                            cake_sigma=np.array(all_sigmas),
                            collapsed_profiles=(np.array(all_cake_profiles)
                                                if all_cake_profiles else None),
                            collapsed_sigmas=(np.array(all_cake_sigmas)
                                              if all_cake_sigmas else None),
                            r_axis=r_ax, eta_axis=eta_ax, frame_ids=frame_ids,
                            omegas=all_omegas,
                            spec=spec, bin_area=cake_bin_area,
                            kernel=self._kernel, weighted=self._weighted,
                            cake_params={
                                'RMin': float(spec.RMin), 'RMax': float(spec.RMax),
                                'RBinSize': float(spec.RBinSize),
                                'EtaMin': float(spec.EtaMin), 'EtaMax': float(spec.EtaMax),
                                'EtaBinSize': float(spec.EtaBinSize),
                            })
                        try:
                            # out_paths is only fully known here (every zarr
                            # sibling from this run, plus this h5's own path).
                            prov_entry['extra']['out_paths'] = out_paths + [str(h5_path)]
                            stamp_h5_provenance(h5_path, prov_entry)
                        except Exception:
                            self.log_line.emit(
                                f"[batch] note: provenance stamp on {h5_path.name} "
                                "failed (non-fatal):\n" + traceback.format_exc())
                        out_paths.append(str(h5_path))
                    except Exception:
                        self.log_line.emit(
                            "[batch] cake HDF5 output failed:\n" + traceback.format_exc())
                else:
                    h5_dir = self._out_dir / "h5"
                    h5_dir.mkdir(parents=True, exist_ok=True)
                    h5_path = h5_dir / f"{combined_stem}.h5"
                    m.write_h5(str(h5_path),
                               profiles=np.array(all_profiles),
                               r_axis=r_ax,
                               frame_ids=frame_ids,
                               sigmas=np.array(all_sigmas),
                               # One rotation angle per profile, same order —
                               # see write_all_profiles for why this is an
                               # extra dataset rather than a metadata field.
                               extra_datasets={"omegas": np.asarray(
                                   all_omegas, dtype=np.float64)})
                    try:
                        prov_entry['extra']['out_paths'] = out_paths + [str(h5_path)]
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
                "omegas": all_omegas,
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

            rebin_unit = rebin_cfg_parts(self._q_cfg)[0]
            if self._q_cfg:
                out_grid, r_ax = rebin_grid_and_r(self._q_cfg, lsd, px, wl)
            two_theta, _, q_ax = axis_conversions(r_ax, lsd, px, wl)

            fields_on = (self._dark is not None or self._bright is not None
                         or self._background is not None)
            total = len(self._frames)
            self.log_line.emit(
                f"[pump] {total} frames | kernel={self._kernel} | "
                f"corrections={'on' if ctx['corr_on'] else 'off'} | "
                f"rebin={_REBIN_LABEL[rebin_unit] if self._q_cfg else 'off'} | "
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
                    prof, _ = rebin_R_to_grid(compute_r_axis(spec), prof, None,
                                              out_grid, lsd, px, wl, rebin_unit)
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
        """Take the mean of repeats per delay → I_by_delay; subtract the reference (mean over
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


class _ChunkCombinedFileSource:
    """Iterate over an arbitrary, already-resolved ``list[str]`` of
    single-frame files (TIFF/``.ge*``), combining every ``chunk_size``
    consecutive FILES into one output frame via ``helpers._COMBINE_OPS`` —
    the TIFF-family counterpart of ``_HDF5StackGlobSource``'s "Combine
    sub-frames", for a Batch Integrate ``unify_combine`` panel (see
    ``widgets.DataLoaderPanel.source_cfg``'s ``chunk_size``/``combine_op``
    on the ``"tiff_glob"``/``"tiff_list"`` types).

    Unlike ``_HDF5StackGlobSource`` — where each file already holds several
    raw sub-frames and chunking never crosses a file boundary — a TIFF-family
    file holds exactly ONE raw frame, so there is no smaller unit to group
    within a file; chunking here groups consecutive FILES instead, freely
    crossing file boundaries. Any start/end file-number filtering the caller
    wants has already been applied to ``paths`` before construction (see
    ``_open_source_cfg``/``_filter_paths_by_frame_number``), so chunk
    boundaries here always start counting from ``paths[0]``.

    ``chunk_size`` falsy (``None``/``0``) combines every file in ``paths``
    into a single output frame, mirroring ``read_hdf5_stack_combined``'s
    "whole file" convention. ``chunk_size == 1`` (the default) combines
    nothing — one output frame per file, identical to ``_ExplicitTIFFSource``/
    ``TIFFGlobSource``, including frame ids (bare filename stem).

    A multi-file chunk's fid is ``"<first_file_stem>.frame_<start>_<end>"``
    (``start``/``end`` the chunk's 0-based, INCLUSIVE file-index range within
    THIS source's own — already start/end-filtered — ``paths``, not a global
    index) — the same ``.frame_<start>_<end>`` convention
    ``_HDF5StackGlobSource`` mints for its own multi-frame chunks, so
    ``froot_and_frame_num`` recovers the first file's own scan number and
    keeps chunks distinct exactly the way it already does there."""

    def __init__(self, paths, *, chunk_size=None, op: str = "mean"):
        self._paths = [Path(p) for p in paths]
        self._chunk_size = int(chunk_size) if chunk_size else (len(self._paths) or 1)
        from midas_gui.helpers import _COMBINE_OPS
        self._combine = _COMBINE_OPS.get(op, _COMBINE_OPS["mean"])

    @property
    def n_frames(self) -> int:
        n = len(self._paths)
        return -(-n // self._chunk_size) if n else 0

    def _group(self, k: int) -> list:
        start = k * self._chunk_size
        return self._paths[start:start + self._chunk_size]

    def _fid(self, k: int, group: list) -> str:
        if len(group) == 1:
            return group[0].stem
        start = k * self._chunk_size
        return f"{group[0].stem}.frame_{start}_{start + len(group) - 1}"

    def raw_window_for_index(self, idx: int) -> tuple:
        """Inclusive, 0-based raw-frame range output frame ``idx`` was built
        from, counted across the whole selection — what
        ``cake_params.omega_for_window`` needs to turn OME_START/OME_STEP
        into an angle.

        A TIFF-family file holds exactly one raw frame, so the "raw index"
        of a file is just its position in ``paths``, and chunking groups
        whole files: the window is the chunk's own file-index range.

        Note this counts across the whole selection, where
        ``_HDF5StackGlobSource`` restarts at every file. The two are not
        inconsistent — the rule in both is that ω is measured from raw
        sub-frame 0 of the rotation. An HDF5 sub-frame stack IS one
        rotation, whereas one-frame-per-file data only becomes a rotation as
        a series, so here the selection is the rotation and ``paths[0]`` is
        its start (``paths`` is already the start/end filtered list — see the
        class docstring)."""
        group = self._group(idx)
        if not group:
            raise IndexError(idx)
        start = idx * self._chunk_size
        return start, start + len(group) - 1

    def zarr_group_key(self, idx: int) -> str:
        """Which zarr archive output frame ``idx`` belongs to under "one zarr
        per source file" — see ``BatchWorker``'s ``zarr_grouping``.

        A zarr group is one ROTATION, the same unit ω is measured from, so
        this is deliberately the same answer ``raw_window_for_index`` is
        counted against: one-frame-per-file data only becomes a rotation as a
        series, so the whole selection is one group and the key is constant.
        (``_HDF5StackGlobSource`` returns the source file instead, because
        there one file already IS one rotation.)"""
        if not self._group(idx):
            raise IndexError(idx)
        return "<selection>"

    def _read(self, group: list) -> np.ndarray:
        imgs = []
        for p in group:
            img = _load_image(p).astype(np.float64)
            imgs.append(img[0] if img.ndim == 3 else img)
        if len(imgs) == 1:
            return imgs[0]
        return self._combine(np.stack(imgs, axis=0)).astype(np.float64)

    def __iter__(self):
        for k in range(self.n_frames):
            group = self._group(k)
            yield self._fid(k, group), self._read(group)

    def get(self, idx: int):
        group = self._group(idx)
        if not group:
            raise IndexError(idx)
        return self._fid(idx, group), self._read(group)


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
    looked hung (no progress, no output, no error).

    Reading has the same rule, and it took a second pass to finish the job:
    counting stopped decoding, but ``get(idx)`` still decoded the whole
    OWNING file to return one frame from it, so a single-file 1442-sub-frame
    VAREX pick cost 23.9 GB (~230 s over NFS) per first access — the
    Detector-view preview and every parallel ``BatchWorker`` chunk each paid
    it. ``_combined_one(i, k)`` now reads only chunk ``k``'s raw sub-frames.
    The pixel cache is bounded to that one most recently used chunk (it was
    the most recent whole file, and before that an unbounded dict retaining
    every decoded file for the life of the run); ``__iter__`` and a chunked
    ``BatchWorker`` both walk frames in order, so one chunk is all the
    lookahead either needs."""

    #: HDF5 paths for the per-acquisition scalars mpe_wf/GSAS-II's zarr
    #: schema carries — fixed regardless of ``dataset`` (the cake-source
    #: dataset the user picked), since these live under the file's
    #: ``instrument/`` group, not under ``exchange/``.
    #: Per-frame instrument metadata, by hutch. The tables themselves live in
    #: :mod:`midas_gui.ion_csv`, which the Batch Correction path also reads
    #: them from — one mapping rather than two that can drift. Kept as class
    #: attributes so existing references (and tests) still resolve.
    _METADATA_H5_PATHS = ion_csv.METADATA_H5_PATHS
    _ION_CHAMBER_H5_PATHS = ion_csv.ION_CHAMBER_H5_PATHS
    _SAMPLE_MOTOR_H5_GROUPS = ion_csv.SAMPLE_MOTOR_H5_GROUPS

    def __init__(self, paths, dataset: str, *, chunk_size=None, op: str = "mean",
                 raw_start=None, raw_end=None):
        self._paths = [Path(p) for p in paths]
        self._dataset = dataset
        self._chunk_size = chunk_size
        self._op = op
        # 0-based inclusive raw sub-frame bounds, only ever set for a
        # single-file "hdf5" cfg (see widgets.DataLoaderPanel.source_cfg's
        # "hdf5" branch / workers._open_source_cfg) — a multi-file
        # "hdf5_stack_glob" source is filtered at the FILE level instead
        # (_filter_paths_by_frame_number), before this class ever sees the
        # survivors, so every other construction site leaves these None.
        self._raw_start = raw_start
        self._raw_end = raw_end
        self._cache: dict = {}   # (path index, chunk index) -> np.ndarray (most-recent chunk only)
        self._counts: Optional[list] = None    # per-file combined-frame count
        self._raw_ns: Optional[list] = None    # per-file raw (pre-combine) sub-frame count
        self._metadata_cache: dict = {}   # path index -> {name: np.ndarray|None}
        self._aligned_cache: dict = {}   # path index -> light-frame count
        self._hutch = self._resolve_hutch()

    def _resolve_hutch(self) -> Optional[str]:
        """This source's hutch, from the first selected path, falling back to
        the active profile — see :func:`midas_gui.ion_csv.resolve_hutch`."""
        return ion_csv.resolve_hutch(self._paths[0] if self._paths else None,
                                     settings.active_profile())

    def _metadata_h5_paths(self) -> dict:
        """Flat ``{key: h5_path}`` table for this source's hutch: the fixed
        temperature/pressure/current entries plus whichever ion-chamber
        entries apply (none, for an unrecognized hutch)."""
        return ion_csv.metadata_h5_paths(self._hutch)

    def _raw_bounds(self, n: int) -> tuple:
        """0-based inclusive ``(lo, hi)`` raw sub-frame bounds within a file
        of ``n`` raw sub-frames, from ``raw_start``/``raw_end`` — ``(0, n-1)``
        (the whole file) when neither is set."""
        lo = max(0, self._raw_start) if self._raw_start is not None else 0
        hi = min(n - 1, self._raw_end) if self._raw_end is not None else n - 1
        return lo, hi

    def _stat(self, i: int) -> tuple:
        """``(n_chunks, raw_n)`` for file ``i`` — from its dataset shape
        alone (no pixel read), one h5py header open covering both. Mirrors
        ``read_hdf5_stack_combined``'s own chunking: a 2-D dataset is one
        frame; otherwise ``ceil(N_eff / chunk_size)`` combined frames out of
        ``N_eff`` (``raw_start``/``raw_end``-filtered) raw ones, with a falsy
        chunk_size meaning "whole (filtered) file" (one combined frame).
        ``raw_n`` (second element) is always the file's TRUE, unfiltered raw
        count — ``_read_metadata``'s light/dark boundary detection needs the
        whole per-acquisition metadata array, not just the filtered slice."""
        import h5py
        try:
            with h5py.File(str(self._paths[i]), "r") as f:
                dset = f[self._dataset]
                if dset.ndim == 2:
                    return 1, 1
                n = int(dset.shape[0])
        except Exception:
            return 1, 1   # unreadable/odd file — assume 1; get() surfaces the real error
        lo, hi = self._raw_bounds(n)
        n_eff = hi - lo + 1
        if not self._chunk_size:
            return (1 if n_eff > 0 else 0), n
        size = int(self._chunk_size)
        return (max(1, -(-n_eff // size)) if n_eff > 0 else 0), n   # ceil

    def _ensure_stats(self) -> None:
        if self._counts is None:
            stats = [self._stat(i) for i in range(len(self._paths))]
            self._counts = [s[0] for s in stats]
            self._raw_ns = [s[1] for s in stats]

    def _combined_one(self, i: int, k: int):
        """Combined frame ``k`` of file ``i``, or ``None`` when ``k`` is past
        that file's last chunk.

        Reads only chunk ``k``'s raw sub-frames. The whole-file
        ``read_hdf5_stack_combined`` this replaced decoded every chunk and
        cached the list, so ``get(0)`` on a 1442-sub-frame VAREX file read all
        23.9 GB (~230 s over NFS) and held ~1.9 GB, to return one 33 MB frame
        — the Detector-view preview and the first frame of every parallel
        ``BatchWorker`` chunk each paid that in full. Same bug as the
        decode-to-count one in the class docstring, on the pixel path rather
        than the header path; fixing counting alone left it live.
        """
        key = (i, k)
        cached = self._cache.get(key)
        if cached is None:
            cached = read_hdf5_stack_chunk(
                self._paths[i], self._dataset, k,
                chunk_size=self._chunk_size, op=self._op,
                raw_start=self._raw_start, raw_end=self._raw_end)
            if cached is None:
                return None
            self._cache = {key: cached}   # one chunk at a time
        return cached

    @property
    def n_frames(self) -> int:
        self._ensure_stats()
        return sum(self._counts)

    def _chunk_range(self, k: int, n_raw: int) -> tuple:
        """Inclusive ``(start, end)`` raw 0-based sub-frame range that
        combined-frame ``k`` was built from — the same range ``_fid`` embeds
        in the frame id, reused here to know exactly which raw metadata
        entries (see ``_read_metadata``) belong to this combined frame.
        ``n_raw`` is the file's TRUE raw count (see ``_stat``); the
        ``raw_start``/``raw_end`` filter is re-applied here via
        ``_raw_bounds`` so the reported range stays in absolute (unfiltered)
        raw indices, matching ``_combined``'s own slicing."""
        lo, hi = self._raw_bounds(n_raw)
        if not self._chunk_size:
            return lo, hi
        size = int(self._chunk_size)
        start = lo + k * size
        end = min(start + size, hi + 1) - 1
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
        instrument group). Keys prefixed ``motor:`` are per-channel
        sample-motion-system positions (see ``_read_sample_motors``); every
        other key is a single scalar-per-acquisition quantity."""
        cached = self._metadata_cache.get(i)
        if cached is not None:
            return cached
        metadata_h5_paths = self._metadata_h5_paths()
        out = {k: None for k in metadata_h5_paths}
        try:
            import h5py
            self._ensure_stats()
            n_data = self._raw_ns[i]
            with h5py.File(str(self._paths[i]), "r") as f:
                n_aligned = self._metadata_frame_count(f, n_data)
                self._aligned_cache[i] = n_aligned
                for key, h5_path in metadata_h5_paths.items():
                    if h5_path in f:
                        arr = np.asarray(f[h5_path][()], dtype=np.float64)
                        if arr.ndim == 1 and arr.size >= n_aligned:
                            out[key] = arr[:n_aligned]
                out.update(self._read_sample_motors(f, n_aligned))
        except Exception:
            pass
        self._metadata_cache[i] = out
        return out

    def _read_sample_motors(self, f, n_aligned: int) -> dict:
        """Every channel under this hutch's sample-motion-system group(s)
        (``_SAMPLE_MOTOR_H5_GROUPS``), keyed ``"motor:<group-leaf>/<channel>"``
        (e.g. ``"motor:HR/samX"``) so E hutch's two coexisting sub-configs
        (HL/HR — no reliable way to tell which is physically in use for a
        given file, same gap as hutch detection itself) don't collide.
        Channel names vary between groups (D's HR set differs from E's), so
        this discovers them from the file rather than hardcoding a list."""
        out: dict = {}
        for group_path in self._SAMPLE_MOTOR_H5_GROUPS.get(self._hutch, []):
            if group_path not in f:
                continue
            leaf = group_path.rsplit("/", 1)[-1]
            group = f[group_path]
            for channel in group.keys():
                arr = np.asarray(group[channel][()], dtype=np.float64)
                if arr.ndim == 1 and arr.size >= n_aligned:
                    out[f"motor:{leaf}/{channel}"] = arr[:n_aligned]
        return out

    def _aligned_count(self, i: int) -> int:
        """How many leading metadata entries in file ``i`` belong to real
        light frames — ``_metadata_frame_count`` memoised, so callers that
        only want the alignment (``h5_context_for_index``) don't have to pull
        the whole per-key metadata dict to get at it."""
        cached = self._aligned_cache.get(i)
        if cached is not None:
            return cached
        self._ensure_stats()
        n = self._raw_ns[i]
        try:
            import h5py
            with h5py.File(str(self._paths[i]), "r") as f:
                n = self._metadata_frame_count(f, n)
        except Exception:
            pass
        self._aligned_cache[i] = n
        return n

    def h5_context_for_index(self, idx: int) -> dict:
        """Which source file combined frame ``idx`` came from, and which raw
        sub-frames of it — everything ``h5_metadata.snapshot`` needs to copy
        that file's ``instrument/`` tree forward with its per-frame arrays
        averaged over the right window.

        Separate from ``metadata_for_index`` because that one answers "what
        were the six named scalars for this frame" (values, already reduced)
        while this answers "where did this frame come from" (a location), and
        the bulk copy needs the location.
        """
        self._ensure_stats()
        remaining = idx
        for i, p in enumerate(self._paths):
            n_here = self._counts[i]
            if remaining < n_here:
                start, end = self._chunk_range(remaining, self._raw_ns[i])
                return {"path": str(p), "frame_ranges": [(start, end)],
                        "n_aligned": self._aligned_count(i)}
            remaining -= n_here
        raise IndexError(idx)

    def omega_channel_window(self, idx: int) -> tuple:
        """``(path, lo, hi)`` — the inclusive, FILE-LOCAL, 0-based raw
        sub-frame range combined frame ``idx`` was built from, and the file
        it came from.

        This is the one rotation window in this class: both consumers of an
        angle go through it. ``_chunk_range`` answers it per file, in that
        file's own absolute (unfiltered) raw indices, so the
        ``raw_start``/``raw_end`` filter SHIFTS the angles rather than
        rebasing them — OME_START is the angle of raw sub-frame 0 of the
        file, and skipping the first ten sub-frames does not move where that
        file's rotation began.

        **Each file restarts at OME_START.** One HDF5 sub-frame stack is one
        rotation, so the index is deliberately local and the walk carries no
        cumulative offset. It used to: ``raw_window_for_index`` added every
        preceding file's raw count to make one continuous ramp across a
        multi-file pick, which meant the computed OME_START/OME_STEP ramp and
        a MEASURED omega channel — a 1-D dataset stored per file, indexed
        from 0 in each, which therefore has no choice but to be file-local —
        ran in two different coordinate systems, and that a file-number
        filter silently rebased the ramp (the dropped files were gone before
        this class saw them, so their sub-frames could not be counted).
        One window removes both discrepancies.

        Still safe in Batch-Parallel mode, where each chunk worker sees only
        its own slice: the window is a pure function of ``idx`` and the
        header counts, both identical in every worker, so two workers cannot
        disagree about a frame's angle."""
        self._ensure_stats()
        remaining = idx
        for i, p in enumerate(self._paths):
            n_here = self._counts[i]
            if remaining < n_here:
                start, end = self._chunk_range(remaining, self._raw_ns[i])
                return str(p), start, end
            remaining -= n_here
        raise IndexError(idx)

    def raw_window_for_index(self, idx: int) -> tuple:
        """The same window as ``omega_channel_window``, without the file —
        what ``cake_params.omega_for_window`` needs to turn OME_START/
        OME_STEP into an angle.

        Defined in terms of it rather than repeating the walk: the computed
        ramp and a measured channel must index the same axis, and the surest
        way to keep them doing so is to leave only one place that decides
        what the axis is."""
        return self.omega_channel_window(idx)[1:]

    def zarr_group_key(self, idx: int) -> str:
        """Which zarr archive output frame ``idx`` belongs to under "one zarr
        per source file" — see ``BatchWorker``'s ``zarr_grouping``.

        A zarr group is one ROTATION, and here one file IS one rotation, so
        the key is the source file. Defined through ``omega_channel_window``
        for the same reason ``raw_window_for_index`` is: that walk is already
        the single place deciding which rotation a frame belongs to, and
        grouping must not get to disagree with the angle."""
        return self.omega_channel_window(idx)[0]

    def metadata_for_index(self, idx: int) -> dict:
        """Chunk-mean metadata (Temperature/Pressure/StorageRing current)
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
            n_here, n_raw = self._counts[i], self._raw_ns[i]
            for k in range(n_here):
                img = self._combined_one(i, k)
                if img is None:   # header count disagreed with the real read
                    break
                yield self._fid(p, k, n_here, n_raw), img.astype(np.float64)

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
                img = self._combined_one(i, remaining)
                if img is None:   # header count disagreed with the real read
                    raise IndexError(idx)
                return (self._fid(p, remaining, n_here, self._raw_ns[i]),
                        img.astype(np.float64))
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
        self._used_names: set = set()   # frame_output_base collision guard
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
            out_grid = None
            rebin_unit = rebin_cfg_parts(self._q_cfg)[0]
            if self._q_cfg:
                out_grid, r_ax = rebin_grid_and_r(self._q_cfg, lsd, px, wl)
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
                        prof, sigma = rebin_R_to_grid(compute_r_axis(spec), prof, sigma,
                                                      out_grid, lsd, px, wl, rebin_unit)
                    self._seen.add(fid)
                    count += 1
                    self.frame_done.emit(fid, r_ax, prof, sigma)
                    self.new_count.emit(count)
                    self.log_line.emit(f"[monitor] +{fid}: peak={prof.max():.1f}")
                    if can_save:
                        self._out_dir.mkdir(parents=True, exist_ok=True)
                        base = frame_output_base(self._out_dir, fid, count,
                                                 self._used_names)
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


def _split_into_chunks_on_groups(indices: list, n_chunks: int, key_fn) -> list:
    """Like :func:`_split_into_chunks`, but no chunk boundary may fall inside
    a run of indices sharing a ``key_fn(idx)``.

    Batch-Parallel splits purely by count, which is fine while every output
    frame writes its own zarr but not once several share one archive: two
    workers handed halves of the same source file would open the same
    ``.zarr.zip`` path and race. Whole groups per worker removes the
    collision by construction, at the cost of capping parallelism at the
    number of groups (a 3-file folder uses 3 workers however many were
    asked for).

    Groups are formed from CONSECUTIVE equal keys, not by gathering every
    index with the same key: the frame order is the source's own, so a
    repeated key after a gap would mean the source interleaves rotations,
    and silently reordering frames to suit the writer would be worse than
    writing two archives. Concatenating the chunks back in order reproduces
    ``indices`` exactly, exactly as the unaligned splitter promises."""
    groups: list = []
    for i in indices:
        k = key_fn(i)
        if groups and groups[-1][0] == k:
            groups[-1][1].append(i)
        else:
            groups.append((k, [i]))
    n_chunks = max(1, min(int(n_chunks), len(groups)))
    if not groups:
        return []
    # Distribute whole groups so chunk sizes stay as even as the group sizes
    # allow: walk the groups, starting a new chunk once the current one has
    # taken its share of the remaining frames.
    chunks: list = []
    remaining_frames = sum(len(g[1]) for g in groups)
    remaining_chunks = n_chunks
    cur: list = []
    for gi, (_k, members) in enumerate(groups):
        cur.extend(members)
        groups_left = len(groups) - gi - 1
        if remaining_chunks > 1 and groups_left >= remaining_chunks - 1 and \
                len(cur) >= remaining_frames // remaining_chunks:
            chunks.append(cur)
            remaining_frames -= len(cur)
            remaining_chunks -= 1
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


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
                       lsd, px, wl, eta_axis=None, spec=None, bin_area=None,
                       calibration_snapshot=None, kernel=None, weighted=None,
                       cake_params=None, omegas=None) -> list:
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
    checked to get that format. ``"h5"`` when ``profiles`` is 3-D is written
    via ``cake_hdf5.write_cake_h5`` when ``spec`` is supplied (the caller has
    the ``IntegrationSpec`` the run used); silently skipped, as before, when
    it isn't. Returns the list of paths written.
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
        used_names: set = set()   # frame_output_base collision guard
        for i, fid in enumerate(frame_ids):
            base = frame_output_base(out_dir, fid, i, used_names)
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
        # The per-frame rotation angle rides along as an extra dataset
        # rather than a ProfileMetadata field: write_h5 has no omega of its
        # own, and a downstream peak fit or pole figure needs one value per
        # profile, in the same order. Skipped rather than padded when the
        # lengths disagree — a misaligned angle is worse than none.
        extras = None
        if omegas is not None and len(omegas) == len(frame_ids):
            extras = {"omegas": np.asarray(omegas, dtype=np.float64)}
        m.write_h5(str(h5_path), profiles=profiles, r_axis=r_axis,
                   frame_ids=frame_ids, sigmas=sigmas,
                   extra_datasets=extras)
        try:
            entry = provenance.build_entry(
                'midas_gui.batch_integrate.save',
                extra={'n_frames': len(frame_ids),
                      'calibration_snapshot': calibration_snapshot,
                      'active_profile': settings.active_profile()})
            stamp_h5_provenance(h5_path, entry)
        except Exception:
            pass   # best-effort — a failed stamp shouldn't fail the save
        out_paths.append(str(h5_path))
    elif "h5" in fmts and len(frame_ids) and multi:
        if spec is None:
            pass   # no IntegrationSpec available — same silent skip as before
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            h5_path = out_dir / "integrated.h5"
            try:
                from midas_gui.cake_hdf5 import write_cake_h5
                write_cake_h5(
                    h5_path, cake=profiles, cake_sigma=sigmas, r_axis=r_axis,
                    eta_axis=eta_axis, frame_ids=frame_ids, spec=spec,
                    bin_area=bin_area, kernel=kernel, weighted=weighted,
                    cake_params=cake_params, omegas=omegas)
                entry = provenance.build_entry(
                    'midas_gui.batch_integrate.save',
                    extra={'n_frames': len(frame_ids), 'multi_azimuth': True,
                          'calibration_snapshot': calibration_snapshot,
                          'kernel': kernel, 'weighted': weighted,
                          'cake_params': cake_params,
                          'active_profile': settings.active_profile()})
                stamp_h5_provenance(h5_path, entry)
                out_paths.append(str(h5_path))
            except Exception:
                pass   # best-effort — a failed cake HDF5 write shouldn't fail the save
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
                 corrections, variance_cfg, q_cfg=None, omega_cfg=None,
                 frame_range=None, monitor_file=None, drift_traj=None,
                 dark=None, bright=None, background=None, bright_mode="divide",
                 weighted=True, context=None, im_trans=(), multi_azimuth=False,
                 run_mode="sequential", n_workers=1, parent=None,
                 calibration_snapshot=None, zarr_grouping="frame",
                 ion_csv_extras=()):
        super().__init__(parent)
        self._args = dict(
            spec=spec, source_cfg=source_cfg, mask=mask, out_dir=out_dir, fmts=fmts,
            kernel=kernel, corrections=corrections, variance_cfg=variance_cfg,
            q_cfg=q_cfg, omega_cfg=omega_cfg,
            frame_range=frame_range, monitor_file=monitor_file,
            drift_traj=drift_traj, dark=dark, bright=bright, background=background,
            bright_mode=bright_mode, weighted=weighted, im_trans=im_trans,
            multi_azimuth=multi_azimuth, calibration_snapshot=calibration_snapshot,
            zarr_grouping=zarr_grouping, ion_csv_extras=ion_csv_extras)
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
        # Several frames sharing one zarr archive must not be split across
        # workers — see _split_into_chunks_on_groups. "run" grouping yields a
        # single group and so a single chunk, which falls through to the
        # sequential path below rather than needing a case of its own.
        grouping = self._args.get("zarr_grouping", "frame")
        key_fn = (getattr(source, "zarr_group_key", None)
                  if grouping == "file" else
                  (lambda _i: "<run>") if grouping == "run" else None)
        if "zarr" in (self._args.get("fmts") or ()) and key_fn is not None:
            self._chunks = _split_into_chunks_on_groups(indices, n_workers, key_fn)
            if len(self._chunks) < n_workers:
                self.log_line.emit(
                    f"[batch] zarr grouping={grouping}: a group cannot be "
                    f"split across workers — using {len(self._chunks)} "
                    f"worker(s) rather than {n_workers}.")
            if len(self._chunks) <= 1:
                self._chunks = []
                self._start_sequential()
                return
        else:
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
                                "frame_ids": [], "omegas": [],
                                "out_paths": [], "aborted": True})
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
        merged_omegas = []
        for cw in self._chunk_workers:
            d = self._chunk_results[id(cw)]
            if d.get("profiles") is not None and len(d["profiles"]):
                merged_profiles.extend(d["profiles"])
            if d.get("sigmas") is not None and len(d["sigmas"]):
                merged_sigmas.extend(d["sigmas"])
            merged_ids.extend(d.get("frame_ids") or [])
            merged_omegas.extend(d.get("omegas") or [])
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
            try:
                spec = self._args["spec"]
                kernel = self._args["kernel"]
                bin_area = cake_params = None
                if multi_azimuth:
                    # Not carried over from any one chunk worker (each built its
                    # own geometry independently) — rebuilt once here so the
                    # combined cake HDF5 still gets a real, non-zero BinArea row.
                    geom = build_geom(spec, kernel, self._args["mask"])
                    bin_area = count_cake(geom, kernel, spec.NrPixelsZ, spec.NrPixelsY)
                    cake_params = {
                        'RMin': float(spec.RMin), 'RMax': float(spec.RMax),
                        'RBinSize': float(spec.RBinSize),
                        'EtaMin': float(spec.EtaMin), 'EtaMax': float(spec.EtaMax),
                        'EtaBinSize': float(spec.EtaBinSize),
                    }
                h5_paths = write_all_profiles(
                    out_dir, ["h5"], r_axis, merged_profiles, merged_sigmas, merged_ids,
                    float(spec.Lsd), float(spec.pxY), float(spec.Wavelength),
                    eta_axis=eta_axis, spec=spec, bin_area=bin_area,
                    calibration_snapshot=self._args.get("calibration_snapshot"),
                    kernel=kernel, weighted=self._args.get("weighted"),
                    cake_params=cake_params, omegas=merged_omegas)
                merged_out.extend(h5_paths)
            except Exception:
                self.log_line.emit(
                    "[batch] combined HDF5 write failed:\n" + traceback.format_exc())
        self.finished.emit({
            "n": len(merged_profiles), "r_axis_px": r_axis,
            "profiles": np.array(merged_profiles) if merged_profiles else np.array([]),
            "sigmas": np.array(merged_sigmas) if merged_sigmas else np.array([]),
            "frame_ids": merged_ids, "omegas": merged_omegas,
            "out_paths": merged_out, "aborted": aborted,
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
            q, I, sig = load_profile_file(path)
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
                q_bg, I_bg, sig_bg = load_profile_file(bg_path)
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
