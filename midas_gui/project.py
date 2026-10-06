"""FAIR provenance "project" files.

A project file is a single, long-lived HDF5 file with two top-level headers
(schema_version 3 — a clean-cutover redesign, no backward compatibility with
the single-``/workspace``-blob v1/v2 layout):

- ``/gui_workspace/<tab_name>/{state, sidecars/}`` — one group per GUI tab
  (e.g. ``"Data Viewer"``, ``"Mask Builder"``, ``"Calibrate"``), each holding
  that tab's most recently saved field state (JSON) plus any binary/text
  sidecars it exports (a live mask array, an unfitted calibration summary).
  Modular by design: a tab's snapshot can be read/restored independently of
  every other tab's. Overwritten in place on each save (mutable), unlike the
  append-only history below. See ``write_gui_workspace``/``read_workspace_tab``/
  ``list_workspace_tabs``/``read_workspace_meta``.
- ``/analysis/{mask,calibrate,integrate}/...`` — append-only FAIR-provenance
  records:
  - ``/analysis/mask/attempt_NNNN`` — one *global* history (Mask Builder is a
    single shared tab, not per-detector) of mask-creation attempts: full
    parameters + the resulting compressed mask array. See
    ``append_mask_attempt``/``list_mask_attempts``/``read_mask_attempt_array``.
  - ``/analysis/calibrate/<panel_key>/attempt_NNNN`` and
    ``/analysis/integrate/<panel_key>/attempt_NNNN`` — one per Calibrate or
    Batch-Integrate run that finishes (single-detector under ``single``,
    Hydra panels under ``ge1``..``ge4``, the Hydra composite under
    ``hydra_composite``). Each attempt group is self-sufficient: a JSON
    ``metadata`` blob (full params, full result, resolved input paths +
    hashes, environment/version snapshot) plus the resulting profile/cake
    arrays. Input correction data (mask/dark/bright/background) is never
    duplicated as raw arrays — only referenced by path + hash — except a
    live/drawn-in-tab mask that was never saved to a file, which has no path
    to hash and is embedded as-is. See ``append_calibration_attempt``/
    ``append_integration_attempt``.

Every mutating write (``write_gui_workspace`` and the three
``append_*_attempt`` functions) builds its new content in a sibling
"staging" child group first and only swaps it into place with a cheap
metadata-only rename (:func:`_stage_and_swap`) — a crash mid-write leaves
whatever was there before fully intact. ``write_gui_workspace`` additionally
makes a rolling ``path + ".bak"`` copy (:func:`backup_before_overwrite`)
before each overwrite. Saving As to a new file (``create_project`` +
:func:`copy_analysis_history`) lets the caller choose how much of the
*source* project's ``/analysis`` history — none, latest-only, or all — to
carry into the fresh destination file; see :func:`analysis_summary` for the
counts shown to the user before they choose.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from midas_gui import provenance

PROJECT_MARKER = "__midas_gui_project__"
SCHEMA_VERSION = 3

_HASH_FULL_MAX_BYTES = 200 * 1024 * 1024
_HASH_PARTIAL_CHUNK = 4 * 1024 * 1024


class ProjectContext:
    """Mutable holder for the currently-open project path, shared by
    reference between MainWindow and the tabs/pages that log attempts."""

    def __init__(self):
        self.path: Optional[str] = None


def backup_before_overwrite(path) -> None:
    """Best-effort single rolling backup (``path + ".bak"``), made just
    before a destructive rewrite of an existing project file. Never
    raises — a failed backup must not block the save/overwrite it exists to
    protect against mistakes for."""
    try:
        import shutil
        shutil.copy2(str(path), str(path) + ".bak")
    except Exception:
        pass


def _stage_and_swap(parent_grp, target_name: str, build_fn) -> None:
    """Build a new child group under ``parent_grp`` via
    ``build_fn(staging_group)``, then swap it into ``target_name`` with a
    metadata-only rename — not the previous delete-then-rebuild-in-place
    pattern. A crash during ``build_fn`` leaves any pre-existing
    ``target_name`` group fully intact and just orphans a harmless
    ``"_<target_name>_staging"`` group (invisible to every reader, which
    only ever looks up exact names like ``"gui_workspace"`` or
    ``attempt_NNNN`` — and automatically cleaned up the next time this same
    parent/name is written, by the check at the top of this function)."""
    staging_name = f"_{target_name}_staging"
    if staging_name in parent_grp:
        del parent_grp[staging_name]
    staging = parent_grp.create_group(staging_name)
    build_fn(staging)
    parent_grp.file.flush()
    if target_name in parent_grp:
        del parent_grp[target_name]
    parent_grp.move(staging_name, target_name)


def create_project(path, name: Optional[str] = None, *, overwrite: bool = False) -> str:
    path = str(path)
    if Path(path).exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists")
        backup_before_overwrite(path)
        Path(path).unlink()
    from midas_gui import settings
    with h5py.File(path, "w") as f:
        f.attrs[PROJECT_MARKER] = True
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["project_name"] = name or Path(path).stem
        f.attrs["created_utc"] = _now_iso()
        try:
            f.attrs["active_profile_at_creation"] = settings.active_profile()
        except Exception:
            pass
    return path


def open_project(path) -> str:
    path = str(path)
    with h5py.File(path, "r") as f:
        if not f.attrs.get(PROJECT_MARKER, False):
            raise ValueError(f"{path} is not a MIDAS GUI project file")
    return path


def project_active_profile(project_path) -> Optional[str]:
    """The beamline Profile active when this project was created (a stable,
    file-level attr — see ``create_project``), or ``None`` for a project
    file predating this feature."""
    with h5py.File(str(project_path), "r") as f:
        return f.attrs.get("active_profile_at_creation")


def project_schema_version(project_path) -> Optional[int]:
    """The recorded ``schema_version`` attr, or ``None`` if the file can't be
    read or isn't a MIDAS GUI project at all. Since schema_version 3 is a
    clean-cutover redesign (no backward compatibility — see the module
    docstring), callers use this to detect a pre-3 project file and warn
    rather than silently show an empty checkbox tree."""
    try:
        with h5py.File(str(project_path), "r") as f:
            if not f.attrs.get(PROJECT_MARKER, False):
                return None
            return int(f.attrs.get("schema_version", 0))
    except Exception:
        return None


def _write_sidecars(parent_grp, sidecars: dict) -> None:
    """Shared by ``write_gui_workspace``'s per-tab groups: write a
    ``sidecars`` child group holding not-yet-exported derived data a tab's
    own ``get_state(sidecar_stem=)`` wrote (e.g. a live/drawn mask array, an
    unfitted-but-in-progress calibration summary)."""
    side_grp = parent_grp.create_group("sidecars")
    for name, data in sidecars.items():
        if isinstance(data, (bytes, bytearray, str)):
            # h5py's VLEN string/bytes type rejects embedded NUL bytes
            # (common in binary sidecars like a mask .npy export), raising
            # "VLEN strings do not support embedded NULLs". Store as opaque
            # uint8 bytes instead, tagged so the reader can tell it apart
            # from a genuine uint8 array sidecar (e.g. a mask array) and
            # reconstruct the original str/bytes.
            is_str = isinstance(data, str)
            raw = data.encode("utf-8") if is_str else bytes(data)
            ds = side_grp.create_dataset(name, data=np.frombuffer(raw, dtype=np.uint8))
            ds.attrs["_midas_gui_encoding"] = "str" if is_str else "bytes"
        else:
            _write_array(side_grp, name, data)


def _read_sidecars(parent_grp) -> dict:
    side_grp = parent_grp.get("sidecars")
    if side_grp is None:
        return {}
    sidecars = {}
    for name in side_grp.keys():
        ds = side_grp[name]
        value = ds[()]
        encoding = ds.attrs.get("_midas_gui_encoding")
        if encoding == "bytes":
            value = value.tobytes()
        elif encoding == "str":
            value = value.tobytes().decode("utf-8")
        sidecars[name] = value
    return sidecars


def write_gui_workspace(project_path, *, tabs: dict, sidecars: Optional[dict] = None,
                         meta: Optional[dict] = None) -> None:
    """Overwrite the project's ``/gui_workspace`` group, one child group per
    tab in ``tabs`` (``{tab_name: state_dict}``) — modular by design, unlike
    the single mutable JSON blob the old ``/workspace`` slot held: a tab's
    snapshot can be read back independently via ``read_workspace_tab``
    without touching any other tab's. ``sidecars`` is
    ``{tab_name: {filename: data}}`` (see ``_write_sidecars``); ``meta`` is a
    flat dict of whole-session facts (``active_tab``/``active_profile``/
    ``saved_utc``) stored as attrs on the ``/gui_workspace`` group itself."""
    sidecars = sidecars or {}
    project_path = str(project_path)
    if Path(project_path).exists():
        backup_before_overwrite(project_path)

    def _build(root):
        root.attrs["saved_utc"] = _now_iso()
        for k, v in (meta or {}).items():
            root.attrs[k] = v if v is not None else ""
        for tab_name, state in tabs.items():
            g = root.create_group(tab_name)
            g.create_dataset("state", data=json.dumps(state, indent=2, default=_json_default))
            tab_sidecars = sidecars.get(tab_name)
            if tab_sidecars:
                _write_sidecars(g, tab_sidecars)

    with h5py.File(project_path, "a") as f:
        _stage_and_swap(f, "gui_workspace", _build)


def list_workspace_tabs(project_path) -> list:
    """Which tabs have a saved ``/gui_workspace`` snapshot in this project,
    in HDF5 iteration order. Empty for a brand-new project, or one with no
    saved session yet — not an error case."""
    with h5py.File(str(project_path), "r") as f:
        grp = f.get("gui_workspace")
        return list(grp.keys()) if grp is not None else []


def read_workspace_meta(project_path) -> dict:
    """The whole-session facts (``active_tab``/``active_profile``/
    ``saved_utc``) recorded alongside ``/gui_workspace``'s per-tab groups, or
    ``{}`` if no session has been saved yet."""
    with h5py.File(str(project_path), "r") as f:
        grp = f.get("gui_workspace")
        return dict(grp.attrs) if grp is not None else {}


def read_workspace_tab(project_path, tab_name: str) -> tuple:
    """One tab's saved snapshot, as ``(state, sidecars)`` — both ``{}`` if
    that tab has no saved snapshot (never saved, or saved before this tab
    existed). Not an error case: callers should treat an empty result as
    "nothing to restore for this tab," not a failure."""
    with h5py.File(str(project_path), "r") as f:
        grp = f.get(f"gui_workspace/{tab_name}")
        if grp is None:
            return {}, {}
        state = json.loads(grp["state"][()])
        return state, _read_sidecars(grp)


def sha256_file(path) -> dict:
    """Full SHA-256 for files under the size threshold; for larger files
    (e.g. multi-thousand-frame raw datasets) a fast head/tail fingerprint
    instead, so logging an attempt never stalls on hashing a huge file."""
    p = Path(path)
    size = p.stat().st_size
    if size <= _HASH_FULL_MAX_BYTES:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"method": "sha256_full", "sha256": h.hexdigest(), "size_bytes": size}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read(_HASH_PARTIAL_CHUNK))
        if size > _HASH_PARTIAL_CHUNK:
            fh.seek(max(size - _HASH_PARTIAL_CHUNK, 0))
            h.update(fh.read(_HASH_PARTIAL_CHUNK))
    return {
        "method": "sha256_partial",
        "sha256_head_tail": h.hexdigest(),
        "size_bytes": size,
        "mtime": p.stat().st_mtime,
    }


def _safe_version(module_name: str):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def _best_effort_git_commit(directory) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(directory),
            capture_output=True, text=True, timeout=2)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def environment_snapshot() -> dict:
    import platform
    import midas_gui

    env = {
        "midas_gui_version": getattr(midas_gui, "__version__", "unknown"),
        "midas_gui_git_commit": _best_effort_git_commit(Path(midas_gui.__file__).resolve().parent),
        "python_version": platform.python_version(),
    }
    for pkg in ("midas_calibrate_v2", "midas_integrate_v2", "midas_calibrate",
                "midas_hkls", "midas_distortion", "h5py", "numpy"):
        env[f"{pkg}_version"] = _safe_version(pkg)
    try:
        from PyQt5 import QtCore
        env["pyqt_version"] = QtCore.PYQT_VERSION_STR
        env["qt_version"] = QtCore.QT_VERSION_STR
    except Exception:
        pass
    env["workstation"] = workstation_snapshot()
    return env


def _cpu_model() -> Optional[str]:
    """Best-effort human-readable CPU model string (e.g. "Apple M2 Pro" or
    an Intel/AMD brand string). Platform-specific lookups are each wrapped
    individually — a failure just falls through to the next, least-specific
    source rather than losing the whole field."""
    import platform
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def workstation_snapshot() -> dict:
    """Best-effort hardware/OS identification for the machine an analysis
    attempt actually ran on, so a project file can still be traced back to
    the exact workstation it was produced on long after the fact. Every
    field is independently best-effort — a lookup failure yields ``None``
    for just that field rather than dropping the whole snapshot."""
    import platform

    ws = {
        "hostname": None,
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": None,
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": None,
        "total_memory_gb": None,
    }
    try:
        ws["hostname"] = socket.gethostname()
    except Exception:
        pass
    try:
        ws["processor"] = _cpu_model()
    except Exception:
        pass
    try:
        import psutil
        ws["cpu_count_physical"] = psutil.cpu_count(logical=False)
        ws["total_memory_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        pass
    return ws


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(obj):
    """``json.dumps(..., default=...)`` handler for the numpy/Path/set values
    that show up in calibration results and other GUI state — shared with
    ``cake_hdf5.py`` so a stamped calibration snapshot uses the same
    JSON-safety guarantee as project files."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


_json_default = json_default   # back-compat alias for existing call sites


def _hash_paths_in(obj):
    """Recursively walk a JSON-like structure; any dict with a string
    'path' key pointing to an existing file gets a sibling 'path_hash' key.
    Returns a new structure (does not mutate the input)."""
    if isinstance(obj, dict):
        out = {k: _hash_paths_in(v) for k, v in obj.items()}
        p = obj.get("path")
        if isinstance(p, str) and p and Path(p).is_file():
            try:
                out["path_hash"] = sha256_file(p)
            except Exception as e:
                out["path_hash"] = {"error": str(e)}
        return out
    if isinstance(obj, list):
        return [_hash_paths_in(v) for v in obj]
    return obj


def sanitize_result_dict(result) -> Optional[dict]:
    """Full result fields, including the underscore-prefixed extras the
    GUI bolts on (_calibrant_name, _panel_unpacked, ...) — unlike the
    existing GUI-state sidecar JSON, nothing is dropped here except
    torch-tensor fields (duck-typed via a '.numpy' attribute), which are
    large and already referenced separately (residual_corr_bin_path)."""
    if result is None:
        return None
    return {k: v for k, v in vars(result).items() if not hasattr(v, "numpy")}


def _write_array(group, name, arr):
    arr = np.asarray(arr)
    kwargs = {}
    if arr.ndim > 0 and arr.size > 0:
        kwargs = dict(compression="gzip", compression_opts=4, chunks=True)
    group.create_dataset(name, data=arr, **kwargs)


def _next_attempt_name(group) -> str:
    existing = [k for k in group.keys() if k.startswith("attempt_")]
    return f"attempt_{len(existing) + 1:04d}"


def _panel_shifts_array(panel_u: Optional[dict]) -> Optional[np.ndarray]:
    """``(N, 6)`` array ``[panel_id, dy, dz, dtheta, dlsd, dp2]`` from a
    calibration result's ``_panel_unpacked`` dict (raw refined-shift
    tensors) — same column layout
    ``midas_calibrate_v2.compat.to_v1.write_panel_shifts_file`` writes, so
    :func:`materialize_panel_shifts` can round-trip through that function
    unmodified. Returns ``None`` when there is no panel data to embed."""
    if not panel_u:
        return None
    dyz = panel_u.get("panel_delta_yz")
    dth = panel_u.get("panel_delta_theta")
    if dyz is None or dth is None:
        return None

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    dyz, dth = _np(dyz), _np(dth)
    n = dyz.shape[0]
    dl = _np(panel_u["panel_delta_lsd"]) if panel_u.get("panel_delta_lsd") is not None else np.zeros(n)
    dp2 = _np(panel_u["panel_delta_p2"]) if panel_u.get("panel_delta_p2") is not None else np.zeros(n)
    return np.column_stack([np.arange(n), dyz[:, 0], dyz[:, 1], dth, dl, dp2])


def materialize_panel_shifts(project_path, ref: str, arr) -> Optional[str]:
    """Write an embedded panel-shifts array (see :func:`_panel_shifts_array`)
    back out to a real ``<attempt>_panelshifts.txt`` file next to the
    project, so a reopened project's multi-panel calibration keeps
    integrating correctly even though the file the *live* Fit run wrote (an
    Output-folder path, or an anonymous tempfile — see
    ``calib._attach_panel_result``) may be long gone or on a different
    machine. Returns the written path, or ``None`` if ``arr`` is empty."""
    if arr is None or len(arr) == 0:
        return None
    from midas_calibrate_v2.compat.to_v1 import write_panel_shifts_file
    arr = np.asarray(arr)
    unpacked = {
        "panel_delta_yz": arr[:, 1:3],
        "panel_delta_theta": arr[:, 3],
        "panel_delta_lsd": arr[:, 4],
        "panel_delta_p2": arr[:, 5],
    }
    project_path = Path(project_path)
    out_dir = project_path.with_name(project_path.stem + "_panel_shifts")
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt_name = ref.rstrip("/").rsplit("/", 1)[-1]
    out_path = out_dir / f"{attempt_name}_panelshifts.txt"
    write_panel_shifts_file(unpacked, out_path)
    return str(out_path)


def read_calib_attempt_panel_shifts(project_path, ref: str) -> Optional[np.ndarray]:
    """The raw ``(N, 6)`` panel-shifts array embedded by
    :func:`append_calibration_attempt` (see :func:`_panel_shifts_array` for
    the column layout), or ``None`` if this attempt has none (a
    single-panel calibration, or one logged before this feature existed)."""
    with h5py.File(project_path, "r") as f:
        grp = f.get(ref.lstrip("/"))
        if grp is None or "panel_shifts" not in grp:
            return None
        return grp["panel_shifts"][()]


def append_calibration_attempt(project_path, panel_key, *, cfg, result, loader_state,
                                mask_is_file_backed: bool = False,
                                results: Optional[dict] = None,
                                extra: Optional[dict] = None) -> str:
    cfg_copy = dict(cfg or {})
    mask = cfg_copy.pop("mask", None)
    embed_mask = mask is not None and not mask_is_file_backed
    panel_arr = _panel_shifts_array(getattr(result, "_panel_unpacked", None))

    metadata = {
        "timestamp_utc": _now_iso(),
        "panel_key": panel_key,
        "cfg": _hash_paths_in(cfg_copy),
        "result": sanitize_result_dict(result),
        "loader_state": _hash_paths_in(loader_state or {}),
        "environment": environment_snapshot(),
        "mask_present": mask is not None,
        "mask_embedded": embed_mask,
        "panel_shifts_embedded": panel_arr is not None,
    }
    if extra:
        metadata.update(extra)

    def _build(att):
        att.create_dataset("metadata", data=json.dumps(metadata, indent=2, default=_json_default))
        att.attrs["timestamp_utc"] = metadata["timestamp_utc"]
        att.attrs["pipeline"] = str(cfg_copy.get("mode") or "")
        # provenance_history: same convention/schema as the new zarr cake
        # output (see midas_gui/provenance.py) — lets tooling from the
        # mpe_wf_saxs_waxs convergence read either project's outputs the
        # same way. Complements, doesn't replace, the metadata/
        # environment_snapshot above (that's MIDAS_GUI's own richer
        # per-attempt record; this is the cross-project-compatible one).
        try:
            entry = provenance.build_entry(
                'midas_gui.calibrate',
                extra={'panel_key': panel_key, 'pipeline': cfg_copy.get('mode')})
            provenance.append_to_hdf5_attrs(att, entry)
        except Exception:
            pass   # best-effort — never block saving the calibration attempt
        for k in ("Lsd", "BC_y", "BC_z"):
            v = getattr(result, k, None)
            if v is not None:
                att.attrs[k] = float(v)
        if embed_mask:
            _write_array(att, "mask", mask)
        if panel_arr is not None:
            _write_array(att, "panel_shifts", panel_arr)
        if results:
            res_grp = att.create_group("results")
            for key in ("profile", "r_axis_px", "cake_2d", "eta_axis_deg", "resid_cake"):
                if results.get(key) is not None:
                    _write_array(res_grp, key, results[key])
            for k in ("lsd_um", "px_um", "wavelength_A"):
                if results.get(k) is not None:
                    res_grp.attrs[k] = float(results[k])

    with h5py.File(project_path, "a") as f:
        grp = f.require_group(f"analysis/calibrate/{panel_key}")
        name = _next_attempt_name(grp)
        _stage_and_swap(grp, name, _build)
        grp.attrs["latest"] = name

    return f"/analysis/calibrate/{panel_key}/{name}"


def append_mask_attempt(project_path, *, cfg, mask, loader_state,
                         extra: Optional[dict] = None) -> str:
    """Append a global (not per-panel — Mask Builder is one shared tab, not
    per-detector) FAIR-provenance record of a mask-creation attempt: the
    full parameter set (``cfg``, typically ``{"fields": widgets_to_dict(...)}``)
    and resolved source (``loader_state``, hashed like every other attempt's
    input paths) alongside the resulting compressed mask array itself. See
    ``list_mask_attempts``/``read_mask_attempt_array`` for the read side, and
    ``MaskTab.apply_project_mask`` for how a recorded attempt repopulates the
    tab it came from."""
    metadata = {
        "timestamp_utc": _now_iso(),
        "cfg": _hash_paths_in(dict(cfg or {})),
        "loader_state": _hash_paths_in(dict(loader_state or {})),
        "environment": environment_snapshot(),
    }
    if extra:
        metadata.update(extra)

    def _build(att):
        att.create_dataset("metadata", data=json.dumps(metadata, indent=2, default=_json_default))
        att.attrs["timestamp_utc"] = metadata["timestamp_utc"]
        _write_array(att, "mask", mask)

    with h5py.File(project_path, "a") as f:
        grp = f.require_group("analysis/mask")
        name = _next_attempt_name(grp)
        _stage_and_swap(grp, name, _build)
        grp.attrs["latest"] = name

    return f"/analysis/mask/{name}"


def list_mask_attempts(project_path) -> list:
    """Global mask-creation attempts recorded under ``/analysis/mask``,
    newest first — same entry shape as :func:`list_attempts` (``name``/
    ``ref``/``timestamp_utc``), just with no ``panel_key`` axis."""
    with h5py.File(str(project_path), "r") as f:
        grp = f.get("analysis/mask")
        if grp is None:
            return []
        names = sorted((k for k in grp.keys() if k.startswith("attempt_")), reverse=True)
        return [{"name": n, "ref": f"/analysis/mask/{n}",
                 "timestamp_utc": grp[n].attrs.get("timestamp_utc", "")}
                for n in names]


def read_mask_attempt_array(project_path, ref: str) -> Optional[np.ndarray]:
    """The embedded mask array for a ``/analysis/mask`` attempt, or ``None``
    if the ref has no ``mask`` dataset (shouldn't happen for a genuine mask
    attempt — every one embeds its mask — but kept defensive like
    :func:`read_calib_attempt_panel_shifts`)."""
    with h5py.File(str(project_path), "r") as f:
        grp = f.get(ref.lstrip("/"))
        if grp is None or "mask" not in grp:
            return None
        return grp["mask"][()]


_PANEL_ORDER = ("single", "ge1", "ge2", "ge3", "ge4", "hydra_composite")


_KIND_DIRS = {"calib": "calibrate", "integrate": "integrate"}


def discover_panels(project_path) -> list:
    """Which of the canonical panel groups (single-detector or the 4 Hydra
    GE panels) have at least one recorded calibrate or integrate attempt, in
    canonical order."""
    with h5py.File(str(project_path), "r") as f:
        calib = set(f.get("analysis/calibrate", {}).keys())
        integrate = set(f.get("analysis/integrate", {}).keys())
        return [p for p in _PANEL_ORDER if p in calib or p in integrate]


def list_attempts(project_path, panel_key: str, kind: str) -> list:
    """Attempts recorded under ``analysis/<calibrate|integrate>/<panel_key>``
    (``kind`` is ``"calib"`` or ``"integrate"``), newest first. Each entry is
    enough to populate a picker without parsing the (possibly large)
    ``metadata`` JSON blob."""
    kind_dir = _KIND_DIRS[kind]
    with h5py.File(str(project_path), "r") as f:
        grp = f.get(f"analysis/{kind_dir}/{panel_key}")
        if grp is None:
            return []
        names = sorted((k for k in grp.keys() if k.startswith("attempt_")), reverse=True)
        return [{"name": n, "ref": f"/analysis/{kind_dir}/{panel_key}/{n}",
                 "timestamp_utc": grp[n].attrs.get("timestamp_utc", "")}
                for n in names]


def read_attempt(project_path, ref: str) -> dict:
    """Parsed ``metadata`` JSON for one attempt, given a ref such as
    ``/analysis/calibrate/ge1/attempt_0003`` (as returned by
    ``append_*_attempt`` / ``list_attempts``/``list_mask_attempts``)."""
    with h5py.File(str(project_path), "r") as f:
        return json.loads(f[ref.lstrip("/")]["metadata"][()])


def read_attempt_results(project_path, ref: str) -> dict:
    """The embedded 1-D result arrays for an *integration* attempt
    (``profiles``/``r_axis_px``/``sigmas``/``omegas``/``frame_ids``) — these live as
    raw HDF5 datasets under ``<ref>/results``, separate from the JSON
    ``metadata`` blob ``read_attempt`` returns (see
    ``append_integration_attempt``). Returns {} if the attempt has no
    ``results`` group (e.g. a calibration attempt, or an aborted run with
    zero frames)."""
    with h5py.File(str(project_path), "r") as f:
        grp = f.get(f"{ref.lstrip('/')}/results")
        if grp is None:
            return {}
        out = {}
        for key in ("profiles", "r_axis_px", "sigmas", "omegas"):
            if key in grp:
                out[key] = grp[key][()]
        if "frame_ids" in grp:
            out["frame_ids"] = [v.decode() if isinstance(v, bytes) else v
                                 for v in grp["frame_ids"][()]]
        return out


def read_calib_attempt_results(project_path, ref: str) -> dict:
    """The embedded cake/profile arrays for a *calibration* attempt (see
    ``append_calibration_attempt``'s ``results`` kwarg) — ``profile``,
    ``r_axis_px``, ``cake_2d``, ``eta_axis_deg``, ``resid_cake`` plus scalar
    ``lsd_um``/``px_um``/``wavelength_A``. Returns {} if the attempt has no
    ``results`` group (e.g. an attempt logged before this feature existed,
    or integration never ran for it)."""
    with h5py.File(str(project_path), "r") as f:
        grp = f.get(f"{ref.lstrip('/')}/results")
        if grp is None:
            return {}
        out = {}
        for key in ("profile", "r_axis_px", "cake_2d", "eta_axis_deg", "resid_cake"):
            if key in grp:
                out[key] = grp[key][()]
        for k in ("lsd_um", "px_um", "wavelength_A"):
            if k in grp.attrs:
                out[k] = float(grp.attrs[k])
        return out


def calib_attempt_gui_fields(meta: dict) -> dict:
    """Map a parsed calibration-attempt ``metadata`` dict to the
    widget-keyed field dict consumed by ``CalibrationTab``/
    ``HydraCalibrationPage``/``HydraCalibPanelCard``'s own
    ``_state_widgets()``/``state_widgets()`` (applied via
    ``helpers.apply_dict_to_widgets``). The single-detector tab, the Hydra
    page's shared "recipe" fields, and one Hydra panel card's seed fields are
    three different (non-overlapping) subsets of the same widget-key
    vocabulary, so one dict can be handed to all three — each just ignores
    the keys it doesn't define."""
    cfg = meta.get("cfg") or {}
    result = meta.get("result") or {}
    refine = cfg.get("refine") or {}
    im_trans = set(cfg.get("im_trans") or [])
    fields = {
        "wl": cfg.get("wavelength"),
        "cal": cfg.get("calibrant"),
        "pxY": cfg.get("pxY"),
        "flip_y": 1 in im_trans, "flip_z": 2 in im_trans, "transp": 3 in im_trans,
        "manual_seed_check": True,
        "seed_bcy": result.get("BC_y"), "seed_bcz": result.get("BC_z"),
        "seed_tx": result.get("tx", 0.0), "seed_ty": result.get("ty", 0.0),
        "seed_tz": result.get("tz", 0.0),
        "ref_lsd": refine.get("Lsd"), "ref_bc": refine.get("BC"),
        "ref_ty": refine.get("ty"), "ref_tz": refine.get("tz"), "ref_tx": refine.get("tx"),
        "ref_wl": refine.get("Wavelength"), "ref_dist": refine.get("Distortion"),
        "build_rc": cfg.get("build_residual_corr"),
        "n_iter": cfg.get("n_iter"), "lm_iter": cfg.get("lm_max_iter"),
        "device": cfg.get("device"),
    }
    lsd = result.get("Lsd")
    if lsd is not None:
        fields["seed_lsd"] = float(lsd) / 1000.0   # µm (stored) -> mm (display)
    pxZ = cfg.get("pxZ")
    if pxZ is not None:
        fields["pxZ_check"] = True
        fields["pxZ_spin"] = pxZ
    return {k: v for k, v in fields.items() if v is not None}


def calib_attempt_dist_coeffs(meta: dict) -> Optional[list]:
    """The attempt's per-coefficient distortion selection, as sorted v2
    harmonic names — the companion to ``calib_attempt_gui_fields``'
    ``ref_dist``, which is only the on/off tick and says nothing about which
    of the fifteen harmonics were free. Recorded as a set by
    ``CalibrationTab._refine_flags`` and written out sorted by
    ``_json_default``. ``None`` — not ``[]`` — when the attempt predates the
    per-coefficient selector, so the caller leaves its own default in place
    rather than restoring a fit that refines nothing."""
    coeffs = ((meta.get("cfg") or {}).get("refine") or {}).get("distortion_coeffs")
    if coeffs is None:
        return None
    return sorted(str(c) for c in coeffs)


def calib_attempt_loader_state(meta: dict) -> dict:
    """The subset of a calibration attempt's ``loader_state`` that
    ``DataLoaderPanel.set_state()`` understands (single-detector mode)."""
    ls = meta.get("loader_state") or {}
    out = {}
    if ls.get("path"):
        out["path"] = ls["path"]
    if ls.get("dataset"):
        out["dataset"] = ls["dataset"]
    if ls.get("frame_index") is not None:
        out["frame_index"] = ls["frame_index"]
    return out


def integrate_attempt_gui_fields(meta: dict) -> dict:
    """Map a parsed integration-attempt ``metadata`` dict to the
    widget-keyed field dict consumed by ``BatchTab``/``HydraBatchPage``'s
    ``_state_widgets()`` (applied via ``helpers.apply_dict_to_widgets``).
    Combo-box fields (kernel, output format) are stored by their short key
    but the widgets are populated by their display label, so both are
    translated via ``constants.KERNELS``/``constants.OUTPUT_FORMATS``."""
    from midas_gui.constants import KERNELS, OUTPUT_FORMATS

    inputs = meta.get("inputs") or {}
    fields = {}
    kernel_label = {v: k for k, v in KERNELS.items()}.get(inputs.get("kernel"))
    if kernel_label:
        fields["kernel"] = kernel_label
    # "fmt" is a list[str] of OUTPUT_FORMATS keys as of the checkbox-list
    # output-format selector; older project files recorded a single string —
    # wrap it so both shapes feed widgets.OutputFormatSelector.set_state the
    # same way.
    fmt_val = inputs.get("fmt")
    if fmt_val:
        fmt_keys = fmt_val if isinstance(fmt_val, list) else [fmt_val]
        valid_keys = set(OUTPUT_FORMATS.values())
        fmt_keys = [k for k in fmt_keys if k in valid_keys]
        if fmt_keys:
            fields["fmt_keys"] = fmt_keys
    # Stored by its short key; the combo is populated by display label, so
    # translate the same way kernel/fmt do above. Absent (every project saved
    # before zarr grouping existed) leaves the widget at its "frame" default,
    # which is the behaviour those attempts actually ran under.
    zg_label = {"frame": "One zarr per output frame",
                "file": "One zarr per source file",
                "run": "One zarr for the whole run"}.get(inputs.get("zarr_grouping"))
    if zg_label:
        fields["zarr_grouping"] = zg_label
    # Beam-monitor CSV column groups. Absent in every project saved before
    # the CSV existed, which correctly leaves both boxes unticked — those
    # attempts ran with the default columns only.
    ion_extras = inputs.get("ion_csv_extras")
    if ion_extras is not None:
        ion_extras = set(ion_extras)
        fields["ion_env_chk"] = "env" in ion_extras
        fields["ion_motors_chk"] = "motors" in ion_extras
    if inputs.get("monitor_file"):
        fields["mon_ed"] = inputs["monitor_file"]
    q_cfg = inputs.get("q_cfg")
    if q_cfg:
        fields["q_check"] = True
        if q_cfg.get("QMin") is not None:
            fields["q_min"] = q_cfg["QMin"]
        if q_cfg.get("QMax") is not None:
            fields["q_max"] = q_cfg["QMax"]
        if q_cfg.get("QBinSize") is not None:
            fields["q_bin"] = q_cfg["QBinSize"]
    return fields


def integrate_attempt_loader_state(meta: dict) -> dict:
    """The subset of an integration attempt's ``inputs`` that
    ``DataLoaderPanel.set_state()`` (stream mode) understands: path/dataset
    plus the frame range as ``fr_start``/``fr_end``/``fr_stride``, and
    "Combine sub-frames" as ``combine_chunk``/``combine_op``.

    A ``unify_combine`` panel's (Batch Integrate's) ``src_cfg`` carries the
    real start/end filter directly as ``frame_start``/``frame_end`` — its
    ``inputs["frame_range"]`` is always ``(0, None, 1)`` (see
    ``widgets.DataLoaderPanel.frame_range``), so those keys are preferred
    when present. Falls back to parsing the legacy ``frame_range`` tuple for
    projects saved before this existed."""
    inputs = meta.get("inputs") or {}
    src = inputs.get("src_cfg") or {}
    out = {}
    if src.get("path"):
        out["path"] = src["path"]
    if src.get("dataset"):
        out["dataset"] = src["dataset"]
    if "frame_start" in src or "frame_end" in src:
        if src.get("frame_start") is not None:
            out["fr_start"] = src["frame_start"]
        out["fr_end"] = src["frame_end"] if src.get("frame_end") is not None else 0
    else:
        frame_range = inputs.get("frame_range")
        if frame_range:
            start, end, stride = (list(frame_range) + [None, None, None])[:3]
            if start is not None:
                out["fr_start"] = start
            out["fr_end"] = end if end is not None else 0
            if stride is not None:
                out["fr_stride"] = stride
    if src.get("chunk_size") is not None:
        out["combine_chunk"] = src["chunk_size"]
    if src.get("combine_op"):
        out["combine_op"] = src["combine_op"]
    return out


def calibration_namespace(calibration_snapshot: dict):
    """Turn a stored ``calibration_snapshot`` (or a calibration attempt's
    ``result`` dict) into a duck-typed object with the attributes
    ``BatchTab.set_calibration``/``HydraBatchPanelCard.set_calibration`` and
    the integration-spec builders expect — mirrors
    ``helpers.result_ns_from_geometry_file``'s shape."""
    from types import SimpleNamespace
    snap = dict(calibration_snapshot or {})
    snap.setdefault("residual_corr_bin_path", None)
    return SimpleNamespace(**snap)


def append_integration_attempt(project_path, panel_key, *, inputs, finished_payload,
                                calibration_snapshot=None, calib_attempt_ref=None,
                                mask=None, mask_is_file_backed: bool = False,
                                extra: Optional[dict] = None,
                                environment: Optional[dict] = None) -> str:
    """Append one integration attempt to ``/analysis/integrate/<panel_key>``.

    ``environment`` lets a caller supply a pre-computed
    :func:`environment_snapshot`; omitted (the default), one is taken here as
    it always was. It exists for callers that log many attempts in a row — the
    Batch Queue writes one project per sample — because a snapshot shells out
    to ``git rev-parse`` and (on macOS) ``sysctl``, each with a 2 s timeout, so
    paying for it once per run rather than once per sample is worth the kwarg.
    The environment cannot meaningfully change mid-run."""
    payload = dict(finished_payload or {})
    profiles = payload.pop("profiles", None)
    r_axis = payload.pop("r_axis_px", None)
    sigmas = payload.pop("sigmas", None)
    frame_ids = payload.pop("frame_ids", None)
    omegas = payload.pop("omegas", None)
    embed_mask = mask is not None and not mask_is_file_backed

    metadata = {
        "timestamp_utc": _now_iso(),
        "panel_key": panel_key,
        "inputs": _hash_paths_in(inputs or {}),
        "n_frames": payload.get("n"),
        "out_paths": payload.get("out_paths"),
        "aborted": payload.get("aborted", False),
        "calibration_snapshot": calibration_snapshot,
        "calib_attempt_ref": calib_attempt_ref,
        "environment": environment if environment is not None else environment_snapshot(),
        "mask_present": mask is not None,
        "mask_embedded": embed_mask,
    }
    if extra:
        metadata.update(extra)

    def _build(att):
        att.create_dataset("metadata", data=json.dumps(metadata, indent=2, default=_json_default))
        att.attrs["timestamp_utc"] = metadata["timestamp_utc"]
        if metadata["n_frames"] is not None:
            att.attrs["n_frames"] = int(metadata["n_frames"])
        if inputs and inputs.get("kernel"):
            att.attrs["kernel"] = str(inputs["kernel"])
        if calib_attempt_ref:
            att.attrs["calib_attempt_ref"] = calib_attempt_ref
        if embed_mask:
            _write_array(att, "mask", mask)

        res_grp = att.create_group("results")
        if profiles is not None:
            _write_array(res_grp, "profiles", profiles)
        if r_axis is not None:
            _write_array(res_grp, "r_axis_px", r_axis)
        if sigmas is not None:
            _write_array(res_grp, "sigmas", sigmas)
        if frame_ids is not None:
            res_grp.create_dataset("frame_ids", data=np.array(list(frame_ids), dtype=h5py.string_dtype()))
        # Stored rather than recomputed at export time: a measured omega
        # channel is read off the frames, and the export path never reopens
        # them. See gsas_export.export_gsas_zarr.
        if omegas is not None and len(omegas):
            _write_array(res_grp, "omegas", np.asarray(omegas, dtype=np.float64))

    with h5py.File(project_path, "a") as f:
        grp = f.require_group(f"analysis/integrate/{panel_key}")
        name = _next_attempt_name(grp)
        _stage_and_swap(grp, name, _build)
        grp.attrs["latest"] = name

    return f"/analysis/integrate/{panel_key}/{name}"


def analysis_summary(project_path) -> dict:
    """Attempt counts per kind/panel: ``{"mask": n, "calibrate": {panel:
    n}, "integrate": {panel: n}}``. Empty kinds/panels are omitted, so a
    caller can cheaply check ``bool(analysis_summary(path))`` for "does
    this project have any provenance to offer copying" — used by both the
    Save-As history-scope dialog and :func:`copy_analysis_history`."""
    summary = {}
    mask_n = len(list_mask_attempts(project_path))
    if mask_n:
        summary["mask"] = mask_n
    panels = discover_panels(project_path)
    for kind, key in (("calib", "calibrate"), ("integrate", "integrate")):
        per_panel = {}
        for panel in panels:
            n = len(list_attempts(project_path, panel, kind))
            if n:
                per_panel[panel] = n
        if per_panel:
            summary[key] = per_panel
    return summary


def copy_analysis_history(src_path, dest_path, scope: str = "all") -> dict:
    """Copy ``/analysis`` attempt history from ``src_path`` (a
    currently-open project) into ``dest_path`` on a Save-As — used only
    when ``dest_path`` was just created empty (see ``create_project``), so
    copied attempts always keep their original names verbatim: no
    collision, no renumbering, and (for "latest") no ``calib_attempt_ref``
    rewriting is ever needed since the referenced calibrate attempt is
    included under the same name it already has.

    ``scope``:
      - ``"none"``   -- no-op, returns ``{}``.
      - ``"all"``    -- every attempt, every kind/panel, plus the mask
        history.
      - ``"latest"`` -- each kind/panel's ``"latest"`` attempt only, plus
        (for integrate) whichever calibrate attempt it references — even if
        that isn't itself the calibrate panel's own latest — so a copied
        integrate attempt is never left with a dangling
        ``calib_attempt_ref``.

    Returns the same shape as :func:`analysis_summary`, but counting only
    what was actually copied.
    """
    if scope == "none":
        return {}
    if scope not in ("all", "latest"):
        raise ValueError(f"unknown scope: {scope!r}")

    copied = {}
    with h5py.File(str(src_path), "r") as src, h5py.File(str(dest_path), "a") as dest:
        src_mask = src.get("analysis/mask")
        if src_mask is not None:
            names = sorted(k for k in src_mask.keys() if k.startswith("attempt_"))
            if scope == "latest":
                latest = src_mask.attrs.get("latest")
                if latest and latest in src_mask:
                    names = [latest]
                elif names:
                    names = names[-1:]
                else:
                    names = []
            if names:
                dest_mask = dest.require_group("analysis/mask")
                for n in names:
                    src.copy(src_mask[n], dest_mask, name=n)
                latest = src_mask.attrs.get("latest")
                dest_mask.attrs["latest"] = latest if latest in names else names[-1]
                copied["mask"] = len(names)

        panels = discover_panels(src_path)

        # "latest" scope: whichever calibrate attempts a to-be-copied
        # "latest" integrate attempt references, so it never ends up
        # dangling even if that calibrate attempt isn't itself that panel's
        # own latest.
        forced_calib = {}
        if scope == "latest":
            for panel in panels:
                src_int = src.get(f"analysis/integrate/{panel}")
                if src_int is None:
                    continue
                latest = src_int.attrs.get("latest")
                if not latest or latest not in src_int:
                    continue
                ref = src_int[latest].attrs.get("calib_attempt_ref")
                if not ref:
                    continue
                parts = ref.lstrip("/").split("/")
                if len(parts) >= 4 and parts[0] == "analysis" and parts[1] == "calibrate":
                    forced_calib.setdefault(parts[2], set()).add(parts[3])

        for kind, key in (("calibrate", "calibrate"), ("integrate", "integrate")):
            panel_counts = {}
            for panel in panels:
                src_grp = src.get(f"analysis/{kind}/{panel}")
                if src_grp is None:
                    continue
                names = sorted(k for k in src_grp.keys() if k.startswith("attempt_"))
                if not names:
                    continue
                if scope == "latest":
                    latest = src_grp.attrs.get("latest")
                    chosen = {latest} if latest and latest in src_grp else set(names[-1:])
                    if kind == "calibrate":
                        chosen |= forced_calib.get(panel, set())
                    names = sorted(chosen & set(names))
                if not names:
                    continue
                dest_grp = dest.require_group(f"analysis/{kind}/{panel}")
                for n in names:
                    src.copy(src_grp[n], dest_grp, name=n)
                latest = src_grp.attrs.get("latest")
                dest_grp.attrs["latest"] = latest if latest in names else names[-1]
                panel_counts[panel] = len(names)
            if panel_counts:
                copied[key] = panel_counts

        dest.attrs["forked_from_path"] = str(src_path)
        dest.attrs["forked_from_project_name"] = src.attrs.get("project_name", "")
        dest.attrs["forked_from_utc"] = _now_iso()
        dest.attrs["fork_scope"] = scope

    return copied
