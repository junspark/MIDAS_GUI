"""Per-user configuration for the GUI (pure stdlib — no Qt, no constants).

Two layers, higher wins:

    active profile file  →  built-in constants

The built-in layer lives in :mod:`midas_gui.constants`, which applies the config
returned by :func:`load_config` to its own module globals at import time (and,
on a profile switch, via :func:`midas_gui.constants.reload_from_config`). This
module imports nothing from the package (no import cycle; works before a
``QApplication`` exists). Config files are JSON; a missing or malformed file is
ignored rather than fatal.

Profiles let a user keep several named sets of defaults (e.g. per beamline) and
switch between them from Preferences ▸ Profile. On disk::

    <config dir>/
        profile_meta.json      {"active": "<name>", "bundled_seeded": ["..."]}
        profiles/
            Default.json
            20-ID-D.json, 20-ID-E.json, 1-ID-E.json   (bundled beamline profiles)
            <name>.json ...

A user upgrading from a version with no profile support has their old single
``config.json`` transparently migrated into ``profiles/Default.json`` the first
time this module touches the filesystem — see :func:`_ensure_profiles`. That
same function also seeds the bundled beamline profiles in
:data:`BUNDLED_PROFILES` (each supplying its own beamline-specific ``devices``
list, e.g. for the Data Viewer's Live Data PV dropdown) the first time it ever
runs on a given machine; deleting a bundled profile does not bring it back.

Sharing between users is done by exporting/importing a JSON file
(Preferences ▸ Save/Load config), not by any shared-file mechanism here.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

APP_DIRNAME = "midas_gui"
CONFIG_FILENAME = "config.json"          # legacy single-profile file (pre-migration)
PROFILES_DIRNAME = "profiles"
META_FILENAME = "profile_meta.json"
DEFAULT_PROFILE = "Default"
RECENT_FILENAME = "recent_files.json"    # global, NOT per-profile (see record_recent)
RECENT_KINDS = ("project", "workspace")
RECENT_MAX = 10

_cache: Optional[dict] = None


def _pva(name: str, prefix: str) -> dict:
    return {"name": name, "prefix": prefix, "backend": "pva", "pva_suffix": "Pva1:Image"}


def _ca(name: str, prefix: str, ca_suffix: str = "image1:") -> dict:
    """A device read over plain EPICS Channel Access via an areaDetector
    NDPluginStdArrays plugin (conventionally named "image1:"), for
    beamlines whose IOC has no NDPluginPva -- see CaLiveSource in
    midas_gui.live_sources."""
    return {"name": name, "prefix": prefix, "backend": "ca", "ca_suffix": ca_suffix}


# Bundled per-beamline device lists (Data Viewer ▸ Live Data PV dropdown), seeded
# once per machine by :func:`_ensure_profiles` so a user can switch beamlines from
# Preferences ▸ Profile instead of hand-editing devices. Extracted from each
# beamline's B-PILOT area-detector blueprint (``instrument/devices/<bl>_devices/
# *_area_detectors.py``, the ``make_det(...)`` calls with ``pva1_exists=True``).
# This module intentionally has no package imports (see module docstring), so
# these lists are independent literals rather than a shared reference with
# ``constants.DEFAULT_DEVICES`` — keep 20-ID-D's list here in sync with that one
# if it ever changes.
BUNDLED_PROFILES = {
    "20-ID-D": {"devices": [
        _pva("20iddNF", "20idOR1:"),
        _pva("s20idPil", "20idPil:"),
        _pva("pg4", "1idPG4:"),
        _pva("20iddTomo", "20idGH1s:"),
        _pva("20iddFF", "20IDFF:"),
        _pva("Sim Detector", "midasSim:"),
    ]},
    "20-ID-E": {"devices": [
        _pva("pimega", "PITEC:D:RAD1_5Mh:"),
        _pva("spl1", "20idsp1:"),
        _pva("s20varex2", "20idVarex2:"),
        _pva("pg6", "20idPG6s:"),
        _pva("gh2", "20idGH2S:"),
        _pva("Sim Detector", "midasSim:"),
    ]},
    "1-ID-E": {"devices": [
        _pva("ge1", "GE1:"),
        _pva("ge2", "GE2:"),
        _pva("ge3", "GE3:"),
        _pva("ge4", "GE4:"),
        _pva("ge5", "GE5:"),
        _pva("pixirad", "s1_pixirad2:"),
        _pva("gh1", "1idGH1:"),
        _pva("pg1", "1idPG1:"),
        _pva("pg5", "1idSP5:"),
        _pva("s1varex1", "1idVarex1:"),
        _pva("Sim Detector", "midasSim:"),
    ]},
    # PLACEHOLDER prefix/plugin -- 17-BM's Varex is only ever driven via
    # plain Channel Access (cam1/TIFF1/BADPIX1 plugins in the beamline's own
    # control repo; no NDPluginPva referenced anywhere there). Confirm with
    # 17-BM staff whether the IOC actually runs an NDPluginStdArrays
    # instance named "image1:" (e.g. `caget 17bmVarex:image1:UniqueId_RBV`)
    # before relying on this in a real deployment.
    "17-BM": {"devices": [
        _ca("varex", "17bmVarex:"),
        _pva("Sim Detector", "midasSim:"),
    ]},
    # Extracted from the beamline's apstools.devices.ad_creator spec (not a
    # B-PILOT blueprint, unlike the others above) -- its "pva" plugin entry
    # has no explicit port/suffix override, so it's the apstools default
    # ("Pva1:Image"), matching _pva()'s own default.
    "S3IDC": {"devices": [
        _pva("eiger2", "dp_eiger_sn:"),
        _pva("Sim Detector", "midasSim:"),
    ]},
}


# ── location ─────────────────────────────────────────────────────────────────
def _config_base_dir() -> Path:
    """Per-OS config base directory, chosen without any Qt dependency."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_DIRNAME


def _profiles_dir() -> Path:
    return _config_base_dir() / PROFILES_DIRNAME


def _meta_path() -> Path:
    return _config_base_dir() / META_FILENAME


def _recent_path() -> Path:
    return _config_base_dir() / RECENT_FILENAME


# ── io ───────────────────────────────────────────────────────────────────────
def read_json(path) -> dict:
    """Read a JSON dict; return {} on any problem (missing / malformed / non-dict)."""
    try:
        p = Path(path)
        if p.is_file():
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


# ── profile migration / bookkeeping ───────────────────────────────────────────
def _ensure_profiles() -> None:
    """Idempotent, cheap-to-repeat setup: make sure ``profiles/`` and the meta
    file exist, migrating a pre-existing single ``config.json`` into a
    ``Default`` profile the first time this runs, then seeding any
    not-yet-seen :data:`BUNDLED_PROFILES` (per-beamline device presets)."""
    pdir = _profiles_dir()
    legacy = _config_base_dir() / CONFIG_FILENAME
    if not pdir.is_dir():
        pdir.mkdir(parents=True, exist_ok=True)
        seed = read_json(legacy) if legacy.is_file() else {}
        _write_json(pdir / f"{DEFAULT_PROFILE}.json", seed)
    elif not any(pdir.glob("*.json")):
        # Directory exists but every profile was removed by hand — reseed.
        _write_json(pdir / f"{DEFAULT_PROFILE}.json", {})

    meta = _meta_path()
    meta_data = read_json(meta)
    seeded = set(meta_data.get("bundled_seeded", []))
    to_seed = [n for n in BUNDLED_PROFILES if n not in seeded]
    if to_seed:
        for name in to_seed:
            path = pdir / f"{name}.json"
            if not path.is_file():
                _write_json(path, dict(BUNDLED_PROFILES[name]))
            seeded.add(name)
        meta_data["bundled_seeded"] = sorted(seeded)
        _write_json(meta, meta_data)

    names = _profile_names()
    fallback = DEFAULT_PROFILE if DEFAULT_PROFILE in names else (names[0] if names else DEFAULT_PROFILE)
    if not meta.is_file():
        _write_json(meta, {"active": fallback})
    elif meta_data.get("active") not in names:
        meta_data["active"] = fallback
        _write_json(meta, meta_data)


def _profile_names() -> list:
    return sorted(p.stem for p in _profiles_dir().glob("*.json"))


_NAME_RE = re.compile(r"^[^/\\:*?\"<>|]{1,80}$")


def _validate_profile_name(name: str) -> str:
    name = (name or "").strip()
    if not name or not _NAME_RE.match(name):
        raise ValueError(
            "Profile name must be 1-80 characters and may not contain "
            '/ \\ : * ? " < > |')
    return name


# ── profile API ────────────────────────────────────────────────────────────
def list_profiles() -> list:
    """Names of all saved profiles, sorted alphabetically."""
    _ensure_profiles()
    return _profile_names()


def active_profile() -> str:
    """Name of the currently active profile (auto-recovers if it went missing)."""
    _ensure_profiles()
    return read_json(_meta_path()).get("active", DEFAULT_PROFILE)


def set_active_profile(name: str) -> None:
    """Switch the active profile. Raises ValueError if it doesn't exist."""
    global _cache
    _ensure_profiles()
    if name not in _profile_names():
        raise ValueError(f"No such profile: {name!r}")
    _write_json(_meta_path(), {"active": name})
    _cache = None


def profile_path(name: Optional[str] = None) -> Path:
    """Path to a profile's JSON file (the active one, by default)."""
    _ensure_profiles()
    return _profiles_dir() / f"{(name or active_profile())}.json"


def create_profile(name: str, seed_cfg: Optional[dict] = None) -> Path:
    """Create a new, empty (or pre-seeded) profile. Raises ValueError if the
    name is invalid or already taken."""
    name = _validate_profile_name(name)
    _ensure_profiles()
    if name in _profile_names():
        raise ValueError(f"A profile named {name!r} already exists")
    path = _profiles_dir() / f"{name}.json"
    _write_json(path, dict(seed_cfg or {}))
    return path


def duplicate_profile(src: str, dst: str) -> Path:
    """Copy an existing profile's contents into a new profile named ``dst``."""
    cfg = read_json(profile_path(src))
    return create_profile(dst, seed_cfg=cfg)


def rename_profile(old: str, new: str) -> Path:
    """Rename a profile, keeping it active if it was the active one."""
    global _cache
    new = _validate_profile_name(new)
    _ensure_profiles()
    names = _profile_names()
    if old not in names:
        raise ValueError(f"No such profile: {old!r}")
    if new == old:
        return profile_path(old)
    if new in names:
        raise ValueError(f"A profile named {new!r} already exists")
    was_active = active_profile() == old
    old_path, new_path = profile_path(old), _profiles_dir() / f"{new}.json"
    old_path.rename(new_path)
    if was_active:
        _write_json(_meta_path(), {"active": new})
        _cache = None
    return new_path


def delete_profile(name: str) -> None:
    """Delete a profile. Refuses to delete the last remaining profile. If the
    deleted profile was active, the active pointer moves to another
    remaining profile (``Default`` if present, else whichever sorts first)."""
    global _cache
    _ensure_profiles()
    names = _profile_names()
    if name not in names:
        raise ValueError(f"No such profile: {name!r}")
    if len(names) <= 1:
        raise ValueError("Cannot delete the only remaining profile")
    was_active = active_profile() == name
    profile_path(name).unlink(missing_ok=True)
    if was_active:
        remaining = _profile_names()
        fallback = DEFAULT_PROFILE if DEFAULT_PROFILE in remaining else remaining[0]
        _write_json(_meta_path(), {"active": fallback})
        _cache = None


# ── active-profile config (unchanged call signatures) ────────────────────────
def user_config_path() -> Path:
    """Path to the *active profile's* config file."""
    return profile_path()


def load_config(*, force: bool = False) -> dict:
    """The active profile's config dict (cached; never raises; {} if no file)."""
    global _cache
    if _cache is not None and not force:
        return _cache
    _cache = read_json(user_config_path())
    return _cache


def reload() -> dict:
    """Re-read the active profile's config file from disk and refresh the cache."""
    return load_config(force=True)


def save_user_config(cfg: dict) -> Path:
    """Write ``cfg`` to the active profile's config file (creating parent dirs)
    and invalidate the cache. Returns the path written."""
    global _cache
    path = user_config_path()
    _write_json(path, cfg)
    _cache = None
    return path


def reset_user_config() -> None:
    """Reset the active profile to the built-in shipped defaults (an empty
    overlay) without removing the profile itself."""
    save_user_config({})


def config_paths() -> dict:
    up = user_config_path()
    return {"user": up, "user_exists": up.is_file()}


# ── last-used Exp ID ─────────────────────────────────────────────────────────
# Deliberately global (in profile_meta.json, alongside "active" — not stored
# inside a profile), mirroring recent files below: an Exp ID like "park_may26"
# tracks the current experiment, not the beamline, so it shouldn't reset when
# switching Profiles. This is only the app-wide fallback shown at startup —
# once a Project is open, its own saved Exp ID (part of /gui_workspace meta,
# see app.py's _serialize_workspace/_apply_workspace_state) takes over.
def get_last_expid() -> str:
    return read_json(_meta_path()).get("last_expid", "")


def set_last_expid(value: str) -> None:
    meta = read_json(_meta_path())
    meta["last_expid"] = value or ""
    _write_json(_meta_path(), meta)


# ── recent files (Projects / Workspaces) ─────────────────────────────────────
# Deliberately global (not stored inside a profile) — a recently-opened
# project/workspace should stay "recent" regardless of which beamline Profile
# happens to be active, matching how every other app's recent-files list works.
def _read_recent() -> dict:
    data = read_json(_recent_path())
    for kind in RECENT_KINDS:
        data.setdefault(kind, [])
    return data


def record_recent(path, kind: str, *, when_utc: Optional[str] = None) -> None:
    """Add ``path`` to the front of the MRU list for ``kind`` ("project" or
    "workspace"), de-duplicating by path and capping at :data:`RECENT_MAX`."""
    if kind not in RECENT_KINDS:
        raise ValueError(f"Unknown recent-files kind: {kind!r}")
    if when_utc is None:
        from datetime import datetime, timezone
        when_utc = datetime.now(timezone.utc).isoformat()
    path = str(Path(path))
    data = _read_recent()
    entries = [e for e in data[kind] if e.get("path") != path]
    entries.insert(0, {"path": path, "name": Path(path).name, "last_opened_utc": when_utc})
    data[kind] = entries[:RECENT_MAX]
    _write_json(_recent_path(), data)


def autosave_draft_path() -> Path:
    """Path to the Workspace autosave/crash-recovery draft file. Global (not
    per-profile) since a crash can happen regardless of which profile is
    active; unrelated to any Project's `.h5` file, which this never touches."""
    return _config_base_dir() / "autosave" / "workspace_draft.json"


def get_recent(kind: str) -> list:
    """MRU-ordered list of ``{"path", "name", "last_opened_utc"}`` dicts for
    ``kind``. Entries whose file no longer exists are silently dropped (and
    the dropped list persisted) rather than shown as dead menu items."""
    if kind not in RECENT_KINDS:
        raise ValueError(f"Unknown recent-files kind: {kind!r}")
    data = _read_recent()
    entries = data[kind]
    live = [e for e in entries if Path(e.get("path", "")).is_file()]
    if len(live) != len(entries):
        data[kind] = live
        _write_json(_recent_path(), data)
    return live
