"""Data model for the Batch Queue tab — many samples, grouped by calibration.

A beamtime rarely produces one thing to integrate. It produces a run of samples
sharing a calibration and a set of dark/bright/background frames, then the
calibrant changes and the next stage begins. This module is the model for that
shape: a three-level tree

    CalibrationNode           calibration + detector mask
      └─ CorrectionsNode      dark / bright / background
           └─ Sample          one HDF5 file, or one folder of frame files

with 1..N corrections nodes under a calibration, because darks are commonly
retaken partway through a stage while the calibration stands.

**This module deliberately imports no Qt.** Everything here — the tree, its
JSON round-trip, the input→output path mirroring, the per-sample source
descriptor — is pure logic so it can be unit-tested without a QApplication and
without the forked-child CoreFoundation hazard that governs this repo's Qt
tests (see STATE.md). The widgets live in ``tab_queue.py``, the execution in
``queue_runner.py``. The one lazy exception is :func:`detect_dataset`, which
reaches into ``helpers`` (and so transitively Qt) only when actually asked to
inspect an HDF5 file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from midas_gui.constants import H5_EXTS

#: A sample is one self-contained thing to integrate.
KIND_HDF5 = "hdf5"      # a single HDF5 container holding the frames
KIND_FOLDER = "folder"  # a directory of single-frame files (TIFF, .ge*, .cbf, …)

#: Where a calibration node's geometry comes from.
SOURCE_TAB2 = "tab2"    # snapshotted from the Calibrate tab's live result
SOURCE_FILE = "file"    # a paramstest / .poni / .json geometry file

#: Fallback when an HDF5 file's frame dataset can't be auto-detected. Matches
#: ``DataLoaderPanel._dataset``'s default so the two agree.
DEFAULT_DATASET = "exchange/data"

#: Samples that don't sit under the data root land here, rather than being
#: silently flattened into the output root alongside properly-mirrored ones.
UNROOTED_DIR = "_unrooted"

#: ``BatchWorker`` writes its per-format output into subfolders of the sample
#: directory named after the format (``csv/``, ``zarr/``, ``h5/``, …). A sample
#: whose label collides with one of those would put its project file at the
#: same path as a directory, so :func:`project_path_for` renames it.
_RESERVED_LABELS = {"csv", "xye", "fxye", "dat", "h5", "2d_csv", "zarr"}


def kind_for_path(path) -> str:
    """``KIND_HDF5`` for an HDF5-suffixed path, ``KIND_FOLDER`` otherwise.

    Suffix-only, like ``helpers.is_h5`` — deliberately does not touch the
    filesystem, so the model stays usable for a path that has gone away (a
    queue restored from a project saved on another machine, say). The tab
    validates existence when it renders the tree."""
    return KIND_HDF5 if Path(str(path)).suffix.lower() in H5_EXTS else KIND_FOLDER


def default_label(path, kind: Optional[str] = None) -> str:
    """The name a sample is shown and its output directory is named after."""
    p = Path(str(path))
    kind = kind or kind_for_path(p)
    return p.stem if kind == KIND_HDF5 else p.name


def detect_dataset(path) -> str:
    """The frame dataset inside an HDF5 sample — first ≥3-D one, else the
    first ≥2-D one, else :data:`DEFAULT_DATASET`.

    Same preference order ``DataLoaderPanel._on_path_changed`` uses when it
    populates its dataset combo, so a sample added here resolves to what the
    Data Viewer would have shown for the same file. Returns the fallback
    rather than raising when the file is missing or unreadable — a queue is
    routinely built before every file has landed, and the run itself will
    report a genuinely broken file far more usefully than tree-building can.

    Imported lazily: ``helpers`` pulls in PyQt5, and keeping that out of this
    module's import graph is what lets the model be tested Qt-free."""
    try:
        from midas_gui.helpers import list_h5_datasets
        items = list_h5_datasets(path)
    except Exception:
        return DEFAULT_DATASET
    for name, shape in items:
        if len(shape) >= 3:
            return name
    return items[0][0] if items else DEFAULT_DATASET


# ═════════════════════════════════════════════════════════════════════════════
#  The tree
# ═════════════════════════════════════════════════════════════════════════════
#
# Each level carries its own to_json/from_json rather than leaning on
# dataclasses.asdict, for two reasons: from_json has to rebuild the nested
# dataclasses anyway, and being explicit lets an older saved queue (missing a
# key added later) load with defaults instead of raising.


@dataclass
class Sample:
    """One HDF5 file or one folder of frames — the unit of work and the unit
    the output tree mirrors."""
    path: str
    kind: str = ""
    dataset: Optional[str] = None   # HDF5 only; None → resolved at run time
    label: str = ""
    enabled: bool = True

    def __post_init__(self):
        self.path = str(self.path)
        if not self.kind:
            self.kind = kind_for_path(self.path)
        if not self.label:
            self.label = default_label(self.path, self.kind)

    def to_json(self) -> dict:
        return {"path": self.path, "kind": self.kind, "dataset": self.dataset,
                "label": self.label, "enabled": self.enabled}

    @classmethod
    def from_json(cls, d: dict) -> "Sample":
        return cls(path=d.get("path", ""), kind=d.get("kind", ""),
                   dataset=d.get("dataset"), label=d.get("label", ""),
                   enabled=bool(d.get("enabled", True)))


@dataclass
class CorrectionsNode:
    """A dark/bright/background set, and the samples collected under it."""
    name: str = "Corrections"
    dark: Optional[str] = None
    bright: Optional[str] = None
    background: Optional[str] = None
    background_scale: float = 1.0
    bright_mode: str = "divide"
    samples: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {"name": self.name, "dark": self.dark, "bright": self.bright,
                "background": self.background, "background_scale": self.background_scale,
                "bright_mode": self.bright_mode,
                "samples": [s.to_json() for s in self.samples]}

    @classmethod
    def from_json(cls, d: dict) -> "CorrectionsNode":
        return cls(name=d.get("name", "Corrections"), dark=d.get("dark"),
                   bright=d.get("bright"), background=d.get("background"),
                   background_scale=float(d.get("background_scale", 1.0)),
                   bright_mode=d.get("bright_mode", "divide"),
                   samples=[Sample.from_json(s) for s in d.get("samples") or []])

    def describe(self) -> str:
        """One-line summary of which correction frames are set, for the tree."""
        bits = [f"{k}={Path(v).name}" for k, v in
                (("dark", self.dark), ("bright", self.bright),
                 ("background", self.background)) if v]
        return "  ".join(bits) if bits else "no corrections"


@dataclass
class CalibrationNode:
    """A calibration and its detector mask, with 1..N corrections nodes.

    The mask lives here rather than on the corrections nodes because it is a
    property of the detector and its geometry — beamstop shadow, dead pixels,
    panel gaps. That placement is also what makes one shared detector map per
    calibration node correct: ``workers.build_integration_context`` depends on
    the spec, kernel, mask, corrections flags and weighting, and *not* on
    dark/bright/background, which are applied per frame afterwards.

    ``calib_snapshot`` is a ``project.sanitize_result_dict`` of the Calibrate
    tab's result, taken when the node is created — so it is JSON-safe, and a
    later re-calibration cannot silently change an already-queued stage.
    Rehydrate with ``project.calibration_namespace``."""
    name: str = "Calibration"
    source: str = SOURCE_TAB2
    file_path: Optional[str] = None
    calib_snapshot: Optional[dict] = None
    mask_sources: Optional[list] = None      # MaskSelector's serialised shape
    corrections: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {"name": self.name, "source": self.source,
                "file_path": self.file_path, "calib_snapshot": self.calib_snapshot,
                "mask_sources": self.mask_sources,
                "corrections": [c.to_json() for c in self.corrections]}

    @classmethod
    def from_json(cls, d: dict) -> "CalibrationNode":
        return cls(name=d.get("name", "Calibration"),
                   source=d.get("source", SOURCE_TAB2),
                   file_path=d.get("file_path"),
                   calib_snapshot=d.get("calib_snapshot"),
                   mask_sources=d.get("mask_sources"),
                   corrections=[CorrectionsNode.from_json(c)
                                for c in d.get("corrections") or []])

    def using_file(self) -> bool:
        return self.source == SOURCE_FILE


@dataclass
class BatchQueue:
    """The whole tree, plus the roots and the queue-wide integration settings.

    ``settings`` is left an open dict on purpose: it is filled by the tab from
    its own widgets (R bin, η bin, Rmin/Rmax, formats, kernel, …) and copied
    wholesale from the Batch Integrate tab, so pinning its keys here would mean
    editing this module every time that tab grows a control."""
    calibrations: list = field(default_factory=list)
    data_root: Optional[str] = None
    out_root: Optional[str] = None
    settings: dict = field(default_factory=dict)

    # ── traversal ────────────────────────────────────────────────────

    def iter_samples(self, enabled_only: bool = False) -> Iterator[tuple]:
        """Yield ``(calibration, corrections, sample)`` in tree order."""
        for cal in self.calibrations:
            for corr in cal.corrections:
                for s in corr.samples:
                    if enabled_only and not s.enabled:
                        continue
                    yield cal, corr, s

    def sample_count(self, enabled_only: bool = False) -> int:
        return sum(1 for _ in self.iter_samples(enabled_only=enabled_only))

    def sample_paths(self, enabled_only: bool = True) -> list:
        return [s.path for _c, _k, s in self.iter_samples(enabled_only=enabled_only)]

    # ── serialisation ────────────────────────────────────────────────

    def to_json(self) -> dict:
        return {"calibrations": [c.to_json() for c in self.calibrations],
                "data_root": self.data_root, "out_root": self.out_root,
                "settings": dict(self.settings)}

    @classmethod
    def from_json(cls, d: Optional[dict]) -> "BatchQueue":
        d = d or {}
        return cls(calibrations=[CalibrationNode.from_json(c)
                                 for c in d.get("calibrations") or []],
                   data_root=d.get("data_root"), out_root=d.get("out_root"),
                   settings=dict(d.get("settings") or {}))


# ═════════════════════════════════════════════════════════════════════════════
#  Source descriptors
# ═════════════════════════════════════════════════════════════════════════════

def sample_source_cfg(sample: Sample, *, chunk_size=None,
                      combine_op: str = "mean") -> dict:
    """The tagged descriptor ``workers._open_source_cfg`` consumes.

    Mirrors ``DataLoaderPanel.source_cfg`` (widgets.py), but built per sample:
    queue samples are never loaded into a loader panel, so that method — which
    reads the panel's own widgets — cannot be reused directly. Only two of its
    four shapes are reachable from here, because a queue sample is by
    definition one container or one directory:

    * ``KIND_HDF5``   → ``{"type": "hdf5", …}``       (one file, its own stack)
    * ``KIND_FOLDER`` → ``{"type": "tiff_glob", …}``  (a directory of frames)

    ``tiff_list``/``hdf5_stack_glob`` belong to the interactive tab's
    multi-file picks and have no queue equivalent — there, N files are one
    sample; here, N files are N samples."""
    if sample.kind == KIND_HDF5:
        return {"type": "hdf5", "path": sample.path,
                "dataset": sample.dataset or DEFAULT_DATASET,
                "chunk_size": chunk_size, "combine_op": combine_op}
    return {"type": "tiff_glob", "path": sample.path}


# ═════════════════════════════════════════════════════════════════════════════
#  Input tree → output tree
# ═════════════════════════════════════════════════════════════════════════════
#
# Nothing in the package did this before: Path.relative_to against a root
# appears nowhere. BatchTab._suggest_output_dir is the nearest precedent but
# reads its segments positionally off ancestor depth for one specific beamline
# layout — a sibling strategy, not something to extend.


def _abs(path) -> Path:
    """Absolute, ``..``-collapsed, without touching the filesystem.

    ``Path.resolve()`` would also follow symlinks and stat the path, which
    makes the mirroring untestable against paths that don't exist yet — and
    the output tree by definition doesn't."""
    return Path(os.path.abspath(str(path)))


def sample_target(sample_path, kind: Optional[str] = None) -> Path:
    """Where this sample sits in the *input* tree, as a directory-shaped path.

    A folder sample is already directory-shaped. An HDF5 sample is a file, and
    its output directory is named after it with the suffix dropped — so
    ``/d/s1/scan_001.h5`` occupies ``/d/s1/scan_001``. ``with_suffix("")``
    rather than ``stem`` so a compound suffix keeps its inner part
    (``scan.vrx.h5`` → ``scan.vrx``), matching ``Path.stem``."""
    p = _abs(sample_path)
    kind = kind or kind_for_path(p)
    return p.with_suffix("") if kind == KIND_HDF5 else p


def infer_data_root(sample_paths) -> Optional[str]:
    """The deepest directory every sample sits under — the default mirror root.

    Computed over the samples' *parent* directories rather than the samples
    themselves, so the relative part always retains at least the sample's own
    name. With one sample that gives its parent (``/d/s1/scan.h5`` → ``/d/s1``,
    mirroring to ``<out>/scan``); with several it gives their common ancestor.

    ``None`` when there are no samples, or when they can't share a root at all
    — different Windows drives, which ``os.path.commonpath`` rejects."""
    parents = [str(sample_target(p).parent) for p in sample_paths]
    if not parents:
        return None
    try:
        return os.path.commonpath(parents)
    except ValueError:
        return None


def mirror_output_dir(sample_path, data_root, out_root,
                      kind: Optional[str] = None, label: str = "") -> Path:
    """This sample's output directory: its position under ``data_root``,
    re-rooted at ``out_root``.

        /data/bt/s1/scan_001.h5   →  <out_root>/s1/scan_001/
        /data/bt/tiffs/sampleA/   →  <out_root>/tiffs/sampleA/

    (both with ``data_root=/data/bt``). Everything for the sample lands in
    there: its ``csv/``/``zarr/``/``h5/`` output subfolders and its project
    file.

    A sample outside ``data_root`` — another volume, an unrelated path a user
    dragged in — goes to ``<out_root>/_unrooted/<label>/`` rather than being
    silently flattened next to the properly-mirrored ones. Same for a sample
    that *is* the data root, which would otherwise mirror onto ``out_root``
    itself and collide with everything."""
    target = sample_target(sample_path, kind)
    out_root = _abs(out_root)
    label = label or default_label(sample_path, kind)
    if data_root:
        try:
            rel = target.relative_to(_abs(data_root))
        except ValueError:
            rel = None
        if rel is not None and rel != Path("."):
            return out_root / rel
    return out_root / UNROOTED_DIR / label


def is_unrooted(out_dir, out_root) -> bool:
    """True when :func:`mirror_output_dir` had to fall back for this sample —
    the tab flags these rather than letting them pass unnoticed."""
    try:
        return _abs(out_dir).relative_to(_abs(out_root)).parts[:1] == (UNROOTED_DIR,)
    except (ValueError, IndexError):
        return False


def project_path_for(out_dir, label: str = "") -> Path:
    """The per-sample project file inside its output directory.

    ``<out_dir>/<label>.h5``, except when the label is one of the format names
    ``BatchWorker`` uses for its output subfolders — a sample called ``h5``
    would otherwise want its project at exactly the path of the ``h5/``
    directory. Those get ``<label>_project.h5``."""
    out_dir = Path(out_dir)
    name = label or out_dir.name
    if name.lower() in _RESERVED_LABELS:
        name = f"{name}_project"
    return out_dir / f"{name}.h5"


def plan_output_dirs(queue: BatchQueue) -> tuple:
    """Resolve every enabled sample's output directory up front.

    Returns ``(rows, problems)`` where ``rows`` is a list of
    ``(calibration, corrections, sample, out_dir)`` and ``problems`` is a list
    of human-readable blocking reasons. The tab shows ``rows`` as the mapping
    preview and refuses to start while ``problems`` is non-empty — better to
    find out that two samples want the same directory before a four-hour run
    than after it.

    Pure: no filesystem access, so it is unit-testable and cheap enough to call
    on every edit. The writability preflight
    (``helpers.check_output_dir_writable``) is the caller's job, since it does
    touch the disk."""
    problems: list = []
    if not queue.out_root:
        problems.append("No output root set.")
    rows: list = []
    if not problems:
        data_root = queue.data_root or infer_data_root(queue.sample_paths())
        for cal, corr, s in queue.iter_samples(enabled_only=True):
            rows.append((cal, corr, s,
                         mirror_output_dir(s.path, data_root, queue.out_root,
                                           kind=s.kind, label=s.label)))
        seen: dict = {}
        for _cal, _corr, s, out_dir in rows:
            seen.setdefault(str(out_dir), []).append(s.label)
        for out_dir, labels in seen.items():
            if len(labels) > 1:
                problems.append(
                    f"{len(labels)} samples map to the same output directory "
                    f"'{out_dir}': {', '.join(labels)}. Rename one, or pick a "
                    "data root they differ under.")
    if not rows and not problems:
        problems.append("No enabled samples in the queue.")
    return rows, problems
