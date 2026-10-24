"""Which dark a given input file gets — the decision, with no Qt attached.

Batch Correction and Batch Integrate used to answer this differently and
nothing said so: ``BatchCorrectionWorker`` resolved a dark per input file
through ``frame_correct.resolve_dark``, while ``BatchWorker`` (caking) took
whatever array the GUI handed it and applied that to every file. The same
selection therefore produced different corrections depending on which tab
ran it. This module is the single answer both of them, and both background
CLIs, now ask.

It lives apart from the workers for the reason ``queue_policy`` does: it is
a decision that is easy to get subtly wrong in the middle of a QThread, and
it has to be reachable identically from two threads and two command-line
entry points, none of which should need a ``QApplication`` to find out
which dark to subtract.

The rule is a function of HOW THE DATA WAS PICKED
-------------------------------------------------
Explicit picks use the Dark field as chosen. Picks that traverse
subfolders resolve a dark per file, because one dark cannot be right for a
tree of scans measured at different times:

====================  =========================================
pick                  dark
====================  =========================================
one file              the Dark field
multiple files        the Dark field
a folder (flat)       the Dark field
a folder, recursive   resolved per file
files sharing a stem  resolved per file
====================  =========================================

A stem match is on the autodetect side because it globs
``<folder>/**/<stem>*`` and crosses subfolder boundaries whatever the
"Include subfolders" box says.

Unresolvable in autodetect mode is a SKIP
------------------------------------------
Not a fallback. ``resolve_dark`` returns at its rung 3 whenever a
``fallback`` is passed and so never reaches "no dark found" — which means
handing it the GUI dark as a safety net silently converts "I could not find
this file's dark" into "I used some other scan's dark". In autodetect mode
this module passes ``fallback=None`` on purpose, and the caller is expected
to skip the file and report it rather than correct it with a guess.

Nothing here imports Qt. Keep it that way.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Picks that traverse subfolders, and so resolve a dark per file.
AUTO_ORIGINS = frozenset({"folder_recursive", "stem"})

#: Every origin ``DataLoaderPanel.pick_origin`` can report.
ORIGINS = frozenset({"file", "files", "folder"}) | AUTO_ORIGINS

#: ``override`` values. ``None`` means "decide from the pick".
ALWAYS_EXPLICIT = "always_explicit"
ALWAYS_AUTO = "always_auto"


@dataclass(frozen=True)
class DarkRequest:
    """What the GUI knows, independent of any one input file."""
    origin: str = "file"
    path: Optional[str] = None          #: the Dark field's file, if any
    dataset: str = "exchange/data_dark"  #: container searched inside it
    array: Optional[np.ndarray] = None   #: the Dark field, already averaged
    panel: Optional[int] = None          #: Hydra panel, for the message
    override: Optional[str] = None
    #: In autodetect mode, the folder the user actually chose. The search
    #: covers this WHOLE TREE, not just each file's own directory: at
    #: 1-ID-E the darks live in sibling folders (``AlCaBa_S1_RT_Darks/``)
    #: one level up from the scans they serve, so a per-directory search
    #: finds nothing. None falls back to each file's own folder.
    search_root: Optional[str] = None
    #: Exposure the data was taken at; a candidate must match it.
    exposure: Optional[float] = None


@dataclass(frozen=True)
class DarkChoice:
    """``status`` is ``"ok"`` or ``"skipped"``; ``why`` is always sayable.

    ``source`` is the file the pixels came from, when one is known. It is
    separate from ``why`` on purpose: ``why`` is prose and embeds a
    per-step distance ("3 scans away"), so it IDENTIFIES nothing and
    cannot be used as a cache key. Keying on it wrote one TIFF per step
    instead of one per dark — caught by
    test_many_steps_sharing_one_dark_write_one_tiff.
    """
    array: Optional[np.ndarray]
    why: str
    status: str = "ok"
    source: Optional[str] = None

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"


def wants_auto(req: DarkRequest) -> bool:
    """Should this request resolve a dark per file?"""
    if req.override == ALWAYS_AUTO:
        return True
    if req.override == ALWAYS_EXPLICIT:
        return False
    return req.origin in AUTO_ORIGINS


def describe(req: DarkRequest) -> str:
    """One line for the UI, so the mode is never a guess."""
    if wants_auto(req):
        return ("Dark: resolved per file (the selection spans subfolders)"
                if req.override is None else
                "Dark: resolved per file (forced)")
    which = os.path.basename(req.path) if req.path else "none set"
    return (f"Dark: the Dark field for every file — {which}"
            if req.override is None else
            f"Dark: the Dark field for every file, forced — {which}")


def resolve(file_path, req: DarkRequest, *, shape=None) -> DarkChoice:
    """The dark for one input file.

    In explicit mode the sibling ladder is skipped entirely
    (``auto=False``) and the Dark field is used as given. In autodetect
    mode the ladder runs with **no fallback**, so a file it cannot resolve
    comes back ``skipped`` instead of quietly inheriting another scan's
    dark.
    """
    from midas_gui.frame_correct import resolve_dark

    if wants_auto(req):
        if req.search_root:
            cands = tree_dark_candidates(file_path, req.search_root,
                                         exposure=req.exposure,
                                         dataset=req.dataset)
            for path, why in cands:
                if path is None:
                    return DarkChoice(
                        None, f"{_skip_reason(file_path, req)} — {why}",
                        "skipped")
                arr = _peek_shaped(path, req.dataset, shape)
                if arr is not None:
                    return DarkChoice(arr, why, source=str(path))
            return DarkChoice(None, _skip_reason(file_path, req), "skipped")
        arr, why = resolve_dark(file_path, dark_dataset=req.dataset,
                                shape=shape, fallback=None, auto=True)
        if arr is None:
            return DarkChoice(None, _skip_reason(file_path, req), "skipped")
        return DarkChoice(arr, why, source=why)

    if req.array is None:
        # Explicit mode with nothing chosen is a legitimate run — correcting
        # without a dark is allowed, it just has to be said out loud.
        return DarkChoice(None, "no Dark field set — correcting without one")
    arr, why = resolve_dark(file_path, dark_dataset=req.dataset, shape=shape,
                            fallback=req.array, auto=False)
    return DarkChoice(arr, why, source=(req.path or "loader Dark field"))


#: Exposures this close (relative) count as the same acquisition setting.
EXPOSURE_RTOL = 0.01


def tree_dark_candidates(data_path, root, *, exposure=None, dataset=
                         "exchange/data_dark") -> list:
    """Dark files anywhere under ``root``, best first, for ``data_path``.

    The tree generalisation of ``frame_correct.dark_candidates``, which
    only ever lists one directory. Three things make it work:

    * **Scan numbers are global.** A dark at ``..._005787.ge1.h5`` and a
      scan at ``..._005826.ge1.h5`` are on one increasing sequence across
      the whole experiment, so "the nearest dark measured BEFORE this
      file" -- the rule the 192-log survey validated within a folder --
      means the same thing across a tree without reinterpretation.
    * **Exposure must match.** A GE panel's offset grows with exposure, so
      a 0.1 s dark is not a dark for a 10 s frame. Mismatches are dropped
      here rather than ranked low, because a wrong dark that is merely
      unpreferred still gets used when it is the only one.
    * **Frames settle doubt.** A name-matched candidate carrying obvious
      diffraction is rejected whatever it is called.

    Returns ``[(path, why)]``.
    """
    import os as _os
    from midas_gui.frame_correct import (acquisition_exposure, split_scan_name,
                                         has_diffraction_structure)
    from midas_gui.helpers import is_dark_like_name, is_h5

    num = split_scan_name(_os.path.basename(str(data_path))).num
    ranked, rejected = [], []
    for dirpath, _dirs, files in _os.walk(str(root)):
        for name in files:
            full = _os.path.join(dirpath, name)
            if full == str(data_path) or not is_h5(full):
                continue
            if not (is_dark_like_name(name) or is_dark_like_name(dirpath)):
                continue
            if exposure is not None:
                got = acquisition_exposure(full)
                if got is None or abs(got - exposure) > EXPOSURE_RTOL * max(
                        abs(exposure), 1e-9):
                    rejected.append((full, f"exposure {got}s != {exposure}s"))
                    continue
            cand = split_scan_name(name).num
            if cand is None or num is None:
                direction, dist = 2, 10 ** 9
            elif cand <= num:
                direction, dist = 0, num - cand        # measured before it
            else:
                direction, dist = 1, cand - num        # measured after it
            ranked.append(((direction, dist, full), full, name))
    ranked.sort(key=lambda row: row[0])

    out = []
    for (direction, dist, _f), full, name in ranked:
        arr = _peek(full, dataset)
        if arr is not None and has_diffraction_structure(arr):
            rejected.append((full, "frames show diffraction, not a dark"))
            continue
        when = ("preceding" if direction == 0 else
                "following" if direction == 1 else "unnumbered")
        out.append((full, f"{when} dark {name} ({dist} scans away)"))
    return out or [(None, _why_none(rejected))]


def _peek_shaped(path, dataset, shape):
    from midas_gui.frame_correct import _mean_dark
    return _mean_dark(path, dataset, shape)


def _peek(path, dataset):
    from midas_gui.frame_correct import _mean_dark
    return _mean_dark(path, dataset, None)


def _why_none(rejected) -> str:
    if not rejected:
        return "no dark-like file anywhere under the selected folder"
    import os as _os
    head = rejected[:3]
    bits = "; ".join(f"{_os.path.basename(p)}: {r}" for p, r in head)
    more = f" (+{len(rejected) - len(head)} more)" if len(rejected) > 3 else ""
    return f"every candidate rejected — {bits}{more}"


def _skip_reason(file_path, req: DarkRequest) -> str:
    panel = f" [{panel_token(req.panel)}]" if req.panel else ""
    return (f"no dark found for {os.path.basename(str(file_path))}{panel} "
            f"in {os.path.dirname(str(file_path))} — file skipped")


# ─────────────────────────────────────────── Hydra panel substitution ──
#: A panel token that is a whole path component or a dot-delimited segment.
#: Anchored on BOTH sides so ``average1/``, ``edge2/`` and ``image3`` are
#: left alone — the existing ``hydra.PANEL_IN_NAME`` rewrites those — while
#: still matching ``scan_ge1.h5``, which ``helpers._HYDRA_PANEL_RE``'s
#: leading ``\b`` misses because ``_`` is a word character.
_PANEL_TOKEN = re.compile(r"(?<![0-9A-Za-z])(ge)\s*([1-4])(?![0-9A-Za-z])",
                          re.IGNORECASE)


def panel_token(n) -> str:
    return f"ge{int(n)}"


def substitute_panel(path, n) -> Optional[str]:
    """Rewrite every panel token in ``path`` to panel ``n``.

    Case-preserving, and applied to the WHOLE path: a real file carries the
    token twice —
    ``/data/white_oct26/ge1/SCAN/SCAN_005826.ge1.h5`` — once as a directory
    and once in the extension, and rewriting only one half yields a path
    that does not exist.

    Returns ``None`` when the path carries no panel token at all, which is
    the caller's signal that a structural substitution is not available
    (rather than that the sibling is missing).
    """
    text = str(path)
    out, count = _PANEL_TOKEN.subn(
        lambda m, _n=int(n): m.group(1) + str(_n), text)
    return out if count else None


def panel_dark_path(anchor, n) -> tuple:
    """``(path, why)`` for panel ``n``'s dark, derived from ``anchor``.

    ``path`` is None when the structure cannot be derived or the derived
    file does not exist — never another panel's file. Applying ge1's dark
    to ge2 data is the specific mistake this function exists to prevent, so
    it fails rather than falls back.
    """
    if not anchor:
        return None, "no dark anchor path set"
    cand = substitute_panel(anchor, n)
    if cand is None:
        return None, (f"{os.path.basename(str(anchor))} carries no ge1-ge4 "
                      f"token, so {panel_token(n)}'s dark cannot be derived "
                      f"from it")
    if not os.path.isfile(cand):
        return None, f"{panel_token(n)} dark not found: {cand}"
    return cand, f"{panel_token(n)} dark derived from the anchor: {cand}"


class DarkArgBuilder:
    """``--dark`` arguments for background-job steps, with no copying.

    Was ``DarkTiffCache``, which wrote each resolved dark out as a
    pre-averaged TIFF. That cost 17 MB per step written during curation
    -- 816 MB and a visibly blocked GUI before a 48-step job launched --
    to duplicate a file already sitting on disk. The job can read the
    original: ``batch_cli --dark <source.h5> --dark-dataset <path>``.

    A TIFF is still written for the one case with no source file behind
    it: an explicitly chosen Dark field, which reaches us as an array.
    That is one file per panel, not one per step.
    """

    def __init__(self, out_dir, prefix: str = "_bg_job_dark"):
        self._dir = out_dir
        self._prefix = prefix
        self._explicit: dict = {}
        self.skipped: list = []

    def written(self) -> int:
        """TIFFs actually written — zero whenever every dark resolved to
        a real file, which is the normal autodetect case."""
        return len(self._explicit)

    def arg_for(self, probe, request, *, tag: str = ""):
        """``["--dark", path, "--dark-dataset", ds]`` for one step.

        ``[]`` when the step legitimately has no dark; ``None`` when it
        could not be resolved, in which case the caller must DROP the
        step -- never substitute another scan's dark.
        """
        import dataclasses
        from midas_gui.frame_correct import acquisition_exposure
        req = dataclasses.replace(request, exposure=acquisition_exposure(probe))
        choice = resolve(probe, req)
        if choice.skipped:
            self.skipped.append((str(probe), choice.why))
            return None
        if choice.array is None:
            return []
        src = choice.source
        # Only in AUTO mode. An explicitly chosen Dark field may carry a
        # frame-range the user set on it, and the array we were handed
        # already honours that -- re-reading the whole stack from the
        # path would quietly ignore it.
        if wants_auto(req) and src and os.path.isfile(src):
            return ["--dark", str(src), "--dark-dataset", req.dataset]
        # No file behind it: the loader's own Dark field. One TIFF per
        # tag, written once.
        path = self._explicit.get(tag)
        if path is None:
            import numpy as _np
            import tifffile as _tiff
            name = f"{self._prefix}{('_' + tag) if tag else ''}.tif"
            path = self._dir / name
            _tiff.imwrite(str(path),
                          _np.asarray(choice.array, dtype=_np.float32))
            self._explicit[tag] = path
        return ["--dark", str(path)]


#: Old name, kept so nothing breaks mid-edit.
DarkTiffCache = DarkArgBuilder
