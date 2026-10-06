"""Persistent per-run screen logs, under ``~/midas_runs/midas_screen_logs``.

Tabs log to an on-screen ``LogPanel``, which is gone the moment the window
closes — so "what dark did that run pick?", asked a week later, had no answer.
The caking workflow this GUI replaces already kept its logs on disk in exactly
the tree this writes into::

    ~/midas_runs/midas_screen_logs/<beamline>/<expid>/<name>.screenlog
    └─ e.g. 20-id-e/pan_jul26/pan_jul26_caking_s20varex2_1_14_A_21986-21986.screenlog

Those 198 historical files are not incidental: they are what the dark-selection
ladder in ``frame_correct`` was derived from and what
``tests/test_dark_autodetect`` still replays. Writing new logs beside them, in
the same shape, keeps that corpus growing rather than starting a second one
somewhere else.

``<beamline>`` is the active profile lowercased (``20-ID-E`` -> ``20-id-e``,
which is what the existing folders are already called) and ``<expid>`` is the
header's Exp ID field.

**Nothing here may ever break a run.** A log is a record of work, not the work;
every failure path degrades to "no log" plus one line in the on-screen panel,
never an exception that reaches the worker.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Optional

#: Root of the screen-log tree. Under ``Path.home()`` rather than hardcoded,
#: so it follows whoever is logged in instead of one operator's account.
LOG_ROOT = Path.home() / "midas_runs" / "midas_screen_logs"

#: Stand-ins when the header fields are blank. Spelled so they sort together
#: and are obviously not a real beamline or experiment, because a log filed
#: under a guess is worse than one filed under "unknown".
NO_EXPID = "_no_expid"
NO_BEAMLINE = "_no_beamline"

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str, fallback: str) -> str:
    """One path segment: lowercased, path separators and spaces collapsed.

    Deliberately strict — these names come from a free-text header field and
    land in a shared directory tree, so ``../`` or an embedded slash must not
    be able to steer the write out of it.
    """
    out = _SAFE.sub("_", (text or "").strip().lower()).strip("._-")
    return out or fallback


def log_dir(profile: Optional[str], expid: Optional[str]) -> Path:
    """``<root>/<beamline>/<expid>`` for this run. Not created here."""
    return LOG_ROOT / _slug(profile, NO_BEAMLINE) / _slug(expid, NO_EXPID)


def log_path(kind: str, *, profile, expid, stem: str = "") -> Path:
    """Full path for one run's log.

    Named ``<expid>_<kind>_<stem>_<timestamp>.screenlog``, echoing the caking
    logs' ``<expid>_caking_<detector>_<froot>_<range>.screenlog``. A timestamp
    rather than a frame range because a correction run can span many files and
    two runs over the same files are a normal thing to want to compare.
    """
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [_slug(expid, NO_EXPID), _slug(kind, "run")]
    if stem:
        parts.append(_slug(stem, ""))
    parts.append(stamp)
    name = "_".join(p for p in parts if p) + ".screenlog"
    return log_dir(profile, expid) / name


class ScreenLog:
    """An open run log. Append-only, flushed per line, never raises.

    Flushed on every write rather than buffered: the reason to want this file
    at all is usually a run that died, and a buffered tail is exactly the part
    that would be missing.

    A ``ScreenLog`` whose file could not be opened is not an error — it is a
    live object that quietly discards writes, so callers need no special case.
    :attr:`path` is None and :attr:`error` says why.
    """

    def __init__(self, path: Optional[Path]):
        self.path: Optional[Path] = None
        self.error: Optional[str] = None
        self._fh = None
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            self.path = path
        except Exception as exc:          # unwritable share, full disk, …
            self.error = f"{type(exc).__name__}: {exc}"

    def write(self, line: str) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(line.rstrip("\n") + "\n")
            self._fh.flush()
        except Exception:
            # Stop trying rather than raise on every subsequent line.
            self.close()

    def header(self, title: str, fields: dict) -> None:
        """The ``====``-delimited preamble the caking logs open with, so the
        two generations of log read the same way."""
        self.write("=" * 40)
        self.write(title)
        self.write("=" * 40)
        for key, value in fields.items():
            self.write(f"{key:<13}: {value}")
        self.write("=" * 40)

    def close(self) -> None:
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def open_log(kind: str, *, profile, expid, stem: str = "") -> ScreenLog:
    """Open this run's log. Returns a discarding :class:`ScreenLog` rather
    than raising if the path cannot be computed or opened."""
    try:
        return ScreenLog(log_path(kind, profile=profile, expid=expid, stem=stem))
    except Exception:
        return ScreenLog(None)
