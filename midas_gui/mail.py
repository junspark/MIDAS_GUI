"""Sending one short report by email, with no Qt and no ``Job`` attached.

Extracted from ``job_queue.send_job_notification`` when a second caller
appeared (the dark-skip report). Two copies of the same ``mail``
invocation would have drifted in exactly the way the contract below is
easiest to get wrong.

**Best effort, by contract.** A mail system that is missing,
misconfigured or slow must cost a log line and nothing else. Every caller
sends *after* the work it describes has finished, so there is no outcome
left to protect, and a notifier that could wedge the GUI would be worse
than no notifier.

**Delivery is NOT confirmed.** ``mail`` returning 0 means the local MTA
accepted the message, not that it reached the recipient; a lab host
relaying to an external domain may still be dropped at the far end. Say
"sent", never "delivered".
"""
from __future__ import annotations

import subprocess


def send_mail(subject: str, body, address: str, *, _runner=None) -> bool:
    """Hand one message to the local ``mail`` command.

    ``body`` may be a string or a list of lines. Returns True when the
    message was handed off — not when it arrived. A blank address is not
    an error: it is how a user says "no mail", and every caller treats it
    as "report to the log instead".
    """
    address = (address or "").strip()
    if not address:
        return False
    text = body if isinstance(body, str) else "\n".join(body)
    runner = _runner or subprocess.run
    try:
        runner(["mail", "-s", subject, address],
               input=text + "\n", text=True, timeout=30, capture_output=True)
        return True
    except Exception:
        return False


def dark_skip_report(skipped, *, context: str = "") -> tuple:
    """``(subject, body)`` for files that ran without being corrected.

    A skip is the deliberate outcome when no valid dark can be found —
    correcting a file with another scan's dark is the error the policy
    exists to prevent — but it is only defensible if it is loud. This is
    what makes it loud.
    """
    n = len(skipped)
    where = f" — {context}" if context else ""
    subject = f"MIDAS: {n} file(s) skipped, no valid dark{where}"
    body = [
        f"{n} file(s) were NOT processed because no valid dark could be",
        "resolved for them.",
        "",
        "They were skipped rather than corrected with another scan's dark,",
        "which would have produced a plausible and wrong result.",
        "",
    ]
    for path, why in skipped[:50]:
        body += [f"  {path}", f"      {why}", ""]
    if n > 50:
        body.append(f"  ... and {n - 50} more.")
    return subject, body


def summarise_skips(skipped, *, limit: int = 8) -> str:
    """A few lines a human can act on, for a dialog.

    NOT the full reasons. Each reason names every candidate it rejected
    and why, which is right for a log and unreadable in a message box --
    four panels of one scan produced a wall of repeated text that the
    user could not read at all. The detail stays in the Log; this says
    WHAT was skipped and the one thing they share.
    """
    import collections
    import os as _os
    if not skipped:
        return ""
    names = []
    for path, _why in skipped:
        b = _os.path.basename(str(path))
        # strip the panel/extension tail so four panels read as one scan
        names.append(b.split(".")[0])
    counts = collections.Counter(names)
    lines = [f"  {n}" + (f"  ({c} panels)" if c > 1 else "")
             for n, c in list(counts.items())[:limit]]
    if len(counts) > limit:
        lines.append(f"  ... and {len(counts) - limit} more scans.")

    # The shared reason, as a CATEGORY. The raw text names every
    # rejected candidate with its exposure, which is the log's job; what
    # a dialog needs is the one sentence that tells you what to do.
    def _category(why):
        w = str(why).lower()
        if "exposure" in w:
            return ("no dark was found at these scans' exposure — the "
                    "darks that exist were taken at a different one")
        if "no dark-like file" in w:
            return "no dark-like file was found anywhere under the "\
                   "selected folder"
        if "diffraction" in w:
            return ("the only candidates carry diffraction, so they are "
                    "not darks")
        if "carries no ge" in w or "not found" in w:
            return "the matching per-panel dark file does not exist"
        return "no valid dark could be resolved"

    common = collections.Counter(_category(w) for _p, w in skipped)
    why = common.most_common(1)[0][0] if common else ""
    if len(common) > 1:
        why += f"  (+{len(common) - 1} other reason(s) — see the Log)"
    return "\n".join(lines), why, len(counts)
