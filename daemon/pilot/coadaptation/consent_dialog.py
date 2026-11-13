"""The lines of the daemon's approval dialog for a co-adaptation consent grant.

The window's consent sheet sends what is being asked for; the person's answer in this dialog is the consent. The
words are the ones the consent sheet shows (``tauri-app/ui/src/locales/coadaptation.en.json``, section
``coadaptation.consent``), so the dialog promises neither less nor more than the text the person read. A test
compares the two.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pilot.coadaptation.core.consent import CoAdaptationPurpose

# Label and detail of each purpose, word for word as the consent sheet shows them.
PURPOSE_WORDS: dict[CoAdaptationPurpose, tuple[str, str]] = {
    CoAdaptationPurpose.SELECTION_STUDY: (
        "Run the selection study",
        "Store study trials, Heliox's choices and your ratings.",
    ),
    CoAdaptationPurpose.ADAPTIVE_ASSISTANCE: (
        "Let Heliox adapt during later sessions",
        "From your second session, Heliox picks help using its current estimate for you, within your rules. For "
        "comparison, some blocks of those sessions use standard selection, and some use the help that scored best in "
        "your first session's study results.",
    ),
    CoAdaptationPurpose.CROSS_SESSION_LEARNING: (
        "Remember across sessions",
        "Keep what Heliox learned about your selections from one day to the next. Without this, every session starts "
        "fresh.",
    ),
    CoAdaptationPurpose.GAZE_PRIOR: (
        "Use coarse gaze direction",
        "Use the left/right/up/down/centre gaze region as a hint about which target you mean. Gaze never selects "
        "anything. While a study session runs, gaze stays inside the study: it is not sent to Heliox's activity log. "
        "Only the gaze region and how sure the tracker was at each selection, and your gaze-check score, are stored, "
        "with your study data.",
    ),
    CoAdaptationPurpose.HAND_POINTER: (
        "Use the hand pointer",
        "Point with your hand inside the study area. Only hand-tracking quality summaries are stored.",
    ),
    CoAdaptationPurpose.RESEARCH_EXPORT: (
        "Allow encrypted export",
        "Let you save an encrypted copy of your study data on this device for analysis.",
    ),
}

# The consent sheet's "Where it is kept" paragraph, without its heading.
WHERE_KEPT = (
    "Only on this device, encrypted with a key held by your operating system's secure storage. Nothing is uploaded. "
    "Anyone who uses Heliox on this device can open Settings and view or erase it."
)

_UNDO = {"kind": "undoable", "reason": "Withdrawing in Settings erases it."}


def _days(count: int) -> str:
    return f"{count} day" if count == 1 else f"{count} days"


def _date(moment: datetime) -> str:
    local = moment.astimezone()
    return f"{local:%B} {local.day}, {local.year}"


def consent_lines(
    purposes: list[CoAdaptationPurpose],
    retention_days: int,
    on_file: tuple[frozenset[str], datetime] | None,
) -> list[dict[str, Any]]:
    """One approval line per purpose the grant adds, then where it is kept and until when.

    ``on_file`` is, for an update, the participant's granted purposes and the erase date the update would set
    (``CoAdaptationService.consent_on_file``); None for a new participant. An update that adds no purpose asks nothing: it can only remove purposes or bring the erase date
    nearer, never extend it.
    """
    granted = on_file[0] if on_file is not None else frozenset()
    added = [purpose for purpose in PURPOSE_WORDS if purpose in purposes and purpose.value not in granted]
    if not added:
        return []
    lines: list[dict[str, Any]] = [
        {"action_type": "record_data", "target": PURPOSE_WORDS[purpose][0], "description": PURPOSE_WORDS[purpose][1]}
        for purpose in added
    ]
    if on_file is None:
        target = f"Keep it on this computer for {_days(retention_days)}"
        erased = f"Your data is erased automatically after {_days(retention_days)}, or immediately when you withdraw."
    else:
        until = on_file[1]
        target = f"Keep it with your study data until {_date(until)}"
        erased = f"It is erased with the rest of your study data on {_date(until)}, or immediately when you withdraw."
    lines.append({"action_type": "record_data", "target": target, "description": f"{WHERE_KEPT} {erased}"})
    for line in lines:
        line.update({"irreversible": False, "undo": dict(_UNDO)})
    return lines
