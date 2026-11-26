"""What Heliox says, shows and keeps about a step whose outcome was not plainly confirmed: one table, here.

Every step that ran and was not plainly verified gets a kind:

- ``not_observable``: by nature nothing can be read (no fixed target such as a volume step or a scroll, a platform
  that cannot be read such as Wayland or a field without an accessibility reader, the default browser that is not
  Heliox's, an action with no check at all);
- ``nothing_happened``: there was something to look at and nothing visible changed (a click on a video);
- ``changed_later``: a later step, or the person, changed what the check would have read;
- ``could_not_look``: a check was expected and could have looked but did not (a time limit, a server that did not
  answer, a sign-in refused, a missing setting such as no IMAP for mail read-back, a comparison cut short);
- ``receipt_only``: the service accepted it and no read-back exists (a webhook that answers "ok");
- ``still_checking``: the check is still running when the reply goes out; its outcome reaches the receipt later;
- ``mismatch``: the check read the state and it is not what was asked.

What reaches the person from each kind:

- **Spoken:** a mismatch, and ``could_not_look`` or ``receipt_only`` when the step sends something to other people
  or cannot be undone (``CONSEQUENTIAL_ACTIONS``). Each kind has one short fixed phrase (``SPOKEN_PHRASES``) that never
  holds the verifier's detail, a window title, a file name, an address or a count.
- **Shown in the window:** at most one line for the whole reply (``window_line``), counting the kinds in
  ``COUNTED_KINDS``. The kinds that are by nature not observable are not counted.
- **Kept in the receipt and the timeline:** every reason in full, with window titles and file names reduced to the
  application's name and a short masked form (``mask``).

The verifier's own detail is never spoken. Change the tables below to tune what is said or shown.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pilot.actions import IRREVERSIBLE_ACTIONS, ActionType


class Kind(StrEnum):
    NOT_OBSERVABLE = "not_observable"
    NOTHING_HAPPENED = "nothing_happened"
    CHANGED_LATER = "changed_later"
    COULD_NOT_LOOK = "could_not_look"
    RECEIPT_ONLY = "receipt_only"
    STILL_CHECKING = "still_checking"
    MISMATCH = "mismatch"


# Seconds the reply waits for the checks of a plan, all together. A check still running then goes on behind the reply
# (bounded by its own time limit) and its outcome reaches the receipt when it arrives.
VERIFY_BUDGET_SECONDS = 3.0
# Added to the budget for the last read of a check whose own wait ends inside it (the app window check counts
# its 3 s from the launch): that read's few milliseconds must not push it to "still being checked".
VERIFY_GRACE_SECONDS = 0.5

# Steps that send something to other people or cannot be undone: a doubt about these is worth saying aloud.
CONSEQUENTIAL_ACTIONS: frozenset[ActionType] = frozenset(
    {
        *IRREVERSIBLE_ACTIONS,
        ActionType.FILE_DELETE,
        ActionType.CALENDAR_CREATE_EVENT,
        ActionType.CALENDAR_DELETE_EVENT,
    }
)

SPOKEN_PHRASES: dict[Kind, str] = {
    Kind.MISMATCH: "The check afterwards found something else",
    Kind.COULD_NOT_LOOK: "I couldn't check that it went through",
    Kind.RECEIPT_ONLY: "It was accepted, but I can't read it back",
}
SPOKEN_ALWAYS: frozenset[Kind] = frozenset({Kind.MISMATCH})
SPOKEN_WHEN_CONSEQUENTIAL: frozenset[Kind] = frozenset({Kind.COULD_NOT_LOOK, Kind.RECEIPT_ONLY})

# Counted in the window's one line. Not counted: what by nature cannot be read, and checks still running (said apart).
COUNTED_KINDS: frozenset[Kind] = frozenset(
    {Kind.NOTHING_HAPPENED, Kind.CHANGED_LATER, Kind.COULD_NOT_LOOK, Kind.RECEIPT_ONLY, Kind.MISMATCH}
)

# Said later, in one sentence, when a check that ran behind the reply finds a mismatch on a consequential step.
LATE_MISMATCH_SENTENCES: dict[ActionType, str] = {
    ActionType.API_SEND_EMAIL: "I checked the mail I sent: its copy in Sent is not what was sent.",
    ActionType.EMAIL_REPLY: "I checked the reply I sent: its copy in Sent is not what was sent.",
    ActionType.API_SLACK: "I checked the Slack message: it is not in the channel as sent.",
    ActionType.API_DISCORD: "I checked the Discord message: it is not in the channel as sent.",
    ActionType.GIT_PUSH: "I checked the push: the remote does not hold what was pushed.",
    ActionType.PACKAGE_UPDATE: "I checked the upgrade: nothing was upgraded.",
    ActionType.FILE_DELETE: "I checked the delete: the file is still there.",
    ActionType.CALENDAR_CREATE_EVENT: "I checked the calendar: the new event is not there as asked.",
    ActionType.CALENDAR_DELETE_EVENT: "I checked the calendar: the event is still there.",
}
LATE_MISMATCH_DEFAULT = "I checked the last task again: one step did not turn out as asked. Details are in the receipt."

# Kinds for steps whose verifier does not name one, by action: the action's nature decides.
ACTION_KINDS: dict[ActionType, Kind] = {
    ActionType.API_WEBHOOK: Kind.RECEIPT_ONLY,
    ActionType.MOUSE_SCROLL: Kind.NOT_OBSERVABLE,
    ActionType.MOUSE_MOVE: Kind.NOT_OBSERVABLE,
}

# Kinds read from the verifier's detail, first match wins. Several verifiers were written before kinds existed; this
# keeps them in one place instead of in each module. Anything unmatched is ``could_not_look``.
DETAIL_KINDS: tuple[tuple[re.Pattern[str], Kind], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), kind)
    for pattern, kind in (
        # A platform or a window that is never read, whatever else the sentence says (Wayland's reason mentions an
        # accessibility bus that "did not answer").
        (r"wayland|never reads|is never read|a password field", Kind.NOT_OBSERVABLE),
        # Time limits, servers that did not answer, refusals, missing settings: a look was expected.
        (
            r"did not finish within|did not answer|could not be reached|timed out|not set up|"
            r"refused|answered HTTP|failed with|could not check|not verified in full",
            Kind.COULD_NOT_LOOK,
        ),
        (r"\breceipt\b|no read-back|cannot be read back|could not read it back", Kind.RECEIPT_ONLY),
        (
            r"later steps?|the person may have|meanwhile|changed after it was|another page|redirected|"
            r"no longer at that place|a pop-up",
            Kind.CHANGED_LATER,
        ),
        (r"no change was observed|pointer did not move|nothing was open under that name", Kind.NOTHING_HAPPENED),
        (
            r"wayland|cannot be read|cannot be listed|does not expose|no ui automation reader|accessibility|"
            r"no fixed (?:target|point)|media key|no key state|not something heliox knows|only (?:linux|windows) has|"
            r"no posix|session-only|changes nothing|not heliox's|no graphical session|needs (?:xprop|xdotool|wmctrl|"
            r"python-xlib|pyobjc|xclip)|was not read|went through xdotool|no reading was made",
            Kind.NOT_OBSERVABLE,
        ),
    )
)


def classify(action_type: ActionType, detail: str, *, observed: bool, passed: bool, hint: str = "") -> Kind | None:
    """The kind of one step's outcome; None for a plainly verified step."""
    if observed and passed:
        return None
    if not passed:
        return Kind.MISMATCH
    if hint:
        try:
            return Kind(hint)
        except ValueError:
            pass
    if action_type in ACTION_KINDS:
        return ACTION_KINDS[action_type]
    text = str(detail or "")
    for pattern, kind in DETAIL_KINDS:
        if pattern.search(text):
            return kind
    return Kind.COULD_NOT_LOOK


def no_check_kind(action_type: ActionType) -> Kind:
    """A step whose action has no check at all: a doubt only when it sends something or cannot be undone."""
    return Kind.RECEIPT_ONLY if action_type in CONSEQUENTIAL_ACTIONS else Kind.NOT_OBSERVABLE


def spoken(action_type: ActionType, kind: Kind | str | None) -> str:
    """The fixed phrase said aloud for this step, or ""."""
    if kind is None:
        return ""
    kind = Kind(kind)
    if kind in SPOKEN_ALWAYS or (kind in SPOKEN_WHEN_CONSEQUENTIAL and action_type in CONSEQUENTIAL_ACTIONS):
        return SPOKEN_PHRASES.get(kind, "")
    return ""


WINDOW_LINE_END = "; details in the receipt."


def window_line(steps: Sequence[tuple[ActionType, Kind | str]]) -> str:
    """The one line the window shows for a whole reply, or "" when nothing worth knowing was left unconfirmed.

    It starts with the spoken phrase when one step's doubt is said aloud, so the window and the voice agree."""
    kinds = [(action_type, Kind(kind)) for action_type, kind in steps]
    phrase = next((spoken(action_type, kind) for action_type, kind in kinds if spoken(action_type, kind)), "")
    counted = sum(kind in COUNTED_KINDS for _action, kind in kinds)
    checking = sum(kind is Kind.STILL_CHECKING for _action, kind in kinds)
    parts = []
    if phrase:
        others = counted - 1
        parts.append(phrase)
        if others > 0:
            parts.append(f"{others} more step{'s' if others != 1 else ''} could not be checked")
    elif counted:
        parts.append(f"{counted} step{'s' if counted != 1 else ''} could not be checked")
    if checking:
        parts.append(f"{checking} step{'s are' if checking != 1 else ' is'} still being checked")
    if not parts:
        return ""
    return "Note: " + ", ".join(parts) + WINDOW_LINE_END


_WINDOW_LINE = re.compile(r"^Note: (?P<lead>[^\n]+?)" + re.escape(WINDOW_LINE_END) + r"$", re.MULTILINE)


def spoken_from_window_line(text: str) -> tuple[str, str]:
    """(the text without the window line, the phrase to say for it or "")."""
    found = _WINDOW_LINE.search(text)
    if found is None:
        return text, ""
    lead = found.group("lead")
    phrase = next((p for p in SPOKEN_PHRASES.values() if lead.startswith(p)), "")
    rest = (text[: found.start()] + text[found.end() :]).strip()
    return rest, phrase


def speakable_message(text: str, *, limit: int = 300) -> str:
    """A reply read aloud as it stands (no shorter sentence was made for it), cut to ``limit`` characters: the
    window's line becomes its spoken phrase, said after the cut, or nothing."""
    rest, phrase = spoken_from_window_line(text)
    tail = f"{phrase}." if phrase else ""
    room = max(0, limit - len(tail) - 1) if tail else limit
    if len(rest) > room:
        rest = rest[: max(0, room - 3)] + "..."
    return "\n".join(part for part in (rest, tail) if part)


# ── masking for the receipt and the timeline ─────────────────────────

_WINDOW = re.compile(r'"(?P<title>[^"\n]{1,400})" \((?P<app>[^()\n]{1,120})\)')
# A quoted text: in double quotes, or in single quotes that are not apostrophes ("the field's text ... the field's").
_QUOTED = re.compile(r"(?P<q>\")(?P<text>[^\"\n]{7,400})\"|(?<!\w)(?P<s>')(?P<stext>[^'\n]{7,400})'(?!\w)")
_PATH = re.compile(
    r"(?<![\w:/\\])(?:[A-Za-z]:[\\/]|\\\\|~[\\/]|/(?=[^\s/]+/))[^\s\"'<>|;,()]*",
)


def _short(text: str) -> str:
    return text[:3] + "…"


def _masked_name(name: str) -> str:
    stem, dot, suffix = name.rpartition(".")
    if dot and stem and 0 < len(suffix) <= 6 and suffix.isalnum():
        return _short(stem) + "." + suffix
    return _short(name)


def _mask_path(found: re.Match[str]) -> str:
    path = found.group(0).rstrip(".:")
    tail = found.group(0)[len(path) :]
    name = re.split(r"[\\/]", path.rstrip("\\/"))[-1]
    if not name or name.endswith(":"):
        return found.group(0)
    return "…/" + _masked_name(name) + tail


def mask(text: str, *, files: bool = True) -> str:
    """A reason for the receipt and the timeline: a window title becomes its application and a short masked form, a
    path its masked file name, a long quoted text its first characters. ``files=False`` masks window titles only
    (a failure shown to the person who named the file)."""
    text = _WINDOW.sub(lambda m: f'{m.group("app")} window "{_short(m.group("title"))}"', str(text or ""))
    if not files:
        return text
    text = _PATH.sub(_mask_path, text)
    return _QUOTED.sub(_mask_quoted, text)


def _mask_quoted(found: re.Match[str]) -> str:
    if found.group("q"):
        return '"' + _short(found.group("text")) + '"'
    return "'" + _short(found.group("stext")) + "'"


@dataclass(frozen=True)
class Step:
    """One step's entry in the receipt: what it was, its kind and the reason, masked."""

    index: int
    action_type: ActionType
    kind: Kind
    reason: str
    late: bool = False

    def for_receipt(self) -> dict[str, object]:
        return {
            "step": self.index + 1,
            "action_type": self.action_type.value,
            "kind": self.kind.value,
            "reason": mask(self.reason)[:500],
            "late": self.late,
        }


def steps_of(
    action_types: Sequence[ActionType],
    kinds: Mapping[int, str],
    reasons: Mapping[int, str],
    *,
    late: Iterable[int] = (),
) -> list[Step]:
    late_set = set(late)
    return [
        Step(index, action_types[index], Kind(kind), str(reasons.get(index, "")), index in late_set)
        for index, kind in sorted(kinds.items())
        if 0 <= index < len(action_types)
    ]


NO_CHECK_REASON = "No independent check exists for this action; the result is the action's own report"


def receipt_checks(results: Sequence[object], verification: object) -> list[dict[str, object]]:
    """Every step of a finished plan that was not plainly verified, for the receipt: kind and reason, masked. A step
    with no check that by nature cannot be read and has no reason (a read, a volume step) is left out."""
    kinds: Mapping[int, str] = getattr(verification, "reason_kinds", None) or {}
    reasons: Mapping[int, str] = getattr(verification, "reasons", None) or {}
    action_types = [result.action.action_type for result in results]  # type: ignore[attr-defined]
    entries = []
    for step in steps_of(action_types, kinds, reasons):
        if not step.reason and step.kind is Kind.NOT_OBSERVABLE:
            continue
        reason = step.reason or NO_CHECK_REASON
        entries.append(Step(step.index, step.action_type, step.kind, reason, step.late).for_receipt())
    return entries


def masked_verification(verification: object, *, files: bool = True) -> object:
    """A copy of a VerificationResult with its details and reasons masked: window titles always, file names too
    unless ``files`` is False (the window, where the person sees the files they named). The planner's repair round
    keeps the original: it needs the names."""
    copy = verification.model_copy(deep=True)  # type: ignore[attr-defined]
    copy.details = [mask(detail, files=files) for detail in copy.details]
    copy.reasons = {index: mask(reason, files=files) for index, reason in copy.reasons.items()}
    return copy


def late_sentence(action_type: ActionType) -> str:
    return LATE_MISMATCH_SENTENCES.get(action_type, LATE_MISMATCH_DEFAULT)


__all__ = [
    "ACTION_KINDS",
    "CONSEQUENTIAL_ACTIONS",
    "COUNTED_KINDS",
    "DETAIL_KINDS",
    "Kind",
    "LATE_MISMATCH_SENTENCES",
    "SPOKEN_PHRASES",
    "Step",
    "VERIFY_BUDGET_SECONDS",
    "VERIFY_GRACE_SECONDS",
    "classify",
    "late_sentence",
    "mask",
    "no_check_kind",
    "spoken",
    "spoken_from_window_line",
    "steps_of",
    "window_line",
]
