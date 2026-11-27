"""What the world model learns from finished steps, and when that may ask the person to approve a plan.

An approval card is the one moment Heliox asks for a person's attention on safety, so learned evidence may only add
one for harm, never for failure alone. Every finished step gets one outcome:

- ``failed_unchanged``: the step failed and changed nothing (the source was missing, no permission, refused, not
  found, rejected before it ran). Harmless: the person heard the failure in its own reply.
- ``failed_partial``: the step failed after it started and its cause is not one that stops before anything is
  written, so it may have left something changed or half done. Harm.
- ``mismatch``: the step reported success and its check then read something else (pilot.agents.unconfirmed). Harm.
- ``undone``: the person undid the step afterwards. Harm: the result was not what they wanted.
- ``verified``: the step reported success and its check confirmed it. Evidence the other way.
- ``approved``: the person approved the step on a card. Evidence the other way: they saw it and said yes.
- ``unchecked``: the step reported success and nothing could confirm it. Neither way.

Evidence is about the same thing: one action type in one place (the folder a file lands in, a site's host, a
repository, else the target itself). Four failed copies of missing files say nothing about copying a file that
exists, and two mismatched copies onto one USB stick say nothing about copies into Documents. Old evidence fades
with a half-life, so a bad afternoon does not follow the person for weeks.

A pause needs at least ``MINIMUM_HARMFUL`` harmful outcomes in the scope (their faded weight at least
``MINIMUM_HARM_WEIGHT``) and harm outweighing the verified runs and approvals there. The sentence it gives names what
happened in the person's terms ("the last two copies into this folder ended with a different file than the source").

A run of harmless failures pauses nothing. It is kept as a note on the assessment (shown with the world model's
details, never on a card): each failure was already said in its own reply, and a card before the next attempt would
only repeat it.

The evidence lives in memory for the daemon's lifetime, as the callback counts it replaces did; nothing is written
to disk.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from pilot.actions import ActionType

# The score a pause carries: above the approval threshold of 0.3 (pilot.agents.destructive_critic), so a learned
# pause asks exactly as any other world-model pause does, and below the 0.9 of an irreversible step.
PAUSE_SCORE = 0.35
MAXIMUM_SCORE = 0.85
MINIMUM_HARMFUL = 2
MINIMUM_HARM_WEIGHT = 1.5
HALF_LIFE_SECONDS = 3 * 24 * 3600.0
# Harmless failures in one scope before the assessment carries a note about them.
NOTE_AFTER_HARMLESS = 3
MAX_RECORDS_PER_SCOPE = 50
MAX_SCOPES = 500


class Outcome(StrEnum):
    FAILED_UNCHANGED = "failed_unchanged"
    FAILED_PARTIAL = "failed_partial"
    MISMATCH = "mismatch"
    UNDONE = "undone"
    VERIFIED = "verified"
    APPROVED = "approved"
    UNCHECKED = "unchecked"


HARMFUL: frozenset[Outcome] = frozenset({Outcome.FAILED_PARTIAL, Outcome.MISMATCH, Outcome.UNDONE})
SUPPORTING: frozenset[Outcome] = frozenset({Outcome.VERIFIED, Outcome.APPROVED})

# Failure causes that stop a step before it writes anything, each with the few words a note uses for it, first match
# wins. Anything else that failed after the step started counts as possibly half done.
_UNCHANGED_CAUSES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), words)
    for pattern, words in (
        (r"permission denied|access (?:is )?denied|not permitted|errno 13\b|winerror 5\b|errno 1\b", "no permission"),
        (r"read-only|read only", "read-only"),
        (
            r"not found|no such file|does not exist|cannot find|could not find|no file at|errno 2\b|winerror [23]\b",
            "not found",
        ),
        (r"winerror 32\b|being used by another process", "in use by another program"),
        (r"refused|not allowed|blocked", "refused"),
        (r"validation failed|invalid|empty target", "rejected before it ran"),
        (
            r"already exists|is not a (?:file|directory|folder)|is a directory|not a git repository|nothing to",
            "nothing to do there",
        ),
    )
)

_FILE_ACTIONS = frozenset(
    {
        ActionType.FILE_WRITE,
        ActionType.FILE_COPY,
        ActionType.FILE_MOVE,
        ActionType.FILE_DELETE,
        ActionType.FILE_PERMISSIONS,
        ActionType.GIT_RESOLVE,
        ActionType.DOWNLOAD_FILE,
    }
)
_GIT_ACTIONS = frozenset(
    {ActionType.GIT_BRANCH, ActionType.GIT_STAGE, ActionType.GIT_COMMIT, ActionType.GIT_PUSH, ActionType.GIT_RESOLVE}
)

# How a sentence names the steps of one action type: (plural noun, place words).
_NOUNS: dict[ActionType, tuple[str, str]] = {
    ActionType.FILE_COPY: ("copies", "into this folder"),
    ActionType.FILE_MOVE: ("moves", "into this folder"),
    ActionType.FILE_WRITE: ("writes", "in this folder"),
    ActionType.FILE_DELETE: ("deletes", "in this folder"),
    ActionType.DOWNLOAD_FILE: ("downloads", "into this folder"),
    ActionType.GIT_COMMIT: ("commits", "in this repository"),
    ActionType.GIT_PUSH: ("pushes", "from this repository"),
    ActionType.GIT_STAGE: ("stagings", "in this repository"),
}
_WHAT_WENT_WRONG: dict[Outcome, str] = {
    Outcome.MISMATCH: "ended differently from what was asked when Heliox checked them",
    Outcome.FAILED_PARTIAL: "stopped part way and may have left something half done",
    Outcome.UNDONE: "were undone by you afterwards",
}
_COPY_MISMATCH = "ended with a different file than the source"
_NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


@dataclass(frozen=True)
class Scope:
    action_type: str
    place: str


@dataclass(frozen=True)
class Assessment:
    """What the evidence in one scope says: a pause (score above zero, with its sentence) or a note, or nothing."""

    score: float = 0.0
    reason: str = ""
    note: str = ""
    harmful: int = 0
    supporting: int = 0


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)) if path else ""


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def scope_of(action: Any) -> Scope:
    """The scope a step's evidence belongs to: its action type and where it acts."""
    action_type = ActionType(action.action_type)
    params = getattr(action, "parameters", None)
    target = str(getattr(action, "target", "") or "").strip()
    place = ""
    if action_type in _GIT_ACTIONS:
        place = _norm_path(str(getattr(params, "repo_path", "") or getattr(params, "path", "") or target))
    elif action_type in _FILE_ACTIONS:
        destination = str(getattr(params, "destination", "") or "")
        path = destination or str(getattr(params, "path", "") or target)
        if path:
            place = os.path.dirname(_norm_path(path))
    else:
        url = str(getattr(params, "url", "") or "")
        if url or target.startswith(("http://", "https://")):
            place = _host(url or target)
        else:
            place = target.lower()[:200]
    return Scope(action_type.value, place)


def failure_outcome(result: Any) -> tuple[Outcome, str]:
    """The outcome of a step that failed, and its cause in a few words."""
    error = str(getattr(result, "error", "") or "")
    if not getattr(result, "executed", True):
        return Outcome.FAILED_UNCHANGED, "stopped before it ran"
    for pattern, words in _UNCHANGED_CAUSES:
        if pattern.search(error):
            return Outcome.FAILED_UNCHANGED, words
    return Outcome.FAILED_PARTIAL, ""


def outcomes_of(results: Sequence[Any], verification: Any) -> list[tuple[Any, Outcome]]:
    """The outcome of every step of a finished plan that reported success, from its check (failed steps are
    recorded as they fail, by ``failure_outcome``). Steps that were skipped or simulated are left out."""
    failed = set(getattr(verification, "failed_actions", None) or [])
    observed = set(getattr(verification, "independently_observed_actions", None) or [])
    pending = set(getattr(verification, "pending_actions", None) or [])
    outcomes: list[tuple[Any, Outcome]] = []
    for index, result in enumerate(results):
        if not getattr(result, "success", False) or not getattr(result, "executed", True):
            continue
        if str(getattr(result, "output", "") or "").startswith("(dry run)") or index in pending:
            continue
        if index in failed:
            outcomes.append((result.action, Outcome.MISMATCH))
        elif index in observed:
            outcomes.append((result.action, Outcome.VERIFIED))
        else:
            outcomes.append((result.action, Outcome.UNCHECKED))
    return outcomes


def _count_words(count: int) -> str:
    return _NUMBERS.get(count, str(count))


def _sentence(action_type: str, outcome: Outcome, count: int, *, in_a_row: bool) -> str:
    kind = ActionType(action_type)
    noun, where = _NOUNS.get(kind, (f"{kind.value.replace('_', ' ')} steps", "here"))
    what = _COPY_MISMATCH if kind == ActionType.FILE_COPY and outcome == Outcome.MISMATCH else _WHAT_WENT_WRONG[outcome]
    # A number only when it is exact (``count`` 0: some of the scope's harmful outcomes were dropped by its cap).
    words = _count_words(count) if count else "several"
    lead = f"The last {words}" if in_a_row else f"{words.capitalize()} recent"
    return f"{lead} {noun} {where} {what}."


@dataclass
class _Record:
    outcome: Outcome
    at: float
    cause: str
    task: str = ""  # the task the step belonged to ("" when unknown)
    step: str = ""  # the step within its task: its action and where it writes (``step_of``)


def step_of(action: Any) -> str:
    """The identity of a step within its task: a repair round's re-run of the same copy is the same step."""
    action_type = ActionType(action.action_type).value
    params = getattr(action, "parameters", None)
    where = str(
        getattr(params, "destination", "") or getattr(params, "path", "") or getattr(action, "target", "") or ""
    )
    return f"{action_type}|{_norm_path(where) if where else ''}"


def _current_task() -> str:
    from pilot.intelligence.experience import get_experience_context

    return get_experience_context().task_id or ""


def _direction(outcome: Outcome) -> str:
    if outcome in HARMFUL:
        return "harm"
    if outcome in SUPPORTING:
        return "support"
    return outcome.value


class OutcomeEvidence:
    """Outcomes of finished steps per scope, and what they say about the next step in the same scope."""

    def __init__(self, *, clock: Callable[[], float] = time.time, half_life: float = HALF_LIFE_SECONDS) -> None:
        self._clock = clock
        self._half_life = max(1.0, half_life)
        self._scopes: OrderedDict[Scope, deque[_Record]] = OrderedDict()
        self._dropped_harm: set[Scope] = set()
        self._lock = threading.Lock()

    def record(self, action: Any, outcome: Outcome | str, *, cause: str = "", task: str | None = None) -> None:
        """One step's outcome. Within one task a step is one piece of evidence each way, however many paths report
        it (the check, a late check, a repair round's re-run, an approval, an undo): a newer report of the same kind
        replaces the earlier one."""
        scope = scope_of(action)
        new = _Record(
            Outcome(outcome), self._clock(), cause, _current_task() if task is None else task, step_of(action)
        )
        with self._lock:
            records = self._scopes.pop(scope, None) or deque(maxlen=MAX_RECORDS_PER_SCOPE)
            if new.task:
                same = [
                    record
                    for record in records
                    if record.task == new.task
                    and record.step == new.step
                    and _direction(record.outcome) == _direction(new.outcome)
                ]
                for record in same:
                    records.remove(record)
            if len(records) == records.maxlen and records[0].outcome in HARMFUL:
                self._dropped_harm.add(scope)  # counts in this scope are no longer exact
            records.append(new)
            self._scopes[scope] = records  # most recent last
            while len(self._scopes) > MAX_SCOPES:
                dropped, _records = self._scopes.popitem(last=False)
                self._dropped_harm.discard(dropped)

    def record_many(self, outcomes: Iterable[tuple[Any, Outcome]]) -> None:
        for action, outcome in outcomes:
            self.record(action, outcome)

    def _weight(self, record: _Record, now: float) -> float:
        return math.pow(0.5, max(0.0, now - record.at) / self._half_life)

    def assess(self, action: Any, *, task: str | None = None) -> Assessment:
        """What the evidence says about ``action``. Evidence that this very step produced in this task is left out:
        a repair round re-running a step is not paused by the outcome it is repairing."""
        scope = scope_of(action)
        now = self._clock()
        current = _current_task() if task is None else task
        step = step_of(action)
        with self._lock:
            exact = scope not in self._dropped_harm
            records = [
                record
                for record in self._scopes.get(scope, ())
                if not (current and record.task == current and record.step == step)
            ]
        harmful = [record for record in records if record.outcome in HARMFUL]
        supporting = [record for record in records if record.outcome in SUPPORTING]
        harm_weight = sum(self._weight(record, now) for record in harmful)
        support_weight = sum(self._weight(record, now) for record in supporting)
        if len(harmful) >= MINIMUM_HARMFUL and harm_weight >= MINIMUM_HARM_WEIGHT and harm_weight > support_weight:
            share = harm_weight / (harm_weight + support_weight)
            score = min(MAXIMUM_SCORE, PAUSE_SCORE + 0.4 * share * min(1.0, harm_weight / 4.0))
            # The sentence names the latest harmful outcomes of one kind, counted back from the newest.
            latest = harmful[-1].outcome
            run = 0
            for record in reversed(harmful):
                if record.outcome != latest:
                    break
                run += 1
            if run < MINIMUM_HARMFUL:
                # Mixed kinds (a mismatch, then one undone): one sentence for all of them.
                count = _count_words(len(harmful)) if exact else "several"
                reason = (
                    f"{count.capitalize()} recent {_plain_steps(scope.action_type)} here did not turn out as asked."
                )
            else:
                in_a_row = all(record.outcome == latest for record in records[-run:])
                reason = _sentence(scope.action_type, latest, run if exact else 0, in_a_row=in_a_row and exact)
            return Assessment(round(score, 4), reason, "", len(harmful), len(supporting))
        harmless = [record for record in records if record.outcome == Outcome.FAILED_UNCHANGED]
        note = ""
        recent = records[-NOTE_AFTER_HARMLESS:]
        if len(harmless) >= NOTE_AFTER_HARMLESS and all(r.outcome == Outcome.FAILED_UNCHANGED for r in recent):
            causes = sorted({record.cause for record in recent if record.cause})
            because = f" ({', '.join(causes)})" if causes else ""
            note = (
                f"The last {_count_words(len(recent))} {_plain_steps(scope.action_type)} here failed before changing "
                f"anything{because}; that alone does not ask for approval."
            )
        return Assessment(0.0, "", note, len(harmful), len(supporting))

    def status(self) -> dict[str, Any]:
        with self._lock:
            records = [record for scope in self._scopes.values() for record in scope]
        counts = {outcome.value: 0 for outcome in Outcome}
        for record in records:
            counts[record.outcome.value] += 1
        return {
            "scopes": len(self._scopes),
            "samples": len(records),
            "outcomes": counts,
            "minimum_harmful": MINIMUM_HARMFUL,
            "half_life_days": round(self._half_life / 86400.0, 2),
        }


def _plain_steps(action_type: str) -> str:
    kind = ActionType(action_type)
    return _NOUNS.get(kind, (f"{kind.value.replace('_', ' ')} steps", ""))[0]


__all__ = [
    "HARMFUL",
    "MINIMUM_HARMFUL",
    "PAUSE_SCORE",
    "SUPPORTING",
    "Assessment",
    "Outcome",
    "OutcomeEvidence",
    "Scope",
    "failure_outcome",
    "outcomes_of",
    "scope_of",
    "step_of",
]
