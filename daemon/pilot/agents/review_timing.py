"""When the companion's review of a plan may hold up the person, decided from what the plan's steps already carry.

The companion (pilot.agents.execution_companion) asks a model whether the plan serves the request. On 26 November
that took about 3 s of a 7 s note, every turn. Its answer only matters for what it could still prevent:

- ``beside``: every step only reads, has nothing at stake (a level, the window or tab in front, a page loaded by
  address: pilot.agents.step_stakes), or is a write, copy or move inside the person's own folders whose previous
  state Heliox's undo journal (pilot.workflows.undo_journal) will keep. Nothing here can hurt, so the plan runs at
  once and the review runs beside it; a concern it raises arrives afterwards as a remark in the window.
- ``with_card``: the plan asks for approval anyway (a step that needs it, the world model's pause, an MCP client's
  plan). The review runs while the card is open, which costs the person nothing; a concern that arrives while the
  card is open is shown beside it, and a STOP or REVISE still holds once the person approves.
- ``before``: any other plan that would run without a card. It waits for the review, at most
  ``BEFORE_LIMIT_SECONDS`` (longer only while the safety critic still reviews the same plan). A review that does not
  answer is no evidence the plan is right, so the plan then asks the person first (``UNANSWERED_REASON``): a failed
  review is never read as approval of something that could hurt.

What the journal keeps, and so what counts as "can be put back" here (``_capture_file``, ``_keep_copy``):

- a write: the file's earlier content when it exists (a copy), or nothing when the file is new (undo deletes it);
- a copy or a move: the file it lands on when one exists (a copy), and where it came from;
- the kept copies of one task together may not pass ``undo.max_task_mb``; a capture that fails (too large, inside
  Heliox's own folders, a folder in the way) does not stop the step, which then runs without undo, except for a step
  that runs beside its review because of that copy: the executor then stops it before it changes anything
  (``kept_copy_required``).

So a step is ``beside`` only when undo is on, every path it touches is inside the person's home folder (not a system
or program folder, another user's folder, a network share or a removable drive) and outside Heliox's own folders,
it acts on one file (not a folder), nothing it would replace is a folder, the files it would replace fit in the
task's undo limit together, and a move stays on one drive. A clipboard write is not ``beside``: the journal keeps
the clipboard's earlier text in memory only (not an image, and not across a restart).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from enum import StrEnum
from pathlib import Path
from typing import Any

from pilot.actions import READ_ONLY_ACTIONS, ActionType, PermissionTier
from pilot.agents.step_stakes import Stake, stake_of, step_stake

# The longest a risky plan without a card waits for the review on its own. On 26 November the small cloud model
# answered in 3.0 s; the configured limit (narration.companion_timeout_seconds) applies when it is shorter.
BEFORE_LIMIT_SECONDS = 4.0
UNANSWERED_REASON = "The double-check of this plan did not answer in time, so Heliox asks you before it runs."

_FILE_CHANGES = frozenset({ActionType.FILE_WRITE, ActionType.FILE_COPY, ActionType.FILE_MOVE})


class ReviewTiming(StrEnum):
    BESIDE = "beside"
    WITH_CARD = "with_card"
    BEFORE = "before"


def _is_read(action: Any) -> bool:
    tier = action.permission_tier
    # A file read is labelled USER_WRITE by the permission tiers for historical reasons; the read set says what reads.
    return tier == PermissionTier.READ_ONLY or (
        action.action_type in READ_ONLY_ACTIONS and tier <= PermissionTier.USER_WRITE
    )


def _home() -> Path:
    return Path.home()


def _device(path: Path) -> int | None:
    """The device of ``path`` or of its nearest existing folder; None when it cannot be read."""
    for candidate in (path, *path.parents):
        try:
            return os.stat(candidate).st_dev
        except OSError:
            continue
    return None


def _private(path: Path) -> bool:
    from pilot.security.validator import inside_heliox_private_dirs

    return inside_heliox_private_dirs(str(path))


class _Probe:
    """What one plan's file changes would leave the undo journal to keep, checked before anything runs."""

    def __init__(
        self,
        undo: Any,
        *,
        home: Path | None,
        device_of: Callable[[Path], int | None],
        private: Callable[[Path], bool],
    ) -> None:
        self.enabled = bool(getattr(undo, "enabled", False))
        self.budget = int(getattr(undo, "max_task_mb", 0) or 0) * 1024 * 1024
        self.home = Path(os.path.realpath(home or _home()))
        self.device_of = device_of
        self.private = private
        self.kept = 0  # bytes the journal would copy for the plan so far

    def own(self, path: Path) -> bool:
        """Inside the person's home folder and outside Heliox's own: not a system, program, other user's, network
        or removable location (those are never under the home folder on Windows, macOS or Linux)."""
        text = str(path)
        if text.startswith(("\\\\", "//")):
            return False  # a network share
        try:
            resolved = Path(os.path.realpath(path))
        except (OSError, ValueError):
            return False
        try:
            resolved.relative_to(self.home)
        except ValueError:
            return False
        return not self.private(resolved)

    def keep(self, path: Path) -> bool:
        """The journal must copy ``path`` (it will be replaced): it must be one file within the undo limit."""
        if path.is_dir():
            return False  # the journal refuses a folder in the way ("the destination folder already exists")
        if not path.is_file():
            return True  # nothing there: nothing to keep
        try:
            self.kept += path.stat().st_size
        except OSError:
            return False
        return self.kept <= self.budget

    def can_put_back(self, action: Any) -> bool:
        params = action.parameters
        raw = str(getattr(params, "path", "") or action.target or "")
        if not raw:
            return False
        path = Path(os.path.abspath(raw))
        if action.action_type == ActionType.FILE_WRITE:
            return self.own(path) and not path.is_dir() and self.keep(path)
        destination_raw = str(getattr(params, "destination", "") or "")
        if not destination_raw or not path.is_file():
            return False  # a folder copied or moved, or a source not there: not this class
        destination = Path(os.path.abspath(destination_raw))
        final = destination / path.name if destination.is_dir() else destination
        if not (self.own(path) and self.own(final)):
            return False
        if action.action_type == ActionType.FILE_MOVE:
            source_device, final_device = self.device_of(path), self.device_of(final.parent)
            if source_device is None or source_device != final_device:
                return False  # across drives a move is a copy and a delete
        return self.keep(final)


def step_cannot_hurt(action: Any, probe: _Probe) -> bool:
    """A read, a step with nothing at stake, or a write, copy or move whose previous state the undo journal will
    keep (pilot.agents.step_stakes)."""
    if action.requires_root or action.dangerous_flags or action.is_irreversible:
        return False
    stake = step_stake(action)
    if stake is Stake.READ:
        return _is_read(action)
    if stake is Stake.NOTHING_AT_STAKE:
        return True
    if (
        stake is not Stake.JOURNAL_KEEPS
        or not probe.enabled
        or action.action_type not in _FILE_CHANGES
        or action.permission_tier > PermissionTier.USER_WRITE
    ):
        return False
    try:
        return probe.can_put_back(action)
    except (OSError, ValueError):
        return False


def review_timing(
    plan: Any,
    *,
    card_anyway: bool,
    undo: Any,
    home: Path | None = None,
    device_of: Callable[[Path], int | None] = _device,
    private: Callable[[Path], bool] = _private,
) -> ReviewTiming:
    """Reads the file system (sizes, folders, devices) for a plan of file changes: call it off the event loop."""
    if card_anyway:
        return ReviewTiming.WITH_CARD
    probe = _Probe(undo, home=home, device_of=device_of, private=private)
    if all(step_cannot_hurt(action, probe) for action in plan.actions):
        return ReviewTiming.BESIDE
    return ReviewTiming.BEFORE


# Steps that run beside their review only because the undo journal will keep the earlier file: the executor stops
# such a step when the journal then cannot keep its copy (a full disk, an earlier file it cannot read). Set by the
# server for the plan it is about to run; every other step keeps the journal's own rule (a failed copy is listed as
# not undoable and the step runs).
_KEPT_COPY_REQUIRED: ContextVar[frozenset[str]] = ContextVar("kept_copy_required", default=frozenset())

KEPT_COPY_FAILED_SKIP = "no_undo_copy"


def _step_key(action: Any) -> str:
    # Where the step writes, not its content: the executor may put an earlier step's output into the content.
    params = action.parameters
    return "|".join(
        (
            action.action_type.value,
            str(getattr(params, "path", "") or ""),
            str(getattr(params, "destination", "") or ""),
        )
    )


def require_kept_copies(actions: Iterable[Any]) -> None:
    """Mark the plan's journal-kept steps (those whose class is ``journal_keeps``) for the executor."""
    _KEPT_COPY_REQUIRED.set(
        frozenset(_step_key(action) for action in actions if stake_of(action.action_type) is Stake.JOURNAL_KEEPS)
    )


def kept_copy_required(action: Any) -> bool:
    keys = _KEPT_COPY_REQUIRED.get()
    return bool(keys) and _step_key(action) in keys


def kept_copy_failed_sentence(action: Any) -> str:
    """What the person reads for a step stopped because the journal could not keep the earlier file."""
    verb = "overwrite" if action.action_type == ActionType.FILE_WRITE else "replace"
    return f"Heliox could not keep a copy of the earlier file, so it did not {verb} it."


__all__ = [
    "BEFORE_LIMIT_SECONDS",
    "KEPT_COPY_FAILED_SKIP",
    "UNANSWERED_REASON",
    "ReviewTiming",
    "kept_copy_failed_sentence",
    "kept_copy_required",
    "require_kept_copies",
    "review_timing",
    "step_cannot_hurt",
]
