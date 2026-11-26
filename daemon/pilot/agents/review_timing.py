"""When the companion's review of a plan may hold up the person, decided from what the plan's steps already carry.

The companion (pilot.agents.execution_companion) asks a model whether the plan serves the request. On 26 November
that took about 3 s of a 7 s note, every turn. Its answer only matters for what it could still prevent:

- ``beside``: every step only reads, or is a local change Heliox's undo can put back (a write, copy or move of the
  person's files, the clipboard, the volume) and is not a delete. Nothing here can hurt, so the plan runs at once and
  the review runs beside it; a concern it raises arrives afterwards as a remark in the window.
- ``with_card``: the plan asks for approval anyway (a step that needs it, the world model's pause, an MCP client's
  plan). The review runs while the card is open, which costs the person nothing; a concern that arrives while the
  card is open is shown beside it, and a STOP or REVISE still holds once the person approves.
- ``before``: any other plan (a step that reaches outside the computer, deletes, cannot be undone, or acts on
  another application) that would run without a card. It waits for the review, at most ``BEFORE_LIMIT_SECONDS``
  (longer only while the safety critic still reviews the same plan). A review that does not answer is no evidence
  the plan is right, so the plan then asks the person first (``UNANSWERED_REASON``): a failed review is never read as
  approval of something that could hurt.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pilot.actions import READ_ONLY_ACTIONS, ActionType, PermissionTier

# The longest a risky plan without a card waits for the review on its own. On 26 November the small cloud model
# answered in 3.0 s; the configured limit (narration.companion_timeout_seconds) applies when it is shorter.
BEFORE_LIMIT_SECONDS = 4.0
UNANSWERED_REASON = "The double-check of this plan did not answer in time, so Heliox asks you before it runs."


class ReviewTiming(StrEnum):
    BESIDE = "beside"
    WITH_CARD = "with_card"
    BEFORE = "before"


def step_cannot_hurt(action: Any, *, undo_enabled: bool) -> bool:
    """A read, or a local change that Heliox's undo can put back and that deletes nothing."""
    if action.requires_root or action.dangerous_flags or action.is_irreversible:
        return False
    tier = action.permission_tier
    # A file read is labelled USER_WRITE by the permission tiers for historical reasons; the read set says what reads.
    if tier == PermissionTier.READ_ONLY or (
        action.action_type in READ_ONLY_ACTIONS and tier <= PermissionTier.USER_WRITE
    ):
        return True
    if not undo_enabled or tier > PermissionTier.USER_WRITE or action.action_type == ActionType.FILE_DELETE:
        return False
    from pilot.workflows.undo_journal import undo_support

    return undo_support(action).kind == "undoable"


def review_timing(plan: Any, *, card_anyway: bool, undo_enabled: bool) -> ReviewTiming:
    if card_anyway:
        return ReviewTiming.WITH_CARD
    if all(step_cannot_hurt(action, undo_enabled=undo_enabled) for action in plan.actions):
        return ReviewTiming.BESIDE
    return ReviewTiming.BEFORE


__all__ = ["BEFORE_LIMIT_SECONDS", "UNANSWERED_REASON", "ReviewTiming", "review_timing", "step_cannot_hurt"]
