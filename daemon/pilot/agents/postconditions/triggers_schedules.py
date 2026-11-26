"""Post-conditions for reactive triggers and scheduled tasks."""

from __future__ import annotations

from pilot.actions import ActionResult, ActionType, ScheduleParams
from pilot.agents.postconditions import Outcome, mismatch, register, verified


# No pre-check: a task name that matches nothing must reach the handler and fail there, not be skipped as
# "already satisfied" because nothing by that name exists.
@register(ActionType.SCHEDULE_DELETE, method="schedule_absence_postcondition")
async def schedule_absent(result: ActionResult) -> Outcome:
    """The task is gone, asked of the platform's scheduler (schtasks, launchctl, crontab) afresh."""
    from pilot.system.scheduler import schedule_exists

    params: ScheduleParams = result.action.parameters  # type: ignore[assignment]
    target = params.task_id or params.name
    if not target:
        raise ValueError("Schedule delete has no task identifier to observe")
    if await schedule_exists(target):
        return mismatch(f"Scheduled task still exists: {target}")
    return verified(f"Scheduled task is absent: {target}")
