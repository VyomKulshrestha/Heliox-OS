"""Post-conditions for reactive triggers and scheduled tasks.

Triggers live only in the daemon: the trigger engine's table is the one place a
trigger exists and the only thing its evaluation loop reads, so the checks read that
table afresh through ``get_engine()`` (never the handler's returned text) and judge
"running" from the loop's asyncio task, not from the flag the start and stop calls set.

Scheduled tasks are read back from the platform's scheduler: Windows through the Task
Scheduler service's COM interface, macOS from the plist file and launchd, Linux from the
installed crontab (``pilot.system.scheduler.observe_schedule``). No check runs a
trigger's command or a scheduled command.
"""

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


def _command_key(command: str, *, windows: bool) -> str:
    """A command compared the way each scheduler may store it: runs of spaces collapsed;
    on Windows quotes dropped and case ignored, because Task Scheduler splits the command
    into a program path and its arguments and may keep or drop the quotes around the path."""
    text = command.replace('"', "") if windows else command
    key = " ".join(text.split())
    return key.casefold() if windows else key


@register(ActionType.SCHEDULE_CREATE, method="schedule_definition_postcondition")
async def schedule_present(result: ActionResult) -> Outcome:
    """The task exists in the platform's scheduler and runs the requested command."""
    from pilot.system import scheduler
    from pilot.system.platform_detect import Platform

    params: ScheduleParams = result.action.parameters  # type: ignore[assignment]
    target = params.name
    if not target:
        raise ValueError("Schedule create has no task name to observe")
    observed = await scheduler.observe_schedule(target)
    if observed is None:
        return mismatch(f"Scheduled task {target} does not exist after it was created")
    if observed.command is None:
        return mismatch(f"Scheduled task {target} has a definition the scheduler cannot read")
    if not observed.loaded:
        return mismatch(f"Scheduled task {target} has a definition file but launchd has not loaded it")
    windows = scheduler.CURRENT_PLATFORM == Platform.WINDOWS
    if _command_key(observed.command, windows=windows) != _command_key(params.command, windows=windows):
        return mismatch(f"Scheduled task {target} runs a different command from the one requested")
    if observed.schedule and " ".join(params.schedule.split()) != observed.schedule:
        return mismatch(
            f"Scheduled task {target} runs on {observed.schedule!r}, not on the requested {params.schedule!r}"
        )
    return verified(f"Scheduled task {target} is in the scheduler with the requested command")
