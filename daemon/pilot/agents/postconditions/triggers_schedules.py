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

import json
from typing import Any

from pilot.actions import ActionResult, ActionType, ScheduleParams, TriggerParams
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


# ── Triggers ─────────────────────────────────────────────────────────


def _created_id(result: ActionResult) -> str:
    try:
        record: Any = json.loads(result.output or "")
    except ValueError:
        return ""
    value = record.get("id") if isinstance(record, dict) else None
    return value if isinstance(value, str) else ""


@register(ActionType.TRIGGER_CREATE, method="trigger_present_postcondition")
async def trigger_present(result: ActionResult) -> Outcome:
    """The engine's table holds the trigger with the requested settings, and the engine's loop is running."""
    from pilot.system.triggers import get_engine

    params: TriggerParams = result.action.parameters  # type: ignore[assignment]
    engine = get_engine()
    name = params.name.strip()
    created = _created_id(result)
    entries = engine.list_triggers()
    if created:
        candidates = [entry for entry in entries if entry.get("id") == created]
    else:
        candidates = [entry for entry in entries if entry.get("name") == name]
    label = f"{name or created} ({created})" if created and name else (name or created)
    if not candidates:
        return mismatch(f"Trigger {label} is not in the trigger engine after it was created")
    expected = {
        "name": name,
        "trigger_type": params.trigger_type,
        "condition": params.condition,
        "action_command": params.action_command.strip(),
        "enabled": True,
    }
    matching = [entry for entry in candidates if all(entry.get(key) == value for key, value in expected.items())]
    if not matching:
        entry = candidates[0]
        differing = sorted(key for key, value in expected.items() if entry.get(key) != value)
        return mismatch(f"Trigger {label} exists but its {', '.join(differing)} differ from the request")
    if not engine.is_running():
        return mismatch(f"Trigger {label} exists but the trigger engine is not running, so it will never fire")
    return verified(f"Trigger {label} is in the running trigger engine with the requested condition and command")


@register(ActionType.TRIGGER_DELETE, method="trigger_absence_postcondition")
async def trigger_absent(result: ActionResult) -> Outcome:
    """No trigger with the given id or name is left in the engine's table."""
    from pilot.system.triggers import get_engine

    params: TriggerParams = result.action.parameters  # type: ignore[assignment]
    target = (params.trigger_id or params.name).strip()
    if not target:
        raise ValueError("Trigger delete has no id or name to observe")
    left = [entry for entry in get_engine().list_triggers() if target in (entry.get("id"), entry.get("name"))]
    if left:
        return mismatch(f"{len(left)} trigger(s) with id or name {target} are still in the trigger engine")
    return verified(f"No trigger with id or name {target} is left in the trigger engine")


@register(ActionType.TRIGGER_START, method="trigger_engine_running_postcondition", precheck=True)
async def trigger_engine_running(result: ActionResult) -> Outcome:
    """The engine's evaluation loop is alive."""
    from pilot.system.triggers import get_engine

    if get_engine().is_running():
        return verified("The trigger engine's evaluation loop is running")
    return mismatch("The trigger engine's evaluation loop is not running after the start")


@register(ActionType.TRIGGER_STOP, method="trigger_engine_stopped_postcondition", precheck=True)
async def trigger_engine_stopped(result: ActionResult) -> Outcome:
    """The engine's evaluation loop has ended, so no trigger fires."""
    from pilot.system.triggers import get_engine

    if get_engine().is_running():
        return mismatch("The trigger engine's evaluation loop is still running after the stop")
    return verified("The trigger engine's evaluation loop is stopped")
