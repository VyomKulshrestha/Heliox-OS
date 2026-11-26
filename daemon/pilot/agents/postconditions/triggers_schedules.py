"""Post-conditions for reactive triggers and scheduled tasks.

Triggers are kept in the trigger store on disk (``DATA_DIR/triggers.json``, written by
the engine) and held in the engine's table, which its evaluation loop reads. The checks
read the store file afresh from disk, a different path than the in-memory table the
handler changed, and also confirm the engine holds the same trigger; "running" is judged
from the stored wish to run and from the loop's asyncio task, not from a flag. Where no
store is attached (a script, a test without one) the triggers exist only in memory and the
outcome is unobserved.

Scheduled tasks are read back from the platform's scheduler: Windows through the Task
Scheduler service's COM interface, macOS from the plist file and launchd, Linux from the
installed crontab (``pilot.system.scheduler.observe_schedule``). No check runs a
trigger's command or a scheduled command.

Deletes: "delete X" where nothing is called X fails in the handler. The pre-check of a
delete counts an absent target as already done only when this same run deleted and
verified it before (``_run_deletions``); otherwise the delete runs and fails.
"""

from __future__ import annotations

import json
from typing import Any

from pilot.actions import ActionResult, ActionType, ScheduleParams, TriggerParams
from pilot.agents.postconditions import (
    Outcome,
    _run_deletions,
    is_precheck,
    mismatch,
    register,
    unobserved,
    verified,
)


# The pre-check skips only a repeat of a delete this run already made and verified; a task name that
# matches nothing otherwise reaches the handler and fails there.
@register(ActionType.SCHEDULE_DELETE, method="schedule_absence_postcondition", precheck=True)
async def schedule_absent(result: ActionResult) -> Outcome:
    """The task is gone, asked of the platform's scheduler (schtasks, launchctl, crontab) afresh."""
    from pilot.system.scheduler import schedule_exists

    params: ScheduleParams = result.action.parameters  # type: ignore[assignment]
    target = params.task_id or params.name
    if not target:
        raise ValueError("Schedule delete has no task identifier to observe")
    present = await schedule_exists(target)
    if is_precheck(result):
        if not present and _run_deletions.deleted_earlier_in_this_run(ActionType.SCHEDULE_DELETE, target):
            return verified(f"Scheduled task {target} was deleted and verified earlier in this run")
        return mismatch(f"Scheduled task {target} was not deleted earlier in this run; the delete runs")
    if present:
        return mismatch(f"Scheduled task still exists: {target}")
    _run_deletions.note_verified_deletion(ActionType.SCHEDULE_DELETE, target)
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


_NO_STORE = "Triggers are kept only in memory here (no trigger store attached), so this cannot be read back"


def _store(engine: Any) -> dict[str, Any] | None:
    """The trigger store read from disk; None when the engine has no store. A damaged file raises (fails closed)."""
    from pilot.system.triggers import read_trigger_store

    if engine.store_path is None:
        return None
    return read_trigger_store(engine.store_path) or {"version": 1, "engine_running": False, "triggers": []}


@register(ActionType.TRIGGER_CREATE, method="trigger_present_postcondition")
async def trigger_present(result: ActionResult) -> Outcome:
    """The stored trigger has the requested settings, the engine holds the same one, and the engine runs."""
    from pilot.system.triggers import get_engine

    params: TriggerParams = result.action.parameters  # type: ignore[assignment]
    engine = get_engine()
    store = _store(engine)
    if store is None:
        return unobserved(_NO_STORE)
    name = params.name.strip()
    created = _created_id(result)
    entries = [entry for entry in store["triggers"] if isinstance(entry, dict)]
    if created:
        candidates = [entry for entry in entries if entry.get("id") == created]
    else:
        candidates = [entry for entry in entries if entry.get("name") == name]
    label = f"{name or created} ({created})" if created and name else (name or created)
    if not candidates:
        return mismatch(f"Trigger {label} is not in the trigger store after it was created")
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
        return mismatch(f"Trigger {label} is stored but its {', '.join(differing)} differ from the request")
    stored = matching[0]
    held = engine.get_trigger(stored.get("id", ""))
    if held is None or held.action_command != stored["action_command"] or held.condition != stored["condition"]:
        return mismatch(f"Trigger {label} is stored but the trigger engine does not hold it")
    if store.get("engine_running") is not True or not engine.is_running():
        return mismatch(f"Trigger {label} exists but the trigger engine is not running, so it will never fire")
    return verified(f"Trigger {label} is stored and held by the running trigger engine as requested")


@register(ActionType.TRIGGER_DELETE, method="trigger_absence_postcondition", precheck=True)
async def trigger_absent(result: ActionResult) -> Outcome:
    """No trigger with the given id or name is left in the trigger store or in the engine."""
    from pilot.system.triggers import get_engine

    params: TriggerParams = result.action.parameters  # type: ignore[assignment]
    target = (params.trigger_id or params.name).strip()
    if not target:
        raise ValueError("Trigger delete has no id or name to observe")
    engine = get_engine()
    precheck = is_precheck(result)
    store = _store(engine)
    if store is None:
        if precheck:
            return mismatch("The trigger store is not attached; the delete runs")
        return unobserved(_NO_STORE)
    stored = [
        entry
        for entry in store["triggers"]
        if isinstance(entry, dict) and target in (entry.get("id"), entry.get("name"))
    ]
    held = [entry for entry in engine.list_triggers() if target in (entry.get("id"), entry.get("name"))]
    if precheck:
        if not stored and not held and _run_deletions.deleted_earlier_in_this_run(ActionType.TRIGGER_DELETE, target):
            return verified(f"Trigger {target} was deleted and verified earlier in this run")
        return mismatch(f"Trigger {target} was not deleted earlier in this run; the delete runs")
    if stored:
        return mismatch(f"{len(stored)} trigger(s) with id or name {target} are still in the trigger store")
    if held:
        return mismatch(f"{len(held)} trigger(s) with id or name {target} are still held by the trigger engine")
    _run_deletions.note_verified_deletion(ActionType.TRIGGER_DELETE, target)
    return verified(f"No trigger with id or name {target} is left in the trigger store or the engine")


@register(ActionType.TRIGGER_START, method="trigger_engine_running_postcondition", precheck=True)
async def trigger_engine_running(result: ActionResult) -> Outcome:
    """The store says the engine should run, and its evaluation loop is alive."""
    from pilot.system.triggers import get_engine

    engine = get_engine()
    store = _store(engine)
    if store is None:
        return unobserved(_NO_STORE)
    if store.get("engine_running") is not True:
        return mismatch("The trigger store does not record the engine as running, so a restart would leave it off")
    if not engine.is_running():
        return mismatch("The trigger engine's evaluation loop is not running after the start")
    return verified("The trigger engine's evaluation loop is running, and the store keeps it running after a restart")


@register(ActionType.TRIGGER_STOP, method="trigger_engine_stopped_postcondition", precheck=True)
async def trigger_engine_stopped(result: ActionResult) -> Outcome:
    """The store says the engine should stay stopped, and its evaluation loop has ended."""
    from pilot.system.triggers import get_engine

    engine = get_engine()
    store = _store(engine)
    if store is None:
        return unobserved(_NO_STORE)
    if store.get("engine_running") is not False:
        return mismatch("The trigger store still records the engine as running, so a restart would start it again")
    if engine.is_running():
        return mismatch("The trigger engine's evaluation loop is still running after the stop")
    return verified("The trigger engine's evaluation loop is stopped, and the store keeps it stopped")
