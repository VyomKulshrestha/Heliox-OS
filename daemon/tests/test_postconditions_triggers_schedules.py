"""Post-conditions for trigger_create, trigger_delete, trigger_start, trigger_stop and schedule_create.

Triggers run on a fresh TriggerEngine put in place of the module's engine; its
triggers watch a file path that does not exist, so evaluating them does nothing.
Scheduled tasks are faked at the scheduler's command runners; no test reaches the
real Task Scheduler, launchd or crontab.
"""

from __future__ import annotations

import asyncio
import base64
import json
import plistlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ScheduleParams, TriggerParams
from pilot.agents.executor import Executor
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.system import scheduler, triggers
from pilot.system.platform_detect import Platform

_HANDLERS = {
    ActionType.TRIGGER_CREATE: Executor._exec_trigger_create,
    ActionType.TRIGGER_DELETE: Executor._exec_trigger_delete,
    ActionType.TRIGGER_START: Executor._exec_trigger_start,
    ActionType.TRIGGER_STOP: Executor._exec_trigger_stop,
    ActionType.SCHEDULE_CREATE: Executor._exec_schedule_create,
    ActionType.SCHEDULE_DELETE: Executor._exec_schedule_delete,
}


async def _run(action_type: ActionType, params) -> ActionResult:
    """Run the action through the executor's own handler and wrap it as the executor does."""
    action = Action(action_type=action_type, target="controlled-target", parameters=params)
    try:
        output = await _HANDLERS[action_type](MagicMock(), action)
    except Exception as error:  # noqa: BLE001 - mirrors Executor._execute_single
        return ActionResult(action=action, success=False, error=str(error))
    return ActionResult(action=action, success=True, output=output)


async def _verify(result: ActionResult):
    return await Verifier(MagicMock()).verify(ActionPlan(actions=[result.action]), [result])


def _assert_verified(verification) -> None:
    assert verification.passed is True, verification.details
    assert verification.independently_observed_actions == [0]
    assert "VERIFIED" in verification.details[0]


def _assert_mismatch(verification, *words: str) -> None:
    assert verification.passed is False
    assert verification.independently_observed_actions == [0]
    assert "MISMATCH" in verification.details[0]
    for word in words:
        assert word in verification.details[0]


def test_verifiers_are_registered() -> None:
    assert POSTCONDITION_VERIFIERS[ActionType.TRIGGER_CREATE] == "trigger_present_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.TRIGGER_DELETE] == "trigger_absence_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.TRIGGER_START] == "trigger_engine_running_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.TRIGGER_STOP] == "trigger_engine_stopped_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.SCHEDULE_CREATE] == "schedule_definition_postcondition"


# ── Triggers ─────────────────────────────────────────────────────────


@pytest.fixture
async def engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A fresh engine keeping its triggers in a store under tmp_path, in place of the module's engine."""
    fresh = triggers.TriggerEngine()
    fresh.attach_store(tmp_path / "store" / "triggers.json")
    monkeypatch.setattr(triggers, "_engine", fresh)
    yield fresh
    await fresh.stop()


def _watch(tmp_path: Path, name: str = "watch-downloads", command: str = "notify new download") -> TriggerParams:
    return TriggerParams(
        name=name,
        trigger_type="file_created",
        condition={"path": str(tmp_path / "missing-folder")},
        action_command=command,
    )


def _stored(engine: triggers.TriggerEngine) -> dict:
    return json.loads(engine.store_path.read_text(encoding="utf-8"))


def _rewrite_store(engine: triggers.TriggerEngine, change) -> None:
    data = _stored(engine)
    change(data)
    engine.store_path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.asyncio
async def test_created_trigger_verifies_from_the_store(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))

    assert result.success is True
    assert engine.is_running() is True
    stored = _stored(engine)
    assert stored["version"] == 1 and stored["engine_running"] is True
    assert [entry["name"] for entry in stored["triggers"]] == ["watch-downloads"]
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_created_trigger_with_a_stopped_engine_is_a_mismatch(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    await engine.stop()

    _assert_mismatch(await _verify(result), "not running")


@pytest.mark.asyncio
async def test_created_trigger_whose_loop_died_is_a_mismatch(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    engine._task.cancel()
    await asyncio.gather(engine._task, return_exceptions=True)

    assert engine._running is True
    _assert_mismatch(await _verify(result), "not running")


@pytest.mark.asyncio
async def test_created_trigger_missing_from_the_store_is_a_mismatch(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    _rewrite_store(engine, lambda data: data.update(triggers=[]))

    _assert_mismatch(await _verify(result), "not in the trigger store")


@pytest.mark.asyncio
async def test_created_trigger_the_engine_no_longer_holds_is_a_mismatch(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    engine._triggers.clear()

    _assert_mismatch(await _verify(result), "does not hold it")


@pytest.mark.asyncio
async def test_created_trigger_stored_with_a_different_command_is_a_mismatch(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    _rewrite_store(engine, lambda data: data["triggers"][0].update(action_command="something else"))

    verification = await _verify(result)

    _assert_mismatch(verification, "action_command")
    assert "something else" not in verification.details[0]


@pytest.mark.asyncio
async def test_without_a_store_the_trigger_is_reported_not_observed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    memory_only = triggers.TriggerEngine()
    monkeypatch.setattr(triggers, "_engine", memory_only)
    try:
        result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
        verification = await _verify(result)
    finally:
        await memory_only.stop()

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


@pytest.mark.asyncio
async def test_invalid_trigger_is_a_failed_action_and_is_not_stored(engine) -> None:
    result = await _run(
        ActionType.TRIGGER_CREATE,
        TriggerParams(name="x", trigger_type="not_a_type", action_command="notify"),
    )

    verification = await _verify(result)
    assert result.success is False
    assert "FAILED" in verification.details[0]
    assert not engine.store_path.exists() or _stored(engine)["triggers"] == []


@pytest.mark.asyncio
async def test_retried_create_keeps_the_identical_trigger_and_says_so(engine, tmp_path: Path) -> None:
    first = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    await engine.stop()
    second = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))

    assert json.loads(first.output)["id"] == json.loads(second.output)["id"]
    assert "note" not in json.loads(first.output)
    assert "already existed and was kept" in json.loads(second.output)["note"]
    assert len(_stored(engine)["triggers"]) == 1
    _assert_verified(await _verify(second))


@pytest.mark.asyncio
async def test_deleted_trigger_verifies(engine, tmp_path: Path) -> None:
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))

    result = await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="watch-downloads"))

    assert result.success is True
    assert _stored(engine)["triggers"] == []
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_a_name_two_triggers_share_asks_which_and_deletes_neither(engine, tmp_path: Path) -> None:
    engine.create_trigger("watch-downloads", "file_created", {"path": str(tmp_path / "a")}, "notify a")
    engine.create_trigger("watch-downloads", "file_created", {"path": str(tmp_path / "b")}, "notify b")

    result = await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="watch-downloads"))

    assert result.success is False
    assert (result.error or "").startswith('2 triggers match "watch-downloads"')
    assert len(_stored(engine)["triggers"]) == 2


@pytest.mark.asyncio
async def test_a_delete_whose_trigger_is_still_stored_is_a_mismatch(engine, tmp_path: Path) -> None:
    """The check looks for the id the delete named, so a delete that said it removed a trigger still there fails."""
    kept = engine.create_trigger("watch-downloads", "file_created", {"path": str(tmp_path / "a")}, "notify a")
    action = Action(
        action_type=ActionType.TRIGGER_DELETE,
        target="the downloads trigger",
        parameters=TriggerParams(name="the downloads trigger"),
    )
    claimed = ActionResult(action=action, success=True, output=f'Deleted the trigger "watch-downloads" (id {kept.id}).')

    _assert_mismatch(await _verify(claimed), "still in the trigger store")


@pytest.mark.asyncio
async def test_delete_by_a_wrong_name_fails_and_keeps_the_real_one(engine, tmp_path: Path) -> None:
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))

    result = await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="no-such-trigger"))
    verification = await _verify(result)

    assert result.success is False
    assert "nothing was deleted" in (result.error or "")
    assert verification.passed is False
    assert len(_stored(engine)["triggers"]) == 1


@pytest.mark.asyncio
async def test_delete_precheck_counts_only_this_runs_verified_deletion(engine, tmp_path: Path) -> None:
    from pilot.agents.postconditions import _run_deletions
    from pilot.intelligence.experience import experience_scope

    _run_deletions.clear()
    action = Action(action_type=ActionType.TRIGGER_DELETE, parameters=TriggerParams(name="watch-downloads"))
    checker = Verifier(MagicMock())
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    with experience_scope(task_id="task-one"):
        assert (await checker.check_already_satisfied(action))[0] is False  # present: the delete runs
        deleted = await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="watch-downloads"))
        _assert_verified(await _verify(deleted))
        # The same run plans the delete again: already done, not a failure.
        assert (await checker.check_already_satisfied(action))[0] is True
    with experience_scope(task_id="task-two"):
        # Another run asking to delete a name that matches nothing: the delete runs and fails.
        assert (await checker.check_already_satisfied(action))[0] is False
        assert (await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="watch-downloads"))).success is False
    # No task id at all: the safer answer, the delete runs and fails.
    assert (await checker.check_already_satisfied(action))[0] is False
    _run_deletions.clear()


@pytest.mark.asyncio
async def test_engine_start_and_stop_verify_and_are_stored(engine) -> None:
    started = await _run(ActionType.TRIGGER_START, TriggerParams())
    assert _stored(engine)["engine_running"] is True
    _assert_verified(await _verify(started))

    stopped = await _run(ActionType.TRIGGER_STOP, TriggerParams())
    assert _stored(engine)["engine_running"] is False
    _assert_verified(await _verify(stopped))


@pytest.mark.asyncio
async def test_engine_start_then_stopped_again_is_a_mismatch(engine) -> None:
    started = await _run(ActionType.TRIGGER_START, TriggerParams())
    await engine.stop()

    _assert_mismatch(await _verify(started), "not running")


@pytest.mark.asyncio
async def test_engine_stop_then_started_again_is_a_mismatch(engine) -> None:
    stopped = await _run(ActionType.TRIGGER_STOP, TriggerParams())
    await engine.start()

    _assert_mismatch(await _verify(stopped), "still running")


@pytest.mark.asyncio
async def test_engine_stop_not_kept_in_the_store_is_a_mismatch(engine) -> None:
    stopped = await _run(ActionType.TRIGGER_STOP, TriggerParams())
    _rewrite_store(engine, lambda data: data.update(engine_running=True))

    _assert_mismatch(await _verify(stopped), "restart would start it again")


@pytest.mark.asyncio
async def test_start_restarts_a_loop_that_ended(engine) -> None:
    await engine.start()
    engine._task.cancel()
    await asyncio.gather(engine._task, return_exceptions=True)

    started = await _run(ActionType.TRIGGER_START, TriggerParams())

    assert engine.is_running() is True
    _assert_verified(await _verify(started))


# ── The trigger store across a restart ───────────────────────────────


@pytest.mark.asyncio
async def test_a_created_trigger_is_there_and_enabled_after_a_restart(engine, tmp_path: Path) -> None:
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    await engine.stop()

    restarted = triggers.TriggerEngine()
    fired = AsyncMock()
    restarted.set_fire_callback(fired)
    restarted.attach_store(engine.store_path)

    [entry] = restarted.list_triggers()
    assert (entry["name"], entry["enabled"], entry["disabled_reason"]) == ("watch-downloads", True, "")
    assert restarted.wanted_running is True
    assert restarted.is_running() is False  # loading starts nothing
    fired.assert_not_called()


@pytest.mark.asyncio
async def test_resume_starts_the_engine_only_when_it_was_left_running(
    monkeypatch: pytest.MonkeyPatch, engine, tmp_path: Path
) -> None:
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    await engine.stop()
    path = engine.store_path

    resumed = triggers.TriggerEngine()
    monkeypatch.setattr(triggers, "_engine", resumed)
    try:
        assert await triggers.resume_triggers(path) is True
        assert resumed.is_running() is True
        await _run(ActionType.TRIGGER_STOP, TriggerParams())
    finally:
        await resumed.stop()

    left_stopped = triggers.TriggerEngine()
    monkeypatch.setattr(triggers, "_engine", left_stopped)
    assert await triggers.resume_triggers(path) is False
    assert left_stopped.is_running() is False


@pytest.mark.asyncio
async def test_a_deleted_trigger_is_gone_after_a_restart(engine, tmp_path: Path) -> None:
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path, name="keep-me", command="notify kept"))
    await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="watch-downloads"))

    restarted = triggers.TriggerEngine()
    restarted.attach_store(engine.store_path)

    assert [entry["name"] for entry in restarted.list_triggers()] == ["keep-me"]


def _entry(tmp_path: Path, **changes) -> dict:
    entry = {
        "id": "abc12345",
        "name": "watch",
        "trigger_type": "file_created",
        "condition": {"path": str(tmp_path / "missing")},
        "action_command": "notify me",
        "enabled": True,
        "fire_count": 0,
        "max_fires": 0,
        "cooldown_seconds": 60,
        "last_fired": 0,
        "created_at": "",
        "disabled_reason": "",
    }
    entry.update(changes)
    return entry


def test_a_damaged_entry_is_skipped_and_the_rest_load(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "triggers.json"
    entries = [
        _entry(tmp_path, id="good0001", name="first"),
        "not an entry",
        _entry(tmp_path, id="bad00001", trigger_type="no_such_type"),
        _entry(tmp_path, id="bad00002", max_fires="many"),
        {"name": "no id"},
        _entry(tmp_path, id="good0002", name="second"),
    ]
    path.write_text(json.dumps({"version": 1, "engine_running": True, "triggers": entries}), encoding="utf-8")

    engine = triggers.TriggerEngine()
    with caplog.at_level("WARNING", logger="pilot.system.triggers"):
        engine.attach_store(path)

    assert sorted(entry["name"] for entry in engine.list_triggers()) == ["first", "second"]
    assert sum("Skipped damaged trigger entry" in record.getMessage() for record in caplog.records) == 4


def test_a_damaged_store_file_is_moved_aside_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "triggers.json"
    path.write_text("{ not json", encoding="utf-8")

    engine = triggers.TriggerEngine()
    engine.attach_store(path)
    engine.create_trigger("fresh", "file_created", {"path": str(tmp_path / "x")}, "notify fresh")

    aside = list(tmp_path.glob("triggers.json.damaged-*"))
    assert len(aside) == 1 and aside[0].read_text(encoding="utf-8") == "{ not json"
    assert [entry["name"] for entry in _stored(engine)["triggers"]] == ["fresh"]


def test_a_store_from_a_newer_version_is_read_but_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "triggers.json"
    original = json.dumps({"version": 2, "engine_running": True, "triggers": [_entry(tmp_path)]})
    path.write_text(original, encoding="utf-8")

    engine = triggers.TriggerEngine()
    engine.attach_store(path)
    engine.create_trigger("fresh", "file_created", {"path": str(tmp_path / "x")}, "notify fresh")

    assert path.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_a_stored_trigger_refused_by_todays_rules_loads_disabled_and_never_fires(tmp_path: Path) -> None:
    path = tmp_path / "triggers.json"
    refused = _entry(
        tmp_path,
        id="refused1",
        trigger_type="time_interval",
        condition={},
        cooldown_seconds=0,
        action_command="   ",
    )
    sneaky = _entry(
        tmp_path,
        id="refused2",
        trigger_type="custom_condition",
        condition={"expression": "__import__('os').system('echo hacked')"},
        cooldown_seconds=0,
    )
    path.write_text(json.dumps({"version": 1, "engine_running": True, "triggers": [sneaky]}), encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["triggers"].append(refused)
    path.write_text(json.dumps(data), encoding="utf-8")

    engine = triggers.TriggerEngine()
    fired = AsyncMock()
    engine.set_fire_callback(fired)
    engine.attach_store(path)

    loaded = {entry["id"]: entry for entry in engine.list_triggers()}
    assert set(loaded) == {"refused1", "refused2"}
    for entry in loaded.values():
        assert entry["enabled"] is False
        assert "Refused by the trigger rules" in entry["disabled_reason"]
    await engine.start()
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        await engine.stop()
    fired.assert_not_called()


# ── schedule_create, mirroring the schedule_delete tests ─────────────


def _schedule_result(command: str = "python backup.py", schedule: str = "0 9 * * *") -> ActionResult:
    action = Action(
        action_type=ActionType.SCHEDULE_CREATE,
        target="controlled-task",
        parameters=ScheduleParams(name="controlled-task", command=command, schedule=schedule),
    )
    return ActionResult(action=action, success=True, output="executor accepted")


@pytest.mark.asyncio
async def test_schedule_create_uses_observed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    observe = AsyncMock(
        side_effect=[
            scheduler.ObservedSchedule(command="python backup.py", schedule="0 9 * * *"),
            None,
        ]
    )
    monkeypatch.setattr(scheduler, "observe_schedule", observe)
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.LINUX)

    _assert_verified(await _verify(_schedule_result()))
    _assert_mismatch(await _verify(_schedule_result()), "does not exist")
    observe.assert_awaited_with("controlled-task")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("observed", "words"),
    [
        (scheduler.ObservedSchedule(command="rm -rf ~", schedule="0 9 * * *"), "different command"),
        (scheduler.ObservedSchedule(command="python backup.py", schedule="*/5 * * * *"), "not on the requested"),
        (scheduler.ObservedSchedule(command=None, loaded=False), "cannot read"),
        (scheduler.ObservedSchedule(command="python backup.py", loaded=False), "has not loaded"),
    ],
)
async def test_schedule_create_differences_are_mismatches(
    monkeypatch: pytest.MonkeyPatch, observed: scheduler.ObservedSchedule, words: str
) -> None:
    monkeypatch.setattr(scheduler, "observe_schedule", AsyncMock(return_value=observed))
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.LINUX)

    verification = await _verify(_schedule_result())

    _assert_mismatch(verification, words)
    assert "rm -rf" not in verification.details[0]


@pytest.mark.asyncio
async def test_schedule_create_observer_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "observe_schedule", AsyncMock(side_effect=RuntimeError("observer unavailable")))

    verification = await _verify(_schedule_result())

    assert verification.passed is False
    assert "Verification error: observer unavailable" in verification.details[0]


@pytest.mark.asyncio
async def test_schedule_create_failure_is_a_failed_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(scheduler, "run_command", AsyncMock(return_value=(1, "", "ERROR: Access is denied.")))

    result = await _run(
        ActionType.SCHEDULE_CREATE, ScheduleParams(name="controlled-task", command="notepad.exe", schedule="DAILY")
    )

    assert result.success is False
    assert "FAILED" in (await _verify(result)).details[0]


def _windows_answer(payload: dict) -> tuple[int, str, str]:
    return 0, base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii") + "\n", ""


@pytest.mark.asyncio
async def test_windows_schedule_is_read_from_the_task_scheduler_service(monkeypatch: pytest.MonkeyPatch) -> None:
    run = AsyncMock(
        return_value=_windows_answer(
            {"found": True, "actions": [{"path": '"C:\\Program Files\\Tool\\tool.exe"', "arguments": "--sync"}]}
        )
    )
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(scheduler, "run_powershell", run)

    observed = await scheduler.observe_schedule("controlled-task")
    verification = await _verify(
        _schedule_result(command='"C:\\Program Files\\Tool\\TOOL.exe" --sync', schedule="DAILY")
    )

    assert observed == scheduler.ObservedSchedule(command='"C:\\Program Files\\Tool\\tool.exe" --sync')
    # The task name is data ($HelioxArgs), never part of the script text.
    assert run.await_args.kwargs["values"] == {"name": "controlled-task"}
    assert "controlled-task" not in run.await_args.args[0]
    _assert_verified(verification)


@pytest.mark.asyncio
async def test_windows_missing_task_reads_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(scheduler, "run_powershell", AsyncMock(return_value=_windows_answer({"found": False})))

    assert await scheduler.observe_schedule("controlled-task") is None


@pytest.mark.asyncio
async def test_windows_scheduler_refusal_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(scheduler, "run_powershell", AsyncMock(return_value=(1, "", "Access is denied.")))

    with pytest.raises(RuntimeError, match="Access is denied"):
        await scheduler.observe_schedule("controlled-task")


@pytest.mark.asyncio
async def test_linux_schedule_is_read_from_the_installed_crontab(monkeypatch: pytest.MonkeyPatch) -> None:
    crontab = (
        "0 9 * * * python other.py # pilot-task:controlled-task-2\n"
        "0 9 * * *   python   backup.py # pilot-task:controlled-task\n"
        "@daily echo hi # pilot-task:daily\n"
    )
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(scheduler, "run_command", AsyncMock(return_value=(0, crontab, "")))

    assert await scheduler.observe_schedule("controlled-task") == scheduler.ObservedSchedule(
        command="python backup.py", schedule="0 9 * * *"
    )
    assert await scheduler.observe_schedule("daily") == scheduler.ObservedSchedule(command="echo hi", schedule="@daily")
    assert await scheduler.observe_schedule("controlled") is None
    _assert_verified(await _verify(_schedule_result()))


@pytest.mark.asyncio
async def test_linux_marker_of_a_longer_name_is_not_this_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(
        scheduler, "run_command", AsyncMock(return_value=(0, "0 9 * * * true # pilot-task:backup-old\n", ""))
    )

    assert await scheduler.schedule_exists("backup") is False
    assert await scheduler.schedule_exists("backup-old") is True


@pytest.mark.asyncio
async def test_linux_without_a_crontab_reads_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(scheduler, "run_command", AsyncMock(return_value=(1, "", "no crontab for user")))

    assert await scheduler.observe_schedule("controlled-task") is None


@pytest.mark.asyncio
async def test_macos_schedule_is_read_from_the_plist_and_launchd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plist = tmp_path / "com.pilot.controlled-task.plist"
    plist.write_bytes(
        plistlib.dumps({"Label": "com.pilot.controlled-task", "ProgramArguments": ["/bin/bash", "-c", "a && b < c"]})
    )
    run = AsyncMock(side_effect=[(0, "", ""), (113, "", "Could not find service")])
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(scheduler, "_launch_agent_path", lambda name: str(tmp_path / f"{name}.plist"))
    monkeypatch.setattr(scheduler, "run_command", run)

    assert await scheduler.observe_schedule("controlled-task") == scheduler.ObservedSchedule(command="a && b < c")
    assert await scheduler.observe_schedule("controlled-task") == scheduler.ObservedSchedule(
        command="a && b < c", loaded=False
    )
    run.assert_awaited_with(["launchctl", "list", "com.pilot.controlled-task"])

    plist.write_text("<plist><dict><string>a && b</string></dict></plist>", encoding="utf-8")
    run.side_effect = None
    run.return_value = (0, "", "")
    assert (await scheduler.observe_schedule("controlled-task")).command is None


@pytest.mark.asyncio
async def test_macos_plist_written_by_create_is_valid_for_shell_characters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "com.pilot.controlled-task.plist"
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(scheduler.os.path, "expanduser", lambda path: str(target))
    monkeypatch.setattr(scheduler, "run_command", AsyncMock(return_value=(0, "", "")))

    await scheduler.schedule_create("controlled-task", "a && b < c", "")

    assert plistlib.loads(target.read_bytes())["ProgramArguments"] == ["/bin/bash", "-c", "a && b < c"]


@pytest.mark.asyncio
async def test_linux_delete_keeps_the_task_whose_name_only_starts_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: list[str] = []

    async def run(args, **_kwargs):
        if args == ["crontab", "-l"]:
            return 0, "0 9 * * * a # pilot-task:backup\n0 9 * * * b # pilot-task:backup-old\n", ""
        installed.append(Path(args[1]).read_text(encoding="utf-8"))
        return 0, "", ""

    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(scheduler, "run_command", run)

    await scheduler.schedule_delete("backup")

    assert installed == ["0 9 * * * b # pilot-task:backup-old\n"]


# ── schedule_delete: a name that matches nothing fails, on every platform ──

_NOT_DELETED = "No scheduled task is called 'controlled-task', so nothing was deleted. List them first."


def _windows_delete_runner(exists: bool, calls: list[list[str]]):
    async def run(args, **_kwargs):
        calls.append(list(args))
        if args[:2] == ["schtasks", "/query"]:
            return (0, "TaskName: controlled-task", "") if exists else (1, "", "ERROR: The system cannot find the file")
        return 0, "SUCCESS", ""

    return run


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", [True, False])
async def test_windows_delete_fails_for_a_name_that_matches_nothing(monkeypatch: pytest.MonkeyPatch, exists: bool):
    calls: list[list[str]] = []
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(scheduler, "run_command", _windows_delete_runner(exists, calls))

    result = await _run(ActionType.SCHEDULE_DELETE, ScheduleParams(name="controlled-task"))

    deleted = ["schtasks", "/delete", "/tn", "controlled-task", "/f"] in calls
    assert (result.success, deleted) == (exists, exists)
    if not exists:
        assert result.error == _NOT_DELETED


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", [True, False])
async def test_macos_delete_fails_for_a_name_that_matches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exists: bool
) -> None:
    plist = tmp_path / "com.pilot.controlled-task.plist"
    if exists:
        plist.write_bytes(plistlib.dumps({"Label": "com.pilot.controlled-task"}))
    calls: list[list[str]] = []

    async def run(args, **_kwargs):
        calls.append(list(args))
        return (113, "", "Could not find service") if args[:2] == ["launchctl", "list"] else (0, "", "")

    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(scheduler.os.path, "expanduser", lambda path: str(plist))
    monkeypatch.setattr(scheduler, "run_command", run)

    result = await _run(ActionType.SCHEDULE_DELETE, ScheduleParams(name="controlled-task"))

    assert result.success is exists
    assert plist.exists() is False
    assert (["launchctl", "unload", str(plist)] in calls) is exists
    if not exists:
        assert result.error == _NOT_DELETED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crontab",
    [
        (1, "", "no crontab for user"),
        (0, "0 9 * * * true # pilot-task:controlled-task-old\n", ""),
    ],
)
async def test_linux_delete_fails_for_a_name_that_matches_nothing(monkeypatch: pytest.MonkeyPatch, crontab) -> None:
    calls: list[list[str]] = []

    async def run(args, **_kwargs):
        calls.append(list(args))
        return crontab

    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(scheduler, "run_command", run)

    result = await _run(ActionType.SCHEDULE_DELETE, ScheduleParams(name="controlled-task"))

    assert result.success is False
    assert result.error == _NOT_DELETED
    assert all(call == ["crontab", "-l"] for call in calls)  # nothing was installed


@pytest.mark.asyncio
async def test_schedule_delete_precheck_counts_only_this_runs_verified_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pilot.agents.postconditions import _run_deletions
    from pilot.intelligence.experience import experience_scope

    _run_deletions.clear()
    monkeypatch.setattr(scheduler, "schedule_exists", AsyncMock(return_value=False))
    action = Action(action_type=ActionType.SCHEDULE_DELETE, parameters=ScheduleParams(name="controlled-task"))
    checker = Verifier(MagicMock())

    with experience_scope(task_id="task-one"):
        # An absent task that this run never deleted is not "already satisfied": the delete runs and fails.
        assert (await checker.check_already_satisfied(action))[0] is False
        done = ActionResult(action=action, success=True, output="Deleted scheduled task: controlled-task")
        _assert_verified(await _verify(done))
        assert (await checker.check_already_satisfied(action))[0] is True
    with experience_scope(task_id="task-two"):
        assert (await checker.check_already_satisfied(action))[0] is False
    _run_deletions.clear()
