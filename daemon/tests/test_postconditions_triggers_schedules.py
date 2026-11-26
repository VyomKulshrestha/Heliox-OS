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
    fresh = triggers.TriggerEngine()
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


@pytest.mark.asyncio
async def test_created_trigger_verifies(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))

    assert result.success is True
    assert engine.is_running() is True
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
async def test_created_trigger_missing_from_the_engine_is_a_mismatch(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    engine._triggers.clear()

    _assert_mismatch(await _verify(result), "not in the trigger engine")


@pytest.mark.asyncio
async def test_created_trigger_with_a_different_command_is_a_mismatch(engine, tmp_path: Path) -> None:
    result = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    created = json.loads(result.output)["id"]
    engine._triggers[created].action_command = "something else"

    _assert_mismatch(await _verify(result), "action_command")


@pytest.mark.asyncio
async def test_invalid_trigger_is_a_failed_action(engine) -> None:
    result = await _run(
        ActionType.TRIGGER_CREATE,
        TriggerParams(name="x", trigger_type="not_a_type", action_command="notify"),
    )

    verification = await _verify(result)
    assert result.success is False
    assert "FAILED" in verification.details[0]


@pytest.mark.asyncio
async def test_retried_create_reuses_the_same_trigger(engine, tmp_path: Path) -> None:
    first = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))
    await engine.stop()
    second = await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))

    assert json.loads(first.output)["id"] == json.loads(second.output)["id"]
    assert len(engine.list_triggers()) == 1
    _assert_verified(await _verify(second))


@pytest.mark.asyncio
async def test_deleted_trigger_verifies(engine, tmp_path: Path) -> None:
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))

    result = await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="watch-downloads"))

    assert result.success is True
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_delete_leaving_a_trigger_of_that_name_is_a_mismatch(engine, tmp_path: Path) -> None:
    engine.create_trigger("watch-downloads", "file_created", {"path": str(tmp_path / "a")}, "notify a")
    engine.create_trigger("watch-downloads", "file_created", {"path": str(tmp_path / "b")}, "notify b")

    result = await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="watch-downloads"))

    assert result.success is True
    _assert_mismatch(await _verify(result), "still in the trigger engine")


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=False,
    reason="trigger_delete raises when nothing matched in the groundwork series (verify1); passes once stacked on it",
)
async def test_delete_by_a_wrong_name_is_not_a_success(engine, tmp_path: Path) -> None:
    await _run(ActionType.TRIGGER_CREATE, _watch(tmp_path))

    result = await _run(ActionType.TRIGGER_DELETE, TriggerParams(name="no-such-trigger"))
    verification = await _verify(result)

    assert result.success is False
    assert verification.passed is False
    assert len(engine.list_triggers()) == 1


@pytest.mark.asyncio
async def test_engine_start_and_stop_verify(engine) -> None:
    started = await _run(ActionType.TRIGGER_START, TriggerParams())
    _assert_verified(await _verify(started))

    stopped = await _run(ActionType.TRIGGER_STOP, TriggerParams())
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
async def test_start_restarts_a_loop_that_ended(engine) -> None:
    await engine.start()
    engine._task.cancel()
    await asyncio.gather(engine._task, return_exceptions=True)

    started = await _run(ActionType.TRIGGER_START, TriggerParams())

    assert engine.is_running() is True
    _assert_verified(await _verify(started))


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
