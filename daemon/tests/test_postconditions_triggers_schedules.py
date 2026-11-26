"""Post-conditions for trigger_create, trigger_delete, trigger_start, trigger_stop and schedule_create.

Triggers run on a fresh TriggerEngine put in place of the module's engine; its
triggers watch a file path that does not exist, so evaluating them does nothing.
Scheduled tasks are faked at the scheduler's command runners; no test reaches the
real Task Scheduler, launchd or crontab.
"""

from __future__ import annotations

import base64
import json
import plistlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ScheduleParams
from pilot.agents.executor import Executor
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.system import scheduler
from pilot.system.platform_detect import Platform

_HANDLERS = {
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
    assert POSTCONDITION_VERIFIERS[ActionType.SCHEDULE_CREATE] == "schedule_definition_postcondition"


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
