"""Focused coverage for independently observed high-risk local effects."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    DiskManageParams,
    PowerParams,
    ProcessParams,
    ScheduleParams,
    VerificationResult,
    WindowParams,
)
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.server import _postcondition_failure_requires_reconciliation
from pilot.system.platform_detect import Platform

HIGH_RISK_VERIFIERS = {
    ActionType.PROCESS_KILL: "process_absence_postcondition",
    ActionType.POWER_SHUTDOWN: "shutdown_transition_postcondition",
    ActionType.POWER_RESTART: "restart_transition_postcondition",
    ActionType.POWER_LOGOUT: "logout_transition_postcondition",
    ActionType.WINDOW_CLOSE: "window_absence_postcondition",
    ActionType.DISK_UNMOUNT: "mount_absence_postcondition",
    ActionType.SCHEDULE_DELETE: "schedule_absence_postcondition",
}


def _result(action_type: ActionType, parameters) -> tuple[ActionPlan, list[ActionResult]]:
    action = Action(action_type=action_type, target="controlled-target", parameters=parameters)
    return ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="executor accepted")]


def test_high_risk_verifier_registry_is_runtime_authority() -> None:
    assert {action: POSTCONDITION_VERIFIERS[action] for action in HIGH_RISK_VERIFIERS} == HIGH_RISK_VERIFIERS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_type", "parameters", "module", "observer", "observer_kwargs"),
    [
        (
            ActionType.PROCESS_KILL,
            ProcessParams(pid=4242),
            "pilot.system.processes",
            "process_exists",
            {"pid": 4242, "name": None},
        ),
        (
            ActionType.WINDOW_CLOSE,
            WindowParams(title="Controlled Window"),
            "pilot.system.window_mgr",
            "window_exists",
            {"window_id": None, "title": "Controlled Window", "process_name": None},
        ),
        (
            ActionType.DISK_UNMOUNT,
            DiskManageParams(mount_point="/mnt/controlled"),
            "pilot.system.disks",
            "mount_exists",
            {"device": None, "mount_point": "/mnt/controlled"},
        ),
        (
            ActionType.SCHEDULE_DELETE,
            ScheduleParams(name="controlled-task"),
            "pilot.system.scheduler",
            "schedule_exists",
            None,
        ),
    ],
)
async def test_absence_postconditions_use_observed_state(
    monkeypatch: pytest.MonkeyPatch,
    action_type: ActionType,
    parameters,
    module: str,
    observer: str,
    observer_kwargs: dict[str, object] | None,
) -> None:
    imported = __import__(module, fromlist=[observer])
    check = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(imported, observer, check)
    plan, results = _result(action_type, parameters)
    verifier = Verifier(MagicMock())

    passed = await verifier.verify(plan, results)
    mismatch = await verifier.verify(plan, results)

    assert passed.passed is True
    assert mismatch.passed is False
    assert mismatch.failed_actions == [0]
    if observer_kwargs is None:
        check.assert_awaited_with("controlled-task")
    else:
        check.assert_awaited_with(**observer_kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_type",
    [ActionType.POWER_SHUTDOWN, ActionType.POWER_RESTART, ActionType.POWER_LOGOUT],
)
async def test_power_postcondition_requires_host_observation(
    monkeypatch: pytest.MonkeyPatch,
    action_type: ActionType,
) -> None:
    from pilot.system import power

    observed = AsyncMock(side_effect=[True, False])
    monkeypatch.setattr(power, "power_transition_observed", observed)
    plan, results = _result(action_type, PowerParams())
    verifier = Verifier(MagicMock())

    passed = await verifier.verify(plan, results)
    mismatch = await verifier.verify(plan, results)

    transition = action_type.value.removeprefix("power_")
    assert passed.passed is True
    assert mismatch.passed is False
    assert observed.await_args_list[0].args == (transition,)


@pytest.mark.asyncio
async def test_observer_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import scheduler

    monkeypatch.setattr(
        scheduler,
        "schedule_exists",
        AsyncMock(side_effect=RuntimeError("observer unavailable")),
    )
    plan, results = _result(ActionType.SCHEDULE_DELETE, ScheduleParams(name="controlled-task"))

    verification = await Verifier(MagicMock()).verify(plan, results)

    assert verification.passed is False
    assert verification.failed_actions == [0]
    assert "Verification error: observer unavailable" in verification.details[0]


def test_destructive_postcondition_mismatch_requires_manual_reconciliation() -> None:
    plan, results = _result(ActionType.PROCESS_KILL, ProcessParams(pid=4242))
    verification = VerificationResult(
        passed=False,
        failed_actions=[0],
        details=["Action 0 (process_kill): MISMATCH"],
    )

    assert plan.actions[0].requires_snapshot is True
    assert _postcondition_failure_requires_reconciliation(results, verification) is True


@pytest.mark.asyncio
async def test_windows_power_observer_requires_matching_host_event(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    run = AsyncMock(side_effect=[(0, "The process initiated restart of computer", ""), (0, "", "")])
    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(power, "run_powershell", run)

    assert await power.power_transition_observed("restart") is True
    assert await power.power_transition_observed("shutdown") is False
    assert "Id=1074" in run.await_args_list[0].args[0]


@pytest.mark.asyncio
async def test_mount_observer_uses_exact_linux_mountpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import disks

    run = AsyncMock(return_value=(1, "", ""))
    monkeypatch.setattr(disks, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(disks, "run_command", run)

    assert await disks.mount_exists(mount_point="/mnt/controlled") is False
    run.assert_awaited_once_with(["findmnt", "--noheadings", "--output", "TARGET", "--mountpoint", "/mnt/controlled"])


@pytest.mark.asyncio
async def test_schedule_observer_treats_missing_windows_task_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pilot.system import scheduler

    run = AsyncMock(return_value=(1, "", "ERROR: The system cannot find the file specified."))
    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(scheduler, "run_command", run)

    assert await scheduler.schedule_exists("controlled-task") is False
    run.assert_awaited_once_with(["schtasks", "/query", "/tn", "controlled-task", "/fo", "LIST", "/nh"])


@pytest.mark.asyncio
async def test_schedule_observer_permission_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import scheduler

    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(scheduler, "run_command", AsyncMock(return_value=(1, "", "Access is denied.")))

    with pytest.raises(RuntimeError, match="Access is denied"):
        await scheduler.schedule_exists("controlled-task")


@pytest.mark.asyncio
async def test_window_observer_escapes_windows_title(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import window_mgr

    run = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(window_mgr, "run_powershell", run)

    assert await window_mgr.window_exists(title="Owner's Controlled Window") is False
    # The title is data ($HelioxArgs), never part of the script text.
    assert run.await_args.kwargs["values"]["title"] == "Owner's Controlled Window"
    assert "Owner" not in run.await_args.args[0]
