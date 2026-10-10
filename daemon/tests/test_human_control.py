"""Human takeover: the person can take the desktop back and Heliox's own input stops."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType, KeyboardParams
from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.human_control import (
    ControlHolder,
    HumanControlGate,
    HumanControlRefused,
    PhysicalInputWatcher,
    agent_initiated,
    is_agent_initiated,
)
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.server import PilotServer
from pilot.system import accessibility


class _FakeWatcher:
    def __init__(self, on_physical_input) -> None:
        self.on_physical_input = on_physical_input
        self.running = False
        self.starts = 0

    def start(self) -> bool:
        self.starts += 1
        self.running = True
        return True

    def stop(self) -> None:
        self.running = False


def _gate() -> tuple[HumanControlGate, dict]:
    created: dict = {}

    def factory(callback):
        created["watcher"] = _FakeWatcher(callback)
        return created["watcher"]

    return HumanControlGate(watcher_factory=factory), created


def test_take_and_release_track_holder_and_generation(monkeypatch):
    cleared = MagicMock()
    monkeypatch.setattr(accessibility.default_store(), "clear", cleared)
    gate, _ = _gate()

    assert gate.take("requested") is True
    assert gate.take("again") is False
    assert gate.status().holder == ControlHolder.HUMAN
    assert gate.release() is True
    assert gate.release() is False
    status = gate.status()
    assert status.holder == ControlHolder.AGENT
    assert status.generation == 1
    cleared.assert_called_once()


def test_only_agent_initiated_desktop_input_is_refused():
    gate, _ = _gate()
    gate.take("requested")

    gate.check_agent_may_act(ActionType.MOUSE_CLICK)  # the person's own command still runs
    with agent_initiated():
        assert is_agent_initiated()
        gate.check_agent_may_act(ActionType.FILE_READ)  # not desktop input
        with pytest.raises(HumanControlRefused, match="person has control"):
            gate.check_agent_may_act(ActionType.KEYBOARD_TYPE)
    assert not is_agent_initiated()


@pytest.mark.asyncio
async def test_wait_for_release_returns_when_handed_back_or_times_out():
    gate, _ = _gate()
    assert await gate.wait_for_release(0.01) is True
    gate.take("requested")
    assert await gate.wait_for_release(0.01) is False

    asyncio.get_running_loop().call_later(0.01, gate.release)
    assert await gate.wait_for_release(1.0) is True


@pytest.mark.asyncio
async def test_physical_input_takes_control_only_while_armed():
    gate, created = _gate()
    watcher = created["watcher"]

    watcher.on_physical_input()  # not armed: no loop, ignored
    assert not gate.holds_control()

    with gate.armed():
        assert watcher.running and gate.status().armed
        thread = threading.Thread(target=watcher.on_physical_input)  # hook threads call in
        thread.start()
        thread.join()
        await asyncio.sleep(0)
        assert gate.holds_control()
        assert "physical" in gate.status().reason
    assert not watcher.running


def test_watcher_ignores_injected_input_and_mouse_moves():
    calls = []
    watcher = PhysicalInputWatcher(lambda: calls.append(1))

    watcher.keyboard_filter(0x0100, SimpleNamespace(flags=0x10))  # injected key
    watcher.mouse_filter(0x0201, SimpleNamespace(flags=0x01))  # injected click
    watcher.mouse_filter(0x0200, SimpleNamespace(flags=0))  # physical move only
    assert calls == []

    watcher.keyboard_filter(0x0100, SimpleNamespace(flags=0))
    watcher.mouse_filter(0x0201, SimpleNamespace(flags=0))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_executor_refuses_agent_keyboard_input_while_the_person_holds_control(tmp_path, monkeypatch):
    pressed = AsyncMock()
    monkeypatch.setattr("pilot.system.input_control.keyboard_press", pressed)
    monkeypatch.setattr("pilot.system.input_control.keyboard_type", pressed)
    config = PilotConfig()
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )
    gate, _ = _gate()
    gate.take("requested")
    executor.set_human_control(gate)
    plan = ActionPlan(
        actions=[Action(action_type=ActionType.KEYBOARD_PRESS, target="", parameters=KeyboardParams(key="enter"))],
        explanation="press enter",
    )

    with agent_initiated():
        results = await executor.execute(plan, plan_id="takeover")

    assert results[0].success is False
    assert results[0].skip_reason == "human_control"
    pressed.assert_not_awaited()


@pytest.mark.asyncio
async def test_autonomous_round_waits_for_the_person_and_then_observes_again():
    gate, _ = _gate()
    gate.take("requested")
    notified = []
    autonomous = AutonomousExecutor(
        planner=None, executor=None, verifier=None, decomposer=MagicMock(), human_control=gate
    )
    autonomous._notify = AsyncMock(side_effect=lambda event, _job: notified.append(event))
    job = AutonomousJob(job_id="j1", goal="do it")
    progress: list[str] = []

    asyncio.get_running_loop().call_later(0.01, gate.release)
    assert await autonomous._wait_for_desktop_control(job, progress) is True

    assert notified == ["autonomous_paused_for_human", "autonomous_resumed"]
    assert "observing again" in progress[-1]


@pytest.mark.asyncio
async def test_take_release_and_status_rpcs():
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._human_control.add_listener(server._broadcast_human_control)

    taken = await server._handle_take_control({"reason": "my turn"}, ws=AsyncMock())
    assert taken["holder"] == "human" and taken["reason"] == "my turn"
    assert (await server._handle_human_control_status({}, ws=AsyncMock()))["holder"] == "human"
    released = await server._handle_release_control({}, ws=AsyncMock())
    assert released["holder"] == "agent" and released["generation"] == 1
    await asyncio.sleep(0)
    events = [call.args[0] for call in server._broadcast_notification.await_args_list]
    assert events == ["human_control", "human_control"]
    assert (await server._handle_take_control({"reason": 5}, ws=AsyncMock()))["status"] == "error"


def test_file_actions_are_not_human_controlled():
    gate, _ = _gate()
    gate.take("requested")
    with agent_initiated():
        gate.check_agent_may_act(ActionType.FILE_WRITE)


@pytest.mark.asyncio
async def test_request_human_help_hands_the_desktop_to_the_person():
    from pilot.actions import HumanHelpParams

    gate, _ = _gate()
    executor = SimpleNamespace(_human_control=gate)
    action = SimpleNamespace(parameters=HumanHelpParams(help_needed="Sign in to Outlook", instructions="then resume"))

    output = await Executor._exec_request_human_help(executor, action)

    assert gate.holds_control()
    assert gate.status().reason == "Heliox asked for help: Sign in to Outlook (then resume)"
    assert "continue after they hand control back" in output
    with agent_initiated():
        gate.check_agent_may_act(ActionType.REQUEST_HUMAN_HELP)  # asking again is never refused


def test_planner_fills_help_needed_from_common_synonyms():
    from pilot.actions import HumanHelpParams
    from pilot.agents.planner import Planner

    planner = Planner(MagicMock(), MagicMock())
    params = planner._parse_parameters(ActionType.REQUEST_HUMAN_HELP, {"reason": "Enter the 2FA code"})

    assert isinstance(params, HumanHelpParams)
    assert params.help_needed == "Enter the 2FA code"
