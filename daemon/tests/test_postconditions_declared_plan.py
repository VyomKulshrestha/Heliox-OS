"""Declared post-conditions in the plan: the planner writes ``expect``, and the approval card says the check."""

from __future__ import annotations

import string
from unittest.mock import MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionResult,
    ActionType,
    Expectation,
    FileParams,
    ShellCommandParams,
    SshCommandParams,
)


def ssh(command: str, host: str = "prod-1") -> Action:
    return Action(
        action_type=ActionType.SSH_COMMAND, target=host, parameters=SshCommandParams(host=host, command=command)
    )


def checked(action: Action, probe: Action) -> Action:
    return action.model_copy(update={"expect": Expectation(probe=probe, kind="exit_code", value=0)})


def _planner():
    from pilot.agents.planner import Planner

    return Planner(MagicMock(), MagicMock())


def test_planner_parses_expect_with_typed_probe_parameters() -> None:
    raw = {
        "action_type": "ssh_command",
        "target": "prod-1",
        "parameters": {"host": "prod-1", "command": "touch /tmp/x"},
        "expect": {
            "probe": {
                "action_type": "ssh_command",
                "target": "prod-1",
                "parameters": {"host": "prod-1", "command": "test -f /tmp/x"},
            },
            "kind": "exit_code",
            "value": 0,
        },
    }

    action = _planner()._parse_action(raw)

    assert action.expect is not None
    assert isinstance(action.expect.probe.parameters, SshCommandParams)
    assert action.expect.probe.parameters.command == "test -f /tmp/x"
    assert "expect" not in action.parameters.model_dump()


def test_planner_parses_ssh_steps_it_offers() -> None:
    from pilot.actions import SshScriptParams

    plan = _planner()._parse_response(
        '{"explanation": "x", "actions": [{"action_type": "ssh_script", "target": "prod-1", '
        '"parameters": {"host": "prod-1", "script": "uptime\\n", "timeout_seconds": 30}}]}',
        "check prod-1",
    )
    assert plan.error is None
    assert isinstance(plan.actions[0].parameters, SshScriptParams)
    assert plan.actions[0].parameters.timeout_seconds == 30


def test_a_plan_without_expect_parses_as_before() -> None:
    action = _planner()._parse_action(
        {"action_type": "ssh_command", "target": "h", "parameters": {"host": "h", "command": "uptime"}}
    )
    assert action.expect is None
    assert "expect" in action.model_dump() and action.model_dump()["expect"] is None


@pytest.mark.parametrize(
    "expect",
    [
        {"kind": "exit_code", "value": 0},
        {"probe": "test -f /tmp/x", "kind": "exit_code", "value": 0},
        {"probe": {"action_type": "ssh_command", "parameters": {"host": "h", "command": "id -u"}}, "kind": "regex"},
        {
            "probe": {
                "action_type": "ssh_command",
                "parameters": {"host": "h", "command": "id -u"},
                "expect": {"probe": {"action_type": "file_delete", "parameters": {"path": "/x"}}, "kind": "exists"},
            },
            "kind": "exists",
        },
        {
            "probe": {
                "action_type": "ssh_command",
                "parameters": {"host": "h", "command": "id -u"},
                "rollback_action": {"action_type": "file_delete", "parameters": {"path": "/x"}},
            },
            "kind": "exists",
        },
    ],
)
def test_planner_drops_a_malformed_expect_and_keeps_the_step(expect: dict) -> None:
    raw = {"action_type": "ssh_command", "target": "h", "parameters": {"host": "h", "command": "uptime"}}
    action = _planner()._parse_action({**raw, "expect": expect})
    assert action.expect is None
    assert action.parameters.command == "uptime"


def test_planner_keeps_at_most_four_checks() -> None:
    from pilot.agents.planner import _limit_expects

    actions = [checked(ssh(f"touch /tmp/{i}"), ssh(f"test -f /tmp/{i}")) for i in range(6)]
    kept = _limit_expects(actions)
    assert [a.expect is not None for a in kept] == [True, True, True, True, False, False]


def test_planner_prompt_teaches_the_field() -> None:
    from pilot.agents.planner import SYSTEM_PROMPT

    names = {name for _, name, _, _ in string.Formatter().parse(SYSTEM_PROMPT) if name}
    text = SYSTEM_PROMPT.format(**{name: "x" for name in names})
    assert '"expect"' in text
    assert '"command": "test -f /srv/app/ready"' in text
    assert 'Leave "expect" out when there is nothing definite to read back' in text


def test_server_adds_the_check_line_to_the_approval_payload() -> None:
    from pilot.server import _declared_check_line

    touch = checked(ssh("touch /tmp/x"), ssh("test -f /tmp/x"))
    write = Action(action_type=ActionType.FILE_WRITE, target="/tmp/x", parameters=FileParams(path="/tmp/x"))
    assert _declared_check_line(touch) == {"check": "Runs `test -f /tmp/x` on prod-1 and expects it to succeed."}
    assert _declared_check_line(ssh("touch /tmp/x")) == {
        "check": "None declared: Heliox will report what the step said, without checking it."
    }
    assert _declared_check_line(write) == {}


def test_a_declared_mismatch_stops_before_the_retry() -> None:
    from pilot.server import _postcondition_failure_requires_reconciliation

    step = Action(action_type=ActionType.SHELL_COMMAND, parameters=ShellCommandParams(command="make", args=["install"]))
    probe = Action(action_type=ActionType.FILE_READ, target="/srv/version", parameters=FileParams(path="/srv/version"))
    action = step.model_copy(update={"expect": Expectation(probe=probe, kind="equals", value="2")})
    verification = MagicMock(failed_actions=[0])

    assert not action.is_irreversible
    assert _postcondition_failure_requires_reconciliation([ActionResult(action=action, success=True)], verification)
    # Without a declared check the step keeps today's path: a retry.
    assert not _postcondition_failure_requires_reconciliation([ActionResult(action=step, success=True)], verification)
