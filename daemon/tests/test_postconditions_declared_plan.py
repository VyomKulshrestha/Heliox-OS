"""Declared post-conditions in the plan: the planner writes ``expect``, and the approval card says the check."""

from __future__ import annotations

import string
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionType, Expectation, SshCommandParams


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
