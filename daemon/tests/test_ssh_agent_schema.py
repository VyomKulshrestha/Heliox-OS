from __future__ import annotations

from pilot.actions import Action, ActionPlan, ActionType, SshCommandParams
from pilot.security.validator import ActionValidator


def test_ssh_action_parses_params() -> None:
    action = Action(
        action_type=ActionType.SSH_COMMAND,
        target="",
        parameters={"host": "prod-1", "command": "echo hi"},
    )
    assert isinstance(action.parameters, SshCommandParams)
    assert action.parameters.host == "prod-1"


def test_ssh_validator_rejects_empty_host(default_config) -> None:
    validator = ActionValidator(default_config)
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SSH_COMMAND,
                target="",
                parameters={"host": "", "command": "echo hi"},
            )
        ]
    )
    errors = validator.validate_plan(plan)
    assert any("Empty SSH host alias" in e for e in errors)
