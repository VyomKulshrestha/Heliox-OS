import os

import pytest
from pydantic import ValidationError

from pilot.actions import Action, ActionType, FileParams, ShellCommandParams
from pilot.config import PilotConfig
from pilot.security.validator import ActionValidator


@pytest.fixture
def validator():
    config = PilotConfig()
    config.restrictions.protected_folders = ["/protected"]
    config.restrictions.blocked_commands = ["blockedcmd"]
    return ActionValidator(config)


def test_elevated_command_requires_destructive_or_root(validator):
    # Missing destructive/requires_root
    action = Action(
        action_type=ActionType.SHELL_COMMAND, parameters=ShellCommandParams(command="rm", args=["-rf", "/tmp"])
    )
    with pytest.raises(Exception) as excinfo:
        validator.validate_action(action, 0)
    assert "requires destructive or root flag" in str(excinfo.value)

    # Has destructive flag
    action.destructive = True
    # Still might fail if it targets a protected thing, but the elevated check will pass
    # We just want to ensure the explicit elevated check doesn't raise
    try:
        validator.validate_action(action, 0)
    except Exception as e:
        assert "requires destructive or root flag" not in str(e)


def test_protected_folder_bypass_mitigated(validator):
    from unittest.mock import patch

    # Path without .. so sanitizer allows it, but it resolves to the protected folder
    action = Action(
        action_type=ActionType.FILE_DELETE,
        target="/unprotected/link_to_secret",
        parameters=FileParams(path="/unprotected/link_to_secret"),
    )

    def fake_realpath(p):
        if "link_to_secret" in p:
            return "/protected/secret.txt"
        return p

    with patch("os.path.realpath", side_effect=fake_realpath):
        with pytest.raises(Exception) as excinfo:
            validator.validate_action(action, 0)
        assert "is in protected folder" in str(excinfo.value)
