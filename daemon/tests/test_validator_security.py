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


@pytest.mark.parametrize(
    ("action_type", "path", "destination"),
    [
        (ActionType.FILE_READ, "{config}/config.toml", None),
        (ActionType.FILE_WRITE, "{config}/mcp_servers.json", None),
        (ActionType.FILE_READ, "{runtime}/auth_token", None),
        (ActionType.FILE_COPY, "{home}/notes.txt", "{config}/skills/evil.md"),
        (ActionType.FILE_READ, "{config}/../heliox-os/config.toml", None),
    ],
)
def test_heliox_config_and_runtime_folders_are_off_limits(
    validator, monkeypatch, tmp_path, action_type, path, destination
):
    import pilot.config

    config_dir = tmp_path / "cfg" / "heliox-os"
    runtime_dir = tmp_path / "run" / "heliox-os"
    monkeypatch.setattr(pilot.config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(pilot.config, "RUNTIME_DIR", runtime_dir)
    values = {"config": str(config_dir), "runtime": str(runtime_dir), "home": str(tmp_path)}
    action = Action(
        action_type=action_type,
        target=path.format(**values),
        parameters=FileParams(
            path=path.format(**values), destination=destination.format(**values) if destination else None
        ),
    )

    with pytest.raises(Exception, match="Heliox's own configuration or runtime folder|Path traversal"):
        validator.validate_action(action, 0)


def test_neighbouring_folders_are_not_mistaken_for_heliox_config(monkeypatch, tmp_path):
    import pilot.config
    from pilot.security.validator import inside_heliox_private_dirs

    monkeypatch.setattr(pilot.config, "CONFIG_DIR", tmp_path / "heliox-os")
    monkeypatch.setattr(pilot.config, "RUNTIME_DIR", tmp_path / "runtime")

    assert inside_heliox_private_dirs(str(tmp_path / "heliox-os" / "config.toml"))
    assert not inside_heliox_private_dirs(str(tmp_path / "heliox-os-notes" / "todo.txt"))
    assert not inside_heliox_private_dirs("")
