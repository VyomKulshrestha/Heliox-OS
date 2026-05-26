"""Tests for the action validator and sanitizer."""

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    FileParams,
    PackageParams,
    ServiceParams,
    ShellCommandParams,
)
from pilot.config import PilotConfig
from pilot.security.validator import ActionValidator, ValidationError


@pytest.fixture
def config():
    cfg = PilotConfig()
    cfg.restrictions.protected_folders = ["/home/user/private"]
    cfg.restrictions.protected_packages = ["firefox"]
    cfg.restrictions.blocked_commands = ["rm"]
    return cfg


@pytest.fixture
def validator(config):
    return ActionValidator(config)


class TestPathValidation:
    def test_valid_absolute_path(self, validator):
        action = Action(
            action_type=ActionType.FILE_READ,
            target="/home/user/test.txt",
            parameters=FileParams(path="/home/user/test.txt"),
        )
        validator.validate_action(action)

    def test_rejects_relative_path(self, validator):
        action = Action(
            action_type=ActionType.FILE_READ,
            target="relative/path.txt",
            parameters=FileParams(path="relative/path.txt"),
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)

    def test_rejects_path_traversal(self, validator):
        action = Action(
            action_type=ActionType.FILE_READ,
            target="/home/user/../etc/passwd",
            parameters=FileParams(path="/home/user/../etc/passwd"),
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)

    def test_rejects_shell_metacharacters(self, validator):
        action = Action(
            action_type=ActionType.FILE_READ,
            target="/home/user/$(whoami).txt",
            parameters=FileParams(path="/home/user/$(whoami).txt"),
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)


class TestPackageValidation:
    def test_valid_package_name(self, validator):
        action = Action(
            action_type=ActionType.PACKAGE_INSTALL,
            target="vim",
            parameters=PackageParams(name="vim"),
        )
        validator.validate_action(action)

    def test_rejects_invalid_package_name(self, validator):
        action = Action(
            action_type=ActionType.PACKAGE_INSTALL,
            target="bad;name",
            parameters=PackageParams(name="bad;name"),
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)

    def test_rejects_protected_package_removal(self, validator):
        action = Action(
            action_type=ActionType.PACKAGE_REMOVE,
            target="firefox",
            parameters=PackageParams(name="firefox"),
            destructive=True,
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)


class TestShellCommandValidation:
    def test_valid_safe_command(self, validator):
        action = Action(
            action_type=ActionType.SHELL_COMMAND,
            target="ls",
            parameters=ShellCommandParams(command="ls", args=["-la", "/home"]),
        )
        validator.validate_action(action)

    def test_rejects_unsafe_command(self, validator):
        action = Action(
            action_type=ActionType.SHELL_COMMAND,
            target="nmap",
            parameters=ShellCommandParams(command="nmap", args=["http://evil.com"]),
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)

    def test_rejects_args_with_metacharacters(self, validator):
        validator._sanitizer._is_windows = False
        action = Action(
            action_type=ActionType.SHELL_COMMAND,
            target="echo",
            parameters=ShellCommandParams(command="echo", args=["hello; rm -rf /"]),
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)


class TestRestrictions:
    def test_rejects_protected_folder(self, validator):
        action = Action(
            action_type=ActionType.FILE_WRITE,
            target="/home/user/private/secret.txt",
            parameters=FileParams(path="/home/user/private/secret.txt", content="hack"),
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)

    def test_rejects_blocked_command(self, validator):
        action = Action(
            action_type=ActionType.SHELL_COMMAND,
            target="rm",
            parameters=ShellCommandParams(command="rm", args=["-rf", "/"]),
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)


class TestRootValidation:
    def test_rejects_root_when_disabled(self, validator):
        action = Action(
            action_type=ActionType.FILE_WRITE,
            target="/etc/hosts",
            parameters=FileParams(path="/etc/hosts", content="127.0.0.1 evil"),
            requires_root=True,
        )
        with pytest.raises(ValidationError):
            validator.validate_action(action)

    def test_allows_root_when_enabled(self, config):
        config.security.root_enabled = True
        validator = ActionValidator(config)
        action = Action(
            action_type=ActionType.FILE_READ,
            target="/etc/hosts",
            parameters=FileParams(path="/etc/hosts"),
            requires_root=True,
        )
        validator.validate_action(action)
