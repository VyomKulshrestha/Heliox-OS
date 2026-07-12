import pytest

from pilot.config import PilotConfig
from pilot.security.sanitizer import SanitizationError, Sanitizer


@pytest.fixture
def sanitizer():
    config = PilotConfig()
    return Sanitizer(config)


def test_absolute_path_windows_posix(sanitizer):
    # Should not raise exception
    sanitizer.validate_path("C:\\Windows\\System32", 0)
    sanitizer.validate_path("/etc/passwd", 0)

    # Should raise exception for relative paths
    with pytest.raises(SanitizationError):
        sanitizer.validate_path("relative/path", 0)

    with pytest.raises(SanitizationError):
        sanitizer.validate_path("../traversal", 0)


def test_destructive_commands_allowed_by_sanitizer(sanitizer):
    destructive = ["rm", "del", "chmod", "kill", "docker"]
    for cmd in destructive:
        # Sanitizer should allow them; validator will block them later
        sanitizer.validate_shell_command(cmd, [], 0)


def test_safe_commands_allowed(sanitizer):
    safe = ["echo", "ls", "cat", "whoami"]
    for cmd in safe:
        sanitizer.validate_shell_command(cmd, [], 0)


def test_url_scheme_validation(sanitizer):
    # Should not raise
    sanitizer.validate_url("http://example.com", 0)
    sanitizer.validate_url("example.com", 0)  # Should normalize to https://example.com

    # Should raise for bad schemes
    with pytest.raises(SanitizationError):
        sanitizer.validate_url("file:///etc/passwd", 0)
    with pytest.raises(SanitizationError):
        sanitizer.validate_url("javascript:alert(1)", 0)
    with pytest.raises(SanitizationError):
        sanitizer.validate_url("data:text/html,<html>", 0)
