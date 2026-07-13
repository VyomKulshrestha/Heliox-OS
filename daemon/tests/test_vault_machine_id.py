"""Tests for KeyVault._get_machine_id() cross-platform behaviour.

Verifies that the correct platform-specific source is used on each OS and
that the hardcoded fallback is only reached when every source fails.
"""

from __future__ import annotations

import subprocess
import types
from unittest.mock import MagicMock, patch

from pilot.security.vault import KeyVault

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linux_path_raises(self, *args, **kwargs):  # noqa: ANN001
    raise OSError("linux paths not present")


def _make_ioreg_output(uuid: str) -> str:
    return '  "IOPlatformUUID" = "' + uuid + '"\n  "IOPlatformSerialNumber" = "X12345678"\n'


def _make_winreg_mock(guid: str) -> types.ModuleType:
    """Build a minimal winreg stub that returns the given GUID."""
    winreg = types.ModuleType("winreg")
    winreg.HKEY_LOCAL_MACHINE = 0x80000002

    mock_key = MagicMock()
    mock_key.__enter__ = MagicMock(return_value=mock_key)
    mock_key.__exit__ = MagicMock(return_value=False)

    winreg.OpenKey = MagicMock(return_value=mock_key)
    winreg.QueryValueEx = MagicMock(return_value=(guid, 1))
    return winreg


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------


def test_machine_id_linux_primary():
    """Returns the content of /etc/machine-id when the file is present."""

    def fake_read_text(self, *args, **kwargs):  # noqa: ANN001
        from pathlib import Path

        if Path(self).as_posix() in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            return "abc123linux\n"
        raise OSError("not found")

    with patch("pilot.security.vault.Path.read_text", fake_read_text):
        result = KeyVault._get_machine_id()

    assert result == "abc123linux"


def test_machine_id_linux_fallback_to_dbus():
    """/var/lib/dbus/machine-id is used when /etc/machine-id is absent."""

    def fake_read_text(self, *args, **kwargs):  # noqa: ANN001
        from pathlib import Path

        posix_path = Path(self).as_posix()
        if posix_path == "/etc/machine-id":
            raise OSError("no such file")
        if posix_path == "/var/lib/dbus/machine-id":
            return "dbus999\n"
        raise OSError("not found")

    with patch("pilot.security.vault.Path.read_text", fake_read_text):
        result = KeyVault._get_machine_id()

    assert result == "dbus999"


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------


def test_machine_id_macos(monkeypatch):
    """IOPlatformUUID is extracted from ioreg output on Darwin."""
    expected = "AABBCCDD-1122-3344-5566-778899AABBCC"

    monkeypatch.setattr("pilot.security.vault.Path.read_text", _linux_path_raises)

    completed = subprocess.CompletedProcess(
        args=["ioreg"],
        returncode=0,
        stdout=_make_ioreg_output(expected),
        stderr="",
    )

    with patch("platform.system", return_value="Darwin"), patch("subprocess.run", return_value=completed):
        result = KeyVault._get_machine_id()

    assert result == expected


def test_machine_id_macos_ioreg_timeout(monkeypatch):
    """Falls back to hardcoded constant when ioreg times out on Darwin."""
    monkeypatch.setattr("pilot.security.vault.Path.read_text", _linux_path_raises)

    with (
        patch("platform.system", return_value="Darwin"),
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ioreg", timeout=5)),
    ):
        result = KeyVault._get_machine_id()

    assert result == "pilot-fallback-id"


def test_machine_id_macos_ioreg_missing_uuid(monkeypatch):
    """Falls back when ioreg output contains no IOPlatformUUID line."""
    monkeypatch.setattr("pilot.security.vault.Path.read_text", _linux_path_raises)

    completed = subprocess.CompletedProcess(
        args=["ioreg"],
        returncode=0,
        stdout="no uuid here\n",
        stderr="",
    )

    with patch("platform.system", return_value="Darwin"), patch("subprocess.run", return_value=completed):
        result = KeyVault._get_machine_id()

    assert result == "pilot-fallback-id"


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_machine_id_windows(monkeypatch):
    """MachineGuid is read from the registry on Windows."""
    expected = "11223344-aabb-ccdd-eeff-001122334455"

    monkeypatch.setattr("pilot.security.vault.Path.read_text", _linux_path_raises)

    winreg_stub = _make_winreg_mock(expected)

    with patch("platform.system", return_value="Windows"), patch.dict("sys.modules", {"winreg": winreg_stub}):
        result = KeyVault._get_machine_id()

    assert result == expected


def test_machine_id_windows_registry_error(monkeypatch):
    """Falls back to hardcoded constant when the registry key is inaccessible."""
    monkeypatch.setattr("pilot.security.vault.Path.read_text", _linux_path_raises)

    winreg_stub = types.ModuleType("winreg")
    winreg_stub.HKEY_LOCAL_MACHINE = 0x80000002
    winreg_stub.OpenKey = MagicMock(side_effect=OSError("access denied"))

    with patch("platform.system", return_value="Windows"), patch.dict("sys.modules", {"winreg": winreg_stub}):
        result = KeyVault._get_machine_id()

    assert result == "pilot-fallback-id"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_machine_id_fallback_unknown_platform(monkeypatch):
    """Returns the hardcoded fallback on an unrecognised platform."""
    monkeypatch.setattr("pilot.security.vault.Path.read_text", _linux_path_raises)

    with patch("platform.system", return_value="FreeBSD"):
        result = KeyVault._get_machine_id()

    assert result == "pilot-fallback-id"


def test_machine_id_never_returns_empty(monkeypatch):
    """The returned value is always a non-empty string."""
    monkeypatch.setattr("pilot.security.vault.Path.read_text", _linux_path_raises)

    with patch("platform.system", return_value="FreeBSD"):
        result = KeyVault._get_machine_id()

    assert isinstance(result, str)
    assert len(result) > 0
