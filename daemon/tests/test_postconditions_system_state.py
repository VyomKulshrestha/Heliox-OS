"""Post-condition verifiers for system state (postconditions/system_state.py).

Every probe is faked: no test locks the screen, changes a real file's mode outside a temp folder, writes the real
registry (except the CI-only read-back under a key the test creates and removes), enables or disables a service,
mounts a disk, changes Wi-Fi or installs anything.
"""

from __future__ import annotations

import os
import plistlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, PowerParams
from pilot.agents.postconditions import system_state
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.system.platform_detect import Platform


async def _verify(action_type: ActionType, parameters, *, success: bool = True, output: str | None = "done"):
    action = Action(action_type=action_type, target="controlled-target", parameters=parameters)
    result = ActionResult(action=action, success=success, output=output, error=None if success else "handler failed")
    return await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result])


async def _detail(action_type: ActionType, parameters, *, output: str | None = "done") -> str:
    """The verifier's own sentence (verify() shows it only for a MISMATCH)."""
    action = Action(action_type=action_type, target="controlled-target", parameters=parameters)
    return (await Verifier(MagicMock())._verify_single(ActionResult(action=action, success=True, output=output)))[1]


def _assert_verified(verification) -> None:
    assert verification.passed is True, verification.details
    assert verification.independently_observed_actions == [0]
    assert "VERIFIED" in verification.details[0]


def _assert_mismatch(verification) -> str:
    assert verification.passed is False, verification.details
    assert verification.independently_observed_actions == [0]
    assert "MISMATCH" in verification.details[0]
    return verification.details[0]


def _assert_reported(verification) -> str:
    assert verification.passed is True, verification.details
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]
    return verification.details[0]


def _assert_failed(verification) -> None:
    assert verification.passed is False
    assert verification.independently_observed_actions == []
    assert "FAILED" in verification.details[0]


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Readings are counted, never timed: the pause between them returns at once."""
    pause = AsyncMock()
    monkeypatch.setattr(system_state, "_pause", pause)
    return pause


def test_system_state_verifiers_are_registered() -> None:
    assert POSTCONDITION_VERIFIERS[ActionType.POWER_LOCK] == "session_locked_postcondition"


# ── power_lock ──────────────────────────────────────────────────────────────────────────────────────────────────


def _wts_buffer(flags: int, level: int = 1) -> bytes:
    return (
        level.to_bytes(4, "little")
        + bytes(4)
        + (1).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + flags.to_bytes(4, "little", signed=True)
        + bytes(212)
    )


def _ioreg(uid: int, locked: bool | None) -> str:
    user = {"kCGSSessionUserIDKey": uid, "kCGSSessionOnConsoleKey": True, "kCGSSessionUserNameKey": "someone"}
    if locked is not None:
        user["CGSSessionScreenIsLocked"] = locked
    return plistlib.dumps({"IOConsoleUsers": [user], "IORegistryEntryName": "Root"}).decode("utf-8")


def test_wts_flags_parse_lock_unlock_and_unknown() -> None:
    from pilot.system.power import parse_wts_session_flags

    assert parse_wts_session_flags(_wts_buffer(0)) is True
    assert parse_wts_session_flags(_wts_buffer(1)) is False
    assert parse_wts_session_flags(_wts_buffer(-1)) is None
    assert parse_wts_session_flags(_wts_buffer(0, level=2)) is None
    assert parse_wts_session_flags(b"\x01\x00") is None


@pytest.mark.skipif(sys.platform != "win32", reason="the Terminal Services API exists only on Windows")
def test_windows_session_record_is_readable_without_elevation() -> None:
    from pilot.system.power import _windows_session_info, parse_wts_session_flags

    # A read of this test process's own session: on an unlocked runner it is never "locked". A runner whose
    # service session has no record at all says so with an OSError, which is the "observer broke" path.
    try:
        raw = _windows_session_info()
    except OSError as error:
        pytest.skip(f"this session has no Terminal Services record: {error}")
    assert parse_wts_session_flags(raw) in {False, None}


async def test_windows_lock_verified_from_session_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.WINDOWS)
    readings = iter([_wts_buffer(1), _wts_buffer(1), _wts_buffer(0)])
    monkeypatch.setattr(power, "_windows_session_info", lambda: next(readings))

    verification = await _verify(ActionType.POWER_LOCK, PowerParams())

    _assert_verified(verification)
    readings = iter([_wts_buffer(0)])
    assert "Windows session" in await _detail(ActionType.POWER_LOCK, PowerParams())


async def test_windows_lock_that_never_lands_is_a_mismatch(monkeypatch: pytest.MonkeyPatch, _no_waiting) -> None:
    from pilot.system import power

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(power, "_windows_session_info", lambda: _wts_buffer(1))

    detail = _assert_mismatch(await _verify(ActionType.POWER_LOCK, PowerParams()))

    assert "Expected the screen to be locked" in detail
    assert _no_waiting.await_count == system_state.LOCK_READINGS - 1


async def test_windows_unknown_lock_state_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(power, "_windows_session_info", lambda: _wts_buffer(-1))

    _assert_reported(await _verify(ActionType.POWER_LOCK, PowerParams()))
    assert "Could not read the lock state" in await _detail(ActionType.POWER_LOCK, PowerParams())


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="macOS and Linux have user ids")
async def test_macos_lock_read_from_console_record(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    run = AsyncMock(side_effect=[(0, _ioreg(os.getuid(), None), ""), (0, _ioreg(os.getuid(), True), "")])
    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(power, "run_command", run)

    _assert_verified(await _verify(ActionType.POWER_LOCK, PowerParams()))
    assert run.await_args_list[0].args[0] == ["ioreg", "-n", "Root", "-d1", "-a"]


def test_macos_console_record_parsing() -> None:
    from pilot.system.power import parse_mac_console_lock

    assert parse_mac_console_lock(_ioreg(501, True).encode(), 501) is True
    assert parse_mac_console_lock(_ioreg(501, False).encode(), 501) is False
    # The key is present only while locked.
    assert parse_mac_console_lock(_ioreg(501, None).encode(), 501) is False
    # Another user's console record says nothing about this user's session.
    assert parse_mac_console_lock(_ioreg(502, True).encode(), 501) is None
    assert parse_mac_console_lock(b"not a plist", 501) is None


async def test_macos_unlocked_screen_is_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(power, "run_command", AsyncMock(return_value=(0, _ioreg(501, None), "")))
    monkeypatch.setattr(power.os, "getuid", lambda: 501, raising=False)

    _assert_mismatch(await _verify(ActionType.POWER_LOCK, PowerParams()))


async def test_linux_lock_verified_from_logind_locked_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    run = AsyncMock(side_effect=[(0, "no\n", ""), (0, "yes\n", "")])
    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(power, "run_command", run)
    monkeypatch.setenv("XDG_SESSION_ID", "c2")

    _assert_verified(await _verify(ActionType.POWER_LOCK, PowerParams()))
    assert run.await_args_list[0].args[0] == ["loginctl", "show-session", "c2", "--property=LockedHint", "--value"]


async def test_linux_lock_signal_with_no_locker_listening_is_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``loginctl lock-session`` exits 0 when no screen locker acts on logind's Lock signal: the handler reports
    success and LockedHint stays "no". The verifier must catch exactly that."""
    from pilot.system import power

    calls: list[list[str]] = []

    async def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:2] == ["loginctl", "lock-session"]:
            return (0, "", "")
        return (0, "no\n", "")

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(power, "run_command", fake_run)
    monkeypatch.delenv("XDG_SESSION_ID", raising=False)

    assert await power.lock_screen() == "Screen locked"
    detail = _assert_mismatch(await _verify(ActionType.POWER_LOCK, PowerParams()))

    assert "screen locker" in detail
    assert calls[1] == ["loginctl", "show-session", "self", "--property=LockedHint", "--value"]


async def test_linux_without_loginctl_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(power, "run_command", AsyncMock(return_value=(-1, "", "Command not found: loginctl")))

    _assert_reported(await _verify(ActionType.POWER_LOCK, PowerParams()))
    assert "no loginctl" in await _detail(ActionType.POWER_LOCK, PowerParams())


async def test_linux_lock_read_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(power, "run_command", AsyncMock(return_value=(1, "", "Failed to get session: denied")))

    verification = await _verify(ActionType.POWER_LOCK, PowerParams())

    assert verification.passed is False
    assert "Verification error" in verification.details[0]


async def test_failed_lock_never_reaches_the_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import power

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(power, "run_command", AsyncMock(return_value=(1, "", "no session")))
    observer = AsyncMock()
    monkeypatch.setattr(power, "screen_locked", observer)

    with pytest.raises(RuntimeError, match="Lock failed"):
        await power.lock_screen()
    _assert_failed(await _verify(ActionType.POWER_LOCK, PowerParams(), success=False))
    observer.assert_not_awaited()


# ── file_permissions ────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no POSIX mode bits")
async def test_chmod_600_verifies_and_a_mode_changed_back_is_a_mismatch(tmp_path) -> None:
    from pilot.actions import FileParams
    from pilot.system.filesystem import file_permissions

    target = tmp_path / "secret.txt"
    target.write_text("x", encoding="utf-8")
    target.chmod(0o644)
    params = FileParams(path=str(target), permissions="600")

    await file_permissions(str(target), "600")
    _assert_verified(await _verify(ActionType.FILE_PERMISSIONS, params))

    target.chmod(0o644)  # changed back between execute and verify
    detail = _assert_mismatch(await _verify(ActionType.FILE_PERMISSIONS, params))
    assert "Expected mode 0600" in detail and "found 0644" in detail


class _FakeOs:
    def __init__(self, mode: int | None) -> None:
        self.mode = mode

    def stat(self, path):
        if self.mode is None:
            raise FileNotFoundError(path)
        return os.stat_result((0o100000 | self.mode, 0, 0, 0, 0, 0, 0, 0, 0, 0))


@pytest.mark.parametrize(
    ("asked", "found", "outcome"),
    [("600", 0o600, "verified"), ("0o750", 0o750, "verified"), ("600", 0o644, "mismatch"), ("600", None, "mismatch")],
)
async def test_file_mode_read_with_stat(monkeypatch: pytest.MonkeyPatch, asked, found, outcome) -> None:
    from pilot.actions import FileParams

    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(system_state, "os", _FakeOs(found))

    verification = await _verify(ActionType.FILE_PERMISSIONS, FileParams(path="/srv/key", permissions=asked))

    if outcome == "verified":
        _assert_verified(verification)
    else:
        detail = _assert_mismatch(verification)
        assert ("no longer exists" in detail) if found is None else ("found 0644" in detail)


async def test_permissions_on_windows_and_plain_reads_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.actions import FileParams

    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.WINDOWS)
    _assert_reported(await _verify(ActionType.FILE_PERMISSIONS, FileParams(path="C:/x.txt", permissions="600")))
    assert "ACL" in await _detail(ActionType.FILE_PERMISSIONS, FileParams(path="C:/x.txt", permissions="600"))

    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    _assert_reported(await _verify(ActionType.FILE_PERMISSIONS, FileParams(path="/srv/key")))


async def test_permissions_on_a_missing_path_fail_in_the_handler(tmp_path) -> None:
    from pilot.actions import FileParams
    from pilot.system.filesystem import file_permissions

    with pytest.raises(FileNotFoundError):
        await file_permissions(str(tmp_path / "absent"), "600")
    params = FileParams(path=str(tmp_path / "absent"), permissions="600")
    _assert_failed(await _verify(ActionType.FILE_PERMISSIONS, params, success=False))


# ── registry_write ──────────────────────────────────────────────────────────────────────────────────────────────


def _registry_answer(kind: str | None, data=None, expected=None, *, key: bool = True) -> tuple[int, str, str]:
    import json

    if kind is None:
        return (0, json.dumps({"key": key, "present": False}), "")
    return (0, json.dumps({"key": True, "present": True, "kind": kind, "data": data, "expected": expected}), "")


def _registry(value_type: str, data: str):
    from pilot.actions import RegistryParams

    return RegistryParams(
        key_path=r"HKCU\Software\Controlled", value_name="Setting", value_data=data, value_type=value_type
    )


@pytest.mark.parametrize(
    ("value_type", "data", "answer", "outcome"),
    [
        ("REG_DWORD", "0x10", _registry_answer("DWord", [16], [16]), "verified"),
        ("REG_DWORD", "4294967295", _registry_answer("DWord", [-1], [4294967295]), "verified"),
        ("REG_SZ", "hello", _registry_answer("String", ["hello"], [None]), "verified"),
        ("REG_MULTI_SZ", "one", _registry_answer("MultiString", ["one"], ["one"]), "verified"),
        ("REG_QWORD", "5", _registry_answer("QWord", [5], [5]), "verified"),
        ("REG_DWORD", "16", _registry_answer("DWord", [15], [16]), "mismatch"),
        ("REG_DWORD", "16", _registry_answer("String", ["16"], [16]), "mismatch"),
        ("REG_SZ", "hello", _registry_answer("String", ["other"], None), "mismatch"),
        ("REG_SZ", "hello", _registry_answer(None), "mismatch"),
        ("REG_SZ", "hello", _registry_answer(None, key=False), "mismatch"),
        ("REG_DWORD", "ten", _registry_answer("DWord", [10], None), "reported"),
    ],
)
async def test_registry_value_read_back(monkeypatch: pytest.MonkeyPatch, value_type, data, answer, outcome) -> None:
    from pilot.system import registry_readback

    run = AsyncMock(return_value=answer)
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(registry_readback, "run_powershell", run)

    verification = await _verify(ActionType.REGISTRY_WRITE, _registry(value_type, data))

    if outcome == "verified":
        _assert_verified(verification)
    elif outcome == "mismatch":
        detail = _assert_mismatch(verification)
        assert "hello" not in detail and "other" not in detail  # stored text never appears in the sentence
    else:
        _assert_reported(verification)
    # The key, name and data are values, never script text.
    assert run.await_args.kwargs["values"]["key"] == r"HKCU\Software\Controlled"
    assert "Controlled" not in run.await_args.args[0]


async def test_registry_powershell_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import registry_readback

    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(registry_readback, "run_powershell", AsyncMock(return_value=(1, "", "Access denied")))

    verification = await _verify(ActionType.REGISTRY_WRITE, _registry("REG_SZ", "x"))

    assert verification.passed is False and "Verification error" in verification.details[0]


@pytest.mark.parametrize("platform", [Platform.LINUX, Platform.MACOS])
async def test_registry_off_windows_is_reported(monkeypatch: pytest.MonkeyPatch, platform) -> None:
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", platform)

    _assert_reported(await _verify(ActionType.REGISTRY_WRITE, _registry("REG_SZ", "x")))


async def test_registry_write_failure_is_a_failed_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.agents.executor import Executor
    from pilot.system import platform_detect

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(platform_detect, "run_powershell", AsyncMock(return_value=(1, "", "Requested registry access")))
    action = Action(action_type=ActionType.REGISTRY_WRITE, target="x", parameters=_registry("REG_SZ", "x"))

    with pytest.raises(RuntimeError, match="Registry write failed"):
        await Executor._exec_registry_write(MagicMock(), action)
    _assert_failed(await _verify(ActionType.REGISTRY_WRITE, _registry("REG_SZ", "x"), success=False))


@pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("CI"),
    reason="writes the real registry: only on a Windows CI runner, under a key the test creates and removes",
)
async def test_registry_write_reads_back_dword_and_string_on_windows_ci() -> None:
    import uuid
    import winreg

    from pilot.actions import RegistryParams
    from pilot.agents.executor import Executor

    subkey = rf"Software\HelioxVerifierTest\{uuid.uuid4().hex}"
    winreg.CreateKey(winreg.HKEY_CURRENT_USER, subkey).Close()
    try:
        for value_type, data in (("REG_DWORD", "0x10"), ("REG_SZ", "hello")):
            params = RegistryParams(
                key_path=rf"HKCU\{subkey}", value_name=value_type, value_data=data, value_type=value_type
            )
            action = Action(action_type=ActionType.REGISTRY_WRITE, target="x", parameters=params)
            await Executor._exec_registry_write(MagicMock(), action)
            _assert_verified(await _verify(ActionType.REGISTRY_WRITE, params))

        # Changed behind Heliox's back: the read-back sees it.
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, subkey, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "REG_DWORD", 0, winreg.REG_DWORD, 17)
            winreg.SetValueEx(key, "REG_SZ", 0, winreg.REG_EXPAND_SZ, "hello")
        params = RegistryParams(
            key_path=rf"HKCU\{subkey}", value_name="REG_DWORD", value_data="0x10", value_type="REG_DWORD"
        )
        assert "found 17" in _assert_mismatch(await _verify(ActionType.REGISTRY_WRITE, params))
        params = RegistryParams(
            key_path=rf"HKCU\{subkey}", value_name="REG_SZ", value_data="hello", value_type="REG_SZ"
        )
        assert "REG_EXPAND_SZ" in _assert_mismatch(await _verify(ActionType.REGISTRY_WRITE, params))
    finally:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\HelioxVerifierTest", 0, winreg.KEY_ALL_ACCESS
        ) as parent:
            if winreg.QueryInfoKey(parent)[0] == 0:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\HelioxVerifierTest")
