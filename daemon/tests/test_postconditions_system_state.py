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
# The handler writes with Set-ItemProperty, which the suite's system guard refuses; this test alone may use it, on CI.
@pytest.mark.real_system("Set-ItemProperty")
async def test_registry_write_reads_back_dword_and_string_on_windows_ci() -> None:
    import contextlib
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
        # Removed whatever happened above: the key holds only values, so DeleteKey takes it with them.
        parent_path = r"Software\HelioxVerifierTest"
        with contextlib.suppress(FileNotFoundError):
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
        empty = False
        with contextlib.suppress(OSError), winreg.OpenKey(winreg.HKEY_CURRENT_USER, parent_path) as parent:
            empty = winreg.QueryInfoKey(parent)[0] == 0
        if empty:
            with contextlib.suppress(OSError):
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, parent_path)


# ── service_enable and service_disable ──────────────────────────────────────────────────────────────────────────


def _service(name: str = "backup.timer", user_scope: bool = False):
    from pilot.actions import ServiceParams

    return ServiceParams(name=name, user_scope=user_scope)


@pytest.mark.parametrize(
    ("action_type", "printed", "code", "outcome"),
    [
        (ActionType.SERVICE_ENABLE, "enabled\n", 0, "verified"),
        (ActionType.SERVICE_ENABLE, "disabled\n", 1, "mismatch"),
        (ActionType.SERVICE_ENABLE, "static\n", 0, "mismatch"),
        (ActionType.SERVICE_ENABLE, "enabled-runtime\n", 0, "mismatch"),
        (ActionType.SERVICE_ENABLE, "masked\n", 1, "mismatch"),
        (ActionType.SERVICE_DISABLE, "disabled\n", 1, "verified"),
        (ActionType.SERVICE_DISABLE, "masked\n", 1, "verified"),
        (ActionType.SERVICE_DISABLE, "masked-runtime\n", 1, "verified"),
        (ActionType.SERVICE_DISABLE, "enabled\n", 0, "mismatch"),
        (ActionType.SERVICE_DISABLE, "static\n", 0, "mismatch"),
        (ActionType.SERVICE_DISABLE, "indirect\n", 0, "mismatch"),
    ],
)
async def test_service_boot_state_from_systemctl_is_enabled(
    monkeypatch: pytest.MonkeyPatch, action_type, printed, code, outcome
) -> None:
    from pilot.system import platform_detect

    run = AsyncMock(return_value=(code, printed, ""))
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(platform_detect, "run_command", run)

    verification = await _verify(action_type, _service())

    if outcome == "verified":
        _assert_verified(verification)
    else:
        detail = _assert_mismatch(verification)
        if printed.strip() == "static":
            assert "no [Install] section" in detail
    assert run.await_args.args[0] == ["systemctl", "is-enabled", "--", "backup.timer"]
    assert "pkexec" not in run.await_args.args[0]


async def test_user_scope_service_is_read_with_user_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import platform_detect

    run = AsyncMock(return_value=(0, "enabled\n", ""))
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(platform_detect, "run_command", run)

    _assert_verified(await _verify(ActionType.SERVICE_ENABLE, _service("sync.service", user_scope=True)))
    assert run.await_args.args[0] == ["systemctl", "--user", "is-enabled", "--", "sync.service"]
    assert run.await_args.kwargs["timeout"] == 15


async def test_systemctl_without_a_state_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import platform_detect

    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(
        platform_detect, "run_command", AsyncMock(return_value=(1, "", "System has not been booted with systemd"))
    )

    verification = await _verify(ActionType.SERVICE_DISABLE, _service())

    assert verification.passed is False and "Verification error" in verification.details[0]


@pytest.mark.parametrize("platform", [Platform.WINDOWS, Platform.MACOS])
@pytest.mark.parametrize("action_type", [ActionType.SERVICE_ENABLE, ActionType.SERVICE_DISABLE])
async def test_services_off_linux_are_reported(monkeypatch: pytest.MonkeyPatch, platform, action_type) -> None:
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", platform)

    _assert_reported(await _verify(action_type, _service()))


async def test_failed_enable_never_reaches_the_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import systemctl

    monkeypatch.setattr(
        systemctl, "_run_systemctl", AsyncMock(return_value=(1, "", "Unit file backup.timer does not exist."))
    )
    observer = AsyncMock()
    monkeypatch.setattr(systemctl, "is_enabled", observer)

    with pytest.raises(RuntimeError, match="Failed to enable"):
        await systemctl.service_enable("backup.timer")
    _assert_failed(await _verify(ActionType.SERVICE_ENABLE, _service(), success=False))
    observer.assert_not_awaited()


# ── disk_mount ──────────────────────────────────────────────────────────────────────────────────────────────────


def _mount(device: str = "/dev/sdb1", mount_point: str = "/mnt/controlled"):
    from pilot.actions import DiskManageParams

    return DiskManageParams(device=device, mount_point=mount_point)


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [((0, "/dev/sdb1\n", ""), "verified"), ((1, "", ""), "mismatch"), ((0, "/dev/sdc2\n", ""), "mismatch")],
)
async def test_linux_mount_read_from_findmnt(monkeypatch: pytest.MonkeyPatch, answer, outcome) -> None:
    from pilot.system import disks

    run = AsyncMock(return_value=answer)
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(disks, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(disks, "run_command", run)

    verification = await _verify(ActionType.DISK_MOUNT, _mount())

    if outcome == "verified":
        _assert_verified(verification)
    else:
        detail = _assert_mismatch(verification)
        assert ("nothing is mounted there" in detail) if answer[0] == 1 else ("/dev/sdc2 is mounted there" in detail)
    assert run.await_args_list[0].args[0] == [
        "findmnt",
        "--noheadings",
        "--first-only",
        "--output",
        "SOURCE",
        "--mountpoint",
        "/mnt/controlled",
    ]


async def test_linux_mount_by_uuid_is_resolved_by_findmnt(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import disks

    run = AsyncMock(side_effect=[(0, "/dev/sdb1\n", ""), (0, "/dev/sdb1\n", "")])
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(disks, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(disks, "run_command", run)

    _assert_verified(await _verify(ActionType.DISK_MOUNT, _mount(device="UUID=1234-ABCD")))
    assert run.await_args_list[1].args[0][-2:] == ["--source", "UUID=1234-ABCD"]


async def test_findmnt_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import disks

    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(disks, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(disks, "run_command", AsyncMock(return_value=(-1, "", "Command not found: findmnt")))

    verification = await _verify(ActionType.DISK_MOUNT, _mount())

    assert verification.passed is False and "Verification error" in verification.details[0]


MAC_MOUNT = (
    "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n"
    "/dev/disk4s1 on /Volumes/Backup Drive (apfs, local, nodev, nosuid, journaled, noowners)\n"
)


@pytest.mark.parametrize(
    ("device", "mount_point", "outcome"),
    [
        ("/dev/disk4s1", "/Volumes/Backup Drive", "verified"),
        ("/dev/disk5s1", "/Volumes/Backup Drive", "mismatch"),
        ("/dev/disk4s1", "/Volumes/Other", "mismatch"),
    ],
)
async def test_macos_mount_read_from_mount_table(monkeypatch: pytest.MonkeyPatch, device, mount_point, outcome) -> None:
    from pilot.system import disks

    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(disks, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(disks, "run_command", AsyncMock(return_value=(0, MAC_MOUNT, "")))

    verification = await _verify(ActionType.DISK_MOUNT, _mount(device, mount_point))

    if outcome == "verified":
        _assert_verified(verification)
    else:
        _assert_mismatch(verification)


async def test_mount_on_windows_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.WINDOWS)

    _assert_reported(await _verify(ActionType.DISK_MOUNT, _mount("E:", "C:/mnt")))


async def test_failed_mount_is_a_failed_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.agents.executor import Executor
    from pilot.system import platform_detect

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(platform_detect, "run_command", AsyncMock(return_value=(32, "", "mount: wrong fs type")))
    action = Action(action_type=ActionType.DISK_MOUNT, target="x", parameters=_mount())

    with pytest.raises(RuntimeError, match="Mount failed"):
        await Executor._exec_disk_mount(MagicMock(), action)
    _assert_failed(await _verify(ActionType.DISK_MOUNT, _mount(), success=False))


# ── wifi_connect ────────────────────────────────────────────────────────────────────────────────────────────────


def _wifi(ssid: str = "Cafe", interface: str | None = None):
    from pilot.actions import WifiParams

    return WifiParams(ssid=ssid, password="never-shown-pass", interface=interface)


def test_nmcli_terse_parsing_unescapes_names() -> None:
    from pilot.system.network import parse_nmcli_active_ssids

    text = "no:Neighbour\nyes:Home\\:5G\nyes:Back\\\\slash\nno:\nyes:\n"
    assert parse_nmcli_active_ssids(text) == ["Home:5G", "Back\\slash"]


async def test_linux_wifi_read_from_networkmanager(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    run = AsyncMock(side_effect=[(0, "no:Cafe\nyes:Home\n", ""), (0, "yes:Cafe\nno:Home\n", "")])
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", run)

    _assert_verified(await _verify(ActionType.WIFI_CONNECT, _wifi(interface="wlan0")))
    args = run.await_args.args[0]
    assert args == ["nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi", "list", "ifname", "wlan0", "--rescan", "no"]
    assert run.await_args.kwargs["env"]["LC_ALL"] == "C"


async def test_linux_other_network_is_a_mismatch(monkeypatch: pytest.MonkeyPatch, _no_waiting) -> None:
    from pilot.system import network

    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", AsyncMock(return_value=(0, "no:Cafe\nyes:Home\n", "")))

    detail = _assert_mismatch(await _verify(ActionType.WIFI_CONNECT, _wifi()))

    assert "Expected to be joined to Cafe" in detail and "on Home" in detail
    assert "never-shown-pass" not in detail
    assert _no_waiting.await_count == system_state.WIFI_READINGS - 1


async def test_windows_wifi_read_from_wlan_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    readings = iter([network.WifiReading([]), network.WifiReading(["Cafe"])])
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(network, "_windows_connected_ssids", lambda: next(readings))

    _assert_verified(await _verify(ActionType.WIFI_CONNECT, _wifi()))


async def test_windows_without_location_access_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    withheld = network.WifiReading(None, "Windows withholds the network name from apps without location access")
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(network, "_windows_connected_ssids", lambda: withheld)

    _assert_reported(await _verify(ActionType.WIFI_CONNECT, _wifi()))
    assert "location access" in await _detail(ActionType.WIFI_CONNECT, _wifi())


@pytest.mark.skipif(sys.platform != "win32", reason="the WLAN API exists only on Windows")
def test_windows_wlan_api_read_runs_without_elevation() -> None:
    from pilot.system.network import WifiReading, _windows_connected_ssids

    # A read only: a runner without Wi-Fi hardware answers that the WLAN service is absent.
    assert isinstance(_windows_connected_ssids(), WifiReading)


@pytest.mark.parametrize(
    ("answers", "outcome"),
    [
        ([(0, "Current Wi-Fi Network: Cafe\n", "")], "verified"),
        ([(0, "Current AirPort Network: Cafe\n", "")], "verified"),
        (
            [
                (0, "You are not associated with an AirPort network.\n", ""),
                (0, "<dictionary> {\n  SSID : Cafe\n}\n", ""),
            ],
            "verified",
        ),
        (
            [
                (0, "You are not associated with an AirPort network.\n", ""),
                (0, "<dictionary> {\n  SSID : <redacted>\n}\n", ""),
            ],
            "reported",
        ),
        ([(0, "Current Wi-Fi Network: Home\n", "")] * 10, "mismatch"),
    ],
)
async def test_macos_wifi_read(monkeypatch: pytest.MonkeyPatch, answers, outcome) -> None:
    from pilot.system import network

    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(network, "run_command", AsyncMock(side_effect=answers))

    verification = await _verify(ActionType.WIFI_CONNECT, _wifi())

    {"verified": _assert_verified, "mismatch": _assert_mismatch, "reported": _assert_reported}[outcome](verification)


async def test_failed_wifi_connect_never_reaches_the_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", AsyncMock(return_value=(10, "", "No network with SSID 'Cafe' found.")))
    observer = AsyncMock()
    monkeypatch.setattr(network, "connected_ssids", observer)

    with pytest.raises(RuntimeError, match="WiFi connect failed"):
        await network.wifi_connect("Cafe")
    _assert_failed(await _verify(ActionType.WIFI_CONNECT, _wifi(), success=False))
    observer.assert_not_awaited()


# ── package_update ──────────────────────────────────────────────────────────────────────────────────────────────

WINGET_TABLE = (
    "\r   - \r   \\ \r"
    "Name                      Id                      Version        Available      Source\n"
    "-----------------------------------------------------------------------------------------\n"
    "Microsoft Edge            Microsoft.Edge          119.0.2151.58  119.0.2151.72  winget\n"
    "7-Zip 23.01 (x64)         7zip.7zip               23.01          24.07          winget\n"
    "2 upgrades available.\n"
    "\n"
    "The following packages have an upgrade available, but require explicit targeting for upgrade:\n"
    "Name     Id          Version  Available  Source\n"
    "-----------------------------------------------\n"
    "Pinned   Pin.Ned     1.0      2.0        winget\n"
)
# The same table from a German Windows: titles and the summary are translated, the layout is not.
WINGET_TABLE_DE = (
    "Name                      ID                      Version        Verfügbar      Quelle\n"
    "-----------------------------------------------------------------------------------------\n"
    "Microsoft Edge            Microsoft.Edge          119.0.2151.58  119.0.2151.72  winget\n"
    "1 Aktualisierungen verfügbar.\n"
)


def test_winget_table_is_read_by_position_in_any_language() -> None:
    from pilot.system.package_mgr import parse_winget_upgrade_table

    assert parse_winget_upgrade_table(WINGET_TABLE) == ["Microsoft Edge", "7-Zip 23.01 (x64)"]
    assert parse_winget_upgrade_table(WINGET_TABLE_DE) == ["Microsoft Edge"]
    assert parse_winget_upgrade_table("No installed package found matching input criteria.\n") == []


def test_dnf_and_choco_lists_are_parsed() -> None:
    from pilot.system.package_mgr import parse_choco_outdated, parse_dnf_check_update

    dnf = (
        "\nkernel.x86_64        6.6.9-200.fc39       updates\n"
        "vim-enhanced.x86_64  2:9.1.016-1.fc39     updates\n"
        "Obsoleting Packages\n"
        "grub2-tools.x86_64   1:2.06-111.fc39      updates\n"
    )
    assert parse_dnf_check_update(dnf) == ["kernel.x86_64", "vim-enhanced.x86_64"]
    assert parse_choco_outdated("git|2.42.0|2.43.0|false\nnode|20.1|21.0|true\n") == ["git"]


def _package_platform(monkeypatch: pytest.MonkeyPatch, platform: Platform, manager: str, run: AsyncMock) -> None:
    from pilot.system import package_mgr

    monkeypatch.setattr(package_mgr, "CURRENT_PLATFORM", platform)
    monkeypatch.setattr(package_mgr, "run_command", run)
    monkeypatch.setattr(package_mgr, "_detect_linux_pkg_manager", AsyncMock(return_value=manager))
    monkeypatch.setattr(package_mgr, "_detect_win_pkg_manager", AsyncMock(return_value=manager))


@pytest.mark.parametrize(
    ("platform", "manager", "command", "root"),
    [
        (Platform.LINUX, "apt-get", ["sh", "-c"], True),
        (Platform.LINUX, "dnf", ["dnf", "upgrade", "-y", "--refresh"], True),
        (Platform.LINUX, "pacman", ["pacman", "-Syu", "--noconfirm"], True),
        (
            Platform.WINDOWS,
            "winget",
            ["winget", "upgrade", "--all", "--accept-package-agreements", "--accept-source-agreements"],
            False,
        ),
        (Platform.WINDOWS, "choco", ["choco", "upgrade", "all", "-y"], False),
    ],
)
async def test_package_update_upgrades_on_every_platform(
    monkeypatch: pytest.MonkeyPatch, platform, manager, command, root
) -> None:
    from pilot.system.package_mgr import APT_UPGRADE_SCRIPT, package_update

    run = AsyncMock(return_value=(0, "", ""))
    _package_platform(monkeypatch, platform, manager, run)

    assert "brought up to date" in await package_update()
    args = run.await_args.args[0]
    assert args[: len(command)] == command
    assert run.await_args.kwargs.get("root", False) is root
    if manager == "apt-get":
        # One pkexec prompt: the list refresh and the upgrade run in one fixed root shell command.
        assert args == ["sh", "-c", APT_UPGRADE_SCRIPT]
        assert "apt-get update && apt-get -y" in APT_UPGRADE_SCRIPT and "upgrade" in APT_UPGRADE_SCRIPT


async def test_macos_update_upgrades_and_a_failed_upgrade_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system.package_mgr import package_update

    run = AsyncMock(side_effect=[(0, "", ""), (0, "", ""), (0, "", ""), (1, "", "Error: openssl@3: failed")])
    _package_platform(monkeypatch, Platform.MACOS, "brew", run)

    assert "brew" in await package_update()
    assert [call.args[0] for call in run.await_args_list] == [["brew", "update"], ["brew", "upgrade"]]
    with pytest.raises(RuntimeError, match="Package update failed"):
        await package_update()


async def test_winget_with_nothing_to_upgrade_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system.package_mgr import package_update

    _package_platform(monkeypatch, Platform.WINDOWS, "winget", AsyncMock(return_value=(0x8A15002B, "No update", "")))

    assert "winget" in await package_update()


@pytest.mark.parametrize(
    ("platform", "manager", "answers", "outcome"),
    [
        (Platform.LINUX, "dnf", [(0, "", "")], "verified"),
        (Platform.LINUX, "dnf", [(100, "kernel.x86_64  6.6.9-200.fc39  updates\n", "")], "mismatch"),
        (Platform.LINUX, "apt-get", [(0, "Conf libc6 (2.39)\n", "")], "verified"),
        (
            Platform.LINUX,
            "apt-get",
            [(0, "Inst libc6 [2.38] (2.39 Ubuntu:24.04/noble-updates [amd64])\n", "")],
            "mismatch",
        ),
        (Platform.LINUX, "pacman", [(1, "", "")], "verified"),
        (Platform.LINUX, "pacman", [(0, "linux 6.6.1-1 -> 6.6.2-1\n", "")], "mismatch"),
        (Platform.LINUX, "zypper", [], "reported"),
        (Platform.MACOS, "brew", [(0, '{"formulae": [], "casks": []}', "")], "verified"),
        (
            Platform.MACOS,
            "brew",
            [(0, '{"formulae": [{"name": "node", "pinned": true}], "casks": []}', "")],
            "verified",
        ),
        (
            Platform.MACOS,
            "brew",
            [(0, '{"formulae": [{"name": "git", "pinned": false}], "casks": []}', "")],
            "mismatch",
        ),
        (
            Platform.WINDOWS,
            "winget",
            [(0x8A15002B, "No installed package found matching input criteria.\n", "")],
            "verified",
        ),
        (Platform.WINDOWS, "winget", [(0, WINGET_TABLE, "")], "mismatch"),
        (Platform.WINDOWS, "choco", [(0, "", "")], "verified"),
        (Platform.WINDOWS, "choco", [(0, "git|2.42.0|2.43.0|false\n", "")], "mismatch"),
    ],
)
async def test_package_update_verified_from_pending_upgrades(
    monkeypatch: pytest.MonkeyPatch, platform, manager, answers, outcome
) -> None:
    from pilot.actions import PackageParams

    run = AsyncMock(side_effect=answers)
    _package_platform(monkeypatch, platform, manager, run)

    verification = await _verify(ActionType.PACKAGE_UPDATE, PackageParams())

    {"verified": _assert_verified, "mismatch": _assert_mismatch, "reported": _assert_reported}[outcome](verification)
    for call in run.await_args_list:
        assert call.kwargs.get("root", False) is False  # the read never asks for a password


async def test_dnf_check_update_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.actions import PackageParams

    _package_platform(monkeypatch, Platform.LINUX, "dnf", AsyncMock(return_value=(1, "", "Cannot download repomd.xml")))

    verification = await _verify(ActionType.PACKAGE_UPDATE, PackageParams())

    assert verification.passed is False and "Verification error" in verification.details[0]


async def test_failed_package_update_never_reaches_the_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.actions import PackageParams
    from pilot.system import package_mgr

    _package_platform(
        monkeypatch, Platform.LINUX, "apt-get", AsyncMock(return_value=(100, "", "E: Could not get lock"))
    )
    observer = AsyncMock()
    monkeypatch.setattr(package_mgr, "pending_upgrades", observer)

    with pytest.raises(RuntimeError, match="Package update failed"):
        await package_mgr.package_update()
    _assert_failed(await _verify(ActionType.PACKAGE_UPDATE, PackageParams(), success=False))
    observer.assert_not_awaited()


def test_package_update_keeps_its_approval_tier() -> None:
    from pilot.actions import SYSTEM_MODIFY_ACTIONS

    # It now upgrades on every platform: never lower than the system-modify tier that asks first.
    assert ActionType.PACKAGE_UPDATE in SYSTEM_MODIFY_ACTIONS


# ── wifi_disconnect ─────────────────────────────────────────────────────────────────────────────────────────────


async def test_wifi_disconnect_verified_when_no_network_is_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    run = AsyncMock(side_effect=[(0, "yes:Home\n", ""), (0, "no:Home\n", "")])
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", run)

    _assert_verified(await _verify(ActionType.WIFI_DISCONNECT, _wifi(interface="wlan0")))


async def test_wifi_still_joined_after_disconnect_is_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(network, "_windows_connected_ssids", lambda: network.WifiReading(["Home"]))

    assert "still joined to Home" in _assert_mismatch(await _verify(ActionType.WIFI_DISCONNECT, _wifi()))


async def test_wifi_disconnect_unreadable_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(network, "_windows_connected_ssids", lambda: network.WifiReading(None, "no WLAN service"))

    _assert_reported(await _verify(ActionType.WIFI_DISCONNECT, _wifi()))


async def test_macos_disconnect_leaves_wifi_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turning Wi-Fi back on rejoins the preferred network at once, so the handler leaves it off."""
    from pilot.system import network

    run = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(network, "run_command", run)

    assert "turned off" in await network.wifi_disconnect()
    assert [call.args[0] for call in run.await_args_list] == [["networksetup", "-setairportpower", "en0", "off"]]


async def test_failed_disconnect_never_reaches_the_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", AsyncMock(return_value=(10, "", "Device 'wlan0' not found")))

    with pytest.raises(RuntimeError, match="WiFi disconnect failed"):
        await network.wifi_disconnect()
    _assert_failed(await _verify(ActionType.WIFI_DISCONNECT, _wifi(), success=False))


# ── env_set ─────────────────────────────────────────────────────────────────────────────────────────────────────


def _env(name: str = "HELIOX_TEST_VAR", value: str = "s3cret value", persistent: bool = True):
    from pilot.actions import EnvParams

    return EnvParams(name=name, value=value, persistent=persistent)


async def test_persistent_variable_read_from_the_shell_profile(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from pilot.system import environment

    profile = tmp_path / ".bashrc"
    profile.write_text("alias ll='ls -l'\n", encoding="utf-8")
    monkeypatch.setattr(environment, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(environment, "persistent_profile", lambda: str(profile))
    monkeypatch.delenv("HELIOX_TEST_VAR", raising=False)

    await environment.env_set("HELIOX_TEST_VAR", "s3cret value", persistent=True)
    _assert_verified(await _verify(ActionType.ENV_SET, _env()))

    # A later line in the profile overrides it: what a new shell sees differs.
    with profile.open("a", encoding="utf-8") as handle:
        handle.write("export HELIOX_TEST_VAR=other\n")
    detail = _assert_mismatch(await _verify(ActionType.ENV_SET, _env()))
    assert "s3cret" not in detail and "other" not in detail

    profile.write_text("", encoding="utf-8")
    assert "not there" in _assert_mismatch(await _verify(ActionType.ENV_SET, _env()))
    monkeypatch.delenv("HELIOX_TEST_VAR", raising=False)


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        ('{"present":true,"value":"s3cret value"}', "verified"),
        ('{"present":true,"value":"truncated"}', "mismatch"),
        ('{"present":false}', "mismatch"),
    ],
)
async def test_windows_user_variable_read_in_a_new_process(monkeypatch: pytest.MonkeyPatch, answer, outcome) -> None:
    from pilot.system import environment, platform_detect

    run = AsyncMock(return_value=(0, answer, ""))
    monkeypatch.setattr(environment, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(platform_detect, "run_powershell", run)

    verification = await _verify(ActionType.ENV_SET, _env())

    if outcome == "verified":
        _assert_verified(verification)
    else:
        assert "s3cret" not in _assert_mismatch(verification)
    assert run.await_args.kwargs["values"] == {"name": "HELIOX_TEST_VAR"}


async def test_session_only_variable_is_reported() -> None:
    _assert_reported(await _verify(ActionType.ENV_SET, _env(persistent=False)))


async def test_failed_setx_is_a_failed_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import environment

    monkeypatch.setattr(environment, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(environment, "run_command", AsyncMock(return_value=(1, "", "ERROR: Invalid syntax.")))
    monkeypatch.delenv("HELIOX_TEST_VAR", raising=False)

    with pytest.raises(RuntimeError, match="setx failed"):
        await environment.env_set("HELIOX_TEST_VAR", "x", persistent=True)
    monkeypatch.delenv("HELIOX_TEST_VAR", raising=False)
    _assert_failed(await _verify(ActionType.ENV_SET, _env(), success=False))


# ── volume_mute, volume_set, brightness_set ─────────────────────────────────────────────────────────────────────


def _volume(**kwargs):
    from pilot.actions import VolumeParams

    return VolumeParams(**kwargs)


@pytest.mark.parametrize(
    ("params", "state", "outcome"),
    [
        ({"mute": True}, (30, True), "verified"),
        ({}, (30, True), "verified"),
        ({"mute": False}, (30, False), "verified"),
        ({"mute": True}, (30, False), "mismatch"),
        ({"mute": True}, (None, None), "reported"),
    ],
)
async def test_volume_mute_read_from_the_mixer(monkeypatch: pytest.MonkeyPatch, params, state, outcome) -> None:
    from pilot.system import volume

    monkeypatch.setattr(volume, "volume_state", AsyncMock(return_value=volume.VolumeState(*state)))

    verification = await _verify(ActionType.VOLUME_MUTE, _volume(**params))

    {"verified": _assert_verified, "mismatch": _assert_mismatch, "reported": _assert_reported}[outcome](verification)


@pytest.mark.parametrize(
    ("params", "level", "outcome"),
    [
        ({"level": 30}, 30, "verified"),
        ({"level": 30}, 31, "verified"),
        ({"level": 30}, 45, "mismatch"),
        ({"level": 30}, None, "reported"),
        ({"step": 10}, 40, "reported"),
    ],
)
async def test_volume_level_read_from_the_mixer(monkeypatch: pytest.MonkeyPatch, params, level, outcome) -> None:
    from pilot.system import volume

    monkeypatch.setattr(volume, "volume_state", AsyncMock(return_value=volume.VolumeState(level, False)))

    verification = await _verify(ActionType.VOLUME_SET, _volume(**params))

    {"verified": _assert_verified, "mismatch": _assert_mismatch, "reported": _assert_reported}[outcome](verification)


async def test_failed_mute_is_a_failed_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import volume

    monkeypatch.setattr(volume, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(volume, "volume_state", AsyncMock(return_value=volume.VolumeState(30, False)))
    monkeypatch.setattr(volume.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="cannot be muted"):
        await volume.volume_mute(True)
    _assert_failed(await _verify(ActionType.VOLUME_MUTE, _volume(mute=True), success=False))


@pytest.mark.parametrize(
    ("platform", "answers", "outcome"),
    [
        (Platform.LINUX, [(0, "960\n", ""), (0, "1920\n", "")], "verified"),
        (Platform.LINUX, [(0, "192\n", ""), (0, "1920\n", "")], "mismatch"),
        (Platform.LINUX, [(-1, "", "Command not found: brightnessctl")], "reported"),
        (
            Platform.MACOS,
            [(0, "display 0: main, active, awake, online, built-in, ID 0x1\ndisplay 0: brightness 0.500000\n", "")],
            "verified",
        ),
        (Platform.MACOS, [(-1, "", "Command not found: brightness")], "reported"),
    ],
)
async def test_brightness_read_back(monkeypatch: pytest.MonkeyPatch, platform, answers, outcome) -> None:
    from pilot.actions import BrightnessParams
    from pilot.system import screen

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", platform)
    monkeypatch.setattr(screen, "run_command", AsyncMock(side_effect=answers))

    verification = await _verify(ActionType.BRIGHTNESS_SET, BrightnessParams(level=50))

    {"verified": _assert_verified, "mismatch": _assert_mismatch, "reported": _assert_reported}[outcome](verification)


@pytest.mark.parametrize(("answer", "outcome"), [((0, "52\n", ""), "verified"), ((0, "", ""), "reported")])
async def test_windows_brightness_read_from_wmi(monkeypatch: pytest.MonkeyPatch, answer, outcome) -> None:
    from pilot.actions import BrightnessParams
    from pilot.system import screen

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(screen, "run_powershell", AsyncMock(return_value=answer))

    verification = await _verify(ActionType.BRIGHTNESS_SET, BrightnessParams(level=50))

    {"verified": _assert_verified, "reported": _assert_reported}[outcome](verification)


async def test_failed_brightness_set_is_a_failed_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.actions import BrightnessParams
    from pilot.system import screen

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(screen, "run_command", AsyncMock(return_value=(1, "", "no backlight")))

    with pytest.raises(RuntimeError, match="Brightness set failed"):
        await screen.brightness_set(50)
    _assert_failed(await _verify(ActionType.BRIGHTNESS_SET, BrightnessParams(level=50), success=False))


def test_every_system_state_verifier_is_registered_under_its_catalog_name() -> None:
    assert {action: POSTCONDITION_VERIFIERS[action] for action in SYSTEM_STATE_VERIFIERS} == SYSTEM_STATE_VERIFIERS


SYSTEM_STATE_VERIFIERS = {
    ActionType.POWER_LOCK: "session_locked_postcondition",
    ActionType.FILE_PERMISSIONS: "conditional_file_mode_postcondition",
    ActionType.REGISTRY_WRITE: "registry_value_postcondition",
    ActionType.SERVICE_ENABLE: "service_enabled_postcondition",
    ActionType.SERVICE_DISABLE: "service_disabled_postcondition",
    ActionType.DISK_MOUNT: "mount_presence_postcondition",
    ActionType.WIFI_CONNECT: "wifi_network_postcondition",
    ActionType.PACKAGE_UPDATE: "packages_current_postcondition",
    ActionType.WIFI_DISCONNECT: "wifi_disconnected_postcondition",
    ActionType.ENV_SET: "conditional_environment_value_postcondition",
    ActionType.VOLUME_MUTE: "conditional_mute_state_postcondition",
    ActionType.VOLUME_SET: "conditional_volume_level_postcondition",
    ActionType.BRIGHTNESS_SET: "conditional_brightness_level_postcondition",
}


# ── package_update is irreversible ──────────────────────────────────────────────────────────────────────────────


def _package_update_action() -> Action:
    from pilot.actions import PackageParams

    return Action(action_type=ActionType.PACKAGE_UPDATE, target="all installed packages", parameters=PackageParams())


def test_package_update_is_irreversible_and_always_asks() -> None:
    from pilot.actions import IRREVERSIBLE_ACTIONS, PermissionTier
    from pilot.security.permissions import PermissionChecker

    action = _package_update_action()

    assert ActionType.PACKAGE_UPDATE in IRREVERSIBLE_ACTIONS
    assert action.is_irreversible is True
    assert action.permission_tier >= PermissionTier.SYSTEM_MODIFY
    assert ActionPlan(actions=[action]).needs_confirmation_unconditional is True
    decision = PermissionChecker(MagicMock()).check_action(action)
    assert decision.requires_confirmation is True and decision.irreversible is True


def test_a_spoken_yes_cannot_approve_a_package_update() -> None:
    """Said back as destructive before any dialog; the dialog still needs the click, and silence waits it out."""
    from pilot.conversation import say_back, turns

    action = _package_update_action()

    assert say_back.destructive([action]) is True
    assert turns.must_wait_out_window([action]) is True


def test_package_update_mismatch_goes_to_reconciliation_not_a_retry() -> None:
    from pilot.actions import VerificationResult
    from pilot.server import _postcondition_failure_requires_reconciliation

    results = [ActionResult(action=_package_update_action(), success=True, output="done")]
    verification = VerificationResult(passed=False, failed_actions=[0], details=["Action 0 (package_update): MISMATCH"])

    assert _postcondition_failure_requires_reconciliation(results, verification) is True


@pytest.mark.parametrize(
    ("platform", "words"),
    [(Platform.WINDOWS, "winget"), (Platform.MACOS, "Homebrew"), (Platform.LINUX, "apt, dnf or pacman")],
)
def test_approval_says_every_package_is_upgraded_on_each_platform(monkeypatch: pytest.MonkeyPatch, platform, words):
    from pilot.server import _approval_effect
    from pilot.system import package_mgr
    from pilot.workflows.undo_journal import undo_support

    monkeypatch.setattr(package_mgr, "CURRENT_PLATFORM", platform)
    effect = _approval_effect(_package_update_action())

    assert effect.startswith("Upgrade every installed package") and words in effect
    assert "cannot put the earlier versions back" in effect
    support = undo_support(_package_update_action())
    assert support.kind == "not_undoable" and "earlier versions" in support.reason
    assert _approval_effect(Action(action_type=ActionType.VOLUME_MUTE, target="x", parameters=_volume())) == ""


# ── power_lock: a second signal when LockedHint says no ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("screensaver", "outcome", "words"),
    [
        ("(true,)\n", "reported", ""),
        ("(false,)\n", "mismatch", "also reports itself inactive"),
        (None, "mismatch", "no desktop screensaver interface answered"),
    ],
)
async def test_linux_lock_hint_no_is_checked_against_the_desktop_screensaver(
    monkeypatch: pytest.MonkeyPatch, screensaver, outcome, words
) -> None:
    from pilot.system import power

    gdbus_calls: list[list[str]] = []

    async def fake_run(args, **kwargs):
        if args[0] == "loginctl":
            return (0, "no\n", "")
        gdbus_calls.append(args)
        assert kwargs["timeout"] == 5
        return (0, screensaver, "") if screensaver else (1, "", "The name is not activatable")

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(system_state, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(power, "run_command", fake_run)

    verification = await _verify(ActionType.POWER_LOCK, PowerParams())

    if outcome == "reported":
        _assert_reported(verification)
        assert "Could not verify the lock" in await _detail(ActionType.POWER_LOCK, PowerParams())
    else:
        detail = _assert_mismatch(verification)
        assert "does not report itself as locked" in detail and words in detail
    # Only GetActive is ever called: a read.
    assert all(call[-1].endswith(".GetActive") for call in gdbus_calls)


# ── wifi_disconnect on Linux finds the Wi-Fi device ─────────────────────────────────────────────────────────────


async def test_linux_disconnect_finds_a_newer_interface_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    run = AsyncMock(
        side_effect=[
            (
                0,
                "enp3s0:ethernet:connected\nwlp2s0:wifi:connected\np2p-dev-wlp2s0:wifi-p2p:disconnected\nlo:loopback:connected (externally)\n",
                "",
            ),
            (0, "Device 'wlp2s0' successfully disconnected.\n", ""),
        ]
    )
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", run)

    assert await network.wifi_disconnect() == "Disconnected from WiFi"
    assert run.await_args_list[0].args[0] == ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "device"]
    assert run.await_args_list[0].kwargs["env"]["LC_ALL"] == "C"
    assert run.await_args_list[1].args[0] == ["nmcli", "device", "disconnect", "wlp2s0"]


async def test_linux_disconnect_without_a_wifi_device_fails_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", AsyncMock(return_value=(0, "enp3s0:ethernet:connected\n", "")))

    with pytest.raises(RuntimeError, match="no Wi-Fi device"):
        await network.wifi_disconnect()


async def test_linux_disconnect_when_not_connected_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    run = AsyncMock(return_value=(0, "wlp2s0:wifi:disconnected\n", ""))
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", run)

    assert await network.wifi_disconnect() == "WiFi was not connected"
    assert run.await_count == 1


async def test_linux_disconnect_of_a_named_interface_asks_nothing_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import network

    run = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", run)

    await network.wifi_disconnect("wlx00c0ca123456")
    run.assert_awaited_once_with(["nmcli", "device", "disconnect", "wlx00c0ca123456"])


# ── systemctl calls are bounded ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("call", "args", "timeout"),
    [
        ("is_active", ("backup.timer",), 15),
        ("service_status", ("backup.timer",), 15),
        ("service_start", ("backup.timer",), 120),
        ("service_enable", ("backup.timer",), 120),
    ],
)
async def test_every_systemctl_call_has_a_time_limit(monkeypatch: pytest.MonkeyPatch, call, args, timeout) -> None:
    from pilot.system import platform_detect, systemctl

    run = AsyncMock(return_value=(0, "active\n", ""))
    monkeypatch.setattr(platform_detect, "run_command", run)

    await getattr(systemctl, call)(*args)

    assert run.await_args.kwargs["timeout"] == timeout
    assert ("pkexec" in run.await_args.args[0]) is (timeout == 120)


async def test_a_hung_systemctl_read_is_reported_as_inactive_not_waited_for(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import platform_detect, systemctl

    monkeypatch.setattr(platform_detect, "run_command", AsyncMock(return_value=(-1, "", "Command timed out")))

    assert await systemctl.is_active("backup.timer") is False


# ── power_sleep says what was requested ─────────────────────────────────────────────────────────────────────────


async def test_sleep_result_says_requested_not_slept(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.actions import PowerParams
    from pilot.agents.postconditions._unobservable import UNOBSERVABLE_POSTCONDITION
    from pilot.system import power

    monkeypatch.setattr(power, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(power, "run_command", AsyncMock(return_value=(0, "", "")))

    said = await power.sleep()

    assert said.startswith("Sleep was requested") and "cannot be checked" in said
    assert ActionType.POWER_SLEEP in UNOBSERVABLE_POSTCONDITION
    assert ActionType.POWER_SLEEP not in POSTCONDITION_VERIFIERS
    _assert_reported(await _verify(ActionType.POWER_SLEEP, PowerParams()))
