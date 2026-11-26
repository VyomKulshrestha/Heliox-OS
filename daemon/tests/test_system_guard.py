"""The test suite cannot change the computer it runs on (pilot.system.system_guard).

Each test here provokes a refusal on purpose and then clears the record, so the suite's own guard does not fail it.
Nothing is run: every refusal happens before a process starts or a connection opens.
"""

from __future__ import annotations

import pytest

from pilot.system import system_guard
from pilot.system.system_guard import SystemChangeRefused


@pytest.fixture
def expected_refusals():
    yield
    system_guard.clear()


def test_the_suite_is_guarded() -> None:
    assert system_guard.guarded() is True


@pytest.mark.parametrize(
    ("args", "refused"),
    [
        (["powershell", "-NoProfile", "-Command", "Checkpoint-Computer -Description 'x'"], True),
        (["powershell", "-Command", "(Get-Command Checkpoint-Computer -ErrorAction SilentlyContinue).Name"], False),
        (["powershell", "-Command", "Set-ItemProperty -LiteralPath x -Name y -Value 1"], True),
        (["powershell", "-Command", "Get-ItemProperty -LiteralPath x"], False),
        (["shutdown", "/s", "/t", "0"], True),
        (["timeshift", "--create"], True),
        (["timeshift", "--list"], False),
        (["btrfs", "subvolume", "snapshot", "/", "/.snapshots/x"], True),
        (["tmutil", "localsnapshot"], True),
        (["pkexec", "apt-get", "install", "-y", "cowsay"], True),
        (["apt", "list", "--upgradable"], False),
        (["winget", "upgrade", "--all"], True),
        (["systemctl", "--user", "enable", "x"], True),
        (["systemctl", "is-active", "x"], False),
        (["schtasks", "/create", "/tn", "x"], True),
        (["schtasks", "/query", "/tn", "x"], False),
        (["crontab", "-l"], False),
        (["crontab", "/tmp/new.cron"], True),
        (["reg", "add", "HKCU\\Software\\x"], True),
        (["reg", "query", "HKCU\\Software\\x"], False),
        (["nmcli", "device", "wifi", "connect", "home"], True),
        (["netsh", "wlan", "show", "interfaces"], False),
        (["mount"], False),
        (["mount", "/dev/sdb1", "/mnt/x"], True),
        (["loginctl", "lock-session"], True),
        (["rundll32.exe", "user32.dll,LockWorkStation"], True),
        (["git", "status"], False),
        (["C:\\Windows\\System32\\shutdown.exe", "/r"], True),
    ],
)
def test_what_counts_as_a_system_change(args: list[str], refused: bool) -> None:
    assert (system_guard.system_change(args) is not None) is refused


@pytest.mark.asyncio
async def test_a_command_runner_refuses_before_any_process_starts(expected_refusals) -> None:
    from pilot.system.platform_detect import run_command, run_powershell

    with pytest.raises(SystemChangeRefused, match="shutdown"):
        await run_command(["shutdown", "/s", "/t", "0"])
    with pytest.raises(SystemChangeRefused, match="Checkpoint-Computer"):
        await run_powershell("Checkpoint-Computer -Description 'test'")
    assert [kind for kind, _ in system_guard.attempts()][-2:] == ["command", "command"]


@pytest.mark.asyncio
async def test_a_real_snapshot_manager_cannot_take_a_restore_point_or_snapshot(expected_refusals) -> None:
    from pilot.config import PilotConfig
    from pilot.system.snapshots import SnapshotManager

    manager = SnapshotManager(PilotConfig())
    with pytest.raises(SystemChangeRefused, match="Checkpoint-Computer"):
        await manager._windows_restore_snapshot("pilot-test", "guard")
    with pytest.raises(SystemChangeRefused, match="timeshift"):
        await manager._timeshift_snapshot("pilot-test", "guard")
    with pytest.raises(SystemChangeRefused, match="btrfs"):
        await manager._btrfs_snapshot("pilot-test", "guard")


def test_executors_built_in_a_test_get_the_fake_system_snapshot(tmp_path) -> None:
    from pilot.agents.executor import Executor
    from pilot.config import PilotConfig
    from pilot.security.audit import AuditLogger
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator

    config = PilotConfig()
    executor = Executor(
        config, ActionValidator(config), PermissionChecker(config), AuditLogger(audit_file=tmp_path / "audit.jsonl")
    )

    assert type(executor._snapshot_mgr).__name__ == "GuardedSnapshotManager"


@pytest.mark.asyncio
async def test_services_mail_and_webhooks_are_refused(monkeypatch: pytest.MonkeyPatch, expected_refusals) -> None:
    from pilot.system import api_client, systemctl

    with pytest.raises(SystemChangeRefused, match="systemctl start"):
        await systemctl.service_start("heliox-test.service", user_scope=True)

    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASS", "app-password")
    with pytest.raises(SystemChangeRefused, match="SMTP connection"):
        await api_client.send_email("friend@example.com", "Hi", "Body")

    with pytest.raises(SystemChangeRefused, match="POST to hooks.example"):
        await api_client.api_request("POST", "https://hooks.example/services/x", body={"text": "hi"})


def test_reads_loopback_and_fakes_are_let_through() -> None:
    before = len(system_guard.attempts())

    system_guard.check_http("GET", "api.example", real_client=True)
    system_guard.check_http("POST", "127.0.0.1", real_client=True)
    system_guard.check_http("POST", "hooks.example", real_client=False)
    system_guard.check_mail("smtp.example", object())

    assert len(system_guard.attempts()) == before


def test_the_real_system_needs_ci_to_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(system_guard.ALLOW_VARIABLE, "1")
    monkeypatch.delenv("CI", raising=False)
    assert system_guard.guarded() is True
    monkeypatch.setenv("CI", "true")
    assert system_guard.guarded() is False


SET_VALUE = "Set-ItemProperty -LiteralPath x -Name y -Value 1"


def test_a_marked_test_may_make_only_the_changes_it_names_and_only_on_ci(
    monkeypatch: pytest.MonkeyPatch, expected_refusals
) -> None:
    monkeypatch.delenv(system_guard.ALLOW_VARIABLE, raising=False)
    monkeypatch.setenv("CI", "true")
    with system_guard.real_system_allowed("Set-ItemProperty"):
        system_guard.check_powershell(SET_VALUE)  # the named change passes
        with pytest.raises(SystemChangeRefused):
            system_guard.check_powershell("Remove-ItemProperty -LiteralPath x -Name y")
    assert ("PowerShell", "Set-ItemProperty") in system_guard.allowed_uses()
    with pytest.raises(SystemChangeRefused):
        system_guard.check_powershell(SET_VALUE)  # only while the marked test runs

    monkeypatch.delenv("CI")
    with system_guard.real_system_allowed("Set-ItemProperty"), pytest.raises(SystemChangeRefused):
        system_guard.check_powershell(SET_VALUE)  # never off CI
