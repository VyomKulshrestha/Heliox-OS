"""Tests for installed-application resolution and truthful launch outcomes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType, OpenApplicationParams
from pilot.agents.executor import Executor
from pilot.agents.verifier import Verifier
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.system import applications, platform_detect


def _shortcut(root: Path, name: str) -> Path:
    shortcut = root / f"{name}.lnk"
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    shortcut.touch()
    return shortcut


def _isolate_discovery(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(applications, "_default_start_menu_roots", lambda: [root])
    monkeypatch.setattr(applications.shutil, "which", lambda _name: None)
    monkeypatch.setattr(applications, "_load_start_apps", AsyncMock(return_value=[]))


@pytest.mark.asyncio
async def test_resolves_case_insensitive_exact_start_menu_name(monkeypatch, tmp_path):
    shortcut = _shortcut(tmp_path, "Openscreen")
    executable = tmp_path / "Openscreen.exe"
    executable.touch()
    _isolate_discovery(monkeypatch, tmp_path)
    resolver = AsyncMock(return_value=str(executable))
    monkeypatch.setattr(applications, "_resolve_shortcut_target", resolver)

    target = await applications.resolve_windows_application("openscreen")

    assert target.display_name == "Openscreen"
    assert target.value == str(executable.resolve())
    assert target.source == "Start menu"
    resolver.assert_awaited_once_with(shortcut)


@pytest.mark.asyncio
async def test_exact_product_name_wins_over_shorter_related_name(monkeypatch, tmp_path):
    antigravity = _shortcut(tmp_path, "Antigravity")
    antigravity_ide = _shortcut(tmp_path, "Antigravity IDE")
    executable = tmp_path / "Antigravity IDE.exe"
    executable.touch()
    _isolate_discovery(monkeypatch, tmp_path)

    async def resolve(shortcut: Path) -> str:
        assert shortcut == antigravity_ide
        return str(executable)

    monkeypatch.setattr(applications, "_resolve_shortcut_target", resolve)

    target = await applications.resolve_windows_application("Antigravity IDE")

    assert target.display_name == "Antigravity IDE"
    assert target.value == str(executable.resolve())
    assert antigravity != antigravity_ide


@pytest.mark.asyncio
async def test_duplicate_shortcuts_to_same_executable_are_not_ambiguous(monkeypatch, tmp_path):
    first_root = tmp_path / "user"
    second_root = tmp_path / "common"
    _shortcut(first_root, "Example App")
    _shortcut(second_root, "Example App")
    executable = tmp_path / "Example.exe"
    executable.touch()
    monkeypatch.setattr(applications, "_default_start_menu_roots", lambda: [first_root, second_root])
    monkeypatch.setattr(applications.shutil, "which", lambda _name: None)
    monkeypatch.setattr(applications, "_load_start_apps", AsyncMock(return_value=[]))
    monkeypatch.setattr(applications, "_resolve_shortcut_target", AsyncMock(return_value=str(executable)))

    target = await applications.resolve_windows_application("Example App")

    assert target.value == str(executable.resolve())


@pytest.mark.asyncio
async def test_ambiguous_registered_app_ids_fail_closed(monkeypatch, tmp_path):
    _isolate_discovery(monkeypatch, tmp_path)
    monkeypatch.setattr(
        applications,
        "_load_start_apps",
        AsyncMock(
            return_value=[
                applications._NamedTarget("Antigravity", "vendor.first"),
                applications._NamedTarget("Antigravity", "vendor.second"),
            ]
        ),
    )

    with pytest.raises(applications.ApplicationResolutionError, match="ambiguous"):
        await applications.resolve_windows_application("Antigravity")


@pytest.mark.asyncio
async def test_unknown_application_fails_instead_of_claiming_launch(monkeypatch, tmp_path):
    _isolate_discovery(monkeypatch, tmp_path)

    with pytest.raises(applications.ApplicationResolutionError, match="was not found"):
        await applications.resolve_windows_application("Definitely Missing Product")


@pytest.mark.asyncio
async def test_launcher_reports_resolved_source(monkeypatch, tmp_path):
    executable = tmp_path / "Real App.exe"
    executable.touch()
    monkeypatch.setattr(
        applications,
        "resolve_windows_application",
        AsyncMock(return_value=applications.ApplicationTarget("Real App", "executable", str(executable), "Start menu")),
    )
    starter = AsyncMock(return_value=1234)
    monkeypatch.setattr(applications, "_start_and_check", starter)

    result = await applications.launch_windows_application("real app")

    assert result == "Launched Real App (resolved via Start menu)."
    starter.assert_awaited_once_with([str(executable)], "Real App")


@pytest.mark.asyncio
async def test_nonzero_startup_is_a_failed_launch(monkeypatch):
    process = AsyncMock()
    process.pid = 1234
    process.wait.return_value = 7
    process.stderr.read.return_value = b"startup failed"
    monkeypatch.setattr(applications.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    with pytest.raises(applications.ApplicationLaunchError, match="code 7: startup failed"):
        await applications._start_and_check(["broken.exe"], "Broken")


@pytest.mark.asyncio
async def test_macos_application_launch_uses_launch_services(monkeypatch):
    run_command = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", platform_detect.Platform.MACOS)
    monkeypatch.setattr(platform_detect, "run_command", run_command)
    monkeypatch.setattr(applications, "mac_application_bundles", lambda: [])

    result = await applications.launch_application("Hermes", ["--safe"])

    run_command.assert_awaited_once_with(["open", "-a", "Hermes", "--args", "--safe"])
    assert result == "Launched Hermes (resolved via macOS Launch Services)."


@pytest.mark.asyncio
async def test_linux_missing_application_fails_before_invoking_missing_launcher(monkeypatch):
    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", platform_detect.Platform.LINUX)
    monkeypatch.setattr(applications.shutil, "which", lambda _name: None)
    from pilot.system import app_names

    monkeypatch.setattr(app_names, "_desktop_entries", lambda: ())

    with pytest.raises(applications.ApplicationResolutionError, match="among the installed applications"):
        await applications.launch_application("Missing App")


@pytest.mark.asyncio
async def test_executor_and_verifier_expose_resolution_failure(monkeypatch, tmp_path):
    config = PilotConfig()
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )
    action = Action(
        action_type=ActionType.OPEN_APPLICATION,
        target="Missing App",
        parameters=OpenApplicationParams(name="Missing App"),
    )
    plan = ActionPlan(actions=[action], raw_input="open Missing App")
    monkeypatch.setattr(
        applications,
        "launch_application",
        AsyncMock(side_effect=applications.ApplicationResolutionError("not installed")),
    )

    result = await executor._execute_single(action, snapshot_id=None)
    verification = await Verifier(MagicMock()).verify(plan, [result])

    assert result.success is False
    assert result.error == "not installed"
    assert verification.passed is False
    assert verification.failed_actions == [0]
    assert "FAILED" in verification.details[0]
