from unittest.mock import AsyncMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType, FileParams
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.system.snapshots import SnapshotBackend, SnapshotManager


def _executor(tmp_path) -> Executor:
    config = PilotConfig()
    config.security.snapshot_on_destructive = True
    return Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )


def _delete_plan(path) -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_DELETE,
                target=str(path),
                parameters=FileParams(path=str(path)),
                destructive=True,
            )
        ],
        explanation="delete test file",
    )


@pytest.mark.asyncio
async def test_destructive_plan_fails_closed_without_snapshot_backend(tmp_path):
    victim = tmp_path / "keep-me.txt"
    victim.write_text("safe", encoding="utf-8")
    executor = _executor(tmp_path)
    executor._snapshot_mgr.create_snapshot = AsyncMock(return_value=None)

    results = await executor.execute(_delete_plan(victim), plan_id="snapshot-none")

    assert victim.exists()
    assert len(results) == 1
    assert results[0].success is False
    assert "No actions ran" in results[0].error


@pytest.mark.asyncio
async def test_destructive_plan_fails_closed_when_snapshot_errors(tmp_path):
    victim = tmp_path / "keep-me-too.txt"
    victim.write_text("safe", encoding="utf-8")
    executor = _executor(tmp_path)
    executor._snapshot_mgr.create_snapshot = AsyncMock(side_effect=RuntimeError("restore point denied"))

    results = await executor.execute(_delete_plan(victim), plan_id="snapshot-error")

    assert victim.exists()
    assert results[0].success is False
    assert "restore point denied" in results[0].error


@pytest.mark.asyncio
async def test_windows_snapshot_status_requires_administrator(monkeypatch):
    config = PilotConfig()
    manager = SnapshotManager(config)
    manager._backend = SnapshotBackend.WINDOWS_RESTORE_POINT
    monkeypatch.setattr("pilot.security.privileges.has_elevated_privileges", lambda: False)

    status = await manager.status()

    assert status["available"] is True
    assert status["ready"] is False
    assert "not Administrator" in status["detail"]
    assert status["retention_supported"] is False
    assert "Windows manages Restore Point retention" in status["retention_detail"]


@pytest.mark.asyncio
async def test_snapshot_creation_enforces_retention_for_supported_backend():
    config = PilotConfig()
    manager = SnapshotManager(config)
    manager._backend = SnapshotBackend.BTRFS
    manager._btrfs_snapshot = AsyncMock(return_value="/.snapshots/new")
    manager.cleanup = AsyncMock(return_value=2)

    snapshot_id = await manager.create_snapshot("plan-1", "before changes")

    assert snapshot_id == "/.snapshots/new"
    manager.cleanup.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cleanup_keeps_newest_pilot_snapshots(monkeypatch):
    config = PilotConfig()
    config.security.snapshot_retention_count = 2
    manager = SnapshotManager(config)
    manager._backend = SnapshotBackend.BTRFS
    manager.list_snapshots = AsyncMock(
        return_value=[
            {"id": "/.snapshots/old", "tag": "pilot-a-20260101-000000"},
            {"id": "/.snapshots/new", "tag": "pilot-b-20260301-000000"},
            {"id": "/.snapshots/middle", "tag": "pilot-c-20260201-000000"},
            {"id": "/.snapshots/user", "tag": "user-created"},
        ]
    )
    run = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr("pilot.system.snapshots._run", run)

    removed = await manager.cleanup()

    assert removed == 1
    run.assert_awaited_once_with(
        ["btrfs", "subvolume", "delete", "/.snapshots/old"],
        root=True,
    )


@pytest.mark.asyncio
async def test_windows_snapshot_label_is_identifiable_as_pilot_owned(monkeypatch):
    config = PilotConfig()
    manager = SnapshotManager(config)
    manager._backend = SnapshotBackend.WINDOWS_RESTORE_POINT
    run = AsyncMock(return_value=(0, "42\n", ""))
    monkeypatch.setattr("pilot.system.snapshots._run", run)

    snapshot_id = await manager.create_snapshot("plan-7", "before protected change")

    assert snapshot_id == "windows-restore:42"
    command = run.await_args.args[0][-1]
    assert "pilot-plan-7-" in command
    assert "before protected change" in command
