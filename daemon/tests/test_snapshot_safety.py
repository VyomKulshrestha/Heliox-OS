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
