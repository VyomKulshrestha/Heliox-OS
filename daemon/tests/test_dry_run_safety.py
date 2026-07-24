from unittest.mock import AsyncMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType, FileParams
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.server import _resolve_dry_run


def _dry_run_executor(tmp_path) -> Executor:
    config = PilotConfig()
    config.security.dry_run = True
    return Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )


@pytest.mark.asyncio
async def test_dry_run_never_deletes_or_snapshots(tmp_path):
    victim = tmp_path / "dry-run-keeps-me.txt"
    victim.write_text("safe", encoding="utf-8")
    executor = _dry_run_executor(tmp_path)
    executor._snapshot_mgr.create_snapshot = AsyncMock()
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_DELETE,
                target=str(victim),
                parameters=FileParams(path=str(victim)),
                destructive=True,
            )
        ],
        explanation="simulate deleting a file",
    )

    results = await executor.execute(plan, plan_id="dry-run-delete")

    assert victim.read_text(encoding="utf-8") == "safe"
    executor._snapshot_mgr.create_snapshot.assert_not_awaited()
    assert results[0].success is True
    assert results[0].output.startswith("(dry run) Would")


def test_global_dry_run_cannot_be_disabled_by_request():
    assert _resolve_dry_run(True, False) is True
    assert _resolve_dry_run(True, None) is True


def test_request_can_enable_dry_run_without_changing_global_default():
    assert _resolve_dry_run(False, True) is True
    assert _resolve_dry_run(False, False) is False
