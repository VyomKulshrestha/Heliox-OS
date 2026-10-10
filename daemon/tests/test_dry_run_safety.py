from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType, FileParams
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.server import PilotServer, _resolve_dry_run


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


@pytest.mark.asyncio
async def test_request_dry_run_simulates_when_global_dry_run_is_off(tmp_path):
    victim = tmp_path / "request-dry-run-keeps-me.txt"
    victim.write_text("safe", encoding="utf-8")
    config = PilotConfig()
    config.security.dry_run = False
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )
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

    results = await executor.execute(plan, plan_id="request-dry-run-delete", dry_run=True)

    assert victim.read_text(encoding="utf-8") == "safe"
    executor._snapshot_mgr.create_snapshot.assert_not_awaited()
    assert results[0].output.startswith("(dry run) Would")


@pytest.mark.asyncio
async def test_execute_rpc_dry_run_skips_specialists_and_changes_nothing(tmp_path):
    # Dry runs skip the critic and the approval card, so execution must be simulated.
    config = PilotConfig()
    config.security.dry_run = False
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "rpc-audit.jsonl"),
    )
    target = tmp_path / "rpc-dry-run-must-not-exist.txt"
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_WRITE,
                target=str(target),
                parameters=FileParams(path=str(target), content="written only for real runs"),
            )
        ],
        explanation="write one file",
        raw_input="write the test file",
    )
    server = PilotServer(config)
    server._executor = executor
    server._planner = AsyncMock()
    server._planner.plan.return_value = plan
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._orchestrator = MagicMock()
    server._orchestrator.get_routing_summary.return_value = {"assigned_agents": []}
    server._orchestrator.execute_plan = AsyncMock(return_value=[])
    server._permission_checker = MagicMock()
    server._permission_checker.plan_requires_confirmation.return_value = True
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()

    result = await server._handle_execute(
        {"input": "write the test file", "session_id": "dry-run-rpc", "dry_run": True},
        ws=AsyncMock(),
    )

    assert result["status"] == "success"
    assert result["dry_run"] is True
    assert not target.exists()
    server._orchestrator.execute_plan.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_rpc_dry_run_reports_an_action_that_would_fail(tmp_path):
    config = PilotConfig()
    config.security.dry_run = False
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "invalid-audit.jsonl"),
    )
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_WRITE,
                target="",
                parameters=FileParams(path="", content="never written"),
            )
        ],
        explanation="write a file without saying where",
        raw_input="write a file",
    )
    server = PilotServer(config)
    server._executor = executor
    server._planner = AsyncMock()
    server._planner.plan.return_value = plan
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = MagicMock()
    server._permission_checker.plan_requires_confirmation.return_value = False
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()

    result = await server._handle_execute(
        {"input": "write a file", "session_id": "dry-run-invalid", "dry_run": True},
        ws=AsyncMock(),
    )

    assert result["status"] != "success"


def test_global_dry_run_cannot_be_disabled_by_request():
    assert _resolve_dry_run(True, False) is True
    assert _resolve_dry_run(True, None) is True


def test_request_can_enable_dry_run_without_changing_global_default():
    assert _resolve_dry_run(False, True) is True
    assert _resolve_dry_run(False, False) is False
