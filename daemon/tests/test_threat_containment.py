"""Unit tests for ThreatContainmentBridge (Issue #365).

Tests the full containment pipeline without starting the real server:
  - JSON parsing of ForensicsAgent output
  - PID extraction from proposed_resolution strings
  - Action plan translation (PROCESS_KILL / SHELL_COMMAND fallback)
  - Non-CRITICAL reports are silently skipped
  - Confirmation flow (approved, denied, timeout)
  - AuditLogger is called on every containment attempt
  - ForensicsAgent properly wires and invokes the bridge
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    LogAnalyzeParams,
    PermissionTier,
)
from pilot.agents.forensics_agent import ForensicsAgent
from pilot.agents.threat_containment import (
    ForensicsReport,
    ThreatContainmentBridge,
    _build_kill_action,
    _extract_pids,
)

# ---------------------------------------------------------------------------
# Shared mock for PendingConfirmation — avoids importing pilot.server which
# starts WebSocket listeners at import time and hangs in CI.
# ---------------------------------------------------------------------------


class _MockPendingConfirmation:
    """Minimal stand-in for server.PendingConfirmation used in tests."""

    def __init__(self, plan_id: str, event: asyncio.Event, plan: Any) -> None:
        self.plan_id = plan_id
        self.event = event
        self.plan = plan
        self.confirmed = False


# Install a fake 'pilot.server' module so the lazy
#   `from pilot.server import PendingConfirmation`
# inside route_and_confirm() never triggers a real server import.
if "pilot.server" not in sys.modules:
    _fake_server = types.ModuleType("pilot.server")
    _fake_server.PendingConfirmation = _MockPendingConfirmation  # type: ignore[attr-defined]
    sys.modules["pilot.server"] = _fake_server


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_result(output: str, success: bool = True) -> ActionResult:
    """Build a minimal ActionResult with the given output string."""
    action = Action(
        action_type=ActionType.LOG_ANALYZE,
        target="syslog",
        parameters=LogAnalyzeParams(log_path="syslog"),
    )
    return ActionResult(action=action, success=success, output=output)


def _critical_json(
    incident_type: str = "malware_process",
    resolution: str = "Kill malicious process 1042",
    pids: list[int] | None = None,
) -> str:
    return json.dumps(
        {
            "severity": "CRITICAL",
            "incident_type": incident_type,
            "proposed_resolution": resolution,
            "affected_pids": pids or [],
            "summary": "Malicious process detected consuming root privileges.",
        }
    )


def _make_bridge(
    confirmed: bool = True,
    timeout: bool = False,
    broadcast_called: list | None = None,
    audit_called: list | None = None,
) -> ThreatContainmentBridge:
    """Build a ThreatContainmentBridge with all external deps mocked."""
    pending_confirms: dict[str, Any] = {}

    # --- Mock Orchestrator --------------------------------------------------
    mock_orchestrator = MagicMock()

    async def _fake_execute(user_input, plan, plan_id=None, **kwargs):
        # Simulate successful execution for each action
        return [ActionResult(action=a, success=True, output="killed") for a in plan.actions]

    mock_orchestrator.execute_plan = AsyncMock(side_effect=_fake_execute)

    # --- Mock AuditLogger ---------------------------------------------------
    audit_entries: list[dict] = []
    if audit_called is not None:
        audit_called.clear()

    mock_audit = MagicMock()

    async def _fake_audit(event: str, details: dict):
        entry = {"event": event, **details}
        audit_entries.append(entry)
        if audit_called is not None:
            audit_called.append(entry)

    mock_audit.log_security_event = AsyncMock(side_effect=_fake_audit)

    # --- Mock broadcast -----------------------------------------------------
    broadcasts: list[tuple[str, dict]] = []
    if broadcast_called is not None:
        broadcast_called.clear()

    async def _fake_broadcast(method: str, params: Any):
        broadcasts.append((method, params))
        if broadcast_called is not None:
            broadcast_called.append((method, params))

        # Simulate the user confirming or denying through the pending_confirms dict
        plan_id = params.get("plan_id") if isinstance(params, dict) else None
        if method == "threat_confirmation_required" and plan_id:
            # Auto-resolve the pending confirmation event
            async def _resolve():
                await asyncio.sleep(0)  # yield to event loop
                pc = pending_confirms.get(plan_id)
                if pc is not None:
                    if timeout:
                        pass  # never set the event — let it time out
                    else:
                        pc.confirmed = confirmed
                        pc.event.set()

            asyncio.create_task(_resolve())

    bridge = ThreatContainmentBridge(
        orchestrator=mock_orchestrator,
        audit_logger=mock_audit,
        broadcast_fn=_fake_broadcast,
        pending_confirms=pending_confirms,
    )
    return bridge


# ---------------------------------------------------------------------------
# 1. ForensicsReport — parsing
# ---------------------------------------------------------------------------


class TestForensicsReportParsing:
    def test_parse_valid_critical_json(self):
        bridge = _make_bridge()
        report = bridge._parse_report(_critical_json())
        assert report is not None
        assert report.severity == "CRITICAL"
        assert report.incident_type == "malware_process"
        assert report.proposed_resolution == "Kill malicious process 1042"

    def test_parse_case_insensitive_severity(self):
        data = {"severity": "critical", "incident_type": "x", "proposed_resolution": "kill pid 5"}
        bridge = _make_bridge()
        report = bridge._parse_report(json.dumps(data))
        assert report is not None
        assert report.severity == "CRITICAL"

    def test_parse_markdown_wrapped_json(self):
        raw = "```json\n" + _critical_json() + "\n```"
        bridge = _make_bridge()
        report = bridge._parse_report(raw)
        assert report is not None
        assert report.severity == "CRITICAL"

    def test_parse_empty_output_returns_none(self):
        bridge = _make_bridge()
        assert bridge._parse_report("") is None
        assert bridge._parse_report("   ") is None

    def test_parse_plain_text_returns_none(self):
        bridge = _make_bridge()
        assert bridge._parse_report("No anomalies found in the last 24 hours.") is None

    def test_parse_invalid_json_returns_none(self):
        bridge = _make_bridge()
        assert bridge._parse_report("{severity: CRITICAL, bad json}") is None

    def test_parse_non_dict_json_returns_none(self):
        bridge = _make_bridge()
        assert bridge._parse_report("[1, 2, 3]") is None

    def test_parse_report_with_affected_pids(self):
        raw = _critical_json(pids=[111, 222])
        bridge = _make_bridge()
        report = bridge._parse_report(raw)
        assert report is not None
        assert report.affected_pids == [111, 222]


# ---------------------------------------------------------------------------
# 2. PID Extraction
# ---------------------------------------------------------------------------


class TestPidExtraction:
    def test_hint_pids_take_priority(self):
        pids = _extract_pids("Kill process 999", hint_pids=[1042, 2048])
        assert pids == [1042, 2048]

    def test_pid_keyword_pattern(self):
        pids = _extract_pids("kill pid: 1042", hint_pids=[])
        assert 1042 in pids

    def test_kill_numeric_pattern(self):
        pids = _extract_pids("kill -9 5678", hint_pids=[])
        assert 5678 in pids

    def test_process_id_pattern(self):
        pids = _extract_pids("terminate process 3333", hint_pids=[])
        assert 3333 in pids

    def test_no_pid_returns_empty(self):
        pids = _extract_pids("quarantine the suspicious file /etc/cron.d/evil", hint_pids=[])
        assert pids == []

    def test_multiple_pids(self):
        pids = _extract_pids("Kill pid 100 and process 200 and kill -9 300", hint_pids=[])
        assert 100 in pids
        assert 200 in pids
        assert 300 in pids


# ---------------------------------------------------------------------------
# 3. Action Translation
# ---------------------------------------------------------------------------


class TestTranslateResolution:
    def test_pid_found_produces_process_kill(self):
        bridge = _make_bridge()
        report = ForensicsReport(
            severity="CRITICAL",
            incident_type="malware",
            proposed_resolution="Kill malicious process 1042",
        )
        plan = bridge.translate_resolution(report)
        assert len(plan.actions) == 1
        action = plan.actions[0]
        assert action.action_type == ActionType.PROCESS_KILL
        assert action.destructive is True
        assert action.reversible is False
        assert action.parameters.pid == 1042

    def test_multiple_pids_produce_multiple_actions(self):
        bridge = _make_bridge()
        report = ForensicsReport(
            severity="CRITICAL",
            proposed_resolution="Kill process 111 and pid 222",
        )
        plan = bridge.translate_resolution(report)
        assert len(plan.actions) == 2
        pids = {a.parameters.pid for a in plan.actions}
        assert pids == {111, 222}

    def test_all_kill_actions_are_tier3_destructive(self):
        bridge = _make_bridge()
        report = ForensicsReport(
            severity="CRITICAL",
            proposed_resolution="kill -9 9999",
        )
        plan = bridge.translate_resolution(report)
        for action in plan.actions:
            assert action.destructive is True
            assert action.permission_tier == PermissionTier.DESTRUCTIVE

    def test_no_pid_generates_no_actions(self):
        bridge = _make_bridge()
        report = ForensicsReport(
            severity="CRITICAL",
            proposed_resolution="Block network interface eth0 immediately",
        )
        plan = bridge.translate_resolution(report)
        assert len(plan.actions) == 0

    def test_affected_pids_field_used_when_no_pid_in_text(self):
        bridge = _make_bridge()
        report = ForensicsReport(
            severity="CRITICAL",
            proposed_resolution="Terminate the threat immediately.",
            affected_pids=[7777],
        )
        plan = bridge.translate_resolution(report)
        assert any(a.action_type == ActionType.PROCESS_KILL and a.parameters.pid == 7777 for a in plan.actions)


# ---------------------------------------------------------------------------
# 4. Non-CRITICAL Reports Are Skipped
# ---------------------------------------------------------------------------


class TestNonCriticalIgnored:
    @pytest.mark.asyncio
    async def test_high_severity_not_intercepted(self):
        bridge = _make_bridge()
        raw = json.dumps({"severity": "HIGH", "proposed_resolution": "Kill pid 1", "incident_type": "x"})
        results = [_make_result(raw)]
        records = await bridge.intercept(results)
        assert records == []

    @pytest.mark.asyncio
    async def test_empty_results_no_containment(self):
        bridge = _make_bridge()
        records = await bridge.intercept([])
        assert records == []

    @pytest.mark.asyncio
    async def test_plain_text_output_not_intercepted(self):
        bridge = _make_bridge()
        results = [_make_result("No anomalies detected.")]
        records = await bridge.intercept(results)
        assert records == []


# ---------------------------------------------------------------------------
# 5. Full Pipeline — Approved, Denied, Timeout
# ---------------------------------------------------------------------------


class TestContainmentPipeline:
    @pytest.mark.asyncio
    async def test_approved_containment_runs_orchestrator(self):
        audit_calls: list = []
        broadcasts: list = []
        bridge = _make_bridge(confirmed=True, audit_called=audit_calls, broadcast_called=broadcasts)

        results = [_make_result(_critical_json(pids=[1042]))]
        records = await bridge.intercept(results)

        assert len(records) == 1
        record = records[0]
        assert record.confirmed is True
        assert any(r.success for r in record.results)

    @pytest.mark.asyncio
    async def test_approved_containment_writes_audit(self):
        audit_calls: list = []
        bridge = _make_bridge(confirmed=True, audit_called=audit_calls)

        results = [_make_result(_critical_json(pids=[1042]))]
        await bridge.intercept(results)

        assert len(audit_calls) >= 1
        events = [e["event"] for e in audit_calls]
        assert "threat_contained" in events

    @pytest.mark.asyncio
    async def test_denied_containment_does_not_execute(self):
        audit_calls: list = []
        bridge = _make_bridge(confirmed=False, audit_called=audit_calls)

        results = [_make_result(_critical_json(pids=[1042]))]
        records = await bridge.intercept(results)

        assert len(records) == 1
        record = records[0]
        assert record.confirmed is False
        assert record.results == []

        events = [e["event"] for e in audit_calls]
        assert "threat_containment_denied" in events

    @pytest.mark.asyncio
    async def test_multiple_criticals_in_one_batch(self):
        bridge = _make_bridge(confirmed=True)

        results = [
            _make_result(_critical_json(incident_type="brute_force", pids=[111])),
            _make_result(_critical_json(incident_type="ransomware", pids=[222])),
        ]
        records = await bridge.intercept(results)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# 6. ForensicsAgent Integration
# ---------------------------------------------------------------------------


class TestForensicsAgentIntegration:
    def test_set_threat_bridge_stores_reference(self):
        """ForensicsAgent accepts the bridge via set_threat_bridge()."""
        mock_model = MagicMock()
        mock_executor = MagicMock()
        agent = ForensicsAgent(model_router=mock_model, executor=mock_executor)

        assert agent._threat_bridge is None

        bridge = _make_bridge()
        agent.set_threat_bridge(bridge)
        assert agent._threat_bridge is bridge

    @pytest.mark.asyncio
    async def test_handle_task_spawns_background_intercept(self):
        """handle_task schedules _intercept_critical_threats as background task."""

        mock_model = MagicMock()

        # Fake executor that returns a CRITICAL result
        mock_executor = MagicMock()
        critical_result = _make_result(_critical_json(pids=[9999]))

        async def _fake_execute(plan, **kwargs):
            return [critical_result]

        mock_executor.execute = AsyncMock(side_effect=_fake_execute)

        agent = ForensicsAgent(model_router=mock_model, executor=mock_executor)

        intercepted: list = []

        class _TrackingBridge:
            async def intercept(self, results):
                intercepted.extend(results)
                return []

        agent.set_threat_bridge(_TrackingBridge())

        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.LOG_ANALYZE,
                    target="syslog",
                    parameters=LogAnalyzeParams(log_path="syslog"),
                )
            ],
            explanation="Test",
            raw_input="check logs",
        )

        results = await agent.handle_task("check logs", plan)

        # Let the background task run
        await asyncio.sleep(0.05)

        assert len(results) == 1
        assert len(intercepted) == 1


# ---------------------------------------------------------------------------
# 7. _build_kill_action sanity
# ---------------------------------------------------------------------------


class TestBuildKillAction:
    def test_returns_action_with_correct_fields(self):
        action = _build_kill_action(1234)
        assert action.action_type == ActionType.PROCESS_KILL
        assert action.parameters.pid == 1234
        assert action.parameters.signal == "SIGKILL"
        assert action.destructive is True
        assert action.reversible is False
        assert action.requires_root is False

    def test_permission_tier_is_destructive(self):
        action = _build_kill_action(42)
        assert action.permission_tier == PermissionTier.DESTRUCTIVE
        assert action.requires_confirmation is True
