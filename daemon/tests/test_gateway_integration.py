"""Integration tests for AgentGateway wired into Executor.execute().

Verifies the gateway is actually consulted (not just unit-testable in
isolation) and that a denial surfaces the same way an existing
PermissionChecker denial does — a single failed ActionResult, not a crash.
"""

from unittest.mock import patch

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams, FileParams
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.gateway import (
    AgentGateway,
    InvocationSource,
    TaskScopeOverride,
    mark_critic_already_reviewed,
)
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator


def _executor(tmp_path, gateway: AgentGateway | None) -> Executor:
    config = PilotConfig()
    validator = ActionValidator(config)
    permissions = PermissionChecker(config)
    audit = AuditLogger(audit_file=tmp_path / "audit.jsonl")
    return Executor(config, validator, permissions, audit, gateway=gateway)


def _plan(action_type: ActionType) -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=action_type, target="https://example.com", parameters=EmptyParams())],
        raw_input="test",
    )


@pytest.fixture
def gateway():
    config = PilotConfig()
    permissions = PermissionChecker(config)
    return AgentGateway(config, permissions)


class TestExecutorGatewayIntegration:
    @pytest.mark.asyncio
    async def test_no_gateway_attached_behaves_as_before(self, tmp_path):
        ex = _executor(tmp_path, gateway=None)
        results = await ex.execute(_plan(ActionType.BROWSER_EXECUTE_JS), invocation_source=InvocationSource.AUTONOMOUS)
        # With no gateway wired in, nothing new denies this — same as pre-gateway behavior.
        assert not any("Gateway denied" in (r.error or "") for r in results)

    @pytest.mark.asyncio
    async def test_autonomous_source_denied_before_dispatch(self, tmp_path, gateway):
        ex = _executor(tmp_path, gateway=gateway)
        results = await ex.execute(_plan(ActionType.BROWSER_EXECUTE_JS), invocation_source=InvocationSource.AUTONOMOUS)
        assert len(results) == 1
        assert results[0].success is False
        assert "Gateway denied" in results[0].error
        assert "browser_execute_js" in results[0].error

    @pytest.mark.asyncio
    async def test_interactive_source_not_denied_by_gateway(self, tmp_path, gateway):
        ex = _executor(tmp_path, gateway=gateway)
        results = await ex.execute(_plan(ActionType.BROWSER_EXECUTE_JS), invocation_source=InvocationSource.INTERACTIVE)
        assert not any("Gateway denied" in (r.error or "") for r in results)

    @pytest.mark.asyncio
    async def test_scope_override_cannot_widen_autonomous_floor_through_executor(self, tmp_path, gateway):
        ex = _executor(tmp_path, gateway=gateway)
        widen = TaskScopeOverride(max_tier={"browsing": 4}, deny_action_types=[], allow_root=True)
        results = await ex.execute(
            _plan(ActionType.BROWSER_EXECUTE_JS),
            invocation_source=InvocationSource.AUTONOMOUS,
            scope_override=widen,
        )
        # browser_execute_js is deny-listed for "autonomous" regardless of tier —
        # an override cannot remove it from the deny list.
        assert results[0].success is False
        assert "Gateway denied" in results[0].error

    @pytest.mark.asyncio
    async def test_default_invocation_source_is_interactive(self, tmp_path, gateway):
        # Callers that haven't been updated to pass invocation_source (the
        # ~20 untagged sub-agent call sites) get the unrestricted floor by
        # default — verifies the fail-open behavior described in gateway.py.
        ex = _executor(tmp_path, gateway=gateway)
        results = await ex.execute(_plan(ActionType.BROWSER_EXECUTE_JS))
        assert not any("Gateway denied" in (r.error or "") for r in results)


class TestRiskGateOutage:
    @pytest.mark.asyncio
    async def test_executor_does_not_dispatch_autonomous_write_during_gate_outage(
        self, tmp_path, gateway,
    ):
        target = tmp_path / "must-not-write.txt"
        plan = ActionPlan(actions=[Action(
            action_type=ActionType.FILE_WRITE, target=str(target),
            parameters=FileParams(content="test"),
        )], raw_input="write a test file")
        with patch("pilot.security.risk_gate.get_risk_gate", side_effect=RuntimeError("boom")):
            results = await _executor(tmp_path, gateway).execute(
                plan, invocation_source=InvocationSource.AUTONOMOUS,
            )
        assert len(results) == 1 and not results[0].success
        assert "Gateway denied" in (results[0].error or "")
        assert not target.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("already_reviewed", [False, True])
    async def test_autonomous_write_is_blocked_when_enabled_gate_fails(
        self, gateway, already_reviewed,
    ):
        plan = _plan(ActionType.FILE_WRITE)
        with patch("pilot.security.risk_gate.get_risk_gate", side_effect=RuntimeError("boom")):
            decision = await gateway.authorize(
                plan, InvocationSource.AUTONOMOUS, critic_already_reviewed=already_reviewed,
            )
        assert not decision.allowed
        assert "unavailable" in decision.reasons[0]
        assert decision.critic_verdict["world_model"]["model_version"] == "evaluation-error"

    @pytest.mark.asyncio
    async def test_read_only_plan_is_not_blocked_by_gate_outage(self, gateway):
        with patch("pilot.security.risk_gate.get_risk_gate", side_effect=RuntimeError("boom")):
            decision = await gateway.authorize(
                _plan(ActionType.SYSTEM_INFO), InvocationSource.AUTONOMOUS,
            )
        assert decision.allowed

    @pytest.mark.asyncio
    async def test_interactive_write_still_reaches_visible_approval_path(self, gateway):
        with patch("pilot.security.risk_gate.get_risk_gate", side_effect=RuntimeError("boom")):
            decision = await gateway.authorize(
                _plan(ActionType.FILE_WRITE), InvocationSource.INTERACTIVE,
            )
        assert decision.allowed
        assert decision.critic_verdict["world_model"]["requires_confirmation"]


class TestCriticBypassFix:
    """Verifies the specific gap that motivated this feature: previously,
    DestructiveCriticAgent only ran inside server.py's interactive `execute`
    handler, so a plan reaching Executor.execute() any other way (e.g. an
    autonomous job) never got a critic review at all. The gateway must now
    invoke an equivalent review for such plans."""

    @pytest.mark.asyncio
    async def test_critic_invoked_for_autonomous_plan_that_trips_heuristic_risk(self, tmp_path):
        config = PilotConfig()
        permissions = PermissionChecker(config)

        class _StubVerdict:
            def to_dict(self):
                return {"verdict": "WARN", "risk_score": 0.5, "recommendation": "reviewed"}

        class _StubCritic:
            def __init__(self):
                self.calls = 0

            async def review(self, user_input, plan):
                self.calls += 1
                return _StubVerdict()

        critic = _StubCritic()
        gateway = AgentGateway(config, permissions, destructive_critic=critic)
        ex = _executor(tmp_path, gateway=gateway)

        # SSH_COMMAND is SYSTEM_MODIFY tier (matches the autonomous profile's
        # shell-family ceiling exactly, so it isn't denied outright) but is
        # irreversible, which alone gives heuristic_risk == 0.3 --
        # HEURISTIC_RISK_THRESHOLD exactly, satisfying the ">=" trigger.
        plan = ActionPlan(
            actions=[Action(action_type=ActionType.SSH_COMMAND, target="host1", parameters=EmptyParams())],
            raw_input="run a command over ssh",
        )

        await ex.execute(plan, invocation_source=InvocationSource.AUTONOMOUS)
        assert critic.calls == 1

    @pytest.mark.asyncio
    async def test_critic_already_reviewed_flag_skips_redundant_call(self, tmp_path):
        config = PilotConfig()
        permissions = PermissionChecker(config)

        class _StubCritic:
            def __init__(self):
                self.calls = 0

            async def review(self, user_input, plan):
                self.calls += 1
                raise AssertionError("critic should not be called when critic_already_reviewed=True")

        critic = _StubCritic()
        gateway = AgentGateway(config, permissions, destructive_critic=critic)
        ex = _executor(tmp_path, gateway=gateway)

        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.FILE_DELETE,
                    target="/tmp/x",
                    parameters=EmptyParams(),
                    dangerous_flags=["rm -rf pattern detected"],
                )
            ],
            raw_input="delete a file",
        )

        await ex.execute(plan, invocation_source=InvocationSource.INTERACTIVE, critic_already_reviewed=True)
        assert critic.calls == 0

    @pytest.mark.asyncio
    async def test_outer_review_context_skips_nested_orchestrator_critic(self, tmp_path):
        config = PilotConfig()
        permissions = PermissionChecker(config)

        class _StubCritic:
            def __init__(self):
                self.calls = 0

            async def review(self, user_input, plan):
                self.calls += 1
                raise AssertionError("nested specialist execution repeated the critic review")

        critic = _StubCritic()
        gateway = AgentGateway(config, permissions, destructive_critic=critic)
        ex = _executor(tmp_path, gateway=gateway)
        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.FILE_DELETE,
                    target="/tmp/x",
                    parameters=EmptyParams(),
                    dangerous_flags=["rm -rf pattern detected"],
                )
            ],
            raw_input="delete a file",
        )

        with mark_critic_already_reviewed():
            await ex.execute(plan, invocation_source=InvocationSource.SYSTEM_AGENT)

        assert critic.calls == 0

    @pytest.mark.asyncio
    async def test_world_model_warns_without_destructive_critic_for_low_tier_plan(self):
        config = PilotConfig()
        config.restrictions.protected_folders = ["/protected"]
        permissions = PermissionChecker(config)

        class _StubVerdict:
            def to_dict(self):
                return {"verdict": "WARN", "risk_score": 0.9, "recommendation": "reviewed"}

        class _StubCritic:
            def __init__(self):
                self.calls = 0

            async def review(self, user_input, plan):
                self.calls += 1
                return _StubVerdict()

        critic = _StubCritic()
        gateway = AgentGateway(config, permissions, destructive_critic=critic)
        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.FILE_READ,
                    target="/protected/config.txt",
                    parameters=EmptyParams(),
                )
            ],
            raw_input="read protected config",
        )

        verdict = await gateway._maybe_run_critic(plan, critic_already_reviewed=False)

        assert critic.calls == 0
        assert verdict["verdict"] == "WARN"
        assert verdict["critic_skipped"] == "no_destructive_authority"
        assert verdict["verdict"] == "WARN"
