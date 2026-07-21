"""Integration tests for ExecutionNarrator wired into Executor.execute().

Uses a fake gateway (a directly-constructed, controllable GatewayDecision)
and a fake narrator that records every call, so these test the WIRING
itself -- whether the right hooks fire at the right times with the right
data -- not the narrator's or gateway's own internal logic (each already
has its own dedicated test file).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType, BrowserParams, EmptyParams
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.gateway import GatewayDecision, InvocationSource
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.system.dom_diff import TargetAssessment


class _FakeBrowserBackend:
    """Stands in for PlaywrightBackend so these wiring tests never launch a
    real browser -- they only need to confirm the interrupt hook fired and
    whether the action was skipped or reached dispatch, not exercise real
    browser automation (covered by system/browser.py's own tests)."""

    async def click(self, selector, button="left", timeout=5000):
        return f"Clicked: {selector}"


class _FakeGateway:
    def __init__(self, critic_verdict=None, allowed=True):
        self._critic_verdict = critic_verdict
        self._allowed = allowed

    async def authorize(
        self, plan, invocation_source, *, scope_override=None, critic_already_reviewed=False, plan_id=None
    ):
        return GatewayDecision(allowed=self._allowed, reasons=[], critic_verdict=self._critic_verdict)


class _FakeNarrator:
    def __init__(self, plan_risk_proceed: bool = True, target_assessment_proceed: bool = True):
        self.plan_risk_calls: list[dict] = []
        self.action_start_calls: list = []
        self.action_complete_calls: list = []
        self.target_assessment_calls: list = []
        self._plan_risk_proceed = plan_risk_proceed
        self._target_assessment_proceed = target_assessment_proceed

    async def on_plan_risk(self, plan, critic_verdict) -> bool:
        self.plan_risk_calls.append(critic_verdict)
        return self._plan_risk_proceed

    async def on_action_start(self, action) -> None:
        self.action_start_calls.append(action)

    async def on_action_complete(self, result) -> None:
        self.action_complete_calls.append(result)

    async def on_target_assessment(self, action, assessment) -> bool:
        self.target_assessment_calls.append((action, assessment))
        return self._target_assessment_proceed


def _executor(tmp_path, gateway=None) -> Executor:
    config = PilotConfig()
    validator = ActionValidator(config)
    permissions = PermissionChecker(config)
    audit = AuditLogger(audit_file=tmp_path / "audit.jsonl")
    return Executor(config, validator, permissions, audit, gateway=gateway)


def _plan(action_type: ActionType = ActionType.CPU_USAGE, target: str = "x") -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=action_type, target=target, parameters=EmptyParams())], raw_input="test"
    )


class TestPlanRiskWiring:
    @pytest.mark.asyncio
    async def test_warn_verdict_invokes_narrator_with_captured_verdict(self, tmp_path):
        gateway = _FakeGateway(critic_verdict={"verdict": "WARN", "recommendation": "risky"})
        narrator = _FakeNarrator(plan_risk_proceed=True)
        ex = _executor(tmp_path, gateway=gateway)
        ex.set_narrator(narrator)

        results = await ex.execute(_plan(), invocation_source=InvocationSource.AUTONOMOUS)

        assert len(narrator.plan_risk_calls) == 1
        assert narrator.plan_risk_calls[0]["verdict"] == "WARN"
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_narrator_denial_aborts_the_plan(self, tmp_path):
        gateway = _FakeGateway(critic_verdict={"verdict": "WARN", "recommendation": "risky"})
        narrator = _FakeNarrator(plan_risk_proceed=False)
        ex = _executor(tmp_path, gateway=gateway)
        ex.set_narrator(narrator)

        results = await ex.execute(_plan(), invocation_source=InvocationSource.AUTONOMOUS)

        assert len(results) == 1
        assert results[0].success is False
        assert "risk interrupt" in results[0].error

    @pytest.mark.asyncio
    async def test_no_critic_verdict_never_invokes_narrator(self, tmp_path):
        gateway = _FakeGateway(critic_verdict=None)
        narrator = _FakeNarrator()
        ex = _executor(tmp_path, gateway=gateway)
        ex.set_narrator(narrator)

        await ex.execute(_plan(), invocation_source=InvocationSource.AUTONOMOUS)

        assert narrator.plan_risk_calls == []

    @pytest.mark.asyncio
    async def test_no_narrator_configured_ignores_critic_verdict(self, tmp_path):
        gateway = _FakeGateway(critic_verdict={"verdict": "WARN"})
        ex = _executor(tmp_path, gateway=gateway)  # set_narrator never called

        results = await ex.execute(_plan(), invocation_source=InvocationSource.AUTONOMOUS)

        assert results[0].success is True


class TestActionNarrationWiring:
    @pytest.mark.asyncio
    async def test_narrator_hooks_fire_automatically_when_caller_passes_no_callbacks(self, tmp_path):
        narrator = _FakeNarrator()
        ex = _executor(tmp_path, gateway=None)
        ex.set_narrator(narrator)

        await ex.execute(_plan(), invocation_source=InvocationSource.AUTONOMOUS)

        assert len(narrator.action_start_calls) == 1
        assert len(narrator.action_complete_calls) == 1

    @pytest.mark.asyncio
    async def test_explicit_caller_callbacks_are_not_overridden_by_narrator(self, tmp_path):
        narrator = _FakeNarrator()
        caller_start_calls = []
        caller_complete_calls = []

        async def on_start(action):
            caller_start_calls.append(action)

        async def on_complete(result):
            caller_complete_calls.append(result)

        ex = _executor(tmp_path, gateway=None)
        ex.set_narrator(narrator)

        await ex.execute(
            _plan(),
            on_action_start=on_start,
            on_action_complete=on_complete,
            invocation_source=InvocationSource.AUTONOMOUS,
        )

        assert len(caller_start_calls) == 1
        assert len(caller_complete_calls) == 1
        assert narrator.action_start_calls == []
        assert narrator.action_complete_calls == []


class TestBrowserTargetAssessmentWiring:
    @pytest.mark.asyncio
    async def test_non_browser_action_never_triggers_target_assessment(self, tmp_path):
        narrator = _FakeNarrator()
        ex = _executor(tmp_path, gateway=None)
        ex.set_narrator(narrator)

        await ex.execute(_plan(ActionType.CPU_USAGE), invocation_source=InvocationSource.AUTONOMOUS)

        assert narrator.target_assessment_calls == []

    @pytest.mark.asyncio
    async def test_browser_action_with_no_active_session_skips_assessment(self, tmp_path):
        # has_active_session() is False by default -- no real browser opened in tests.
        narrator = _FakeNarrator()
        ex = _executor(tmp_path, gateway=None)
        ex.set_narrator(narrator)
        ex._browser_backend = _FakeBrowserBackend()
        plan = ActionPlan(
            actions=[
                Action(action_type=ActionType.BROWSER_CLICK, target="#btn", parameters=BrowserParams(selector="#btn"))
            ],
            raw_input="test",
        )

        await ex.execute(plan, invocation_source=InvocationSource.AUTONOMOUS)

        assert narrator.target_assessment_calls == []

    @pytest.mark.asyncio
    async def test_flagged_assessment_skips_action_when_narrator_denies(self, tmp_path, monkeypatch):
        fake_assessment = TargetAssessment(matchable=True, found=False, reason="selector '#btn' not found")
        monkeypatch.setattr(
            "pilot.system.dom_diff.assess_browser_action_target",
            AsyncMock(return_value=fake_assessment),
        )
        narrator = _FakeNarrator(target_assessment_proceed=False)
        ex = _executor(tmp_path, gateway=None)
        ex.set_narrator(narrator)
        plan = ActionPlan(
            actions=[
                Action(action_type=ActionType.BROWSER_CLICK, target="#btn", parameters=BrowserParams(selector="#btn"))
            ],
            raw_input="test",
        )

        results = await ex.execute(plan, invocation_source=InvocationSource.AUTONOMOUS)

        assert len(narrator.target_assessment_calls) == 1
        assert results[0].success is False
        assert "Skipped after interrupt" in results[0].error

    @pytest.mark.asyncio
    async def test_flagged_assessment_proceeds_when_narrator_allows(self, tmp_path, monkeypatch):
        fake_assessment = TargetAssessment(matchable=True, found=False, reason="selector '#btn' not found")
        monkeypatch.setattr(
            "pilot.system.dom_diff.assess_browser_action_target",
            AsyncMock(return_value=fake_assessment),
        )
        narrator = _FakeNarrator(target_assessment_proceed=True)
        ex = _executor(tmp_path, gateway=None)
        ex.set_narrator(narrator)
        ex._browser_backend = _FakeBrowserBackend()
        plan = ActionPlan(
            actions=[
                Action(action_type=ActionType.BROWSER_CLICK, target="#btn", parameters=BrowserParams(selector="#btn"))
            ],
            raw_input="test",
        )

        results = await ex.execute(plan, invocation_source=InvocationSource.AUTONOMOUS)

        assert len(narrator.target_assessment_calls) == 1
        # It still proceeds to the real dispatch (which will itself fail/succeed
        # on its own merits since no real browser exists here) rather than
        # being force-skipped by the interrupt.
        assert "Skipped after interrupt" not in (results[0].error or "")
