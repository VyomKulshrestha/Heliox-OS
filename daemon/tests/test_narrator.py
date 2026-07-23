"""Tests for pilot.agents.narrator.ExecutionNarrator.

Covers the two independent trigger sources: ambient (always non-blocking)
narration, and risk-triggered interrupt-and-wait (reusing the same
PendingConfirmation/confirm mechanism AutonomousHealingEngine's
propose-and-wait branch already uses -- see test_autonomous_healing.py's
TestProposeAndWait for the mirrored shape).
"""

from __future__ import annotations

import asyncio

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams
from pilot.agents.narrator import ExecutionNarrator
from pilot.config import PilotConfig
from pilot.system.action_preview import ActionPreview
from pilot.system.dom_diff import TargetAssessment


class _Broadcast:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, method: str, params: dict) -> None:
        self.calls.append((method, params))


def _action(action_type: ActionType = ActionType.PROCESS_LIST, target: str = "x") -> Action:
    return Action(action_type=action_type, target=target, parameters=EmptyParams())


def _config(**overrides) -> PilotConfig:
    cfg = PilotConfig()
    cfg.narration.enabled = True
    for k, v in overrides.items():
        setattr(cfg.narration, k, v)
    return cfg


def _narrator(**cfg_overrides) -> tuple[ExecutionNarrator, dict, _Broadcast]:
    pending_confirms: dict = {}
    broadcast = _Broadcast()
    narrator = ExecutionNarrator(
        config=_config(**cfg_overrides), pending_confirms=pending_confirms, broadcast_fn=broadcast
    )
    return narrator, pending_confirms, broadcast


class TestDisabled:
    @pytest.mark.asyncio
    async def test_on_action_start_noop_when_disabled(self):
        narrator, _, broadcast = _narrator()
        narrator._config.narration.enabled = False
        await narrator.on_action_start(_action())
        assert broadcast.calls == []

    @pytest.mark.asyncio
    async def test_on_plan_risk_proceeds_when_disabled(self):
        narrator, _, broadcast = _narrator()
        narrator._config.narration.enabled = False
        result = await narrator.on_plan_risk(ActionPlan(actions=[], raw_input="x"), {"verdict": "WARN"})
        assert result is True
        assert broadcast.calls == []


class TestAmbientNarration:
    @pytest.mark.asyncio
    async def test_on_action_start_broadcasts_when_enabled(self):
        narrator, _, broadcast = _narrator()
        await narrator.on_action_start(_action())
        assert len(broadcast.calls) == 1
        method, params = broadcast.calls[0]
        assert method == "execution_narration"
        assert params["phase"] == "start"
        assert "Starting" in params["text"]

    @pytest.mark.asyncio
    async def test_on_action_complete_success_text(self):
        narrator, _, broadcast = _narrator()
        result = ActionResult(action=_action(), success=True, output="ok")
        await narrator.on_action_complete(result)
        method, params = broadcast.calls[0]
        assert params["success"] is True
        assert "Done" in params["text"]

    @pytest.mark.asyncio
    async def test_on_action_complete_failure_text_includes_error(self):
        narrator, _, broadcast = _narrator()
        result = ActionResult(action=_action(), success=False, error="boom")
        await narrator.on_action_complete(result)
        method, params = broadcast.calls[0]
        assert params["success"] is False
        assert "Failed" in params["text"]
        assert "boom" in params["text"]

    @pytest.mark.asyncio
    async def test_narrate_steps_off_suppresses_ambient_narration(self):
        narrator, _, broadcast = _narrator(narrate_steps=False)
        await narrator.on_action_start(_action())
        assert broadcast.calls == []


class TestPlanRiskInterrupt:
    @pytest.mark.asyncio
    async def test_no_critic_verdict_proceeds(self):
        narrator, _, broadcast = _narrator()
        result = await narrator.on_plan_risk(ActionPlan(actions=[], raw_input="x"), None)
        assert result is True
        assert broadcast.calls == []

    @pytest.mark.asyncio
    async def test_approve_verdict_proceeds_without_interrupt(self):
        narrator, _, broadcast = _narrator()
        result = await narrator.on_plan_risk(ActionPlan(actions=[], raw_input="x"), {"verdict": "APPROVE"})
        assert result is True
        assert broadcast.calls == []

    @pytest.mark.asyncio
    async def test_interrupt_on_risk_off_proceeds_even_on_warn(self):
        narrator, _, broadcast = _narrator(interrupt_on_risk=False)
        result = await narrator.on_plan_risk(ActionPlan(actions=[], raw_input="x"), {"verdict": "WARN"})
        assert result is True
        assert broadcast.calls == []

    @pytest.mark.asyncio
    async def test_warn_verdict_registers_pending_confirmation_and_broadcasts(self):
        narrator, pending_confirms, broadcast = _narrator(confirm_timeout_seconds=0.2)
        plan = ActionPlan(actions=[], raw_input="test plan")
        task = asyncio.create_task(narrator.on_plan_risk(plan, {"verdict": "WARN", "recommendation": "risky"}))
        await asyncio.sleep(0.05)
        assert len(pending_confirms) == 1
        assert broadcast.calls[0][0] == "execution_interrupt"
        assert broadcast.calls[0][1]["reason"] == "risky"
        await task

    @pytest.mark.asyncio
    async def test_warn_verdict_confirmed_returns_true(self):
        narrator, pending_confirms, broadcast = _narrator(confirm_timeout_seconds=5.0)
        plan = ActionPlan(actions=[], raw_input="test plan")
        task = asyncio.create_task(narrator.on_plan_risk(plan, {"verdict": "WARN", "recommendation": "risky"}))
        await asyncio.sleep(0.05)
        plan_id = next(iter(pending_confirms.keys()))
        pending_confirms[plan_id].confirmed = True
        pending_confirms[plan_id].event.set()
        result = await task
        assert result is True

    @pytest.mark.asyncio
    async def test_warn_verdict_denied_returns_false(self):
        narrator, pending_confirms, broadcast = _narrator(confirm_timeout_seconds=5.0)
        plan = ActionPlan(actions=[], raw_input="test plan")
        task = asyncio.create_task(narrator.on_plan_risk(plan, {"verdict": "WARN", "recommendation": "risky"}))
        await asyncio.sleep(0.05)
        plan_id = next(iter(pending_confirms.keys()))
        pending_confirms[plan_id].confirmed = False
        pending_confirms[plan_id].event.set()
        result = await task
        assert result is False
        assert any(m == "execution_interrupt_denied" for m, _ in broadcast.calls)

    @pytest.mark.asyncio
    async def test_warn_verdict_timeout_returns_false(self):
        narrator, pending_confirms, broadcast = _narrator(confirm_timeout_seconds=0.05)
        plan = ActionPlan(actions=[], raw_input="test plan")
        result = await narrator.on_plan_risk(plan, {"verdict": "WARN", "recommendation": "risky"})
        assert result is False
        assert pending_confirms == {}
        assert any(m == "execution_interrupt_timeout" for m, _ in broadcast.calls)


class TestTargetAssessmentInterrupt:
    @pytest.mark.asyncio
    async def test_unmatchable_assessment_proceeds(self):
        narrator, _, broadcast = _narrator()
        assessment = TargetAssessment(matchable=False, reason="too complex")
        result = await narrator.on_target_assessment(_action(ActionType.BROWSER_CLICK), assessment)
        assert result is True
        assert broadcast.calls == []

    @pytest.mark.asyncio
    async def test_clean_assessment_proceeds(self):
        narrator, _, broadcast = _narrator()
        assessment = TargetAssessment(
            matchable=True, found=True, visible=True, ambiguous=False, reason="found and visible"
        )
        result = await narrator.on_target_assessment(_action(ActionType.BROWSER_CLICK), assessment)
        assert result is True
        assert broadcast.calls == []

    @pytest.mark.asyncio
    async def test_problem_assessment_triggers_interrupt(self):
        narrator, pending_confirms, broadcast = _narrator(confirm_timeout_seconds=5.0)
        assessment = TargetAssessment(matchable=True, found=False, reason="selector '#x' not found")
        task = asyncio.create_task(narrator.on_target_assessment(_action(ActionType.BROWSER_CLICK), assessment))
        await asyncio.sleep(0.05)
        assert len(pending_confirms) == 1
        plan_id = next(iter(pending_confirms.keys()))
        pending_confirms[plan_id].confirmed = True
        pending_confirms[plan_id].event.set()
        result = await task
        assert result is True
        assert broadcast.calls[0][1]["kind"] == "target_assessment"

    @pytest.mark.asyncio
    async def test_ambiguous_assessment_triggers_interrupt(self):
        narrator, pending_confirms, broadcast = _narrator(confirm_timeout_seconds=0.05)
        assessment = TargetAssessment(
            matchable=True, found=True, visible=True, ambiguous=True, reason="matches 2 elements"
        )
        result = await narrator.on_target_assessment(_action(ActionType.BROWSER_CLICK), assessment)
        assert result is False  # times out with no responder

    @pytest.mark.asyncio
    async def test_interrupt_on_risk_off_skips_target_assessment_interrupt(self):
        narrator, _, broadcast = _narrator(interrupt_on_risk=False)
        assessment = TargetAssessment(matchable=True, found=False, reason="not found")
        result = await narrator.on_target_assessment(_action(ActionType.BROWSER_CLICK), assessment)
        assert result is True
        assert broadcast.calls == []


def _preview_narrator(**preview_overrides) -> tuple[ExecutionNarrator, dict, _Broadcast]:
    """Same shape as `_narrator()` but configures PreviewConfig (a separate
    section from NarrationConfig) since on_action_preview() is gated by
    config.preview, not config.narration."""
    pending_confirms: dict = {}
    broadcast = _Broadcast()
    config = PilotConfig()
    config.preview.enabled = True
    for k, v in preview_overrides.items():
        setattr(config.preview, k, v)
    narrator = ExecutionNarrator(config=config, pending_confirms=pending_confirms, broadcast_fn=broadcast)
    return narrator, pending_confirms, broadcast


def _preview(caption: str = "About to click: Save") -> ActionPreview:
    return ActionPreview(screenshot_base64="abc", bbox=None, target_label=None, caption=caption)


class TestActionPreviewInterrupt:
    @pytest.mark.asyncio
    async def test_disabled_proceeds_without_broadcasting(self):
        narrator, _, broadcast = _preview_narrator()
        narrator._config.preview.enabled = False
        result = await narrator.on_action_preview(_action(ActionType.BROWSER_CLICK), _preview())
        assert result is True
        assert broadcast.calls == []

    @pytest.mark.asyncio
    async def test_enabled_always_interrupts_and_waits(self):
        """Unlike on_target_assessment (only interrupts on a PROBLEM),
        on_action_preview interrupts for every autonomous action when
        enabled -- the whole point is showing the preview before every
        gated commit, not just risky ones."""
        narrator, pending_confirms, broadcast = _preview_narrator(confirm_timeout_seconds=5.0)
        task = asyncio.create_task(narrator.on_action_preview(_action(ActionType.BROWSER_CLICK), _preview()))
        await asyncio.sleep(0.05)
        assert len(pending_confirms) == 1
        plan_id = next(iter(pending_confirms.keys()))
        pending_confirms[plan_id].confirmed = True
        pending_confirms[plan_id].event.set()
        result = await task
        assert result is True

        assert broadcast.calls[0][0] == "execution_interrupt"
        payload = broadcast.calls[0][1]
        assert payload["kind"] == "action_preview"
        assert payload["preview"]["caption"] == "About to click: Save"

    @pytest.mark.asyncio
    async def test_denied_returns_false(self):
        narrator, pending_confirms, broadcast = _preview_narrator(confirm_timeout_seconds=5.0)
        task = asyncio.create_task(narrator.on_action_preview(_action(ActionType.BROWSER_CLICK), _preview()))
        await asyncio.sleep(0.05)
        plan_id = next(iter(pending_confirms.keys()))
        pending_confirms[plan_id].confirmed = False
        pending_confirms[plan_id].event.set()
        result = await task
        assert result is False
        assert broadcast.calls[-1][0] == "execution_interrupt_denied"

    @pytest.mark.asyncio
    async def test_timeout_with_no_responder_returns_false(self):
        narrator, _, broadcast = _preview_narrator(confirm_timeout_seconds=0.05)
        result = await narrator.on_action_preview(_action(ActionType.BROWSER_CLICK), _preview())
        assert result is False
        assert broadcast.calls[-1][0] == "execution_interrupt_timeout"

    @pytest.mark.asyncio
    async def test_uses_preview_timeout_not_narration_timeout(self):
        """PreviewConfig.confirm_timeout_seconds must actually be the value
        used -- not silently falling back to NarrationConfig's (different)
        default, which would be a config-wiring bug."""
        narrator, _, broadcast = _preview_narrator(confirm_timeout_seconds=0.05)
        narrator._config.narration.confirm_timeout_seconds = 999.0  # would hang the test if used by mistake
        result = await narrator.on_action_preview(_action(ActionType.BROWSER_CLICK), _preview())
        assert result is False
        assert broadcast.calls[0][1]["timeout_seconds"] == 0.05
