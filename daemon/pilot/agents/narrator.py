"""ExecutionNarrator — live narration and risk-triggered interrupt-and-wait
for in-progress plan execution.

Two independent trigger sources, both wired automatically by `Executor`
(see `agents/executor.py`'s `set_narrator()`) once a narrator is configured:

- **Ambient narration** (`on_action_start`/`on_action_complete`) — always
  non-blocking, broadcasts a short spoken-style description of each action
  as it starts/finishes. Never registers a confirmation, never pauses
  anything.
- **Risk-triggered interrupt** (`on_plan_risk`/`on_target_assessment`) —
  gates a plan or a single browser action *before* it runs, using signals
  that already exist elsewhere in this codebase: the Agent Gateway's own
  critic verdict (`on_plan_risk`, a WARN this codebase previously computed
  and then silently discarded whenever a plan was otherwise allowed) and
  `pilot.system.dom_diff.assess_target()`'s pre-execution target check
  (`on_target_assessment`, previously dry-run-only).

Interrupt-and-wait reuses the exact `PendingConfirmation`/`confirm` RPC
mechanism `ThreatContainmentBridge` and `AutonomousHealingEngine` already
established for background-initiated confirm-then-continue flows — no new
safety primitive, no new RPC.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from pilot.actions import Action, ActionPlan, ActionResult
    from pilot.config import NarrationConfig, PilotConfig, PreviewConfig
    from pilot.system.action_preview import ActionPreview
    from pilot.system.dom_diff import TargetAssessment

logger = logging.getLogger("pilot.agents.narrator")


def _describe_action(action: Action) -> str:
    target = getattr(action, "target", "") or ""
    action_type = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
    return f"{action_type} ({target})" if target else action_type


class ExecutionNarrator:
    """Turns in-flight plan execution into live narration + pre-emptive
    interrupts, without introducing any new way to bypass or weaken the
    existing safety pipeline."""

    def __init__(
        self,
        config: PilotConfig,
        pending_confirms: dict[str, Any],
        broadcast_fn: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._config = config
        self._pending_confirms = pending_confirms
        self._broadcast_fn = broadcast_fn

    def _cfg(self) -> NarrationConfig:
        return self._config.narration

    def _preview_cfg(self) -> PreviewConfig:
        return self._config.preview

    # ── Ambient narration (never blocks) ──

    async def on_action_start(self, action: Action) -> None:
        cfg = self._cfg()
        if not cfg.enabled or not cfg.narrate_steps or not self._broadcast_fn:
            return
        await self._broadcast_fn(
            "execution_narration",
            {
                "phase": "start",
                "text": f"Starting: {_describe_action(action)}",
                "action_type": action.action_type.value,
                "target": getattr(action, "target", "") or "",
            },
        )

    async def on_action_complete(self, result: ActionResult) -> None:
        cfg = self._cfg()
        if not cfg.enabled or not cfg.narrate_steps or not self._broadcast_fn:
            return
        description = _describe_action(result.action)
        text = (
            f"Done: {description}" if result.success else f"Failed: {description} — {result.error or 'unknown error'}"
        )
        await self._broadcast_fn(
            "execution_narration",
            {
                "phase": "complete",
                "text": text,
                "success": result.success,
                "action_type": result.action.action_type.value,
            },
        )

    # ── Risk-triggered interrupt-and-wait ──

    async def on_plan_risk(self, plan: ActionPlan, critic_verdict: dict[str, Any] | None) -> bool:
        """Called once per plan with whatever critic_verdict the Agent
        Gateway produced. A BLOCK verdict already halts the plan before
        this is ever reached (the gateway denies it outright); an APPROVE
        needs no interrupt. Only a WARN on an otherwise-allowed plan gets
        surfaced here -- the exact signal `Executor.execute()` used to
        silently discard."""
        cfg = self._cfg()
        if not cfg.enabled or not cfg.interrupt_on_risk or not critic_verdict:
            return True
        if critic_verdict.get("verdict") != "WARN":
            return True

        reason = (
            critic_verdict.get("recommendation")
            or "; ".join(critic_verdict.get("issues", []))
            or "This plan was flagged as risky."
        )
        return await self._interrupt_and_wait(
            reason=reason,
            context={
                "kind": "plan_risk",
                "plan_summary": plan.explanation or plan.raw_input,
                "critic_verdict": critic_verdict,
            },
        )

    async def on_target_assessment(self, action: Action, assessment: TargetAssessment) -> bool:
        """Called for a single browser click/type/select/fill_form action,
        right before it actually executes, only when the assessment could
        be evaluated and flagged a problem."""
        cfg = self._cfg()
        if not cfg.enabled or not cfg.interrupt_on_risk:
            return True
        if not assessment.matchable:
            return True
        if assessment.found and assessment.visible and not assessment.ambiguous:
            return True

        return await self._interrupt_and_wait(
            reason=assessment.reason,
            context={
                "kind": "target_assessment",
                "action_type": action.action_type.value,
                "target": getattr(action, "target", "") or "",
            },
        )

    async def on_action_preview(self, action: Action, preview: ActionPreview) -> bool:
        """ "Simulate before executing" gate for autonomous background tasks
        (see `pilot.system.action_preview`) — pauses and shows a real
        screenshot with the action's target highlighted (plus, for browser
        actions, a real dry-run DOM diff), waiting for confirm/deny. Only
        ever called by `Executor.execute()` for `InvocationSource.AUTONOMOUS`
        plans — interactive/voice/gesture invocations already have the user
        watching in real time and skip this entirely."""
        cfg = self._preview_cfg()
        if not cfg.enabled:
            return True

        return await self._interrupt_and_wait(
            reason=preview.caption,
            context={
                "kind": "action_preview",
                "action_type": action.action_type.value,
                "target": getattr(action, "target", "") or "",
                "preview": preview.to_dict(),
            },
            timeout_seconds=cfg.confirm_timeout_seconds,
        )

    async def _interrupt_and_wait(
        self, *, reason: str, context: dict[str, Any], timeout_seconds: float | None = None
    ) -> bool:
        if not self._broadcast_fn:
            return True

        from pilot.server import PendingConfirmation

        plan_id = f"interrupt_{uuid.uuid4().hex[:8]}"
        pending = PendingConfirmation(plan_id=plan_id, event=asyncio.Event())
        self._pending_confirms[plan_id] = pending

        effective_timeout = timeout_seconds if timeout_seconds is not None else self._cfg().confirm_timeout_seconds
        await self._broadcast_fn(
            "execution_interrupt",
            {
                "plan_id": plan_id,
                "reason": reason,
                "timeout_seconds": effective_timeout,
                **context,
            },
        )

        try:
            await asyncio.wait_for(pending.event.wait(), timeout=effective_timeout)
        except TimeoutError:
            logger.warning("Execution interrupt timed out (plan_id=%s)", plan_id)
            await self._broadcast_fn("execution_interrupt_timeout", {"plan_id": plan_id})
            return False
        finally:
            self._pending_confirms.pop(plan_id, None)

        if not pending.confirmed:
            await self._broadcast_fn("execution_interrupt_denied", {"plan_id": plan_id})
        return pending.confirmed
