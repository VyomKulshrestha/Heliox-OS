"""Autonomous Healing Engine — passive system-health monitoring with tiered
auto-remediation.

Borrows IBM Power Autonomous Operations' pattern: instead of only reacting
to on-demand commands, the daemon watches its own system-health signals
(CPU/memory/disk, via the existing `BackgroundTaskManager` monitors) and,
when one crosses its threshold, generates a remediation goal and plans it
through the same `Planner`/`Executor` pipeline every other command uses.

Autonomy is tiered, per the confirmed design:
  - If the resulting plan is entirely low-tier (<= `auto_execute_max_tier`,
    default USER_WRITE) AND contains no irreversible action, it is executed
    immediately through the normal safety pipeline (PermissionChecker /
    AgentGateway / Learned Risk Gate) — the same pipeline every other
    `Executor.execute()` caller goes through, not a bypass of it.
  - Otherwise the plan is proposed and held pending explicit user
    confirmation, reusing the exact `PendingConfirmation` / `confirm` RPC
    mechanism `ThreatContainmentBridge` already established for
    background-initiated (non-request-driven) confirm-then-execute flows.

This module intentionally adds no new safety primitive — it composes
Planner + Executor + PendingConfirmation exactly as `AutonomousExecutor`
and `ThreatContainmentBridge` already do, just triggered by a health alert
instead of a voice/text goal or a forensics report.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.actions import ActionPlan
from pilot.security.gateway import InvocationSource

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.agents.planner import Planner
    from pilot.config import PilotConfig, SelfHealingConfig

logger = logging.getLogger("pilot.agents.autonomous_healing")

# Fallback goal templates, used when SelfHealingConfig.goal_templates has no
# entry for a given metric. `{detail}` is filled with the triggering
# BackgroundTask check's own human-readable message/data.
DEFAULT_GOAL_TEMPLATES: dict[str, str] = {
    "cpu": (
        "The system is under high CPU load ({detail}). Identify what is "
        "consuming the most CPU and take an appropriate, minimal, safe "
        "action to reduce it if one exists."
    ),
    "memory": (
        "The system is running low on available memory ({detail}). "
        "Identify what is consuming the most memory and take an "
        "appropriate, minimal, safe action to free some up if one exists."
    ),
    "disk": (
        "The system is running low on disk space ({detail}). Identify "
        "reclaimable space such as temp files, caches, or old logs, and "
        "free it up if it's safe to do so."
    ),
}


class HealingOutcome(StrEnum):
    AUTO_EXECUTED = "auto_executed"
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    TIMED_OUT = "timed_out"
    NO_ACTION = "no_action"
    PLAN_ERROR = "plan_error"


@dataclass
class HealingAttempt:
    """One health-alert-to-remediation attempt, tracked for the UI/RPCs."""

    attempt_id: str
    metric: str
    trigger: dict[str, Any]
    goal: str
    plan_id: str = ""
    outcome: str = HealingOutcome.NO_ACTION.value
    max_tier: int = 0
    irreversible: bool = False
    explanation: str = ""
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "metric": self.metric,
            "trigger": self.trigger,
            "goal": self.goal,
            "plan_id": self.plan_id,
            "outcome": self.outcome,
            "max_tier": self.max_tier,
            "irreversible": self.irreversible,
            "explanation": self.explanation,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


class AutonomousHealingEngine:
    """Turns BackgroundTaskManager health alerts into tiered remediation."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        config: PilotConfig,
        pending_confirms: dict[str, Any],
        broadcast_fn: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._config = config
        self._pending_confirms = pending_confirms
        self._broadcast_fn = broadcast_fn
        self._last_attempt_at: dict[str, float] = {}
        self._attempts: dict[str, HealingAttempt] = {}

    def _cfg(self) -> SelfHealingConfig:
        return self._config.self_healing

    def _in_cooldown(self, metric: str) -> bool:
        last = self._last_attempt_at.get(metric)
        if last is None:
            return False
        return (time.time() - last) < self._cfg().cooldown_seconds

    # ── BackgroundTask.on_trigger hooks (one per watched metric) ──

    async def on_cpu_alert(self, result: dict[str, Any]) -> HealingAttempt | None:
        return await self.on_health_alert("cpu", result)

    async def on_memory_alert(self, result: dict[str, Any]) -> HealingAttempt | None:
        return await self.on_health_alert("memory", result)

    async def on_disk_alert(self, result: dict[str, Any]) -> HealingAttempt | None:
        return await self.on_health_alert("disk", result)

    # ── Core flow ──

    async def on_health_alert(self, metric: str, result: dict[str, Any]) -> HealingAttempt | None:
        """Handle one triggered health check. Returns None if self-healing
        is disabled, this metric isn't watched, or it's still in cooldown
        from a previous attempt on the same metric."""
        cfg = self._cfg()
        if not cfg.enabled:
            return None
        if metric not in cfg.watched_metrics:
            return None
        if self._in_cooldown(metric):
            logger.debug("Self-healing: %s alert suppressed (cooldown)", metric)
            return None
        self._last_attempt_at[metric] = time.time()

        goal_template = cfg.goal_templates.get(metric) or DEFAULT_GOAL_TEMPLATES.get(metric, "")
        if not goal_template:
            return None
        detail = result.get("message") or ", ".join(
            f"{k}={v}" for k, v in result.items() if k not in ("triggered", "message")
        )
        goal = goal_template.format(detail=detail)

        attempt = HealingAttempt(attempt_id=f"heal_{uuid.uuid4().hex[:8]}", metric=metric, trigger=result, goal=goal)
        self._attempts[attempt.attempt_id] = attempt
        logger.info("Self-healing: %s alert -> goal: %s", metric, goal[:120])

        try:
            plan = await self._planner.plan(goal)
        except Exception as exc:
            logger.warning("Self-healing: planning failed for %s alert: %s", metric, exc)
            attempt.outcome = HealingOutcome.PLAN_ERROR.value
            attempt.explanation = str(exc)
            attempt.resolved_at = time.time()
            return attempt

        if plan.error or not plan.actions:
            attempt.outcome = HealingOutcome.NO_ACTION.value
            attempt.explanation = plan.error or "Planner produced no actions"
            attempt.resolved_at = time.time()
            return attempt

        attempt.max_tier = int(plan.max_tier)
        attempt.irreversible = plan.needs_confirmation_unconditional

        if self._is_auto_executable(plan, cfg):
            await self._auto_execute(attempt, plan)
        else:
            await self._propose_and_wait(attempt, plan)
        return attempt

    def _is_auto_executable(self, plan: ActionPlan, cfg: SelfHealingConfig) -> bool:
        return int(plan.max_tier) <= cfg.auto_execute_max_tier and not plan.needs_confirmation_unconditional

    async def _auto_execute(self, attempt: HealingAttempt, plan: ActionPlan) -> None:
        attempt.plan_id = attempt.attempt_id
        if self._broadcast_fn:
            await self._broadcast_fn("self_healing_auto_executing", attempt.to_dict())
        results = await self._executor.execute(
            plan, invocation_source=InvocationSource.SELF_HEALING, plan_id=attempt.plan_id
        )
        attempt.outcome = HealingOutcome.AUTO_EXECUTED.value
        attempt.explanation = "; ".join((r.output or r.error or "") for r in results)[:500]
        attempt.resolved_at = time.time()
        if self._broadcast_fn:
            await self._broadcast_fn("self_healing_complete", attempt.to_dict())

    async def _propose_and_wait(self, attempt: HealingAttempt, plan: ActionPlan) -> None:
        # Local import: PendingConfirmation lives in server.py, which imports
        # this module's package indirectly — importing at call time (not
        # module load time) avoids a circular import, same as
        # ThreatContainmentBridge.route_and_confirm does.
        from pilot.server import PendingConfirmation

        plan_id = attempt.attempt_id
        attempt.plan_id = plan_id
        attempt.outcome = HealingOutcome.PROPOSED.value

        pending = PendingConfirmation(plan_id=plan_id, event=asyncio.Event(), plan=plan)
        self._pending_confirms[plan_id] = pending

        cfg = self._cfg()
        if self._broadcast_fn:
            await self._broadcast_fn(
                "self_healing_confirmation_required",
                {
                    **attempt.to_dict(),
                    "actions": [a.model_dump() for a in plan.actions],
                    "timeout_seconds": cfg.confirm_timeout_seconds,
                },
            )

        try:
            await asyncio.wait_for(pending.event.wait(), timeout=cfg.confirm_timeout_seconds)
        except TimeoutError:
            logger.warning("Self-healing: confirmation timed out for %s (plan_id=%s)", attempt.metric, plan_id)
            attempt.outcome = HealingOutcome.TIMED_OUT.value
            attempt.resolved_at = time.time()
            if self._broadcast_fn:
                await self._broadcast_fn("self_healing_timeout", attempt.to_dict())
            return
        finally:
            self._pending_confirms.pop(plan_id, None)

        if not pending.confirmed:
            attempt.outcome = HealingOutcome.DENIED.value
            attempt.resolved_at = time.time()
            if self._broadcast_fn:
                await self._broadcast_fn("self_healing_denied", attempt.to_dict())
            return

        results = await self._executor.execute(plan, invocation_source=InvocationSource.SELF_HEALING, plan_id=plan_id)
        attempt.outcome = HealingOutcome.CONFIRMED.value
        attempt.explanation = "; ".join((r.output or r.error or "") for r in results)[:500]
        attempt.resolved_at = time.time()
        if self._broadcast_fn:
            await self._broadcast_fn("self_healing_complete", attempt.to_dict())

    # ── Read-only accessors for RPCs ──

    def list_attempts(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in sorted(self._attempts.values(), key=lambda a: a.created_at, reverse=True)[:50]]

    def get_attempt(self, attempt_id: str) -> HealingAttempt | None:
        return self._attempts.get(attempt_id)
