"""The world model and Gnomon review every plan an autonomous job is about to run.

Autonomous jobs, scheduled runs, MCP tasks and voice workflows all go through
``AutonomousExecutor``. Before each round runs, this reviewer asks:

* the world model (``assess_plan_risk``): predicted risk of the plan's effects;
* Gnomon, when it is loaded: scope, harm and untrusted-instruction checks.

Either can only add an approval requirement, and the approval prompt then shows
why. Neither can approve or remove one: deterministic tiers, the gateway and the
person stay the authority. Completion claims get Gnomon's postcondition check,
which can only send the loop back to work.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pilot.actions import READ_ONLY_ACTIONS, ActionType, PermissionTier

# READ_ONLY_ACTIONS also lists a few effectful entries (opening a URL or app, a notification).
# Gnomon should see those: an injected link is exactly what its untrusted-instruction check is for.
_EFFECTFUL = frozenset({ActionType.OPEN_URL, ActionType.OPEN_APPLICATION, ActionType.NOTIFY})

# Actions that only read or inspect. Several sit in the "always safe" list and report the
# USER_WRITE tier, so the tier alone cannot say whether a plan changes anything.
OBSERVATION_ACTIONS = (frozenset(READ_ONLY_ACTIONS) - _EFFECTFUL) | {
    ActionType.FILE_READ,
    ActionType.FILE_LIST,
    ActionType.FILE_SEARCH,
    ActionType.CLIPBOARD_READ,
    ActionType.SYSTEM_INFO,
    ActionType.SYSTEM_HEALTH_REVIEW,
    ActionType.DISK_USAGE,
    ActionType.MEMORY_USAGE,
    ActionType.CPU_USAGE,
    ActionType.NETWORK_INFO,
    ActionType.BATTERY_INFO,
    ActionType.PROCESS_LIST,
    ActionType.PROCESS_INFO,
    ActionType.SERVICE_STATUS,
    ActionType.WINDOW_LIST,
    ActionType.VOLUME_GET,
    ActionType.BRIGHTNESS_GET,
    ActionType.WIFI_LIST,
    ActionType.DISK_LIST,
    ActionType.USER_LIST,
    ActionType.USER_INFO,
    ActionType.ENV_GET,
    ActionType.ENV_LIST,
    ActionType.SCREEN_OCR,
    ActionType.SCREEN_ANALYZE,
    ActionType.SCREEN_FIND_TEXT,
    ActionType.SCREEN_ELEMENT_MAP,
    ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT,
    ActionType.SCREEN_DETECT_ELEMENTS,
    ActionType.SCHEDULE_LIST,
    ActionType.TRIGGER_LIST,
    ActionType.MOUSE_POSITION,
    ActionType.SCREENSHOT,
    ActionType.REGISTRY_READ,
    ActionType.PACKAGE_SEARCH,
    ActionType.GNOME_SETTING_READ,
}


def changes_state(plan: Any) -> bool:
    """True when any action does more than observe (read-only MCP tools and GET requests observe)."""
    return any(
        action.action_type not in OBSERVATION_ACTIONS and action.permission_tier > PermissionTier.READ_ONLY
        for action in plan.actions
    )


logger = logging.getLogger("pilot.agents.autonomous_review")


@dataclass
class PlanReview:
    requires_approval: bool = False
    reasons: list[str] = field(default_factory=list)
    risk_assessment: dict[str, Any] | None = None  # set when the world model asked for approval
    gnomon: dict[str, Any] | None = None

    def summary(self) -> str:
        return " ".join(self.reasons)


class AutonomousReviewer:
    def __init__(
        self,
        config_getter: Callable[[], Any],
        *,
        gnomon: Any | None = None,
        on_assessment: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._config_getter = config_getter
        self._gnomon = gnomon
        self._on_assessment = on_assessment

    def warm(self) -> None:
        """Start loading Gnomon in the background when a job begins; never blocks."""
        if self._gnomon is not None:
            self._gnomon.warm()

    async def review_plan(
        self, job: Any, plan: Any, source: Any, plan_id: str, untrusted_context: str = ""
    ) -> PlanReview:
        from pilot.agents.destructive_critic import assess_plan_risk

        assessment = assess_plan_risk(plan, self._config_getter(), source=getattr(source, "value", str(source)))
        review = PlanReview()
        if assessment.requires_confirmation:
            review.requires_approval = True
            review.risk_assessment = assessment.to_dict()
            detail = f": {'; '.join(assessment.reasons[:2])}" if assessment.reasons else ""
            review.reasons.append(f"The world model predicts {assessment.world_model_score:.0%} risk{detail}.")
        mutating = changes_state(plan)
        if mutating and assessment.model_version == "evaluation-error":
            review.requires_approval = True
            review.reasons.append("The world model could not evaluate this plan, so Heliox asks first.")

        if self._gnomon is not None and mutating:
            try:
                verdict = await self._gnomon.review_plan(job.goal, plan, untrusted_context)
            except Exception:
                logger.warning("Gnomon plan review failed", exc_info=True)
                verdict = None
            if verdict is not None:
                review.gnomon = verdict.to_dict()
                if verdict.escalate:
                    review.requires_approval = True
                    review.reasons.extend(verdict.reasons)

        if self._on_assessment is not None:
            try:
                await self._on_assessment(
                    plan_id, {"job_id": job.job_id, **assessment.to_dict(), "gnomon": review.gnomon}
                )
            except Exception:
                logger.debug("Could not publish the autonomous risk assessment", exc_info=True)
        return review

    async def review_completion(self, goal: str, evidence: str, receipt: str, expected: str) -> str:
        """Return a reason when Gnomon rejects a completion claim, or "" to accept it."""
        if self._gnomon is None:
            return ""
        try:
            verdict = await self._gnomon.review_completion(goal, evidence, receipt, expected)
        except Exception:
            logger.warning("Gnomon completion review failed", exc_info=True)
            return ""
        return " ".join(verdict.reasons) if verdict is not None and verdict.escalate else ""
