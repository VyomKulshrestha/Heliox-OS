"""Permission tier checker.

Tier 0: Read-only         — no confirmation, no snapshot, no root
Tier 1: User-space write  — no confirmation, no snapshot, no root
Tier 2: System modify     — confirmation required, optional snapshot
Tier 3: Destructive       — explicit confirmation, snapshot required
Tier 4: Root / Critical   — explicit confirmation + passphrase, snapshot required, root toggle ON
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pilot.actions import Action, ActionPlan, PermissionTier

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.security.permissions")


@dataclass
class PermissionDecision:
    allowed: bool
    tier: PermissionTier
    requires_confirmation: bool
    requires_snapshot: bool
    requires_root_toggle: bool
    denial_reason: str = ""


class PermissionChecker:
    """Evaluates actions against the permission tier system."""

    def __init__(self, config: PilotConfig) -> None:
        self._config = config

    def check_action(self, action: Action) -> PermissionDecision:
        tier = action.permission_tier

        if tier == PermissionTier.ROOT_CRITICAL and not self._config.security.root_enabled:
            return PermissionDecision(
                allowed=False,
                tier=tier,
                requires_confirmation=True,
                requires_snapshot=True,
                requires_root_toggle=True,
                denial_reason="Root access is disabled. Enable it in settings to proceed.",
            )

        return PermissionDecision(
            allowed=True,
            tier=tier,
            requires_confirmation=tier >= PermissionTier.SYSTEM_MODIFY,
            requires_snapshot=(tier >= PermissionTier.DESTRUCTIVE and self._config.security.snapshot_on_destructive),
            requires_root_toggle=tier >= PermissionTier.ROOT_CRITICAL,
        )

    def check_plan(self, plan: ActionPlan) -> list[PermissionDecision]:
        return [self.check_action(a) for a in plan.actions]

    def plan_requires_confirmation(self, plan: ActionPlan) -> bool:
        return any(d.requires_confirmation for d in self.check_plan(plan))

    def plan_requires_snapshot(self, plan: ActionPlan) -> bool:
        return any(d.requires_snapshot for d in self.check_plan(plan))

    def plan_allowed(self, plan: ActionPlan) -> tuple[bool, list[str]]:
        """Check if all actions in a plan are allowed. Returns (allowed, reasons)."""
        decisions = self.check_plan(plan)
        denied = [d for d in decisions if not d.allowed]
        return len(denied) == 0, [d.denial_reason for d in denied]
