"""Agent Gateway — source-scoped permissions for shell/browsing/system-control.

Motivated by a real gap: `PermissionChecker` (permissions.py) applies the
same global tier policy no matter where a plan came from, and
`DestructiveCriticAgent` only runs inside server.py's interactive `execute`
RPC handler, before `Executor.execute()` is ever called. Any caller that
reaches `Executor.execute()` a different way — most importantly
`autonomous_submit`'s fire-and-forget background jobs — never passes
through the critic at all. Combined with browser actions previously being
mis-tiered to always-safe (see actions.py's retiered ALWAYS_SAFE set), an
autonomous job could drive the browser with zero confirmation, zero critic
review, and zero audit trail.

This module adds a second, orthogonal gate that sits ALONGSIDE
`PermissionChecker`, never replacing it: `AgentGateway.authorize()` is
called first in `Executor.execute()`, and both it and the existing
`plan_allowed()` check must pass. The gateway can only add denials on top
of the existing tier logic, never relax it.

Scoping model: a small set of source-based profiles (interactive,
autonomous, web_agent, voice, gesture) define an enforced FLOOR per
ActionFamily (shell/browsing/system_control/other). A caller may attach an
optional `TaskScopeOverride` that can only NARROW that floor further, never
widen it — see `resolve_effective_profile()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from pilot.actions import ActionType, PermissionTier

if TYPE_CHECKING:
    from pilot.actions import ActionPlan
    from pilot.agents.destructive_critic import DestructiveCriticAgent
    from pilot.config import PilotConfig
    from pilot.security.gateway_audit import AgentGatewayAuditStore
    from pilot.security.permissions import PermissionChecker

# Sentinel action_index for a plan-level (not per-action) audit row, e.g. the
# critic verdict, which evaluates the whole plan rather than one action.
PLAN_LEVEL_AUDIT_INDEX = -1

logger = logging.getLogger("pilot.security.gateway")


class InvocationSource(StrEnum):
    """Where a plan being executed originated from. Threaded through
    `Executor.execute()` rather than stored on `Action`/`ActionPlan` itself,
    to avoid touching the ~150 existing action-dispatch call sites."""

    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"
    WEB_AGENT = "web_agent"
    VOICE = "voice"
    GESTURE = "gesture"
    UNKNOWN = "unknown"  # fail-open bucket for call sites not yet tagged


class ActionFamily(StrEnum):
    SHELL = "shell"
    BROWSING = "browsing"
    SYSTEM_CONTROL = "system_control"
    OTHER = "other"


# ── Action-type family groupings ──
# These are the gateway's own classification, distinct from actions.py's
# tier-oriented sets (READ_ONLY_ACTIONS/DESTRUCTIVE_ACTIONS/etc.) — a family
# groups *what kind of surface* an action touches, not how risky it is.

SHELL_ACTIONS: set[ActionType] = {
    ActionType.SHELL_COMMAND,
    ActionType.SHELL_SCRIPT,
    ActionType.PTY_EXEC,
    ActionType.CODE_EXECUTE,
    ActionType.CODE_GENERATE_AND_RUN,
    ActionType.SSH_COMMAND,
    ActionType.SSH_SCRIPT,
}

BROWSING_ACTIONS: set[ActionType] = {a for a in ActionType if a.value.startswith("browser_")}

SYSTEM_CONTROL_ACTIONS: set[ActionType] = {
    ActionType.MOUSE_CLICK,
    ActionType.MOUSE_DOUBLE_CLICK,
    ActionType.MOUSE_RIGHT_CLICK,
    ActionType.MOUSE_MOVE,
    ActionType.MOUSE_DRAG,
    ActionType.MOUSE_SCROLL,
    ActionType.KEYBOARD_TYPE,
    ActionType.KEYBOARD_PRESS,
    ActionType.KEYBOARD_HOTKEY,
    ActionType.KEYBOARD_HOLD,
    ActionType.PROCESS_KILL,
    ActionType.FILE_DELETE,
    ActionType.FILE_PERMISSIONS,
    ActionType.SERVICE_START,
    ActionType.SERVICE_STOP,
    ActionType.SERVICE_RESTART,
    ActionType.SERVICE_ENABLE,
    ActionType.SERVICE_DISABLE,
    ActionType.POWER_SHUTDOWN,
    ActionType.POWER_RESTART,
    ActionType.POWER_LOGOUT,
    ActionType.WIFI_CONNECT,
    ActionType.WIFI_DISCONNECT,
    ActionType.DISK_MOUNT,
    ActionType.DISK_UNMOUNT,
    ActionType.REGISTRY_WRITE,
    ActionType.WINDOW_CLOSE,
}


def action_family(action_type: ActionType) -> ActionFamily:
    if action_type in SHELL_ACTIONS:
        return ActionFamily.SHELL
    if action_type in BROWSING_ACTIONS:
        return ActionFamily.BROWSING
    if action_type in SYSTEM_CONTROL_ACTIONS:
        return ActionFamily.SYSTEM_CONTROL
    return ActionFamily.OTHER


@dataclass
class SourceProfile:
    """The enforced floor for one InvocationSource: a per-family tier
    ceiling, an explicit deny list, and whether root/Tier-4 actions are
    reachable at all (independent of the per-family tier ceiling, since a
    ceiling of ROOT_CRITICAL would otherwise implicitly allow root)."""

    max_tier: dict[str, int] = field(
        default_factory=lambda: {f.value: int(PermissionTier.ROOT_CRITICAL) for f in ActionFamily}
    )
    deny_action_types: list[str] = field(default_factory=list)
    allow_root: bool = True


DEFAULT_SOURCE_PROFILES: dict[str, SourceProfile] = {
    # Strict no-op floor — interactive traffic already goes through
    # PermissionChecker's tier/confirmation gate and (for Tier 3/4 plans)
    # the critic; the gateway must not add any additional restriction here.
    "interactive": SourceProfile(
        max_tier={f.value: int(PermissionTier.ROOT_CRITICAL) for f in ActionFamily},
    ),
    "autonomous": SourceProfile(
        max_tier={
            ActionFamily.SHELL.value: int(PermissionTier.SYSTEM_MODIFY),
            ActionFamily.BROWSING.value: int(PermissionTier.SYSTEM_MODIFY),
            ActionFamily.SYSTEM_CONTROL.value: int(PermissionTier.USER_WRITE),
            ActionFamily.OTHER.value: int(PermissionTier.SYSTEM_MODIFY),
        },
        deny_action_types=["browser_execute_js", "power_shutdown", "power_restart", "registry_write"],
        allow_root=False,
    ),
    "web_agent": SourceProfile(
        max_tier={
            ActionFamily.SHELL.value: int(PermissionTier.READ_ONLY),
            ActionFamily.BROWSING.value: int(PermissionTier.SYSTEM_MODIFY),
            ActionFamily.SYSTEM_CONTROL.value: int(PermissionTier.READ_ONLY),
            ActionFamily.OTHER.value: int(PermissionTier.USER_WRITE),
        },
        deny_action_types=["browser_execute_js"],
        allow_root=False,
    ),
    "voice": SourceProfile(
        max_tier={
            ActionFamily.SHELL.value: int(PermissionTier.USER_WRITE),
            ActionFamily.BROWSING.value: int(PermissionTier.USER_WRITE),
            ActionFamily.SYSTEM_CONTROL.value: int(PermissionTier.USER_WRITE),
            ActionFamily.OTHER.value: int(PermissionTier.SYSTEM_MODIFY),
        },
        allow_root=False,
    ),
    "gesture": SourceProfile(
        max_tier={
            ActionFamily.SHELL.value: int(PermissionTier.READ_ONLY),
            ActionFamily.BROWSING.value: int(PermissionTier.READ_ONLY),
            ActionFamily.SYSTEM_CONTROL.value: int(PermissionTier.SYSTEM_MODIFY),
            ActionFamily.OTHER.value: int(PermissionTier.USER_WRITE),
        },
        allow_root=False,
    ),
}

# Untagged call sites (the ~20 sub-agent paths not explicitly wired in this
# pass — see the plan's Context section) fail OPEN to "interactive" to
# preserve today's behavior, rather than fail closed and break them.
DEFAULT_UNKNOWN_SOURCE_PROFILE = "interactive"


class TaskScopeOverride(BaseModel):
    """Untrusted, caller-supplied restriction on top of a SourceProfile —
    e.g. an `autonomous_submit` RPC param. Pydantic validates the shape;
    `resolve_effective_profile()` guarantees it can only narrow the floor,
    never widen it, regardless of what values are supplied here."""

    max_tier: dict[str, int] | None = None
    deny_action_types: list[str] = Field(default_factory=list)
    allow_root: bool | None = None


def resolve_effective_profile(profile: SourceProfile, override: TaskScopeOverride | None) -> SourceProfile:
    """Combine a SourceProfile floor with an optional override, guaranteeing
    the result is never less restrictive than `profile` — per-family tiers
    take the MIN of floor and override, deny lists take the UNION, and
    allow_root is ANDed. A malicious override claiming a wider tier or
    root access is silently clamped back to the floor, not honored."""
    if override is None:
        return profile

    effective_tier = dict(profile.max_tier)
    if override.max_tier:
        for family, ceiling in override.max_tier.items():
            if family in effective_tier:
                effective_tier[family] = min(effective_tier[family], ceiling)

    effective_deny = sorted(set(profile.deny_action_types) | set(override.deny_action_types))

    effective_allow_root = profile.allow_root
    if override.allow_root is not None:
        effective_allow_root = effective_allow_root and override.allow_root

    return SourceProfile(
        max_tier=effective_tier,
        deny_action_types=effective_deny,
        allow_root=effective_allow_root,
    )


@dataclass
class GatewayDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    critic_verdict: dict[str, Any] | None = None


class AgentGateway:
    """Source-aware second gate, checked alongside `PermissionChecker`.

    Not a replacement for `PermissionChecker` — `authorize()` only ever
    tightens what a plan may do relative to the existing tier system; it
    never grants anything `PermissionChecker` would otherwise deny.
    """

    def __init__(
        self,
        config: PilotConfig,
        permissions: PermissionChecker,
        destructive_critic: DestructiveCriticAgent | None = None,
        audit_store: AgentGatewayAuditStore | None = None,
    ) -> None:
        self._config = config
        self._permissions = permissions
        self._destructive_critic = destructive_critic
        self._audit_store = audit_store

    def _profile_for(self, source: InvocationSource) -> SourceProfile:
        profiles = self._config.gateway.source_profiles
        key = source.value if source != InvocationSource.UNKNOWN else DEFAULT_UNKNOWN_SOURCE_PROFILE
        return profiles.get(key) or DEFAULT_SOURCE_PROFILES["interactive"]

    async def authorize(
        self,
        plan: ActionPlan,
        source: InvocationSource,
        scope_override: TaskScopeOverride | None = None,
        critic_already_reviewed: bool = False,
        plan_id: str = "",
    ) -> GatewayDecision:
        if not self._config.gateway.enabled:
            return GatewayDecision(allowed=True)

        profile = resolve_effective_profile(self._profile_for(source), scope_override)
        policy_snapshot = {
            "max_tier": profile.max_tier,
            "deny_action_types": profile.deny_action_types,
            "allow_root": profile.allow_root,
        }
        override_applied = scope_override is not None
        override_restricted = override_applied and profile != self._profile_for(source)

        reasons: list[str] = []
        for index, action in enumerate(plan.actions):
            family = action_family(action.action_type)
            tier = int(action.permission_tier)
            ceiling = profile.max_tier.get(family.value, int(PermissionTier.ROOT_CRITICAL))

            action_reason = ""
            if action.action_type.value in profile.deny_action_types:
                action_reason = f"{action.action_type.value} is denied for source '{source.value}' by gateway policy."
            elif tier > ceiling:
                action_reason = (
                    f"{action.action_type.value} (tier={PermissionTier(tier).name}, family={family.value}) "
                    f"exceeds the '{source.value}' source's allowed ceiling "
                    f"({PermissionTier(ceiling).name})."
                )
            elif action.permission_tier == PermissionTier.ROOT_CRITICAL and not profile.allow_root:
                action_reason = (
                    f"{action.action_type.value} requires root, which is denied for source '{source.value}'."
                )

            if action_reason:
                reasons.append(action_reason)

            await self._record_decision(
                plan_id=plan_id,
                action_index=index,
                action_type=action.action_type.value,
                action_family=family.value,
                target=action.target,
                source=source,
                permission_tier=PermissionTier(tier).name,
                override_applied=override_applied,
                override_restricted=override_restricted,
                decision="denied" if action_reason else "allowed",
                denial_reason=action_reason,
                policy_snapshot=policy_snapshot,
            )

        if reasons:
            return GatewayDecision(allowed=False, reasons=reasons)

        critic_verdict = await self._maybe_run_critic(plan, critic_already_reviewed)
        if critic_verdict is not None and critic_verdict.get("verdict") == "BLOCK":
            reason = f"Blocked by safety critic: {critic_verdict.get('recommendation', '')}"
            await self._record_decision(
                plan_id=plan_id,
                action_index=PLAN_LEVEL_AUDIT_INDEX,
                action_type="__critic_review__",
                action_family=ActionFamily.OTHER.value,
                target="",
                source=source,
                permission_tier="",
                override_applied=override_applied,
                override_restricted=override_restricted,
                decision="denied",
                denial_reason=reason,
                policy_snapshot=policy_snapshot,
            )
            return GatewayDecision(allowed=False, reasons=[reason], critic_verdict=critic_verdict)

        return GatewayDecision(allowed=True, critic_verdict=critic_verdict)

    async def _record_decision(
        self,
        *,
        plan_id: str,
        action_index: int,
        action_type: str,
        action_family: str,
        target: str,
        source: InvocationSource,
        permission_tier: str,
        override_applied: bool,
        override_restricted: bool,
        decision: str,
        denial_reason: str,
        policy_snapshot: dict[str, Any],
    ) -> None:
        if self._audit_store is None:
            return
        try:
            await self._audit_store.record_event(
                plan_id=plan_id,
                action_index=action_index,
                action_type=action_type,
                action_family=action_family,
                target=target,
                source_profile=source.value,
                permission_tier=permission_tier,
                override_applied=override_applied,
                override_restricted=override_restricted,
                decision=decision,
                denial_reason=denial_reason,
                dry_run=self._config.security.dry_run,
                policy_snapshot=policy_snapshot,
            )
        except Exception:
            logger.warning("Failed to record agent gateway audit event (non-fatal)", exc_info=True)

    async def _maybe_run_critic(self, plan: ActionPlan, critic_already_reviewed: bool) -> dict[str, Any] | None:
        """Re-implements server.py's interactive critic-trigger predicate so
        non-interactive plans (autonomous jobs, web-agent plans) get the same
        safety review interactive requests already receive, instead of
        silently skipping it because they never pass through server.py's
        confirmation-gate code path."""
        if critic_already_reviewed or self._destructive_critic is None:
            return None

        from pilot.agents.destructive_critic import HEURISTIC_RISK_THRESHOLD
        from pilot.agents.destructive_critic import risk_score as compute_risk_score

        plan_has_tier4 = any(a.permission_tier == PermissionTier.ROOT_CRITICAL for a in plan.actions)
        plan_has_tier3 = any(a.permission_tier == PermissionTier.DESTRUCTIVE for a in plan.actions)
        plan_has_irreversible = any(getattr(a, "is_irreversible", False) for a in plan.actions)

        needs_review = plan_has_tier4 or plan_has_tier3 or plan_has_irreversible
        if not needs_review:
            return None

        risk_score = compute_risk_score(plan, self._config) if (plan_has_tier3 or plan_has_irreversible) else 0.0
        if not (plan_has_tier4 or risk_score >= HEURISTIC_RISK_THRESHOLD):
            return None

        verdict = await self._destructive_critic.review(plan.raw_input, plan)
        return verdict.to_dict()
