"""Destructive Critic Agent â€” safety reviewer for Tier 4 (ROOT_CRITICAL) plans.

For any plan that contains Tier 4 destructive actions, the orchestration
pipeline runs a two-agent handshake:

  1. Planner              â†’ produces the ActionPlan as normal.
  2. DestructiveCriticAgent â†’ independently reviews the plan and returns a
     structured safety verdict BEFORE the user confirmation gate fires.

If the critic blocks the plan, execution is aborted immediately and the
reason is surfaced to the user. The critic never executes anything â€” it
only reads the plan and reasons about it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pilot.actions import PermissionTier

if TYPE_CHECKING:
    from pilot.actions import ActionPlan
    from pilot.config import PilotConfig
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.destructive_critic")

# Below this heuristic risk score, a Tier-3-only plan (no Tier 4, no flagged
# dangerous arguments) skips the LLM critic round-trip entirely and goes
# straight to the confirmation dialog — see heuristic_risk().
HEURISTIC_RISK_THRESHOLD = 0.3


def heuristic_risk(plan: ActionPlan) -> float:
    """Cheap, non-LLM risk estimate for a plan, used to decide whether a
    Tier-3-only (non-root) plan is worth an LLM critic round-trip.

    Tier 4 (ROOT_CRITICAL) plans always go through the LLM critic regardless
    of this score — it only gates the *additional* Tier 3 coverage so trivial
    single-file deletes don't all pay for a model call.
    """
    if not plan.actions:
        return 0.0

    score = 0.0

    if len(plan.actions) > 3:
        score += 0.2

    distinct_targets = {a.target for a in plan.actions if a.target}
    if len(distinct_targets) > 2:
        score += 0.2

    if any(getattr(a, "dangerous_flags", None) for a in plan.actions):
        score += 0.4

    if any(a.is_irreversible for a in plan.actions):
        score += 0.3

    tiers = {a.permission_tier for a in plan.actions}
    low_tiers = {PermissionTier.READ_ONLY, PermissionTier.USER_WRITE}
    high_tiers = {PermissionTier.DESTRUCTIVE, PermissionTier.ROOT_CRITICAL}
    if tiers & low_tiers and tiers & high_tiers:
        score += 0.1

    return min(1.0, score)


def risk_score(plan: ActionPlan, config: PilotConfig | None = None) -> float:
    """Decides whether a Tier-3-only/irreversible-only plan is worth the
    LLM critic round-trip — combines the cheap rule-based heuristic_risk()
    above with the Learned Risk Gate (pilot.security.risk_gate), when
    enabled, via max().

    max() rather than a blend/average is deliberate: the Learned Risk
    Gate's checks (predicted disk-usage exhaustion, fork-bomb-like
    process-count deltas, protected-path/package collisions) are things
    heuristic_risk() has NO visibility into at all — it only looks at
    plan length, distinct targets, dangerous_flags, and tier mixing. So
    for the common case (neither signal finds anything), max() just
    returns heuristic_risk() unchanged; the gate only ever adds a reason
    to escalate to critic review that heuristic_risk() alone would have
    missed, never removes one. This never affects Tier-4 plans, which
    always get critic review regardless of any risk score — see the
    `needs_review` predicate at both of this function's call sites
    (server.py, gateway.py's _maybe_run_critic).

    Falls back to heuristic_risk() unchanged if config is None, the gate
    is disabled, or no learned weights are staged — see RiskGate's own
    graceful-degradation philosophy.
    """
    base = heuristic_risk(plan)
    if config is None or not config.gateway.risk_gate_enabled:
        return base

    try:
        from pilot.security.risk_gate import get_risk_gate

        gate = get_risk_gate()
        learned_risk, _reasons = gate.evaluate_plan(plan, config)
    except Exception:
        logger.warning("Learned Risk Gate evaluation failed (non-fatal), using heuristic_risk() alone", exc_info=True)
        return base

    return max(base, learned_risk)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_PROMPT = """\
You are the Destructive Action Safety Critic for Heliox OS.
Your ONLY job is to review action plans that contain Tier 4 (ROOT_CRITICAL) or
Tier 3 (DESTRUCTIVE) operations and decide whether they are safe to proceed.

You are a second, independent agent. You did NOT create this plan. Approach it
with healthy skepticism.

Evaluate the plan against these five criteria:

1. INTENT ALIGNMENT  â€” Does every action directly serve the stated user request?
   Flag any action that seems unrelated or excessive.
2. BLAST RADIUS      â€” What is the worst-case irreversible damage if something
   goes wrong? (e.g. deleting /home vs deleting a temp file)
3. REVERSIBILITY     â€” Can the damage be undone without a snapshot?
4. PRIVILEGE CREEP   â€” Does the plan request root/elevated access beyond what is
   strictly necessary for the task?
5. INJECTION RISK    â€” Could any parameter (path, command, script) be exploited
   via path traversal, shell injection, or wildcard expansion?

Respond with ONLY a JSON object â€” no markdown, no prose outside the JSON:
{
  "verdict": "APPROVE" | "WARN" | "BLOCK",
  "risk_score": 0.0-1.0,
  "issues": ["list of specific concerns, empty if none"],
  "safe_actions": ["action_types that are fine"],
  "flagged_actions": ["action_types that are risky"],
  "recommendation": "One sentence summary for the user"
}

Verdict rules:
- APPROVE : Plan is safe to proceed (risk_score < 0.4, no critical issues).
- WARN    : Plan has concerns but can proceed with user awareness (0.4 <= risk_score < 0.75).
- BLOCK   : Plan must NOT execute â€” too dangerous or misaligned (risk_score >= 0.75).
"""

_CRITIC_USER_TEMPLATE = """\
User request: {user_input}

Planned actions ({action_count} total, max permission tier: {max_tier}):
{action_list}

Plan explanation: {explanation}
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CriticVerdict:
    """Structured output from the DestructiveCriticAgent."""

    verdict: str  # "APPROVE" | "WARN" | "BLOCK"
    risk_score: float
    issues: list[str] = field(default_factory=list)
    safe_actions: list[str] = field(default_factory=list)
    flagged_actions: list[str] = field(default_factory=list)
    recommendation: str = ""
    raw_response: str = ""

    @property
    def is_blocked(self) -> bool:
        """Return True when the critic has hard-blocked the plan."""
        return self.verdict == "BLOCK"

    @property
    def has_warnings(self) -> bool:
        """Return True when the critic approved with caveats."""
        return self.verdict == "WARN"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON broadcast."""
        return {
            "verdict": self.verdict,
            "risk_score": self.risk_score,
            "issues": self.issues,
            "safe_actions": self.safe_actions,
            "flagged_actions": self.flagged_actions,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DestructiveCriticAgent:
    """Secondary critic agent that reviews Tier 4 plans before execution.

    This agent is intentionally stateless and has no side effects â€” it only
    reads the plan and calls the LLM to produce a safety verdict.

    The critic is invoked by the server's execution pipeline whenever a plan
    contains at least one ROOT_CRITICAL (Tier 4) action.  It runs *before*
    the user confirmation gate so that a BLOCK verdict can abort the pipeline
    without ever prompting the user to approve something dangerous.

    Usage::

        critic = DestructiveCriticAgent(model_router)
        verdict = await critic.review(user_input, plan)
        if verdict.is_blocked:
            # abort â€” surface verdict.recommendation to the user
        elif verdict.has_warnings:
            # surface warnings alongside the normal confirmation prompt
    """

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router

    async def review(self, user_input: str, plan: ActionPlan) -> CriticVerdict:
        """Review a Tier 4 plan and return a structured safety verdict.

        Args:
            user_input: The original natural-language request from the user.
            plan: The ActionPlan produced by the Planner.

        Returns:
            A CriticVerdict with verdict, risk_score, issues, and recommendation.
        """
        action_list = self._format_action_list(plan)
        max_tier = plan.max_tier.name if plan.actions else "UNKNOWN"

        user_message = _CRITIC_USER_TEMPLATE.format(
            user_input=user_input,
            action_count=len(plan.actions),
            max_tier=max_tier,
            action_list=action_list,
            explanation=plan.explanation or "(no explanation provided)",
        )

        logger.info(
            "[DestructiveCritic] Reviewing plan â€” %d action(s), max_tier=%s",
            len(plan.actions),
            max_tier,
        )

        raw = ""
        try:
            raw = await self._model.generate(
                user_message,
                system_prompt=_CRITIC_SYSTEM_PROMPT,
            )
            verdict = self._parse_verdict(raw)
            logger.info(
                "[DestructiveCritic] verdict=%s risk=%.2f flagged=%s",
                verdict.verdict,
                verdict.risk_score,
                verdict.flagged_actions,
            )
            return verdict

        except Exception as exc:
            # Fail-safe: if the critic itself errors, emit WARN rather than
            # silently approving or hard-blocking â€” let the user decide.
            logger.warning("[DestructiveCritic] Review failed (%s), defaulting to WARN", exc)
            return CriticVerdict(
                verdict="WARN",
                risk_score=0.5,
                issues=[f"Critic agent encountered an error: {exc}"],
                recommendation="Critic review failed â€” proceed with extra caution.",
                raw_response=raw,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_action_list(self, plan: ActionPlan) -> str:
        """Render the plan's actions as a numbered list for the critic prompt.

        Only the fields most relevant to safety analysis are included to keep
        the prompt concise and focused.
        """
        lines: list[str] = []
        for i, action in enumerate(plan.actions, start=1):
            tier = action.permission_tier.name
            target = action.target or "(no target)"
            flags = ""
            if action.requires_root:
                flags += " [REQUIRES ROOT]"
            if action.destructive:
                flags += " [DESTRUCTIVE]"
            lines.append(f"  {i}. {action.action_type.value} | tier={tier} | target={target}{flags}")

            # Surface key parameters so the critic can reason about concrete values
            if action.parameters:
                try:
                    param_dict = action.parameters.model_dump(exclude_none=True)
                    # Only include fields that carry safety-relevant information
                    relevant = {
                        k: v
                        for k, v in param_dict.items()
                        if k
                        in (
                            "path",
                            "command",
                            "script",
                            "name",
                            "destination",
                            "recursive",
                            "elevated",
                            "force",
                        )
                        and v not in (None, "", [], {}, False)
                    }
                    if relevant:
                        lines.append(f"     params: {json.dumps(relevant)}")
                except Exception:
                    pass  # Parameter serialisation is best-effort

        return "\n".join(lines) if lines else "  (empty plan)"

    def _parse_verdict(self, raw: str) -> CriticVerdict:
        """Parse the LLM JSON response into a CriticVerdict.

        Handles markdown code fences that some models wrap around JSON output.
        """
        text = raw.strip()

        # Strip markdown code fences if present (e.g. ```json ... ```)
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        data: dict[str, Any] = json.loads(text)

        # Normalise verdict â€” default to WARN for any unexpected value
        verdict_str = str(data.get("verdict", "WARN")).upper()
        if verdict_str not in ("APPROVE", "WARN", "BLOCK"):
            verdict_str = "WARN"

        # Clamp risk_score to [0.0, 1.0]
        risk_score = float(data.get("risk_score", 0.5))
        risk_score = max(0.0, min(1.0, risk_score))

        return CriticVerdict(
            verdict=verdict_str,
            risk_score=risk_score,
            issues=list(data.get("issues", [])),
            safe_actions=list(data.get("safe_actions", [])),
            flagged_actions=list(data.get("flagged_actions", [])),
            recommendation=str(data.get("recommendation", "")),
            raw_response=raw,
        )
