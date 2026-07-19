"""Learned Risk Gate — ties Layers 1/3/4/5 together for a whole plan.

Mirrors Ferrum-OS's cognitive/world_model/mod.rs's evaluate_action(), which
composes the encoder/transition/safety layers for one proposed action.
Heliox's RiskGate.evaluate_plan() does the plan-level equivalent: capture
one OS snapshot, run every action in the plan through
encode -> predict -> score, and take the worst (max) risk seen across the
plan — the same "one bad action anywhere in the plan is enough" principle
heuristic_risk() already uses for its own signals (dangerous_flags,
irreversibility), just extended to the predicted-outcome checks this
module adds.

Not implemented (a documented, deliberate Phase 1 scope boundary, same as
Ferrum's own staged Phase 1/2/3 rollout): Ferrum's mod.rs additionally
simulates *repeating* one action several times (MAX_LOOKAHEAD) to catch
risk that only compounds after several repetitions. Heliox's
heuristic_risk() already has its own (cruder) proxy for this — flagging
plans with >3 actions — so this isn't a total gap, but a real
self-composition lookahead would be a natural follow-up.

Strictly additive and optional: if no weights are staged, evaluate_plan()
still runs (using the rule-based transition fallback throughout), and if
config.gateway.risk_gate_enabled is False, callers shouldn't even call in
here — see destructive_critic.py's risk_score().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pilot.security.risk_model import RiskTransitionModel
from pilot.security.risk_observation import capture_os_snapshot
from pilot.security.risk_safety import score_outcome

if TYPE_CHECKING:
    from pilot.actions import ActionPlan
    from pilot.config import PilotConfig


class RiskGate:
    """Owns the (optional) learned transition model and evaluates whole
    plans against it. One instance is enough for the daemon's lifetime —
    see get_risk_gate() below; construct your own only in tests."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._transition = RiskTransitionModel(weights_path)

    @property
    def available(self) -> bool:
        """True once a learned transition model is actually staged — the
        rule-based fallback runs regardless, so this only reports whether
        predictions for LEARNABLE_ACTION_TYPES come from real training or
        the honest rule-table default."""
        return self._transition.is_loaded

    def evaluate_plan(self, plan: ActionPlan, config: PilotConfig) -> tuple[float, list[str]]:
        """Returns (worst risk in [0,1] seen across the plan's actions,
        the reasons that fired for that worst action — empty if none)."""
        if not plan.actions:
            return 0.0, []

        # One snapshot for the whole plan: this runs before execution
        # (predicting what WOULD happen), not interleaved with it, so OS
        # state isn't expected to shift meaningfully action-to-action —
        # and psutil calls aren't free, so one call beats N.
        snapshot = capture_os_snapshot()

        worst_risk = 0.0
        worst_reasons: list[str] = []
        for action in plan.actions:
            outcome = self._transition.predict(snapshot, action)
            risk, reasons = score_outcome(action, outcome, config)
            if risk > worst_risk:
                worst_risk = risk
                worst_reasons = reasons

        return worst_risk, worst_reasons


_gate: RiskGate | None = None


def get_risk_gate() -> RiskGate:
    """Lazily-constructed process-wide singleton — the learned weights (if
    any) only need loading once per daemon lifetime."""
    global _gate
    if _gate is None:
        _gate = RiskGate()
    return _gate
