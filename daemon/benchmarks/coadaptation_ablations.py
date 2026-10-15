"""Benchmark-only ablations of the constrained co-adaptation policy (spec section 18.2, policy 11).

Each subclass overrides exactly one of the two protected hooks of ``ConstrainedThompsonPolicy`` and is passed to
``CoAdaptationEngine(policy=...)`` by ``benchmarks.coadaptation_suite``. They exist only to measure what each
constraint costs on simulated users. Nothing under ``pilot/`` imports this module (a test checks), so no runtime path
can run an ablated policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pilot.coadaptation.core.constraints import ExclusionReason
from pilot.coadaptation.core.outcome_model import InteractionPrediction
from pilot.coadaptation.core.policy import ConstrainedThompsonPolicy
from pilot.coadaptation.core.vocabulary import DecisionLevel


class NoAmbiguityRulePolicy(ConstrainedThompsonPolicy):
    """The "no ambiguity rule" ablation: rule C2 at L2 is skipped, so learned L2 tables may give ``act`` mass."""

    def _ambiguity_constraint(
        self, candidates: Sequence[str], *, pin_act: bool
    ) -> tuple[list[str], dict[str, ExclusionReason]]:
        return list(candidates), {}


class NoSafetyFilterPolicy(ConstrainedThompsonPolicy):
    """The "no safety filter" ablation: the UCB screen of section 9.2 step 3 keeps every candidate."""

    def _safety_filter(
        self,
        level: DecisionLevel,
        candidates: Sequence[str],
        predictions: Mapping[str, InteractionPrediction],
        *,
        baseline_mean_ua: float,
        settings: Any,
    ) -> tuple[list[str], dict[str, ExclusionReason]]:
        return list(candidates), {}


ABLATIONS: Mapping[str, type[ConstrainedThompsonPolicy]] = {
    "no_ambiguity_rule": NoAmbiguityRulePolicy,
    "no_safety_filter": NoSafetyFilterPolicy,
}
