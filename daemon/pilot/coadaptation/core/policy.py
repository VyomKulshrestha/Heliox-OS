"""Constrained Thompson policy (sections 9.2-9.5 of the co-adaptation spec).

Every decision follows the same procedure: candidates and applicability; hard constraints (C1-C8) before any model
output is read; the probabilistic safety filter in adaptive mode only; a fallback when nothing survives; then the
choice by mode with exact logged propensities over every option of the level (zeros included). Random numbers come
only from the decision's seed: ``rng_ts`` for Thompson sampling and ``rng_u`` for the selecting draw of an L1 or L3
decision and for the pre-drawn rows of an ambiguity table, so every draw can be re-derived from the logged seed.

Authority: every ``Decision`` and ``AmbiguityTable`` has ``authority = "presentation_only"`` and
``execution_authority = False`` and carries no plan id, action index, action type, coordinate command or callable.
At L2 no table outside a ``pin_act`` collapse gives ``act`` any mass (P3), so neither learning nor exploration can
commit an ambiguous literal hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, final

import numpy as np

from .constraints import (
    ConstraintState,
    ExclusionReason,
    Fallbacks,
    apply_hard_constraints,
    fixed_l2_rule,
    l2_applicable,
    pin_act_collapse,
)
from .explain import Explanation, build_explanation
from .preferences import PreferenceSet, caution_of
from .utility import UtilityWeights, l1_utility, l2_utility, l3_utility
from .vocabulary import L2_OPTIONS, L3_OPTIONS, VARIANTS, DecisionLevel, l1_arms

if TYPE_CHECKING:
    from .outcome_model import InteractionPrediction
    from .state import ActivationContext, LayoutSummary

POLICY_VERSION = "coadapt-policy-v1"
MAX_DRAWS_PER_TRIAL = 4
AFTER_MAX = "wait_for_evidence"
PRESENTATION_ONLY: Final = "presentation_only"
FIXTURE_SEED = 20261015
FIXTURE_CASES = 200

# Thompson samplers return ``outcome_model.OutcomeDraws``: ``.variants`` and ``.values[outcome]`` of shape
# ``(draws, len(variants))``. L1 outcomes: complete, ua, correction, veto (probabilities), log_time (log seconds),
# effort and intrusiveness (expected ratings); L2: the first five; L3: accepted and dismissed.
L1Sampler = Callable[[np.random.Generator, Sequence[str], int], Any]
CellSampler = Callable[[np.random.Generator, "ActivationContext", Sequence[str], int], Any]
CellPredictor = Callable[["ActivationContext", Sequence[str]], Mapping[str, "InteractionPrediction"]]
OfferSampler = Callable[[np.random.Generator, Sequence[str], int], Any]


class PolicyMode(StrEnum):
    RANDOMIZED = "randomized"
    ADAPTIVE = "adaptive"
    PROBE = "probe"
    COMPARATOR = "comparator"
    PINNED = "pinned"
    BASELINE = "baseline"


class NoAllowedRegime(Exception):
    """No L1 regime is allowed: ``fallbacks(modality).l1`` is None and nothing survived (section 9.2 step 4)."""


# ---------------------------------------------------------------------------------------------------- seeds and draws


def decision_seed(participant_key: str, seed_salt: str, decision_id: str) -> int:
    """``int.from_bytes(sha256(f"{participant_key}|{seed_salt}|{decision_id}|{POLICY_VERSION}")[:8], "big")``."""
    digest = hashlib.sha256(f"{participant_key}|{seed_salt}|{decision_id}|{POLICY_VERSION}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def seed_hex(seed: int) -> str:
    """The logged form of a seed: 16 hexadecimal characters."""
    return f"{seed:016x}"


def draw_stream(seed: int) -> np.random.Generator:
    """``rng_u``: ``Generator(PCG64(seed_u))`` with ``seed_u = int.from_bytes(sha256(f"{seed:016x}|u")[:8], "big")``."""
    seed_u = int.from_bytes(hashlib.sha256(f"{seed:016x}|u".encode()).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(seed_u))


def thompson_stream(seed: int) -> np.random.Generator:
    """``rng_ts = Generator(PCG64(seed))``, used only for Thompson sampling."""
    return np.random.Generator(np.random.PCG64(seed))


def select_from_cell(order: Sequence[str], propensities: Mapping[str, float], u: float) -> str:
    """Return the first variant in ``order`` whose cumulative propensity exceeds ``u``; if rounding leaves none,
    return the last variant in ``order`` with positive propensity."""
    cumulative = 0.0
    for variant in order:
        cumulative += float(propensities.get(variant, 0.0))
        if cumulative > u:
            return variant
    for variant in reversed(order):
        if float(propensities.get(variant, 0.0)) > 0.0:
            return variant
    raise ValueError("no variant has positive propensity")


# -------------------------------------------------------------------------------------------------- decision types


def _prediction_dict(prediction: Any) -> dict[str, Any]:
    """PredictionV1 (section 14.3) from an ``InteractionPrediction``."""

    def estimate(value: Any) -> dict[str, float]:
        return {"mean": float(value.mean), "lo90": float(value.lo90), "hi90": float(value.hi90)}

    evidence = prediction.evidence
    return {
        "variant": prediction.variant,
        "p_complete": estimate(prediction.p_complete),
        "p_unintended": estimate(prediction.p_unintended),
        "ucb_unintended": float(prediction.ucb_unintended),
        "p_correction": estimate(prediction.p_correction),
        "p_veto": estimate(prediction.p_veto),
        "time_ms": estimate(prediction.time_ms),
        "burden": None if prediction.burden is None else {key: float(v) for key, v in prediction.burden.items()},
        "evidence": {
            "trials_person": int(evidence.trials_person),
            "trials_today": int(evidence.trials_today),
            "source": str(evidence.source),
        },
        "authority": PRESENTATION_ONLY,
    }


def _params(variant: str) -> dict[str, Any]:
    return {key: value for key, value in VARIANTS[variant].params}


def _versions(versions: Mapping[str, str]) -> dict[str, str]:
    return {key: str(value) for key, value in versions.items()}


@dataclass(frozen=True, slots=True)
class Decision:
    """An L1 regime or L3 offer decision (RecommendationV1). Presentation only."""

    decision_id: str
    participant_key: str
    level: DecisionLevel
    mode: PolicyMode
    variant: str
    options: tuple[str, ...]  # every option of the level for the modality, in caution / table order
    eligible: tuple[str, ...]
    excluded: dict[str, ExclusionReason]
    propensities: dict[str, float]  # over `options`, zeros included
    p_thompson: dict[str, float] | None
    exploration: float
    fallback: bool
    protocol_deviation: bool
    round_index: int | None
    spread: tuple[float, float] | None  # (r, floor) the UI uses for every posterior of the block; L1 only
    priorities: tuple[float, float, float] | None  # (fewer_mistakes, faster, fewer_interruptions); L1 only
    predictions: dict[str, Any]  # variant -> InteractionPrediction
    explanation: Explanation
    seed: int
    u: float  # the selecting draw from rng_u (not serialized; re-derived from the seed)
    versions: dict[str, str]
    authority: Literal["presentation_only"] = "presentation_only"
    execution_authority: Literal[False] = False

    def __post_init__(self) -> None:
        if self.authority != PRESENTATION_ONLY or self.execution_authority is not False:
            raise ValueError("decisions are presentation-only")

    @property
    def propensity(self) -> float:
        return self.propensities[self.variant]

    def to_dict(self) -> dict[str, Any]:
        """RecommendationV1 (section 14.3); floats are never rounded."""
        return {
            "decision_id": self.decision_id,
            "level": DecisionLevel(self.level).value,
            "mode": PolicyMode(self.mode).value,
            "variant": self.variant,
            "params": _params(self.variant),
            "eligible": list(self.eligible),
            "excluded": {variant: ExclusionReason(reason).value for variant, reason in self.excluded.items()},
            "propensities": {variant: float(self.propensities[variant]) for variant in self.options},
            "p_thompson": None
            if self.p_thompson is None
            else {variant: float(self.p_thompson.get(variant, 0.0)) for variant in self.options},
            "exploration": float(self.exploration),
            "fallback": self.fallback,
            "protocol_deviation": self.protocol_deviation,
            "round_index": self.round_index,
            "spread": None if self.spread is None else {"r": float(self.spread[0]), "floor": float(self.spread[1])},
            "priorities": None
            if self.priorities is None
            else {
                "fewer_mistakes": float(self.priorities[0]),
                "faster": float(self.priorities[1]),
                "fewer_interruptions": float(self.priorities[2]),
            },
            "predictions": {variant: _prediction_dict(p) for variant, p in self.predictions.items()},
            "explanation": self.explanation.to_dict(),
            "seed": seed_hex(self.seed),
            "versions": _versions(self.versions),
            "authority": PRESENTATION_ONLY,
            "execution_authority": False,
        }


@dataclass(frozen=True, slots=True)
class TableCell:
    """One cell of an ambiguity table: the responses and their propensities for one kind of ambiguous activation."""

    cell_id: str
    bin: str
    n_plausible: int
    has_hit: bool
    order: tuple[str, ...]  # always L2_OPTIONS
    propensities: dict[str, float]  # over L2_OPTIONS, zeros included
    eligible: tuple[str, ...]
    excluded: dict[str, ExclusionReason]
    p_thompson: dict[str, float] | None = None
    pinned: bool = False  # the pin_act collapse

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "bin": self.bin,
            "n_plausible": self.n_plausible,
            "has_hit": self.has_hit,
            "order": list(self.order),
            "propensities": {variant: float(self.propensities[variant]) for variant in self.order},
            "eligible": list(self.eligible),
            "excluded": {variant: ExclusionReason(reason).value for variant, reason in self.excluded.items()},
        }


@dataclass(frozen=True, slots=True)
class AmbiguityTable:
    """A per-block L2 table with pre-drawn uniforms (AmbiguityTableV1). Presentation only."""

    decision_id: str
    participant_key: str
    block_decision_id: str
    mode: PolicyMode
    ambiguity_threshold: float
    bins: tuple[tuple[str, float, float], ...]
    cells: tuple[TableCell, ...]
    start_trial_index: int
    draws: tuple[tuple[float, ...], ...]  # u[k][j]: row k is trial start_trial_index + k
    explanation: Explanation
    seed: int
    versions: dict[str, str]
    pin_act: bool = False
    max_draws_per_trial: int = MAX_DRAWS_PER_TRIAL
    after_max: str = AFTER_MAX
    authority: Literal["presentation_only"] = "presentation_only"
    execution_authority: Literal[False] = False

    def __post_init__(self) -> None:
        if self.authority != PRESENTATION_ONLY or self.execution_authority is not False:
            raise ValueError("tables are presentation-only")

    @property
    def level(self) -> DecisionLevel:
        return DecisionLevel.AMBIGUITY_RESPONSE

    def cell(self, cell_id: str) -> TableCell | None:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None

    def draw(self, trial_index: int, draw_index: int) -> float | None:
        """``u[trial_index - start_trial_index][draw_index]``, or None outside the table."""
        row = trial_index - self.start_trial_index
        if not 0 <= row < len(self.draws) or not 0 <= draw_index < self.max_draws_per_trial:
            return None
        return self.draws[row][draw_index]

    def variant_for(self, cell_id: str, trial_index: int, draw_index: int) -> str | None:
        """The response the UI must apply: ``select_from_cell`` on the drawn uniform, ``wait_for_evidence`` after
        the maximum number of draws, None when the cell or the row does not exist."""
        cell = self.cell(cell_id)
        if cell is None:
            return None
        if draw_index >= self.max_draws_per_trial:
            return self.after_max
        u = self.draw(trial_index, draw_index)
        if u is None:
            return None
        return select_from_cell(cell.order, cell.propensities, u)

    def to_dict(self) -> dict[str, Any]:
        """AmbiguityTableV1 (section 14.3); the RPC level name is ``ambiguity_table``."""
        return {
            "decision_id": self.decision_id,
            "level": "ambiguity_table",
            "block_decision_id": self.block_decision_id,
            "mode": PolicyMode(self.mode).value,
            "ambiguity_threshold": float(self.ambiguity_threshold),
            "bins": [{"id": bin_id, "lo": float(lo), "hi": float(hi)} for bin_id, lo, hi in self.bins],
            "cells": [cell.to_dict() for cell in self.cells],
            "start_trial_index": self.start_trial_index,
            "draws": [[float(u) for u in row] for row in self.draws],
            "max_draws_per_trial": self.max_draws_per_trial,
            "after_max": self.after_max,
            "explanation": self.explanation.to_dict(),
            "seed": seed_hex(self.seed),
            "versions": _versions(self.versions),
            "authority": PRESENTATION_ONLY,
            "execution_authority": False,
        }


# ----------------------------------------------------------------------------------------------------- requests


@dataclass(frozen=True, slots=True)
class PolicySettings:
    """The settings a decision reads (an ``EngineSettings`` satisfies this structurally)."""

    ambiguity_threshold: float = 0.25
    unintended_activation_tolerance: float = 0.05
    noninferiority_margin: float = 0.01
    upper_bound_credibility: float = 0.95
    exploration_floor: float = 0.1
    switching_penalty: float = 0.1
    thompson_draws: int = 256


@dataclass(frozen=True, slots=True)
class BlockRequest:
    """Everything an L1 decision reads. The engine builds it from session state and non-mutating predictions."""

    decision_id: str
    participant_key: str
    seed: int
    mode: PolicyMode
    modality: str
    preferences: PreferenceSet
    state: ConstraintState
    fallbacks: Fallbacks
    settings: Any
    predictions: Mapping[str, Any]  # variant -> InteractionPrediction for every L1 arm of the modality
    sampler: L1Sampler | None
    previous_regime: str | None
    priorities: tuple[float, float, float]
    t_ref_s: float
    spread: tuple[float, float] | None
    round_remaining: frozenset[str] | None  # randomized mode: arms not yet used in the round (None = new round)
    round_index: int
    comparator_ranking: tuple[str, ...]
    protocol_deviation: bool
    versions: Mapping[str, str]
    context: str = "study"


@dataclass(frozen=True, slots=True)
class CellSpec:
    """A reachable table cell with its representative activation context (section 9.3)."""

    cell_id: str
    bin: str
    n_plausible: int
    has_hit: bool
    context: Any  # ActivationContext


@dataclass(frozen=True, slots=True)
class TableRequest:
    decision_id: str
    participant_key: str
    block_decision_id: str
    seed: int
    mode: PolicyMode
    modality: str
    preferences: PreferenceSet
    state: ConstraintState
    fallbacks: Fallbacks
    settings: Any
    cells: tuple[CellSpec, ...]
    predict_cell: CellPredictor
    sample_cell: CellSampler | None
    priorities: tuple[float, float, float]
    interruptions_used: int
    start_trial_index: int
    trials_in_block: int
    versions: Mapping[str, str]
    context: str = "study"


@dataclass(frozen=True, slots=True)
class OfferRequest:
    decision_id: str
    participant_key: str
    seed: int
    mode: PolicyMode
    modality: str
    keyboard_available: bool
    preferences: PreferenceSet
    state: ConstraintState
    settings: Any
    sampler: OfferSampler | None
    priorities: tuple[float, float, float]
    versions: Mapping[str, str]
    predictions: Mapping[str, Any] = field(default_factory=dict)
    context: str = "study"


# ----------------------------------------------------------------------------------------------------- helpers


def bin_bounds(tau: float) -> tuple[tuple[str, float, float], ...]:
    """The session's bins ``(id, lo, hi)`` with ``lo < A_gate <= hi`` (section 6.3); empty bins are left out."""
    bounds = (("b1", tau, max(tau, 0.4)), ("b2", max(tau, 0.4), max(tau, 0.6)), ("b3", max(tau, 0.6), 1.0))
    return tuple(bound for bound in bounds if bound[2] > bound[1])


def representative_ambiguity(bin_id: str, tau: float) -> float:
    """The bin midpoint ``A_mid`` of section 9.3."""
    if bin_id == "b1":
        return (tau + max(tau, 0.4)) / 2.0
    if bin_id == "b2":
        return (max(tau, 0.4) + max(tau, 0.6)) / 2.0
    return (max(tau, 0.6) + 1.0) / 2.0


def reachable_cells(tau: float) -> tuple[tuple[str, int, bool], ...]:
    """(bin, n_plausible, has_hit) of every cell that can occur given tau; ``h0`` occurs only in b3."""
    cells = []
    for bin_id, _, _ in bin_bounds(tau):
        for n_plausible in (1, 2, 3):
            hits = (False, True) if bin_id == "b3" else (True,)
            for has_hit in hits:
                cells.append((bin_id, n_plausible, has_hit))
    return tuple(cells)


def cell_specs(tau: float, layout: LayoutSummary, modality: str) -> tuple[CellSpec, ...]:
    """Representative activation contexts of every reachable cell (section 9.3): bin midpoint ambiguity,
    ``p_hit = 1 - A_mid`` with a hit (0 without), ``p_top = max(p_hit, 1/n_plausible)``, the block's first layout."""
    from .state import ActivationContext

    specs = []
    for bin_id, n_plausible, has_hit in reachable_cells(tau):
        a_mid = representative_ambiguity(bin_id, tau)
        p_hit = 1.0 - a_mid if has_hit else 0.0
        context = ActivationContext(
            ambiguity=a_mid,
            bin=bin_id,  # type: ignore[arg-type]
            n_plausible=n_plausible,
            has_hit=has_hit,
            p_hit=p_hit,
            p_top=max(p_hit, 1.0 / n_plausible),
            modality=modality,  # type: ignore[arg-type]
            layout=layout,
        )
        cell = f"{bin_id}:p{n_plausible}:h{1 if has_hit else 0}"
        specs.append(CellSpec(cell_id=cell, bin=bin_id, n_plausible=n_plausible, has_hit=has_hit, context=context))
    return tuple(specs)


def _normalise(probabilities: Mapping[str, float], options: Sequence[str]) -> dict[str, float]:
    total = sum(probabilities.get(variant, 0.0) for variant in options)
    if total <= 0:
        raise ValueError("propensities must have positive mass")
    return {variant: float(probabilities.get(variant, 0.0)) / total for variant in options}


def _point(variant: str, options: Sequence[str]) -> dict[str, float]:
    return {option: 1.0 if option == variant else 0.0 for option in options}


def _uniform(support: Sequence[str], options: Sequence[str]) -> dict[str, float]:
    share = 1.0 / len(support)
    return {option: share if option in support else 0.0 for option in options}


def _draw_column(sampled: Any, outcome: str, variant: str, draws: int) -> np.ndarray:
    """One variant's draws of one outcome from an ``OutcomeDraws`` (shape ``(draws,)``)."""
    column = list(sampled.variants).index(variant)
    return np.broadcast_to(np.asarray(sampled.values[outcome], dtype=np.float64)[:, column], (draws,))


def argmax_frequency(utilities: np.ndarray) -> np.ndarray:
    """Share of draws (rows) in which each column is the argmax; ties split evenly."""
    # Column by column: numpy reduces a short row axis slowly, and Thompson sampling calls this for every cell.
    best = utilities[:, 0].copy()
    for column in range(1, utilities.shape[1]):
        np.maximum(best, utilities[:, column], out=best)
    ties = utilities == best[:, None]
    counts = ties.sum(axis=1)
    if np.all(counts == 1):  # no tie: dividing by the counts would change nothing
        return ties.mean(axis=0)
    return (ties / counts[:, None]).mean(axis=0)


def thompson_mixture(p_ts: Mapping[str, float], support: Sequence[str], epsilon: float) -> dict[str, float]:
    """``pi(v) = (1 - eps) p_TS(v) + eps / |S|`` on the safe set."""
    share = epsilon / len(support)
    return {variant: (1.0 - epsilon) * float(p_ts.get(variant, 0.0)) + share for variant in support}


# ------------------------------------------------------------------------------------------------------- policy


class ConstrainedThompsonPolicy:
    """Constrained Thompson sampling over the safe set, with an exploration floor (section 9.2).

    ``_ambiguity_constraint`` (rule C2 at L2) and ``_safety_filter`` (section 9.2 step 3) are the only methods a
    subclass may override: the benchmark ablations "no ambiguity rule" and "no safety filter" replace them to measure
    the price of each constraint. Every other method is final. Hard constraints C1 and C5-C8, the fallbacks, the
    modes and the propensities do not depend on these hooks.
    """

    def __init__(self) -> None:
        self._name = type(self).__name__

    # -------------------------------------------------------------------------------------------------- hooks

    def _ambiguity_constraint(
        self, candidates: Sequence[str], *, pin_act: bool
    ) -> tuple[list[str], dict[str, ExclusionReason]]:
        """Rule C2 at L2, called for every table cell in every mode (``fallbacks().l2`` never contains ``act``
        outside pin_act, section 9.2 step 4): removes ``act`` with reason ambiguous_activation unless the pin_act
        collapse applies. The 'no ambiguity rule' ablation returns the candidates unchanged and no exclusions."""
        if pin_act:
            return list(candidates), {}
        kept = [variant for variant in candidates if variant != "act"]
        excluded = {"act": ExclusionReason.AMBIGUOUS_ACTIVATION} if "act" in candidates else {}
        return kept, excluded

    def _safety_filter(
        self,
        level: DecisionLevel,
        candidates: Sequence[str],
        predictions: Mapping[str, InteractionPrediction],
        *,
        baseline_mean_ua: float,
        settings: Any,
    ) -> tuple[list[str], dict[str, ExclusionReason]]:
        """Section 9.2 step 3, called in adaptive mode only, with the exemptions listed there (reason
        slip_risk_above_limit). The 'no safety filter' ablation returns the candidates unchanged.

        Removes ``v`` when ``UCB_c[p_ua(v)] > max(eps_user, mean[p_ua(baseline)] + delta_ni)``. A heuristic screen
        on model predictions with no error-rate guarantee. Exempt: ``wait_for_evidence``, ``no_offer`` and offers.
        """
        limit = max(
            float(settings.unintended_activation_tolerance), baseline_mean_ua + float(settings.noninferiority_margin)
        )
        kept: list[str] = []
        excluded: dict[str, ExclusionReason] = {}
        for variant in candidates:
            if variant == "wait_for_evidence" or level == DecisionLevel.OFFER:
                kept.append(variant)
                continue
            prediction = predictions.get(variant)
            if prediction is not None and float(prediction.ucb_unintended) > limit:
                excluded[variant] = ExclusionReason.SLIP_RISK_ABOVE_LIMIT
            else:
                kept.append(variant)
        return kept, excluded

    # ------------------------------------------------------------------------------------------------------ L1

    @final
    def _enforces_c2(self) -> bool:
        """True unless a subclass replaced the C2 hook (the benchmark's 'no ambiguity rule' ablation)."""
        return type(self)._ambiguity_constraint is ConstrainedThompsonPolicy._ambiguity_constraint

    @final
    def block_safe_set(self, request: BlockRequest) -> tuple[list[str], dict[str, ExclusionReason]]:
        """Steps 1-3 of an adaptive L1 decision: arms, C1, C2, C5, C6, C7 and the safety filter."""
        arms = list(l1_arms(request.modality))
        safe, excluded = apply_hard_constraints(
            DecisionLevel.BLOCK_REGIME,
            arms,
            preferences=request.preferences,
            state=request.state,
            modality=request.modality,
            participant_key=request.participant_key,
            adaptive=True,
            context=request.context,
        )
        baseline = request.predictions.get("assistance_off")
        baseline_ua = float(baseline.p_unintended.mean) if baseline is not None else 0.0
        safe, filtered = self._safety_filter(
            DecisionLevel.BLOCK_REGIME,
            safe,
            request.predictions,
            baseline_mean_ua=baseline_ua,
            settings=request.settings,
        )
        excluded.update(filtered)
        return safe, excluded

    @final
    def adaptive_l1_probabilities(
        self,
        request: BlockRequest,
        safe: Sequence[str],
        rng: np.random.Generator,
        draws: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """(pi, p_TS) over the safe set by Thompson sampling of the L1 utility (section 9.1) plus the floor."""
        weights = UtilityWeights.from_priorities(*request.priorities)
        if request.sampler is None or len(safe) == 1:
            p_ts = {variant: 1.0 / len(safe) for variant in safe}
        else:
            sampled = request.sampler(rng, list(safe), draws)
            columns = []
            for variant in safe:

                def draw(outcome: str, variant: str = variant) -> np.ndarray:
                    return _draw_column(sampled, outcome, variant, draws)

                columns.append(
                    l1_utility(
                        variant,
                        p_complete=draw("complete"),
                        p_ua=draw("ua"),
                        p_correction=draw("correction"),
                        p_veto=draw("veto"),
                        log_time_s=draw("log_time"),
                        effort=draw("effort"),
                        intrusiveness=draw("intrusiveness"),
                        weights=weights,
                        t_ref_s=request.t_ref_s,
                        previous_regime=request.previous_regime,
                        switching_penalty=float(request.settings.switching_penalty),
                    )
                )
            frequency = argmax_frequency(np.column_stack([np.broadcast_to(c, (draws,)) for c in columns]))
            p_ts = {variant: float(frequency[i]) for i, variant in enumerate(safe)}
        pi = thompson_mixture(p_ts, safe, float(request.settings.exploration_floor))
        return pi, p_ts

    @final
    def decide_block(self, request: BlockRequest) -> Decision:
        """An L1 regime decision in the request's mode (section 9.2 steps 1-6)."""
        options = tuple(l1_arms(request.modality))
        mode = PolicyMode(request.mode)
        rng_u = draw_stream(request.seed)
        excluded: dict[str, ExclusionReason] = {}
        p_ts: dict[str, float] | None = None
        exploration = 0.0
        fallback = False
        deviation = request.protocol_deviation
        round_index: int | None = None
        held = request.preferences.held_regime(
            context=request.context, modality=request.modality, participant_key=request.participant_key
        )

        if mode is PolicyMode.PINNED:
            if held is None:
                raise ValueError("pinned mode needs a freeze or pin")
            chosen_support = [held]
            for variant in options:
                if variant != held:
                    excluded[variant] = ExclusionReason.PINNED_OTHER
            pi = _point(held, options)
        elif mode is PolicyMode.BASELINE:
            practice = request.fallbacks.practice
            if practice is None:
                raise NoAllowedRegime("no practice regime is allowed")
            chosen_support = [practice]
            pi = _point(practice, options)
            fallback = True
        else:
            adaptive = mode is PolicyMode.ADAPTIVE
            if adaptive:
                safe, excluded = self.block_safe_set(request)
            else:
                safe, excluded = apply_hard_constraints(
                    DecisionLevel.BLOCK_REGIME,
                    list(options),
                    preferences=request.preferences,
                    state=request.state,
                    modality=request.modality,
                    participant_key=request.participant_key,
                    adaptive=False,
                    context=request.context,
                )
            pi, chosen_support = {}, []
            if mode is PolicyMode.RANDOMIZED:
                round_index = request.round_index
                remaining = request.round_remaining
                if remaining is None or not (set(remaining) & set(safe)):
                    remaining = frozenset(options)
                    round_index = request.round_index + (0 if request.round_remaining is None else 1)
                support = [variant for variant in options if variant in safe and variant in remaining]
                for variant in options:
                    if variant in safe and variant not in remaining:
                        excluded[variant] = ExclusionReason.USED_THIS_ROUND
                if support:
                    chosen_support = support
                    pi = _uniform(support, options)
            elif mode is PolicyMode.PROBE:
                if "assistance_off" in safe:
                    chosen_support = ["assistance_off"]
                elif safe:
                    chosen_support = [min(safe, key=caution_of)]
                    deviation = True
                for variant in safe:
                    if variant not in chosen_support:
                        excluded[variant] = ExclusionReason.PROBE_SCHEDULE
                if chosen_support:
                    pi = _point(chosen_support[0], options)
            elif mode is PolicyMode.COMPARATOR:
                ranking = [variant for variant in request.comparator_ranking if variant in options]
                allowed = [variant for variant in ranking if variant in safe]
                if allowed:
                    chosen_support = [allowed[0]]
                    if not ranking or allowed[0] != ranking[0]:
                        deviation = True
                    for variant in safe:
                        if variant != allowed[0]:
                            excluded[variant] = ExclusionReason.COMPARATOR_SCHEDULE
                    pi = _point(allowed[0], options)
                else:
                    deviation = True
            elif mode is PolicyMode.ADAPTIVE:
                if safe:
                    draws = max(1, int(request.settings.thompson_draws))
                    pi_safe, p_ts = self.adaptive_l1_probabilities(request, safe, thompson_stream(request.seed), draws)
                    exploration = float(request.settings.exploration_floor)
                    chosen_support = list(safe)
                    pi = {variant: pi_safe.get(variant, 0.0) for variant in options}
            if not chosen_support:
                fallback_regime = request.fallbacks.l1
                if fallback_regime is None:
                    raise NoAllowedRegime("no L1 regime is allowed")
                fallback = True
                chosen_support = [fallback_regime]
                pi = _point(fallback_regime, options)
                excluded.pop(fallback_regime, None)

        pi = _normalise(pi, options)
        u = float(rng_u.random())
        chosen = select_from_cell(options, pi, u)
        eligible = tuple(variant for variant in options if variant in chosen_support)
        # arms a fixed-regime block leaves out without another reason: the schedule (or the pin) decided
        schedule = {
            PolicyMode.PINNED: ExclusionReason.PINNED_OTHER,
            PolicyMode.COMPARATOR: ExclusionReason.COMPARATOR_SCHEDULE,
        }.get(mode, ExclusionReason.PROBE_SCHEDULE)
        for variant in options:
            if variant not in eligible and variant not in excluded:
                excluded[variant] = schedule
        _check_decision(pi, chosen, eligible, excluded)
        shown = {
            variant: request.predictions[variant]
            for variant in options
            if variant in request.predictions and (variant in eligible or variant == chosen)
        }
        explain_with = dict(shown)
        if "assistance_off" in request.predictions:
            explain_with.setdefault("assistance_off", request.predictions["assistance_off"])
        explanation = build_explanation(
            mode=mode.value,
            chosen=chosen,
            predictions=explain_with,
            excluded=excluded,
            baseline="assistance_off",
            fallback=fallback,
        )
        return Decision(
            decision_id=request.decision_id,
            participant_key=request.participant_key,
            level=DecisionLevel.BLOCK_REGIME,
            mode=mode,
            variant=chosen,
            options=options,
            eligible=eligible,
            excluded={variant: excluded[variant] for variant in options if variant in excluded},
            propensities=pi,
            p_thompson=None if p_ts is None else {variant: p_ts.get(variant, 0.0) for variant in options},
            exploration=exploration,
            fallback=fallback,
            protocol_deviation=deviation,
            round_index=round_index,
            spread=request.spread,
            priorities=tuple(request.priorities),  # type: ignore[arg-type]
            predictions=shown,
            explanation=explanation,
            seed=request.seed,
            u=u,
            versions=dict(request.versions),
        )

    # ------------------------------------------------------------------------------------------------------ L2

    @final
    def cell_safe_set(
        self, request: TableRequest, spec: CellSpec, predictions: Mapping[str, Any], *, adaptive: bool
    ) -> tuple[list[str], dict[str, ExclusionReason], bool]:
        """Candidates and applicability, C2 (the hook), C1 and C5 and, in adaptive mode, the safety filter for one
        cell. Returns (safe set, exclusions, pin_act collapse). C2 runs before the other rules so that ``act`` is
        logged with ``ambiguous_activation`` in every table outside a pin_act collapse (section 14.3)."""
        candidates = []
        excluded: dict[str, ExclusionReason] = {}
        for variant in L2_OPTIONS:
            if l2_applicable(variant, n_plausible=spec.n_plausible, has_hit=spec.has_hit):
                candidates.append(variant)
            else:
                excluded[variant] = ExclusionReason.NOT_APPLICABLE
        pin_act = request.preferences.pin_act(
            context=request.context, modality=request.modality, participant_key=request.participant_key
        )
        if pin_act:
            safe, collapsed = pin_act_collapse(candidates)
            excluded.update(collapsed)
            return safe, excluded, True
        candidates, a_excluded = self._ambiguity_constraint(candidates, pin_act=False)
        excluded.update(a_excluded)
        safe, c_excluded = apply_hard_constraints(
            DecisionLevel.AMBIGUITY_RESPONSE,
            candidates,
            preferences=request.preferences,
            state=request.state,
            modality=request.modality,
            participant_key=request.participant_key,
            adaptive=adaptive,
            interruptions_used=request.interruptions_used,
            context=request.context,
        )
        excluded.update(c_excluded)
        if adaptive:
            baseline = predictions.get("act")
            baseline_ua = float(baseline.p_unintended.mean) if baseline is not None else 0.0
            safe, s_excluded = self._safety_filter(
                DecisionLevel.AMBIGUITY_RESPONSE,
                safe,
                predictions,
                baseline_mean_ua=baseline_ua,
                settings=request.settings,
            )
            excluded.update(s_excluded)
        return safe, excluded, False

    @final
    def adaptive_cell_probabilities(
        self,
        request: TableRequest,
        spec: CellSpec,
        safe: Sequence[str],
        rng: np.random.Generator,
        draws: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        weights = UtilityWeights.from_priorities(*request.priorities)
        if request.sample_cell is None or len(safe) == 1:
            p_ts = {variant: 1.0 / len(safe) for variant in safe}
        else:
            sampled = request.sample_cell(rng, spec.context, list(safe), draws)
            columns = []
            for variant in safe:

                def draw(outcome: str, variant: str = variant) -> np.ndarray:
                    return _draw_column(sampled, outcome, variant, draws)

                columns.append(
                    l2_utility(
                        variant,
                        p_complete=draw("complete"),
                        p_ua=draw("ua"),
                        p_correction=draw("correction"),
                        p_veto=draw("veto"),
                        log_time_s=draw("log_time"),
                        weights=weights,
                    )
                )
            frequency = argmax_frequency(np.column_stack([np.broadcast_to(c, (draws,)) for c in columns]))
            p_ts = {variant: float(frequency[i]) for i, variant in enumerate(safe)}
        return thompson_mixture(p_ts, safe, float(request.settings.exploration_floor)), p_ts

    @final
    def decide_table(self, request: TableRequest) -> AmbiguityTable:
        """An ambiguity table for the block's remaining trials (section 9.3)."""
        mode = PolicyMode(request.mode)
        rng_ts = thompson_stream(request.seed)
        draws_ts = max(1, int(request.settings.thompson_draws))
        cells: list[TableCell] = []
        any_pin_act = False
        explain_cell: tuple[CellSpec, Mapping[str, Any], str] | None = None
        for spec in request.cells:
            adaptive = mode is PolicyMode.ADAPTIVE
            predictions: Mapping[str, Any] = (
                request.predict_cell(spec.context, list(L2_OPTIONS)) if adaptive or explain_cell is None else {}
            )
            safe, excluded, pin_act = self.cell_safe_set(request, spec, predictions, adaptive=adaptive)
            any_pin_act = any_pin_act or pin_act
            p_ts: dict[str, float] | None = None
            if pin_act:
                propensities = _point(safe[0], L2_OPTIONS)
                eligible = list(safe)
            elif mode in (PolicyMode.PROBE, PolicyMode.PINNED, PolicyMode.BASELINE):
                # the fixed L2 rule: the first entry of fallbacks.l2 applicable to the cell (and still in its safe
                # set); the safe set stays the cell's eligible set, so wait_for_evidence is always in it (P4)
                rule = next(
                    (variant for variant in request.fallbacks.l2 if variant in safe),
                    "wait_for_evidence",
                )
                propensities = _point(rule, L2_OPTIONS)
                eligible = list(safe)
            elif mode is PolicyMode.ADAPTIVE:
                pi_safe, p_ts = self.adaptive_cell_probabilities(request, spec, safe, rng_ts, draws_ts)
                propensities = {variant: pi_safe.get(variant, 0.0) for variant in L2_OPTIONS}
                eligible = list(safe)
            else:  # randomized and comparator: uniform over the cell's safe set
                propensities = _uniform(safe, L2_OPTIONS)
                eligible = list(safe)
            propensities = _normalise(propensities, L2_OPTIONS)
            _check_cell(propensities, eligible, pin_act=pin_act, has_hit=spec.has_hit, enforce_c2=self._enforces_c2())
            cell = TableCell(
                cell_id=spec.cell_id,
                bin=spec.bin,
                n_plausible=spec.n_plausible,
                has_hit=spec.has_hit,
                order=tuple(L2_OPTIONS),
                propensities=propensities,
                eligible=tuple(variant for variant in L2_OPTIONS if variant in eligible),
                excluded={variant: excluded[variant] for variant in L2_OPTIONS if variant in excluded},
                p_thompson=p_ts,
                pinned=pin_act,
            )
            cells.append(cell)
            if explain_cell is None or (spec.has_hit and spec.n_plausible == 2 and spec.bin == "b2"):
                top = max(L2_OPTIONS, key=lambda v: (propensities[v], -L2_OPTIONS.index(v)))
                if predictions:
                    explain_cell = (spec, predictions, top)

        rng_u = draw_stream(request.seed)
        rows = tuple(
            tuple(float(rng_u.random()) for _ in range(MAX_DRAWS_PER_TRIAL)) for _ in range(request.trials_in_block)
        )
        table_mode = PolicyMode.PINNED if any_pin_act else mode
        if explain_cell is not None:
            spec, predictions, top = explain_cell
            explanation = build_explanation(
                mode=table_mode.value, chosen=top, predictions=predictions, excluded={}, baseline="act"
            )
        else:
            explanation = build_explanation(
                mode=table_mode.value, chosen="wait_for_evidence", predictions={}, excluded={}, baseline=None
            )
        return AmbiguityTable(
            decision_id=request.decision_id,
            participant_key=request.participant_key,
            block_decision_id=request.block_decision_id,
            mode=table_mode,
            ambiguity_threshold=float(request.settings.ambiguity_threshold),
            bins=bin_bounds(float(request.settings.ambiguity_threshold)),
            cells=tuple(cells),
            start_trial_index=request.start_trial_index,
            draws=rows,
            explanation=explanation,
            seed=request.seed,
            versions=dict(request.versions),
            pin_act=any_pin_act,
        )

    # ------------------------------------------------------------------------------------------------------ L3

    @final
    def offer_safe_set(self, request: OfferRequest, *, adaptive: bool) -> tuple[list[str], dict[str, ExclusionReason]]:
        candidates = []
        excluded: dict[str, ExclusionReason] = {}
        for variant in L3_OPTIONS:
            applicable = True
            if variant == "offer_modality:keyboard":
                applicable = request.keyboard_available and request.modality != "keyboard"
            elif variant == "offer_modality:pointer":
                applicable = request.modality == "hand_pointer"
            if applicable:
                candidates.append(variant)
            else:
                excluded[variant] = ExclusionReason.NOT_APPLICABLE
        safe, c_excluded = apply_hard_constraints(
            DecisionLevel.OFFER,
            candidates,
            preferences=request.preferences,
            state=request.state,
            modality=request.modality,
            participant_key=request.participant_key,
            adaptive=adaptive,
            context=request.context,
        )
        excluded.update(c_excluded)
        if "no_offer" not in safe:  # no_offer is always applicable and cannot be turned off
            safe.insert(0, "no_offer")
            excluded.pop("no_offer", None)
        return safe, excluded

    @final
    def adaptive_offer_probabilities(
        self, request: OfferRequest, safe: Sequence[str], rng: np.random.Generator, draws: int
    ) -> tuple[dict[str, float], dict[str, float]]:
        weights = UtilityWeights.from_priorities(*request.priorities)
        offers = [variant for variant in safe if variant != "no_offer"]
        if request.sampler is None or len(safe) == 1:
            p_ts = {variant: 1.0 / len(safe) for variant in safe}
        else:
            sampled = request.sampler(rng, offers, draws)
            columns = []
            for variant in safe:
                if variant == "no_offer":
                    columns.append(np.zeros(draws))
                else:
                    columns.append(
                        l3_utility(
                            variant,
                            p_accept=_draw_column(sampled, "accepted", variant, draws),
                            p_dismiss=_draw_column(sampled, "dismissed", variant, draws),
                            weights=weights,
                        )
                    )
            frequency = argmax_frequency(np.column_stack([np.broadcast_to(c, (draws,)) for c in columns]))
            p_ts = {variant: float(frequency[i]) for i, variant in enumerate(safe)}
        return thompson_mixture(p_ts, safe, float(request.settings.exploration_floor)), p_ts

    @final
    def decide_offer(self, request: OfferRequest) -> Decision:
        """An L3 offer decision (section 9.4)."""
        mode = PolicyMode(request.mode)
        options = tuple(L3_OPTIONS)
        p_ts: dict[str, float] | None = None
        exploration = 0.0
        if mode in (PolicyMode.PROBE, PolicyMode.PINNED, PolicyMode.BASELINE):
            safe, excluded = self.offer_safe_set(request, adaptive=False)
            for variant in safe:
                if variant != "no_offer":
                    excluded[variant] = (
                        ExclusionReason.PINNED_OTHER if mode is PolicyMode.PINNED else ExclusionReason.PROBE_SCHEDULE
                    )
            eligible = ["no_offer"]
            pi = _point("no_offer", options)
        elif mode is PolicyMode.ADAPTIVE:
            safe, excluded = self.offer_safe_set(request, adaptive=True)
            pi_safe, p_ts = self.adaptive_offer_probabilities(
                request, safe, thompson_stream(request.seed), max(1, int(request.settings.thompson_draws))
            )
            exploration = float(request.settings.exploration_floor)
            eligible = list(safe)
            pi = {variant: pi_safe.get(variant, 0.0) for variant in options}
        else:
            safe, excluded = self.offer_safe_set(request, adaptive=False)
            eligible = list(safe)
            pi = _uniform(safe, options)
        pi = _normalise(pi, options)
        u = float(draw_stream(request.seed).random())
        chosen = select_from_cell(options, pi, u)
        eligible_t = tuple(variant for variant in options if variant in eligible)
        _check_decision(pi, chosen, eligible_t, excluded)
        explanation = build_explanation(
            mode=mode.value, chosen=chosen, predictions=request.predictions, excluded=excluded, baseline=None
        )
        return Decision(
            decision_id=request.decision_id,
            participant_key=request.participant_key,
            level=DecisionLevel.OFFER,
            mode=mode,
            variant=chosen,
            options=options,
            eligible=eligible_t,
            excluded={variant: excluded[variant] for variant in options if variant in excluded},
            propensities=pi,
            p_thompson=None if p_ts is None else {variant: p_ts.get(variant, 0.0) for variant in options},
            exploration=exploration,
            fallback=False,
            protocol_deviation=False,
            round_index=None,
            spread=None,
            priorities=None,
            predictions=dict(request.predictions),
            explanation=explanation,
            seed=request.seed,
            u=u,
            versions=dict(request.versions),
        )


def _check_decision(
    pi: Mapping[str, float], chosen: str, eligible: Sequence[str], excluded: Mapping[str, ExclusionReason]
) -> None:
    """P1 and P5: support only on the eligible set, sum 1, positive propensity for the chosen variant."""
    if abs(sum(pi.values()) - 1.0) > 1e-9:
        raise RuntimeError("propensities must sum to 1")
    if pi.get(chosen, 0.0) <= 0.0:
        raise RuntimeError("the chosen variant must have positive propensity")
    for variant, value in pi.items():
        if value > 0.0 and variant not in eligible:
            raise RuntimeError("propensity outside the eligible set")
        if value > 0.0 and variant in excluded:
            raise RuntimeError("propensity on an excluded variant")


def _check_cell(
    propensities: Mapping[str, float], eligible: Sequence[str], *, pin_act: bool, has_hit: bool, enforce_c2: bool
) -> None:
    """P1, P3 and P5 for one cell: act only in a pin_act collapse (unless the 'no ambiguity rule' ablation replaced
    the C2 hook), support only on the eligible set, propensities summing to 1."""
    if abs(sum(propensities.values()) - 1.0) > 1e-9:
        raise RuntimeError("cell propensities must sum to 1")
    collapse = pin_act and has_hit and propensities.get("act", 0.0) == 1.0
    if enforce_c2 and propensities.get("act", 0.0) > 0.0 and not collapse:
        raise RuntimeError("act has mass outside a pin_act collapse")
    for variant, value in propensities.items():
        if value > 0.0 and variant not in eligible:
            raise RuntimeError("cell propensity outside the eligible set")


# -------------------------------------------------------------------------------------------- decision-table fixture


def fixture_cases(seed: int = FIXTURE_SEED, n: int = FIXTURE_CASES) -> list[dict[str, Any]]:
    """Seeded ``select_from_cell`` cases shared with the TypeScript twin ``selectFromCell``."""
    rng = np.random.Generator(np.random.PCG64(seed))
    orders = [
        list(L2_OPTIONS),
        list(l1_arms("mouse")),
        list(l1_arms("hand_pointer")),
        list(l1_arms("keyboard")),
        list(L3_OPTIONS),
    ]
    cases: list[dict[str, Any]] = []
    while len(cases) < n:
        order = orders[int(rng.integers(0, len(orders)))]
        kind = int(rng.integers(0, 6))
        raw = rng.random(len(order))
        mask = rng.random(len(order)) < 0.6
        if not mask.any():
            mask[int(rng.integers(0, len(order)))] = True
        if kind == 0:  # a point mass
            mask = np.zeros(len(order), dtype=bool)
            mask[int(rng.integers(0, len(order)))] = True
        weights = np.where(mask, raw, 0.0)
        if kind == 1:  # uniform over the support
            weights = mask.astype(float)
        weights = weights / weights.sum()
        propensities = {variant: float(weights[i]) for i, variant in enumerate(order)}
        cumulative = np.cumsum(weights)
        if kind == 2:  # u exactly on a cumulative boundary
            u = float(cumulative[int(rng.integers(0, len(order)))])
        elif kind == 3:  # u just below 1 (rounding may leave no cumulative sum above it)
            u = float(np.nextafter(1.0, 0.0))
        elif kind == 4:
            u = 0.0
        else:
            u = float(rng.random())
        cases.append(
            {"order": order, "propensities": propensities, "u": u, "expected": select_from_cell(order, propensities, u)}
        )
    return cases


def fixture_json(seed: int = FIXTURE_SEED) -> str:
    return json.dumps({"seed": seed, "cases": fixture_cases(seed)}, indent=1) + "\n"


def write_fixture(path: Path, seed: int = FIXTURE_SEED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture_json(seed).encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the decision-table parity fixture (spec 9.3).")
    parser.add_argument("--write-fixture", metavar="PATH", required=True)
    args = parser.parse_args(argv)
    write_fixture(Path(args.write_fixture))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
