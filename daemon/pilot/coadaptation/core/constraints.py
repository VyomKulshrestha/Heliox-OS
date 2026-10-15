"""Hard constraints, fallbacks and the cap rule (sections 9.2-9.3 and 9.7 of the co-adaptation spec).

Everything here is deterministic and reads no model output: rule C1 comes from the person's preferences, C5, C6 and
C8 from session-scoped state, and C7 from the calibration-suspension flag that ``observe_trial`` set. The safety
filter (step 3) and the ambiguity rule C2 at L2 live in ``policy.py`` as the policy's two overridable hooks. Rule C9
(irreversible candidates) is applied by the host's commit reducer, never by the policy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .preferences import (
    BLOCK_TRIALS,
    Preference,
    PreferenceConflict,
    PreferenceKind,
    PreferenceSet,
    caution_of,
    family_name,
)
from .vocabulary import L2_OPTIONS, VARIANTS, DecisionLevel, l1_arms

if TYPE_CHECKING:
    from .state import Modality

# Ordinal cutpoints of the burden head (section 7.1), used by the C6 closed form.
BURDEN_CUTPOINTS: tuple[float, ...] = (-math.inf, -1.5, -0.9, -0.3, 0.3, 0.9, 1.5, math.inf)
CONTROL_FLOOR_MEAN = 3.0
CONTROL_FLOOR_UPPER = 3.5
CONTROL_FLOOR_Z = 1.2816  # 90th percentile of the expected rating (80% two-sided bound)
OFFER_COOLDOWN_TRIALS = 10


class ExclusionReason(StrEnum):
    FORBIDDEN_BY_YOU = "forbidden_by_you"
    BELOW_YOUR_MIN_CAUTION = "below_your_min_caution"
    PINNED_OTHER = "pinned_other"
    DIRECT_SELECTION_PINNED = "direct_selection_pinned"
    NOT_APPLICABLE = "not_applicable"
    AMBIGUOUS_ACTIVATION = "ambiguous_activation"
    SLIP_RISK_ABOVE_LIMIT = "slip_risk_above_limit"
    RECENT_VETO = "recent_veto"
    LOW_CONTROL_RATING = "low_control_rating"
    CALIBRATION_SUSPENDED = "calibration_suspended"
    INTERRUPTION_LIMIT = "interruption_limit"
    OFFER_COOLDOWN = "offer_cooldown"
    USED_THIS_ROUND = "used_this_round"
    PROBE_SCHEDULE = "probe_schedule"
    COMPARATOR_SCHEDULE = "comparator_schedule"
    PURPOSE_MISSING = "purpose_missing"


def _level_key(level: DecisionLevel | str) -> str:
    return DecisionLevel(level).value


@dataclass
class ConstraintState:
    """Session-scoped constraint state (never snapshotted): C5 vetoes, C6 exclusions, C7 flag, C8 cooldown and the
    interruptions used per block (cap rule, section 9.3)."""

    vetoed: dict[str, set[str]] = field(default_factory=dict)  # level value -> variants excluded at the next decision
    control_floor: set[str] = field(default_factory=set)  # C6: L1 regimes excluded for the rest of the session
    calibration_suspended: bool = False  # C7, copied from the model's suspension flag
    offers_blocked_until: int | None = None  # C8: session trial count before which offers stay excluded
    trials_recorded: int = 0  # trials recorded in this session (the C8 clock)
    interruptions_used: dict[int, int] = field(default_factory=dict)  # block_index -> k

    def record_veto(self, level: DecisionLevel | str, variant: str) -> None:
        """C5: exclude ``variant`` at the next decision of ``level`` (not called for pinned-mode decisions)."""
        if variant == "wait_for_evidence":
            return
        self.vetoed.setdefault(_level_key(level), set()).add(variant)

    def vetoed_at(self, level: DecisionLevel | str) -> frozenset[str]:
        return frozenset(self.vetoed.get(_level_key(level), ()))

    def consume_vetoes(self, level: DecisionLevel | str) -> None:
        """Called after a decision of ``level`` was made: the cooldown covers exactly that next decision."""
        self.vetoed.pop(_level_key(level), None)

    def add_control_floor(self, regime: str) -> None:
        self.control_floor.add(regime)

    def dismiss_offer(self) -> None:
        """C8: after a dismissed offer, offers are excluded for the next 10 trials."""
        self.offers_blocked_until = self.trials_recorded + OFFER_COOLDOWN_TRIALS

    def offers_blocked(self) -> bool:
        return self.offers_blocked_until is not None and self.trials_recorded < self.offers_blocked_until

    def interruptions(self, block_index: int) -> int:
        return self.interruptions_used.get(block_index, 0)


def _mark(excluded: dict[str, ExclusionReason], variant: str, reason: ExclusionReason) -> None:
    excluded.setdefault(variant, reason)


def apply_hard_constraints(
    level: DecisionLevel,
    candidates: Sequence[str],
    *,
    preferences: PreferenceSet,
    state: ConstraintState,
    modality: Modality,
    participant_key: str | None,
    adaptive: bool,
    interruptions_used: int = 0,
    context: str = "study",
    use_c5: bool = True,
    use_c6: bool = True,
) -> tuple[list[str], dict[str, ExclusionReason]]:
    """Rules C1, C2 (L1 part), C5, C6, C7 and C8 for one decision (section 9.2 step 2).

    C1 comes from ``PreferenceSet.apply``. At L1, ``assistance_off`` leaves the adaptive set (``probe_schedule``),
    and in adaptive mode C7 removes the acting regimes. C2 at L2 (no ``act``) is the policy's
    ``_ambiguity_constraint`` hook and the ``pin_act`` collapse is ``pin_act_collapse``; neither is applied here.
    """
    allowed, excluded = preferences.apply(
        level,
        candidates,
        context=context,
        modality=modality,
        participant_key=participant_key,
        interruptions_used=interruptions_used,
    )
    excluded = dict(excluded)
    vetoed = state.vetoed_at(level) if use_c5 else frozenset()
    kept: list[str] = []
    for variant in allowed:
        if level == DecisionLevel.BLOCK_REGIME and adaptive and variant == "assistance_off":
            _mark(excluded, variant, ExclusionReason.PROBE_SCHEDULE)
        elif variant in vetoed:
            _mark(excluded, variant, ExclusionReason.RECENT_VETO)
        elif level == DecisionLevel.BLOCK_REGIME and use_c6 and variant in state.control_floor:
            _mark(excluded, variant, ExclusionReason.LOW_CONTROL_RATING)
        elif (
            level == DecisionLevel.BLOCK_REGIME
            and adaptive
            and state.calibration_suspended
            and VARIANTS[variant].acting
        ):
            _mark(excluded, variant, ExclusionReason.CALIBRATION_SUSPENDED)
        elif level == DecisionLevel.OFFER and variant != "no_offer" and state.offers_blocked():
            _mark(excluded, variant, ExclusionReason.OFFER_COOLDOWN)
        else:
            kept.append(variant)
    return kept, excluded


def pin_act_collapse(candidates: Sequence[str]) -> tuple[list[str], dict[str, ExclusionReason]]:
    """The ``pin_act`` collapse of one L2 cell: ``act`` when the cell has a literal hit (``act`` applicable), else
    ``wait_for_evidence``; every other response is excluded (``pinned_other``)."""
    chosen = "act" if "act" in candidates else "wait_for_evidence"
    excluded = {variant: ExclusionReason.PINNED_OTHER for variant in candidates if variant != chosen}
    return [chosen], excluded


# ------------------------------------------------------------------------------------------------- fallbacks


@dataclass(frozen=True, slots=True)
class FreezeConflict:
    message: str
    conflicting: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "conflicting": list(self.conflicting)}

    @classmethod
    def from_conflict(cls, conflict: PreferenceConflict) -> FreezeConflict:
        return cls(message=conflict.message, conflicting=tuple(conflict.conflicting_pref_ids))


@dataclass(frozen=True, slots=True)
class Fallbacks:
    l1: str | None
    plain: str | None
    refresh: str | None
    practice: str | None
    freezable: tuple[str, ...]
    freeze_conflict: FreezeConflict | None
    l2: tuple[str, ...]
    interruption_cap: int | None
    issued_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """FallbacksV1 (section 14.4)."""
        return {
            "l1": self.l1,
            "plain": self.plain,
            "refresh": self.refresh,
            "practice": self.practice,
            "freezable": list(self.freezable),
            "freeze_conflict": None if self.freeze_conflict is None else self.freeze_conflict.to_dict(),
            "l2": list(self.l2),
            "interruption_cap": self.interruption_cap,
            "issued_at": self.issued_at.isoformat(),
        }


def _most_cautious(candidates: Iterable[str], *, suspended: bool) -> str | None:
    pool = [variant for variant in candidates if not (suspended and VARIANTS[variant].acting)]
    return max(pool, key=caution_of) if pool else None


def fallbacks(
    preferences: PreferenceSet,
    modality: Modality,
    *,
    participant_key: str | None,
    calibration_suspended: bool,
    now: datetime,
    current_regime: str | None = None,
) -> Fallbacks:
    """Section 9.2 step 4, from the active preferences (C1) and the C7 flag only; reads no model output.

    ``l1``, ``refresh`` and ``practice`` are the frozen or pinned regime while one is in force for the modality (C7
    does not apply to it). Otherwise ``l1`` is the most cautious regime C1 allows that, during a suspension, is not
    acting; ``plain`` is ``assistance_off`` only when C1 allows it and calibration is not suspended, else that most
    cautious regime; ``refresh`` and ``practice`` equal ``plain``. Pass ``preferences.as_of(t)`` to recompute the
    fallbacks issued at an earlier time ``t``.
    """
    arms = l1_arms(modality)
    c1_allowed, _ = preferences.apply(
        DecisionLevel.BLOCK_REGIME,
        arms,
        context="study",
        modality=modality,
        participant_key=participant_key,
        interruptions_used=0,
        hold=False,
    )
    held = preferences.held_regime(context="study", modality=modality, participant_key=participant_key)
    cautious = _most_cautious(c1_allowed, suspended=calibration_suspended)
    plain = "assistance_off" if ("assistance_off" in c1_allowed and not calibration_suspended) else cautious
    if held is not None:
        l1 = refresh = practice = held
    else:
        l1 = cautious
        refresh = practice = plain

    freezable = preferences.freezable(modality=modality, participant_key=participant_key)
    target = current_regime or l1 or max(arms, key=caution_of)
    conflict = preferences.freeze_conflict(target, modality=modality, participant_key=participant_key)

    asks, _ = preferences.apply(
        DecisionLevel.AMBIGUITY_RESPONSE,
        ["ask_clarification:3", "ask_clarification:2"],
        context="study",
        modality=modality,
        participant_key=participant_key,
        interruptions_used=0,
    )
    if preferences.pin_act(context="study", modality=modality, participant_key=participant_key):
        l2: tuple[str, ...] = ("act", "wait_for_evidence")
    else:
        l2 = (*asks, "wait_for_evidence")
    return Fallbacks(
        l1=l1,
        plain=plain,
        refresh=refresh,
        practice=practice,
        freezable=freezable,
        freeze_conflict=None if conflict is None else FreezeConflict.from_conflict(conflict),
        l2=l2,
        interruption_cap=preferences.interruption_cap(
            context="study", modality=modality, participant_key=participant_key
        ),
        issued_at=now,
    )


def parse_cell_id(cell: str) -> tuple[str, int, bool]:
    """``"b2:p3:h1"`` -> (``"b2"``, 3, True)."""
    parts = cell.split(":")
    if len(parts) != 3 or parts[0] not in ("b1", "b2", "b3") or parts[1][:1] != "p" or parts[2][:1] != "h":
        raise ValueError("cell id must look like b1:p2:h1")
    n_plausible = int(parts[1][1:])
    has_hit = parts[2][1:]
    if n_plausible not in (1, 2, 3) or has_hit not in ("0", "1"):
        raise ValueError("cell id must look like b1:p2:h1")
    return parts[0], n_plausible, has_hit == "1"


def l2_applicable(variant: str, *, n_plausible: int, has_hit: bool) -> bool:
    """Section 4.5 at L2: act and second_click need a literal hit; ask_clarification:k needs k plausible candidates."""
    if variant in ("act", "confirm_selection:second_click"):
        return has_hit
    if variant == "ask_clarification:2":
        return n_plausible >= 2
    if variant == "ask_clarification:3":
        return n_plausible >= 3
    return variant == "wait_for_evidence"


def _cell_shape(cell: Any) -> tuple[int, bool]:
    if isinstance(cell, str):
        _, n_plausible, has_hit = parse_cell_id(cell)
        return n_plausible, has_hit
    return int(cell.n_plausible), bool(cell.has_hit)


def fixed_l2_rule(fallbacks: Fallbacks, cell: Any) -> str:
    """The first entry of ``fallbacks.l2`` applicable to ``cell`` (a cell id, a table cell or an activation context).

    ``wait_for_evidence`` ends every list and is always applicable, so the rule never runs out.
    """
    n_plausible, has_hit = _cell_shape(cell)
    for variant in fallbacks.l2:
        if l2_applicable(variant, n_plausible=n_plausible, has_hit=has_hit):
            return variant
    return "wait_for_evidence"


def cap_applies(interruption_cap: int | None, interruptions_used: int, *, pin_act: bool) -> bool:
    """The cap rule of section 9.3: ``wait_for_evidence`` (source ``cap``) replaces any L2 response once the block's
    count reached the cap, except in a ``pin_act`` collapse."""
    return interruption_cap is not None and interruptions_used >= interruption_cap and not pin_act


def is_l1_step(activation: Any) -> bool:
    """An L1 confirm or hold step (section 9.3): result ``confirming`` or ``held``, no L2 response, not C9."""
    return activation.result in ("confirming", "held") and activation.l2 is None and not activation.irreversible_rule


def trial_interruptions(activations: Sequence[Any]) -> int:
    """A trial's contribution to the block's cap count k (section 9.3), from each activation's immediate ``result``
    with ``l2`` and ``irreversible_rule`` (never ``outcome``): the trial's L1 confirm and hold steps add 1 together,
    however often they were retried; every L2 confirm, C9 confirm and clarification prompt adds 1."""
    l1_step = False
    count = 0
    for activation in activations:
        if is_l1_step(activation):
            l1_step = True
        elif activation.result in ("confirming", "held", "asked"):
            count += 1
    return count + (1 if l1_step else 0)


# ------------------------------------------------------------------------------------------------ C6 closed form


def expected_rating(eta: float, cutpoints: Sequence[float] = BURDEN_CUTPOINTS) -> float:
    """E[y | eta] = sum_k k * [Phi(c_k - eta) - Phi(c_{k-1} - eta)] for the 7-point ordinal probit."""
    total = 0.0
    for k in range(1, len(cutpoints)):
        total += k * (_phi_cdf(cutpoints[k] - eta) - _phi_cdf(cutpoints[k - 1] - eta))
    return total


def predictive_mean_rating(mu: float, s: float, cutpoints: Sequence[float] = BURDEN_CUTPOINTS) -> float:
    """Posterior predictive mean rating: P(k) uses (c_k - mu) / sqrt(1 + s^2) (section 7.6)."""
    scale = math.sqrt(1.0 + s * s)
    total = 0.0
    for k in range(1, len(cutpoints)):
        total += k * (_phi_cdf((cutpoints[k] - mu) / scale) - _phi_cdf((cutpoints[k - 1] - mu) / scale))
    return total


def control_upper_bound(mu: float, s: float) -> float:
    """The 90th percentile of the expected control rating, ``E[y | eta = mu + 1.2816 s]`` (exact: E[y | eta] is
    increasing in eta, so no sampling is needed)."""
    return expected_rating(mu + CONTROL_FLOOR_Z * s)


def control_floor_excludes(mu: float, s: float) -> bool:
    """C6: posterior mean control below 3.0 and its 80% upper bound below 3.5."""
    return predictive_mean_rating(mu, s) < CONTROL_FLOOR_MEAN and control_upper_bound(mu, s) < CONTROL_FLOOR_UPPER


def _phi_cdf(x: float) -> float:
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------------------- preference violations


def _history(preference_history: PreferenceSet | Iterable[Preference]) -> PreferenceSet:
    if isinstance(preference_history, PreferenceSet):
        return preference_history
    return PreferenceSet(preference_history)


def _rules_at(
    history: PreferenceSet, when: datetime, *, modality: str, participant_key: str | None
) -> tuple[Preference, ...]:
    rules = [
        pref
        for pref in history.all
        if pref.active_at(when)
        and pref.scope.context in ("study", "all")
        and (pref.scope.modality is None or pref.scope.modality == modality)
    ]
    if participant_key is not None:
        rules = [pref for pref in rules if pref.scope.participant_key in (None, participant_key)]
    return tuple(rules)


def _excluded_by(rules: Sequence[Preference], variant: str) -> str | None:
    if variant != "wait_for_evidence" and any(pref.forbids(variant) for pref in rules):
        return "forbidden"
    minima = [pref.value for pref in rules if pref.kind is PreferenceKind.MIN_CAUTION]
    if (
        minima
        and variant != "wait_for_evidence"
        and 0 <= caution_of(variant) < max(caution_of(value) for value in minima)
    ):
        return "below_min_caution"
    return None


def _held_at(rules: Sequence[Preference], modality: str) -> str | None:
    held = [
        pref
        for pref in rules
        if pref.kind in (PreferenceKind.FREEZE_ADAPTATION, PreferenceKind.PIN_REGIME)
        and pref.regime in l1_arms(modality)
    ]
    if not held:
        return None
    return max(held, key=lambda p: (p.created_at, p.pref_id)).regime


def preference_violations(
    trial: Any,
    preference_history: PreferenceSet | Iterable[Preference],
    *,
    decision_times: Mapping[str, datetime],
    participant_key: str | None = None,
) -> tuple[str, ...]:
    """Applied variants that the rules in force when their source was issued did not allow (section 9.7).

    Binding times: the decision's ``created_at`` for a regime or table draw applied as recommended; the trial's
    ``fallbacks_issued_at`` for an L1 fallback and for fixed-rule, fallback and cap responses before a revert; the
    revert's ``fallbacks_issued_at`` for the revert regime and every response from the revert on. Flags a variant
    forbidden (or below the minimum caution) at its binding time and, while a freeze or pin for the modality was in
    force, an applied L1 regime other than the frozen or pinned one; a "Not this time" revert is exempt from that
    second check. ``preference_history`` holds the rules that applied to the participant (pass ``participant_key``
    to filter a device's full rule set). Returns ``"<where>:<variant>:<reason>"`` strings.
    """
    history = _history(preference_history)
    modality = trial.modality
    violations: list[str] = []

    def judge(where: str, variant: str, when: datetime | None, *, l1: bool, veto_revert: bool = False) -> None:
        if when is None:
            return
        rules = _rules_at(history, when, modality=modality, participant_key=participant_key)
        reason = _excluded_by(rules, variant)
        if reason is not None:
            violations.append(f"{where}:{variant}:{reason}")
            return
        if l1 and not veto_revert:
            held = _held_at(rules, modality)
            if held is not None and variant != held:
                violations.append(f"{where}:{variant}:not_held_regime")

    if trial.l1_fallback_reason is None:
        l1_time = decision_times.get(trial.regime_decision_id)
    else:
        l1_time = trial.fallbacks_issued_at
    judge("l1", trial.applied_regime, l1_time, l1=True)

    revert = trial.revert
    revert_from = revert.from_activation if revert is not None else None
    if revert is not None:
        judge("revert", revert.regime, revert.fallbacks_issued_at, l1=True, veto_revert=revert.reason == "veto")
    for activation in trial.activations:
        applied = activation.l2
        if applied is None:
            continue
        if revert_from is not None and activation.index >= revert_from:
            when = revert.fallbacks_issued_at
        elif applied.source == "table":
            when = decision_times.get(applied.decision_id)
        else:
            when = trial.fallbacks_issued_at
        judge(f"activation[{activation.index}]", applied.variant, when, l1=False)
    return tuple(violations)
