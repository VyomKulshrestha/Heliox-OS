"""Off-policy evaluation of frozen co-adaptation snapshots (spec section 19).

Only frozen snapshot policies are evaluated: the adaptive engine learns after every trial, so it is not a fixed policy
and "the value of the adaptive policy" has no off-policy target. The unit is the block. Every randomized draw of a
block stays in its weight: each scored trial is weighted by the block's L1 factor and the L2 and L3 draws made up to
and including that trial (per-decision importance sampling, Precup, Sutton and Singh 2000), and the block's burden
rating takes the full product. The estimand is the expected utility of one block run by the target given the logged
previous regime and the arms the design still allowed (``R_b``); target mass outside ``R_b``, and mass on arms the
logging policy could not choose, is withheld and reported, never imputed.

Rules fixed before any human data exist: weights are clipped at 20 and an estimate whose effective sample size is
below 30 is refused (``Estimate.refused = "low_ess"``). A refused estimate is a result, reported as "not estimated"
with its ESS; it is never re-run with another threshold, clip, block set or estimator.

Standard library, numpy and ``pilot.coadaptation.core`` only.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

import numpy as np
from numpy.random import PCG64, Generator

from .consent import ConsentPeriod, parse_datetime
from .engine import CoAdaptationEngine, EngineEvent, EngineSettings, PolicyProbabilities
from .preferences import Preference, PreferenceSet, preference_from_dict
from .records import (
    RECORD_SCHEMA_VERSION,
    BurdenReport,
    DecisionRecord,
    SessionRow,
    StoredTrial,
)
from .state import BlockContext, ClockContext, OfferContext
from .study_plan import physics_reference_time_s
from .utility import PROTOCOL_WEIGHTS, UtilityWeights, block_burden_term, realized_utility
from .vocabulary import L2_OPTIONS, DecisionLevel

CLIP = 20.0  # fixed before any human data (section 19)
MIN_ESS = 30.0  # an estimate below this effective sample size is refused, never re-run
OPE_DRAWS = 4096  # Thompson draws behind a frozen target's probabilities
BUNDLE_SCHEMA = 1
BLOCK_PHASES = ("A", "A_lite")  # design-based propensities: the block-level logs
_PHASE_B = ("B", "probe", "comparator")
L2_UNIT_PHASES = ("A", "A_lite", "comparator")  # uniform L2 draws: the trial-level L2 logs
SESSION_EXCLUSION_REASONS = frozenset(
    {
        # C1, C5, C6 and C8 reasons: session state the target cannot recompute from the snapshot
        "forbidden_by_you",
        "below_your_min_caution",
        "pinned_other",
        "direct_selection_pinned",
        "interruption_limit",
        "recent_veto",
        "low_control_rating",
        "offer_cooldown",
    }
)

WeightsKind = Literal["participant", "protocol"]


# ------------------------------------------------------------------------------------------------ logged units


@dataclass(frozen=True, slots=True)
class LoggedStep:
    level: DecisionLevel
    key: str  # cell id for L2; "" for L1 and L3
    trial_index: int | None  # trial of the block the draw belongs to; None for L1
    chosen: str
    propensities: Mapping[str, float]  # full logged vector, zeros included
    context: Any  # what the target needs for this step (opaque to estimators)


@dataclass(frozen=True, slots=True)
class LoggedBlock:
    unit_id: str  # the block's L1 decision id (the trial id for trial-level L2 units)
    group_id: str  # participant key (bootstrap group)
    session_id: str
    context: Any  # BlockLog: BlockContext plus the logged session-state exclusions; used by q_hat and the target
    remaining_arms: tuple[str, ...]  # R_b (empty for a trial-level L2 unit, which has no L1 step)
    steps: tuple[LoggedStep, ...]  # every randomized draw of the unit, in order
    # u_{b,t} of the scored trials, in order (realized utility without burden)
    trial_utilities: tuple[float, ...]
    trial_positions: tuple[int, ...]  # trial index within the block of each scored trial
    burden_term: float  # w_B * B_b, the block's weighted burden term (21.2)
    reward: float  # r_b = mean(trial_utilities) - burden_term; no switching penalty
    behavior: Literal["policy", "baseline"]


@dataclass(frozen=True, slots=True)
class BlockLog:
    """The logged context of one block: what a frozen target needs to compute its probabilities there."""

    participant_key: str
    session_id: str
    session_number: int
    block_index: int
    phase: str
    task: str
    decision_id: str  # the block's L1 decision
    created_at: datetime  # the L1 decision's time: preferences and diffusion are evaluated here
    block: BlockContext
    # C1, C5, C6 and C8 exclusions logged with the L1 decision
    session_exclusions: Mapping[str, str]
    applied_regime: str


TargetPolicy = Callable[[LoggedBlock, LoggedStep], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float | None  # None when refused
    # Grouped bootstrap (participants, then sessions); None with one group.
    ci95: tuple[float, float] | None
    ess: float
    unsupported_fraction: float  # mean target mass on choices the logging policy could not make
    withheld_mass: float  # mean 1 - pi_e(R_b | x_b) over the blocks (supported part of R_b only)
    refused: str | None = None  # "low_ess" or "no_support"
    estimator: str = ""
    n_units: int = 0
    clip: float | None = CLIP

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "value": self.value,
            "ci95": None if self.ci95 is None else [self.ci95[0], self.ci95[1]],
            "ess": self.ess,
            "unsupported_fraction": self.unsupported_fraction,
            "withheld_mass": self.withheld_mass,
            "refused": self.refused,
            "reported_as": "not estimated" if self.refused is not None else "estimated",
            "n_units": self.n_units,
            "clip": self.clip,
        }


@dataclass(frozen=True, slots=True)
class SupportReport:
    n_units: int
    withheld_mass: float  # mean over units of 1 - pi_e(R_b ∩ supp | x_b)
    # Mean over randomized steps of the target mass the logging policy gave 0.
    unsupported_fraction: float
    unsupported_by_level: Mapping[str, float]
    unsupported_units: int  # units whose logged L1 choice lies outside the supported part of R_b
    # Units whose logged L1 choice is supported but gets target probability 0 (lambda_b = 0 with full support, for
    # example assistance_off under an adaptive target, which C2 never lets choose it); not a support problem.
    zero_target_units: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_units": self.n_units,
            "withheld_mass": self.withheld_mass,
            "unsupported_fraction": self.unsupported_fraction,
            "unsupported_by_level": dict(self.unsupported_by_level),
            "unsupported_units": self.unsupported_units,
            "zero_target_units": self.zero_target_units,
        }


# --------------------------------------------------------------------------------------------- weights per unit


@dataclass(frozen=True, slots=True)
class _UnitWeights:
    lam: float  # L1 factor (0 when the logged choice is outside the supported part of R_b; 1 without an L1 step)
    rho_t: np.ndarray  # rho_{b,t} for each scored trial (unclipped)
    rho_b: float  # full product over every draw of the unit (unclipped)
    mass: float  # pi_e(R_b ∩ supp | x_b), the unit's weight in the estimand (1 without an L1 step)
    support: tuple[str, ...]  # R_b ∩ {v : pi_b(v) > 0}
    target_l1: Mapping[str, float]  # pi_e over the L1 arms ({} without an L1 step)
    chosen_l1: str | None
    # (level, target mass with logging propensity 0) per randomized step
    unsupported: tuple[tuple[str, float], ...]

    @property
    def unsupported_unit(self) -> bool:
        """The logged L1 choice lies outside the supported part of R_b (``R_b ∩ supp(pi_b)``)."""
        return self.chosen_l1 is not None and self.chosen_l1 not in self.support

    @property
    def zero_target_unit(self) -> bool:
        """The logged L1 choice is supported, but the target gives it probability 0, so ``lambda_b`` is 0."""
        return self.chosen_l1 is not None and self.chosen_l1 in self.support and self.lam == 0.0


def _is_level(step: LoggedStep, level: DecisionLevel) -> bool:
    return str(step.level) == level.value


def _unit_weights(block: LoggedBlock, target: TargetPolicy) -> _UnitWeights:
    lam = 1.0
    mass = 1.0
    support: tuple[str, ...] = ()
    target_l1: Mapping[str, float] = {}
    chosen_l1: str | None = None
    unsupported: list[tuple[str, float]] = []
    factors: list[tuple[int, float]] = []
    for step in block.steps:
        logged = step.propensities
        chosen_p = float(logged.get(step.chosen, 0.0))
        if chosen_p <= 0.0:
            raise ValueError("a logged choice must have positive logged propensity")
        probs = target(block, step)
        if _is_level(step, DecisionLevel.BLOCK_REGIME):
            arms = block.remaining_arms or tuple(arm for arm, p in logged.items() if p > 0)
            support = tuple(arm for arm in arms if float(logged.get(arm, 0.0)) > 0.0)
            target_l1 = dict(probs)
            chosen_l1 = step.chosen
            mass = float(sum(float(probs.get(arm, 0.0)) for arm in support))
            lam = float(probs.get(step.chosen, 0.0)) / chosen_p if step.chosen in support else 0.0
            unsupported.append(
                (
                    str(step.level),
                    float(sum(float(probs.get(arm, 0.0)) for arm in arms if arm not in support)),
                )
            )
            continue
        if step.trial_index is None:
            raise ValueError("an L2 or L3 step needs the trial it belongs to")
        factors.append((step.trial_index, float(probs.get(step.chosen, 0.0)) / chosen_p))
        off_support = [float(p) for o, p in probs.items() if float(logged.get(o, 0.0)) <= 0.0]
        unsupported.append((str(step.level), float(sum(off_support))))
    rho_t = np.empty(len(block.trial_positions), dtype=float)
    for i, position in enumerate(block.trial_positions):
        product = lam
        for trial_index, factor in factors:
            if trial_index <= position:
                product *= factor
        rho_t[i] = product
    rho_b = lam * math.prod(factor for _, factor in factors)
    return _UnitWeights(
        lam=lam,
        rho_t=rho_t,
        rho_b=rho_b,
        mass=mass,
        support=support,
        target_l1=target_l1,
        chosen_l1=chosen_l1,
        unsupported=tuple(unsupported),
    )


def _clip(value: Any, clip: float | None) -> Any:
    return value if clip is None else np.minimum(value, clip)


@dataclass(frozen=True, slots=True)
class UnitContribution:
    """One unit's pieces of the estimators, after clipping (section 19)."""

    unit_id: str
    group_id: str
    session_id: str
    weighted_reward: float  # r~_b = mean_t rho_{b,t} u_{b,t} - rho_b w_B B_b
    mean_weight: float  # w-bar_b = mean_t rho_{b,t}
    mass: float  # pi_e(R_b ∩ supp | x_b)
    lam: float  # L1 factor lambda_b
    rho_b: float  # full product over every draw of the unit
    # sum_{v in R_b} pi_e(v) q(x_b, v) + r~_b - lambda_b q(x_b, v_b); None without q_hat
    dr_term: float | None
    unsupported: tuple[tuple[str, float], ...]
    unsupported_unit: bool  # the logged L1 choice lies outside the supported part of R_b
    chosen: str | None = None  # the logged L1 choice (None for a trial-level L2 unit)
    # pi_e(v | x_b) for v in R_b ∩ supp (the control variate)
    support_probs: tuple[tuple[str, float], ...] = ()
    zero_target_unit: bool = False  # the logged L1 choice is supported but has target probability 0

    def with_q_hat(self, q: Callable[[str], float]) -> UnitContribution:
        """This unit with its DR term for ``q`` (the unit's ``q_hat`` as a function of the regime)."""
        control = sum(probability * q(arm) for arm, probability in self.support_probs)
        if self.chosen is not None:
            control -= self.lam * q(self.chosen)
        return replace(self, dr_term=control + self.weighted_reward)


def unit_contributions(
    blocks: Sequence[LoggedBlock],
    target: TargetPolicy,
    *,
    clip: float | None = CLIP,
    q_hat: Callable[[LoggedBlock, str], float] | None = None,
) -> list[UnitContribution]:
    """Every unit's weighted reward, effective weight and target mass. ``rho_{b,t}`` is ``lambda_b`` times the
    factors of the L2 and L3 draws made in trials ``0..t`` only; ``rho_b`` takes every draw of the unit (interrupted
    trials included); each is clipped at ``clip``."""
    out: list[UnitContribution] = []
    for block in blocks:
        if len(block.trial_utilities) != len(block.trial_positions):
            raise ValueError("every scored trial needs its position")
        w = _unit_weights(block, target)
        rho_t = _clip(w.rho_t, clip)
        rho_b = float(_clip(w.rho_b, clip))
        lam = float(_clip(w.lam, clip))
        if len(block.trial_utilities):
            utilities = np.asarray(block.trial_utilities, dtype=float)
            weighted = float(np.mean(rho_t * utilities)) - rho_b * block.burden_term
            mean_weight = float(np.mean(rho_t))
        else:
            weighted = -rho_b * block.burden_term
            mean_weight = rho_b
        unit = UnitContribution(
            unit_id=block.unit_id,
            group_id=block.group_id,
            session_id=block.session_id,
            weighted_reward=weighted,
            mean_weight=mean_weight,
            mass=w.mass,
            lam=lam,
            rho_b=rho_b,
            dr_term=None,
            unsupported=w.unsupported,
            unsupported_unit=w.unsupported_unit,
            chosen=w.chosen_l1,
            support_probs=tuple((arm, float(w.target_l1.get(arm, 0.0))) for arm in w.support),
            zero_target_unit=w.zero_target_unit,
        )
        if q_hat is not None:
            unit = unit.with_q_hat(lambda arm, _block=block: q_hat(_block, arm))
        out.append(unit)
    return out


def effective_sample_size(weights: Iterable[float]) -> float:
    """``(sum w)^2 / sum w^2`` (0 when every weight is 0)."""
    w = np.asarray(list(weights), dtype=float)
    denominator = float(np.sum(w * w))
    return float(np.sum(w)) ** 2 / denominator if denominator > 0 else 0.0


# ------------------------------------------------------------------------------------------------- bootstrap


def grouped_bootstrap(
    values_by_group: Mapping[str, Sequence[Any]],
    stat: Callable[[np.ndarray], float],
    *,
    n: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval of ``stat`` under a two-stage resample: groups (participants) with replacement, then
    the items (sessions) within each drawn group with replacement.

    Each group maps to its sessions' values (a float, or a fixed-length vector such as a session's numerator and
    denominator); ``stat`` receives the resampled items stacked as an ``(m, d)`` array. Needs at least two groups.
    """
    groups = [key for key in sorted(values_by_group) if len(values_by_group[key])]
    if len(groups) < 2:
        raise ValueError("the grouped bootstrap needs at least two groups")
    if n < 1:
        raise ValueError("n must be positive")
    rows: list[np.ndarray] = []
    offsets = np.empty(len(groups), dtype=np.int64)
    counts = np.empty(len(groups), dtype=np.int64)
    position = 0
    for g, key in enumerate(groups):
        items = np.atleast_2d(
            np.asarray(values_by_group[key], dtype=float).reshape(len(values_by_group[key]), -1),
        )
        rows.append(items)
        offsets[g] = position
        counts[g] = items.shape[0]
        position += items.shape[0]
    flat = np.concatenate(rows, axis=0)
    rng = Generator(PCG64(seed))
    picks = rng.integers(0, len(groups), size=(n, len(groups)))
    width = int(counts.max())
    uniform = rng.random((n, len(groups), width))
    picked_counts = counts[picks][..., None]
    sessions = np.minimum((uniform * picked_counts).astype(np.int64), picked_counts - 1)
    valid = np.arange(width)[None, None, :] < picked_counts
    index = offsets[picks][..., None] + sessions
    stats = np.empty(n, dtype=float)
    for r in range(n):
        stats[r] = float(stat(flat[index[r][valid[r]]]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _ratio(sample: np.ndarray) -> float:
    denominator = float(np.sum(sample[:, 1]))
    return float(np.sum(sample[:, 0])) / denominator if denominator > 0 else math.nan


def _session_sums(
    units: Sequence[UnitContribution], numerators: Sequence[float], denominators: Sequence[float]
) -> dict[str, list[tuple[float, ...]]]:
    by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0]),
    )
    for unit, num, den in zip(units, numerators, denominators):
        cell = by_group[unit.group_id][unit.session_id]
        cell[0] += num
        cell[1] += den
    out: dict[str, list[tuple[float, ...]]] = {}
    for group, sessions in by_group.items():
        out[group] = [tuple(values) for _, values in sorted(sessions.items())]
    return out


# ------------------------------------------------------------------------------------------------- estimators


def estimate_from_contributions(
    name: Literal["ips", "snips", "dr"],
    contributions: Sequence[UnitContribution],
    *,
    clip: float | None = CLIP,
    bootstrap_n: int = 5000,
    bootstrap_seed: int = 0,
) -> Estimate:
    """IPS, SNIPS or DR from unit contributions, with the refusal rule and the grouped bootstrap interval."""
    contributions = list(contributions)
    if not contributions:
        return Estimate(
            None,
            None,
            0.0,
            0.0,
            0.0,
            refused="no_support",
            estimator=name,
            n_units=0,
            clip=clip,
        )
    if name == "dr":
        if any(c.dr_term is None for c in contributions):
            raise ValueError("the DR estimate needs contributions computed with q_hat")
        numerators = [float(c.dr_term) for c in contributions]  # type: ignore[arg-type]
    else:
        numerators = [c.weighted_reward for c in contributions]
    denominators = [c.mean_weight if name == "snips" else c.mass for c in contributions]
    ess = effective_sample_size(c.mean_weight for c in contributions)
    unsupported = [mass for c in contributions for _, mass in c.unsupported]
    common = {
        "ess": ess,
        "unsupported_fraction": float(np.mean(unsupported)) if unsupported else 0.0,
        "withheld_mass": float(np.mean([1.0 - c.mass for c in contributions])),
        "estimator": name,
        "n_units": len(contributions),
        "clip": clip,
    }
    if sum(denominators) <= 0:
        return Estimate(None, None, refused="no_support", **common)
    if ess < MIN_ESS:
        return Estimate(None, None, refused="low_ess", **common)
    value = sum(numerators) / sum(denominators)
    sessions = _session_sums(contributions, numerators, denominators)
    ci = None
    if len(sessions) >= 2:
        ci = grouped_bootstrap(sessions, _ratio, n=bootstrap_n, seed=bootstrap_seed)
    return Estimate(float(value), ci, **common)


def _estimate(
    name: Literal["ips", "snips", "dr"],
    blocks: Sequence[LoggedBlock],
    target: TargetPolicy,
    *,
    clip: float | None,
    q_hat: Callable[[LoggedBlock, str], float] | None = None,
    bootstrap_seed: int = 0,
) -> Estimate:
    contributions = unit_contributions(blocks, target, clip=clip, q_hat=q_hat)
    return estimate_from_contributions(
        name,
        contributions,
        clip=clip,
        bootstrap_seed=bootstrap_seed,
    )


def ips(
    blocks: Sequence[LoggedBlock],
    target: TargetPolicy,
    *,
    clip: float | None = CLIP,
    bootstrap_seed: int = 0,
) -> Estimate:
    """``sum_b r~_b / sum_b pi_e(R_b | x_b)``; for a target fixed independently of the evaluated blocks its
    unclipped expectation is ``V(pi_e)`` (section 19)."""
    return _estimate("ips", blocks, target, clip=clip, bootstrap_seed=bootstrap_seed)


def snips(
    blocks: Sequence[LoggedBlock],
    target: TargetPolicy,
    *,
    clip: float | None = CLIP,
    bootstrap_seed: int = 0,
) -> Estimate:
    """Self-normalised IPS (Swaminathan and Joachims 2015): divides by ``sum_b w-bar_b``."""
    return _estimate("snips", blocks, target, clip=clip, bootstrap_seed=bootstrap_seed)


def doubly_robust(
    blocks: Sequence[LoggedBlock],
    target: TargetPolicy,
    q_hat: Callable[[LoggedBlock, str], float],
    *,
    clip: float | None = CLIP,
    bootstrap_seed: int = 0,
) -> Estimate:
    """Doubly robust (Dudík, Langford and Li 2011) with the L1 control variate and the per-decision weighted reward:
    ``sum_b [sum_{v in R_b} pi_e(v | x_b) q(x_b, v) + r~_b - lambda_b q(x_b, v_b)] / sum_b pi_e(R_b | x_b)``."""
    return _estimate("dr", blocks, target, clip=clip, q_hat=q_hat, bootstrap_seed=bootstrap_seed)


def support_check(blocks: Sequence[LoggedBlock], target: TargetPolicy) -> SupportReport:
    """Unsupported target mass and the mean withheld mass ``1 - pi_e(R_b | x_b)`` (both reported next to every
    estimate; neither is imputed). ``unsupported_units`` counts units whose logged L1 choice lies outside
    ``R_b ∩ supp(pi_b)``; a supported choice the target gives probability 0 is counted in ``zero_target_units``."""
    weights = [_unit_weights(block, target) for block in blocks]
    by_level: dict[str, list[float]] = defaultdict(list)
    for w in weights:
        for level, mass in w.unsupported:
            by_level[level].append(mass)
    masses = [mass for values in by_level.values() for mass in values]
    return SupportReport(
        n_units=len(weights),
        withheld_mass=float(np.mean([1.0 - w.mass for w in weights])) if weights else 0.0,
        unsupported_fraction=float(np.mean(masses)) if masses else 0.0,
        unsupported_by_level={level: float(np.mean(values)) for level, values in by_level.items()},
        unsupported_units=sum(1 for w in weights if w.unsupported_unit),
        zero_target_units=sum(1 for w in weights if w.zero_target_unit),
    )


def clipping_sensitivity(
    blocks: Sequence[LoggedBlock],
    target: TargetPolicy,
    q_hat: Callable[[LoggedBlock, str], float],
    clips: Sequence[float | None] = (5, 10, 20, None),
) -> dict[str, dict[str, Any]]:
    """The three estimators at several clips, reported next to the fixed-clip estimate (never used to replace it)."""
    report: dict[str, dict[str, Any]] = {}
    for clip in clips:
        key = "none" if clip is None else f"{float(clip):g}"
        values = {
            "ips": ips(blocks, target, clip=clip),
            "snips": snips(blocks, target, clip=clip),
            "dr": doubly_robust(blocks, target, q_hat, clip=clip),
        }
        report[key] = {
            "ess": values["ips"].ess,
            **{name: estimate.value for name, estimate in values.items()},
            "refused": {name: estimate.refused for name, estimate in values.items()},
        }
    return report


# --------------------------------------------------------------------------------------------- outcome model q_hat


def fit_regime_means(blocks: Sequence[LoggedBlock]) -> Callable[[LoggedBlock, str], float]:
    """A simple ``q_hat``: the mean logged reward of blocks run under each (task, regime), falling back to the
    regime's mean and then the overall mean."""
    by_task_regime: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_regime: dict[str, list[float]] = defaultdict(list)
    rewards: list[float] = []
    for block in blocks:
        regime = _l1_choice(block)
        if regime is None:
            continue
        task = _task_of(block)
        by_task_regime[(task, regime)].append(block.reward)
        by_regime[regime].append(block.reward)
        rewards.append(block.reward)
    overall = float(np.mean(rewards)) if rewards else 0.0
    task_means = {key: float(np.mean(values)) for key, values in by_task_regime.items()}
    regime_means = {key: float(np.mean(values)) for key, values in by_regime.items()}

    def q_hat(block: LoggedBlock, regime: str) -> float:
        key = (_task_of(block), regime)
        if key in task_means:
            return task_means[key]
        return regime_means.get(regime, overall)

    return q_hat


def _l1_choice(block: LoggedBlock) -> str | None:
    for step in block.steps:
        if _is_level(step, DecisionLevel.BLOCK_REGIME):
            return step.chosen
    return None


def _task_of(block: LoggedBlock) -> str:
    context = block.context
    return str(getattr(context, "task", "") or "")


def cross_fit_q_hat(
    blocks: Sequence[LoggedBlock],
    fit: Callable[[Sequence[LoggedBlock]], Callable[[LoggedBlock, str], float]] = fit_regime_means,
    *,
    fold_by: str = "session_id",
) -> Callable[[LoggedBlock, str], float]:
    """Leave-one-fold-out ``q_hat`` (folds are sessions by default): a block is predicted by the model fitted
    without its own fold, so its reward never enters its own control variate."""
    blocks = list(blocks)
    folds = sorted({str(getattr(block, fold_by)) for block in blocks})
    fold_of = [str(getattr(block, fold_by)) for block in blocks]
    models = {fold: fit([b for b, own in zip(blocks, fold_of) if own != fold]) for fold in folds}
    pooled = fit(blocks)

    def q_hat(block: LoggedBlock, regime: str) -> float:
        model = models.get(str(getattr(block, fold_by)), pooled)
        return model(block, regime)

    return q_hat


# ------------------------------------------------------------------------------------------------ bundle reading


@dataclass(frozen=True, slots=True)
class BundleRows:
    """An export bundle (section 10.9) read only through the ``from_dict`` methods of section 13.1.2."""

    consents: tuple[ConsentPeriod, ...]
    sessions: tuple[SessionRow, ...]
    decisions: tuple[DecisionRecord, ...]
    trials: tuple[StoredTrial, ...]
    burden: tuple[BurdenReport, ...]
    events: tuple[EngineEvent, ...]
    preferences: tuple[Preference, ...]
    learning_reset_at: datetime | None
    versions: Mapping[str, str] = field(default_factory=dict)

    def participant_keys(self) -> tuple[str, ...]:
        return tuple(sorted({row.participant_key for row in self.sessions}))


def read_bundle(bundle: Mapping[str, Any] | BundleRows) -> BundleRows:
    """Read a decrypted bundle with the records' ``from_dict``; refuses unknown schema values."""
    if isinstance(bundle, BundleRows):
        return bundle
    if bundle.get("bundle_schema") != BUNDLE_SCHEMA:
        raise ValueError("unknown bundle_schema")
    if bundle.get("record_schema") != RECORD_SCHEMA_VERSION:
        raise ValueError("unknown record_schema")
    reset = bundle.get("learning_reset_at")
    return BundleRows(
        consents=tuple(ConsentPeriod.from_dict(d) for d in bundle.get("consents", ())),
        sessions=tuple(SessionRow.from_dict(d) for d in bundle.get("sessions", ())),
        decisions=tuple(DecisionRecord.from_dict(d) for d in bundle.get("decisions", ())),
        trials=tuple(StoredTrial.from_dict(d) for d in bundle.get("trials", ())),
        burden=tuple(BurdenReport.from_dict(d) for d in bundle.get("burden", ())),
        events=tuple(EngineEvent.from_dict(d) for d in bundle.get("events", ())),
        preferences=tuple(preference_from_dict(d) for d in bundle.get("preferences", ())),
        learning_reset_at=None if reset is None else parse_datetime(reset, "learning_reset_at"),
        versions=dict(bundle.get("versions", {})),
    )


def merge_bundles(bundles: Iterable[Mapping[str, Any] | BundleRows]) -> BundleRows:
    """Several participants' bundles as one set of rows (preferences and consents kept per participant)."""
    parts = [read_bundle(bundle) for bundle in bundles]
    resets = [part.learning_reset_at for part in parts if part.learning_reset_at is not None]
    seen: dict[str, Preference] = {}
    for part in parts:
        for pref in part.preferences:
            seen.setdefault(pref.pref_id, pref)
    return BundleRows(
        consents=tuple(row for part in parts for row in part.consents),
        sessions=tuple(row for part in parts for row in part.sessions),
        decisions=tuple(row for part in parts for row in part.decisions),
        trials=tuple(row for part in parts for row in part.trials),
        burden=tuple(row for part in parts for row in part.burden),
        events=tuple(row for part in parts for row in part.events),
        preferences=tuple(seen.values()),
        learning_reset_at=max(resets) if resets else None,
        versions=parts[0].versions if parts else {},
    )


def _block_plan(session: SessionRow, block_index: int) -> Any:
    for plan in session.plan.blocks:
        if plan.index == block_index:
            return plan
    raise ValueError("a trial names a block that its session plan does not have")


def _modality(session: SessionRow) -> str:
    return session.device.modality


@dataclass(frozen=True, slots=True)
class _ParticipantRows:
    sessions: tuple[SessionRow, ...]  # in replay order
    decisions: Mapping[str, DecisionRecord]
    trials_by_session: Mapping[str, tuple[StoredTrial, ...]]  # in seq order
    burden_by_session: Mapping[str, tuple[BurdenReport, ...]]


def _participant_rows(rows: BundleRows) -> dict[str, _ParticipantRows]:
    out: dict[str, _ParticipantRows] = {}
    for key in rows.participant_keys():
        sessions = tuple(
            sorted(
                (row for row in rows.sessions if row.participant_key == key),
                key=lambda row: (row.started_at, row.session_id),
            )
        )
        session_ids = {row.session_id for row in sessions}
        trials: dict[str, list[StoredTrial]] = defaultdict(list)
        for stored in rows.trials:
            if stored.trial.session_id in session_ids:
                trials[stored.trial.session_id].append(stored)
        burden: dict[str, list[BurdenReport]] = defaultdict(list)
        for report in rows.burden:
            if report.session_id in session_ids:
                burden[report.session_id].append(report)
        out[key] = _ParticipantRows(
            sessions=sessions,
            decisions={d.decision_id: d for d in rows.decisions if d.participant_key == key},
            trials_by_session={sid: _sorted_by(items, "trial") for sid, items in trials.items()},
            burden_by_session={sid: _sorted_by(items) for sid, items in burden.items()},
        )
    return out


def _sorted_by(items: Iterable[Any], inner: str | None = None) -> tuple[Any, ...]:
    """Rows in ``seq`` order (a stored trial's ``seq`` lives on its ``trial``)."""
    return tuple(sorted(items, key=lambda row: (getattr(row, inner) if inner else row).seq))


def _all_reports(data: _ParticipantRows) -> list[BurdenReport]:
    return [r for s in data.sessions for r in data.burden_by_session.get(s.session_id, ())]


def _completed_time_s(stored: StoredTrial) -> float | None:
    labels = stored.labels
    if labels is None or not labels.complete:
        return None
    return float(labels.time_ms) / 1000.0


def t_ref_by_task(rows: BundleRows | Mapping[str, Any], participant_key: str) -> dict[str, float]:
    """``T_ref`` per task (section 21.2): the median completed time of the participant's session-1
    ``assistance_off`` trials of the task; else of all their ``assistance_off`` trials of the task, probes
    included; else the physics median for ``assistance_off`` on the participant's layouts of that task."""
    rows = read_bundle(rows)
    data = _participant_rows(rows)[participant_key]
    out: dict[str, float] = {}
    first = next((s.session_id for s in data.sessions if s.session_number == 1), None)
    for task in ("pointing", "menu"):
        session_one: list[float] = []
        everywhere: list[float] = []
        physics: list[float] = []
        for session in data.sessions:
            for stored in data.trials_by_session.get(session.session_id, ()):
                trial = stored.trial
                if trial.task != task or trial.interrupted or trial.intended_index is None:
                    continue
                physics.append(physics_reference_time_s(trial, trial.modality))
                if trial.applied_regime != "assistance_off" or trial.revert is not None:
                    continue
                time_s = _completed_time_s(stored)
                if time_s is None:
                    continue
                everywhere.append(time_s)
                if session.session_id == first:
                    session_one.append(time_s)
        for values in (session_one, everywhere, physics):
            if values:
                out[task] = float(statistics.median(values))
                break
    return out


def _burden_means(
    reports: Sequence[BurdenReport], regime_of_block: Mapping[tuple[str, int], str]
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, dict[str, float]]]:
    """Per (session, regime) and per regime means of effort and intrusiveness (imputation sources, 21.2)."""
    session_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )
    all_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for report in reports:
        if report.scope != "block" or report.block_index is None:
            continue
        regime = regime_of_block.get((report.session_id, report.block_index))
        if regime is None:
            continue
        for item in ("effort", "intrusiveness"):
            value = getattr(report, item)
            if value is not None:
                session_values[(report.session_id, regime)][item].append(float(value))
                all_values[regime][item].append(float(value))
    return _item_means(session_values), _item_means(all_values)


def _item_means(values: Mapping[Any, Mapping[str, Sequence[float]]]) -> dict[Any, dict[str, float]]:
    out: dict[Any, dict[str, float]] = {}
    for key, items in values.items():
        out[key] = {item: float(np.mean(v)) for item, v in items.items()}
    return out


@dataclass(frozen=True, slots=True)
class BlockBurden:
    value: float  # B_obs (unweighted)
    imputed: tuple[str, ...]  # the items that were imputed


def block_burden(
    session_id: str,
    block_index: int,
    regime: str,
    reports: Sequence[BurdenReport],
    regime_of_block: Mapping[tuple[str, int], str],
) -> BlockBurden:
    """``B_obs`` of one block with the imputation order of section 21.2 (same regime in the same session, then
    across sessions, then the scale midpoint 4; intrusiveness 1 for ``assistance_off``)."""
    session_means, all_means = _burden_means(reports, regime_of_block)
    block_reports = [r for r in reports if r.scope == "block" and r.session_id == session_id]
    report = next((r for r in block_reports if r.block_index == block_index), None)
    effort = None if report is None else report.effort
    intrusiveness = None if report is None else report.intrusiveness
    imputed = []
    if effort is None:
        imputed.append("effort")
    if intrusiveness is None and regime != "assistance_off":
        imputed.append("intrusiveness")
    value = block_burden_term(
        effort,
        intrusiveness,
        regime=regime,
        session_means=session_means.get((session_id, regime)),
        all_means=all_means.get(regime),
    )
    return BlockBurden(value, tuple(imputed))


def weights_for(decision: DecisionRecord | None, kind: WeightsKind) -> UtilityWeights:
    """The block's utility weights: the priorities logged with its L1 decision (primary) or the protocol weights."""
    if kind == "protocol" or decision is None:
        return PROTOCOL_WEIGHTS
    priorities = decision.payload.get("priorities")
    if not isinstance(priorities, Mapping):
        return PROTOCOL_WEIGHTS
    return UtilityWeights.from_mapping(priorities)


def weighted_burden(value: float, weights: UtilityWeights) -> float:
    """``w_B * B_obs``: the burden term in utility units."""
    return weights.w_b * value


def trial_utility(stored: StoredTrial, *, t_ref_s: float, weights: UtilityWeights) -> float | None:
    """Realized utility of one scored trial without the burden term (no switching penalty); None without labels."""
    labels = stored.labels
    if labels is None or stored.trial.interrupted:
        return None
    time_ms = labels.time_ms if labels.complete else stored.trial.limit_ms
    time_s = float(time_ms) / 1000.0
    return realized_utility(
        labels,
        time_s=time_s,
        t_ref_s=t_ref_s,
        burden_term=0.0,
        weights=weights,
    )


def _main_trials(trials: Iterable[StoredTrial]) -> list[StoredTrial]:
    return [stored for stored in trials if stored.trial.task != "idle"]


def regime_of_blocks(rows: BundleRows) -> dict[tuple[str, int], str]:
    """(session id, block index) -> the regime applied at the block's first trial."""
    out: dict[tuple[str, int], str] = {}
    for stored in sorted(rows.trials, key=lambda s: (s.trial.session_id, s.trial.seq)):
        trial = stored.trial
        if trial.task == "idle":
            continue
        out.setdefault((trial.session_id, trial.block_index), trial.applied_regime)
    return out


def reconstruct_block_context(
    data: _ParticipantRows, session: SessionRow, block_index: int, decision: DecisionRecord
) -> BlockContext:
    """The BlockContext of a logged L1 decision, rebuilt from stored rows: the block's trial layouts, the session's
    device and day, the minutes since the session started, today's earlier trials, the previous block's regime and
    hand quality. The same reconstruction serves the frozen target and the simulated truth."""
    plan = _block_plan(session, block_index)
    trials = [
        stored.trial
        for stored in data.trials_by_session.get(session.session_id, ())
        if stored.trial.block_index == block_index and stored.trial.task != "idle"
    ]
    trials.sort(key=lambda trial: trial.trial_index)
    layouts = tuple(trial.layout for trial in trials)
    earlier = [
        stored.trial
        for stored in data.trials_by_session.get(session.session_id, ())
        if stored.trial.block_index < block_index and stored.trial.task != "idle"
    ]
    previous_regime = None
    hand_quality = None
    if earlier:
        last_block = max(trial.block_index for trial in earlier)
        last = [trial for trial in earlier if trial.block_index == last_block]
        previous_regime = last[-1].applied_regime
        qualities = [trial.hand.quality_mean for trial in last if trial.hand is not None]
        if qualities and session.device.modality == "hand_pointer":
            hand_quality = float(np.mean(qualities))
    trials_today = 0
    previous_start = None
    for other in data.sessions:
        if other.started_at < session.started_at:
            previous_start = other.started_at
        if other.day_index != session.day_index:
            continue
        for stored in data.trials_by_session.get(other.session_id, ()):
            if stored.trial.task != "idle" and stored.trial.recorded_at < decision.created_at:
                trials_today += 1
    days_since = None
    if previous_start is not None:
        days_since = (session.started_at - previous_start).total_seconds() / 86400.0
    clock = ClockContext(
        day_index=session.day_index,
        session_minute=max(0.0, (decision.created_at - session.started_at).total_seconds() / 60.0),
        trials_today=trials_today,
        days_since_last_session=days_since,
    )
    return BlockContext(
        session_id=session.session_id,
        block_index=block_index,
        phase=plan.phase,
        task=plan.task,
        layouts=layouts,
        device=session.device,
        clock=clock,
        previous_regime=previous_regime,
        hand_quality_recent=hand_quality,
    )


def offer_context(trial: Any, session: SessionRow) -> OfferContext:
    """The OfferContext of a logged offer, rebuilt from its trial (the struggle kind is inferred: a hand-pointer
    trial with two tracking losses fired on tracking loss, every other one on corrections)."""
    hand = trial.hand
    struggle = "tracking_loss" if hand is not None and hand.tracking_losses >= 2 else "corrections"
    modality = session.device.modality
    return OfferContext(
        session_id=session.session_id,
        block_index=trial.block_index,
        trial_index=trial.trial_index,
        modality=modality,
        task=trial.task,
        struggle_kind=struggle,
        keyboard_available=modality != "keyboard",
    )


def session_exclusions(decision: DecisionRecord) -> dict[str, str]:
    """The C1, C5, C6 and C8 exclusions logged with a decision (for a table, the union over its cells)."""
    payload = decision.payload
    sources = [payload.get("excluded") or {}]
    sources += [cell.get("excluded") or {} for cell in payload.get("cells", ())]
    out: dict[str, str] = {}
    for excluded in sources:
        for variant, reason in excluded.items():
            if str(reason) in SESSION_EXCLUSION_REASONS:
                out.setdefault(str(variant), str(reason))
    return out


def _l1_step(decision: DecisionRecord, block_log: BlockLog) -> LoggedStep:
    payload = decision.payload
    if decision.behavior == "baseline":
        applied = decision.applied or block_log.applied_regime
        return LoggedStep(
            DecisionLevel.BLOCK_REGIME,
            "",
            None,
            applied,
            {applied: 1.0},
            {"decision": decision},
        )
    propensities = {str(k): float(v) for k, v in payload["propensities"].items()}
    return LoggedStep(
        DecisionLevel.BLOCK_REGIME,
        "",
        None,
        str(payload["variant"]),
        propensities,
        {"decision": decision},
    )


def _remaining_arms(decision: DecisionRecord) -> tuple[str, ...]:
    payload = decision.payload
    remaining = payload.get("remaining_arms")
    if isinstance(remaining, list | tuple) and remaining:
        return tuple(str(arm) for arm in remaining)
    return tuple(str(arm) for arm, p in payload["propensities"].items() if float(p) > 0.0)


def _table_steps(
    trials: Sequence[StoredTrial],
    decisions: Mapping[str, DecisionRecord],
) -> list[LoggedStep]:
    steps: list[LoggedStep] = []
    for stored in trials:
        trial = stored.trial
        for activation in trial.activations:
            applied = activation.l2
            if applied is None or applied.source != "table" or applied.draw_index >= 4:
                continue  # fixed-rule, fallback, cap and after-max responses are deterministic steps
            table = decisions.get(applied.decision_id)
            if table is None or (table.behavior == "baseline" and table.fallback_reason != "stale_table"):
                # A table relabelled for a stale_table fallback is the one requested after a veto: the trials that
                # applied its draws once it arrived ran under its logged propensities (the service keeps it policy
                # when the same trial applied it).
                continue
            cell = next(
                (c for c in table.payload.get("cells", ()) if c["cell_id"] == applied.cell_id),
                None,
            )
            if cell is None:
                raise ValueError("an applied table response names a cell its table does not have")
            propensities = {str(k): float(v) for k, v in cell["propensities"].items()}
            if propensities.get(applied.variant, 0.0) >= 1.0:
                continue  # a deterministic cell (fixed rule or pin_act collapse)
            steps.append(
                LoggedStep(
                    DecisionLevel.AMBIGUITY_RESPONSE,
                    applied.cell_id,
                    trial.trial_index,
                    applied.variant,
                    propensities,
                    {"decision": table},
                )
            )
    return steps


def _offer_steps(
    trials: Sequence[StoredTrial], decisions: Mapping[str, DecisionRecord], session: SessionRow
) -> list[LoggedStep]:
    steps: list[LoggedStep] = []
    for stored in trials:
        trial = stored.trial
        for offer in trial.offers:
            decision = decisions.get(offer.decision_id)
            if decision is None:
                continue
            if decision.behavior == "baseline" or offer.fallback_reason is not None:
                chosen = offer.variant
                propensities = {chosen: 1.0}
            else:
                chosen = str(decision.payload["variant"])
                logged = decision.payload["propensities"]
                propensities = {str(k): float(v) for k, v in logged.items()}
            if propensities.get(chosen, 0.0) >= 1.0 and decision.behavior != "baseline":
                continue  # probe and pinned blocks offer nothing: a deterministic step
            steps.append(
                LoggedStep(
                    DecisionLevel.OFFER,
                    "",
                    trial.trial_index,
                    chosen,
                    propensities,
                    {"decision": decision, "offer": offer_context(trial, session)},
                )
            )
    return steps


def logged_blocks(
    bundle: Mapping[str, Any] | BundleRows,
    *,
    weights: WeightsKind,
    phases: Sequence[str] = BLOCK_PHASES,
    sessions: Sequence[int] | None = None,
) -> list[LoggedBlock]:
    """The block-level logs (section 19): Phase A and A-lite blocks with design-based propensities, one unit per
    L1 decision, each with its per-decision steps, scored trials and burden term.

    ``sessions`` restricts the units to those session numbers (``S_2`` in sample: sessions 1-2). A decision
    relabelled ``baseline`` (UI fallback) enters with propensity 1 for the applied regime. Rewards use the priorities
    logged with the block's L1 decision (``weights="participant"``) or the protocol weights (``"protocol"``).
    """
    rows = read_bundle(bundle)
    regime_of_block = regime_of_blocks(rows)
    units: list[LoggedBlock] = []
    for key, data in _participant_rows(rows).items():
        t_ref = t_ref_by_task(rows, key)
        for session in data.sessions:
            if sessions is not None and session.session_number not in sessions:
                continue
            session_trials = data.trials_by_session.get(session.session_id, ())
            reports = data.burden_by_session.get(session.session_id, ())
            all_reports = _all_reports(data)
            by_block: dict[int, list[StoredTrial]] = defaultdict(list)
            for stored in _main_trials(session_trials):
                by_block[stored.trial.block_index].append(stored)
            for block_index, block_trials in sorted(by_block.items()):
                plan = _block_plan(session, block_index)
                if plan.phase not in phases:
                    continue
                block_trials.sort(key=lambda s: s.trial.trial_index)
                decision = data.decisions.get(block_trials[0].trial.regime_decision_id)
                if decision is None:
                    continue
                unit = _block_unit(
                    data,
                    session,
                    plan,
                    block_trials,
                    decision,
                    reports,
                    all_reports,
                    regime_of_block,
                    t_ref,
                    weights,
                )
                if unit is not None:
                    units.append(unit)
    return units


def _block_unit(
    data: _ParticipantRows,
    session: SessionRow,
    plan: Any,
    block_trials: Sequence[StoredTrial],
    decision: DecisionRecord,
    reports: Sequence[BurdenReport],
    all_reports: Sequence[BurdenReport],
    regime_of_block: Mapping[tuple[str, int], str],
    t_ref: Mapping[str, float],
    weights_kind: WeightsKind,
) -> LoggedBlock | None:
    applied = block_trials[0].trial.applied_regime
    weights = weights_for(decision, weights_kind)
    utilities: list[float] = []
    positions: list[int] = []
    for stored in block_trials:
        if stored.trial.interrupted or stored.labels is None:
            continue
        task_ref = t_ref.get(stored.trial.task)
        if task_ref is None:
            continue
        value = trial_utility(stored, t_ref_s=task_ref, weights=weights)
        if value is None:
            continue
        utilities.append(value)
        positions.append(stored.trial.trial_index)
    if not utilities:
        return None
    burden = block_burden(session.session_id, plan.index, applied, all_reports, regime_of_block)
    burden_term = weighted_burden(burden.value, weights)
    block_log = BlockLog(
        participant_key=session.participant_key,
        session_id=session.session_id,
        session_number=session.session_number,
        block_index=plan.index,
        phase=plan.phase,
        task=plan.task,
        decision_id=decision.decision_id,
        created_at=decision.created_at,
        block=reconstruct_block_context(data, session, plan.index, decision),
        session_exclusions=session_exclusions(decision),
        applied_regime=applied,
    )
    steps = [
        _l1_step(decision, block_log),
        *_table_steps(block_trials, data.decisions),
        *_offer_steps(block_trials, data.decisions, session),
    ]
    return LoggedBlock(
        unit_id=decision.decision_id,
        group_id=session.participant_key,
        session_id=session.session_id,
        context=block_log,
        remaining_arms=_remaining_arms(decision),
        steps=tuple(steps),
        trial_utilities=tuple(utilities),
        trial_positions=tuple(positions),
        burden_term=burden_term,
        reward=float(np.mean(utilities)) - burden_term,
        behavior=decision.behavior,
    )


def logged_l2_trials(
    bundle: Mapping[str, Any] | BundleRows,
    *,
    weights: WeightsKind,
    phases: Sequence[str] = L2_UNIT_PHASES,
) -> list[LoggedBlock]:
    """Trial-level L2 units: every scored trial of an act-guarded Phase A, A-lite or comparator block, weighted by
    the product over its own randomized L2 draws (comparator blocks contribute their uniform draws); the reward is
    the trial's realized utility, burden term included."""
    rows = read_bundle(bundle)
    regime_of_block = regime_of_blocks(rows)
    units: list[LoggedBlock] = []
    for key, data in _participant_rows(rows).items():
        t_ref = t_ref_by_task(rows, key)
        all_reports = _all_reports(data)
        for session in data.sessions:
            for stored in _main_trials(data.trials_by_session.get(session.session_id, ())):
                trial = stored.trial
                plan = _block_plan(session, trial.block_index)
                if plan.phase not in phases or trial.applied_regime not in _ACT_GUARDED:
                    continue
                decision = data.decisions.get(trial.regime_decision_id)
                task_ref = t_ref.get(trial.task)
                if decision is None or task_ref is None:
                    continue
                weights_ = weights_for(decision, weights)
                value = trial_utility(stored, t_ref_s=task_ref, weights=weights_)
                if value is None:
                    continue
                burden = block_burden(
                    session.session_id,
                    plan.index,
                    trial.applied_regime,
                    all_reports,
                    regime_of_block,
                )
                burden_term = weighted_burden(burden.value, weights_)
                block_log = BlockLog(
                    participant_key=session.participant_key,
                    session_id=session.session_id,
                    session_number=session.session_number,
                    block_index=plan.index,
                    phase=plan.phase,
                    task=plan.task,
                    decision_id=decision.decision_id,
                    created_at=decision.created_at,
                    block=reconstruct_block_context(data, session, plan.index, decision),
                    session_exclusions=session_exclusions(decision),
                    applied_regime=trial.applied_regime,
                )
                units.append(
                    LoggedBlock(
                        unit_id=trial.trial_id,
                        group_id=session.participant_key,
                        session_id=session.session_id,
                        context=block_log,
                        remaining_arms=(),
                        steps=tuple(_table_steps([stored], data.decisions)),
                        trial_utilities=(value,),
                        trial_positions=(trial.trial_index,),
                        burden_term=burden_term,
                        reward=value - burden_term,
                        behavior="policy",
                    )
                )
    return units


_ACT_GUARDED = frozenset({"act", "enlarge_target:1.5", "enlarge_target:2.0"})


# ------------------------------------------------------------------------------------------------ frozen targets


def rows_hash(ids: Iterable[str], learning_reset_at: datetime | None = None) -> str:
    """sha256 over the sorted row ids (and the learning-reset marker) that a snapshot was rebuilt from."""
    digest = hashlib.sha256()
    for row_id in sorted(ids):
        digest.update(row_id.encode("utf-8"))
        digest.update(b"\n")
    digest.update(
        b"" if learning_reset_at is None else learning_reset_at.isoformat().encode("utf-8"),
    )
    return digest.hexdigest()


def ope_seed(rows_hash_value: str, decision_id: str) -> int:
    """``int.from_bytes(sha256(f"ope|{rows_hash}|{decision_id}").digest()[:8], "big")`` (section 19)."""
    return int.from_bytes(
        hashlib.sha256(f"ope|{rows_hash_value}|{decision_id}".encode()).digest()[:8],
        "big",
    )


class FrozenTarget:
    """A frozen snapshot as a target policy: adaptive-mode probabilities from
    ``CoAdaptationEngine.policy_probabilities`` with 4,096 Thompson draws seeded per evaluated decision, under the
    preferences in force at that decision and the session-state exclusions logged with it. Never learns.

    One ``policy_probabilities`` call returns a block's L1 probabilities, every cell of its L2 table and, given an
    offer context, the L3 probabilities of that offer, so the block's L1 decision is the evaluated decision for all
    three, seeded by its id; the call carries the block's first offer context. The L3 probabilities depend on the
    offer's task, struggle kind and keyboard availability and on the offer cooldown only, so the other offers of
    the block share them unless one of those differs. A table requested after a veto carries the veto cooldown in its
    cells, and an offer under the offer cooldown, and each is evaluated with its own id, time and exclusions."""

    def __init__(
        self,
        snapshot: bytes,
        *,
        participant_key: str,
        settings: EngineSettings,
        preferences_history: Sequence[Preference],
        rows_hash: str,
        draws: int = OPE_DRAWS,
    ) -> None:
        self._history = PreferenceSet(preferences_history)
        self._engine = CoAdaptationEngine.restore(
            snapshot, participant_key=participant_key, settings=settings, preferences=self._history
        )
        self._rows_hash = rows_hash
        self._draws = draws
        self._cache: dict[str, tuple[PolicyProbabilities, tuple[str, str, bool] | None]] = {}

    @property
    def rows_hash(self) -> str:
        return self._rows_hash

    def snapshot(self) -> bytes:
        """The restored engine's learned state; equal to the evaluated snapshot, since a target never learns."""
        return self._engine.snapshot()

    def probabilities(
        self,
        context: BlockLog,
        *,
        decision: DecisionRecord | None = None,
        offer: OfferContext | None = None,
    ) -> PolicyProbabilities:
        """The target's L1, L2 (per cell) and, with ``offer``, L3 probabilities at a logged block.

        ``decision`` is the evaluated decision (the block's L1 decision by default): its id seeds the Thompson
        draws, its time selects the preferences in force and the diffusion of the snapshot's filters, and the
        session-state exclusions logged with it are passed on with its level, so a logged "Not this time" excludes
        its variant at that level only, as live (C5). The L1, L2 and L3 draws come from one stream in that order, so
        passing an offer context never changes the L1 and L2 probabilities of the same decision.
        """
        if decision is None:
            decision_id, now, exclusions, level = (
                context.decision_id,
                context.created_at,
                dict(context.session_exclusions),
                DecisionLevel.BLOCK_REGIME,
            )
        else:
            decision_id, now, exclusions, level = (
                decision.decision_id,
                decision.created_at,
                session_exclusions(decision),
                DecisionLevel(decision.level),
            )
        cached = self._cache.get(decision_id)
        if cached is not None and (offer is None or cached[1] == _offer_key(offer)):
            return cached[0]
        probs = self._engine.policy_probabilities(
            block=context.block,
            session_exclusions=exclusions,
            preferences=self._history.as_of(now),
            seed=ope_seed(self._rows_hash, decision_id),
            draws=self._draws,
            now=now,
            offer=offer,
            exclusions_level=level,
        )
        if cached is None or offer is not None:
            self._cache[decision_id] = (probs, None if offer is None else _offer_key(offer))
        return probs

    def __call__(self, block: LoggedBlock, step: LoggedStep) -> Mapping[str, float]:
        context: BlockLog = block.context
        step_context = step.context if isinstance(step.context, Mapping) else {}
        decision = step_context.get("decision")
        first_offer = _first_offer(block)
        if _is_level(step, DecisionLevel.BLOCK_REGIME):
            return self.probabilities(context, offer=first_offer).l1
        if _is_level(step, DecisionLevel.AMBIGUITY_RESPONSE):
            if decision is not None and _has_veto_cooldown(decision):
                return self.probabilities(context, decision=decision).l2.get(step.key) or _zeros(
                    L2_OPTIONS,
                )
            return self.probabilities(context, offer=first_offer).l2.get(step.key) or _zeros(
                L2_OPTIONS,
            )
        offer = step_context.get("offer")
        if decision is not None and "offer_cooldown" in session_exclusions(decision).values():
            return self.probabilities(context, decision=decision, offer=offer).l3 or {}
        other_offer = offer is not None and first_offer is not None
        if other_offer and _offer_key(offer) != _offer_key(first_offer):
            return self.probabilities(context, decision=decision, offer=offer).l3 or {}
        return self.probabilities(context, offer=first_offer).l3 or {}


def _zeros(options: Sequence[str]) -> dict[str, float]:
    return {option: 0.0 for option in options}


def _offer_key(offer: OfferContext) -> tuple[str, str, bool]:
    """The parts of an offer context the target's L3 probabilities depend on."""
    return (offer.task, offer.struggle_kind, offer.keyboard_available)


def _first_offer(block: LoggedBlock) -> OfferContext | None:
    for step in block.steps:
        if _is_level(step, DecisionLevel.OFFER) and isinstance(step.context, Mapping):
            offer = step.context.get("offer")
            if offer is not None:
                return offer
    return None


def _has_veto_cooldown(decision: DecisionRecord) -> bool:
    return any(
        str(reason) == "recent_veto"
        for cell in decision.payload.get("cells", ())
        for reason in (cell.get("excluded") or {}).values()
    )


def frozen_target(
    snapshot: bytes,
    *,
    participant_key: str,
    settings: EngineSettings,
    preferences_history: Sequence[Preference],
    rows_hash: str,
) -> TargetPolicy:
    """The frozen snapshot ``snapshot`` as a target policy (section 19). ``settings`` are
    ``EngineSettings(**row.settings)`` of the evaluated session, so the target decides with that session's tau,
    epsilon and filter settings. Restoring and evaluating leave ``snapshot`` unchanged; the result is deterministic."""
    return FrozenTarget(
        snapshot,
        participant_key=participant_key,
        settings=settings,
        preferences_history=preferences_history,
        rows_hash=rows_hash,
    )


def mixed_with_uniform(probs: Mapping[str, float], weight: float = 0.5) -> dict[str, float]:
    """``weight * pi + (1 - weight) * uniform`` over the support of ``pi`` (the step's safe set)."""
    support = [option for option, p in probs.items() if float(p) > 0.0]
    if not support:
        return dict(probs)
    uniform = 1.0 / len(support)
    return {
        option: weight * float(p) + ((1.0 - weight) * uniform if option in support else 0.0)
        for option, p in probs.items()
    }


class MixtureTarget:
    """The estimator-check target of section 19: half a frozen snapshot, half uniform over each step's safe set."""

    def __init__(self, base: FrozenTarget, weight: float = 0.5) -> None:
        self._base = base
        self._weight = weight

    def probabilities(
        self,
        context: BlockLog,
        *,
        decision: DecisionRecord | None = None,
        offer: OfferContext | None = None,
    ) -> PolicyProbabilities:
        probs = self._base.probabilities(context, decision=decision, offer=offer)
        return PolicyProbabilities(
            l1=mixed_with_uniform(probs.l1, self._weight),
            l2={c: mixed_with_uniform(vector, self._weight) for c, vector in probs.l2.items()},
            l3=None if probs.l3 is None else mixed_with_uniform(probs.l3, self._weight),
        )

    def __call__(self, block: LoggedBlock, step: LoggedStep) -> Mapping[str, float]:
        return mixed_with_uniform(self._base(block, step), self._weight)


# ------------------------------------------------------------------------------------- snapshots from stored rows


def replay_rows(rows: BundleRows, participant_key: str) -> list[Any]:
    """The participant's retained rows in replay order (section 10.8): sessions by (started_at, session_id), and
    within a session trials and burden reports by ``seq`` with each event right after the trial it names."""
    events_by_trial: dict[str, list[EngineEvent]] = defaultdict(list)
    for event in rows.events:
        events_by_trial[event.trial_id].append(event)
    data = _participant_rows(rows).get(participant_key)
    if data is None:
        return []
    ordered: list[Any] = []
    for session in data.sessions:
        if rows.learning_reset_at is not None and session.started_at < rows.learning_reset_at:
            continue
        ordered.append(session)
        session_trials = data.trials_by_session.get(session.session_id, ())
        items: list[tuple[int, Any]] = [(s.trial.seq, s) for s in session_trials]
        items += [(r.seq, r) for r in data.burden_by_session.get(session.session_id, ())]
        for _, item in sorted(items, key=lambda pair: pair[0]):
            ordered.append(item)
            if isinstance(item, StoredTrial):
                ordered.extend(events_by_trial.get(item.trial.trial_id, ()))
    return ordered


def _row_id(row: Any) -> str:
    if isinstance(row, SessionRow):
        return row.session_id
    if isinstance(row, StoredTrial):
        return row.trial.trial_id
    if isinstance(row, BurdenReport):
        return row.report_id
    return row.event_id


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A frozen snapshot rebuilt from stored rows (``S_k``)."""

    data: bytes
    settings: EngineSettings  # EngineSettings(**row.settings) of session k
    rows_hash: str
    session_number: int


def snapshot_before_phase_b(
    rows: BundleRows | Mapping[str, Any],
    participant_key: str,
    session_number: int,
    *,
    preferences: Sequence[Preference] = (),
) -> Snapshot | None:
    """``S_k``: the engine rebuilt from the participant's rows up to the first Phase B decision of session ``k``
    (every row of sessions 1..k-1 and the rows of session k recorded before that decision), with learning then
    switched off. None when session ``k`` has no Phase B decision."""
    rows = read_bundle(rows)
    data = _participant_rows(rows).get(participant_key)
    if data is None:
        return None
    target = next((s for s in data.sessions if s.session_number == session_number), None)
    if target is None:
        return None
    phase_b = {plan.index for plan in target.plan.blocks if plan.phase in _PHASE_B}
    first_b: DecisionRecord | None = None
    for stored in data.trials_by_session.get(target.session_id, ()):
        if stored.trial.block_index in phase_b:
            decision = data.decisions.get(stored.trial.regime_decision_id)
            if decision is not None and (first_b is None or decision.seq < first_b.seq):
                first_b = decision
    if first_b is None:
        return None
    engine = CoAdaptationEngine(
        participant_key=participant_key,
        settings=EngineSettings(**target.settings),
        preferences=PreferenceSet(preferences),
    )
    used: list[str] = []
    current: SessionRow | None = None
    for row in replay_rows(rows, participant_key):
        if isinstance(row, SessionRow):
            if current is not None and current.ended_at is not None:
                engine.end_session(
                    ended_at=current.ended_at,
                    reason=current.end_reason or "completed",
                )
            if row.started_at > target.started_at:
                break
            current = row
            engine.begin_session(row)
        elif isinstance(row, StoredTrial):
            if row.trial.session_id == target.session_id and row.trial.seq > first_b.seq:
                break
            engine.replay_trial(row)
        elif isinstance(row, BurdenReport):
            if row.session_id == target.session_id and row.seq > first_b.seq:
                break
            engine.observe_burden(row)
        else:
            engine.apply_event(row)
        used.append(_row_id(row))
    return Snapshot(
        data=engine.snapshot(),
        settings=EngineSettings(**target.settings),
        rows_hash=rows_hash(used, rows.learning_reset_at),
        session_number=session_number,
    )
