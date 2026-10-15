"""Session plans of the Selection Study and the comparator ranking (section 12.3 of the co-adaptation spec).

Session 1 (or any session without adaptive assistance): a practice block and 12 Phase A blocks in permuted rounds.
Later sessions with adaptive assistance: a practice block, one A-lite round and 10 Phase B blocks in two halves, each
half with three adaptive blocks, one probe block (``assistance_off``) and one comparator block (the participant's
best fixed assisted regime from session 1) at seeded, counterbalanced positions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from .preferences import caution_of
from .priors import l1_physics_update
from .utility import PROTOCOL_WEIGHTS, UtilityWeights, block_burden_term, realized_utility
from .vocabulary import ASSISTANCE_OFF, l1_arms

if TYPE_CHECKING:
    from .records import BurdenReport, DecisionRecord, StoredTrial

STUDY_PROTOCOL_ID = "coadapt-study-v1"
PLAN_RULES_VERSION = "plan-v1"
PROTOCOL_HYPERPARAMETERS: dict[str, float | int] = {
    "ambiguity_threshold": 0.25,
    "unintended_activation_tolerance": 0.05,
    "noninferiority_margin": 0.01,
    "upper_bound_credibility": 0.95,
    "exploration_floor": 0.1,
    "switching_penalty": 0.1,
    "thompson_draws": 256,
}
LAYOUT_GRID: dict[str, Any] = {
    "pointing": {"w_px": [16, 28, 44], "gap_px": [2, 8], "d_px": [120, 240, 360], "grids": ["4x3", "6x2"]},
    "menu": {"n_items": [6, 12, 16], "item_h_px": 24, "item_w_px": 220, "button_h_px": 32},
    "fit_px": [640, 420],
}
BLOCK_TRIALS = 8
PHASE_A_BLOCKS = 12
PHASE_B_BLOCKS = 10
PHASE_A_IDLE_AFTER = (4, 10)  # main block indices followed by an idle epoch (session 1)
MIN_RANKING_TRIALS = 8

PHASES = ("practice", "A", "A_lite", "B", "probe", "comparator")
TASKS = ("pointing", "menu", "idle")


@dataclass(frozen=True, slots=True)
class BlockPlan:
    index: int
    phase: Literal["practice", "A", "A_lite", "B", "probe", "comparator"]
    task: Literal["pointing", "menu", "idle"]
    trials: int
    idle_after: bool
    burden_after: bool

    def __post_init__(self) -> None:
        if self.phase not in PHASES or self.task not in TASKS:
            raise ValueError("unknown phase or task")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "phase": self.phase,
            "task": self.task,
            "trials": self.trials,
            "idle_after": self.idle_after,
            "burden_after": self.burden_after,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        _check_keys(d, ("index", "phase", "task", "trials", "idle_after", "burden_after"), "BlockPlan")
        for name in ("index", "trials"):
            if not isinstance(d[name], int) or isinstance(d[name], bool):
                raise ValueError(f"BlockPlan.{name}: expected int")
        for name in ("idle_after", "burden_after"):
            if not isinstance(d[name], bool):
                raise ValueError(f"BlockPlan.{name}: expected bool")
        if d["phase"] not in PHASES or d["task"] not in TASKS:
            raise ValueError("BlockPlan: unknown phase or task")
        return cls(**{key: d[key] for key in ("index", "phase", "task", "trials", "idle_after", "burden_after")})


@dataclass(frozen=True, slots=True)
class SessionPlan:
    session_number: int
    blocks: tuple[BlockPlan, ...]
    end_questionnaire: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_number": self.session_number,
            "end_questionnaire": self.end_questionnaire,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        _check_keys(d, ("session_number", "end_questionnaire", "blocks"), "SessionPlan")
        if not isinstance(d["session_number"], int) or isinstance(d["session_number"], bool):
            raise ValueError("SessionPlan.session_number: expected int")
        if not isinstance(d["end_questionnaire"], bool):
            raise ValueError("SessionPlan.end_questionnaire: expected bool")
        if not isinstance(d["blocks"], list | tuple):
            raise ValueError("SessionPlan.blocks: expected a list")
        return cls(
            session_number=d["session_number"],
            blocks=tuple(BlockPlan.from_dict(block) for block in d["blocks"]),
            end_questionnaire=d["end_questionnaire"],
        )

    def block(self, index: int) -> BlockPlan:
        for block in self.blocks:
            if block.index == index:
                return block
        raise KeyError(index)

    @property
    def active_trials(self) -> int:
        return sum(block.trials for block in self.blocks)


def _check_keys(d: Any, keys: Sequence[str], record: str) -> None:
    if not isinstance(d, Mapping):
        raise ValueError(f"{record}: expected an object")
    unknown = sorted(str(key) for key in d if key not in keys)
    if unknown:
        raise ValueError(f"{record}: unknown keys {unknown}")
    missing = sorted(key for key in keys if key not in d)
    if missing:
        raise ValueError(f"{record}: missing keys {missing}")


def plan_seed(participant_key: str, session_number: int) -> int:
    digest = hashlib.sha256(f"{participant_key}|{session_number}|{PLAN_RULES_VERSION}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _task(index: int) -> Literal["pointing", "menu"]:
    return "pointing" if index % 2 == 0 else "menu"


def phase_b_positions(*, n_arms: int) -> tuple[tuple[int, int, int, int], ...]:
    """Every valid (probe, comparator) position pair of the two halves, as (p1, c1, p2, c2), sorted.

    Positions run 2-5 within each half and are distinct inside a half; across the halves the probe falls once on a
    pointing and once on a menu block, and so does the comparator.
    """
    first = n_arms + 1  # block index of B1
    plans = []
    for p1, c1, p2, c2 in itertools.product(range(2, 6), repeat=4):
        if p1 == c1 or p2 == c2:
            continue
        probe_tasks = {_task(first + p1 - 1), _task(first + 5 + p2 - 1)}
        comparator_tasks = {_task(first + c1 - 1), _task(first + 5 + c2 - 1)}
        if probe_tasks == {"pointing", "menu"} and comparator_tasks == {"pointing", "menu"}:
            plans.append((p1, c1, p2, c2))
    return tuple(sorted(plans))


def build_session_plan(
    *,
    session_number: int,
    adaptive_allowed: bool,
    comparator_available: bool,
    plan_seed: int,
    burden_every_blocks: int,
    n_arms: int = 6,
    block_trials: int = BLOCK_TRIALS,
) -> SessionPlan:
    """The block sequence of one session (section 12.3).

    ``n_arms`` is ``len(l1_arms(modality))`` (6, or 4 for keyboard). Tasks alternate by block index (even pointing,
    odd menu; the practice block 0 is mixed and labelled pointing). ``burden_after`` falls on every main block whose
    index is a multiple of ``burden_every_blocks``.
    """
    if session_number < 1:
        raise ValueError("session_number starts at 1")
    if not 1 <= burden_every_blocks <= 8:
        raise ValueError("burden_every_blocks lies in [1, 8]")
    if n_arms < 1 or block_trials < 1:
        raise ValueError("n_arms and block_trials must be positive")

    phases: list[str] = []
    idle_after: set[int] = set()
    if session_number == 1 or not adaptive_allowed:
        phases = ["A"] * PHASE_A_BLOCKS
        idle_after = set(PHASE_A_IDLE_AFTER)
    else:
        phases = ["A_lite"] * n_arms
        b_phases = ["B"] * PHASE_B_BLOCKS
        options = phase_b_positions(n_arms=n_arms)
        rng = np.random.Generator(np.random.PCG64(plan_seed))
        p1, c1, p2, c2 = options[int(rng.integers(0, len(options)))]
        b_phases[p1 - 1] = "probe"
        b_phases[5 + p2 - 1] = "probe"
        if comparator_available:
            b_phases[c1 - 1] = "comparator"
            b_phases[5 + c2 - 1] = "comparator"
        phases.extend(b_phases)
        idle_after = {3, n_arms + 6}

    blocks = [BlockPlan(0, "practice", "pointing", block_trials, False, False)]
    for index, phase in enumerate(phases, start=1):
        blocks.append(
            BlockPlan(
                index=index,
                phase=phase,  # type: ignore[arg-type]
                task=_task(index),
                trials=block_trials,
                idle_after=index in idle_after,
                burden_after=index % burden_every_blocks == 0,
            )
        )
    return SessionPlan(session_number=session_number, blocks=tuple(blocks), end_questionnaire=True)


def study_protocol_sha256() -> str:
    """sha256 of the protocol id, hyperparameters, block size, plan rules version and layout grid."""
    payload = {
        "protocol_id": STUDY_PROTOCOL_ID,
        "hyperparameters": PROTOCOL_HYPERPARAMETERS,
        "block_trials": BLOCK_TRIALS,
        "plan_rules_version": PLAN_RULES_VERSION,
        "layout_grid": LAYOUT_GRID,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------- comparator ranking


def physics_reference_time_s(trial: Any, modality: str) -> float:
    """Physics median time (s) of an ``assistance_off`` trial on this trial's layout and target (section 7.3), for a
    novice (menu expertise 0) on the first trial (``M_p`` at ``n_variant = 0``).

    This is the update-mode prior of the outcome model itself (``priors.l1_physics_update``), so the fallback T_ref
    uses the same constants and the same target width, the constraining dimension ``min(w, h)`` (the item height for
    menus), and the same keyboard presses."""
    prior = l1_physics_update(
        trial.layout,
        trial.candidates,
        trial.intended_index,
        ASSISTANCE_OFF,
        modality=modality,
        n_item_eff=0.0,
        n_variant=0.0,
        r=None,
        tau=float(PROTOCOL_HYPERPARAMETERS["ambiguity_threshold"]),  # the band does not enter assistance_off time
    )
    return prior.time_ms / 1000.0


def reference_times_s(trials: Sequence[Any], *, modality: str) -> dict[str, float]:
    """T_ref per task (section 21.2): the median completed time of the ``assistance_off`` trials of the task, else
    the physics median for ``assistance_off`` on the task's layouts."""
    result: dict[str, float] = {}
    for task in ("pointing", "menu"):
        rows = [item for item in trials if getattr(item, "trial", item).task == task]
        observed = [
            item.labels.time_ms / 1000.0
            for item in rows
            if item.labels is not None
            and item.labels.complete
            and item.trial.applied_regime == "assistance_off"
            and not item.trial.interrupted
        ]
        if observed:
            result[task] = float(median(observed))
        elif rows:
            result[task] = float(median(physics_reference_time_s(item.trial, modality) for item in rows))
    return result


def _weights_for(decision: Any | None) -> UtilityWeights:
    if decision is None:
        return PROTOCOL_WEIGHTS
    priorities = decision.payload.get("priorities") if isinstance(decision.payload, Mapping) else None
    if not isinstance(priorities, Mapping):
        return PROTOCOL_WEIGHTS
    return UtilityWeights.from_mapping(priorities)


def rank_fixed_regimes(
    trials: Sequence[StoredTrial],
    burden: Sequence[BurdenReport],
    decisions: Sequence[DecisionRecord],
    *,
    modality: str,
) -> tuple[str, ...]:
    """The comparator ranking from session-1 Phase A rows (section 12.3).

    The assisted arms of ``l1_arms(modality)`` that ran in this modality with at least 8 non-interrupted trials,
    ordered by mean realized utility per trial (observed outcomes only, each block under the priorities logged with
    its L1 decision, no switching penalty), ties broken toward the more cautious arm.
    """
    arms = [arm for arm in l1_arms(modality) if arm != "assistance_off"]
    by_id = {decision.decision_id: decision for decision in decisions}
    rows = [
        item
        for item in trials
        if item.trial.modality == modality
        and item.trial.task in ("pointing", "menu")
        and not item.trial.interrupted
        and item.labels is not None
    ]
    if not rows:
        return ()
    t_ref = reference_times_s(rows, modality=modality)

    reports: dict[tuple[str, int], Any] = {}
    for report in burden:
        if report.scope == "block" and report.block_index is not None:
            reports[(report.session_id, report.block_index)] = report
    session_means = _burden_means(rows, reports, by_session=True)
    all_means = _burden_means(rows, reports, by_session=False)

    scores: dict[str, list[float]] = {arm: [] for arm in arms}
    for item in rows:
        trial = item.trial
        regime = trial.applied_regime
        if regime not in scores:
            continue
        report = reports.get((trial.session_id, trial.block_index))
        burden_value = block_burden_term(
            None if report is None else report.effort,
            None if report is None else report.intrusiveness,
            regime=regime,
            session_means=session_means.get((trial.session_id, regime)),
            all_means=all_means.get(("", regime)),
        )
        weights = _weights_for(by_id.get(trial.regime_decision_id))
        utility = realized_utility(
            item.labels, time_s=None, t_ref_s=t_ref[trial.task], burden_term=burden_value, weights=weights
        )
        scores[regime].append(utility)

    eligible = [arm for arm in arms if len(scores[arm]) >= MIN_RANKING_TRIALS]
    return tuple(sorted(eligible, key=lambda arm: (-float(np.mean(scores[arm])), -caution_of(arm))))


def _burden_means(
    rows: Sequence[Any], reports: Mapping[tuple[str, int], Any], *, by_session: bool
) -> dict[tuple[str, str], dict[str, float | None]]:
    """Mean effort and intrusiveness per (session, regime) or per regime, over the blocks that have a rating."""
    seen: dict[tuple[str, str], dict[str, list[float]]] = {}
    blocks: set[tuple[str, int]] = set()
    for item in rows:
        trial = item.trial
        key_block = (trial.session_id, trial.block_index)
        if key_block in blocks or key_block not in reports:
            continue
        blocks.add(key_block)
        report = reports[key_block]
        key = (trial.session_id if by_session else "", trial.applied_regime)
        bucket = seen.setdefault(key, {"effort": [], "intrusiveness": []})
        if report.effort is not None:
            bucket["effort"].append(float(report.effort))
        if report.intrusiveness is not None:
            bucket["intrusiveness"].append(float(report.intrusiveness))
    return {
        key: {item: (float(np.mean(values)) if values else None) for item, values in bucket.items()}
        for key, bucket in seen.items()
    }
