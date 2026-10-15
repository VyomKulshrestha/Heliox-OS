"""Simulated users for the co-adaptation benchmark and the OPE validation (spec section 18.1).

Simulated people run the full study protocol of section 12.3 against a policy under test, through the same records,
posteriors (``core.target_posterior``), commit rules (a Python port of section 12.5) and preference layer as the real
study, so the engine verifies and learns from their trials exactly as it does from people's. Every result is
``evidence_kind = "simulated"`` and is never presented as a human result.

The tuning families (F1, F3, F5) share the model's own functional forms, so on them the full model wins over its
ablations largely by construction. The validation family F2 and the holdout families F4 and F6 use forms the model
does not have: delayed or hyperbolic switch costs, skewed or heteroscedastic endpoints, regime-dependent aiming,
burden that is not driven by time, and a mid-session step, a within-session random walk or no day effect at all in
place of the constant day offset of the model's ``gamma``. Family descriptions are nonclinical parameter sets only.

Standard library, numpy and ``pilot.coadaptation.core`` only.
"""

from __future__ import annotations

import copy
import hashlib
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
from numpy.random import PCG64, Generator

from . import SDK_VERSION
from .consent import CoAdaptationPurpose, ConsentPeriod, ConsentRecord
from .constraints import (
    ExclusionReason,
    Fallbacks,
    fallbacks,
    fixed_l2_rule,
    l2_applicable,
    preference_violations,
)
from .engine import CoAdaptationEngine, EngineEvent, EngineSettings, PolicyProbabilities
from .evidence import EvidenceKind
from .ope import BUNDLE_SCHEMA, BundleRows, t_ref_by_task, weights_for
from .outcome_model import MODEL_VERSION, ModelVariant
from .policy import (
    POLICY_VERSION,
    ConstrainedThompsonPolicy,
    NoAllowedRegime,
    PolicyMode,
    decision_seed,
    draw_stream,
    reachable_cells,
    representative_ambiguity,
    select_from_cell,
)
from .preferences import (
    Preference,
    PreferenceConflict,
    PreferenceKind,
    PreferenceScope,
    PreferenceSet,
    preference_to_dict,
)
from .priors import PRIOR_VERSION, enlarge_effective_width, floor_px, keyboard_presses, spread_r0
from .records import (
    RECORD_SCHEMA_VERSION,
    ActivationRecord,
    BurdenReport,
    DecisionRecord,
    HandSummary,
    L2Applied,
    MovementSummary,
    OfferResponse,
    RevertRecord,
    SessionRow,
    SpreadUsed,
    StoredTrial,
    TrialRecord,
    VetoRecord,
    derive_labels,
)
from .state import (
    FEATURE_SCHEMA,
    BlockContext,
    ClockContext,
    DeviceContext,
    LayoutSummary,
    OfferContext,
    Rect,
)
from .study_plan import (
    LAYOUT_GRID,
    PROTOCOL_HYPERPARAMETERS,
    STUDY_PROTOCOL_ID,
    build_session_plan,
    plan_seed,
    rank_fixed_regimes,
    study_protocol_sha256,
)
from .target_posterior import ambiguity_bin, cell_id, compute_posterior, literal_hit, rendered_rects
from .utility import UtilityWeights, block_burden_term, realized_utility
from .vocabulary import (
    CAUTION_ORDER,
    L2_OPTIONS,
    L3_OPTIONS,
    STRATEGY_VERSION,
    VARIANTS,
    DecisionLevel,
    family_of,
    l1_arms,
)

SIM_EPOCH = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
# Simulated sessions run on days 0, 2 and 4: different days within 14 days (12.9).
SESSION_DAY_GAP = 2
SURFACE: tuple[float, float] = (float(LAYOUT_GRID["fit_px"][0]), float(LAYOUT_GRID["fit_px"][1]))
TRIAL_LIMIT_MS = {"pointing": 10000.0, "menu": 15000.0, "idle": 30000.0}
IDLE_EPOCH_MS = 30000.0
# Centre of the 64x32 home button at the bottom centre.
HOME_ORIGIN = (SURFACE[0] / 2.0, SURFACE[1] - 20.0)
MAX_ACTIVATIONS = 32
_VETOABLE_L2 = frozenset(
    {
        "confirm_selection:second_click",
        "ask_clarification:2",
        "ask_clarification:3",
    },
)
ACT_GUARDED = frozenset({"act", "enlarge_target:1.5", "enlarge_target:2.0"})
SIM_CONSENT_TEXT_SHA256 = hashlib.sha256(
    b"heliox simulated participant: no person consented",
).hexdigest()
# p_x of section 7.3: the expansion lands on the aimed target
ENLARGE_RIGHT = {"pointer": 0.85, "hand_pointer": 0.70}
BURDEN_CUTPOINTS = (-1.5, -0.9, -0.3, 0.3, 0.9, 1.5)
DESIGN_PHASES = frozenset({"A", "A_lite"})
_C1_REASONS = frozenset(
    {
        "forbidden_by_you",
        "below_your_min_caution",
        "pinned_other",
        "direct_selection_pinned",
        "interruption_limit",
    }
)


# ----------------------------------------------------------------------------------------------- family forms


@dataclass(frozen=True, slots=True)
class FamilyForms:
    switch_form: Literal["exp3", "hyperbolic", "delayed"]
    endpoint_form: Literal["gaussian", "student_t3", "skew_overshoot", "heteroscedastic_d"]
    aiming_form: Literal["regime_independent", "regime_dependent"]
    burden_form: Literal["time_driven", "annoyance_driven"]
    practice_form: Literal["power", "exponential"]
    # Day and within-session structure.
    day_form: Literal["offset_fatigue", "step", "random_walk", "none"]


FAMILY_FORMS: Mapping[str, FamilyForms] = MappingProxyType(
    {
        "F1": FamilyForms(
            "exp3",
            "gaussian",
            "regime_independent",
            "time_driven",
            "power",
            "offset_fatigue",
        ),
        "F2": FamilyForms(
            "delayed",
            "heteroscedastic_d",
            "regime_independent",
            "time_driven",
            "exponential",
            "step",
        ),
        "F3": FamilyForms(
            "exp3",
            "gaussian",
            "regime_independent",
            "time_driven",
            "power",
            "offset_fatigue",
        ),
        "F4": FamilyForms(
            "hyperbolic",
            "skew_overshoot",
            "regime_dependent",
            "annoyance_driven",
            "power",
            "random_walk",
        ),
        "F5": FamilyForms(
            "exp3",
            "student_t3",
            "regime_independent",
            "time_driven",
            "power",
            "offset_fatigue",
        ),
        "F6": FamilyForms(
            "delayed",
            "heteroscedastic_d",
            "regime_dependent",
            "annoyance_driven",
            "exponential",
            "none",
        ),
    }
)
# The forms the model itself has (section 18.1): the tuning families use only these.
MODEL_FORMS = FamilyForms(
    "exp3",
    "gaussian",
    "regime_independent",
    "time_driven",
    "power",
    "offset_fatigue",
)
FAMILY_SPLITS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"tune": ("F1", "F3", "F5"), "validate": ("F2",), "holdout": ("F4", "F6")}
)
DAY_PARAMS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "offset_fatigue": MappingProxyType({}),
        "step": MappingProxyType(
            {"step_log_spread": 0.25, "step_log_time": 0.10, "step_lo": 0.3, "step_hi": 0.7},
        ),
        "random_walk": MappingProxyType({"walk_sd_log_spread": 0.03, "walk_sd_log_time": 0.015}),
        "none": MappingProxyType({}),
    }
)


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Parameter distributions of one simulated family: (mean, sd) pairs, log-normal for the spread."""

    description: str
    modality: str
    fitts_a_ms: tuple[float, float]
    fitts_b_ms_per_bit: tuple[float, float]
    spread_r: tuple[float, float]  # median and sd of log
    day_sd_log_spread: float
    day_sd_log_time: float
    fatigue_per_min: tuple[float, float]
    practice_alpha: tuple[float, float]
    switch_cost_ms: tuple[float, float]
    confirm_catch: tuple[float, float]
    idle_rate_per_min: tuple[float, float]
    interrupt_dislike: tuple[float, float]  # added to the latent burden of confirm and ask variants
    veto_base: float
    veto_dislike_gain: float
    selection_error: float
    tracking_loss_per_trial: float


FAMILY_SPECS: Mapping[str, FamilySpec] = MappingProxyType(
    {
        "F1": FamilySpec(
            "precise mouse",
            "mouse",
            (100.0, 20.0),
            (230.0, 30.0),
            (1.0, 0.10),
            0.15,
            0.15,
            (0.004, 0.001),
            (0.4, 0.05),
            (250.0, 60.0),
            (0.9, 0.04),
            (0.05, 0.02),
            (0.3, 0.2),
            0.01,
            0.03,
            0.01,
            0.0,
        ),
        "F2": FamilySpec(
            "touchpad",
            "touchpad",
            (120.0, 25.0),
            (330.0, 50.0),
            (1.15, 0.12),
            0.0,
            0.0,
            (0.0, 0.0),
            (0.4, 0.05),
            (300.0, 70.0),
            (0.88, 0.05),
            (0.06, 0.02),
            (0.4, 0.2),
            0.012,
            0.03,
            0.015,
            0.0,
        ),
        "F3": FamilySpec(
            "hand-pointer novice: wide spread, tracking loss",
            "hand_pointer",
            (150.0, 30.0),
            (600.0, 100.0),
            (1.6, 0.15),
            0.15,
            0.15,
            (0.006, 0.002),
            (0.45, 0.05),
            (400.0, 90.0),
            (0.85, 0.05),
            (0.25, 0.08),
            (0.3, 0.2),
            0.015,
            0.04,
            0.03,
            0.15,
        ),
        "F4": FamilySpec(
            "high-frequency jitter (motor-noise parameter)",
            "mouse",
            (110.0, 20.0),
            (260.0, 40.0),
            (1.35, 0.12),
            0.0,
            0.0,
            (0.0, 0.0),
            (0.4, 0.05),
            (300.0, 70.0),
            (0.85, 0.05),
            (0.12, 0.04),
            (0.4, 0.2),
            0.012,
            0.03,
            0.02,
            0.0,
        ),
        "F5": FamilySpec(
            "fast and slip-prone",
            "mouse",
            (80.0, 15.0),
            (200.0, 25.0),
            (1.25, 0.10),
            0.15,
            0.15,
            (0.004, 0.001),
            (0.4, 0.05),
            (250.0, 60.0),
            (0.8, 0.06),
            (0.08, 0.03),
            (0.3, 0.2),
            0.01,
            0.03,
            0.025,
            0.0,
        ),
        "F6": FamilySpec(
            "interruption-averse: strong dislike of confirm and ask, more vetoes",
            "mouse",
            (100.0, 20.0),
            (240.0, 30.0),
            (1.05, 0.10),
            0.0,
            0.0,
            (0.0, 0.0),
            (0.4, 0.05),
            (260.0, 60.0),
            (0.9, 0.04),
            (0.05, 0.02),
            (1.5, 0.3),
            0.02,
            0.06,
            0.012,
            0.0,
        ),
    }
)


def verify_disjoint(splits: Mapping[str, Sequence[str]]) -> None:
    """Raise ValueError when a family appears in more than one split (leakage between tuning and evaluation)."""
    seen: dict[str, str] = {}
    for split, families in splits.items():
        for family in families:
            if family in seen and seen[family] != split:
                raise ValueError(f"family {family} appears in splits {seen[family]} and {split}")
            seen[family] = split


def split_of(family: str) -> str:
    for split, families in FAMILY_SPLITS.items():
        if family in families:
            return split
    raise ValueError("unknown family")


# ----------------------------------------------------------------------------------------------- people


@dataclass(frozen=True, slots=True)
class SimulatedPersonParams:
    family: str
    person_seed: int
    modality: str
    fitts_a_ms: float
    fitts_b_ms_per_bit: float
    spread_r: float
    spread_floor_px: float
    day_sd_log_spread: float  # day offsets (day_form "offset_fatigue" only)
    day_sd_log_time: float
    # Multiplicative spread and time growth per session minute ("offset_fatigue" only).
    fatigue_per_min: float
    # Parameters of the other day forms (step sizes, walk increments).
    day_params: Mapping[str, float]
    practice_alpha: float  # power-law exponent of each variant's extra time (exponential forms use 1/30 per trial)
    switch_cost_ms: float  # size of the transient after a regime change; its shape is switch_form
    forms: FamilyForms
    variant_overhead_ms: Mapping[str, float]
    confirm_catch: float  # probability a wrong highlighted candidate is cancelled
    idle_activation_rate_per_min: float
    # Added to the latent burden for interrupting variants (by family).
    burden_dislike: Mapping[str, float]
    burden_cut_shift: float
    burden_skip_prob: float
    veto_base: float
    veto_dislike_gain: float
    reliance_drift: float  # spread growth per 100 enlarged trials in unassisted blocks (0 for most)
    stated_preferences: tuple[Preference, ...]
    participant_key: str = "p-00000000"
    selection_error: float = 0.01  # share of first activations aimed at another candidate
    tracking_loss_per_trial: float = 0.0  # hand pointer: mean tracking losses per trial
    offer_accept: float = 0.25
    offer_dismiss: float = 0.3


def _participant_key(seed: int, family: str, index: int) -> str:
    return "p-" + hashlib.sha256(f"sim|{seed}|{family}|{index}".encode()).hexdigest()[:8]


def _stated_preferences(rng: Generator, participant_key: str) -> tuple[Preference, ...]:
    """20% forbid ask_clarification, 10% pin act, 10% min_caution = confirm, drawn per person and passed through
    ``PreferenceSet.validate_new``; a combination it rejects is redrawn (18.1)."""
    scope = PreferenceScope(context="study", modality=None, participant_key=participant_key)
    created = SIM_EPOCH - timedelta(hours=23)
    for _ in range(50):
        drawn: list[Preference] = []
        if rng.random() < 0.2:
            drawn.append(
                _pref(
                    participant_key,
                    "forbid",
                    PreferenceKind.FORBID_VARIANT,
                    scope,
                    "ask_clarification",
                    created,
                )
            )
        if rng.random() < 0.1:
            drawn.append(
                _pref(participant_key, "pin-act", PreferenceKind.PIN_ACT, scope, True, created),
            )
        if rng.random() < 0.1:
            drawn.append(
                _pref(
                    participant_key,
                    "min-caution",
                    PreferenceKind.MIN_CAUTION,
                    scope,
                    "confirm_selection:second_click",
                    created,
                )
            )
        accepted: list[Preference] = []
        try:
            for pref in drawn:
                PreferenceSet(accepted).validate_new(pref)
                accepted.append(pref)
        except PreferenceConflict:
            continue
        return tuple(accepted)
    return ()


def _pref(
    participant_key: str,
    name: str,
    kind: PreferenceKind,
    scope: PreferenceScope,
    value: Any,
    created: datetime,
) -> Preference:
    return Preference(
        pref_id=f"sim-{participant_key}-{name}",
        kind=kind,
        scope=scope,
        value=value,
        origin="explicit_user",
        created_at=created,
    )


def _normal(
    rng: Generator,
    pair: tuple[float, float],
    lo: float = 0.0,
    hi: float = math.inf,
) -> float:
    return float(min(max(rng.normal(pair[0], pair[1]), lo), hi))


def _day_offset(rng: Generator, sd: float) -> float:
    """One day's log offset (no draw when the family has no day offset)."""
    return float(rng.normal(0.0, sd)) if sd else 0.0


def make_population(
    families: Sequence[str],
    people_per_family: int,
    seed: int,
) -> list[SimulatedPersonParams]:
    """``people_per_family`` simulated people for each family, deterministic in ``seed``. Reliance drift is above
    zero in half of F3 and F4 (it checks that probes detect de-skilling)."""
    people: list[SimulatedPersonParams] = []
    for family in families:
        spec = FAMILY_SPECS[family]
        forms = FAMILY_FORMS[family]
        for index in range(people_per_family):
            person_seed = int.from_bytes(
                hashlib.sha256(f"{seed}|{family}|{index}".encode()).digest()[:8],
                "big",
            )
            rng = Generator(PCG64(person_seed))
            key = _participant_key(seed, family, index)
            b = _normal(rng, spec.fitts_b_ms_per_bit, 80.0)
            offset_fatigue = forms.day_form == "offset_fatigue"
            fatigue = spec.fatigue_per_min
            dislike = _normal(rng, spec.interrupt_dislike, 0.0)
            people.append(
                SimulatedPersonParams(
                    family=family,
                    person_seed=person_seed,
                    modality=spec.modality,
                    fitts_a_ms=_normal(rng, spec.fitts_a_ms, 20.0),
                    fitts_b_ms_per_bit=b,
                    spread_r=float(spec.spread_r[0] * math.exp(rng.normal(0.0, spec.spread_r[1]))),
                    spread_floor_px=floor_px(spec.modality),
                    day_sd_log_spread=spec.day_sd_log_spread if offset_fatigue else 0.0,
                    day_sd_log_time=spec.day_sd_log_time if offset_fatigue else 0.0,
                    fatigue_per_min=_normal(rng, fatigue, 0.0) if offset_fatigue else 0.0,
                    day_params=DAY_PARAMS[forms.day_form],
                    practice_alpha=_normal(rng, spec.practice_alpha, 0.05),
                    switch_cost_ms=_normal(rng, spec.switch_cost_ms, 0.0),
                    forms=forms,
                    variant_overhead_ms=MappingProxyType(
                        {
                            "confirm_selection": _normal(rng, (60.0, 20.0), 0.0),
                            "enlarge_target": 20.0,
                        }
                    ),
                    confirm_catch=_normal(rng, spec.confirm_catch, 0.5, 0.99),
                    idle_activation_rate_per_min=_normal(rng, spec.idle_rate_per_min, 0.0),
                    burden_dislike=MappingProxyType(
                        {
                            "confirm_selection": dislike,
                            "ask_clarification": dislike,
                            "enlarge_target": 0.1 * dislike,
                        }
                    ),
                    burden_cut_shift=float(rng.normal(0.0, 0.2)),
                    burden_skip_prob=0.1,
                    veto_base=spec.veto_base,
                    veto_dislike_gain=spec.veto_dislike_gain,
                    reliance_drift=0.06 if family in ("F3", "F4") and index % 2 == 0 else 0.0,
                    stated_preferences=_stated_preferences(rng, key),
                    participant_key=key,
                    selection_error=spec.selection_error,
                    tracking_loss_per_trial=spec.tracking_loss_per_trial,
                    offer_accept=0.25,
                    offer_dismiss=0.3,
                )
            )
    return people


# ----------------------------------------------------------------------------------------------- layouts


@dataclass(frozen=True, slots=True)
class PlannedTrial:
    """One cued trial of the layout port: the candidates, the summary sent to the policy and the hidden cue."""

    task: str
    summary: LayoutSummary
    candidates: tuple[Rect, ...]
    intended: int | None  # the cued candidate; None for idle epochs (never in a decision context)
    distance: float  # D of the cued target from the layout origin
    limit_ms: float
    # The grid values drawn from LAYOUT_GRID (w_px, gap_px, d_px, grid, n_items).
    drawn: tuple[tuple[str, Any], ...] = ()


def _id_bar(candidates: Sequence[Rect], origin: tuple[float, float]) -> float:
    values = [
        math.log2(math.hypot(rect.cx - origin[0], rect.cy - origin[1]) / min(rect.w, rect.h) + 1.0)
        for rect in candidates
    ]
    return float(np.mean(values))


def _pointing_layout(rng: Generator, task: str) -> PlannedTrial:
    grid = LAYOUT_GRID["pointing"]
    grid_name = str(grid["grids"][int(rng.integers(0, len(grid["grids"])))])
    cols, rows = (int(part) for part in grid_name.split("x"))
    w = float(grid["w_px"][int(rng.integers(0, len(grid["w_px"])))])
    gap = float(grid["gap_px"][int(rng.integers(0, len(grid["gap_px"])))])
    d = float(grid["d_px"][int(rng.integers(0, len(grid["d_px"])))])
    target = int(rng.integers(0, cols * rows))
    pitch = w + gap
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    offsets = [((c - (cols - 1) / 2.0) * pitch, (r - (rows - 1) / 2.0) * pitch) for c, r in cells]
    half_w = (cols * w + (cols - 1) * gap) / 2.0
    half_h = (rows * w + (rows - 1) * gap) / 2.0
    lo_x, hi_x = 8.0 + half_w, SURFACE[0] - 8.0 - half_w
    lo_y, hi_y = 8.0 + half_h, HOME_ORIGIN[1] - 24.0 - half_h
    centre = None
    for _ in range(24):
        angle = math.radians(float(rng.uniform(-160.0, -20.0)))
        tx = HOME_ORIGIN[0] + d * math.cos(angle)
        ty = HOME_ORIGIN[1] + d * math.sin(angle)
        gx, gy = tx - offsets[target][0], ty - offsets[target][1]
        if lo_x <= gx <= hi_x and lo_y <= gy <= hi_y:
            centre = (gx, gy)
            break
    # D not reachable for this grid: clamp, so the realized D is the nearest reachable one.
    if centre is None:
        centre = (min(max(gx, lo_x), hi_x), min(max(gy, lo_y), hi_y))
    candidates = tuple(Rect(cx=centre[0] + dx, cy=centre[1] + dy, w=w, h=w) for dx, dy in offsets)
    summary = LayoutSummary(
        task=task,  # type: ignore[arg-type]
        n_candidates=len(candidates),
        w_px=w,
        h_px=w,
        gap_px=gap,
        id_bar=_id_bar(candidates, HOME_ORIGIN),
        menu_n=0,
        origin_x=HOME_ORIGIN[0],
        origin_y=HOME_ORIGIN[1],
    )
    # The grid's mean keyboard presses, as the surface sends it, so keyboard decisions use the live prior (7.3).
    summary = replace(summary, presses_mean=keyboard_presses(summary, None, candidates=candidates))
    cued = candidates[target]
    return PlannedTrial(
        task=task,
        summary=summary,
        candidates=candidates,
        intended=None if task == "idle" else target,
        distance=math.hypot(cued.cx - HOME_ORIGIN[0], cued.cy - HOME_ORIGIN[1]),
        limit_ms=TRIAL_LIMIT_MS[task],
        drawn=(("w_px", w), ("gap_px", gap), ("d_px", d), ("grid", f"{cols}x{rows}")),
    )


def _menu_layout(rng: Generator) -> PlannedTrial:
    menu = LAYOUT_GRID["menu"]
    n_items = int(menu["n_items"][int(rng.integers(0, len(menu["n_items"])))])
    item_h = float(menu["item_h_px"])
    item_w = float(menu["item_w_px"])
    button_h = float(menu["button_h_px"])
    bx = float(rng.uniform(item_w / 2.0 + 8.0, SURFACE[0] - item_w / 2.0 - 8.0))
    origin = (bx, button_h / 2.0)
    top = button_h + item_h / 2.0
    candidates = tuple(Rect(cx=bx, cy=top + item_h * i, w=item_w, h=item_h) for i in range(n_items))
    # Zipf(1) over item indices, so familiarity grows.
    weights = np.array([1.0 / (i + 1) for i in range(n_items)])
    target = int(rng.choice(n_items, p=weights / weights.sum()))
    summary = LayoutSummary(
        task="menu",
        n_candidates=n_items,
        w_px=item_w,
        h_px=item_h,
        gap_px=0.0,
        id_bar=_id_bar(candidates, origin),
        menu_n=n_items,
        origin_x=origin[0],
        origin_y=origin[1],
    )
    cued = candidates[target]
    return PlannedTrial(
        task="menu",
        summary=summary,
        candidates=candidates,
        intended=target,
        distance=math.hypot(cued.cx - origin[0], cued.cy - origin[1]),
        limit_ms=TRIAL_LIMIT_MS["menu"],
        drawn=(
            ("n_items", n_items),
            ("item_h_px", item_h),
            ("item_w_px", item_w),
            ("button_h_px", button_h),
        ),
    )


def generate_layout(
    task: str,
    layout_seed: int,
    block_index: int,
    trial_index: int,
) -> PlannedTrial:
    """The Python port of the study's layout generator over ``study_plan.LAYOUT_GRID`` (section 12.4), seeded like
    ``geometry.ts`` with ``layout_seed + block_index * 1000 + trial_index`` (a numpy stream, not mulberry32: the
    port needs the same grid and rules, not the same draws). Idle epochs show a pointing layout with no cue."""
    rng = Generator(PCG64(layout_seed + block_index * 1000 + trial_index))
    if task == "menu":
        return _menu_layout(rng)
    return _pointing_layout(rng, task)


def first_aim_width(plan: PlannedTrial, regime: str, modality: str) -> float:
    """The width ``W_eff`` in the first aim's Fitts time ``ID_eff = log2(D / W_eff + 1)`` (spec 7.3, the model's own
    form): the intended candidate's cue-time width (its height for a menu item), and under ``enlarge_target:k`` the
    ``p_x``-weighted mixture of ``k W`` (the expansion lands on the intended target) and ``W_bad`` (it lands on a
    neighbour), so an enlarged target is also quicker to reach."""
    if plan.intended is None:
        raise ValueError("an idle epoch has no aimed target")
    rect = plan.candidates[plan.intended]
    width = rect.w if plan.task != "menu" else rect.h
    if regime.startswith("enlarge_target"):
        scale = float(VARIANTS[regime].param("scale"))  # type: ignore[arg-type]
        _, _, width = enlarge_effective_width(width, scale, modality)
    return float(width)


# ----------------------------------------------------------------------------------------------- person model


@dataclass
class PersonState:
    """What a simulated person carries from trial to trial; copied at every block start for counterfactuals."""

    day: int = 0
    day_log_spread: float = 0.0
    day_log_time: float = 0.0
    walk_log_spread: float = 0.0
    walk_log_time: float = 0.0
    step_trial: int | None = None
    session_trials: int = 0
    session_minute: float = 0.0
    abs_minute: float = 0.0
    practice: dict[str, int] = field(default_factory=dict)
    practice_total: int = 0
    # (n_eff, at minute)
    item_exposure: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)
    regime: str | None = None
    # Trials since the regime last changed this session (None: no change yet).
    since_switch: int | None = None
    enlarged_trials: int = 0
    vetoes_in_block: int = 0


@dataclass(frozen=True, slots=True)
class BlockSummary:
    """What the person rates after a block (burden prompt, section 12.8)."""

    regime: str
    trials: int
    # Mean log trial time relative to the person's expectation for assistance_off.
    mean_time_z: float
    interruptions_per_trial: float
    errors_per_trial: float
    vetoes: int


class SimulatedPerson:
    """A simulated person: aiming, timing, practice, switch costs, day structure, prompt responses and ratings."""

    def __init__(self, params: SimulatedPersonParams) -> None:
        self.params = params
        self.state = PersonState()

    # ------------------------------------------------------------------------------------------- day structure

    def begin_day(self, day: int, rng: Generator) -> None:
        p = self.params
        self.state.day = day
        self.state.day_log_spread = _day_offset(rng, p.day_sd_log_spread)
        self.state.day_log_time = _day_offset(rng, p.day_sd_log_time)

    def begin_session(self, planned_trials: int, rng: Generator) -> None:
        state = self.state
        state.session_trials = 0
        state.session_minute = 0.0
        state.walk_log_spread = 0.0
        state.walk_log_time = 0.0
        state.regime = None
        state.since_switch = None
        state.step_trial = None
        if self.params.forms.day_form == "step":
            lo = self.params.day_params["step_lo"]
            hi = self.params.day_params["step_hi"]
            state.step_trial = int(
                rng.integers(int(lo * planned_trials), int(hi * planned_trials) + 1),
            )

    def set_regime(self, regime: str) -> None:
        state = self.state
        if state.regime is not None and regime != state.regime:
            state.since_switch = 0
        state.regime = regime
        state.vetoes_in_block = 0

    def _log_offsets(self) -> tuple[float, float]:
        state = self.state
        p = self.params
        log_spread = state.day_log_spread + state.walk_log_spread
        log_time = state.day_log_time + state.walk_log_time
        if p.forms.day_form == "offset_fatigue":
            minutes = min(state.session_minute, 60.0)
            log_spread += p.fatigue_per_min * minutes
            log_time += p.fatigue_per_min * minutes
        if state.step_trial is not None and state.session_trials >= state.step_trial:
            log_spread += p.day_params["step_log_spread"]
            log_time += p.day_params["step_log_time"]
        return log_spread, log_time

    def switch_transient(self) -> float:
        """The switch cost in ms on this trial, shaped by ``switch_form`` (k = trials since the change)."""
        k = self.state.since_switch
        if k is None:
            return 0.0
        c = self.params.switch_cost_ms
        form = self.params.forms.switch_form
        if form == "exp3":
            return c * math.exp(-k / 3.0)
        if form == "hyperbolic":
            return c / (1.0 + k)
        return c * (k / 2.0) * math.exp(1.0 - k / 2.0)

    def spread(self, regime: str) -> float:
        """The person's spread factor ``r`` on this trial."""
        p = self.params
        log_spread, _ = self._log_offsets()
        r = p.spread_r * math.exp(log_spread)
        if p.forms.aiming_form == "regime_dependent":
            if regime.startswith("confirm_selection"):
                r *= 1.3
            elif regime.startswith("enlarge_target"):
                r *= 0.85
        if regime == "assistance_off" and p.reliance_drift > 0:
            r *= 1.0 + p.reliance_drift * self.state.enlarged_trials / 100.0
        return r

    def practice_multiplier(self, regime: str) -> float:
        n = self.state.practice.get(regime, 0)
        if self.params.forms.practice_form == "power":
            return 1.0 + 0.3 * (n + 1) ** (-self.params.practice_alpha)
        return 1.0 + 0.3 * math.exp(-n / 30.0)

    def fitts_ms(self, id_bits: float) -> float:
        return self.params.fitts_a_ms + self.params.fitts_b_ms_per_bit * id_bits

    def menu_search_ms(self, menu_n: int, item: int) -> float:
        n_eff = self._item_exposure(menu_n, item)
        expertise = 1.0 - math.exp(-n_eff / 5.0)
        novice = 0.2 + 0.1 * menu_n
        expert = 0.2 + 0.15 * math.log2(menu_n + 1)
        seconds = (1.0 - expertise) * novice + expertise * expert
        return seconds * 1000.0

    def _item_exposure(self, menu_n: int, item: int) -> float:
        value = self.state.item_exposure.get((menu_n, item))
        if value is None:
            return 0.0
        n_eff, at = value
        days = max(0.0, self.state.abs_minute - at) / (24.0 * 60.0)
        return n_eff * 2.0 ** (-days / 7.0)

    def first_aim_ms(
        self,
        trial: PlannedTrial,
        regime: str,
        aim_width: float,
        rng: Generator,
    ) -> float:
        """Cue to first activation: reaction, menu search, pointing, practice, day effects, switch cost, overhead."""
        _, log_time = self._log_offsets()
        id_bits = math.log2(max(trial.distance, 1.0) / max(aim_width, 1.0) + 1.0)
        base = 250.0 + self.fitts_ms(id_bits)
        if trial.task == "menu" and trial.intended is not None:
            base += self.menu_search_ms(trial.summary.menu_n, trial.intended)
        overhead = self.params.variant_overhead_ms.get(family_of(regime), 0.0)
        practised = base * self.practice_multiplier(regime) * math.exp(log_time)
        total = practised + self.switch_transient() + overhead
        return float(total * math.exp(rng.normal(0.0, 0.2)))

    def endpoint(
        self,
        target: Rect,
        layout: PlannedTrial,
        rendered: Sequence[Rect],
        rng: Generator,
        *,
        regime: str = "act",
    ) -> tuple[float, float]:
        """Where an activation aimed at ``target`` lands, per the family's endpoint form. ``target`` is the aimed
        candidate's cue-time rect: under ``enlarge_target:*`` the scatter stays planned for the original ``W``
        (spec 7.3), and only the literal hit and the posterior use the ``rendered`` rects."""
        p = self.params
        r = self.spread(regime)
        sx = r * math.sqrt((target.w / 4.133) ** 2 + p.spread_floor_px**2)
        sy = r * math.sqrt((target.h / 4.133) ** 2 + p.spread_floor_px**2)
        form = p.forms.endpoint_form
        origin = (layout.summary.origin_x, layout.summary.origin_y)
        if form == "heteroscedastic_d":
            distance = math.hypot(target.cx - origin[0], target.cy - origin[1])
            factor = 0.7 + 0.3 * distance / 240.0
            sx, sy = sx * factor, sy * factor
        if form == "student_t3":
            return target.cx + sx * float(rng.standard_t(3)), target.cy + sy * float(
                rng.standard_t(3),
            )
        if form == "skew_overshoot":
            ux, uy = target.cx - origin[0], target.cy - origin[1]
            norm = math.hypot(ux, uy) or 1.0
            ux, uy = ux / norm, uy / norm
            delta = 4.0 / math.sqrt(17.0)  # skew-normal with shape 4 along the approach axis
            along = delta * abs(float(rng.normal())) + math.sqrt(1.0 - delta * delta) * float(
                rng.normal(),
            )
            across = float(rng.normal())
            scale = (sx + sy) / 2.0
            return (
                target.cx + scale * (along * ux - across * uy),
                target.cy + scale * (along * uy + across * ux),
            )
        return target.cx + sx * float(rng.normal()), target.cy + sy * float(rng.normal())

    # ------------------------------------------------------------------------------------------------ responses

    def dislike(self, variant: str) -> float:
        return float(self.params.burden_dislike.get(family_of(variant), 0.0))

    def responds(
        self,
        prompt_kind: str,
        *,
        correct: bool = True,
        variant: str = "",
        rng: Generator,
    ) -> str:
        """The person's answer to a prompt: ``confirm`` (a highlighted candidate), ``hold``, ``offer``, ``veto_l1``
        and ``veto_l2``."""
        p = self.params
        if prompt_kind == "confirm":
            if correct:
                return "confirm"
            return "cancel" if rng.random() < p.confirm_catch else "confirm"
        if prompt_kind == "hold":
            if p.modality == "hand_pointer" and rng.random() < 0.5 * p.tracking_loss_per_trial:
                return "abort"
            if correct:
                return "release" if rng.random() < 0.03 else "complete"
            return "release" if rng.random() < p.confirm_catch else "complete"
        if prompt_kind == "offer":
            u = rng.random()
            if u < p.offer_accept:
                return "accepted"
            return "dismissed" if u < p.offer_accept + p.offer_dismiss else "ignored"
        if prompt_kind in ("veto_l1", "veto_l2"):
            switched = self.state.since_switch is not None and self.state.since_switch < 3
            disliked = p.veto_dislike_gain * self.dislike(variant)
            probability = p.veto_base + disliked + (0.02 if switched else 0.0)
            if prompt_kind == "veto_l2":
                probability *= 0.5
            return "veto" if rng.random() < probability else "keep"
        raise ValueError("unknown prompt kind")

    def burden(
        self,
        block_summary: BlockSummary,
        rng: Generator,
        *,
        skips: bool = True,
    ) -> dict[str, Any]:
        """The 7-point ratings of a block (B2 is not asked after ``assistance_off``); skipped items are None."""
        p = self.params
        regime = block_summary.regime
        dislike = self.dislike(regime)
        if p.forms.burden_form == "time_driven":
            ipt = block_summary.interruptions_per_trial
            errors = block_summary.errors_per_trial
            time_z = block_summary.mean_time_z
            enlarged = 0.2 if regime.startswith("enlarge_target") else 0.0
            latents = {
                "effort": 0.6 * time_z + 0.8 * ipt + 2.0 * errors + 0.5 * dislike - 0.3,
                "intrusiveness": 0.6 * ipt + dislike + enlarged - 0.6,
                "control": 0.4 - 0.4 * ipt - 3.0 * errors - 0.4 * dislike,
            }
        else:
            annoyance = dislike + (0.2 if regime != "assistance_off" else 0.0)
            vetoes = 0.5 * block_summary.vetoes
            latents = {
                "effort": annoyance + vetoes - 0.3,
                "intrusiveness": annoyance + vetoes - 0.3,
                "control": 0.3 - annoyance - vetoes,
            }
        ratings: dict[str, Any] = {"skipped": []}
        for item in ("effort", "intrusiveness", "control"):
            if item == "intrusiveness" and regime == "assistance_off":
                ratings[item] = None  # not asked; 1 by definition, not counted as skipped (12.8)
                continue
            latent = latents[item] + float(rng.normal(0.0, 0.5))
            rating = 1 + sum(1 for cut in BURDEN_CUTPOINTS if latent > cut + p.burden_cut_shift)
            if skips and rng.random() < p.burden_skip_prob:
                ratings[item] = None
                ratings["skipped"].append(item)
            else:
                ratings[item] = int(rating)
        return ratings

    # ------------------------------------------------------------------------------------------------ learning

    def after_trial(self, trial: PlannedTrial, regime: str, minutes: float, rng: Generator) -> None:
        """Practice, menu familiarity, session time, switch age and the day form's within-session dynamics."""
        state = self.state
        state.practice[regime] = state.practice.get(regime, 0) + 1
        state.practice_total += 1
        if regime.startswith("enlarge_target"):
            state.enlarged_trials += 1
        if trial.task == "menu" and trial.intended is not None:
            key = (trial.summary.menu_n, trial.intended)
            n_eff = self._item_exposure(*key)
            state.item_exposure[key] = (n_eff + 1.0, state.abs_minute)
        state.session_trials += 1
        state.session_minute += minutes
        state.abs_minute += minutes
        if state.since_switch is not None:
            state.since_switch += 1
        if self.params.forms.day_form == "random_walk":
            state.walk_log_spread += float(
                rng.normal(0.0, self.params.day_params["walk_sd_log_spread"]),
            )
            state.walk_log_time += float(
                rng.normal(0.0, self.params.day_params["walk_sd_log_time"]),
            )

    def practise(self, regime: str, trials: int, minutes: float) -> None:
        """The unrecorded practice block: practice counters and session time only."""
        self.state.practice[regime] = self.state.practice.get(regime, 0) + trials
        self.state.practice_total += trials
        self.state.session_minute += minutes
        self.state.abs_minute += minutes

    def copy_state(self) -> PersonState:
        return copy.deepcopy(self.state)

    def restore_state(self, state: PersonState) -> None:
        self.state = copy.deepcopy(state)


# ---------------------------------------------------------------------------------------------- policies


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    stored: StoredTrial
    events: tuple[EngineEvent, ...]
    refresh_required: bool


class StudyPolicy(Protocol):
    """What the runner drives, exactly like the service drives the engine (section 13.3)."""

    name: str
    preference_replacements: int

    def start(self, *, participant_key: str, preferences: PreferenceSet, modality: str) -> None: ...

    def begin_session(self, row: SessionRow) -> None: ...

    def fallbacks(self, modality: str, *, now: datetime) -> Fallbacks: ...

    def decide_block(
        self,
        *,
        decision_id: str,
        block: BlockContext,
        now: datetime,
    ) -> dict[str, Any]: ...

    def ambiguity_table(
        self,
        *,
        decision_id: str,
        block_decision_id: str,
        block: BlockContext,
        start_trial_index: int,
        trials_in_block: int,
        now: datetime,
    ) -> dict[str, Any]: ...

    def offer(
        self,
        *,
        decision_id: str,
        context: OfferContext,
        now: datetime,
    ) -> dict[str, Any]: ...

    def observe_trial(self, trial: TrialRecord) -> PolicyObservation: ...

    def observe_burden(self, report: BurdenReport) -> None: ...

    def end_session(self, *, ended_at: datetime, reason: str) -> None: ...

    def snapshot(self) -> bytes | None: ...


class EnginePolicy:
    """The co-adaptation engine (any ``ModelVariant``, optionally with an ablated policy) as the policy under test."""

    def __init__(
        self,
        variant: ModelVariant = ModelVariant.FULL,
        *,
        policy: Callable[[], ConstrainedThompsonPolicy] | None = None,
        name: str | None = None,
    ) -> None:
        self.variant = ModelVariant(variant)
        self._policy_factory = policy
        self.name = name or f"engine_{self.variant.value}"
        self.preference_replacements = 0
        self.engine: CoAdaptationEngine | None = None

    def start(self, *, participant_key: str, preferences: PreferenceSet, modality: str) -> None:
        self.engine = CoAdaptationEngine(
            participant_key=participant_key,
            settings=EngineSettings(),
            preferences=preferences,
            evidence_kind=EvidenceKind.SIMULATED,
            model_variant=self.variant,
            policy=None if self._policy_factory is None else self._policy_factory(),
        )

    def _engine(self) -> CoAdaptationEngine:
        if self.engine is None:
            raise RuntimeError("start() first")
        return self.engine

    def begin_session(self, row: SessionRow) -> None:
        self._engine().begin_session(row)

    def fallbacks(self, modality: str, *, now: datetime) -> Fallbacks:
        return self._engine().fallbacks(modality, now=now)

    def decide_block(
        self,
        *,
        decision_id: str,
        block: BlockContext,
        now: datetime,
    ) -> dict[str, Any]:
        return self._engine().decide_block(decision_id=decision_id, block=block, now=now).to_dict()

    def ambiguity_table(
        self,
        *,
        decision_id: str,
        block_decision_id: str,
        block: BlockContext,
        start_trial_index: int,
        trials_in_block: int,
        now: datetime,
    ) -> dict[str, Any]:
        return (
            self._engine()
            .decide_ambiguity_table(
                decision_id=decision_id,
                block_decision_id=block_decision_id,
                block=block,
                start_trial_index=start_trial_index,
                trials_in_block=trials_in_block,
                now=now,
            )
            .to_dict()
        )

    def offer(self, *, decision_id: str, context: OfferContext, now: datetime) -> dict[str, Any]:
        decision = self._engine().decide_offer(decision_id=decision_id, context=context, now=now)
        return decision.to_dict()

    def observe_trial(self, trial: TrialRecord) -> PolicyObservation:
        observation = self._engine().observe_trial(trial)
        return PolicyObservation(
            observation.stored,
            tuple(observation.events),
            bool(observation.refresh_required),
        )

    def observe_burden(self, report: BurdenReport) -> None:
        self._engine().observe_burden(report)

    def end_session(self, *, ended_at: datetime, reason: str) -> None:
        self._engine().end_session(ended_at=ended_at, reason=reason)

    def snapshot(self) -> bytes | None:
        return self._engine().snapshot()


BaselineKind = Literal["assistance_off", "best_fixed", "rule_heuristic", "uniform", "oracle"]


class BaselinePolicy:
    """Policies 1-4 and 10 of section 18.2. The design blocks (Phase A and A-lite permuted rounds, probes and
    comparators) run exactly as in the study; the adaptive positions (phase ``B``) take the baseline's own choice.
    Every choice passes through the same preference layer as the engine (C1 through ``PreferenceSet.apply`` and the
    ambiguity rule C2 at L2); a choice the person's rules exclude is replaced by the first allowed entry of
    ``fallbacks(modality)`` and counted in ``preference_replacements``."""

    def __init__(
        self,
        kind: BaselineKind,
        *,
        oracle: Callable[[BlockContext, Sequence[str]], str] | None = None,
    ):
        self.kind = kind
        self.name = {
            "assistance_off": "fixed_assistance_off",
            "best_fixed": "best_fixed_regime",
            "rule_heuristic": "rule_heuristic",
            "uniform": "uniform_random",
            "oracle": "oracle",
        }[kind]
        self.preference_replacements = 0
        self._oracle = oracle
        self._preferences = PreferenceSet(())
        self._participant_key = ""
        self._modality = "mouse"
        self._row: SessionRow | None = None
        self._remaining: list[str] = []
        self._round = 0
        self._decisions: dict[str, dict[str, Any]] = {}
        self._decision_times: dict[str, datetime] = {}
        self._block_modes: dict[str, str] = {}
        # C5: level -> variants excluded at that level's next decision.
        self._vetoed: dict[str, set[str]] = {}

    def set_oracle(self, oracle: Callable[[BlockContext, Sequence[str]], str]) -> None:
        self._oracle = oracle

    def start(self, *, participant_key: str, preferences: PreferenceSet, modality: str) -> None:
        self._participant_key = participant_key
        self._preferences = preferences
        self._modality = modality

    def begin_session(self, row: SessionRow) -> None:
        self._row = row
        self._remaining = []
        self._round = 0
        self._vetoed = {}

    def end_session(self, *, ended_at: datetime, reason: str) -> None:
        self._row = None

    def snapshot(self) -> bytes | None:
        return None

    def fallbacks(self, modality: str, *, now: datetime) -> Fallbacks:
        return fallbacks(
            self._preferences,
            modality,
            participant_key=self._participant_key,
            calibration_suspended=False,
            now=now,
        )

    # ------------------------------------------------------------------------------------------------ helpers

    def _settings(self) -> Mapping[str, Any]:
        return self._row.settings if self._row is not None else dict(PROTOCOL_HYPERPARAMETERS)

    def _c1(
        self,
        level: DecisionLevel,
        options: Sequence[str],
    ) -> tuple[list[str], dict[str, ExclusionReason]]:
        return self._preferences.apply(
            level,
            options,
            context="study",
            modality=self._modality,
            participant_key=self._participant_key,
            interruptions_used=0,
        )

    def _hard(
        self,
        level: DecisionLevel,
        options: Sequence[str],
    ) -> tuple[list[str], dict[str, ExclusionReason]]:
        """C1 and the veto cooldown C5 (a variant vetoed in this session is excluded at the level's next
        decision)."""
        allowed, excluded = self._c1(level, options)
        vetoed = self._vetoed.get(level.value, set())
        kept = []
        for variant in allowed:
            if variant in vetoed and variant != "wait_for_evidence":
                excluded[variant] = ExclusionReason.RECENT_VETO
            else:
                kept.append(variant)
        return kept, excluded

    def _seed(self, decision_id: str) -> int:
        salt = self._row.seed_salt if self._row is not None else "0" * 32
        return decision_seed(self._participant_key, salt, decision_id)

    def _priorities(self) -> dict[str, float]:
        m, f, i = self._preferences.priorities(
            context="study", modality=self._modality, participant_key=self._participant_key
        )
        return {"fewer_mistakes": m, "faster": f, "fewer_interruptions": i}

    def _explanation(self, mode: str, excluded: Mapping[str, str]) -> dict[str, Any]:
        return {
            "summary_key": f"coadaptation.explain.{mode}",
            "values": {},
            "contrasts": [],
            "excluded": [[variant, str(reason)] for variant, reason in excluded.items()],
            "evidence_source": "literature_prior",
            "trials_person": 0,
            "trials_today": 0,
        }

    @staticmethod
    def _versions() -> dict[str, str]:
        return {
            "policy": POLICY_VERSION,
            "model": MODEL_VERSION,
            "prior": PRIOR_VERSION,
            "feature_schema": FEATURE_SCHEMA,
        }

    def _recommendation(
        self,
        decision_id: str,
        level: DecisionLevel,
        mode: str,
        options: Sequence[str],
        propensities: Mapping[str, float],
        eligible: Sequence[str],
        excluded: Mapping[str, str],
        *,
        protocol_deviation: bool = False,
        round_index: int | None = None,
        fallback: bool = False,
    ) -> dict[str, Any]:
        seed = self._seed(decision_id)
        u = float(draw_stream(seed).random())
        order = [option for option in options if propensities.get(option, 0.0) > 0]
        variant = select_from_cell(options, propensities, u) if order else options[0]
        l1 = level == DecisionLevel.BLOCK_REGIME
        return {
            "decision_id": decision_id,
            "level": level.value,
            "mode": mode,
            "variant": variant,
            "params": {key: value for key, value in VARIANTS[variant].params},
            "eligible": list(eligible),
            "excluded": {variant_: str(reason) for variant_, reason in excluded.items()},
            "propensities": {option: float(propensities.get(option, 0.0)) for option in options},
            "p_thompson": None,
            "exploration": 0.0,
            "fallback": fallback,
            "protocol_deviation": protocol_deviation,
            "round_index": round_index,
            "spread": {"r": spread_r0(self._modality), "floor": floor_px(self._modality)}
            if l1 and self._modality != "keyboard"
            else None,
            "priorities": self._priorities() if l1 else None,
            "predictions": {},
            "explanation": self._explanation(mode, excluded),
            "seed": f"{seed:016x}",
            "versions": self._versions(),
            "authority": "presentation_only",
            "execution_authority": False,
        }

    def _replace(
        self, choice: str, allowed: Sequence[str], excluded: Mapping[str, str], fallback: str | None
    ) -> str | None:
        """The baseline's own choice, or the first allowed fallback when the hard constraints exclude it. Only an
        exclusion by the person's rules (C1) counts as a preference replacement; a veto cooldown (C5) does not."""
        if choice in allowed:
            return choice
        if str(excluded.get(choice, "")) in _C1_REASONS:
            self.preference_replacements += 1
        if fallback in allowed:
            return fallback
        return max(allowed, key=lambda v: VARIANTS[v].caution) if allowed else None

    # ------------------------------------------------------------------------------------------------ L1

    def decide_block(
        self,
        *,
        decision_id: str,
        block: BlockContext,
        now: datetime,
    ) -> dict[str, Any]:
        arms = l1_arms(self._modality)
        allowed, excluded = self._hard(DecisionLevel.BLOCK_REGIME, arms)
        excluded = {variant: str(reason) for variant, reason in excluded.items()}
        held = self._preferences.held_regime(
            context="study", modality=self._modality, participant_key=self._participant_key
        )
        fb = self.fallbacks(self._modality, now=now)
        if not allowed and fb.l1 is None:
            raise NoAllowedRegime("no allowed regime")
        planned_mode = {
            "A": "randomized",
            "A_lite": "randomized",
            "probe": "probe",
            "comparator": "comparator",
        }
        if held is not None:
            payload = self._recommendation(
                decision_id,
                DecisionLevel.BLOCK_REGIME,
                "pinned",
                arms,
                {held: 1.0},
                [held],
                excluded,
            )
        elif not allowed:
            payload = self._recommendation(
                decision_id,
                DecisionLevel.BLOCK_REGIME,
                planned_mode.get(block.phase, "adaptive"),
                arms,
                {fb.l1: 1.0},
                [],
                excluded,
                fallback=True,
            )
        elif block.phase in DESIGN_PHASES:
            payload = self._randomized(decision_id, arms, allowed, excluded)
        elif block.phase == "probe":
            if "assistance_off" in allowed:
                choice = "assistance_off"
            else:
                choice = min(allowed, key=CAUTION_ORDER.index) if allowed else fb.l1
            payload = self._recommendation(
                decision_id,
                DecisionLevel.BLOCK_REGIME,
                "probe",
                arms,
                {choice: 1.0},
                [choice],
                {
                    **{a: "probe_schedule" for a in arms if a != choice and a not in excluded},
                    **excluded,
                },
                protocol_deviation=choice != "assistance_off",
            )
        elif block.phase == "comparator":
            ranking = self._row.comparator_ranking if self._row is not None else ()
            permitted = [arm for arm in ranking if arm in allowed]
            choice = permitted[0] if permitted else fb.l1
            payload = self._recommendation(
                decision_id,
                DecisionLevel.BLOCK_REGIME,
                "comparator",
                arms,
                {choice: 1.0},
                [choice],
                {
                    **{a: "comparator_schedule" for a in arms if a != choice and a not in excluded},
                    **excluded,
                },
                protocol_deviation=not ranking or choice != ranking[0],
            )
        else:
            payload = self._adaptive(decision_id, block, arms, allowed, excluded, fb)
        self._decisions[decision_id] = payload
        self._decision_times[decision_id] = now
        self._block_modes[decision_id] = payload["mode"]
        self._vetoed.pop(DecisionLevel.BLOCK_REGIME.value, None)
        return payload

    def _randomized(
        self,
        decision_id: str,
        arms: Sequence[str],
        allowed: Sequence[str],
        excluded: Mapping[str, str],
    ) -> dict[str, Any]:
        remaining = [arm for arm in self._remaining if arm in allowed]
        if not remaining:
            self._round += 1
            remaining = list(allowed)
        propensities = {arm: 1.0 / len(remaining) for arm in remaining}
        reasons = dict(excluded)
        for arm in allowed:
            if arm not in remaining:
                reasons[arm] = "used_this_round"
        payload = self._recommendation(
            decision_id,
            DecisionLevel.BLOCK_REGIME,
            "randomized",
            arms,
            propensities,
            list(allowed),
            reasons,
            round_index=self._round,
        )
        remaining.remove(payload["variant"])
        self._remaining = remaining
        return payload

    def _adaptive(
        self,
        decision_id: str,
        block: BlockContext,
        arms: Sequence[str],
        allowed: Sequence[str],
        excluded: Mapping[str, str],
        fb: Fallbacks,
    ) -> dict[str, Any]:
        if self.kind == "uniform":
            propensities = {arm: 1.0 / len(allowed) for arm in allowed}
            return self._recommendation(
                decision_id,
                DecisionLevel.BLOCK_REGIME,
                "adaptive",
                arms,
                propensities,
                list(allowed),
                excluded,
            )
        if self.kind == "assistance_off":
            choice = "assistance_off"
        elif self.kind == "best_fixed":
            ranking = self._row.comparator_ranking if self._row is not None else ()
            choice = ranking[0] if ranking else "assistance_off"
        elif self.kind == "rule_heuristic":
            if self._modality == "hand_pointer":
                choice = "confirm_selection:hold_400"
            elif float(np.mean([layout.id_bar for layout in block.layouts])) > 4.0:
                choice = "enlarge_target:1.5"
            else:
                choice = "act"
        else:
            if self._oracle is None:
                raise RuntimeError("the oracle needs the runner's block simulator")
            choice = self._oracle(block, allowed)
        choice = self._replace(choice, allowed, excluded, fb.l1)
        if choice is None:
            raise NoAllowedRegime("no allowed regime")
        return self._recommendation(
            decision_id,
            DecisionLevel.BLOCK_REGIME,
            "adaptive",
            arms,
            {choice: 1.0},
            list(allowed),
            excluded,
        )

    # ------------------------------------------------------------------------------------------------ L2

    def ambiguity_table(
        self,
        *,
        decision_id: str,
        block_decision_id: str,
        block: BlockContext,
        start_trial_index: int,
        trials_in_block: int,
        now: datetime,
    ) -> dict[str, Any]:
        tau = float(self._settings()["ambiguity_threshold"])
        mode = self._block_modes.get(block_decision_id, "randomized")
        fb = self.fallbacks(self._modality, now=now)
        pin_act = self._preferences.pin_act(
            context="study", modality=self._modality, participant_key=self._participant_key
        )
        if pin_act:
            mode = "pinned"
        cells = []
        for bin_id, n_plausible, has_hit in reachable_cells(tau):
            shape = {"n_plausible": n_plausible, "has_hit": has_hit}
            applicable = [v for v in L2_OPTIONS if l2_applicable(v, **shape)]
            excluded = {v: "not_applicable" for v in L2_OPTIONS if v not in applicable}
            allowed, c1 = self._hard(DecisionLevel.AMBIGUITY_RESPONSE, applicable)
            excluded.update({v: str(r) for v, r in c1.items()})
            if pin_act:
                chosen = "act" if "act" in allowed else "wait_for_evidence"
                propensities = {chosen: 1.0}
                eligible = [chosen]
                excluded.update({v: "pinned_other" for v in allowed if v != chosen})
            else:
                if "act" in allowed:
                    allowed.remove("act")
                    excluded["act"] = "ambiguous_activation"
                if "wait_for_evidence" not in allowed:
                    allowed.append("wait_for_evidence")
                eligible = list(allowed)
                cell = cell_id(bin_id, n_plausible, has_hit)
                propensities = self._cell_propensities(
                    mode,
                    cell,
                    eligible,
                    excluded,
                    fb,
                    tau,
                    bin_id,
                )
            cells.append(
                {
                    "cell_id": cell_id(bin_id, n_plausible, has_hit),
                    "bin": bin_id,
                    "n_plausible": n_plausible,
                    "has_hit": has_hit,
                    "order": list(L2_OPTIONS),
                    "propensities": {v: float(propensities.get(v, 0.0)) for v in L2_OPTIONS},
                    "eligible": eligible,
                    "excluded": excluded,
                }
            )
        seed = self._seed(decision_id)
        rng_u = draw_stream(seed)
        draws = [[float(rng_u.random()) for _ in range(4)] for _ in range(trials_in_block)]
        bounds = [
            ("b1", tau, max(tau, 0.4)),
            ("b2", max(tau, 0.4), max(tau, 0.6)),
            ("b3", max(tau, 0.6), 1.0),
        ]
        payload = {
            "decision_id": decision_id,
            "level": "ambiguity_table",
            "block_decision_id": block_decision_id,
            "mode": mode,
            "ambiguity_threshold": tau,
            "bins": [{"id": b, "lo": lo, "hi": hi} for b, lo, hi in bounds],
            "cells": cells,
            "start_trial_index": start_trial_index,
            "draws": draws,
            "max_draws_per_trial": 4,
            "after_max": "wait_for_evidence",
            "explanation": self._explanation(mode, {}),
            "seed": f"{seed:016x}",
            "versions": self._versions(),
            "authority": "presentation_only",
            "execution_authority": False,
        }
        self._decisions[decision_id] = payload
        self._decision_times[decision_id] = now
        self._block_modes[decision_id] = mode
        self._vetoed.pop(DecisionLevel.AMBIGUITY_RESPONSE.value, None)
        return payload

    def _cell_propensities(
        self,
        mode: str,
        cell: str,
        eligible: Sequence[str],
        excluded: Mapping[str, str],
        fb: Fallbacks,
        tau: float,
        bin_id: str,
    ) -> dict[str, float]:
        if mode in ("probe", "pinned"):
            return {fixed_l2_rule(fb, cell): 1.0}
        if mode in ("randomized", "comparator") or self.kind in ("uniform", "best_fixed"):
            return {v: 1.0 / len(eligible) for v in eligible}
        if self.kind == "rule_heuristic":
            n_plausible, has_hit = int(cell.split(":")[1][1:]), cell.endswith("h1")
            if representative_ambiguity(bin_id, tau) > 0.3 and n_plausible >= 2:
                choice = "ask_clarification:2"
            elif has_hit:
                choice = "confirm_selection:second_click"
            else:
                choice = "wait_for_evidence"
            rule = fixed_l2_rule(fb, cell)
            choice = self._replace(choice, eligible, excluded, rule) or "wait_for_evidence"
            return {choice: 1.0}
        fixed = fixed_l2_rule(fb, cell)
        return {fixed if fixed in eligible else "wait_for_evidence": 1.0}

    # ------------------------------------------------------------------------------------------------ L3

    def offer(self, *, decision_id: str, context: OfferContext, now: datetime) -> dict[str, Any]:
        block_mode = next(reversed(self._block_modes.values()), "randomized")
        applicable = [v for v in L3_OPTIONS if _offer_applicable(v, context)]
        allowed, excluded = self._c1(DecisionLevel.OFFER, applicable)
        reasons = {v: "not_applicable" for v in L3_OPTIONS if v not in applicable}
        reasons.update({v: str(r) for v, r in excluded.items()})
        if "no_offer" not in allowed:
            allowed.append("no_offer")
        if block_mode in ("randomized", "comparator") or self.kind == "uniform":
            propensities = {v: 1.0 / len(allowed) for v in allowed}
        else:
            propensities = {"no_offer": 1.0}
        payload = self._recommendation(
            decision_id, DecisionLevel.OFFER, block_mode, L3_OPTIONS, propensities, allowed, reasons
        )
        self._decisions[decision_id] = payload
        self._decision_times[decision_id] = now
        return payload

    # ------------------------------------------------------------------------------------------------ trials

    def observe_trial(self, trial: TrialRecord) -> PolicyObservation:
        veto = trial.veto
        if veto is not None and self._block_modes.get(veto.decision_id) != "pinned":
            level = DecisionLevel(veto.level)
            if level == DecisionLevel.BLOCK_REGIME:
                self._vetoed.setdefault(level.value, set()).add(trial.applied_regime)
            else:
                limit = len(trial.activations) + 1
                if trial.revert is not None:
                    limit = trial.revert.from_activation
                responded = [a for a in trial.activations if a.l2 is not None and a.index < limit]
                before = [a.l2.variant for a in responded]
                if before and before[-1] != "wait_for_evidence":
                    self._vetoed.setdefault(level.value, set()).add(before[-1])
        if trial.interrupted:
            verification = "interrupted"
        elif trial.l1_fallback_reason is not None:
            verification = "fallback"
        else:
            verification = "ok"
        labels = None if trial.interrupted else derive_labels(trial)
        violations = preference_violations(
            trial,
            self._preferences,
            decision_times=self._decision_times,
            participant_key=self._participant_key,
        )
        day_index = self._row.day_index if self._row is not None else 0
        stored = StoredTrial(
            trial=trial,
            verification=verification,
            day_index=day_index,
            regime_ucb_unintended=None,
            labels=labels,
            episodes=(),
            preference_violations=violations,
        )
        return PolicyObservation(stored, (), False)

    def observe_burden(self, report: BurdenReport) -> None:
        return None


def _offer_applicable(variant: str, context: OfferContext) -> bool:
    if variant == "offer_modality:keyboard":
        return context.keyboard_available and context.modality != "keyboard"
    if variant == "offer_modality:pointer":
        return context.modality == "hand_pointer"
    return True


# ----------------------------------------------------------------------------------------------- runner


@dataclass(frozen=True, slots=True)
class TrialTruth:
    """What really happened in one simulated trial (the simulator knows the cue; the policy never does)."""

    trial_id: str
    session_number: int
    block_index: int
    trial_index: int
    phase: str
    task: str
    regime: str  # the regime the trial applied (the revert regime counts from the revert on)
    idle: bool
    interrupted: bool
    complete: bool
    ua: bool
    correction: bool
    veto: bool
    time_s: float  # cue to correct commit, or the limit
    limit_s: float
    corrections: int
    idle_commits: int
    idle_minutes: float
    # A commit with A_gate > tau under an act-guarded regime outside pin_act.
    acting_under_ambiguity: bool
    activations: int
    interruptions: int  # the trial's addition to the block's interruption count (9.3)


@dataclass(frozen=True, slots=True)
class BlockStart:
    """The state at one block's start: what ``counterfactual_block_value`` restores."""

    decision_id: str
    session_number: int
    block_index: int
    phase: str
    task: str
    person: PersonState
    trials: tuple[PlannedTrial, ...]
    previous_regime: str | None
    plain: str | None
    fallbacks_issued_at: datetime
    interruption_cap: int | None
    l2_fallback: tuple[str, ...]
    tau: float
    spread: tuple[float, float]
    modality: str
    pin_act: bool
    clock: datetime
    regime: str


@dataclass
class SimulatedRunResult:
    """Everything a simulated study produced: the stored rows (real records), the truth and the block states."""

    participant_key: str
    family: str
    policy: str
    params: SimulatedPersonParams
    sessions: list[SessionRow] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    trials: list[StoredTrial] = field(default_factory=list)
    burden: list[BurdenReport] = field(default_factory=list)
    events: list[EngineEvent] = field(default_factory=list)
    preferences: tuple[Preference, ...] = ()
    consents: tuple[ConsentPeriod, ...] = ()
    truth: list[TrialTruth] = field(default_factory=list)
    block_starts: dict[str, BlockStart] = field(default_factory=dict)
    # Engine state before each session's first Phase B.
    snapshots: dict[int, bytes] = field(default_factory=dict)
    preference_replacements: int = 0
    refused_blocks: int = 0
    blocks_ended_by_refresh: int = 0

    def rows(self) -> BundleRows:
        return BundleRows(
            consents=tuple(self.consents),
            sessions=tuple(self.sessions),
            decisions=tuple(self.decisions),
            trials=tuple(self.trials),
            burden=tuple(self.burden),
            events=tuple(self.events),
            preferences=tuple(self.preferences),
            learning_reset_at=None,
            versions=_versions_block(),
        )

    def bundle(self) -> dict[str, Any]:
        """The export bundle of section 10.9 built from the records' ``to_dict()``; marked simulated in every
        session row (``evidence_kind``)."""
        preferences = sorted(self.preferences, key=lambda p: p.created_at)
        return {
            "bundle_schema": BUNDLE_SCHEMA,
            "record_schema": RECORD_SCHEMA_VERSION,
            "consents": [period.to_dict() for period in self.consents],
            "sessions": [row.to_dict() for row in self.sessions],
            "decisions": [row.to_dict() for row in self.decisions],
            "trials": [row.to_dict() for row in self.trials],
            "burden": [row.to_dict() for row in self.burden],
            "events": [row.to_dict() for row in self.events],
            "preferences": [preference_to_dict(p) for p in preferences],
            "learning_reset_at": None,
            "versions": _versions_block(),
        }


def _versions_block() -> dict[str, str]:
    return {
        "strategy": STRATEGY_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "model": MODEL_VERSION,
        "prior": PRIOR_VERSION,
        "policy": POLICY_VERSION,
        "sdk": SDK_VERSION,
    }


def _uuid(seed: int, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in (seed, *parts)).encode()).digest()
    return str(uuid.UUID(bytes=digest[:16], version=4))


def _round_px(value: float) -> int:
    """Endpoints are integer surface px before any posterior (``Math.round`` in the UI)."""
    return int(math.floor(value + 0.5))


@dataclass
class _TrialContext:
    """The rules a trial runs under: what the UI holds at cue."""

    regime: str
    l1_decision_id: str
    table: Mapping[str, Any] | None
    table_decision_id: str | None
    block_mode: str
    plain: str | None
    fallbacks_issued_at: datetime
    interruption_cap: int | None
    l2_fallback: tuple[str, ...]
    pin_act: bool
    tau: float
    spread: tuple[float, float]
    modality: str
    k_start: int
    idle: bool
    trial_index: int
    # Counterfactual replays draw L2 from a target.
    l2_chooser: Callable[[str, int, Generator], str] | None = None
    allow_veto: bool = True


@dataclass
class _TrialRun:
    activations: list[ActivationRecord]
    committed_index: int | None
    commit_t_ms: float | None
    commit_via: str
    corrections: int
    elapsed_ms: float
    veto: VetoRecord | None
    revert: RevertRecord | None
    interrupted: bool
    interruption_reason: str | None
    k_end: int
    l2_veto: bool
    acting_under_ambiguity: bool
    first_endpoint: tuple[int, int] | None
    first_t_ms: float
    regime_at_end: str


class SimulatedStudyRunner:
    """Runs the full 3-session protocol (section 12.3) for one simulated person against a policy under test:
    generates layouts (Python port of the study generator over study_plan.LAYOUT_GRID: pointing W in {16, 28, 44},
    gap in {2, 8}, D in {120, 240, 360}; menus N in {6, 12, 16} items of 24 px, section 12.4), simulates endpoints
    and responses, computes posteriors with core.target_posterior, applies commit rules (Python port of section
    12.5), builds TrialRecords and BurdenReports, and drives the engine (or a baseline policy) exactly like the
    service does: SessionRows with simulated day indices, a seed_salt derived from the runner seed and the
    comparator ranking from rank_fixed_regimes over the simulated session 1, recorded_at and seq set per row. It
    stores the person's state at every block start for counterfactual_block_value."""

    def __init__(
        self,
        person: SimulatedPerson,
        policy_factory: Callable[[], StudyPolicy],
        *,
        seed: int,
    ) -> None:
        self.person = person
        self.policy_factory = policy_factory
        self.seed = int(seed)
        self._counter = 0
        self._t_ref_cache: dict[int, Mapping[str, float]] = {}
        self._hand_rng = Generator(PCG64([self.seed, person.params.person_seed, 7]))
        # The block the oracle evaluates.
        self._pending: tuple[tuple[PlannedTrial, ...], _SessionRun] | None = None

    # ------------------------------------------------------------------------------------------- identifiers

    def _id(self, kind: str) -> str:
        self._counter += 1
        return _uuid(self.seed, self.person.params.participant_key, kind, self._counter)

    # ------------------------------------------------------------------------------------------- the study

    def run(self, sessions: int = 3) -> SimulatedRunResult:
        params = self.person.params
        key = params.participant_key
        preferences = PreferenceSet(params.stated_preferences)
        policy = self.policy_factory()
        policy.start(participant_key=key, preferences=preferences, modality=params.modality)
        if isinstance(policy, BaselinePolicy) and policy.kind == "oracle":
            policy.set_oracle(self._oracle_for(policy))
        result = SimulatedRunResult(
            participant_key=key,
            family=params.family,
            policy=policy.name,
            params=params,
            preferences=params.stated_preferences,
            consents=(_sim_consent(key),),
        )
        rng = Generator(PCG64([self.seed, params.person_seed]))
        self.person.state = PersonState()
        days_seen = 0
        for number in range(1, sessions + 1):
            self._run_session(policy, result, number, preferences, rng, days_seen)
            days_seen += SESSION_DAY_GAP
        result.preference_replacements = policy.preference_replacements
        return result

    def _run_session(
        self,
        policy: StudyPolicy,
        result: SimulatedRunResult,
        number: int,
        preferences: PreferenceSet,
        rng: Generator,
        day: int,
    ) -> None:
        params = self.person.params
        key = params.participant_key
        modality = params.modality
        started = SIM_EPOCH + timedelta(days=day, minutes=float(rng.uniform(0.0, 120.0)))
        self.person.state.abs_minute = (started - SIM_EPOCH).total_seconds() / 60.0
        self.person.begin_day(day, rng)
        adaptive = number >= 2
        ranking: tuple[str, ...] = ()
        if adaptive and result.sessions:
            first = result.sessions[0]
            ranking = rank_fixed_regimes(
                [s for s in result.trials if s.trial.session_id == first.session_id],
                [r for r in result.burden if r.session_id == first.session_id],
                [d for d in result.decisions if d.session_id == first.session_id],
                modality=modality,
            )
        plan = build_session_plan(
            session_number=number,
            adaptive_allowed=adaptive,
            comparator_available=bool(ranking),
            plan_seed=plan_seed(key, number),
            burden_every_blocks=1,
            n_arms=len(l1_arms(modality)),
        )
        settings = asdict(
            EngineSettings(
                **PROTOCOL_HYPERPARAMETERS,
                adaptive_allowed=adaptive,
                learn_across_sessions=True,
                gaze_allowed=False,
                hand_pointer_allowed=modality == "hand_pointer",
            )
        )
        session_id = self._id("session")
        row = SessionRow(
            session_id=session_id,
            participant_key=key,
            context="study",
            day_index=SIM_EPOCH.date().toordinal() + day,
            session_number=number,
            protocol_id=STUDY_PROTOCOL_ID,
            protocol_sha256=study_protocol_sha256(),
            evidence_kind=EvidenceKind.SIMULATED,
            started_at=started,
            ended_at=None,
            end_reason=None,
            device=DeviceContext(
                modality=modality,  # type: ignore[arg-type]
                pointer_types=("touch",) if modality == "touchpad" else ("mouse",),
                dpr=1.0,
                surface_w=SURFACE[0],
                surface_h=SURFACE[1],
                gaze_available=False,
                hand_pointer=modality == "hand_pointer",
            ),
            plan=plan,
            layout_seed=int.from_bytes(
                hashlib.sha256(f"{self.seed}|{key}|{number}|layout".encode()).digest()[:4], "big"
            ),
            seed_salt=hashlib.sha256(f"{self.seed}|{key}|{number}|salt".encode()).hexdigest()[:32],
            gaze_check_accuracy=None,
            comparator_ranking=ranking,
            settings=settings,
            protocol_deviations=(),
            versions=_versions_block(),
        )
        policy.begin_session(row)
        spread = (1.0, 0.0)
        if modality != "keyboard":
            spread = (spread_r0(modality), floor_px(modality))
        session = _SessionRun(
            row=row,
            clock=started,
            seq=0,
            fallbacks=policy.fallbacks(modality, now=started),
            spread=spread,
            corrections_recent=[],
        )
        self.person.begin_session(sum(block.trials for block in plan.blocks), rng)
        previous_regime: str | None = None
        hand_quality: float | None = None
        for block in plan.blocks:
            if block.phase == "practice":
                practice = session.fallbacks.practice
                if practice is None:
                    result.refused_blocks += 1
                    break
                self.person.practise(practice, block.trials, minutes=block.trials * 3.0 / 60.0)
                session.clock += timedelta(seconds=block.trials * 3.0)
                continue
            if block.phase in ("B", "probe", "comparator") and number not in result.snapshots:
                snapshot = policy.snapshot()
                if snapshot is not None:
                    result.snapshots[number] = snapshot
            outcome = self._run_block(
                policy,
                result,
                session,
                block,
                previous_regime,
                hand_quality,
                rng,
            )
            if outcome is None:
                continue
            previous_regime, hand_quality, ended_for_refresh, regime, decision_id = outcome
            if block.idle_after and not ended_for_refresh:
                self._run_idle(policy, result, session, block, regime, decision_id, rng)
        ended = session.clock + timedelta(seconds=30)
        policy.end_session(ended_at=ended, reason="completed")
        result.sessions.append(replace(row, ended_at=ended, end_reason="completed"))

    # ------------------------------------------------------------------------------------------- one block

    def _record_decision(
        self,
        result: SimulatedRunResult,
        session: _SessionRun,
        payload: Mapping[str, Any],
        now: datetime,
    ) -> DecisionRecord:
        level_name = payload["level"]
        level = DecisionLevel.AMBIGUITY_RESPONSE
        if level_name != "ambiguity_table":
            level = DecisionLevel(level_name)
        record = DecisionRecord(
            decision_id=payload["decision_id"],
            session_id=session.row.session_id,
            participant_key=session.row.participant_key,
            seq=session.next_seq(),
            level=level,
            mode=PolicyMode(payload["mode"]),
            behavior="policy",
            created_at=now,
            superseded_by=None,
            payload=dict(payload),
            applied=None,
            fallback_reason=None,
            versions=dict(payload.get("versions", {})),
        )
        result.decisions.append(record)
        return record

    def _block_context(
        self,
        session: _SessionRun,
        block: Any,
        trials: Sequence[PlannedTrial],
        previous: str | None,
        hand_quality: float | None,
    ) -> BlockContext:
        row = session.row
        return BlockContext(
            session_id=row.session_id,
            block_index=block.index,
            phase=block.phase,
            task=block.task,
            layouts=tuple(trial.summary for trial in trials),
            device=row.device,
            clock=ClockContext(
                day_index=row.day_index,
                session_minute=(session.clock - row.started_at).total_seconds() / 60.0,
                trials_today=session.trials_today,
                days_since_last_session=None if row.session_number == 1 else float(SESSION_DAY_GAP),
            ),
            previous_regime=previous,
            hand_quality_recent=hand_quality,
        )

    def _run_block(
        self,
        policy: StudyPolicy,
        result: SimulatedRunResult,
        session: _SessionRun,
        block: Any,
        previous_regime: str | None,
        hand_quality: float | None,
        rng: Generator,
    ) -> tuple[str | None, float | None, bool, str, str] | None:
        row = session.row
        modality = row.device.modality
        tau = float(row.settings["ambiguity_threshold"])
        layout_of = (block.task, row.layout_seed, block.index)
        planned = tuple(generate_layout(*layout_of, t) for t in range(block.trials))
        context = self._block_context(session, block, planned, previous_regime, hand_quality)
        decision_id = self._id("decision")
        now = session.clock
        self._pending = (planned, session)
        try:
            payload = policy.decide_block(decision_id=decision_id, block=context, now=now)
        except NoAllowedRegime:
            result.refused_blocks += 1
            return None
        finally:
            self._pending = None
        self._record_decision(result, session, payload, now)
        regime = str(payload["variant"])
        mode = str(payload["mode"])
        if payload.get("spread"):
            session.spread = (float(payload["spread"]["r"]), float(payload["spread"]["floor"]))
        pin_act = PreferenceSet(self.person.params.stated_preferences).pin_act(
            context="study", modality=modality, participant_key=row.participant_key
        )
        table = None
        table_id = None
        if regime in ACT_GUARDED:
            table_id = self._id("table")
            table = policy.ambiguity_table(
                decision_id=table_id,
                block_decision_id=decision_id,
                block=context,
                start_trial_index=0,
                trials_in_block=block.trials,
                now=now,
            )
            table_record = self._record_decision(result, session, table, now)
        result.block_starts[decision_id] = BlockStart(
            decision_id=decision_id,
            session_number=row.session_number,
            block_index=block.index,
            phase=block.phase,
            task=block.task,
            person=self.person.copy_state(),
            trials=planned,
            previous_regime=previous_regime,
            plain=session.fallbacks.plain,
            fallbacks_issued_at=session.fallbacks.issued_at,
            interruption_cap=session.fallbacks.interruption_cap,
            l2_fallback=tuple(session.fallbacks.l2),
            tau=tau,
            spread=session.spread,
            modality=modality,
            pin_act=pin_act,
            clock=now,
            regime=regime,
        )
        self.person.set_regime(regime)
        session.clock += timedelta(seconds=4)
        k = 0
        ended_for_refresh = False
        qualities: list[float] = []
        times: list[tuple[PlannedTrial, float]] = []
        errors = 0
        vetoes = 0
        counted = 0
        for index, trial_plan in enumerate(planned):
            offers = self._offers(policy, result, session, block, index, mode, rng)
            ctx = _TrialContext(
                regime=regime,
                l1_decision_id=decision_id,
                table=table,
                table_decision_id=table_id,
                block_mode=mode,
                plain=session.fallbacks.plain,
                fallbacks_issued_at=session.fallbacks.issued_at,
                interruption_cap=session.fallbacks.interruption_cap,
                l2_fallback=tuple(session.fallbacks.l2),
                pin_act=pin_act,
                tau=tau,
                spread=session.spread,
                modality=modality,
                k_start=k,
                idle=False,
                trial_index=index,
            )
            run = self._simulate_trial(trial_plan, ctx, rng)
            k = run.k_end
            record, truth = self._trial_record(
                session,
                block,
                trial_plan,
                ctx,
                run,
                offers,
                table_ids=(table_id,) if table_id else (),
            )
            observation = policy.observe_trial(record)
            result.trials.append(observation.stored)
            result.events.extend(observation.events)
            result.truth.append(truth)
            session.trials_today += 1
            session.corrections_recent = (session.corrections_recent + [run.corrections > 0])[-3:]
            if record.hand is not None:
                qualities.append(record.hand.quality_mean)
            if not record.interrupted:
                times.append((trial_plan, truth.time_s))
                errors += int(truth.ua)
                counted += 1
            vetoes += int(record.veto is not None)
            self.person.after_trial(trial_plan, run.regime_at_end, run.elapsed_ms / 60000.0, rng)
            if run.l2_veto and mode != "pinned" and index + 1 < len(planned) and table is not None:
                new_id = self._id("table")
                new_table = policy.ambiguity_table(
                    decision_id=new_id,
                    block_decision_id=decision_id,
                    block=context,
                    start_trial_index=index + 1,
                    trials_in_block=len(planned) - index - 1,
                    now=session.clock,
                )
                result.decisions[result.decisions.index(table_record)] = replace(
                    table_record,
                    superseded_by=new_id,
                )
                table_record = self._record_decision(result, session, new_table, session.clock)
                table, table_id = new_table, new_id
            if observation.refresh_required:
                session.fallbacks = policy.fallbacks(modality, now=session.clock)
                result.blocks_ended_by_refresh += 1
                ended_for_refresh = True
                break
            if run.interruption_reason == "no_allowed_regime":
                break
        if block.burden_after and counted:
            summary = BlockSummary(
                regime=regime,
                trials=counted,
                mean_time_z=self._time_z(times),
                interruptions_per_trial=k / max(counted, 1),
                errors_per_trial=errors / max(counted, 1),
                vetoes=vetoes,
            )
            ratings = self.person.burden(summary, rng)
            session.clock += timedelta(seconds=8)
            report = BurdenReport(
                report_id=self._id("burden"),
                session_id=row.session_id,
                seq=session.next_seq(),
                block_index=block.index,
                scope="block",
                effort=ratings["effort"],
                intrusiveness=ratings["intrusiveness"],
                control=ratings["control"],
                tlx=None,
                predictability=None,
                preferred_regime=None,
                skipped=tuple(ratings["skipped"]),
                recorded_at=session.clock,
            )
            policy.observe_burden(report)
            result.burden.append(report)
        quality = float(np.mean(qualities)) if qualities else None
        return regime, quality, ended_for_refresh, regime, decision_id

    def _time_z(self, pairs: Sequence[tuple[PlannedTrial, float]]) -> float:
        """Mean log trial time over the person's plain time for the same target (reaction, pointing and, for menus,
        search at the current familiarity), in units of 0.35 (the model's log-time noise)."""
        if not pairs:
            return 0.0
        values = []
        for plan, seconds in pairs:
            width = min(plan.summary.w_px, plan.summary.h_px)
            expected_ms = 250.0 + self.person.fitts_ms(
                math.log2(max(plan.distance, 1.0) / width + 1.0),
            )
            if plan.task == "menu" and plan.intended is not None:
                expected_ms += self.person.menu_search_ms(plan.summary.menu_n, plan.intended)
            values.append(math.log(max(seconds, 0.05)) - math.log(expected_ms / 1000.0))
        return float(np.mean(values)) / 0.35

    def _offers(
        self,
        policy: StudyPolicy,
        result: SimulatedRunResult,
        session: _SessionRun,
        block: Any,
        index: int,
        mode: str,
        rng: Generator,
    ) -> tuple[OfferResponse, ...]:
        """The struggle signal at trial start (two corrections among the last three trials) requests an offer,
        except in probe and pinned blocks, which make no offers (9.4)."""
        if mode in ("probe", "pinned") or sum(session.corrections_recent) < 2:
            return ()
        row = session.row
        context = OfferContext(
            session_id=row.session_id,
            block_index=block.index,
            trial_index=index,
            modality=row.device.modality,  # type: ignore[arg-type]
            task=block.task,
            struggle_kind="corrections",
            keyboard_available=row.device.modality != "keyboard",
        )
        decision_id = self._id("offer")
        payload = policy.offer(decision_id=decision_id, context=context, now=session.clock)
        self._record_decision(result, session, payload, session.clock)
        variant = str(payload["variant"])
        response = "ignored" if variant == "no_offer" else self.person.responds("offer", rng=rng)
        session.corrections_recent = []
        return (
            OfferResponse(
                decision_id=decision_id,
                variant=variant,
                response=response,
                fallback_reason=None,
            ),
        )

    # ------------------------------------------------------------------------------------------- one trial

    def _simulate_trial(self, plan: PlannedTrial, ctx: _TrialContext, rng: Generator) -> _TrialRun:
        """The commit-rule port of section 12.5 for one trial (or one idle epoch)."""
        person = self.person
        limit = plan.limit_ms
        activations: list[ActivationRecord] = []
        committed: int | None = None
        commit_t: float | None = None
        via = "none"
        corrections = 0
        k = ctx.k_start
        l1_counted = False
        draws_used = 0
        veto: VetoRecord | None = None
        revert: RevertRecord | None = None
        l2_veto = False
        acting = False
        first_endpoint: tuple[int, int] | None = None
        first_t = 0.0
        regime = ctx.regime
        reverted_from: int | None = None

        if ctx.idle:
            return self._simulate_idle(plan, ctx, rng)

        if (
            ctx.allow_veto
            and regime != "assistance_off"
            and person.responds("veto_l1", variant=regime, rng=rng) == "veto"
        ):
            veto = VetoRecord(decision_id=ctx.l1_decision_id, level=DecisionLevel.BLOCK_REGIME)
            if ctx.plain is None:
                return _TrialRun(
                    activations,
                    None,
                    None,
                    "none",
                    0,
                    0.0,
                    veto,
                    None,
                    True,
                    "no_allowed_regime",
                    k,
                    False,
                    False,
                    None,
                    0.0,
                    regime,
                )
            revert = RevertRecord(
                reason="veto",
                from_activation=0,
                regime=ctx.plain,
                fallbacks_issued_at=ctx.fallbacks_issued_at,
            )
            reverted_from = 0
            person.state.vetoes_in_block += 1

        intended = plan.intended
        assert intended is not None
        current = revert.regime if revert is not None else regime
        t = person.first_aim_ms(plan, current, first_aim_width(plan, current, ctx.modality), rng)
        done = False
        while t < limit and len(activations) < MAX_ACTIVATIONS and not done:
            index = len(activations)
            reverted = reverted_from is not None and index >= reverted_from
            current = revert.regime if (revert is not None and reverted) else regime
            first = index == 0
            aimed = intended
            slip = person.params.selection_error * (1.0 + person.switch_transient() / 500.0)
            if first and rng.random() < slip:
                others = [i for i in range(len(plan.candidates)) if i != intended]
                aimed = others[int(rng.integers(0, len(others)))]
            expanded: int | None = None
            scale = 1.0
            if current.startswith("enlarge_target"):
                scale = float(VARIANTS[current].param("scale"))
                p_x = ENLARGE_RIGHT["hand_pointer" if ctx.modality == "hand_pointer" else "pointer"]
                lands_on_aimed = rng.random() < p_x
                expanded = aimed if lands_on_aimed else _nearest_neighbour(plan.candidates, aimed)
            rendered = rendered_rects(plan.candidates, expanded, scale)
            # The scatter is planned for the cue-time rect (spec 7.3); the hit and the posterior read the rendered ones.
            x, y = person.endpoint(plan.candidates[aimed], plan, rendered, rng, regime=current)
            xi = min(max(_round_px(x), -100), 20000)
            yi = min(max(_round_px(y), -100), 20000)
            press = literal_hit(rendered, xi, yi)
            press_index = press if press is not None else -1
            post = compute_posterior(
                rendered,
                (xi, yi),
                modality=ctx.modality,
                r=ctx.spread[0],
                floor=ctx.spread[1],
                surface=SURFACE,
                press_index=press_index,
            )
            hit = post.hit_index
            a_gate = float(post.ambiguity_gate)
            if first:
                first_endpoint = (xi, yi)
                first_t = t
            l2: L2Applied | None = None
            result_kind = "none"
            # The regime row of the transition table.
            if current == "assistance_off":
                result_kind = "commit" if hit is not None else "none"
            elif current in ACT_GUARDED:
                if hit is not None and a_gate <= ctx.tau:
                    result_kind = "commit"
                else:
                    cell = cell_id(
                        ambiguity_bin(a_gate, ctx.tau) or "b3",
                        post.n_plausible,
                        hit is not None,
                    )
                    l2, draws_used = self._l2_response(ctx, cell, k, draws_used, reverted, rng)
                    result_kind = {
                        "act": "commit" if hit is not None else "none",
                        "confirm_selection:second_click": "confirming",
                        "wait_for_evidence": "none",
                        "ask_clarification:2": "asked",
                        "ask_clarification:3": "asked",
                    }[l2.variant]
                    ambiguous_hit = hit is not None and a_gate > ctx.tau
                    if l2.variant == "act" and ambiguous_hit and not ctx.pin_act:
                        acting = True
            elif current.startswith("confirm_selection:hold"):
                result_kind = "held" if hit is not None else "none"
            elif current.startswith("confirm_selection"):
                result_kind = "confirming" if hit is not None else "none"

            outcome = "none"
            record_result = {
                "commit": "committed",
                "confirming": "confirming",
                "held": "held",
                "asked": "asked",
            }.get(result_kind, "none")
            committed_here: int | None = None
            if record_result in ("confirming", "held"):
                if l2 is None:
                    if not l1_counted:
                        k += 1
                        l1_counted = True
                else:
                    k += 1
            elif record_result == "asked":
                k += 1
            step_end = t
            if result_kind == "commit":
                outcome, committed_here = "committed", hit
                via = "direct"
            elif result_kind == "confirming":
                answer = person.responds("confirm", correct=hit == intended, rng=rng)
                step_end = t + 250.0 + person.fitts_ms(2.0)
                if step_end > limit:
                    outcome = "open_at_end"
                elif answer == "confirm":
                    outcome, committed_here, via = "committed", hit, "confirm"
                else:
                    outcome = "cancelled"
                    corrections += 1
            elif result_kind == "held":
                hold_ms = float(VARIANTS[current].param("hold_ms") or 400)
                answer = person.responds("hold", correct=hit == intended, rng=rng)
                step_end = t + hold_ms + 150.0
                if answer == "complete" and step_end > limit:
                    outcome = "open_at_end"
                elif answer == "complete":
                    outcome, committed_here, via = "committed", hit, "hold"
                elif answer == "release":
                    outcome = "cancelled"
                    corrections += 1
                    step_end = t + 0.5 * hold_ms
                else:
                    outcome = "aborted"
                    step_end = t + 0.5 * hold_ms
            elif result_kind == "asked":
                k_options = int(VARIANTS[l2.variant].param("k"))  # type: ignore[union-attr]
                listed = post.top_k(k_options)
                step_end = t + 250.0 + 150.0 * math.log2(k_options + 2) + person.fitts_ms(2.0)
                if step_end > limit:
                    outcome = "open_at_end"
                elif intended in listed:
                    outcome, committed_here, via = "committed", intended, "ask"
                else:
                    outcome = "cancelled"
                    corrections += 1
            activations.append(
                ActivationRecord(
                    index=index,
                    t_ms=float(min(t, 60000.0)),
                    modality=ctx.modality,
                    x=xi,
                    y=yi,
                    hit_index=hit,
                    a_gate=a_gate,
                    n_plausible=int(post.n_plausible),
                    press_index=press_index,
                    expanded_index=expanded,
                    hit_rect=rendered[hit] if hit is not None else None,
                    gaze=None,
                    l2=l2,
                    irreversible_rule=False,
                    result=record_result,  # type: ignore[arg-type]
                    outcome=outcome,  # type: ignore[arg-type]
                    committed_index=committed_here,
                )
            )
            if committed_here is not None:
                committed = committed_here
                commit_t = step_end
                if committed_here == intended:
                    done = True
                    t = step_end
                    break
                corrections += 1  # the person undoes the wrong commit
                t = step_end + 500.0
            elif outcome == "open_at_end":
                t = limit
                break
            else:
                t = step_end + 250.0 + person.fitts_ms(1.0) * float(math.exp(rng.normal(0.0, 0.2)))
            if (
                ctx.allow_veto
                and l2 is not None
                and l2.variant in _VETOABLE_L2
                and veto is None
                and not reverted
                and person.responds("veto_l2", variant=l2.variant, rng=rng) == "veto"
            ):
                veto = VetoRecord(
                    decision_id=l2.decision_id,
                    level=DecisionLevel.AMBIGUITY_RESPONSE,
                )
                person.state.vetoes_in_block += 1
                if ctx.plain is None:
                    return _TrialRun(
                        activations,
                        committed,
                        commit_t,
                        via if committed is not None else "none",
                        corrections,
                        min(t, limit),
                        veto,
                        None,
                        True,
                        "no_allowed_regime",
                        k,
                        False,
                        acting,
                        first_endpoint,
                        first_t,
                        regime,
                    )
                revert = RevertRecord(
                    reason="veto",
                    from_activation=index + 1,
                    regime=ctx.plain,
                    fallbacks_issued_at=ctx.fallbacks_issued_at,
                )
                reverted_from = index + 1
                l2_veto = l2.source == "table"
        if committed is None:
            via = "none"
        elapsed = min(t, limit)
        regime_at_end = revert.regime if revert is not None else regime
        return _TrialRun(
            activations,
            committed,
            commit_t if committed is not None else None,
            via,
            corrections,
            elapsed,
            veto,
            revert,
            False,
            None,
            k,
            l2_veto,
            acting,
            first_endpoint,
            first_t,
            regime_at_end,
        )

    def _l2_response(
        self, ctx: _TrialContext, cell: str, k: int, draws_used: int, reverted: bool, rng: Generator
    ) -> tuple[L2Applied, int]:
        """The L2 response of one ambiguous activation: the cap rule first (outside a pin_act collapse), then the
        table draw, the fixed rule (idle epochs, reverted trials) or the fallback (no table)."""
        decision_id = ctx.table_decision_id or ctx.l1_decision_id
        if not ctx.pin_act and ctx.interruption_cap is not None and k >= ctx.interruption_cap:
            return (
                L2Applied(decision_id, cell, min(draws_used, 4), "wait_for_evidence", "cap", None),
                draws_used,
            )
        fixed = _first_applicable(ctx.l2_fallback, cell)
        if ctx.idle or reverted:
            return L2Applied(
                decision_id,
                cell,
                min(draws_used, 4),
                fixed,
                "fixed_rule",
                None,
            ), draws_used
        if ctx.table is None and ctx.l2_chooser is None:
            return L2Applied(
                decision_id,
                cell,
                min(draws_used, 4),
                fixed,
                "fallback",
                "late",
            ), draws_used
        # After the fourth draw of a trial: wait_for_evidence, deterministic (9.3).
        if draws_used >= 4:
            return L2Applied(decision_id, cell, 4, "wait_for_evidence", "table", None), draws_used
        # Counterfactual replay: the target's probabilities, sampled directly.
        if ctx.l2_chooser is not None:
            variant = ctx.l2_chooser(cell, draws_used, rng)
            return L2Applied(decision_id, cell, draws_used, variant, "table", None), draws_used + 1
        row = ctx.trial_index - int(ctx.table["start_trial_index"])
        entry = next(c for c in ctx.table["cells"] if c["cell_id"] == cell)
        variant = select_from_cell(
            entry["order"],
            entry["propensities"],
            float(ctx.table["draws"][row][draws_used]),
        )
        return L2Applied(decision_id, cell, draws_used, variant, "table", None), draws_used + 1

    def _simulate_idle(self, plan: PlannedTrial, ctx: _TrialContext, rng: Generator) -> _TrialRun:
        """Stray activations during a 30 s idle epoch under the preceding block's regime (fixed L2 rule, its own
        interruption count). Any commit is an unintended activation; the person cancels every step it opens."""
        person = self.person
        n_stray = int(
            rng.poisson(person.params.idle_activation_rate_per_min * IDLE_EPOCH_MS / 60000.0),
        )
        n_times = min(n_stray, MAX_ACTIVATIONS)
        times = sorted(float(rng.uniform(0.0, IDLE_EPOCH_MS)) for _ in range(n_times))
        activations: list[ActivationRecord] = []
        committed = None
        commit_t = None
        via = "none"
        corrections = 0
        k = 0
        l1_counted = False
        for index, t in enumerate(times):
            aimed = int(rng.integers(0, len(plan.candidates)))
            rect = plan.candidates[aimed]
            # A careless, unaimed contact.
            loose = Rect(cx=rect.cx, cy=rect.cy, w=rect.w * 4.0, h=rect.h * 4.0)
            x, y = person.endpoint(loose, plan, plan.candidates, rng, regime=ctx.regime)
            xi, yi = min(max(_round_px(x), -100), 20000), min(max(_round_px(y), -100), 20000)
            press = literal_hit(plan.candidates, xi, yi)
            press_index = press if press is not None else -1
            post = compute_posterior(
                plan.candidates,
                (xi, yi),
                modality=ctx.modality,
                r=ctx.spread[0],
                floor=ctx.spread[1],
                surface=SURFACE,
                press_index=press_index,
            )
            hit = post.hit_index
            a_gate = float(post.ambiguity_gate)
            l2 = None
            kind = "none"
            if ctx.regime == "assistance_off":
                kind = "commit" if hit is not None else "none"
            elif ctx.regime in ACT_GUARDED:
                if hit is not None and a_gate <= ctx.tau:
                    kind = "commit"
                else:
                    cell = cell_id(
                        ambiguity_bin(a_gate, ctx.tau) or "b3",
                        post.n_plausible,
                        hit is not None,
                    )
                    l2, _ = self._l2_response(ctx, cell, k, 0, True, rng)
                    kind = {
                        "act": "commit" if hit is not None else "none",
                        "wait_for_evidence": "none",
                    }.get(l2.variant, "asked" if l2.variant.startswith("ask") else "confirming")
            elif hit is not None:
                kind = "held" if ctx.regime.startswith("confirm_selection:hold") else "confirming"
            result_kind = {"commit": "committed"}.get(kind, kind)
            if result_kind in ("confirming", "held") and l2 is None:
                if not l1_counted:
                    k += 1
                    l1_counted = True
            elif result_kind in ("confirming", "asked"):
                k += 1
            outcome = "none"
            committed_here = None
            if result_kind == "committed":
                outcome, committed_here = "committed", hit
                committed, commit_t, via = hit, t, "direct"
            elif result_kind in ("confirming", "held", "asked"):
                outcome = "cancelled"
                corrections += 1
            activations.append(
                ActivationRecord(
                    index=index,
                    t_ms=t,
                    modality=ctx.modality,
                    x=xi,
                    y=yi,
                    hit_index=hit,
                    a_gate=a_gate,
                    n_plausible=int(post.n_plausible),
                    press_index=press_index,
                    expanded_index=None,
                    hit_rect=plan.candidates[hit] if hit is not None else None,
                    gaze=None,
                    l2=l2,
                    irreversible_rule=False,
                    result=result_kind,  # type: ignore[arg-type]
                    outcome=outcome,  # type: ignore[arg-type]
                    committed_index=committed_here,
                )
            )
        return _TrialRun(
            activations,
            committed,
            commit_t,
            via,
            corrections,
            IDLE_EPOCH_MS,
            None,
            None,
            False,
            None,
            k,
            False,
            False,
            None,
            0.0,
            ctx.regime,
        )

    def _build_record(
        self,
        plan: PlannedTrial,
        ctx: _TrialContext,
        run: _TrialRun,
        offers: tuple[OfferResponse, ...],
        *,
        session_id: str,
        block_index: int,
        cue_at: datetime,
        seq: int,
        table_ids: tuple[str, ...],
        hand: bool,
    ) -> TrialRecord:
        """The TrialRecord the study UI would send for a simulated trial (``recorded_at`` is the end of the trial;
        a service sets ``recorded_at`` and ``seq`` itself)."""
        time_ms = (
            run.commit_t_ms
            if run.committed_index is not None and run.committed_index == plan.intended
            else run.elapsed_ms
        )
        time_ms = float(min(max(time_ms or 0.0, 0.0), 60000.0))
        movement = None
        if not ctx.idle and run.first_endpoint is not None and plan.intended is not None:
            target = plan.candidates[plan.intended]
            width = min(target.w, target.h)
            rt = 250.0
            movement = MovementSummary(
                rt_ms=rt,
                mt_ms=float(min(max(run.first_t_ms - rt, 1.0), 60000.0)),
                path_ratio=1.2,
                submovements=2,
                peak_speed_px_s=float(
                    min(plan.distance / max(run.first_t_ms - rt, 50.0) * 1800.0, 50000.0),
                ),
                reentries=0,
                endpoint_dx_norm=float(
                    min(max((run.first_endpoint[0] - target.cx) / width, -20.0), 20.0),
                ),
                endpoint_dy_norm=float(
                    min(max((run.first_endpoint[1] - target.cy) / width, -20.0), 20.0),
                ),
            )
        hand_summary = None
        if hand:
            loss_rate = self.person.params.tracking_loss_per_trial
            quality = float(
                min(max(0.85 - 2.0 * loss_rate + 0.05 * self._hand_rng.normal(), 0.0), 1.0),
            )
            losses = int(self._hand_rng.poisson(loss_rate))
            hand_summary = HandSummary(
                quality_mean=quality,
                quality_min=max(quality - 0.2, 0.0),
                tracking_losses=min(losses, 100),
            )
        spread = None
        if ctx.modality != "keyboard":
            spread = SpreadUsed(r=ctx.spread[0], floor=ctx.spread[1])
        return TrialRecord(
            trial_id=self._id("trial"),
            session_id=session_id,
            block_index=block_index,
            trial_index=ctx.trial_index,
            task="idle" if ctx.idle else plan.task,  # type: ignore[arg-type]
            regime=ctx.regime,
            regime_decision_id=ctx.l1_decision_id,
            table_decision_ids=table_ids,
            applied_regime=ctx.regime,
            l1_fallback_reason=None,
            fallbacks_issued_at=ctx.fallbacks_issued_at,
            spread=spread,
            layout=plan.summary,
            candidates=plan.candidates,
            surface_w=SURFACE[0],
            surface_h=SURFACE[1],
            dpr=1.0,
            reversible=tuple(True for _ in plan.candidates),
            cue_at=cue_at,
            intended_index=None if ctx.idle else plan.intended,
            modality=ctx.modality,
            activations=tuple(run.activations),
            revert=run.revert,
            committed_index=run.committed_index,
            commit_t_ms=None if run.commit_t_ms is None else float(min(run.commit_t_ms, 60000.0)),
            commit_via=run.commit_via,  # type: ignore[arg-type]
            corrections=min(run.corrections, 50),
            time_ms=time_ms,
            limit_ms=plan.limit_ms,
            duration_ms=IDLE_EPOCH_MS if ctx.idle else time_ms,
            movement=movement,
            veto=run.veto,
            offers=offers,
            hand=hand_summary,
            gaze=None,
            interrupted=run.interrupted,
            interruption_reason=run.interruption_reason,
            explanation_views=0,
            recorded_at=cue_at + timedelta(milliseconds=time_ms + 300.0),
            seq=seq,
        )

    def _trial_record(
        self,
        session: _SessionRun,
        block: Any,
        plan: PlannedTrial,
        ctx: _TrialContext,
        run: _TrialRun,
        offers: tuple[OfferResponse, ...],
        *,
        table_ids: tuple[str, ...],
    ) -> tuple[TrialRecord, TrialTruth]:
        row = session.row
        cue_at = session.clock
        record = self._build_record(
            plan,
            ctx,
            run,
            offers,
            session_id=row.session_id,
            block_index=block.index,
            cue_at=cue_at,
            seq=session.next_seq(),
            table_ids=table_ids,
            hand=bool(row.settings.get("hand_pointer_allowed")),
        )
        session.clock = cue_at + timedelta(milliseconds=record.time_ms + 1300.0)
        record = replace(record, recorded_at=session.clock)
        labels = derive_labels(record)
        idle = ctx.idle
        complete = bool(labels.complete) and not record.interrupted
        truth = TrialTruth(
            trial_id=record.trial_id,
            session_number=row.session_number,
            block_index=block.index,
            trial_index=ctx.trial_index,
            phase="idle" if idle else block.phase,
            task=record.task,
            regime=run.regime_at_end,
            idle=idle,
            interrupted=record.interrupted,
            complete=complete,
            ua=bool(labels.ua),
            correction=bool(labels.correction),
            veto=record.veto is not None,
            time_s=float(labels.time_ms) / 1000.0,
            limit_s=plan.limit_ms / 1000.0,
            corrections=record.corrections,
            idle_commits=int(labels.idle_commits),
            idle_minutes=float(labels.idle_minutes),
            acting_under_ambiguity=run.acting_under_ambiguity,
            activations=len(run.activations),
            interruptions=run.k_end - ctx.k_start,
        )
        return record, truth

    def simulate_trial(
        self,
        plan: PlannedTrial,
        *,
        session_id: str,
        block_index: int,
        trial_index: int,
        regime: str,
        regime_decision_id: str,
        table: Mapping[str, Any] | None,
        fallbacks: Fallbacks,
        spread: tuple[float, float],
        tau: float,
        modality: str,
        k_start: int,
        cue_at: datetime,
        rng: Generator,
        pin_act: bool = False,
        hand: bool = False,
        idle: bool = False,
        allow_veto: bool = False,
    ) -> tuple[TrialRecord, int]:
        """One simulated trial through the commit-rule port under the decisions and fallbacks a study UI holds, for
        hosts and tests that drive a real service with simulated input. Returns the record (``seq`` 0: the service
        sets ``recorded_at`` and ``seq``) and the block's interruption count after the trial (section 9.3)."""
        ctx = _TrialContext(
            regime=regime,
            l1_decision_id=regime_decision_id,
            table=table,
            table_decision_id=None if table is None else str(table["decision_id"]),
            block_mode="service",
            plain=fallbacks.plain,
            fallbacks_issued_at=fallbacks.issued_at,
            interruption_cap=fallbacks.interruption_cap,
            l2_fallback=tuple(fallbacks.l2),
            pin_act=pin_act,
            tau=tau,
            spread=spread,
            modality=modality,
            k_start=0 if idle else k_start,
            idle=idle,
            trial_index=trial_index,
            allow_veto=allow_veto,
        )
        run = self._simulate_trial(plan, ctx, rng)
        record = self._build_record(
            plan,
            ctx,
            run,
            (),
            session_id=session_id,
            block_index=block_index,
            cue_at=cue_at,
            seq=0,
            table_ids=() if table is None else (str(table["decision_id"]),),
            hand=hand,
        )
        if idle:
            self.person.state.session_minute += IDLE_EPOCH_MS / 60000.0
            self.person.state.abs_minute += IDLE_EPOCH_MS / 60000.0
        else:
            self.person.after_trial(plan, run.regime_at_end, run.elapsed_ms / 60000.0, rng)
        return record, run.k_end

    def _run_idle(
        self,
        policy: StudyPolicy,
        result: SimulatedRunResult,
        session: _SessionRun,
        block: Any,
        regime: str,
        decision_id: str,
        rng: Generator,
    ) -> None:
        row = session.row
        plan = generate_layout("idle", row.layout_seed, block.index, 900)
        ctx = _TrialContext(
            regime=regime,
            l1_decision_id=decision_id,
            table=None,
            table_decision_id=None,
            block_mode="idle",
            plain=session.fallbacks.plain,
            fallbacks_issued_at=session.fallbacks.issued_at,
            interruption_cap=session.fallbacks.interruption_cap,
            l2_fallback=tuple(session.fallbacks.l2),
            pin_act=PreferenceSet(self.person.params.stated_preferences).pin_act(
                context="study", modality=row.device.modality, participant_key=row.participant_key
            ),
            tau=float(row.settings["ambiguity_threshold"]),
            spread=session.spread,
            modality=row.device.modality,
            k_start=0,
            idle=True,
            trial_index=block.trials,
        )
        run = self._simulate_trial(plan, ctx, rng)
        record, truth = self._trial_record(session, block, plan, ctx, run, (), table_ids=())
        observation = policy.observe_trial(record)
        result.trials.append(observation.stored)
        result.events.extend(observation.events)
        result.truth.append(truth)
        self.person.state.session_minute += IDLE_EPOCH_MS / 60000.0
        self.person.state.abs_minute += IDLE_EPOCH_MS / 60000.0
        if observation.refresh_required:
            session.fallbacks = policy.fallbacks(row.device.modality, now=session.clock)

    # ------------------------------------------------------------------------------------------- counterfactuals

    def _t_ref(self, result: SimulatedRunResult) -> Mapping[str, float]:
        cached = self._t_ref_cache.get(id(result))
        if cached is None:
            cached = t_ref_by_task(result.rows(), result.participant_key)
            self._t_ref_cache[id(result)] = cached
        return cached

    def replay_block(
        self,
        start: BlockStart,
        regime: str,
        *,
        rng: Generator,
        l2_probabilities: Mapping[str, Mapping[str, float]] | None,
        offer_probabilities: Mapping[str, float] | None,
        weights: UtilityWeights,
        t_ref: Mapping[str, float],
    ) -> float:
        """The realized utility per trial of one replay of a block from its stored start state (no switching
        penalty; burden term from the person's ratings without skips)."""
        self.person.restore_state(start.person)
        self.person.set_regime(regime)
        k = 0
        utilities: list[float] = []
        times: list[tuple[PlannedTrial, float]] = []
        errors = 0
        vetoes = 0
        recent: list[bool] = []

        def chooser(cell: str, _draw: int, local: Generator) -> str:
            if l2_probabilities is None:
                return _first_applicable(start.l2_fallback, cell)
            probs = l2_probabilities.get(cell) or {}
            options = [option for option in L2_OPTIONS if float(probs.get(option, 0.0)) > 0]
            if not options:
                return _first_applicable(start.l2_fallback, cell)
            p = np.array([float(probs[option]) for option in options])
            return options[int(local.choice(len(options), p=p / p.sum()))]

        for index, plan in enumerate(start.trials):
            if offer_probabilities is not None and sum(recent) >= 2:
                options = [o for o in L3_OPTIONS if float(offer_probabilities.get(o, 0.0)) > 0]
                if options:
                    p = np.array([float(offer_probabilities[o]) for o in options])
                    chosen = options[int(rng.choice(len(options), p=p / p.sum()))]
                    if chosen != "no_offer":
                        self.person.responds("offer", rng=rng)
                recent = []
            ctx = _TrialContext(
                regime=regime,
                l1_decision_id=start.decision_id,
                table=None,
                table_decision_id=start.decision_id,
                block_mode="replay",
                plain=start.plain,
                fallbacks_issued_at=start.fallbacks_issued_at,
                interruption_cap=start.interruption_cap,
                l2_fallback=start.l2_fallback,
                pin_act=start.pin_act,
                tau=start.tau,
                spread=start.spread,
                modality=start.modality,
                k_start=k,
                idle=False,
                trial_index=index,
                l2_chooser=chooser,
            )
            run = self._simulate_trial(plan, ctx, rng)
            k = run.k_end
            if run.interrupted:
                self.person.after_trial(plan, run.regime_at_end, run.elapsed_ms / 60000.0, rng)
                continue
            labels = _truth_labels(run, plan)
            utilities.append(
                realized_utility(
                    labels,
                    time_s=labels["time_ms"] / 1000.0,
                    t_ref_s=t_ref[plan.task],
                    burden_term=0.0,
                    weights=weights,
                )
            )
            times.append((plan, labels["time_ms"] / 1000.0))
            errors += int(labels["ua"])
            vetoes += int(run.veto is not None)
            recent = (recent + [run.corrections > 0])[-3:]
            self.person.after_trial(plan, run.regime_at_end, run.elapsed_ms / 60000.0, rng)
        if not utilities:
            return math.nan
        summary = BlockSummary(
            regime=regime,
            trials=len(utilities),
            mean_time_z=self._time_z(times),
            interruptions_per_trial=k / len(utilities),
            errors_per_trial=errors / len(utilities),
            vetoes=vetoes,
        )
        ratings = self.person.burden(summary, rng, skips=False)
        burden = block_burden_term(ratings["effort"], ratings["intrusiveness"], regime=regime)
        return float(np.mean(utilities)) - weights.w_b * burden

    def counterfactual_block_value(
        self,
        result: SimulatedRunResult,
        block_decision_id: str,
        target: Callable[..., PolicyProbabilities],
        *,
        remaining_arms: Sequence[str],
        replays: int = 32,
        seed: int,
        weights: UtilityWeights | None = None,
    ) -> tuple[float, float]:
        """OPE ground truth (section 19): restores the person's state stored at that block's start (practice, day
        state, fatigue, the logged previous regime and switch age), replays the block ``replays`` times with the
        regime drawn from the target restricted to ``remaining_arms`` (``R_b``) and renormalised, the target's L2
        and L3 probabilities and common random numbers, and returns ``(V_b, pi_e(R_b | x_b))``: the mean realized
        utility per trial (no switching penalty) and the target mass on ``R_b``.

        ``target(block_decision_id, offer=None)`` returns the target's probabilities at the logged block. Replay
        ``r`` uses the stream ``(seed, r)`` for the person and a separate stream for the regime draw, so two targets
        evaluated with one seed see the same person randomness."""
        start = result.block_starts[block_decision_id]
        probs = target(block_decision_id)
        arms = [arm for arm in remaining_arms if float(probs.l1.get(arm, 0.0)) > 0]
        mass = float(sum(float(probs.l1.get(arm, 0.0)) for arm in remaining_arms))
        if mass <= 0 or not arms:
            return math.nan, 0.0
        restricted = np.array([float(probs.l1[arm]) for arm in arms]) / mass
        decision = next((d for d in result.decisions if d.decision_id == block_decision_id), None)
        weights = weights or weights_for(decision, "participant")
        t_ref = self._t_ref(result)
        saved = self.person.copy_state()
        values = []
        try:
            for r in range(replays):
                regime_rng = Generator(PCG64([seed, r, 1]))
                regime = arms[int(regime_rng.choice(len(arms), p=restricted))]
                value = self.replay_block(
                    start,
                    regime,
                    rng=Generator(PCG64([seed, r])),
                    l2_probabilities=probs.l2,
                    offer_probabilities=probs.l3,
                    weights=weights,
                    t_ref=t_ref,
                )
                if math.isfinite(value):
                    values.append(value)
        finally:
            self.person.restore_state(saved)
        return (float(np.mean(values)) if values else math.nan), mass

    def _oracle_for(self, policy: BaselinePolicy) -> Callable[[BlockContext, Sequence[str]], str]:
        """Policy 10: per block, the allowed regime whose true expected utility (8 common-random-number replays of
        the planned block from the person's current state, fixed L2 rule) is highest."""

        def choose(block: BlockContext, allowed: Sequence[str]) -> str:
            if self._pending is None:
                raise RuntimeError("the oracle runs only inside a simulated block")
            planned, session = self._pending
            fb = session.fallbacks
            start = BlockStart(
                decision_id="oracle",
                session_number=0,
                block_index=block.block_index,
                phase=block.phase,
                task=block.task,
                person=self.person.copy_state(),
                trials=planned,
                previous_regime=block.previous_regime,
                plain=fb.plain,
                fallbacks_issued_at=fb.issued_at,
                interruption_cap=fb.interruption_cap,
                l2_fallback=tuple(fb.l2),
                tau=float(session.row.settings["ambiguity_threshold"]),
                spread=session.spread,
                modality=block.device.modality,
                pin_act=False,
                clock=session.clock,
                regime="",
            )
            t_ref = {"pointing": 1.0, "menu": 2.0}
            saved = self.person.copy_state()
            best, best_value = allowed[0], -math.inf
            try:
                for arm in allowed:
                    values = [
                        self.replay_block(
                            start,
                            arm,
                            rng=Generator(PCG64([self.seed, block.block_index, r, 99])),
                            l2_probabilities=None,
                            offer_probabilities=None,
                            weights=weights_for(None, "protocol"),
                            t_ref=t_ref,
                        )
                        for r in range(8)
                    ]
                    value = float(np.nanmean(values))
                    if value > best_value:
                        best, best_value = arm, value
            finally:
                self.person.restore_state(saved)
            return best

        return choose


@dataclass
class _SessionRun:
    row: SessionRow
    clock: datetime
    seq: int
    fallbacks: Fallbacks
    spread: tuple[float, float]
    corrections_recent: list[bool]
    trials_today: int = 0

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq


def _first_applicable(l2: Sequence[str], cell: str) -> str:
    parts = cell.split(":")
    n_plausible, has_hit = int(parts[1][1:]), parts[2] == "h1"
    for variant in l2:
        if l2_applicable(variant, n_plausible=n_plausible, has_hit=has_hit):
            return variant
    return "wait_for_evidence"


def _nearest_neighbour(candidates: Sequence[Rect], index: int) -> int:
    target = candidates[index]
    others = [i for i in range(len(candidates)) if i != index]
    return min(
        others,
        key=lambda i: (math.hypot(candidates[i].cx - target.cx, candidates[i].cy - target.cy), i),
    )


def _truth_labels(run: _TrialRun, plan: PlannedTrial) -> dict[str, Any]:
    """Trial labels of section 7.1 straight from a simulated trial (counterfactual replays build no record)."""
    intended = plan.intended
    in_time = run.commit_t_ms is not None and run.commit_t_ms <= plan.limit_ms
    complete = run.committed_index == intended and in_time
    ua = any(a.outcome == "committed" and a.committed_index != intended for a in run.activations)
    cancelled = any(a.outcome == "cancelled" for a in run.activations)
    return {
        "complete": complete,
        "ua": ua,
        "correction": run.corrections > 0 or cancelled,
        "veto": run.veto is not None,
        "time_ms": max(float(run.commit_t_ms), 150.0) if complete else plan.limit_ms,
        "limit_ms": plan.limit_ms,
    }


def _sim_consent(participant_key: str) -> ConsentPeriod:
    granted = SIM_EPOCH - timedelta(days=1)
    return ConsentPeriod(
        record=ConsentRecord(
            # Deterministic: runs repeat by seed.
            consent_id=uuid.UUID(_uuid(0, participant_key, "consent")),
            participant_key=participant_key,
            purposes=frozenset(CoAdaptationPurpose),
            consent_text_version="coadapt-consent-v1",
            consent_text_sha256=SIM_CONSENT_TEXT_SHA256,
            protocol_id=STUDY_PROTOCOL_ID,
            protocol_sha256=study_protocol_sha256(),
            granted_at=granted,
            expires_at=granted + timedelta(days=365),
            retention_days=365,
            authorized=True,
        ),
        replaced_at=None,
    )
