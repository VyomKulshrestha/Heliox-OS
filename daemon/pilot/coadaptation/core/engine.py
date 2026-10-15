"""The co-adaptation engine: one participant's world model, policy and session state (section 13.1.1 of the spec).

``CoAdaptationEngine`` is pure and in memory: no I/O, no clock reads (time is passed in) and no knowledge of consent
or storage. Every returned decision is presentation-only. The live path (``observe_trial``) verifies a recorded
trial, derives its labels, builds the stored row and applies it through the same routine a rebuild uses
(``replay_trial``), so replaying the stored rows reproduces the live snapshot bytes exactly. Drift and calibration
alarms fire only live; a rebuild re-applies them from their stored events (section 10.8).
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from statistics import median
from typing import Any, Literal, Self

import numpy as np

from .constraints import (
    ConstraintState,
    ExclusionReason,
    Fallbacks,
    apply_hard_constraints,
    control_floor_excludes,
    fixed_l2_rule,
    is_l1_step,
    preference_violations,
    trial_interruptions,
)
from .constraints import (
    fallbacks as compute_fallbacks,
)
from .evidence import EvidenceKind
from .outcome_model import (
    MODEL_VERSION,
    CoAdaptationWorldModel,
    EpisodeOutcome,
    InteractionPrediction,
    ModelVariant,
    PopulationPrior,
    TrialContext,
    TrialOutcome,
    read_npz,
    write_npz,
)
from .policy import (
    POLICY_VERSION,
    AmbiguityTable,
    BlockRequest,
    ConstrainedThompsonPolicy,
    Decision,
    NoAllowedRegime,
    OfferRequest,
    PolicyMode,
    TableRequest,
    cell_specs,
    decision_seed,
    select_from_cell,
)
from .preferences import Preference, PreferenceSet
from .priors import PRIOR_VERSION
from .records import (
    BurdenReport,
    SessionRow,
    StoredTrial,
    TrialLabels,
    TrialRecord,
    derive_episodes,
    derive_labels,
    vetoed_variant,
)
from .state import FEATURE_SCHEMA, ActivationContext, BlockContext, OfferContext
from .target_posterior import GazeInput, ambiguity_bin, cell_id, compute_posterior, keyboard_endpoint, rendered_rects
from .vocabulary import L2_OPTIONS, L3_OPTIONS, STRATEGY_VERSION, VARIANTS, DecisionLevel, l1_arms

ENGINE_SNAPSHOT_SCHEMA = "coadapt-engine-snapshot-v1"
_A_GATE_TOLERANCE = 1e-6
_ACT_GUARDED = frozenset({"act", "enlarge_target:1.5", "enlarge_target:2.0"})
_C7_REGIMES = frozenset({"act", "enlarge_target:1.5", "enlarge_target:2.0"})

__all__ = [
    "CoAdaptationEngine",
    "EngineEvent",
    "EngineSettings",
    "EngineStatus",
    "NoAllowedRegime",
    "PolicyProbabilities",
    "SessionSummary",
    "TrialObservation",
]


# ------------------------------------------------------------------------------------------------------- settings


_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "ambiguity_threshold": (0.05, 0.25),
    "unintended_activation_tolerance": (0.0, 0.2),
    "noninferiority_margin": (0.0, 0.05),
    "upper_bound_credibility": (0.8, 0.99),
    "exploration_floor": (0.0, 0.3),
    "switching_penalty": (0.0, 1.0),
}
_BOOL_FIELDS = ("adaptive_allowed", "learn_across_sessions", "gaze_allowed", "hand_pointer_allowed")


@dataclass(frozen=True, slots=True)
class EngineSettings:
    ambiguity_threshold: float = 0.25
    unintended_activation_tolerance: float = 0.05
    noninferiority_margin: float = 0.01
    upper_bound_credibility: float = 0.95
    exploration_floor: float = 0.1
    switching_penalty: float = 0.1
    adaptive_allowed: bool = False
    learn_across_sessions: bool = False
    gaze_allowed: bool = False
    hand_pointer_allowed: bool = False
    thompson_draws: int = 256
    day_boundary_hour: int = 4

    def __post_init__(self) -> None:
        for name, (lo, hi) in _FLOAT_RANGES.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a number")
            if not lo <= float(value) <= hi:
                raise ValueError(f"{name} must lie in [{lo}, {hi}]")
            object.__setattr__(self, name, float(value))
        for name in _BOOL_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        draws = self.thompson_draws
        if isinstance(draws, bool) or not isinstance(draws, int) or not 1 <= draws <= 100_000:
            raise ValueError("thompson_draws must be an integer in [1, 100000]")
        hour = self.day_boundary_hour
        if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23:
            raise ValueError("day_boundary_hour must be an integer in [0, 23]")

    def as_dict(self) -> dict[str, float | int | bool]:
        """Every field, as ``SessionRow.settings`` stores them."""
        return asdict(self)


# ---------------------------------------------------------------------------------------------- returned records


def _event_id(trial_id: str, kind: str, head: str | None, outcome: str | None) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{trial_id}|{kind}|{head}|{outcome}"))


@dataclass(frozen=True, slots=True)
class EngineEvent:
    """A drift or calibration-suspension alarm, stored with the trial whose observation fired it."""

    event_id: str  # uuid5(NAMESPACE_URL, f"{trial_id}|{kind}|{head}|{outcome}")
    kind: Literal["drift", "calibration_suspended"]
    session_id: str
    trial_id: str
    head: str | None  # drift only
    outcome: str | None  # drift only
    created_at: datetime  # = the firing trial's recorded_at

    def __post_init__(self) -> None:
        if self.kind not in ("drift", "calibration_suspended"):
            raise ValueError("event kind must be drift or calibration_suspended")
        if (self.kind == "drift") != (self.head is not None and self.outcome is not None):
            raise ValueError("drift events name a head and an outcome; suspension events name neither")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must carry a UTC offset")

    @classmethod
    def create(
        cls,
        *,
        kind: Literal["drift", "calibration_suspended"],
        session_id: str,
        trial_id: str,
        created_at: datetime,
        head: str | None = None,
        outcome: str | None = None,
    ) -> EngineEvent:
        return cls(
            event_id=_event_id(trial_id, kind, head, outcome),
            kind=kind,
            session_id=session_id,
            trial_id=trial_id,
            head=head,
            outcome=outcome,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "head": self.head,
            "outcome": self.outcome,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        keys = ("event_id", "kind", "session_id", "trial_id", "head", "outcome", "created_at")
        if not isinstance(d, Mapping):
            raise ValueError("EngineEvent: expected an object")
        unknown = sorted(str(key) for key in d if key not in keys)
        if unknown:
            raise ValueError(f"EngineEvent: unknown keys {unknown}")
        missing = sorted(key for key in keys if key not in d)
        if missing:
            raise ValueError(f"EngineEvent: missing keys {missing}")
        for name in ("event_id", "kind", "session_id", "trial_id", "created_at"):
            if not isinstance(d[name], str):
                raise ValueError(f"EngineEvent.{name}: expected str")
        for name in ("head", "outcome"):
            if d[name] is not None and not isinstance(d[name], str):
                raise ValueError(f"EngineEvent.{name}: expected str or null")
        try:
            created = datetime.fromisoformat(d["created_at"])
        except ValueError:
            raise ValueError("EngineEvent.created_at: expected an ISO-8601 string") from None
        return cls(
            event_id=d["event_id"],
            kind=d["kind"],
            session_id=d["session_id"],
            trial_id=d["trial_id"],
            head=d["head"],
            outcome=d["outcome"],
            created_at=created,
        )


@dataclass(frozen=True, slots=True)
class TrialObservation:
    stored: StoredTrial  # the trials row: trial, verification, labels, replay inputs (13.1.2)
    learned: bool  # stored.verification == "ok"
    refresh_required: bool  # an alarm of this trial marked the session stale (mark_stale)
    events: tuple[EngineEvent, ...]  # alarms fired by this trial; put_trial writes them with the row


@dataclass(frozen=True, slots=True)
class SessionSummary:
    trials: int
    completion: float
    ua_rate: float
    idle_ua_per_min: float | None
    median_time_ms: float | None
    burden_mean: dict[str, float | None]
    overrides: int
    regime_entropy_bits: float
    switch_rate: float
    allowed_now: tuple[str, ...]
    excluded_now: dict[str, str]
    predictions: dict[str, InteractionPrediction]


@dataclass(frozen=True, slots=True)
class EngineStatus:
    trials_seen: int
    sessions_seen: int
    last_day_index: int | None
    drift_flags: tuple[str, ...]
    calibration_suspended: bool
    learn_across_sessions_effective: bool
    model_version: str
    prior_version: str
    policy_version: str
    feature_schema: str


@dataclass(frozen=True, slots=True)
class PolicyProbabilities:  # evaluation only (section 19)
    l1: dict[str, float]  # over l1_arms(modality), zeros included
    l2: dict[str, dict[str, float]]  # cell_id -> over L2_OPTIONS, zeros included (act always 0)
    l3: dict[str, float] | None  # over L3_OPTIONS when an offer context was given


# ------------------------------------------------------------------------------------------------ session state


@dataclass
class _BlockStats:
    """Per-block bookkeeping shared by live observation and replay (both go through ``replay_trial``)."""

    interruptions: int = 0  # the cap count k of section 9.3 over every recorded active trial of the block
    active_trials: int = 0  # recorded active trials that were not interrupted
    regime: str | None = None  # the applied regime of the block's first active trial


@dataclass
class _Session:
    row: SessionRow
    constraint: ConstraintState
    decisions: dict[str, Decision | AmbiguityTable] = field(default_factory=dict)
    created: dict[str, datetime] = field(default_factory=dict)
    open: set[str] = field(default_factory=set)
    stale: set[str] = field(default_factory=set)
    refresh_required: bool = False
    round_remaining: frozenset[str] | None = None
    round_index: int = 0
    last_l1: dict[str, str] = field(default_factory=dict)  # modality -> regime of the latest L1 decision
    block_decisions: dict[int, str] = field(default_factory=dict)  # block index -> latest L1 decision id
    blocks: dict[int, _BlockStats] = field(default_factory=dict)
    previous_issued_at: datetime | None = None
    suspended_since: datetime | None = None
    last_block: BlockContext | None = None
    trials: list[StoredTrial] = field(default_factory=list)
    burden: list[BurdenReport] = field(default_factory=list)


def _scale(regime: str) -> float:
    value = VARIANTS[regime].param("scale")
    return 1.0 if value is None else float(value)


def _entropy_bits(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = len(values)
    return float(-sum(c / total * math.log2(c / total) for c in counts.values()))


# --------------------------------------------------------------------------------------------------------- engine


class CoAdaptationEngine:
    """Pure, in-memory co-adaptation engine for one participant. No I/O, no clock reads (time is passed in),
    no knowledge of consent or storage. Every returned decision is presentation-only."""

    def __init__(
        self,
        *,
        participant_key: str,
        settings: EngineSettings,
        preferences: PreferenceSet,
        evidence_kind: EvidenceKind = EvidenceKind.LIVE_HUMAN,
        model_variant: ModelVariant = ModelVariant.FULL,
        population_prior: PopulationPrior | None = None,
        policy: ConstrainedThompsonPolicy | None = None,
    ) -> None:
        if not isinstance(settings, EngineSettings):
            raise ValueError("settings must be EngineSettings")
        if not isinstance(preferences, PreferenceSet):
            raise ValueError("preferences must be a PreferenceSet")
        self.participant_key = participant_key
        self._settings = settings
        self._preferences = preferences
        self.evidence_kind = EvidenceKind(evidence_kind)
        self.model_variant = ModelVariant(model_variant)
        self.population_prior = population_prior
        self.policy = policy if policy is not None else ConstrainedThompsonPolicy()
        self.model = CoAdaptationWorldModel(variant=self.model_variant, population_prior=population_prior)
        self._session: _Session | None = None
        self._sessions_seen = 0
        self._trials_seen = 0
        self._off_times_s: list[float] = []  # completed assistance_off times (T_ref; cancels between regimes)

    # ------------------------------------------------------------------------------------------- properties

    @property
    def settings(self) -> EngineSettings:
        return self._settings

    @property
    def preferences(self) -> PreferenceSet:
        return self._preferences

    @property
    def session_open(self) -> bool:
        return self._session is not None

    def _versions(self) -> dict[str, str]:
        return {
            "strategy": STRATEGY_VERSION,
            "feature_schema": FEATURE_SCHEMA,
            "model": MODEL_VERSION,
            "prior": PRIOR_VERSION,
            "policy": POLICY_VERSION,
        }

    def _require_session(self) -> _Session:
        if self._session is None:
            raise ValueError("no open session")
        return self._session

    # ------------------------------------------------------------------------------------ preferences, staleness

    def _own_rules(self, preferences: PreferenceSet) -> tuple[Preference, ...]:
        """The rules that apply to this participant (device-wide or scoped to them), retracted ones included."""
        return tuple(pref for pref in preferences.all if pref.scope.participant_key in (None, self.participant_key))

    def set_preferences(self, preferences: PreferenceSet) -> None:
        """Takes the person's new rule set; a change to a rule that applies to this participant marks the open session
        stale (mark_stale). A rule scoped to another participant changes nothing here."""
        if not isinstance(preferences, PreferenceSet):
            raise ValueError("preferences must be a PreferenceSet")
        changed = self._own_rules(preferences) != self._own_rules(self._preferences)
        self._preferences = preferences
        if changed:
            self.mark_stale("preferences")

    def mark_stale(self, reason: Literal["preferences", "calibration_suspended", "drift"]) -> bool:
        """Marks every open decision of the current session stale and asks for a refresh; True whenever a session is
        open, also before its first decision or after a failed one. A practice block, or a block whose L1 request
        failed, runs on the fallbacks issued at its cue and holds no open decision, yet the change makes those
        fallbacks stale too (preference changes take effect at once, section 9.7). observe_trial calls it on a C7 or
        drift event. A stale decision still verifies trials recorded against it."""
        if reason not in ("preferences", "calibration_suspended", "drift"):
            raise ValueError("unknown stale reason")
        session = self._session
        if session is None:
            return False
        session.stale |= session.open
        session.open.clear()
        session.refresh_required = True
        return True

    def refresh_required(self) -> bool:
        """True from ``mark_stale`` until the next ``decide_block``."""
        return self._session is not None and self._session.refresh_required

    def _c7_at(self, when: datetime) -> bool:
        """The C7 state in force at ``when`` within the open session (a suspension clears only at a day boundary or
        a learning reset, which happen at ``begin_session``)."""
        session = self._session
        if session is None:
            return bool(self.model.calibration_suspended)
        return session.suspended_since is not None and when >= session.suspended_since

    def fallbacks(self, modality: str, *, now: datetime) -> Fallbacks:
        """Section 9.2 step 4, from the preferences and the C7 state at ``now`` only; ``issued_at = now``."""
        session = self._session
        current = session.last_l1.get(modality) if session is not None else None
        return compute_fallbacks(
            self._preferences,
            modality,  # type: ignore[arg-type]
            participant_key=self.participant_key,
            calibration_suspended=bool(self.model.calibration_suspended),
            now=now,
            current_regime=current,
        )

    def _fallbacks_at(self, modality: str, when: datetime) -> Fallbacks:
        """The fallbacks issued at an earlier binding time: the rules and the C7 state in force then."""
        return compute_fallbacks(
            self._preferences.as_of(when),
            modality,  # type: ignore[arg-type]
            participant_key=self.participant_key,
            calibration_suspended=self._c7_at(when),
            now=when,
        )

    # --------------------------------------------------------------------------------------------- sessions

    def begin_session(self, row: SessionRow) -> None:
        """Called by the service after it stored the row, and by replay. Sets the engine's settings to
        EngineSettings(**row.settings) for the whole session (τ, ε_user, δ_ni, credibility, exploration floor,
        switching penalty, the allowed flags, Thompson draws, day boundary hour): decisions, tables, bins,
        verification, physics priors (7.3, 7.5) and the drift and C7 tests of this session use them, live and in
        replay, whatever the config says later. Applies day-boundary and device-change dynamics from row.day_index
        and row.device (never a clock or a time zone, section 8). When row.settings["learn_across_sessions"] is False
        (the stored value, so replay follows what happened live even if the setting changed later) it resets the
        learned state and the practice counters and n_item_eff to the prior; otherwise, under
        ModelVariant.SESSION_ONLY, it resets the filters, spread, idle rates and drift statistics only (13.1.1
        table). Takes row.seed_salt for decision seeds (9.2 step 6), row.gaze_check_accuracy for posteriors and
        row.comparator_ranking for comparator blocks (9.2 step 5), starts a new permuted round and starts
        session-scoped state empty."""
        if self._session is not None:
            raise ValueError("a session is already open")
        settings = EngineSettings(**row.settings)
        self._settings = settings
        if not settings.learn_across_sessions:
            self._off_times_s = []
        self.model.begin_session(
            day_index=row.day_index, device=row.device, learn_across_sessions=settings.learn_across_sessions
        )
        suspended = bool(self.model.calibration_suspended)
        self._session = _Session(
            row=row,
            constraint=ConstraintState(calibration_suspended=suspended),
            suspended_since=row.started_at if suspended else None,
        )
        self._sessions_seen += 1

    def end_session(self, *, ended_at: datetime, reason: str) -> SessionSummary:
        session = self._require_session()
        summary = self._summary(session, ended_at)
        self.model.end_session()
        self._session = None
        return summary

    def _summary(self, session: _Session, ended_at: datetime) -> SessionSummary:
        active = [s for s in session.trials if s.trial.task != "idle" and not s.trial.interrupted and s.labels]
        idle = [s for s in session.trials if s.trial.task == "idle" and not s.trial.interrupted and s.labels]
        completion = float(np.mean([s.labels.complete for s in active])) if active else 0.0  # type: ignore[union-attr]
        ua_rate = float(np.mean([s.labels.ua for s in active])) if active else 0.0  # type: ignore[union-attr]
        idle_minutes = sum(s.labels.idle_minutes for s in idle)  # type: ignore[union-attr]
        idle_rate = sum(s.labels.idle_commits for s in idle) / idle_minutes if idle_minutes > 0 else None  # type: ignore[union-attr]
        times = [s.labels.time_ms for s in active if s.labels.complete]  # type: ignore[union-attr]
        burden_mean: dict[str, float | None] = {}
        for item in ("effort", "intrusiveness", "control"):
            values = [getattr(r, item) for r in session.burden if r.scope == "block" and getattr(r, item) is not None]
            burden_mean[item] = float(np.mean(values)) if values else None
        regimes = [s.trial.applied_regime for s in session.trials if s.trial.task != "idle"]
        block_regimes = [stats.regime for _, stats in sorted(session.blocks.items()) if stats.regime is not None]
        switches = sum(1 for a, b in zip(block_regimes, block_regimes[1:]) if a != b)
        modality = session.row.device.modality
        candidates = [arm for arm in l1_arms(modality) if arm != "assistance_off"]
        allowed, excluded = apply_hard_constraints(
            DecisionLevel.BLOCK_REGIME,
            candidates,
            preferences=self._preferences,
            state=session.constraint,
            modality=modality,  # type: ignore[arg-type]
            participant_key=self.participant_key,
            adaptive=True,
        )
        predictions: dict[str, InteractionPrediction] = {}
        if session.last_block is not None:
            predictions = self.predictions_for_block(session.last_block, now=ended_at)
        return SessionSummary(
            trials=len([s for s in session.trials if s.trial.task != "idle"]),
            completion=completion,
            ua_rate=ua_rate,
            idle_ua_per_min=idle_rate,
            median_time_ms=float(median(times)) if times else None,
            burden_mean=burden_mean,
            overrides=sum(1 for s in session.trials if s.trial.veto is not None),
            regime_entropy_bits=_entropy_bits(regimes),
            switch_rate=switches / (len(block_regimes) - 1) if len(block_regimes) > 1 else 0.0,
            allowed_now=tuple(allowed),
            excluded_now={variant: ExclusionReason(reason).value for variant, reason in excluded.items()},
            predictions=predictions,
        )

    # -------------------------------------------------------------------------------------------- decisions

    def _block_mode(self, phase: str) -> PolicyMode:
        if phase == "practice":
            return PolicyMode.BASELINE
        if phase in ("A", "A_lite"):
            return PolicyMode.RANDOMIZED
        if phase == "B":
            return PolicyMode.ADAPTIVE if self._settings.adaptive_allowed else PolicyMode.RANDOMIZED
        if phase == "probe":
            return PolicyMode.PROBE
        if phase == "comparator":
            return PolicyMode.COMPARATOR
        raise ValueError("unknown phase")

    def _t_ref_s(self, predictions: Mapping[str, InteractionPrediction]) -> float:
        """T_ref (section 9.1): the median completed assistance_off time seen, else the predicted median for
        assistance_off on the block's layouts. It is common to every regime of a decision, so it never changes
        which regime is chosen."""
        if self._off_times_s:
            return float(median(self._off_times_s))
        baseline = predictions.get("assistance_off")
        if baseline is not None and baseline.time_ms.mean > 0:
            return float(baseline.time_ms.mean) / 1000.0
        return 1.0

    def _spread_pair(self, modality: str) -> tuple[float, float] | None:
        if modality == "keyboard":
            return None
        r = self.model.spread_r(modality)
        floor = self.model.floor(modality)
        if r is None or floor is None:
            return None
        return (float(r), float(floor))

    def _register(self, session: _Session, decision: Decision | AmbiguityTable, now: datetime) -> None:
        if decision.decision_id in session.decisions:
            raise ValueError("Duplicate decision_id")
        session.decisions[decision.decision_id] = decision
        session.created[decision.decision_id] = now
        session.open.add(decision.decision_id)

    def decide_block(self, *, decision_id: str, block: BlockContext, now: datetime) -> Decision:
        """L1. Mode from the plan phase and settings: practice→BASELINE, A/A_lite→RANDOMIZED,
        B→ADAPTIVE if adaptive_allowed else RANDOMIZED, probe→PROBE, comparator→COMPARATOR; freeze/pin
        preferences→PINNED. The decision carries spread = {r, floor} for the session's modality (None for keyboard,
        section 6.1) and the priorities (m, f, i) active at `now` (section 9.1).
        Raises NoAllowedRegime when fallbacks(modality).l1 is None and nothing survives (section 9.2 step 4).
        decide_ambiguity_table and decide_offer inherit the block's PROBE or PINNED mode (fixed L2 rule, no offers)
        and its COMPARATOR mode (uniform L2 and L3 as in Phase A).
        Every decide_* reads the model only through non-mutating predictions (copies diffused up to `now`, 7.6)
        and changes only session-scoped state (round, decision cache, stale marks)."""
        session = self._require_session()
        if decision_id in session.decisions:
            raise ValueError("Duplicate decision_id")
        modality = session.row.device.modality
        settings = self._settings
        planned = self._block_mode(block.phase)
        held = self._preferences.held_regime(context="study", modality=modality, participant_key=self.participant_key)
        mode = PolicyMode.PINNED if held is not None else planned
        deviation = held is not None and planned in (PolicyMode.RANDOMIZED, PolicyMode.PROBE, PolicyMode.COMPARATOR)
        predictions = self.model.predict_l1_block(
            block,
            l1_arms(modality),
            now=now,
            tau=settings.ambiguity_threshold,
            credibility=settings.upper_bound_credibility,
        )

        def sampler(rng: np.random.Generator, variants: Sequence[str], draws: int) -> Any:
            return self.model.sample_l1(block, variants, rng, draws, now=now, tau=settings.ambiguity_threshold)

        request = BlockRequest(
            decision_id=decision_id,
            participant_key=self.participant_key,
            seed=decision_seed(self.participant_key, session.row.seed_salt, decision_id),
            mode=mode,
            modality=modality,
            preferences=self._preferences,
            state=session.constraint,
            fallbacks=self.fallbacks(modality, now=now),
            settings=settings,
            predictions=predictions,
            sampler=sampler,
            previous_regime=block.previous_regime,
            priorities=self._preferences.priorities(
                context="study", modality=modality, participant_key=self.participant_key
            ),
            t_ref_s=self._t_ref_s(predictions),
            spread=self._spread_pair(modality),
            round_remaining=session.round_remaining,
            round_index=session.round_index,
            comparator_ranking=tuple(session.row.comparator_ranking),
            protocol_deviation=deviation,
            versions=self._versions(),
        )
        decision = self.policy.decide_block(request)
        session.open.clear()
        session.refresh_required = False
        self._register(session, decision, now)
        session.constraint.consume_vetoes(DecisionLevel.BLOCK_REGIME)
        if decision.mode is PolicyMode.RANDOMIZED and not decision.fallback:
            support = {variant for variant, p in decision.propensities.items() if p > 0.0}
            session.round_remaining = frozenset(support - {decision.variant})
            session.round_index = decision.round_index if decision.round_index is not None else session.round_index
        session.last_l1[modality] = decision.variant
        session.block_decisions[block.block_index] = decision.decision_id
        session.last_block = block
        return decision

    def _block_decision_mode(self, session: _Session, block_decision_id: str | None, phase: str) -> PolicyMode:
        decision = session.decisions.get(block_decision_id) if block_decision_id is not None else None
        if isinstance(decision, Decision):
            return decision.mode
        return self._block_mode(phase)

    def decide_ambiguity_table(
        self,
        *,
        decision_id: str,
        block_decision_id: str,
        block: BlockContext,
        start_trial_index: int,
        trials_in_block: int,
        now: datetime,
    ) -> AmbiguityTable:
        session = self._require_session()
        if decision_id in session.decisions:
            raise ValueError("Duplicate decision_id")
        if not block.layouts:
            raise ValueError("a table needs the block's planned layouts")
        if start_trial_index < 0 or trials_in_block < 0:
            raise ValueError("start_trial_index and trials_in_block must be non-negative")
        modality = session.row.device.modality
        settings = self._settings
        mode = self._block_decision_mode(session, block_decision_id, block.phase)

        def predict_cell(context: ActivationContext, variants: Sequence[str]) -> Mapping[str, InteractionPrediction]:
            return self.model.predict_l2(context, variants, now=now, credibility=settings.upper_bound_credibility)

        def sample_cell(
            rng: np.random.Generator, context: ActivationContext, variants: Sequence[str], draws: int
        ) -> Any:
            return self.model.sample_l2(context, variants, rng, draws, now=now)

        request = TableRequest(
            decision_id=decision_id,
            participant_key=self.participant_key,
            block_decision_id=block_decision_id,
            seed=decision_seed(self.participant_key, session.row.seed_salt, decision_id),
            mode=mode,
            modality=modality,
            preferences=self._preferences,
            state=session.constraint,
            fallbacks=self.fallbacks(modality, now=now),
            settings=settings,
            cells=cell_specs(settings.ambiguity_threshold, block.layouts[0], modality),
            predict_cell=predict_cell,
            sample_cell=sample_cell,
            priorities=self._preferences.priorities(
                context="study", modality=modality, participant_key=self.participant_key
            ),
            interruptions_used=self._block(session, block.block_index).interruptions,
            start_trial_index=start_trial_index,
            trials_in_block=trials_in_block,
            versions=self._versions(),
        )
        table = self.policy.decide_table(request)
        self._register(session, table, now)
        session.constraint.consume_vetoes(DecisionLevel.AMBIGUITY_RESPONSE)
        return table

    def decide_offer(self, *, decision_id: str, context: OfferContext, now: datetime) -> Decision:
        session = self._require_session()
        if decision_id in session.decisions:
            raise ValueError("Duplicate decision_id")
        phase = self._phase_of(session, context.block_index)
        mode = self._block_decision_mode(session, session.block_decisions.get(context.block_index), phase)

        def sampler(rng: np.random.Generator, variants: Sequence[str], draws: int) -> Any:
            return self.model.sample_offers(context.modality, context.task, variants, rng, draws)

        request = OfferRequest(
            decision_id=decision_id,
            participant_key=self.participant_key,
            seed=decision_seed(self.participant_key, session.row.seed_salt, decision_id),
            mode=mode,
            modality=context.modality,
            keyboard_available=context.keyboard_available,
            preferences=self._preferences,
            state=session.constraint,
            settings=self._settings,
            sampler=sampler,
            priorities=self._preferences.priorities(
                context="study", modality=context.modality, participant_key=self.participant_key
            ),
            versions=self._versions(),
        )
        decision = self.policy.decide_offer(request)
        self._register(session, decision, now)
        return decision

    def _phase_of(self, session: _Session, block_index: int) -> str:
        for block in session.row.plan.blocks:
            if block.index == block_index:
                return block.phase
        return "A"

    def interruptions_used(self, block_index: int) -> int:
        """The open session's cap count k of a block (section 9.3), over every recorded active trial of the block
        whatever its verification; idle epochs count from 0 on their own and are not included."""
        session = self._session
        if session is None or block_index not in session.blocks:
            return 0
        return session.blocks[block_index].interruptions

    def decision(self, decision_id: str) -> Decision | AmbiguityTable | None:
        """This session's decisions."""
        session = self._session
        return None if session is None else session.decisions.get(decision_id)

    # ---------------------------------------------------------------------------------------------- verification

    def _block(self, session: _Session, block_index: int) -> _BlockStats:
        return session.blocks.setdefault(block_index, _BlockStats())

    def _verify(self, session: _Session, trial: TrialRecord) -> tuple[str, dict[int, Any]]:
        """Verification of section 13.1.1 step 1; also returns the recomputed posteriors by activation index."""
        posteriors = self._recompute_posteriors(session, trial)
        if trial.interrupted:
            return "interrupted", posteriors
        if not self._draws_ok(session, trial):
            return "draw_mismatch", posteriors
        if not self._posteriors_ok(session, trial, posteriors):
            return "posterior_mismatch", posteriors
        if trial.l1_fallback_reason is not None:
            return "fallback", posteriors
        return "ok", posteriors

    def _governing(self, trial: TrialRecord, index: int) -> tuple[str, datetime, bool]:
        """(regime, binding time, after revert) for one activation."""
        revert = trial.revert
        if revert is not None and index >= revert.from_activation:
            return revert.regime, revert.fallbacks_issued_at, True
        return trial.applied_regime, trial.fallbacks_issued_at, False

    def _draws_ok(self, session: _Session, trial: TrialRecord) -> bool:
        row = session.row
        modality = trial.modality
        tau = self._settings.ambiguity_threshold
        issued = trial.fallbacks_issued_at
        if issued < row.started_at or issued > trial.recorded_at:
            return False
        if session.previous_issued_at is not None and issued < session.previous_issued_at:
            return False
        if trial.applied_regime not in l1_arms(modality):
            return False
        cue_fallbacks = self._fallbacks_at(modality, issued)
        if trial.l1_fallback_reason is None:
            decision = session.decisions.get(trial.regime_decision_id)
            if not isinstance(decision, Decision) or decision.level != DecisionLevel.BLOCK_REGIME:
                return False
            if trial.applied_regime != decision.variant or trial.regime != decision.variant:
                return False
        elif trial.applied_regime != cue_fallbacks.l1:
            return False

        revert = trial.revert
        revert_fallbacks: Fallbacks | None = None
        if revert is not None:
            if revert.fallbacks_issued_at < row.started_at or revert.fallbacks_issued_at > trial.recorded_at:
                return False
            revert_fallbacks = self._fallbacks_at(modality, revert.fallbacks_issued_at)
            expected = revert_fallbacks.plain if revert.reason == "veto" else revert_fallbacks.refresh
            if expected is None or revert.regime != expected:
                return False

        idle = trial.task == "idle"
        k = 0 if idle else self._block(session, trial.block_index).interruptions
        l1_counted = False
        table_draws = 0
        for activation in trial.activations:
            regime, _, after_revert = self._governing(trial, activation.index)
            binding = revert_fallbacks if after_revert and revert_fallbacks is not None else cue_fallbacks
            has_hit = activation.hit_index is not None
            ambiguous = activation.a_gate > tau or not has_hit
            routed_to_l2 = regime in _ACT_GUARDED and ambiguous
            applied = activation.l2
            if routed_to_l2 != (applied is not None):
                return False
            if applied is not None:
                bin_id = ambiguity_bin(activation.a_gate, tau) or "b3"
                if applied.cell_id != cell_id(bin_id, activation.n_plausible, has_hit):
                    return False
                if applied.draw_index != min(table_draws, 4):
                    return False
                pin_act = False
                if applied.source == "table":
                    if idle or after_revert:
                        return False
                    table = session.decisions.get(applied.decision_id)
                    if not isinstance(table, AmbiguityTable):
                        return False
                    if trial.l1_fallback_reason is None and table.block_decision_id != trial.regime_decision_id:
                        return False
                    cell = table.cell(applied.cell_id)
                    if cell is None:
                        return False
                    pin_act = cell.pinned
                    if not pin_act and binding.interruption_cap is not None and k >= binding.interruption_cap:
                        return False
                    expected = table.variant_for(applied.cell_id, trial.trial_index, applied.draw_index)
                    if expected is None or expected != applied.variant:
                        return False
                    if applied.draw_index < 4:
                        table_draws += 1
                elif applied.source in ("fixed_rule", "fallback"):
                    pin_act = binding.l2 == ("act", "wait_for_evidence")
                    if not pin_act and binding.interruption_cap is not None and k >= binding.interruption_cap:
                        return False
                    if applied.variant != fixed_l2_rule(binding, applied.cell_id):
                        return False
                else:  # cap
                    if binding.interruption_cap is None or k < binding.interruption_cap:
                        return False
            if is_l1_step(activation):
                if not l1_counted:
                    k += 1
                    l1_counted = True
            elif activation.result in ("confirming", "held", "asked"):
                k += 1
        return True

    def _recompute_posteriors(self, session: _Session, trial: TrialRecord) -> dict[int, Any]:
        """The daemon's posterior for every activation it can recompute (None where an input is missing)."""
        results: dict[int, Any] = {}
        n = len(trial.candidates)
        accuracy = session.row.gaze_check_accuracy
        for activation in trial.activations:
            regime, _, _ = self._governing(trial, activation.index)
            try:
                if activation.modality == "keyboard":
                    if not 0 <= activation.press_index < n:
                        results[activation.index] = None
                        continue
                    results[activation.index] = compute_posterior(
                        trial.candidates,
                        (activation.x, activation.y),
                        modality="keyboard",
                        r=None,
                        floor=None,
                        surface=(trial.surface_w, trial.surface_h),
                        press_index=activation.press_index,
                    )
                    continue
                if trial.spread is None:
                    results[activation.index] = None
                    continue
                gaze = None
                if activation.gaze is not None:
                    if accuracy is None:
                        results[activation.index] = None
                        continue
                    gaze = GazeInput(region=activation.gaze.region, conf=activation.gaze.conf, acc=accuracy)
                rendered = rendered_rects(trial.candidates, activation.expanded_index, _scale(regime))
                results[activation.index] = compute_posterior(
                    rendered,
                    (activation.x, activation.y),
                    modality=activation.modality,
                    r=trial.spread.r,
                    floor=trial.spread.floor,
                    surface=(trial.surface_w, trial.surface_h),
                    gaze=gaze,
                    press_index=activation.press_index,
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                results[activation.index] = None
        return results

    def _posteriors_ok(self, session: _Session, trial: TrialRecord, posteriors: Mapping[int, Any]) -> bool:
        if trial.l1_fallback_reason is None:
            decision = session.decisions.get(trial.regime_decision_id)
            decision_spread = decision.spread if isinstance(decision, Decision) else None
            if (trial.spread is None) != (decision_spread is None):
                return False
            if trial.spread is not None and decision_spread is not None:
                if trial.spread.r != decision_spread[0] or trial.spread.floor != decision_spread[1]:
                    return False
        n = len(trial.candidates)
        for activation in trial.activations:
            result = posteriors.get(activation.index)
            if result is None:
                return False
            if activation.modality == "keyboard":
                focused = activation.press_index
                if activation.hit_index != focused or not 0 <= focused < n:
                    return False
                rect = trial.candidates[focused]
                if (activation.x, activation.y) != keyboard_endpoint(rect) or activation.hit_rect != rect:
                    return False
                if activation.expanded_index is not None or activation.gaze is not None:
                    return False
                if activation.a_gate != 0 or activation.n_plausible != 1:
                    return False
                continue
            regime, _, _ = self._governing(trial, activation.index)
            if activation.expanded_index is not None and _scale(regime) == 1.0:
                return False
            if result.hit_index != activation.hit_index or result.n_plausible != activation.n_plausible:
                return False
            if abs(result.ambiguity_gate - activation.a_gate) > _A_GATE_TOLERANCE:
                return False
            if activation.hit_index is not None:
                rendered = rendered_rects(trial.candidates, activation.expanded_index, _scale(regime))
                if activation.hit_rect != rendered[activation.hit_index]:
                    return False
        return True

    def _contexts(self, trial: TrialRecord, posteriors: Mapping[int, Any]) -> dict[int, ActivationContext]:
        tau = self._settings.ambiguity_threshold
        contexts: dict[int, ActivationContext] = {}
        for activation in trial.activations:
            result = posteriors.get(activation.index)
            if activation.l2 is None or result is None:
                continue
            bin_id = ambiguity_bin(result.ambiguity_gate, tau) or "b3"
            contexts[activation.index] = ActivationContext(
                ambiguity=float(result.ambiguity_gate),
                bin=bin_id,  # type: ignore[arg-type]
                n_plausible=int(result.n_plausible),
                has_hit=bool(result.has_hit),
                p_hit=float(result.p_hit),
                p_top=float(result.p_top),
                modality=activation.modality,  # type: ignore[arg-type]
                layout=trial.layout,
            )
        return contexts

    # ------------------------------------------------------------------------------------------------- trials

    def observe_trial(self, trial: TrialRecord) -> TrialObservation:
        """Live path; the update time is trial.recorded_at (set by the service). Steps:
        1. Verification; the first failing check wins, in this order: 'interrupted'; the applied L1 regime, every
           L2 response, the revert regime, the cap rule (9.3) and the fallbacks_issued_at values ('draw_mismatch');
           then every activation's posterior ('posterior_mismatch'); then an L1 fallback ('fallback'); else 'ok'.
           L2 and L3 fallbacks do not make a trial 'fallback': they relabel only their own decision (9.7).
        2. Derives labels and L2 episodes (section 7.1) and checks every applied variant against the preferences in
           force when its source was issued (preference_violations, section 9.7).
        3. Builds the StoredTrial and applies it with the replay_trial routine; also updates session-scoped state
           (veto cooldown, except for a veto of a decision whose mode is pinned; offer cooldown).
        4. Runs the drift and C7 tests on the updated statistics; each alarm becomes an EngineEvent, is applied
           through apply_event and marks the session stale (mark_stale)."""
        session = self._require_session()
        if trial.session_id != session.row.session_id:
            raise ValueError("the trial belongs to another session")
        verification, posteriors = self._verify(session, trial)
        labels: TrialLabels | None = None
        episodes: tuple = ()
        if verification != "interrupted":
            labels = derive_labels(trial)
            episodes = derive_episodes(trial, self._contexts(trial, posteriors), labels)
        history = [pref for pref in self._preferences.all if pref.scope.participant_key in (None, self.participant_key)]
        violations = preference_violations(
            trial, history, decision_times=session.created, participant_key=self.participant_key
        )
        ucb: float | None = None
        if verification == "ok" and trial.task != "idle" and trial.applied_regime in _C7_REGIMES:
            decision = session.decisions.get(trial.regime_decision_id)
            if isinstance(decision, Decision) and trial.applied_regime in decision.predictions:
                ucb = float(decision.predictions[trial.applied_regime].ucb_unintended)
        stored = StoredTrial(
            trial=trial,
            verification=verification,  # type: ignore[arg-type]
            day_index=session.row.day_index,
            regime_ucb_unintended=ucb,
            labels=labels,
            episodes=episodes,
            preference_violations=violations,
        )
        self.replay_trial(stored)
        self._session_bookkeeping(session, trial)

        events: list[EngineEvent] = []
        refresh = False
        if verification == "ok":
            for head, outcome in self.model.drift_alarms():
                events.append(
                    EngineEvent.create(
                        kind="drift",
                        session_id=trial.session_id,
                        trial_id=trial.trial_id,
                        created_at=trial.recorded_at,
                        head=head,
                        outcome=outcome,
                    )
                )
            # Only a row that entered the C7 buffer (an acting regime with its decision's UCB) can change the test, so
            # only such a row may name the suspension it fires.
            if ucb is not None and self.model.calibration_test():
                events.append(
                    EngineEvent.create(
                        kind="calibration_suspended",
                        session_id=trial.session_id,
                        trial_id=trial.trial_id,
                        created_at=trial.recorded_at,
                    )
                )
        for event in events:
            self.apply_event(event)
            refresh = self.mark_stale(event.kind) or refresh
        return TrialObservation(
            stored=stored, learned=verification == "ok", refresh_required=refresh, events=tuple(events)
        )

    def _session_bookkeeping(self, session: _Session, trial: TrialRecord) -> None:
        """Live-only session state: C5 vetoes, C8 offer cooldown and the previous fallbacks_issued_at."""
        constraint = session.constraint
        constraint.trials_recorded += 1
        session.previous_issued_at = trial.fallbacks_issued_at
        veto = trial.veto
        if veto is not None:
            decision = session.decisions.get(veto.decision_id)
            pinned = (isinstance(decision, Decision | AmbiguityTable) and decision.mode is PolicyMode.PINNED) or (
                isinstance(decision, AmbiguityTable) and decision.pin_act
            )
            vetoed = vetoed_variant(trial)
            if not pinned and vetoed is not None:
                # The variant the chip showed: after a refresh revert that is the revert regime or a later response,
                # never the cue regime the refresh had already replaced (records.vetoed_variant).
                constraint.record_veto(DecisionLevel(veto.level), vetoed)
        if any(offer.response == "dismissed" for offer in trial.offers):
            constraint.dismiss_offer()

    def _trial_context(self, trial: TrialRecord) -> TrialContext:
        session = self._require_session()
        minutes = max(0.0, (trial.cue_at - session.row.started_at).total_seconds() / 60.0)
        return TrialContext(
            at=trial.recorded_at,
            layout=trial.layout,
            candidates=tuple(trial.candidates),
            target_index=int(trial.intended_index),  # type: ignore[arg-type]
            variant=trial.applied_regime,
            device=session.row.device,
            session_minute=minutes,
            hand_quality=None if trial.hand is None else float(trial.hand.quality_mean),
            tau=self._settings.ambiguity_threshold,
        )

    def replay_trial(self, stored: StoredTrial) -> None:
        """Rebuild path (section 10.8): applies a stored row without verifying and without reading any decision.
        'ok' → the updates of section 7.5; 'fallback' → practice counters only; any other value → nothing.
        Feeds the drift detectors and the C7 buffer but never fires an alarm."""
        session = self._require_session()
        trial = stored.trial
        session.trials.append(stored)
        idle = trial.task == "idle"
        if not idle:
            stats = self._block(session, trial.block_index)
            stats.interruptions += trial_interruptions(trial.activations)
            if not trial.interrupted:
                stats.active_trials += 1
            if stats.regime is None:
                stats.regime = trial.applied_regime
            session.constraint.interruptions_used[trial.block_index] = stats.interruptions
        if stored.verification == "ok" and stored.labels is not None:
            self._trials_seen += 1
            labels = stored.labels
            if idle:
                self.model.observe_idle(trial.modality, trial.applied_regime, labels.idle_commits, labels.idle_minutes)
                return
            ctx = self._trial_context(trial)
            outcome = TrialOutcome(
                complete=labels.complete,
                ua=labels.ua,
                correction=labels.correction,
                veto=labels.veto,
                log_time=math.log(labels.time_ms / 1000.0),
                censored=labels.censored,
            )
            self.model.observe_l1_trial(ctx, outcome, regime_ucb_unintended=stored.regime_ucb_unintended)
            for episode in stored.episodes:
                self.model.observe_l2_episode(
                    trial.recorded_at,
                    episode.context,
                    episode.variant,
                    EpisodeOutcome(
                        complete=episode.complete,
                        ua=episode.ua,
                        correction=episode.correction,
                        veto=episode.veto,
                        log_time=math.log(episode.time_ms / 1000.0),
                        censored=episode.censored,
                    ),
                )
            # Spread (7.5): the first activation aims at the cue-time rects only when no enlarge regime governed it.
            # A refresh or veto before the first activation (from_activation 0) puts the revert regime in force for it.
            first = trial.activations[0] if trial.activations else None
            if (
                first is not None
                and first.modality != "keyboard"
                and not self._governing(trial, first.index)[0].startswith("enlarge_target")
                and first.expanded_index is None
                and trial.intended_index is not None
            ):
                self.model.observe_spread(first.modality, first.x, first.y, trial.candidates[trial.intended_index])
            for offer in trial.offers:
                if offer.variant != "no_offer" and offer.response in ("accepted", "dismissed"):
                    self.model.observe_offer(trial.modality, trial.task, offer.variant, offer.response)
            if trial.applied_regime == "assistance_off" and labels.complete:
                self._off_times_s.append(labels.time_ms / 1000.0)
        elif stored.verification == "fallback" and not idle and trial.intended_index is not None:
            self.model.observe_practice_only(self._trial_context(trial))

    def apply_event(self, event: EngineEvent) -> None:
        """Applies a drift or calibration-suspension event, from a live alarm or a stored row: drift widens the slow
        block of that (head, outcome) filter (widen_beta, capped at the prior), resets its detector and sets the
        day's drift flag (section 8); calibration_suspended sets C7 until the next day boundary or learning reset
        (section 7.7)."""
        if event.kind == "drift":
            if event.head is None or event.outcome is None:
                raise ValueError("a drift event names a head and an outcome")
            self.model.apply_drift(event.head, event.outcome)
            return
        self.model.apply_calibration_suspension()
        session = self._session
        if session is not None:
            session.constraint.calibration_suspended = True
            if session.suspended_since is None:
                session.suspended_since = event.created_at

    # -------------------------------------------------------------------------------------------------- burden

    def observe_burden(self, report: BurdenReport) -> None:
        """Block ratings update the burden head (time from report.recorded_at); a control rating evaluates C6."""
        session = self._require_session()
        session.burden.append(report)
        if report.scope != "block" or report.block_index is None:
            return
        stats = session.blocks.get(report.block_index)
        if stats is None or stats.regime is None:
            return
        modality = session.row.device.modality
        ipt = stats.interruptions / max(stats.active_trials, 1)
        ratings = {"effort": report.effort, "intrusiveness": report.intrusiveness, "control": report.control}
        self.model.observe_burden(report.recorded_at, stats.regime, modality, ipt=ipt, ratings=ratings)
        if report.control is not None:
            latent = self.model.burden_latent(stats.regime, modality, ipt=ipt, now=report.recorded_at)["control"]
            mu, s2 = latent
            if control_floor_excludes(float(mu), math.sqrt(max(float(s2), 0.0))):
                session.constraint.add_control_floor(stats.regime)

    # ----------------------------------------------------------------------------------------- predictions etc.

    def predictions_for_block(self, block: BlockContext, *, now: datetime) -> dict[str, InteractionPrediction]:
        return self.model.predict_l1_block(
            block,
            l1_arms(block.device.modality),
            now=now,
            tau=self._settings.ambiguity_threshold,
            credibility=self._settings.upper_bound_credibility,
        )

    def policy_probabilities(
        self,
        *,
        block: BlockContext,
        session_exclusions: Mapping[str, ExclusionReason],
        preferences: PreferenceSet,
        seed: int,
        draws: int = 4096,
        now: datetime,
        offer: OfferContext | None = None,
        exclusions_level: DecisionLevel | str | None = None,
    ) -> PolicyProbabilities:
        """Evaluation only (WP4 off-policy evaluation, section 19); never called by the service. The adaptive-mode
        π this engine state would use for the block's L1 decision, for every cell of its L2 table and, when `offer`
        is given, for that offer, given the session-state exclusions logged with the evaluated decision (C1, C5, C6
        and C8 reasons) and the preferences active then; the engine applies its own C2, C7 and safety filter.
        Thompson draws come from Generator(PCG64(seed)) with `draws` samples: one set per filter for the L1 decision
        and one for the whole L2 table, which every cell shares. Never mutates state and logs nothing.

        `exclusions_level` is the level of the decision that logged `session_exclusions`. A "Not this time"
        excludes its variant at the next decision of the same level only (C5): an L1 decision logs the regime vetoes
        and a table logs the response vetoes in its cells, and live, a veto of the regime
        ``confirm_selection:second_click`` never removes the response of that name from the block's table. With a
        level, a logged ``recent_veto`` applies at that level only, as live. None treats the map as the union of
        both levels' exclusions and applies each ``recent_veto`` at every level that offers the variant (a variant
        that is both an L1 regime and an L2 response is then excluded from both); evaluation passes the level."""
        modality = block.device.modality
        settings = self._settings
        state = ConstraintState(calibration_suspended=bool(self.model.calibration_suspended))
        veto_levels = (
            (DecisionLevel.BLOCK_REGIME, DecisionLevel.AMBIGUITY_RESPONSE)
            if exclusions_level is None
            else (DecisionLevel(exclusions_level),)
        )
        level_options = {
            DecisionLevel.BLOCK_REGIME: l1_arms(modality),
            DecisionLevel.AMBIGUITY_RESPONSE: L2_OPTIONS,
            DecisionLevel.OFFER: L3_OPTIONS,
        }
        for variant, reason in session_exclusions.items():
            reason = ExclusionReason(reason)
            if reason is ExclusionReason.RECENT_VETO:
                for level in veto_levels:
                    if variant in level_options.get(level, ()):
                        state.record_veto(level, variant)
            elif reason is ExclusionReason.LOW_CONTROL_RATING:
                state.add_control_floor(variant)
            elif reason is ExclusionReason.OFFER_COOLDOWN:
                state.offers_blocked_until = 1
        rng = np.random.Generator(np.random.PCG64(seed))
        fallbacks_now = compute_fallbacks(
            preferences,
            modality,  # type: ignore[arg-type]
            participant_key=self.participant_key,
            calibration_suspended=state.calibration_suspended,
            now=now,
        )
        predictions = self.model.predict_l1_block(
            block,
            l1_arms(modality),
            now=now,
            tau=settings.ambiguity_threshold,
            credibility=settings.upper_bound_credibility,
        )

        def l1_sampler(rng_: np.random.Generator, variants: Sequence[str], n: int) -> Any:
            return self.model.sample_l1(block, variants, rng_, n, now=now, tau=settings.ambiguity_threshold)

        priorities = preferences.priorities(context="study", modality=modality, participant_key=self.participant_key)
        request = BlockRequest(
            decision_id="policy_probabilities",
            participant_key=self.participant_key,
            seed=seed,
            mode=PolicyMode.ADAPTIVE,
            modality=modality,
            preferences=preferences,
            state=state,
            fallbacks=fallbacks_now,
            settings=settings,
            predictions=predictions,
            sampler=l1_sampler,
            previous_regime=block.previous_regime,
            priorities=priorities,
            t_ref_s=self._t_ref_s(predictions),
            spread=None,
            round_remaining=None,
            round_index=0,
            comparator_ranking=(),
            protocol_deviation=False,
            versions=self._versions(),
        )
        arms = l1_arms(modality)
        safe, _ = self.policy.block_safe_set(request)
        if safe:
            pi_safe, _ = self.policy.adaptive_l1_probabilities(request, safe, rng, max(1, int(draws)))
            l1 = {variant: float(pi_safe.get(variant, 0.0)) for variant in arms}
        elif fallbacks_now.l1 is not None:
            l1 = {variant: 1.0 if variant == fallbacks_now.l1 else 0.0 for variant in arms}
        else:
            l1 = {variant: 0.0 for variant in arms}

        # One set of L2 parameter draws for the whole table, taken from the stream when the first cell needs them:
        # every cell's p_TS is the argmax frequency over the same `draws` samples of each L2 filter (9.2 step 5).
        # Drawing 4,096 samples of every filter again for each of the twelve cells made this evaluation-only call
        # dominate the OPE validation's run time (spec 22.4 budget). decide_ambiguity_table still draws per cell.
        # The model does not change during this call, so the cells also share the predictions and draw columns of
        # responses whose physics prior and features repeat across cells.
        l2_thetas: dict[tuple[str, str], np.ndarray] = {}
        l2_draw_memo: dict[Any, np.ndarray] = {}
        l2_prediction_memo: dict[Any, Any] = {}

        def l2_sampler(rng_: np.random.Generator, ctx: Any, variants: Sequence[str], n: int) -> Any:
            if not l2_thetas:
                l2_thetas.update(self.model.sample_l2_parameters(rng_, n, now=now))
            return self.model.l2_outcome_draws(ctx, variants, l2_thetas, memo=l2_draw_memo)

        def l2_predictor(ctx: Any, variants: Sequence[str]) -> Any:
            return self.model.predict_l2(
                ctx, variants, now=now, credibility=settings.upper_bound_credibility, memo=l2_prediction_memo
            )

        table_request = TableRequest(
            decision_id="policy_probabilities",
            participant_key=self.participant_key,
            block_decision_id="policy_probabilities",
            seed=seed,
            mode=PolicyMode.ADAPTIVE,
            modality=modality,
            preferences=preferences,
            state=state,
            fallbacks=fallbacks_now,
            settings=settings,
            cells=cell_specs(settings.ambiguity_threshold, block.layouts[0], modality),
            predict_cell=l2_predictor,
            sample_cell=l2_sampler,
            priorities=priorities,
            interruptions_used=0,
            start_trial_index=0,
            trials_in_block=0,
            versions=self._versions(),
        )
        l2: dict[str, dict[str, float]] = {}
        for spec in table_request.cells:
            cell_predictions = table_request.predict_cell(spec.context, list(L2_OPTIONS))
            safe_cell, _, pin_act = self.policy.cell_safe_set(table_request, spec, cell_predictions, adaptive=True)
            if pin_act:
                probabilities = {variant: 1.0 if variant == safe_cell[0] else 0.0 for variant in L2_OPTIONS}
            elif safe_cell:
                pi_cell, _ = self.policy.adaptive_cell_probabilities(
                    table_request, spec, safe_cell, rng, max(1, int(draws))
                )
                probabilities = {variant: float(pi_cell.get(variant, 0.0)) for variant in L2_OPTIONS}
            else:
                rule = fixed_l2_rule(fallbacks_now, spec.cell_id)
                probabilities = {variant: 1.0 if variant == rule else 0.0 for variant in L2_OPTIONS}
            l2[spec.cell_id] = probabilities

        l3: dict[str, float] | None = None
        if offer is not None:
            offer_request = OfferRequest(
                decision_id="policy_probabilities",
                participant_key=self.participant_key,
                seed=seed,
                mode=PolicyMode.ADAPTIVE,
                modality=offer.modality,
                keyboard_available=offer.keyboard_available,
                preferences=preferences,
                state=state,
                settings=settings,
                sampler=lambda rng_, variants, n: self.model.sample_offers(
                    offer.modality, offer.task, variants, rng_, n
                ),
                priorities=priorities,
                versions=self._versions(),
            )
            safe_offers, _ = self.policy.offer_safe_set(offer_request, adaptive=True)
            pi_offer, _ = self.policy.adaptive_offer_probabilities(offer_request, safe_offers, rng, max(1, int(draws)))
            l3 = {variant: float(pi_offer.get(variant, 0.0)) for variant in L3_OPTIONS}
        return PolicyProbabilities(l1=l1, l2=l2, l3=l3)

    def prequential_prediction(self, stored: StoredTrial) -> dict[str, float]:
        """Analysis and results only (H1, H4, H5, H8, H9 through WP4's prequential_scores; the calibration of the
        results RPC, section 13.3). Called before replay_trial(stored): the L1 trial head's predictive quantities
        for this trial, which are the ones the drift detectors score (section 8): realized regime and target
        (physics prior in update mode), filter state diffused to stored.trial.recorded_at. Keys "complete", "ua",
        "correction", "veto" (predictive probabilities Φ(μ/√(1 + s²))) and "log_time_mean", "log_time_sd"
        (μ and √(s² + σ_T²) of log seconds). Never mutates state."""
        if stored.trial.task == "idle" or stored.trial.intended_index is None:
            raise ValueError("prequential predictions exist for active trials only")
        return dict(self.model.l1_trial_prediction(self._trial_context(stored.trial)))

    def beta_means(self) -> dict[tuple[str, str], np.ndarray]:
        """Analysis only (core.evaluation.population_prior): copies of the current mean of every slow block β,
        keyed by (head, outcome)."""
        return self.model.beta_means()

    def spread(self) -> dict[str, float]:
        """modality -> r̂"""
        return self.model.spread()

    def status(self) -> EngineStatus:
        return EngineStatus(
            trials_seen=self._trials_seen,
            sessions_seen=self._sessions_seen,
            last_day_index=self.model.last_day_index,
            drift_flags=tuple(sorted(self.model.drift_flags)),
            calibration_suspended=bool(self.model.calibration_suspended),
            learn_across_sessions_effective=self._settings.learn_across_sessions,
            model_version=MODEL_VERSION,
            prior_version=PRIOR_VERSION,
            policy_version=POLICY_VERSION,
            feature_schema=FEATURE_SCHEMA,
        )

    # ------------------------------------------------------------------------------------------------- snapshot

    def _metadata(self) -> dict[str, Any]:
        return {
            "schema": ENGINE_SNAPSHOT_SCHEMA,
            "model": self.model.metadata(),
            "policy_version": POLICY_VERSION,
        }

    def snapshot(self) -> bytes:
        """Learned arrays only (section 10.6): the model's arrays, the engine's counters and a JSON metadata array,
        written as a deterministic npz (equal states give equal bytes). Session-scoped state is never included."""
        arrays = {f"model.{name}": value for name, value in self.model.to_arrays().items()}
        arrays["engine.counters"] = np.array([float(self._sessions_seen), float(self._trials_seen)])
        arrays["metadata"] = np.array(json.dumps(self._metadata(), sort_keys=True))
        return write_npz(arrays)

    @classmethod
    def restore(
        cls,
        data: bytes,
        *,
        participant_key: str,
        settings: EngineSettings,
        preferences: PreferenceSet,
        evidence_kind: EvidenceKind = EvidenceKind.LIVE_HUMAN,
        model_variant: ModelVariant | None = None,
        population_prior: PopulationPrior | None = None,
        policy: ConstrainedThompsonPolicy | None = None,
    ) -> CoAdaptationEngine:
        """Loads a snapshot with ``allow_pickle=False`` and strict version and shape checks; the restored engine has
        no open session."""
        try:
            arrays = read_npz(data)
        except Exception as exc:  # numpy raises several types for a corrupt archive
            raise ValueError("snapshot is not a valid npz archive") from exc
        if "metadata" not in arrays or "engine.counters" not in arrays:
            raise ValueError("snapshot is missing its metadata or counters")
        try:
            meta = json.loads(str(arrays.pop("metadata").item()))
        except (TypeError, ValueError) as exc:
            raise ValueError("snapshot metadata is not valid JSON") from exc
        if not isinstance(meta, dict) or meta.get("schema") != ENGINE_SNAPSHOT_SCHEMA:
            raise ValueError("snapshot schema does not match this engine")
        if meta.get("policy_version") != POLICY_VERSION:
            raise ValueError("snapshot policy version does not match")
        model_meta = meta.get("model")
        if not isinstance(model_meta, dict):
            raise ValueError("snapshot has no model metadata")
        variant = ModelVariant(model_variant) if model_variant is not None else ModelVariant(model_meta.get("variant"))
        engine = cls(
            participant_key=participant_key,
            settings=settings,
            preferences=preferences,
            evidence_kind=evidence_kind,
            model_variant=variant,
            population_prior=population_prior,
            policy=policy,
        )
        if model_meta != engine.model.metadata():
            raise ValueError("snapshot versions or model variant do not match this engine")
        counters = np.asarray(arrays.pop("engine.counters"), dtype=np.float64)
        if counters.shape != (2,) or not np.all(np.isfinite(counters)) or np.any(counters < 0):
            raise ValueError("snapshot counters have the wrong shape or values")
        model_arrays = {}
        for name, value in arrays.items():
            if not name.startswith("model."):
                raise ValueError("snapshot holds an unknown array")
            model_arrays[name[len("model.") :]] = value
        engine.model.from_arrays(model_arrays)
        engine._sessions_seen = int(counters[0])
        engine._trials_seen = int(counters[1])
        return engine


def settings_fields() -> tuple[str, ...]:
    """Every ``EngineSettings`` field name (``SessionRow.settings`` holds exactly these)."""
    return tuple(f.name for f in fields(EngineSettings))
