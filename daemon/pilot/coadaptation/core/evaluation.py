"""Interaction metrics and prequential scoring for the co-adaptation study (sections 2, 7.7, 12.7 and 13.1).

``evaluate_interaction_trials`` scores what happened in selection trials and idle epochs; ``prequential_scores``
replays stored rows into a fresh engine of one ``ModelVariant`` and scores each trial with the prediction made just
before the trial updated the model (Dawid 1984), which is how H1, H4, H5, H8 and H9 compare model variants.
Standard library, numpy and ``pilot.coadaptation.core`` only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import numpy as np

from .engine import CoAdaptationEngine, EngineEvent, EngineSettings
from .outcome_model import ModelVariant, PopulationPrior
from .preferences import Preference, PreferenceSet
from .records import BurdenReport, SessionRow, StoredTrial
from .special import log_norm_cdf, norm_cdf, norm_pdf
from .utility import pointing_throughput

BINARY_OUTCOMES = ("complete", "ua", "correction", "veto")
OUTCOMES = (*BINARY_OUTCOMES, "log_time")
# H1's primary scoring set: regimes the full model's own policy did not choose.
PRIMARY_PHASES = ("A", "A_lite")
_P_CLIP = 1e-6

# The throughput of section 12.7, shared with the service's results (core.utility).
throughput = pointing_throughput


# ------------------------------------------------------------------------------------------- interaction trials


@dataclass(frozen=True, slots=True)
class InteractionTrial:
    """One active selection trial or one idle epoch, as scored by ``evaluate_interaction_trials``."""

    task: str  # "pointing", "menu" or "idle"
    duration_seconds: float  # trial time from cue to its end, or the idle epoch's duration
    intended_index: int | None  # None exactly for idle epochs
    committed: bool = False
    committed_index: int | None = None
    corrections: int = 0
    time_ms: float | None = None  # cue to the correct commit; completed trials only
    # Commits of a non-intended candidate (undone ones included); in idle epochs any commit.
    unintended_commits: int = 0

    def __post_init__(self) -> None:
        if self.task not in ("pointing", "menu", "idle"):
            raise ValueError("task must be pointing, menu or idle")
        if not np.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("trial duration must be finite and positive")
        if (self.intended_index is None) != (self.task == "idle"):
            raise ValueError(
                "an idle epoch has no intended target and every selection trial has one",
            )
        if self.committed and self.committed_index is None:
            raise ValueError("a committed trial requires a committed candidate")
        if not self.committed and self.committed_index is not None:
            raise ValueError("a committed candidate belongs only to committed trials")
        for name in ("corrections", "unintended_commits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.time_ms is not None:
            if not np.isfinite(self.time_ms) or self.time_ms < 0:
                raise ValueError("trial time must be finite and non-negative")
            if not self.completed:
                raise ValueError("a trial time belongs only to completed trials")

    @property
    def completed(self) -> bool:
        hit = self.committed and self.committed_index == self.intended_index
        return self.task != "idle" and hit

    @property
    def unintended(self) -> bool:
        """At least one unintended activation: a commit of a non-intended candidate, in an idle epoch any commit."""
        return self.unintended_count > 0

    @property
    def unintended_count(self) -> int:
        final = int(
            self.committed and (self.task == "idle" or self.committed_index != self.intended_index),
        )
        return max(self.unintended_commits, final)

    @classmethod
    def from_stored(cls, stored: StoredTrial) -> InteractionTrial:
        """The scored view of a stored trial (labels as the daemon derived them)."""
        trial = stored.trial
        labels = stored.labels
        idle = trial.task == "idle"
        duration_ms = trial.duration_ms if idle else max(trial.time_ms, trial.duration_ms)
        complete = labels is not None and labels.complete
        wrong = sum(
            1
            for activation in trial.activations
            if activation.outcome == "committed"
            and activation.committed_index is not None
            and (idle or activation.committed_index != trial.intended_index)
        )
        return cls(
            task=trial.task,
            duration_seconds=max(float(duration_ms), 1.0) / 1000.0,
            intended_index=trial.intended_index,
            committed=trial.committed_index is not None,
            committed_index=trial.committed_index,
            corrections=int(trial.corrections),
            time_ms=float(labels.time_ms) if complete and labels is not None else None,
            unintended_commits=wrong,
        )


@dataclass(frozen=True, slots=True)
class InteractionEvaluation:
    trial_count: int
    active_trials: int
    idle_epochs: int
    completed_trials: int
    completion: float
    ua_per_trial: float
    idle_minutes: float
    idle_commits: int
    ua_per_idle_minute: float
    corrections_per_trial: float
    median_time_ms: float | None
    p95_time_ms: float | None
    throughput_bits_s: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_interaction_trials(
    trials: Sequence[InteractionTrial], *, throughput_trials: Sequence[StoredTrial] = ()
) -> InteractionEvaluation:
    """Completion, unintended activations per trial and per idle minute, corrections per trial, median and p95
    time of completed trials, and pointing throughput.

    Unintended activations per idle minute use the arithmetic of ``pilot/neural/evaluation.py:83-100`` (false
    commits during no-control epochs over their summed duration), per minute instead of per hour. Throughput comes
    from ``core.utility.pointing_throughput`` over ``throughput_trials`` (None without them).
    """
    if not trials:
        raise ValueError("at least one trial is required")
    active = [trial for trial in trials if trial.task != "idle"]
    idle = [trial for trial in trials if trial.task == "idle"]
    completed = [trial for trial in active if trial.completed]
    unintended = sum(trial.unintended for trial in active)
    idle_seconds = sum(trial.duration_seconds for trial in idle)
    idle_false_commits = sum(trial.unintended_count for trial in idle)
    times = sorted(float(trial.time_ms) for trial in completed if trial.time_ms is not None)
    idle_rate = round(idle_false_commits / (idle_seconds / 60), 6) if idle_seconds else 0.0
    corrections = sum(trial.corrections for trial in active)
    return InteractionEvaluation(
        trial_count=len(trials),
        active_trials=len(active),
        idle_epochs=len(idle),
        completed_trials=len(completed),
        completion=round(len(completed) / len(active), 6) if active else 0.0,
        ua_per_trial=round(unintended / len(active), 6) if active else 0.0,
        idle_minutes=round(idle_seconds / 60, 6),
        idle_commits=idle_false_commits,
        ua_per_idle_minute=idle_rate,
        corrections_per_trial=round(corrections / len(active), 6) if active else 0.0,
        median_time_ms=round(median(times), 3) if times else None,
        p95_time_ms=round(float(np.percentile(times, 95)), 3) if times else None,
        throughput_bits_s=throughput(throughput_trials) if throughput_trials else None,
    )


# ------------------------------------------------------------------------------------------- scoring rules


def brier(p: Sequence[float], y: Sequence[float]) -> float:
    """Mean squared difference between predicted probabilities and binary outcomes."""
    p_arr, y_arr = _paired(p, y)
    return float(np.mean((p_arr - y_arr) ** 2))


def ece(p: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error over ``bins`` equal-width probability bins (empty bins left out)."""
    p_arr, y_arr = _paired(p, y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p_arr, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            gap = abs(float(p_arr[mask].mean()) - float(y_arr[mask].mean()))
            total += mask.sum() / len(p_arr) * gap
    return float(total)


def log_score(p: Sequence[float], y: Sequence[float]) -> float:
    """Mean log predictive probability of binary outcomes (higher is better); ``p`` clipped to [1e-6, 1 - 1e-6]."""
    p_arr, y_arr = _paired(p, y)
    p_arr = np.clip(p_arr, _P_CLIP, 1.0 - _P_CLIP)
    return float(np.mean(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log1p(-p_arr)))


def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting and Raftery 2007); lower is better."""
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    z = (y - mu) / sigma
    return float(
        sigma * (z * (2.0 * norm_cdf(z) - 1.0) + 2.0 * norm_pdf(z) - 1.0 / math.sqrt(math.pi)),
    )


def binary_log_score(p: float, y: bool) -> float:
    p = min(max(float(p), _P_CLIP), 1.0 - _P_CLIP)
    return math.log(p) if y else math.log1p(-p)


def censored_time_log_score(mean: float, sd: float, *, log_time: float, censored: bool) -> float:
    """Log predictive density of a completed trial's log time, or ``log P(T >= limit)`` of a censored one."""
    if not sd > 0:
        raise ValueError("sd must be positive")
    z = (log_time - mean) / sd
    if censored:
        return float(log_norm_cdf(-z))
    return float(-0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi))


def _paired(p: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.shape != y_arr.shape or p_arr.ndim != 1 or not len(p_arr):
        raise ValueError("predictions and outcomes must be non-empty and paired")
    return p_arr, y_arr


# ---------------------------------------------------------------------------------------- prequential scoring


@dataclass(frozen=True, slots=True)
class TrialScore:
    """The prequential score of one trial: each outcome's log score under the prediction made before it."""

    trial_id: str
    participant_key: str
    session_id: str
    session_number: int
    block_index: int
    trial_index: int
    phase: str
    primary: bool  # Phase A or A-lite (H1's primary scoring set)
    active_index: int  # position among the session's scored active trials (H9 scores the first 24 of sessions 2, 3)
    scores: Mapping[str, float]  # complete, ua, correction, veto, log_time
    predictions: Mapping[str, float]  # complete, ua, correction, veto, log_time_mean, log_time_sd
    outcomes: Mapping[str, float]  # complete, ua, correction, veto (0/1), log_time, censored (0/1)

    @property
    def total(self) -> float:
        return float(sum(self.scores[outcome] for outcome in OUTCOMES))


@dataclass(frozen=True, slots=True)
class PrequentialScores:
    variant: str
    trials: tuple[TrialScore, ...]

    def select(
        self,
        *,
        primary: bool | None = None,
        sessions: Iterable[int] | None = None,
        first: int | None = None,
    ) -> tuple[TrialScore, ...]:
        """A scoring set: ``primary`` restricts to Phase A and A-lite trials, ``sessions`` to session numbers and
        ``first`` to the first scored active trials of each session."""
        wanted = None if sessions is None else set(sessions)
        return tuple(
            score
            for score in self.trials
            if (primary is None or score.primary == primary)
            and (wanted is None or score.session_number in wanted)
            and (first is None or score.active_index < first)
        )

    def mean(self, scores: Sequence[TrialScore] | None = None) -> dict[str, float]:
        """Mean per-trial log score per outcome and summed (``total``)."""
        chosen = self.trials if scores is None else scores
        if not chosen:
            return {**{outcome: math.nan for outcome in OUTCOMES}, "total": math.nan}
        out = {outcome: float(np.mean([s.scores[outcome] for s in chosen])) for outcome in OUTCOMES}
        out["total"] = float(np.mean([s.total for s in chosen]))
        return out

    def by_participant(
        self,
        scores: Sequence[TrialScore] | None = None,
    ) -> dict[str, dict[str, float]]:
        chosen = self.trials if scores is None else scores
        groups: dict[str, list[TrialScore]] = {}
        for score in chosen:
            groups.setdefault(score.participant_key, []).append(score)
        return {key: self.mean(values) for key, values in sorted(groups.items())}


def _phase_of(row: SessionRow, block_index: int) -> str:
    for plan in row.plan.blocks:
        if plan.index == block_index:
            return plan.phase
    return ""


def score_trial(
    prediction: Mapping[str, float],
    stored: StoredTrial,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-outcome log scores of one trial under ``prediction`` (the engine's ``prequential_prediction``)."""
    labels = stored.labels
    if labels is None:
        raise ValueError("a scored trial needs labels")
    outcomes = {name: float(bool(getattr(labels, name))) for name in BINARY_OUTCOMES}
    scores = {n: binary_log_score(prediction[n], bool(outcomes[n])) for n in BINARY_OUTCOMES}
    censored = not labels.complete
    seconds = (float(labels.limit_ms) if censored else max(float(labels.time_ms), 1.0)) / 1000.0
    log_time = math.log(seconds)
    scores["log_time"] = censored_time_log_score(
        prediction["log_time_mean"], prediction["log_time_sd"], log_time=log_time, censored=censored
    )
    outcomes["log_time"] = log_time
    outcomes["censored"] = float(censored)
    return scores, outcomes


def _scored(stored: StoredTrial) -> bool:
    return stored.verification == "ok" and stored.labels is not None and stored.trial.task != "idle"


def _replays_stored_drift(variant: ModelVariant, population_prior: PopulationPrior | None) -> bool:
    """Whether a replay engine takes its drift alarms from the stored ``EngineEvent`` rows (section 10.8).

    Stored drift events are the alarms of the live engine that recorded the rows, and the service always runs
    ``FULL`` from the literature prior (section 13.1.1). Only that configuration replays them. Every other
    configuration (each ablation variant, and ``FULL`` started from an H8 population prior) has its own detectors fed
    by its own predictions, so it ignores the stored drift events and raises its own alarms after each ``ok`` trial,
    as a live run of it would (``_raise_own_drift_alarms``). A variant without drift detectors (``PERSON_STATIC``,
    ``PHYSICS_ONLY``, ``POPULATION``) therefore never sees a drift widening.
    """
    return ModelVariant(variant) is ModelVariant.FULL and population_prior is None


def _raise_own_drift_alarms(engine: CoAdaptationEngine, stored: StoredTrial) -> None:
    """The drift step of the live path (``CoAdaptationEngine.observe_trial`` step 4) for a replay engine that does
    not replay the stored drift events: after an ``ok`` trial every alarm of the engine's own detectors becomes a
    drift event applied through ``apply_event``."""
    if stored.verification != "ok":
        return
    trial = stored.trial
    for head, outcome in engine.model.drift_alarms():
        engine.apply_event(
            EngineEvent.create(
                kind="drift",
                session_id=trial.session_id,
                trial_id=trial.trial_id,
                created_at=trial.recorded_at,
                head=head,
                outcome=outcome,
            )
        )


def _replay_stored_event(engine: CoAdaptationEngine, event: EngineEvent, *, stored_drift: bool) -> None:
    """A stored event row: calibration suspensions always (C7 reads the logged UCBs, not the variant's predictions);
    drift events only when ``stored_drift`` (see ``_replays_stored_drift``)."""
    if event.kind == "drift" and not stored_drift:
        return
    engine.apply_event(event)


def prequential_scores(
    rows: Sequence[Any],
    *,
    variant: ModelVariant,
    population_prior: PopulationPrior | None = None,
    preferences: Sequence[Preference] = (),
) -> PrequentialScores:
    """Replay ``rows`` (``SessionRow``, ``StoredTrial``, ``BurdenReport`` and ``EngineEvent`` in replay order,
    section 10.8) into a fresh engine of ``variant`` and score every ``ok`` active trial with
    ``prequential_prediction`` called before its ``replay_trial``.

    Stored drift events are replayed only by ``FULL`` without a population prior, the configuration that recorded
    them; any other configuration raises its own drift alarms instead (``_replays_stored_drift``), so an ablation is
    never widened at the full model's alarm times.

    ``log_time`` contributes the log predictive density of a completed trial and the log predictive survival
    probability ``log P(T >= limit)`` of a censored one; each outcome is kept separately and summed. Every trial is
    tagged with its phase, so the caller can take H1's primary set (Phase A and A-lite) or all trials.
    """
    stored_drift = _replays_stored_drift(variant, population_prior)
    engine: CoAdaptationEngine | None = None
    session: SessionRow | None = None
    active_index = 0
    scored: list[TrialScore] = []
    for row in rows:
        if isinstance(row, SessionRow):
            if engine is None:
                engine = CoAdaptationEngine(
                    participant_key=row.participant_key,
                    settings=EngineSettings(**row.settings),
                    preferences=PreferenceSet(preferences),
                    model_variant=variant,
                    population_prior=population_prior,
                )
            elif session is not None:
                _end(engine, session)
            engine.begin_session(row)
            session = row
            active_index = 0
            continue
        if engine is None or session is None:
            raise ValueError("rows must start with a session")
        if isinstance(row, StoredTrial):
            if _scored(row):
                prediction = engine.prequential_prediction(row)
                scores, outcomes = score_trial(prediction, row)
                phase = _phase_of(session, row.trial.block_index)
                scored.append(
                    TrialScore(
                        trial_id=row.trial.trial_id,
                        participant_key=session.participant_key,
                        session_id=session.session_id,
                        session_number=session.session_number,
                        block_index=row.trial.block_index,
                        trial_index=row.trial.trial_index,
                        phase=phase,
                        primary=phase in PRIMARY_PHASES,
                        active_index=active_index,
                        scores=scores,
                        predictions=dict(prediction),
                        outcomes=outcomes,
                    )
                )
                active_index += 1
            engine.replay_trial(row)
            if not stored_drift:
                _raise_own_drift_alarms(engine, row)
        elif isinstance(row, BurdenReport):
            engine.observe_burden(row)
        elif isinstance(row, EngineEvent):
            _replay_stored_event(engine, row, stored_drift=stored_drift)
        else:
            raise ValueError("unknown replay row")
    if engine is not None and session is not None:
        _end(engine, session)
    return PrequentialScores(variant=ModelVariant(variant).value, trials=tuple(scored))


def _end(engine: CoAdaptationEngine, session: SessionRow) -> None:
    if session.ended_at is not None:
        engine.end_session(ended_at=session.ended_at, reason=session.end_reason or "completed")


def rebuild_engine(
    rows: Sequence[Any],
    *,
    variant: ModelVariant = ModelVariant.FULL,
    population_prior: PopulationPrior | None = None,
    preferences: Sequence[Preference] = (),
) -> CoAdaptationEngine:
    """A fresh engine of ``variant`` with ``rows`` replayed into it (section 10.8): the per-participant engines
    ``population_prior`` pools. Drift events follow ``prequential_scores``: stored ones for ``FULL`` without a
    population prior, the engine's own alarms for every other configuration."""
    stored_drift = _replays_stored_drift(variant, population_prior)
    engine: CoAdaptationEngine | None = None
    session: SessionRow | None = None
    for row in rows:
        if isinstance(row, SessionRow):
            if engine is None:
                engine = CoAdaptationEngine(
                    participant_key=row.participant_key,
                    settings=EngineSettings(**row.settings),
                    preferences=PreferenceSet(preferences),
                    model_variant=variant,
                    population_prior=population_prior,
                )
            elif session is not None:
                _end(engine, session)
            engine.begin_session(row)
            session = row
        elif engine is None:
            raise ValueError("rows must start with a session")
        elif isinstance(row, StoredTrial):
            engine.replay_trial(row)
            if not stored_drift:
                _raise_own_drift_alarms(engine, row)
        elif isinstance(row, BurdenReport):
            engine.observe_burden(row)
        elif isinstance(row, EngineEvent):
            _replay_stored_event(engine, row, stored_drift=stored_drift)
    if engine is None:
        raise ValueError("no session to replay")
    if session is not None:
        _end(engine, session)
    return engine


def population_prior(
    engines: Mapping[str, CoAdaptationEngine] | Sequence[CoAdaptationEngine],
    *,
    leave_out: str | None = None,
) -> PopulationPrior:
    """The H8 population prior: the equal-weight mean of the other participants' slow-block means
    (``engine.beta_means()``) and the median of their spread estimates (``engine.spread()``) per modality.

    Pass a mapping from participant key to that participant's ``FULL`` engine (rebuilt from their rows) and
    ``leave_out`` the participant being scored, who is then never among the sources.
    """
    if isinstance(engines, Mapping):
        sources = [engine for key, engine in sorted(engines.items()) if key != leave_out]
    else:
        if leave_out is not None:
            raise ValueError("leave_out needs engines keyed by participant")
        sources = list(engines)
    if not sources:
        raise ValueError("the population prior needs at least one other participant")
    means: dict[tuple[str, str], list[np.ndarray]] = {}
    spreads: dict[str, list[float]] = {}
    for engine in sources:
        for key, value in engine.beta_means().items():
            means.setdefault(key, []).append(np.asarray(value, dtype=float))
        for modality, r in engine.spread().items():
            spreads.setdefault(modality, []).append(float(r))
    return PopulationPrior(
        beta_means={key: np.mean(np.stack(values), axis=0) for key, values in means.items()},
        spread_r0={modality: float(np.median(values)) for modality, values in spreads.items()},
        source_participants=len(sources),
    )
