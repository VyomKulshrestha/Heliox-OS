"""Interaction metrics, scoring rules and prequential scoring of the co-adaptation evaluation layer."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from pilot.coadaptation.core import utility
from pilot.coadaptation.core.dynamics import DRIFT_THRESHOLDS
from pilot.coadaptation.core.engine import EngineEvent
from pilot.coadaptation.core.evaluation import (
    InteractionTrial,
    binary_log_score,
    brier,
    censored_time_log_score,
    crps_gaussian,
    ece,
    evaluate_interaction_trials,
    log_score,
    population_prior,
    prequential_scores,
    rebuild_engine,
    throughput,
)
from pilot.coadaptation.core.ope import replay_rows
from pilot.coadaptation.core.outcome_model import ModelVariant, PopulationPrior
from pilot.coadaptation.core.simulator import (
    EnginePolicy,
    SimulatedPerson,
    SimulatedStudyRunner,
    make_population,
)
from pilot.neural.evaluation import NeuralIntentTrial, evaluate_neural_intent_trials
from pilot.neural.protocol import NeuralIntentClass


def test_interaction_trial_invariants() -> None:
    with pytest.raises(ValueError):
        InteractionTrial(task="pointing", duration_seconds=math.nan, intended_index=0)
    with pytest.raises(ValueError):
        InteractionTrial(task="pointing", duration_seconds=0.0, intended_index=0)
    with pytest.raises(ValueError):
        InteractionTrial(task="pointing", duration_seconds=1.0, intended_index=0, committed=True)
    with pytest.raises(ValueError):
        InteractionTrial(task="idle", duration_seconds=30.0, intended_index=2)
    with pytest.raises(ValueError):
        InteractionTrial(task="menu", duration_seconds=1.0, intended_index=None)
    with pytest.raises(ValueError):
        InteractionTrial(task="pointing", duration_seconds=1.0, intended_index=0, corrections=-1)
    with pytest.raises(ValueError):
        InteractionTrial(task="pointing", duration_seconds=1.0, intended_index=0, time_ms=900.0)
    trial = InteractionTrial(
        task="pointing",
        duration_seconds=1.2,
        intended_index=3,
        committed=True,
        committed_index=3,
        time_ms=1100.0,
    )
    assert trial.completed and not trial.unintended
    undone = InteractionTrial(
        task="pointing",
        duration_seconds=2.0,
        intended_index=3,
        committed=True,
        committed_index=3,
        unintended_commits=1,
    )
    assert undone.completed and undone.unintended


def test_idle_ua_per_minute_uses_the_neural_arithmetic() -> None:
    durations = [30.0, 30.0, 28.5, 31.0]
    commits = [True, False, True, False]
    neural = evaluate_neural_intent_trials(
        [
            NeuralIntentTrial(
                duration_seconds=d,
                expected_intent=None,
                predicted_intent=next(iter(NeuralIntentClass)) if c else None,
                committed=c,
                latency_ms=10.0 if c else None,
            )
            for d, c in zip(durations, commits)
        ]
    )
    ours = evaluate_interaction_trials(
        [
            InteractionTrial(
                task="idle",
                duration_seconds=d,
                intended_index=None,
                committed=c,
                committed_index=0 if c else None,
            )
            for d, c in zip(durations, commits)
        ]
    )
    assert ours.ua_per_idle_minute == pytest.approx(
        neural.false_commits_per_idle_hour / 60.0,
        abs=1e-5,
    )
    assert ours.idle_commits == 2
    assert ours.idle_minutes == pytest.approx(sum(durations) / 60.0)


def test_interaction_metrics() -> None:
    trials = [
        InteractionTrial(
            task="pointing",
            duration_seconds=1.0,
            intended_index=0,
            committed=True,
            committed_index=0,
            time_ms=800.0,
        ),
        InteractionTrial(
            task="pointing",
            duration_seconds=2.0,
            intended_index=0,
            committed=True,
            committed_index=1,
            corrections=1,
        ),
        InteractionTrial(task="menu", duration_seconds=15.0, intended_index=4),
        InteractionTrial(
            task="menu",
            duration_seconds=1.5,
            intended_index=2,
            committed=True,
            committed_index=2,
            time_ms=1400.0,
        ),
    ]
    result = evaluate_interaction_trials(trials)
    assert result.active_trials == 4
    assert result.completion == pytest.approx(0.5)
    assert result.ua_per_trial == pytest.approx(0.25)
    assert result.corrections_per_trial == pytest.approx(0.25)
    assert result.median_time_ms == pytest.approx(1100.0)
    assert result.p95_time_ms == pytest.approx(float(np.percentile([800.0, 1400.0], 95)))
    assert result.throughput_bits_s is None
    with pytest.raises(ValueError):
        evaluate_interaction_trials([])


def test_scoring_rule_arithmetic() -> None:
    p = [0.1, 0.8, 0.5, 0.95]
    y = [0, 1, 1, 1]
    assert brier(p, y) == pytest.approx(np.mean([(a - b) ** 2 for a, b in zip(p, y)]))
    assert log_score(p, y) == pytest.approx(
        np.mean([math.log(a) if b else math.log(1 - a) for a, b in zip(p, y)]),
    )
    assert binary_log_score(0.2, True) == pytest.approx(math.log(0.2))
    assert binary_log_score(0.2, False) == pytest.approx(math.log(0.8))
    # Ten-bin ECE: |mean p - mean y| per bin, weighted by the bin's share.
    p_bins = [0.05, 0.07, 0.55, 0.58, 0.92]
    y_bins = [0, 1, 1, 0, 1]
    expected = (2 / 5) * abs(0.06 - 0.5) + (2 / 5) * abs(0.565 - 0.5) + (1 / 5) * abs(0.92 - 1.0)
    assert ece(p_bins, y_bins) == pytest.approx(expected)
    with pytest.raises(ValueError):
        brier([0.1], [0, 1])


def test_censored_time_log_score_matches_the_normal_density_and_survival() -> None:
    mean, sd = math.log(1.4), 0.4
    completed = censored_time_log_score(mean, sd, log_time=math.log(1.1), censored=False)
    assert completed == pytest.approx(
        stats.norm.logpdf(math.log(1.1), loc=mean, scale=sd),
        abs=1e-10,
    )
    censored = censored_time_log_score(mean, sd, log_time=math.log(10.0), censored=True)
    assert censored == pytest.approx(
        stats.norm.logsf(math.log(10.0), loc=mean, scale=sd),
        abs=1e-10,
    )
    far = censored_time_log_score(0.0, 0.3, log_time=12.0, censored=True)
    assert math.isfinite(far) and far < -500


def test_crps_gaussian_closed_form_matches_integration() -> None:
    mu, sigma, y = 0.3, 0.7, 1.1

    def integrand(x: float) -> float:
        return (stats.norm.cdf(x, mu, sigma) - (1.0 if x >= y else 0.0)) ** 2

    numeric = integrate.quad(integrand, -10, y)[0] + integrate.quad(integrand, y, 10)[0]
    assert crps_gaussian(mu, sigma, y) == pytest.approx(numeric, abs=1e-7)
    with pytest.raises(ValueError):
        crps_gaussian(0.0, 0.0, 1.0)


def test_throughput_is_the_shared_implementation() -> None:
    assert throughput is utility.pointing_throughput


class _FakeEngine:
    def __init__(self, beta: float, r: float) -> None:
        self._beta = beta
        self._r = r

    def beta_means(self) -> dict[tuple[str, str], np.ndarray]:
        return {
            ("l1", "ua"): np.full(3, self._beta),
            ("l1", "log_time"): np.full(3, 2 * self._beta),
        }

    def spread(self) -> dict[str, float]:
        return {"mouse": self._r}


def test_population_prior_pools_other_participants_only() -> None:
    engines = {
        "p-00000001": _FakeEngine(1.0, 1.0),
        "p-00000002": _FakeEngine(3.0, 1.2),
        "p-00000003": _FakeEngine(100.0, 9.0),
    }
    prior = population_prior(engines, leave_out="p-00000003")
    assert prior.source_participants == 2
    assert np.allclose(prior.beta_means[("l1", "ua")], 2.0)
    assert np.allclose(prior.beta_means[("l1", "log_time")], 4.0)
    assert prior.spread_r0["mouse"] == pytest.approx(1.1)
    everyone = population_prior(
        [_FakeEngine(1.0, 1.0), _FakeEngine(2.0, 3.0), _FakeEngine(3.0, 2.0)],
    )
    assert everyone.spread_r0["mouse"] == pytest.approx(2.0)  # the median, not the mean
    with pytest.raises(ValueError):
        population_prior([_FakeEngine(1.0, 1.0)], leave_out="p-00000001")
    with pytest.raises(ValueError):
        population_prior({"p-00000001": _FakeEngine(1.0, 1.0)}, leave_out="p-00000001")


def test_prequential_scores_predict_before_each_update() -> None:
    person = make_population(["F1"], 1, seed=20261015)[0]
    result = SimulatedStudyRunner(SimulatedPerson(person), lambda: EnginePolicy(), seed=7).run(
        sessions=2,
    )
    rows = replay_rows(result.rows(), result.participant_key)
    full = prequential_scores(rows, variant=ModelVariant.FULL, preferences=result.preferences)
    physics = prequential_scores(
        rows,
        variant=ModelVariant.PHYSICS_ONLY,
        preferences=result.preferences,
    )
    ok_active = [s for s in result.trials if s.verification == "ok" and s.trial.task != "idle"]
    assert len(full.trials) == len(ok_active)
    assert {score.phase for score in full.select(primary=True)} <= {"A", "A_lite"}
    assert any(not score.primary for score in full.trials)  # Phase B trials form the secondary set
    for score in full.trials:
        outcomes = ("complete", "ua", "correction", "veto", "log_time")
        assert all(math.isfinite(score.scores[outcome]) for outcome in outcomes)
    censored = [score for score in full.trials if score.outcomes["censored"]]
    for score in censored:
        assert score.scores["log_time"] == pytest.approx(
            stats.norm.logsf(
                score.outcomes["log_time"],
                score.predictions["log_time_mean"],
                score.predictions["log_time_sd"],
            ),
            abs=1e-9,
        )
    # The first trial is scored before any update, so every variant predicts it from the same prior.
    assert full.trials[0].predictions == pytest.approx(physics.trials[0].predictions)
    assert full.mean()["total"] == pytest.approx(np.mean([s.total for s in full.trials]))
    early = full.select(sessions=(2,), first=24)
    assert len(early) == 24 and all(s.session_number == 2 and s.active_index < 24 for s in early)


def test_only_the_recording_configuration_replays_stored_drift_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored drift events are the live FULL engine's alarms. FULL without a population prior replays them (10.8);
    every other configuration raises its own alarms, so PERSON_STATIC (no drift detectors) is never widened at FULL's
    alarm times and the ablations of H4, H5 and H9 get their own alarm times."""
    # Low thresholds, for the live run and every replay, so the live FULL engine raises L1 alarms on any run.
    monkeypatch.setitem(DRIFT_THRESHOLDS, "binary", 2.0)
    monkeypatch.setitem(DRIFT_THRESHOLDS, "log_time", 2.0)
    person = make_population(["F4"], 1, seed=20261015)[0]
    result = SimulatedStudyRunner(SimulatedPerson(person), lambda: EnginePolicy(), seed=7).run(sessions=2)
    rows = replay_rows(result.rows(), result.participant_key)
    drift = [row for row in rows if isinstance(row, EngineEvent) and row.kind == "drift"]
    assert any(event.head == "l1" for event in drift)  # the live FULL engine raised an L1 alarm
    stripped = [row for row in rows if not (isinstance(row, EngineEvent) and row.kind == "drift")]
    prefs = result.preferences

    def scores(replayed: list, variant: ModelVariant, prior: PopulationPrior | None = None) -> list:
        return [
            (s.trial_id, s.scores, s.predictions)
            for s in prequential_scores(replayed, variant=variant, population_prior=prior, preferences=prefs).trials
        ]

    full = scores(rows, ModelVariant.FULL)
    assert full != scores(stripped, ModelVariant.FULL)  # the recording configuration replays the stored alarm
    # A FULL engine started from a neutral population prior is not the recording configuration: it raises its own
    # alarms, which on these rows are exactly the live engine's, whether or not the stored events are present.
    neutral = PopulationPrior(beta_means={}, spread_r0={}, source_participants=0)
    assert scores(stripped, ModelVariant.FULL, neutral) == full
    assert scores(rows, ModelVariant.FULL, neutral) == full
    for variant in (
        ModelVariant.PERSON_STATIC,
        ModelVariant.NO_TODAY,
        ModelVariant.NO_PRACTICE_SWITCH,
        ModelVariant.SESSION_ONLY,
    ):
        assert scores(rows, variant) == scores(stripped, variant), variant
        rebuilt = rebuild_engine(rows, variant=variant, preferences=prefs)
        assert rebuilt.snapshot() == rebuild_engine(stripped, variant=variant, preferences=prefs).snapshot()
    static = rebuild_engine(rows, variant=ModelVariant.PERSON_STATIC, preferences=prefs)
    assert static.status().drift_flags == ()
    assert rebuild_engine(rows, preferences=prefs).snapshot() != rebuild_engine(stripped, preferences=prefs).snapshot()
