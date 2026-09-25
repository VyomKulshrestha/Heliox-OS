"""Verified online adaptation over the canonical Heliox experience stream.

This module deliberately has no browser, planner, executor, or permission
dependencies.  It learns only from events that Heliox has already observed and
its predictions are advisory ranking signals, never execution authority.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from river import drift, linear_model, optim

from pilot.intelligence.experience import ExperienceEvent, ExperienceEventType, ExperienceLedger

MODEL_VERSION = "verified-online-v2"
MINIMUM_PROMOTION_LABELS = 6
REPLAY_CAPACITY = 256
MAX_ROUTINES = 256
MAX_PENDING_PLANS = 256
MAX_PENDING_ACTIONS = 512
DECAY_HALF_LIFE_DAYS = 30.0


@dataclass(frozen=True, slots=True)
class AdaptationScore:
    """One bounded advisory score with its evidence state."""

    probability: float
    labels: int
    state: str
    source: str = MODEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": round(self.probability, 3),
            "labels": self.labels,
            "state": self.state,
            "source": self.source,
            "authority": "ranking_only",
        }


@dataclass(frozen=True, slots=True)
class _ReplaySample:
    features: dict[str, float]
    label: bool
    occurred_at: str


def _new_classifier() -> Any:
    return linear_model.LogisticRegression(
        optimizer=optim.SGD(0.2),
        intercept_lr=0.2,
    )


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_bucket(value: object, *, fallback: str = "unknown") -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "_", str(value).strip().lower()).strip("_")
    return normalized[:64] or fallback


def _time_features(occurred_at: str) -> dict[str, float]:
    moment = _parse_time(occurred_at)
    hour_angle = 2.0 * math.pi * moment.hour / 24.0
    day_angle = 2.0 * math.pi * moment.weekday() / 7.0
    return {
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "day_sin": math.sin(day_angle),
        "day_cos": math.cos(day_angle),
    }


class _VerifiedClassifier:
    """River classifier with bounded replay and ADWIN drift detection."""

    def __init__(self) -> None:
        self.model = _new_classifier()
        self.detector = drift.ADWIN()
        self.samples: deque[_ReplaySample] = deque(maxlen=REPLAY_CAPACITY)
        self.labels = 0
        self.positive = 0
        self.negative = 0
        self.drift_events = 0

    def learn(self, features: dict[str, float], label: bool, occurred_at: str) -> None:
        probability = float(self.model.predict_proba_one(features).get(True, 0.5))
        self.detector.update(abs(float(label) - probability))
        self.model.learn_one(features, label)
        self.samples.append(_ReplaySample(dict(features), label, occurred_at))
        self.labels += 1
        if label:
            self.positive += 1
        else:
            self.negative += 1
        if self.detector.drift_detected:
            self.drift_events += 1
            self._rebuild_after_drift()

    def score(self, features: dict[str, float]) -> AdaptationScore:
        probability = float(self.model.predict_proba_one(features).get(True, 0.5))
        if self.labels < MINIMUM_PROMOTION_LABELS:
            return AdaptationScore(0.5, self.labels, "candidate")
        support = min(1.0, self.labels / 10.0)
        bounded = 0.5 + (probability - 0.5) * support
        state = "promoted" if abs(bounded - 0.5) >= 0.08 else "candidate"
        return AdaptationScore(max(0.05, min(0.95, bounded)), self.labels, state)

    def reset(self) -> None:
        self.model = _new_classifier()
        self.detector = drift.ADWIN()
        self.samples.clear()
        self.labels = 0
        self.positive = 0
        self.negative = 0

    def _rebuild_after_drift(self) -> None:
        retained = list(self.samples)[-(REPLAY_CAPACITY // 2) :]
        self.model = _new_classifier()
        self.detector = drift.ADWIN()
        self.labels = 0
        self.positive = 0
        self.negative = 0
        self.samples.clear()
        for sample in retained:
            self.model.learn_one(sample.features, sample.label)
            self.samples.append(sample)
            self.labels += 1
            if sample.label:
                self.positive += 1
            else:
                self.negative += 1

    def status(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "positive": self.positive,
            "negative": self.negative,
            "replay_samples": len(self.samples),
            "drift_events": self.drift_events,
            "promotion_threshold": MINIMUM_PROMOTION_LABELS,
        }


class VerifiedOnlineLearner:
    """Consume verified events and expose bounded personalization signals."""

    def __init__(self, state_path: str | Path) -> None:
        self._state_path = Path(state_path)
        self._suggestions = _VerifiedClassifier()
        self._transitions = _VerifiedClassifier()
        self._lock = asyncio.Lock()
        self._seen_sequences: set[int] = set()
        self._suggestion_features: dict[str, dict[str, float]] = {}
        self._action_features: dict[str, dict[str, float]] = {}
        self._plan_actions: dict[str, dict[int, tuple[str, str]]] = {}
        self._completed_actions: dict[str, dict[int, tuple[str, bool]]] = {}
        self._routines: Counter[str] = Counter()
        self._routine_epoch = datetime.now(timezone.utc)
        self._workflow_outcomes: Counter[str] = Counter()
        self._prediction_errors = 0
        self._corrections = 0
        self._explicit_rules = 0
        self._cursor = 0
        self._reset_before_sequence = 0
        self._load_checkpoint()

    async def initialize(self, ledger: ExperienceLedger) -> None:
        """Rebuild from the append-only ledger after the last explicit reset."""

        after = self._reset_before_sequence
        while True:
            events = await ledger.list_events(after_sequence=after, limit=1000)
            if not events:
                break
            for event in events:
                await self.consume(event)
            after = events[-1].sequence
            if len(events) < 1000:
                break

    async def consume(self, event: ExperienceEvent) -> None:
        """Learn from one event exactly once without creating any action."""

        async with self._lock:
            if event.sequence <= self._reset_before_sequence or event.sequence in self._seen_sequences:
                return
            self._seen_sequences.add(event.sequence)
            if len(self._seen_sequences) > 4096:
                floor = max(0, self._cursor - 2048)
                self._seen_sequences = {sequence for sequence in self._seen_sequences if sequence >= floor}
            self._cursor = max(self._cursor, event.sequence)
            self._consume_locked(event)

    def score_suggestion(
        self,
        *,
        pattern_id: str,
        app_name: str,
        priority: str,
        occurred_at: str = "",
    ) -> AdaptationScore:
        features = self._suggestion_vector(
            pattern_id=pattern_id,
            app_name=app_name,
            priority=priority,
            occurred_at=occurred_at or datetime.now(timezone.utc).isoformat(),
        )
        return self._suggestions.score(features)

    def score_transition(
        self,
        *,
        action_type: str,
        source: str = "interactive",
        occurred_at: str = "",
    ) -> AdaptationScore:
        features = self._transition_vector(
            action_type=action_type,
            source=source,
            occurred_at=occurred_at or datetime.now(timezone.utc).isoformat(),
        )
        return self._transitions.score(features)

    def status(self) -> dict[str, Any]:
        routine_items = self._decayed_routines()[:10]
        return {
            "enabled": True,
            "model_version": MODEL_VERSION,
            "backend": "river-0.23.0",
            "authority": "ranking_only",
            "event_cursor": self._cursor,
            "reset_before_sequence": self._reset_before_sequence,
            "suggestions": self._suggestions.status(),
            "transitions": self._transitions.status(),
            "prediction_errors": self._prediction_errors,
            "corrections": self._corrections,
            "explicit_rules": self._explicit_rules,
            "routine_patterns": [
                {"pattern": pattern, "decayed_evidence": round(evidence, 3)} for pattern, evidence in routine_items
            ],
            "workflow_patterns": len(self._workflow_outcomes),
            "privacy": {
                "raw_media_stored": False,
                "secret_browsing": False,
                "external_observation_requires_permission": True,
            },
        }

    async def reset(self) -> dict[str, Any]:
        """Forget trained state while preserving the immutable audit ledger."""

        async with self._lock:
            self._reset_before_sequence = self._cursor
            self._suggestions.reset()
            self._transitions.reset()
            self._seen_sequences.clear()
            self._suggestion_features.clear()
            self._action_features.clear()
            self._plan_actions.clear()
            self._completed_actions.clear()
            self._routines.clear()
            self._routine_epoch = datetime.now(timezone.utc)
            self._workflow_outcomes.clear()
            self._prediction_errors = 0
            self._corrections = 0
            self._explicit_rules = 0
            self._save_checkpoint()
            return self.status()

    def _consume_locked(self, event: ExperienceEvent) -> None:
        event_type = event.event_type
        payload = event.payload
        if event_type == ExperienceEventType.SUGGESTION_SHOWN:
            suggestion_id = str(payload.get("suggestion_id", ""))
            if suggestion_id:
                self._suggestion_features[suggestion_id] = self._suggestion_vector(
                    pattern_id=str(payload.get("pattern_id", "")),
                    app_name=str(payload.get("context_app", "unknown")),
                    priority=str(payload.get("priority", "low")),
                    occurred_at=event.occurred_at,
                )
        elif event_type == ExperienceEventType.SUGGESTION_FEEDBACK:
            suggestion_id = str(payload.get("suggestion_id", ""))
            features = self._suggestion_features.pop(suggestion_id, None)
            if features is None:
                features = self._suggestion_vector(
                    pattern_id=str(payload.get("pattern_id", "")),
                    app_name=str(payload.get("context_app", "unknown")),
                    priority=str(payload.get("priority", "low")),
                    occurred_at=event.occurred_at,
                )
            decision = str(payload.get("decision", "")).lower()
            if decision in {"accepted", "dismissed", "ignored"}:
                self._suggestions.learn(features, decision == "accepted", event.occurred_at)
        elif event_type in {
            ExperienceEventType.CANDIDATE_ACTION,
            ExperienceEventType.ACTION_STARTED,
        }:
            action = payload.get("action")
            action_data = action if isinstance(action, dict) else {}
            action_type = str(action_data.get("action_type") or payload.get("action_type") or "")
            index = payload.get("index")
            if (event.plan_id and event.action_id and action_type
                    and isinstance(index, int) and not isinstance(index, bool) and index >= 0):
                if event.plan_id not in self._plan_actions and len(self._plan_actions) >= MAX_PENDING_PLANS:
                    self._forget_plan(next(iter(self._plan_actions)))
                self._plan_actions.setdefault(event.plan_id, {})[index] = (event.action_id, _safe_bucket(action_type))
                if event_type == ExperienceEventType.ACTION_STARTED:
                    self._action_features[event.action_id] = self._transition_vector(
                        action_type=action_type,
                        source=event.source,
                        occurred_at=event.occurred_at,
                    )
                    if len(self._action_features) > MAX_PENDING_ACTIONS:
                        self._action_features.pop(next(iter(self._action_features)))
        elif event_type == ExperienceEventType.ACTION_COMPLETED:
            if payload.get("callback_observed") is not True or payload.get("executed") is not True:
                return
            if str(payload.get("output_excerpt", "")).startswith("(dry run)"):
                return
            index = payload.get("index")
            if (event.plan_id and event.action_id in self._action_features
                    and isinstance(index, int) and not isinstance(index, bool) and index >= 0):
                planned = self._plan_actions.get(event.plan_id, {}).get(index)
                if planned is not None and planned[0] == event.action_id:
                    self._completed_actions.setdefault(event.plan_id, {})[index] = (
                        event.action_id, payload.get("success") is True,
                    )
        elif event_type == ExperienceEventType.OBSERVATION:
            observation_time = _parse_time(event.occurred_at)
            active_app = _safe_bucket(payload.get("active_app"))
            hour = observation_time.hour
            if observation_time > self._routine_epoch:
                self._decay_routines(observation_time)
                evidence = 1.0
            else:
                age_days = (self._routine_epoch - observation_time).total_seconds() / 86_400.0
                evidence = math.pow(0.5, age_days / DECAY_HALF_LIFE_DAYS)
            self._routines[f"app:{active_app}:hour:{hour // 3}"] += evidence
            if len(self._routines) > MAX_ROUTINES:
                for key, _ in self._routines.most_common()[MAX_ROUTINES:]:
                    del self._routines[key]
        elif event_type == ExperienceEventType.OUTCOME_VERIFIED:
            if event.source == "verifier":
                self._learn_verified_plan(event)
        elif event_type == ExperienceEventType.PREDICTION_ERROR:
            self._prediction_errors += 1
        elif event_type == ExperienceEventType.USER_CORRECTION:
            self._corrections += 1
        elif event_type == ExperienceEventType.MEMORY_PROMOTED:
            if bool(event.provenance.get("user_confirmed", False)):
                self._explicit_rules += 1

    @staticmethod
    def _suggestion_vector(
        *,
        pattern_id: str,
        app_name: str,
        priority: str,
        occurred_at: str,
    ) -> dict[str, float]:
        features = _time_features(occurred_at)
        features[f"pattern={_safe_bucket(pattern_id)}"] = 1.0
        features[f"app={_safe_bucket(app_name)}"] = 1.0
        features[f"priority={_safe_bucket(priority)}"] = 1.0
        return features

    @staticmethod
    def _transition_vector(
        *,
        action_type: str,
        source: str,
        occurred_at: str,
    ) -> dict[str, float]:
        features = _time_features(occurred_at)
        features[f"action={_safe_bucket(action_type)}"] = 1.0
        features[f"source={_safe_bucket(source)}"] = 1.0
        return features

    def _learn_verified_plan(self, event: ExperienceEvent) -> None:
        plan_actions = self._plan_actions.get(event.plan_id, {})
        completed = self._completed_actions.get(event.plan_id, {})
        verification = event.payload.get("verification")
        result_count = event.payload.get("result_count")
        try:
            if (not isinstance(verification, dict) or not isinstance(result_count, int)
                    or isinstance(result_count, bool) or result_count <= 0
                    or set(plan_actions) != set(range(result_count))):
                return
            failed = verification.get("failed_actions")
            observed = verification.get("independently_observed_actions")
            passed = verification.get("passed")
            if (not isinstance(failed, list) or not isinstance(observed, list)
                    or not isinstance(passed, bool)
                    or any(not isinstance(index, int) or isinstance(index, bool)
                           or index not in plan_actions for index in (*failed, *observed))
                    or len(set(failed)) != len(failed) or len(set(observed)) != len(observed)
                    or passed != (not failed)):
                return
            failed_indices = set(failed)
            for index in observed:
                action_id = plan_actions[index][0]
                completion = completed.get(index)
                features = self._action_features.get(action_id)
                if completion is not None and completion[0] == action_id and features is not None:
                    self._transitions.learn(features, completion[1] and index not in failed_indices,
                                            event.occurred_at)
            if len(observed) == result_count:
                signature = ">".join(plan_actions[index][1] for index in range(result_count))
                self._workflow_outcomes[f"{signature}:{'success' if passed else 'failure'}"] += 1
        finally:
            self._forget_plan(event.plan_id)

    def _forget_plan(self, plan_id: str) -> None:
        for action_id, _ in self._plan_actions.pop(plan_id, {}).values():
            self._action_features.pop(action_id, None)
        self._completed_actions.pop(plan_id, None)

    def _decayed_routines(self) -> list[tuple[str, float]]:
        # Routine observations are intentionally coarse. Count decay keeps an
        # old habit from remaining permanently dominant without retaining
        # window titles or browser content.
        age_days = max(
            0.0,
            (datetime.now(timezone.utc) - self._routine_epoch).total_seconds() / 86_400.0,
        )
        decay = math.pow(0.5, age_days / DECAY_HALF_LIFE_DAYS)
        return sorted(
            ((pattern, count * decay) for pattern, count in self._routines.items()),
            key=lambda item: item[1],
            reverse=True,
        )

    def _decay_routines(self, at: datetime) -> None:
        if at <= self._routine_epoch:
            return
        age_days = (at - self._routine_epoch).total_seconds() / 86_400.0
        decay = math.pow(0.5, age_days / DECAY_HALF_LIFE_DAYS)
        if decay < 0.999:
            self._routines = Counter(
                {pattern: evidence * decay for pattern, evidence in self._routines.items() if evidence * decay >= 0.05}
            )
            self._routine_epoch = at

    def _load_checkpoint(self) -> None:
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._reset_before_sequence = max(0, int(raw.get("reset_before_sequence", 0)))
        except (OSError, TypeError, ValueError):
            self._reset_before_sequence = 0

    def _save_checkpoint(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "reset_before_sequence": self._reset_before_sequence,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._state_path)
