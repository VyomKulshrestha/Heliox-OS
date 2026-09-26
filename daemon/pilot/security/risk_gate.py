"""Learned Risk Gate — ties Layers 1/3/4/5 together for a whole plan.

Mirrors Ferrum-OS's cognitive/world_model/mod.rs's evaluate_action(), which
composes the encoder/transition/safety layers for one proposed action.
Heliox's RiskGate.evaluate_plan() captures one OS snapshot, runs every
action through both learned and deterministic transitions, scores the
riskier result, and advances a conservative simulated state. This preserves
the "one bad action anywhere in the plan is enough" rule while also catching
resource impact that compounds across actions already present in the plan.

Strictly additive: if no weights are staged, evaluate_plan() still runs
using deterministic transitions. When a learned model is staged, a learned
prediction can add caution but cannot replace a stronger rule prediction.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable

from pilot.intelligence.world_model import (
    HistoricalFailureRisk,
    HybridWorldModel,
    WorldState,
)
from pilot.security.risk_model import EMBEDDING_SIZE, LEARNABLE_ACTION_TYPES, RiskTransitionModel
from pilot.security.risk_observation import NOMINAL_PROC_CAPACITY, capture_os_snapshot
from pilot.security.risk_safety import score_outcome

if TYPE_CHECKING:
    from pilot.actions import ActionPlan
    from pilot.config import PilotConfig
    from pilot.intelligence.online_learning import VerifiedOnlineLearner


class RiskGate:
    """Owns the (optional) learned transition model and evaluates whole
    plans against it. One instance is enough for the daemon's lifetime —
    see get_risk_gate() below; construct your own only in tests."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._transition = RiskTransitionModel(weights_path)
        self._world_model = HybridWorldModel()
        self._history = HistoricalFailureRisk()
        self._online_learner: VerifiedOnlineLearner | None = None
        self._ui_observation_provider: Callable[[], dict[str, Any] | None] | None = None
        self._last_evaluation: dict[str, object] | None = None

    def set_online_learner(self, learner: VerifiedOnlineLearner) -> None:
        """Attach verified outcome evidence without changing policy authority."""
        self._online_learner = learner

    def set_ui_observation_provider(
        self, provider: Callable[[], dict[str, Any] | None] | None,
    ) -> None:
        """Use an existing fresh screen observation; never capture in this gate."""
        self._ui_observation_provider = provider

    def record_outcome(self, action_type: object, success: bool) -> None:
        """Feed only observed execution outcomes into historical caution."""
        self._history.record(action_type, success)

    @property
    def available(self) -> bool:
        """True once a learned transition model is actually staged — the
        rule-based fallback runs regardless, so this only reports whether
        predictions for LEARNABLE_ACTION_TYPES come from real training or
        the honest rule-table default."""
        return self._transition.is_loaded

    @property
    def last_evaluation(self) -> dict[str, object] | None:
        """Return a defensive copy of the latest plan assessment."""
        return dict(self._last_evaluation) if self._last_evaluation is not None else None

    def status(self, enabled: bool) -> dict[str, object]:
        """Return user-safe runtime/model metadata for Settings."""
        validation_mae = self._transition.validation_mae
        baseline_mae = self._transition.baseline_mae
        return {
            "enabled": enabled,
            "weights_loaded": self.available,
            "model_version": self._transition.model_version,
            "training_samples": self._transition.training_samples,
            "validation_samples": self._transition.validation_samples,
            "calibrated": self._transition.is_calibrated,
            "validation_mae": (
                {"disk_delta": validation_mae[0], "process_delta": validation_mae[1]} if validation_mae else None
            ),
            "baseline_mae": (
                {"disk_delta": baseline_mae[0], "process_delta": baseline_mae[1]} if baseline_mae else None
            ),
            "embedding_size": EMBEDDING_SIZE,
            "learnable_action_types": sorted(action_type.value for action_type in LEARNABLE_ACTION_TYPES),
            "prediction_contract": self._world_model.status(),
            "historical_risk": self._history.status(),
            "verified_online_transition": (
                self._online_learner.status()["transitions"] if self._online_learner is not None else None
            ),
            "last_evaluation": self._last_evaluation,
        }

    def evaluate_plan(self, plan: ActionPlan, config: PilotConfig, *, source: str = "") -> tuple[float, list[str]]:
        """Returns (worst risk in [0,1] seen across the plan's actions,
        the reasons that fired for that worst action — empty if none)."""
        if not plan.actions:
            self._last_evaluation = {
                "evaluated_at": datetime.now(UTC).isoformat(),
                "action_count": 0,
                "risk_score": 0.0,
                "reasons": [],
                "worst_action_type": None,
                "prediction_sources": [],
                "prediction_confidence": None,
                "predictions": [],
            }
            return 0.0, []

        # One snapshot for the whole plan: this runs before execution
        # (predicting what WOULD happen), not interleaved with it, so OS
        # state isn't expected to shift meaningfully action-to-action —
        # and psutil calls aren't free, so one call beats N.
        snapshot = capture_os_snapshot()
        ui_observation: dict[str, Any] = {}
        if self._ui_observation_provider is not None:
            try:
                ui_observation = self._ui_observation_provider() or {}
            except Exception:
                # Screen availability must not decide whether policy runs.
                ui_observation = {}

        simulated = snapshot
        cumulative_proc_delta = 0.0
        worst_risk = 0.0
        worst_reasons: list[str] = []
        worst_action_type: str | None = None
        worst_sources: list[str] = []
        worst_prediction_confidence: float | None = None
        world_predictions: list[dict[str, object]] = []
        rollout_ui = ui_observation
        rollout_browser: dict[str, Any] = {}
        for step_index, action in enumerate(plan.actions):
            contract_prediction = self._world_model.predict(
                WorldState.from_os_snapshot(simulated, ui=rollout_ui, browser=rollout_browser),
                action,
            )
            rollout_ui = dict(contract_prediction.predicted_state.ui)
            rollout_browser = dict(contract_prediction.predicted_state.browser)
            learned_or_rule = self._transition.predict(simulated, action)
            deterministic = self._transition.predict_rule(simulated, action)
            candidates = [learned_or_rule]
            if learned_or_rule.source != "rule":
                candidates.append(deterministic)

            step_proc_delta = max(
                (outcome.proc_count_delta_normalized for outcome in candidates),
                key=abs,
            )
            cumulative_proc_delta += step_proc_delta

            action_risk = 0.0
            action_reasons: list[str] = []
            for outcome in candidates:
                cumulative_outcome = replace(outcome, proc_count_delta_normalized=cumulative_proc_delta)
                risk, reasons = score_outcome(
                    action,
                    cumulative_outcome,
                    config,
                    baseline_disk_usage=simulated.disk_usage_fraction,
                )
                if risk > action_risk or not action_reasons:
                    action_risk = risk
                    action_reasons = reasons
                if risk > worst_risk or worst_action_type is None:
                    worst_risk = risk
                    worst_reasons = reasons
                    worst_action_type = action.action_type.value
                    worst_sources = sorted({candidate.source for candidate in candidates})
                    worst_prediction_confidence = (
                        learned_or_rule.confidence if learned_or_rule.source != "rule" else None
                    )

            transition_risk = action_risk
            transition_reasons = list(action_reasons)
            historical_risk, historical_reason = self._history.score(action.action_type)
            if historical_risk > action_risk:
                action_risk = historical_risk
                action_reasons = [historical_reason]
            if historical_risk > worst_risk:
                worst_risk = historical_risk
                worst_reasons = [historical_reason]
                worst_action_type = action.action_type.value
                worst_sources = ["execution_history"]
                worst_prediction_confidence = min(
                    1.0,
                    self._history.status()["samples"] / 10.0,
                )

            online_risk = 0.0
            online_reason = ""
            if self._online_learner is not None and source:
                adaptation = self._online_learner.score_transition(
                    action_type=action.action_type.value,
                    source=source,
                )
                if adaptation.state == "promoted" and adaptation.probability < 0.42:
                    online_risk = min(0.7, max(0.0, (0.5 - adaptation.probability) * 2.0))
                    online_reason = (
                        f"{adaptation.labels} independently verified {action.action_type.value} "
                        "outcomes suggest elevated failure risk"
                    )
            if online_risk > action_risk:
                action_risk = online_risk
                action_reasons = [online_reason]
            if online_risk > worst_risk:
                worst_risk = online_risk
                worst_reasons = [online_reason]
                worst_action_type = action.action_type.value
                worst_sources = ["verified_online_transition"]
                worst_prediction_confidence = None

            contract_payload = contract_prediction.to_dict()
            contract_payload["input_state_source"] = "observed" if step_index == 0 else "predicted"
            contract_payload["sources"] = list(
                dict.fromkeys(
                    [
                        *contract_payload["sources"],
                        *(candidate.source for candidate in candidates),
                    ]
                )
            )
            contract_payload["risk_score"] = action_risk
            if historical_risk > 0:
                contract_payload["sources"] = list(dict.fromkeys([*contract_payload["sources"], "execution_history"]))
                contract_payload["risk_evidence"].append(
                    {
                        "source": "execution_history",
                        "score": historical_risk,
                        "reason": historical_reason,
                        "deterministic": False,
                    }
                )
            if online_risk > 0:
                contract_payload["sources"] = list(
                    dict.fromkeys([*contract_payload["sources"], "verified_online_transition"])
                )
                contract_payload["risk_evidence"].append(
                    {
                        "source": "verified_online_transition",
                        "score": online_risk,
                        "reason": online_reason,
                        "deterministic": False,
                    }
                )
            contract_payload["risk_evidence"].extend(
                {
                    "source": "risk_transition",
                    "score": transition_risk,
                    "reason": reason,
                    "deterministic": "rule" in contract_payload["sources"],
                }
                for reason in transition_reasons
            )
            world_predictions.append(contract_payload)

            simulated = replace(
                simulated,
                disk_usage_fraction=max(outcome.disk_usage_after for outcome in candidates),
                proc_count=max(
                    0,
                    round(snapshot.proc_count + cumulative_proc_delta * NOMINAL_PROC_CAPACITY),
                ),
            )

        self._last_evaluation = {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "action_count": len(plan.actions),
            "risk_score": worst_risk,
            "reasons": worst_reasons,
            "worst_action_type": worst_action_type,
            "prediction_sources": worst_sources,
            "prediction_confidence": worst_prediction_confidence,
            "predictions": world_predictions,
        }
        return worst_risk, worst_reasons


_gate: RiskGate | None = None


def get_risk_gate() -> RiskGate:
    """Lazily-constructed process-wide singleton — the learned weights (if
    any) only need loading once per daemon lifetime."""
    global _gate
    if _gate is None:
        _gate = RiskGate()
    return _gate
