"""Unified, inspectable world-model prediction contract.

The structured predictor describes likely state transitions before execution.
It is advisory: validators, deterministic policy, approval, and verification
remain the authority for whether an action may run and whether it succeeded.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from pilot.actions import ActionType, MouseParams

if TYPE_CHECKING:
    from pilot.actions import Action
    from pilot.security.risk_observation import OsSnapshot


class EffectDomain(StrEnum):
    BROWSER = "browser"
    UI = "ui"
    WINDOW = "window"
    PROCESS = "process"
    FILESYSTEM = "filesystem"
    SERVICE = "service"
    PACKAGE = "package"
    SYSTEM = "system"
    APPROVAL = "approval"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ExpectedEffect:
    domain: EffectDomain
    operation: str
    target: str
    before: Any = None
    after: Any = None
    confidence: float = 1.0
    reversible: bool = True
    requires_observation: bool = True
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class RiskEvidence:
    source: str
    score: float
    reason: str
    deterministic: bool = False


@dataclass(frozen=True, slots=True)
class WorldState:
    observed_at: str
    system: dict[str, Any] = field(default_factory=dict)
    ui: dict[str, Any] = field(default_factory=dict)
    browser: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> WorldState:
        return cls(observed_at=datetime.now(UTC).isoformat())

    @classmethod
    def from_os_snapshot(
        cls,
        snapshot: OsSnapshot,
        *,
        ui: dict[str, Any] | None = None,
        browser: dict[str, Any] | None = None,
    ) -> WorldState:
        return cls(
            observed_at=datetime.now(UTC).isoformat(),
            system={
                "process_count": snapshot.proc_count,
                "disk_usage_fraction": snapshot.disk_usage_fraction,
                "memory_usage_fraction": snapshot.memory_usage_fraction,
            },
            ui=dict(ui or {}),
            browser=dict(browser or {}),
        )


@dataclass(frozen=True, slots=True)
class WorldPrediction:
    action_type: str
    predicted_state: WorldState
    expected_effects: tuple[ExpectedEffect, ...]
    uncertainty: float
    risk_evidence: tuple[RiskEvidence, ...]
    sources: tuple[str, ...]
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        predicted_state = asdict(self.predicted_state)
        # The visual latent can encode private on-screen content. It is an
        # internal model input, not part of the user-facing prediction trace.
        predicted_state["ui"].pop("latent_embedding", None)
        return {
            "action_type": self.action_type,
            "predicted_state": predicted_state,
            "expected_effects": [{**asdict(effect), "domain": effect.domain.value} for effect in self.expected_effects],
            "uncertainty": self.uncertainty,
            "risk_evidence": [asdict(item) for item in self.risk_evidence],
            "sources": list(self.sources),
            "model_version": self.model_version,
        }


class WorldPredictor(Protocol):
    def predict(
        self,
        current_state: WorldState,
        candidate_action: Action,
    ) -> WorldPrediction: ...


class OptionalWorldPredictor(Protocol):
    def predict_optional(
        self,
        current_state: WorldState,
        candidate_action: Action,
    ) -> WorldPrediction | None: ...


_BROWSER_NAVIGATION = {
    ActionType.BROWSER_NAVIGATE,
    ActionType.OPEN_URL,
    ActionType.BROWSER_BACK,
    ActionType.BROWSER_FORWARD,
    ActionType.BROWSER_REFRESH,
}
_BROWSER_MUTATION = {
    ActionType.BROWSER_CLICK,
    ActionType.BROWSER_CLICK_TEXT,
    ActionType.BROWSER_TYPE,
    ActionType.BROWSER_SELECT,
    ActionType.BROWSER_FILL_FORM,
    ActionType.BROWSER_EXECUTE_JS,
}
_WINDOW_ACTIONS = {
    ActionType.WINDOW_FOCUS,
    ActionType.WINDOW_CLOSE,
    ActionType.WINDOW_MINIMIZE,
    ActionType.WINDOW_MAXIMIZE,
}
_PROCESS_START = {
    ActionType.OPEN_APPLICATION,
    ActionType.SERVICE_START,
}
_TRANSIENT_PROCESS_ACTIONS = {
    ActionType.SHELL_COMMAND,
    ActionType.SHELL_SCRIPT,
    ActionType.CODE_EXECUTE,
}
_PROCESS_STOP = {
    ActionType.PROCESS_KILL,
    ActionType.SERVICE_STOP,
}
_FILE_ACTIONS = {
    ActionType.FILE_WRITE,
    ActionType.FILE_DELETE,
    ActionType.FILE_MOVE,
    ActionType.FILE_COPY,
    ActionType.FILE_PERMISSIONS,
    ActionType.DOWNLOAD_FILE,
}
_SERVICE_ACTIONS = {
    ActionType.SERVICE_START,
    ActionType.SERVICE_STOP,
    ActionType.SERVICE_RESTART,
    ActionType.SERVICE_ENABLE,
    ActionType.SERVICE_DISABLE,
}
_PACKAGE_ACTIONS = {
    ActionType.PACKAGE_INSTALL,
    ActionType.PACKAGE_REMOVE,
    ActionType.PACKAGE_UPDATE,
}


class StructuredTransitionPredictor:
    """Predict first-party OS/UI effects without executing the action."""

    model_version = "structured-transition-v1"

    def predict(
        self,
        current_state: WorldState,
        candidate_action: Action,
    ) -> WorldPrediction:
        action_type = candidate_action.action_type
        params = candidate_action.parameters.model_dump()
        predicted = WorldState(
            observed_at=current_state.observed_at,
            system=deepcopy(current_state.system),
            ui=deepcopy(current_state.ui),
            browser=deepcopy(current_state.browser),
        )
        # Rules can describe UI effects but cannot synthesize a next frame.
        predicted.ui.pop("latent_embedding", None)
        predicted.ui.pop("latent_source", None)
        predicted.ui.pop("latent_transition", None)
        effects: list[ExpectedEffect] = []
        uncertainty = self._uncertainty(action_type)

        if action_type in _BROWSER_NAVIGATION:
            target = self._safe_target(candidate_action, params)
            before_url = current_state.browser.get("url")
            after_url = target if action_type in {ActionType.BROWSER_NAVIGATE, ActionType.OPEN_URL} else "changed"
            predicted.browser["url"] = after_url
            predicted.browser["dom_state"] = "requires_observation"
            effects.extend(
                (
                    ExpectedEffect(
                        EffectDomain.BROWSER,
                        "navigate",
                        target or action_type.value,
                        before=before_url,
                        after=after_url,
                        confidence=0.9 if target else 0.65,
                        evidence="navigation action contract",
                    ),
                    ExpectedEffect(
                        EffectDomain.UI,
                        "replace_accessibility_tree",
                        "active browser document",
                        before=current_state.browser.get("dom_state"),
                        after="requires_observation",
                        confidence=0.85,
                        evidence="navigation invalidates prior DOM and accessibility state",
                    ),
                )
            )
        elif action_type in _BROWSER_MUTATION:
            target = self._safe_target(candidate_action, params)
            predicted.browser["dom_state"] = "possibly_changed"
            effects.append(
                ExpectedEffect(
                    EffectDomain.BROWSER,
                    "mutate_document",
                    target or action_type.value,
                    after="possibly_changed",
                    confidence=0.65,
                    reversible=action_type != ActionType.BROWSER_EXECUTE_JS,
                    evidence="interactive browser actions can mutate DOM, accessibility, or navigation state",
                )
            )
        elif action_type in _WINDOW_ACTIONS or action_type == ActionType.OPEN_APPLICATION:
            target = self._safe_target(candidate_action, params)
            operation = action_type.value.removeprefix("window_").removeprefix("open_")
            predicted.ui["active_window"] = target if operation in {"focus", "application"} else "changed"
            effects.append(
                ExpectedEffect(
                    EffectDomain.WINDOW,
                    operation,
                    target or "selected window",
                    before=current_state.ui.get("active_window"),
                    after=predicted.ui["active_window"],
                    confidence=0.85,
                    evidence="window-manager action contract",
                )
            )

        process_delta = int(action_type in _PROCESS_START) - int(action_type in _PROCESS_STOP)
        if process_delta:
            before_count = current_state.system.get("process_count")
            after_count = before_count + process_delta if isinstance(before_count, int) else "changed"
            predicted.system["process_count"] = after_count
            effects.append(
                ExpectedEffect(
                    EffectDomain.PROCESS,
                    "start" if process_delta > 0 else "stop",
                    self._safe_target(candidate_action, params) or action_type.value,
                    before=before_count,
                    after=after_count,
                    confidence=0.7,
                    reversible=action_type not in {ActionType.PROCESS_KILL, ActionType.SHELL_SCRIPT},
                    evidence="process lifecycle action contract",
                )
            )

        if action_type in _TRANSIENT_PROCESS_ACTIONS:
            count = current_state.system.get("process_count")
            predicted.system["process_count"] = "requires_observation"
            effects.append(
                ExpectedEffect(
                    EffectDomain.PROCESS,
                    "run_transient",
                    action_type.value,
                    before=count,
                    after="requires_observation",
                    confidence=0.5,
                    evidence="handler waits for its direct child; background side effects require observation",
                )
            )

        if action_type in _FILE_ACTIONS:
            path = self._safe_target(candidate_action, params)
            destination = str(params.get("destination") or "").strip()[:300]
            if action_type == ActionType.DOWNLOAD_FILE:
                path = str(params.get("output_path") or "").strip()[:300]
            elif action_type == ActionType.FILE_COPY:
                path = destination
            elif action_type == ActionType.FILE_MOVE:
                source = str(params.get("path") or "").strip()[:300]
                effects.append(
                    ExpectedEffect(
                        EffectDomain.FILESYSTEM,
                        "move",
                        source or "declared source",
                        after="absent",
                        confidence=0.9 if source and destination else 0.35,
                        reversible=candidate_action.reversible,
                        evidence="filesystem move source postcondition",
                    )
                )
                path = destination
            operation = "move_destination" if action_type == ActionType.FILE_MOVE else (
                action_type.value.removeprefix("file_").removeprefix("download_")
            )
            effects.append(
                ExpectedEffect(
                    EffectDomain.FILESYSTEM,
                    operation,
                    path or "declared file target",
                    after=self._file_after_state(action_type, params),
                    confidence=0.9 if path else 0.35,
                    reversible=candidate_action.reversible,
                    evidence="filesystem action contract",
                )
            )

        if action_type in _SERVICE_ACTIONS:
            state = action_type.value.removeprefix("service_")
            effects.append(
                ExpectedEffect(
                    EffectDomain.SERVICE,
                    state,
                    self._safe_target(candidate_action, params) or "declared service",
                    after=state,
                    confidence=0.9,
                    evidence="service-manager action contract",
                )
            )

        if action_type in _PACKAGE_ACTIONS:
            operation = action_type.value.removeprefix("package_")
            effects.append(
                ExpectedEffect(
                    EffectDomain.PACKAGE,
                    operation,
                    self._safe_target(candidate_action, params) or "declared package",
                    after=operation,
                    confidence=0.8,
                    reversible=operation != "remove",
                    evidence="package-manager action contract",
                )
            )

        approval_state = "required" if candidate_action.requires_confirmation else "not_required"
        predicted.system["approval_state"] = approval_state
        effects.append(
            ExpectedEffect(
                EffectDomain.APPROVAL,
                "gate",
                action_type.value,
                after=approval_state,
                confidence=1.0,
                evidence="deterministic permission tier",
            )
        )

        if len(effects) == 1:
            effects.insert(
                0,
                ExpectedEffect(
                    EffectDomain.UNKNOWN,
                    "observe_result",
                    action_type.value,
                    after="requires_observation",
                    confidence=0.35,
                    evidence="no structured transition rule for this action type",
                ),
            )

        return WorldPrediction(
            action_type=action_type.value,
            predicted_state=predicted,
            expected_effects=tuple(effects),
            uncertainty=uncertainty,
            risk_evidence=self._risk_evidence(candidate_action),
            sources=("structured_rules",),
            model_version=self.model_version,
        )

    @staticmethod
    def _safe_target(action: Action, params: dict[str, Any]) -> str:
        if action.action_type in {
            ActionType.SHELL_COMMAND,
            ActionType.SHELL_SCRIPT,
            ActionType.PTY_EXEC,
            ActionType.CODE_EXECUTE,
            ActionType.BROWSER_EXECUTE_JS,
        }:
            return action.action_type.value
        for key in (
            "path",
            "output_path",
            "destination",
            "url",
            "name",
            "title",
            "process_name",
            "value_name",
            "selector",
            "text",
        ):
            value = params.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()[:300]
        return action.target.strip()[:300]

    @staticmethod
    def _file_after_state(action_type: ActionType, params: dict[str, Any]) -> str:
        if action_type == ActionType.FILE_DELETE:
            return "absent"
        if action_type == ActionType.FILE_MOVE:
            return "present_or_updated"
        if action_type in {ActionType.FILE_WRITE, ActionType.FILE_COPY, ActionType.DOWNLOAD_FILE}:
            return "present_or_updated"
        return "metadata_changed"

    @staticmethod
    def _uncertainty(action_type: ActionType) -> float:
        if action_type in {
            ActionType.SHELL_COMMAND,
            ActionType.SHELL_SCRIPT,
            ActionType.PTY_EXEC,
            ActionType.CODE_EXECUTE,
            ActionType.BROWSER_EXECUTE_JS,
        }:
            return 0.85
        if action_type in _BROWSER_MUTATION:
            return 0.45
        if action_type in _BROWSER_NAVIGATION:
            return 0.2
        if action_type in _FILE_ACTIONS | _SERVICE_ACTIONS | _PACKAGE_ACTIONS | _WINDOW_ACTIONS:
            return 0.15
        return 0.35

    @staticmethod
    def _risk_evidence(action: Action) -> tuple[RiskEvidence, ...]:
        evidence: list[RiskEvidence] = []
        if action.requires_root:
            evidence.append(RiskEvidence("permission_tier", 1.0, "Action requests root privileges", True))
        if action.is_irreversible:
            evidence.append(RiskEvidence("reversibility", 0.9, "Action may be irreversible", True))
        if action.dangerous_flags:
            evidence.append(
                RiskEvidence(
                    "validator",
                    min(1.0, 0.55 + 0.1 * len(action.dangerous_flags)),
                    f"Validator reported {len(action.dangerous_flags)} dangerous flag(s)",
                    True,
                )
            )
        return tuple(evidence)


class UiJepaPredictor:
    """Optional action-conditioned UI latent predictor.

    The model predicts embeddings, not pixels. It activates only when staged
    weights and an on-device latent observation are both present. Predictions
    remain shadow-only unless the artifact explicitly records that it passed
    gating validation.
    """

    MIN_GATING_SAMPLES = 100
    MAX_GATING_VALIDATION_ERROR = 0.35
    ACTION_GEOMETRY_VERSION = "screen-point-v1"
    POINTER_ACTIONS = {
        ActionType.MOUSE_CLICK,
        ActionType.MOUSE_DOUBLE_CLICK,
        ActionType.MOUSE_RIGHT_CLICK,
    }

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self._weights_path = Path(weights_path) if weights_path else Path(__file__).with_name("ui_jepa_weights.npz")
        self._state_weight: np.ndarray | None = None
        self._action_weight: np.ndarray | None = None
        self._action_geometry_weight: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._model_version = "ui-jepa-unavailable"
        self._training_samples = 0
        self._validation_error: float | None = None
        self._validated_for_gating = False
        self._action_vocabulary_bound = False
        self._load_error = "weights file not found"
        self._gating_rejection_reasons: tuple[str, ...] = ()
        self._try_load()

    @property
    def is_loaded(self) -> bool:
        return self._state_weight is not None

    @property
    def validated_for_gating(self) -> bool:
        return self._validated_for_gating

    def status(self) -> dict[str, Any]:
        latent_dim = int(self._bias.shape[0]) if self._bias is not None else 0
        return {
            "weights_loaded": self.is_loaded,
            "model_version": self._model_version,
            "training_samples": self._training_samples,
            "validation_error": self._validation_error,
            "validated_for_gating": self._validated_for_gating,
            "action_vocabulary_bound": self._action_vocabulary_bound,
            "action_geometry_bound": self._action_geometry_weight is not None,
            "gating_action_types": (
                sorted(action.value for action in self.POINTER_ACTIONS)
                if self._validated_for_gating else []
            ),
            "latent_dimension": latent_dim,
            "mode": "gating" if self._validated_for_gating else "shadow",
            "load_error": self._load_error,
            "gating_rejection_reasons": list(self._gating_rejection_reasons),
        }

    def _try_load(self) -> None:
        try:
            with np.load(self._weights_path, allow_pickle=False) as data:
                state_weight = data["state_weight"]
                action_weight = data["action_weight"]
                geometry_weight = data["action_geometry_weight"] if "action_geometry_weight" in data.files else None
                bias = data["bias"]
                expected_actions = tuple(action_type.value for action_type in ActionType)
                stored_actions = (
                    tuple(str(value) for value in data["action_types"].tolist()) if "action_types" in data.files else ()
                )
                self._action_vocabulary_bound = stored_actions == expected_actions
                if not self._action_vocabulary_bound:
                    self._load_error = "artifact action vocabulary is missing or does not match this build"
                    return
                if (
                    state_weight.ndim != 2
                    or state_weight.shape[0] != state_weight.shape[1]
                    or bias.shape != (state_weight.shape[0],)
                    or action_weight.shape != (len(expected_actions), state_weight.shape[0])
                    or not np.all(np.isfinite(state_weight))
                    or not np.all(np.isfinite(action_weight))
                    or not np.all(np.isfinite(bias))
                ):
                    self._load_error = "artifact tensor shapes or values are invalid"
                    return
                if geometry_weight is not None:
                    geometry_version = (
                        str(data["action_geometry_version"].item())
                        if "action_geometry_version" in data.files else ""
                    )
                    if (geometry_version != self.ACTION_GEOMETRY_VERSION
                            or geometry_weight.shape != (2, state_weight.shape[0])
                            or not np.all(np.isfinite(geometry_weight))):
                        self._load_error = "artifact action geometry version, shape, or values are invalid"
                        return
                self._state_weight = state_weight.astype(np.float32)
                self._action_weight = action_weight.astype(np.float32)
                self._action_geometry_weight = (
                    geometry_weight.astype(np.float32) if geometry_weight is not None else None
                )
                self._bias = bias.astype(np.float32)
                self._model_version = (
                    str(data["model_version"].item())[:120] if "model_version" in data.files else "ui-jepa-unversioned"
                )
                self._training_samples = int(data["training_samples"].item()) if "training_samples" in data.files else 0
                raw_validation_error = (
                    float(data["validation_error"].item()) if "validation_error" in data.files else None
                )
                self._validation_error = (
                    raw_validation_error
                    if raw_validation_error is not None
                    and np.isfinite(raw_validation_error)
                    and 0.0 <= raw_validation_error <= 1.0
                    else None
                )
                requested_gating = (
                    bool(data["validated_for_gating"].item()) if "validated_for_gating" in data.files else False
                )
                reasons: list[str] = []
                if not requested_gating:
                    reasons.append("artifact is designated for shadow evaluation")
                if self._action_geometry_weight is None:
                    reasons.append("artifact cannot distinguish click locations for gating")
                if self._training_samples < self.MIN_GATING_SAMPLES:
                    reasons.append(f"training sample count is below {self.MIN_GATING_SAMPLES}")
                if self._validation_error is None:
                    reasons.append("validation error is missing or invalid")
                elif self._validation_error > self.MAX_GATING_VALIDATION_ERROR:
                    reasons.append(f"validation error exceeds {self.MAX_GATING_VALIDATION_ERROR:.2f}")
                self._gating_rejection_reasons = tuple(reasons)
                self._validated_for_gating = requested_gating and not reasons
                self._load_error = ""
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            self._load_error = str(exc) or exc.__class__.__name__
            return

    def predict_optional(
        self,
        current_state: WorldState,
        candidate_action: Action,
    ) -> WorldPrediction | None:
        if self._state_weight is None or self._action_weight is None or self._bias is None:
            return None
        latent = current_state.ui.get("latent_embedding")
        if not isinstance(latent, list):
            return None
        state = np.asarray(latent, dtype=np.float32)
        if state.shape != self._bias.shape or not np.all(np.isfinite(state)):
            return None

        action_index = tuple(ActionType).index(candidate_action.action_type)
        action_offset = self._action_weight[action_index].copy()
        geometry_used = False
        if candidate_action.action_type in self.POINTER_ACTIONS and self._action_geometry_weight is not None:
            if not isinstance(candidate_action.parameters, MouseParams) or candidate_action.parameters.relative:
                return None
            # Zero/zero is an executor sentinel for a detected target, not a screen point.
            if (candidate_action.use_previous_output
                    or (candidate_action.action_type == ActionType.MOUSE_CLICK
                        and candidate_action.parameters.x == candidate_action.parameters.y == 0)):
                return None
            bounds = current_state.ui.get("screen_bounds")
            if (not isinstance(bounds, (tuple, list)) or len(bounds) != 4
                    or any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in bounds)
                    or bounds[2] <= 0 or bounds[3] <= 0):
                return None
            x = (candidate_action.parameters.x - bounds[0]) / bounds[2]
            y = (candidate_action.parameters.y - bounds[1]) / bounds[3]
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                return None
            action_offset += np.asarray((x, y), dtype=np.float32) @ self._action_geometry_weight
            geometry_used = True
        predicted_latent = np.tanh(state @ self._state_weight + action_offset + self._bias)
        delta_norm = float(np.linalg.norm(predicted_latent - state))
        state_norm = float(np.linalg.norm(state))
        predicted_norm = float(np.linalg.norm(predicted_latent))
        cosine = float(np.dot(state, predicted_latent) / max(state_norm * predicted_norm, 1e-6))
        validation_error = self._validation_error
        uncertainty = float(
            np.clip(
                validation_error if validation_error is not None else 0.75,
                0.05,
                1.0,
            )
        )
        predicted = WorldState(
            observed_at=current_state.observed_at,
            system=deepcopy(current_state.system),
            ui={
                **deepcopy(current_state.ui),
                "latent_embedding": predicted_latent.tolist(),
                "latent_source": "predicted",
                "latent_transition": {
                    "delta_norm": round(delta_norm, 6),
                    "cosine_similarity": round(cosine, 6),
                    "requires_observation": True,
                },
            },
            browser=deepcopy(current_state.browser),
        )
        risk_evidence: tuple[RiskEvidence, ...] = ()
        if self._validated_for_gating and geometry_used and delta_norm >= 1.0:
            risk_evidence = (
                RiskEvidence(
                    "ui_jepa",
                    min(0.95, 0.5 + delta_norm / 10.0),
                    "Validated latent predictor expects a large UI-state transition",
                    False,
                ),
            )
        return WorldPrediction(
            action_type=candidate_action.action_type.value,
            predicted_state=predicted,
            expected_effects=(
                ExpectedEffect(
                    EffectDomain.UI,
                    "predict_latent_transition",
                    "active visual surface",
                    after=predicted.ui["latent_transition"],
                    confidence=1.0 - uncertainty,
                    evidence="action-conditioned joint-embedding predictor",
                ),
            ),
            uncertainty=uncertainty,
            risk_evidence=risk_evidence,
            sources=("ui_jepa",),
            model_version=self._model_version,
        )


class HybridWorldModel:
    """Fuse structured and optional latent predictions conservatively."""

    def __init__(
        self,
        *,
        structured: WorldPredictor | None = None,
        visual: OptionalWorldPredictor | None = None,
    ) -> None:
        self._structured = structured or StructuredTransitionPredictor()
        self._visual = visual or UiJepaPredictor()

    def status(self) -> dict[str, Any]:
        visual_status = (
            self._visual.status() if hasattr(self._visual, "status") else {"weights_loaded": False, "mode": "shadow"}
        )
        return {
            "contract_version": "world-prediction-v1",
            "structured_model_version": getattr(
                self._structured,
                "model_version",
                "custom",
            ),
            "ui_jepa": visual_status,
        }

    def predict(
        self,
        current_state: WorldState,
        candidate_action: Action,
    ) -> WorldPrediction:
        structured = self._structured.predict(current_state, candidate_action)
        visual = self._visual.predict_optional(current_state, candidate_action)
        if visual is None:
            return structured
        visual_updates = {
            key: deepcopy(value)
            for key, value in visual.predicted_state.ui.items()
            if key not in structured.predicted_state.ui
            or key not in current_state.ui
            or value != current_state.ui[key]
        }
        fused_state = WorldState(
            observed_at=structured.predicted_state.observed_at,
            system=deepcopy(structured.predicted_state.system),
            ui={
                **deepcopy(structured.predicted_state.ui),
                **visual_updates,
            },
            browser=deepcopy(structured.predicted_state.browser),
        )
        return WorldPrediction(
            action_type=structured.action_type,
            predicted_state=fused_state,
            expected_effects=structured.expected_effects + visual.expected_effects,
            uncertainty=max(structured.uncertainty, visual.uncertainty),
            risk_evidence=structured.risk_evidence + visual.risk_evidence,
            sources=tuple(dict.fromkeys(structured.sources + visual.sources)),
            model_version=f"{structured.model_version}+{visual.model_version}",
        )


class HistoricalFailureRisk:
    """Bounded empirical risk from executor callbacks, not postconditions."""

    def __init__(self, *, minimum_samples: int = 3) -> None:
        self._minimum_samples = max(1, minimum_samples)
        self._counts: dict[str, list[int]] = {}

    def record(self, action_type: ActionType | str, success: bool) -> None:
        key = ActionType(action_type).value
        counts = self._counts.setdefault(key, [0, 0])
        counts[0 if success else 1] += 1

    def score(self, action_type: ActionType | str) -> tuple[float, str]:
        key = ActionType(action_type).value
        successes, failures = self._counts.get(key, [0, 0])
        total = successes + failures
        if total < self._minimum_samples:
            return 0.0, ""
        failure_rate = failures / total
        support = min(1.0, total / 10.0)
        score = min(0.85, failure_rate * (0.5 + 0.5 * support))
        if score < 0.2:
            return 0.0, ""
        return (
            score,
            f"{failures} of {total} observed {key} executor callbacks failed",
        )

    def status(self) -> dict[str, Any]:
        return {
            "minimum_samples": self._minimum_samples,
            "action_types_observed": len(self._counts),
            "samples": sum(sum(counts) for counts in self._counts.values()),
        }
