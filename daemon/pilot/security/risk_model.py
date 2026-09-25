"""Learned Risk Gate — Layers 3/4: Encoder + Transition Model.

Mirrors Ferrum-OS's cognitive/world_model/encoder.rs + transition.rs/
learned.rs, adapted for Heliox: given a real OS-state snapshot
(risk_observation.py) and a proposed Action, predict a couple of
CONCRETE, interpretable outcome fields — how much this action would move
process count and disk usage — rather than an opaque risk scalar. The
prediction is scored for risk separately, by hardcoded rules in
risk_safety.py; nothing in this module decides what counts as dangerous.

Two interchangeable prediction sources behind one signature, exactly like
Ferrum's rule_based_delta/learned::predict_delta split:
  - Rule-based (`_rule_based_outcome`): a small lookup table for the file
    and process-affecting ActionTypes with well-understood effects.
    Always available, zero dependencies beyond numpy.
  - Learned (`RiskTransitionModel`): a small MLP trained on real telemetry
    collected from repeatable, safe-to-run actions in a throwaway sandbox
    (scripts/collect_risk_training_data.py) — used only for the action
    families that data was actually collected for; everything else falls
    back to the rule table, same honest-default philosophy as Ferrum's
    rule_based_delta's exhaustive-but-modest coverage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pilot.actions import ActionType, PermissionTier
from pilot.security.gateway import ActionFamily, action_family
from pilot.security.risk_observation import NOMINAL_PROC_CAPACITY, OsSnapshot

if TYPE_CHECKING:
    from pilot.actions import Action

logger = logging.getLogger("pilot.security.risk_model")

# ── Fixed embedding layout (mirrors encoder.rs's named-slot philosophy —
# every index here is documented and stable so risk_safety.py's rules can
# read exact fields, not guess at a black-box vector's meaning) ──
IDX_PROC_COUNT = 0
IDX_DISK_USAGE = 1
IDX_MEMORY_USAGE = 2
IDX_TIER = 3
IDX_IRREVERSIBLE = 4
IDX_REQUIRES_ROOT = 5
IDX_DANGEROUS_FLAGS = 6
IDX_FAMILY_BASE = 7  # 4 contiguous one-hot slots: shell/browsing/system_control/other

FAMILY_ORDER = [ActionFamily.SHELL, ActionFamily.BROWSING, ActionFamily.SYSTEM_CONTROL, ActionFamily.OTHER]

# Action families the rule table (and, once trained, the learned model)
# actually have well-understood, repeatable effects for. Mirrors
# transition.rs's honest "no predicted change" default for every action it
# doesn't explicitly model.
_DISK_DELTA_RULES: dict[ActionType, float] = {
    ActionType.FILE_WRITE: 0.02,
    ActionType.DOWNLOAD_FILE: 0.03,
    ActionType.FILE_COPY: 0.01,
    ActionType.FILE_DELETE: -0.01,
}

_PROC_DELTA_RULES: dict[ActionType, float] = {
    ActionType.SHELL_COMMAND: 1.0,
    ActionType.SHELL_SCRIPT: 1.0,
    ActionType.PTY_EXEC: 1.0,
    ActionType.CODE_EXECUTE: 1.0,
    ActionType.OPEN_APPLICATION: 1.0,
    ActionType.SERVICE_START: 1.0,
    ActionType.SERVICE_STOP: -1.0,
    ActionType.PROCESS_KILL: -1.0,
}

# Modeled action types the learned transition model (when trained) applies
# to — a subset of the rule table's keys, exactly the ones real sandboxed
# telemetry was collected for. See collect_risk_training_data.py.
LEARNABLE_ACTION_TYPES = frozenset(_DISK_DELTA_RULES) | frozenset(_PROC_DELTA_RULES)
LEARNABLE_ACTION_TYPE_ORDER = tuple(sorted(LEARNABLE_ACTION_TYPES, key=lambda action_type: action_type.value))
IDX_ACTION_TYPE_BASE = IDX_FAMILY_BASE + len(FAMILY_ORDER)
EMBEDDING_SIZE = IDX_ACTION_TYPE_BASE + len(LEARNABLE_ACTION_TYPE_ORDER)

MODEL_VERSION = "risk-mlp-v4-heldout"


@dataclass(frozen=True)
class PredictedOutcome:
    """Concrete, interpretable prediction — never an opaque score. See
    risk_safety.py for how these fields turn into an actual risk verdict."""

    disk_usage_after: float
    proc_count_delta_normalized: float
    source: str  # "learned_calibrated" | "learned" | "rule"
    confidence: float = 1.0


def encode(snapshot: OsSnapshot, action: Action) -> np.ndarray:
    """Encodes (OS state, proposed action) into the fixed embedding both
    the rule-based and learned transition paths read."""
    v = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
    v[IDX_PROC_COUNT] = snapshot.proc_count_normalized
    v[IDX_DISK_USAGE] = snapshot.disk_usage_fraction
    v[IDX_MEMORY_USAGE] = snapshot.memory_usage_fraction
    v[IDX_TIER] = float(action.permission_tier) / float(PermissionTier.ROOT_CRITICAL)
    v[IDX_IRREVERSIBLE] = 1.0 if action.is_irreversible else 0.0
    v[IDX_REQUIRES_ROOT] = 1.0 if action.requires_root else 0.0
    v[IDX_DANGEROUS_FLAGS] = min(1.0, len(action.dangerous_flags) / 3.0)

    family = action_family(action.action_type)
    if family in FAMILY_ORDER:
        v[IDX_FAMILY_BASE + FAMILY_ORDER.index(family)] = 1.0

    if action.action_type in LEARNABLE_ACTION_TYPE_ORDER:
        v[IDX_ACTION_TYPE_BASE + LEARNABLE_ACTION_TYPE_ORDER.index(action.action_type)] = 1.0

    return v


def _rule_based_outcome(snapshot: OsSnapshot, action: Action) -> PredictedOutcome:
    """Honest default: every ActionType not in the two rule tables above
    predicts NO change, exactly like transition.rs's rule_based_delta —
    Phase 1 doesn't try to model low-consequence/unrecognized actions."""
    disk_delta = _DISK_DELTA_RULES.get(action.action_type, 0.0)
    proc_delta = _PROC_DELTA_RULES.get(action.action_type, 0.0) / NOMINAL_PROC_CAPACITY
    return PredictedOutcome(
        disk_usage_after=min(1.0, max(0.0, snapshot.disk_usage_fraction + disk_delta)),
        proc_count_delta_normalized=proc_delta,
        source="rule",
        confidence=1.0,
    )


class RiskTransitionModel:
    """Loads an optional learned MLP (a few hundred bytes, pure numpy) that
    refines the rule table's predictions for LEARNABLE_ACTION_TYPES, trained
    on real sandboxed telemetry (see collect_risk_training_data.py /
    train_risk_gate.py). Falls back to the rule table for every other
    action type, and for ALL action types if no weights are staged —
    strictly additive and optional, same as Ferrum's learned.rs."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._loaded = False
        self._w1 = self._b1 = self._w2 = self._b2 = None
        self._model_version = "rule-fallback"
        self._training_samples = 0
        self._validation_samples = 0
        self._validation_mae: np.ndarray | None = None
        self._baseline_mae: np.ndarray | None = None
        self._action_medians: np.ndarray | None = None
        self._calibration_alpha: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_scale: np.ndarray | None = None
        self._weights_path = weights_path or _default_weights_path()
        self._try_load()

    def _try_load(self) -> None:
        try:
            data = np.load(self._weights_path)
            w1, b1, w2, b2 = data["w1"], data["b1"], data["w2"], data["b2"]
            if w1.shape[0] != EMBEDDING_SIZE or w2.shape[1] != 2:
                logger.warning(
                    "Risk gate weights at %s have unexpected shape (input=%d output=%d), ignoring",
                    self._weights_path,
                    w1.shape[0],
                    w2.shape[1],
                )
                return
            self._w1, self._b1, self._w2, self._b2 = w1, b1, w2, b2
            self._model_version = (
                str(data["model_version"].item()) if "model_version" in data.files else "legacy-learned-model"
            )
            self._training_samples = int(data["training_samples"].item()) if "training_samples" in data.files else 0
            self._validation_samples = (
                int(data["validation_samples"].item()) if "validation_samples" in data.files else 0
            )
            self._validation_mae = data["validation_mae"] if "validation_mae" in data.files else None
            self._baseline_mae = data["baseline_mae"] if "baseline_mae" in data.files else None
            self._action_medians = data["action_medians"] if "action_medians" in data.files else None
            self._calibration_alpha = data["calibration_alpha"] if "calibration_alpha" in data.files else None
            self._feature_mean = data["feature_mean"] if "feature_mean" in data.files else None
            self._feature_scale = data["feature_scale"] if "feature_scale" in data.files else None

            expected_calibration_shape = (len(LEARNABLE_ACTION_TYPE_ORDER), 2)
            calibration_shapes_valid = (
                self._action_medians is not None
                and self._action_medians.shape == expected_calibration_shape
                and self._calibration_alpha is not None
                and self._calibration_alpha.shape == expected_calibration_shape
            )
            if not calibration_shapes_valid:
                self._action_medians = None
                self._calibration_alpha = None
            self._loaded = True
            logger.info("Loaded learned risk transition model from %s", self._weights_path)
        except FileNotFoundError:
            logger.debug("No risk gate weights staged at %s — rule-based fallback only", self._weights_path)
        except Exception:
            logger.warning("Failed to load risk gate weights at %s", self._weights_path, exc_info=True)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def training_samples(self) -> int:
        return self._training_samples

    @property
    def validation_samples(self) -> int:
        return self._validation_samples

    @property
    def is_calibrated(self) -> bool:
        return self._action_medians is not None and self._calibration_alpha is not None

    @property
    def validation_mae(self) -> tuple[float, float] | None:
        if self._validation_mae is None or self._validation_mae.shape != (2,):
            return None
        return float(self._validation_mae[0]), float(self._validation_mae[1])

    @property
    def baseline_mae(self) -> tuple[float, float] | None:
        if self._baseline_mae is None or self._baseline_mae.shape != (2,):
            return None
        return float(self._baseline_mae[0]), float(self._baseline_mae[1])

    def _prediction_confidence(self, x: np.ndarray) -> float:
        """Report empirical support without treating a prediction as certain."""
        quality = 0.5
        validation_mae = self.validation_mae
        baseline_mae = self.baseline_mae
        if validation_mae and baseline_mae:
            relative_quality = [
                1.0 - min(1.0, error / max(baseline, 1e-12))
                for error, baseline in zip(validation_mae, baseline_mae, strict=True)
            ]
            quality = float(np.mean(relative_quality))

        support = 1.0
        if (
            self._feature_mean is not None
            and self._feature_scale is not None
            and self._feature_mean.shape == (3,)
            and self._feature_scale.shape == (3,)
        ):
            z = np.abs((x[:3] - self._feature_mean) / np.maximum(self._feature_scale, 1e-4))
            excess = float(np.mean(np.maximum(0.0, z - 3.0)))
            support = float(np.exp(-0.2 * excess))

        return float(np.clip(quality * support, 0.1, 0.99))

    def predict_rule(self, snapshot: OsSnapshot, action: Action) -> PredictedOutcome:
        """Return the deterministic baseline even when learned weights exist.

        The gate scores this in parallel with learned output, so a noisy model
        can add caution but can never erase a stronger deterministic warning.
        """
        return _rule_based_outcome(snapshot, action)

    def predict(self, snapshot: OsSnapshot, action: Action) -> PredictedOutcome:
        if not self._loaded or action.action_type not in LEARNABLE_ACTION_TYPES:
            return _rule_based_outcome(snapshot, action)

        x = encode(snapshot, action)
        hidden = np.tanh(x @ self._w1 + self._b1)
        out = hidden @ self._w2 + self._b2  # [disk_delta, proc_delta_normalized]
        source = "learned"

        if self.is_calibrated:
            action_index = LEARNABLE_ACTION_TYPE_ORDER.index(action.action_type)
            alpha = self._calibration_alpha[action_index]
            median = self._action_medians[action_index]
            out = alpha * out + (1.0 - alpha) * median
            source = "learned_calibrated"

        disk_after = float(np.clip(snapshot.disk_usage_fraction + out[0], 0.0, 1.0))
        proc_delta = float(out[1])
        return PredictedOutcome(
            disk_usage_after=disk_after,
            proc_count_delta_normalized=proc_delta,
            source=source,
            confidence=self._prediction_confidence(x),
        )


def _default_weights_path() -> str:
    from pathlib import Path

    return str(Path(__file__).parent / "risk_gate_weights.npz")
