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
from pilot.security.risk_observation import OsSnapshot

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
EMBEDDING_SIZE = IDX_FAMILY_BASE + len(FAMILY_ORDER)

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


@dataclass(frozen=True)
class PredictedOutcome:
    """Concrete, interpretable prediction — never an opaque score. See
    risk_safety.py for how these fields turn into an actual risk verdict."""

    disk_usage_after: float
    proc_count_delta_normalized: float
    source: str  # "learned" | "rule" — for audit/debugging, not a risk signal itself


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

    return v


def _rule_based_outcome(snapshot: OsSnapshot, action: Action) -> PredictedOutcome:
    """Honest default: every ActionType not in the two rule tables above
    predicts NO change, exactly like transition.rs's rule_based_delta —
    Phase 1 doesn't try to model low-consequence/unrecognized actions."""
    disk_delta = _DISK_DELTA_RULES.get(action.action_type, 0.0)
    proc_delta = _PROC_DELTA_RULES.get(action.action_type, 0.0)
    return PredictedOutcome(
        disk_usage_after=min(1.0, max(0.0, snapshot.disk_usage_fraction + disk_delta)),
        proc_count_delta_normalized=proc_delta,
        source="rule",
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
            self._loaded = True
            logger.info("Loaded learned risk transition model from %s", self._weights_path)
        except FileNotFoundError:
            logger.debug("No risk gate weights staged at %s — rule-based fallback only", self._weights_path)
        except Exception:
            logger.warning("Failed to load risk gate weights at %s", self._weights_path, exc_info=True)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, snapshot: OsSnapshot, action: Action) -> PredictedOutcome:
        if not self._loaded or action.action_type not in LEARNABLE_ACTION_TYPES:
            return _rule_based_outcome(snapshot, action)

        x = encode(snapshot, action)
        hidden = np.tanh(x @ self._w1 + self._b1)
        out = hidden @ self._w2 + self._b2  # [disk_delta, proc_delta_normalized]

        disk_after = float(np.clip(snapshot.disk_usage_fraction + out[0], 0.0, 1.0))
        proc_delta = float(out[1])
        return PredictedOutcome(disk_usage_after=disk_after, proc_count_delta_normalized=proc_delta, source="learned")


def _default_weights_path() -> str:
    from pathlib import Path

    return str(Path(__file__).parent / "risk_gate_weights.npz")
