"""Stress-Aware Task Gating — Pauses destructive actions under cognitive stress.

Prevents catastrophic mistakes by detecting when the user is in a
high-stress cognitive state (see pilot.cognitive.cognitive_engine) and
gating destructive (Tier 2+) actions.

Behavior:
  - If the cognitive engine predicts stress > STRESS_GATE_THRESHOLD ──→ PAUSE the action
  - Notify the user with a calming confirmation dialog
  - Allow override with explicit confirmation
  - Never gates read-only or information-gathering actions

Integration point: Executor._execute_single() in executor.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pilot.actions import ActionType
from pilot.cognitive.cognitive_engine import CognitiveEngine

logger = logging.getLogger("pilot.cognitive.stress_gate")

# ── Configuration ──

# Stress level threshold to trigger gating (0.0 - 1.0)
STRESS_GATE_THRESHOLD = 0.75

# Cognitive load threshold (compounds with stress)
LOAD_GATE_THRESHOLD = 0.85

# Cool-down: don't re-gate the same action type within N seconds
GATE_COOLDOWN_S = 30.0

# Actions that are NEVER gated (read-only, information-gathering)
SAFE_ACTIONS: set[ActionType] = {
    ActionType.FILE_READ,
    ActionType.FILE_LIST,
    ActionType.FILE_SEARCH,
    ActionType.CLIPBOARD_READ,
    ActionType.SYSTEM_INFO,
    ActionType.DISK_USAGE,
    ActionType.MEMORY_USAGE,
    ActionType.CPU_USAGE,
    ActionType.NETWORK_INFO,
    ActionType.BATTERY_INFO,
    ActionType.PROCESS_LIST,
    ActionType.PROCESS_INFO,
    ActionType.SERVICE_STATUS,
    ActionType.WINDOW_LIST,
    ActionType.VOLUME_GET,
    ActionType.BRIGHTNESS_GET,
    ActionType.WIFI_LIST,
    ActionType.DISK_LIST,
    ActionType.USER_LIST,
    ActionType.USER_INFO,
    ActionType.ENV_GET,
    ActionType.ENV_LIST,
    ActionType.SCREEN_OCR,
    ActionType.SCREEN_ANALYZE,
    ActionType.SCREEN_FIND_TEXT,
    ActionType.SCREEN_ELEMENT_MAP,
    ActionType.BROWSER_EXTRACT,
    ActionType.BROWSER_EXTRACT_TABLE,
    ActionType.BROWSER_EXTRACT_LINKS,
    ActionType.BROWSER_PAGE_INFO,
    ActionType.BROWSER_LIST_TABS,
    ActionType.BROWSER_SCREENSHOT,
    ActionType.SCHEDULE_LIST,
    ActionType.TRIGGER_LIST,
    ActionType.MOUSE_POSITION,
    ActionType.SCREENSHOT,
    ActionType.REGISTRY_READ,
    ActionType.PACKAGE_SEARCH,
    ActionType.GNOME_SETTING_READ,
    ActionType.NOTIFY,
}

# High-risk actions that are ALWAYS gated under stress (even moderate)
HIGH_RISK_ACTIONS: set[ActionType] = {
    ActionType.FILE_DELETE,
    ActionType.POWER_SHUTDOWN,
    ActionType.POWER_RESTART,
    ActionType.POWER_LOGOUT,
    ActionType.PROCESS_KILL,
    ActionType.PACKAGE_REMOVE,
    ActionType.SERVICE_STOP,
    ActionType.SERVICE_DISABLE,
    ActionType.DISK_UNMOUNT,
    ActionType.REGISTRY_WRITE,
}

# Lower threshold for high-risk actions
HIGH_RISK_STRESS_THRESHOLD = 0.55


@dataclass
class GateDecision:
    """Result of the stress gate evaluation."""

    action_type: ActionType
    gated: bool = False  # True = action should be paused
    stress_level: float = 0.0
    cognitive_load: float = 0.0
    reason: str = ""
    suggestion: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "gated": self.gated,
            "stress_level": round(self.stress_level, 3),
            "cognitive_load": round(self.cognitive_load, 3),
            "reason": self.reason,
            "suggestion": self.suggestion,
        }


class StressGate:
    """Evaluates whether an action should be paused based on cognitive stress.

    Sits in the execution pipeline between the planner and executor.
    For destructive actions, checks the user's predicted cognitive state
    and returns a GateDecision indicating whether to proceed or pause.
    """

    def __init__(self, cognitive_engine: CognitiveEngine | None = None) -> None:
        self._engine = cognitive_engine or CognitiveEngine.get_instance()
        self._enabled = True
        self._gate_cooldowns: dict[str, float] = {}  # action_type → last gate time

        # Stats
        self._total_evaluated = 0
        self._total_gated = 0
        self._total_overridden = 0
        self._gate_history: list[GateDecision] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self, enabled: bool | None = None) -> bool:
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = not self._enabled
        return self._enabled

    async def evaluate(self, action_type: ActionType) -> GateDecision:
        """Evaluate whether an action should be gated due to stress.

        Returns a GateDecision. If gated=True, the executor should pause
        and request explicit user confirmation.
        """
        self._total_evaluated += 1

        # Safe actions are never gated
        if action_type in SAFE_ACTIONS:
            return GateDecision(
                action_type=action_type,
                gated=False,
                reason="safe_action",
            )

        if not self._enabled:
            return GateDecision(
                action_type=action_type,
                gated=False,
                reason="stress_gate_disabled",
            )

        # Check cooldown — don't nag repeatedly
        cooldown_key = action_type.value
        now = time.time()
        last_gate = self._gate_cooldowns.get(cooldown_key, 0.0)
        if now - last_gate < GATE_COOLDOWN_S:
            return GateDecision(
                action_type=action_type,
                gated=False,
                reason="cooldown_active",
            )

        # Get cognitive state
        state = await self._engine.predict_cognitive_state(stimulus_description=f"executing {action_type.value}")

        # Determine threshold based on action risk
        is_high_risk = action_type in HIGH_RISK_ACTIONS
        stress_threshold = HIGH_RISK_STRESS_THRESHOLD if is_high_risk else STRESS_GATE_THRESHOLD
        load_threshold = 0.65 if is_high_risk else LOAD_GATE_THRESHOLD

        # Gate decision logic
        should_gate = False
        reason = ""
        suggestion = ""

        if state.stress_level > stress_threshold:
            should_gate = True
            reason = f"high_stress ({state.stress_level:.2f} > {stress_threshold:.2f})"
            suggestion = "Your stress level appears elevated. Take a moment before proceeding with this action."

        elif state.cognitive_load > load_threshold and state.stress_level > 0.4:
            should_gate = True
            reason = f"compound_overload (load={state.cognitive_load:.2f}, stress={state.stress_level:.2f})"
            suggestion = "You seem to be handling a lot right now. Would you like to confirm this action?"

        if should_gate:
            self._total_gated += 1
            self._gate_cooldowns[cooldown_key] = now

        decision = GateDecision(
            action_type=action_type,
            gated=should_gate,
            stress_level=state.stress_level,
            cognitive_load=state.cognitive_load,
            reason=reason or "within_normal_range",
            suggestion=suggestion,
        )

        # Keep history
        self._gate_history.append(decision)
        if len(self._gate_history) > 50:
            self._gate_history = self._gate_history[-50:]

        # Record this interaction for future predictions
        self._engine.record_interaction(
            event_type="action_gate_check",
            modality="cognitive",
            intensity=state.stress_level,
            metadata={"action_type": action_type.value, "gated": should_gate},
        )

        if should_gate:
            logger.warning(
                "STRESS GATE: Pausing %s — stress=%.2f load=%.2f reason=%s",
                action_type.value,
                state.stress_level,
                state.cognitive_load,
                reason,
            )

        return decision

    def record_override(self, action_type: ActionType) -> None:
        """Record that the user explicitly overrode a gate decision."""
        self._total_overridden += 1
        logger.info("User overrode stress gate for %s", action_type.value)

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "total_evaluated": self._total_evaluated,
            "total_gated": self._total_gated,
            "total_overridden": self._total_overridden,
            "gate_rate": (round(self._total_gated / self._total_evaluated, 3) if self._total_evaluated > 0 else 0.0),
            "recent_gates": [g.to_dict() for g in self._gate_history[-5:] if g.gated],
        }
