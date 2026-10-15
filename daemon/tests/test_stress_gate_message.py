"""The stress-gate pause says what it is: a ten-second window to cancel, not a confirmation (F7)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pilot.actions import Action, ActionType, EmptyParams
from pilot.agents.executor import STRESS_PAUSE_PHRASE, Executor
from pilot.cognitive.stress_gate import GATE_COOLDOWN_S, GateDecision, StressGate
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator


class _Engine:
    """A cognitive engine that predicts high stress for every action."""

    def __init__(self) -> None:
        self.recorded: list[dict] = []

    async def predict_cognitive_state(self, stimulus_description: str = ""):
        return SimpleNamespace(stress_level=0.9, cognitive_load=0.5)

    def record_interaction(self, **kwargs) -> None:
        self.recorded.append(kwargs)


class _GatedStressGate:
    enabled = True

    def __init__(self) -> None:
        self.evaluated = 0

    async def evaluate(self, action_type: ActionType) -> GateDecision:
        self.evaluated += 1
        return GateDecision(action_type=action_type, gated=True)


def _executor(tmp_path) -> Executor:
    config = PilotConfig()
    return Executor(config, ActionValidator(config), PermissionChecker(config), AuditLogger(tmp_path / "audit.jsonl"))


def test_the_phrase_describes_a_cancel_window_and_never_a_confirmation():
    assert STRESS_PAUSE_PHRASE == "Pausing ten seconds before continuing. Say or press Stop to cancel."
    assert "confirm" not in STRESS_PAUSE_PHRASE.casefold()
    assert "focus" not in STRESS_PAUSE_PHRASE.casefold()


def test_the_gate_docstring_describes_a_cancel_window():
    doc = inspect.getdoc(StressGate.evaluate) or ""
    assert "cancel" in doc
    assert "request explicit user confirmation" not in doc


@pytest.mark.asyncio
async def test_gated_action_speaks_the_phrase_pauses_ten_seconds_once_and_then_runs(tmp_path):
    executor = _executor(tmp_path)
    gate = _GatedStressGate()
    executor._stress_gate = gate
    speech = AsyncMock()
    broadcast = AsyncMock()
    executor.set_speech(speech)
    executor.set_broadcast(broadcast)

    with patch("asyncio.sleep", new=AsyncMock()) as sleep:
        result = await executor._execute_single(
            Action(action_type=ActionType.CPU_USAGE, target="x", parameters=EmptyParams()), snapshot_id=None
        )

    assert gate.evaluated == 1
    sleep.assert_awaited_once_with(10)
    speech.assert_awaited_once_with(STRESS_PAUSE_PHRASE, "approval_risk", "stress-gate:cpu_usage")
    method, params = broadcast.await_args.args
    assert method == "daemon_speech"
    assert params["text"] == STRESS_PAUSE_PHRASE
    assert result.success is True


@pytest.mark.asyncio
async def test_gate_counters_and_cooldown_are_unchanged():
    engine = _Engine()
    gate = StressGate(engine)  # type: ignore[arg-type]

    first = await gate.evaluate(ActionType.FILE_DELETE)
    second = await gate.evaluate(ActionType.FILE_DELETE)
    safe = await gate.evaluate(ActionType.CPU_USAGE)
    stats = gate.get_stats()

    assert first.gated is True
    assert second.gated is False and second.reason == "cooldown_active"
    assert safe.gated is False and safe.reason == "safe_action"
    assert stats["total_evaluated"] == 3
    assert stats["total_gated"] == 1
    assert stats["total_overridden"] == 0
    assert GATE_COOLDOWN_S == 30.0
