"""Tests for Executor's cognitive-stress-gate daemon_speech pairing.

The stress-gate pause speaks directly on the daemon's own OS audio
(pilot.system.voice.speak()), not the frontend's speechSynthesis --
without a paired notification, the user gets zero visual trace of the
10-second pause or why it's happening. daemon_speech must carry the exact
spoken phrase and is display-only (session.ts appends a chat bubble via
addSystemMessage; it must never also call speakText() for this
notification, or the phrase would be spoken twice).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator


class _Broadcast:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, method: str, params: dict) -> None:
        self.calls.append((method, params))


class _FakeStressGate:
    def __init__(self, gated: bool, enabled: bool = True):
        self.enabled = enabled
        self._gated = gated

    async def evaluate(self, action_type):
        from pilot.cognitive.stress_gate import GateDecision

        return GateDecision(action_type=action_type, gated=self._gated)


def _executor(tmp_path) -> Executor:
    config = PilotConfig()
    validator = ActionValidator(config)
    permissions = PermissionChecker(config)
    audit = AuditLogger(audit_file=tmp_path / "audit.jsonl")
    return Executor(config, validator, permissions, audit)


def _action() -> Action:
    return Action(action_type=ActionType.CPU_USAGE, target="x", parameters=EmptyParams())


@pytest.mark.asyncio
async def test_gated_action_speaks_and_broadcasts_matching_text(tmp_path):
    ex = _executor(tmp_path)
    ex._stress_gate = _FakeStressGate(gated=True)
    broadcast = _Broadcast()
    ex.set_broadcast(broadcast)

    with (
        patch("pilot.system.voice.speak", new=AsyncMock(return_value="")) as mock_speak,
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        await ex._execute_single(_action(), snapshot_id=None)

    spoken_text = mock_speak.call_args.args[0]
    assert len(broadcast.calls) == 1
    method, params = broadcast.calls[0]
    assert method == "daemon_speech"
    assert params["text"] == spoken_text
    assert params["source"] == "stress_gate"


@pytest.mark.asyncio
async def test_not_gated_never_speaks_or_broadcasts(tmp_path):
    ex = _executor(tmp_path)
    ex._stress_gate = _FakeStressGate(gated=False)
    broadcast = _Broadcast()
    ex.set_broadcast(broadcast)

    with patch("pilot.system.voice.speak", new=AsyncMock(return_value="")) as mock_speak:
        await ex._execute_single(_action(), snapshot_id=None)

    mock_speak.assert_not_awaited()
    assert broadcast.calls == []


@pytest.mark.asyncio
async def test_explicitly_confirmed_action_does_not_pause_or_speak(tmp_path):
    ex = _executor(tmp_path)
    stress_gate = _FakeStressGate(gated=True)
    stress_gate.evaluate = AsyncMock(wraps=stress_gate.evaluate)
    ex._stress_gate = stress_gate
    broadcast = _Broadcast()
    ex.set_broadcast(broadcast)

    with (
        patch("pilot.system.voice.speak", new=AsyncMock(return_value="")) as mock_speak,
        patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        result = await ex._execute_single(
            _action(),
            snapshot_id=None,
            user_confirmed=True,
        )

    stress_gate.evaluate.assert_not_awaited()
    mock_speak.assert_not_awaited()
    mock_sleep.assert_not_awaited()
    assert broadcast.calls == []
    assert result.success is True


@pytest.mark.asyncio
async def test_no_broadcast_configured_is_a_safe_noop(tmp_path):
    ex = _executor(tmp_path)  # set_broadcast() never called
    ex._stress_gate = _FakeStressGate(gated=True)

    with (
        patch("pilot.system.voice.speak", new=AsyncMock(return_value="")) as mock_speak,
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        await ex._execute_single(_action(), snapshot_id=None)  # must not raise

    mock_speak.assert_awaited_once()


@pytest.mark.asyncio
async def test_speak_failure_does_not_propagate_or_skip_the_pause(tmp_path):
    ex = _executor(tmp_path)
    ex._stress_gate = _FakeStressGate(gated=True)
    broadcast = _Broadcast()
    ex.set_broadcast(broadcast)

    with (
        patch("pilot.system.voice.speak", new=AsyncMock(side_effect=RuntimeError("no tts engine"))),
        patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        result = await ex._execute_single(_action(), snapshot_id=None)  # must not raise

    assert broadcast.calls == []  # never reached the broadcast after speak() raised
    mock_sleep.assert_awaited_once_with(10)
    assert result.success is True  # execution still proceeds after the (failed) announcement


@pytest.mark.asyncio
async def test_stress_gate_uses_shared_priority_speech_when_wired(tmp_path):
    ex = _executor(tmp_path)
    ex._stress_gate = _FakeStressGate(gated=True)
    speech = AsyncMock()
    ex.set_speech(speech)

    with (
        patch("pilot.system.voice.speak", new=AsyncMock()) as direct_speak,
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        await ex._execute_single(_action(), snapshot_id=None)

    direct_speak.assert_not_awaited()
    speech.assert_awaited_once_with(
        "Pausing ten seconds before continuing. Say or press Stop to cancel.",
        "approval_risk",
        "stress-gate:cpu_usage",
    )


@pytest.mark.asyncio
async def test_real_action_outcome_feeds_historical_world_model(tmp_path):
    ex = _executor(tmp_path)
    recorder = MagicMock()
    ex.set_world_model_outcome_recorder(recorder)

    results = await ex.execute(ActionPlan(actions=[_action()], raw_input="cpu"))

    assert results[0].success is True
    recorder.assert_called_once_with(ActionType.CPU_USAGE, True)


@pytest.mark.asyncio
async def test_dry_run_does_not_train_historical_world_model(tmp_path):
    ex = _executor(tmp_path)
    ex._config.security.dry_run = True
    recorder = MagicMock()
    ex.set_world_model_outcome_recorder(recorder)

    results = await ex.execute(ActionPlan(actions=[_action()], raw_input="cpu"))

    assert results[0].output.startswith("(dry run)")
    recorder.assert_not_called()
