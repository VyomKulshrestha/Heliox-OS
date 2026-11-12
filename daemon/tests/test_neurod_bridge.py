from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.neural.protocol import NeuralEvidenceKind, NeuralStreamDescriptorV1, NeuralTransport
from pilot.neural.rpc_client import NeurodBridge


class FakeTransport:
    def __init__(self, stop: asyncio.Event) -> None:
        self.stop = stop
        self.calls: list[tuple[str, dict]] = []
        self.status_count = 0
        self.connected = False
        self.closed = False
        self.markers: list[dict] = []

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def call(self, method: str, params: dict | None = None) -> dict:
        payload = params or {}
        self.calls.append((method, payload))
        if method == "neural_status":
            self.status_count += 1
            return (
                {"status": "ok", "state": "calibrating", "state_revision": 2}
                if self.status_count == 1
                else {"status": "ok", "state": "armed_safe_ui", "state_revision": 4}
            )
        if method == "neural_stimulus_markers":
            markers, self.markers = self.markers, []
            return {"status": "ok", "markers": markers}
        if method == "neural_intent_preview":
            self.stop.set()
            return {"status": "previewed"}
        return {"status": "ok"}


class FakeService:
    def __init__(self) -> None:
        self.descriptor = NeuralStreamDescriptorV1(
            session_id=uuid.uuid4(),
            source_id="synthetic-test",
            board_kind="synthetic",
            transport=NeuralTransport.SYNTHETIC,
            evidence_kind=NeuralEvidenceKind.SYNTHETIC,
            sample_rate_hz=250,
            channel_count=2,
            channel_names=("O1", "Oz"),
            reference="synthetic-reference",
            sequence_start=0,
            started_monotonic_ns=1,
        )
        self.buffer_health = SimpleNamespace(buffered_samples=500, dropped_samples=0)
        self.started = False
        self.stopped = False
        self.markers = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def observe_once(self, *, state_revision: int, requested_scope):
        quality = MagicMock()
        quality.model_dump.return_value = {
            "quality": "good",
            "artifact_flags": [],
            "channel_std_uv": [4.2, 4.1],
            "line_noise_ratio": 0.01,
            "muscle_ratio": 0.02,
            "estimated_missing_samples": 0,
            "timestamp_jitter_ratio": 0.0,
            "reasons": [],
        }
        intent = MagicMock()
        intent.model_dump.return_value = {
            "schema_version": 1,
            "class": "select",
            "signature": "a" * 64,
        }
        return SimpleNamespace(quality=quality, intent=intent)

    def record_stimulus_marker(self, marker) -> None:
        self.markers.append(marker)


@pytest.mark.asyncio
async def test_bridge_activates_calibration_streams_only_summaries_and_closes() -> None:
    stop = asyncio.Event()
    transport = FakeTransport(stop)
    service = FakeService()
    transport.markers = [
        {
            "schema_version": 1,
            "session_id": str(service.descriptor.session_id),
            "sequence": 0,
            "target_id": "focus_right",
            "event": "target_on",
            "received_monotonic_ns": 100,
            "client_performance_ms": 50.0,
        }
    ]
    metrics = MagicMock()
    metrics.model_dump.return_value = {"balanced_accuracy": 1.0}
    artifact = SimpleNamespace(
        calibration_id="b" * 64,
        decoder_version="a" * 64,
        subject_key="local-subject",
        metrics=metrics,
    )
    bridge = NeurodBridge(
        transport=transport,
        service=service,
        artifact=artifact,
        poll_seconds=0.05,
    )
    await asyncio.wait_for(bridge.run(stop), timeout=2)

    methods = [method for method, _ in transport.calls]
    assert methods[:4] == [
        "neural_connect",
        "neural_stimulus_markers",
        "neural_status",
        "neural_finish_calibration",
    ]
    assert "neural_observation" in methods
    assert "neural_intent_preview" in methods
    observation = next(params for method, params in transport.calls if method == "neural_observation")
    assert "samples_uv" not in observation and "timestamps_ns" not in observation
    assert transport.connected and transport.closed
    assert service.started and service.stopped
    assert service.markers[0].target_id == "focus_right"


class FlakyTransport(FakeTransport):
    """Fails the way a saturated machine or a restarted daemon does, then works."""

    def __init__(self, stop: asyncio.Event, failures: list[str]) -> None:
        super().__init__(stop)
        self.failures = failures
        self.connects = 0
        self.closes = 0

    async def connect(self) -> None:
        self.connects += 1
        if self.failures and self.failures[0] == "connect":
            self.failures.pop(0)
            raise OSError("connection refused")
        self.connected = True

    async def close(self) -> None:
        self.closes += 1

    async def call(self, method: str, params: dict | None = None) -> dict:
        if method == "neural_status" and self.failures and self.failures[0] == "slow":
            self.failures.pop(0)
            from pilot.neural.rpc_client import NeuralRpcError

            raise NeuralRpcError("RPC timed out: neural_status")
        return await super().call(method, params)


def _artifact():
    metrics = MagicMock()
    metrics.model_dump.return_value = {"balanced_accuracy": 1.0}
    return SimpleNamespace(calibration_id="b" * 64, decoder_version="a" * 64, subject_key="s", metrics=metrics)


@pytest.mark.asyncio
async def test_a_slow_rpc_or_a_lost_daemon_reconnects_with_a_back_off(monkeypatch) -> None:
    from pilot.neural import rpc_client

    monkeypatch.setattr(rpc_client, "RECONNECT_BASE_SECONDS", 0.01)
    stop = asyncio.Event()
    transport = FlakyTransport(stop, ["slow", "connect", "connect"])
    seen: list[dict] = []
    bridge = NeurodBridge(transport=transport, service=FakeService(), artifact=_artifact(), poll_seconds=0.05)
    original = bridge._session

    async def watched(stop_event):
        seen.append(dict(bridge.status))
        return await original(stop_event)

    monkeypatch.setattr(bridge, "_session", watched)

    await asyncio.wait_for(bridge.run(stop), timeout=5)

    # One slow RPC after connecting, then the daemon refused twice; the bridge kept going and finished its work.
    assert transport.connects == 4
    assert [method for method, _ in transport.calls].count("neural_connect") == 2  # one per established link
    assert seen[1]["state"] == "reconnecting" and seen[1]["attempt"] == 1  # reset: the first link was established
    assert seen[2]["attempt"] == 2 and seen[3]["attempt"] == 3
    assert seen[3]["retry_in_s"] == pytest.approx(0.04)
    assert "timed out" in seen[1]["reason"]
    assert bridge.status == {"state": "stopped"}


@pytest.mark.asyncio
async def test_stop_during_the_back_off_ends_the_bridge_at_once(monkeypatch) -> None:
    from pilot.neural import rpc_client

    monkeypatch.setattr(rpc_client, "RECONNECT_BASE_SECONDS", 30.0)
    stop = asyncio.Event()
    transport = FlakyTransport(stop, ["connect"] * 5)
    service = FakeService()
    bridge = NeurodBridge(transport=transport, service=service, artifact=_artifact(), poll_seconds=0.05)

    task = asyncio.create_task(bridge.run(stop))
    while bridge.status.get("state") != "reconnecting":
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, timeout=5)

    assert transport.connects == 1 and service.stopped and bridge.status == {"state": "stopped"}


class QueuedConnection:
    """A daemon connection that answers with queued messages; ``None`` means nothing more arrives."""

    def __init__(self, messages: list) -> None:
        self.messages = list(messages)
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def recv(self) -> str:
        if not self.messages or self.messages[0] is None:
            await asyncio.sleep(3600)
        message = self.messages.pop(0)
        return message(self.sent) if callable(message) else message


def _reply_to_last(sent: list[str]) -> str:
    import json

    request = json.loads(sent[-1])
    return json.dumps({"jsonrpc": "2.0", "result": {"status": "ok", "state": "idle"}, "id": request["id"]})


@pytest.mark.asyncio
async def test_a_call_skips_notifications_and_late_replies_and_returns_its_own() -> None:
    import json

    from pilot.neural.rpc_client import NeuralRpcTransport

    transport = NeuralRpcTransport(url="ws://unused", token="t", timeout_seconds=2)
    transport._connection = QueuedConnection(
        [
            json.dumps({"jsonrpc": "2.0", "method": "startup_status", "params": {"phase": "starting"}}),
            json.dumps({"jsonrpc": "2.0", "result": {"status": "ok"}, "id": "neurod-0"}),  # an earlier call's, late
            _reply_to_last,
        ]
    )

    result = await transport.call("neural_status")

    assert result == {"status": "ok", "state": "idle"}


@pytest.mark.asyncio
async def test_notifications_alone_end_in_a_timeout_within_one_limit() -> None:
    import json
    import time

    from pilot.neural.rpc_client import NeuralRpcError, NeuralRpcTransport

    transport = NeuralRpcTransport(url="ws://unused", token="t", timeout_seconds=0.3)
    transport._connection = QueuedConnection(
        [json.dumps({"jsonrpc": "2.0", "method": "startup_status", "params": {}})] * 3 + [None]
    )
    started = time.monotonic()

    with pytest.raises(NeuralRpcError, match="timed out: neural_status"):
        await transport.call("neural_status")
    assert time.monotonic() - started < 1.5  # one limit for the whole wait, not one per message
