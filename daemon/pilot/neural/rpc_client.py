"""Runnable neurod-to-Heliox paired WebSocket bridge."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from pilot.config import DATA_DIR, RUNTIME_DIR
from pilot.neural.acquisition import (
    BrainFlowNeuralSource,
    LSLNeuralSource,
    PlaybackNeuralSource,
    SyntheticNeuralSource,
)
from pilot.neural.decoder import SSVEPCalibrationArtifact, SSVEPDecoder
from pilot.neural.gate import NeuralIntentSigner
from pilot.neural.protocol import NeuralScope, NeuralStimulusMarkerV1
from pilot.neural.recording import (
    EncryptedNeuralRecorder,
    NeuralRecordingConsentV1,
    prune_expired_neural_recordings,
)
from pilot.neural.service import NeuralDecoderService
from pilot.neural.simulator import ensure_synthetic_calibration_artifact
from pilot.security.rpc_identity import derive_neural_signing_key
from pilot.timeouts import bounded

logger = logging.getLogger("pilot.neural.rpc_client")


class NeuralRpcError(RuntimeError):
    pass


# After losing the daemon (a slow RPC on a saturated machine, a daemon restart), the bridge connects again after 1 s,
# doubling to 60 s; the back-off starts again once a connection is established.
RECONNECT_BASE_SECONDS = 1.0
RECONNECT_MAX_SECONDS = 60.0


class NeuralRpcTransport:
    """Sequential JSON-RPC transport; neurod is excluded from UI broadcasts."""

    def __init__(self, *, url: str, token: str, timeout_seconds: float = 10.0) -> None:
        self._url = url
        self._token = token
        self._timeout = timeout_seconds
        self._connection: ClientConnection | None = None
        self._request_id = 0

    async def connect(self) -> None:
        if self._connection is not None:
            return
        self._connection = await connect(
            self._url,
            open_timeout=self._timeout,
            ping_interval=30,
            ping_timeout=30,
        )
        result = await self.call("auth", {"token": self._token})
        if result.get("role") != "neural_sidecar":
            await self.close()
            raise NeuralRpcError("daemon did not grant the neural_sidecar role")

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._connection is None:
            raise NeuralRpcError("neural RPC transport is not connected")
        self._request_id += 1
        request_id = f"neurod-{self._request_id}"
        await self._connection.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params or {},
                    "id": request_id,
                },
                separators=(",", ":"),
            )
        )
        response = await self._reply_to(request_id, method)
        if "error" in response:
            raise NeuralRpcError(str(response["error"].get("message") or "RPC failed"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise NeuralRpcError("neural RPC returned a non-object result")
        return result

    async def _reply_to(self, request_id: str, method: str) -> dict[str, Any]:
        """Read until the reply to ``request_id``, within one timeout for the whole wait.

        The daemon sends every connection some notifications of its own (``startup_status`` right after auth while
        it starts, for one), and a reply to an earlier call that timed out can still arrive late. Neither is this
        call's reply: both are skipped, as the Rust app's send_rpc does.
        """
        connection = self._connection
        if connection is None:
            raise NeuralRpcError("neural RPC transport is not connected")
        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NeuralRpcError(f"RPC timed out: {method}")
            try:
                raw = await bounded(connection.recv(), timeout=remaining)
            except TimeoutError as exc:
                raise NeuralRpcError(f"RPC timed out: {method}") from exc
            try:
                response = json.loads(str(raw))
            except ValueError as exc:
                raise NeuralRpcError("neural RPC received a message that is not JSON") from exc
            if isinstance(response, dict) and response.get("id") == request_id:
                return response


class BridgeTransport(Protocol):
    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]: ...


class NeurodBridge:
    """Pair acquisition with Heliox while keeping raw samples process-local."""

    _ARMED_STATES = {"armed_safe_ui", "armed_safe_desktop"}

    def __init__(
        self,
        *,
        transport: BridgeTransport,
        service: NeuralDecoderService,
        artifact: SSVEPCalibrationArtifact,
        poll_seconds: float = 0.5,
    ) -> None:
        if not 0.05 <= poll_seconds <= 5:
            raise ValueError("neurod poll interval must be between 50 ms and 5 seconds")
        self._transport = transport
        self._service = service
        self._artifact = artifact
        self._poll_seconds = poll_seconds
        self._activated_revision = -1
        self._last_marker_sequence = -1
        # "connecting", "connected", "reconnecting" (with attempt, retry_in_s and reason) or "stopped".
        self.status: dict[str, Any] = {"state": "stopped"}

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Pair with the daemon until ``stop`` is set, connecting again with a back-off whenever the link fails.

        A failed RPC (timed out on a saturated machine, a daemon that restarted, a closed socket) no longer ends the
        bridge for the session. Each new connection starts a new session on the daemon side: the daemon disarmed
        when the sidecar went away, so nothing is armed again without the person (fail closed).
        """
        stop = stop_event or asyncio.Event()
        # In a worker thread: starting opens the recording, whose key comes from the system keyring.
        await asyncio.to_thread(self._service.start)
        failures = 0
        try:
            while not stop.is_set():
                try:
                    await self._session(stop)
                    return
                except (NeuralRpcError, OSError, TimeoutError, ValueError, ConnectionClosed) as error:
                    if stop.is_set():
                        return
                    failures = 0 if self.status.get("state") == "connected" else failures
                    failures += 1
                    delay = min(RECONNECT_BASE_SECONDS * 2 ** (failures - 1), RECONNECT_MAX_SECONDS)
                    reason = str(error)[:200] or type(error).__name__
                    self.status = {"state": "reconnecting", "attempt": failures, "retry_in_s": delay, "reason": reason}
                    logger.warning("Neural bridge lost the daemon (%s); connecting again in %.0f s", reason, delay)
                    with contextlib.suppress(Exception):
                        await self._transport.close()
                    self._activated_revision = -1
                    self._last_marker_sequence = -1
                    with contextlib.suppress(TimeoutError):
                        await bounded(stop.wait(), timeout=delay)
        finally:
            self.status = {"state": "stopped"}
            self._service.stop()
            await self._transport.close()

    async def _session(self, stop: asyncio.Event) -> None:
        """One connection: connect, announce the stream, then poll until ``stop``. Raises when the link fails."""
        self.status = {"state": "connecting"}
        await self._transport.connect()
        connected = await self._transport.call(
            "neural_connect",
            {"descriptor": self._service.descriptor.model_dump(mode="json")},
        )
        self._require_ok(connected, "connect")
        self.status = {"state": "connected"}
        while not stop.is_set():
            marker_result = await self._transport.call(
                "neural_stimulus_markers",
                {"after_sequence": self._last_marker_sequence},
            )
            self._require_ok(marker_result, "stimulus marker sync")
            for payload in marker_result.get("markers", []):
                marker = NeuralStimulusMarkerV1.model_validate(payload)
                self._service.record_stimulus_marker(marker)
                self._last_marker_sequence = marker.sequence
            status = await self._transport.call("neural_status")
            state = str(status.get("state") or "")
            revision = int(status.get("state_revision", 0))
            if state == "calibrating" and revision != self._activated_revision:
                result = await self._transport.call(
                    "neural_finish_calibration",
                    {
                        "session_id": str(self._service.descriptor.session_id),
                        "calibration_id": self._artifact.calibration_id,
                        "decoder_version": self._artifact.decoder_version,
                        "subject_key": self._artifact.subject_key,
                        "metrics": self._artifact.metrics.model_dump(mode="json"),
                    },
                )
                self._require_ok(result, "calibration activation")
                self._activated_revision = revision
            elif state in self._ARMED_STATES:
                scope = NeuralScope.NAVIGATE if state == "armed_safe_ui" else NeuralScope.SAFE_DESKTOP
                observation = self._service.observe_once(
                    state_revision=revision,
                    requested_scope=scope,
                )
                health = self._service.buffer_health
                telemetry = await self._transport.call(
                    "neural_observation",
                    {
                        "quality": observation.quality.model_dump(mode="json"),
                        "buffered_samples": health.buffered_samples,
                        "dropped_samples": health.dropped_samples,
                        "observed_at_ns": time.monotonic_ns(),
                    },
                )
                self._require_ok(telemetry, "quality telemetry")
                if observation.intent is not None:
                    preview = await self._transport.call(
                        "neural_intent_preview",
                        {"intent": observation.intent.model_dump(mode="json", by_alias=True)},
                    )
                    # Low dwell/margin/confidence is a normal abstention,
                    # not a retryable delivery error.
                    if preview.get("status") not in {"previewed", "cancelled", "rejected"}:
                        raise NeuralRpcError("daemon returned an invalid preview state")
            try:
                await bounded(stop.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                pass

    @staticmethod
    def _require_ok(result: dict[str, Any], operation: str) -> None:
        if result.get("status") != "ok":
            raise NeuralRpcError(f"neural {operation} rejected: {result.get('error', 'unknown error')}")


def _source_from_args(args: argparse.Namespace, artifact: SSVEPCalibrationArtifact):
    if args.source == "synthetic":
        return SyntheticNeuralSource(
            sample_rate_hz=artifact.sample_rate_hz,
            channel_names=artifact.channel_names,
            target_hz=args.synthetic_frequency,
            seed=args.seed,
        )
    if args.source == "playback":
        if not args.playback:
            raise ValueError("--playback is required for playback source")
        return PlaybackNeuralSource.from_npz(Path(args.playback))
    if args.source == "brainflow":
        parameters = {"serial_port": args.serial_port} if args.serial_port else {}
        return BrainFlowNeuralSource(
            board_id=args.board_id,
            channel_names=artifact.channel_names,
            reference=artifact.reference,
            input_parameters=parameters,
        )
    return LSLNeuralSource(
        stream_name=args.lsl_name,
        channel_names=artifact.channel_names,
        sample_rate_hz=artifact.sample_rate_hz,
        reference=artifact.reference,
    )


async def _run_from_args(args: argparse.Namespace) -> None:
    token = Path(args.token_file).expanduser().resolve().read_text(encoding="utf-8").strip()
    if args.artifact:
        artifact = SSVEPCalibrationArtifact.load(Path(args.artifact))
    elif args.source == "synthetic":
        artifact = ensure_synthetic_calibration_artifact(RUNTIME_DIR / "synthetic_ssvep_calibration.json")
    else:
        raise ValueError("--artifact is required for live and playback neural sources")
    source = _source_from_args(args, artifact)
    signer = NeuralIntentSigner(derive_neural_signing_key(token))
    recorder_factory = None
    if args.record_raw:
        granted_at = datetime.now(UTC)
        recording_path = (
            Path(args.recording_file)
            if args.recording_file
            else DATA_DIR / "neural" / f"{artifact.subject_key}-{granted_at:%Y%m%dT%H%M%SZ}.neeg"
        )
        prune_expired_neural_recordings(recording_path.expanduser().resolve().parent)

        def recorder_factory(descriptor):
            consent = NeuralRecordingConsentV1(
                session_id=descriptor.session_id,
                subject_key=artifact.subject_key,
                purpose=args.recording_purpose,
                granted_at=granted_at,
                expires_at=granted_at + timedelta(days=args.retention_days),
                retention_days=args.retention_days,
                allow_bids_export=args.allow_bids_export,
                authorized=True,
            )
            return EncryptedNeuralRecorder(
                destination=recording_path,
                descriptor=descriptor,
                consent=consent,
            )

    service = NeuralDecoderService(
        source=source,
        decoder=SSVEPDecoder(artifact),
        signer=signer,
        recorder_factory=recorder_factory,
    )
    bridge = NeurodBridge(
        transport=NeuralRpcTransport(url=args.url, token=token),
        service=service,
        artifact=artifact,
        poll_seconds=args.poll_seconds,
    )
    await bridge.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Heliox least-privileged neural gateway")
    parser.add_argument("--url", default="ws://127.0.0.1:8785")
    parser.add_argument("--token-file", default=str(RUNTIME_DIR / "neural_auth_token"))
    parser.add_argument(
        "--artifact",
        help="Content-addressed calibration JSON (generated automatically only for synthetic mode)",
    )
    parser.add_argument("--source", choices=("synthetic", "playback", "brainflow", "lsl"), required=True)
    parser.add_argument("--playback")
    parser.add_argument("--board-id", type=int, default=0)
    parser.add_argument("--serial-port", default="")
    parser.add_argument("--lsl-name", default="HelioxEEG")
    parser.add_argument("--synthetic-frequency", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--record-raw", action="store_true", help="Explicitly consent to encrypted local raw EEG")
    parser.add_argument("--recording-file", help="New .neeg destination; never overwritten")
    parser.add_argument("--recording-purpose", default="local accessibility calibration")
    parser.add_argument("--retention-days", type=int, choices=range(1, 366), default=7, metavar="1-365")
    parser.add_argument("--allow-bids-export", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(_run_from_args(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
