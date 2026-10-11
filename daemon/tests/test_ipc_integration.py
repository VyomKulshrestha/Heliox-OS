import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from pilot.config import PilotConfig, _merge_config
from pilot.server import JsonRpcRequest, PilotServer


@pytest.fixture
async def server_port(unused_tcp_port):
    """Provides a random unused TCP port for the test server."""
    return unused_tcp_port


async def _wait_for_port(
    host: str,
    port: int,
    server_task: asyncio.Task,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        if server_task.done():
            if server_task.cancelled():
                raise RuntimeError("Server task was cancelled before port became ready.")
            raise RuntimeError("Server task failed before port became ready.") from server_task.exception()
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Server did not bind on {host}:{port} within {timeout_seconds}s.") from None
            await asyncio.sleep(0.05)


@pytest.fixture
async def daemon_server(server_port, tmp_path, monkeypatch):
    """
    Fixture that starts a PilotServer in the background with a
    clean temporary environment.
    """
    # Isolate the test environment using temporary directories
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "runtime"
    config_dir.mkdir()
    data_dir.mkdir()
    state_dir.mkdir()
    runtime_dir.mkdir()

    monkeypatch.setattr("pilot.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("pilot.config.CONFIG_FILE", config_dir / "config.toml")
    monkeypatch.setattr("pilot.config.RESTRICTIONS_FILE", config_dir / "restrictions.toml")
    monkeypatch.setattr("pilot.config.DATA_DIR", data_dir)
    monkeypatch.setattr("pilot.config.STATE_DIR", state_dir)
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr("pilot.config.DB_FILE", data_dir / "pilot.db")
    monkeypatch.setattr("pilot.config.LOG_FILE", state_dir / "pilot.log")

    config = PilotConfig()
    config.server.host = "127.0.0.1"
    config.server.port = server_port
    config.server.auth_token = "test-token"

    # Mock heavy subsystems to avoid requiring a real LLM or OCR
    # These mocks ensure the test focuses on the IPC layer.
    with (
        patch("pilot.models.router.ModelRouter.initialize", new_callable=AsyncMock),
        patch("pilot.models.cache.LLMCache.initialize", new_callable=AsyncMock),
        patch("pilot.memory.store.MemoryStore.initialize", new_callable=AsyncMock),
        patch("pilot.agents.screen_vision.ScreenVisionAgent.start", new_callable=AsyncMock),
        patch("pilot.cognitive.cognitive_engine.CognitiveEngine.load_model", new_callable=AsyncMock),
        patch("pilot.models.budget_tracker.BudgetTracker.initialize", new_callable=AsyncMock),
        patch("pilot.agents.prompt_improver.PromptImprover.initialize", new_callable=AsyncMock),
        patch("pilot.agents.subconscious.SubconsciousAgent.initialize", new_callable=AsyncMock),
        patch("pilot.security.audit.AuditLogger", return_value=MagicMock()),
        patch("pilot.agents.code_agent.CodeAgent", new_callable=MagicMock),
    ):
        server = PilotServer(config)
        server_task = asyncio.create_task(server.start())

        # Give the server a moment to start the websocket listener
        await _wait_for_port("127.0.0.1", server_port, server_task)

        uri = f"ws://127.0.0.1:{server_port}"
        yield uri

        # Cleanup
        await server.stop()
        assert server._orchestrator is not None
        assert server._orchestrator._scheduler_task.done()
        await server_task


@pytest.mark.asyncio
async def test_ipc_ping_pong(daemon_server):
    """Verify that the basic ping-pong handshake works."""
    async with websockets.connect(daemon_server) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "auth"}))
        auth_resp = json.loads(await ws.recv())
        assert auth_resp["result"]["status"] == "authenticated"

        request = {"jsonrpc": "2.0", "method": "ping", "params": {}, "id": "test-1"}
        await ws.send(json.dumps(request))

        response = json.loads(await ws.recv())
        assert response["id"] == "test-1"
        assert response["result"]["pong"] is True
        assert "version" in response["result"]


@pytest.mark.asyncio
async def test_daemon_rotates_a_role_scoped_local_mcp_token(daemon_server):
    from pilot import config as config_module

    token_path = config_module.RUNTIME_DIR / "mcp_auth_token"
    token = token_path.read_text(encoding="utf-8")
    assert len(token) >= 32

    async with websockets.connect(daemon_server) as ws:
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "auth",
                    "params": {"token": token},
                    "id": "mcp-auth",
                }
            )
        )
        authenticated = json.loads(await ws.recv())
        assert authenticated["result"] == {"status": "authenticated", "role": "mcp_local"}

        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "execute", "id": "forbidden"}))
        denied = json.loads(await ws.recv())
        assert denied["error"]["code"] == -32601
        assert "mcp_local" in denied["error"]["message"]


@pytest.mark.asyncio
async def test_neural_sidecar_token_is_role_scoped_and_cannot_call_ping(daemon_server):
    from pilot import config as config_module

    token = (config_module.RUNTIME_DIR / "neural_auth_token").read_text(encoding="utf-8")
    async with websockets.connect(daemon_server) as ws:
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "auth",
                    "params": {"token": token},
                    "id": "neural-auth",
                }
            )
        )
        auth = json.loads(await ws.recv())
        assert auth["result"] == {"status": "authenticated", "role": "neural_sidecar"}

        async with websockets.connect(daemon_server) as duplicate:
            await duplicate.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "auth",
                        "params": {"token": token},
                        "id": "duplicate-neural-auth",
                    }
                )
            )
            duplicate_denied = json.loads(await duplicate.recv())
            assert duplicate_denied["error"]["code"] == -32002

        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "neural_status", "id": "neural-status"}))
        status = json.loads(await ws.recv())
        assert status["result"]["status"] == "ok"
        assert status["result"]["connected"] is False
        assert status["result"]["capabilities"]["physical_control"] is False

        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": "forbidden"}))
        denied = json.loads(await ws.recv())
        assert denied["error"]["code"] == -32601
        assert "neural_sidecar" in denied["error"]["message"]


@pytest.mark.asyncio
async def test_real_neurod_bridge_calibrates_previews_commits_and_disconnects_fail_closed(
    daemon_server,
):
    """Exercise the product boundary with separate UI and sidecar sockets."""

    from pilot import config as config_module
    from pilot.neural.acquisition import SyntheticNeuralSource
    from pilot.neural.decoder import SSVEPDecoder
    from pilot.neural.gate import NeuralIntentSigner
    from pilot.neural.rpc_client import NeuralRpcTransport, NeurodBridge
    from pilot.neural.service import NeuralDecoderService
    from pilot.neural.simulator import build_synthetic_calibration_artifact
    from pilot.security.rpc_identity import derive_neural_signing_key

    token = (config_module.RUNTIME_DIR / "neural_auth_token").read_text(encoding="utf-8")
    artifact = build_synthetic_calibration_artifact()
    source = SyntheticNeuralSource(
        sample_rate_hz=artifact.sample_rate_hz,
        channel_names=artifact.channel_names,
        target_hz=10.0,
        noise_uv=0.5,
        seed=91,
    )
    service = NeuralDecoderService(
        source=source,
        decoder=SSVEPDecoder(artifact),
        signer=NeuralIntentSigner(derive_neural_signing_key(token)),
    )
    stop = asyncio.Event()
    bridge = NeurodBridge(
        transport=NeuralRpcTransport(url=daemon_server, token=token),
        service=service,
        artifact=artifact,
        poll_seconds=0.5,
    )
    bridge_task = asyncio.create_task(bridge.run(stop))
    notifications: list[dict] = []
    request_index = 0

    async def rpc(ws, method: str, params: dict | None = None) -> dict:
        nonlocal request_index
        request_index += 1
        request_id = f"ui-neural-{request_index}"
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params or {},
                    "id": request_id,
                }
            )
        )
        deadline = asyncio.get_running_loop().time() + 10
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.01, deadline - asyncio.get_running_loop().time()))
            message = json.loads(raw)
            if message.get("id") == request_id:
                if "error" in message:
                    raise AssertionError(message["error"])
                return message["result"]
            if message.get("method"):
                notifications.append(message)

    async def wait_status(ws, predicate, timeout: float = 12.0, *, require_bridge: bool = True) -> dict:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if require_bridge and bridge_task.done():
                raise AssertionError("neurod bridge stopped unexpectedly") from bridge_task.exception()
            status = await rpc(ws, "neural_status")
            if predicate(status):
                return status
            await asyncio.sleep(0.05)
        raise AssertionError("neural status did not reach the expected state")

    async def wait_notification(ws, method: str, timeout: float = 5.0) -> dict:
        for message in notifications:
            if message.get("method") == method:
                return message

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if bridge_task.done():
                raise AssertionError("neurod bridge stopped unexpectedly") from bridge_task.exception()
            raw = await asyncio.wait_for(
                ws.recv(),
                timeout=max(0.01, deadline - asyncio.get_running_loop().time()),
            )
            message = json.loads(raw)
            if message.get("method"):
                notifications.append(message)
                if message["method"] == method:
                    return message
        raise AssertionError(f"notification {method!r} was not received")

    try:
        async with websockets.connect(daemon_server) as ui:
            auth = await rpc(ui, "auth", {"token": "test-token"})
            assert auth["role"] == "ui"
            connected = await wait_status(ui, lambda status: status.get("connected") is True)
            session_id = connected["session_id"]

            calibration = await rpc(ui, "neural_begin_calibration", {"session_id": session_id})
            assert calibration["status"] == "ok"
            await wait_status(ui, lambda status: status.get("state") == "observe_only")

            armed = await rpc(
                ui,
                "neural_arm",
                {"session_id": session_id, "scope": "navigate", "user_authorized": True},
            )
            assert armed["state"] == "armed_safe_ui"

            async def pending_preview() -> tuple[dict, dict]:
                """Return the status and the announcement of the preview pending right now."""
                status = await wait_status(ui, lambda status: status.get("state") == "previewed")
                await wait_notification(ui, "neural_preview")
                deadline = asyncio.get_running_loop().time() + 5
                while asyncio.get_running_loop().time() < deadline:
                    for message in reversed(notifications):
                        params = message.get("params") or {}
                        if message.get("method") == "neural_preview" and params.get("preview_id") == status.get(
                            "pending_preview"
                        ):
                            return status, params
                    status = await rpc(ui, "neural_status")  # also collects newer notifications
                    await asyncio.sleep(0.05)
                raise AssertionError("the pending preview was never announced")

            # The bridge keeps polling, so on a slow host a newer preview can replace the
            # one just read; commit whichever preview is pending, with bounded retries.
            for _attempt in range(3):
                preview_status, preview = await pending_preview()
                assert preview["intent_class"] == "focus_right"
                assert preview["fusion"]["raw_media_excluded"] is True

                wait_seconds = max(
                    0.0,
                    (int(preview["eligible_at_ns"]) - time.monotonic_ns()) / 1_000_000_000,
                )
                await asyncio.sleep(wait_seconds + 0.05)
                result = await rpc(
                    ui,
                    "neural_commit",
                    {
                        "preview_id": preview["preview_id"],
                        "expected_revision": preview_status["state_revision"],
                        "world_model_approved": False,
                    },
                )
                if result["status"] == "committed":
                    break
                assert any(
                    reason in result.get("error", "")
                    for reason in ("no longer current", "changed after preview", "expired")
                ), result
            assert result["status"] == "committed", result
            assert result["canonical_goal"] == "neural_ui.focus_right"
            assert "retry_allowed" not in result

            stop.set()
            await asyncio.wait_for(bridge_task, timeout=5)
            disarmed = await wait_status(
                ui,
                lambda status: status.get("armed_scope") == "observe",
                require_bridge=False,
            )
            assert disarmed["state"] == "observe_only"
            assert disarmed["audit"]["valid"] is True
    finally:
        stop.set()
        if not bridge_task.done():
            await asyncio.wait_for(bridge_task, timeout=5)


@pytest.mark.asyncio
async def test_ipc_parse_error(daemon_server):
    """Verify that invalid JSON returns a Parse error (-32700)."""
    async with websockets.connect(daemon_server) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "auth"}))
        auth_resp = json.loads(await ws.recv())
        assert auth_resp["result"]["status"] == "authenticated"

        await ws.send("invalid json {")

        response = json.loads(await ws.recv())
        assert response["id"] is None
        assert response["error"]["code"] == -32700, response
        assert "Parse error" in response["error"]["message"]


def test_json_rpc_parser_rejects_non_object_payload():
    with pytest.raises(ValueError, match="expected an object"):
        JsonRpcRequest.parse("[]")


@pytest.mark.asyncio
async def test_ipc_method_not_found(daemon_server):
    """Verify that calling a non-existent method returns -32601."""
    async with websockets.connect(daemon_server) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "auth"}))
        auth_resp = json.loads(await ws.recv())
        assert auth_resp["result"]["status"] == "authenticated"

        request = {"jsonrpc": "2.0", "method": "non_existent_method", "id": 99}
        await ws.send(json.dumps(request))

        response = json.loads(await ws.recv())
        assert response["id"] == 99
        assert response["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_ipc_get_config(daemon_server):
    """Verify that get_config returns the expected configuration sections."""
    async with websockets.connect(daemon_server) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "auth"}))
        auth_resp = json.loads(await ws.recv())
        assert auth_resp["result"]["status"] == "authenticated"

        request = {"jsonrpc": "2.0", "method": "get_config", "id": "cfg-1"}
        await ws.send(json.dumps(request))

        response = json.loads(await ws.recv())
        result = response["result"]
        assert "model" in result
        assert "security" in result
        assert result["screen_vision"]["capture_interval_seconds"] == 3.0
        assert "first_run_complete" in result


@pytest.mark.asyncio
async def _reply(ws):
    """The next JSON-RPC response; notifications such as config_updated may arrive in between."""
    while True:
        message = json.loads(await ws.recv())
        if "id" in message:
            return message


async def test_ipc_update_screen_vision_interval(daemon_server):
    """Verify screen vision interval updates round-trip through config IPC."""
    async with websockets.connect(daemon_server) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "auth"}))
        auth_resp = await _reply(ws)
        assert auth_resp["result"]["status"] == "authenticated"

        request = {
            "jsonrpc": "2.0",
            "method": "update_config",
            "params": {
                "section": "screen_vision",
                "values": {"capture_interval_seconds": 7.5},
            },
            "id": "screen-cfg-1",
        }
        await ws.send(json.dumps(request))

        response = await _reply(ws)
        assert response["result"]["status"] == "ok"

        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "get_config", "id": "screen-cfg-2"}))
        response = await _reply(ws)
        assert response["result"]["screen_vision"]["capture_interval_seconds"] == 7.5


@pytest.mark.asyncio
async def test_screen_vision_toggle_uses_configured_interval_by_default():
    """Verify toggling screen vision without params uses the persisted interval."""
    config = PilotConfig()
    config.screen_vision.capture_interval_seconds = 8.5
    server = PilotServer(config)
    screen_vision = MagicMock()
    screen_vision.start = AsyncMock()
    server._screen_vision = screen_vision

    result = await server._handle_screen_vision_toggle({"enabled": True}, MagicMock())

    assert result == {"status": "ok", "enabled": True}
    screen_vision.start.assert_awaited_once_with(8.5, False)


def test_screen_vision_config_merges_numeric_interval():
    config = PilotConfig()
    raw = {"screen_vision": {"capture_interval_seconds": 12}}

    merged = _merge_config(config, raw)

    assert merged.screen_vision.capture_interval_seconds == 12.0


@pytest.mark.asyncio
async def test_ipc_execute_flow_broadcast(daemon_server):
    """
    Verify that notifications (status updates) are broadcast correctly.
    """
    async with websockets.connect(daemon_server) as ws:
        # Mock the planner so it doesn't try to call an LLM
        with patch("pilot.agents.planner.Planner.plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = MagicMock(error=None, actions=[], explanation="Mocked plan")

            await ws.send(
                json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "auth"})
            )
            auth_resp = json.loads(await ws.recv())
            assert auth_resp["result"]["status"] == "authenticated"

            request = {"jsonrpc": "2.0", "method": "execute", "params": {"input": "test command"}, "id": "exec-1"}
            await ws.send(json.dumps(request))

            # We expect several notifications and eventually a response
            # 1. status: receiving input
            # 2. status: recalling memory
            # 3. status: routing agents
            # ...

            messages = []
            for _ in range(5):
                msg = json.loads(await ws.recv())
                messages.append(msg)
                if "id" in msg and msg["id"] == "exec-1":
                    break

            # Check that we got at least one status notification
            status_notifications = [m for m in messages if m.get("method") == "status"]
            assert len(status_notifications) > 0
            assert status_notifications[0]["params"]["phase"] == "receiving input"
