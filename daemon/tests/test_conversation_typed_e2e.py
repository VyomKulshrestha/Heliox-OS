"""A typed question answered by the conversation layer, through a real server over its WebSocket: the messages the
chat view needs arrive in order and leave it finished (no spinner, the reply shown and stored)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from pilot.config import PilotConfig
from pilot.server import PilotServer


class StubTalk:
    """The fast model: a short markdown reply in pieces."""

    name, provider, streams, local = "stub", "stub", True, False

    def __init__(self) -> None:
        self.calls = []

    async def stream(self, system, messages):
        for piece in ("Merge keeps history. ", "**Rebase** rewrites it.\n"):
            yield piece


async def _wait_for_port(port: int, task: asyncio.Task, timeout: float = 120.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if task.done():
            raise RuntimeError("server stopped before it listened") from task.exception()
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_running_loop().time() >= deadline:
                raise
            await asyncio.sleep(0.05)


@pytest.fixture
async def running(unused_tcp_port, tmp_path, monkeypatch):
    for name in ("config", "data", "state", "runtime"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr("pilot.config.CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr("pilot.config.CONFIG_FILE", tmp_path / "config" / "config.toml")
    monkeypatch.setattr("pilot.config.RESTRICTIONS_FILE", tmp_path / "config" / "restrictions.toml")
    monkeypatch.setattr("pilot.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("pilot.config.STATE_DIR", tmp_path / "state")
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr("pilot.config.DB_FILE", tmp_path / "data" / "pilot.db")
    monkeypatch.setattr("pilot.config.LOG_FILE", tmp_path / "state" / "pilot.log")
    config = PilotConfig()
    config.server.host = "127.0.0.1"
    config.server.port = unused_tcp_port
    config.server.auth_token = "test-token"
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
        patch("pilot.agents.planner.Planner.plan", new_callable=AsyncMock) as planner,
    ):
        server = PilotServer(config)
        task = asyncio.create_task(server.start())
        await _wait_for_port(unused_tcp_port, task)
        from pilot.conversation import layer_for

        layer_for(server)._model = lambda: StubTalk()
        server._memory.record = AsyncMock(return_value=None)
        yield f"ws://127.0.0.1:{unused_tcp_port}", server, planner
        await server.stop()
        await task


@pytest.mark.asyncio
async def test_a_typed_question_answered_by_the_layer_leaves_the_chat_view_finished(running):
    uri, server, planner = running
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "a"}))
        assert json.loads(await ws.recv())["result"]["status"] == "authenticated"
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "execute",
                    "params": {"input": "What's the difference between merge and rebase?"},
                    "id": "q",
                }
            )
        )
        messages = []
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), 30))
            messages.append(message)
            if message.get("id") == "q":
                break
        await asyncio.sleep(0.2)
    planner.assert_not_awaited()
    methods = [m.get("method") for m in messages if "method" in m]
    statuses = [m["params"]["phase"] for m in messages if m.get("method") == "status"]
    assert statuses == ["receiving input"]  # once, and no later phase that would leave a spinner running
    assert methods.index("status") < methods.index("token_stream") < methods.index("conversation_response")
    assert "task_complete" in methods
    streamed = "".join(m["params"]["token"] for m in messages if m.get("method") == "token_stream")
    assert "**Rebase** rewrites it." in streamed  # the markdown reaches the chat view
    assert "result" in messages[-1], messages[-1]
    result = messages[-1]["result"]
    assert result["status"] == "success" and "**Rebase**" in result["message"]
    server._memory.record.assert_awaited_once()  # stored in the session history like a planner reply


@pytest.mark.asyncio
async def test_a_typed_request_handed_over_sends_the_first_status_once(running):
    uri, server, planner = running
    from pilot.actions import ActionPlan

    planner.return_value = ActionPlan(actions=[], explanation="Nothing to do.", raw_input="x")
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "a"}))
        await ws.recv()
        await ws.send(
            json.dumps(
                {"jsonrpc": "2.0", "method": "execute", "params": {"input": "Find the invoice from October"}, "id": "r"}
            )
        )
        messages = []
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), 30))
            messages.append(message)
            if message.get("id") == "r":
                break
    planner.assert_awaited()
    statuses = [m["params"]["phase"] for m in messages if m.get("method") == "status"]
    assert statuses.count("receiving input") == 1
