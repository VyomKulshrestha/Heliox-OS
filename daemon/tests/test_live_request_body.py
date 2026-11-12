"""What a cloud model is actually sent about the focused field, captured at the HTTP request.

The conversation layer is built the way the daemon builds it (``pilot.conversation.adapters.layer_for``), with the
real Claude client from ``build_fast_model``; only the network is replaced, by an ``httpx.MockTransport`` standing in
for the provider. It records each request body and streams back a short reply. Live's last look at the focused field
holds a marker. With ``share_screen_text`` on auto and a cloud model, the marker must not be in any body; with "on" it
must be, which shows the capture can see it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.live.engine import Observation

MARK = "HLXMARK7731 the quarterly numbers draft"


def claude_stream(text: str) -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def build(monkeypatch, share: str):
    bodies: list[dict] = []

    def provider(request: httpx.Request) -> httpx.Response:
        bodies.append({"url": str(request.url), "body": json.loads(request.content or b"{}")})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=claude_stream("Sure."))

    import pilot.system.http_client as http_client

    monkeypatch.setattr(
        http_client, "create_httpx_client", lambda *_a, **_k: httpx.AsyncClient(transport=httpx.MockTransport(provider))
    )
    config = PilotConfig()
    config.model.provider = "cloud"
    config.model.cloud_provider = "claude"
    config.live.enabled = True
    config.conversation.share_screen_text = share

    async def get_key(_name):
        return "sk-ant-test-not-a-real-key"

    async def speak(_text, **_kwargs):
        return SimpleNamespace(status="spoken")

    obs = Observation(
        text=f"Notes for Friday. {MARK}", app="notepad.exe", title="Draft - Notepad", source="focused_field"
    )
    server = SimpleNamespace(
        config=config,
        _vault=SimpleNamespace(get_key=get_key),
        _model_router=object(),
        _budget_tracker=None,
        _live=SimpleNamespace(_last_obs=obs),
        _voice_listener=None,
        _speak_companion_text=speak,
    )
    from pilot.conversation.adapters import layer_for

    return layer_for(server), bodies


async def ask(layer) -> None:
    await layer.take("what do you think of what I'm writing", source="voice")


def sent(bodies) -> str:
    return json.dumps([b["body"] for b in bodies if "anthropic" in b["url"]])


@pytest.mark.asyncio
async def test_with_share_on_auto_a_cloud_model_is_never_sent_the_field_text(monkeypatch):
    layer, bodies = build(monkeypatch, "auto")
    await ask(layer)
    assert any("anthropic" in b["url"] for b in bodies), "no request reached the stand-in provider"
    assert "HLXMARK7731" not in sent(bodies)
    assert "Notepad" in sent(bodies)  # the app's display name is what it gets


@pytest.mark.asyncio
async def test_the_capture_sees_the_field_text_when_sharing_is_on(monkeypatch):
    layer, bodies = build(monkeypatch, "on")
    await ask(layer)
    assert "HLXMARK7731" in sent(bodies)
