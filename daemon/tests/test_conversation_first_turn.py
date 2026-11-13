"""The first spoken turn of the day pays for nothing the later ones do not (the bench, 13 November: the screen read
took 0.55 to 0.75 s and the code handlers 0.25 s on the first model turn after start-up, 0.00 on every later one).

The screen is read from the wake phrase on, the request never waits for it beyond SCREEN_WAIT_SECONDS, the focused
window's module is imported off the event loop, and the modules the code handlers import on first use are imported
when the listener starts."""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from pilot.conversation import layer as layer_module
from pilot.conversation import trace
from pilot.conversation.context import ContextSources, Screen
from pilot.conversation.fast_model import HttpFastModel
from pilot.conversation.layer import ConversationLayer

QUESTION = "How long does pasta take to cook?"


def _model(seen: list) -> HttpFastModel:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if not body.get("stream"):  # the warm-up at the wake phrase, not the reply
            return httpx.Response(404)
        seen.append(body)
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "About ten minutes. "}}
        return httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "x"})

    async def key() -> str:
        return "k"

    return HttpFastModel(
        provider="claude",
        name="claude-haiku-5-5",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
    )


async def _ear(text: str, channel: str, key: str) -> str:
    return "spoken"


def _layer(model: HttpFastModel, read) -> ConversationLayer:
    return ConversationLayer(sink=_ear, model=lambda: model, hold_after=60.0, sources=ContextSources(screen_async=read))


async def test_the_screen_read_started_at_the_wake_phrase_is_used_by_the_turn():
    reads: list[float] = []

    async def read(local: bool):
        reads.append(time.monotonic())
        await asyncio.sleep(0.3)  # slower than the turn may wait, but started early enough
        return Screen("Notepad", "the shopping list")

    seen: list = []
    layer = _layer(_model(seen), read)
    layer.prewarm()  # the wake phrase
    await asyncio.sleep(0.35)  # the person is still talking
    await layer.take(QUESTION)
    assert len(reads) == 1  # read once, at the wake phrase
    assert "the shopping list" in json.dumps(seen[0])


async def test_a_slow_screen_never_holds_the_request_beyond_the_bound(monkeypatch):
    async def read(local: bool):
        await asyncio.sleep(5)
        return Screen("Notepad", "never in time")

    stamps: list[tuple[str, dict]] = []
    monkeypatch.setattr(trace, "listener", lambda name, at, **data: stamps.append((name, data)))
    trace.begin()
    seen: list = []
    started = time.monotonic()
    await _layer(_model(seen), read).take(QUESTION)
    assert time.monotonic() - started < layer_module.SCREEN_WAIT_SECONDS + 0.5
    assert "never in time" not in json.dumps(seen[0])  # sent without it
    context = dict(stamps)["context ready"]
    assert context["screen_ms"] <= layer_module.SCREEN_WAIT_SECONDS * 1000 + 100


async def test_a_read_from_long_before_the_turn_is_not_used(monkeypatch):
    now = [1000.0]
    texts = iter(["the old window", "the window now"])

    async def read(local: bool):
        return Screen("Notepad", next(texts))

    seen: list = []
    layer = _layer(_model(seen), read)
    layer._clock = lambda: now[0]
    layer.prewarm()
    await asyncio.sleep(0.01)
    now[0] += layer_module.SCREEN_FRESH_SECONDS + 1
    await layer.take(QUESTION)
    sent = json.dumps(seen[0])
    assert "the window now" in sent and "the old window" not in sent


async def test_the_focused_window_is_read_off_the_event_loop(monkeypatch):
    from pilot.conversation import adapters

    threads: list[str] = []

    def active() -> tuple[str, str]:
        import threading

        threads.append(threading.current_thread().name)
        return "Notepad", "notes.txt"

    monkeypatch.setattr(adapters, "_active_window", active)
    server = type("S", (), {"_live": None, "config": None})()
    focus = await adapters.FocusSource(server).focus()
    assert focus is not None and focus.app == "Notepad"
    assert threads and threads[0] != "MainThread"


async def test_the_code_handlers_modules_are_imported_when_the_listener_starts(monkeypatch):
    """Imports only, read from the handlers' source: no handler is called."""
    import importlib

    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    imported: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))
    await server._warm_code_handlers()
    assert "pilot.agents.screen_vision" in imported  # the screen read's
    assert "pilot.workflows.user_profile" in imported  # read from _profile_request's own source
    assert all(name.startswith("pilot.") for name in imported)
