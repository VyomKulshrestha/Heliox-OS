"""The local voice starts loading at the first honest sign of use: Heliox's own window says it gained focus or that
typing began in its box (voice_prepare), or the person answers a Live card. Nothing on this path reads other programs:
no front-window title, no focused text. Fakes only: no model, no window system."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import pilot.server as server_module
from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import local_tts_worker


def server(engine: str = "kokoro_tts") -> PilotServer:
    config = PilotConfig()
    config.save = lambda *a, **k: None
    config.voice.tts_engine = engine
    item = PilotServer(config)
    item.prewarmed = 0

    def prewarm(*, model: bool = False) -> None:
        item.prewarmed += 1

    item._prewarm_voice = prewarm
    return item


@pytest.fixture
def released(monkeypatch):
    voice_worker = SimpleNamespace(running=False, failing=False)
    monkeypatch.setattr(local_tts_worker, "manager", lambda: voice_worker)
    return voice_worker


@pytest.fixture
def no_platform_readers(monkeypatch):
    """Every reader of other programs fails the test if it is called."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a platform reader was called")

    monkeypatch.setattr("pilot.agents.screen_vision._get_active_window", forbidden)
    monkeypatch.setattr("pilot.agents.screen_vision._get_active_window_windows", forbidden)
    monkeypatch.setattr("pilot.agents.screen_vision._get_active_window_macos", forbidden)
    monkeypatch.setattr("pilot.agents.screen_vision._get_active_window_linux", forbidden)


@pytest.mark.asyncio
async def test_the_window_gaining_focus_or_typing_loads_the_released_voice(released, no_platform_readers, monkeypatch):
    item = server()
    clock = {"now": 100.0}
    monkeypatch.setattr(server_module.time, "monotonic", lambda: clock["now"])
    assert await item._handle_voice_prepare({"reason": "focus"}, None) == {"status": "ok", "started": True}
    clock["now"] += 0.5
    assert (await item._handle_voice_prepare({"reason": "typing"}, None))["started"] is False  # a repeat: ignored
    clock["now"] += 3
    assert (await item._handle_voice_prepare({"reason": "typing"}, None))["started"] is True
    assert item.prewarmed == 2


@pytest.mark.asyncio
async def test_harmless_from_another_or_a_stale_window(released, no_platform_readers):
    item = server()
    assert (await item._handle_voice_prepare({"reason": "click"}, None))["status"] == "error"
    assert (await item._handle_voice_prepare({}, None))["status"] == "error"
    released.running = True  # already loaded: nothing to do
    assert (await item._handle_voice_prepare({"reason": "focus"}, None)) == {"status": "ok", "started": False}
    assert item.prewarmed == 0


@pytest.mark.asyncio
async def test_the_system_voice_needs_no_loading(released, no_platform_readers):
    item = server("os_native")
    assert (await item._handle_voice_prepare({"reason": "focus"}, None))["started"] is False
    assert item.prewarmed == 0


@pytest.mark.asyncio
async def test_answering_a_live_card_loads_the_voice():
    item = server()

    class Live:
        async def feedback(self, card_id, verdict, from_person=False):
            return {"status": "ok"}

    item._live = Live()
    assert (await item._handle_live_feedback({"id": "c1", "verdict": "helpful"}, None))["status"] == "ok"
    assert item.prewarmed == 1
    await item._handle_live_feedback({"id": "c1", "verdict": "nonsense"}, None)
    assert item.prewarmed == 1  # a refused request is not a sign of use


def test_nothing_on_this_path_reads_other_programs():
    source = inspect.getsource(PilotServer._handle_voice_prepare)
    for reader in ("screen_vision", "_get_active_window", "focused_text", "window_title"):
        assert reader not in source
    assert not hasattr(PilotServer, "_watch_for_use")  # no polling of the front window
    assert "voice_prepare" in inspect.getsource(PilotServer)
