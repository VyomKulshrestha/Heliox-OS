"""Health's ready {answers, hears, speaks}: the one readiness answer, each part from its own owner's flag.

answers from start-up, hears from the listener's own "hears" (get_stats) and its microphone fault, speaks from the
local voice's state. Each part is checked through the real health handler, with the parts it reads put in each state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import local_tts_worker


def _listener(*, hears: bool, notice: str = "", running: bool = True) -> SimpleNamespace:
    return SimpleNamespace(is_running=running, capture_notice=notice, get_stats=lambda: {"hears": hears, "warm_ms": {}})


async def _ready(server: PilotServer) -> dict:
    return (await server._handle_health({}, MagicMock()))["ready"]


@pytest.fixture
def server(monkeypatch):
    server = PilotServer(PilotConfig())
    server.config.voice.tts_engine = "os_native"  # the system voice: no load
    return server


@pytest.mark.asyncio
async def test_while_starting_it_does_not_answer_and_says_the_stage(server):
    server._startup = SimpleNamespace(is_ready=False, snapshot=lambda: {"stage": "opening Heliox's records"})
    answers = (await _ready(server))["answers"]
    assert answers == {"ok": False, "state": "starting", "reason": "Starting: opening Heliox's records"}

    server._startup = SimpleNamespace(is_ready=True, snapshot=lambda: {"stage": ""})
    assert (await _ready(server))["answers"] == {"ok": True, "state": "on", "reason": ""}


@pytest.mark.asyncio
async def test_hears_follows_the_listeners_own_flag_and_a_microphone_fault_wins(server):
    server._voice_listener = None
    assert (await _ready(server))["hears"]["state"] == "off"  # no microphone, or listening turned off

    server._voice_listener = _listener(hears=False)
    assert (await _ready(server))["hears"] == {"ok": False, "state": "loading", "reason": "Getting ready to listen…"}

    fault = "The microphone stopped sending sound; reopening it."
    server._voice_listener = _listener(hears=False, notice=fault)  # a fault while still loading
    assert (await _ready(server))["hears"] == {"ok": False, "state": "fault", "reason": fault}

    server._voice_listener = _listener(hears=True)
    assert (await _ready(server))["hears"] == {"ok": True, "state": "on", "reason": ""}


@pytest.mark.asyncio
async def test_speaks_the_system_voice_always_a_local_voice_once_loaded(server, monkeypatch):
    assert (await _ready(server))["speaks"] == {"ok": True, "state": "on", "reason": ""}

    server.config.voice.tts_engine = "kokoro_tts"
    states = iter(
        [
            {"voice_loaded": False, "voice_loading": True},
            {"voice_loaded": False, "voice_loading": False},
            {"voice_loaded": False, "voice_loading": False},
            {"voice_loaded": True, "voice_loading": False},
        ]
    )
    monkeypatch.setattr(local_tts_worker, "local_voice_state", lambda _config: next(states))
    manager = SimpleNamespace(_failed_until=None)
    monkeypatch.setattr(local_tts_worker, "_manager", manager)

    assert (await _ready(server))["speaks"]["state"] == "loading"
    assert (await _ready(server))["speaks"] == {
        "ok": False,
        "state": "not_loaded",
        "reason": "Heliox's voice loads when it first speaks.",
    }
    manager._failed_until = 123.0  # the local voice failed: the system voice speaks meanwhile
    assert (await _ready(server))["speaks"]["state"] == "system_voice"
    assert (await _ready(server))["speaks"] == {"ok": True, "state": "on", "reason": ""}
