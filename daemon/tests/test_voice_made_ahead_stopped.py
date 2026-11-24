"""The fixed sentences made ahead are a task the daemon starts (make_fixed_speech_ahead, when the voice is warmed);
the daemon's stop cancels it and waits for it, so none is left pending when the event loop closes ("Task was
destroyed but it is pending"). Fakes only: nothing is made, no worker starts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import fixed_speech, local_tts_worker


@pytest.fixture
def making_waits(monkeypatch):
    """The store is there and making ahead waits until it is cancelled; counts how often it was asked."""
    asked: list[str] = []
    monkeypatch.setattr(local_tts_worker._manager, "head_cache", SimpleNamespace())
    monkeypatch.setattr(fixed_speech, "gather", lambda: ["Okay."])

    async def make_ahead(engine, voice, pieces, **_kwargs):
        asked.append(voice)
        await asyncio.Event().wait()

    monkeypatch.setattr(local_tts_worker._manager, "make_ahead", make_ahead)
    return asked


async def test_stopping_cancels_the_task_and_waits_for_it(making_waits):
    task = local_tts_worker.make_fixed_speech_ahead(SimpleNamespace(tts_engine="kokoro_tts", tts_voice="af_heart"))
    assert task is not None
    for _ in range(20):
        if making_waits:
            break
        await asyncio.sleep(0)
    assert making_waits == ["af_heart"] and not task.done()
    await local_tts_worker.stop_making_ahead()
    assert task.cancelled() and local_tts_worker._making_ahead is None
    await local_tts_worker.stop_making_ahead()  # nothing left: a second stop does nothing


async def test_the_daemon_s_stop_ends_the_task_its_warm_up_started(making_waits, monkeypatch):
    monkeypatch.setattr(local_tts_worker, "prewarm_local_voice", lambda voice: None)
    server = PilotServer(PilotConfig())
    server.config.voice.tts_engine, server.config.voice.tts_voice = "kokoro_tts", "af_heart"
    server._prewarm_voice()
    started = local_tts_worker._making_ahead
    assert started is not None and not started[1].done()
    await server.stop()
    assert started[1].done() and local_tts_worker._making_ahead is None
