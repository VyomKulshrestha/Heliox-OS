import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


@pytest.mark.asyncio
async def test_speak_text_uses_daemon_voice_engine(monkeypatch):
    speak = AsyncMock(return_value="Spoken: hello...")
    monkeypatch.setattr("pilot.system.voice.speak", speak)
    server = PilotServer(PilotConfig())

    result = await server._handle_speak_text({"text": "  hello  "}, MagicMock())

    speak.assert_awaited_once_with("hello")
    assert result == {"status": "spoken", "message": "Spoken: hello..."}


@pytest.mark.asyncio
async def test_superseded_speech_still_returns_a_terminal_rpc_response(monkeypatch):
    speak = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr("pilot.system.voice.speak", speak)
    server = PilotServer(PilotConfig())

    result = await server._handle_speak_text({"text": "old message"}, MagicMock())

    assert result == {"status": "cancelled", "message": "Speech superseded"}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " ", None, 42])
async def test_speak_text_rejects_invalid_text(text):
    server = PilotServer(PilotConfig())

    result = await server._handle_speak_text({"text": text}, MagicMock())

    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_speak_text_rejects_oversized_text():
    server = PilotServer(PilotConfig())

    result = await server._handle_speak_text({"text": "x" * 4001}, MagicMock())

    assert result["status"] == "error"
    assert "4000" in result["message"]


@pytest.mark.asyncio
async def test_stop_speech_stops_daemon_playback(monkeypatch):
    stop_speaking = AsyncMock(return_value="Speech stopped")
    monkeypatch.setattr("pilot.system.voice.stop_speaking", stop_speaking)
    server = PilotServer(PilotConfig())

    result = await server._handle_stop_speech({}, MagicMock())

    stop_speaking.assert_awaited_once_with()
    assert result == {"status": "stopped", "message": "Speech stopped"}
