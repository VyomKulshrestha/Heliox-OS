"""Speech nobody heard is not "spoken": with a failed or absent voice on each system, the coordinator's outcome is
"no_voice"; the server opens no follow-up window, shows the reply as text and says what to check once; speak_text
returns the failure; Live opens no reply window; list_voices finds espeak-ng and reports a failed listing. Fake
system programs only: nothing is spoken."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.system import voice
from pilot.system.companion_speech import CompanionSpeechCoordinator, SpeechChannel, SpeechOutcome
from pilot.system.platform_detect import Platform


class Proc:
    def __init__(self, code: int, out: bytes = b"", err: bytes = b"") -> None:
        self.returncode, self._out, self._err = code, out, err

    async def communicate(self):
        return self._out, self._err

    def kill(self):
        return None


@pytest.fixture
def os_voice(monkeypatch):
    """speak() goes straight to the system's own voice (no local voice), on the system the test picks."""
    config = PilotConfig()
    config.voice.tts_engine = "os_native"
    monkeypatch.setattr(voice.PilotConfig, "load", staticmethod(lambda *a, **k: config))
    monkeypatch.setattr(voice, "_SAY_WARNED", False)

    def on(system: Platform):
        monkeypatch.setattr(voice, "CURRENT_PLATFORM", system)

    return on


def failing_programs(monkeypatch, code: int = 1):
    async def create(*cmd, **_kwargs):
        return Proc(code, err=b"audio device busy\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    async def powershell(_script):
        return code, "", "No voice installed"

    monkeypatch.setattr(voice, "run_powershell", powershell)


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [Platform.WINDOWS, Platform.MACOS, Platform.LINUX])
async def test_a_failing_voice_is_no_voice_not_spoken_on_every_system(system, os_voice, monkeypatch):
    os_voice(system)
    failing_programs(monkeypatch)
    monkeypatch.setattr("pilot.system.linux_voice.espeak_program", lambda: "espeak-ng")
    coordinator = CompanionSpeechCoordinator()
    outcome = await coordinator.speak("The backup is done.", channel=SpeechChannel.FINAL_ANSWER)
    await coordinator.close()
    assert outcome.status == "no_voice"
    assert "Heliox's voice did not work" in outcome.message and "Check" in outcome.message


@pytest.mark.asyncio
async def test_no_voice_installed_on_linux_is_no_voice(os_voice, monkeypatch):
    os_voice(Platform.LINUX)
    monkeypatch.setattr("pilot.system.linux_voice.espeak_program", lambda: None)
    coordinator = CompanionSpeechCoordinator()
    outcome = await coordinator.speak("Hello.", channel=SpeechChannel.FINAL_ANSWER)
    await coordinator.close()
    assert outcome.status == "no_voice" and "espeak-ng" in outcome.message


@pytest.mark.asyncio
async def test_a_failing_voice_while_listening_is_no_voice_not_spoken(os_voice, monkeypatch):
    os_voice(Platform.MACOS)
    failing_programs(monkeypatch)
    watcher = MagicMock()
    watcher.is_active = True

    async def never(timeout=None):
        await asyncio.sleep(10)

    watcher.wait_for_speech_start = never
    with pytest.raises(voice.SpeechUnheard):
        await voice.speak_interruptible("Hello.", recorder=watcher)


@pytest.mark.asyncio
async def test_a_working_voice_is_still_spoken(os_voice, monkeypatch):
    os_voice(Platform.MACOS)
    failing_programs(monkeypatch, code=0)
    coordinator = CompanionSpeechCoordinator()
    outcome = await coordinator.speak("Hello.", channel=SpeechChannel.FINAL_ANSWER)
    await coordinator.close()
    assert outcome.status == "spoken"


def server_with(outcomes: list[SpeechOutcome]):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    listener = MagicMock()
    listener.is_running = True
    listener.in_call = ""
    listener.wait_for_floor = AsyncMock(return_value=0.0)
    server._voice_listener = listener
    server.config.voice.barge_in_enabled = False
    server._speech_coordinator = SimpleNamespace(speak=AsyncMock(side_effect=outcomes))
    return server, listener


def failure_statuses(server):
    return [
        call.args[1]
        for call in server._broadcast_notification.await_args_list
        if call.args[0] == "voice_status" and call.args[1].get("status") == "voice_output_failed"
    ]


@pytest.mark.asyncio
async def test_after_speech_nobody_heard_no_window_opens_and_the_reply_is_shown_with_one_line_said_once():
    reason = "Heliox's voice did not work, so its replies are shown here as text. Check the output device."
    unheard = SpeechOutcome("no_voice", SpeechChannel.FINAL_ANSWER, reason)
    spoken = SpeechOutcome("spoken", SpeechChannel.FINAL_ANSWER, "Speech completed")
    server, listener = server_with([unheard, unheard, spoken, unheard])
    await server._speak_companion_text("Which file, the blue one?", channel=SpeechChannel.FINAL_ANSWER)
    await server._speak_companion_text("It is done.", channel=SpeechChannel.FINAL_ANSWER)
    listener.arm_follow_up_window.assert_not_called()  # nothing heard, so nothing to reply to
    failed = failure_statuses(server)
    assert [f["text"] for f in failed] == ["Which file, the blue one?", "It is done."]
    assert [f["message"] for f in failed] == [reason, ""]  # said once
    await server._speak_companion_text("Back again.", channel=SpeechChannel.FINAL_ANSWER)  # the voice works again
    listener.arm_follow_up_window.assert_called_once()
    await server._speak_companion_text("Gone again.", channel=SpeechChannel.FINAL_ANSWER)
    assert failure_statuses(server)[-1]["message"] == reason  # broken again after working: said again


@pytest.mark.asyncio
async def test_speak_text_returns_the_failure():
    reason = "Heliox's voice did not work, so its replies are shown here as text. Check the output device."
    server, _listener = server_with([SpeechOutcome("no_voice", SpeechChannel.FINAL_ANSWER, reason)])
    result = await server._handle_speak_text({"text": "Hello."}, None)
    assert result == {"status": "no_voice", "message": reason}


@pytest.mark.asyncio
async def test_live_opens_no_reply_window_after_a_remark_nobody_heard():
    from pilot.personal.wiring import live_speak

    listener = SimpleNamespace(is_running=False, note_live_spoke=MagicMock())

    async def speak(_text, *, channel, dedupe_key):
        return SpeechOutcome("no_voice", SpeechChannel(str(channel)), "no voice")

    server = SimpleNamespace(_voice_listener=listener, _live=None, _speak_companion_text=speak)
    assert await live_speak(server, "The nineteenth is a Thursday.", False) == "no_voice"
    listener.note_live_spoke.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("found", "code", "expect"),
    [("espeak-ng", 0, "en-gb"), (None, 0, "Error: TTS unavailable"), ("espeak-ng", 2, "Error: espeak-ng exited")],
)
async def test_list_voices_finds_espeak_ng_and_reports_a_failed_listing(found, code, expect, monkeypatch):
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr("pilot.system.linux_voice.espeak_program", lambda: found)
    ran: list[tuple] = []

    async def create(*cmd, **_kwargs):
        ran.append(cmd)
        return Proc(code, out=b"en-gb\n", err=b"no such voice\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    result = await voice.list_voices()
    assert result.startswith(expect)
    if found:
        assert ran[0][0] == "espeak-ng"  # not a hard-coded "espeak"
