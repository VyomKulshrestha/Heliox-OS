"""Quiet while the person is in a call: how a call is recognised, and what Heliox does then."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import call_detection
from pilot.system.call_detection import MicUser, decide
from pilot.system.voice import ContinuousVoiceListener

OWN = [r"C:\Users\me\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe"]


def test_another_app_using_the_microphone_is_a_call_but_heliox_itself_is_not():
    teams = MicUser("MSTeams_8wekyb3d8bbwe", in_use=True)
    heliox = MicUser(r"C:#Users#me#AppData#Roaming#uv#python#cpython-3.13-windows-x86_64-none#python.exe", True)
    finished = MicUser(r"C:#Program Files#Zoom#bin#Zoom.exe", in_use=False)
    assert decide([heliox, finished], [], [], OWN) == ""
    assert decide([heliox, teams], [], [], OWN) == "MSTeams_8wekyb3d8bbwe"


@pytest.mark.parametrize(
    ("title", "app"),
    [
        ("Meet - abc-defg-hij - Google Chrome", "Google Meet"),
        ("Zoom Meeting", "Zoom"),
        ("Meeting with Priya | Microsoft Teams", "Microsoft Teams"),
        ("Call with Sam | Microsoft Teams", "Microsoft Teams"),
        ("WhatsApp - Voice call", "WhatsApp"),
        ("Huddle in #launch - Slack", "Slack"),
    ],
)
def test_call_windows_are_recognised(title, app):
    assert decide([], [], [title]) == app


@pytest.mark.parametrize(
    "title",
    ["Chat | Microsoft Teams", "Calls | Microsoft Teams", "X - Google Chrome", "Inbox - Outlook", "WhatsApp", "Slack"],
)
def test_ordinary_windows_are_not_calls(title):
    assert decide([], [], [title]) == ""


def test_a_zoom_meeting_process_is_a_call():
    assert decide([], ["explorer.exe", "CptHost.exe"], []) == "Zoom"
    assert decide([], ["Zoom.exe"], []) == ""  # Zoom open, no meeting


def test_when_nothing_can_be_read_it_is_not_a_call(monkeypatch):
    monkeypatch.setattr(call_detection, "_mic_users_linux", lambda _own: None)  # no pactl
    monkeypatch.setattr(call_detection, "_processes", lambda: [])
    monkeypatch.setattr(call_detection.sys, "platform", "linux")
    assert call_detection.current_call() == ""
    monkeypatch.setattr(call_detection.sys, "platform", "freebsd13")
    assert call_detection.current_call() == ""


@pytest.mark.asyncio
async def test_during_a_call_speech_is_not_taken_as_addressed_to_heliox():
    dispatched: list[str] = []
    overheard: list[str] = []

    async def dispatch(text):
        dispatched.append(text)

    async def overhear(text):
        overheard.append(text)

    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"], on_command=dispatch, on_overheard=overhear, config=PilotConfig()
    )
    listener._wake_calibrator = MagicMock()
    listener.in_call = "Microsoft Teams"
    listener.arm_follow_up_window(30)
    queue = iter(["Hey Heliox, open my mail", "Skip the backup.", "the launch is Monday the twelfth of November"])

    async def transcribe(**_kwargs):
        try:
            return next(queue)
        except StopIteration:
            listener._running = False
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True
    await listener._listen_loop()
    await asyncio.gather(*tuple(listener._command_tasks))
    assert dispatched == [] and overheard == []


@pytest.mark.asyncio
async def test_during_a_call_heliox_says_nothing():
    server = PilotServer(PilotConfig())
    listener = MagicMock()
    listener.is_running = True
    listener.in_call = "Zoom"
    server._voice_listener = listener
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.speak = AsyncMock()
    outcome = await server._speak_companion_text("The twelfth of November is a Thursday. You said Monday.")
    assert outcome.status == "suppressed"
    server._speech_coordinator.speak.assert_not_awaited()
    server.config.voice.quiet_in_calls = False
    server._speech_coordinator.speak = AsyncMock(return_value=MagicMock(status="spoken"))
    listener.wait_for_floor = AsyncMock(return_value=0.0)
    await server._speak_companion_text("Hello")
    server._speech_coordinator.speak.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_voice_bar_is_told_when_a_call_starts_and_ends(monkeypatch):
    statuses: list[tuple[str, dict]] = []

    async def on_status(status, data):
        statuses.append((status, data))

    listener = ContinuousVoiceListener(wake_words=["hey heliox"], on_status=on_status, config=PilotConfig())
    answers = iter(["Zoom", "Zoom", ""])
    monkeypatch.setattr(call_detection, "current_call", lambda: next(answers, ""))
    listener._running = True
    task = asyncio.create_task(listener._watch_calls(every_seconds=0.01))
    await asyncio.sleep(0.1)
    listener._running = False
    await task
    assert statuses[0] == ("in_call", {"app": "Zoom", "message": "Quiet while you're in a call"})
    assert statuses[1] == ("call_ended", {})


def test_an_app_path_in_either_form_is_named_by_its_program():
    assert decide([MicUser("C:#Program Files#Zoom#bin#Zoom.exe", True)], [], [], OWN) == "Zoom"
    assert decide([MicUser(r"C:\Program Files\Zoom\bin\Zoom.exe", True)], [], [], OWN) == "Zoom"
