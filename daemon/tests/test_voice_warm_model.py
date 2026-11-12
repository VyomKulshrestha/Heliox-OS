"""The listener asks for the cloud model's connection to be opened while the person is still talking, but only for
speech known to be for Heliox (the wake phrase read from the opening, the microphone button, an open follow-up),
never for speech that is not, and never by waiting on it. Fakes only: no models, no audio device."""

from __future__ import annotations

import asyncio
import time

import pytest

from pilot.config import PilotConfig
from pilot.system import voice
from pilot.system.voice import ContinuousVoiceListener


class Speculation:
    def __init__(self, _listener) -> None:
        self.started = 0

    def start(self, _frames) -> None:
        self.started += 1

    def cancel(self) -> None:
        return None

    def take(self):
        return None


def listener_with(monkeypatch, opening_words: str = "") -> tuple[ContinuousVoiceListener, list[str]]:
    monkeypatch.setattr(voice, "_Speculation", Speculation)
    monkeypatch.setattr(voice, "_recognisers_running", lambda: False)
    monkeypatch.setattr(voice, "_write_wav", lambda _frames, _rate: "opening.wav")
    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    listener._loop = asyncio.get_running_loop()
    listener._parakeet = lambda: object()

    async def read_opening(_path, language="en"):
        return opening_words

    listener._transcribe_whisper_text = read_opening
    warmed: list[str] = []

    def warm(addressed: str) -> bool:
        warmed.append(addressed)
        return True

    listener.warm_model = warm
    return listener, warmed


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_the_wake_phrase_read_from_the_opening_warms_the_model_at_once(monkeypatch):
    listener, warmed = listener_with(monkeypatch, "Hey Heliox, remind me")
    listener._start_early_wake([b"frames"])
    await settle()
    assert warmed == ["wake"]  # 1.6 s in, while the person is still talking
    speculation = listener._speculate_final_pass()
    listener._recorder.on_candidate_end([b"frames"])
    await settle()
    assert speculation.started == 1 and warmed == ["wake", "wake"]  # again at the end-of-speech candidate


@pytest.mark.asyncio
async def test_speech_not_for_heliox_never_warms_anything(monkeypatch):
    listener, warmed = listener_with(monkeypatch, "so I told her the meeting")
    listener.task_open = lambda: False
    listener._start_early_wake([b"frames"])
    await settle()
    speculation = listener._speculate_final_pass()
    listener._recorder.on_candidate_end([b"frames"])
    await settle()
    assert speculation.started == 1  # the final pass still starts as before
    assert warmed == []


@pytest.mark.asyncio
async def test_an_open_follow_up_or_a_running_task_warms_at_the_candidate_end_only(monkeypatch):
    listener, warmed = listener_with(monkeypatch, "and the one on the desktop")
    listener._follow_up_until = time.monotonic() + 10
    listener._start_early_wake([b"frames"])
    await settle()
    assert warmed == []  # the opening read warms only for the wake phrase
    listener._speculate_final_pass()
    listener._recorder.on_candidate_end([b"frames"])
    await settle()
    assert warmed == ["follow_up"]

    listener._follow_up_until = 0.0
    listener.task_open = lambda: True
    listener._recorder.on_candidate_end([b"frames"])
    await settle()
    assert warmed == ["follow_up", "follow_up"]


@pytest.mark.asyncio
async def test_while_heliox_speaks_a_follow_up_window_does_not_warm(monkeypatch):
    listener, warmed = listener_with(monkeypatch)
    listener._follow_up_until = time.monotonic() + 10
    listener._wake_free_suppression_depth = 1  # possibly Heliox's own voice
    listener._speculate_final_pass()
    listener._recorder.on_candidate_end([b"frames"])
    await settle()
    assert warmed == []


@pytest.mark.asyncio
async def test_the_microphone_button_warms_as_button(monkeypatch):
    listener, warmed = listener_with(monkeypatch)
    listener.note_interruption(by_button=True)
    listener._speculate_final_pass()
    listener._recorder.on_candidate_end([b"frames"])
    await settle()
    assert warmed == ["button"]


@pytest.mark.asyncio
async def test_the_warm_up_is_never_waited_for_and_its_failure_changes_nothing(monkeypatch):
    listener, _warmed = listener_with(monkeypatch, "Hey Heliox")
    calls: list[str] = []

    def failing(addressed: str) -> bool:
        calls.append(addressed)
        raise RuntimeError("no network")

    listener.warm_model = failing
    listener._early_wake = asyncio.get_running_loop().create_future()
    listener._early_wake.set_result("Hey Heliox")
    speculation = listener._speculate_final_pass()
    listener._recorder.on_candidate_end([b"frames"])
    assert calls == []  # handed to the loop, not called inside the recorder's callback
    assert speculation.started == 1
    await settle()
    assert calls == ["wake"]
