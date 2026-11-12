"""The microphone survives a real desk: a stream that fails or stalls, a call app taking the device, the default
microphone changing, a configured microphone coming back, and a muted device sending pure zeros. Fake capture
backends only; no audio packages are needed."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.system import audio_health
from pilot.system.audio_health import CaptureState, CaptureWatchdog


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def state(frames: int, **changes) -> CaptureState:
    values = {"frames": frames, "error": "", "zero_seconds": 0.0, "active": True, "default_name": "USB mic"}
    values.update(changes)
    return CaptureState(**values)


def test_a_stream_that_stops_delivering_is_reopened_after_three_seconds():
    clock = Clock()
    dog = CaptureWatchdog(clock)
    dog.opened("USB mic")
    assert dog.check(state(10)).action == "ok"
    clock.now += 2.9
    assert dog.check(state(10)).action == "ok"
    clock.now += 0.2
    decision = dog.check(state(10))
    assert (decision.action, decision.reason) == ("reopen", "no_audio")
    assert "stopped sending sound" in decision.message


def test_a_failed_stream_is_reopened_at_once_and_recovery_is_reported_once():
    clock = Clock()
    dog = CaptureWatchdog(clock)
    dog.opened("USB mic")
    dog.check(state(10))
    decision = dog.check(state(11, error="AUDCLNT_E_DEVICE_INVALIDATED"))  # a call app took the headset
    assert (decision.action, decision.reason) == ("reopen", "stream_failed")
    dog.reopen_succeeded("USB mic")
    assert not dog.recovered(state(12))  # not yet: hardly any sound since
    assert dog.recovered(state(20))
    assert not dog.recovered(state(30))
    assert dog.problem == ""


def test_the_default_microphone_is_followed_only_when_the_setting_is_auto():
    clock = Clock()
    dog = CaptureWatchdog(clock)
    dog.opened("USB mic")
    assert dog.check(state(1, default_name="Headset (Jabra)", auto=False)).action == "ok"
    decision = dog.check(state(2, default_name="Headset (Jabra)"))
    assert (decision.action, decision.reason) == ("reopen", "default_changed")
    assert "Headset (Jabra)" in decision.message
    dog.reopen_succeeded("Headset (Jabra)")
    assert dog.check(state(3, default_name="Headset (Jabra)")).action == "ok"
    assert dog.check(state(4, default_name="")).action == "ok"  # unknown default: no switching


def test_the_configured_microphone_is_taken_back_when_it_returns():
    dog = CaptureWatchdog(Clock())
    dog.opened("Laptop mic")
    decision = dog.check(state(1, auto=False, using_fallback=True, configured_available=True))
    assert (decision.action, decision.reason) == ("reopen", "device_returned")


def test_pure_zeros_are_reported_then_retried_every_thirty_seconds():
    clock = Clock()
    dog = CaptureWatchdog(clock)
    dog.opened("USB mic")
    frames = 0
    actions = []
    for _ in range(70):
        clock.now += 1.0
        frames += 20
        zero = max(0.0, clock.now - 105.0)  # silent from 5 s in
        actions.append(dog.check(state(frames, zero_seconds=zero)).action)
    assert actions.count("notice") == 1
    assert actions.count("reopen") == 2  # cheap, and sometimes wins the device back from another app
    assert dog.problem == "silent"
    assert dog.recovered(state(frames + 50, zero_seconds=0.0))


def test_reopening_backs_off_while_it_brings_no_sound():
    clock = Clock()
    dog = CaptureWatchdog(clock)
    dog.opened("USB mic")
    reopened_at = []
    for _ in range(1200):  # two minutes of a dead device, checked every 0.1 s
        clock.now += 0.1
        if dog.check(state(5)).action == "reopen":
            reopened_at.append(round(clock.now - 100.0, 1))
            dog.reopen_succeeded("USB mic")  # it opens, but nothing comes
    gaps = [b - a for a, b in zip(reopened_at, reopened_at[1:], strict=False)]
    assert gaps[0] == pytest.approx(3.0, abs=0.15)
    assert max(gaps) >= 29.0  # settles at one try every 30 s
    assert len(reopened_at) <= 10


def test_device_probes_are_empty_without_the_audio_packages(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "soundcard", None)
    assert audio_health.default_microphone_name() == ""
    assert audio_health.default_speaker_name() == ""
    assert not audio_health.microphone_present("USB mic")


# ── the recorder ──


def test_the_recorder_measures_runs_of_exact_zeros():
    from pilot.system.voice import _ContinuousRecorder

    recorder = _ContinuousRecorder()
    recorder._publish_audio_block(np.zeros(800, dtype=np.int16))
    assert recorder.zero_seconds >= 0.0 and recorder._zeros_since is not None
    recorder._publish_audio_block(np.full(800, 3, dtype=np.int16))  # one LSB of hiss is signal
    assert recorder.zero_seconds == 0.0


@pytest.mark.asyncio
async def test_reopening_keeps_the_queue_so_a_waiting_utterance_carries_on(monkeypatch):
    from pilot.system import voice

    recorder = voice._ContinuousRecorder()
    opened = []

    def fake_open() -> bool:
        opened.append(True)
        recorder._stream = object()
        return True

    monkeypatch.setattr(recorder, "_open_capture", fake_open)
    assert recorder.start()
    queue = recorder._queue
    assert recorder.reopen()
    assert recorder._queue is queue and queue.empty()  # no stop sentinel: the wait is not ended
    assert len(opened) == 2


# ── the listener ──


class FakeRecorder:
    """A capture backend that can die and come back."""

    def __init__(self) -> None:
        self.alive = True
        self._frames = 0
        self.last_error = ""
        self.zero_seconds = 0.0
        self.is_active = True
        self.using_fallback_input = False
        self.preferred_device = "auto"
        self.input_device_name = "Microphone (USBAudio1.0)"
        self.reopened = 0
        self.lock = threading.Lock()

    @property
    def frames_received(self) -> int:
        with self.lock:
            if self.alive:
                self._frames += 5
            return self._frames

    def reopen(self) -> bool:
        self.reopened += 1
        self.alive = True
        self.last_error = ""
        return True


@pytest.mark.asyncio
async def test_the_listener_reopens_a_dead_microphone_and_says_so(monkeypatch):
    from pilot.system.voice import ContinuousVoiceListener

    monkeypatch.setattr(audio_health, "STALL_SECONDS", 0.05)
    statuses: list[tuple[str, dict]] = []

    async def on_status(name, data):
        statuses.append((name, data))

    listener = ContinuousVoiceListener(wake_words=["hey heliox"], on_status=on_status, config=PilotConfig())
    fake = FakeRecorder()
    listener._recorder = fake
    listener._running = True
    watch = asyncio.ensure_future(listener._watch_capture(every_seconds=0.01))
    await asyncio.sleep(0.05)
    fake.alive = False  # 15:41:58: a call app takes the headset and the stream stops
    for _ in range(500):
        if any(name == "microphone_recovered" for name, _data in statuses):
            break
        await asyncio.sleep(0.01)
    listener._running = False
    await asyncio.wait_for(watch, 5)
    names = [name for name, _data in statuses]
    assert names[:2] == ["microphone_problem", "microphone_recovered"]
    assert statuses[0][1]["reason"] == "no_audio" and fake.reopened == 1
    assert listener._capture_watchdog.problem == "" and listener.capture_notice == ""


@pytest.mark.asyncio
async def test_the_listener_follows_a_new_default_microphone(monkeypatch):
    from pilot.system import voice
    from pilot.system.platform_detect import Platform

    names = iter(["USB mic", "USB mic", "Headset (Jabra)"] + ["Headset (Jabra)"] * 100)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(audio_health, "default_microphone_name", lambda: next(names))
    monkeypatch.setattr(audio_health, "default_speaker_name", lambda: "")
    statuses: list[tuple[str, dict]] = []

    async def on_status(name, data):
        statuses.append((name, data))

    listener = voice.ContinuousVoiceListener(wake_words=["hey heliox"], on_status=on_status, config=PilotConfig())
    fake = FakeRecorder()
    listener._recorder = fake
    listener._running = True
    watch = asyncio.ensure_future(listener._watch_capture(every_seconds=0.01))
    for _ in range(300):
        if fake.reopened:
            break
        await asyncio.sleep(0.01)
    listener._running = False
    await asyncio.wait_for(watch, 5)
    assert fake.reopened == 1
    assert statuses[0][1]["reason"] == "default_changed"


def test_on_macos_silence_also_points_at_the_microphone_permission():
    assert "Privacy & Security" in audio_health.silence_message("darwin")
    assert "Privacy" not in audio_health.silence_message("win32")
    assert "Privacy" not in audio_health.silence_message("linux")


def test_reopening_a_portaudio_microphone_rereads_the_device_list(monkeypatch):
    from pilot.system import voice

    recorder = voice._ContinuousRecorder()
    recorder.capture_backend = "portaudio"
    refreshed = []
    monkeypatch.setattr(voice, "_refresh_audio_devices", lambda: refreshed.append(True))
    monkeypatch.setattr(recorder, "_open_capture", lambda: True)
    assert recorder.reopen() and refreshed == [True]
    recorder.capture_backend = "windows_wasapi"  # WASAPI sees new devices without a refresh
    assert recorder.reopen() and refreshed == [True]


@pytest.mark.asyncio
async def test_a_refused_macos_permission_is_said_at_once(monkeypatch):
    from pilot.system import macos_permissions
    from pilot.system.voice import ContinuousVoiceListener

    monkeypatch.setattr(macos_permissions, "microphone_state", lambda: macos_permissions.PermissionState.DENIED)
    statuses: list[tuple[str, dict]] = []

    async def on_status(name, data):
        statuses.append((name, data))

    listener = ContinuousVoiceListener(wake_words=["hey heliox"], on_status=on_status, config=PilotConfig())
    fake = FakeRecorder()
    fake.zero_seconds = 5.0  # macOS gives a refused app pure zeros
    listener._recorder = fake
    listener._running = True
    watch = asyncio.ensure_future(listener._watch_capture(every_seconds=0.01))
    for _ in range(300):
        if statuses:
            break
        await asyncio.sleep(0.01)
    listener._running = False
    await asyncio.wait_for(watch, 5)
    name, data = statuses[0]
    assert (name, data["reason"]) == ("microphone_problem", "permission")
    assert "Privacy & Security" in data["message"]
    assert len([s for s in statuses if s[1]["reason"] == "permission"]) == 1  # said once, not every second
    assert not listener._capture_watchdog.recovered(
        CaptureState(frames=10_000, error="", zero_seconds=5.0, active=True)
    )  # still zeros: not recovered


@pytest.mark.asyncio
async def test_off_macos_the_permission_check_says_nothing(monkeypatch):
    from pilot.system import macos_permissions
    from pilot.system.voice import ContinuousVoiceListener

    monkeypatch.setattr(macos_permissions, "microphone_state", lambda: macos_permissions.PermissionState.CANNOT_TELL)
    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    assert not await listener._check_microphone_permission()


class FakeSoundcard:
    """soundcard as the WASAPI path uses it: the first recorder dies with "device invalidated", the next works."""

    class SoundcardRuntimeWarning(Warning):
        pass

    def __init__(self) -> None:
        self.opened = 0

    def all_microphones(self, include_loopback=False):
        return [self.default_microphone()]

    def default_microphone(self):
        card = self

        class Recorder:
            def __init__(self, dies: bool) -> None:
                self.dies, self.blocks = dies, 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def record(self, numframes):
                import time as _time

                self.blocks += 1
                if self.dies and self.blocks > 3:
                    raise RuntimeError("Error 0x88890004")  # AUDCLNT_E_DEVICE_INVALIDATED: a call app took it
                _time.sleep(0.005)
                return np.full((numframes, 1), 0.01, dtype=np.float32)

        class Microphone:
            name = "Headset (Jabra)"

            def recorder(self, samplerate, channels, blocksize):
                card.opened += 1
                return Recorder(dies=card.opened == 1)

        return Microphone()


@pytest.mark.asyncio
async def test_a_wasapi_stream_invalidated_by_a_call_app_is_reopened(monkeypatch):
    """10 November 15:41:58: record() raised 0x88890004, the pump thread ended, but the recorder still looked open."""
    import sys

    from pilot.system import voice
    from pilot.system.platform_detect import Platform

    card = FakeSoundcard()
    monkeypatch.setitem(sys.modules, "soundcard", card)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(audio_health, "STALL_SECONDS", 0.2)
    monkeypatch.setattr(audio_health, "default_microphone_name", lambda: "")
    monkeypatch.setattr(audio_health, "default_speaker_name", lambda: "")
    statuses: list[tuple[str, dict]] = []

    async def on_status(name, data):
        statuses.append((name, data))

    listener = voice.ContinuousVoiceListener(wake_words=["hey heliox"], on_status=on_status, config=PilotConfig())
    recorder = listener._recorder
    assert recorder.start() and recorder.capture_backend == "windows_wasapi"
    listener._running = True
    watch = asyncio.ensure_future(listener._watch_capture(every_seconds=0.02))
    for _ in range(500):
        if any(name == "microphone_recovered" for name, _data in statuses):
            break
        await asyncio.sleep(0.01)
    listener._running = False
    await asyncio.wait_for(watch, 5)
    recorder.stop()
    assert recorder.is_active is False
    assert statuses[0][0] == "microphone_problem" and statuses[0][1]["reason"] == "stream_failed"
    assert "0x88890004" in statuses[0][1]["message"]
    assert card.opened == 2 and any(name == "microphone_recovered" for name, _data in statuses)
