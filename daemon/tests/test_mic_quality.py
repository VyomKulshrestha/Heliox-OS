"""Telling a Bluetooth headset in call mode, a very quiet microphone and a clipping one from the sound alone.

The thresholds were set on the owner's recordings (desk microphone: 8e-4 to 1.3e-2 of the speech band's energy
above 3.8 kHz; Bluetooth headset in call mode: none); here synthetic speech-like signals stand in for them."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from pilot.system.mic_quality import MESSAGES, MicQuality, hands_free_name

RATE = 16000


def speech_like(seconds: float, level: float = 0.05, seed: int = 0) -> np.ndarray:
    """Noise shaped like voiced speech: syllables of 0.25 s with short gaps, wideband up to 8 kHz."""
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(int(seconds * RATE)).astype(np.float32)
    envelope = (np.arange(len(audio)) // (RATE // 4)) % 3 != 2  # two syllables, one gap
    audio = audio * envelope
    # A gentle tilt, as speech has: more energy low than high, but still some above 4 kHz.
    audio = sosfilt(butter(1, 1500, "low", fs=RATE, output="sos"), audio) + 0.15 * audio
    return (audio / np.abs(audio).max() * level * 4).astype(np.float32)


def narrowband(audio: np.ndarray) -> np.ndarray:
    """What a headset in call mode delivers: an 8 kHz link, brought back to 16 kHz by the system."""
    from scipy.signal import resample_poly

    return resample_poly(resample_poly(audio, 1, 2), 2, 1).astype(np.float32)[: len(audio)]


def judge(chunks) -> MicQuality:
    quality = MicQuality()
    for chunk in chunks:
        quality.observe(chunk)
    return quality


def test_a_wideband_microphone_is_ok():
    assert judge([speech_like(2.5, seed=n) for n in range(3)]).verdict == "ok"


def test_a_headset_in_call_mode_is_narrowband():
    quality = judge([narrowband(speech_like(2.5, seed=n)) for n in range(3)])
    assert quality.verdict == "narrowband"
    assert "Bluetooth call mode" in MESSAGES["narrowband"] and "computer's own microphone" in MESSAGES["narrowband"]


def test_one_short_word_is_not_enough_to_judge():
    quality = MicQuality()
    assert quality.observe(narrowband(speech_like(1.0))) == ""  # "Wait, stop." has no sibilants on any mic
    assert quality.verdict == ""


def test_a_very_quiet_and_a_clipping_microphone():
    assert judge([speech_like(2.5, level=0.0008, seed=n) for n in range(3)]).verdict == "quiet"
    loud = [np.clip(speech_like(2.5, level=0.6, seed=n) * 3, -1.0, 1.0) for n in range(3)]
    assert judge(loud).verdict == "clipping"


def test_the_verdict_is_given_once_and_judged_afresh_for_another_microphone():
    quality = MicQuality()
    verdicts = [quality.observe(narrowband(speech_like(2.5, seed=n))) for n in range(4)]
    assert [v for v in verdicts if v] == ["narrowband"]
    quality.reset()
    assert judge([speech_like(2.5, seed=n) for n in range(3)]).verdict == "ok"


def test_device_names_that_say_call_mode():
    assert hands_free_name("Headset (Rockerz 415 Hands-Free AG Audio)")
    assert hands_free_name("bluez_input.XX.headset_head_unit")
    assert not hands_free_name("Microphone (USBAudio1.0)")


@pytest.mark.asyncio
async def test_the_listener_says_once_when_the_microphone_is_narrowband(monkeypatch, tmp_path):
    from pilot.config import PilotConfig
    from pilot.system import voice

    statuses = []

    async def on_status(name, data):
        statuses.append((name, data))

    listener = voice.ContinuousVoiceListener(wake_words=["hey heliox"], on_status=on_status, config=PilotConfig())
    monkeypatch.setattr(listener, "_other_input_for", lambda headset: None)  # no other microphone here
    clips = iter([narrowband(speech_like(2.5, seed=n)) for n in range(4)])
    monkeypatch.setattr(voice, "_load_pcm_wav_for_whisper", lambda _path: next(clips))
    for _ in range(4):
        listener._schedule_mic_quality(str(tmp_path / "u.wav"))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if statuses:
                break
    assert [(name, data["reason"]) for name, data in statuses] == [("microphone_quality", "narrowband")]
    assert listener.get_stats()["mic_quality"] == "narrowband"


# ── offering another microphone for a headset in call mode ──

WINDOWS_DEVICES = [
    {"id": "Windows WASAPI::Headset (Rockerz 415 Hands-Free AG Audio)", "name": "Headset (Rockerz 415 Hands-Free AG Audio)",
     "hostapi": "Windows WASAPI"},
    {"id": "MME::Headset Microphone (Rockerz 415)", "name": "Headset Microphone (Rockerz 415)", "hostapi": "MME"},
    {"id": "Windows WASAPI::Stereo Mix (Realtek(R) Audio)", "name": "Stereo Mix (Realtek(R) Audio)",
     "hostapi": "Windows WASAPI"},
    {"id": "MME::Microphone Array (Realtek(R) Audio)", "name": "Microphone Array (Realtek(R) Audio)", "hostapi": "MME"},
    {"id": "Windows WASAPI::Microphone Array (Realtek(R) Audio)", "name": "Microphone Array (Realtek(R) Audio)",
     "hostapi": "Windows WASAPI"},
]  # fmt: skip


def test_the_offer_is_the_built_in_microphone_never_the_headset_or_a_recording_of_the_output():
    from pilot.system.mic_quality import suggest_input

    offer = suggest_input(WINDOWS_DEVICES, "Headset (Rockerz 415 Hands-Free AG Audio)", "win32")
    assert offer == {
        "id": "Windows WASAPI::Microphone Array (Realtek(R) Audio)",
        "name": "Microphone Array (Realtek(R) Audio)",
    }
    mac = [
        {"id": "Core Audio::AirPods Pro", "name": "AirPods Pro", "hostapi": "Core Audio"},
        {"id": "Core Audio::MacBook Pro Microphone", "name": "MacBook Pro Microphone", "hostapi": "Core Audio"},
        {"id": "Core Audio::BlackHole 2ch", "name": "BlackHole 2ch", "hostapi": "Core Audio"},
    ]
    assert suggest_input(mac, "AirPods Pro", "darwin")["name"] == "MacBook Pro Microphone"
    linux = [  # PulseAudio/PipeWire sources by their stable names, as audio_health.pulse_sources lists them
        {"id": "pulse::bluez_input.F4_4E.headset-head-unit", "name": "Rockerz 415", "hostapi": "PulseAudio"},
        {"id": "pulse::alsa_input.pci-0000_00_1f.3.analog-stereo", "name": "Built-in Audio Analog Stereo",
         "hostapi": "PulseAudio"},
    ]  # fmt: skip
    offer = suggest_input(linux, "bluez_input.F4_4E.headset-head-unit", "linux")
    assert offer == {"id": "pulse::alsa_input.pci-0000_00_1f.3.analog-stereo", "name": "Built-in Audio Analog Stereo"}
    assert suggest_input(WINDOWS_DEVICES[:3], "Headset (Rockerz 415 Hands-Free AG Audio)", "win32") is None


def test_a_headset_the_person_chose_another_microphone_for_listens_with_that_one(monkeypatch):
    from pilot.system import audio_health
    from pilot.system.voice import _ContinuousRecorder

    recorder = _ContinuousRecorder()
    recorder.headset_input_choices = {"Headset (Rockerz)": "Windows WASAPI::Microphone Array (Realtek)"}
    monkeypatch.setattr(audio_health, "default_microphone_key", lambda: "Headset (Rockerz)")
    assert recorder.effective_preferred_device() == "Windows WASAPI::Microphone Array (Realtek)"
    monkeypatch.setattr(audio_health, "default_microphone_key", lambda: "Microphone (USBAudio1.0)")
    assert recorder.effective_preferred_device() == "auto"  # another default: the choice does not apply
    recorder.preferred_device = "MME::Microphone (USBAudio1.0)"
    monkeypatch.setattr(audio_health, "default_microphone_key", lambda: "Headset (Rockerz)")
    assert recorder.effective_preferred_device() == "MME::Microphone (USBAudio1.0)"  # a fixed choice wins


@pytest.mark.asyncio
async def test_the_listener_offers_another_microphone_once_for_a_narrowband_headset(monkeypatch, tmp_path):
    from pilot.config import PilotConfig
    from pilot.system import voice

    statuses = []

    async def on_status(name, data):
        statuses.append((name, data))

    from pilot.system import audio_health

    # The headset is the device the listener opened, not whatever this computer's real default microphone is: the
    # test read the real one whenever an earlier test had left soundcard working on the thread it ran on.
    monkeypatch.setattr(audio_health, "default_microphone_key", lambda platform=None: "")
    listener = voice.ContinuousVoiceListener(wake_words=["hey heliox"], on_status=on_status, config=PilotConfig())
    listener._recorder.input_device_name = "Headset (Rockerz 415 Hands-Free AG Audio)"
    monkeypatch.setattr(listener, "_other_input_for", lambda headset: {"id": "W::Mic", "name": "Mic"})
    clips = iter([narrowband(speech_like(2.5, seed=n)) for n in range(4)])
    monkeypatch.setattr(voice, "_load_pcm_wav_for_whisper", lambda _path: next(clips))
    for _ in range(4):
        listener._schedule_mic_quality(str(tmp_path / "u.wav"))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if statuses:
                break
    name, data = statuses[0]
    assert name == "microphone_quality" and data["suggested_input"] == {"id": "W::Mic", "name": "Mic"}
    assert data["headset"] == "Headset (Rockerz 415 Hands-Free AG Audio)" and len(statuses) == 1


@pytest.mark.asyncio
async def test_the_setting_is_validated_and_reopens_the_microphone():
    from unittest.mock import AsyncMock

    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server.config.save = lambda *a, **k: None
    bad = await server._handle_update_config({"section": "voice", "values": {"headset_input_choices": ["x"]}}, None)
    assert bad["status"] == "error"
    server._replace_voice_listener = AsyncMock()
    good = await server._handle_update_config(
        {"section": "voice", "values": {"headset_input_choices": {"Headset (Rockerz)": "W::Mic"}}}, None
    )
    assert good["status"] == "ok" and server.config.voice.headset_input_choices == {"Headset (Rockerz)": "W::Mic"}


def test_a_pulse_source_is_opened_through_the_pulse_device_by_its_stable_name(monkeypatch):
    import os
    import sys
    from types import SimpleNamespace

    from pilot.system import voice

    opened = {}

    class Stream:
        def __init__(self, **kwargs):
            opened.update(kwargs, source=os.environ.get("PULSE_SOURCE"))

        def start(self):
            pass

    fake_sd = SimpleNamespace(
        InputStream=Stream,
        query_devices=lambda device=None: {"name": "pulse", "hostapi": 0},
        query_hostapis=lambda index: {"name": "ALSA"},
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", voice.Platform.LINUX)
    monkeypatch.setattr(voice, "_resolve_input_device", lambda sd, **kwargs: 3)
    monkeypatch.delenv("PULSE_SOURCE", raising=False)
    recorder = voice._ContinuousRecorder(preferred_device="auto")
    recorder.headset_input_choices = {"bluez_input.F4_4E.headset-head-unit": "pulse::alsa_input.analog-stereo"}
    monkeypatch.setattr(recorder, "device_key", lambda: "bluez_input.F4_4E.headset-head-unit")
    assert recorder._open_capture()
    assert opened["source"] == "alsa_input.analog-stereo" and opened["device"] == 3
    recorder._stream = None
    monkeypatch.setattr(recorder, "device_key", lambda: "alsa_input.usb-mic")  # another default: no choice applies
    assert recorder._open_capture() and opened["source"] is None
