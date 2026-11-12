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
    clips = iter([narrowband(speech_like(2.5, seed=n)) for n in range(4)])
    monkeypatch.setattr(voice, "_load_pcm_wav_for_whisper", lambda _path: next(clips))
    for _ in range(4):
        listener._schedule_mic_quality(str(tmp_path / "u.wav"))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if listener._mic_quality.verdict:
                break
    await asyncio.sleep(0.05)
    assert [(name, data["reason"]) for name, data in statuses] == [("microphone_quality", "narrowband")]
    assert listener.get_stats()["mic_quality"] == "narrowband"
