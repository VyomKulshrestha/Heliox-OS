import wave

import numpy as np

from pilot.system.voice import (
    _band_limit_voice_audio,
    _load_pcm_wav_for_whisper,
    _normalize_voice_audio,
)


def test_load_pcm_wav_for_whisper_without_ffmpeg(tmp_path):
    path = tmp_path / "voice.wav"
    source = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(source.tobytes())

    loaded = _load_pcm_wav_for_whisper(str(path))

    assert loaded is not None
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(
        loaded,
        source.astype(np.float32) / 32768.0,
        rtol=0,
        atol=1e-7,
    )


def test_load_pcm_wav_for_whisper_downmixes_and_resamples(tmp_path):
    path = tmp_path / "stereo.wav"
    left = np.full(8000, 16384, dtype=np.int16)
    right = np.zeros(8000, dtype=np.int16)
    stereo = np.column_stack([left, right]).reshape(-1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(stereo.tobytes())

    loaded = _load_pcm_wav_for_whisper(str(path))

    assert loaded is not None
    assert loaded.shape == (16000,)
    np.testing.assert_allclose(loaded, 0.25, rtol=0, atol=1e-5)


def test_normalize_voice_audio_raises_quiet_signal_with_bounded_gain():
    audio = np.array([-0.08, 0.0, 0.08], dtype=np.float32)

    normalized = _normalize_voice_audio(audio)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [-0.5, 0.0, 0.5], atol=1e-6)


def test_normalize_voice_audio_does_not_amplify_near_silence():
    audio = np.array([-0.001, 0.0, 0.001], dtype=np.float32)

    normalized = _normalize_voice_audio(audio)

    np.testing.assert_array_equal(normalized, audio)


def test_band_limit_voice_audio_keeps_speech_and_removes_rumble():
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    rumble = np.sin(2 * np.pi * 20 * time).astype(np.float32)
    speech = np.sin(2 * np.pi * 1000 * time).astype(np.float32)

    filtered_rumble = _band_limit_voice_audio(rumble, sample_rate)
    filtered_speech = _band_limit_voice_audio(speech, sample_rate)

    assert filtered_rumble.dtype == np.float32
    assert filtered_rumble.shape == rumble.shape
    assert np.sqrt(np.mean(filtered_rumble**2)) < 0.01
    assert np.sqrt(np.mean(filtered_speech**2)) > 0.65
