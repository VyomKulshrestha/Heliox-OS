from pilot.config import PilotConfig, _merge_config


def test_defaults():
    config = PilotConfig()
    assert config.voice.vad_energy_threshold == 0.02
    assert config.voice.vad_silence_ms == 700.0
    assert config.voice.vad_max_utterance_seconds == 20.0
    assert config.voice.barge_in_enabled is True
    assert config.voice.tts_engine == "pocket_tts"
    assert config.voice.tts_voice == "alba"


def test_voice_section_merges_vad_and_barge_in_settings():
    config = PilotConfig()
    raw = {
        "voice": {
            "vad_energy_threshold": 0.05,
            "vad_silence_ms": 500.0,
            "vad_max_utterance_seconds": 15.0,
            "barge_in_enabled": False,
        }
    }

    merged = _merge_config(config, raw)

    assert merged.voice.vad_energy_threshold == 0.05
    assert merged.voice.vad_silence_ms == 500.0
    assert merged.voice.vad_max_utterance_seconds == 15.0
    assert merged.voice.barge_in_enabled is False


def test_voice_missing_section_leaves_defaults():
    config = PilotConfig()
    merged = _merge_config(config, {})
    assert merged.voice.barge_in_enabled is True


def test_voice_section_merges_tts_engine_and_voice():
    config = PilotConfig()
    raw = {"voice": {"tts_engine": "os_native", "tts_voice": "giovanni"}}

    merged = _merge_config(config, raw)

    assert merged.voice.tts_engine == "os_native"
    assert merged.voice.tts_voice == "giovanni"
