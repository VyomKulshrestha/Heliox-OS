from pilot.config import PilotConfig, _merge_config


def test_default_is_disabled():
    config = PilotConfig()
    assert config.supervision.enabled is False
    assert config.supervision.keyboard_mouse_hook_enabled is False
    assert config.supervision.cognitive_coaching_enabled is True
    assert config.supervision.risk_pattern_detection_enabled is True
    assert config.supervision.tick_interval_seconds == 1.5
    assert config.supervision.ocr_interval_seconds == 8.0
    assert config.supervision.stress_coaching_threshold == 0.75
    assert config.supervision.cognitive_load_coaching_threshold == 0.8
    assert config.supervision.coaching_cooldown_seconds == 900.0
    assert config.supervision.risk_cooldown_seconds == 30.0
    assert config.supervision.keystroke_buffer_max_chars == 256
    assert config.supervision.ocr_snippet_max_chars == 400


def test_supervision_section_merges_scalars():
    config = PilotConfig()
    merged = _merge_config(
        config,
        {
            "supervision": {
                "enabled": True,
                "keyboard_mouse_hook_enabled": True,
                "cognitive_coaching_enabled": False,
                "risk_pattern_detection_enabled": False,
                "tick_interval_seconds": 2.0,
                "ocr_interval_seconds": 15.0,
                "stress_coaching_threshold": 0.6,
                "cognitive_load_coaching_threshold": 0.7,
                "coaching_cooldown_seconds": 600.0,
                "risk_cooldown_seconds": 10.0,
                "keystroke_buffer_max_chars": 128,
                "ocr_snippet_max_chars": 200,
            }
        },
    )
    assert merged.supervision.enabled is True
    assert merged.supervision.keyboard_mouse_hook_enabled is True
    assert merged.supervision.cognitive_coaching_enabled is False
    assert merged.supervision.risk_pattern_detection_enabled is False
    assert merged.supervision.tick_interval_seconds == 2.0
    assert merged.supervision.ocr_interval_seconds == 15.0
    assert merged.supervision.stress_coaching_threshold == 0.6
    assert merged.supervision.cognitive_load_coaching_threshold == 0.7
    assert merged.supervision.coaching_cooldown_seconds == 600.0
    assert merged.supervision.risk_cooldown_seconds == 10.0
    assert merged.supervision.keystroke_buffer_max_chars == 128
    assert merged.supervision.ocr_snippet_max_chars == 200


def test_missing_section_leaves_default():
    config = PilotConfig()
    merged = _merge_config(config, {})
    assert merged.supervision.enabled is False
    assert merged.supervision.keyboard_mouse_hook_enabled is False
