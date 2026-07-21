from pilot.config import PilotConfig, _merge_config


def test_default_is_disabled():
    config = PilotConfig()
    assert config.narration.enabled is False
    assert config.narration.narrate_steps is True
    assert config.narration.interrupt_on_risk is True
    assert config.narration.confirm_timeout_seconds == 120.0


def test_narration_section_merges_scalars():
    config = PilotConfig()
    merged = _merge_config(
        config,
        {
            "narration": {
                "enabled": True,
                "narrate_steps": False,
                "interrupt_on_risk": False,
                "confirm_timeout_seconds": 30.0,
            }
        },
    )
    assert merged.narration.enabled is True
    assert merged.narration.narrate_steps is False
    assert merged.narration.interrupt_on_risk is False
    assert merged.narration.confirm_timeout_seconds == 30.0


def test_missing_section_leaves_default():
    config = PilotConfig()
    merged = _merge_config(config, {})
    assert merged.narration.enabled is False
