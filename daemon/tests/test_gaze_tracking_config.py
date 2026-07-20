from pilot.config import PilotConfig, _merge_config


def test_default_is_disabled():
    config = PilotConfig()
    assert config.vision.gaze_tracking_enabled is False


def test_vision_section_merges_gaze_tracking_toggle():
    config = PilotConfig()
    merged = _merge_config(config, {"vision": {"gaze_tracking_enabled": True}})
    assert merged.vision.gaze_tracking_enabled is True


def test_missing_section_leaves_default():
    config = PilotConfig()
    merged = _merge_config(config, {})
    assert merged.vision.gaze_tracking_enabled is False
