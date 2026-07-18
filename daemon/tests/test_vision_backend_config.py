from pilot.config import PilotConfig, _merge_config


def test_default_backend_is_legacy() -> None:
    config = PilotConfig()
    assert config.vision.mediapipe_backend == "legacy"


def test_vision_section_merges_backend_choice() -> None:
    config = PilotConfig()
    merged = _merge_config(config, {"vision": {"mediapipe_backend": "tasks"}})
    assert merged.vision.mediapipe_backend == "tasks"


def test_vision_missing_section_leaves_default() -> None:
    config = PilotConfig()
    merged = _merge_config(config, {})
    assert merged.vision.mediapipe_backend == "legacy"
