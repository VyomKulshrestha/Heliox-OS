from pilot.config import PilotConfig, _merge_config


def test_default_is_disabled():
    config = PilotConfig()
    assert config.self_healing.enabled is False
    assert config.self_healing.auto_execute_max_tier == 1
    assert config.self_healing.watched_metrics == ["cpu", "memory", "disk"]


def test_self_healing_section_merges_scalars():
    config = PilotConfig()
    merged = _merge_config(
        config,
        {
            "self_healing": {
                "enabled": True,
                "auto_execute_max_tier": 2,
                "cooldown_seconds": 120.0,
                "confirm_timeout_seconds": 60.0,
            }
        },
    )
    assert merged.self_healing.enabled is True
    assert merged.self_healing.auto_execute_max_tier == 2
    assert merged.self_healing.cooldown_seconds == 120.0
    assert merged.self_healing.confirm_timeout_seconds == 60.0


def test_self_healing_section_merges_watched_metrics_and_goal_templates():
    config = PilotConfig()
    merged = _merge_config(
        config,
        {
            "self_healing": {
                "watched_metrics": ["disk"],
                "goal_templates": {"disk": "Free up disk space."},
            }
        },
    )
    assert merged.self_healing.watched_metrics == ["disk"]
    assert merged.self_healing.goal_templates == {"disk": "Free up disk space."}


def test_missing_section_leaves_default():
    config = PilotConfig()
    merged = _merge_config(config, {})
    assert merged.self_healing.enabled is False
