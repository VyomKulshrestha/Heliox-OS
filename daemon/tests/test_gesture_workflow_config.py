from pilot.config import GestureWorkflowBinding, PilotConfig, _merge_config


def test_defaults_are_disabled_with_no_bindings() -> None:
    config = PilotConfig()
    assert config.gesture_workflows.enabled is False
    assert config.gesture_workflows.bindings == []
    assert config.gesture_workflows.pending_trigger_window_seconds == 90.0
    assert config.gesture_workflows.paused_window_seconds == 1800.0


def test_gesture_workflows_section_merges_to_config() -> None:
    config = PilotConfig()
    raw = {
        "gesture_workflows": {
            "enabled": True,
            "bindings": [
                {"gesture_name": "swipe_up", "goal_template": "run my daily briefing", "enabled": True},
                {"gesture_name": "rock_sign", "goal_template": "back up today's screenshots"},
            ],
            "pending_trigger_window_seconds": 45.0,
            "paused_window_seconds": 600.0,
        }
    }

    merged = _merge_config(config, raw)

    assert merged.gesture_workflows.enabled is True
    assert merged.gesture_workflows.pending_trigger_window_seconds == 45.0
    assert merged.gesture_workflows.paused_window_seconds == 600.0
    assert len(merged.gesture_workflows.bindings) == 2
    assert merged.gesture_workflows.bindings[0] == GestureWorkflowBinding(
        gesture_name="swipe_up", goal_template="run my daily briefing", enabled=True
    )
    # enabled defaults to True when omitted from a binding's raw dict
    assert merged.gesture_workflows.bindings[1].enabled is True


def test_gesture_workflows_ignores_non_dict_binding_entries() -> None:
    config = PilotConfig()
    raw = {"gesture_workflows": {"bindings": [{"gesture_name": "palm", "goal_template": "x"}, "not-a-dict", 42]}}

    merged = _merge_config(config, raw)

    assert len(merged.gesture_workflows.bindings) == 1
    assert merged.gesture_workflows.bindings[0].gesture_name == "palm"


def test_gesture_workflows_missing_section_leaves_defaults() -> None:
    config = PilotConfig()
    merged = _merge_config(config, {})
    assert merged.gesture_workflows.enabled is False
    assert merged.gesture_workflows.bindings == []
