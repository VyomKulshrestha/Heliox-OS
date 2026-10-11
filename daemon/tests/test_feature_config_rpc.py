from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["yes", 1, None])
async def test_first_run_update_requires_boolean(value):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "", "values": {"first_run_complete": value}},
        MagicMock(),
    )

    assert result == {"status": "error", "message": "first_run_complete must be a boolean"}
    assert config.first_run_complete is False
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_first_run_save_failure_restores_live_value():
    config = PilotConfig()
    config.save = MagicMock(side_effect=OSError("disk full"))
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "", "values": {"first_run_complete": True}},
        MagicMock(),
    )

    assert result == {"status": "error", "message": "Config update was not saved: disk full"}
    assert config.first_run_complete is False


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [None, "not-an-object", []])
async def test_config_update_requires_values_object(values):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "screen_vision", "values": values},
        MagicMock(),
    )

    assert result == {"status": "error", "message": "Config values must be an object"}
    config.save.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0.49, 60.1, "3", True, float("nan")])
async def test_screen_vision_interval_rejects_invalid_or_unbounded_values(value):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "screen_vision", "values": {"capture_interval_seconds": value}},
        MagicMock(),
    )

    assert result["status"] == "error"
    # NaN is refused before the range check, as every non-finite number is.
    expected = "must be a finite number" if value != value else "capture_interval_seconds must be from 0.5 to 60.0"
    assert expected in result["message"]
    assert config.screen_vision.capture_interval_seconds == 3.0
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_screen_vision_runtime_failure_restores_persisted_and_live_interval():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    screen_vision = SimpleNamespace(set_interval=MagicMock(side_effect=[RuntimeError("timer failed"), None]))
    server._screen_vision = screen_vision

    result = await server._handle_update_config(
        {"section": "screen_vision", "values": {"capture_interval_seconds": 5}},
        MagicMock(),
    )

    assert result == {"status": "error", "message": "Screen-vision setting was not applied: timer failed"}
    assert config.screen_vision.capture_interval_seconds == 3.0
    assert config.save.call_count == 2
    assert screen_vision.set_interval.call_args_list == [call(5.0), call(3.0)]


@pytest.mark.asyncio
async def test_preview_enabled_update_requires_boolean():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "preview", "values": {"enabled": "yes"}},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.preview.enabled is False
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_preview_enabled_update_is_applied_and_saved():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "preview", "values": {"enabled": True}},
        MagicMock(),
    )

    assert result["status"] == "ok"
    assert config.preview.enabled is True
    config.save.assert_called_once_with()


@pytest.mark.asyncio
async def test_unknown_config_key_is_rejected_by_rpc():
    server = object.__new__(PilotServer)
    config = PilotConfig()
    config.save = MagicMock()
    server.config = config

    result = await server._handle_update_config(
        {"section": "calendar", "values": {"caldav_usernme": "typo"}},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert "calendar.caldav_usernme" in result["message"]
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_subscription_provider_update_is_validated_and_saved():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "model",
            "values": {
                "provider": "subscription",
                "subscription_provider": "codex",
                "subscription_model": "  gpt-test  ",
                "subscription_timeout_seconds": 90,
                "subscription_max_prompt_chars": 24000,
            },
        },
        MagicMock(),
    )

    assert result["status"] == "ok"
    assert config.model.provider == "subscription"
    assert config.model.subscription_provider == "codex"
    assert config.model.subscription_model == "gpt-test"
    config.save.assert_called_once_with()


@pytest.mark.asyncio
async def test_model_update_reconfigures_live_runtime():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    model_router = SimpleNamespace(reconfigure=AsyncMock())
    server._model_router = model_router
    server._planner = SimpleNamespace(_model=model_router)

    result = await server._handle_update_config(
        {"section": "model", "values": {"subscription_model": "gpt-test"}},
        MagicMock(),
    )

    assert result["status"] == "ok"
    model_router.reconfigure.assert_awaited_once_with({"subscription_model"})


@pytest.mark.asyncio
async def test_config_save_failure_restores_live_values_without_reconfiguring_model():
    config = PilotConfig()
    config.save = MagicMock(side_effect=OSError("disk full"))
    server = PilotServer(config)
    model_router = SimpleNamespace(reconfigure=AsyncMock())
    server._model_router = model_router

    result = await server._handle_update_config(
        {"section": "model", "values": {"subscription_model": "gpt-test"}},
        MagicMock(),
    )

    assert result == {"status": "error", "message": "Config update was not saved: disk full"}
    assert config.model.subscription_model == ""
    model_router.reconfigure.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_reconfigure_failure_rolls_back_persisted_and_live_values():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    model_router = SimpleNamespace(
        reconfigure=AsyncMock(side_effect=[RuntimeError("reload failed"), None]),
    )
    server._model_router = model_router

    result = await server._handle_update_config(
        {"section": "model", "values": {"subscription_model": "gpt-test"}},
        MagicMock(),
    )

    assert result == {"status": "error", "message": "Model setting was not applied: reload failed"}
    assert config.model.subscription_model == ""
    assert config.save.call_count == 2
    assert model_router.reconfigure.await_count == 2
    model_router.reconfigure.assert_awaited_with({"subscription_model"})


@pytest.mark.asyncio
async def test_failure_threshold_update_reconfigures_live_circuit_breaker():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    model_router = SimpleNamespace(reconfigure=AsyncMock())
    circuit_breaker = SimpleNamespace(reconfigure=MagicMock())
    server._model_router = model_router
    server._circuit_breaker = circuit_breaker

    result = await server._handle_update_config(
        {"section": "model", "values": {"max_consecutive_failures": 7}},
        MagicMock(),
    )

    assert result["status"] == "ok"
    model_router.reconfigure.assert_awaited_once_with({"max_consecutive_failures"})
    circuit_breaker.reconfigure.assert_called_once_with(7)


@pytest.mark.asyncio
async def test_circuit_breaker_failure_restores_model_router_and_threshold():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    model_router = SimpleNamespace(reconfigure=AsyncMock())
    circuit_breaker = SimpleNamespace(reconfigure=MagicMock(side_effect=[RuntimeError("breaker failed"), None]))
    server._model_router = model_router
    server._circuit_breaker = circuit_breaker

    result = await server._handle_update_config(
        {"section": "model", "values": {"max_consecutive_failures": 7}},
        MagicMock(),
    )

    assert result == {"status": "error", "message": "Model setting was not applied: breaker failed"}
    assert config.model.max_consecutive_failures == 3
    assert config.save.call_count == 2
    assert model_router.reconfigure.await_count == 2
    assert circuit_breaker.reconfigure.call_args_list == [call(7), call(3)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"provider": "browser_cookie"}, "model.provider"),
        ({"subscription_provider": "unknown"}, "subscription_provider"),
        ({"subscription_timeout_seconds": 5}, "subscription_timeout_seconds"),
        ({"subscription_max_prompt_chars": 15999}, "subscription_max_prompt_chars"),
        ({"idle_unload_seconds": -1}, "idle_unload_seconds"),
        ({"idle_unload_seconds": 86401}, "idle_unload_seconds"),
        ({"rate_limit_rpm": 0}, "rate_limit_rpm"),
        ({"rate_limit_burst": 0}, "rate_limit_burst"),
        ({"budget_monthly_limit_usd": -1}, "budget_monthly_limit_usd"),
        ({"max_tokens_per_task": -1}, "max_tokens_per_task"),
        ({"max_usd_per_task": -0.01}, "max_usd_per_task"),
        ({"max_consecutive_failures": 0}, "max_consecutive_failures"),
    ],
)
async def test_subscription_provider_update_rejects_unsafe_values(values, message):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "model", "values": values},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert message in result["message"]


@pytest.mark.asyncio
async def test_reasoning_mode_update_rejects_unknown_value():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "model", "values": {"mode": "turbo"}},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.model.mode == "lightweight"
    config.save.assert_not_called()
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_rejected_multi_field_update_does_not_partially_mutate_runtime_config():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "model",
            "values": {
                "provider": "subscription",
                "subscription_timeout_seconds": 5,
            },
        },
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.model.provider == "ollama"
    assert config.model.subscription_timeout_seconds == 120
    config.save.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"enabled": 1}, "enabled must be a boolean"),
        ({"sensitivity": 0}, "sensitivity must be from 0.1 to 3"),
        ({"prediction_ms": 251}, "prediction_ms must be from 0 to 250"),
        ({"blend": 1.1}, "blend must be from 0 to 1"),
    ],
)
async def test_gesture_cursor_update_rejects_invalid_values(values, message):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "gesture_cursor", "values": values},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert message in result["message"]
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_gesture_cursor_update_applies_runtime_tuning():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "gesture_cursor",
            "values": {"enabled": True, "sensitivity": 1.7, "blend": 0.45},
        },
        MagicMock(),
    )

    assert result["status"] == "ok"
    assert config.gesture_cursor.enabled is True
    assert config.gesture_cursor.sensitivity == 1.7
    assert config.gesture_cursor.blend == 0.45
    config.save.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"mediapipe_backend": "remote"}, "mediapipe_backend must be legacy or tasks"),
        ({"gaze_tracking_enabled": 1}, "gaze_tracking_enabled must be a boolean"),
    ],
)
async def test_camera_intelligence_update_rejects_invalid_values(values, message):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "vision", "values": values},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert message in result["message"]
    assert config.vision.mediapipe_backend == "tasks"
    assert config.vision.gaze_tracking_enabled is False
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_gesture_calibration_update_requires_boolean():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "adaptive_calibration",
            "values": {"gesture_enabled": "enabled"},
        },
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.adaptive_calibration.gesture_enabled is True
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_voice_calibration_update_requires_boolean():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "adaptive_calibration",
            "values": {"voice_wake_word_enabled": 1},
        },
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.adaptive_calibration.voice_wake_word_enabled is True
    config.save.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"tts_engine": "cloud"}, "tts_engine must be kokoro_tts, pocket_tts, or os_native"),
        ({"tts_voice": "unknown"}, "tts_voice must be a supported Kokoro or Pocket TTS voice"),
        ({"input_device": ""}, "input_device must be a valid microphone identifier"),
    ],
)
async def test_voice_output_update_rejects_unknown_options(values, message):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "voice", "values": values},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert message in result["message"]
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_active_voice_listener_restarts_for_new_input_device(monkeypatch):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    current = SimpleNamespace(
        is_running=True,
        wake_words=["hey heliox"],
        stop=AsyncMock(return_value="stopped"),
    )
    server._voice_listener = current
    created = []

    class FakeListener:
        def __init__(self, *, wake_words, config, **kwargs):
            self.wake_words = wake_words
            self.input_device = config.voice.input_device
            self.is_running = False
            created.append(self)

        async def start(self):
            self.is_running = True
            return "started"

        async def stop(self, *, cancel_commands=True):
            self.is_running = False
            self.cancelled_commands = cancel_commands
            return "stopped"

        def adopt_command_tasks(self, tasks):
            self.adopted = set(tasks)

    monkeypatch.setattr("pilot.system.voice.ContinuousVoiceListener", FakeListener)

    result = await server._handle_update_config(
        {"section": "voice", "values": {"input_device": "WASAPI::Headset"}},
        MagicMock(),
    )

    assert result == {"status": "ok"}
    current.stop.assert_awaited_once()
    assert config.voice.input_device == "WASAPI::Headset"
    assert created[0].input_device == "WASAPI::Headset"
    assert server._voice_listener is created[0]
    assert created[0].is_running is True


@pytest.mark.asyncio
async def test_failed_voice_device_restart_rolls_back_config_and_listener(monkeypatch):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    current = SimpleNamespace(
        is_running=True,
        wake_words=["hey heliox"],
        stop=AsyncMock(return_value="stopped"),
    )
    server._voice_listener = current
    attempted_devices = []

    class FakeListener:
        def __init__(self, *, wake_words, config, **kwargs):
            self.wake_words = wake_words
            self.input_device = config.voice.input_device
            self.is_running = False
            attempted_devices.append(self.input_device)

        async def start(self):
            self.is_running = self.input_device != "missing-device"
            return "started" if self.is_running else "selected microphone is unavailable"

        async def stop(self, *, cancel_commands=True):
            self.is_running = False
            self.cancelled_commands = cancel_commands
            return "stopped"

        def adopt_command_tasks(self, tasks):
            self.adopted = set(tasks)

    monkeypatch.setattr("pilot.system.voice.ContinuousVoiceListener", FakeListener)

    result = await server._handle_update_config(
        {"section": "voice", "values": {"input_device": "missing-device"}},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert "selected microphone is unavailable" in result["message"]
    assert config.voice.input_device == "auto"
    assert attempted_devices == ["missing-device", "auto"]
    assert server._voice_listener.input_device == "auto"
    assert server._voice_listener.is_running is True
    assert config.save.call_count == 2
