"""Tests for PilotServer's supervision_status/supervision_config_update RPC
handlers -- including the start/stop-on-transition wiring that (unlike
narration/self-healing's simpler config-flip-only handlers) actually starts
or stops the background task and the keyboard/mouse hook."""

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


class _FakeHook:
    def __init__(self):
        self.started = False
        self.stopped_count = 0
        self.start_count = 0

    def start(self) -> None:
        self.started = True
        self.start_count += 1

    def stop(self) -> None:
        self.started = False
        self.stopped_count += 1

    def is_running(self) -> bool:
        return self.started


class _FakeBackground:
    def __init__(self):
        self.started_tasks: list[str] = []
        self.stopped_tasks: list[str] = []

    def start(self, task_id: str) -> bool:
        self.started_tasks.append(task_id)
        return True

    def stop(self, task_id: str) -> bool:
        self.stopped_tasks.append(task_id)
        return True


@pytest.mark.asyncio
async def test_status_reports_config_defaults():
    server = PilotServer(PilotConfig())
    result = await server._handle_supervision_status({}, ws=None)
    assert result["enabled"] is False
    assert result["keyboard_mouse_hook_enabled"] is False
    assert result["cognitive_coaching_enabled"] is True
    assert result["risk_pattern_detection_enabled"] is True
    assert result["hook_healthy"] is False


@pytest.mark.asyncio
async def test_status_without_background_or_hook_initialized_is_safe():
    """Mirrors how a freshly-constructed PilotServer (before initialize())
    has neither _background nor _supervision_hook set -- must not raise."""
    server = PilotServer(PilotConfig())
    assert not hasattr(server, "_supervision_hook")
    result = await server._handle_supervision_status({}, ws=None)
    assert result["hook_healthy"] is False


@pytest.mark.asyncio
async def test_config_update_persists_enabled_toggle(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_supervision_config_update({"enabled": True}, ws=None)
    assert result["status"] == "ok"
    assert result["enabled"] is True
    assert server.config.supervision.enabled is True


@pytest.mark.asyncio
async def test_config_update_without_background_or_hook_is_safe(monkeypatch):
    """Toggling enabled/keyboard_mouse_hook_enabled before initialize() has
    run (no _background/_supervision_hook yet) must not raise."""
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_supervision_config_update(
        {"enabled": True, "keyboard_mouse_hook_enabled": True}, ws=None
    )
    assert result["status"] == "ok"
    assert result["hook_healthy"] is False


@pytest.mark.asyncio
async def test_config_update_sets_sub_toggles_and_thresholds(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_supervision_config_update(
        {
            "cognitive_coaching_enabled": False,
            "risk_pattern_detection_enabled": False,
            "stress_coaching_threshold": 0.5,
            "ocr_snippet_max_chars": 100,
        },
        ws=None,
    )
    assert result["cognitive_coaching_enabled"] is False
    assert result["risk_pattern_detection_enabled"] is False
    assert server.config.supervision.stress_coaching_threshold == 0.5
    assert server.config.supervision.ocr_snippet_max_chars == 100


@pytest.mark.asyncio
async def test_status_reflects_updated_config(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    await server._handle_supervision_config_update({"enabled": True}, ws=None)
    result = await server._handle_supervision_status({}, ws=None)
    assert result["enabled"] is True


@pytest.mark.asyncio
async def test_enabling_starts_background_task(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)
    server._background = _FakeBackground()

    await server._handle_supervision_config_update({"enabled": True}, ws=None)

    assert server._background.started_tasks == ["user_supervision"]
    assert server._background.stopped_tasks == []


@pytest.mark.asyncio
async def test_disabling_stops_background_task(monkeypatch):
    config = PilotConfig()
    config.supervision.enabled = True
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)
    server._background = _FakeBackground()

    await server._handle_supervision_config_update({"enabled": False}, ws=None)

    assert server._background.stopped_tasks == ["user_supervision"]
    assert server._background.started_tasks == []


@pytest.mark.asyncio
async def test_enabling_hook_while_already_enabled_starts_hook(monkeypatch):
    config = PilotConfig()
    config.supervision.enabled = True
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)
    server._background = _FakeBackground()
    server._supervision_hook = _FakeHook()

    result = await server._handle_supervision_config_update({"keyboard_mouse_hook_enabled": True}, ws=None)

    assert server._supervision_hook.started is True
    assert result["hook_healthy"] is True


@pytest.mark.asyncio
async def test_disabling_hook_stops_hook(monkeypatch):
    config = PilotConfig()
    config.supervision.enabled = True
    config.supervision.keyboard_mouse_hook_enabled = True
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)
    server._background = _FakeBackground()
    server._supervision_hook = _FakeHook()
    server._supervision_hook.started = True

    result = await server._handle_supervision_config_update({"keyboard_mouse_hook_enabled": False}, ws=None)

    assert server._supervision_hook.started is False
    assert result["hook_healthy"] is False


@pytest.mark.asyncio
async def test_disabling_overall_enabled_also_stops_hook(monkeypatch):
    config = PilotConfig()
    config.supervision.enabled = True
    config.supervision.keyboard_mouse_hook_enabled = True
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)
    server._background = _FakeBackground()
    server._supervision_hook = _FakeHook()
    server._supervision_hook.started = True

    await server._handle_supervision_config_update({"enabled": False}, ws=None)

    assert server._supervision_hook.started is False
    assert server._background.stopped_tasks == ["user_supervision"]


@pytest.mark.asyncio
async def test_already_enabled_transition_does_not_restart_hook(monkeypatch):
    """Flipping an unrelated field while enabled/hook stay both True must
    not re-issue start()."""
    config = PilotConfig()
    config.supervision.enabled = True
    config.supervision.keyboard_mouse_hook_enabled = True
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)
    server._background = _FakeBackground()
    server._supervision_hook = _FakeHook()
    server._supervision_hook.started = True

    await server._handle_supervision_config_update({"ocr_snippet_max_chars": 50}, ws=None)

    assert server._supervision_hook.start_count == 0
    assert server._background.started_tasks == []
