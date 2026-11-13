"""Tests for PilotServer's self_healing_status/self_healing_config_update
RPC handlers."""

from types import SimpleNamespace

import pytest

from pilot.agents.background import TaskStatus
from pilot.config import PilotConfig
from pilot.server import PilotServer


class _BackgroundStub:
    def __init__(self):
        self._tasks = {
            task_id: SimpleNamespace(status=TaskStatus.STOPPED)
            for task_id in ("monitor_cpu", "monitor_memory", "monitor_disk")
        }
        self.started: list[str] = []
        self.stopped: list[str] = []

    def start(self, task_id):
        self.started.append(task_id)
        self._tasks[task_id].status = TaskStatus.RUNNING
        return True

    def stop(self, task_id):
        self.stopped.append(task_id)
        self._tasks[task_id].status = TaskStatus.STOPPED
        return True

    def list_tasks(self):
        return [
            {
                "task_id": task_id,
                "status": task.status.value,
                "condition": f"{task_id} threshold",
                "interval_seconds": 10,
                "last_run": 0,
                "run_count": 0,
                "error_count": 0,
                "last_result": {},
            }
            for task_id, task in self._tasks.items()
        ]


@pytest.mark.asyncio
async def test_status_reports_config_defaults_with_no_engine():
    server = PilotServer(PilotConfig())
    result = await server._handle_self_healing_status({}, ws=None)
    assert result["status"] == "ok"
    assert result["enabled"] is False
    assert result["attempts"] == []


@pytest.mark.asyncio
async def test_config_update_persists_enabled_toggle(tmp_path, monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_self_healing_config_update({"enabled": True}, ws=None)
    assert result["status"] == "ok"
    assert result["enabled"] is True
    assert server.config.self_healing.enabled is True


@pytest.mark.asyncio
async def test_enabling_starts_only_watched_monitors(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)
    server._background = _BackgroundStub()

    result = await server._handle_self_healing_config_update(
        {"enabled": True, "watched_metrics": ["cpu", "disk"]},
        ws=None,
    )

    assert result["status"] == "ok"
    assert set(server._background.started) == {"monitor_cpu", "monitor_disk"}
    assert server._self_healing_started_monitors == {"monitor_cpu", "monitor_disk"}


@pytest.mark.asyncio
async def test_disabling_stops_only_self_healing_owned_monitors(monkeypatch):
    config = PilotConfig()
    config.self_healing.enabled = True
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)
    server._background = _BackgroundStub()
    server._background._tasks["monitor_memory"].status = TaskStatus.RUNNING
    server._sync_self_healing_monitors()

    await server._handle_self_healing_config_update({"enabled": False}, ws=None)

    assert set(server._background.stopped) == {"monitor_cpu", "monitor_disk"}
    assert server._background._tasks["monitor_memory"].status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_status_exposes_monitor_runtime():
    server = PilotServer(PilotConfig())
    server._background = _BackgroundStub()

    result = await server._handle_self_healing_status({}, ws=None)

    assert result["monitors"]["cpu"]["status"] == "stopped"


@pytest.mark.asyncio
async def test_config_update_sets_tiering_and_metrics(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_self_healing_config_update(
        {"auto_execute_max_tier": 2, "watched_metrics": ["cpu"]}, ws=None
    )
    assert result["status"] == "ok"
    assert result["auto_execute_max_tier"] == 2
    assert result["watched_metrics"] == ["cpu"]


@pytest.mark.asyncio
async def test_config_update_rejects_non_list_watched_metrics(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_self_healing_config_update({"watched_metrics": "cpu"}, ws=None)
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_config_update_rejects_unsupported_metric_without_partial_mutation(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_self_healing_config_update(
        {"enabled": True, "watched_metrics": ["gpu"]},
        ws=None,
    )

    assert result["status"] == "error"
    assert config.self_healing.enabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"enabled": "false"}, "enabled must be a boolean"),
        ({"auto_execute_max_tier": 1.5}, "auto_execute_max_tier must be an integer"),
        ({"cooldown_seconds": float("nan")}, "cooldown_seconds must be finite"),
    ],
)
async def test_config_update_rejects_coercive_values_atomically(monkeypatch, params, message):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: pytest.fail("invalid config was saved"))
    server = PilotServer(config)

    result = await server._handle_self_healing_config_update(params, ws=None)

    assert result == {"status": "error", "message": message}
    assert config.self_healing.enabled is False
    assert config.self_healing.auto_execute_max_tier == 1
    assert config.self_healing.cooldown_seconds == 600.0


@pytest.fixture(autouse=True)
def _the_person_approves_setting_changes(monkeypatch):
    """These tests are about checking and saving settings; asking before a widening is test_guard_changes.py's."""
    from pilot.server import PilotServer

    async def approved(self, ws, found):
        return None

    monkeypatch.setattr(PilotServer, "_approve_widenings", approved)
