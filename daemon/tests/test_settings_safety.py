"""Settings changes are saved safely, never destroy the person's files, and announce themselves."""

from __future__ import annotations

import json
import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import pilot.config as config_module
from pilot.agents.autonomous import AutonomousJob, JobStatus
from pilot.config import ConfigWriteBlocked, PilotConfig
from pilot.server import CONFIG_MUTATING_RPC_METHODS, SETTINGS_RPC_METHODS, JsonRpcRequest, PilotServer


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(config_module, "RESTRICTIONS_FILE", tmp_path / "restrictions.toml")
    return tmp_path


def test_an_unreadable_config_file_is_kept_and_never_overwritten(config_dir):
    broken = '[voice]\ntts_engine = "kokoro"\n[mystery_section]\nx = 1\n'
    (config_dir / "config.toml").write_text(broken, encoding="utf-8")

    config = PilotConfig.load()

    assert config.load_errors and "config.toml could not be read" in config.load_errors[0]
    assert list(config_dir.glob("config.toml.unreadable-*"))[0].read_text(encoding="utf-8") == broken
    with pytest.raises(ConfigWriteBlocked):
        config.save()
    assert (config_dir / "config.toml").read_text(encoding="utf-8") == broken

    config.clear_load_errors()  # what Reset to defaults does
    config.save()
    assert tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))


def test_hand_edits_made_while_running_are_picked_up_and_overrides_survive(config_dir):
    config = PilotConfig()
    config.save()
    loaded = PilotConfig.load()
    loaded.set_runtime_override("security", "dry_run", True)
    text = (config_dir / "config.toml").read_text(encoding="utf-8")
    (config_dir / "config.toml").write_text(
        text.replace("rate_limit_rpm = 60", "rate_limit_rpm = 42"), encoding="utf-8"
    )

    assert loaded.reload_if_edited() is True
    assert loaded.model.rate_limit_rpm == 42
    assert loaded.security.dry_run is True  # this run's --dry-run stays on
    assert loaded.reload_if_edited() is False


def test_runtime_overrides_are_never_saved_as_settings(config_dir):
    config = PilotConfig()
    config.set_runtime_override("security", "dry_run", True)
    config.set_runtime_override("model", "ollama_model", "fallback:latest")

    config.save()

    saved = tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))
    assert saved["security"]["dry_run"] is False
    assert saved["model"]["ollama_model"] == PilotConfig().model.ollama_model
    assert config.security.dry_run is True


def test_configs_built_in_memory_never_read_the_real_files(config_dir):
    (config_dir / "config.toml").write_text("first_run_complete = true\n", encoding="utf-8")

    assert PilotConfig().reload_if_edited() is False


def _server() -> PilotServer:
    config = PilotConfig()
    config.save = MagicMock()
    return PilotServer(config)


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("server", {"auth_token": ""}, "only be changed in config.toml"),
        ("narration", {"enabled": True}, "narration_config_update"),
        ("self_healing", {"auto_execute_max_tier": 4}, "self_healing_config_update"),
        ("gateway", {"source_profiles": {}}, "gateway_policy_update"),
        ("model", {"budget_monthly_limit_usd": float("nan")}, "finite number"),
        ("model", {"gpu_memory_limit_mb": -5}, "cannot be negative"),
        ("calendar", {"caldav_password_provider": "heliox_mesh"}, "caldav vault entry"),
        ("restrictions", {"sandbox_allowed_commands": ["rm"]}, "cannot be changed from Settings"),
        ("restrictions", {"protected_folders": ["", "C:/Data"]}, "non-empty text"),
    ],
)
async def test_update_config_refuses_unsafe_or_dedicated_changes(section, values, message):
    server = _server()

    result = await server._handle_update_config({"section": section, "values": values}, MagicMock())

    assert result["status"] == "error" and message in result["message"]
    server.config.save.assert_not_called()


async def test_restrictions_can_be_edited_from_settings(config_dir):
    server = PilotServer(PilotConfig())

    result = await server._handle_update_config(
        {"section": "restrictions", "values": {"protected_folders": ["C:/Finance", "C:/Finance", " D:/Photos "]}},
        MagicMock(),
    )

    assert result == {"status": "ok", "values": {"protected_folders": ["C:/Finance", "D:/Photos"]}}
    saved = tomllib.loads((config_dir / "restrictions.toml").read_text(encoding="utf-8"))
    assert saved["protected_folders"] == ["C:/Finance", "D:/Photos"]


async def test_a_change_that_cannot_be_saved_is_not_left_active():
    server = _server()
    server.config.save.side_effect = OSError("disk full")
    before = server.config.narration.follow_up_enabled

    result = await server._handle_narration_config_update({"follow_up_enabled": not before}, MagicMock())

    assert result["status"] == "error" and "disk full" in result["message"]
    assert server.config.narration.follow_up_enabled == before


async def test_supervision_does_not_start_the_input_hook_when_saving_fails():
    server = _server()
    server.config.save.side_effect = OSError("locked")
    server._supervision_hook = SimpleNamespace(start=MagicMock(), stop=MagicMock(), is_running=lambda: False)

    result = await server._handle_supervision_config_update(
        {"enabled": True, "keyboard_mouse_hook_enabled": True}, MagicMock()
    )

    assert result["status"] == "error"
    server._supervision_hook.start.assert_not_called()
    assert server.config.supervision.enabled is False


async def test_reset_refuses_while_a_background_job_runs():
    server = _server()
    job = AutonomousJob(job_id="j1", goal="g", status=JobStatus.RUNNING)
    server._autonomous = SimpleNamespace(_active_tasks={"j1": object()}, get_job=lambda job_id: job)

    result = await server._handle_reset_config({}, MagicMock())

    assert result["status"] == "error" and "Stop the running tasks" in result["message"]
    server.config.save.assert_not_called()


async def test_reset_keeps_the_auth_token_restrictions_and_finished_setup(monkeypatch):
    server = _server()
    server.config.server.auth_token = "secret-token"
    server.config.restrictions.protected_folders = ["C:/Finance"]
    server.config.first_run_complete = True
    server.config.security.dry_run = True
    server._vault = SimpleNamespace(available=True)
    monkeypatch.setattr("pilot.air_handoff.AirHandoffManager", lambda vault, **kwargs: SimpleNamespace())
    monkeypatch.setattr("pilot.air_handoff.AirHandoffServer", lambda manager, **kwargs: SimpleNamespace(running=False))

    result = await server._handle_reset_config({}, MagicMock())

    assert result["status"] == "ok"
    assert server.config.server.auth_token == "secret-token"
    assert server.config.restrictions.protected_folders == ["C:/Finance"]
    assert server.config.first_run_complete is True
    assert server.config.security.dry_run is False


@pytest.mark.parametrize("name", ["heliox_mesh", "ssh:prod:private-key", "air_handoff:phone"])
async def test_api_key_rpcs_cannot_touch_vault_entries_owned_by_other_features(name):
    server = _server()
    server._vault = SimpleNamespace(store_key=AsyncMock(), delete_key=AsyncMock())

    stored = await server._handle_store_api_key({"provider": name, "api_key": "x"}, MagicMock())
    deleted = await server._handle_delete_api_key({"provider": name}, MagicMock())

    assert stored["status"] == deleted["status"] == "error"
    server._vault.store_key.assert_not_awaited()
    server._vault.delete_key.assert_not_awaited()


async def test_settings_changes_announce_config_updated_only_after_success():
    server = _server()
    server._broadcast_notification = AsyncMock()
    server._handlers = {
        "risk_gate_config_update": server._handle_risk_gate_config_update,
        "update_config": server._handle_update_config,
    }
    ok = JsonRpcRequest(method="risk_gate_config_update", params={"enabled": False}, id=1)
    response = await server._dispatch(ok, MagicMock())
    await server._announce_config_change(ok, response)
    server._broadcast_notification.assert_awaited_once_with(
        "config_updated", {"method": "risk_gate_config_update", "section": ""}
    )

    server._broadcast_notification.reset_mock()
    bad = JsonRpcRequest(method="update_config", params={"section": "server", "values": {"port": 1}}, id=2)
    response = await server._dispatch(bad, MagicMock())
    await server._announce_config_change(bad, response)
    assert json.loads(response)["result"]["status"] == "error"
    server._broadcast_notification.assert_not_awaited()


def test_settings_rpcs_do_not_wait_behind_a_running_task():
    assert {"update_config", "narration_config_update", "get_config"} <= SETTINGS_RPC_METHODS
    assert CONFIG_MUTATING_RPC_METHODS <= SETTINGS_RPC_METHODS


async def test_get_config_reports_unreadable_files(config_dir):
    (config_dir / "config.toml").write_text("[mystery]\nx = 1\n", encoding="utf-8")
    server = PilotServer(PilotConfig.load())

    data = await server._handle_get_config({}, MagicMock())

    assert data["load_errors"] and "server" not in data
