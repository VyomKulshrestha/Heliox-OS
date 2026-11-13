from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


@pytest.fixture(autouse=True)
def _the_person_approves(monkeypatch):
    """The person says yes in the daemon's dialog; test_guard_changes.py drives that dialog. This file is about the
    work that follows."""

    async def approve(self, ws, found):
        return None

    async def yes(self, ws, lines):
        return True

    monkeypatch.setattr(PilotServer, "_approve_widenings", approve)
    monkeypatch.setattr(PilotServer, "_ask_about_data", yes)


class _Vault:
    def __init__(self, secret=None):
        self.secret = secret
        self.stored = []
        self.deleted = []

    async def get_key(self, provider):
        assert provider == "heliox_mesh"
        return self.secret

    async def store_key(self, provider, value):
        assert provider == "heliox_mesh"
        self.secret = value
        self.stored.append(value)

    async def delete_key(self, provider):
        assert provider == "heliox_mesh"
        self.secret = None
        self.deleted.append(provider)


class _Mesh:
    def __init__(self, *, start_error=None):
        self.instance_id = "mesh-local"
        self.peer_ids = []
        self.start_error = start_error
        self.started = False
        self.stopped = False
        self.collab_executor = object()

    async def start(self):
        if self.start_error:
            raise self.start_error
        self.started = True

    async def stop(self):
        self.stopped = True


def _server(monkeypatch, secret=None):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", MagicMock())
    server = PilotServer(config)
    server._vault = _Vault(secret)
    return server


@pytest.mark.asyncio
async def test_mesh_status_distinguishes_configured_and_live_state(monkeypatch):
    server = _server(monkeypatch, "s" * 32)
    server.config.network.enabled = True
    server._mesh_error = "port unavailable"

    result = await server._handle_mesh_status({}, ws=None)

    assert result["status"] == "ok"
    assert result["enabled"] is False
    assert result["configured_enabled"] is True
    assert result["authenticated"] is False
    assert result["secret_configured"] is True
    assert result["reason"] == "port unavailable"


@pytest.mark.asyncio
async def test_mesh_configure_requires_secret_before_enable(monkeypatch):
    server = _server(monkeypatch)

    result = await server._handle_mesh_configure({"enabled": True}, ws=None)

    assert result["status"] == "error"
    assert "shared secret" in result["message"]
    assert server.config.network.enabled is False
    server.config.save.assert_not_called()


@pytest.mark.asyncio
async def test_mesh_configure_stores_secret_and_starts_runtime(monkeypatch):
    server = _server(monkeypatch)
    server._executor = SimpleNamespace(set_collab_executor=MagicMock())
    mesh = _Mesh()
    monkeypatch.setattr(server, "_new_mesh", lambda secret: mesh)

    result = await server._handle_mesh_configure(
        {
            "enabled": True,
            "skill_sync_enabled": True,
            "collab_exec_enabled": False,
            "shared_secret": "s" * 32,
        },
        ws=None,
    )

    assert result["status"] == "ok"
    assert result["enabled"] is True
    assert result["authenticated"] is True
    assert mesh.started is True
    assert server._vault.stored == ["s" * 32]
    assert server.config.network.skill_sync_enabled is True
    assert server.config.network.collab_exec_enabled is False
    assert server._executor.set_collab_executor.call_args_list == [call(None), call(mesh.collab_executor)]
    server.config.save.assert_called_once()


@pytest.mark.asyncio
async def test_mesh_configure_disable_stops_runtime(monkeypatch):
    server = _server(monkeypatch, "s" * 32)
    server.config.network.enabled = True
    server._mesh = _Mesh()

    result = await server._handle_mesh_configure({"enabled": False}, ws=None)

    assert result["status"] == "ok"
    assert result["enabled"] is False
    assert server._mesh is None
    assert server.config.network.enabled is False


@pytest.mark.asyncio
async def test_mesh_start_failure_rolls_back_config_and_new_secret(monkeypatch):
    server = _server(monkeypatch)
    failed_mesh = _Mesh(start_error=OSError("port busy"))
    monkeypatch.setattr(server, "_new_mesh", lambda secret: failed_mesh)

    result = await server._handle_mesh_configure(
        {"enabled": True, "shared_secret": "n" * 32},
        ws=None,
    )

    assert result == {"status": "error", "message": "Peer Mesh was not changed: port busy"}
    assert server.config.network.enabled is False
    assert server._vault.secret is None
    assert server._vault.deleted == ["heliox_mesh"]
    assert failed_mesh.stopped is True


@pytest.mark.asyncio
async def test_mesh_persistence_failure_restores_previous_runtime_and_secret(monkeypatch):
    previous_secret = "p" * 32
    replacement_secret = "n" * 32
    server = _server(monkeypatch, previous_secret)
    server.config.network.enabled = True
    previous_mesh = _Mesh()
    restored_mesh = _Mesh()
    server._mesh = previous_mesh
    server.config.save.side_effect = [OSError("disk full"), None]

    def new_mesh(secret):
        assert secret == previous_secret
        return restored_mesh

    monkeypatch.setattr(server, "_new_mesh", new_mesh)

    result = await server._handle_mesh_configure(
        {"enabled": True, "shared_secret": replacement_secret},
        ws=None,
    )

    assert result == {"status": "error", "message": "Peer Mesh was not changed: disk full"}
    assert previous_mesh.stopped is True
    assert restored_mesh.started is True
    assert server._mesh is restored_mesh
    assert server.config.network.enabled is True
    assert server._vault.secret == previous_secret


@pytest.mark.asyncio
async def test_mesh_configure_rejects_truthy_string_atomically(monkeypatch):
    server = _server(monkeypatch, "s" * 32)

    result = await server._handle_mesh_configure({"collab_exec_enabled": "false"}, ws=None)

    assert result == {"status": "error", "message": "collab_exec_enabled must be a boolean"}
    assert server.config.network.collab_exec_enabled is False
    server.config.save.assert_not_called()


@pytest.mark.asyncio
async def test_mesh_secret_generation_and_disabled_clear(monkeypatch):
    server = _server(monkeypatch, "s" * 32)

    generated = await server._handle_mesh_generate_secret({}, ws=None)
    cleared = await server._handle_mesh_clear_secret({}, ws=None)

    assert generated["status"] == "ok"
    assert len(generated["shared_secret"].encode("utf-8")) >= 32
    assert cleared == {"status": "ok"}
    assert server._vault.secret is None


@pytest.mark.asyncio
async def test_mesh_secret_clear_is_blocked_while_enabled(monkeypatch):
    server = _server(monkeypatch, "s" * 32)
    server.config.network.enabled = True
    server._vault.delete_key = AsyncMock()

    result = await server._handle_mesh_clear_secret({}, ws=None)

    assert result["status"] == "error"
    server._vault.delete_key.assert_not_awaited()
