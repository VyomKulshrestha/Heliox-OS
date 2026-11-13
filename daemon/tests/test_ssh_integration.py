"""Secure SSH configuration and connection-test contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType
from pilot.agents.ssh_agent import SshAgent
from pilot.config import PilotConfig, SshHostConfig
from pilot.server import PilotServer


def _server() -> PilotServer:
    server = PilotServer.__new__(PilotServer)
    server.config = PilotConfig()
    server.config.save = MagicMock()
    server._vault = MagicMock()
    server._vault.get_key = AsyncMock(return_value=None)
    server._vault.store_key = AsyncMock()
    server._vault.delete_key = AsyncMock()
    # The person approves in the daemon's dialog (test_guard_changes.py drives it); this file is about storage.
    server._approve_widenings = AsyncMock(return_value=None)
    return server


@pytest.mark.asyncio
async def test_save_host_keeps_private_key_out_of_config():
    server = _server()

    result = await server._handle_ssh_save_host(
        {
            "name": "build-box",
            "hostname": "10.0.0.7",
            "port": 22,
            "username": "builder",
            "private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n-----END OPENSSH PRIVATE KEY-----",
            "passphrase": "key-passphrase",
            "strict_host_key_checking": True,
            "enabled": True,
        },
        None,
    )

    assert result["status"] == "ok"
    host = server.config.ssh.allowed_hosts[0]
    assert host.private_key_provider == "ssh:build-box:private-key"
    assert host.passphrase_provider == "ssh:build-box:passphrase"
    assert not hasattr(host, "private_key")
    stored_providers = [call.args[0] for call in server._vault.store_key.await_args_list]
    assert stored_providers == ["ssh:build-box:private-key", "ssh:build-box:passphrase"]
    assert server._vault.store_key.await_count == 2
    server.config.save.assert_called_once()


@pytest.mark.asyncio
async def test_save_host_requires_key_for_new_alias_and_validates_port():
    server = _server()

    missing_key = await server._handle_ssh_save_host(
        {"name": "host", "hostname": "example.test", "username": "user", "port": 22},
        None,
    )
    bad_port = await server._handle_ssh_save_host(
        {
            "name": "host",
            "hostname": "example.test",
            "username": "user",
            "port": 70000,
            "private_key": "key",
        },
        None,
    )

    assert "private key" in missing_key["message"].lower()
    assert "port" in bad_port["message"].lower()
    server.config.save.assert_not_called()


@pytest.mark.asyncio
async def test_enable_switch_is_persisted_and_type_checked():
    server = _server()

    enabled = await server._handle_ssh_set_enabled({"enabled": True}, None)
    invalid = await server._handle_ssh_set_enabled({"enabled": "yes"}, None)

    assert enabled == {"status": "ok", "enabled": True}
    assert server.config.ssh.enabled is True
    assert "boolean" in invalid["message"]
    server.config.save.assert_called_once()


@pytest.mark.asyncio
async def test_list_and_delete_host_expose_no_secret_material():
    server = _server()
    server.config.ssh.allowed_hosts = [
        SshHostConfig(
            name="host",
            hostname="example.test",
            username="user",
            private_key_provider="ssh:host:private-key",
            passphrase_provider="ssh:host:passphrase",
        )
    ]
    server._vault.get_key = AsyncMock(side_effect=["private-key-material", "passphrase"])

    listed = await server._handle_ssh_list_hosts({}, None)

    assert listed["hosts"][0]["has_private_key"] is True
    assert listed["hosts"][0]["has_passphrase"] is True
    assert "private_key_provider" not in listed["hosts"][0]

    deleted = await server._handle_ssh_delete_host({"name": "host"}, None)
    assert deleted["status"] == "ok"
    assert server.config.ssh.allowed_hosts == []
    assert server._vault.delete_key.await_count == 2


@pytest.mark.asyncio
async def test_connection_test_fails_closed_for_unknown_alias():
    config = PilotConfig()
    config.ssh.enabled = True
    router = MagicMock()
    router.get_config.return_value = config
    router.get_vault.return_value = SimpleNamespace(get_key=AsyncMock())
    agent = SshAgent(router)

    result = await agent.test_connection("missing")

    assert result["status"] == "error"
    assert "unknown" in result["message"].lower()


@pytest.mark.asyncio
async def test_disabled_action_points_user_to_secure_settings():
    config = PilotConfig()
    router = MagicMock()
    router.get_config.return_value = config
    router.get_vault.return_value = SimpleNamespace(get_key=AsyncMock())
    agent = SshAgent(router)
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SSH_COMMAND,
                target="",
                parameters={"host": "build-box", "command": "uname -a"},
            )
        ]
    )

    result = await agent.handle_task("inspect build box", plan)

    assert len(result) == 1
    assert result[0].success is False
    assert result[0].error == "SSH is disabled. Enable it in Settings > Integrations > SSH."
