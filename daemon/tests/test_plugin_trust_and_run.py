"""A plugin is trusted only when a trusted key vouched for it, and running its tool goes through the plan path.

plugin_create signs every plugin it writes with a new key of its own, so a signature made with the key beside the
plugin says only that the files did not change since signing. Such a plugin loads, marked unverified. Running a tool
from the Plugins tab (plugin_run_tool) now asks in the approval dialog first and runs through the executor, as a
plan's plugin_call does; until 13 November 2026 it ran the plugin's code at once.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pilot.actions import ActionResult, ActionType
from pilot.config import PilotConfig
from pilot.plugins import PluginRegistry, sign_plugin_directory, verify_plugin_signature
from pilot.plugins.capabilities import PluginCapabilities
from pilot.server import PilotServer

MANIFEST = {
    "name": "greeter",
    "version": "1.0.0",
    "description": "says hello",
    "tools": [{"name": "greet", "description": "greet someone", "parameters": {}}],
    "runtime_type": "python",
    "entry_point": "plugin.py",
    "capabilities": PluginCapabilities().to_dict(),
}


def _plugin(folder) -> Any:
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (folder / "plugin.py").write_text("def handle_tool(name, args):\n    return {'ok': True}\n", encoding="utf-8")
    sign_plugin_directory(folder)
    return folder


def test_a_plugin_signed_only_by_its_own_key_is_unverified(tmp_path):
    folder = _plugin(tmp_path / "plugins" / "greeter")

    assert verify_plugin_signature(folder) is False
    registry = PluginRegistry(plugin_dirs=[tmp_path / "plugins"])
    registry.discover()
    assert registry.find_tool_plugin("greet").verified is False


def test_a_plugin_a_trusted_key_vouched_for_is_verified(tmp_path):
    folder = _plugin(tmp_path / "plugins" / "greeter")
    trusted = Ed25519PrivateKey.generate()
    # Re-sign with the trusted key, as a release would.
    from pilot.plugins import PLUGIN_SIGNATURE_FILE, _plugin_digest

    signature = trusted.sign(_plugin_digest(folder))
    import base64

    (folder / PLUGIN_SIGNATURE_FILE).write_text(base64.b64encode(signature).decode("ascii"), encoding="utf-8")

    assert verify_plugin_signature(folder, (trusted.public_key(),)) is True
    registry = PluginRegistry(plugin_dirs=[tmp_path / "plugins"], trusted_public_keys=[trusted.public_key()])
    registry.discover()
    assert registry.find_tool_plugin("greet").verified is True


class Person:
    def __init__(self, confirmed: bool) -> None:
        self.confirmed = confirmed
        self.asked: list[dict[str, Any]] = []

    async def __call__(self, plan_id, plan, ws, **kwargs):
        self.asked.append({"plan": plan, **kwargs})
        everything = set(range(len(plan.actions)))
        return self.confirmed, (everything if self.confirmed else set()), everything


def _server(person: Person, tmp_path) -> tuple[PilotServer, MagicMock]:
    server = PilotServer(PilotConfig())
    _plugin(tmp_path / "plugins" / "greeter")
    registry = PluginRegistry(plugin_dirs=[tmp_path / "plugins"])
    registry.discover()
    registry.call_tool = MagicMock(return_value={"ok": True})  # never reached directly
    server._plugin_registry = registry
    server._wait_for_confirmation = person  # type: ignore[method-assign]

    async def execute(plan, **kwargs):
        action = plan.actions[0]
        return [ActionResult(action=action, success=True, output=json.dumps({"ok": True}))]

    server._executor = MagicMock()
    server._executor.execute = AsyncMock(side_effect=execute)
    return server, registry.call_tool


@pytest.mark.asyncio
async def test_running_a_tool_asks_first_and_runs_through_the_executor(tmp_path):
    person = Person(confirmed=True)
    server, direct = _server(person, tmp_path)

    reply = await server._handle_plugin_run_tool({"tool_name": "greet", "args": {"who": "sam"}}, MagicMock())

    assert reply == {"result": {"ok": True}}
    plan = person.asked[0]["plan"]
    assert plan.actions[0].action_type == ActionType.PLUGIN_CALL and plan.actions[0].parameters.args == {"who": "sam"}
    assert "signed only by its own key" in person.asked[0]["reason"]
    assert server._executor.execute.await_args.kwargs["user_confirmed"] is True
    direct.assert_not_called()


@pytest.mark.asyncio
async def test_a_tool_the_person_does_not_approve_does_not_run(tmp_path):
    person = Person(confirmed=False)
    server, direct = _server(person, tmp_path)

    reply = await server._handle_plugin_run_tool({"tool_name": "greet"}, MagicMock())

    assert reply["status"] == "cancelled"
    server._executor.execute.assert_not_awaited()
    direct.assert_not_called()


@pytest.mark.asyncio
async def test_a_tool_no_plugin_provides_is_refused_without_asking(tmp_path):
    person = Person(confirmed=True)
    server, _ = _server(person, tmp_path)

    reply = await server._handle_plugin_run_tool({"tool_name": "rm_everything"}, MagicMock())

    assert "No installed plugin" in reply["error"] and person.asked == []
