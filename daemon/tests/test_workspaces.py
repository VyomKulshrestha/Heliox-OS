"""Workspaces, task history search and timers."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.system.platform_detect import Platform
from pilot.workflows.reminders import ReminderStore
from pilot.workflows.task_receipts import TaskReceipts
from pilot.workflows.workspaces import WorkspaceStore, open_apps
from tests.test_task_receipts import FakeVault, _record


def test_store_saves_opens_lists_and_deletes(tmp_path):
    store = WorkspaceStore(tmp_path / "w.json")

    store.save("Coding!", ["Visual Studio Code", "Slack"])
    store.save("coding", ["Visual Studio Code", "Slack", "Google Chrome"])

    assert [item["name"] for item in store.list()] == ["coding"]
    assert store.open_request("Open my coding setup") == (
        "open Visual Studio Code, Slack and Google Chrome",
        'Workspace "coding": open Visual Studio Code, Slack and Google Chrome',
    )
    assert store.open_request("open my gaming setup") is None
    assert store.delete("coding") is True and store.list() == []
    with pytest.raises(ValueError, match="No app windows"):
        store.save("empty", [])


@pytest.mark.asyncio
async def test_open_apps_uses_friendly_names_and_skips_desktop_furniture(monkeypatch):
    processes = [
        {"ProcessName": "Code", "Description": "Visual Studio Code"},
        {"ProcessName": "explorer", "Description": "Windows Explorer"},
        {"ProcessName": "slack", "Description": ""},
        {"ProcessName": "Code", "Description": "Visual Studio Code"},
        {"ProcessName": "heliox", "Description": "Heliox OS"},
    ]

    async def fake_powershell(script):
        return 0, json.dumps(processes), ""

    monkeypatch.setattr("pilot.workflows.workspaces.CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr("pilot.workflows.workspaces.run_powershell", fake_powershell)

    assert await open_apps() == ["Visual Studio Code", "slack"]


def _server(tmp_path, monkeypatch, apps=("Visual Studio Code", "Slack")):
    from pilot.server import PilotServer

    async def fake_open_apps():
        return list(apps)

    monkeypatch.setattr("pilot.workflows.workspaces.open_apps", fake_open_apps)
    server = PilotServer(PilotConfig())
    server._workspaces = WorkspaceStore(tmp_path / "w.json")
    server._broadcast_notification = AsyncMock()
    return server


@pytest.mark.asyncio
async def test_saying_save_and_open_a_setup(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)

    reply = await server._control_reply("Save this as my coding setup")
    assert reply == (
        'Saved your coding (Visual Studio Code and Slack) setup. Say "open my coding setup" to bring it back.'
    )
    assert await server._expand_request("open my coding setup") == (
        "open Visual Studio Code and Slack",
        'Workspace "coding": open Visual Studio Code and Slack',
    )
    assert (await server._control_reply("what setups do I have")).startswith("Your setups: coding")
    assert await server._control_reply("delete my coding setup") == 'Deleted your "coding" setup.'


@pytest.mark.asyncio
async def test_workspace_rpcs(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)

    saved = await server._handle_workspace_save({"name": "writing"}, MagicMock())
    assert saved["status"] == "ok" and saved["workspace"]["apps"] == ["Visual Studio Code", "Slack"]
    assert (await server._handle_workspace_list({}, MagicMock()))["workspaces"][0]["name"] == "writing"
    assert (await server._handle_workspace_delete({"name": "writing"}, MagicMock()))["workspaces"] == []
    assert (await server._handle_workspace_delete({"name": "writing"}, MagicMock()))["status"] == "error"


@pytest.mark.asyncio
async def test_when_did_i_ask_you_finds_the_task(tmp_path):
    receipts = TaskReceipts(tmp_path / "r", vault=FakeVault())
    for task, request in (("a", "email the invoice to Sam"), ("b", "tidy my downloads")):
        _record(receipts, task)
        await receipts.finalize(task, request=request, status="success", summary="ok", source="voice")

    answer = receipts.answer("When did I ask you to email the invoice?")

    assert answer.startswith("On ") and answer.endswith('you asked: "email the invoice to Sam" (1 action, success).')
    assert receipts.answer("when did I ask you about the weather") == (
        "I have no task where you asked me about the weather."
    )
    assert [row["request"] for row in receipts.search("downloads")] == ["tidy my downloads"]


@pytest.mark.asyncio
async def test_receipts_search_rpc(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._task_receipts = TaskReceipts(tmp_path / "r", vault=FakeVault())
    _record(server._task_receipts, "a")
    await server._task_receipts.finalize("a", request="email Sam", status="success", summary="ok", source="voice")

    found = await server._handle_receipts_search({"query": "sam"}, MagicMock())
    assert [row["request"] for row in found["receipts"]] == ["email Sam"]
    assert (await server._handle_receipts_search({"query": ""}, MagicMock()))["status"] == "error"


@pytest.mark.parametrize(
    ("text", "reply", "seconds"),
    [
        ("set a timer for 10 minutes", "Timer set for 10 minutes.", 600),
        ("set a 5 min timer", "Timer set for 5 minutes.", 300),
        ("start a timer for 30 seconds", "Timer set for 30 seconds.", 30),
        ("set a timer for one hour", "Timer set for 1 hour.", 3600),
    ],
)
def test_timers_are_reminders(tmp_path, text, reply, seconds):
    now = datetime(2026, 10, 24, 10, 0)
    store = ReminderStore(tmp_path / "r.json", clock=lambda: now.timestamp())

    assert store.answer(text, now=now) == reply
    assert store.list()[0]["due"] == now.timestamp() + seconds
    assert store.list()[0]["text"].endswith("timer is done")


def test_timer_limits(tmp_path):
    now = datetime(2026, 10, 24, 10, 0)
    store = ReminderStore(tmp_path / "r.json", clock=lambda: now.timestamp())

    assert store.answer("set a timer for 5 seconds", now=now) == "A timer can run from 10 seconds to 24 hours."
