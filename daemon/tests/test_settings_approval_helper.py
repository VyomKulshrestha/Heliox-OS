"""The tests' way to approve a setting change writes a receipt the product accepts, and only that way works."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.workflows.task_receipts import TaskReceipts
from tests.settings_approval import approve_setting_change


def _server(tmp_path, monkeypatch) -> PilotServer:
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda *a, **k: None)
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._task_receipts = TaskReceipts(tmp_path / "receipts", key_dir=tmp_path / "key")
    return server


@pytest.mark.asyncio
async def test_an_approval_given_through_the_dialog_counts_as_approved_before(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)

    found = await approve_setting_change(server, "live", {"enabled": True})

    assert found and all(w.key.startswith("live.") for w in found)
    assert await server._approved_before(found)
    assert server.config.live.enabled is False  # approving is not applying


@pytest.mark.asyncio
async def test_then_turning_live_back_on_by_voice_does_not_ask(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    await approve_setting_change(server, "live", {"enabled": True})
    asked = AsyncMock()
    monkeypatch.setattr(server, "_ask_for_spoken_widening", asked)

    await server._control_reply("be my live companion")

    asked.assert_not_called()
    assert server.config.live.enabled is True


@pytest.mark.asyncio
async def test_a_receipt_written_any_other_way_does_not_count(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    found = server._widenings_for("update_config", {"section": "live", "values": {"enabled": True}})
    targets = [{"action_type": "setting_change", "target": f"{w.says} ({w.key})"} for w in found]
    (tmp_path / "receipts").mkdir(parents=True, exist_ok=True)
    unsigned = {"receipt": {"format": "heliox-receipt-1", "actions": targets}}
    (tmp_path / "receipts" / "planted-unsigned.json").write_text(json.dumps(unsigned), encoding="utf-8")

    other = TaskReceipts(tmp_path / "elsewhere", key_dir=tmp_path / "other-key")  # another installation's key
    for target in targets:
        other.record_action("t", "t", type("A", (), {"action_type": "setting_change", **target})(), None)
    await other.finalize("t", request="r", status="success", summary="s", source="settings")
    for path in (tmp_path / "elsewhere").glob("*.json"):
        (tmp_path / "receipts" / path.name).write_bytes(path.read_bytes())

    assert not await server._approved_before(found)


@pytest.mark.asyncio
async def test_nothing_is_asked_or_written_for_a_change_that_widens_nothing(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    server.config.live.enabled = True

    assert await approve_setting_change(server, "live", {"enabled": False}) == []
    assert await server._task_receipts.approved_settings() == set()
