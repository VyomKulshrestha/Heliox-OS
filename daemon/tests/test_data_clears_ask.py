"""Clearing or restoring the person's data waits for their approval in the daemon's own dialog.

A "confirmed": true a client sends is a request, never the approval: before 13 November 2026 it was all privacy_clear,
undo_clear and backup_restore checked. Now the daemon asks, saying what will be deleted and how much, or what a
backup replaces; denied or unanswered, nothing changes and the reply is "cancelled".
"""

from __future__ import annotations

import io
import zipfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.workflows import privacy


class Person:
    def __init__(self, confirmed: bool) -> None:
        self.confirmed = confirmed
        self.asked: list[dict[str, Any]] = []

    async def __call__(self, plan_id, plan, ws, **kwargs):
        self.asked.append({"plan_id": plan_id, **kwargs})
        everything = set(range(len(plan.actions)))
        return self.confirmed, (everything if self.confirmed else set()), everything


def _server(person: Person, monkeypatch, tmp_path) -> PilotServer:
    import pilot.server as server_module

    monkeypatch.setattr(server_module, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()
    server = PilotServer(PilotConfig())
    server._wait_for_confirmation = person  # type: ignore[method-assign]
    server._privacy_counts = AsyncMock(return_value={"notes": 3})
    server._undo_journal = MagicMock(clear=AsyncMock())
    server._undo_busy_message = lambda: ""
    server._export_dir = lambda: tmp_path / "Downloads"
    return server


def _ws() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_clearing_a_store_says_what_and_how_much_and_a_no_keeps_it(monkeypatch, tmp_path):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch, tmp_path)
    notes = tmp_path / "data" / "notes.json"
    notes.write_text('[{"text": "a"}, {"text": "b"}, {"text": "c"}]', encoding="utf-8")

    reply = await server._handle_privacy_clear({"key": "notes", "confirmed": True}, _ws())

    assert reply["status"] == "cancelled" and notes.exists()
    line = person.asked[0]["shown"][0]
    assert line["target"] == "Delete Notes" and line["description"].startswith("3 items (")
    assert "cannot be undone" in line["description"] and person.asked[0]["plan_id"].startswith("data-")


@pytest.mark.asyncio
async def test_an_approved_clear_deletes(monkeypatch, tmp_path):
    person = Person(confirmed=True)
    server = _server(person, monkeypatch, tmp_path)
    notes = tmp_path / "data" / "notes.json"
    notes.write_text("[]", encoding="utf-8")

    reply = await server._handle_privacy_clear({"key": "notes"}, _ws())

    assert reply["status"] == "ok" and not notes.exists()


@pytest.mark.asyncio
async def test_clearing_undo_history_asks_and_a_no_keeps_it(monkeypatch, tmp_path):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch, tmp_path)

    reply = await server._handle_undo_clear({"confirmed": True}, _ws())

    assert reply["status"] == "cancelled"
    server._undo_journal.clear.assert_not_awaited()
    assert person.asked[0]["shown"][0]["target"] == "Delete Undo copies"


def _backup(folder, files: dict[str, str]) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    name = "heliox-backup-20261113-120000.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("FORMAT", privacy.BACKUP_FORMAT)
        for path, content in files.items():
            archive.writestr(path, content)
    (folder / name).write_bytes(buffer.getvalue())
    return name


@pytest.mark.asyncio
async def test_restoring_a_backup_names_what_it_replaces_and_a_no_restores_nothing(monkeypatch, tmp_path):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch, tmp_path)
    name = _backup(tmp_path / "Downloads", {"notes.json": "[]", "user_profile.json": "{}"})
    notes = tmp_path / "data" / "notes.json"
    notes.write_text('[{"text": "mine"}]', encoding="utf-8")

    reply = await server._handle_backup_restore({"name": name, "confirmed": True}, _ws())

    assert reply["status"] == "cancelled" and "mine" in notes.read_text(encoding="utf-8")
    described = person.asked[0]["shown"][0]["description"]
    assert "Notes" in described and "About you" in described


@pytest.mark.asyncio
async def test_an_approved_restore_restores(monkeypatch, tmp_path):
    person = Person(confirmed=True)
    server = _server(person, monkeypatch, tmp_path)
    name = _backup(tmp_path / "Downloads", {"notes.json": '[{"text": "from the backup"}]'})

    reply = await server._handle_backup_restore({"name": name}, _ws())

    assert reply["status"] == "ok" and "from the backup" in (tmp_path / "data" / "notes.json").read_text("utf-8")


def test_a_file_that_is_not_a_backup_is_refused_before_anything_is_asked(tmp_path):
    with pytest.raises(ValueError):
        privacy.backup_contents(tmp_path, "notes.json")


@pytest.mark.asyncio
async def test_each_recording_asks_naming_what_is_recorded_and_a_no_records_nothing(monkeypatch, tmp_path):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch, tmp_path)
    server._recorded_workflows = MagicMock()

    reply = await server._handle_recorded_workflow_start(
        {"name": "Expenses", "consent": True, "capture_manual": True, "manual_consent": True, "capture_visual": True},
        _ws(),
    )

    assert reply["status"] == "cancelled"
    server._recorded_workflows.start.assert_not_called()
    lines = {line["target"]: line["description"] for line in person.asked[0]["shown"]}
    assert "Record a workflow: Expenses" in lines
    assert "every key you press and every click" in lines["Record your own keyboard and mouse while recording"]
    assert "Take pictures of the screen while recording" in lines


@pytest.mark.asyncio
async def test_a_recording_is_asked_about_every_time(monkeypatch, tmp_path):
    person = Person(confirmed=True)
    server = _server(person, monkeypatch, tmp_path)
    server._recorded_workflows = MagicMock(start=MagicMock(return_value={"active": True}))
    server._announce_recorded_workflow_status = AsyncMock()

    for _ in range(2):
        await server._handle_recorded_workflow_start({"name": "Review", "consent": True}, _ws())

    assert len(person.asked) == 2
