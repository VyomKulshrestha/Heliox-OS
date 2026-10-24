"""Privacy dashboard, forgetting a stretch of activity, and backup and restore."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, TimelineConfig
from pilot.workflows.activity_timeline import ActivityTimeline
from pilot.workflows.notes import NoteStore
from pilot.workflows.privacy import (
    clear_store,
    create_backup,
    forget_window,
    inventory,
    list_backups,
    restore_backup,
)

NOW = datetime(2026, 10, 24, 15, 0)


def test_inventory_lists_each_store_with_its_size(tmp_path):
    NoteStore(tmp_path / "notes.json").add("buy milk")

    rows = {row["key"]: row for row in inventory(tmp_path, {"notes": 1})}

    assert rows["notes"]["items"] == 1 and rows["notes"]["bytes"] > 0
    assert rows["timeline"]["bytes"] == 0 and rows["timeline"]["size"] == "0 bytes"
    assert set(rows) >= {"timeline", "receipts", "undo", "reminders", "shortcuts", "workspaces"}


@pytest.mark.parametrize(
    ("text", "minutes", "label"),
    [
        ("forget my activity from the last hour", 60, "the last 1 hour(s)"),
        ("delete my activity from the last 30 minutes", 30, "the last 30 minutes"),
        ("erase what I was doing in the past two hours", 120, "the last 2 hour(s)"),
    ],
)
def test_forget_window_recent(text, minutes, label):
    start, end, got_label = forget_window(text, NOW)

    assert (end - start, got_label) == (minutes * 60, label)


def test_forget_window_named_periods_and_other_text():
    start, end, label = forget_window("forget my activity from this morning", NOW)
    assert label == "this morning"
    assert forget_window("forget the milk", NOW) is None


def test_timeline_forget_removes_only_that_stretch(tmp_path):
    timeline = ActivityTimeline(tmp_path / "t.db", config=TimelineConfig(enabled=True))
    base = 1_800_000_000
    for second in range(0, 601, 30):
        timeline.observe("Code", "a.py", base + second)
    for second in range(4000, 4601, 30):
        timeline.observe("Chrome", "news", base + second)

    assert timeline.forget(base + 3900, base + 5000) == 1
    assert [span["app"] for span in timeline.spans(0, 2e10)] == ["Code"]


def test_backup_round_trip_without_secrets(tmp_path):
    data = tmp_path / "data"
    downloads = tmp_path / "Downloads"
    NoteStore(data / "notes.json").add("buy milk")
    (data / "receipts").mkdir(parents=True)
    (data / "receipts" / "r1.json").write_text("{}", encoding="utf-8")
    (data / "config.toml").write_text("token = 'secret'", encoding="utf-8")

    path = create_backup(data, downloads)

    names = set(zipfile.ZipFile(path).namelist())
    assert {"FORMAT", "notes.json", "receipts/r1.json"} <= names
    assert "config.toml" not in names and not any(name.startswith("timeline") for name in names)

    (data / "notes.json").unlink()
    restored = restore_backup(data, downloads, path.name)
    assert "notes.json" in restored
    assert json.loads((data / "notes.json").read_text(encoding="utf-8"))[0]["text"] == "buy milk"
    assert [row["name"] for row in list_backups(downloads)] == [path.name]


def test_restore_refuses_other_files_and_paths(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    evil = downloads / "heliox-backup-evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("FORMAT", "heliox-backup/1")
        archive.writestr("../../outside.json", "x")
        archive.writestr("config.toml", "token = 'x'")

    restore_backup(tmp_path / "data", downloads, evil.name)

    assert not (tmp_path / "outside.json").exists()
    assert not (tmp_path / "data" / "config.toml").exists()
    with pytest.raises(ValueError):
        restore_backup(tmp_path / "data", downloads, "../secrets.zip")
    plain = downloads / "heliox-backup-plain.zip"
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr("notes.json", "[]")
    with pytest.raises(ValueError, match="not a Heliox backup"):
        restore_backup(tmp_path / "data", downloads, plain.name)


def test_clear_store(tmp_path):
    NoteStore(tmp_path / "notes.json").add("x")

    clear_store(tmp_path, "notes")

    assert not (tmp_path / "notes.json").exists()
    with pytest.raises(ValueError):
        clear_store(tmp_path, "config")


def _server(tmp_path, monkeypatch):
    from pilot.server import PilotServer

    monkeypatch.setattr("pilot.server.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / "Downloads").mkdir()
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._activity_timeline = ActivityTimeline(
        tmp_path / "data" / "timeline" / "timeline.db", config=TimelineConfig(enabled=True)
    )
    server._notes = NoteStore(tmp_path / "data" / "notes.json")
    return server


@pytest.mark.asyncio
async def test_privacy_rpcs(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    server._notes.add("buy milk")

    listed = await server._handle_privacy_inventory({}, MagicMock())
    assert {row["key"]: row["items"] for row in listed["stores"]}["notes"] == 1
    assert (await server._handle_privacy_clear({"key": "notes"}, MagicMock()))["status"] == "error"
    cleared = await server._handle_privacy_clear({"key": "notes", "confirmed": True}, MagicMock())
    assert cleared["status"] == "ok" and server._notes.list() == []

    server._notes.add("keep me")
    made = await server._handle_backup_create({}, MagicMock())
    assert made["status"] == "ok"
    assert (await server._handle_backup_list({}, MagicMock()))["backups"][0]["name"] == made["name"]
    server._notes.delete(server._notes.list()[0]["id"])
    restored = await server._handle_backup_restore({"name": made["name"], "confirmed": True}, MagicMock())
    assert restored["status"] == "ok" and server._notes.list()[0]["text"] == "keep me"


@pytest.mark.asyncio
async def test_forget_by_voice(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    now = datetime.now().timestamp()
    server._activity_timeline.observe("Chrome", "news", now - 120)
    server._activity_timeline.observe("Chrome", "news", now - 60)

    reply = await server._control_reply("forget my activity from the last hour")

    assert reply == "Done. I removed 1 timeline entry from the last 1 hour(s)."
    assert server._activity_timeline.spans(0, 2e10) == []
