"""Tests for ReAct trace JSON export handler."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pilot.server import PilotServer


@pytest.mark.asyncio
async def test_export_react_trace_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr("pilot.server.Path.home", lambda: tmp_path)

    server = PilotServer.__new__(PilotServer)
    trace = {
        "version": 1,
        "exportedAt": "2026-05-18T00:00:00.000Z",
        "summary": {"eventCount": 2},
        "stages": [],
        "events": [{"timestamp": 1, "method": "status", "payload": {"phase": "planning"}}],
    }

    result = await server._handle_export_react_trace({"trace": trace}, MagicMock())

    assert result["status"] == "ok"
    assert result["format"] == "json"
    out_path = Path(result["path"])
    assert out_path.exists()
    assert out_path.suffix == ".json"
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["eventCount"] == 2


@pytest.mark.asyncio
async def test_export_react_trace_rejects_invalid_payload():
    server = PilotServer.__new__(PilotServer)
    result = await server._handle_export_react_trace({"trace": "not-an-object"}, MagicMock())
    assert result["status"] == "error"
