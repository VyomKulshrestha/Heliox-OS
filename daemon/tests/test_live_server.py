"""Heliox Live wired into the daemon: voice toggles and replies, RPCs, supervision routing, hand-off."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, _merge_config
from pilot.live.engine import LiveCompanion, Observation
from pilot.workflows.routines import normalize_phrase


def make_server(tmp_path, text: str = "Lunch on Tuesday, 14 November with Sam"):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()

    async def observe():
        return Observation(text=text, app="notepad.exe", title="plan.txt - Notepad", source="focused_field")

    server._live = LiveCompanion(
        settings=lambda: server.config.live,
        observe=observe,
        emit=server._broadcast_notification,
        speak=server._live_speak,
        relevance_path=tmp_path / "rel.json",
        mute_kind=server._live_mute_kind,
        today=lambda: date(2026, 11, 9),
    )
    return server


def test_config_defaults_off_and_validates_on_load():
    assert PilotConfig().live.enabled is False
    merged = _merge_config(
        PilotConfig(), {"live": {"enabled": True, "level": "loud", "muted_kinds": ["arithmetic", "bogus"]}}
    )
    assert merged.live.enabled is True
    assert merged.live.level == "balanced"  # an unknown level is ignored
    assert merged.live.muted_kinds == ["arithmetic"]


@pytest.mark.asyncio
async def test_voice_toggles_turn_it_on_and_off_and_are_reserved(tmp_path):
    server = make_server(tmp_path)
    for phrase in ("be my live companion", "stop correcting me", "turn on the live companion"):
        assert server._phrase_reserved(normalize_phrase(phrase)), phrase
    assert (await server._control_reply("be my live companion")).startswith("Okay, I'm with you")
    assert server.config.live.enabled is True
    assert (await server._control_reply("stop correcting me")).startswith("Okay, I'll stay quiet")
    assert server.config.live.enabled is False


@pytest.mark.asyncio
async def test_rpcs_set_status_feedback_and_explain(tmp_path):
    server = make_server(tmp_path)
    status = await server._handle_live_set({"enabled": True, "level": "proactive"}, MagicMock())
    assert status["enabled"] is True and status["level"] == "proactive"
    assert {k["kind"] for k in status["kinds"]} >= {"weekday_date", "risky_command"}
    bad = await server._handle_live_set({"level": "loud"}, MagicMock())
    assert bad["status"] == "error"

    clock = [100.0]
    server._live._clock = lambda: clock[0]
    for _ in range(40):  # 8 seconds of micro-turns
        clock[0] += 0.2
        await server._live.tick()
    sent = [call.args for call in server._broadcast_notification.await_args_list]
    [card] = [payload for method, payload in sent if method == "live_interjection"]
    assert card["message"] == "14 November is a Saturday, not a Tuesday."

    explained = await server._handle_live_explain({"id": card["id"]}, MagicMock())
    assert "17 November" in explained["text"]
    assert (await server._handle_live_feedback({"id": card["id"], "verdict": "nope"}, MagicMock()))["status"] == "error"
    muted = await server._handle_live_feedback({"id": card["id"], "verdict": "mute"}, MagicMock())
    assert muted["status"] == "ok" and server.config.live.muted_kinds == ["weekday_date"]


@pytest.mark.asyncio
async def test_spoken_replies_follow_the_last_correction_and_fix_it_hands_off(tmp_path):
    server = make_server(tmp_path, "Error: listen EADDRINUSE: address already in use :::3000")
    server.config.live.enabled = True
    server._voice_command_dispatch = AsyncMock()
    assert await server._live_reply("fix it") is None  # nothing caught yet: left to the normal pipeline
    clock = [100.0]
    server._live._clock = lambda: clock[0]
    for _ in range(50):
        clock[0] += 0.2
        await server._live.tick()
    assert "port 3000" in await server._control_reply("why")
    assert await server._control_reply("fix it") == "On it."
    import asyncio

    await asyncio.sleep(0)
    server._voice_command_dispatch.assert_awaited_with("Show me which program is using port 3000")


@pytest.mark.asyncio
async def test_supervision_risk_warnings_become_live_cards_when_live_is_on(tmp_path):
    server = make_server(tmp_path, "nothing risky")
    await server._supervision_broadcast("supervision_risk_warning", {"pattern": "destructive_sql", "source": "ocr"})
    assert server._broadcast_notification.await_args.args[0] == "supervision_risk_warning"  # live off: the dialog

    server._broadcast_notification.reset_mock()
    server.config.live.enabled = True
    clock = [100.0]
    server._live._clock = lambda: clock[0]
    await server._supervision_broadcast("supervision_risk_warning", {"pattern": "destructive_sql", "source": "ocr"})
    assert not server._broadcast_notification.await_args_list  # not the dialog
    for _ in range(10):
        clock[0] += 0.2
        await server._live.tick()
    cards = [c.args[1] for c in server._broadcast_notification.await_args_list if c.args[0] == "live_interjection"]
    assert cards and cards[0]["urgency"] == "urgent" and "permanently" in cards[0]["message"]
