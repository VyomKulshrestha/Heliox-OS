"""Reminders that came due while away reach the person once: one card, at most one sentence, the old ones logged."""

import time
from unittest.mock import AsyncMock

from pilot.config import PilotConfig
from pilot.server import PilotServer


async def test_one_card_one_sentence_and_a_log_line_for_the_old_ones():
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._memory = AsyncMock()
    now = time.time()
    items = [
        {"id": "a", "text": "call mum", "due": now - 3 * 3600, "late": True, "late_seconds": 3 * 3600},
        {"id": "b", "text": "renew the passport", "due": now - 9 * 86400, "late": True, "late_seconds": 9 * 86400},
    ]

    await server._deliver_missed_reminders(items, True)

    server._broadcast_notification.assert_awaited_once()
    method, payload = server._broadcast_notification.await_args.args
    assert method == "reminders_missed"
    assert [item["id"] for item in payload["reminders"]] == ["a", "b"]
    assert all("due" in item for item in payload["reminders"])
    server._speak_companion_text.assert_awaited_once()
    assert server._speak_companion_text.await_args.args[0] == (
        "While Heliox was off, two reminders came due: renew the passport and call mum."
    )
    server._memory.record_note.assert_awaited_once()  # only the one more than a week old
    assert server._memory.record_note.await_args.args[0] == "Reminder: renew the passport"
