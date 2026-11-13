"""Settings > Live: what Live raised lately and how each kind was answered, counts only (LiveCompanion.answers)."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation


def build(tmp_path, text: list[str], muted: list[str] | None = None) -> tuple[LiveCompanion, list, dict]:
    clock = {"t": 100.0}
    events: list[tuple[str, dict]] = []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=muted or [])

    async def observe():
        return Observation(text=text[0], app="notepad.exe", title="notes.txt - Notepad", source="focused_field")

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        return "spoken"

    live = LiveCompanion(
        settings=lambda: settings, observe=observe, emit=emit, speak=speak, clock=lambda: clock["t"],
        today=lambda: date(2026, 11, 9), wallclock=lambda: datetime(2026, 11, 9, 11), relevance_path=tmp_path / "r.json",
    )  # fmt: skip
    return live, events, clock


async def run_for(live, clock, seconds: float) -> None:
    for _ in range(round(seconds / TICK_SECONDS)):
        clock["t"] += TICK_SECONDS
        await live.tick()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_counts_per_kind_with_the_persons_own_answers_and_no_text(tmp_path):
    text = ["Hotel 12 x 15 = 170 units. Meet on Tuesday, 14 November."]
    live, events, clock = build(tmp_path, text, muted=["repeated_word"])
    await run_for(live, clock, 4.0)
    cards = {p["kind"]: p for e, p in events if e == "live_interjection"}
    assert set(cards) == {"arithmetic", "weekday_date"}
    await live.feedback(cards["arithmetic"]["id"], "helpful", from_person=True)
    await live.feedback(cards["weekday_date"]["id"], "dismissed", from_person=True)
    rows = {row["kind"]: row for row in live.answers()}
    assert rows["arithmetic"] == {
        "kind": "arithmetic", "label": rows["arithmetic"]["label"], "raised": 1, "spoken": rows["arithmetic"]["spoken"],
        "thanks": 1, "not_now": 0, "muted_from_card": 0, "not_now_days": 0, "quiet": False, "muted": False,
    }  # fmt: skip
    assert rows["weekday_date"]["not_now"] == 1 and rows["weekday_date"]["not_now_days"] == 1
    assert rows["repeated_word"]["muted"] and rows["repeated_word"]["raised"] == 0
    status = live.status()
    assert status["answers"] == live.answers() and status["answers_of_last"] == 20
    blob = json.dumps(status["answers"])
    assert "170" not in blob and "14 November" not in blob  # counts only: no card text, no evidence


@pytest.mark.asyncio
async def test_a_card_left_alone_or_waved_away_at_the_end_is_not_counted_as_an_answer(tmp_path):
    text = ["Hotel 12 x 15 = 170 units."]
    live, events, clock = build(tmp_path, text)
    await run_for(live, clock, 4.0)
    [card] = [p for e, p in events if e == "live_interjection"]
    await live.feedback(card["id"], "dismissed")  # not from the person: a card that simply went unread
    [row] = live.answers()
    assert row["raised"] == 1 and row["thanks"] == row["not_now"] == row["muted_from_card"] == 0
