"""Heliox Live, part six: promises with a deadline, and an offer to set a reminder before it."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from pilot.live.commitments import deadline_of, find_commitments, remind_at
from pilot.live.detectors import TextContext
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation
from pilot.workflows.reminders import parse_reminder

MONDAY_11 = datetime(2026, 11, 9, 11, 0)
CHAT = TextContext(today=MONDAY_11.date(), app="slack.exe", title="#team - Slack")


def offers(text: str, now: datetime = MONDAY_11, ctx: TextContext = CHAT, reminders=None) -> list[str]:
    return [c.fix for c in find_commitments(text, ctx, now, reminders)]


@pytest.mark.parametrize(
    ("text", "wanted"),
    [
        ("Sounds good, I'll send you the deck by Friday.", "Remind me on Friday at 10am to send them the deck"),
        (
            "Let me check with legal and get back to you tomorrow.",
            "Remind me tomorrow at 10am to check with legal and get back to them",
        ),
        ("I will review the PR by 3pm.", "Remind me at 2pm to review the PR"),
        ("I can share the numbers by end of day.", "Remind me at 4pm to share the numbers"),
        ("I'll update the doc next week.", "Remind me on 16 November at 10am to update the doc"),
        ("Will do, I'll ping Sam tonight.", "Remind me at 7pm to ping Sam"),
        ("I'll have it ready on 18 November.", "Remind me on 18 November at 10am to have it ready"),
    ],
)
def test_promises_with_a_deadline_get_a_reminder_offer_that_the_reminder_parser_reads_back(text, wanted):
    assert offers(text) == [wanted]
    due, what = parse_reminder(wanted, MONDAY_11)
    assert due > MONDAY_11.timestamp() and what in wanted.casefold()


@pytest.mark.parametrize(
    "text",
    [
        "Did you send the deck by Friday?",
        "I'll think about it.",
        "He'll send it by Friday.",
        "I sent the deck on Friday.",
        "I'll be out on Friday.",
        "Let me know by Friday.",
        "If I finish early I'll send it Friday.",
        "I'll send it.",
        "I'll have to think about it.",
    ],
)
def test_questions_states_other_people_and_promises_without_a_deadline_are_left_alone(text):
    assert offers(text) == []


def test_the_message_quotes_their_words_and_the_offer_skips_what_is_already_set_or_already_late():
    [card] = find_commitments("Sure. I'll send you the deck by Friday.", CHAT, MONDAY_11)
    assert (card.title, card.message) == (
        "Want a reminder on Friday at 10am?",
        "You said you'd send them the deck by Friday.",
    )
    assert card.kind == "commitment" and card.fix_kind == "ask" and card.urgency == "low"
    assert offers("I'll send you the deck by Friday.", reminders=[{"text": "send the deck to Ana"}]) == []
    assert offers("I will review the PR by 11:30am.") == []  # the reminder would be due in the past
    terminal = TextContext(today=MONDAY_11.date(), app="WindowsTerminal.exe", title="bash")
    assert offers("echo I'll send it by Friday", ctx=terminal) == []


def test_deadlines_and_reminder_times():
    today = date(2026, 11, 9)
    assert deadline_of("by Friday", today) == (datetime(2026, 11, 13, 0, 0), False)
    assert deadline_of("by 3pm", today) == (datetime(2026, 11, 9, 15, 0), True)
    assert deadline_of("tonight", today) == (datetime(2026, 11, 9, 20, 0), True)
    assert deadline_of("next week", today) == (datetime(2026, 11, 16, 9, 0), False)
    assert deadline_of("whenever", today) == (None, False)
    assert remind_at(datetime(2026, 11, 13), False, MONDAY_11) == datetime(2026, 11, 13, 10, 0)
    assert remind_at(datetime(2026, 11, 9, 15), True, MONDAY_11) == datetime(2026, 11, 9, 14, 0)
    assert remind_at(datetime(2026, 11, 9, 11, 30), True, MONDAY_11) is None


def test_reminders_understand_weekdays_and_dates_now():
    monday_afternoon = datetime(2026, 11, 9, 14, 30)
    due, what = parse_reminder("remind me to call mom on friday at 5", monday_afternoon)
    assert datetime.fromtimestamp(due) == datetime(2026, 11, 13, 17, 0) and what == "call mom"  # not today at 5
    assert datetime.fromtimestamp(parse_reminder("remind me on monday at 4pm to stretch", monday_afternoon)[0]) == (
        datetime(2026, 11, 9, 16, 0)
    )
    assert datetime.fromtimestamp(parse_reminder("remind me on monday to pay rent", monday_afternoon)[0]) == (
        datetime(2026, 11, 16, 9, 0)
    )
    assert datetime.fromtimestamp(parse_reminder("remind me on 2 january to renew", monday_afternoon)[0]) == (
        datetime(2027, 1, 2, 9, 0)
    )
    assert parse_reminder("remind me on 31 february to party", monday_afternoon) is None


@pytest.mark.asyncio
async def test_the_offer_survives_sending_the_message():
    clock = {"t": 100.0}
    text = ["Sounds good, I'll send you the deck by Friday."]
    events: list[tuple[str, dict]] = []
    settings = SimpleNamespace(
        enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[], personal_timing=False
    )

    async def observe():
        return Observation(text=text[0], app="slack.exe", title="#team - Slack", source="focused_field")

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        return None

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        clock=lambda: clock["t"],
        today=lambda: MONDAY_11.date(),
        wallclock=lambda: MONDAY_11,
    )
    for _ in range(4):
        clock["t"] += TICK_SECONDS
        await live.tick()
    text[0] = ""  # sent before the pause was long enough
    for _ in range(40):
        clock["t"] += TICK_SECONDS
        await live.tick()
        await asyncio.sleep(0)
    [card] = [p for e, p in events if e == "live_interjection"]
    assert card["kind"] == "commitment" and card["fix"] == "Remind me on Friday at 10am to send them the deck"
    assert not [p for e, p in events if e == "live_interjection_withdrawn"]
