"""Heliox Live, part three: corrections grounded in the person's own calendar, to-dos and reminders."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, time
from types import SimpleNamespace

import pytest

from pilot.live import grounded
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation
from pilot.live.grounded import (
    GroundedChecker,
    build_facts,
    clash,
    judge,
    linked,
    stated_day,
    stated_time,
    verify,
)
from pilot.live.grounded_eval import evaluate, load

MONDAY = date(2026, 11, 9)
WEEK = load()
FACTS = build_facts(today=MONDAY, events=WEEK["events"], todos=WEEK["todos"], reminders=WEEK["reminders"])
BY_ID = {fact.id: fact for fact in FACTS}


# ── reading days and times ──


def test_days_and_times_are_read_the_way_people_write_them():
    assert stated_day("Thursday works", MONDAY)[0] == date(2026, 11, 12)
    assert stated_day("see you Monday", MONDAY)[0] == MONDAY  # today is Monday
    assert stated_day("next Monday", MONDAY)[0] == date(2026, 11, 16)
    assert stated_day("tomorrow at 9", MONDAY)[0] == date(2026, 11, 10)
    assert stated_day("on 18 November", MONDAY)[0] == date(2026, 11, 18)
    assert stated_day("November 2", MONDAY)[0] == date(2027, 11, 2)  # already past this year
    assert stated_day("Is it Tuesday the 17th?", MONDAY) == (date(2026, 11, 17), "Tuesday the 17th")
    assert stated_day("we sat in the sun", MONDAY) == (None, "")  # "sat" and "sun" are words, not days
    assert stated_time("at 3pm") == time(15, 0) and stated_time("11:30") == time(11, 30)
    assert stated_time("lunch at 1") == time(13, 0) and stated_time("at 9") == time(9, 0)
    assert stated_time("noon") == time(12, 0) and stated_time("our 1:1") is None


def test_the_facts_are_one_line_each_and_only_the_ones_that_can_be_contradicted():
    facts = build_facts(
        today=MONDAY,
        events=[
            {"summary": "Offsite", "start": "2026-11-18"},
            {"summary": "Long ago", "start": "2026-01-01T09:00:00"},
            {"summary": "", "start": "2026-11-10T09:00:00"},
        ],
        todos=[{"text": "Write the plan"}, {"text": "Ship it", "due": "2026-11-13", "done": True}],
        reminders=[{"text": "Stretch", "due": datetime(2026, 11, 10, 17, 0).timestamp()}],
    )
    assert [f.text for f in facts] == [
        "Calendar: Offsite, Wednesday 18 November, all day",
        "Reminder: Stretch, Tuesday 10 November at 17:00",
    ]
    assert BY_ID["event:0"].text == "Calendar: Dentist, Thursday 12 November, 15:00 to 16:00"


# ── code: clashes, named things, deliberate changes ──


def test_a_plan_on_top_of_an_event_is_a_clash_but_talking_about_the_event_is_not():
    found = clash("Sure, Thursday at 3pm works for me.", FACTS, MONDAY)
    assert found.message == "You have Dentist from 15:00 to 16:00 on Thursday 12 November."
    assert found.evidence == "Thursday at 3pm" and found.kind == "personal_fact"
    assert clash("Sure, Thursday at 5pm works for me.", FACTS, MONDAY) is None
    assert clash("The dentist on Thursday is at 3pm, so I'll leave early.", FACTS, MONDAY) is None
    assert clash("Thursday at 3pm was fun.", FACTS, MONDAY) is None  # not a plan
    assert "all day" in clash("Let's meet on 18 November at 10am.", FACTS, MONDAY).message


def test_a_named_thing_on_the_wrong_day_is_corrected_in_code_with_its_fix():
    sentence = "The design review is on Thursday, right?"
    fact = linked(sentence, FACTS)
    assert fact.id == "event:1"
    found = verify(fact, sentence, sentence, MONDAY, linked_by_words=True)
    assert found.message == "Design review with Sam is on Tuesday 10 November, not Thursday 12 November."
    assert found.evidence == "Thursday" and found.fix == "Tuesday"
    assert linked("The review slides are due Thursday.", FACTS) is None  # the slides, not the review
    assert linked("Lunch with Priya on Monday", FACTS) is None  # a name in two facts links neither
    assert linked("Is our 1:1 still on for Tuesday the 17th?", FACTS).id == "event:4"


@pytest.mark.asyncio
async def test_deliberate_changes_are_left_alone():
    checker = GroundedChecker(None, lambda: FACTS, today=lambda: MONDAY)
    assert await checker.check("Remind me Wednesday to call the plumber again.") == []
    assert await checker.check("Can we move the design review to Thursday?") == []
    assert checker.last_verdict == "a deliberate change"
    # A change that lands on another event is still a clash.
    assert await checker.check("Can we move our call to Thursday at 3pm?")


# ── the model: links only, checked in code ──


def test_a_model_link_is_used_only_when_every_claim_checks_out():
    sentence = "Heading to the airport on Saturday for Berlin."
    facts = [BY_ID["event:2"], BY_ID["event:0"]]

    def answer(**fields):
        return json.dumps({"refers": True, "confidence": 0.9, **fields})

    found, why = judge(answer(fact="event:2", quote="Saturday"), sentence, facts, MONDAY)
    assert why == "accepted" and found.message == "Flight to Berlin is on Friday 13 November, not Saturday 14 November."
    assert judge('{"refers": false}', sentence, facts, MONDAY) == (None, "no_reference")
    assert judge("not json", sentence, facts, MONDAY)[1] == "rejected: not JSON"
    assert judge(answer(fact="event:9", quote="Saturday"), sentence, facts, MONDAY)[1] == "rejected: unknown fact"
    assert (
        judge(answer(fact="event:2", quote="Sunday"), sentence, facts, MONDAY)[1]
        == "rejected: quote not in the sentence"
    )
    weak = json.dumps({"refers": True, "fact": "event:0", "quote": "Saturday", "confidence": 0.6})
    assert judge(weak, sentence, facts, MONDAY)[1] == "rejected: weak link"
    agrees = "Heading to the airport on Friday for Berlin."
    assert judge(answer(fact="event:2", quote="Friday"), agrees, facts, MONDAY)[1] == "rejected: the facts agree"


@pytest.mark.asyncio
async def test_the_model_is_asked_only_when_code_cannot_tell_and_its_failure_is_harmless():
    asked: list[str] = []

    async def model(system: str, prompt: str) -> str:
        asked.append(prompt)
        return '{"refers": true, "fact": "event:2", "quote": "Saturday", "confidence": 0.9}'

    checker = GroundedChecker(model, lambda: FACTS, today=lambda: MONDAY)
    assert await checker.check("The design review is on Tuesday at 11.") == []  # named and right: no model
    assert await checker.check("I think the review went well.") == []  # no day: nothing to check
    assert asked == []
    [card] = await checker.check("Heading to the airport on Saturday for Berlin.")
    assert card.detail.startswith("From your calendar: Flight to Berlin") and len(asked) == 1
    assert "[event:2] Calendar: Flight to Berlin" in asked[0]

    async def down(system: str, prompt: str) -> str:
        raise ConnectionError("Ollama is not running")

    offline = GroundedChecker(down, lambda: FACTS, today=lambda: MONDAY)
    assert await offline.check("Heading to the airport on Saturday for Berlin.") == []
    assert "Ollama is not running" in offline.stats()["error"]
    assert await offline.check("Sure, Thursday at 3pm works for me.")  # code checks need no model


@pytest.mark.asyncio
async def test_code_alone_is_right_every_time_it_speaks_on_the_fixed_week():
    result = await evaluate(None)
    assert result["precision"] == 1.0 and result["false_alarms"] == 0
    assert result["recall"] >= 0.84 and result["model_calls"] == 0
    misses = [row["text"] for row in result["rows"] if not row["pass"]]
    assert all("dental" in text or "airport" in text for text in misses), misses


def test_the_prompt_is_local_and_never_sees_more_than_a_few_facts():
    prompt = grounded.build_prompt("x", FACTS[:2], MONDAY)
    assert prompt.startswith("Today is Monday 9 November 2026.") and "[event:0]" in prompt
    assert len(grounded.candidates("Thursday was a long day.", FACTS, MONDAY)) <= grounded.MAX_FACTS


# ── in the live engine ──


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


async def run_for(live, clock, seconds: float) -> None:
    for _ in range(round(seconds / TICK_SECONDS)):
        clock.t = round(clock.t + TICK_SECONDS, 3)
        await live.tick()
        for _ in range(3):
            await asyncio.sleep(0)


def build(text: list[str], enabled: bool = True):
    clock = Clock()
    events: list[tuple[str, dict]] = []
    settings = SimpleNamespace(
        enabled=True,
        level="balanced",
        listen_to_speech=True,
        muted_kinds=[],
        personal_timing=False,
        grounded_checks=enabled,
    )
    speaking = {"now": False}

    async def observe():
        return Observation(text=text[0], app="OUTLOOK.EXE", title="Re: next week - Message", source="focused_field")

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        return None

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        user_speaking=lambda: speaking["now"],
        grounded=GroundedChecker(None, lambda: FACTS, today=lambda: MONDAY),
        clock=clock,
        today=lambda: MONDAY,
    )
    return live, clock, events, speaking


@pytest.mark.asyncio
async def test_a_clash_becomes_a_card_at_the_pause_and_goes_away_when_fixed():
    text = ["Sure, Thursday at 3pm works for me."]
    live, clock, events, _ = build(text)
    await run_for(live, clock, 3.0)
    [card] = [p for e, p in events if e == "live_interjection"]
    assert card["kind"] == "personal_fact" and card["evidence"] == "Thursday at 3pm"
    await run_for(live, clock, 3.0)  # still there: the rule detectors never find it, but its words are on screen
    assert not [p for e, p in events if e == "live_interjection_withdrawn"]
    text[0] = "Sure, Thursday at 5pm works for me."
    await run_for(live, clock, 3.0)
    assert ("live_interjection_withdrawn", {"id": card["id"], "reason": "fixed"}) in events
    assert live.status()["grounded"]["checks"] >= 1


@pytest.mark.asyncio
async def test_off_means_no_checks_and_speech_is_checked_too():
    text = ["Sure, Thursday at 3pm works for me."]
    live, clock, events, speaking = build(text, enabled=False)
    await run_for(live, clock, 4.0)
    assert not [p for e, p in events if e == "live_interjection"]

    live, clock, events, speaking = build(["nothing here"])
    speaking["now"] = True
    live.offer_speech("book lunch with Sam on Thursday at 3pm")
    await run_for(live, clock, 1.0)
    assert not [p for e, p in events if e == "live_interjection"]  # never over their voice
    speaking["now"] = False
    await run_for(live, clock, 1.0)
    [card] = [p for e, p in events if e == "live_interjection"]
    assert card["kind"] == "personal_fact" and card["source"] == "speech"
