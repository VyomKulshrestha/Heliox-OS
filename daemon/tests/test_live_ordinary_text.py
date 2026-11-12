"""Ordinary writing and talk that must not raise anything, found by running Live over this repository's commit
messages and source comments and over the conversation bench's not-addressed sentences; plus the settings and
"why" for the kinds added since."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live import detectors
from pilot.live.detectors import TextContext, detect, only_quoted
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation, because

TODAY = date(2026, 11, 12)


def found(text):
    ctx = TextContext(today=TODAY, app="notepad.exe", title="notes.txt - Notepad", hour=11)
    return [(c.kind, c.evidence) for c in detect(text, ctx)]


@pytest.mark.parametrize(
    "text",
    [
        # Quoted examples are mentions, not slips (all found in this repository's history).
        'The card for "Tuesday 14 November" now says which day it is.',
        "Ships writing-idioms (misheard phrases such as “could of” and “sneak peak”).",
        'Answered from the evidence ("In Notepad, you wrote "Tuesday 14 November", and so on").',
        '"20% of 150, so 35" and "at 25:00" are checked only after a word that makes them a claim.',
        "Use `Tuesday the 12th` in the test.",
        # Doubled words that are not slips.
        "A tool call is read-only only when the server says so.",
        "Require an explicit opt-in in Settings.",
        "2026 is twenty twenty-six when read aloud.",
        "Probe the shared head on public public-corpus domains.",
        "Dev mode no longer renders [object Object] in the picker.",
        '"Got it" and okay okay stop the voice.',
        # About the rule, not a date.
        "Dates that do not exist, including 29 February in a year that isn't a leap year.",
    ],
)
def test_ordinary_writing_raises_nothing(text):
    assert found(text) == []


def test_the_same_slips_unquoted_are_still_caught():
    assert [kind for kind, _ in found("The review is on Tuesday 14 November.")] == ["weekday_date"]
    assert found("Allow bounded reference reference evaluation") == [("repeated_word", "reference reference")]
    assert not only_quoted('he said "x" and then x', "x")
    assert only_quoted('he said "x" twice', "x")


def test_a_failing_detector_is_logged_and_the_others_still_run(monkeypatch, caplog):
    def broken(text, ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(detectors.DETECTORS, "arithmetic", broken)
    with caplog.at_level(logging.WARNING, logger="pilot.live.detectors"):
        result = found("The review is on Tuesday 14 November and 2 + 2 = 5.")
    assert ("weekday_date", "Tuesday 14 November") in result
    assert "Live detector arithmetic failed" in caplog.text


def test_why_works_for_the_newer_kinds():
    said_vs_typed = {
        "kind": "said_vs_typed",
        "title": "You wrote Thursday 12 November for the dentist",
        "message": "You said Friday, which is Friday 13 November.",
        "detail": "A few minutes ago in Notepad: Thursday 12 November.",
        "evidence": "Friday",
        "source": "speech",
        "app": "",
    }
    assert because(said_vs_typed) == (
        "You said Friday, which is Friday 13 November, and you wrote Thursday 12 November for the dentist. "
        "A few minutes ago in Notepad: Thursday 12 November."
    )


@pytest.mark.asyncio
async def test_speech_not_said_to_heliox_gives_no_card(tmp_path):
    clock = SimpleNamespace(t=100.0)
    events, spoken = [], []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[])

    async def observe():
        return Observation(text="", app="notepad.exe", title="x - Notepad", source="focused_field", editable=False)

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        spoken.append(message)

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        relevance_path=tmp_path / "r.json",
        clock=lambda: clock.t,
        today=lambda: date(2026, 11, 9),
    )
    live.offer_speech("I'll send Tom the deck by tomorrow.", addressed=False)
    live.offer_speech("lunch is on Tuesday 14 November", addressed=False)
    for _ in range(int(3 / TICK_SECONDS)):
        clock.t += TICK_SECONDS
        await live.tick()
    for _ in range(3):
        await asyncio.sleep(0)
    assert [p for e, p in events if e == "live_interjection"] == [] and spoken == []


def test_offers_status_carries_the_getting_to_know_switch(tmp_path):
    from pilot.personal.offers import INTRO, Offers

    offers = Offers(mode=lambda: "auto", profile_empty=lambda: True, state_path=tmp_path / "o.json")
    assert offers.status()["intro"] is True
    offers.silence(INTRO)
    status = offers.status()
    assert status["intro"] is False and status["silenced"] == []  # its own switch, not a line in the list
    assert offers.next_offer() is None
    offers.unsilence(INTRO.key)
    assert offers.next_offer() is INTRO
