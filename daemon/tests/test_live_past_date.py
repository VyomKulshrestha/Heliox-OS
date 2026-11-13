"""A date that has already passed, written as if it were ahead (pilot.live.detectors.past_date): every sentence of
its bench pack (pilot/live/bench_scenarios/watch-past-date.json) fires or stays quiet as the pack says."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from pilot.live.detectors import KIND_LABELS, KINDS, TextContext, detect
from pilot.live.engine import OWN_WORDS_KINDS, SENT_KINDS

PACK = json.loads(
    (Path(__file__).parents[1] / "pilot/live/bench_scenarios/watch-past-date.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("scenario", PACK, ids=[s["name"] for s in PACK])
def test_the_pack(scenario):
    ctx = TextContext(
        today=date.fromisoformat(scenario["today"]), app=scenario["app"], title=scenario["title"], hour=11
    )
    text = " ".join(step["type"] for step in scenario["steps"] if "type" in step)
    kinds = [c.kind for c in detect(text, ctx) if c.kind == "past_date"]
    assert kinds == scenario["expect"], text


def test_the_card_says_when_it_was_and_what_today_is():
    ctx = TextContext(today=date(2026, 11, 13), app="slack.exe", title="#team - Slack", hour=11)
    [card] = [c for c in detect("See you on 5 November at the office.", ctx) if c.kind == "past_date"]
    assert (card.title, card.message, card.evidence) == (
        "5 November has already passed",
        "That was 8 days ago; today is Friday 13 November.",
        "5 November",
    )
    assert card.fix == "" and card.detail == "If you meant next year, add the year."


def test_a_date_more_than_45_days_back_or_spoken_is_left_alone():
    ctx = TextContext(today=date(2026, 11, 13), app="slack.exe", title="#team - Slack", hour=11)
    assert not [c for c in detect("See you on 20 September.", ctx) if c.kind == "past_date"]
    said = TextContext(today=date(2026, 11, 13), source="speech", hour=11)
    assert not [c for c in detect("see you on the fifth of November", said) if c.kind == "past_date"]


def test_a_switch_a_you_sent_card_and_the_persons_own_words():
    assert "past_date" in KINDS and KIND_LABELS["past_date"]
    assert "past_date" in SENT_KINDS and "past_date" in OWN_WORDS_KINDS
