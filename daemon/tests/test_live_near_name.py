"""A name one letter off someone the person told Heliox about (pilot.live.detectors.near_name): every sentence of
its bench pack (pilot/live/bench_scenarios/watch-near-name.json) fires or stays quiet as the pack says."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from pilot.live.detectors import KIND_LABELS, KINDS, TextContext, _one_letter_off, detect
from pilot.live.engine import OWN_WORDS_KINDS, SENT_KINDS

PACK = json.loads(
    (Path(__file__).parents[1] / "pilot/live/bench_scenarios/watch-near-name.json").read_text(encoding="utf-8")
)


def ctx_for(scenario: dict) -> TextContext:
    return TextContext(
        today=date.fromisoformat(scenario["today"]),
        app=scenario["app"],
        title=scenario["title"],
        hour=11,
        people=scenario["people"],
    )


@pytest.mark.parametrize("scenario", PACK, ids=[s["name"] for s in PACK])
def test_the_pack(scenario):
    text = " ".join(step["type"] for step in scenario["steps"] if "type" in step)
    kinds = [c.kind for c in detect(text, ctx_for(scenario)) if c.kind == "near_name"]
    assert kinds == scenario["expect"], text


@pytest.mark.parametrize(
    "word,name,off",
    [
        ("Priay", "Priya", True),  # the last two swapped
        ("Pirya", "Priya", True),
        ("Prya", "Priya", True),  # an inner letter dropped
        ("Danniel", "Daniel", True),  # an inner letter doubled
        ("Daniela", "Daniel", False),  # a letter added at the end: another name
        ("Hanna", "Hannah", False),  # the last letter dropped: another name
        ("Maria", "Mario", False),  # one letter for another: both exist
        ("Rpiya", "Priya", False),  # the first letter: not a slip of the hand
        ("Priya", "Priya", False),
    ],
)
def test_one_letter_off(word, name, off):
    assert _one_letter_off(word, name) is off


def test_the_card_asks_and_offers_the_spelling():
    scenario = PACK[0]
    [card] = [c for c in detect("Hi Priay, the deck is attached.", ctx_for(scenario)) if c.kind == "near_name"]
    assert (card.title, card.message, card.evidence, card.fix, card.fix_kind) == (
        "Did you mean Priya?",
        "You wrote Priay; your manager is Priya.",
        "Priay",
        "Priya",
        "replace",
    )
    assert card.detail.endswith("If Priay is someone else, ignore this.")


@pytest.mark.parametrize(
    "title,text,people",
    [
        # The window spells it that way: the recipient or the subject is that person.
        ("RE: plans - Hanah Jones - Message (HTML)", "Hi Hanah, thanks for the notes.", {"sister": "Hannah"}),
        # A surname that is not the known person's: someone else.
        ("RE: plans - Message (HTML)", "Hi Priay Kumar, welcome to the team.", {"manager": "Priya Shah"}),
        # Real names of different people, one letter for another or too short: never.
        ("RE: plans - Message (HTML)", "Hi Jan, see you Monday.", {"colleague": "Jon"}),
        ("RE: plans - Message (HTML)", "Hi Maria, see you Monday.", {"colleague": "Mario"}),
        ("RE: plans - Message (HTML)", "Hi Anna, see you Monday.", {"colleague": "Anne"}),
        ("RE: plans - Message (HTML)", "Hi Marie, see you Monday.", {"colleague": "Maria"}),
    ],
)
def test_a_different_persons_own_name_is_left_alone(title, text, people):
    ctx = TextContext(today=date(2026, 11, 13), app="OUTLOOK.EXE", title=title, hour=11, people=people)
    assert not [c for c in detect(text, ctx) if c.kind == "near_name"]


def test_the_known_surname_after_it_still_asks():
    people = {"manager": "Priya Shah"}
    ctx = TextContext(today=date(2026, 11, 13), app="OUTLOOK.EXE", title="RE: x - Message (HTML)", people=people)
    assert [c.kind for c in detect("Hi Priay Shah, welcome.", ctx) if c.kind == "near_name"] == ["near_name"]


def test_never_spoken_whatever_the_level():
    from pilot.live.policy import CARD_ONLY_KINDS, LEVELS, SHOW, Activity, Pending, decide

    [card] = [c for c in detect("Hi Priay, ok.", ctx_for(PACK[0])) if c.kind == "near_name"]
    item = Pending(card, "focused_field", 0.0, 0.0, window="app|title")
    assert "near_name" in CARD_ONLY_KINDS
    for name, level in LEVELS.items():
        assert decide(item, Activity(now=100.0, last_text_change=50.0), level, 0.9, frozenset()) == SHOW, name


def test_two_people_it_could_be_or_nobody_known_says_nothing():
    two = TextContext(
        today=date(2026, 11, 13), app="slack.exe", title="#t - Slack", people={"a": "Priya", "b": "Pryia"}
    )
    assert not [c for c in detect("Hi Prya, ok.", two) if c.kind == "near_name"]  # Priya or Pirya
    nobody = TextContext(today=date(2026, 11, 13), app="slack.exe", title="#t - Slack")
    assert not [c for c in detect("Hi Priay, ok.", nobody) if c.kind == "near_name"]


def test_a_switch_the_persons_own_words_and_no_you_sent_card():
    assert "near_name" in KINDS and KIND_LABELS["near_name"]
    assert "near_name" in OWN_WORDS_KINDS and "near_name" not in SENT_KINDS


def test_why_says_what_was_written_then_asks():
    from pilot.live.engine import because

    entry = {
        "kind": "near_name",
        "title": "Did you mean Priya?",
        "message": "You wrote Priay; your manager is Priya.",
        "detail": "You told me your manager is Priya. If Priay is someone else, ignore this.",
        "evidence": "Priay",
        "source": "focused_field",
        "app": "Outlook",
    }
    assert because(entry) == (
        'In Outlook, you wrote "Priay". Did you mean Priya? You told me your manager is Priya. '
        "If Priay is someone else, ignore this."
    )
