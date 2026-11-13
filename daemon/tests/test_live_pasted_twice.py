"""The same sentence twice in one field (pilot.live.detectors.pasted_twice): every text of its bench pack
(pilot/live/bench_scenarios/watch-pasted-twice.json) fires or stays quiet as the pack says."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from pilot.live.detectors import KIND_LABELS, KINDS, LOW, TextContext, detect
from pilot.live.engine import OWN_WORDS_KINDS, SENT_KINDS

PACK = json.loads(
    (Path(__file__).parents[1] / "pilot/live/bench_scenarios/watch-pasted-twice.json").read_text(encoding="utf-8")
)
TODAY = date(2026, 11, 13)


@pytest.mark.parametrize("scenario", PACK, ids=[s["name"] for s in PACK])
def test_the_pack(scenario):
    ctx = TextContext(today=TODAY, app=scenario["app"], title=scenario["title"], hour=11)
    text = " ".join(step["type"] for step in scenario["steps"] if "type" in step)
    kinds = [c.kind for c in detect(text, ctx) if c.kind == "pasted_twice"]
    assert kinds == scenario["expect"], text


def test_a_low_card_quoting_the_sentence():
    ctx = TextContext(today=TODAY, app="notepad.exe", title="notes.txt - Notepad", hour=11)
    text = "We should ship the first part on its own next week. We should ship the first part on its own next week."
    [card] = [c for c in detect(text, ctx) if c.kind == "pasted_twice"]
    assert (card.title, card.message, card.evidence, card.urgency, card.fix) == (
        "This sentence is in there twice",
        "Delete one of them.",
        "We should ship the first part on its own next week.",
        LOW,
        "",
    )


def test_a_switch_the_persons_own_words_and_no_you_sent_card():
    assert "pasted_twice" in KINDS and KIND_LABELS["pasted_twice"]
    assert "pasted_twice" in OWN_WORDS_KINDS and "pasted_twice" not in SENT_KINDS
