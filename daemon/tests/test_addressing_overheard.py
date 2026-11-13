"""The count of utterances not taken since Heliox last spoke: a number only, reset when Heliox finishes speaking, and
never in the way of "stop", "no" and "wait"."""

from __future__ import annotations

import pytest

from pilot.system.addressing import Addressing, Heard

LINES = [
    "So I told him the meeting was moved to Thursday",
    "Did you see the game last night",
    "Pass me the charger please, the one on the shelf",
    "Thanks for watching and see you in the next video",
    "My password is hunter two, don't tell anyone",
    "The invoice for Acme is two thousand four hundred",
    "Honestly I think we should just order pizza",
    "Hang on, where did I put my keys this morning",
    "Mum called about Sunday lunch again",
    "Let's look at the third slide once more",
]


def open_after_an_answer(now: float = 100.0) -> Addressing:
    rule = Addressing(clock=lambda: now)
    rule.open_since = now - 3
    rule.heliox_finished()
    return rule


def heard(text: str, at: float, voiced: float = 2.5, *, over_heliox: bool = False) -> Heard:
    return Heard(text=text, started=at, ended=at + voiced, voiced_seconds=voiced, heliox_was_speaking=over_heliox)


def test_ten_overheard_lines_are_counted_and_no_word_of_them_is_kept():
    rule = open_after_an_answer()
    for n, line in enumerate(LINES):
        assert not rule.decide(heard(line, 100.5 + n * 0.1, voiced=9.0)).act  # a monologue: never taken
    assert rule.overheard_since_heliox == 10
    kept = repr(vars(rule)) + repr(rule.history)
    for line in LINES:
        for word in ("Thursday", "charger", "hunter", "Acme", "pizza", "keys", "Sunday", "slide"):
            if word in line:
                assert word not in kept
    rule.heliox_finished()
    assert rule.overheard_since_heliox == 0


def test_a_taken_follow_up_reads_the_count_and_does_not_add_to_it():
    rule = open_after_an_answer()
    rule.decide(heard("So anyway we went to lunch and then the whole thing fell apart", 100.5, voiced=9.0))
    assert rule.overheard_since_heliox == 1
    assert rule.decide(heard("Open my mail.", 101.5, voiced=1.0)).act
    assert rule.overheard_since_heliox == 1


@pytest.mark.parametrize("word", ["Stop.", "No.", "Wait.", "Stop", "no", "wait"])
def test_stop_no_and_wait_are_decided_before_the_count_whatever_it_is(word):
    rule = open_after_an_answer()
    rule.overheard_since_heliox = 7  # however much was overheard
    decision = rule.decide(heard(word, 101.0, voiced=0.6))
    assert decision.act and decision.why == "follow-up: control"
    over = open_after_an_answer()
    over.overheard_since_heliox = 7
    assert over.decide(heard(word, 101.0, voiced=0.6, over_heliox=True)).act  # over Heliox: the cut-off, a stop


def test_the_listener_shows_its_rule_for_the_count():
    from pilot.config import PilotConfig
    from pilot.system.voice import ContinuousVoiceListener

    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    assert listener.addressing.overheard_since_heliox == 0
