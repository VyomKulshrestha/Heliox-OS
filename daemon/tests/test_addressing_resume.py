"""'Sorry, what were you saying?' after a reply was cut short is a follow-up, never doubted into 'Sorry?'; only within
two minutes of the cut (the conversation layer's cut_reply_at), so these words widen nothing otherwise."""

from __future__ import annotations

import pytest

from pilot.system.addressing import RESUME_SECONDS, Addressing, Heard


def after_a_cut(cut_at: float = 100.0, *, open_conversation: bool = True) -> Addressing:
    rule = Addressing(clock=lambda: cut_at + 2)
    rule.reply_cut_at = cut_at
    if open_conversation:
        rule.open_since = cut_at
    return rule


def heard(text: str, at: float, *, over_heliox: bool = False) -> Heard:
    return Heard(text=text, started=at, ended=at + 1.0, voiced_seconds=1.0, heliox_was_speaking=over_heliox)


@pytest.mark.parametrize(
    "text",
    [
        "Sorry, what were you saying?",
        "What were you saying?",
        "Go on.",
        "You were saying?",
        "Carry on.",
        "Finish what you were saying.",
        "Sorry, what were you going to say?",
        "Keep going.",
    ],
)
def test_asking_for_the_rest_after_a_cut_is_taken(text):
    decision = after_a_cut().decide(heard(text, 101.5))
    assert decision.act and decision.why in {"follow-up: resume", "follow-up: control"}
    assert not decision.say_sorry


def test_after_a_quick_question_in_between_and_with_the_conversation_closed():
    # the person talked over the reply, asked the time, then wanted the rest 90 s after the cut
    decision = after_a_cut(open_conversation=False).decide(heard("Sorry, what were you saying?", 190.0))
    assert decision.act and decision.why == "follow-up: resume"


def test_not_without_a_cut_reply_nor_after_two_minutes_nor_over_heliox():
    plain = Addressing(clock=lambda: 102.0)
    plain.open_since = 100.0
    plain.heliox_finished()
    assert plain.decide(heard("Sorry, what were you saying?", 101.5)).why != "follow-up: resume"
    late = after_a_cut(open_conversation=False)
    assert not late.decide(heard("What were you saying?", 100.0 + RESUME_SECONDS + 1)).act
    over = after_a_cut().decide(heard("What were you saying, then, about the DNS settings", 101.0, over_heliox=True))
    assert over.why != "follow-up: resume"
    assert not after_a_cut().decide(heard("So what were you saying about Priya's party", 101.5)).why.endswith("resume")
