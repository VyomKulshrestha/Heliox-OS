"""Statements about the person in an open follow-up window, and a conversation that a long answer does not use up.

"Call me Sam." is taken; then "I start work at nine." and "I stop work at six." were "follow-up: talk", and
"Remember that I'm working on the podcast launch." came "too long after Heliox spoke" (the Live agent's day, 13
November). A sentence in a form the personal layer keeps is now taken in the window as "follow-up: about me", said
back at once, at most ABOUT_ME_IN_ROW in a row without the wake phrase, never over Heliox's voice and never when the
second-voice rule is on and the room was heard.

And after a long answer, a follow-up said seconds after its last sound was "no conversation open": the
conversation's 20 s of quiet was counted from the person's request, so a 20-second answer used it up. It now runs
from Heliox's last word, on every road that ends a reply (spoken, shown with the voice off, cut by the person).
"""

from __future__ import annotations

import pytest

from pilot.system.addressing import ABOUT_ME_IN_ROW, CLOSE_SECONDS, Addressing, Heard, hand_about_me
from pilot.workflows.user_profile import tells_about_self


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def said(rule: Addressing, clock: Clock, text: str, *, over_heliox: bool = False, button: bool = False):
    voiced = max(0.6, len(text.split()) / 2.5)
    started = clock.t
    clock.t += voiced
    return rule.decide(
        Heard(
            text=text,
            started=started,
            ended=clock.t,
            voiced_seconds=voiced,
            heliox_was_speaking=over_heliox,
            by_button=button,
        )
    )


def heliox_says(rule: Addressing, clock: Clock, seconds: float, *, after: float = 0.5) -> None:
    """Heliox answers after ``after`` seconds and speaks for ``seconds``; the listener is told when its voice ends."""
    clock.t += after + seconds
    rule.heliox_finished()


def opened(**kwargs) -> tuple[Addressing, Clock]:
    clock = Clock()
    rule = Addressing(clock=clock, **kwargs)
    hand_about_me(rule, None)
    taken = said(rule, clock, "Hey Heliox, good morning.")
    assert taken.act and taken.why == "wake"
    heliox_says(rule, clock, 1.5)
    return rule, clock


# ── about me ──


@pytest.mark.parametrize(
    "text",
    [
        "I start work at nine.",
        "I stop work at six.",
        "Remember that I'm working on the podcast launch.",
        "My sister is Ana.",
        "I prefer short answers.",
        "My goal is to run a marathon.",
    ],
)
def test_the_forms_the_profile_keeps_are_taken_in_the_window(text):
    rule, clock = opened()
    clock.t += 1.0
    decision = said(rule, clock, text)
    assert decision.act and decision.why == "follow-up: about me"


def test_a_corrected_statement_is_recognised_and_taken():
    assert tells_about_self("Actually, I start work at ten.")
    rule, clock = opened()
    clock.t += 1.0
    assert said(rule, clock, "Actually, I start work at ten.").act  # "actually" leads a correction: taken anyway


@pytest.mark.parametrize(
    "text",
    [
        "I always forget my keys.",
        "I usually take the bus.",
        "My boss is annoying today.",
        "My brother is in Berlin this week.",
        "He is my brother.",
        "I'd rather not talk about it.",
        "I went to the market and bought too much fruit.",
    ],
)
def test_talk_that_only_looks_like_it_is_not_taken(text):
    assert not tells_about_self(text)
    rule, clock = opened()
    clock.t += 1.0
    assert not said(rule, clock, text).act


def test_the_live_day_keeps_every_statement_said_in_one_go():
    rule, clock = opened()
    for text in ("Call me Sam.", "I start work at nine.", "I stop work at six."):
        clock.t += 1.0
        assert said(rule, clock, text).act, text
        heliox_says(rule, clock, 2.0)  # the say-back ("I'll call you Sam."): the window starts again from its end
    clock.t += 1.0
    taken = said(rule, clock, "Remember that I'm working on the podcast launch.")
    assert taken.act and taken.why == "follow-up: about me"


def test_a_few_in_a_row_then_the_wake_phrase_again():
    rule, clock = opened()
    statements = ["I start work at nine.", "I stop work at six.", "My sister is Ana.", "My brother is Tom."]
    for text in statements[:ABOUT_ME_IN_ROW]:
        clock.t += 1.0
        assert said(rule, clock, text).act, text
        heliox_says(rule, clock, 2.0)
    clock.t += 1.0
    assert not said(rule, clock, statements[ABOUT_ME_IN_ROW]).act
    clock.t += 1.0
    assert said(rule, clock, "Hey Heliox, my brother is Tom.").act  # the wake phrase starts the count again
    heliox_says(rule, clock, 2.0)
    clock.t += 1.0
    assert said(rule, clock, "I prefer short answers.").act


def test_never_over_heliox_voice():
    rule, clock = opened()
    clock.t += 1.0
    assert not said(rule, clock, "I start work at nine.", over_heliox=True).act


def test_not_when_the_second_voice_rule_is_on_and_the_room_was_heard():
    rule, clock = opened()
    rule.second_voice_threshold = 1
    clock.t += 0.5
    assert not said(rule, clock, "Then we went to the market.").act  # someone else in the room
    clock.t += 0.5
    assert not said(rule, clock, "I start work at nine.").act
    rule.second_voice_threshold = 0  # the rule off (the default): the same sentence is taken
    assert said(rule, clock, "I start work at nine.").act


def test_the_settings_give_the_threshold_only_when_the_rule_is_on():
    from types import SimpleNamespace

    rule = Addressing()
    hand_about_me(
        rule, SimpleNamespace(conversation=SimpleNamespace(second_voice_guard=False, second_voice_threshold=2))
    )
    assert rule.second_voice_threshold == 0 and rule.about_me is tells_about_self
    hand_about_me(
        rule, SimpleNamespace(conversation=SimpleNamespace(second_voice_guard=True, second_voice_threshold=2))
    )
    assert rule.second_voice_threshold == 2


def test_not_without_an_open_conversation():
    clock = Clock()
    rule = Addressing(clock=clock)
    hand_about_me(rule, None)
    assert not said(rule, clock, "I start work at nine.").act


# ── the conversation's quiet runs from Heliox's last word ──


@pytest.mark.parametrize(
    "road",
    [
        "spoken",  # the voice played the whole answer: the listener is told when it ended
        "voice off",  # shown as text: told as soon as it is shown, after a long wait for the model
        "cut",  # the person talked over it: told when the voice stopped
    ],
)
def test_a_follow_up_three_seconds_after_a_long_answer_is_taken(road):
    clock = Clock()
    rule = Addressing(clock=clock)
    assert said(rule, clock, "Hey Heliox, what do you know about me?").act
    if road == "spoken":
        heliox_says(rule, clock, 20.0, after=1.0)
    elif road == "voice off":
        heliox_says(rule, clock, 0.0, after=CLOSE_SECONDS + 5.0)  # a slow model, then the text at once
    else:
        heliox_says(rule, clock, 21.0, after=1.0)  # stopped by the person 21 s in
    assert clock.t > CLOSE_SECONDS  # longer than the conversation's quiet allowed before
    clock.t += 3.0
    decision = said(rule, clock, "What time is it?")
    assert decision.act and decision.why == "follow-up: question"


def test_an_answer_spoken_sentence_by_sentence_keeps_the_conversation_open():
    clock = Clock()
    rule = Addressing(clock=clock)
    assert said(rule, clock, "Hey Heliox, tell me about the week.").act
    for _ in range(4):  # four sentences of 7 s each: 28 s of voice
        heliox_says(rule, clock, 7.0, after=0.2)
    clock.t += 3.0
    assert said(rule, clock, "What time is it?").act


def test_speech_nobody_asked_for_long_after_still_opens_nothing():
    clock = Clock()
    rule = Addressing(clock=clock)
    assert said(rule, clock, "Hey Heliox, what time is it?").act
    heliox_says(rule, clock, 2.0)
    clock.t += 100.0
    rule.heliox_finished()  # a reminder spoken later: the conversation closed long ago
    clock.t += 1.0
    decision = said(rule, clock, "What time is it?")
    assert not decision.act and decision.why == "no conversation open"
