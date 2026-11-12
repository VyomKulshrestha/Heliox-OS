"""When Heliox takes speech as meant for it: the wake phrase only in the vocative position, a narrow follow-up window
after Heliox speaks, no cascade, one "Sorry?" at most, and never during a call. Transcripts and an injected clock."""

from __future__ import annotations

import pytest

from pilot.system.addressing import Addressing, Heard, plausibly_addressed, vocative_wake


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("Hey Heliox, what can I say?", "what can I say"),
        ("Heliox, open my mail.", "open my mail"),
        ("Heliox open my mail", "open my mail"),
        ("Hey Heliox.", ""),
        ("Here Heliox, remind me that the launch call is Monday", "remind me that the launch call is Monday"),
        ("Okay Heliox, can you read that again?", "can you read that again"),
        ("Heliox, is it raining?", "is it raining"),
    ],
)
def test_the_wake_phrase_at_the_start_said_to_heliox(text, command):
    wake = vocative_wake(text)
    assert wake is not None and wake.command == command


@pytest.mark.parametrize(
    "text",
    [
        "Heliox is the app I use for this.",  # about Heliox
        "Heliox can do that later.",
        "Heliox, is a nice name for a project.",  # the bench's false hit: a pause, then talk about Heliox
        "Heliox's settings are in the corner.",
        "I asked Heliox yesterday.",  # not at the start
        "that Heliox thing is clever",
        "Helix",  # not the name
        "Hello, Alex.",
        "Hello there, everyone.",
        "Helium is light.",
        "Heliox.com, nice video.",  # a web address
        "Helios mission control update",  # a loose spelling without a second opinion
    ],
)
def test_the_name_said_about_heliox_or_something_like_it_does_not_wake_it(text):
    assert vocative_wake(text) is None


def test_a_misheard_name_counts_only_when_the_second_recogniser_heard_the_same_start():
    assert vocative_wake("Hey Heliocs, undo that", "Hey Lokes, undo that.") is not None
    assert vocative_wake("Hey Heliocs, undo that", "Hey Felix, open the door.") is None
    assert vocative_wake("Hey Heliocs, undo that") is None  # no second opinion: not enough for a misspelling


def heard(text, started, **kwargs):
    return Heard(text=text, started=started, ended=started + 1.5, voiced_seconds=kwargs.pop("voiced", 1.2), **kwargs)


def test_outside_a_conversation_only_the_wake_phrase_or_the_button_is_taken():
    clock = Clock()
    rule = Addressing(clock=clock)
    assert not rule.decide(heard("Open my mail", clock.now)).act  # looks like a command: not enough
    assert not rule.decide(heard("Can you set a timer for five minutes?", clock.now)).act
    decision = rule.decide(heard("Hey Heliox, open my mail", clock.now))
    assert decision.act and decision.command == "open my mail" and decision.why == "wake"
    assert rule.decide(heard("the deck please", clock.now, by_button=True)).act


def test_a_follow_up_soon_after_heliox_spoke_needs_no_wake_phrase():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, what's on my calendar?", clock.now))
    clock.now += 3
    rule.heliox_finished(asked=False)
    clock.now += 2
    decision = rule.decide(heard("And tomorrow?", clock.now))
    assert not decision.act  # third-person-ish fragment: not a direct request
    clock.now += 1
    decision = rule.decide(heard("What about tomorrow?", clock.now))
    assert decision.act and decision.why == "follow-up: question"


def test_after_the_window_the_wake_phrase_is_needed_again():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, what time is it?", clock.now))
    rule.heliox_finished()
    clock.now += 9  # past the 8 s window
    assert not rule.decide(heard("Set a timer for ten minutes", clock.now)).act


def test_a_reply_to_a_question_is_taken_even_when_it_is_only_a_name():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, open the report", clock.now))
    rule.heliox_finished(asked=True)  # "Which report?"
    clock.now += 2
    decision = rule.decide(heard("The quarterly one", clock.now))
    assert decision.act and decision.why == "follow-up: reply"


@pytest.mark.parametrize(
    "text",
    [
        "So then we went to the client and they said no.",  # narration
        "He thinks the budget is fine.",  # third person
        "I'll send Priya the deck by Friday.",
    ],
)
def test_talk_inside_the_window_is_not_taken(text):
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, what time is it?", clock.now))
    rule.heliox_finished()
    clock.now += 2
    assert not rule.decide(heard(text, clock.now)).act


def test_speech_that_began_while_heliox_was_talking_is_not_a_follow_up_unless_it_is_a_control_word():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, read my mail", clock.now))
    rule.heliox_finished()
    clock.now += 1
    # A sentence over Heliox (longer than a cut-off) is not taken; a control word is.
    assert not rule.decide(heard("Open the deck for everyone", clock.now, heliox_was_speaking=True, voiced=2.5)).act
    assert rule.decide(heard("Stop, that is enough now", clock.now, heliox_was_speaking=True, voiced=2.0)).act


def test_a_long_monologue_is_talk_not_a_reply():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, read my mail", clock.now))
    rule.heliox_finished()
    clock.now += 1
    assert not rule.decide(heard("Open the deck and walk everyone through it", clock.now, voiced=12.0)).act


def test_no_cascade_unclear_speech_gets_one_sorry_and_does_not_keep_the_conversation_open():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, what time is it?", clock.now))
    rule.heliox_finished()
    clock.now += 2
    first = rule.decide(heard("Your thing over there", clock.now))
    assert not first.act and first.say_sorry
    rule.heliox_finished()  # Heliox said "Sorry?"
    clock.now += 2
    second = rule.decide(heard("Your other thing", clock.now))
    assert not second.act and not second.say_sorry  # once only
    clock.now += 17  # 21 s since the last confidently addressed utterance
    assert not rule.conversation_open
    rule.heliox_finished()  # a reply now opens nothing
    clock.now += 1
    assert not rule.decide(heard("Open my mail", clock.now)).act


def test_only_a_confidently_addressed_follow_up_keeps_the_conversation_open():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, what time is it?", clock.now))
    for _ in range(3):  # a real exchange keeps going
        rule.heliox_finished()
        clock.now += 5
        assert rule.decide(heard("What about in Tokyo?", clock.now)).act
    rule.heliox_finished()
    clock.now += 5
    assert not rule.decide(heard("I think we should wrap up now.", clock.now)).act
    clock.now += 21
    assert not rule.conversation_open


def test_in_a_call_the_conversation_is_closed_and_nothing_is_taken():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, what time is it?", clock.now))
    rule.heliox_finished()
    clock.now += 1
    assert not rule.decide(heard("Hey Heliox, stop", clock.now, in_call=True)).act
    assert not rule.conversation_open


def test_a_running_task_keeps_the_conversation_open_for_control_words():
    clock = Clock()
    rule = Addressing(clock=clock, task_open=True)
    clock.now += 60
    assert rule.decide(heard("Skip the backup", clock.now)).act
    assert not rule.decide(heard("We had lunch at noon.", clock.now)).act


def test_heliox_says_once_why_it_did_not_answer_only_when_the_person_clearly_tried():
    clock = Clock()
    rule = Addressing(clock=clock)
    first = rule.decide(heard("Can you open my calendar?", clock.now))
    assert not first.act and "Hey Heliox" in first.hint
    assert rule.decide(heard("Can you open my mail?", clock.now)).hint == ""  # once
    assert Addressing(clock=clock).decide(heard("We met at noon.", clock.now)).hint == ""


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Stop", "control"),
        ("No, the other one", "control"),
        ("Undo that", "control"),
        ("Open my calendar", "request"),
        ("What time is it in Tokyo?", "question"),
        ("Can you read that again?", "question"),
        ("I'll send Priya the deck by Friday.", "talk"),
        ("He thinks the budget is fine.", "talk"),
    ],
)
def test_how_a_follow_up_reads(text, kind):
    assert plausibly_addressed(text, question_open=False) == kind


def test_a_word_or_two_over_heliox_is_a_stop_whatever_it_reads_as():
    clock = Clock()
    rule = Addressing(clock=clock)
    decision = rule.decide(heard("Great score.", clock.now, heliox_was_speaking=True, voiced=0.74))
    assert decision.act and decision.command == "Stop" and decision.why == "cut-off"
    assert not rule.decide(heard("Great score.", clock.now, heliox_was_speaking=False, voiced=0.74)).act
