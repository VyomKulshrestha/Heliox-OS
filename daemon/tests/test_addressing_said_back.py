"""What the person says right after Heliox answered them is taken as said to Heliox, unless something says it was
not (pilot.system.addressing: _said_back, said_elsewhere).

The companion acceptance run (24 November, spoken, real listener and model) found a spoken debate impossible without
"Hey Heliox" before every sentence: "And roughly how many people live there?" said 1.3 to 2.9 s after Heliox's
answer was overheard on 3 of 3 runs, and so were "It'll be fine." and the person's decision itself. Those lines, as
the recogniser wrote them, are taken here; speech said to someone else, read aloud, played by a video, said in a
call, or said later than TALK_TAKEN_SECONDS still is not. Transcripts and a clock the test controls."""

from __future__ import annotations

import pytest

from pilot.system.addressing import TALK_TAKEN_SECONDS, Addressing, Heard, hand_about_me, said_elsewhere


class Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


def answered(*, asked: bool = False) -> tuple[Addressing, Clock]:
    """The person said "Hey Heliox, ..." and Heliox has just finished its answer."""
    clock = Clock()
    rule = Addressing(clock=clock)
    hand_about_me(rule, None)
    assert rule.decide(Heard("Hey Heliox, what's the capital of Australia?", clock.now, clock.now + 3.0, 2.4)).act
    clock.now += 5.0
    rule.heliox_finished(asked=asked)  # "Canberra."
    return rule, clock


def say(rule: Addressing, clock: Clock, text: str, after: float, *, voiced: float | None = None, **kwargs):
    clock.now += after
    seconds = voiced if voiced is not None else max(0.5, 0.33 * len(text.split()))
    return rule.decide(Heard(text, clock.now, clock.now + seconds, seconds, **kwargs))


# The acceptance run's lines without the wake phrase, as Whisper wrote them, and when they were said after Heliox's
# last word (s1r, second and shake2 runs: 1.3 to 2.9 s; s3free: 0.5 s; the rest "reply" at 1.0 s).
RUN_LINES = [
    "And roughly how many people live there.",
    "How many people live there?",
    "I'm going to watch a series tonight and do the report in the morning.",
    "It'll be fine.",
    "Seriously, I'll manage.",
    "Actually, my manager moved the deadline to Friday, so I have time.",
    "I know. I'll do it anyway.",
    "Anyway, what should I cook for dinner?",
    "I think I'll wear the blue shirt tomorrow.",
    "I'm going to call my mum this evening.",
    "Oh, I meant Monday the thirtieth.",
    "What were you saying before I cut you off?",
]


@pytest.mark.parametrize("text", RUN_LINES)
@pytest.mark.parametrize("after", [0.5, 1.3, 2.9])
def test_the_run_s_replies_without_the_wake_phrase_are_taken(text, after):
    rule, clock = answered()
    decision = say(rule, clock, text, after)
    assert decision.act, decision.why
    assert decision.command == text.strip(" .?!")


def test_a_debate_goes_on_without_the_wake_phrase():
    rule, clock = answered()
    for line in ["I'm going to watch a series tonight and do the report in the morning.", "It'll be fine.",
                 "Seriously, I'll manage."]:  # fmt: skip
        assert say(rule, clock, line, 1.0).act, line
        clock.now += 4.0
        rule.heliox_finished()  # Heliox pushed back, or gave way
    assert rule.conversation_open


def test_later_than_the_short_while_only_a_question_or_request_is_taken():
    rule, clock = answered()
    assert not say(rule, clock, "It'll be fine.", TALK_TAKEN_SECONDS + 1.0).act
    rule2, clock2 = answered()
    assert say(rule2, clock2, "How many people live there?", TALK_TAKEN_SECONDS + 1.0).act  # as before: a question


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("Hey Felix, are you coming to lunch?", "said to someone by name"),
        ("Hello, Alex, good to see you.", "said to someone by name"),
        ("Priya, could you stop sharing your screen for a second?", "said to someone by name"),
        ("Can you take notes today, Ken?", "said to someone by name"),
        ("Can you open the deck so everyone can see it?", "said to the room"),
        ("Does anyone mind if we start without Omar?", "said to the room"),
        ("Can somebody set a timer for five minutes?", "said to the room"),
        ("Open Excel and show us the error you got.", "said to the room"),
        ("Sorry, you were on mute. Could you say that again?", "a call or a video"),
        ("Please like and subscribe if this helped you.", "a call or a video"),
        ("I've been using Heliox for a month and it's fine.", "talk about Heliox"),
        ("My brother asked whether Heliox works on a Mac.", "talk about Heliox"),
    ],
)
def test_speech_said_to_someone_else_is_not_taken(text, why):
    assert said_elsewhere(text) == why
    rule, clock = answered()
    decision = say(rule, clock, text, 1.0)
    assert not decision.act and decision.why == f"follow-up: {why}"


@pytest.mark.parametrize(
    "text",
    [
        "It'll be fine.",
        "Seriously, I'll manage.",
        "Anyway, what should I cook for dinner?",
        "Actually, my manager moved the deadline to Friday, so I have time.",
        "I'll send Priya the deck by Friday.",  # a name in the sentence is not a name said to
        "Is that right, Heliox?",  # the name at the end is Heliox's
    ],
)
def test_nothing_in_a_reply_to_heliox_reads_as_said_elsewhere(text):
    assert said_elsewhere(text) == ""


@pytest.mark.parametrize(
    ("text", "voiced"),
    [
        ("PowerShell.", 0.8),  # one word: not a reply
        ("Thanks for watching.", 1.0),  # what Whisper writes for noise
        ("Open, open, open, open.", 1.5),
        ("It is fine.", 0.1),  # too little voice for three words: noise read as words
        ("First, open Chrome from the start menu.", 2.0),  # steps read out to viewers
        ("The rain had stopped by the time the train reached the coast, and the platform was empty.", 5.0),
        ("So I told him the meeting was moved to Thursday and then the whole thing fell apart", 9.0),  # monologue
    ],
)
def test_noise_steps_a_passage_and_a_monologue_are_not_replies(text, voiced):
    rule, clock = answered()
    assert not say(rule, clock, text, 1.0, voiced=voiced).act


def test_only_after_heliox_answered_the_person():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.open_since = clock.now - 3.0  # a conversation, but what Heliox just said answered nobody (a reminder)
    rule.heliox_finished()
    assert not say(rule, clock, "It'll be fine.", 1.0).act
    after_sorry, clock2 = answered()
    first = say(after_sorry, clock2, "Your thing over there", TALK_TAKEN_SECONDS + 1.0)  # later: unclear
    assert not first.act and first.say_sorry
    clock2.now += 0.7
    after_sorry.heliox_finished()  # "Sorry?"
    assert not say(after_sorry, clock2, "It'll be fine.", 1.0).act  # a "Sorry?" answers nobody: nothing said back


def test_not_when_another_voice_was_heard_since_heliox_spoke():
    rule, clock = answered()
    assert not say(rule, clock, "Hey Felix, are you coming to lunch?", 0.5).act
    assert rule.overheard_since_heliox == 1
    assert not say(rule, clock, "It'll be fine.", 0.5).act  # the room is not just the person now
    assert say(rule, clock, "What about Sydney?", 0.5).act  # a question is a question, as before


def test_not_over_heliox_and_not_in_a_call():
    rule, clock = answered()
    assert not say(rule, clock, "It'll be fine.", 1.0, voiced=2.0, heliox_was_speaking=True).act
    call, clock2 = answered()
    assert not say(call, clock2, "It'll be fine.", 1.0, in_call=True).act
    assert not call.conversation_open


def test_stop_and_no_said_to_anyone_still_stop():
    rule, clock = answered()
    decision = say(rule, clock, "No, Sam, stop.", 1.0)
    assert decision.act and decision.why == "follow-up: control"


def test_nothing_of_what_was_said_elsewhere_is_kept():
    rule, clock = answered()
    for text in ["Priya, could you stop sharing your screen for a second?", "Can somebody set a timer?"]:
        assert not say(rule, clock, text, 0.5).act
    kept = repr(vars(rule)) + repr(rule.history)
    assert "Priya" not in kept and "timer" not in kept
    assert rule.overheard_since_heliox == 2
