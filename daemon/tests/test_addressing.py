"""When Heliox takes speech as meant for it: the wake phrase only in the vocative position, a narrow follow-up window
after Heliox speaks, no cascade, one "Sorry?" at most, and never during a call. Transcripts and an injected clock."""

from __future__ import annotations

import pytest

from pilot.system.addressing import CONFIDENT, Addressing, Heard, plausibly_addressed, vocative_wake


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
    assert decision.act and decision.why == "follow-up: elliptical"  # right after Heliox's answer
    rule.heliox_finished(asked=False)
    clock.now += 6
    decision = rule.decide(heard("What about tomorrow?", clock.now))
    assert decision.act and decision.why == "follow-up: question"  # later: read as a question


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


# ── other languages: the follow-up words are read in English; a quick short reply to a question needs no words ──


@pytest.mark.parametrize(
    "answer",
    ["Ja, die blaue Datei.", "Sí, la segunda.", "हाँ, नीली वाली फ़ाइल।", "はい、青いほうです。"],
)
def test_a_quick_short_answer_to_heliox_s_question_is_a_reply_in_any_language(answer):
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(Heard("Hey Heliox, open the file.", clock(), clock(), 1.5))
    clock.now += 2.0
    rule.heliox_finished(asked=True)  # Heliox: "Which file, the blue one or the red one?"
    clock.now += 1.2
    decision = rule.decide(Heard(answer, clock(), clock() + 1.4, 1.4))
    assert decision.act and decision.why == "follow-up: reply"


@pytest.mark.parametrize("answer", ["हाँ, नीली वाली फ़ाइल।", "はい、青いほうです。"])
def test_the_word_free_reply_needs_heliox_s_question_a_quick_start_a_short_answer_and_a_clear_floor(answer):
    def asked(after: float, voiced: float = 1.4, *, question: bool = True, over: bool = False):
        clock = Clock()
        rule = Addressing(clock=clock)
        rule.decide(Heard("Hey Heliox, open the file.", clock(), clock(), 1.5))
        clock.now += 2.0
        rule.heliox_finished(asked=question)
        clock.now += after
        return rule.decide(Heard(answer, clock(), clock() + voiced, voiced, heliox_was_speaking=over))

    assert asked(1.5).act
    assert not asked(3.0).act  # not quick: the words would have to say it, and these are not read
    assert not asked(1.0, voiced=6.0).act  # not short
    assert not asked(1.0, question=False).act  # Heliox asked nothing
    assert not asked(0.5, voiced=2.5, over=True).act  # began over Heliox's voice (longer than a cut-off)


@pytest.mark.parametrize("noise", ["Thanks for watching.", "Thank you for watching!", "Open, open, open, open."])
def test_a_line_whisper_invents_from_noise_is_never_a_quick_reply(noise):
    from pilot.system.voice import looks_hallucinated

    assert looks_hallucinated(noise)
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(Heard("Hey Heliox, open the file.", clock(), clock(), 1.5))
    clock.now += 2.0
    rule.heliox_finished(asked=True)  # a question is open: the quick-reply window
    clock.now += 0.5
    assert not rule.decide(Heard(noise, clock(), clock() + 1.0, 1.0)).act


# ── elliptical follow-ups right after Heliox's answer ──


def answered(after: float, text: str, *, over: bool = False, in_call: bool = False, voiced: float = 1.2):
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, what's the weather today?", clock.now))
    clock.now += 3
    rule.heliox_finished(asked=False)  # Heliox answered; no question open
    clock.now += after
    return rule.decide(Heard(text, clock.now, clock.now + voiced, voiced, heliox_was_speaking=over, in_call=in_call))


@pytest.mark.parametrize(
    "text",
    ["And tomorrow?", "What about the second one?", "Shorter.", "In short.", "Once more.", "Again.", "Say that again.",
     "How about Friday?", "Or the red one?"],
)  # fmt: skip
def test_a_short_elliptical_follow_up_right_after_heliox_s_answer_is_taken(text):
    decision = answered(1.0, text)
    assert decision.act and decision.why == "follow-up: elliptical"


def test_an_elliptical_follow_up_is_not_taken_late_over_heliox_or_in_a_call():
    assert not answered(6.0, "And tomorrow?").act  # later than 4 s: read as words, and "And ..." is narration
    assert not answered(0.5, "And tomorrow?", over=True, voiced=2.0).act  # began over Heliox's voice
    assert not answered(1.0, "And tomorrow?", in_call=True).act
    assert not answered(1.0, "And then we went home after the movie.").act  # too long: narration
    assert answered(6.0, "Say that again.").act  # later it still reads as a request ("Say ...")


@pytest.mark.parametrize(
    "text",
    ["Say that again.", "What did you say?", "Pardon?", "Repeat that, please.", "One more time.", "Do that again.",
     "Run it again.", "Repeat my last task.", "Make it shorter.", "The gist.", "TL;DR", "Just the short version."],
)  # fmt: skip
def test_the_repeat_and_shorter_phrases_the_conversation_layer_answers_are_elliptical(text):
    # The phrases pilot/conversation/followups.py answers in code (its AGAIN-like and SHORTER lists).
    from pilot.system.addressing import elliptical_kind

    assert elliptical_kind(text) in {"again", "shorter"}


def test_a_softened_imperative_is_a_request():
    from pilot.system.addressing import plausibly_addressed

    assert plausibly_addressed("Just delete all of that.", question_open=False) == "request"
    assert plausibly_addressed("Okay so just open the deck.", question_open=False) == "request"
    assert plausibly_addressed("So we went to lunch.", question_open=False) == "talk"  # narration stays talk


@pytest.mark.parametrize(
    ("said", "invites"),
    [('Say "more" for others.', True), ("Which file?", True), ("Just say yes or no.", True),
     ("You can say “stop” at any time.", True), ("Opened Notepad.", False), ("I can't say more about that.", False),
     ("I'll say it once more.", False)],
)  # fmt: skip
def test_heliox_s_own_sentence_invites_a_reply_when_it_asks_or_names_a_word_to_say(said, invites):
    from pilot.system.addressing import invites_reply

    assert invites_reply(said) is invites


def test_the_word_heliox_invited_is_taken_in_the_window():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.decide(heard("Hey Heliox, what can I say?", clock.now))
    clock.now += 4
    rule.heliox_finished(asked=True)  # 'You can ask me for the time, ... Say "more" for others.'
    clock.now += 1.5
    decision = rule.decide(heard("More.", clock.now))
    assert decision.act and decision.why == "follow-up: reply"


# ── answering Live: its own replies for a few seconds after it spoke, with no conversation open ──


def live_rule():
    clock = Clock()
    rule = Addressing(clock=clock)
    rule.live_reply = lambda text: text.strip(" .?!").casefold() in {"why", "not now", "thanks", "stop"}
    rule.live_spoke()  # Live finished its remark at clock.now
    return rule, clock


def test_a_reply_to_live_is_judged_when_it_was_said_not_when_it_was_heard():
    rule, clock = live_rule()
    said_at = clock.now + 2.0
    clock.now += 9.0  # a recognition backlog: the decision comes 9 s after Live spoke
    decision = rule.decide(Heard("Why?", said_at, said_at + 0.6, 0.6))
    assert decision.act and decision.why == "live reply"
    assert not rule.conversation_open  # nothing opens


def test_one_reply_per_remark_and_nothing_else_or_later():
    rule, clock = live_rule()
    assert not rule.decide(Heard("Open the door.", clock.now + 1.0, clock.now + 2.0, 1.0)).act
    assert rule.decide(Heard("Not now.", clock.now + 2.0, clock.now + 2.5, 0.5)).act
    assert not rule.decide(Heard("Thanks.", clock.now + 3.0, clock.now + 3.5, 0.5)).act  # the window is used
    late, late_clock = live_rule()
    assert not late.decide(Heard("Why?", late_clock.now + 7.0, late_clock.now + 7.5, 0.5)).act


def test_no_reply_to_live_in_a_call_or_after_the_computer_woke():
    rule, clock = live_rule()
    assert not rule.decide(Heard("Why?", clock.now + 1.0, clock.now + 1.5, 0.5, in_call=True)).act
    woke, woke_clock = live_rule()
    woke.close("cleared")  # the server's wake-from-sleep path clears every window
    assert not woke.decide(Heard("Why?", woke_clock.now + 1.0, woke_clock.now + 1.5, 0.5)).act


def test_over_live_s_voice_a_short_cut_off_is_still_stop_and_a_sentence_is_not_a_reply():
    rule, clock = live_rule()
    cut = rule.decide(Heard("Why?", clock.now - 0.5, clock.now, 0.6, heliox_was_speaking=True))
    assert cut.act and cut.command == "Stop" and cut.why == "cut-off"
    rule2, clock2 = live_rule()
    long = rule2.decide(Heard("Why?", clock2.now - 1.0, clock2.now + 1.5, 2.5, heliox_was_speaking=True))
    assert not long.act


@pytest.mark.parametrize(
    "text",
    [
        "Forget that I'm a founder.",
        "Forget that.",
        "Please forget what I said about the budget.",
        "Forget my address.",
        "Don't keep that.",
        "Remember that.",
    ],  # fmt: skip
)
def test_a_command_about_what_heliox_keeps_is_a_follow_up(text):
    assert plausibly_addressed(text, question_open=False) in CONFIDENT


@pytest.mark.parametrize(
    "text",
    [
        "Forget it.",
        "Forget about it, we can do it tomorrow.",
        "Remember that the meeting moved to Friday.",
        "Remember to save your work before you close the editor.",
    ],  # fmt: skip
)
def test_the_same_verbs_said_to_someone_else_are_not(text):
    assert plausibly_addressed(text, question_open=False) not in CONFIDENT
