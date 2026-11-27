"""The second live run's spoken bench, with a to-do "report due by 9" and no day: "how long until" is counted in code,
never left to the model; a sentence that gives the to-do a day or a duration the grounds do not hold goes whole (the
question about the day is asked in its place), never with words cut out of it; the grounds hold for the turns after
it; and the speech path never says a sentence with the marks of a hole (pilot.conversation.due, pilot.reply_rules)."""

from __future__ import annotations

import re
from datetime import datetime

import pytest

from pilot.conversation.context import ContextSources, ConversationMemory, build_messages
from pilot.conversation.due import WHICH_DAY, due_line, invented_day, invented_duration, time_left
from pilot.conversation.layer import ConversationLayer
from pilot.reply_rules import ear_faults, hole

FRIDAY_7 = datetime(2026, 11, 27, 7, 3)  # the run's clock: Friday morning
NO_DAY = {"text": "Finish the quarterly report by 9", "done": False}
TOMORROW_9 = {"text": "Finish the quarterly report by 9 am tomorrow", "done": False}
PLAN = "I'm going to watch a series tonight and do the report in the morning"

# The talk model's own replies in the run (k2h, calls 1 to 3), before any filter.
TURN_1 = "Your report is due by nine tomorrow morning, but the day isn't set, so which day is it due?"
TURN_2 = (
    "It's Friday morning and the report is due by nine, so tonight's series would leave you with about two hours. "
    "Want me to set a reminder for eight?"
)
TURN_3 = (
    "Your report is due by nine, and it's Friday now, so a morning session may be tight. Is the report due tomorrow "
    "morning?\n\nSTANCE: report timing vs. Friday morning"
)


# ── how long until, counted here ──


@pytest.mark.parametrize(
    ("now", "words"),
    [
        (datetime(2026, 11, 27, 7, 3), "about twenty-six hours from now"),  # morning
        (datetime(2026, 11, 27, 14, 0), "about nineteen hours from now"),  # afternoon
        (datetime(2026, 11, 27, 22, 30), "about ten hours from now"),  # late evening
        (datetime(2026, 11, 28, 8, 20), "about forty minutes from now"),
        (datetime(2026, 11, 28, 9, 30), "already past"),
        (datetime(2026, 11, 24, 9, 0), ""),  # more than two days ahead: not counted
    ],
)
def test_time_left_is_counted_in_code(now, words):
    assert time_left(datetime(2026, 11, 28, 9, 0), now) == words


def test_the_grounds_carry_the_count_when_the_day_is_known_and_no_count_when_it_is_not():
    known = due_line(PLAN, todos=[TOMORROW_9], reminders=[], now=FRIDAY_7, local=False)
    assert "due tomorrow at 9:00 in the morning (about twenty-six hours from now)" in known
    unknown = due_line(PLAN, todos=[NO_DAY], reminders=[], now=FRIDAY_7, local=False)
    assert "day not said, so how long is left is not known" in unknown
    assert "hours from now" not in unknown


# ── a day or a duration the grounds do not hold ──


@pytest.mark.parametrize(
    ("sentence", "invented"),
    [
        ("Your report is due by nine tomorrow morning, but the day isn't set,", True),  # turn 1
        ("Is the report due tomorrow morning?", True),  # turn 3's question
        ("It's due Friday at nine.", True),  # today's name put to the deadline
        ("Your report is due by nine, and it's Friday now,", False),  # today's name, about now
        ("It's Friday morning and the report is due by nine,", False),
        ("so tonight's series would leave you with little time.", False),  # "tonight" was the person's word
    ],
)
def test_a_day_given_to_the_to_do(sentence, invented):
    assert invented_day(sentence, PLAN, "Friday") is invented


def test_a_duration_the_grounds_do_not_hold():
    turn_2 = "so tonight's series would leave you with about two hours."
    assert invented_duration(turn_2, "", PLAN)  # the day is not known: no duration may be stated
    grounds = due_line(PLAN, todos=[TOMORROW_9], reminders=[], now=FRIDAY_7, local=False)
    assert not invented_duration("That leaves about twenty-six hours.", grounds, PLAN)
    assert invented_duration("That leaves about two hours.", grounds, PLAN)


# ── the three turns, through the layer, at the run's clock ──


class Replies:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.local, self.streams, self.name, self.provider = False, True, "fake", "fake"
        self.seen: list[list[dict[str, str]]] = []

    async def stream(self, system, messages):
        self.seen.append(messages)
        yield self.replies.pop(0)


def _layer(model, todos):
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: model, disagree=lambda: "when_it_matters", clock=lambda: 0.0)
    layer.sources = ContextSources(
        things_due=lambda text: due_line(text, todos=todos, reminders=[], now=FRIDAY_7, local=False),
        now=lambda: FRIDAY_7,
    )
    return layer, said


async def test_turn_one_asks_which_day_and_never_says_tomorrow():
    layer, said = _layer(Replies(TURN_1), [NO_DAY])
    await layer.take(PLAN)
    assert said == [WHICH_DAY]
    assert not any(hole(line) for line in said)


async def test_turn_two_keeps_the_grounds_and_states_no_duration():
    model = Replies("Your report is due by nine, but which day is it due?", TURN_2)
    layer, said = _layer(model, [NO_DAY])
    await layer.take(PLAN)
    said.clear()
    await layer.take("it'll be fine")
    assert "day not said" in model.seen[1][-1]["content"]  # "the day is not known" carried into turn two
    spoken = " ".join(said)
    assert "two hours" not in spoken and "set a reminder" not in spoken
    assert said[-1] == WHICH_DAY  # the sentence whose end was dropped closes on the question, not a hanging clause


async def test_turn_three_leaves_no_hole():
    layer, said = _layer(Replies(TURN_3), [NO_DAY])
    await layer.take(PLAN)
    spoken = " ".join(said)
    assert "it's Friday now" in spoken  # nothing cut out of a sentence
    assert "and it's now," not in spoken and "Is the report due?" not in spoken
    assert said[-1] == WHICH_DAY
    assert not any(hole(line) for line in said)


# ── the marks of a hole ──

HOLED = [
    "Your report is due by nine, and it's now, so a morning session may be tight.",  # the run's sentence
    "The report is due , so start early.",
    "Your report is due at nine  so that's tight.",
    "That leaves you with the.",
    "I'll put it into your.",
]
GOOD = [
    "Your report is due by nine, but the day isn't set, so which day is it due?",
    "Blue shirt it is.",
    "The task on screen is still working on the 404 check you asked for.",
    "I won't change its permissions without your say-so.",
    "The append is in progress.  Tell me which you still want.",
    "It's checking the file at ...work, and the result will show on screen.",
    "About four hundred and seventy thousand people live in Canberra.",
    "Okay, your call.",
    "It's Friday morning, so you have the whole day.",
]


@pytest.mark.parametrize("sentence", HOLED)
def test_a_hole_is_found(sentence):
    assert hole(sentence)
    assert any(fault.startswith("hole") for fault in ear_faults(sentence))


@pytest.mark.parametrize("sentence", GOOD)
def test_a_good_sentence_is_left_alone(sentence):
    assert hole(sentence) == ""


async def test_the_speech_path_never_says_a_holed_sentence():
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: Replies(HOLED[0]), clock=lambda: 0.0)
    await layer.take("How long do I have for the report?")
    assert not any(hole(line) for line in said)
    assert said and not re.search(r"it'?s now,", " ".join(said))


def test_the_grounds_go_with_the_next_turn_of_the_same_conversation_only():
    memory = ConversationMemory(clock=lambda: 0.0)
    line = due_line(PLAN, todos=[NO_DAY], reminders=[], now=FRIDAY_7, local=False)
    first = ContextSources(things_due=lambda _t: line, now=lambda: FRIDAY_7)
    build_messages(PLAN, first, memory, screen=None)
    memory.add_user(PLAN)
    memory.add_heliox("Which day is it due?")
    later = ContextSources(things_due=lambda _t: "", now=lambda: FRIDAY_7)
    assert "day not said" in str(build_messages("it'll be fine", later, memory, screen=None))
    memory.clear()  # a new conversation
    assert "day not said" not in str(build_messages("it'll be fine", later, memory, screen=None))


# ── from the real-model replay (live12voice_after, first pass) ──


def test_this_morning_put_to_the_deadline_is_a_day_even_when_the_person_said_in_the_morning():
    sentence = "Your report is due at nine this morning, so doing it in the morning is tight."
    assert invented_day(sentence, PLAN, "Friday")


async def test_a_later_sentence_with_an_invented_day_is_replaced_by_the_question():
    reply = "Sounds good. Your report is due at nine tomorrow morning, so you may want to set an alarm."
    layer, said = _layer(Replies(reply), [NO_DAY])
    await layer.take(PLAN)
    assert said == ["Sounds good.", WHICH_DAY]


async def test_no_second_day_question_when_the_reply_asks_it():
    reply = (
        "The report is due at nine, and I don't know which day, so if it's tomorrow morning that plan is tight. "
        "Which day is the report due?"
    )
    layer, said = _layer(Replies(reply), [NO_DAY])
    await layer.take(PLAN)
    spoken = " ".join(said)
    assert "tomorrow" not in spoken
    assert spoken.casefold().count("which day") == 1  # asked once, whichever words ask it
