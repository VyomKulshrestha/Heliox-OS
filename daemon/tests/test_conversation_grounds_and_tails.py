"""From the seventh spoken run: timers are never grounds; a to-do with no day gets none in the reply; a plain fact is
said without a hedge at its end; times are said as a person says them; the recogniser's "my2do" is a to-do list; a
change during an approval has one line, and a dropped change is said in a few words."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from pilot.conversation.action_talk import wont_change_sentence
from pilot.conversation.context import ContextSources, ConversationMemory, build_messages, clock_words
from pilot.conversation.due import due_line, without_invented_day
from pilot.conversation.handover import APPROVAL_SENTENCE, keep_company
from pilot.conversation.layer import ConversationLayer
from pilot.conversation.style import without_hedge_tail
from pilot.workflows.reminders import ReminderStore
from pilot.workflows.todos import TodoStore, spelled_out

NOW = datetime(2026, 11, 24, 17, 4)  # a Tuesday afternoon


def _reminder(text: str, minutes: int, **extra) -> dict:
    return {"id": text[:6], "text": text, "due": NOW.timestamp() + 60 * minutes, **extra}


# ── 1. what may be grounds ──


def test_timers_and_reminders_that_went_off_are_never_grounds():
    timer = _reminder("Your 10 minutes timer is done", 10, kind="timer")
    old_timer = _reminder("Your 10 minutes timer is done", 11)  # stored before timers were marked
    held = _reminder("call mum", -5, pending=True)
    said = _reminder("call the plumber", 20, said_at=NOW.timestamp())
    real = _reminder("call the dentist about the report", 30)
    line = due_line(
        "I'll do the report at six", todos=[], reminders=[timer, old_timer, held, said, real], now=NOW, local=True
    )
    assert "timer" not in line and "mum" not in line and "plumber" not in line
    assert "dentist" in line or line == ""  # health is never sent; the point is that timers are gone


def test_a_timer_set_by_voice_is_marked_as_one(tmp_path):
    store = ReminderStore(tmp_path / "reminders.json", clock=lambda: NOW.timestamp())
    assert store.answer("set a timer for 10 minutes", now=NOW) == "Timer set for 10 minutes."
    [item] = store.list()
    assert item["kind"] == "timer"
    assert due_line("what's on at five fifteen today", todos=[], reminders=[item], now=NOW, local=True) == ""


def test_done_to_dos_are_never_grounds():
    done = {"text": "finish the report by 9", "done": True}
    assert due_line("I'll do the report later", todos=[done], reminders=[], now=NOW, local=True) == ""


# ── 2. no day for a to-do that has none ──

NO_DAY = {"text": "finish the quarterly report by 9", "done": False}


@pytest.mark.parametrize(
    ("reply", "given", "said"),
    [
        (  # the spoken run, all six replies
            "Your report is due at nine tomorrow morning, so that's tight.",
            "I'm going to watch a series tonight and do the report in the morning.",
            "Your report is due at nine, so that's tight.",
        ),
        ("The report is due tomorrow at nine.", "I'll do the report after dinner.", "The report is due at nine."),
        ("It's due Friday at nine.", "I'll start later.", "It's due at nine."),
        (  # a day the person said stays
            "Watching tonight leaves only the morning before nine.",
            "I'm going to watch a series tonight.",
            "Watching tonight leaves only the morning before nine.",
        ),
    ],
)
def test_a_day_the_to_do_never_carried_is_not_said(reply, given, said):
    assert without_invented_day(reply, given) == said


def test_the_grounds_line_says_what_to_do_about_the_missing_day():
    line = due_line("I'll do the report later", todos=[NO_DAY], reminders=[], now=NOW, local=False)
    assert "by 9:00 in the morning, day not said (never give it a day; ask which day if it matters)" in line


class Replies:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.local, self.streams, self.name, self.provider = False, True, "fake", "fake"

    async def stream(self, system, messages):
        yield self.replies.pop(0)


async def _said(reply: str, words: str, sources: ContextSources | None = None) -> list[str]:
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: Replies(reply), disagree=lambda: "when_it_matters")
    if sources is not None:
        layer.sources = sources
    await layer.take(words)
    return said


async def test_the_layer_takes_the_invented_day_out_when_the_grounds_hold_a_dayless_to_do():
    line = due_line("I'll do the report in the morning", todos=[NO_DAY], reminders=[], now=NOW, local=False)
    sources = ContextSources(things_due=lambda _text: line, now=lambda: NOW)
    said = await _said(
        "Your report is due at nine tomorrow morning, so that's tight.",
        "I'm going to watch a series tonight and do the report in the morning.",
        sources,
    )
    assert said == ["Your report is due at nine, so that's tight."]


# ── 3. no hedge at the end of a plain fact ──


@pytest.mark.parametrize(
    ("reply", "said"),
    [  # the spoken run's own replies
        (
            "Canberra has roughly four hundred thousand people, a bit over that in recent years.",
            "Canberra has roughly four hundred thousand people.",
        ),
        (
            "Roughly four hundred and fifty thousand people live in Canberra, as of recent estimates.",
            "Roughly four hundred and fifty thousand people live in Canberra.",
        ),
        (
            "About four hundred and seventy thousand, though that's from memory and may be a bit out of date.",
            "About four hundred and seventy thousand.",
        ),
        ("About 330 metres, including its antennas.", "About 330 metres, including its antennas."),
    ],
)
def test_a_trailing_hedge_is_not_said(reply, said):
    assert without_hedge_tail(reply, "How many people live in Canberra?") == said


def test_the_hedge_stays_when_they_asked_how_current():
    reply = "About four hundred and sixty thousand, as of the latest census."
    assert without_hedge_tail(reply, "What's the latest population of Canberra?") == reply


async def test_the_layer_says_the_fact_without_its_tail():
    said = await _said(
        "Canberra has roughly four hundred thousand people, as of recent estimates.",
        "How many people live in Canberra?",
    )
    # The voice gets the sentence in pieces cut at commas, so the first piece is already said when the tail comes:
    # the tail is the part not said.
    assert said == ["Canberra has roughly four hundred thousand people,"]


# ── 4. times as a person says them ──


@pytest.mark.parametrize(
    ("hour", "minute", "words"),
    [
        (11, 59, "11:59 in the morning"),
        (12, 0, "noon"),
        (12, 1, "12:01 in the afternoon"),
        (17, 40, "5:40 in the afternoon"),
        (18, 0, "6:00 in the evening"),
        (23, 59, "11:59 at night"),
        (0, 0, "midnight"),
        (0, 1, "12:01 at night"),
        (5, 30, "5:30 in the morning"),
    ],
)
def test_a_time_said_the_way_a_person_says_it(hour, minute, words):
    now = datetime(2026, 11, 24, hour, minute)
    assert clock_words(now) == words
    sent = str(build_messages("hello", ContextSources(now=lambda: now), ConversationMemory(), screen=None))
    assert f"Now: Tuesday 24 November 2026, {words}." in sent
    assert "17:40" not in sent and "seventeen" not in sent


# ── 5. how a recogniser writes "to-do" ──


@pytest.mark.parametrize(
    ("heard", "written"),
    [
        (
            "Add finish the quarterly report by 9 to my2do list.",
            "Add finish the quarterly report by 9 to my to-do list.",
        ),
        ("what's on my 2do list", "what's on my to-do list"),
        ("add milk to the 2-do list", "add milk to the to-do list"),
    ],
)
def test_the_recognisers_spelling_is_written_out(heard, written):
    assert spelled_out(heard) == written


def test_the_spoken_run_s_my2do_is_added_to_the_list(tmp_path):
    store = TodoStore(tmp_path / "todos.json")
    assert store.answer("Add finish the quarterly report by 9 to my2do list.").startswith("Added")
    assert store.open_items()[0]["text"] == "finish the quarterly report by 9"
    assert store.answer("what's on my2do list").startswith("You have 1 thing")


# ── 6. lines around a changed request ──


def test_a_dropped_change_is_said_in_a_few_words():
    assert wont_change_sentence("make it the all draft instead") == "Okay, I won't change it to the all draft."
    assert (
        wont_change_sentence("Actually, make it the all draft instead") == "Okay, I won't change it to the all draft."
    )
    assert wont_change_sentence("instead") == "Okay, I won't change it."


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def test_a_change_during_an_approval_gets_no_second_line_from_the_company():
    clock = Clock()
    said: list[str] = []

    def status():
        clock.t += 0.5
        if clock.t < 3:
            return {"phase": "awaiting_approval", "message": "Waiting"}
        return {"phase": "correcting", "message": "Applying your spoken correction"}

    async def speak(text, channel, key):
        said.append(text)
        return "spoken"

    task = asyncio.ensure_future(keep_company(acknowledgement_text="", speak=speak, status=status, clock=clock, poll=0))
    while clock.t < 30:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert said == [APPROVAL_SENTENCE]  # the server's own "I've closed that one and I'm changing it." is the one line
