"""Reminders, change diffs and "where did you put X": answered in code from local records."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionResult, ActionType, FileParams
from pilot.config import PilotConfig, UndoConfig
from pilot.workflows.reminders import ReminderStore, describe_time, parse_reminder
from pilot.workflows.undo_journal import UndoJournal

NOW = datetime(2026, 10, 23, 15, 20)


@pytest.mark.parametrize(
    ("text", "due", "what"),
    [
        ("remind me in 20 minutes to stretch", datetime(2026, 10, 23, 15, 40), "stretch"),
        ("Remind me to call mom in 2 hours", datetime(2026, 10, 23, 17, 20), "call mom"),
        ("remind me at 5 pm to leave", datetime(2026, 10, 23, 17, 0), "leave"),
        ("remind me at 5 to leave", datetime(2026, 10, 23, 17, 0), "leave"),
        ("remind me tomorrow at 9:30 am to send the report", datetime(2026, 10, 24, 9, 30), "send the report"),
        ("remind me at 9 am to water the plants", datetime(2026, 10, 24, 9, 0), "water the plants"),
        ("remind me tomorrow to pay rent", datetime(2026, 10, 24, 9, 0), "pay rent"),
    ],
)
def test_parse_reminder(text, due, what):
    assert parse_reminder(text, NOW) == (due.timestamp(), what)


def test_unclear_reminders_are_not_guessed():
    assert parse_reminder("remind me about the thing", NOW) is None
    assert parse_reminder("remind me at 27 to sleep", NOW) is None
    assert parse_reminder("open the report", NOW) is None


def test_describe_time():
    assert describe_time(datetime(2026, 10, 23, 15, 40).timestamp(), NOW) == "at 3:40 PM"
    assert describe_time(datetime(2026, 10, 24, 9, 0).timestamp(), NOW) == "tomorrow at 9:00 AM"


class Clock:
    def __init__(self, when: datetime) -> None:
        self.when = when.timestamp()

    def __call__(self) -> float:
        return self.when


def test_set_list_and_cancel_by_sentence(tmp_path):
    store = ReminderStore(tmp_path / "r.json", clock=Clock(NOW))

    assert store.answer("remind me in 20 minutes to stretch", now=NOW) == "Okay, I'll remind you at 3:40 PM to stretch."
    store.answer("remind me tomorrow to pay rent", now=NOW)
    assert store.answer("what are my reminders", now=NOW) == (
        "You have 2 reminders. at 3:40 PM: stretch; tomorrow at 9:00 AM: pay rent."
    )
    assert store.answer("cancel the reminder about rent", now=NOW) == "Cancelled 1 reminder."
    assert store.answer("cancel the reminder about rent", now=NOW) == "There was no reminder like that to cancel."
    assert store.answer("open notepad", now=NOW) is None
    assert "Tell me when" in store.answer("remind me about the thing", now=NOW)


@pytest.mark.asyncio
async def test_due_reminders_are_delivered_once_and_late_ones_marked(tmp_path):
    clock = Clock(NOW)
    delivered = []

    async def deliver(item):
        delivered.append((item["text"], item["late"]))

    store = ReminderStore(tmp_path / "r.json", deliver=deliver, clock=clock)
    store.add(NOW.timestamp() + 60, "stretch")
    store.add(NOW.timestamp() + 3600, "call mom")

    assert await store.tick() == 0
    clock.when += 61
    assert await store.tick() == 1
    assert await store.tick() == 0
    clock.when += 2 * 3600  # Heliox was not running when "call mom" fell due
    assert await store.tick() == 1
    assert delivered == [("stretch", False), ("call mom", True)]
    assert store.list() == []


@pytest.mark.asyncio
async def test_missed_reminders_are_never_dropped_and_go_out_once_together(tmp_path):
    clock = Clock(NOW)
    deliver = AsyncMock()
    missed: list = []

    async def deliver_missed(items, first_tick):
        missed.append(([item["text"] for item in items], first_tick, [item["late"] for item in items]))

    store = ReminderStore(tmp_path / "r.json", deliver=deliver, deliver_missed=deliver_missed, clock=clock)
    store.add(NOW.timestamp() + 60, "call mum")
    store.add(NOW.timestamp() + 3600, "stretch")
    store.add(NOW.timestamp() + 30 * 86400, "renew the passport")
    clock.when += 3 * 86400  # the laptop was asleep for three days

    assert await store.tick() == 2
    assert missed == [(["call mum", "stretch"], True, [True, True])]  # the first check since Heliox started
    deliver.assert_not_awaited()  # none goes out on its own as well
    assert await store.tick() == 0  # never twice
    assert [item["text"] for item in store.list()] == ["renew the passport"]


@pytest.mark.asyncio
async def test_after_the_first_check_a_missed_reminder_is_said_to_come_from_sleep(tmp_path):
    clock = Clock(NOW)
    missed: list = []

    async def deliver_missed(items, first_tick):
        missed.append(first_tick)

    store = ReminderStore(tmp_path / "r.json", deliver=AsyncMock(), deliver_missed=deliver_missed, clock=clock)
    await store.tick()  # Heliox started; nothing due
    store.add(NOW.timestamp() + 60, "stand up")
    clock.when += 4 * 3600  # asleep
    await store.tick()
    assert missed == [False]


@pytest.mark.asyncio
async def test_a_repeating_reminder_missed_while_away_goes_out_once_and_keeps_its_cadence(tmp_path):
    clock = Clock(NOW)
    missed: list = []

    async def deliver_missed(items, first_tick):
        missed.extend(item["text"] for item in items)

    store = ReminderStore(tmp_path / "r.json", deliver=AsyncMock(), deliver_missed=deliver_missed, clock=clock)
    store.add(NOW.timestamp() + 60, "drink water", repeat={"minutes": 30})
    clock.when += 5 * 3600

    await store.tick()
    await store.tick()
    assert missed == ["drink water"]
    assert store.list()[0]["due"] > clock()


def test_the_missed_sentence_names_at_most_three():
    from pilot.workflows.reminders import describe_missed

    def items(*texts):
        return [{"text": text, "due": index} for index, text in enumerate(texts)]

    assert describe_missed(items("call mum"), False, now=0)[1] == (
        "While this computer was asleep, a reminder came due: call mum."
    )
    assert describe_missed(items("call mum", "stretch"), True, now=0)[1] == (
        "While Heliox was off, two reminders came due: call mum and stretch."
    )
    assert describe_missed(items("a", "b", "c", "d", "e"), False, now=0)[1] == (
        "While this computer was asleep, five reminders came due: a, b, c and two more."
    )


def _write(path, content):
    return Action(
        action_type=ActionType.FILE_WRITE, target=str(path), parameters=FileParams(path=str(path), content=content)
    )


async def _run(journal, action, effect, task="t1"):
    token = await journal.before(action, task_key=task)
    effect()
    await journal.after(token, ActionResult(action=action, success=True, output="ok"))


@pytest.mark.asyncio
async def test_diff_shows_what_changed_in_each_written_file(tmp_path):
    journal = UndoJournal(tmp_path / "undo", config=UndoConfig())
    notes = tmp_path / "notes.txt"
    notes.write_text("milk\neggs\n", encoding="utf-8")
    new = tmp_path / "todo.md"

    await _run(journal, _write(notes, "milk\nbread\n"), lambda: notes.write_text("milk\nbread\n", encoding="utf-8"))
    await _run(journal, _write(new, "call mom\n"), lambda: new.write_text("call mom\n", encoding="utf-8"))

    changes = await journal.diff("t1")

    assert [Path(item["path"]).name for item in changes] == ["todo.md", "notes.txt"]
    assert changes[0]["created"] is True and "+call mom" in changes[0]["diff"]
    assert "-eggs" in changes[1]["diff"] and "+bread" in changes[1]["diff"]


@pytest.mark.asyncio
async def test_binary_files_are_not_diffed(tmp_path):
    journal = UndoJournal(tmp_path / "undo", config=UndoConfig())
    image = tmp_path / "logo.bin"
    image.write_bytes(b"\xff\xfe\x00")

    await _run(journal, _write(image, "x"), lambda: image.write_bytes(b"\xff\x00\x01"))

    assert (await journal.diff("t1"))[0]["diff"] is None


@pytest.mark.asyncio
async def test_where_did_you_put_answers_from_moves(tmp_path):
    journal = UndoJournal(tmp_path / "undo", config=UndoConfig())
    report = tmp_path / "Q3 report.pdf"
    report.write_bytes(b"%PDF")
    archive = tmp_path / "Archive"
    archive.mkdir()
    move = Action(
        action_type=ActionType.FILE_MOVE,
        target=str(report),
        parameters=FileParams(path=str(report), destination=str(archive)),
    )

    await _run(journal, move, lambda: shutil.move(str(report), str(archive)))

    answer = await journal.answer_async("Where did you put my Q3 report?")
    assert answer.startswith(f"I moved Q3 report.pdf to {archive} at ")
    assert await journal.answer_async("where did you put the cat photo") == (
        "I have no record of moving, copying or saving cat photo."
    )
    assert await journal.answer_async("open the report") is None


@pytest.mark.asyncio
async def test_server_answers_reminders_and_where_questions_in_code(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._reminders = ReminderStore(tmp_path / "r.json")
    server._undo_journal = UndoJournal(tmp_path / "undo", config=UndoConfig())

    reply = await server._control_reply("remind me in 10 minutes to check the oven")
    assert reply.startswith("Okay, I'll remind you ") and reply.endswith("to check the oven.")
    assert (await server._control_reply("where did you put my essay")).startswith("I have no record")


@pytest.mark.asyncio
async def test_reminder_and_diff_rpcs(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._reminders = ReminderStore(tmp_path / "r.json")
    server._undo_journal = UndoJournal(tmp_path / "undo", config=UndoConfig())
    server._broadcast_notification = AsyncMock()

    added = await server._handle_reminder_add({"text": "stretch", "in_minutes": 5}, MagicMock())
    assert added["status"] == "ok"
    listed = await server._handle_reminder_list({}, MagicMock())
    assert [item["text"] for item in listed["reminders"]] == ["stretch"]
    assert (await server._handle_reminder_add({"text": "x", "in_minutes": 0}, MagicMock()))["status"] == "error"
    cancelled = await server._handle_reminder_cancel({"id": added["reminder"]["id"]}, MagicMock())
    assert cancelled["status"] == "ok" and cancelled["reminders"] == []
    assert (await server._handle_undo_diff({"task_key": "none"}, MagicMock()))["changes"] == []
