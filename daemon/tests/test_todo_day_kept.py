"""A to-do keeps the day said with it ("by nine on Friday morning"), a reminder too, and the grounds line never states
a day a to-do does not carry (from the sixth spoken run)."""

from __future__ import annotations

from datetime import datetime

import pytest

from pilot.conversation.due import due_line
from pilot.workflows.reminders import parse_reminder
from pilot.workflows.todos import TodoStore

NOW = datetime(2026, 11, 24, 14, 0)  # a Tuesday


@pytest.mark.parametrize(
    ("said", "kept"),
    [
        (
            "add finish the quarterly report by nine on Friday morning to my to-do list",
            "finish the quarterly report by nine on Friday morning",
        ),
        ("add finish the report by 9 on Friday morning to my to-do list.", "finish the report by 9 on Friday morning"),
        ("add talk to Sam about the launch to my list", "talk to Sam about the launch"),
        ("put renew the passport next week on my to do list", "renew the passport next week"),
        ("add send the deck on 4 December to the list", "send the deck on 4 December"),
    ],
)
def test_a_to_do_keeps_all_of_what_was_said_about_when(tmp_path, said, kept):
    store = TodoStore(tmp_path / "todos.json")
    assert store.answer(said).startswith("Added")
    assert store.open_items()[0]["text"] == kept


def test_a_reminder_keeps_its_day_too():
    due, what = parse_reminder("remind me on Friday morning at 9 to finish the report", NOW)
    assert datetime.fromtimestamp(due) == datetime(2026, 11, 27, 9, 0) and what == "finish the report"


def test_the_grounds_never_state_a_day_the_to_do_does_not_carry():
    decision = "I'm going to watch a series tonight and do the report in the morning"
    no_day = {"text": "finish the quarterly report by 9", "done": False}
    line = due_line(decision, todos=[no_day], reminders=[], now=NOW, local=False)
    assert "by 09:00 in the morning, day not said" in line and "tomorrow" not in line and "today" not in line
    friday = {"text": "finish the quarterly report by nine on Friday morning", "done": False}
    assert "due Friday 27 November at 09:00 in the morning" in due_line(
        decision, todos=[friday], reminders=[], now=NOW, local=False
    )
