"""A reminder's date and time come out of the sentence wherever the person puts them (pilot.workflows.reminders).

On 13 November "remind me on Thursday 19 November to send the report" (Live's answer to a weekday that did not
match its date) was kept as "19 november to send the report": the weekday was read, the date after it stayed in.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from benchmarks.conversation_eval.sessions import PACK, load_pack, run_session
from pilot.workflows.reminders import ReminderStore, parse_reminder

NOW = datetime(2026, 11, 12, 10, 0)  # a Thursday morning
REPORT = "send the report"


class Clock:
    def __init__(self, when: datetime) -> None:
        self.when = when.timestamp()

    def __call__(self) -> float:
        return self.when


@pytest.mark.parametrize(
    ("said", "due", "what"),
    [
        # before the thing
        ("remind me on Thursday 19 November at 9 am to send the report", datetime(2026, 11, 19, 9), REPORT),
        ("remind me on 19 November to send the report", datetime(2026, 11, 19, 9), REPORT),
        ("remind me on November 19th at 3 pm to send the report", datetime(2026, 11, 19, 15), REPORT),
        # after it
        ("remind me to send the report on Thursday 19 November at 9 am", datetime(2026, 11, 19, 9), REPORT),
        ("remind me to send the report on the 19th of November", datetime(2026, 11, 19, 9), REPORT),
        # both sides
        ("remind me on Thursday 19 November to send the report at 4 pm", datetime(2026, 11, 19, 16), REPORT),
        ("remind me at 4 pm to send the report on the 19th", datetime(2026, 11, 19, 16), REPORT),
        # a weekday
        ("remind me on friday to call mom", datetime(2026, 11, 13, 9), "call mom"),
        ("remind me to call mom on friday afternoon", datetime(2026, 11, 13, 14), "call mom"),
        ("remind me on thursday to stretch", datetime(2026, 11, 19, 9), "stretch"),  # today's 9 has passed
        # tomorrow
        ("remind me tomorrow morning to water the plants", datetime(2026, 11, 13, 9), "water the plants"),
        ("remind me to water the plants tomorrow at 7 pm", datetime(2026, 11, 13, 19), "water the plants"),
        ("remind me tonight to lock the door", datetime(2026, 11, 12, 20), "lock the door"),
        # in an hour
        ("remind me in an hour to stretch", datetime(2026, 11, 12, 11), "stretch"),
        ("remind me to stretch in half an hour", datetime(2026, 11, 12, 10, 30), "stretch"),
        # a date written two ways in one sentence
        ("remind me on Thursday the 19th, 19 November, to send the report", datetime(2026, 11, 19, 9), REPORT),
        ("remind me on the 19th of November, the 19th, at noon to send the report", datetime(2026, 11, 19, 12),
         REPORT),
        # as a recogniser writes it
        ("Remind me on Thursday the nineteenth of November at nine a.m. to send the report.",
         datetime(2026, 11, 19, 9), REPORT),
    ],
)  # fmt: skip
def test_the_date_and_time_come_out_of_the_sentence(said, due, what):
    parsed = parse_reminder(said, NOW)
    assert parsed is not None, said
    assert (datetime.fromtimestamp(parsed[0]), parsed[1]) == (due, what)


@pytest.mark.parametrize(
    "said",
    [
        "remind me about the thing",  # no time at all
        "remind me on 19 November to move it to 20 November",  # two different dates: not guessed
        "remind me on 31 November to party",  # no such day
        "remind me today at 9 to call",  # already past
        "remind me at 27 to sleep",
    ],
)
def test_what_is_not_guessed(said):
    assert parse_reminder(said, NOW) is None


def test_the_store_says_it_without_the_date_in_the_text(tmp_path):
    store = ReminderStore(tmp_path / "r.json", clock=Clock(NOW))
    assert store.answer("remind me on Thursday 19 November to send the report", now=NOW) == (
        "Okay, I'll remind you on Thu 19 Nov at 9:00 AM to send the report."
    )


@pytest.mark.asyncio
async def test_lives_weekday_answer_becomes_a_clean_reminder():
    """Through the real server: "Tuesday the nineteenth", Live asks, "Thursday.", and the reminder is set."""
    session = next(s for s in load_pack(PACK) if s["name"] == "slips said to Heliox, held and answered")
    record, failures = await run_session(session, "none")
    assert failures == []
    set_line = next(said.text for said in record.said if said.turn == 1)
    assert set_line.endswith("to send the report.") and "november to" not in set_line.casefold(), set_line
