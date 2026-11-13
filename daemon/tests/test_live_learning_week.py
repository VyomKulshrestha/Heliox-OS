"""What "not now" and "don't mention these" do to the following days (pilot.live.learning_week): three made-up
people, the same four slips each day, through the real LiveCompanion kept across four days. The numbers are the
check.

Monday and Tuesday: "not now" to the first weekday card. Tuesday: "don't mention these" to the doubled word.
Wednesday: one person says "thanks" to a weekday card, one uses its fix, one ignores it."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import pytest

from pilot.live import learning_week as week
from pilot.live.engine import LiveCompanion

EVERYONE = [p.name for p in week.PEOPLE]


@pytest.fixture(scope="module")
def days():
    return asyncio.run(week.run_all())


@pytest.mark.parametrize("person", EVERYONE)
def test_monday_and_tuesday_not_now_holds_for_the_moment(days, person):
    for day in days[person][:2]:
        assert day.delivered["weekday_date"] == ["spoken"]  # the second weekday slip, minutes later, held back
        assert day.count("repeated_word") == 1


@pytest.mark.parametrize("person", EVERYONE)
def test_wednesday_after_two_days_of_not_now(days, person):
    wednesday = days[person][2]
    assert wednesday.delivered["weekday_date"] == ["shown", "shown"]  # shown, not spoken
    assert wednesday.count("repeated_word") == 0  # "don't mention these" on Tuesday: never again
    assert wednesday.count("arithmetic") == 1


def test_thursday_follows_what_they_did_on_wednesday(days):
    assert days["says thanks"][3].delivered["weekday_date"][0] == "spoken"  # "thanks": spoken again at once
    assert days["fixes it"][3].delivered["weekday_date"][0] == "spoken"  # using the fix is as welcome
    assert days["ignores the rest"][3].delivered["weekday_date"] == ["shown", "shown"]  # still a week of quiet
    for person in EVERYONE:
        assert days[person][3].count("repeated_word") == 0


def test_the_table(days):
    assert week.table(days).count("\n") == 3 * len(week.DAYS)


# ── the rule on its own ──


def _live(tmp_path, wall):
    async def nothing(*_args):
        return None

    return LiveCompanion(
        settings=lambda: None, observe=nothing, emit=nothing, speak=nothing,
        relevance_path=tmp_path / "live_relevance.json", wallclock=lambda: wall[0],
    )  # fmt: skip


def test_two_days_within_a_week_and_a_week_after_the_last(tmp_path):
    wall = [datetime(2026, 11, 9, 11)]
    live = _live(tmp_path, wall)
    live.snooze("weekday_date")
    live.snooze("weekday_date")  # the same day twice is one day
    assert live.quiet_kinds() == frozenset()
    wall[0] += timedelta(days=1)
    live.snooze("weekday_date")
    assert live.quiet_kinds() == {"weekday_date"}
    again = _live(tmp_path, wall)  # kept across a restart
    assert again.quiet_kinds() == {"weekday_date"}
    wall[0] += timedelta(days=7)
    assert again.quiet_kinds() == frozenset()  # a week after the last "not now"


def test_undoing_not_now_speaks_it_again_and_the_old_file_still_reads(tmp_path):
    wall = [datetime(2026, 11, 9, 11)]
    live = _live(tmp_path, wall)
    live.snooze("arithmetic")
    wall[0] += timedelta(days=1)
    live.snooze("arithmetic")
    assert live.unsnooze("arithmetic") and live.quiet_kinds() == frozenset()
    old = tmp_path / "live_snoozed.json"
    old.write_text(json.dumps({"arithmetic": wall[0].timestamp() + 600}), encoding="utf-8")  # the file before
    assert "arithmetic" in _live(tmp_path, wall).snoozed_kinds()


def test_not_now_no_longer_lowers_relevance(tmp_path):
    """One mechanism for "not now": its days. Relevance moves only for thanks, fixes, mutes and cards left alone."""
    live = _live(tmp_path, [datetime(2026, 11, 9, 11)])
    before = live.relevance.score("weekday_date")
    entry = {"id": "c1", "kind": "weekday_date", "urgency": "normal", "at": 0.0}
    live._recent.append(entry)
    asyncio.run(live.feedback("c1", "dismissed", from_person=True))
    assert live.relevance.score("weekday_date") == before and "weekday_date" in live.snoozed_kinds()
