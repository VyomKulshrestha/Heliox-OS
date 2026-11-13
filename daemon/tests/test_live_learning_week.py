"""What Monday's "not now" and Tuesday's "don't mention these" do to Wednesday (pilot.live.learning_week): three
made-up people, the same four slips each day, through the real LiveCompanion kept across the days. The numbers are
the check."""

from __future__ import annotations

import asyncio

import pytest

from pilot.live import learning_week as week


@pytest.fixture(scope="module")
def days():
    return asyncio.run(week.run_all())


@pytest.mark.parametrize("person", [p.name for p in week.PEOPLE])
def test_monday(days, person):
    monday = days[person][0]
    # "Not now" to the first weekday card: the second weekday slip, minutes later, is held back.
    assert monday.delivered["weekday_date"] == ["spoken"]
    assert monday.count("arithmetic") == 1 and monday.count("repeated_word") == 1


@pytest.mark.parametrize("person", [p.name for p in week.PEOPLE])
def test_tuesday_not_now_was_for_monday_only(days, person):
    tuesday = days[person][1]
    assert tuesday.count("weekday_date") == 2 and tuesday.delivered["weekday_date"][0] == "spoken"
    assert tuesday.count("repeated_word") == 1  # then "don't mention these"


@pytest.mark.parametrize("person", [p.name for p in week.PEOPLE])
def test_wednesday(days, person):
    wednesday = days[person][2]
    assert wednesday.count("repeated_word") == 0  # muted on Tuesday: never again
    assert wednesday.count("weekday_date") == 2 and wednesday.delivered["weekday_date"][0] == "spoken"
    assert wednesday.count("arithmetic") == 1


def test_the_table(days):
    text = week.table(days)
    assert text.count("\n") == 3 * len(week.DAYS) and "Wednesday" in text
