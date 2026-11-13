"""Offers tied to the working day the person told Heliox (pilot.personal.offers.day_moments): their day when it
starts, wrapping up just before it ends, each once a day with "why" answered from what they said, and the next day
shaped by what they answered (a simulated week, minute by minute through the offer rules)."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from pilot.personal.offers import Offers, day_moments
from pilot.personal.wiring import quiet_now
from pilot.workflows.user_profile import UserProfile, parse_work_start

MONDAY = datetime(2026, 11, 16)


class Clock:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at.timestamp()


@pytest.mark.parametrize(
    ("said", "minute"),
    [("I start work at nine", 9 * 60), ("I start work at 8:30", 8 * 60 + 30), ("my working day starts at 13:00", 13 * 60),
     ("I begin work at 12", 12 * 60), ("I start work at noon-ish", None)],
)  # fmt: skip
def test_parse_work_start(said, minute):
    assert parse_work_start(said) == minute


@pytest.mark.asyncio
async def test_told_and_said_back(tmp_path):
    profile = UserProfile(tmp_path / "p.json")
    assert await profile.answer_async("I start work at nine") == "Okay, you start work at 9 am."
    assert await profile.answer_async("I stop work at six") == (
        "Okay, you stop work at 6 pm. After that I'll hold offers and suggestions until morning."
    )
    assert (profile.work_start_minute(), profile.work_end_minute()) == (9 * 60, 18 * 60)


def test_the_moments_themselves():
    at = lambda h, m=0: MONDAY.replace(hour=h, minute=m)  # noqa: E731
    assert [o.key for o in day_moments(at(9, 10), start=540, end=1080)] == ["routine:day_start"]
    assert day_moments(at(9, 50), start=540, end=1080) == []
    wrap = day_moments(at(17, 45), start=540, end=1080)
    assert [o.text for o in wrap] == ["It's nearly 6 pm, when you stop work. Want me to wrap up the day?"]
    assert wrap[0].why == "You told me you stop work at 6 pm." and wrap[0].request == "wrap up my day"
    assert day_moments(at(18, 0), start=540, end=1080) == []  # the evening is quiet
    assert day_moments(MONDAY.replace(day=21, hour=9, minute=10), start=540, end=1080) == []  # a Saturday
    assert day_moments(at(9, 10), start=None, end=None) == []  # nothing told, nothing offered


async def _week(tmp_path, answers: dict[tuple[int, str], str]) -> list[tuple[int, str, str]]:
    """Five weekdays, looking every five minutes from 7:00 to 20:00 as the offer loop does at quiet moments.
    ``answers``: (day, offer key) -> what the person said to it. Returns (day, time, key) for every offer made."""
    clock = Clock(MONDAY.replace(hour=7))
    profile = UserProfile(tmp_path / "p.json", clock=clock)
    await profile.answer_async("I start work at nine")
    await profile.answer_async("I stop work at six")
    server = SimpleNamespace(_user_profile=profile, _focus=None)

    async def broadcast(method, params):
        return None

    server._broadcast_notification = broadcast
    offers = Offers(
        mode=lambda: "auto",
        profile_empty=profile.is_empty,
        moments=lambda now: day_moments(now, start=profile.work_start_minute(), end=profile.work_end_minute()),
        now=lambda: clock.at,
        clock=clock,
        state_path=tmp_path / "offers.json",
    )
    made: list[tuple[int, str, str]] = []
    for day in range(5):
        clock.at = (MONDAY + timedelta(days=day)).replace(hour=7)
        while clock.at.hour < 20:
            if not quiet_now(server, clock.at):
                offer = offers.next_offer()
                if offer is not None:
                    offers.mark_offered(offer)
                    made.append((day, clock.at.strftime("%H:%M"), offer.key))
                    said = answers.get((day, offer.key))
                    if said:
                        offers.reply(said)
            clock.at += timedelta(minutes=5)
    return made


@pytest.mark.asyncio
async def test_a_week_each_once_a_day_at_its_moment(tmp_path):
    made = await _week(tmp_path, {})
    assert [(d, k) for d, _t, k in made] == [(d, k) for d in range(5) for k in ("routine:day_start", "routine:wrap_up")]
    assert all("09:00" <= t < "09:45" for _d, t, k in made if k == "routine:day_start")
    assert all("17:40" <= t < "18:00" for _d, t, k in made if k == "routine:wrap_up")


@pytest.mark.asyncio
async def test_the_next_day_follows_the_answer(tmp_path):
    """ "No" holds an offer for the rest of that day only; "don't offer that" ends it for good, and the other stays."""
    made = await _week(
        tmp_path, {(0, "routine:day_start"): "not today", (1, "routine:wrap_up"): "don't offer that again"}
    )
    keys = [(d, k) for d, _t, k in made]
    assert (1, "routine:day_start") in keys  # "not today" on Monday: offered again on Tuesday
    assert (1, "routine:wrap_up") in keys and not any(k == "routine:wrap_up" for d, k in keys if d >= 2)
    assert all((d, "routine:day_start") in keys for d in range(5))


@pytest.mark.asyncio
async def test_quiet_unless_asked_holds_them(tmp_path):
    clock = Clock(MONDAY.replace(hour=9, minute=5))
    profile = UserProfile(tmp_path / "p.json", clock=clock)
    await profile.answer_async("I start work at nine")
    await profile.answer_async("stay quiet unless I ask")
    assert quiet_now(SimpleNamespace(_user_profile=profile, _focus=None), clock.at) == "asked"
