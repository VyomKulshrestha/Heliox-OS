"""How many offers a day brings, when, and how many are declined: a simulated working week for made-up people.

Each person is a profile, goals, habits, a to-do list, the requests they finish during the day, and how they answer
each kind of offer. The week runs Monday to Friday, 07:00 to 23:00, looking every five minutes as the offer loop does
at quiet moments, through the real rules: ``Offers`` (caps, "not today", "don't offer that", the getting-to-know
offer), ``Suggester`` (habits, quiet goals, the to-do list), ``day_moments`` (the working day the person told) and
``goal_progress`` (a finished request that matches a goal), with ``quiet_now`` holding everything after the working
day and while asked to stay quiet. Nothing is spoken and nothing leaves the machine; it is pure arithmetic on a fake
clock. ``python -m pilot.personal.offer_day`` prints the table.

What it does not show: whether a real person wants these offers, how a real day is spread, or anything about the
timeline's usual-app offers (the timeline is off by default and not simulated).
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pilot.personal.offers import Offers, day_moments, goal_progress
from pilot.personal.wiring import quiet_now
from pilot.workflows.suggestions import Suggester
from pilot.workflows.user_profile import UserProfile

MONDAY = datetime(2026, 11, 16)
STEP = timedelta(minutes=5)


@dataclass
class Person:
    name: str
    told: list[str] = field(default_factory=list)  # what they told Heliox about themselves, in their words
    goals: list[dict[str, Any]] = field(default_factory=list)
    habits: list[dict[str, Any]] = field(default_factory=list)  # {"name", "streak"}
    todos: int = 0
    requests: dict[str, str] = field(default_factory=dict)  # "HH:MM" -> a request finished then, every weekday
    answers: dict[str, str] = field(default_factory=dict)  # offer kind -> what they say ("yes", "no", "don't ...")
    answers_by_day: dict[tuple[int, str], str] = field(default_factory=dict)  # (day, kind) overrides


def _stale(title: str, step: str, gid: str) -> dict[str, Any]:
    return {"id": gid, "title": title, "status": "active", "next_step": step, "updated_at": 0.0}


PEOPLE = (
    Person(
        "office, nine to six",
        told=["call me Sam", "I start work at nine", "I stop work at six", "I prefer short answers"],
        goals=[_stale("launch the podcast", "book a guest", "g1")],
        habits=[{"name": "water", "streak": 4}],
        todos=3,
        requests={"10:30": "open the podcast folder", "14:00": "find the quarterly report"},
        answers={"routine": "yes", "goal_progress": "yes", "habit": "yes", "stale_goal": "no", "todos": "no"},
        answers_by_day={(2, "routine"): "not today", (3, "stale_goal"): "don't offer that again"},
    ),
    Person(
        "student, no fixed hours",
        told=["call me Alex", "I'm studying chemistry"],
        goals=[_stale("finish the thesis", "write the methods section", "g2")],
        habits=[{"name": "reading", "streak": 3}],
        todos=5,
        answers={"habit": "yes", "stale_goal": "no", "todos": "no"},
        answers_by_day={(1, "todos"): "don't offer that again"},
    ),
    Person("told Heliox nothing"),
    Person(
        "a week of use, nothing told",
        requests={"10:00": "open outlook", "15:00": "play some music"},
        answers={"timeline": "no"},
    ),
)


@dataclass
class Day:
    offers: list[tuple[str, str]] = field(default_factory=list)  # (HH:MM, kind)
    declined: int = 0
    accepted: int = 0


async def run_week(person: Person, folder: Path) -> list[Day]:
    """Five weekdays for ``person``; one ``Day`` each."""
    clock_at = [MONDAY.replace(hour=7)]
    tick = lambda: clock_at[0].timestamp()  # noqa: E731
    profile = UserProfile(folder / "profile.json", clock=tick)
    for said in person.told:
        await profile.answer_async(said)
    habits = {h["name"]: {"name": h["name"], "streak": h["streak"], "today": False} for h in person.habits}
    goals = [dict(g) for g in person.goals]
    todos = [{"id": str(i), "text": f"thing {i}"} for i in range(person.todos)]
    suggester = Suggester(
        stale_goals=lambda: [g for g in goals if g.get("next_step")],
        habits=lambda: list(habits.values()),
        open_todos=lambda: todos,
        now=lambda: clock_at[0],
    )
    offers = Offers(
        mode=lambda: "auto",
        profile_empty=profile.is_empty,
        suggestions=suggester.picks,
        moments=lambda now: day_moments(now, start=profile.work_start_minute(), end=profile.work_end_minute()),
        now=lambda: clock_at[0],
        clock=tick,
        state_path=folder / "offers.json",
        timeline_off=lambda: True,  # the default
    )
    server = SimpleNamespace(_user_profile=profile, _focus=None)

    async def broadcast(method: str, params: Any) -> None:
        return None

    server._broadcast_notification = broadcast
    week: list[Day] = []
    for index in range(5):
        day = Day()
        clock_at[0] = (MONDAY + timedelta(days=index)).replace(hour=7)
        for habit in habits.values():
            habit["today"] = False
        while clock_at[0].hour < 23:
            hhmm = clock_at[0].strftime("%H:%M")
            done = person.requests.get(hhmm)
            if done:
                offers.note_used(clock_at[0].date().isoformat())  # a request taken: a day of use
                offer = goal_progress(done, goals)
                if offer is not None:
                    offers.offer_soon(offer)
            if not quiet_now(server, clock_at[0]):
                offer = offers.next_offer()
                if offer is not None:
                    offers.mark_offered(offer)
                    day.offers.append((hhmm, offer.kind))
                    said = person.answers_by_day.get((index, offer.kind)) or person.answers.get(offer.kind, "")
                    if said:
                        offers.reply(said)
                    if said == "yes":
                        day.accepted += 1
                        if offer.kind == "habit":
                            for habit in habits.values():
                                habit["today"] = True
                    elif said:
                        day.declined += 1
            clock_at[0] += STEP
        week.append(day)
    return week


async def run_all() -> dict[str, list[Day]]:
    results: dict[str, list[Day]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for number, person in enumerate(PEOPLE):
            folder = Path(tmp) / str(number)
            folder.mkdir()
            results[person.name] = await run_week(person, folder)
    return results


def table(results: dict[str, list[Day]]) -> str:
    lines = ["person | offers per day (Mon-Fri) | total | accepted | declined | kinds"]
    for name, week in results.items():
        per_day = [len(day.offers) for day in week]
        kinds: dict[str, int] = {}
        for day in week:
            for _hhmm, kind in day.offers:
                kinds[kind] = kinds.get(kind, 0) + 1
        lines.append(
            f"{name} | {per_day} | {sum(per_day)} | {sum(d.accepted for d in week)} | "
            f"{sum(d.declined for d in week)} | {dict(sorted(kinds.items()))}"
        )
    return "\n".join(lines)


def main() -> int:
    from pilot.real_folders import use_scratch_folders

    use_scratch_folders()  # nothing here writes Heliox's folders, but a script outside pytest always starts so
    print(table(asyncio.run(run_all())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
