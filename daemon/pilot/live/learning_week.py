"""Does what the person told Live on Monday and Tuesday change Wednesday? A simulated three days, measured.

Three made-up people type the same four slips each day (two weekdays that do not match their dates, a wrong sum, a
doubled word, from the Live Bench's own scenarios) through the real ``LiveCompanion``, with one engine kept across
the days, as one run of Heliox would be (relevance, snoozes and the muted kinds carry over; a day apart on the clock).

On Monday each person says "not now" to the first weekday card; on Tuesday "don't mention these" to the doubled word.
They differ in what they do with every other card: one says "thanks", one fixes the text, one ignores it. The result
is, per person and day, what was delivered of each kind, and whether it was spoken or only shown.

``python -m pilot.live.learning_week`` prints the table; ``tests/test_live_learning_week.py`` asserts it.
What it does not show: a real person's week, or how relevance moves over months.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from pilot.live.bench import PERSONAS, Session, expand, load_scenarios, warm_up
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday")
SCENARIOS = (
    "email: weekday that doesn't match the date, mid-sentence",
    "notes: wrong product in a budget line",
    "notes: a doubled word",
    "chat: offsite on the wrong weekday",
)
DAY_SECONDS = 86400.0


@dataclass
class Person:
    name: str
    rest: str  # what they do with every other card: "thanks", "fix" or "ignore"


PEOPLE = (Person("says thanks", "thanks"), Person("fixes it", "fix"), Person("ignores the rest", "ignore"))


@dataclass
class Day:
    delivered: dict[str, list[str]] = field(default_factory=dict)  # kind -> ["spoken" | "shown", ...]

    def count(self, kind: str) -> int:
        return len(self.delivered.get(kind, []))


async def run_person(person: Person) -> list[Day]:
    by_name = {s["name"]: s for s in load_scenarios()}
    scenarios = [by_name[name] for name in SCENARIOS]
    persona = PERSONAS["steady"]
    clock = {"t": 1000.0}
    settings = SimpleNamespace(
        enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[], personal_timing=True
    )
    state: dict[str, Any] = {"session": None, "day": 0}
    deliveries: list[dict[str, Any]] = []

    async def observe() -> Observation | None:
        session = state["session"]
        if session is None:
            return None
        app, _, title = session.window.partition("|")
        return Observation(text=session.doc, app=app, title=title, source="focused_field", editable=True)

    async def emit(event: str, payload: dict[str, Any]) -> None:
        if event == "live_interjection":
            deliveries.append(payload)

    async def speak(message: str, urgent: bool) -> None:
        return None

    async def mute(kind: str) -> None:
        settings.muted_kinds = sorted({*settings.muted_kinds, kind})

    def today() -> date:
        session = state["session"]
        return date.fromisoformat(session.scenario.get("today", "2026-11-09")) if session else date(2026, 11, 9)

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        user_speaking=lambda: False,
        in_call=lambda: "",
        people=lambda: dict(state["session"].scenario.get("people", {})) if state["session"] else {},
        clock=lambda: clock["t"],
        today=today,
        mute_kind=mute,
        wallclock=lambda: datetime(2026, 11, 9, 11) + timedelta(seconds=clock["t"] - 1000.0),  # days pass
    )

    async def play(scenario: dict[str, Any], day: Day | None, index: int) -> None:
        rng = random.Random(f"learning:{person.name}:{scenario['name']}")
        session = Session(scenario, expand(scenario, persona, rng), clock["t"])
        state["session"] = session
        seen = len(deliveries)
        while clock["t"] < session.end:
            session.advance(clock["t"])
            await live.tick()
            await asyncio.sleep(0)
            for payload in deliveries[seen:]:
                seen += 1
                if day is None:
                    continue
                kind = payload["kind"]
                day.delivered.setdefault(kind, []).append("spoken" if payload.get("spoken") else "shown")
                said = _answer(person, state["day"], kind, index)
                if said == "fix":
                    session.reactions.append((clock["t"] + 1.2, payload))
                    await live.feedback(payload["id"], "used_fix", from_person=True)
                elif said:
                    await live.handle_reply(said)
            clock["t"] = round(clock["t"] + TICK_SECONDS, 3)
        live._pending.clear()
        state["session"] = None
        clock["t"] += 2.0
        await live.tick()

    week: list[Day] = []
    await play(warm_up(persona), None, -1)
    for number in range(len(DAYS)):
        state["day"] = number
        day = Day()
        for index, scenario in enumerate(scenarios):
            await play(scenario, day, index)
        week.append(day)
        clock["t"] += DAY_SECONDS  # the next day: anything held "for now" has long run out
        # A new day brings new documents: what Live remembers of today's fields and cards goes, and only what it
        # learned stays (relevance, snoozes, the muted kinds, the person's timing).
        for remembered in (live._baselines, live._field_text, live._recent, live._visible):
            remembered.clear()
    return week


def _answer(person: Person, day: int, kind: str, index: int) -> str:
    if day in (0, 1) and kind == "weekday_date" and index == 0:
        return "not now"  # Monday and Tuesday: from Wednesday that kind is shown, not spoken
    if day == 2 and kind == "weekday_date" and index == 3 and person.rest == "thanks":
        return "thanks"  # Wednesday: "thanks" for it, so Thursday speaks it again for this person
    if day == 1 and kind == "repeated_word":
        return "don't mention these"
    return {"thanks": "thanks", "fix": "fix", "ignore": ""}[person.rest]


async def run_all() -> dict[str, list[Day]]:
    return {person.name: await run_person(person) for person in PEOPLE}


def table(results: dict[str, list[Day]]) -> str:
    kinds = ("weekday_date", "arithmetic", "repeated_word")
    lines = ["person | day | " + " | ".join(kinds)]
    for name, days in results.items():
        for label, day in zip(DAYS, days, strict=True):
            cells = [",".join(day.delivered.get(kind, [])) or "-" for kind in kinds]
            lines.append(f"{name} | {label} | " + " | ".join(cells))
    return "\n".join(lines)


def main() -> int:
    from pilot.real_folders import use_scratch_folders

    use_scratch_folders()
    print(table(asyncio.run(run_all())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
