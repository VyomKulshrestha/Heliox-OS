"""A day in the life: one made-up person's working day and the next morning, through the real pieces, on fake clocks.

The day is in tests/data/a_day_in_the_life.json, readable by a person. What it uses, through their public faces:
- the conversation sessions' own runner (benchmarks/conversation_eval/sessions.run_session) for everything said:
  the real server, addressing rule, conversation layer and handlers, with a stub planner and executor, and either no
  model or a stub cloud model, with the profile switch on "auto" and on "off";
- the not-addressed corpus (benchmarks/conversation_bench/corpus.lines) for an hour of talk not meant for Heliox;
- the real Live engine on the Live Bench's typing sessions (pilot.live.bench.Session) for what is typed;
- the real offer rules (pilot.personal.offers) for what is offered, held by pilot.personal.wiring.quiet_now;
- the real reminder store and the server's reminder delivery for the lid closed over a reminder.

The spoken part and the rest run on two servers on the same day's clock: the session runner builds its own server and
has no hook for typed text, offers or a freeze. Each assertion says what a person would have lived through if it failed.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.conversation_bench.corpus import lines as corpus_lines
from benchmarks.conversation_eval.sessions import run_session

DAY = json.loads((Path(__file__).parent / "data" / "a_day_in_the_life.json").read_text(encoding="utf-8"))
THURSDAY = datetime(2026, 11, 12)


def at(hhmm: str, day: datetime = THURSDAY) -> datetime:
    hour, minute = (int(part) for part in hhmm.split(":"))
    return day.replace(hour=hour, minute=minute)


# ── what is said: three runs of the same day ──


def spoken_session(profile_off: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """The day's spoken turns as one session, and where each part of it starts."""
    spoken = DAY["spoken"]
    turns: list[dict[str, Any]] = []
    marks: dict[str, Any] = {"corpus": [], "turn": {}}
    if profile_off:
        turns.append({"say": spoken["profile_off"]})
    for turn in spoken["morning"]:
        marks["turn"][len(turns)] = turn
        turns.append({key: turn[key] for key in ("say", "gap") if key in turn})
    hour = spoken["talk_not_addressed"]
    for line in corpus_lines(count=hour["lines"], seed=20261112):
        marks["corpus"].append(len(turns))
        turns.append({"say": line.text, "gap": hour["gap"]})
    for turn in spoken["evening"]:
        marks["turn"][len(turns)] = turn
        turns.append({key: turn[key] for key in ("say", "gap") if key in turn})
    turns.append({"event": "restart"})
    if profile_off:
        # The session runner's restart builds a fresh config instead of loading the one saved (the daemon loads it),
        # so the switch is said again; a real restart keeps it.
        turns.append({"say": spoken["profile_off"]})
    for turn in spoken["next_morning"]:
        marks["turn"][len(turns)] = turn
        turns.append({key: turn[key] for key in ("say", "gap") if key in turn})
    return {"name": "a day in the life", "plans": spoken["plans"], "turns": turns}, marks


@pytest.fixture(scope="module")
def days() -> dict[str, tuple[Any, dict[str, Any]]]:
    runs = {}
    for label, kind, off in (
        ("no model", "none", False),
        ("cloud, auto", "claude", False),
        ("cloud, off", "claude", True),
    ):
        session, marks = spoken_session(off)
        record, _expectations = asyncio.run(run_session(session, kind))
        runs[label] = (record, marks)
    return runs


def said_on(record: Any, turn: int) -> str:
    return " ".join(s.text for s in record.said if s.turn == turn)


@pytest.mark.parametrize("label", ["no model", "cloud, auto", "cloud, off"])
def test_nothing_not_addressed_is_acted_on(days, label):
    record, marks = days[label]
    for turn in marks["corpus"]:
        acted = [r.action_type for r in record.ran if r.turn == turn] + record.planned.get(turn, [])
        sent = record.requests.get(turn, [])
        assert not acted and not sent, (
            f"During the meeting, Heliox acted on something not said to it ({label}): turn {turn} planned or ran "
            f"{acted!r} and sent {len(sent)} model request(s)"
        )


@pytest.mark.parametrize("label", ["no model", "cloud, auto", "cloud, off"])
def test_the_requests_that_act(days, label):
    record, marks = days[label]
    for turn, mark in marks["turn"].items():
        if "corrected" in mark:
            targets = " ".join(r.target for r in record.ran if r.turn == turn)
            assert mark["corrected"]["never_ran"] not in targets, (
                f"Sam said 'open the notes folder, no, the podcast folder' and Heliox opened the notes folder ({label})"
            )
            if label != "no model":  # with no model the planner road says it needs one
                assert mark["corrected"]["ran"] in targets, f"Sam's corrected request opened nothing ({label})"
        if "stopped" in mark:
            assert not any(r.action_type == mark["stopped"] for r in record.ran), (
                f"Sam said 'wait, stop' and the screenshots were deleted anyway ({label})"
            )
        if "held_slip" in mark:
            assert mark["held_slip"].casefold() in said_on(record, turn).casefold(), (
                f"Sam asked for a reminder on 'Tuesday the nineteenth' (a Thursday) and Heliox did not ask which ({label})"
            )


@pytest.mark.parametrize("label", ["no model", "cloud, auto", "cloud, off"])
def test_the_next_morning(days, label):
    record, marks = days[label]
    for turn, mark in marks["turn"].items():
        reply = said_on(record, turn)
        for word in mark.get("remembers", []):
            assert word.casefold() in reply.casefold(), (
                f"The next morning Heliox had forgotten {word!r} ({label}): it said {reply!r}"
            )
        for word in mark.get("forgot", []):
            assert word.casefold() not in reply.casefold(), (
                f"The next morning Heliox still knew {word!r}, which it never keeps ({label}): {reply!r}"
            )
        if "not_from_memory" in mark:
            assert mark["not_from_memory"] not in reply, f"Heliox recalled a flight Sam never mentioned ({label})"


def test_what_left_the_machine(days):
    none, _ = days["no model"]
    assert not any(none.requests.values()) and not any(none.model_calls.values()), (
        "With no model set up, something was sent to a model"
    )
    auto, _ = days["cloud, auto"]
    sent_auto = " ".join(" ".join(bodies) for bodies in auto.requests.values())
    assert sent_auto, "With a cloud model, the day sent nothing at all, so this run proves nothing"
    off, _ = days["cloud, off"]
    sent_off = " ".join(" ".join(bodies) for bodies in off.requests.values())
    planned_off = " ".join(" ".join(given) for given in off.planned.values())
    for word in DAY["spoken"]["never_sent_when_off"]:
        assert word not in sent_off and word not in planned_off, (
            f"Sam asked Heliox to keep the profile on this computer, and {word!r} went to the cloud model"
        )


# ── what is typed, offered, and the lid: the real Live engine, offer rules and reminder delivery ──


class DayClock:
    def __init__(self) -> None:
        self.now = at("07:00")

    def wall(self) -> datetime:
        return self.now

    def tick(self) -> float:
        return self.now.timestamp()


@pytest.fixture(scope="module")
def person(tmp_path_factory):
    from pilot.workflows.user_profile import UserProfile

    clock = DayClock()
    profile = UserProfile(tmp_path_factory.mktemp("person") / "profile.json", clock=clock.tick)
    for said in DAY["person"]["told"]:
        asyncio.run(profile.answer_async(said))
    server = SimpleNamespace(_user_profile=profile, _focus=None, _broadcast_notification=_nothing)
    return SimpleNamespace(clock=clock, profile=profile, server=server)


async def _nothing(*_args, **_kwargs):
    return None


def test_what_live_raised_while_typing(person):
    from pilot.live.bench import PERSONAS, Session, expand
    from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation
    from pilot.personal.wiring import quiet_now

    state: dict[str, Any] = {"session": None}
    delivered: list[dict[str, Any]] = []
    settings = SimpleNamespace(
        enabled=True, level="balanced", listen_to_speech=False, muted_kinds=[], personal_timing=True
    )
    sim = {"t": 1000.0}

    async def observe():
        session = state["session"]
        if session is None:
            return None
        app, _, title = session.window.partition("|")
        return Observation(text=session.doc, app=app, title=title, source="focused_field", editable=True)

    async def emit(event, payload):
        if event == "live_interjection":
            delivered.append({**payload, "at": person.clock.now.strftime("%H:%M")})

    live = LiveCompanion(
        settings=lambda: settings, observe=observe, emit=emit, speak=_speak_nothing,
        clock=lambda: sim["t"], today=lambda: THURSDAY.date(),
        focus_active=lambda: bool(quiet_now(person.server, person.clock.now)),  # as the server wires it
        wallclock=lambda: person.clock.now,
    )  # fmt: skip
    unwelcome: list[str] = []
    for entry in DAY["typed"]:
        person.clock.now = at(entry["at"])
        scenario = {"name": entry["name"], "app": entry["app"], "title": entry["title"], "expect": entry["welcome"],
                    "steps": entry["steps"], "today": entry["today"]}  # fmt: skip
        session = Session(scenario, expand(scenario, PERSONAS["steady"], random.Random(entry["name"])), sim["t"])
        state["session"] = session
        seen = len(delivered)

        async def play(session=session):
            while sim["t"] < session.end + 20:
                session.advance(sim["t"])
                await live.tick()
                sim["t"] = round(sim["t"] + TICK_SECONDS, 3)

        asyncio.run(play())
        got = delivered[seen:]
        for card in got:
            if card["kind"] not in entry["welcome"]:
                unwelcome.append(f"{card['kind']} at {card['at']} while typing {entry['name']!r}")
        for kind in entry["welcome"]:
            assert any(c["kind"] == kind for c in got), (
                f"Sam typed {entry['name']!r} at {entry['at']} and Live missed it"
            )
        live._pending.clear()
        state["session"] = None
    assert not unwelcome, f"Live raised what it should not have: {unwelcome}"
    assert not any(c["at"] >= "18:00" for c in delivered), (
        f"Live spoke up after Sam's working day had ended: {[c for c in delivered if c['at'] >= '18:00']}"
    )


async def _speak_nothing(_text, _urgent):
    return "spoken"


@pytest.fixture(scope="module")
def offered(person, tmp_path_factory) -> list[tuple[str, str, str]]:
    """The day's offers, (time, kind, key), through the real offer rules, with the working day the person told, their
    habits, goals and list, and a routine record holding a request they made at ten on the weekdays before."""
    from pilot.personal.offers import Offers, day_moments
    from pilot.personal.routines import RoutineRecord
    from pilot.personal.wiring import quiet_now
    from pilot.workflows.suggestions import Suggester

    habits = {h["name"]: {**h, "today": False} for h in DAY["person"]["habits"]}
    goals = [dict(g) for g in DAY["person"]["goals"]]
    todos = [{"id": str(i), "text": f"thing {i}"} for i in range(DAY["person"]["todos"])]
    clock = person.clock
    history = DAY["offers"]["routine_history"]
    record_path = tmp_path_factory.mktemp("routines") / "routines.json"
    days, back = [], 1
    while len(days) < history["weekdays_before"]:
        day = THURSDAY - timedelta(days=back)
        if day.weekday() < 5:
            days.append(day)
        back += 1
    verb, _, target = history["request"].partition(" ")
    record_path.write_text(json.dumps({"requests": [
        {"kind": verb, "target": target, "day": d.date().isoformat(), "hour": history["hour"]} for d in days
    ]}), encoding="utf-8")  # fmt: skip
    record = RoutineRecord(record_path, now=clock.wall)
    suggester = Suggester(stale_goals=lambda: goals, habits=lambda: list(habits.values()), open_todos=lambda: todos,
                          now=clock.wall)  # fmt: skip
    offers = Offers(
        mode=lambda: "auto", profile_empty=person.profile.is_empty, suggestions=suggester.picks,
        moments=lambda now: day_moments(now, start=person.profile.work_start_minute(),
                                        end=person.profile.work_end_minute()) + record.offers(now),
        now=clock.wall, clock=clock.tick, timeline_off=lambda: False,
    )  # fmt: skip
    meeting = DAY["offers"]["meeting"]
    answers = DAY["offers"]["answers"]
    made: list[tuple[str, str, str]] = []
    clock.now = at("07:00")
    while clock.now < at("23:00"):
        hhmm = clock.now.strftime("%H:%M")
        in_meeting = meeting["from"] <= hhmm < meeting["to"]  # a call: the offer rules hold everything
        if not in_meeting and not quiet_now(person.server, clock.now):
            offer = offers.next_offer()
            if offer is not None:
                offers.mark_offered(offer)
                made.append((hhmm, offer.kind, offer.key))
                said = answers.get(offer.key) or answers.get(offer.kind)
                if said:
                    offers.reply(said)
        clock.now += timedelta(minutes=5)
    return made


def test_what_was_offered(offered):
    made = [(t, k) for t, k, _key in offered]
    meeting = DAY["offers"]["meeting"]
    unwelcome = [m for m in made if m[1] not in DAY["offers"]["welcome"] + ["learned_routine"]]
    assert not unwelcome, f"Heliox offered what Sam's day did not call for: {unwelcome}"
    assert not [m for m in made if m[0] >= "18:00"], f"Heliox made an offer after six: {made}"
    assert not [m for m in made if meeting["from"] <= m[0] < meeting["to"]], (
        f"Heliox offered during the meeting: {made}"
    )
    assert "routine" in [k for _t, k in made], f"Sam told Heliox their working day and got no offer from it: {made}"
    assert len(made) <= 3, f"Heliox made {len(made)} offers in one day, more than its own cap: {made}"


# ── personal: the moments a person would expect Heliox to use what it knows, unasked ──


def personal_moments(
    record: Any, marks: dict[str, Any], offered: list[tuple[str, str, str]]
) -> list[tuple[str, bool, str]]:
    """(the moment, delivered, what happened) for every moment the day file marks."""
    found = []
    for turn, mark in marks["turn"].items():
        moment = mark.get("moment")
        if not moment:
            continue
        reply = said_on(record, turn)
        given = " ".join(record.planned.get(turn, []))
        hit = all(word.casefold() in f"{reply} {given}".casefold() for word in moment["expects"])
        found.append((moment["name"], hit, f"said {reply!r}"))
    wrap = DAY["wrap_up_moment"]
    from pilot.workflows.day_briefing import compose_night

    night = compose_night(at("17:55"), brief="", focus_seconds=0, finished_todos=0, open_todos=0, reminders=[])
    found.append((wrap["name"], wrap["expects"].casefold() in night.casefold(), f"said {night!r}"))
    for moment in DAY["offers"]["moments"]:
        if "quiet" in moment:
            begin, end = moment["quiet"]
            during = [o for o in offered if begin <= o[0] < end]
            found.append((moment["name"], not during, f"offers then: {during}"))
            continue
        begin, end = moment["between"]
        hits = [o for o in offered if begin <= o[0] < end and o[1] == moment["kind"]
                and ("key" not in moment or o[2] == moment["key"])]  # fmt: skip
        found.append((moment["name"], bool(hits), f"offers: {[o for o in offered]}"))
    return found


def test_personal_moments_delivered_against_expected(days, offered, capsys):
    """The number first: each run's delivered moments against the expected ones, never fewer than the day file's
    baseline. The misses are printed in a person's words."""
    baseline = DAY["personal_moments_baseline"]
    lines = []
    worse = []
    for label in ("no model", "cloud, auto", "cloud, off"):
        record, marks = days[label]
        found = personal_moments(record, marks, offered)
        delivered = sum(hit for _name, hit, _what in found)
        lines.append(f"{label}: {delivered} of {len(found)}")
        lines += [f"    missed: {name} ({what})" for name, hit, what in found if not hit]
        if baseline.get(label) is not None and delivered < baseline[label]:
            worse.append(f"{label}: {delivered}, was {baseline[label]}")
    with capsys.disabled():
        print("\nPersonal moments, delivered of expected:\n" + "\n".join(lines))
    assert not worse, f"Heliox used less of what it knows about Sam than before: {worse}"


def test_the_lid_closed_over_a_reminder(tmp_path):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer
    from pilot.workflows.reminders import ReminderStore

    lid = DAY["lid"]
    server = PilotServer(PilotConfig())
    shown: list[tuple[str, dict[str, Any]]] = []

    async def broadcast(method, params, **_kwargs):
        if server._clients:
            shown.append((method, params))
            return len(server._clients)
        return 0

    server._broadcast_notification = broadcast
    now = {"t": at("10:00").timestamp()}
    server._reminders = ReminderStore(
        tmp_path / "reminders.json", deliver=server._deliver_reminder, deliver_missed=server._deliver_missed_reminders,
        deliver_held=server._deliver_held_reminders, clock=lambda: now["t"],
    )  # fmt: skip
    server._reminders.add(at(lid["due"]).timestamp(), lid["reminder"])
    now["t"] = at(lid["opened"]).timestamp()  # the lid was closed from 15:50: nothing ran until it opened

    async def wake_then_window():
        await server._on_wake(SimpleNamespace(kind="wake", seconds=1800))
        server._clients.add(object())  # Sam opens Heliox's window at 16:25
        server._on_window_connected()
        for task in list(server.__dict__.get("_notice_tasks", ())):
            await task

    asyncio.run(wake_then_window())
    cards = [p for m, p in shown if m in ("reminder", "reminders_missed")]
    assert len(cards) == 1, f"The four o'clock reminder reached Sam's window {len(cards)} times, not once: {cards}"
    message = cards[0].get("message", "")
    assert message.startswith(lid["message_starts"].split(" at ")[0]), (
        f"The reminder Sam missed while the lid was closed did not say when it was due: {message!r}"
    )
    assert server._reminders.held() == [] and server._reminders.list() == [], "The reminder was given and still waits"


@pytest.mark.xfail(
    strict=True,
    reason="with no model, 'what do you know about me' ends 'Your name, style and what bears on a question go to "
    "none.' (PilotServer._profile_sent_sentence, the conversation agent's): nothing goes anywhere",
)
def test_with_no_model_nothing_is_said_to_go_anywhere(days):
    record, marks = days["no model"]
    for turn, mark in marks["turn"].items():
        if "remembers" in mark and "Sam" in mark["remembers"]:
            reply = said_on(record, turn)
            assert " go to " not in reply, f"With no model, Heliox told Sam their profile goes somewhere: {reply!r}"
