"""The person's routines, learned from what they ask Heliox for (pilot.personal.routines), under every limit the
coordinator set: only requests addressed and taken; only kind, target, weekday and hour; 60 days; never in a prompt;
seen, forgotten and stopped by voice and in the window; one offer a day, with why."""

from __future__ import annotations

import ast
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.personal import routines as R
from pilot.personal.offers import Offers
from pilot.personal.routines import RoutineRecord, what_to_keep
from pilot.server import PilotServer
from pilot.workflows.privacy import BACKUP_FILES, STORES, clear_store
from pilot.workflows.user_profile import UserProfile

MONDAY = datetime(2026, 11, 16, 9, 10)
DAEMON = Path(__file__).resolve().parents[1]


class Now:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def weekdays_before(day: datetime, count: int) -> list[datetime]:
    found, n = [], 1
    while len(found) < count:
        d = day - timedelta(days=n)
        if d.weekday() < 5:
            found.append(d)
        n += 1
    return found


def seed(path: Path, entries: list[tuple[str, str, datetime]]) -> None:
    items = [{"kind": k, "target": t, "day": d.date().isoformat(), "hour": d.hour} for k, t, d in entries]
    path.write_text(json.dumps({"requests": items}), encoding="utf-8")


# ── what is kept ──


@pytest.mark.parametrize(
    ("said", "planned", "kept"),
    [
        ("open outlook", True, ("open", "outlook")),
        ("please open Outlook now", True, ("open", "outlook")),
        ("play some music", True, ("play", "some music")),
        ("delete the old files", True, None),  # never learned: anything that changes or sends
        ("send an email to Ana", True, None),
        ("good morning", False, ("day", "")),
        ("what's on my list", False, ("list", "")),
        ("what's the time", False, None),
    ],
)
def test_what_is_kept(said, planned, kept):
    assert what_to_keep(said, planned=planned) == kept


def test_only_kind_target_day_and_hour_on_disk(tmp_path):
    record = RoutineRecord(tmp_path / "r.json", now=Now(MONDAY))
    assert record.note("please open Outlook now", planned=True)
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data == {"requests": [{"kind": "open", "target": "outlook", "day": "2026-11-16", "hour": 9}]}
    assert "please" not in (tmp_path / "r.json").read_text(encoding="utf-8")


def test_learning_off_keeps_nothing(tmp_path):
    record = RoutineRecord(tmp_path / "r.json", enabled=lambda: False, now=Now(MONDAY))
    assert not record.note("open outlook", planned=True) and not (tmp_path / "r.json").exists()
    assert record.offers(MONDAY) == []


def test_kept_sixty_days_then_dropped(tmp_path):
    path = tmp_path / "r.json"
    seed(path, [("open", "old", MONDAY - timedelta(days=61)), ("open", "kept", MONDAY - timedelta(days=59))])
    record = RoutineRecord(path, now=Now(MONDAY))
    assert [i["target"] for i in record.list()] == ["kept"]  # read: already without it
    assert record.purge() == 1
    assert [i["target"] for i in json.loads(path.read_text(encoding="utf-8"))["requests"]] == ["kept"]
    record.note("open outlook", planned=True)  # every write drops what is too old, too
    assert "old" not in path.read_text(encoding="utf-8")


# ── what it adds up to ──


def test_four_of_the_last_ten_weekdays_in_the_band(tmp_path):
    path = tmp_path / "r.json"
    days = weekdays_before(MONDAY, 10)
    seed(path, [("open", "outlook", d.replace(hour=9 if i % 2 else 10)) for i, d in enumerate(days[:3])])
    record = RoutineRecord(path, now=Now(MONDAY))
    assert record.routines() == []  # three days: not yet
    seed(path, [("open", "outlook", d.replace(hour=9)) for d in days[:4]])
    assert record.routines() == [{"kind": "open", "target": "outlook", "hour": 9, "days": 4}]


def test_often_at_any_hour_is_not_a_routine(tmp_path):
    path = tmp_path / "r.json"
    days = weekdays_before(MONDAY, 10)
    seed(path, [("play", "music", d.replace(hour=h)) for d in days for h in (9, 13, 17, 21)])
    assert RoutineRecord(path, now=Now(MONDAY)).routines() == []


def test_one_offer_a_day_with_why(tmp_path):
    path = tmp_path / "r.json"
    days = weekdays_before(MONDAY, 10)
    seed(path, [(k, t, d.replace(hour=9)) for d in days[:5] for k, t in (("open", "outlook"), ("day", ""))])
    record = RoutineRecord(path, now=Now(MONDAY))
    offers = record.offers(MONDAY)
    assert len(offers) == 1
    assert offers[0].why == "You've asked for this around nine on five days."
    assert offers[0].text.endswith("around nine. Want me to do it now?")
    record.note("open outlook", planned=True)
    record.note("good morning", planned=False)
    assert record.offers(MONDAY) == []  # asked for already today
    assert record.offers(MONDAY.replace(hour=14)) == []  # not its hour

    clock = [1_000_000.0]
    picker = Offers(
        mode=lambda: "on", profile_empty=lambda: False, moments=lambda now: RoutineRecord(path, now=Now(now)).offers(now),
        now=lambda: MONDAY.replace(day=17), clock=lambda: clock[0],
    )  # fmt: skip
    first = picker.next_offer()
    picker.mark_offered(first)
    clock[0] += 4 * 3600
    assert picker.next_offer() is None or picker.next_offer().kind != "learned_routine"  # one a day from this source


def test_the_threshold_from_a_simulation():
    """Sparse, varied use (about 7 requests a day from about 60 kinds, at any hour) and one
    real routine (asked at nine on 70% of weekdays): over 50 simulated months the real one is found every time and
    false routines stay under one a month on average."""
    start = datetime(2026, 9, 1)
    falses, found = 0, 0
    for run in range(50):
        rnd = random.Random(run)
        weights = [1 / (i + 1) for i in range(62)]
        items = []
        for n in range(30):
            day = start + timedelta(days=n)
            if day.weekday() < 5 and rnd.random() < 0.7:
                items.append({"kind": "open", "target": "outlook", "day": day.date().isoformat(), "hour": 9})
            for _ in range(rnd.randint(0, 14)):
                k = rnd.choices(range(62), weights)[0]
                items.append(
                    {"kind": "open", "target": f"t{k}", "day": day.date().isoformat(), "hour": rnd.randint(8, 22)}
                )
        record = RoutineRecord(Path("unused"), now=Now(start + timedelta(days=30, hours=8)))
        record._load = lambda items=items: items  # type: ignore[method-assign]
        result = record.routines()
        falses += sum(1 for r in result if r["target"] != "outlook")
        found += any(r["target"] == "outlook" for r in result)
    assert found == 50 and falses / 50 < 1.0, (found, falses / 50)


# ── seen, forgotten, stopped: by voice through the real handler, and in the window ──


@pytest.fixture
def server(tmp_path):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._user_profile = UserProfile(tmp_path / "profile.json")
    record = RoutineRecord(
        tmp_path / "routines.json", enabled=lambda: server.config.memory.learn_routines, now=Now(MONDAY)
    )

    async def stop():
        await server._handle_update_config({"section": "memory", "values": {"learn_routines": False}}, None)

    record.stop = stop
    server._routines = record
    server._user_profile.routines = record
    seed(tmp_path / "routines.json", [("open", "outlook", d.replace(hour=9)) for d in weekdays_before(MONDAY, 4)])
    return server


@pytest.mark.asyncio
async def test_seen_by_voice_and_in_about_me(server):
    assert await server._control_reply("what do you know about my routines") == (
        "From what you ask me: you ask me to open outlook around nine."
    )
    await server._control_reply("call me Sam")
    about = await server._control_reply("what do you know about me")
    assert about.endswith("From what you ask me: you ask me to open outlook around nine.")


@pytest.mark.asyncio
async def test_forget_my_routines(server):
    reply = await server._control_reply("forget my routines")
    assert reply.startswith("Done, I've forgotten ") and "routines" in reply
    assert server._routines.list() == []


@pytest.mark.asyncio
async def test_stop_learning_my_routines(server):
    reply = await server._control_reply("stop learning my routines")
    assert reply.startswith("Okay, I've stopped learning your routines")
    assert server.config.memory.learn_routines is False and server._routines.list() == []
    assert not server._routines.note("open outlook", planned=True)  # nothing more is kept
    assert server._phrase_reserved("stop learning my routines")


@pytest.mark.asyncio
async def test_a_taken_request_is_noted_and_an_unknown_one_is_not(server):
    server._routines.forget()
    assert await server._control_reply("good morning") is not None  # taken by the day's handler: noted
    assert [(i["kind"], i["target"]) for i in server._routines.list()] == [("day", "")]
    assert await server._control_reply("tell me a joke about cats") is None  # no handler took it: nothing kept
    assert len(server._routines.list()) == 1


def test_in_your_data_the_backup_and_removal(tmp_path):
    assert STORES["routines"][1] == "routines.json" and "routines.json" in BACKUP_FILES
    (tmp_path / "routines.json").write_text('{"requests": []}', encoding="utf-8")
    clear_store(tmp_path, "routines")
    assert not (tmp_path / "routines.json").exists()


def test_never_put_into_a_prompt():
    """No module that builds what a model is sent reads the routine record."""
    for path in (DAEMON / "pilot" / "conversation").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert "pilot.personal.routines" not in names, path
        assert "routines.json" not in path.read_text(encoding="utf-8"), path
    context = (DAEMON / "pilot" / "personal" / "context.py").read_text(encoding="utf-8")
    assert "routines" not in context


def test_no_record_no_noise():
    """A server without the record (tests, a failed start) notes nothing and raises nothing."""
    from pilot.personal.wiring import note_request

    note_request(SimpleNamespace(), "open outlook", planned=True)
    assert R.NEED == 4 and R.OF_LAST == 10


def test_turning_learning_back_on_in_settings_does_not_ask():
    """Not a widening (pilot.security.guard_changes): on by default, it keeps nothing a fresh install does not keep."""
    from pilot.security.guard_changes import section_widenings

    assert section_widenings("memory", SimpleNamespace(learn_routines=False), {"learn_routines": True}) == []


@pytest.mark.asyncio
async def test_a_routine_offer_never_reaches_a_cloud_models_history():
    from pilot.personal.offers import Offer
    from pilot.personal.wiring import OFFERED_HERE, deliver_offer

    offer = Offer(key="learned:open:outlook", kind="learned_routine", text="You often ask me to open outlook around "
                  "nine. Want me to do it now?", request="open outlook", why="")  # fmt: skip
    for local, kept in ((False, OFFERED_HERE), (True, offer.text)):
        history: list[str] = []
        layer = SimpleNamespace(memory=SimpleNamespace(add_heliox=history.append),
                                _model=lambda local=local: SimpleNamespace(local=local))  # fmt: skip
        server = SimpleNamespace(_broadcast_notification=AsyncMock(), _voice_listener=None, _conversation=layer)
        await deliver_offer(server, offer)
        assert history == [kept]
        assert "outlook" not in history[0] or local
