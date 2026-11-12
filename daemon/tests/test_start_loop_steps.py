"""From process start to every late part started, how much of Heliox's own work any one step of the event loop does:
a real start-up (initialize, ready, the learner's replay and the late parts). After ready no step of Heliox's may use
more than AFTER_READY_BOUND of the processor: those first seconds are when a person speaks first, and imports of the
learner's and neural input's libraries used to hold the loop for over a second. Steps before ready are printed, not
asserted: while starting, ping and startup_status wait for them (docs/RELIABILITY.md).

What is measured, and why: every callback and task step the loop runs is timed with time.thread_time (the processor
time of the loop's own thread) as well as wall time. Wall time alone failed whenever the machine was busy: a process
that is not scheduled for 0.28 s makes any step, even asyncio's own self-reading handle, look slow. Only steps that
run Heliox's code count: a task whose coroutines are in the pilot package, or a callback defined there. Each is named
by where its coroutines stopped after the step (the step's code is what ran just before that) and by the late part
that was starting, so a failure says whose code it was. Waiting on a disk inside a step uses no processor time and is
not caught here; the non-blocking CI step "Event-loop steps over 100 ms" (tests/loop_watch.py, wall time) still
prints those.

The learner replays a ledger of SEEDED_EVENTS made for the test (two full pages and a part page), so its replay runs
on every machine. Every folder Heliox writes to is a temporary one: each path constant under the real config, data,
state and runtime folders, in every loaded pilot module, is pointed into tmp_path. (Before, the server's own
EXPERIENCE_DB_FILE still named the real ledger, so the test replayed whatever the machine's ledger held.)"""

from __future__ import annotations

import asyncio
import asyncio.events
import importlib.util
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pilot
import pilot.config as config_module
from pilot.config import PilotConfig
from pilot.server import PilotServer

AFTER_READY_BOUND = 0.1  # seconds of the loop thread's processor time in one step of Heliox's
REPORT = 0.05  # steps over this (processor or wall time) are printed
LIMIT = 600.0  # failure limit only
SEEDED_EVENTS = 2_500
PILOT = str(Path(pilot.__file__).resolve().parent)


def _load(name: str, folder: Path):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Step:
    cpu: float
    wall: float
    code: str  # "" when the step ran no code of Heliox's
    part: str
    after_ready: bool

    def line(self) -> str:
        return f"{self.cpu:.3f} s processor, {self.wall:.3f} s wall [{self.part or '-'}] {self.code or 'not Heliox'}"


def _chain(coro) -> list[str]:
    """Where each coroutine of a task stopped, outermost first, Heliox's frames only."""
    out = []
    while coro is not None:
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None) or getattr(coro, "ag_frame", None)
        if frame is None:
            break
        if str(Path(frame.f_code.co_filename).resolve()).startswith(PILOT):
            out.append(f"{frame.f_code.co_qualname}:{frame.f_lineno}")
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None) or getattr(coro, "ag_await", None)
    return out


def heliox_code(handle) -> str:
    callback = getattr(handle, "_callback", None)
    owner = getattr(callback, "__self__", None)
    if isinstance(owner, asyncio.Task):
        coro = owner.get_coro()
        return " > ".join(_chain(coro)) if coro is not None else ""
    function = getattr(callback, "__func__", callback)
    code = getattr(function, "__code__", None)
    if code is not None and str(Path(code.co_filename).resolve()).startswith(PILOT):
        return f"{code.co_qualname}:{code.co_firstlineno}"
    return ""


class StepMeter:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.active = False
        self.ready = False
        self.part = lambda: ""

    def install(self, monkeypatch) -> None:
        original = asyncio.events.Handle._run
        meter = self

        def run(handle):
            if not meter.active:
                return original(handle)
            cpu, wall = time.thread_time(), time.perf_counter()
            try:
                return original(handle)
            finally:
                cpu, wall = time.thread_time() - cpu, time.perf_counter() - wall
                if cpu >= REPORT or wall >= REPORT:
                    meter.steps.append(
                        Step(cpu, wall, heliox_code(handle), meter.part() if meter.ready else "", meter.ready)
                    )

        monkeypatch.setattr(asyncio.events.Handle, "_run", run)


def point_every_folder_at(monkeypatch, tmp_path: Path) -> Path:
    """Every Path constant under Heliox's real folders, in every loaded pilot module, moved into tmp_path. Returns
    the experience ledger's new path."""
    roots = {
        Path(config_module.CONFIG_DIR): tmp_path / "config",
        Path(config_module.DATA_DIR): tmp_path / "data",
        Path(config_module.STATE_DIR): tmp_path / "state",
        Path(config_module.RUNTIME_DIR): tmp_path / "runtime",
    }
    for folder in roots.values():
        folder.mkdir(parents=True, exist_ok=True)
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "pilot" or name.startswith("pilot.")):
            continue
        for attribute, value in list(vars(module).items()):
            if not isinstance(value, Path):
                continue
            for real, fake in roots.items():
                if value == real or real in value.parents:
                    monkeypatch.setattr(module, attribute, fake / value.relative_to(real))
                    break
    return Path(config_module.EXPERIENCE_DB_FILE)


async def seed(ledger_file: Path, count: int) -> None:
    from pilot.intelligence.experience import ExperienceLedger

    seed_ledger = _load("seed_ledger", Path(__file__).parent / "platform_real")
    ledger = ExperienceLedger(ledger_file)
    await ledger.initialize()  # the daemon's own table
    await ledger.close()

    def insert() -> None:
        with sqlite3.connect(ledger_file) as db:
            db.executemany(
                "INSERT INTO experience_events (event_id, event_type, occurred_at, schema_version, session_id, "
                "task_id, user_id, plan_id, action_id, parent_event_id, idempotency_key, source, payload_json, "
                "provenance_json, confidence, privacy_class) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                seed_ledger.rows(count),
            )

    await asyncio.to_thread(insert)  # the test's own set-up, kept off the loop so the loop watch shows Heliox's steps


@pytest.mark.asyncio
async def test_after_ready_no_step_holds_the_loop_for_long(unused_tcp_port, tmp_path, monkeypatch):
    ledger_file = point_every_folder_at(monkeypatch, tmp_path)
    assert tmp_path in ledger_file.parents, ledger_file
    await seed(ledger_file, SEEDED_EVENTS)
    monkeypatch.setattr("pilot.security.vault.KeyVault.available", property(lambda _self: False))  # no real keyring
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", unused_tcp_port, "test-token"

    meter = StepMeter()
    meter.install(monkeypatch)
    original = PilotServer._begin_after_ready

    def begin_after_ready(self):
        meter.ready = True
        original(self)

    monkeypatch.setattr(PilotServer, "_begin_after_ready", begin_after_ready)
    with (
        patch("pilot.models.router.ModelRouter.initialize", new_callable=AsyncMock),
        patch("pilot.models.cache.LLMCache.initialize", new_callable=AsyncMock),
        patch("pilot.memory.store.MemoryStore.initialize", new_callable=AsyncMock),
        patch("pilot.agents.screen_vision.ScreenVisionAgent.start", new_callable=AsyncMock),
        patch("pilot.cognitive.cognitive_engine.CognitiveEngine.load_model", new_callable=AsyncMock),
        patch("pilot.security.audit.AuditLogger", return_value=MagicMock()),
    ):
        server = PilotServer(config)
        late = server._late_parts()
        meter.part = lambda: next((name for name, state in late.snapshot().items() if state == "starting"), "")
        meter.active = True
        try:
            await asyncio.wait_for(server.start(), LIMIT)
            deadline = time.monotonic() + LIMIT
            while time.monotonic() < deadline and any(
                state not in ("ready", "failed") for state in late.snapshot().values()
            ):
                await asyncio.sleep(0.1)
        finally:
            meter.active = False  # the stop is not part of what is measured
        assert all(state in ("ready", "failed") for state in late.snapshot().values()), late.snapshot()
        replayed = (getattr(server._online_learning, "last_load", None) or {}).get("replayed")
        await server.stop()

    assert replayed is not None and replayed >= SEEDED_EVENTS, replayed  # the replay this test is about ran
    assert meter.ready, "never reached ready"
    for step in sorted(meter.steps, key=lambda step: step.cpu, reverse=True):
        print(f"{'after' if step.after_ready else 'before'} ready: {step.line()}")
    heavy = [step for step in meter.steps if step.after_ready and step.code and step.cpu > AFTER_READY_BOUND]
    assert not heavy, [step.line() for step in heavy]


def test_the_learner_hands_the_loop_back_during_a_replay():
    """Without timing: a replay of two and a half pages yields between every REPLAY_YIELD_EVERY events, so no page is
    learned in one piece, however slow the machine."""
    from pilot.intelligence.online_learning import REPLAY_PAGE, REPLAY_YIELD_EVERY, VerifiedOnlineLearner

    class Ledger:
        def __init__(self, count):
            # Plain objects, made before the loop runs: 2,500 MagicMocks took 0.39 s inside the test's own task on
            # the Windows runner, which the loop watch then reported as a slow step.
            self.events = [SimpleNamespace(sequence=n) for n in range(1, count + 1)]

        async def list_events(self, *, after_sequence=0, limit=REPLAY_PAGE, **_filters):
            return [event for event in self.events if event.sequence > after_sequence][:limit]

    longest = 0
    ledger = Ledger(2 * REPLAY_PAGE + REPLAY_PAGE // 2)

    async def run():
        nonlocal longest
        learner = VerifiedOnlineLearner(Path("unused.json"))
        consumed = 0

        async def consume(_event, *, replaying=False):
            nonlocal consumed
            consumed += 1

        learner.consume = consume
        learner.begin_loading()
        ticks: list[int] = []

        async def ticker():
            while True:
                ticks.append(consumed)
                await asyncio.sleep(0)

        task = asyncio.create_task(ticker())
        await learner.initialize(ledger)
        task.cancel()
        longest = max(later - earlier for earlier, later in zip(ticks, ticks[1:], strict=False))
        return consumed

    assert asyncio.run(run()) == 2 * REPLAY_PAGE + REPLAY_PAGE // 2
    assert longest <= REPLAY_YIELD_EVERY


def test_the_meter_names_heliox_code_and_ignores_asyncio_own_handles():
    """A step of asyncio's own (the proactor's self-reading, a timer that sets a future) is not Heliox's; a task
    running pilot code is, named by where it stopped."""
    from pilot.system.late_start import LateStarts

    async def run():
        loop = asyncio.get_running_loop()
        timer = loop.call_later(10, lambda: None)
        assert heliox_code(timer) == ""
        starts = LateStarts(sleep=asyncio.sleep, pause=0.0)
        gate = asyncio.Event()

        async def part():
            await gate.wait()

        starts.add("demo", "Demo", "", (), part)
        task = asyncio.create_task(starts.run())
        await asyncio.sleep(0)
        names = " > ".join(_chain(task.get_coro()))
        assert names.startswith("LateStarts.run:"), names  # the test's own part() is not Heliox's code
        assert heliox_code(asyncio.events.Handle(task.cancel, (), loop)).startswith("LateStarts.run:")  # its task
        gate.set()
        await task
        timer.cancel()

    asyncio.run(run())
