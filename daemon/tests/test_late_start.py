"""After ready: the learner loads first and the other parts start one after another. Until then, a request that needs
one of them gets a plain sentence, and the risk gate asks for approval where the learner could have added one."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from pilot.actions import Action, ActionPlan, ActionType, FileParams
from pilot.config import PilotConfig
from pilot.intelligence.experience import ExperienceEvent, ExperienceEventType
from pilot.intelligence.online_learning import VerifiedOnlineLearner
from pilot.server import JsonRpcRequest, PilotServer
from pilot.system.late_start import HELIOX_PARTS, LateStarts, LearnerLoading

LIMIT = 30.0  # failure limit only


async def _ask(server: PilotServer, method: str) -> dict:
    reply = await server._dispatch(JsonRpcRequest(method=method, params={}, id=1), None)
    return json.loads(reply)["result"]


def _ready_server_with_nothing_started() -> tuple[PilotServer, list[str]]:
    """A server whose handlers exist but whose late parts have not started (the moment after ready)."""
    server = PilotServer(PilotConfig())
    called: list[str] = []

    def handler(method):
        async def handle(params, ws):
            called.append(method)
            return {"status": "ok"}

        return handle

    server._handlers = {method: handler(method) for _n, _l, _w, methods in HELIOX_PARTS for method in methods}
    server._handlers["ping"] = handler("ping")
    for name in server._late_parts().parts:  # as initialize does: every part has its start, none has run yet
        if name != "online_learning":
            server._late_parts().set_start(name, lambda: asyncio.sleep(0))
    return server, called


def test_a_part_the_server_never_built_is_not_held_back():
    late = LateStarts()
    late.add("x", "X", "X is still starting.", ("x_status",))
    assert late.ready("x") and late.waiting_reply("x_status") is None  # never given a start: not managed here
    late.set_start("x", lambda: asyncio.sleep(0))
    assert not late.ready("x") and late.waiting_reply("x_status") == {
        "status": "starting",
        "message": "X is still starting.",
    }


STARTED_BY_THE_SERVER = [part for part in HELIOX_PARTS if part[0] != "online_learning"]


@pytest.mark.parametrize(
    ("name", "label", "words", "methods"), STARTED_BY_THE_SERVER, ids=[part[0] for part in STARTED_BY_THE_SERVER]
)
async def test_a_request_for_a_part_not_started_starts_it_out_of_turn_and_gets_the_answer(name, label, words, methods):
    server, called = _ready_server_with_nothing_started()
    method = methods[0]
    assert await _ask(server, "ping") == {"status": "ok"}  # everything else answers at once
    assert await _ask(server, method) == {"status": "ok"}  # started now, not after the parts queued before it
    snapshot = server._late_parts().snapshot()
    assert snapshot[name] == "ready"
    others = {state for other, state in snapshot.items() if other not in (name, "online_learning")}
    assert others == {"waiting"}  # nothing else was started for it (the learner is the server's own)
    assert called == ["ping", method]


async def test_a_part_still_starting_after_the_bound_gets_its_sentence_then_the_answer(monkeypatch):
    """The learner is loaded by the server, not by a start of its own: a request for it waits the bound, then hears
    the sentence; once it is ready, the answer."""
    monkeypatch.setattr("pilot.system.late_start.LATE_PART_WAIT_SECONDS", 0.05)
    server, called = _ready_server_with_nothing_started()
    server._late_parts().mark("online_learning", "starting")
    words = {part[0]: part[2] for part in HELIOX_PARTS}["online_learning"]
    assert await _ask(server, "online_learning_status") == {"status": "starting", "message": words}
    assert words.endswith("Ask again in a moment.") and "Error" not in words
    assert called == []
    server._late_parts().mark("online_learning", "ready")
    assert await _ask(server, "online_learning_status") == {"status": "ok"}


async def test_a_part_that_failed_to_start_says_so_and_heliox_carries_on():
    server, called = _ready_server_with_nothing_started()
    server._late_parts().mark("screen_vision", "failed", "RuntimeError: no display")
    reply = await _ask(server, "screen_context")
    assert reply["status"] == "error"
    assert reply["message"] == (
        "Screen awareness could not start (RuntimeError: no display). Heliox works without it; restarting Heliox "
        "tries again."
    )
    assert called == []


async def test_parts_start_one_after_another_with_a_pause_and_a_failure_does_not_stop_the_rest():
    order: list[str] = []
    paused: list[float] = []

    async def sleep(seconds):
        order.append("pause")
        paused.append(seconds)

    def start(name, fail=False):
        async def run():
            order.append(name)
            if fail:
                raise OSError("disk")

        return run

    late = LateStarts(pause=2.0, sleep=sleep)
    late.add("a", "A", "A is starting.", start=start("a"))
    late.add("b", "B", "B is starting.", start=start("b", fail=True))
    late.add("c", "C", "C is starting.", start=start("c"))
    await late.run()
    assert order == ["a", "pause", "b", "pause", "c"]
    assert paused == [2.0, 2.0]
    assert late.snapshot() == {"a": "ready", "b": "failed", "c": "ready"}


def _plan() -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=ActionType.FILE_WRITE, target="notes.txt", parameters=FileParams(path="notes.txt"))]
    )


@pytest.fixture
def quiet_snapshot(monkeypatch):
    from pilot.security.risk_observation import OsSnapshot

    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(proc_count=100, disk_usage_fraction=0.5, memory_usage_fraction=0.5, disk_path="/"),
    )


async def test_while_the_learner_loads_the_gate_adds_the_approval_and_says_why(quiet_snapshot, tmp_path):
    from pilot.agents.destructive_critic import HEURISTIC_RISK_THRESHOLD
    from pilot.security.risk_gate import LEARNER_LOADING_REASON, RiskGate

    gate = RiskGate(weights_path="/nonexistent.npz")
    gate.set_online_learner(LearnerLoading())
    risk, reasons = gate.evaluate_plan(_plan(), PilotConfig(), source="interactive")
    assert risk >= HEURISTIC_RISK_THRESHOLD  # an approval is asked for
    assert reasons == [LEARNER_LOADING_REASON]
    assert "still loading what it has learned" in reasons[0]
    assert gate.status(enabled=True)["verified_online_transition"] is None

    # Loaded, with nothing learned: the learner adds nothing, as before.
    learner = VerifiedOnlineLearner(tmp_path / "state.json")
    gate.set_online_learner(learner)
    risk, reasons = gate.evaluate_plan(_plan(), PilotConfig(), source="interactive")
    assert LEARNER_LOADING_REASON not in reasons


async def test_proactive_holds_suggestions_until_the_learner_has_loaded():
    from types import SimpleNamespace

    from pilot.agents.proactive import ProactiveSuggestionEngine

    engine = ProactiveSuggestionEngine(screen_vision=None)
    engine.set_online_learner(LearnerLoading())
    current = SimpleNamespace(active_app="Visual Studio Code", active_window_title="server.py", timestamp=0)
    await engine._check_patterns(current)
    assert engine._pending_suggestions == [] and engine._cooldowns == {}  # nothing decided, nothing used up


class _SlowLedger:
    """A ledger with many events whose first page waits for the test: the learner is still loading."""

    def __init__(self):
        self.release = asyncio.Event()
        self.reads = 0
        start = datetime(2026, 1, 5, 9, tzinfo=UTC)
        self.events = [
            ExperienceEvent(
                event_id=f"e{n}",
                sequence=n,
                event_type=ExperienceEventType.OBSERVATION,
                occurred_at=(start + timedelta(minutes=n)).isoformat(),
                schema_version=1,
                session_id="s",
                task_id="",
                user_id="",
                plan_id="",
                action_id="",
                parent_event_id="",
                idempotency_key=f"k{n}",
                source="screen",
                payload={"active_app": "code"},
            )
            for n in range(1, 2_501)
        ]
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    async def list_events(self, *, after_sequence=0, limit=200, **_filters):
        await self.release.wait()
        self.reads += 1
        return self.events[after_sequence : after_sequence + min(limit, 1000)]


async def test_after_ready_the_first_request_is_answered_while_the_learner_still_loads(tmp_path, monkeypatch):
    monkeypatch.setattr("pilot.server.DATA_DIR", tmp_path)
    server, called = _ready_server_with_nothing_started()
    server._experience_ledger = _SlowLedger()
    server._online_learning = LearnerLoading()
    late = server._late_parts()
    late.pause = 0.0
    from pilot.security.risk_gate import get_risk_gate

    gate_learner_before = get_risk_gate()._online_learner
    after_ready = asyncio.create_task(server._after_ready(snapshot_every=3600))
    try:
        await asyncio.sleep(0)
        assert await _ask(server, "ping") == {"status": "ok"}  # answered while the learner loads
        assert (await _ask(server, "online_learning_status"))["status"] == "starting"
        assert not server._online_learning.is_ready
        server._experience_ledger.release.set()
        for _ in range(int(LIMIT / 0.01)):
            if all(state == "ready" for state in late.snapshot().values()):
                break
            await asyncio.sleep(0.01)
        assert late.snapshot() == dict.fromkeys(late.parts, "ready")
        assert isinstance(server._online_learning, VerifiedOnlineLearner) and server._online_learning.is_ready
        assert server._experience_ledger.reads == 3  # two full pages and the short one that ends the replay
        assert get_risk_gate()._online_learner is server._online_learning
    finally:
        after_ready.cancel()
        await asyncio.gather(after_ready, return_exceptions=True)
        get_risk_gate().set_online_learner(gate_learner_before)  # the gate is process-wide


async def test_every_change_of_a_late_part_is_announced_so_the_window_can_ask_again():
    """A panel told "starting" asks again when its part is ready: the server sends each change as ``late_parts``."""
    from pilot.system.late_start import LateStarts

    seen: list[dict[str, str]] = []

    async def no_pause(_seconds):
        return None

    async def fails():
        raise RuntimeError("no device")

    async def starts():
        return None

    late = LateStarts(sleep=no_pause, pause=0.0, on_change=seen.append)
    late.add("one", "One", "One is starting.", ("one_status",), starts)
    late.add("two", "Two", "Two is starting.", ("two_status",), fails)
    await late.run()
    assert seen[0] == {"one": "starting", "two": "waiting"}
    assert seen[-1] == {"one": "ready", "two": "failed"}
    assert {"one": "ready", "two": "starting"} in seen

    server = PilotServer(PilotConfig())
    sent: list[tuple[str, dict]] = []
    server._spawn_notification = lambda method, params: sent.append((method, params))
    server._late_parts().mark("neural", "ready")
    assert sent == [("late_parts", {"late_parts": server._late_parts().snapshot()})]
    assert sent[0][1]["late_parts"]["neural"] == "ready"


# ── on demand: a request moves its part to the front, within a bound ──


def _slow_part(late: LateStarts, name: str, seconds: float, starts: list[str]) -> None:
    async def start():
        starts.append(name)
        await asyncio.sleep(seconds)

    late.add(name, name.title(), f"{name} is still starting.", (f"{name}_status",), start)


async def test_on_demand_a_request_starts_its_part_and_waits_for_it_within_the_bound():
    late, starts = LateStarts(), []
    _slow_part(late, "screen", 0.05, starts)
    loop = asyncio.get_running_loop()
    began = loop.time()
    assert await late.ready_for("screen_status", 2.0) is None  # go ahead: it is ready
    waited = loop.time() - began
    assert starts == ["screen"] and late.snapshot() == {"screen": "ready"}
    assert 0.04 <= waited < 1.0  # as long as the part took, not the bound


async def test_the_bound_answers_a_slow_part_with_its_sentence_and_the_part_carries_on():
    late, starts = LateStarts(), []
    _slow_part(late, "neural", 0.3, starts)
    loop = asyncio.get_running_loop()
    began = loop.time()
    assert await late.ready_for("neural_status", 0.05) == {"status": "starting", "message": "neural is still starting."}
    assert loop.time() - began < 0.25  # the bound, not the part's own time
    assert late.snapshot() == {"neural": "starting"}  # the request giving up does not stop the part
    await asyncio.sleep(0.4)
    assert late.snapshot() == {"neural": "ready"}
    assert await late.ready_for("neural_status", 0.05) is None
    assert starts == ["neural"]


async def test_two_requests_for_the_same_part_at_once_share_one_start():
    late, starts = LateStarts(), []
    _slow_part(late, "mesh", 0.05, starts)
    first, second = await asyncio.gather(late.ready_for("mesh_status", 2.0), late.ready_for("mesh_status", 2.0))
    assert first is None and second is None
    assert starts == ["mesh"]


async def test_the_queue_does_not_start_again_a_part_a_request_started():
    late, starts = LateStarts(pause=0.0), []
    for name in ("a", "b", "c"):
        _slow_part(late, name, 0.01, starts)
    assert await late.ready_for("c_status", 2.0) is None  # c asked for first
    await late.run()
    assert starts == ["c", "a", "b"] and set(late.snapshot().values()) == {"ready"}


async def test_the_parts_few_need_early_wait_while_a_request_runs():
    from pilot.system import late_start

    busy = {"now": True}
    late, starts = LateStarts(pause=0.0, idle_pause=0.0, busy=lambda: busy["now"]), []
    _slow_part(late, "screen_vision", 0.0, starts)
    _slow_part(late, "neural", 0.0, starts)
    assert "neural" in late_start.IDLE_PARTS and "screen_vision" not in late_start.IDLE_PARTS
    run = asyncio.create_task(late.run())
    await asyncio.sleep(0.2)
    assert starts == ["screen_vision"]  # neural waits: a request is running
    busy["now"] = False
    await run  # the queue ends once the daemon is idle
    assert starts == ["screen_vision", "neural"]


async def test_planning_starts_what_it_reads_together_within_one_bound(monkeypatch):
    monkeypatch.setattr("pilot.system.late_start.PLAN_PART_WAIT_SECONDS", 1.0)
    server, _called = _ready_server_with_nothing_started()
    started: list[str] = []
    for name in ("screen_vision", "subconscious", "strategy_evolution"):

        async def start(name=name):
            started.append(name)
            await asyncio.sleep(0.1)

        server._late_parts().set_start(name, start)
    loop = asyncio.get_running_loop()
    began = loop.time()
    await server._ensure_plan_parts("screen_vision", "subconscious", "strategy_evolution")
    assert sorted(started) == ["screen_vision", "strategy_evolution", "subconscious"]
    assert loop.time() - began < 0.5  # together, not one after another
    snapshot = server._late_parts().snapshot()
    assert all(snapshot[name] == "ready" for name in started)
    assert snapshot["neural"] == "waiting"  # what the plan does not read is left to the queue


def test_the_order_is_by_use():
    names = [part[0] for part in HELIOX_PARTS]
    assert names[:4] == ["online_learning", "screen_vision", "subconscious", "strategy_evolution"]
    assert names[-3:] == ["neural", "agent_mesh", "evolution_harness"]
