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


@pytest.mark.parametrize(("name", "label", "words", "methods"), HELIOX_PARTS, ids=[part[0] for part in HELIOX_PARTS])
async def test_a_request_for_a_part_still_starting_gets_a_sentence_then_the_answer(name, label, words, methods):
    server, called = _ready_server_with_nothing_started()
    method = methods[0]
    assert await _ask(server, "ping") == {"status": "ok"}  # everything else answers at once
    reply = await _ask(server, method)
    assert reply == {"status": "starting", "message": words}
    assert words.endswith("Ask again in a moment.") and "Error" not in words
    assert called == ["ping"]  # the part was not touched

    server._late_parts().mark(name, "ready")
    assert await _ask(server, method) == {"status": "ok"}
    assert called == ["ping", method]


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
