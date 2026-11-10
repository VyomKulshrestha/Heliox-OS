"""A correction during planning cancels the planning call at once, and the companion review never wastes a call.

Everything here runs on fake planners, executors and models; no model is called.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    FileParams,
    SystemInfoParams,
    VerificationResult,
)
from pilot.agents.execution_companion import ExecutionCompanion
from pilot.server import PilotServer
from tests.test_execute_tracked import _FakePermissionChecker, _FakeWs, _server_ready_for_handle_execute


def _plan(name: str = "notes.txt") -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=ActionType.FILE_READ, target=name, parameters=FileParams(path=name))],
        explanation=f"Read {name}.",
    )


class _Verifier:
    async def verify(self, plan, results):
        return VerificationResult(passed=True, details=["verified"], failed_actions=[])


class _Executor:
    def __init__(self) -> None:
        self.plans: list[str] = []

    async def execute(self, plan, **kwargs):
        self.plans.append(plan.actions[0].target)
        return [ActionResult(action=plan.actions[0], success=True, output="ok")]


class _Planner:
    """First planning call blocks until cancelled (or released); later calls answer at once."""

    def __init__(self, block_calls: int = 1) -> None:
        self.inputs: list[str] = []
        self.cancelled: list[int] = []
        self.entered = asyncio.Event()
        self.entered_count = 0
        self.release = asyncio.Event()
        self.block_calls = block_calls

    async def plan(self, user_input, **kwargs):
        self.inputs.append(user_input)
        call = len(self.inputs)
        self.entered_count = call
        self.entered.set()
        if call <= self.block_calls:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.append(call)
                raise
        return _plan(f"plan{call}.txt")


def _server(planner, executor=None) -> PilotServer:
    server = _server_ready_for_handle_execute(executor or _Executor())
    server._verifier = _Verifier()
    server._planner = planner
    return server


async def _interject(server: PilotServer, text: str) -> dict:
    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        return await server._handle_interject({"input": text}, MagicMock())


def _methods(server: PilotServer) -> list[str]:
    return [method for method, _ in server.test_broadcasts]


def _assert_no_orphans(server: PilotServer) -> None:
    assert server._active_execution_task is None
    assert server._pending_confirms == {}
    assert server._interactive_request_active is False


@pytest.mark.asyncio
async def test_correction_during_planning_cancels_the_call_and_replans_at_once():
    planner = _Planner()
    executor = _Executor()
    server = _server(planner, executor)
    running = asyncio.create_task(server._handle_execute({"input": "back up and ship"}, _FakeWs()))
    await asyncio.wait_for(planner.entered.wait(), 5)

    started = asyncio.get_running_loop().time()
    reply = await _interject(server, "skip the backup")
    result = await asyncio.wait_for(running, 5)
    elapsed = asyncio.get_running_loop().time() - started

    assert reply["status"] == "revising"
    assert planner.cancelled == [1]
    assert len(planner.inputs) == 2
    assert "skip the backup" in planner.inputs[1] and planner.inputs[1].startswith("back up and ship")
    assert executor.plans == ["plan2.txt"]
    assert result["status"] == "success"
    assert elapsed < 2
    assert _methods(server).count("task_complete") == 1
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_correction_while_planning_context_is_gathered_never_starts_the_old_planning_call():
    gate = asyncio.Event()
    entered = asyncio.Event()

    class _Subconscious:
        async def get_persona_context(self):
            entered.set()
            await gate.wait()
            return ""

    planner = _Planner(block_calls=0)
    executor = _Executor()
    server = _server(planner, executor)
    server._subconscious = _Subconscious()
    running = asyncio.create_task(server._handle_execute({"input": "back up and ship"}, _FakeWs()))
    await asyncio.wait_for(entered.wait(), 5)

    await _interject(server, "skip the backup")
    gate.set()
    result = await asyncio.wait_for(running, 5)

    assert len(planner.inputs) == 1
    assert "skip the backup" in planner.inputs[0]
    assert executor.plans == ["plan1.txt"]
    assert result["status"] == "success"
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_correction_before_planning_begins_replans_instead_of_cancelling_the_task():
    gate = asyncio.Event()
    entered = asyncio.Event()
    server = _server(_Planner(block_calls=0))

    async def slow_context(user_input):
        entered.set()
        await gate.wait()
        return ""

    server._reflector.get_improvement_context = slow_context
    planner = server._planner
    running = asyncio.create_task(server._handle_execute({"input": "back up and ship"}, _FakeWs()))
    await asyncio.wait_for(entered.wait(), 5)
    await _interject(server, "skip the backup")
    gate.set()
    result = await asyncio.wait_for(running, 5)

    assert result["status"] == "success"
    assert len(planner.inputs) == 1 and "skip the backup" in planner.inputs[0]
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_correction_while_waiting_for_approval_withdraws_it_and_never_approves():
    first = _plan("risky.txt")
    inputs: list[str] = []

    class _TwoPlans:
        async def plan(self, user_input, **kwargs):
            inputs.append(user_input)
            return first if len(inputs) == 1 else _plan("safe.txt")

    class _Checker(_FakePermissionChecker):
        def plan_requires_confirmation(self, plan):
            return plan.actions[0].target == "risky.txt"

    executor = _Executor()
    server = _server(_TwoPlans(), executor)
    server._permission_checker = _Checker()
    ws = _FakeWs()  # never answers the dialog
    running = asyncio.create_task(server._handle_execute({"input": "do the risky thing"}, ws))
    for _ in range(200):
        if any(json.loads(m).get("method") == "confirm_required" for m in ws.sent):
            break
        await asyncio.sleep(0.01)
    assert server._pending_confirms

    reply = await _interject(server, "do the safe thing instead")
    result = await asyncio.wait_for(running, 5)

    assert reply["status"] == "revising"
    assert executor.plans == ["safe.txt"]
    assert result["status"] == "success"
    assert len(inputs) == 2
    confirms = [m for m in map(json.loads, ws.sent) if m.get("method") == "confirm_required"]
    assert len(confirms) == 1
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_approval_is_not_requested_for_a_plan_the_person_already_corrected_away():
    server = _server(_Planner(block_calls=0))
    server._cancel_event = asyncio.Event()
    server._cancel_event.set()
    ws = _FakeWs()

    confirmed, approved, required = await server._wait_for_confirmation("p1", _plan(), ws)

    assert confirmed is False and approved == set() and required == {0}
    assert ws.sent == []
    assert server._pending_confirms == {}


@pytest.mark.asyncio
async def test_two_quick_corrections_before_the_restart_both_apply_to_one_replan():
    planner = _Planner()
    executor = _Executor()
    server = _server(planner, executor)
    running = asyncio.create_task(server._handle_execute({"input": "back up and ship"}, _FakeWs()))
    await asyncio.wait_for(planner.entered.wait(), 5)

    await _interject(server, "skip the backup")
    await _interject(server, "and use the Desktop")
    result = await asyncio.wait_for(running, 5)

    assert result["status"] == "success"
    assert planner.cancelled == [1]
    assert len(planner.inputs) == 2
    assert "skip the backup" in planner.inputs[1] and "and use the Desktop" in planner.inputs[1]
    assert executor.plans == ["plan2.txt"]
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_second_correction_during_the_replan_cancels_it_and_the_last_one_wins():
    planner = _Planner(block_calls=2)
    executor = _Executor()
    server = _server(planner, executor)
    running = asyncio.create_task(server._handle_execute({"input": "back up and ship"}, _FakeWs()))
    await asyncio.wait_for(planner.entered.wait(), 5)
    await _interject(server, "skip the backup")

    for _ in range(200):
        if planner.entered_count >= 2:
            break
        await asyncio.sleep(0.01)
    assert planner.entered_count == 2
    await _interject(server, "actually only copy the post")
    result = await asyncio.wait_for(running, 5)

    assert result["status"] == "success"
    assert planner.cancelled == [1, 2]
    assert len(planner.inputs) == 3
    assert "only copy the post" in planner.inputs[2]
    assert executor.plans == ["plan3.txt"]
    assert _methods(server).count("task_complete") == 1
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_correction_in_the_gap_between_two_revisions_is_not_lost():
    planner = _Planner(block_calls=1)
    executor = _Executor()
    server = _server(planner, executor)
    original = server._broadcast_notification

    async def broadcast(method, params):
        await original(method, params)
        if method == "companion_revision_started" and not getattr(server, "_gap_done", False):
            server._gap_done = True
            # The previous revision has handed over and the new one has no cancel event yet.
            await _interject(server, "second thought, keep the screenshots")

    server._broadcast_notification = broadcast
    running = asyncio.create_task(server._handle_execute({"input": "back up and ship"}, _FakeWs()))
    await asyncio.wait_for(planner.entered.wait(), 5)
    await _interject(server, "skip the backup")
    result = await asyncio.wait_for(running, 5)

    assert result["status"] == "success"
    assert "keep the screenshots" in planner.inputs[-1]
    assert executor.plans == [f"plan{len(planner.inputs)}.txt"]
    assert len(executor.plans) == 1
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_correction_as_the_plan_completes_runs_no_action_from_the_old_plan():
    planner = _Planner(block_calls=0)
    executor = _Executor()
    server = _server(planner, executor)
    ws = _FakeWs()
    sent = ws.send
    fired = False

    async def send(message):
        nonlocal fired
        await sent(message)
        if json.loads(message).get("method") == "plan_preview" and not fired:
            fired = True
            await _interject(server, "skip the backup")

    ws.send = send
    result = await asyncio.wait_for(server._handle_execute({"input": "back up and ship"}, ws), 5)

    assert result["status"] == "success"
    assert len(planner.inputs) == 2
    assert "skip the backup" in planner.inputs[1]
    assert executor.plans == ["plan2.txt"]
    previews = [m for m in map(json.loads, ws.sent) if m.get("method") == "plan_preview"]
    assert len(previews) == 2
    assert _methods(server).count("task_complete") == 1
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_correction_during_execution_still_replans_once():
    class _Hang:
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()
            self.plans: list[str] = []

        async def execute(self, plan, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await asyncio.Event().wait()
            self.plans.append(plan.actions[0].target)
            return [ActionResult(action=plan.actions[0], success=True, output="ok")]

    executor = _Hang()
    planner = _Planner(block_calls=0)
    server = _server(planner, executor)
    running = asyncio.create_task(server._handle_execute({"input": "back up and ship"}, _FakeWs()))
    await asyncio.wait_for(executor.started.wait(), 5)
    await _interject(server, "skip the backup")
    result = await asyncio.wait_for(running, 5)

    assert result["status"] == "success"
    assert len(planner.inputs) == 2
    assert executor.plans == ["plan2.txt"]
    _assert_no_orphans(server)


@pytest.mark.asyncio
async def test_stop_before_a_phase_registers_cancels_it_without_replanning():
    server = _server(_Planner(block_calls=0))
    server._cancel_event = asyncio.Event()
    server._cancel_event.set()
    started = False

    async def work():
        nonlocal started
        started = True
        return "ran"

    with pytest.raises(asyncio.CancelledError):
        await server._await_execution_tracked(work())

    assert started is False
    assert server._active_execution_task is None


# -- companion review latency --


def _review_plan() -> ActionPlan:
    return _plan()


class _Model:
    def __init__(self, delay: float = 0.0, reply: str = '{"decision":"CONTINUE","reason":"Aligned."}') -> None:
        self.delay = delay
        self.reply = reply
        self.calls = 0
        self.cancelled = 0

    async def generate(self, prompt, **kwargs):
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return self.reply


def _companion(model, timeout: float = 0.05, clock=None) -> ExecutionCompanion:
    companion = ExecutionCompanion(model, **({"clock": clock} if clock else {}))
    companion._timeout_seconds = timeout
    return companion


@pytest.mark.asyncio
async def test_fast_provider_review_arrives_and_says_so():
    model = _Model(delay=0.0, reply='{"decision":"WARN","reason":"Check the path."}')
    review = await _companion(model).review("read notes", _review_plan())

    assert review.decision == "WARN"
    assert review.reviewed is True and review.status == "reviewed"
    payload = review.to_dict()
    assert payload["reviewed"] is True and payload["latency_seconds"] is not None


@pytest.mark.asyncio
async def test_slow_provider_stops_being_called_after_two_timeouts_and_the_event_says_so():
    now = [0.0]
    model = _Model(delay=10)
    companion = _companion(model, clock=lambda: now[0])

    first = await companion.review("read notes", _review_plan())
    second = await companion.review("read notes", _review_plan())
    third = await companion.review("read notes", _review_plan())

    assert model.calls == 2
    assert model.cancelled == 2
    assert [first.status, second.status, third.status] == ["unavailable", "unavailable", "skipped_slow"]
    for review in (first, second, third):
        assert review.decision == "CONTINUE"
        assert review.reviewed is False and review.to_dict()["reviewed"] is False
    assert "skipped" in third.reason


@pytest.mark.asyncio
async def test_skipped_review_is_probed_again_after_the_cooldown_and_recovers():
    now = [0.0]
    model = _Model(delay=10)
    companion = _companion(model, clock=lambda: now[0])
    await companion.review("read notes", _review_plan())
    await companion.review("read notes", _review_plan())
    assert (await companion.review("read notes", _review_plan())).status == "skipped_slow"

    now[0] += 601
    model.delay = 0
    probe = await companion.review("read notes", _review_plan())
    after = await companion.review("read notes", _review_plan())

    assert probe.status == "reviewed" and after.status == "reviewed"
    assert model.calls == 4


@pytest.mark.asyncio
async def test_a_review_cancelled_by_a_correction_does_not_count_as_a_slow_provider():
    model = _Model(delay=10)
    companion = _companion(model, timeout=30)
    for _ in range(3):
        task = asyncio.create_task(companion.review("read notes", _review_plan()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    model.delay = 0
    review = await companion.review("read notes", _review_plan())

    assert review.status == "reviewed"
    assert model.calls == 4


@pytest.mark.asyncio
async def test_the_ui_event_reports_a_skipped_review_and_the_task_still_runs_without_a_model_call():
    now = [0.0]
    model = _Model(delay=10)
    companion = _companion(model, clock=lambda: now[0])
    executor = _Executor()
    server = _server(_Planner(block_calls=0), executor)
    server._execution_companion = companion

    reviews: list[dict] = []
    for _ in range(3):
        ws = _FakeWs()
        result = await asyncio.wait_for(server._handle_execute({"input": "read notes"}, ws), 10)
        assert result["status"] == "success"
        reviews.append(
            next(m["params"] for m in map(json.loads, ws.sent) if m.get("method") == "companion_plan_review")
        )

    assert [r["status"] for r in reviews] == ["unavailable", "unavailable", "skipped_slow"]
    assert all(r["reviewed"] is False for r in reviews)
    assert all(r["decision"] == "CONTINUE" for r in reviews)
    assert model.calls == 2
    assert len(executor.plans) == 3


@pytest.mark.asyncio
async def test_the_ui_event_reports_a_real_review_from_a_fast_provider():
    model = _Model(delay=0.0)
    server = _server(_Planner(block_calls=0))
    server._execution_companion = _companion(model)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "read notes"}, ws), 10)

    event = next(m["params"] for m in map(json.loads, ws.sent) if m.get("method") == "companion_plan_review")
    assert result["status"] == "success"
    assert event["reviewed"] is True and event["status"] == "reviewed"
    assert model.calls == 1


@pytest.mark.asyncio
async def test_a_deterministic_local_review_reports_that_no_model_was_used():
    plan = ActionPlan(
        actions=[Action(action_type=ActionType.SYSTEM_INFO, target="system", parameters=SystemInfoParams())],
        explanation="os",
    )
    model = _Model()
    server = _server_ready_for_handle_execute(_Executor(), plan)
    server._verifier = _Verifier()
    server._execution_companion = _companion(model)
    ws = _FakeWs()

    await asyncio.wait_for(server._handle_execute({"input": "show system information"}, ws), 10)

    event = next(m["params"] for m in map(json.loads, ws.sent) if m.get("method") == "companion_plan_review")
    assert event["status"] == "local" and event["reviewed"] is True
    assert model.calls == 0
