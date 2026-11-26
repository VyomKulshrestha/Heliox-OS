"""The companion's review holds up only a plan that could hurt and would run without a card.

Fake planners, executors, verifiers and models; the companion's clock is moved only by its own polls. Nothing is
written, run or sent.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    FileParams,
    KeyboardParams,
    VerificationResult,
)
from pilot.agents.execution_companion import ExecutionCompanion
from pilot.agents.review_timing import (
    BEFORE_LIMIT_SECONDS,
    UNANSWERED_REASON,
    ReviewTiming,
    review_timing,
    step_cannot_hurt,
)
from tests.test_execute_tracked import _FakePermissionChecker, _FakeWs, _server_ready_for_handle_execute

NOTE = os.path.abspath(os.path.join("work", "timing", "note_timing.txt"))


def _write() -> Action:
    return Action(
        action_type=ActionType.FILE_WRITE, target=NOTE, parameters=FileParams(path=NOTE, content="timing check.")
    )


def _delete() -> Action:
    return Action(action_type=ActionType.FILE_DELETE, target=NOTE, parameters=FileParams(path=NOTE))


def _read() -> Action:
    return Action(action_type=ActionType.FILE_READ, target=NOTE, parameters=FileParams(path=NOTE))


def _type() -> Action:
    return Action(action_type=ActionType.KEYBOARD_TYPE, target="Notepad", parameters=KeyboardParams(text="hello"))


def _plan(*actions: Action) -> ActionPlan:
    return ActionPlan(actions=list(actions), explanation="plan")


# ── the rule, from what the plan carries ──


def test_reads_and_undoable_local_changes_cannot_hurt():
    assert step_cannot_hurt(_read(), undo_enabled=True)
    assert step_cannot_hurt(_write(), undo_enabled=True)
    assert review_timing(_plan(_read(), _write()), card_anyway=False, undo_enabled=True) is ReviewTiming.BESIDE


def test_a_delete_typing_into_another_app_or_a_write_without_undo_waits_for_the_review():
    assert review_timing(_plan(_delete()), card_anyway=False, undo_enabled=True) is ReviewTiming.BEFORE
    assert review_timing(_plan(_read(), _type()), card_anyway=False, undo_enabled=True) is ReviewTiming.BEFORE
    assert review_timing(_plan(_write()), card_anyway=False, undo_enabled=False) is ReviewTiming.BEFORE


def test_a_plan_with_a_card_is_reviewed_while_the_card_is_open():
    assert review_timing(_plan(_write()), card_anyway=True, undo_enabled=True) is ReviewTiming.WITH_CARD


# ── fakes ──


class _GatedModel:
    """Answers only when released: the order of events is decided by the test, not by a clock."""

    def __init__(self, reply: str = '{"decision": "CONTINUE", "reason": "Matches the request."}') -> None:
        self.reply = reply
        self.calls = 0
        self.answered = 0
        self.release = asyncio.Event()

    async def generate(self, prompt, **kwargs):
        self.calls += 1
        await self.release.wait()
        self.answered += 1
        return self.reply


class _NeverAnswers:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt, **kwargs):
        self.calls += 1
        await asyncio.Event().wait()


class _PolledClock:
    """The companion's limit clock, moved on only by its own polls."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.waited = 0.0

    def __call__(self) -> float:
        return self.now

    async def wait(self, tasks, timeout=None):
        self.now += timeout
        self.waited += timeout
        await asyncio.sleep(0)
        done = {task for task in tasks if task.done()}
        return done, set(tasks) - done


class _Executor:
    def __init__(self) -> None:
        self.ran = 0

    async def execute(self, plan, **kwargs):
        self.ran += 1
        return [ActionResult(action=action, success=True, output="ok") for action in plan.actions]


class _Verifier:
    async def verify(self, plan, results):
        return VerificationResult(passed=True, independently_observed_actions=list(range(len(results))))


def _server(plan: ActionPlan, companion: ExecutionCompanion, executor: _Executor, *, card: bool = False):
    server = _server_ready_for_handle_execute(executor, plan)
    server._verifier = _Verifier()
    server._execution_companion = companion
    server._permission_checker = _FakePermissionChecker(requires_confirmation=card)
    return server


def _sent(ws: _FakeWs, method: str) -> list[dict]:
    return [m["params"] for m in map(json.loads, ws.sent) if m.get("method") == method]


# ── a note: runs at once, reviewed beside ──


@pytest.mark.asyncio
async def test_a_note_runs_and_replies_before_its_review_answers_and_a_concern_comes_after_as_a_remark():
    model = _GatedModel(reply='{"decision": "WARN", "reason": "The note went to a folder you did not name."}')
    executor = _Executor()
    server = _server(_plan(_write()), ExecutionCompanion(model), executor)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "write a note"}, ws), 10)

    assert result["status"] == "success" and executor.ran == 1
    assert model.calls == 1 and model.answered == 0  # the reply went out while the review was still waiting
    assert _sent(ws, "companion_plan_review") == []

    model.release.set()
    await asyncio.gather(*list(server._companion_follow_up_tasks), return_exceptions=True)

    review = _sent(ws, "companion_plan_review")
    assert len(review) == 1 and review[0]["beside_run"] is True and review[0]["decision"] == "WARN"
    remark = [params for method, params in server.test_broadcasts if method == "companion_plan_intervention"]
    assert len(remark) == 1
    # Not "WARN", "REVISE" or "STOP": the window shows it as a remark and nothing speaks a stop for a finished task.
    assert remark[0]["decision"] == "REMARK" and remark[0]["original_decision"] == "WARN"


# ── a risky plan without a card: waits, short limit, a card when unanswered ──


@pytest.mark.asyncio
async def test_a_risky_plan_whose_review_does_not_answer_asks_the_person_first(monkeypatch):
    clock = _PolledClock()
    model = _NeverAnswers()
    companion = ExecutionCompanion(model, wait=clock.wait, limit_clock=clock)
    executor = _Executor()
    server = _server(_plan(_type()), companion, executor)
    asked: list[dict] = []

    async def _confirm(plan_id, plan, ws, **kwargs):
        asked.append(kwargs)
        return False, set(), set(range(len(plan.actions)))

    monkeypatch.setattr(server, "_wait_for_confirmation", _confirm)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "type hello in notepad"}, ws), 10)

    assert clock.waited == pytest.approx(BEFORE_LIMIT_SECONDS)  # its short limit, counted by its own polls
    assert len(asked) == 1 and UNANSWERED_REASON in asked[0]["reason"]
    assert asked[0]["force_all_actions"] is True  # no step needed approval by itself: all of them are asked
    assert executor.ran == 0 and result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_a_risky_plan_whose_review_answers_runs_without_a_card(monkeypatch):
    model = _GatedModel()
    model.release.set()
    executor = _Executor()
    server = _server(_plan(_type()), ExecutionCompanion(model), executor)

    async def _confirm(*args, **kwargs):
        raise AssertionError("no card for a reviewed plan that needs none")

    monkeypatch.setattr(server, "_wait_for_confirmation", _confirm)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "type hello in notepad"}, ws), 10)

    assert result["status"] == "success" and executor.ran == 1
    assert _sent(ws, "companion_plan_review")[0]["status"] == "reviewed"


# ── a plan with a card: reviewed while the card is open ──


@pytest.mark.asyncio
async def test_the_card_opens_before_the_review_answers_and_a_stop_still_holds_after_approval(monkeypatch):
    model = _GatedModel(reply='{"decision": "STOP", "reason": "This deletes a file you asked to keep."}')
    executor = _Executor()
    server = _server(_plan(_delete()), ExecutionCompanion(model), executor, card=True)
    seen_on_card: list[int] = []

    async def _confirm(plan_id, plan, ws, **kwargs):
        seen_on_card.append(model.answered)
        server._pending_confirms[plan_id] = object()  # the card is open while the review answers
        model.release.set()
        for _ in range(500):  # counted turns of the loop until the concern is shown beside the card
            if any(method == "companion_plan_intervention" for method, _params in server.test_broadcasts):
                break
            await asyncio.sleep(0)
        server._pending_confirms.pop(plan_id, None)
        return True, {0}, {0}

    monkeypatch.setattr(server, "_wait_for_confirmation", _confirm)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "delete the note"}, ws), 10)

    assert seen_on_card == [0]  # the card did not wait for the review
    concern = [params for method, params in server.test_broadcasts if method == "companion_plan_intervention"]
    assert concern and concern[0]["while_approval_open"] is True
    assert result["status"] == "blocked_by_companion" and executor.ran == 0


@pytest.mark.asyncio
async def test_a_denied_card_cancels_its_review(monkeypatch):
    model = _GatedModel()
    executor = _Executor()
    server = _server(_plan(_delete()), ExecutionCompanion(model), executor, card=True)

    async def _confirm(plan_id, plan, ws, **kwargs):
        await asyncio.sleep(0)
        return False, set(), {0}

    monkeypatch.setattr(server, "_wait_for_confirmation", _confirm)

    result = await asyncio.wait_for(server._handle_execute({"input": "delete the note"}, _FakeWs()), 10)

    assert result["status"] == "cancelled" and executor.ran == 0
    assert model.answered == 0 and all(task.done() for task in server._companion_follow_up_tasks)


# ── the limit's message ──


@pytest.mark.asyncio
async def test_a_review_that_only_rode_along_says_so_instead_of_naming_the_configured_limit(caplog):
    clock = _PolledClock()
    companion = ExecutionCompanion(_NeverAnswers(), timeout_seconds=5.0, wait=clock.wait, limit_clock=clock)
    first = await companion.review("type hello", _plan(_type()), others_running=lambda: False)
    assert first.status == "unavailable"

    others_until = clock.now + 0.1
    with caplog.at_level("WARNING", logger="pilot.agents.execution_companion"):
        second = await companion.review("type hello", _plan(_type()), others_running=lambda: clock() < others_until)

    assert second.status == "unavailable"
    message = caplog.records[-1].getMessage()
    assert "had not answered when the other plan reviews finished, after 0.1 s" in message
    assert "limit 5 s" not in message
