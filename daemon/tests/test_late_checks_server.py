"""The daemon replies within the verification budget and follows the rest: a late outcome reaches the receipt and
the person, never a repair round, and a stop cancels it. Fakes only: the executor and the check are stand-ins."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams
from pilot.agents import postconditions
from pilot.agents import verifier as verifier_module
from pilot.agents.late_checks import LateReport, LateVerifications
from pilot.agents.postconditions import Outcome, mismatch, register
from pilot.agents.verifier import Verifier
from tests.test_execute_tracked import _FakeWs, _server_ready_for_handle_execute


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry = dict(postconditions._REGISTRY)
    monkeypatch.setattr(postconditions, "_REGISTRY", registry)
    return registry


class _Check:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.cancelled = False

    async def __call__(self, _result: ActionResult) -> Outcome:
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return mismatch("Copy destination content mismatch")


class _Executor:
    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, plan, **kwargs):
        self.runs += 1
        return [ActionResult(action=action, success=True, output="Copied the video") for action in plan.actions]


class _Sink:
    def __init__(self) -> None:
        self.recorded: list[LateReport] = []
        self.told: list[str] = []
        self.done = asyncio.Event()

    async def record(self, report: LateReport) -> None:
        self.recorded.append(report)
        self.done.set()

    async def tell(self, sentence: str, report: LateReport, speak: bool) -> None:
        if speak:
            self.told.append(sentence)


def _setup(monkeypatch: pytest.MonkeyPatch):
    check = _Check()
    postconditions._REGISTRY.pop(ActionType.FILE_COPY, None)
    register(ActionType.FILE_COPY, method="test_postcondition")(check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())

    async def budget_spent_once_started(task: asyncio.Task, _budget: float) -> None:
        waiter = asyncio.ensure_future(check.started.wait())
        await asyncio.wait({waiter, task}, return_when=asyncio.FIRST_COMPLETED)
        waiter.cancel()

    monkeypatch.setattr(Verifier, "_wait_budget", staticmethod(budget_spent_once_started))
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_COPY,
                target="video.mp4",
                parameters=FileParams(path="video.mp4", destination="backup/video.mp4"),
            )
        ],
        explanation="Copy the video.",
    )
    executor = _Executor()
    server = _server_ready_for_handle_execute(executor, plan)
    server._verifier = Verifier(MagicMock())
    sink = _Sink()
    server._late_verifications = LateVerifications(record=sink.record, tell=sink.tell)
    return server, executor, check, sink


@pytest.mark.asyncio
async def test_the_reply_goes_out_while_a_slow_check_runs_and_its_outcome_is_kept_later(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    server, executor, check, sink = _setup(monkeypatch)

    response = await server._handle_execute({"input": "copy the video to backup"}, _FakeWs())

    assert response["status"] == "success"
    assert response["message"].endswith("Note: 1 step is still being checked; details in the receipt.")
    assert response["verification"]["pending_actions"] == [0]
    assert check.calls == 1 and not check.release.is_set()

    check.release.set()
    await sink.done.wait()
    await server._late_verifications.stop()

    # A copy changes nothing for other people: its late mismatch is kept, not said.
    assert sink.told == []
    assert sink.recorded[0].checks[0]["kind"] == "mismatch"
    # The late mismatch only informed: the plan ran once and was not checked or repaired again.
    assert executor.runs == 1
    assert check.calls == 1


@pytest.mark.asyncio
async def test_a_stop_cancels_a_check_still_running_behind_the_reply(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    server, _executor, check, sink = _setup(monkeypatch)
    await server._handle_execute({"input": "copy the video to backup"}, _FakeWs())
    assert server._late_verifications.running >= 1

    await server._late_verifications.stop()

    assert check.cancelled is True
    assert server._late_verifications.running == 0
    assert sink.recorded == [] and sink.told == []


def test_the_daemon_stops_late_checks_when_it_stops() -> None:
    import inspect

    from pilot.server import PilotServer

    source = inspect.getsource(PilotServer.stop)
    assert 'self.__dict__.get("_late_verifications")' in source
    assert "late_verifications.stop()" in source


@pytest.mark.asyncio
async def test_a_late_mismatch_is_shown_and_said_as_a_watcher_notice_is() -> None:
    from pilot.agents.late_checks import FollowUp

    server = _server_ready_for_handle_execute(None)
    said: list[tuple[str, str]] = []

    async def speak(text, channel=None, dedupe_key=""):
        said.append((text, str(channel)))

    server._speak_companion_text = speak
    report = LateReport(FollowUp(task_key="t", plan_id="p", request="email Sam", receipt_id="r1"), checks=[])

    await server._tell_late_check("I checked the mail I sent: its copy in Sent is not what was sent.", report, True)
    await server._tell_late_check("2 packages were not upgraded: a, b", report, False)

    methods = [(method, params["text"], params["spoken"]) for method, params in server.test_broadcasts]
    assert methods == [
        ("verification_late", "I checked the mail I sent: its copy in Sent is not what was sent.", True),
        ("verification_late", "2 packages were not upgraded: a, b", False),
    ]
    assert said == [("I checked the mail I sent: its copy in Sent is not what was sent.", "task_narration")]


@pytest.mark.asyncio
async def test_a_late_outcome_is_written_to_a_receipt_that_follows_the_task_receipt() -> None:
    from pilot.agents.late_checks import FollowUp

    server = _server_ready_for_handle_execute(None)
    written: list[dict] = []

    class _Receipts:
        async def record_late_checks(self, task_key, **fields):
            written.append({"task_key": task_key, **fields})

    server._task_receipts = _Receipts()
    checks = [{"step": 1, "action_type": "file_copy", "kind": "verified", "reason": "same", "late": True}]
    report = LateReport(FollowUp(task_key="t", plan_id="p", request="copy", receipt_id="r1"), checks=checks)

    await server._record_late_checks(report)

    assert written == [
        {
            "task_key": "t",
            "request": "copy",
            "plan_id": "p",
            "follows": "r1",
            "summary": "Checked after the reply: 1 verified.",
            "checks": checks,
        }
    ]
