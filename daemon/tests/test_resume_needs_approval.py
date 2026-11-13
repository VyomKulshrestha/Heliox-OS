"""Resuming a plan never stands in for approving it.

Until 13 November 2026 resume_plan ran any checkpoint's remaining actions for real with no approval: a plan the
person denied, a dry run, or one saved before its approval step. A durable plan was guarded only by a parameter the
client could send itself ("_authorized_task_id"). The window never called resume_plan (it resumes through
resume_task), so no person could do this from the screen; any program holding the window's token could.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, NotifyParams, VerificationResult
from pilot.config import PilotConfig
from pilot.server import JsonRpcRequest, PilotServer
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.durable_tasks import DurableTaskStore, TaskStatus


def _plan(*targets: str) -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(action_type=ActionType.NOTIFY, target=t, parameters=NotifyParams(summary=t, body=t)) for t in targets
        ],
        explanation="test plan",
        raw_input="do it",
    )


class Recorder:
    """The executor: what ran, in order."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def execute(self, plan: ActionPlan, **kwargs: Any) -> list[ActionResult]:
        results = []
        for action in plan.actions:
            if "on_action_start" in kwargs:
                await kwargs["on_action_start"](action)
            self.ran.append(action.target)
            result = ActionResult(action=action, success=True, output=action.target)
            if "on_action_complete" in kwargs:
                await kwargs["on_action_complete"](result)
            results.append(result)
        return results


class Person:
    """The approval dialog: what it was shown, and the answer it gives."""

    def __init__(self, confirmed: bool, approve: set[int] | None = None) -> None:
        self.confirmed = confirmed
        self.approve = approve
        self.shown: list[list[str]] = []

    async def __call__(self, plan_id: str, plan: ActionPlan, ws: Any, **kwargs: Any):
        self.shown.append([action.target for action in plan.actions])
        required = set(range(len(plan.actions)))
        approved = required if self.approve is None else self.approve
        return self.confirmed, (approved if self.confirmed else set()), required


def _server(checkpoints: WorkflowCheckpointStore, person: Person, durable=None) -> tuple[PilotServer, Recorder]:
    config = PilotConfig()
    config.security.dry_run = False
    server = PilotServer(config)
    server._checkpoint_store = checkpoints
    server._durable_tasks = durable
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    server._wait_for_confirmation = person  # type: ignore[method-assign]
    recorder = Recorder()
    server._executor = recorder
    return server, recorder


def _ws() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


def _through_execute(server: PilotServer, plan: ActionPlan, *, dry_run: bool = False) -> PilotServer:
    """The parts of the real execute pipeline the plan path needs, with the plan given."""
    server.config.security.dry_run = dry_run
    server._planner = AsyncMock()
    server._planner.plan.return_value = plan
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = MagicMock()
    server._permission_checker.plan_requires_confirmation.return_value = True
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()
    server._orchestrator = None
    return server


@pytest.fixture
def checkpoints(tmp_path) -> WorkflowCheckpointStore:
    return WorkflowCheckpointStore(tmp_path / "checkpoints.db")


@pytest.mark.asyncio
async def test_a_plan_the_person_denied_is_never_resumed(checkpoints):
    server, recorder = _server(checkpoints, Person(confirmed=False))
    _through_execute(server, _plan("delete-folder"))
    denied = await server._handle_execute({"input": "do it", "session_id": "s"}, AsyncMock())
    plan_id = next(iter(await _all_plan_ids(checkpoints)))

    resumed = await server._handle_resume_plan({"plan_id": plan_id}, _ws())

    assert denied["status"] == "cancelled"
    assert (await checkpoints.get(plan_id)).approval == "denied"
    assert resumed["status"] == "error" and "turned this plan down" in resumed["message"]
    assert recorder.ran == []


@pytest.mark.asyncio
async def test_a_dry_run_asks_before_its_checkpoint_runs_for_real(checkpoints):
    person = Person(confirmed=True)
    server, recorder = _server(checkpoints, person)
    _through_execute(server, _plan("write-file"), dry_run=True)
    await server._handle_execute({"input": "do it", "session_id": "s"}, AsyncMock())
    plan_id = next(iter(await _all_plan_ids(checkpoints)))
    checkpoint = await checkpoints.get(plan_id)
    assert checkpoint.approval == "preview"
    # Whatever the preview recorded, a real run of what is left is asked for again.
    await checkpoints.replace_plan(plan_id, _plan("write-file", "send-mail"))
    shown_before = len(person.shown)
    recorder.ran.clear()

    resumed = await server._handle_resume_plan({"plan_id": plan_id}, _ws())

    assert len(person.shown) == shown_before + 1
    assert resumed["status"] in {"success", "partial_failure"}
    assert (await checkpoints.get(plan_id)).approval == "approved"


@pytest.mark.asyncio
async def test_a_checkpoint_saved_before_its_approval_asks_and_runs_only_what_is_approved(checkpoints):
    person = Person(confirmed=True, approve={1})
    server, recorder = _server(checkpoints, person)
    plan = _plan("a", "b", "delete-folder")
    await checkpoints.start_plan("plan-1", "do it", plan)  # the daemon stopped before the approval was answered
    await checkpoints.record_result("plan-1", ActionResult(action=plan.actions[0], success=True, output="a"))

    resumed = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert person.shown == [["b", "delete-folder"]]  # every remaining action shown
    assert recorder.ran == ["delete-folder"]  # only the one the person checked
    assert [a.target for a in (await checkpoints.get("plan-1")).plan.actions] == ["a", "delete-folder"]
    assert resumed["status"] in {"success", "partial_failure"}


@pytest.mark.asyncio
async def test_denying_a_resume_runs_nothing_and_marks_the_plan_denied(checkpoints):
    server, recorder = _server(checkpoints, Person(confirmed=False))
    await checkpoints.start_plan("plan-1", "do it", _plan("a"))

    resumed = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert resumed["status"] == "cancelled" and recorder.ran == []
    assert (await checkpoints.get("plan-1")).approval == "denied"


@pytest.mark.asyncio
async def test_an_approved_plan_that_was_interrupted_resumes_without_asking_again(checkpoints):
    person = Person(confirmed=True)
    server, recorder = _server(checkpoints, person)
    plan = _plan("a", "b")
    await checkpoints.start_plan("plan-1", "do it", plan)
    await checkpoints.mark_approval("plan-1", "approved")
    await checkpoints.record_result("plan-1", ActionResult(action=plan.actions[0], success=True, output="a"))

    resumed = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert person.shown == [] and recorder.ran == ["b"]
    assert resumed["status"] == "success"


@pytest.mark.asyncio
async def test_a_checkpoint_from_before_approvals_were_recorded_asks(tmp_path):
    import aiosqlite

    db = tmp_path / "old.db"
    async with aiosqlite.connect(db) as conn:  # the table as it was before 13 November 2026
        await conn.execute(
            "CREATE TABLE workflow_checkpoints (plan_id TEXT PRIMARY KEY, user_input TEXT NOT NULL, plan_json TEXT "
            "NOT NULL, results_json TEXT NOT NULL, completed_count INTEGER NOT NULL, last_output TEXT NOT NULL, "
            "snapshot_ids_json TEXT NOT NULL, status TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        await conn.execute(
            "INSERT INTO workflow_checkpoints VALUES (?, ?, ?, '[]', 0, '', '[]', 'running', 'then')",
            ("old-plan", "do it", _plan("a").model_dump_json()),
        )
        await conn.commit()
    person = Person(confirmed=False)
    server, recorder = _server(WorkflowCheckpointStore(db), person)

    resumed = await server._handle_resume_plan({"plan_id": "old-plan"}, _ws())

    assert person.shown == [["a"]] and recorder.ran == [] and resumed["status"] == "cancelled"


def test_no_client_can_send_a_parameter_that_starts_with_an_underscore():
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "resume_plan",
            "params": {"plan_id": "p", "_authorized_task_id": "task-1", "_interaction_id": "x", "_quick_plan": {}},
            "id": 1,
        }
    )
    assert JsonRpcRequest.parse(raw).params == {"plan_id": "p"}


@pytest.mark.asyncio
async def test_a_durable_plan_is_not_resumed_by_naming_its_task(checkpoints, tmp_path):
    durable = DurableTaskStore(tmp_path / "durable.db")
    await durable.initialize()
    try:
        await durable.create_task(task_id="task-1", session_id="s", user_input="do it")
        await durable.transition("task-1", TaskStatus.PLANNING)
        await durable.transition("task-1", TaskStatus.AWAITING_APPROVAL, plan_id="plan-1")
        await checkpoints.start_plan("plan-1", "do it", _plan("delete-folder"))
        await checkpoints.mark_approval("plan-1", "approved")
        server, recorder = _server(checkpoints, Person(confirmed=True), durable)
        request = JsonRpcRequest.parse(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "resume_plan",
                    "params": {"plan_id": "plan-1", "_authorized_task_id": "task-1"},
                    "id": 1,
                }
            )
        )

        resumed = await server._handle_resume_plan(request.params, _ws())

        assert resumed["status"] == "error" and "resume_task" in resumed["message"]
        assert recorder.ran == []
    finally:
        await durable.close()


async def _all_plan_ids(checkpoints: WorkflowCheckpointStore) -> list[str]:
    async with checkpoints._connect() as db:
        cursor = await db.execute("SELECT plan_id FROM workflow_checkpoints")
        rows = await cursor.fetchall()
    return [row[0] for row in rows]
