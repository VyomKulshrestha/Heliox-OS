"""What a person reads about a task is their request and their corrections, never Heliox's planner prompt."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, VerificationResult
from pilot.agents.execution_companion import CompanionReview
from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.task_request import TaskRequest, readable_request, split_corrections

LAUNCH = (
    "Get my launch folder on the Desktop ready to ship. Back up my C drive to C:\\Backup first. Then put the final "
    "post and the screenshots into a new folder called launch-release on the Desktop, delete the old drafts and the "
    "build junk, and zip the release."
)
# The planner prompt Heliox built for a live correction before 11 November 2026, as stored in old entries.
OLD_REVISION = (
    f"{LAUNCH}\n\n[LIVE USER CORRECTION]\nSkip the backup.\n\n[ALREADY OBSERVED OUTCOMES]\n"
    "- No action completed before the correction arrived.\n\nRevise the plan to honor the live correction. "
    "Do not repeat a completed action unless the correction explicitly requires it."
)


# ---------------------------------------------------------------- the request itself


def test_a_request_without_corrections_reads_as_typed():
    request = TaskRequest(LAUNCH)
    assert request.display() == LAUNCH
    assert request.for_reviewers() == LAUNCH


def test_corrections_read_as_corrections_in_order():
    request = TaskRequest(LAUNCH).with_corrections("Skip the backup.")
    request = request.with_corrections("keep the drafts\nAdditional correction: Zip it as release.zip")
    assert request.display() == (f"{LAUNCH} Then: Skip the backup. Then: keep the drafts. Then: Zip it as release.zip.")
    assert "[" not in request.display() and "Revise the plan" not in request.display()


def test_reviewers_get_the_correction_and_what_already_ran_in_plain_sentences():
    request = TaskRequest(LAUNCH).with_corrections("Skip the backup.", ["file_copy on C:\\a.md: completed"])
    assert request.for_reviewers() == (
        f"{LAUNCH}\nThe person then said: Skip the backup.\n"
        "Already done before the correction: file_copy on C:\\a.md: completed."
    )
    nothing_ran = TaskRequest(LAUNCH).with_corrections("Skip the backup.")
    assert nothing_ran.for_reviewers().endswith("Already done before the correction: nothing.")


def test_split_corrections_keeps_each_one():
    assert split_corrections("a\nAdditional correction: b\n  Additional correction:  c ") == ["a", "b", "c"]


# ---------------------------------------------------------------- entries written before the split


def test_old_entries_are_shown_with_the_correction_and_without_heliox_headings():
    assert readable_request(OLD_REVISION) == f"{LAUNCH} Then: Skip the backup."


def test_old_entries_with_two_corrections_or_a_flattened_prompt_read_the_same_way():
    nested = (
        f"{OLD_REVISION}\n\n[LIVE USER CORRECTION]\nKeep the drafts.\n\n[ALREADY OBSERVED OUTCOMES]\n"
        "- file_copy on x: completed\n\nRevise the plan to honor the live correction."
    )
    assert readable_request(nested) == f"{LAUNCH} Then: Skip the backup. Then: Keep the drafts."
    flattened = " ".join(OLD_REVISION.split())  # as some views store and show it
    assert readable_request(flattened) == f"{LAUNCH} Then: Skip the backup."


def test_old_companion_revisions_drop_heliox_feedback_and_plain_text_is_untouched():
    companion = (
        "Read my notes\n\n[INDEPENDENT COMPANION REVIEW]\nProblem: too many steps\nRequired correction: one read\n\n"
        "Create a smaller corrected plan that still follows the original user request."
    )
    assert readable_request(companion) == "Read my notes"
    assert readable_request("Rename report.txt [draft]") == "Rename report.txt [draft]"
    assert readable_request("") == ""


# ---------------------------------------------------------------- every place a task's request is kept or shown


def _plan() -> ActionPlan:
    write = Action.model_validate(
        {
            "action_type": "file_write",
            "target": "C:/n/todo.txt",
            "parameters": {"path": "C:/n/todo.txt", "content": "x"},
        }
    )
    return ActionPlan(actions=[write], explanation="write the note")


def _server(seen: dict[str, Any]) -> PilotServer:
    config = PilotConfig()
    config.security.dry_run = False
    config.typed_decision.enabled = False
    config.narration.proactive_review_enabled = True
    server = PilotServer(config)
    plans = [_plan(), _plan()]

    async def plan(user_input, **kwargs):
        seen.setdefault("planned", []).append(user_input)
        if seen.get("correct_during_planning") and len(plans) == 2:
            server._live_correction = seen["correct_during_planning"]
            server._cancel_event.set()
        return plans.pop()

    server._planner = SimpleNamespace(plan=plan)
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = MagicMock()
    server._permission_checker.plan_requires_confirmation.return_value = False
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    server._task_receipts = AsyncMock()
    server._task_receipts.finalize.return_value = None
    server._checkpoint_store = AsyncMock()
    server._orchestrator = None
    server._control_reply = AsyncMock(return_value=None)  # not a timeline question

    async def history(**kwargs):
        seen.setdefault("history", []).append(kwargs["raw_input"])

    server._record_plan_history = history

    class _Companion:
        async def review(self, user_input, plan, *, screen_context="", others_running=None):
            seen.setdefault("companion", []).append(user_input)
            return CompanionReview(decision="CONTINUE", reason="ok")

    server._execution_companion = _Companion()

    async def execute(run_plan: ActionPlan, **kwargs: Any) -> list[Any]:
        return [ActionResult(action=action, success=True, output="ok") for action in run_plan.actions]

    server._executor = SimpleNamespace(execute=execute)
    return server


async def _run(seen: dict[str, Any]) -> PilotServer:
    server = _server(seen)
    ws = MagicMock()
    ws.send = AsyncMock()
    await server._handle_execute({"input": LAUNCH, "session_id": "s"}, ws)
    await asyncio.sleep(0.05)  # memory and reflection are recorded in background tasks
    return server


def _kept(server: PilotServer, seen: dict[str, Any]) -> dict[str, Any]:
    return {
        "activity log and planner history (memory)": [c.args[0] for c in server._memory.record.await_args_list],
        "receipt (Today view, daily brief, exports)": [
            c.kwargs["request"] for c in server._task_receipts.finalize.await_args_list
        ],
        "checkpoint": [c.args[1] for c in server._checkpoint_store.start_plan.await_args_list],
        "plan history": seen.get("history", []),
        "reflection": [c.args[0] for c in server._reflector.reflect.await_args_list],
        "lessons looked up": [c.args[0] for c in server._reflector.get_improvement_context.await_args_list],
    }


@pytest.mark.asyncio
async def test_without_a_correction_every_record_holds_the_request_as_typed():
    seen: dict[str, Any] = {}
    server = await _run(seen)
    for place, values in _kept(server, seen).items():
        assert values and set(values) == {LAUNCH}, place
    assert seen["companion"] == [LAUNCH]


@pytest.mark.asyncio
async def test_after_a_correction_every_record_holds_the_request_and_the_correction_never_the_prompt():
    seen: dict[str, Any] = {"correct_during_planning": "Skip the backup."}
    server = await _run(seen)

    # The planner still gets its revision prompt; nothing a person reads does.
    assert "[LIVE USER CORRECTION]" in seen["planned"][-1]
    for place, values in _kept(server, seen).items():
        assert values, place
        assert values[-1] == f"{LAUNCH} Then: Skip the backup.", place
        assert not any("[LIVE USER CORRECTION]" in value or "Revise the plan" in value for value in values), place
    assert seen["companion"][-1] == (
        f"{LAUNCH}\nThe person then said: Skip the backup.\nAlready done before the correction: nothing."
    )


@pytest.mark.asyncio
async def test_old_history_rows_are_read_back_without_the_planner_prompt(tmp_path, monkeypatch):
    from pilot.memory import store as store_module

    rows = [(1, "2026-11-10T07:50:28", OLD_REVISION, 1, "Skipping the backup as you asked.")]

    class _Cursor:
        async def fetchall(self):
            return rows

        async def close(self):
            return None

    class _Db:
        async def execute(self, *args):
            return _Cursor()

    class _Read:
        async def __aenter__(self):
            return _Db()

        async def __aexit__(self, *exc):
            return False

    memory = store_module.MemoryStore.__new__(store_module.MemoryStore)
    memory._pool = SimpleNamespace(read=lambda: _Read())
    [entry] = await memory.get_history(limit=1)
    assert entry["user_input"] == f"{LAUNCH} Then: Skip the backup."


def test_old_receipts_are_listed_without_the_planner_prompt():
    from pilot.workflows.task_receipts import TaskReceipts

    short = TaskReceipts._short(
        {
            "receipt": {
                "receipt_id": "ab",
                "task_key": "t",
                "finished_at": "2026-11-10T07:50:28+00:00",
                "status": "success",
                "summary": "done",
                "request": OLD_REVISION,
                "source": "text",
                "actions": [],
            },
            "signature": "sig",
        }
    )
    assert short["request"] == f"{LAUNCH} Then: Skip the backup."[:200]
