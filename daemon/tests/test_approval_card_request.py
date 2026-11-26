"""Every approval card carries what the person asked, so a step that does something else is plain to see.

In the live run "Delete note_backup.txt" showed a card with one file_copy step and nothing saying what was asked. The
card now gets "request": the stored, readable request with any token URL masked, one line, cut at a word. A repair
round's card shows the original request; a background job's card (a trigger's request) shows the job's goal. Fakes
only: the executor, the verifier and the window are stand-ins."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams, VerificationResult
from pilot.config import PilotConfig
from pilot.server import APPROVAL_REQUEST_CHARS, PilotServer, approval_request_line
from tests.test_execute_tracked import _FakePermissionChecker, _FakeWs, _server_ready_for_handle_execute

W = "C:\\w"


def test_the_line_is_the_readable_request_masked_and_cut_at_a_word() -> None:
    assert approval_request_line("  Delete   C:\\w\\a.txt\n") == "Delete C:\\w\\a.txt"
    masked = approval_request_line("Post it to https://hooks.slack.com/services/T000/B000/abcdefSECRET123")
    assert "abcdefSECRET123" not in masked
    long = approval_request_line("Copy " + "every file " * 60 + "to the backup")
    assert len(long) <= APPROVAL_REQUEST_CHARS and long.endswith("…") and not long.endswith(" …")
    assert approval_request_line("") == ""


@pytest.mark.asyncio
async def test_the_first_card_and_a_repair_rounds_card_both_show_the_original_request() -> None:
    request_text = f"Delete {W}\\archive\\note_backup.txt"
    target = f"{W}\\archive\\note_backup.txt"
    plan = ActionPlan(
        actions=[Action(action_type=ActionType.FILE_DELETE, target=target, parameters=FileParams(path=target))],
        explanation="Delete the file.",
    )

    class _Executor:
        runs = 0

        async def execute(self, plan, **kwargs):
            _Executor.runs += 1
            return [ActionResult(action=a, success=False, error="The file is in use") for a in plan.actions]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(
                passed=False, details=["Action 0 (file_delete): FAILED \u2014 The file is in use"], failed_actions=[0]
            )

    server = _server_ready_for_handle_execute(_Executor(), plan)
    server._permission_checker = _FakePermissionChecker(requires_confirmation=True)
    server._verifier = _Verifier()
    ws = _FakeWs(confirmation=True)
    ws.server = server

    await server._handle_execute({"input": request_text}, ws)

    cards = [json.loads(m)["params"] for m in ws.sent if '"method": "confirm_required"' in m]
    assert len(cards) >= 2 and _Executor.runs >= 2
    assert {card["request"] for card in cards} == {request_text}


@pytest.mark.asyncio
async def test_a_background_jobs_card_shows_the_jobs_request() -> None:
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    target = f"{W}\\trigger\\fired.log"
    plan = ActionPlan(
        actions=[Action(action_type=ActionType.FILE_DELETE, target=target, parameters=FileParams(path=target))]
    )
    job = SimpleNamespace(
        job_id="job-1", source="scheduled", goal=f"Append the line 'seen a change' to {W}\\trigger\\fired.log"
    )
    task = asyncio.create_task(server._wait_for_autonomous_confirmation(job, plan, "trigger-plan"))
    for _ in range(20):
        if server._broadcast_notification.await_count:
            break
        await asyncio.sleep(0)
    method, payload = server._broadcast_notification.await_args.args
    assert method == "confirm_required"
    assert payload["request"] == f"Append the line 'seen a change' to {W}\\trigger\\fired.log"
    await server._handle_confirm({"plan_id": "trigger-plan", "confirmed": False}, SimpleNamespace())
    assert await task is False
