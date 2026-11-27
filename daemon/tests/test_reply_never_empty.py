"""A reply is never empty, and a failure asking again cannot change is never retried or approved twice.

Each way a plan can end is driven through the daemon's execute handler with a fake planner, executor and check:
done, failed after the repair rounds gave up, denied, cancelled, a repair round that answered in prose (the live
run's approved copy of a missing file, which gave a blank reply), and a step that failed for good (the run's write
into a read-only folder, approved three times). Every reply carries a sentence to show and one to say, and the spoken
one holds no raw system text.
"""

from __future__ import annotations

import json

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams, VerificationResult
from pilot.reply_rules import ear_faults, screen_faults
from pilot.task_reply import STATUS_SENTENCES, with_reply
from tests.test_execute_tracked import _FakePermissionChecker, _FakeWs, _server_ready_for_handle_execute

WORK = r"C:\Users\me\work"
BLOCKED = WORK + r"\readonly\blocked.txt"
MISSING = WORK + r"\notes\does_not_exist.txt"


def _write(path: str = BLOCKED) -> Action:
    return Action(action_type=ActionType.FILE_WRITE, target=path, parameters=FileParams(path=path, content="hello"))


def _copy(source: str = MISSING) -> Action:
    destination = WORK + r"\notes\x.txt"
    return Action(
        action_type=ActionType.FILE_COPY, target=source, parameters=FileParams(path=source, destination=destination)
    )


class _Planner:
    """Returns its plans in turn; the last one again when they run out."""

    def __init__(self, *plans: ActionPlan) -> None:
        self.plans = list(plans)
        self.calls = 0

    async def plan(self, user_input, error_context="", screen_context="", stream_callback=None, **kwargs):
        self.calls += 1
        return self.plans[min(self.calls, len(self.plans)) - 1]


class _Executor:
    """Each step fails with the error given for its action type, or succeeds."""

    def __init__(self, errors: dict[ActionType, str] | None = None) -> None:
        self.errors = errors or {}
        self.runs = 0

    async def execute(self, plan, **kwargs):
        self.runs += 1
        results = []
        for action in plan.actions:
            error = self.errors.get(action.action_type)
            output = "" if error else f"Written 5 bytes to {action.target} (sha256 {'ab12' * 16}, 5 bytes on disk)"
            results.append(ActionResult(action=action, success=error is None, output=output, error=error))
        return results


class _Verifier:
    async def verify(self, plan, results):
        failed = [index for index, result in enumerate(results) if not result.success]
        return VerificationResult(
            passed=not failed,
            failed_actions=failed,
            details=[f"Action {i} ({r.action.action_type.value}): {'FAILED' if not r.success else 'VERIFIED'}"
                     for i, r in enumerate(results)],
            independently_observed_actions=[i for i, r in enumerate(results) if r.success],
        )  # fmt: skip


def _server(planner: _Planner, executor: _Executor, *, confirm: bool = False):
    server = _server_ready_for_handle_execute(executor)
    server._planner = planner
    server._verifier = _Verifier()
    server._permission_checker = _FakePermissionChecker(requires_confirmation=confirm)
    return server


def _ws(server, confirmation: bool | None = None) -> _FakeWs:
    ws = _FakeWs(confirmation=confirmation)
    ws.server = server
    return ws


def _approvals(ws: _FakeWs) -> int:
    return sum(1 for sent in ws.sent if json.loads(sent).get("method") == "confirm_required")


def _assert_said(response: dict) -> None:
    assert str(response.get("message") or "").strip(), response
    assert screen_faults(response["message"]) == [], response["message"]
    assert ear_faults(response["spoken"]) == [], response["spoken"]
    for raw in ("Errno", "WinError", "\\", "sha256"):
        assert raw not in response["spoken"], response["spoken"]


@pytest.mark.asyncio
async def test_done() -> None:
    planner = _Planner(ActionPlan(actions=[_write(WORK + r"\notes\note.txt")], explanation="Write the note."))
    server = _server(planner, _Executor())

    response = await server._handle_execute({"input": "write hello to note.txt"}, _ws(server))

    assert response["status"] == "success"
    assert response["message"] == response["spoken"] == "Wrote 5 bytes to note.txt in notes."
    _assert_said(response)


@pytest.mark.asyncio
async def test_failed_after_the_repair_rounds_gave_up() -> None:
    planner = _Planner(ActionPlan(actions=[_write(WORK + r"\notes\note.txt")], explanation="Write the note."))
    executor = _Executor({ActionType.FILE_WRITE: "Process exited with code 125: {'detail': 'C:\\\\x\\\\y'}"})
    server = _server(planner, executor)

    response = await server._handle_execute({"input": "write hello to the note"}, _ws(server))

    assert response["status"] == "partial_failure"
    assert executor.runs == 1 + server.MAX_RETRIES  # an unknown failure is left to the repair rounds
    # The repair rounds ran and failed the same way: said in one clause after the failure.
    assert response["spoken"] == (
        "I couldn't write note.txt in notes. The details are on screen. I tried again, and it failed the same way."
    )
    assert "Process exited with code 125" in response["message"]
    _assert_said(response)


@pytest.mark.asyncio
async def test_denied() -> None:
    planner = _Planner(ActionPlan(actions=[_write()], explanation="Write the file."))
    executor = _Executor()
    server = _server(planner, executor, confirm=True)

    response = await server._handle_execute({"input": "write hello to blocked.txt"}, _ws(server, confirmation=False))

    assert response["status"] == "cancelled"
    assert executor.runs == 0
    _assert_said(response)


@pytest.mark.asyncio
async def test_cancelled_before_planning() -> None:
    planner = _Planner(ActionPlan(actions=[_write()], explanation="Write the file."))

    class _StopsAtOnce(_Planner):
        async def plan(self, *args, **kwargs):
            server._cancel_event.set()  # a stop that arrives while the plan is being made
            return await super().plan(*args, **kwargs)

    stopping = _StopsAtOnce(*planner.plans)
    executor = _Executor()
    server = _server(stopping, executor)

    response = await server._handle_execute({"input": "write hello to blocked.txt"}, _ws(server))

    assert response["status"] in {"cancelled", "interrupted"}
    assert executor.runs == 0
    _assert_said(response)


@pytest.mark.asyncio
async def test_a_repair_round_that_answers_in_prose_ends_the_task_with_the_failure() -> None:
    # The source is named only by the planner (the request says "the draft"), so the failure is not final and a
    # repair round runs; it answers in prose. The run's reply was blank, its sentence left in "explanation".
    prose = "That file is not in notes. Did you mean draft_v2.txt instead?"
    planner = _Planner(ActionPlan(actions=[_copy()], explanation="Copy it."), ActionPlan(actions=[], explanation=prose))
    executor = _Executor({ActionType.FILE_COPY: f"Source not found: {MISSING}"})
    server = _server(planner, executor)

    response = await server._handle_execute({"input": "copy the draft to x.txt"}, _ws(server))

    assert response["status"] == "partial_failure"
    assert executor.runs == 1 and planner.calls == 2
    assert response["message"].endswith(prose)
    assert "I couldn't find does_not_exist.txt in notes." in response["message"]
    assert response["spoken"] == (
        "I couldn't find does_not_exist.txt in notes. Check the name, or tell me where it is. Did you mean "
        "draft_v2.txt instead?"
    )
    _assert_said(response)


@pytest.mark.asyncio
async def test_a_missing_file_the_request_named_is_final() -> None:
    planner = _Planner(ActionPlan(actions=[_copy()], explanation="Copy it."))
    executor = _Executor({ActionType.FILE_COPY: f"Source not found: {MISSING}"})
    server = _server(planner, executor, confirm=True)
    ws = _ws(server, confirmation=True)

    response = await server._handle_execute({"input": f"Copy {MISSING} to x.txt"}, ws)

    assert response["status"] == "partial_failure"
    assert (executor.runs, planner.calls, _approvals(ws)) == (1, 1, 1)
    assert response["spoken"] == "I couldn't find does_not_exist.txt in notes. Check the name, or tell me where it is."
    _assert_said(response)


@pytest.mark.asyncio
async def test_a_write_into_a_read_only_folder_is_approved_once_and_not_retried() -> None:
    # The run: "[Errno 13] Permission denied: 'C:\\\\Users\\\\...'" spoken whole, after three approval cards.
    planner = _Planner(ActionPlan(actions=[_write()], explanation="Write the file."))
    executor = _Executor({ActionType.FILE_WRITE: f"[Errno 13] Permission denied: {BLOCKED!r}"})
    server = _server(planner, executor, confirm=True)
    ws = _ws(server, confirmation=True)

    response = await server._handle_execute({"input": "write hello into blocked.txt in the readonly folder"}, ws)

    assert response["status"] == "partial_failure"
    assert (executor.runs, planner.calls, _approvals(ws)) == (1, 1, 1)
    assert response["spoken"] == (
        "I couldn't write blocked.txt in readonly: permission was denied. You can change its permissions, or choose "
        "another place."
    )
    _assert_said(response)


@pytest.mark.asyncio
async def test_a_repair_plan_that_repeats_a_step_that_failed_for_good_is_not_put_to_the_person_again() -> None:
    # Two steps fail: the write for good, the copy for a reason a repair could change. The repair plan repeats the
    # write; it is never shown for approval a second time.
    plan = ActionPlan(actions=[_write(), _copy(WORK + r"\notes\a.txt")], explanation="Write and copy.")
    executor = _Executor(
        {
            ActionType.FILE_WRITE: f"[Errno 13] Permission denied: {BLOCKED!r}",
            ActionType.FILE_COPY: "The copy stopped half way",
        }
    )
    server = _server(_Planner(plan), executor, confirm=True)
    ws = _ws(server, confirmation=True)

    response = await server._handle_execute({"input": "write hello to blocked.txt and copy a.txt"}, ws)

    assert response["status"] == "partial_failure"
    assert (executor.runs, _approvals(ws)) == (1, 1)
    assert response["spoken"].startswith("I couldn't write blocked.txt in readonly: permission was denied.")
    _assert_said(response)


@pytest.mark.parametrize("status", [*STATUS_SENTENCES, "", "something_new"])
def test_every_status_with_nothing_to_say_gets_a_sentence(status: str) -> None:
    for response in (
        {"status": status},
        {"status": status, "message": ""},
        {"status": status, "message": '{"success": true, "output": "x"}'},
        {"status": status, "explanation": "", "message": None},
    ):
        final = with_reply(response)
        assert final["message"].strip() and final["spoken"].strip(), (status, response)
        assert ear_faults(final["spoken"]) == []
        assert screen_faults(final["message"]) == []


def test_a_failure_with_no_message_says_its_failed_step() -> None:
    action = _write()
    raw = ActionResult(action=action, success=False, error=f"[WinError 32] being used: {BLOCKED!r}").model_dump()

    final = with_reply({"status": "partial_failure", "results": [raw]})

    assert final["spoken"] == (
        "I couldn't write blocked.txt in readonly: another program is using it. Close that program and ask me again."
    )
    assert final["message"] == final["spoken"]


def test_a_conversational_answer_in_explanation_is_the_message() -> None:
    final = with_reply({"status": "success", "conversational": True, "explanation": r"Saved it to C:\x\y\notes.txt."})
    assert final["message"] == r"Saved it to C:\x\y\notes.txt."
    assert final["spoken"] == "Saved it to notes.txt."
