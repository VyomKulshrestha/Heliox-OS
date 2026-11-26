"""In a session of several turns the planner plans the current request, and code refuses a plan for another one.

The live run of 26 November: with all turns in one session the planner answered in prose and its retry planned an
earlier turn ("Delete note_backup.txt" became a copy of the previous turn's missing file), and a repair round copied
to the previous turn's destination. The causes were the earlier turns sent as chat turns, a history summary sent as a
second system message that a provider keeping one system text put in place of the planner prompt, and a retry that
pasted the prose back. Fakes only: a scripted model, a memory with recorded turns, a fake executor."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams, VerificationResult
from pilot.agents import request_match
from pilot.agents.planner import EARLIER_TURNS_HEADING, Planner
from pilot.models import cloud

W = "C:\\w"


def plan_json(*actions: dict[str, Any], explanation: str = "Plan.") -> str:
    return json.dumps({"explanation": explanation, "actions": list(actions)})


def copy_json(src: str, dst: str) -> dict[str, Any]:
    return {"action_type": "file_copy", "target": src, "parameters": {"path": src, "destination": dst}}


def delete_json(path: str) -> dict[str, Any]:
    return {"action_type": "file_delete", "target": path, "parameters": {"path": path}}


class ScriptedModel:
    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.sent: list[list[dict[str, Any]]] = []

    async def generate(self, messages, **_kwargs) -> str:
        self.sent.append([dict(m) for m in messages])
        return self.answers.pop(0)


class SessionMemory:
    def __init__(self, turns: list[tuple[str, str]]) -> None:
        self.turns = turns  # oldest first: (request, explanation)

    async def get_context(self, *_args, **_kwargs) -> str:
        return ""

    async def get_history(self, *, limit: int = 50, **_kwargs) -> list[dict[str, Any]]:
        rows = [
            {"id": i, "timestamp": "", "user_input": r, "success": False, "explanation": e}
            for i, (r, e) in enumerate(self.turns)
        ]
        return list(reversed(rows))[:limit]


EARLIER = [
    (f"Copy {W}\\big\\big300.bin to {W}\\big\\big300_copy.bin", "Copy big300.bin to big300_copy.bin."),
    (f"Copy {W}\\notes\\does_not_exist.txt to {W}\\notes\\x.txt", "Copy does_not_exist.txt to x.txt."),
]
DELETE = f"Delete {W}\\archive\\note_backup.txt"


@pytest.mark.asyncio
async def test_earlier_turns_are_background_in_one_request_message_after_one_system_message() -> None:
    model = ScriptedModel(plan_json(delete_json(f"{W}\\archive\\note_backup.txt")))
    plan = await Planner(model, SessionMemory(EARLIER)).plan(DELETE, session_id="s1")  # type: ignore[arg-type]

    assert plan.error is None and plan.actions[0].action_type == ActionType.FILE_DELETE
    [sent] = model.sent
    assert [m["role"] for m in sent] == ["system", "user"]
    request_message = sent[1]["content"]
    assert EARLIER_TURNS_HEADING in request_message
    assert "1. The person asked: Copy" in request_message and "does_not_exist.txt" in request_message
    # The current request comes after the background, and is the last thing the model reads.
    assert request_message.index("Current user request (the task to fulfill): " + DELETE) > request_message.index(
        EARLIER_TURNS_HEADING
    )
    assert "Plan this request and nothing else." in request_message


@pytest.mark.asyncio
async def test_a_retry_after_prose_carries_the_same_request_and_not_the_prose() -> None:
    prose = "I can't run the copy from here, so I haven't changed any files."
    model = ScriptedModel(prose, plan_json(delete_json(f"{W}\\archive\\note_backup.txt")))
    plan = await Planner(model, SessionMemory(EARLIER)).plan(DELETE, session_id="s1")  # type: ignore[arg-type]

    assert plan.error is None
    retry = model.sent[1][1]["content"]
    assert "it was prose, not the JSON object" in retry
    assert prose not in retry
    assert "Current user request (the task to fulfill): " + DELETE in retry
    assert EARLIER_TURNS_HEADING in retry


@pytest.mark.asyncio
async def test_a_plan_for_an_earlier_turn_is_planned_again_once_with_the_reason() -> None:
    wrong = plan_json(copy_json(f"{W}\\notes\\does_not_exist.txt", f"{W}\\notes\\x.txt"))
    right = plan_json(delete_json(f"{W}\\archive\\note_backup.txt"))
    model = ScriptedModel(wrong, right)
    plan = await Planner(model, SessionMemory(EARLIER)).plan(DELETE, session_id="s1")  # type: ignore[arg-type]

    assert plan.error is None and plan.actions[0].action_type == ActionType.FILE_DELETE
    second = model.sent[1][1]["content"]
    assert second.startswith("Your previous plan was not used: Step 1 would copy does_not_exist.txt to x.txt")
    assert "Current user request (the task to fulfill): " + DELETE in second


@pytest.mark.asyncio
async def test_a_second_plan_for_an_earlier_turn_fails_honestly_and_nothing_is_planned() -> None:
    wrong = plan_json(copy_json(f"{W}\\notes\\does_not_exist.txt", f"{W}\\notes\\x.txt"))
    model = ScriptedModel(wrong, wrong)
    plan = await Planner(model, SessionMemory(EARLIER)).plan(DELETE, session_id="s1")  # type: ignore[arg-type]

    assert plan.actions == []
    assert plan.error == (
        "I stopped before doing anything: the plan I made would copy does_not_exist.txt to x.txt, which is not what "
        "you asked. Nothing was changed. Please say it again, naming the file or folder."
    )
    assert len(model.sent) == 2


@pytest.mark.asyncio
async def test_a_repair_round_shows_the_failed_steps_and_refuses_an_earlier_turns_destination() -> None:
    request_text = f"Copy {W}\\big\\big300.bin to {W}\\big\\big300_tampered.bin"
    failed = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_COPY,
                target=f"{W}\\big\\big300.bin",
                parameters=FileParams(path=f"{W}\\big\\big300.bin", destination=f"{W}\\big\\big300_tampered.bin"),
            )
        ]
    )
    model = ScriptedModel(plan_json(copy_json(f"{W}\\big\\big300.bin", f"{W}\\big\\big300_copy.bin")))
    plan = await Planner(model, SessionMemory(EARLIER)).plan(  # type: ignore[arg-type]
        request_text,
        error_context="Action 0 (file_copy): MISMATCH \u2014 Copy destination content mismatch",
        session_id="s1",
        repair_of=failed,
    )

    assert plan.error == (
        "I did not try again: the retry would copy big300.bin to big300_copy.bin, which this request did not name."
    )
    assert len(model.sent) == 1  # a repair gets one plan
    sent = model.sent[0][1]["content"]
    assert f"1. file_copy {W}\\big\\big300.bin -> {W}\\big\\big300_tampered.bin" in sent
    assert "This is a repair of the current request only." in sent


@pytest.mark.asyncio
async def test_a_log_tail_planned_with_code_is_planned_once_more_as_a_read() -> None:
    log = f"{W}\\logs\\app.log"
    with_code = plan_json(
        {"action_type": "file_read", "target": log, "parameters": {"path": log}},
        {
            "action_type": "code_execute",
            "target": "",
            "parameters": {"code": "lines = PREV_OUTPUT.splitlines()\nprint('\\n'.join(lines[-5:]))"},
            "use_previous_output": True,
        },
    )
    typed = plan_json({"action_type": "file_read", "target": log, "parameters": {"path": log, "tail_lines": 5}})
    model = ScriptedModel(with_code, typed)
    plan = await Planner(model, SessionMemory([])).plan(f"Show me the last 5 lines of the log file {log}")  # type: ignore[arg-type]

    assert [a.action_type for a in plan.actions] == [ActionType.FILE_READ]
    assert plan.actions[0].parameters.tail_lines == 5
    assert "tail_lines" in model.sent[1][1]["content"]


@pytest.mark.asyncio
async def test_the_first_plan_stays_when_the_read_retry_is_no_better() -> None:
    log = f"{W}\\logs\\app.log"
    with_code = plan_json(
        {"action_type": "file_read", "target": log, "parameters": {"path": log}},
        {
            "action_type": "code_execute",
            "target": "",
            "parameters": {"code": "print('\\n'.join(PREV_OUTPUT.splitlines()[-5:]))"},
            "use_previous_output": True,
        },
    )
    model = ScriptedModel(with_code, "not json")
    plan = await Planner(model, SessionMemory([])).plan(f"Show me the last 5 lines of {log}")  # type: ignore[arg-type]
    assert [a.action_type for a in plan.actions] == [ActionType.FILE_READ, ActionType.CODE_EXECUTE]


@pytest.mark.asyncio
async def test_anthropic_and_gemini_get_every_system_message(monkeypatch) -> None:
    payloads: list[dict[str, Any]] = []

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "content": [{"type": "text", "text": "{}"}],
                "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
            }

    async def fake_post(self, provider, model, url, payload, headers):
        payloads.append(payload)
        return _Response()

    monkeypatch.setattr(cloud.CloudClient, "_post", fake_post)
    monkeypatch.setattr(cloud, "_api_json", lambda resp, provider: resp.json())
    client = cloud.CloudClient.__new__(cloud.CloudClient)
    messages = [
        {"role": "system", "content": "PLANNER PROMPT"},
        {"role": "system", "content": "SUMMARY"},
        {"role": "user", "content": "Delete it"},
    ]
    await client._call_anthropic("k", "claude-haiku-5-5", messages, "", 0.1)
    await client._call_gemini_native("k", "gemini-x", messages, "", True, 0.1, max_retries=0)

    assert payloads[0]["system"] == "PLANNER PROMPT\n\nSUMMARY"
    assert payloads[1]["systemInstruction"]["parts"][0]["text"] == "PLANNER PROMPT\n\nSUMMARY"
    assert cloud.joined_system([{"role": "user", "content": "x"}], "fallback") == "fallback"


@pytest.mark.asyncio
async def test_the_server_does_not_run_a_refused_repair_and_says_why() -> None:
    from tests.test_execute_tracked import _FakeWs, _server_ready_for_handle_execute

    request_text = f"Copy {W}\\big\\big300.bin to {W}\\big\\big300_tampered.bin"

    class _Executor:
        runs = 0

        async def execute(self, plan, **kwargs):
            _Executor.runs += 1
            return [ActionResult(action=a, success=False, error="No space left on device") for a in plan.actions]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(
                passed=False,
                details=["Action 0 (file_copy): FAILED \u2014 No space left on device"],
                failed_actions=[0],
            )

    server = _server_ready_for_handle_execute(_Executor())
    server._verifier = _Verifier()
    model = ScriptedModel(
        plan_json(copy_json(f"{W}\\big\\big300.bin", f"{W}\\big\\big300_tampered.bin")),
        plan_json(copy_json(f"{W}\\big\\big300.bin", f"{W}\\big\\big300_copy.bin")),
    )
    server._planner = Planner(model, SessionMemory(EARLIER))  # type: ignore[arg-type]

    result = await server._handle_execute({"input": request_text}, _FakeWs())

    assert _Executor.runs == 1
    assert result["status"] == "partial_failure"
    assert result["message"].endswith(
        "I did not try again: the retry would copy big300.bin to big300_copy.bin, which this request did not name."
    )
    assert request_match.is_refusal(result["message"].splitlines()[-1])
