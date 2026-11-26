"""Why a step was not confirmed: kept in the verification result; the reply and the voice say only a fixed phrase."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams, FileParams
from pilot.agents import postconditions
from pilot.agents import verifier as verifier_module
from pilot.agents.postconditions import Outcome, register, unobserved, verified
from pilot.agents.verifier import Verifier
from pilot.conversation.action_talk import result_sentence
from pilot.response_contract import success_message

NO_IMAP = "The server accepted the message; its copy in Sent could not be checked: no IMAP settings"


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry = dict(postconditions._REGISTRY)
    monkeypatch.setattr(postconditions, "_REGISTRY", registry)
    return registry


def _use(monkeypatch: pytest.MonkeyPatch, action_type: ActionType, outcome: Outcome) -> None:
    async def check(_result: ActionResult) -> Outcome:
        return outcome

    # The scratch copy may already hold the action's real verifier (api_send_email has one): replace it here.
    postconditions._REGISTRY.pop(action_type, None)
    register(action_type, method="test_postcondition")(check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())


def _email() -> tuple[ActionPlan, list[ActionResult]]:
    action = Action(
        action_type=ActionType.API_SEND_EMAIL,
        target="friend@example.com",
        parameters=ApiRequestParams(to="friend@example.com", subject="Hi", message="Body"),
    )
    return ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="Email sent: Hi")]


@pytest.mark.asyncio
async def test_a_step_that_could_not_be_checked_carries_its_reason_to_the_reply(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    _use(monkeypatch, ActionType.API_SEND_EMAIL, unobserved(NO_IMAP))
    plan, results = _email()

    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)

    assert verification.passed is True
    assert verification.reasons == {0: NO_IMAP}
    assert verification.reason_kinds == {0: "could_not_look"}
    assert f"REPORTED — no independent postcondition ({NO_IMAP})" in verification.details[0]
    # Mail is sent to someone else: its doubt is said in a fixed phrase; the verifier's sentence is not in the reply.
    assert reply.endswith("Note: I couldn't check that it went through; details in the receipt.")
    assert NO_IMAP not in reply
    assert "Body" not in reply


@pytest.mark.asyncio
async def test_a_plainly_verified_step_and_a_step_without_a_check_add_no_reason(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    _use(monkeypatch, ActionType.API_SEND_EMAIL, verified("found in Sent"))
    plan, results = _email()
    read = Action(action_type=ActionType.FILE_LIST, target="/tmp/a.txt", parameters=FileParams(path="/tmp/a.txt"))
    plan = ActionPlan(actions=[*plan.actions, read])
    results = [*results, ActionResult(action=read, success=True, output="text")]

    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)

    assert verification.reasons == {}
    assert "Note:" not in reply


def test_a_long_plan_says_one_line() -> None:
    from pilot.actions import VerificationResult

    actions = [
        Action(action_type=ActionType.API_WEBHOOK, target=f"hook {index}", parameters=ApiRequestParams())
        for index in range(5)
    ]
    results = [ActionResult(action=action, success=True, output="200") for action in actions]
    verification = VerificationResult(
        passed=True,
        reasons={index: "Receipt only: the server answered 200" for index in range(5)},
        reason_kinds={index: "receipt_only" for index in range(5)},
    )

    reply = success_message(ActionPlan(actions=actions), results, verification, dry_run=False)

    assert reply.count("Note:") == 1
    assert reply.endswith(
        "Note: It was accepted, but I can't read it back, 4 more steps could not be checked; details in the receipt."
    )
    assert "server answered 200" not in reply


def test_the_spoken_sentence_says_the_fixed_phrase_never_the_reason() -> None:
    message = "Email sent: Hi\nNote: I couldn't check that it went through; details in the receipt."

    spoken = result_sentence("success", message)

    assert spoken == "Email sent: Hi. I couldn't check that it went through."
    assert result_sentence("success", "Email sent: Hi") == "Email sent: Hi."
