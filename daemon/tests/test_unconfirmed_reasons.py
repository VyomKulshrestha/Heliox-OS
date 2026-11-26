"""Why a step was not confirmed reaches the person: in the verification result, the reply and the spoken sentence."""

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
    assert f"REPORTED — no independent postcondition ({NO_IMAP})" in verification.details[0]
    assert f"Not confirmed, friend@example.com: {NO_IMAP}" in reply
    assert "Body" not in reply


@pytest.mark.asyncio
async def test_a_plainly_verified_step_and_a_step_without_a_check_add_no_reason(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    _use(monkeypatch, ActionType.API_SEND_EMAIL, verified("found in Sent"))
    plan, results = _email()
    read = Action(action_type=ActionType.FILE_READ, target="/tmp/a.txt", parameters=FileParams(path="/tmp/a.txt"))
    plan = ActionPlan(actions=[*plan.actions, read])
    results = [*results, ActionResult(action=read, success=True, output="text")]

    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)

    assert verification.reasons == {}
    assert "Not confirmed" not in reply


def test_a_long_plan_says_at_most_three_reasons() -> None:
    from pilot.actions import VerificationResult

    actions = [
        Action(action_type=ActionType.API_WEBHOOK, target=f"hook {index}", parameters=ApiRequestParams())
        for index in range(5)
    ]
    results = [ActionResult(action=action, success=True, output="200") for action in actions]
    verification = VerificationResult(
        passed=True, reasons={index: "Receipt only: the server answered 200" for index in range(5)}
    )

    reply = success_message(ActionPlan(actions=actions), results, verification, dry_run=False)

    assert reply.count("Not confirmed, ") == 3
    assert "2 more steps not confirmed." in reply


def test_the_spoken_sentence_names_only_what_was_not_confirmed() -> None:
    message = f"Email sent: Hi\nNot confirmed, friend@example.com: {NO_IMAP}"

    spoken = result_sentence("success", message)

    assert spoken == (
        "Email sent: Hi. I couldn't confirm it: the server accepted the message; its copy in Sent could not be "
        "checked: no IMAP settings."
    )
    assert result_sentence("success", "Email sent: Hi") == "Email sent: Hi."
