"""What Heliox says about an action follows the action's state (pilot.conversation.action_talk): never "Deleting
that now." before anything runs, the approval sentence first for a plan that needs one, the end state said as it
is. The real path, for every kind of person, is the session "what is said about an action follows its state"
(benchmarks/conversation_eval/sessions.json, run by test_conversation_sessions.py); these pin the rule itself."""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.conversation_eval.sessions import KINDS, PACK, check_session, load_pack, run_session
from pilot.conversation import handover
from pilot.conversation.action_talk import (
    ActionState,
    asked_sentence,
    claims_done,
    claims_running,
    may_say,
    needs_approval_first,
    result_sentence,
    said_again,
)


@pytest.mark.parametrize(
    "sentence",
    ["Deleting that now.", "Okay, messaging your manager.", "Opening Notepad.", "Emailing the acme invoice."],
)
def test_an_under_way_sentence_is_said_only_while_running(sentence):
    assert claims_running(sentence)
    assert may_say(ActionState.RUNNING, sentence)
    for state in ActionState:
        if state is not ActionState.RUNNING:
            assert not may_say(state, sentence), state


@pytest.mark.parametrize("sentence", ["Notepad is open.", "Done.", "I've sent it.", "The files are deleted."])
def test_a_finished_sentence_is_said_only_when_done(sentence):
    assert claims_done(sentence)
    assert [state for state in ActionState if may_say(state, sentence)] == [ActionState.DONE]


@pytest.mark.parametrize(
    "sentence",
    [
        "I'll open your launch folder.",
        handover.APPROVAL_SENTENCE,
        "Okay, I've dropped that: delete the screenshots. Nothing ran.",
        "That didn't work: access is denied.",
    ],
)
def test_what_claims_nothing_may_be_said_in_any_state(sentence):
    assert all(may_say(state, sentence) for state in ActionState)


@pytest.mark.parametrize(
    "verb", ["delete", "remove", "uninstall", "send", "email", "message", "close", "shut", "restart", "pay"]
)
def test_a_request_whose_plan_needs_approval_says_nothing_before_the_approval(verb):
    assert needs_approval_first(verb)
    assert asked_sentence(verb, "the old installers") == ""
    intent = handover.HandoverIntent(verb=verb, object="the old installers", words="", request="", source="router")
    assert handover.acknowledgement(intent) == ""


def test_the_fast_models_own_sentence_is_used_only_when_it_says_what_will_happen():
    assert asked_sentence("find", "your notes", "I'll look through your notes folder.") == (
        "I'll look through your notes folder."
    )
    assert asked_sentence("find", "your notes", "Looking through your notes now.") == "I'll look for your notes."
    assert asked_sentence("message", "your manager", "Messaging your manager.") == ""


def test_a_sentence_said_again_keeps_the_capital_i():
    assert said_again("I'll open Notepad.") == "Okay, I'll open Notepad."
    assert said_again("Opening Notepad.") == "Okay, opening Notepad."


@pytest.mark.parametrize(
    ("status", "message", "said"),
    [
        (
            "success",
            "Deleted\nNote: 1 action effect was reported successful but not independently verified.",
            "Deleted. I couldn't check that it worked.",
        ),
        ("success", "Notepad is open.", "Notepad is open."),
        ("success", "Execution results:\n- a: one\n- b: two", ""),  # a list stays in the window
        (
            "partial_failure",
            "Task did not complete successfully: 1 of 1 action failed or could not be verified. Access is denied.",
            "That didn't work: access is denied.",
        ),
        ("error", "Task did not complete successfully: no action result was verified.", "That didn't work."),
        ("denied", "Okay, I've dropped that. Nothing ran.", ""),  # the refusal is already said in plain words
    ],
)
def test_the_end_state_is_said_as_it_is(status, message, said):
    assert result_sentence(status, message) == said
    if said:
        assert not claims_running(said)


@pytest.mark.asyncio
async def test_a_joined_sentence_does_not_repeat_the_approval_its_first_half_said():
    said: list[str] = []

    async def speak(text: str, channel: str, key: str) -> str:
        said.append(text)
        return "spoken"

    company = asyncio.ensure_future(
        handover.keep_company(
            acknowledgement_text="",
            speak=speak,
            status=lambda: {"phase": "awaiting_approval"},
            poll=0.01,
            carried=(handover.APPROVAL_SENTENCE, 0.0),
        )
    )
    await asyncio.sleep(0.1)
    company.cancel()
    assert said == []


SESSIONS = {session["name"]: session for session in load_pack(PACK)}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [kind for kind in KINDS if kind != "none"])
async def test_the_session_check_catches_an_under_way_claim_before_the_approval(kind, monkeypatch):
    """The runner's rule bites: with the old acknowledgement put back, the destructive turn fails it. (A person with
    no model is promised nothing before the planner answers, so there is no sentence to put back.)"""
    session = SESSIONS["what is said about an action follows its state"]
    monkeypatch.setattr(handover, "asked_sentence", lambda verb, target, model_say="": "Deleting that now.")
    record, _failures = await run_session(session, kind)
    lines = check_session(session, kind, record)
    assert any("'Deleting that now.' says it is under way" in line for line in lines), lines
