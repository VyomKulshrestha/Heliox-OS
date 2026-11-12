"""Follow-ups on what Heliox just said, inside the follow-up window: which resolve in code from the last words, which
run the last action again through the normal road, and which go to the model with the last answer in memory."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, VerificationResult
from pilot.config import PilotConfig
from pilot.conversation.followups import FOLLOW_UP_SECONDS, SHORT_ALREADY, follow_up_reply, repeat_meaning
from pilot.conversation.layer import ConversationLayer
from pilot.server import PilotServer

ANSWER = "The battery is at 80 percent. It's charging and will be full in about 40 minutes."
REPORT = "Volume is at 60. I raised it by ten."
EITHER = ["Again.", "Once more", "One more time", "Repeat.", "Repeat that.", "Repeat it"]


# ── the rule ──


@pytest.mark.parametrize("said", ["Say that again.", "What did you say?", "What was that?", "Come again?", "Pardon?"])
@pytest.mark.parametrize("last_turn", ["answer", "action", ""])
def test_words_only_follow_ups_repeat_the_words_whatever_the_last_turn(said, last_turn):
    assert follow_up_reply(said, last_spoken=REPORT, seconds_since=2.0, last_turn=last_turn) == REPORT


@pytest.mark.parametrize("said", EITHER)
def test_after_an_answer_again_repeats_the_words(said):
    assert follow_up_reply(said, last_spoken=ANSWER, seconds_since=2.0, last_turn="answer") == ANSWER


@pytest.mark.parametrize("said", EITHER)
def test_after_an_action_again_means_the_action(said):
    assert repeat_meaning(said, last_turn="action", seconds_since=2.0) == "action"
    assert follow_up_reply(said, last_spoken=REPORT, seconds_since=2.0, last_turn="action") is None


@pytest.mark.parametrize(
    ("said", "meaning"),
    [("Again.", ""), ("Once more", ""), ("Repeat that.", "action"), ("Repeat it", "action")],
)
def test_when_the_last_turn_is_not_known_the_old_meaning_wins(said, meaning):
    assert repeat_meaning(said, last_turn="", seconds_since=2.0) == meaning


@pytest.mark.parametrize("last_turn", ["answer", "action"])
def test_shorter_is_about_the_words_after_an_answer_and_after_an_action_report(last_turn):
    assert follow_up_reply("Shorter.", last_spoken=ANSWER, seconds_since=2.0, last_turn=last_turn) == (
        "The battery is at 80 percent."
    )
    assert follow_up_reply("In short", last_spoken=REPORT, seconds_since=2.0, last_turn=last_turn) == "Volume is at 60."
    one = "Volume is at 60."
    assert follow_up_reply("TL;DR", last_spoken=one, seconds_since=2.0, last_turn=last_turn) == SHORT_ALREADY


def test_after_the_window_again_is_not_about_the_words_and_repeat_that_is_the_task():
    late = FOLLOW_UP_SECONDS + 1.0
    assert repeat_meaning("Again.", last_turn="answer", seconds_since=late) == ""
    assert repeat_meaning("Shorter.", last_turn="answer", seconds_since=late) == ""
    assert repeat_meaning("Repeat that.", last_turn="answer", seconds_since=late) == "action"
    assert repeat_meaning("Do that again.", last_turn="answer", seconds_since=2.0) == "action"


@pytest.mark.parametrize("said", ["And tomorrow?", "What about the second one?", "Why?"])
def test_follow_ups_that_change_the_content_are_left_to_the_model(said):
    assert follow_up_reply(said, last_spoken=ANSWER, seconds_since=2.0, last_turn="answer") is None


# ── on the server's voice road ──


def voice_server(last_turn: str, spoken: str, seconds_ago: float, *, task: str | None = "turn the volume up"):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._spawn_interaction_speech = MagicMock()
    server._handle_execute = AsyncMock(
        return_value={"status": "success", "message": "Volume is at 70.", "results": [{"success": True}]}
    )
    server._latest_receipt = AsyncMock(return_value={"receipt": {"request": task}} if task else None)
    server._last_spoken, server._last_spoken_at = spoken, time.monotonic() - seconds_ago
    server._last_turn_kind = last_turn
    return server


def said_aloud(server) -> list[str]:
    return [call.args[0] for call in server._speak_voice_response.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["Again.", "Once more.", "One more time."])
async def test_an_action_then_again_runs_the_action_road_and_says_nothing_twice(said):
    server = voice_server("action", REPORT, 3.0)
    await server._voice_command_dispatch(said)
    server._handle_execute.assert_awaited_once()
    sent = server._handle_execute.await_args.args[0]
    # The last request, taken again by the road it took the first time: one allow-listed step goes by the quick
    # road with its own checks, anything else by the planner with its reviews and approvals.
    assert sent["input"] == "turn the volume up"
    assert REPORT not in said_aloud(server)
    assert server._last_turn_kind == "action"


@pytest.mark.asyncio
async def test_an_answer_then_again_repeats_the_words_and_runs_nothing():
    server = voice_server("answer", ANSWER, 3.0)
    await server._voice_command_dispatch("Again.")
    server._handle_execute.assert_not_awaited()
    assert said_aloud(server) == [ANSWER]


@pytest.mark.asyncio
async def test_repeat_that_nine_seconds_later_is_the_task_as_before():
    server = voice_server("answer", ANSWER, 9.0)
    await server._voice_command_dispatch("Repeat that.")
    assert server._handle_execute.await_args.args[0]["input"] == "turn the volume up"
    nothing = voice_server("answer", ANSWER, 9.0, task=None)
    await nothing._voice_command_dispatch("Repeat that.")
    nothing._handle_execute.assert_not_awaited()
    assert said_aloud(nothing) == ["There is no earlier task for me to repeat yet."]


@pytest.mark.asyncio
async def test_the_last_turn_is_recorded_from_what_ran_not_from_the_wording():
    server = voice_server("", "", 99.0)
    server._handle_execute.return_value = {"status": "success", "message": "I opened Notepad.", "results": []}
    await server._voice_command_dispatch("open notepad")
    assert server._last_turn_kind == "answer"  # it says it opened something, but nothing ran
    server._handle_execute.return_value = {"status": "denied", "message": "Okay.", "results": []}
    await server._voice_command_dispatch("delete the old backups")
    assert server._last_turn_kind == ""  # denied: not known as either, so the old meaning wins next


# ── an action that needed approval still needs it the second time ──


def _approving_server(seen: dict[str, Any]) -> PilotServer:
    config = PilotConfig()
    config.security.dry_run = False
    config.typed_decision.enabled = False
    server = PilotServer(config)
    move = Action.model_validate(
        {
            "action_type": "file_move",
            "target": "C:/d/a.txt",
            "parameters": {"source": "C:/d/a.txt", "destination": "C:/e/a.txt"},
        }
    )

    async def plan(user_input, **kwargs):
        seen.setdefault("planned", []).append(user_input)
        return ActionPlan(actions=[move], explanation="move it")

    async def wait_for_confirmation(plan_id, plan, ws, **kwargs):
        seen["asked_approval"] = seen.get("asked_approval", 0) + 1
        return False, [], []

    async def execute(run_plan: ActionPlan, **kwargs: Any) -> list[Any]:
        seen["executed"] = True
        return [ActionResult(action=action, success=True, output="ok") for action in run_plan.actions]

    server._planner = SimpleNamespace(plan=plan)
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = MagicMock()
    server._permission_checker.plan_requires_confirmation.return_value = True
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    server._task_receipts = AsyncMock()
    server._task_receipts.finalize.return_value = None
    server._checkpoint_store = AsyncMock()
    server._orchestrator = None
    server._control_reply = AsyncMock(return_value=None)
    server._wait_for_confirmation = wait_for_confirmation
    server._executor = SimpleNamespace(execute=execute)
    server._latest_receipt = AsyncMock(return_value={"receipt": {"request": "move a.txt to the e folder"}})
    return server


@pytest.mark.asyncio
async def test_an_action_that_needed_approval_still_asks_for_it_when_said_again():
    seen: dict[str, Any] = {}
    server = _approving_server(seen)
    ws = MagicMock()
    ws.send = AsyncMock()
    await asyncio.wait_for(server._handle_execute({"input": "do that again", "session_id": "s"}, ws), 10)
    assert seen["planned"] == ["move a.txt to the e folder"]  # the last task's own words, planned afresh
    assert seen["asked_approval"] == 1 and not seen.get("executed")


# ── content follow-ups go to the model ──


@pytest.mark.asyncio
async def test_a_content_follow_up_reaches_the_model_with_the_quick_answer_in_memory():
    calls = []

    class Model:
        name, provider, streams, local = "fake", "fake", True, False

        async def stream(self, system, messages):
            calls.append(" ".join(m["content"] for m in messages))
            yield "I can only see the battery now, not a forecast for tomorrow."

    async def sink(text, channel, key):
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: Model(), hold_after=60.0)
    asked = await layer.take("what's the battery level")
    assert not asked.done and asked.quick is not None  # the quick road, no model
    asked.finish("success", ANSWER)
    reply = await layer.take("And tomorrow?")
    assert reply.done and calls and "80 percent" in calls[0]


@pytest.mark.asyncio
async def test_again_after_a_planned_task_goes_back_through_the_planner_road():
    server = voice_server("action", REPORT, 3.0, task="delete the old installers in my Downloads folder")
    await server._voice_command_dispatch("Again.")
    sent = server._handle_execute.await_args.args[0]
    assert sent["input"] == "delete the old installers in my Downloads folder"
    assert "_quick_plan" not in sent  # not an allow-listed step: the planner, its reviews and approvals


@pytest.mark.asyncio
async def test_an_answer_given_in_code_is_part_of_the_conversation_unless_the_turn_was_doubted():
    from pilot.conversation import layer_for
    from pilot.system.voice import voice_turn_addressed
    from pilot.workflows.quick_answers import QuickAnswers

    server = voice_server("", "", 99.0)
    server._quick_answers = QuickAnswers()
    layer_for(server)._model = lambda: None
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("what's two plus two")
    kept = [(turn.role, turn.text) for turn in layer_for(server).memory.recent()]
    assert ("user", "what's two plus two") in kept and ("assistant", "That's 4.") in kept
    voice_turn_addressed.set("doubtful_follow_up")
    await server._voice_command_dispatch("what's three plus three")
    assert not any("three" in text for _, text in [(t.role, t.text) for t in layer_for(server).memory.recent()])
