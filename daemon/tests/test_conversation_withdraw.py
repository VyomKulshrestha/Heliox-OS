"""A sentence split at a pause: what can still be withdrawn, and what happens then (conv_interfaces.md, section b)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan
from pilot.config import PilotConfig
from pilot.conversation.turns import WITHDRAW_SECONDS, TurnLog, said_about, voice_turn_id
from pilot.server import PilotServer


def plan_of(kind: str, target: str) -> ActionPlan:
    params = {"name": target} if kind == "open_application" else {"path": target}
    return ActionPlan(actions=[Action.model_validate({"action_type": kind, "target": target, "parameters": params})])


# ── the log ──


async def _begin(log: TurnLog, turn_id: str):
    voice_turn_id.set(turn_id)
    return log.begin()


async def test_a_turn_that_has_run_nothing_is_withdrawn_and_its_task_stopped():
    log = TurnLog()
    stalled = asyncio.Event()

    async def first_half():
        await _begin(log, "t1")
        await stalled.wait()

    task = asyncio.ensure_future(first_half())
    await asyncio.sleep(0)
    result = await log.withdraw("t1", deny_quietly=AsyncMock(), pending=list)
    assert result.status == "withdrawn"
    assert task.cancelled()


async def test_once_the_executor_has_the_plan_it_is_too_late_and_heliox_says_what_was_done():
    log = TurnLog()
    await _begin(log, "t2")
    log.note_executing(plan_of("open_application", "notepad"))
    result = await log.withdraw("t2", deny_quietly=AsyncMock(), pending=list)
    assert result.status == "too_late" and result.ran == ("open_application",)
    assert result.said == "I've already opened Notepad."


async def test_a_change_kept_in_code_is_too_late_without_a_sentence():
    log = TurnLog()
    await _begin(log, "t3")
    log.note_kept()
    assert (await log.withdraw("t3", deny_quietly=AsyncMock(), pending=list)).status == "too_late"


async def test_an_approval_waiting_for_the_first_half_is_denied_quietly():
    log = TurnLog()
    voice_turn_id.set("t4")
    log.begin(approvals=["older"])
    deny = AsyncMock()
    result = await log.withdraw("t4", deny_quietly=deny, pending=lambda: ["older", "plan-of-t4"])
    assert result.status == "withdrawn"
    deny.assert_awaited_once_with("plan-of-t4")  # an approval from before this turn is left alone


async def test_an_unknown_or_old_turn_is_unknown():
    now = [100.0]
    log = TurnLog(clock=lambda: now[0])
    assert (await log.withdraw("nope", deny_quietly=AsyncMock(), pending=list)).status == "unknown"
    await _begin(log, "t5")
    now[0] += WITHDRAW_SECONDS + 0.1
    assert (await log.withdraw("t5", deny_quietly=AsyncMock(), pending=list)).status == "unknown"


def test_what_was_done_is_said_plainly():
    assert said_about([("file_delete", "C:/Users/me/old.zip")]) == "I've already deleted old.zip."
    assert said_about([("mystery_action", "x")]) == "I've already done the first part."


# ── through the server ──


def server_with_stalled_planner():
    server = PilotServer(PilotConfig())
    heard: list[str] = []

    async def speak(text, channel="final_answer", dedupe_key="", recorder=None):
        from pilot.system.companion_speech import SpeechChannel, SpeechOutcome

        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    server._speech_coordinator.speak = speak
    server._broadcast_notification = AsyncMock()
    server._spawn_interaction_speech = MagicMock()
    server._model_router = object()
    from pilot.conversation import layer_for

    layer_for(server)._model = lambda: None
    calls: list[str] = []
    release = asyncio.Event()

    async def execute(params, ws, *args, **kwargs):
        calls.append(params["input"])
        await release.wait()  # the planner is still thinking
        return {"status": "success", "message": "Done.", "results": []}

    server._handle_execute = execute
    return server, heard, calls, release


async def test_the_joined_sentence_replaces_the_first_half_and_its_acknowledgement_is_not_said_twice():
    server, heard, calls, release = server_with_stalled_planner()
    first = asyncio.ensure_future(server._voice_command_dispatch("Open Notepad"))
    for _ in range(50):
        if calls:
            break
        await asyncio.sleep(0.01)
    first_id = server.last_voice_turn_id
    assert heard == ["Opening Notepad."]  # the first half's acknowledgement
    result = await server.withdraw_voice_turn(first_id, "Open Notepad please")
    assert result.status == "withdrawn" and first.cancelled()
    for _ in range(50):
        if len(calls) == 2:
            break
        await asyncio.sleep(0.01)
    release.set()
    await asyncio.sleep(0.05)
    assert calls == ["Open Notepad", "Open Notepad please"]
    assert heard.count("Opening Notepad.") == 1  # not said again for the joined sentence
    assert heard[-1] == "Notepad is open."  # its result is said as usual


async def test_too_late_says_what_was_done_once():
    server, heard, calls, release = server_with_stalled_planner()
    voice_turn_id.set("done-turn")
    server._voice_turns.begin()
    server._voice_turns.note_executing(plan_of("open_application", "notepad"))
    voice_turn_id.set("")
    result = await server.withdraw_voice_turn("done-turn", "Open Notepad and Spotify")
    assert result.status == "too_late"
    assert heard == ["I've already opened Notepad."]
    assert calls == []  # the listener takes the second half as a new turn


@pytest.mark.parametrize("turn_id", ["", "never-seen"])
async def test_unknown_turns_change_nothing(turn_id):
    server, heard, calls, _ = server_with_stalled_planner()
    result = await server.withdraw_voice_turn(turn_id, "anything")
    assert result.status == "unknown" and heard == [] and calls == []


def test_the_interface_is_what_conv_interfaces_md_names():
    assert callable(PilotServer.withdraw_voice_turn)
    assert SimpleNamespace(status="withdrawn").status in {"withdrawn", "too_late", "unknown"}


async def test_the_limit_counts_on_the_capture_clock_when_the_listener_gives_it():
    now = [100.0]
    log = TurnLog(clock=lambda: now[0])
    await _begin(log, "slow")
    now[0] += 6.1  # recognising the second half took long on a busy machine
    assert log.get("slow") is None  # by processing time it would be over
    assert log.get("slow", gap=2.6) is not None  # but the person paused only 2.6 s
    assert log.get("slow", gap=WITHDRAW_SECONDS + 0.1) is None
    assert (await log.withdraw("slow", deny_quietly=AsyncMock(), pending=list, gap=2.6)).status == "withdrawn"


async def test_a_sentence_already_acknowledged_is_not_acknowledged_again_whatever_it_became():
    server, heard, calls, release = server_with_stalled_planner()
    first = asyncio.ensure_future(server._voice_command_dispatch("Open Notepad"))
    for _ in range(50):
        if calls:
            break
        await asyncio.sleep(0.01)
    assert heard == ["Opening Notepad."]
    result = await server.withdraw_voice_turn(server.last_voice_turn_id, "Open Spotify", gap=1.5)
    assert result.status == "withdrawn" and first.cancelled()
    for _ in range(50):
        if len(calls) == 2:
            break
        await asyncio.sleep(0.01)
    release.set()
    await asyncio.sleep(0.05)
    assert calls == ["Open Notepad", "Open Spotify"]
    assert not any(line.startswith("Opening Spotify") for line in heard)  # one acknowledgement per sentence
