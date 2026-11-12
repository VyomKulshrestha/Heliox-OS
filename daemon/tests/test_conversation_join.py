"""A sentence split at a pause and joined: the person is acknowledged once, hears each progress line once, and hears
nothing more from the withdrawn first half (scratchpad bench_split_withdraw.txt: "Hey Heliox, remind me to" |
"call the bank tomorrow morning", where pauses of 1.0 to 2.0 s said "I'm working out the steps." twice)."""

from __future__ import annotations

import asyncio
import functools

import pytest

from pilot.config import PilotConfig
from pilot.conversation import handover, layer_for
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed, voice_turn_id

FIRST, REST = "remind me too", "call the bank tomorrow morning"


@pytest.fixture(autouse=True)
def quick_company(monkeypatch):
    """The company's timing scaled down (progress after 0.05 s instead of 2 s) so the test does not wait seconds."""
    fast = functools.partial(handover.keep_company, progress_after=0.05, poll=0.01)
    monkeypatch.setattr("pilot.conversation.layer.keep_company", fast)


def server_with_planner():
    server = PilotServer(PilotConfig())
    heard: list[str] = []

    async def speak(text, channel="final_answer", dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._model_router = object()
    layer_for(server)._model = lambda: None
    calls: list[str] = []
    finish: dict[str, asyncio.Event] = {}

    async def execute(params, ws, *args, **kwargs):
        calls.append(params["input"])
        finish[params["input"]] = asyncio.Event()
        await finish[params["input"]].wait()  # the planner works until the test lets it finish
        return {"status": "success", "message": "All set.", "results": []}

    server._handle_execute = execute
    return server, heard, calls, finish


async def wait_until(condition, seconds=3.0):
    for _ in range(int(seconds / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


async def first_half(server):
    voice_turn_addressed.set("wake")
    voice_turn_id.set("v1")
    await server._voice_command_dispatch(FIRST)


@pytest.mark.parametrize("pause", [0.6, 1.0, 1.5, 2.0])
async def test_a_joined_sentence_says_each_line_once(pause):
    server, heard, calls, finish = server_with_planner()
    task = asyncio.ensure_future(first_half(server))
    await wait_until(lambda: calls)
    if pause >= 1.0:  # the longer pauses: the first half's progress line was already said
        await wait_until(lambda: "I'm working out the steps." in heard)
    result = await server.withdraw_voice_turn("v1", f"{FIRST} {REST}", gap=pause)
    assert result.status == "withdrawn"
    await asyncio.gather(task, return_exceptions=True)
    await wait_until(lambda: len(calls) == 2)
    await asyncio.sleep(0.2)  # long enough for a repeated progress line to have been said
    finish[f"{FIRST} {REST}"].set()
    await asyncio.gather(*server._background_dispatches)
    acknowledgements = [line for line in heard if "remind" in line.casefold() or line == "Working on it."]
    assert len(acknowledgements) == 1, heard
    assert heard.count("I'm working out the steps.") <= 1, heard
    assert heard[-1] == "All set." and heard.count("All set.") == 1, heard


async def test_a_first_half_that_finishes_as_it_is_withdrawn_says_nothing():
    server, heard, calls, finish = server_with_planner()
    task = asyncio.ensure_future(first_half(server))
    await wait_until(lambda: calls)
    before = list(heard)
    finish[FIRST].set()  # its planner call ends in the same moment the rest of the sentence arrives
    result = await server.withdraw_voice_turn("v1", f"{FIRST} {REST}", gap=1.0)
    assert result.status == "withdrawn"
    await asyncio.gather(task, return_exceptions=True)
    await wait_until(lambda: len(calls) == 2)
    finish[f"{FIRST} {REST}"].set()
    await asyncio.gather(*server._background_dispatches)
    after_withdrawal = heard[len(before) :]
    assert after_withdrawal.count("All set.") == 1, heard  # the joined sentence's result, and only it
