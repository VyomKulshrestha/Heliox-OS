"""Stop and correction while Heliox is talking or working: how fast the voice is cut and the model stream closed.

  python -m benchmarks.conversation_eval.interrupt_timing [--runs 10] [--wait-minutes 0]

Runs inside one process with the real server dispatch (PilotServer._voice_command_dispatch), the real speech
coordinator and the real conversation layer; the speaker is silent (it "plays" each sentence for as long as it
would take to say it) and the talk model is a stub that streams slowly, so nothing is heard and nothing is billed.

Scenarios, each repeated --runs times:
* reply: a spoken reply is streaming; the person says "stop". Measured from the dispatch of "stop" to the
  sentence being cut, and to the model stream being closed.
* task: a planned task is running and Heliox is narrating; the person says "no, the other folder". Measured from
  the dispatch to the narration being cut, and to the correction reaching the running task.

The listener's own delay (last word to dispatch) is the speech side's part and is not included. Timings are only
taken when the machine is quiet (the brief: never quote a timing measured under load): the script refuses, or
waits up to --wait-minutes for a quiet machine.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import sys
import time
from typing import Any

from .__main__ import machine_busy


class SlowTalk:
    """A talk model that writes a long reply slowly and notes when its stream was closed."""

    name, provider, streams, local = "stub", "stub", True, False

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.closed_at: float | None = None

    async def stream(self, system: str, messages: list[dict[str, str]]):
        try:
            for index in range(40):
                yield f"This is sentence number {index} of a long explanation. "
                await asyncio.sleep(0.15)
        finally:
            if self.closed_at is None:  # the first stream, the one the person interrupted
                self.closed_at = time.perf_counter()


async def _server():
    from unittest.mock import AsyncMock

    from pilot.config import PilotConfig
    from pilot.conversation.layer import ConversationLayer
    from pilot.server import PilotServer
    from pilot.system.companion_speech import CompanionSpeechCoordinator

    server = PilotServer(PilotConfig())
    server._voice_listener = None
    server._broadcast_notification = AsyncMock()
    server._model_router = object()
    marks: dict[str, float] = {}

    async def silent_speaker(text: str, recorder: Any) -> bool:
        marks.setdefault("speaking", time.perf_counter())
        try:
            await asyncio.sleep(0.06 * len(text))  # as long as the sentence would take to say
        except asyncio.CancelledError:
            marks.setdefault("voice_cut", time.perf_counter())
            raise
        return False

    server._speech_coordinator = CompanionSpeechCoordinator(speaker=silent_speaker)
    talk = SlowTalk()

    async def sink(text: str, channel: str, key: str) -> str:
        outcome = await server._speak_companion_text(text, channel=channel, dedupe_key=key)
        return str(outcome.status)

    server._conversation = ConversationLayer(sink=sink, model=lambda: talk, hold_after=60.0)
    return server, marks, talk


async def reply_once(words: str = "stop") -> dict[str, float]:
    server, marks, talk = await _server()
    turn = asyncio.ensure_future(server._voice_command_dispatch("Explain how a compiler works"))
    while "speaking" not in marks:
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.3)  # mid-sentence
    t0 = time.perf_counter()
    await server._voice_command_dispatch(words)
    await asyncio.wait_for(turn, 5)
    for _ in range(200):
        if talk.closed_at is not None:
            break
        await asyncio.sleep(0.005)
    return {
        "voice_cut_ms": (marks["voice_cut"] - t0) * 1000 if "voice_cut" in marks else float("nan"),
        "stream_closed_ms": (talk.closed_at - t0) * 1000 if talk.closed_at else float("nan"),
    }


async def task_once(words: str = "no, the other folder") -> dict[str, float]:
    from unittest.mock import AsyncMock

    server, marks, _talk = await _server()
    reached: dict[str, float] = {}
    started = asyncio.Event()

    async def long_task(params: dict, ws: Any) -> dict:
        server._interactive_request_active = True
        started.set()
        await asyncio.sleep(30)
        return {"status": "success", "message": "done"}

    async def interject(params: dict, ws: Any) -> dict:
        reached.setdefault("correction", time.perf_counter())
        return {"status": "revising", "message": "Revising the plan"}

    server._handle_execute = long_task
    server._handle_interject = interject
    server._interaction_runtime.transition = AsyncMock()
    task = asyncio.ensure_future(server._voice_command_dispatch("Find the invoice from October"))
    await started.wait()
    while "speaking" not in marks:  # the acknowledgement is being said
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.2)
    t0 = time.perf_counter()
    await server._voice_command_dispatch(words)
    for _ in range(200):  # the cut lands when the speech task next runs; its time is when it happened
        if "voice_cut" in marks:
            break
        await asyncio.sleep(0.005)
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    return {
        "voice_cut_ms": (marks["voice_cut"] - t0) * 1000 if "voice_cut" in marks else float("nan"),
        "correction_reached_ms": (reached["correction"] - t0) * 1000 if reached else float("nan"),
    }


def _stats(rows: list[dict[str, float]]) -> dict[str, Any]:
    out = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] == row[key]]
        out[key] = (
            {"median": round(statistics.median(values), 1), "worst": round(max(values), 1), "n": len(values)}
            if values
            else {"missing": len(rows)}
        )
    return out


async def main_async(runs: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for words in ("stop", "no, the other folder"):
        out[f"reply, then {words!r}"] = _stats([await reply_once(words) for _ in range(runs)])
        out[f"task, then {words!r}"] = _stats([await task_once(words) for _ in range(runs)])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_eval.interrupt_timing")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--wait-minutes", type=float, default=0.0)
    args = parser.parse_args(argv)
    deadline = time.time() + args.wait_minutes * 60
    while busy := machine_busy():
        if time.time() >= deadline:
            print(json.dumps({"owed": f"not measured: the machine is busy ({busy})"}))
            return 2
        time.sleep(30)
    print(json.dumps(asyncio.run(main_async(args.runs)), indent=1))
    busy = machine_busy()
    if busy:
        print(json.dumps({"warning": f"the machine became busy during the run ({busy}); do not quote these"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
