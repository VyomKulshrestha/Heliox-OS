"""The session runner gives the same replies however fast the machine is: a change said to a waiting plan is waited
on (its old dialog closing, the plan made again) before the next turn, by the server's own state, never a sleep.
CI run 43341856533 on 5111230: on the Windows runners "undo that" arrived while the change was being applied, so the
plan came back afterwards and a later "undo that" dropped it instead of saying there was nothing to undo."""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.conversation_eval import sessions as S

CORRECTION = next(s for s in S.load_pack(S.PACK) if s["name"].startswith("a correction, then undo"))


def _run_with_planner_delay(delay: float, monkeypatch) -> list[tuple[int, str]]:
    built = S.build_server
    real_sleep = asyncio.sleep

    async def slow_planner_server(*args, **kwargs):
        server = await built(*args, **kwargs)
        plan = server._planner.plan

        async def slow(*a, **k):
            await real_sleep(delay)  # a slow machine: planning takes real time
            return await plan(*a, **k)

        server._planner.plan = slow
        return server

    monkeypatch.setattr(S, "build_server", slow_planner_server)
    record, failures = asyncio.run(S.run_session(CORRECTION, "claude"))
    assert failures == []
    return [(line.turn, line.text) for line in record.said]


@pytest.mark.parametrize("delay", [0.05, 0.3])
def test_a_slow_planner_gives_the_same_replies(delay, monkeypatch):
    fast = _run_with_planner_delay(0.0, monkeypatch)
    assert _run_with_planner_delay(delay, monkeypatch) == fast
