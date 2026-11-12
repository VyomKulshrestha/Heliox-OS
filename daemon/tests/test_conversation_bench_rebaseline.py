"""Re-baseline helpers: the load guard, timings owed on a busy machine, microphone labels, the stub fast model."""

from __future__ import annotations

import asyncio

from benchmarks.conversation_bench import machine, rebaseline, scoring, stack
from benchmarks.conversation_bench import timeline as tl
from benchmarks.conversation_bench.models import KINDS
from benchmarks.conversation_bench.scenarios import parse_scenario
from benchmarks.conversation_bench.scoring import Row


def test_unknown_or_high_load_is_busy_and_waiting_gives_up():
    assert machine.is_busy(None) and machine.is_busy(90.0) and not machine.is_busy(20.0)
    loads = iter([95.0, 92.0, 30.0])
    slept = []
    assert machine.wait_for_quiet(120, sample=lambda: next(loads), sleep=slept.append) == (True, 30.0)
    assert slept == [30.0, 30.0]
    assert machine.wait_for_quiet(0, sample=lambda: 99.0, sleep=slept.append) == (False, 99.0)


def test_timing_rows_are_owed_on_a_busy_machine_and_others_kept():
    rows = [
        Row("1.1a", "real", "as-recorded|desk", "cloud", 5.7, 8, False),
        Row("2.1a", "real", "x", "cloud", 0.2, 9, False),
    ]
    out = rebaseline.owe_timings(rows, {("real", "as-recorded|desk", "cloud"): True})
    assert out[0].value is None and out[0].detail == "owed, machine busy"
    assert out[1].value == 0.2
    quiet = rebaseline.owe_timings(rows, {("real", "as-recorded|desk", "cloud"): False})
    assert quiet[0].value == 5.7


def test_real_rows_carry_their_microphone():
    s = parse_scenario({"id": "u", "set": "understanding", "voice": "real", "mic": "bluetooth-headset-call-mode",
                        "steps": [{"say": "real/a.wav", "text": "Hi"}], "expect": {}})  # fmt: skip
    o = rebaseline.relabel(scoring.score(s, tl.measure([]), condition="bt-call"))
    assert o.condition == "bt-call|headset"


def test_the_stub_fast_model_streams_with_the_kinds_delays():
    async def collect():
        model = stack.StubFastModel(KINDS["cloud"], "One two.")
        return [piece async for piece in model.stream("system", [])], model.calls

    pieces, calls = asyncio.run(collect())
    assert "".join(pieces).strip() == "One two." and len(calls) == 1
    assert not stack.StubFastModel(KINDS["cloud"]).local and stack.StubFastModel(KINDS["local"]).local
