"""Heliox Live in a call: no voice, urgent cards only, the rest said after the call if it still applies.

Injected clocks and a fake call signal; the real call detection is never run."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live.calls import CallWatch
from pilot.live.engine import AFTER_CALL_SECONDS, PENDING_TTL_SECONDS, TICK_SECONDS, LiveCompanion, Observation
from pilot.live.timing import PersonalTiming

MONDAY = date(2026, 11, 9)


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def build(tmp_path, text, call, outcome="spoken"):
    clock = Clock()
    events, spoken = [], []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[])

    async def observe():
        return Observation(text=text[0], app="notepad.exe", title="notes.txt - Notepad", source="focused_field")

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        spoken.append(message)
        return outcome

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        relevance_path=tmp_path / "rel.json",
        clock=clock,
        today=lambda: MONDAY,
        in_call=lambda: call[0],
    )
    return live, clock, events, spoken


async def run_for(live, clock, seconds):
    for _ in range(int(round(seconds / TICK_SECONDS))):
        clock.t += TICK_SECONDS
        await live.tick()
    for _ in range(3):
        await asyncio.sleep(0)


def cards(events):
    return [p for e, p in events if e == "live_interjection"]


@pytest.mark.asyncio
async def test_in_a_call_live_holds_its_voice_and_says_what_still_applies_afterwards(tmp_path):
    call, text = ["Zoom"], ["The total is 12 x 15 = 170 so we order more."]
    live, clock, events, spoken = build(tmp_path, text, call)
    await run_for(live, clock, PENDING_TTL_SECONDS + 30)  # a long call: longer than a catch normally waits
    assert spoken == [] and cards(events) == []
    call[0] = ""
    await run_for(live, clock, AFTER_CALL_SECONDS - 1)
    assert spoken == []  # a moment after the call first
    await run_for(live, clock, 3)
    assert spoken == ["While you were on the call: 12 × 15 is 180."]  # the short form; the card says the rest


@pytest.mark.asyncio
async def test_a_slip_fixed_during_the_call_is_not_raised_afterwards(tmp_path):
    call, text = ["Teams"], ["The total is 12 x 15 = 170 so we order more."]
    live, clock, events, spoken = build(tmp_path, text, call)
    await run_for(live, clock, 10)
    text[0] = "The total is 12 x 15 = 180 so we order more."
    await run_for(live, clock, 10)
    call[0] = ""
    await run_for(live, clock, 15)
    assert spoken == [] and cards(events) == []


@pytest.mark.asyncio
async def test_an_urgent_risk_in_a_call_is_a_card_and_never_a_voice(tmp_path):
    call, text = ["Google Meet"], ["$ git reset --hard HEAD~3"]
    live, clock, events, spoken = build(tmp_path, text, call)
    await run_for(live, clock, 2)
    [card] = cards(events)
    assert card["kind"] == "risky_command" and card["spoken"] is False and spoken == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    # suppressed: a focus session or a call; no_voice: the voice failed or there is none on this system (as a status,
    # and as the speech path's outcome object)
    ["suppressed", "no_voice", SimpleNamespace(status="no_voice")],
)
async def test_a_remark_nobody_heard_does_not_count_against_the_next_one(tmp_path, outcome):
    call, text = [""], ["The total is 12 x 15 = 170 so we order more."]
    live, clock, events, spoken = build(tmp_path, text, call, outcome=outcome)
    before = live._last_spoken_at
    await run_for(live, clock, 8)
    assert len(spoken) == 1  # asked to speak; the speech path suppressed it (a focus session, a call)
    assert live._last_spoken_at == before  # so the spoken rate limit is given back
    assert cards(events)[0]["spoken"] is True and live._recent[-1]["spoken"] is False


def test_call_watch_uses_the_listener_when_it_runs_and_polls_otherwise():
    async def scenario():
        detected = []

        def detect():
            detected.append(1)
            return "Slack"

        listener = SimpleNamespace(is_running=True, in_call="Zoom")
        enabled = [True]
        clock = Clock()
        watch = CallWatch(listener=lambda: listener, enabled=lambda: enabled[0], detect=detect, clock=clock)
        assert watch() == "Zoom" and detected == []
        listener.is_running = False
        assert watch() == ""  # the first answer comes from a worker thread
        for _ in range(50):
            await asyncio.sleep(0.01)
            if watch() == "Slack":
                break
        assert watch() == "Slack" and len(detected) == 1  # not asked again within EVERY_SECONDS
        enabled[0] = False
        assert watch() == ""  # quiet_in_calls off

    asyncio.run(scenario())


def test_well_timed_cards_shrink_the_wait_only_when_live_spoke_near_its_threshold():
    timing = PersonalTiming()
    timing.record_outcome("helpful", None, paused_for=30.0, needed=2.0)  # they were long done: no evidence
    assert timing.scale == 1.0 and timing.well_timed == 1
    timing.record_outcome("helpful", None, paused_for=2.2, needed=2.0)
    assert timing.scale < 1.0


@pytest.mark.asyncio
async def test_on_the_bench_nothing_goes_into_a_call_and_not_now_holds():
    from pilot.live import bench

    pack = [s for s in bench.load_scenarios() if s["name"].startswith(("call:", "not now:"))]
    [blind] = await bench.run_all(pack, personas=["steady"], policies=("personal",), calls=False)
    [aware] = await bench.run_all(pack, personas=["steady"], policies=("personal",))
    assert blind.into_call >= 2 and aware.into_call == 0
    assert aware.catches == aware.expected and aware.false_alarms == 0
