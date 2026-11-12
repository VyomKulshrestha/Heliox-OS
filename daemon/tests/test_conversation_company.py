"""While a slow planner works, Heliox keeps the person company: a new, specific line every 8 to 10 seconds, never
the same sentence twice running, nothing over the person, and why it is slow the first time a local model plans."""

from __future__ import annotations

import asyncio

from pilot.conversation.handover import LOCAL_SLOW, PROGRESS_SPACING_SECONDS, keep_company


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def run_planner(seconds: float, *, speaking=lambda _t: False, local=False, state=None):
    """A planner that takes ``seconds`` on the injected clock; the clock moves 0.5 s per poll."""
    clock = Clock()
    said: list[tuple[float, str]] = []
    once = [local]

    async def speak(text, channel, key):
        said.append((clock.t, text))
        return "spoken"

    def status():
        clock.t += 0.5
        return (state or (lambda _t: {"phase": "planning", "message": "Planning the safest useful action"}))(clock.t)

    def local_slow_once():
        first, once[0] = once[0], False
        return first

    task = asyncio.ensure_future(
        keep_company(
            acknowledgement_text="Okay, working on that.",
            speak=speak,
            status=status,
            clock=clock,
            poll=0,
            user_speaking=lambda: speaking(clock.t),
            local_slow_once=local_slow_once,
        )
    )
    while clock.t < seconds:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return said


async def test_a_forty_second_plan_is_never_silent_for_long_and_never_repeats_itself():
    said = await run_planner(40.0)
    times = [t for t, _ in said]
    gaps = [b - a for a, b in zip([0.0, *times], [*times, 40.0])]
    assert max(gaps) <= PROGRESS_SPACING_SECONDS + 1.0  # a new line every 8 to 10 seconds, to the end
    lines = [text for _, text in said]
    assert all(a != b for a, b in zip(lines, lines[1:]))  # never the same sentence twice running
    assert lines[0] == "Okay, working on that." and lines[1] == "I'm working out the steps."
    assert len(lines) >= 5


async def test_nothing_is_said_while_the_person_is_talking():
    said = await run_planner(30.0, speaking=lambda t: 9.0 <= t <= 20.0)
    assert not any(9.0 <= t <= 20.0 for t, _ in said)
    assert any(t > 20.0 for t, _ in said)  # and it goes on after they stop


async def test_the_first_long_wait_on_a_local_model_says_why_once():
    said = await run_planner(40.0, local=True)
    lines = [text for _, text in said]
    assert lines.count(LOCAL_SLOW) == 1


async def test_when_the_state_names_a_step_the_line_says_which():
    def state(t):
        if t < 12:
            return {"phase": "planning", "message": "Planning the safest useful action"}
        return {"phase": "acting", "message": "Running step 2 of 4"}

    said = await run_planner(40.0, state=state)
    lines = [text for _, text in said]
    assert "Doing it now." in lines and "On step 2 of 4." in lines
