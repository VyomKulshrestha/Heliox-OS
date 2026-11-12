"""Sleep and resume, with injected clocks: the wall clock jumps, the monotonic clock stops (Linux, macOS) or jumps too
(Windows), or the wall clock is set back."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system.wake import ClockEvent, WakeDetector, watch


class Clocks:
    def __init__(self) -> None:
        self.wall = 1_800_000_000.0
        self.mono = 1000.0

    def detector(self) -> WakeDetector:
        return WakeDetector(wall=lambda: self.wall, monotonic=lambda: self.mono)

    def tick(self, seconds: float = 5.0) -> None:
        self.wall += seconds
        self.mono += seconds


def test_ordinary_checks_are_not_a_wake():
    clocks = Clocks()
    detector = clocks.detector()
    for _ in range(10):
        clocks.tick()
        assert detector.check() is None
    clocks.tick(30)  # a busy machine running the loop late is not a sleep either
    assert detector.check() is None


def test_an_hour_asleep_where_the_monotonic_clock_stops_linux_macos():
    clocks = Clocks()
    detector = clocks.detector()
    clocks.wall += 3600
    clocks.mono += 5
    event = detector.check()
    assert event is not None and event.kind == "wake" and event.seconds == pytest.approx(3595)


def test_an_hour_asleep_where_both_clocks_jump_windows():
    clocks = Clocks()
    detector = clocks.detector()
    clocks.tick(3600)
    event = detector.check()
    assert event is not None and event.kind == "wake" and event.seconds == pytest.approx(3595)


def test_a_wall_clock_set_back_is_not_a_wake():
    clocks = Clocks()
    detector = clocks.detector()
    clocks.wall -= 7200
    clocks.mono += 5
    assert detector.check() == ClockEvent("wall_clock_set_back", 7200)
    clocks.tick()
    assert detector.check() is None


async def test_the_watch_hands_each_event_on_and_survives_a_failing_handler():
    clocks = Clocks()
    detector = clocks.detector()
    seen: list[ClockEvent] = []
    steps = iter([3600.0, 5.0, 3600.0])

    async def fake_sleep(_seconds):
        try:
            clocks.tick(next(steps))
        except StopIteration:
            raise RuntimeError("done") from None

    async def handler(event):
        seen.append(event)
        if len(seen) == 1:
            raise ValueError("a handler that fails once")

    with pytest.raises(RuntimeError, match="done"):
        await watch(handler, detector=detector, sleep=fake_sleep)
    assert [event.kind for event in seen] == ["wake", "wake"]


async def test_waking_closes_the_no_wake_phrase_windows_and_checks_reminders_now():
    server = PilotServer(PilotConfig())
    listener = MagicMock()
    server._voice_listener = listener
    server._conversation = MagicMock()
    server._reminders = MagicMock(tick=AsyncMock(return_value=2))

    await server._on_wake(ClockEvent("wake", 3600))

    listener.clear_follow_up_window.assert_called_once()
    listener.end_interruption.assert_called_once()
    server._conversation.interrupt.assert_called_once()
    server._reminders.tick.assert_awaited_once()


async def test_a_clock_set_back_changes_nothing():
    server = PilotServer(PilotConfig())
    server._voice_listener = MagicMock()
    server._reminders = MagicMock(tick=AsyncMock())

    await server._on_wake(ClockEvent("wall_clock_set_back", 7200))

    server._voice_listener.clear_follow_up_window.assert_not_called()
    server._reminders.tick.assert_not_awaited()
