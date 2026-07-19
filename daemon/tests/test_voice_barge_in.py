"""Tests for speak_interruptible() (barge-in: stop TTS playback the instant
the user starts talking) and the underlying subprocess-cancellation fix
that makes it actually work (run_command previously left the TTS
subprocess running in the background after its owning coroutine was
cancelled -- CancelledError was never caught to kill it)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.system import voice
from pilot.system.platform_detect import run_command


class _FakeRecorder:
    def __init__(self, is_active: bool, speech_starts: bool, delay: float = 0.0):
        self.is_active = is_active
        self._speech_starts = speech_starts
        self._delay = delay

    async def wait_for_speech_start(self, timeout=None) -> bool:
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._speech_starts:
            # Never resolves within the test's timeframe -- simulate "no
            # interruption" by sleeping far longer than speak() takes.
            await asyncio.sleep(10)
        return True


@pytest.mark.asyncio
async def test_no_recorder_falls_back_to_plain_speak():
    with patch.object(voice, "speak", new=AsyncMock(return_value="ok")) as mock_speak:
        interrupted = await voice.speak_interruptible("hello", recorder=None)

    mock_speak.assert_awaited_once()
    assert interrupted is False


@pytest.mark.asyncio
async def test_inactive_recorder_falls_back_to_plain_speak():
    recorder = _FakeRecorder(is_active=False, speech_starts=True)
    with patch.object(voice, "speak", new=AsyncMock(return_value="ok")) as mock_speak:
        interrupted = await voice.speak_interruptible("hello", recorder=recorder)

    mock_speak.assert_awaited_once()
    assert interrupted is False


@pytest.mark.asyncio
async def test_speech_detected_mid_playback_interrupts():
    speak_started = asyncio.Event()

    async def _slow_speak(text, **kwargs):
        speak_started.set()
        await asyncio.sleep(10)  # would run "forever" relative to the test
        return "ok"

    recorder = _FakeRecorder(is_active=True, speech_starts=True, delay=0.01)
    with patch.object(voice, "speak", new=_slow_speak):
        interrupted = await voice.speak_interruptible("hello", recorder=recorder)

    assert interrupted is True


@pytest.mark.asyncio
async def test_playback_completes_without_interruption():
    recorder = _FakeRecorder(is_active=True, speech_starts=False)
    with patch.object(voice, "speak", new=AsyncMock(return_value="ok")) as mock_speak:
        interrupted = await voice.speak_interruptible("hi", recorder=recorder)

    mock_speak.assert_awaited_once()
    assert interrupted is False


@pytest.mark.asyncio
async def test_run_command_kills_subprocess_on_cancellation():
    """Regression test for the leak this barge-in fix depends on: before
    it, cancelling the coroutine awaiting run_command() left the
    subprocess running in the background (CancelledError was never caught
    to kill it) -- it would keep talking even after speak_interruptible()
    thought it had stopped it."""

    async def _hang_communicate(*args, **kwargs):
        await asyncio.sleep(30)

    fake_proc = AsyncMock()
    fake_proc.communicate = _hang_communicate
    fake_proc.kill = MagicMock()  # Process.kill() is sync in real asyncio, not awaited
    fake_proc.returncode = 0

    async def _fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc

    with patch("asyncio.create_subprocess_exec", new=_fake_create_subprocess_exec):
        task = asyncio.create_task(run_command(["some-command"]))
        await asyncio.sleep(0.05)  # let run_command reach the communicate() await
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    fake_proc.kill.assert_called_once()
