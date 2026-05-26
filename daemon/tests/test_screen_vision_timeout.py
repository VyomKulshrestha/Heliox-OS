"""Tests for ScreenVisionAgent display-off timeout handling."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pilot.agents.screen_vision import ScreenVisionAgent

# Short timeout for fast tests
TIMEOUT = 0.05
THRESHOLD = 3


@pytest.fixture
def agent():
    return ScreenVisionAgent(
        capture_timeout_seconds=TIMEOUT,
        max_consecutive_timeouts=THRESHOLD,
        auto_resume_after_seconds=30.0,
    )


async def wait_until(cond, poll=0.02, max_wait=10.0):
    for _ in range(int(max_wait / poll)):
        if cond():
            return True
        await asyncio.sleep(poll)
    return False


class TestTimeoutPausesAgent:
    """Agent must pause after N consecutive capture timeouts."""

    @pytest.mark.asyncio
    async def test_pauses_after_consecutive_timeouts(self, agent):
        async def always_timeout():
            raise asyncio.TimeoutError()

        agent._capture_state = always_timeout
        await agent.start(interval_seconds=0.01)
        paused = await wait_until(lambda: agent.is_paused(), max_wait=5.0)

        assert paused
        assert agent._consecutive_timeouts >= agent._max_consecutive_timeouts
        await agent.stop()

    @pytest.mark.asyncio
    async def test_does_not_pause_before_threshold(self, agent):
        timeouts = 0

        async def timeout_then_pass():
            nonlocal timeouts
            timeouts += 1
            if timeouts <= THRESHOLD - 1:
                raise asyncio.TimeoutError()
            return AsyncMock()

        agent._capture_state = timeout_then_pass
        await agent.start(interval_seconds=0.01)
        await wait_until(lambda: agent._consecutive_timeouts > 0, max_wait=3.0)
        await wait_until(lambda: agent._consecutive_timeouts == 0, max_wait=3.0)

        assert not agent.is_paused()
        assert agent._consecutive_timeouts == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_resets_timeout_counter_on_success(self, agent):
        agent._consecutive_timeouts = 2

        async def succeed():
            return AsyncMock()

        agent._capture_state = succeed
        await agent.start(interval_seconds=0.01)
        await wait_until(lambda: agent._consecutive_timeouts == 0, max_wait=3.0)

        assert agent._consecutive_timeouts == 0
        await agent.stop()


class TestAutoResume:
    """Agent must auto-resume when capture succeeds after being paused."""

    @pytest.mark.asyncio
    async def test_auto_resumes_after_timeout_clears(self, agent):
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count <= THRESHOLD + 1:
                raise asyncio.TimeoutError()
            return AsyncMock()

        agent._capture_state = fail_then_succeed
        await agent.start(interval_seconds=0.01)

        await wait_until(
            lambda: not agent.is_paused() and call_count > THRESHOLD + 1,
            max_wait=10.0,
        )

        assert not agent.is_paused()
        assert agent._consecutive_timeouts == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_auto_resume_logs_message(self, agent):
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count <= THRESHOLD + 1:
                raise asyncio.TimeoutError()
            return AsyncMock()

        agent._capture_state = fail_then_succeed

        with patch("pilot.agents.screen_vision.logger") as mock_logger:
            await agent.start(interval_seconds=0.01)
            await wait_until(lambda: agent.is_paused(), max_wait=5.0)
            await wait_until(lambda: not agent.is_paused(), max_wait=5.0)

            info_messages = [c for c in mock_logger.info.call_args_list if "auto-resumed" in str(c)]
            assert info_messages

        await agent.stop()


class TestManualPauseResume:
    """Manual pause() and resume() must work correctly."""

    @pytest.mark.asyncio
    async def test_pause_sets_paused_state(self, agent):
        await agent.start(interval_seconds=0.1)
        assert not agent.is_paused()

        agent.pause()
        assert agent.is_paused()

        await agent.stop()

    @pytest.mark.asyncio
    async def test_resume_clears_paused_and_counter(self, agent):
        agent._consecutive_timeouts = 5
        agent._paused = True

        agent.resume()

        assert not agent.is_paused()
        assert agent._consecutive_timeouts == 0

    def test_resume_while_not_paused_resets_counter(self):
        agent = ScreenVisionAgent()
        agent._consecutive_timeouts = 2

        agent.resume()

        assert not agent.is_paused()
        assert agent._consecutive_timeouts == 0


class TestStats:
    """get_stats must expose timeout and paused state."""

    @pytest.mark.asyncio
    async def test_stats_includes_paused_and_timeout_info(self, agent):
        await agent.start(interval_seconds=0.1)
        stats = agent.get_stats()

        assert "paused" in stats
        assert "consecutive_timeouts" in stats
        assert "max_consecutive_timeouts" in stats
        assert "capture_timeout_seconds" in stats
        assert stats["running"] is True
        assert stats["paused"] is False

        await agent.stop()

    @pytest.mark.asyncio
    async def test_stats_reflects_paused_state(self, agent):
        await agent.start(interval_seconds=0.1)
        agent.pause()
        stats = agent.get_stats()

        assert stats["paused"] is True

        await agent.stop()

    def test_stats_timeout_values(self):
        a = ScreenVisionAgent(
            capture_timeout_seconds=5.0,
            max_consecutive_timeouts=2,
        )
        stats = a.get_stats()
        assert stats["capture_timeout_seconds"] == 5.0
        assert stats["max_consecutive_timeouts"] == 2


class TestConfigurability:
    """Timeout parameters must be configurable."""

    def test_custom_timeout_values(self):
        a = ScreenVisionAgent(
            capture_timeout_seconds=5.0,
            max_consecutive_timeouts=2,
            auto_resume_after_seconds=15.0,
        )
        assert a._capture_timeout == 5.0
        assert a._max_consecutive_timeouts == 2
        assert a._auto_resume_after == 15.0

    def test_default_values(self):
        a = ScreenVisionAgent()
        assert a._capture_timeout == 10.0
        assert a._max_consecutive_timeouts == 3
        assert a._auto_resume_after == 30.0


class TestCaptureLoopResilience:
    """Capture loop must survive exceptions and continue running."""

    @pytest.mark.asyncio
    async def test_loop_continues_after_random_exception(self, agent):
        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("unexpected error")
            return AsyncMock()

        agent._capture_state = fail_once
        await agent.start(interval_seconds=0.01)
        await wait_until(lambda: call_count > 1, max_wait=3.0)

        assert call_count > 1
        await agent.stop()

    @pytest.mark.asyncio
    async def test_start_resets_timeout_counter(self, agent):
        agent._consecutive_timeouts = 99
        agent._paused = True

        async def succeed():
            return AsyncMock()

        agent._capture_state = succeed
        await agent.start(interval_seconds=0.01)
        assert agent._consecutive_timeouts == 0
        assert not agent._paused

        await agent.stop()
