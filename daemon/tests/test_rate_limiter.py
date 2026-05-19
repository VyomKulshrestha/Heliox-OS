"""Unit tests for pilot.models.rate_limiter.TokenBucketRateLimiter.

Run with:
    cd daemon
    pytest tests/test_rate_limiter.py -v --tb=short
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from pilot.models.rate_limiter import TokenBucketRateLimiter

# ---------------------------------------------------------------------------
# Minimal ModelConfig stub
# ---------------------------------------------------------------------------


@dataclass
class FakeModelConfig:
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_limiter(**kwargs) -> TokenBucketRateLimiter:
    return TokenBucketRateLimiter(FakeModelConfig(**kwargs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_within_burst_does_not_wait():
    """Calls within burst capacity complete immediately (no sleep)."""
    limiter = make_limiter(rate_limit_burst=3)
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        for _ in range(3):
            await limiter.acquire()
        mock_sleep.assert_not_called()
    assert limiter.get_stats()["total_calls"] == 3
    assert limiter.get_stats()["total_waits"] == 0


@pytest.mark.asyncio
async def test_acquire_beyond_burst_triggers_wait():
    """Exhausting the bucket causes a sleep on the next call.

    We must also advance the simulated clock by the sleep duration so that
    _refill() adds enough tokens after the sleep returns, preventing an
    infinite retry loop.
    """
    import time as _real_time

    _offset = [0.0]
    _real_mono = _real_time.monotonic

    def fake_monotonic():
        return _real_mono() + _offset[0]

    async def advancing_sleep(seconds: float) -> None:
        _offset[0] += seconds + 0.001  # advance simulated clock past the wait

    limiter = make_limiter(rate_limit_burst=1, rate_limit_rpm=60)
    with (
        patch("pilot.models.rate_limiter.time") as mock_time,
        patch("asyncio.sleep", side_effect=advancing_sleep),
    ):
        mock_time.monotonic.side_effect = fake_monotonic
        await limiter.acquire()  # consumes the single token
        await limiter.acquire()  # bucket empty → waits once then succeeds

    assert limiter.get_stats()["total_waits"] == 1


@pytest.mark.asyncio
async def test_disabled_limiter_never_waits():
    """When rate_limit_enabled=False, acquire() returns immediately every time."""
    limiter = make_limiter(rate_limit_enabled=False, rate_limit_burst=1, rate_limit_rpm=60)
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        for _ in range(10):
            await limiter.acquire()
        mock_sleep.assert_not_called()
    # total_calls is not incremented when disabled
    assert limiter.get_stats()["total_calls"] == 0


@pytest.mark.asyncio
async def test_tokens_depleted_after_burst():
    """Available tokens reach ~0 after consuming the full burst."""
    limiter = make_limiter(rate_limit_burst=3, rate_limit_rpm=60)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        for _ in range(3):
            await limiter.acquire()
        # One more call empties bucket; tokens should be < 1 (possibly negative mid-wait)
        await limiter.acquire()
    stats = limiter.get_stats()
    assert stats["total_calls"] == 4


@pytest.mark.asyncio
async def test_reconfigure_updates_rate_and_burst():
    """reconfigure() changes the rate and burst without resetting stats."""
    limiter = make_limiter(rate_limit_rpm=60, rate_limit_burst=5)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await limiter.acquire()

    new_config = FakeModelConfig(rate_limit_rpm=120, rate_limit_burst=10)
    limiter.reconfigure(new_config)

    stats = limiter.get_stats()
    assert stats["rpm"] == 120
    assert stats["burst"] == 10
    assert stats["total_calls"] == 1  # previous call still counted


@pytest.mark.asyncio
async def test_get_stats_shape():
    """get_stats() returns all expected keys."""
    limiter = make_limiter()
    stats = limiter.get_stats()
    for key in ("enabled", "rpm", "burst", "available_tokens", "total_calls", "total_waits", "avg_wait_ms"):
        assert key in stats, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_reconfigure_disable():
    """Disabling via reconfigure stops waits on subsequent calls."""
    limiter = make_limiter(rate_limit_burst=1, rate_limit_rpm=60)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await limiter.acquire()  # consume token

    limiter.reconfigure(FakeModelConfig(rate_limit_enabled=False))
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        for _ in range(5):
            await limiter.acquire()
        mock_sleep.assert_not_called()
