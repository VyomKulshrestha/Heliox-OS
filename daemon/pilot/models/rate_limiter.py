"""Token bucket rate limiter for LLM API calls."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pilot.config import ModelConfig

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.models.rate_limiter")


class TokenBucketRateLimiter:
    """Async token bucket rate limiter.

    Tokens refill continuously at ``rate_limit_rpm / 60`` per second up to
    ``rate_limit_burst`` capacity.  Each call to :meth:`acquire` consumes one
    token, waiting if the bucket is empty.  Sleep happens outside the lock so
    other coroutines can still acquire concurrently.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._enabled = config.rate_limit_enabled
        self._rate = config.rate_limit_rpm / 60.0  # tokens per second
        self._capacity = float(config.rate_limit_burst)
        self._tokens = float(config.rate_limit_burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._total_calls = 0
        self._total_waits = 0
        self._total_wait_ms = 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(
            self._capacity,
            self._tokens + (now - self._last_refill) * self._rate,
        )
        self._last_refill = now

    async def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        if not self._enabled:
            return
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._total_calls += 1
                    return
                # Compute wait outside the lock to avoid blocking other callers
                wait_time = (1.0 - self._tokens) / self._rate

            t0 = time.monotonic()
            logger.warning("Rate limit reached — waiting %.2fs before next LLM call", wait_time)
            await asyncio.sleep(wait_time)
            elapsed = time.monotonic() - t0
            self._total_waits += 1
            self._total_wait_ms += elapsed * 1000

    def reconfigure(self, config: ModelConfig) -> None:
        """Update rate/burst settings at runtime (e.g. after config change)."""
        self._enabled = config.rate_limit_enabled
        self._rate = config.rate_limit_rpm / 60.0
        self._capacity = float(config.rate_limit_burst)

    def get_stats(self) -> dict:
        """Return current limiter state and lifetime counters."""
        return {
            "enabled": self._enabled,
            "rpm": self._rate * 60,
            "burst": self._capacity,
            "available_tokens": round(self._tokens, 2),
            "total_calls": self._total_calls,
            "total_waits": self._total_waits,
            "avg_wait_ms": round(self._total_wait_ms / max(self._total_waits, 1), 1),
        }
