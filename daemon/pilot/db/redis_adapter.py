"""Redis cache adapter for high-throughput token caching.

Provides a unified cache interface that uses Redis when available,
falling back to a local in-memory LRU cache for single-instance setups.

Usage:
    adapter = RedisCacheAdapter.from_config(config.redis)
    await adapter.initialize()

    await adapter.set("key", "value", ttl=300)
    value = await adapter.get("key")   # None on miss
    await adapter.delete("key")
    await adapter.close()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pilot.db.redis_adapter")

# ── In-memory LRU fallback ────────────────────────────────────────────────────

_SENTINEL = object()


class _LRUCache:
    """Thread-safe LRU cache with per-entry TTL."""

    def __init__(self, max_size: int = 512) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()  # key → (value, expires_at)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._store.get(key, _SENTINEL)
            if entry is _SENTINEL:
                return None
            value, expires_at = entry  # type: ignore[misc]
            if expires_at != -1 and time.monotonic() > expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        expires_at = time.monotonic() + ttl if ttl > 0 else -1
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def flush(self) -> None:
        async with self._lock:
            self._store.clear()

    def size(self) -> int:
        return len(self._store)


# ── Adapter ───────────────────────────────────────────────────────────────────

@dataclass
class RedisConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    key_prefix: str = "pilot:"
    default_ttl: int = 300          # seconds
    max_memory_cache_size: int = 512  # fallback LRU size


class RedisCacheAdapter:
    """Cache adapter: Redis when available, in-memory LRU otherwise."""

    def __init__(self, config: RedisConfig) -> None:
        self._config = config
        self._redis: Any = None          # aioredis / redis.asyncio client
        self._fallback = _LRUCache(max_size=config.max_memory_cache_size)
        self._using_redis = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: RedisConfig) -> RedisCacheAdapter:
        return cls(config)

    async def initialize(self) -> None:
        if not self._config.enabled:
            logger.info("Redis disabled — using in-memory LRU cache")
            return

        try:
            import redis.asyncio as aioredis  # type: ignore[import]

            kwargs: dict[str, Any] = {
                "host": self._config.host,
                "port": self._config.port,
                "db": self._config.db,
                "ssl": self._config.ssl,
                "decode_responses": True,
            }
            if self._config.password:
                kwargs["password"] = self._config.password

            client = aioredis.Redis(**kwargs)
            await client.ping()

            self._redis = client
            self._using_redis = True
            logger.info(
                "Redis cache connected at %s:%s (db=%s)",
                self._config.host,
                self._config.port,
                self._config.db,
            )

        except ImportError:
            logger.warning("redis package not installed — falling back to in-memory LRU cache")

        except Exception as exc:
            logger.warning("Redis connection failed (%s) — falling back to in-memory LRU cache", exc)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            self._using_redis = False

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return "redis" if self._using_redis else "memory"

    def _prefixed(self, key: str) -> str:
        return f"{self._config.key_prefix}{key}"

    async def get(self, key: str) -> str | None:
        """Return cached value or None on miss / expiry."""
        if self._using_redis:
            try:
                return await self._redis.get(self._prefixed(key))
            except Exception as exc:
                logger.warning("Redis GET failed (%s) — using fallback", exc)
                return await self._fallback.get(key)
        return await self._fallback.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Store a value. ttl=0 → no expiry."""
        effective_ttl = ttl if ttl is not None else self._config.default_ttl
        if self._using_redis:
            try:
                if effective_ttl > 0:
                    await self._redis.setex(self._prefixed(key), effective_ttl, value)
                else:
                    await self._redis.set(self._prefixed(key), value)
                return
            except Exception as exc:
                logger.warning("Redis SET failed (%s) — writing to fallback", exc)
        await self._fallback.set(key, value, ttl=effective_ttl)

    async def delete(self, key: str) -> None:
        """Delete a key from both backends."""
        if self._using_redis:
            try:
                await self._redis.delete(self._prefixed(key))
                return
            except Exception as exc:
                logger.warning("Redis DELETE failed (%s) — deleting from fallback", exc)
        await self._fallback.delete(key)

    async def exists(self, key: str) -> bool:
        """Return True if key exists and has not expired."""
        return await self.get(key) is not None

    async def flush(self) -> None:
        """Flush all pilot-prefixed keys (Redis) or all entries (memory)."""
        if self._using_redis:
            try:
                pattern = f"{self._config.key_prefix}*"
                keys = await self._redis.keys(pattern)
                if keys:
                    await self._redis.delete(*keys)
                return
            except Exception as exc:
                logger.warning("Redis FLUSH failed (%s) — flushing fallback", exc)
        await self._fallback.flush()

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "fallback_size": self._fallback.size(),
            "redis_connected": self._using_redis,
            "key_prefix": self._config.key_prefix,
        }