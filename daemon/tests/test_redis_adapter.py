"""Tests for the Redis cache adapter.

Covers:
- In-memory LRU fallback (no Redis dependency needed)
- Cache hit / miss / TTL expiry
- Redis-mode: set/get/delete/exists/flush
- Fallback on Redis failure
- Key prefixing
- stats()
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.db.redis_adapter import RedisCacheAdapter, RedisConfig, _LRUCache

# ─── _LRUCache ────────────────────────────────────────────────────────────────


class TestLRUCache:
    @pytest.fixture
    def cache(self):
        return _LRUCache(max_size=3)

    async def test_miss_returns_none(self, cache):
        assert await cache.get("missing") is None

    async def test_set_and_get(self, cache):
        await cache.set("k", "v", ttl=60)
        assert await cache.get("k") == "v"

    async def test_ttl_expiry(self, cache):
        await cache.set("k", "v", ttl=1)
        # Manually expire by patching time
        with patch("pilot.db.redis_adapter.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 10
            assert await cache.get("k") is None

    async def test_no_ttl_never_expires(self, cache):
        await cache.set("k", "v", ttl=0)
        with patch("pilot.db.redis_adapter.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 99999
            assert await cache.get("k") == "v"

    async def test_delete(self, cache):
        await cache.set("k", "v", ttl=60)
        await cache.delete("k")
        assert await cache.get("k") is None

    async def test_delete_nonexistent_no_error(self, cache):
        await cache.delete("nonexistent")  # should not raise

    async def test_lru_eviction(self, cache):
        await cache.set("a", "1", ttl=60)
        await cache.set("b", "2", ttl=60)
        await cache.set("c", "3", ttl=60)
        # Access "a" to make it recently used
        await cache.get("a")
        # Insert "d" — should evict "b" (LRU)
        await cache.set("d", "4", ttl=60)
        assert await cache.get("b") is None
        assert await cache.get("a") == "1"

    async def test_flush_clears_all(self, cache):
        await cache.set("a", "1", ttl=60)
        await cache.set("b", "2", ttl=60)
        await cache.flush()
        assert cache.size() == 0

    async def test_size(self, cache):
        assert cache.size() == 0
        await cache.set("a", "1", ttl=60)
        assert cache.size() == 1

    async def test_overwrite_existing_key(self, cache):
        await cache.set("k", "old", ttl=60)
        await cache.set("k", "new", ttl=60)
        assert await cache.get("k") == "new"
        assert cache.size() == 1


# ─── RedisCacheAdapter — memory fallback mode ─────────────────────────────────


class TestAdapterMemoryMode:
    @pytest.fixture
    def config(self):
        return RedisConfig(enabled=False)

    @pytest.fixture
    async def adapter(self, config):
        a = RedisCacheAdapter(config)
        await a.initialize()
        yield a
        await a.close()

    async def test_backend_is_memory(self, adapter):
        assert adapter.backend == "memory"

    async def test_set_and_get(self, adapter):
        await adapter.set("key", "value")
        assert await adapter.get("key") == "value"

    async def test_miss_returns_none(self, adapter):
        assert await adapter.get("no-such-key") is None

    async def test_delete(self, adapter):
        await adapter.set("k", "v")
        await adapter.delete("k")
        assert await adapter.get("k") is None

    async def test_exists_true(self, adapter):
        await adapter.set("k", "v")
        assert await adapter.exists("k") is True

    async def test_exists_false(self, adapter):
        assert await adapter.exists("nope") is False

    async def test_flush(self, adapter):
        await adapter.set("a", "1")
        await adapter.set("b", "2")
        await adapter.flush()
        assert await adapter.get("a") is None
        assert await adapter.get("b") is None

    async def test_stats(self, adapter):
        s = adapter.stats()
        assert s["backend"] == "memory"
        assert s["redis_connected"] is False
        assert "fallback_size" in s

    async def test_default_ttl_used(self, adapter):
        # With default TTL, key should be stored
        await adapter.set("k", "v")
        assert await adapter.get("k") == "v"

    async def test_key_prefix_not_applied_in_memory(self, adapter):
        # Memory mode doesn't prefix — direct key lookup works
        await adapter.set("mykey", "myval")
        assert await adapter.get("mykey") == "myval"


# ─── RedisCacheAdapter — Redis mode ──────────────────────────────────────────


class TestAdapterRedisMode:
    @pytest.fixture
    def config(self):
        return RedisConfig(enabled=True, host="127.0.0.1", port=6379, key_prefix="test:")

    def _make_mock_redis(self):
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock()
        mock.setex = AsyncMock()
        mock.delete = AsyncMock()
        mock.keys = AsyncMock(return_value=[])
        mock.aclose = AsyncMock()
        return mock

    @pytest.fixture
    async def adapter_redis(self, config):
        mock_redis = self._make_mock_redis()
        with (
            patch.dict("sys.modules", {"redis": MagicMock(), "redis.asyncio": MagicMock()}),
            patch("pilot.db.redis_adapter.RedisCacheAdapter.initialize", new_callable=AsyncMock) as mock_init,
        ):
            a = RedisCacheAdapter(config)
            a._redis = mock_redis
            a._using_redis = True
            yield a, mock_redis

    async def test_backend_is_redis(self, adapter_redis):
        adapter, _ = adapter_redis
        assert adapter.backend == "redis"

    async def test_get_calls_redis(self, adapter_redis):
        adapter, mock = adapter_redis
        mock.get.return_value = "cached_value"
        result = await adapter.get("mykey")
        mock.get.assert_called_once_with("test:mykey")
        assert result == "cached_value"

    async def test_get_miss_returns_none(self, adapter_redis):
        adapter, mock = adapter_redis
        mock.get.return_value = None
        assert await adapter.get("missing") is None

    async def test_set_with_ttl_uses_setex(self, adapter_redis):
        adapter, mock = adapter_redis
        await adapter.set("k", "v", ttl=60)
        mock.setex.assert_called_once_with("test:k", 60, "v")

    async def test_set_no_ttl_uses_set(self, adapter_redis):
        adapter, mock = adapter_redis
        await adapter.set("k", "v", ttl=0)
        mock.set.assert_called_once_with("test:k", "v")

    async def test_delete_calls_redis(self, adapter_redis):
        adapter, mock = adapter_redis
        await adapter.delete("k")
        mock.delete.assert_called_once_with("test:k")

    async def test_flush_deletes_prefixed_keys(self, adapter_redis):
        adapter, mock = adapter_redis
        mock.keys.return_value = ["test:a", "test:b"]
        await adapter.flush()
        mock.keys.assert_called_once_with("test:*")
        mock.delete.assert_called_once_with("test:a", "test:b")

    async def test_flush_no_keys_no_delete(self, adapter_redis):
        adapter, mock = adapter_redis
        mock.keys.return_value = []
        await adapter.flush()
        mock.delete.assert_not_called()

    async def test_stats_redis_mode(self, adapter_redis):
        adapter, _ = adapter_redis
        s = adapter.stats()
        assert s["backend"] == "redis"
        assert s["redis_connected"] is True
        assert s["key_prefix"] == "test:"


# ─── Fallback on Redis failure ────────────────────────────────────────────────


class TestAdapterRedisFallback:
    @pytest.fixture
    def config(self):
        return RedisConfig(enabled=True)

    async def test_import_error_falls_back_to_memory(self, config):
        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            a = RedisCacheAdapter(config)
            await a.initialize()
            assert a.backend == "memory"
            await a.close()

    async def test_redis_get_failure_uses_fallback(self, config):
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = ConnectionError("Redis down")

        a = RedisCacheAdapter(config)
        a._redis = mock_redis
        a._using_redis = True

        # Pre-populate fallback
        await a._fallback.set("k", "fallback_value", ttl=60)

        result = await a.get("k")
        assert result == "fallback_value"

    async def test_redis_set_failure_writes_to_fallback(self, config):
        mock_redis = AsyncMock()
        mock_redis.setex.side_effect = ConnectionError("Redis down")

        a = RedisCacheAdapter(config)
        a._redis = mock_redis
        a._using_redis = True

        await a.set("k", "v", ttl=60)
        assert await a._fallback.get("k") == "v"

    async def test_redis_delete_failure_deletes_from_fallback(self, config):
        mock_redis = AsyncMock()
        mock_redis.delete.side_effect = ConnectionError("Redis down")

        a = RedisCacheAdapter(config)
        a._redis = mock_redis
        a._using_redis = True

        await a._fallback.set("k", "v", ttl=60)
        await a.delete("k")
        assert await a._fallback.get("k") is None


# ─── from_config ──────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_from_config_returns_adapter(self):
        cfg = RedisConfig(enabled=False)
        adapter = RedisCacheAdapter.from_config(cfg)
        assert isinstance(adapter, RedisCacheAdapter)

    def test_config_defaults(self):
        cfg = RedisConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 6379
        assert cfg.db == 0
        assert cfg.default_ttl == 300
        assert cfg.key_prefix == "pilot:"
        assert cfg.enabled is False
