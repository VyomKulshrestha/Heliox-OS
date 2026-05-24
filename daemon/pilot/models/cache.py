"""Local cache for LLM responses.

L1: Redis (optional, distributed, fast) — keyed by prompt hash
L2: SQLite (local, persistent)          — exact-match fallback

Cache key components:
  - prompt hash (SHA256)
  - system prompt hash (SHA256)
  - model string  (e.g., "gpt-4o", "llama3.1:8b")
  - provider      (e.g., "openai", "ollama", "gemini")
  - temperature
  - json_mode flag
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from pilot.config import PilotConfig
    from pilot.db.redis_adapter import RedisCacheAdapter
    from pilot.db.sqlite_pool import AsyncSqlitePool

logger = logging.getLogger("pilot.models.cache")
CACHE_SCHEMA_VERSION = 1


class LLMCache:
    """Two-tier LLM response cache: Redis L1 → SQLite L2 Connection Pool."""

    def __init__(
        self,
        db_path: Path,
        redis: RedisCacheAdapter | None = None,
        pool: AsyncSqlitePool | None = None,
    ) -> None:
        self._db_path = db_path
        self._pool = pool
        self._initialized = False
        self._redis = redis  # optional L1 cache

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._pool is None:
            from pilot.db.sqlite_pool import AsyncSqlitePool
            self._pool = AsyncSqlitePool(self._db_path, read_pool_size=4)
            await self._pool.start()

        try:
            await self._create_schema()
            self._initialized = True
            logger.debug("LLM cache initialized at %s", self._db_path)
        except Exception as e:
            logger.error("Failed to initialize LLM cache: %s", e)
            raise

    async def _create_schema(self) -> None:
        if not self._pool:
            raise RuntimeError("Cache not initialized")

        async with self._pool.write() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT NOT NULL,
                    system_hash TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    json_mode INTEGER NOT NULL DEFAULT 0,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(prompt_hash, system_hash, model, provider, temperature, json_mode)
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_key ON llm_cache"
                "(prompt_hash, system_hash, model, provider, temperature, json_mode)"
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_provider ON llm_cache(provider)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_model ON llm_cache(model)")
            await db.commit()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _hash_string(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _make_cache_key(
        self,
        prompt: str | list[dict[str, Any]],
        model: str,
        provider: str,
        temperature: float,
        json_mode: bool,
        system: str = "",
    ) -> tuple[str, str, str, str, float, int]:
        """Generate a cache key from prompt parameters.

        Args:
            prompt: The user prompt or message list.
            model: The model identifier.
            provider: The provider name.
            temperature: The temperature parameter.
            json_mode: Whether JSON mode is enabled.
            system: The system prompt (optional).

        Returns:
            Tuple of (prompt_hash, system_hash, model, provider, temperature, json_mode_int)
        """
        if isinstance(prompt, list):
            import json

            prompt_str = json.dumps(prompt, sort_keys=True)
        else:
            prompt_str = prompt
        prompt_hash = self._hash_string(prompt_str)
        system_hash = self._hash_string(system) if system else ""
        return (prompt_hash, system_hash, model, provider, temperature, int(json_mode))

    def _redis_key(
        self,
        prompt_hash: str,
        system_hash: str,
        model: str,
        provider: str,
        temperature: float,
        json_mode_int: int,
    ) -> str:
        return f"llm:{provider}:{model}:{temperature}:{json_mode_int}:{system_hash}:{prompt_hash}"

    # ── public API ────────────────────────────────────────────────────────────

    async def get(
        self,
        prompt: str | list[dict[str, Any]],
        model: str,
        provider: str,
        temperature: float,
        json_mode: bool,
        system: str = "",
    ) -> str | None:
        prompt_hash, system_hash, model, provider, temperature, json_mode_int = self._make_cache_key(
            prompt, model, provider, temperature, json_mode, system
        )

        # L1 — Redis
        if self._redis is not None:
            rkey = self._redis_key(prompt_hash, system_hash, model, provider, temperature, json_mode_int)
            cached = await self._redis.get(rkey)
            if cached is not None:
                logger.debug("Redis cache hit: %s/%s", provider, model)
                return cached

        # L2 — SQLite
        if not self._pool:
            return None

        try:
            async with self._pool.read() as db:
                cursor = await db.execute(
                    """SELECT response FROM llm_cache
                       WHERE prompt_hash=? AND system_hash=? AND model=?
                         AND provider=? AND temperature=? AND json_mode=?
                       LIMIT 1""",
                    (prompt_hash, system_hash, model, provider, temperature, json_mode_int),
                )
                row = await cursor.fetchone()
                await cursor.close()

                if row:
                    logger.debug("SQLite cache hit: %s/%s (prompt: %s...)", provider, model, prompt[:30])
                    # Promote to Redis L1
                    if self._redis is not None:
                        rkey = self._redis_key(prompt_hash, system_hash, model, provider, temperature, json_mode_int)
                        await self._redis.set(rkey, row[0])
                    return row[0]

            return None
        except Exception as e:
            logger.warning("Cache lookup failed: %s", e)
            return None

    async def set(
        self,
        prompt: str | list[dict[str, Any]],
        model: str,
        provider: str,
        temperature: float,
        json_mode: bool,
        response: str,
        system: str = "",
    ) -> bool:
        prompt_hash, system_hash, model, provider, temperature, json_mode_int = self._make_cache_key(
            prompt, model, provider, temperature, json_mode, system
        )

        # Write to Redis L1
        if self._redis is not None:
            rkey = self._redis_key(prompt_hash, system_hash, model, provider, temperature, json_mode_int)
            await self._redis.set(rkey, response)

        # Write to SQLite L2
        if not self._pool:
            return False

        try:
            async with self._pool.write() as db:
                await db.execute(
                    """INSERT INTO llm_cache
                           (prompt_hash, system_hash, model, provider, temperature, json_mode, response)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(prompt_hash, system_hash, model, provider, temperature, json_mode)
                       DO UPDATE SET response=excluded.response, created_at=CURRENT_TIMESTAMP""",
                    (prompt_hash, system_hash, model, provider, temperature, json_mode_int, response),
                )
                await db.commit()
            logger.debug("Cached response: %s/%s (prompt: %s...)", provider, model, prompt[:30])
            return True
        except Exception as e:
            logger.warning("Failed to cache response: %s", e)
            return False

    async def stats(self) -> dict:
        base: dict = {}
        if self._pool:
            try:
                async with self._pool.read() as db:
                    cursor = await db.execute(
                        "SELECT COUNT(*), COUNT(DISTINCT provider), COUNT(DISTINCT model) FROM llm_cache"
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                    if row:
                        base = {
                            "total_cached_responses": row[0],
                            "unique_providers": row[1],
                            "unique_models": row[2],
                        }
            except Exception as e:
                logger.warning("Failed to get cache stats: %s", e)

        if self._redis is not None:
            base["redis"] = self._redis.stats()

        return base

    async def clear(self, provider: str | None = None, model: str | None = None) -> int:
        # Flush Redis L1 entirely (no fine-grained provider/model filter in Redis)
        if self._redis is not None:
            await self._redis.flush()

        if not self._pool:
            return 0

        try:
            async with self._pool.write() as db:
                if provider is None and model is None:
                    cursor = await db.execute("DELETE FROM llm_cache")
                    logger.info("Cleared entire LLM cache")
                elif provider and model:
                    cursor = await db.execute(
                        "DELETE FROM llm_cache WHERE provider=? AND model=?", (provider, model)
                    )
                    logger.info("Cleared cache for %s/%s", provider, model)
                elif provider:
                    cursor = await db.execute("DELETE FROM llm_cache WHERE provider=?", (provider,))
                    logger.info("Cleared cache for provider %s", provider)
                else:
                    cursor = await db.execute("DELETE FROM llm_cache WHERE model=?", (model,))
                    logger.info("Cleared cache for model %s", model)

                await db.commit()
                deleted = cursor.rowcount
                await cursor.close()
                return deleted
        except Exception as e:
            logger.warning("Failed to clear cache: %s", e)
            return 0

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.debug("LLM cache connection closed")