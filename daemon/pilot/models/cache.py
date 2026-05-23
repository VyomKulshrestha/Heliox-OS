"""Local SQLite cache for LLM responses.

Provides fast exact-match caching for LLM prompts to reduce API usage and speed up testing.
Cache is keyed by:
  - prompt hash (SHA256)
  - system prompt hash (SHA256)
  - model string (e.g., "gpt-4o", "llama3.1:8b")
  - provider (e.g., "openai", "ollama", "gemini")
  - temperature
  - json_mode flag

This ensures cache hits only for exact prompt + model + provider combinations.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.models.cache")
CACHE_SCHEMA_VERSION = 1


class LLMCache:
    """SQLite-backed cache for LLM responses with provider and model specificity."""

    def __init__(self, db_path: Path) -> None:
        """Initialize cache with path to SQLite database.
        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database connection and create schema if needed."""
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = await aiosqlite.connect(str(self._db_path))
            await self._conn.execute("PRAGMA journal_mode = WAL")
            await self._conn.execute("PRAGMA synchronous = NORMAL")
            await self._conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
            await self._create_schema()
            self._initialized = True
            logger.debug("LLM cache initialized at %s", self._db_path)
        except Exception as e:
            logger.error("Failed to initialize LLM cache: %s", e)
            raise

    async def _create_schema(self) -> None:
        """Create cache table if it doesn't exist."""
        if not self._conn:
            raise RuntimeError("Cache not initialized")

        await self._conn.execute(
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

        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_key ON llm_cache(prompt_hash, system_hash, model, provider, temperature, json_mode)"
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_provider ON llm_cache(provider)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON llm_cache(model)")

        await self._conn.commit()

    def _hash_string(self, text: str) -> str:
        """Generate SHA256 hash of a string.

        Args:
            text: The string to hash.

        Returns:
            Hex-encoded SHA256 hash.
        """
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

    async def get(
        self,
        prompt: str | list[dict[str, Any]],
        model: str,
        provider: str,
        temperature: float,
        json_mode: bool,
        system: str = "",
    ) -> str | None:
        """Retrieve a cached response for an exact prompt match.

        Args:
            prompt: The user prompt.
            model: The model identifier.
            provider: The provider name.
            temperature: The temperature parameter.
            json_mode: Whether JSON mode is enabled.
            system: The system prompt (optional).

        Returns:
            The cached response if found, None otherwise.
        """
        if not self._conn:
            return None

        try:
            prompt_hash, system_hash, model, provider, temperature, json_mode_int = self._make_cache_key(
                prompt, model, provider, temperature, json_mode, system
            )

            cursor = await self._conn.execute(
                """
                SELECT response FROM llm_cache
                WHERE prompt_hash = ? AND system_hash = ? AND model = ? AND provider = ? AND temperature = ? AND json_mode = ?
                LIMIT 1
                """,
                (prompt_hash, system_hash, model, provider, temperature, json_mode_int),
            )
            row = await cursor.fetchone()
            await cursor.close()

            if row:
                logger.debug(
                    "Cache hit: %s/%s (prompt: %s...)",
                    provider,
                    model,
                    prompt[:30],
                )
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
        """Store a response in the cache.

        Args:
            prompt: The user prompt or message list.
            model: The model identifier.
            provider: The provider name.
            temperature: The temperature parameter.
            json_mode: Whether JSON mode is enabled.
            response: The response text to cache.
            system: The system prompt (optional).

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self._conn:
            return False

        try:
            prompt_hash, system_hash, model, provider, temperature, json_mode_int = self._make_cache_key(
                prompt, model, provider, temperature, json_mode, system
            )

            await self._conn.execute(
                """
                INSERT INTO llm_cache (prompt_hash, system_hash, model, provider, temperature, json_mode, response)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prompt_hash, system_hash, model, provider, temperature, json_mode) DO UPDATE SET
                    response = excluded.response,
                    created_at = CURRENT_TIMESTAMP
                """,
                (prompt_hash, system_hash, model, provider, temperature, json_mode_int, response),
            )
            await self._conn.commit()
            logger.debug(
                "Cached response: %s/%s (prompt: %s...)",
                provider,
                model,
                prompt[:30],
            )
            return True
        except Exception as e:
            logger.warning("Failed to cache response: %s", e)
            return False

    async def stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache size info.
        """
        if not self._conn:
            return {}

        try:
            cursor = await self._conn.execute(
                "SELECT COUNT(*) as total, COUNT(DISTINCT provider) as providers, COUNT(DISTINCT model) as models FROM llm_cache"
            )
            row = await cursor.fetchone()
            await cursor.close()

            if row:
                return {
                    "total_cached_responses": row[0],
                    "unique_providers": row[1],
                    "unique_models": row[2],
                }
            return {}
        except Exception as e:
            logger.warning("Failed to get cache stats: %s", e)
            return {}

    async def clear(self, provider: str | None = None, model: str | None = None) -> int:
        """Clear cache entries.

        Args:
            provider: If specified, only clear entries for this provider.
            model: If specified, only clear entries for this model.

        Returns:
            Number of entries deleted.
        """
        if not self._conn:
            return 0

        try:
            if provider is None and model is None:
                # Clear all
                cursor = await self._conn.execute("DELETE FROM llm_cache")
                logger.info("Cleared entire LLM cache")
            elif provider and model:
                # Clear specific provider + model
                cursor = await self._conn.execute(
                    "DELETE FROM llm_cache WHERE provider = ? AND model = ?",
                    (provider, model),
                )
                logger.info("Cleared cache for %s/%s", provider, model)
            elif provider:
                # Clear by provider
                cursor = await self._conn.execute(
                    "DELETE FROM llm_cache WHERE provider = ?",
                    (provider,),
                )
                logger.info("Cleared cache for provider %s", provider)
            else:
                # Clear by model
                cursor = await self._conn.execute(
                    "DELETE FROM llm_cache WHERE model = ?",
                    (model,),
                )
                logger.info("Cleared cache for model %s", model)

            await self._conn.commit()
            deleted = cursor.rowcount
            await cursor.close()
            return deleted
        except Exception as e:
            logger.warning("Failed to clear cache: %s", e)
            return 0

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False
            logger.debug("LLM cache connection closed")
