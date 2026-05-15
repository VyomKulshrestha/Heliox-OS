#!/usr/bin/env python3
"""Practical demonstration of the LLM cache in action.

This script shows:
  1. Cache miss on first call
  2. Cache hit on identical second call
  3. Cache miss when model changes
  4. Cache miss when provider changes
  5. Cache statistics
"""

import asyncio
import tempfile
from pathlib import Path

from pilot.models.cache import LLMCache


async def main():
    """Demonstrate cache functionality."""
    print("=" * 70)
    print("LLM Cache Demonstration")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "demo_cache.db"
        cache = LLMCache(db_path)
        await cache.initialize()

        # Scenario 1: Cache miss
        print("\n[1] CACHE MISS: First request for OpenAI GPT-4o")
        print("-" * 70)
        prompt = "Explain the concept of recursion in Python"
        system = "You are a helpful programming tutor"

        result = await cache.get(
            prompt,
            model="gpt-4o",
            provider="openai",
            temperature=0.1,
            json_mode=False,
            system=system,
        )
        if result is None:
            print("✗ Cache miss (as expected)")
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Model: gpt-4o | Provider: openai")
            print("  Action: Would call LLM API here")
        else:
            print("✓ Unexpected cache hit")

        # Store simulated LLM response
        print("\n[2] Storing LLM response in cache...")
        response = (
            "Recursion is a programming technique where a function calls itself. "
            "A recursive function must have a base case to avoid infinite loops. "
            "Classic examples: factorial, fibonacci, tree traversal."
        )
        success = await cache.set(
            prompt,
            model="gpt-4o",
            provider="openai",
            temperature=0.1,
            json_mode=False,
            response=response,
            system=system,
        )
        print(f"✓ Response stored (success={success})")

        # Scenario 2: Cache hit with identical parameters
        print("\n[3] CACHE HIT: Second request with identical parameters")
        print("-" * 70)
        result = await cache.get(
            prompt,
            model="gpt-4o",
            provider="openai",
            temperature=0.1,
            json_mode=False,
            system=system,
        )
        if result:
            print("✓ Cache hit!")
            print(f"  Cached response: {result[:60]}...")
            print("  Action: Returned immediately from cache (~0.5ms)")
        else:
            print("✗ Unexpected cache miss")

        # Scenario 3: Different model = cache miss
        print("\n[4] CACHE MISS: Same prompt, different model (gpt-3.5-turbo)")
        print("-" * 70)
        result = await cache.get(
            prompt,
            model="gpt-3.5-turbo",
            provider="openai",
            temperature=0.1,
            json_mode=False,
            system=system,
        )
        if result is None:
            print("✓ Cache miss (as expected - different model)")
            print(f"  Model difference: gpt-4o → gpt-3.5-turbo")
            print("  Action: Would call LLM API with different model")
        else:
            print("✗ Unexpected cache hit (models shouldn't share cache!)")

        # Scenario 4: Different provider = cache miss
        print("\n[5] CACHE MISS: Same prompt, different provider (gemini)")
        print("-" * 70)
        result = await cache.get(
            prompt,
            model="gemini-1.5-pro",
            provider="gemini",
            temperature=0.1,
            json_mode=False,
            system=system,
        )
        if result is None:
            print("✓ Cache miss (as expected - different provider)")
            print(f"  Provider difference: openai → gemini")
            print("  Action: Would call LLM API with different provider")
        else:
            print("✗ Unexpected cache hit (providers shouldn't share cache!)")

        # Scenario 5: Different temperature = cache miss
        print("\n[6] CACHE MISS: Same prompt, different temperature (0.7)")
        print("-" * 70)
        result = await cache.get(
            prompt,
            model="gpt-4o",
            provider="openai",
            temperature=0.7,  # Different!
            json_mode=False,
            system=system,
        )
        if result is None:
            print("✓ Cache miss (as expected - different temperature)")
            print(f"  Temperature difference: 0.1 → 0.7")
            print("  Action: Would call LLM API with different temperature")
        else:
            print("✗ Unexpected cache hit (temperatures shouldn't share cache!)")

        # Scenario 6: Store additional responses
        print("\n[7] Storing additional responses for different models...")
        print("-" * 70)

        # GPT-3.5-turbo response
        gpt35_response = "Recursion: a function that calls itself with a base case."
        await cache.set(
            prompt,
            model="gpt-3.5-turbo",
            provider="openai",
            temperature=0.1,
            json_mode=False,
            response=gpt35_response,
            system=system,
        )
        print("✓ Stored response for gpt-3.5-turbo")

        # Gemini response
        gemini_response = (
            "Recursion is when a function invokes itself to solve a problem by breaking it into smaller subproblems."
        )
        await cache.set(
            prompt,
            model="gemini-1.5-pro",
            provider="gemini",
            temperature=0.1,
            json_mode=False,
            response=gemini_response,
            system=system,
        )
        print("✓ Stored response for gemini-1.5-pro")

        # Ollama response
        ollama_response = "Recursion = function calling itself. Need base case."
        await cache.set(
            prompt,
            model="llama3.1:8b",
            provider="ollama",
            temperature=0.1,
            json_mode=False,
            response=ollama_response,
            system=system,
        )
        print("✓ Stored response for llama3.1:8b (ollama)")

        # Scenario 7: Cache statistics
        print("\n[8] CACHE STATISTICS")
        print("-" * 70)
        stats = await cache.stats()
        print(f"Total cached responses: {stats.get('total_cached_responses', 0)}")
        print(f"Unique providers: {stats.get('unique_providers', 0)}")
        print(f"Unique models: {stats.get('unique_models', 0)}")

        # Scenario 8: Verify each model still has separate cache
        print("\n[9] CACHE ISOLATION: Verify each model has its own response")
        print("-" * 70)

        result_gpt4 = await cache.get(
            prompt,
            model="gpt-4o",
            provider="openai",
            temperature=0.1,
            json_mode=False,
            system=system,
        )
        result_gpt35 = await cache.get(
            prompt,
            model="gpt-3.5-turbo",
            provider="openai",
            temperature=0.1,
            json_mode=False,
            system=system,
        )
        result_gemini = await cache.get(
            prompt,
            model="gemini-1.5-pro",
            provider="gemini",
            temperature=0.1,
            json_mode=False,
            system=system,
        )
        result_ollama = await cache.get(
            prompt,
            model="llama3.1:8b",
            provider="ollama",
            temperature=0.1,
            json_mode=False,
            system=system,
        )

        if (
            result_gpt4
            and result_gpt35
            and result_gemini
            and result_ollama
            and result_gpt4 != result_gpt35
            and result_gpt35 != result_gemini
        ):
            print("✓ Each model has its own cached response")
            print(f"  GPT-4o:      {result_gpt4[:40]}...")
            print(f"  GPT-3.5:     {result_gpt35[:40]}...")
            print(f"  Gemini:      {result_gemini[:40]}...")
            print(f"  Ollama:      {result_ollama[:40]}...")
        else:
            print("✗ Cache isolation verification failed")

        # Scenario 9: Clear cache for specific provider
        print("\n[10] CACHE CLEARING: Remove OpenAI entries")
        print("-" * 70)
        deleted = await cache.clear(provider="openai")
        print(f"✓ Deleted {deleted} OpenAI cache entries")

        # Verify OpenAI entries are gone
        result_gpt4_after = await cache.get(
            prompt,
            model="gpt-4o",
            provider="openai",
            temperature=0.1,
            json_mode=False,
            system=system,
        )
        result_gemini_after = await cache.get(
            prompt,
            model="gemini-1.5-pro",
            provider="gemini",
            temperature=0.1,
            json_mode=False,
            system=system,
        )

        print(f"✓ OpenAI cache cleared: {result_gpt4_after is None}")
        print(f"✓ Gemini cache intact: {result_gemini_after is not None}")

        # Final statistics
        print("\n[11] FINAL STATISTICS")
        print("-" * 70)
        stats = await cache.stats()
        print(f"Remaining cached responses: {stats.get('total_cached_responses', 0)}")
        print(f"Remaining providers: {stats.get('unique_providers', 0)}")
        print(f"Remaining models: {stats.get('unique_models', 0)}")

        await cache.close()

        print("\n" + "=" * 70)
        print("✓ LLM Cache demonstration complete!")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
