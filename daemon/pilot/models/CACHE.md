# Local LLM Response Cache

A fast, SQLite-backed caching layer for LLM responses to reduce API costs and accelerate
testing and development workflows.

## Overview

The LLM cache is an optional, transparent layer that sits between the application and
LLM backends. It automatically caches successful LLM responses and returns cached
results for exact-match prompts, significantly reducing API calls and speeding up testing.

## Features

**Exact-match caching**: Cache is keyed by prompt content, not just prompt structure

**Model-specific**: Different models (gpt-4o vs gpt-3.5-turbo) have separate cache entries

**Provider-specific**: Different providers (OpenAI, Gemini, Claude, Ollama) don't share cache

**Parameter-aware**: Cache keys include temperature, system prompt, and JSON mode flags

**Fast lookups**: SQLite with indexed queries for sub-millisecond cache hits

**Transparent**: Works seamlessly with existing ModelRouter, no API changes

**Modular**: Can be extended with TTL, invalidation, and eviction policies

## Cache Key Components

The cache key is deterministically generated from:

| Component | Details |
|-----------|---------|
| **Prompt Hash** | SHA256 hash of user prompt |
| **System Hash** | SHA256 hash of system prompt (if provided) |
| **Model** | Exact model string (e.g., `gpt-4o`, `llama3.1:8b`, `gemini-1.5-pro`) |
| **Provider** | LLM provider (e.g., `openai`, `ollama`, `gemini`, `claude`) |
| **Temperature** | Temperature parameter (0.0-2.0) |
| **JSON Mode** | Whether JSON mode is enabled (0 or 1) |

This multi-dimensional key ensures:
- **Consistency**: Same prompt + model always returns the same cached response
- **Isolation**: Different models/providers have separate cache entries
- **Correctness**: Temperature and JSON mode changes invalidate the cache

## Architecture

```
┌─────────────┐
│ Application │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   ModelRouter       │ ◄─── initialize() → loads cache
├─────────────────────┤
│ generate(prompt)    │
│   ▼                 │
│ Check cache ────────┤─── Cache Hit ───► return cached response
│   │                 │
│   ├─ Miss ──┐       │
│   │         │       │
│   │    ▼    ▼       │
│   │  Rate Limit     │
│   │    │            │
│   │    ▼            │
│   │  Call LLM       │
│   │    │            │
│   │    ▼            │
│   │  Store in Cache │
│   │    │            │
│   └────┴────────────┼─── return response
└─────────────────────┘
```


## Database Schema

```sql
CREATE TABLE llm_cache (
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
);

CREATE INDEX idx_cache_key ON llm_cache(
    prompt_hash, system_hash, model, provider, temperature, json_mode
);
CREATE INDEX idx_provider ON llm_cache(provider);
CREATE INDEX idx_model ON llm_cache(model);
```
