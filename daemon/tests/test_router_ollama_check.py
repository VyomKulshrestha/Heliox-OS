"""A call to a cloud provider never waits on asking Ollama whether it runs.

On Windows a refused connection to 127.0.0.1 takes about 2 s (the system retries the connect), and the router asked
Ollama's port before every cloud call, so a person on a cloud provider without Ollama waited 2 s before every plan
and every other model call. A local call asks once. Fakes only."""

from __future__ import annotations

import asyncio

import pytest


class _Ollama:
    def __init__(self, up: bool) -> None:
        self.up = up
        self.asked = 0

    async def is_available(self) -> bool:
        self.asked += 1
        return self.up

    async def list_models(self) -> list[str]:
        return ["llama3.2:3b"]

    async def generate(self, *_args, **_kwargs) -> str:
        return "local answer"

    async def close(self) -> None:
        return None


class _Cloud:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return "cloud answer"

    async def close(self) -> None:
        return None


class _NoCache:
    async def get(self, *_args, **_kwargs):
        return None

    async def set(self, *_args, **_kwargs) -> None:
        return None


def _router(provider: str, ollama_up: bool):
    from pilot.config import PilotConfig
    from pilot.models.router import ModelRouter

    router = ModelRouter.__new__(ModelRouter)
    config = PilotConfig()
    config.model.provider = provider
    config.model.cloud_provider = "claude" if provider == "cloud" else ""
    config.model.cloud_model = "claude-haiku-5-5"
    router._config = config
    router._ollama = _Ollama(ollama_up)
    router._cloud = _Cloud() if provider == "cloud" else None
    router._cache = _NoCache()
    router._cloud_unavailable_until = 0.0
    router._cloud_unavailable_reason = ""
    router._resolved_ollama_model = None
    router._llamacpp = None
    router._try_llamacpp = lambda: False

    class _Limiter:
        async def acquire(self) -> None:
            return None

    router._rate_limiter = _Limiter()
    router._local_llm_semaphore = asyncio.Semaphore(2)
    router._local_llm_timeout = 30
    return router


@pytest.mark.asyncio
async def test_a_cloud_call_never_waits_on_the_ollama_check() -> None:
    router = _router("cloud", ollama_up=False)
    assert await router._generate_with_cache([{"role": "user", "content": "x"}]) == "cloud answer"
    assert router._ollama.asked == 0
    assert router._cloud.calls == 1


@pytest.mark.asyncio
async def test_a_local_call_asks_ollama_once() -> None:
    router = _router("local", ollama_up=True)
    assert await router._generate_with_cache([{"role": "user", "content": "x"}]) == "local answer"
    assert router._ollama.asked == 1
