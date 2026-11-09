from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.config import PilotConfig
from pilot.models.ollama import OllamaClient
from pilot.models.router import ModelRouter


class _StreamingResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        yield '{"response":"ready"}'


@pytest.mark.asyncio
async def test_ollama_applies_configured_idle_timeout_to_every_request_path():
    config = PilotConfig()
    config.model.idle_unload_seconds = 37
    transport = MagicMock()
    response = MagicMock(status_code=200)
    response.json.side_effect = [
        {"response": "generated"},
        {"message": {"content": "chatted"}},
    ]
    transport.post = AsyncMock(return_value=response)
    transport.stream.return_value = _StreamingResponse()

    with patch("pilot.models.ollama.create_httpx_client", return_value=transport):
        client = OllamaClient(config=config)
    client._model_max["model"] = 32768  # its context length is known, so no /api/show call here

    assert await client.generate("model", "prompt") == "generated"
    assert await client.generate("model", "prompt", stream_callback=AsyncMock()) == "ready"
    assert await client.generate("model", [{"role": "user", "content": "hello"}]) == "chatted"

    assert transport.post.await_args_list[0].kwargs["json"]["keep_alive"] == 37
    assert transport.post.await_args_list[0].kwargs["json"]["options"]["num_ctx"] == 4096
    assert transport.stream.call_args.kwargs["json"]["keep_alive"] == 37
    assert transport.post.await_args_list[1].kwargs["json"]["keep_alive"] == 37

    config.model.idle_unload_seconds = 9
    response.json.side_effect = [{"response": "updated"}]
    assert await client.generate("model", "new prompt") == "updated"
    assert transport.post.await_args.kwargs["json"]["keep_alive"] == 9


@pytest.mark.asyncio
async def test_llamacpp_unloads_after_configured_idle_period():
    config = PilotConfig()
    config.model.idle_unload_seconds = 0
    client = MagicMock()
    client.generate = AsyncMock(return_value="complete")
    router = object.__new__(ModelRouter)
    router._config = config
    router._llamacpp = client
    router._llamacpp_active_calls = 0
    router._llamacpp_idle_task = None
    router._llamacpp_reload_pending = False

    assert await router._llamacpp_generate("prompt") == "complete"
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    client.unload.assert_called_once_with()
    await router._cancel_llamacpp_idle_unload()


@pytest.mark.asyncio
async def test_llamacpp_stays_loaded_until_all_concurrent_calls_finish():
    config = PilotConfig()
    config.model.idle_unload_seconds = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def generate(prompt, **_kwargs):
        if prompt == "first":
            first_started.set()
            await release_first.wait()
        return prompt

    client = MagicMock()
    client.generate = AsyncMock(side_effect=generate)
    router = object.__new__(ModelRouter)
    router._config = config
    router._llamacpp = client
    router._llamacpp_active_calls = 0
    router._llamacpp_idle_task = None
    router._llamacpp_reload_pending = False

    first = asyncio.create_task(router._llamacpp_generate("first"))
    await first_started.wait()
    assert await router._llamacpp_generate("second") == "second"
    await asyncio.sleep(0)
    client.unload.assert_not_called()

    release_first.set()
    assert await first == "first"
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    client.unload.assert_called_once_with()
    await router._cancel_llamacpp_idle_unload()


@pytest.mark.asyncio
async def test_llamacpp_vram_change_defers_reload_until_active_call_finishes():
    config = PilotConfig()
    started = asyncio.Event()
    release = asyncio.Event()

    async def generate(*_args, **_kwargs):
        started.set()
        await release.wait()
        return "done"

    client = MagicMock()
    client.generate = AsyncMock(side_effect=generate)
    router = object.__new__(ModelRouter)
    router._config = config
    router._llamacpp = client
    router._llamacpp_active_calls = 0
    router._llamacpp_idle_task = None
    router._llamacpp_reload_pending = False

    inference = asyncio.create_task(router._llamacpp_generate("prompt"))
    await started.wait()
    await router.reconfigure({"gpu_memory_limit_mb"})

    assert router._llamacpp_reload_pending is True
    client.unload.assert_not_called()
    release.set()
    assert await inference == "done"
    client.unload.assert_called_once_with()
    assert router._llamacpp is None


@pytest.mark.asyncio
async def test_model_router_closes_every_provider_and_unloads_embedded_model():
    router = object.__new__(ModelRouter)
    router._llamacpp_idle_task = None
    router._llamacpp_reload_pending = False
    router._llamacpp = MagicMock()
    router._subscription = MagicMock(close=AsyncMock())
    router._cache = MagicMock(close=AsyncMock())
    router._ollama = MagicMock(close=AsyncMock())
    router._cloud = MagicMock(close=AsyncMock())

    await router.close()

    router._subscription.close.assert_awaited_once_with()
    router._cache.close.assert_awaited_once_with()
    router._ollama.close.assert_awaited_once_with()
    router._cloud.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_model_reconfigure_refreshes_cached_and_stateful_backends():
    config = PilotConfig()
    router = object.__new__(ModelRouter)
    router._config = config
    router._vault = MagicMock()
    router._resolved_ollama_model = "old-model"
    router._ollama = MagicMock(close=AsyncMock())
    router._cloud = MagicMock(close=AsyncMock())
    router._subscription = MagicMock(close=AsyncMock())
    router._cloud_unavailable_reason = "old failure"
    router._cloud_unavailable_until = 100.0
    router._llamacpp = MagicMock()
    router._llamacpp_active_calls = 0
    router._llamacpp_idle_task = None
    router._llamacpp_reload_pending = False
    config.model.ollama_base_url = "http://127.0.0.1:22434"
    config.model.cloud_provider = ""

    with (
        patch("pilot.models.router.OllamaClient") as ollama_type,
        patch("pilot.models.router.SubscriptionCLIClient") as subscription_type,
    ):
        await router.reconfigure(
            {
                "ollama_model",
                "ollama_base_url",
                "cloud_provider",
                "subscription_model",
                "gpu_memory_limit_mb",
            }
        )

    assert router._resolved_ollama_model is None
    ollama_type.assert_called_once_with("http://127.0.0.1:22434", config)
    subscription_type.assert_called_once_with(config)
    assert router._cloud is None
    assert router._llamacpp is None
    assert router._cloud_unavailable_reason == ""
    assert router._cloud_unavailable_until == 0.0


@pytest.mark.asyncio
async def test_model_reconfigure_updates_live_limits_and_budgets():
    config = PilotConfig()
    router = object.__new__(ModelRouter)
    router._config = config
    router._rate_limiter = MagicMock()
    router._budget_tracker = MagicMock()

    await router.reconfigure({"rate_limit_rpm", "max_tokens_per_task"})

    router._rate_limiter.reconfigure.assert_called_once_with(config.model)
    router._budget_tracker.reconfigure.assert_called_once_with(config.model)
