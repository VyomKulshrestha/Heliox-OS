"""Tests for pilot.models.cloud.CloudClient.

No prior test coverage existed for CloudClient at all (every existing
router/budget test mocks the whole class out, see
test_router_budget_gating.py). These use httpx.MockTransport to actually
exercise the request-building/response-parsing code, focused on the new
"meta" (Muse Spark 1.1) provider plus one regression check for "openai"
since both share `_call_openai_compat`.
"""

from __future__ import annotations

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.models.cloud import DEFAULT_MODELS, PROVIDER_ENDPOINTS, CloudClient


class _FakeVault:
    def __init__(self, keys: dict[str, str]):
        self._keys = keys

    async def get_key(self, provider: str) -> str | None:
        return self._keys.get(provider)


def _client_with_mock_transport(config: PilotConfig, vault: _FakeVault, handler) -> CloudClient:
    client = CloudClient(config, vault)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _openai_format_response(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


class TestMetaProvider:
    @pytest.mark.asyncio
    async def test_generate_hits_the_registered_meta_endpoint_with_bearer_auth(self):
        config = PilotConfig()
        config.model.cloud_provider = "meta"
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = request.read()
            return _openai_format_response("hello from muse spark")

        client = _client_with_mock_transport(config, _FakeVault({"meta": "test-key"}), handler)
        result = await client.generate("hi")

        assert result == "hello from muse spark"
        assert seen["url"] == PROVIDER_ENDPOINTS["meta"]
        assert seen["auth"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_generate_uses_default_model_when_cloud_model_unset(self):
        config = PilotConfig()
        config.model.cloud_provider = "meta"
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen["model"] = json.loads(request.read())["model"]
            return _openai_format_response("ok")

        client = _client_with_mock_transport(config, _FakeVault({"meta": "test-key"}), handler)
        await client.generate("hi")

        assert seen["model"] == DEFAULT_MODELS["meta"]

    @pytest.mark.asyncio
    async def test_generate_respects_explicit_cloud_model_override(self):
        config = PilotConfig()
        config.model.cloud_provider = "meta"
        config.model.cloud_model = "muse-spark-1.1-mini"
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen["model"] = json.loads(request.read())["model"]
            return _openai_format_response("ok")

        client = _client_with_mock_transport(config, _FakeVault({"meta": "test-key"}), handler)
        await client.generate("hi")

        assert seen["model"] == "muse-spark-1.1-mini"

    @pytest.mark.asyncio
    async def test_generate_raises_when_no_meta_key_configured(self):
        config = PilotConfig()
        config.model.cloud_provider = "meta"
        client = _client_with_mock_transport(config, _FakeVault({}), lambda r: _openai_format_response(""))

        with pytest.raises(RuntimeError, match="No API key configured for meta"):
            await client.generate("hi")


class TestOpenAiProviderRegression:
    @pytest.mark.asyncio
    async def test_generate_still_hits_openai_endpoint(self):
        """Sanity check that sharing _call_openai_compat with meta didn't
        change openai's own request shape."""
        config = PilotConfig()
        config.model.cloud_provider = "openai"
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return _openai_format_response("hello from gpt")

        client = _client_with_mock_transport(config, _FakeVault({"openai": "test-key"}), handler)
        result = await client.generate("hi")

        assert result == "hello from gpt"
        assert seen["url"] == PROVIDER_ENDPOINTS["openai"]
