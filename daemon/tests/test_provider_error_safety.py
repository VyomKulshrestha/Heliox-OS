import asyncio

import httpx
import pytest

import pilot.models.cloud as cloud_module
from pilot.config import PilotConfig
from pilot.models.cloud import CloudClient, safe_provider_error
from pilot.models.router import ModelRouter


class _Vault:
    async def get_key(self, provider: str) -> str | None:
        return "super-secret-api-key" if provider == "gemini" else None


class _BackupVault:
    async def get_key(self, provider: str) -> str | None:
        return {
            "gemini": "invalid-primary-key",
            "gemini_backup_1": "healthy-backup-key",
        }.get(provider)


def _http_error(status: int, *, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/demo:generateContent?key=super-secret-api-key",
    )
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError("request failed", request=request, response=response)


def test_http_provider_error_never_contains_endpoint_or_api_key():
    message = safe_provider_error(_http_error(429), "gemini")

    assert message == "Gemini API unavailable (429): quota or rate limit reached."
    assert "super-secret" not in message
    assert "googleapis.com" not in message
    assert "?key=" not in message


def test_invalid_key_error_is_actionable_without_echoing_provider_body():
    message = safe_provider_error(
        _http_error(400, body='{"error":{"status":"API_KEY_INVALID","message":"API key not valid"}}'),
        "gemini",
    )

    assert message == "Gemini API unavailable (400): the configured API key is invalid."


def test_openrouter_credit_error_is_actionable_without_echoing_provider_body():
    message = safe_provider_error(
        _http_error(402, body='{"error":{"message":"Insufficient credits for sk-or-v1-secret"}}'),
        "openrouter",
    )

    assert message == "OpenRouter API unavailable (402): the account or API key has insufficient credits."
    assert "sk-or" not in message


@pytest.mark.asyncio
async def test_cloud_client_raises_sanitized_error_without_original_exception_chain():
    config = PilotConfig()
    config.model.cloud_provider = "gemini"
    client = CloudClient(config, _Vault())

    async def _fail(*args, **kwargs):
        raise _http_error(429)

    client._call_gemini_native = _fail

    with pytest.raises(RuntimeError) as raised:
        await client.generate("hello")

    assert str(raised.value) == "Gemini API unavailable (429): quota or rate limit reached."
    assert raised.value.__suppress_context__ is True
    await client.close()


@pytest.mark.asyncio
async def test_cloud_client_rotates_past_an_invalid_key():
    config = PilotConfig()
    config.model.cloud_provider = "gemini"
    client = CloudClient(config, _BackupVault())
    attempted_keys: list[str] = []

    async def _generate(api_key, *args, **kwargs):
        attempted_keys.append(api_key)
        if api_key == "invalid-primary-key":
            raise _http_error(400, body='{"error":{"status":"API_KEY_INVALID"}}')
        return "healthy response"

    client._call_gemini_native = _generate

    assert await client.generate("hello") == "healthy response"
    assert attempted_keys == ["invalid-primary-key", "healthy-backup-key"]
    await client.close()


@pytest.mark.asyncio
async def test_cloud_client_skips_a_known_bad_key_on_the_next_generation():
    config = PilotConfig()
    config.model.cloud_provider = "gemini"
    client = CloudClient(config, _BackupVault())
    attempted_keys: list[str] = []

    async def _generate(api_key, *args, **kwargs):
        attempted_keys.append(api_key)
        if api_key == "invalid-primary-key":
            raise _http_error(400, body='{"error":{"status":"API_KEY_INVALID"}}')
        return "healthy response"

    client._call_gemini_native = _generate

    assert await client.generate("first") == "healthy response"
    assert await client.generate("second") == "healthy response"
    assert attempted_keys == [
        "invalid-primary-key",
        "healthy-backup-key",
        "healthy-backup-key",
    ]
    await client.close()


@pytest.mark.asyncio
async def test_cloud_client_rotates_past_a_transient_timeout():
    config = PilotConfig()
    config.model.cloud_provider = "gemini"
    client = CloudClient(config, _BackupVault())
    attempted_keys: list[str] = []

    async def _generate(api_key, *args, **kwargs):
        attempted_keys.append(api_key)
        if api_key == "invalid-primary-key":
            raise httpx.ReadTimeout("provider stalled")
        return "healthy response"

    client._call_gemini_native = _generate

    assert await client.generate("hello") == "healthy response"
    assert attempted_keys == ["invalid-primary-key", "healthy-backup-key"]
    await client.close()


@pytest.mark.asyncio
async def test_cloud_client_bounds_a_hung_generation(monkeypatch):
    config = PilotConfig()
    config.model.cloud_provider = "gemini"
    client = CloudClient(config, _Vault())
    monkeypatch.setattr(cloud_module, "CLOUD_GENERATION_BUDGET_SECONDS", 0.01)

    async def _hang(*args, **kwargs):
        await asyncio.sleep(1)
        return "too late"

    client._call_gemini_native = _hang

    with pytest.raises(RuntimeError, match="request timed out"):
        await client.generate("hello")
    await client.close()


@pytest.mark.asyncio
async def test_cloud_client_allows_a_healthy_generation_within_budget(monkeypatch):
    config = PilotConfig()
    config.model.cloud_provider = "gemini"
    client = CloudClient(config, _Vault())
    monkeypatch.setattr(cloud_module, "CLOUD_GENERATION_BUDGET_SECONDS", 0.08)

    async def _respond(*args, **kwargs):
        await asyncio.sleep(0.03)
        return "healthy response"

    client._call_gemini_native = _respond

    assert await client.generate("hello") == "healthy response"
    await client.close()


def test_cloud_circuit_breaker_fails_fast_with_sanitized_reason():
    config = PilotConfig()
    config.model.cloud_provider = "gemini"
    router = ModelRouter(config, _Vault())
    router._open_cloud_circuit("Gemini API unavailable (429): quota or rate limit reached.")

    with pytest.raises(RuntimeError) as raised:
        router._raise_if_cloud_circuit_open()

    assert "Retry shortly" in str(raised.value)
    assert "key=" not in str(raised.value)


@pytest.mark.parametrize(
    ("status", "body", "provider", "name"),
    [
        (
            400,
            '{"error":{"type":"invalid_request_error","message":"Your credit balance is too low"}}',
            "claude",
            "Claude",
        ),
        (
            429,
            '{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}',
            "openai",
            "OpenAI",
        ),
    ],
)
def test_an_account_out_of_credit_says_so_rather_than_rate_limit_or_rejected(status, body, provider, name):
    message = safe_provider_error(_http_error(status, body=body), provider)
    assert message.endswith(f"({status}): the account or API key has insufficient credits.")
    assert message.startswith(name)
