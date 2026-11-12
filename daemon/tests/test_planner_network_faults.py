"""The planner's cloud calls when the network goes away: a bounded wait, a plain sentence, and the next request works
without a restart. A stub transport plays the provider; no real call is made."""

import asyncio

import httpx
import pytest

import pilot.models.cloud as cloud_module
from pilot.config import PilotConfig
from pilot.models.cloud import CloudClient

OK = {"content": [{"type": "text", "text": "a plan"}], "stop_reason": "end_turn", "usage": {"input_tokens": 3}}


class _Vault:
    async def get_key(self, provider: str):
        return "test-key" if provider == "claude" else None


def _client(behaviour: list) -> CloudClient:
    config = PilotConfig()
    config.model.cloud_provider = "claude"
    client = CloudClient(config, _Vault())

    async def handler(request: httpx.Request) -> httpx.Response:
        mode = behaviour.pop(0) if behaviour else "ok"
        if mode == "dns":
            raise httpx.ConnectError("[Errno 11001] getaddrinfo failed", request=request)
        if mode == "html":
            return httpx.Response(200, html="<html><body>Sign in to the hotel Wi-Fi</body></html>")
        if mode == "stall":
            await asyncio.sleep(30)
        if mode == "cut":
            raise httpx.RemoteProtocolError("peer closed connection", request=request)
        return httpx.Response(200, json=OK)

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.parametrize(
    ("mode", "sentence"),
    [
        ("dns", "Claude API unavailable: could not connect (no network, or the address could not be found)."),
        ("cut", "Claude API unavailable: the connection closed before the answer was complete."),
    ],
)
async def test_a_lost_connection_is_said_plainly_and_the_next_request_works(mode, sentence):
    client = _client([mode])
    try:
        with pytest.raises(RuntimeError) as raised:
            await client.generate("plan this")
        assert str(raised.value) == sentence
        assert await client.generate("plan this") == "a plan"  # no restart needed
    finally:
        await client.close()


async def test_a_proxy_or_sign_in_page_answering_with_html_is_named():
    client = _client(["html"])
    try:
        with pytest.raises(RuntimeError) as raised:
            await client.generate("plan this")
        message = str(raised.value)
        assert message.startswith("Claude API unavailable: the answer was not from the API (text/html)")
        assert "sign-in page for this network" in message
        assert await client.generate("plan this") == "a plan"
    finally:
        await client.close()


async def test_a_stream_that_sends_no_bytes_is_given_up_on_within_the_budget(monkeypatch):
    monkeypatch.setattr(cloud_module, "CLOUD_GENERATION_BUDGET_SECONDS", 0.2)
    client = _client(["stall"])
    try:
        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(client.generate("plan this"), 10)  # the 10 s only matters if the bound is broken
        assert str(raised.value) == "Claude API unavailable: the request timed out."
        assert await client.generate("plan this") == "a plan"
    finally:
        await client.close()
