"""The planner's cloud connection is opened while the person types or speaks to Heliox, so the first plan after an
idle period does not set it up (0.13 to 0.15 s with claude-haiku-5-5, measured 27 November).

The warm-up is a HEAD to the API's own address: no key, no body, nothing billed. At most once per idle period, never
while a request is in flight or another warm-up runs, never for a model that is not a cloud provider, a request
never waits for it, and stopping cancels it. A local model server that does not answer is reported in one sentence
after a short wait. Fakes only and a counted clock: no network."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.models import cloud
from pilot.models.cloud import WARM_SKIP_SECONDS, CloudClient


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeHttp:
    """Records each request; a HEAD can be held open until ``release`` is set."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.release = asyncio.Event()
        self.release.set()
        self.closed = False

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": method, "url": url, **kwargs})
        await self.release.wait()
        return httpx.Response(404, request=httpx.Request(method, url))

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": "POST", "url": url})
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    async def aclose(self) -> None:
        self.closed = True


def client(provider: str = "claude") -> tuple[CloudClient, FakeHttp, Clock]:
    config = PilotConfig()
    config.model.provider = "cloud"
    config.model.cloud_provider = provider
    item = CloudClient.__new__(CloudClient)
    item._config = config
    item._client = FakeHttp()
    item._clock = Clock()
    return item, item._client, item._clock


async def settle(item: CloudClient) -> None:
    if item._warming is not None:
        await item._warming


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "url"),
    [
        ("claude", "https://api.anthropic.com/"),
        ("openai", "https://api.openai.com/"),
        ("gemini", "https://generativelanguage.googleapis.com/"),
        ("openrouter", "https://openrouter.ai/"),
    ],
)
async def test_a_warm_up_is_a_head_to_the_providers_address_with_nothing_in_it(provider, url) -> None:
    item, http, _clock = client(provider)
    assert item.warm() is True
    await settle(item)
    assert http.requests == [{"method": "HEAD", "url": url, "timeout": cloud.WARM_TIMEOUT_SECONDS}]


@pytest.mark.asyncio
async def test_once_per_idle_period() -> None:
    item, http, clock = client()
    assert item.warm() is True
    assert item.warm() is False  # one is already under way
    await settle(item)
    clock.now += WARM_SKIP_SECONDS - 1
    assert item.warm() is False  # the connection it opened is still open
    clock.now += 2
    assert item.warm() is True  # idle again
    await settle(item)
    assert [r["method"] for r in http.requests] == ["HEAD", "HEAD"]


@pytest.mark.asyncio
async def test_never_beside_a_request_and_a_request_counts_as_use() -> None:
    item, http, clock = client()
    held = asyncio.Event()
    entered = asyncio.Event()

    async def slow_post(url: str, **kwargs: Any) -> httpx.Response:
        entered.set()
        await held.wait()
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    http.post = slow_post
    request = asyncio.ensure_future(item._post("claude", "m", "https://api.anthropic.com/v1/messages", {}, {}))
    await entered.wait()
    assert item.warm() is False  # a request is in flight
    held.set()
    await request
    clock.now += 1
    assert item.warm() is False  # the request just used the connection
    assert http.requests == []


@pytest.mark.asyncio
async def test_a_request_never_waits_for_a_warm_up() -> None:
    item, http, _clock = client()
    http.release.clear()  # the HEAD does not answer
    assert item.warm() is True
    await asyncio.sleep(0)
    response = await item._post("claude", "m", "https://api.anthropic.com/v1/messages", {}, {})
    assert response.status_code == 200
    assert not item._warming.done()
    http.release.set()
    await settle(item)


@pytest.mark.asyncio
async def test_stopping_cancels_a_warm_up_under_way() -> None:
    item, http, _clock = client()
    http.release.clear()
    assert item.warm() is True
    await asyncio.sleep(0)
    await item.close()
    assert item._warming.cancelled() and http.closed


@pytest.mark.asyncio
async def test_a_failed_warm_up_is_silent_and_does_not_count_as_use() -> None:
    item, http, _clock = client()

    async def offline(method: str, url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("no network")

    http.request = offline
    assert item.warm() is True
    await settle(item)
    assert item.warm() is True  # nothing was opened, so the next sign may try again


def router(provider: str) -> Any:
    from pilot.models.router import ModelRouter

    item = ModelRouter.__new__(ModelRouter)
    config = PilotConfig()
    config.model.provider = provider
    item._config = config
    item._cloud = SimpleNamespace(warmed=0)

    def warm() -> bool:
        item._cloud.warmed += 1
        return True

    item._cloud.warm = warm
    return item


@pytest.mark.parametrize(("provider", "warmed"), [("cloud", 1), ("ollama", 0), ("local", 0), ("subscription", 0)])
def test_only_a_cloud_planner_is_warmed(provider, warmed) -> None:
    item = router(provider)
    assert item.warm_planner_connection() is bool(warmed)
    assert item._cloud.warmed == warmed


@pytest.mark.asyncio
async def test_typing_and_speech_said_to_heliox_warm_it_and_focus_does_not(monkeypatch) -> None:
    import pilot.conversation as conversation
    from pilot.server import PilotServer
    from pilot.system import local_tts_worker, voice

    monkeypatch.setattr(local_tts_worker, "manager", lambda: SimpleNamespace(running=True, failing=False))
    monkeypatch.setattr(voice, "warm_output_line", lambda *a, **k: None)
    monkeypatch.setattr(conversation, "layer_for", lambda _s: SimpleNamespace(prewarm=lambda: None))
    config = PilotConfig()
    config.save = lambda *a, **k: None
    config.voice.tts_engine = "os_native"
    item = PilotServer(config)
    warmed: list[str] = []
    item._model_router = SimpleNamespace(warm_planner_connection=lambda: warmed.append("x") or True)

    await item._handle_voice_prepare({"reason": "focus"}, None)
    assert warmed == []
    await item._handle_voice_prepare({"reason": "typing"}, None)
    assert warmed == ["x"]
    assert item.warm_for_speech("overheard") is False and warmed == ["x"]
    item.warm_for_speech("wake")
    assert warmed == ["x", "x"]


@pytest.mark.asyncio
async def test_a_local_model_server_that_does_not_answer_is_said_in_one_sentence(monkeypatch) -> None:
    from pilot.models.ollama import AVAILABILITY_CONNECT_LOCAL, OllamaClient
    from pilot.models.router import ModelRouter

    config = PilotConfig()
    config.model.provider = "ollama"
    ollama = OllamaClient(config=config)
    timeouts: list[Any] = []

    async def refused(url: str, **kwargs: Any) -> httpx.Response:
        timeouts.append(kwargs.get("timeout"))
        raise httpx.ConnectTimeout("no answer")

    monkeypatch.setattr(ollama._client, "get", refused)
    item = ModelRouter.__new__(ModelRouter)
    item._config = config
    item._ollama = ollama
    item._cloud = None
    item._resolved_ollama_model = None
    item._try_llamacpp = lambda: False

    class NoCache:
        async def get(self, *_a: Any, **_k: Any) -> None:
            return None

    class Limiter:
        async def acquire(self) -> None:
            return None

    item._cache = NoCache()
    item._rate_limiter = Limiter()
    try:
        with pytest.raises(RuntimeError) as raised:
            await item._generate_with_cache([{"role": "user", "content": "x"}])
    finally:
        await ollama.close()
    assert str(raised.value) == (
        "The local model server (Ollama at http://127.0.0.1:11434) is not answering. "
        "Start Ollama, or choose a cloud model in Settings."
    )
    assert len(timeouts) == 1  # asked once, not once per fallback
    assert timeouts[0].connect == AVAILABILITY_CONNECT_LOCAL
