"""Ollama HTTP API client."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("pilot.models.ollama")

DEFAULT_TIMEOUT = 600.0  # 10 minutes — local LLMs can be slow for complex plans


class OllamaModelNotFoundError(RuntimeError):
    def __init__(self, model: str, available: list[str]) -> None:
        self.model = model
        self.available = available
        avail_str = ", ".join(available) if available else "none"
        super().__init__(
            f"Model '{model}' is not installed in Ollama. "
            f"Available models: {avail_str}. "
            f"Run 'ollama pull {model}' to install it, or change the model in Settings."
        )


class OllamaClient:
    """Client for the Ollama local inference server."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except (httpx.ConnectError, httpx.TimeoutException):
            return []

    async def generate(
        self,
        model: str,
        prompt: str | list[dict[str, Any]],
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        stream: bool = False,
        stream_callback: callable | None = None,
    ) -> str:
        """Generate a completion. Returns the full response text.

        If stream_callback is provided, tokens are streamed via the callback
        and the full response is returned at the end.
        """
        if isinstance(prompt, list):
            return await self.chat(
                model,
                prompt,
                json_mode=json_mode,
                temperature=temperature,
                stream_callback=stream_callback,
            )

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": stream_callback is not None,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        if stream_callback is not None:
            return await self._generate_stream(model, prompt, system, json_mode, temperature, stream_callback)

        resp = await self._client.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )

        if resp.status_code == 404:
            available = await self.list_models()
            raise OllamaModelNotFoundError(model, available)

        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    async def _generate_stream(
        self,
        model: str,
        prompt: str,
        system: str,
        json_mode: bool,
        temperature: float,
        stream_callback: callable,
    ) -> str:
        """Generate with streaming - calls stream_callback for each token."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        full_response = ""
        async with self._client.stream("POST", f"{self._base_url}/api/generate", json=payload) as resp:
            if resp.status_code == 404:
                available = await self.list_models()
                raise OllamaModelNotFoundError(model, available)

            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = __import__("json").loads(line)
                    token = data.get("response", "")
                    if token:
                        full_response += token
                        await stream_callback(token)
                except Exception:
                    continue

        return full_response

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        temperature: float = 0.1,
        stream_callback: callable | None = None,
    ) -> str:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": stream_callback is not None,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        if stream_callback is not None:
            full_response = ""
            async with self._client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                if resp.status_code == 404:
                    available = await self.list_models()
                    raise OllamaModelNotFoundError(model, available)
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = __import__("json").loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                            await stream_callback(token)
                    except Exception:
                        continue
            return full_response

        resp = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        if resp.status_code == 404:
            available = await self.list_models()
            raise OllamaModelNotFoundError(model, available)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def close(self) -> None:
        await self._client.aclose()
