"""Fakes for pilot.system.outbound: a resolver and a client, so no request or name lookup leaves a test.

``install(monkeypatch, client)`` makes every outbound request go to ``client`` (anything with an async
``request(method, url, **kwargs)``), or, with ``handler``, to a real httpx client over a MockTransport, which
builds each request as it would go on the wire and sends nothing. Names resolve to ``answers`` (by default to
93.184.216.34, a public address); an address written as an address resolves to itself. By default the client sees the address as the step wrote it:
the judged address the request was sent to is put back into the URL, so a fake can route by host. Pass
``as_sent=True`` to see what outbound really hands the HTTP client (the judged address, the Host header and the TLS
name).
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any

import httpx

PUBLIC = "93.184.216.34"


class ClientContext:
    def __init__(self, client: object) -> None:
        self._client = client

    async def __aenter__(self) -> object:
        return self._client

    async def __aexit__(self, *_args: object) -> None:
        return None


class _AsWritten:
    """Hands the fake the URL as the step wrote it, without the Host header and TLS name outbound added."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def request(self, method: str, url: str, **kwargs: Any) -> Any:
        headers = dict(kwargs.get("headers") or {})
        host = headers.pop("Host", None)
        extensions = kwargs.pop("extensions", None)
        if host is not None:
            url = str(httpx.URL(url).copy_with(netloc=host.encode("ascii")))
            kwargs["headers"] = headers
        elif extensions:
            kwargs["extensions"] = extensions
        return await self._client.request(method, url, **kwargs)


def resolver(answers: dict[str, Any] | None = None, default: list[str] | None = None, calls: list | None = None):
    """A fake ``getaddrinfo``. ``answers`` maps a name to its addresses, or to a list of answers given one per lookup
    (a name whose answer changes). A name not listed gets ``default`` ([PUBLIC] unless given; [] means it does not
    resolve). An address resolves to itself."""
    table = dict(answers or {})
    fallback = [PUBLIC] if default is None else default

    def _getaddrinfo(host: str, port: int, family: int = 0, type: int = 0, *_rest: Any, **_kwargs: Any) -> list:
        if calls is not None:
            calls.append(host)
        try:
            ipaddress.ip_address(host.partition("%")[0])
            found = [host]
        except ValueError:
            entry = table.get(host, fallback)
            if entry and isinstance(entry[0], list):
                found = entry.pop(0) if len(entry) > 1 else entry[0]
            else:
                found = list(entry)
        if not found:
            raise socket.gaierror(11001, "getaddrinfo failed")
        return [
            (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
            for address in found
        ]

    return _getaddrinfo


def install(
    monkeypatch: Any,
    client: Any = None,
    *,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
    answers: dict[str, Any] | None = None,
    default: list[str] | None = None,
    as_sent: bool = False,
    calls: list | None = None,
    client_kwargs: list | None = None,
) -> None:
    from pilot.system import outbound

    def _factory(*_args: Any, **kwargs: Any) -> ClientContext:
        if client_kwargs is not None:
            client_kwargs.append(kwargs)
        if handler is not None:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)  # type: ignore[return-value]
        return ClientContext(client if as_sent else _AsWritten(client))

    monkeypatch.setattr(outbound.PilotConfig, "load", lambda: object())
    monkeypatch.setattr(outbound, "proxy_for", lambda *_a, **_k: None)
    monkeypatch.setattr(outbound, "_nat64_cache", [])
    monkeypatch.setattr(outbound, "_getaddrinfo", resolver(answers, default, calls))
    monkeypatch.setattr(outbound, "create_httpx_client", _factory)
