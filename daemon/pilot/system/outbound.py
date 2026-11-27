"""The one way a plan step's request leaves this computer.

Every request a step makes goes through ``fetch``: api_request, api_scrape, download_file, webhook, Slack, Discord
and GitHub sends, the checks that read a sent message or a written resource back, and feed fetches. It:

- accepts http and https only, with no credentials written into the address;
- looks the host name up once and judges every address the name gives (pilot.security.addresses). The connection
  goes to a judged address: the request names the host in its Host header and in the TLS name indication, and the
  certificate is checked against that name, so a name that answers differently a moment later cannot move it;
- refuses addresses on this computer, the local network, carrier-grade NAT, link-local and cloud metadata addresses,
  in every spelling, unless the step carries ``allow_private_network`` (which makes it need the person's approval);
- follows redirects itself, one hop at a time, with the same check on each; never from https to http; never round a
  loop; and once a local address was allowed, never on to another host.

Through a configured proxy the proxy makes the connection by name; the name is still judged by this computer's own
lookup first.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import socket
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import urljoin

import httpx

from pilot.config import PilotConfig
from pilot.security import addresses
from pilot.security.secret_urls import refuse_masked
from pilot.system.http_client import create_httpx_client, proxy_for
from pilot.timeouts import bounded

# The real client factory; a test that replaced it with a fake reaches no server (system_guard.check_http).
_REAL_CLIENT_FACTORY = create_httpx_client
# The resolver. Each request looks each host up once, here; the suite replaces it with a fake.
_getaddrinfo = socket.getaddrinfo

# Reads a streamed response's body while its connection is open (fetch_streamed).
Consumer = Callable[[Any], Awaitable[Any]]

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 5
_SENSITIVE_REDIRECT_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization"})
_BODY_HEADERS = frozenset({"content-length", "content-type", "transfer-encoding"})

# The failure kind of a refusal (pilot.failure_sentences): asking again without the person's approval cannot change it.
REFUSAL_KIND = "local_address_refused"
REFUSAL = "I don't fetch from addresses on this computer or your own network unless you ask for that address yourself."


class OutboundRefused(ValueError):
    """A request to an address on this computer, the local network or a metadata service that nobody approved."""

    failure_kind = REFUSAL_KIND

    def __init__(self, host: str, reason: str, address: str = "", *, text: str | None = None) -> None:
        shown = host if not address or address == host else f"{host} ({address})"
        super().__init__(text or f"{REFUSAL} {shown} is {reason}.")
        self.host = host
        self.reason = reason


def _redirect_refused(host: str) -> OutboundRefused:
    return OutboundRefused(
        host,
        "another host",
        text=f"The address you approved on this computer or your own network sent me on to {host}, which you did "
        "not approve, so I did not follow it.",
    )


_ssl_context_cache: list[ssl.SSLContext] = []


def _ssl_context() -> ssl.SSLContext:
    """Certificate checks on, with the system's SSL_CERT_FILE and SSL_CERT_DIR honoured as before."""
    if not _ssl_context_cache:
        _ssl_context_cache.append(httpx.create_ssl_context())
    return _ssl_context_cache[0]


def parse(url: str) -> httpx.URL:
    """The URL as the HTTP client reads it, refused unless it is http or https with a host and no credentials."""
    refuse_masked(url)
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        if "port" in str(exc).lower():
            raise ValueError("Outbound URL contains an invalid port") from exc
        raise ValueError(f"Outbound URL is not a valid address: {exc}") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Outbound URLs must use http:// or https://")
    if not parsed.raw_host:
        raise ValueError("Outbound URL must include a hostname")
    if parsed.userinfo:
        raise ValueError("Credentials must be supplied as request authentication, not embedded in the URL")
    return parsed


def _host(parsed: httpx.URL) -> str:
    return parsed.raw_host.decode("ascii")


def _port(parsed: httpx.URL) -> int:
    return parsed.port or (443 if parsed.scheme == "https" else 80)


async def _resolve(host: str, port: int, timeout: float | None) -> list[str]:
    try:
        found = await bounded(asyncio.to_thread(_getaddrinfo, host, port, type=socket.SOCK_STREAM), timeout)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not resolve outbound hostname {host!r}") from exc
    answers = list(dict.fromkeys(str(entry[4][0]) for entry in found))
    if not answers:
        raise ValueError(f"Could not resolve outbound hostname {host!r}")
    return answers


_IPV4ONLY = {ipaddress.IPv4Address("192.0.0.170"), ipaddress.IPv4Address("192.0.0.171")}
_nat64_cache: list[tuple[ipaddress.IPv6Network, ...]] = []


async def _nat64_prefixes(timeout: float | None) -> tuple[ipaddress.IPv6Network, ...]:
    """The NAT64 prefixes this network announces (RFC 7050: the AAAA answers for ipv4only.arpa), looked up once.

    An address in one of them is judged by the IPv4 address it carries. None found means the well-known and local-use
    prefixes only."""
    if _nat64_cache:
        return _nat64_cache[0]
    prefixes: list[ipaddress.IPv6Network] = []
    try:
        found = await bounded(
            asyncio.to_thread(_getaddrinfo, "ipv4only.arpa", 80, socket.AF_INET6, socket.SOCK_STREAM), timeout
        )
    except (OSError, UnicodeError, TimeoutError):
        found = []
    for entry in found:
        try:
            address = ipaddress.IPv6Address(str(entry[4][0]).partition("%")[0])
        except ValueError:
            continue
        for length in (96, 64, 56, 48, 40, 32):
            network = ipaddress.IPv6Network((int(address) >> (128 - length) << (128 - length), length))
            if addresses.nat64_ipv4(address, network) in _IPV4ONLY:
                prefixes.append(network)
                break
    _nat64_cache.append(tuple(dict.fromkeys(prefixes)))
    return _nat64_cache[0]


async def addresses_for(
    parsed: httpx.URL, *, allow_private_network: bool, timeout: float | None, known: dict[str, list[str]]
) -> list[str]:
    """The judged addresses to connect to for this URL's host, looked up once per request (``known``)."""
    host = _host(parsed)
    if host in known:
        return known[host]
    if not allow_private_network:
        reason = addresses.host_refusal(host)
        if reason:
            raise OutboundRefused(host, reason)
    found = await _resolve(host, _port(parsed), timeout)
    if not allow_private_network:
        prefixes: tuple[ipaddress.IPv6Network, ...] = ()
        if any(":" in answer for answer in found):
            prefixes = await _nat64_prefixes(timeout)
        for answer in found:
            try:
                address = ipaddress.ip_address(answer)
            except ValueError:
                raise OutboundRefused(host, addresses.UNREADABLE, answer) from None
            reason = addresses.address_refusal(address, prefixes)
            if reason:
                raise OutboundRefused(host, reason, answer)
    known[host] = found
    return found


def _restore_url(response: Any, url: str) -> None:
    """The response names the address that was asked for, not the judged address it was sent to."""
    if isinstance(response, httpx.Response):
        response.request.url = httpx.URL(url)


async def _send_once(
    method: str,
    parsed: httpx.URL,
    url: str,
    found: list[str],
    headers: dict[str, str],
    request_kwargs: dict[str, Any],
    timeout: float | None,
    config: object,
    consume: Consumer | None = None,
) -> tuple[Any, Any]:
    """(response, what ``consume`` made of it). With ``consume`` the body is streamed: ``consume`` reads it while the
    connection is open, and is not called for a redirect, whose body is never read."""
    host = _host(parsed)
    proxy = proxy_for(config, parsed.scheme, host)
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "follow_redirects": False,
        "verify": _ssl_context(),
        "trust_env": False,
    }
    if proxy:
        client_kwargs["proxy"] = proxy
    literal = addresses.parse_address(host) is not None
    last_error: Exception | None = None
    for address in [None] if proxy else found:
        target = url
        sent_headers = {name: value for name, value in headers.items() if name.lower() != "host"}
        extensions: dict[str, Any] = {}
        bare = (address or "").partition("%")[0]
        if address is not None and bare != host:
            target = str(parsed.copy_with(host=bare))
            sent_headers["Host"] = parsed.netloc.decode("ascii")
            if parsed.scheme == "https" and not literal:
                extensions["sni_hostname"] = host
        call: dict[str, Any] = dict(request_kwargs)
        call["headers"] = sent_headers
        if extensions:
            call["extensions"] = extensions
        try:
            async with create_httpx_client(config, **client_kwargs) as client:
                if consume is None:
                    response = await client.request(method, target, **call)
                    _restore_url(response, url)
                    return response, None
                async with client.stream(method, target, **call) as response:
                    _restore_url(response, url)
                    if response.status_code in REDIRECT_STATUSES and response.headers.get("location"):
                        return response, None
                    return response, await consume(response)
        except httpx.ConnectError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def _redirect_method(method: str, status_code: int) -> str:
    if status_code == 303 and method != "HEAD":
        return "GET"
    if status_code in {301, 302} and method == "POST":
        return "GET"
    return method


def _same_place(first: httpx.URL, other: httpx.URL) -> bool:
    """The same host and port, or the same host moving from http to https on the standard ports."""
    if _host(first) != _host(other):
        return False
    if _port(first) == _port(other):
        return True
    return first.scheme == "http" and other.scheme == "https" and first.port is None and other.port is None


async def fetch(method: str, url: str, **options: Any) -> Any:
    """Send one request for a plan step and follow its redirects, every hop checked. Returns the final response.

    Takes the options of ``_exchange``. Raises ``OutboundRefused`` for an address on this computer or the local
    network that was not approved.
    """
    response, _ = await _exchange(method, url, **options)
    return response


@contextlib.asynccontextmanager
async def body_chunks(response: Any) -> AsyncIterator[AsyncIterator[bytes]]:
    """The decoded body of a streamed ``response``, in chunks. httpx's ``aiter_bytes`` and ``aiter_raw`` each iterate
    an inner async generator that closing them does not close, so a read cut short (a size limit, an error) left those
    for the garbage collector. Here the transport's stream iterator and the decoding are each closed, in order, on every
    exit; the response itself is closed by the ``client.stream`` block around the read."""
    if hasattr(response, "_content"):  # a body already read in full (a response made with its content)

        async def whole() -> AsyncIterator[bytes]:
            if response.content:
                yield response.content

        chunks = whole()
        try:
            yield chunks
        finally:
            await chunks.aclose()
        return
    stream = response.stream.__aiter__()
    decoder = response._get_content_decoder()  # httpx's decoder for the response's content-encoding

    async def decoded() -> AsyncIterator[bytes]:
        async for part in stream:
            data = decoder.decode(part)
            if data:
                yield data
        tail = decoder.flush()
        if tail:
            yield tail

    chunks = decoded()
    try:
        yield chunks
    finally:
        await chunks.aclose()
        closing = getattr(stream, "aclose", None)
        if closing is not None:
            await closing()


async def fetch_streamed(method: str, url: str, consume: Consumer, **options: Any) -> Any:
    """As ``fetch``, with the final response's body streamed to ``consume`` (an async callable given the open
    response) instead of read into memory. Returns what ``consume`` returned."""
    _, consumed = await _exchange(method, url, consume=consume, **options)
    return consumed


async def _exchange(
    method: str,
    url: str,
    *,
    consume: Consumer | None = None,
    allow_private_network: bool = False,
    headers: dict[str, str] | None = None,
    params: Any = None,
    json: Any = None,
    content: Any = None,
    auth: Any = None,
    timeout: float | None = 30,
    config: object | None = None,
) -> tuple[Any, Any]:
    method = method.upper()
    current = parse(url)
    from pilot.system.system_guard import check_http

    check_http(method, _host(current), real_client=create_httpx_client is _REAL_CLIENT_FACTORY)
    config = config if config is not None else PilotConfig.load()
    request_kwargs: dict[str, Any] = {
        name: value
        for name, value in (("params", params), ("json", json), ("content", content), ("auth", auth))
        if value is not None
    }
    sent_headers = dict(headers or {})
    first = current
    current_url = url
    current_method = method
    known: dict[str, list[str]] = {}
    seen = {(method, str(current))}

    for hop in range(MAX_REDIRECTS + 1):
        found = await addresses_for(current, allow_private_network=allow_private_network, timeout=timeout, known=known)
        response, consumed = await _send_once(
            current_method, current, current_url, found, sent_headers, request_kwargs, timeout, config, consume
        )
        location = response.headers.get("location")
        if response.status_code not in REDIRECT_STATUSES or not location:
            return response, consumed
        if hop == MAX_REDIRECTS:
            raise ValueError(f"Outbound request exceeded {MAX_REDIRECTS} redirects")

        next_url = urljoin(current_url, location)
        following = parse(next_url)
        if not allow_private_network:
            reason = addresses.host_refusal(_host(following))
            if reason:
                raise OutboundRefused(_host(following), reason)
        if current.scheme == "https" and following.scheme == "http":
            raise ValueError(
                f"{_host(current)} sent me from a secure (https) address to an insecure (http) one, so I did not "
                "follow it."
            )
        if allow_private_network and not _same_place(first, following):
            raise _redirect_refused(_host(following))
        next_method = _redirect_method(current_method, response.status_code)
        if (next_method, str(following)) in seen:
            raise ValueError(f"{_host(current)} sent me round a redirect loop, so I stopped.")
        seen.add((next_method, str(following)))

        if (current.scheme, _host(current), _port(current)) != (following.scheme, _host(following), _port(following)):
            request_kwargs.pop("auth", None)
            sent_headers = {
                key: value for key, value in sent_headers.items() if key.lower() not in _SENSITIVE_REDIRECT_HEADERS
            }
        if next_method != current_method:
            request_kwargs.pop("content", None)
            request_kwargs.pop("json", None)
            sent_headers = {key: value for key, value in sent_headers.items() if key.lower() not in _BODY_HEADERS}
        request_kwargs.pop("params", None)
        current, current_url, current_method = following, next_url, next_method

    raise AssertionError("redirect loop must return or raise")
