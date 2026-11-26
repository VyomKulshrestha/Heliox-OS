"""Whether a model's requests stay on this computer.

What Heliox puts into a model request (the profile, to-dos and reminders in the person's own words, the end of the
text being typed, the foreground app) depends on where the model runs. A model counts as on this computer only when
its requests cannot leave it:

- the provider is Ollama or the local backend (``model.provider`` "ollama" or "local"), and
- the Ollama address (``model.ollama_base_url``) names this computer: a loopback address (``127.0.0.0/8``, ``::1``,
  an IPv4-mapped ``::ffff:127.x.x.x``), the unspecified address (a connection to ``0.0.0.0`` or ``::`` reaches this
  computer or nothing), ``localhost`` or a name under ``.localhost``, or a Unix socket; or a host name that resolved
  to loopback addresses only.

Anything else is another computer, whatever the provider is called: a private network address (``192.168.x.x``,
``10.x.x.x``), ``host.docker.internal``, a WSL or container bridge address, a host name that resolves to any address
that is not loopback, a name whose lookup failed and a name not looked up yet. A model on another computer gets
exactly what a cloud model gets: the same settings (``conversation.share_profile``, ``share_screen_text``) decide,
and the same shortened forms are sent.

A host name is judged by what it resolved to, never by a lookup on the reply path: the first time a name is asked
about after the daemon starts (Settings asks when it opens) it is looked up once in the background, and until that
answer is in the name counts as another computer. Saving the model settings looks every name up again.

Two things cannot be seen from here and are the person's own set-up: a loopback port that forwards to a server
elsewhere (an SSH tunnel, a proxy), and a name the system's hosts file points somewhere unusual after the lookup.

Nothing here logs an address: an address can carry a user name and password.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("pilot.models.model_place")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_PROVIDERS = frozenset({"ollama", "local"})
LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
SOCKET_SCHEMES = frozenset({"unix", "http+unix", "https+unix"})

Resolver = Callable[[str], Iterable[str]]


def host_of(url: Any) -> str:
    """The host a URL or a bare host name points at, lower case, without brackets, a user name or a port; "" for
    nothing."""
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text if "://" in text else f"//{text}")
        return (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def address_on_this_computer(address: str) -> bool | None:
    """True for a loopback or unspecified address, False for any other address, None when ``address`` is a name."""
    text = address.strip().strip("[]").split("%", 1)[0]  # an IPv6 zone ("fe80::1%eth0") names no other host
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return None
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_loopback or ip.is_unspecified)


def literal_place(url: Any) -> bool | None:
    """Where ``url`` points without a lookup: True for this computer, False for another one, None when only a lookup
    can tell (a host name). An empty address is the default one, on this computer."""
    text = str(url or "").strip()
    if not text:
        return True
    if text.split("://", 1)[0].casefold() in SOCKET_SCHEMES and "://" in text:
        return True
    host = host_of(text)
    if not host:
        return False  # an address with no host in it: nothing to vouch for
    if host in LOOPBACK_NAMES or host.endswith(".localhost"):
        return True
    return address_on_this_computer(host)


def _system_resolve(host: str) -> list[str]:
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)]


def _in_background(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="model-place-lookup", daemon=True).start()


class Places:
    """Where model addresses point, with host names looked up off the caller's thread and remembered.

    ``resolve`` turns a name into addresses (``socket.getaddrinfo`` by default; tests pass their own). ``run``
    starts a lookup (a background thread by default; tests run it at once)."""

    def __init__(self, resolve: Resolver | None = None, run: Callable[[Callable[[], None]], None] | None = None):
        self._resolve = resolve or _system_resolve
        self._run = run or _in_background
        self._known: dict[str, bool] = {}
        self._asked: set[str] = set()
        self._lock = threading.Lock()

    def on_this_computer(self, url: Any) -> bool:
        """Whether requests to ``url`` stay on this computer. Never waits: a name not looked up yet counts as another
        computer, and its lookup starts."""
        found = literal_place(url)
        if found is not None:
            return found
        host = host_of(url)
        with self._lock:
            if host in self._known:
                return self._known[host]
            start = host not in self._asked
            self._asked.add(host)
        if start:
            self._run(lambda: self.learn(host))
        with self._lock:
            return self._known.get(host, False)

    def learn(self, host: str) -> bool:
        """Look ``host`` up now (blocking) and remember the answer: True only when it resolved, and every address
        is on this computer."""
        try:
            addresses = [str(address) for address in self._resolve(host)]
            local = bool(addresses) and all(address_on_this_computer(address) is True for address in addresses)
        except Exception as error:  # a failed lookup is another computer
            logger.debug("A model host name could not be looked up (%s)", type(error).__name__)
            local = False
        with self._lock:
            self._known[host] = local
        return local

    def forget(self) -> None:
        """Look every name up again the next time it is asked about (after the model settings change)."""
        with self._lock:
            self._known.clear()
            self._asked.clear()


PLACES = Places()


def endpoint_on_this_computer(url: Any, places: Places | None = None) -> bool:
    return (places or PLACES).on_this_computer(url or DEFAULT_OLLAMA_URL)


def model_on_this_computer(model: Any, places: Places | None = None) -> bool:
    """Whether the configured model (``config.model``, a dataclass or a dict) runs on this computer: Ollama or the
    local backend at an address on this computer. Every other provider, and Ollama elsewhere, is False."""
    get = model.get if isinstance(model, dict) else lambda key, default=None: getattr(model, key, default)
    if str(get("provider", "") or "").casefold() not in OLLAMA_PROVIDERS:
        return False
    return endpoint_on_this_computer(get("ollama_base_url", "") or DEFAULT_OLLAMA_URL, places)


def model_elsewhere(model: Any, places: Places | None = None) -> str:
    """For Settings: the host of an Ollama or local model on another computer, else "". Only the host name: never
    the rest of the address."""
    get = model.get if isinstance(model, dict) else lambda key, default=None: getattr(model, key, default)
    if str(get("provider", "") or "").casefold() not in OLLAMA_PROVIDERS or model_on_this_computer(model, places):
        return ""
    return host_of(get("ollama_base_url", "")) or "another computer"


__all__ = [
    "DEFAULT_OLLAMA_URL",
    "OLLAMA_PROVIDERS",
    "PLACES",
    "Places",
    "address_on_this_computer",
    "endpoint_on_this_computer",
    "host_of",
    "literal_place",
    "model_elsewhere",
    "model_on_this_computer",
]
