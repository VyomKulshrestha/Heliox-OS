"""Start-up readiness: what the daemon is doing while it starts, and holding requests until it can answer them.

On a slow or loaded machine start-up takes minutes (importing the model stack, opening the stores, loading speech
models). The window used to see nothing but a refused connection ("Reconnecting...") all that time. Now the daemon
listens first, answers ``ping``, ``health``, ``ready`` and ``startup_status`` at once, and tells every window which
stage it is in ("starting: opening Heliox's records"). Anything else the person asks while it starts waits until
start-up is done and is then answered as usual; nothing is refused with an error that only means "not ready yet".
If start-up fails, the waiting requests get that failure in plain words.

``StartupProgress`` is pure bookkeeping with an injected clock; the server reports stages to it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("pilot.system.startup")

# How long a request waits for start-up before it is answered with the reason start-up has not finished.
WAIT_FOR_READY_SECONDS = 600.0


class StartupProgress:
    """starting -> (waiting_for_storage ->) ready, or failed. Stages are said in plain words."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self.state = "starting"
        self.stage = "starting"
        self.reason = ""
        self.stages: list[tuple[str, float]] = []  # (words, seconds since start) for the log and the bench
        self._ready = asyncio.Event()
        self.on_change: Callable[[dict[str, Any]], None] | None = None

    @property
    def is_ready(self) -> bool:
        return self.state == "ready"

    def begin(self, words: str) -> None:
        """A new stage starts, e.g. "opening Heliox's records"."""
        elapsed = self._clock() - self._started
        self.stages.append((words, elapsed))
        self.stage = words
        logger.info("Start-up: %s (%.1f s since start)", words, elapsed)
        self._changed()

    def waiting_for_storage(self, reason: str) -> None:
        self.state = "waiting_for_storage"
        self.reason = reason
        self.stage = "waiting until the data folder can be written"
        self._changed()

    def ready(self) -> None:
        self.state = "ready"
        self.stage = ""
        self.reason = ""
        self.stages.append(("ready", self._clock() - self._started))
        logger.info("Start-up finished in %.1f s", self._clock() - self._started)
        self._event().set()
        self._changed()

    def failed(self, reason: str) -> None:
        self.state = "failed"
        self.reason = reason
        self._event().set()
        self._changed()

    async def wait_ready(self, timeout: float = WAIT_FOR_READY_SECONDS) -> str:
        """Wait until start-up has finished. Returns "" when ready, else the plain reason it is not."""
        if self.state == "ready":
            return ""
        if self.state != "failed":
            try:
                await asyncio.wait_for(self._event().wait(), timeout=timeout)
            except TimeoutError:
                return f"Heliox is still starting ({self.stage}). Try again in a moment."
        if self.state == "failed":
            return f"Heliox could not finish starting: {self.reason}"
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "stage": self.stage,
            "reason": self.reason,
            "stages_done": [words for words, _ in self.stages],
        }

    def _event(self) -> asyncio.Event:
        return self._ready

    def _changed(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change(self.snapshot())
            except Exception:
                logger.debug("Could not announce the start-up stage", exc_info=True)


class AlreadyRunningError(RuntimeError):
    """The daemon's port is taken, most likely by another copy of Heliox. Raised before this copy touches anything."""

    def __init__(self, port: int) -> None:
        super().__init__(
            f"Port {port} is already in use, most likely by another copy of Heliox, so this copy did not start and "
            "left the running one alone. Close the other copy (or the program using that port) and start again."
        )
        self.port = port


def is_address_in_use(error: OSError) -> bool:
    """EADDRINUSE on Linux (98) and macOS (48), WSAEADDRINUSE on Windows (10048)."""
    import errno

    return error.errno in {errno.EADDRINUSE, 48, 98, 10048} or getattr(error, "winerror", None) == 10048


def claim_address(host: str, port: int) -> list[socket.socket]:
    """Bind the daemon's address without listening, so a second copy is found before anything is written.

    A bound socket that is not listening refuses connections (on Windows the connect times out after its SYN
    retries), so no client can reach this copy until ``serve`` starts listening on these sockets, which happens after
    the token files are fresh. Every address ``host`` resolves to is claimed, as ``loop.create_server(host, port)``
    would: "localhost" is usually ::1 and 127.0.0.1, and both are served. With port 0 the first address's port is used
    for the rest, so all of them share one port.

    Windows: SO_EXCLUSIVEADDRUSE, never SO_REUSEADDR (there it lets two processes bind one port).
    Linux and macOS: SO_REUSEADDR, as asyncio sets, so a restart is not refused while the old socket is in TIME_WAIT.

    Raises ``AlreadyRunningError`` when the address is taken; any socket already bound is closed on every failure.
    """
    infos = socket.getaddrinfo(host or None, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)
    claimed: list[socket.socket] = []
    seen: set[tuple] = set()
    try:
        for family, kind, proto, _name, address in infos:
            if (family, address[:2]) in seen:
                continue
            seen.add((family, address[:2]))
            sock = socket.socket(family, kind, proto)
            claimed.append(sock)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            elif os.name == "posix":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and hasattr(socket, "IPPROTO_IPV6"):
                # As asyncio does: the IPv6 socket serves IPv6 only, so it does not collide with the IPv4 one.
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            if port == 0 and len(claimed) > 1:
                address = (address[0], claimed[0].getsockname()[1], *address[2:])
            try:
                sock.bind(address)
            except OSError as error:
                if is_address_in_use(error):
                    raise AlreadyRunningError(port or address[1]) from error
                raise
    except BaseException:
        for sock in claimed:
            sock.close()
        raise
    return claimed


def hold_port_lock(folder: Any, port: int) -> Any:
    """Linux and macOS: hold ``daemon-<port>.lock`` in ``folder`` for this copy's life (released when it exits).

    Linux lets two SO_REUSEADDR sockets bind one address while neither listens, so two copies started in the same
    instant can both pass ``claim_address``. The lock settles it before either writes its tokens. Windows needs no lock
    (the exclusive bind refuses the second claim) and port 0 never collides; both return None.
    Raises ``AlreadyRunningError`` when another copy holds the lock.
    """
    if os.name != "posix" or port == 0:
        return None
    import fcntl
    from pathlib import Path

    Path(folder).mkdir(parents=True, exist_ok=True)
    handle = open(Path(folder) / f"daemon-{port}.lock", "a")  # noqa: SIM115 - held open for the process's life
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise AlreadyRunningError(port) from error
    return handle


class ServerGroup:
    """One websockets server per claimed address, seen as one: ``sockets``, ``close()`` and ``wait_closed()``."""

    def __init__(self, servers: list[Any]) -> None:
        self.servers = servers

    @property
    def sockets(self) -> tuple[socket.socket, ...]:
        return tuple(sock for server in self.servers for sock in server.sockets)

    def close(self, *args: Any, **kwargs: Any) -> None:
        for server in self.servers:
            server.close(*args, **kwargs)

    async def wait_closed(self) -> None:
        for server in self.servers:
            await server.wait_closed()


async def serve_claimed(
    sockets: list[socket.socket], serve: Callable[..., Any], handler: Callable[..., Any], **kwargs: Any
) -> ServerGroup:
    """Start listening on the claimed sockets. On any failure every socket and every started server is closed.

    ``serve`` is ``websockets.serve``. A listen that fails with address-in-use (Linux allows two non-listening
    SO_REUSEADDR binds, so two copies started in the same instant can both claim) raises ``AlreadyRunningError``.
    """
    started: list[Any] = []
    port = sockets[0].getsockname()[1] if sockets else 0
    try:
        for sock in sockets:
            started.append(await serve(handler, sock=sock, **kwargs))
    except BaseException as error:
        for server in started:
            server.close()
        for sock in sockets:
            sock.close()
        if isinstance(error, OSError) and is_address_in_use(error):
            raise AlreadyRunningError(port) from error
        raise
    return ServerGroup(started)
