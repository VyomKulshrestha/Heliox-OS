"""LAN peer discovery via mDNS/DNS-SD (zeroconf).

Broadcasts a ``_helioxos._tcp.local.`` service record so every Heliox OS
instance on the same LAN can find each other without any manual configuration.

Usage
-----
    discovery = PeerDiscovery(port=8786, instance_id="abc123")
    discovery.on_peer_found = lambda info: ...
    discovery.on_peer_lost  = lambda peer_id: ...
    await discovery.start()
    ...
    await discovery.stop()

Graceful degradation
--------------------
If ``zeroconf`` is not installed the module logs a warning and all methods
become no-ops so the rest of the daemon starts normally.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("pilot.network.peer_discovery")

_ZEROCONF_AVAILABLE = False
try:
    from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf
    from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

    _ZEROCONF_AVAILABLE = True
except ImportError:
    logger.warning("zeroconf not installed — LAN peer discovery disabled. Install with: pip install zeroconf")

_SERVICE_TYPE = "_helioxos._tcp.local."


@dataclass
class PeerInfo:
    """Metadata about a discovered peer instance."""

    peer_id: str  # unique instance UUID
    host: str  # IPv4 address
    port: int  # P2P WebSocket port
    hostname: str = ""  # human-readable hostname


class PeerDiscovery:
    """Advertises this instance on the LAN and discovers other instances.

    Callbacks
    ---------
    on_peer_found(PeerInfo)  — called when a new peer is discovered
    on_peer_lost(peer_id)    — called when a peer deregisters or times out
    """

    def __init__(self, port: int, instance_id: str | None = None) -> None:
        self._port = port
        self._instance_id = instance_id or str(uuid.uuid4())[:8]
        self._zc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._service_info: ServiceInfo | None = None
        self._known_peers: dict[str, PeerInfo] = {}

        # Public callbacks — set by HelioxMesh
        self.on_peer_found: Callable[[PeerInfo], None] | None = None
        self.on_peer_lost: Callable[[str], None] | None = None

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def known_peers(self) -> dict[str, PeerInfo]:
        return dict(self._known_peers)

    async def start(self) -> None:
        """Start advertising and browsing."""
        if not _ZEROCONF_AVAILABLE:
            logger.warning("PeerDiscovery.start() skipped — zeroconf not available")
            return

        self._zc = AsyncZeroconf()

        # Register our own service
        hostname = socket.gethostname()
        local_ip = _get_local_ip()
        service_name = f"helioxos-{self._instance_id}.{_SERVICE_TYPE}"

        self._service_info = ServiceInfo(
            type_=_SERVICE_TYPE,
            name=service_name,
            addresses=[socket.inet_aton(local_ip)],
            port=self._port,
            properties={
                "id": self._instance_id.encode(),
                "host": hostname.encode(),
            },
            server=f"{hostname}.local.",
        )

        await self._zc.async_register_service(self._service_info)
        logger.info(
            "PeerDiscovery: registered as %s @ %s:%d",
            self._instance_id,
            local_ip,
            self._port,
        )

        # Browse for other instances
        self._browser = AsyncServiceBrowser(
            self._zc.zeroconf,
            _SERVICE_TYPE,
            handlers=[self._on_service_state_change],
        )
        logger.info("PeerDiscovery: browsing for %s", _SERVICE_TYPE)

    async def stop(self) -> None:
        """Deregister and shut down."""
        if not _ZEROCONF_AVAILABLE or self._zc is None:
            return
        if self._service_info:
            await self._zc.async_unregister_service(self._service_info)
        await self._zc.async_close()
        self._zc = None
        logger.info("PeerDiscovery: stopped")

    def _on_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        """Handle mDNS service state changes (runs in zeroconf's thread)."""
        asyncio.get_event_loop().call_soon_threadsafe(
            self._handle_state_change_sync, zeroconf, service_type, name, state_change
        )

    def _handle_state_change_sync(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        asyncio.ensure_future(self._handle_state_change(zeroconf, service_type, name, state_change))

    async def _handle_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change is ServiceStateChange.Added:
            info = ServiceInfo(service_type, name)
            await info.async_request(zeroconf, timeout=3000)

            if not info.addresses:
                return

            peer_id_bytes = info.properties.get(b"id", b"")
            peer_id = peer_id_bytes.decode() if peer_id_bytes else name
            host_bytes = info.properties.get(b"host", b"")
            hostname = host_bytes.decode() if host_bytes else ""

            # Skip ourselves
            if peer_id == self._instance_id:
                return

            host = socket.inet_ntoa(info.addresses[0])
            peer = PeerInfo(
                peer_id=peer_id,
                host=host,
                port=info.port,
                hostname=hostname,
            )
            self._known_peers[peer_id] = peer
            logger.info("PeerDiscovery: found peer %s @ %s:%d", peer_id, host, info.port)

            if self.on_peer_found:
                self.on_peer_found(peer)

        elif state_change is ServiceStateChange.Removed:
            # Extract peer_id from service name: "helioxos-<id>._helioxos._tcp.local."
            peer_id = name.split(".")[0].replace("helioxos-", "")
            if peer_id in self._known_peers:
                del self._known_peers[peer_id]
                logger.info("PeerDiscovery: lost peer %s", peer_id)
                if self.on_peer_lost:
                    self.on_peer_lost(peer_id)


def _get_local_ip() -> str:
    """Return the primary LAN IPv4 address of this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
