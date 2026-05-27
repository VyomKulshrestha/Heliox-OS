"""Skill/plugin synchronisation across LAN peers.

When a plugin is loaded locally it is broadcast to all connected peers.
When a peer sends a plugin payload it is validated and installed into
``PLUGINS_DIR`` (from ``pilot.config``) via the existing ``PluginManager``.

Security model
--------------
- Full filename validated against ``^[a-zA-Z0-9_]{1,64}\\.py$`` — prevents
  directory traversal attacks (e.g. ``../../evil.py``) that would bypass a
  stem-only check
- Plugin source is validated as syntactically correct Python before writing
- Plugins received from peers are stored in a dedicated ``peer_plugins/``
  sub-directory so they can be audited or removed independently
- No arbitrary code is executed during sync — the plugin is only *installed*;
  it is loaded by the PluginManager on the next discovery cycle
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pilot.config import PLUGINS_DIR

if TYPE_CHECKING:
    from pilot.network.mesh import HelioxMesh

logger = logging.getLogger("pilot.network.skill_sync")

_PEER_PLUGIN_SUBDIR = "peer_plugins"


class SkillSync:
    """Handles plugin serialisation, broadcast, and installation.

    Parameters
    ----------
    mesh:
        The ``HelioxMesh`` instance used to broadcast to peers.
    plugin_base_dir:
        Base directory for plugins (default: ``PLUGINS_DIR`` from ``pilot.config``).
    """

    def __init__(self, mesh: HelioxMesh, plugin_base_dir: str | None = None) -> None:
        self._mesh = mesh
        base = Path(plugin_base_dir or PLUGINS_DIR)
        self._peer_dir = base / _PEER_PLUGIN_SUBDIR
        self._peer_dir.mkdir(parents=True, exist_ok=True)

    # ── Outbound: broadcast a locally loaded plugin ───────────────────────────

    async def broadcast_plugin(self, plugin_name: str, file_path: str) -> None:
        """Read a plugin file and broadcast its source to all peers.

        Parameters
        ----------
        plugin_name:
            The plugin's canonical name (from ``PLUGIN_NAME``).
        file_path:
            Absolute path to the ``.py`` plugin file.
        """
        try:
            source = Path(file_path).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("SkillSync: cannot read plugin %s: %s", file_path, exc)
            return

        payload: dict[str, Any] = {
            "name": plugin_name,
            "filename": Path(file_path).name,
            "source": source,
        }
        await self._mesh.broadcast("skill_sync", payload)
        logger.info("SkillSync: broadcast plugin '%s' to %d peer(s)", plugin_name, len(self._mesh.peer_ids))

    # ── Inbound: install a plugin received from a peer ────────────────────────

    async def handle_incoming(self, peer_id: str, payload: dict[str, Any]) -> None:
        """Validate and install a plugin received from a peer.

        Parameters
        ----------
        peer_id:
            The sending peer's instance ID (used for logging).
        payload:
            Dict with keys ``name``, ``filename``, ``source``.
        """
        name = payload.get("name", "")
        filename = payload.get("filename", "")
        source = payload.get("source", "")

        # Validate the full filename with a strict regex — this prevents directory
        # traversal attacks (e.g. "../../evil.py") where Path().stem would pass the
        # stem check but the full path would escape the peer_plugins directory.
        if not re.match(r"^[a-zA-Z0-9_]{1,64}\.py$", filename):
            logger.warning(
                "SkillSync: rejected plugin from %s — unsafe or malformed filename '%s'",
                peer_id,
                filename,
            )
            return

        # Validate Python syntax before writing
        try:
            ast.parse(source)
        except SyntaxError as exc:
            logger.warning("SkillSync: rejected plugin '%s' from %s — syntax error: %s", name, peer_id, exc)
            return

        dest = self._peer_dir / filename
        # Never overwrite a locally authored plugin
        local_plugin = self._peer_dir.parent / filename
        if local_plugin.exists():
            logger.info("SkillSync: skipping '%s' — local version already exists", filename)
            return

        dest.write_text(source, encoding="utf-8")
        logger.info("SkillSync: installed plugin '%s' from peer %s → %s", name, peer_id, dest)

        # Acknowledge back to the sender
        await self._mesh.send_to(peer_id, "skill_ack", {"name": name, "status": "installed"})

    def list_peer_plugins(self) -> list[str]:
        """Return names of all plugins received from peers."""
        return [p.stem for p in self._peer_dir.glob("*.py")]
