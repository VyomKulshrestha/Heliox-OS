"""LAN mesh network — peer discovery, skill sync, and collaborative execution.

Enable in config.toml:

    [network]
    enabled = true
    port = 8786
    skill_sync_enabled = true
    collab_exec_enabled = true
"""

from pilot.network.mesh import HelioxMesh

__all__ = ["HelioxMesh"]
