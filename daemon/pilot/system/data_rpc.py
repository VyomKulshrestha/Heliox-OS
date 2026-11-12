"""The daemon's side of "how big is Heliox's data" and "archive the older audit log" (Settings, Storage).

Kept out of server.py: the server registers ``storage_sizes`` and ``audit_archive`` with thin handlers that call
these, starts ``watch`` after start-up, and cancels it in stop(). ``server`` is the PilotServer; only its config,
its audit logger and its notification broadcast are used.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path
from typing import Any

from pilot.system.data_size import SizeWatch, sizes

logger = logging.getLogger("pilot.system.data_rpc")


class DataSize:
    """One per server: the folder, the once-per-crossing warning, and the two RPCs."""

    def __init__(self, server: Any, folder: Path) -> None:
        self._server = server
        self._folder = folder
        self.watch = SizeWatch(lambda: server.config.storage.data_folder_warn_gb)

    async def snapshot(self) -> dict[str, Any]:
        """Each unbounded store's size and the data folder's; says the warning once per crossing of the limit."""
        found = await asyncio.to_thread(sizes, self._folder)
        said = self.watch.check(found)
        if said:
            logger.warning(said)
            # The same channel as a full disk: one system line in every open window.
            await self._server._broadcast_notification(
                "storage_problem", {"kind": "size", "component": "", "message": said}
            )
        return {**found, "warn_gb": self._server.config.storage.data_folder_warn_gb, "over": self.watch.over}

    async def watch_forever(self, every: float = 3600.0, first_after: float = 60.0) -> None:
        """A minute after start-up, then hourly."""
        await asyncio.sleep(first_after)
        while True:
            try:
                await self.snapshot()
            except Exception:
                logger.debug("Could not measure the data folder", exc_info=True)
            await asyncio.sleep(every)

    async def storage_sizes(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", **await self.snapshot()}

    async def audit_archive(self, params: dict[str, Any], audit: Any) -> dict[str, Any]:
        """Move audit entries dated before ``older_than`` (YYYY-MM-DD) to a dated file beside the audit log."""
        from pilot.security.audit_archive import ArchiveRefused

        try:
            older_than = datetime.date.fromisoformat(str(params.get("older_than", "")))
        except ValueError:
            return {"status": "error", "message": "Choose the day as YYYY-MM-DD."}
        if audit is None:
            return {"status": "error", "message": "Heliox is still starting; try again in a moment."}
        try:
            record = await audit.archive_older_than(older_than)
        except ArchiveRefused as refused:
            return {"status": "error", "message": str(refused)}
        except OSError as error:
            from pilot.system.storage_health import storage

            said = storage().report("audit log", error, "the audit archive") or f"The archive failed: {error}"
            return {"status": "error", "message": said}
        return {
            "status": "ok",
            "archived": record,
            "sizes": await self.snapshot(),
            "message": f"Moved {record['lines']} audit entries from before {record['older_than']} to "
            f"{record['archive']}, beside the audit log. Nothing was deleted.",
        }
