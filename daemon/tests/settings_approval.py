"""For tests: approve a setting change the way a person does, so the signed receipt is the product's own.

``approve_setting_change(server, section, values)`` works out what the change widens with the daemon's own rules,
lets the daemon ask (``PilotServer._approve_widenings``: its approval dialog, sent to a window connection), and
answers that dialog through the daemon's ``confirm`` handler from the window, with ``input_channel`` "dialog",
approving every line. The daemon then writes the signed ``setting_change`` receipt itself, exactly as after a click,
so ``_approved_before`` and ``TaskReceipts.approved_settings()`` count it. Nothing here writes a receipt.

It does not apply the change: call the handler (or ``_control_reply``) afterwards as the session needs. It lives in
the tests, never in the product: the product has no way to record an approval nobody gave.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

from pilot.security.rpc_identity import RpcClientRole


class _Window:
    """A window connection: keeps what the daemon sends it."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.remote_address = ("127.0.0.1", 0)

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


async def approve_setting_change(
    server: Any, section: str, values: dict[str, Any], *, timeout: float = 10.0
) -> list[Any]:
    """Approve, as a person in the window does, what ``values`` for ``section`` would widen; returns those
    widenings (empty when the change widens nothing, and then nothing is asked or written)."""
    found = server._widenings_for("update_config", {"section": section, "values": values})
    if not found:
        return []
    window = _Window()
    server._client_roles[window] = RpcClientRole.UI
    if getattr(server, "_broadcast_notification", None) is None:
        server._broadcast_notification = MagicMock()
    asking = asyncio.create_task(server._approve_widenings(window, found))
    deadline = asyncio.get_running_loop().time() + timeout
    request = None
    while request is None:
        if asking.done():
            raise AssertionError(f"the daemon did not ask: {asking.result()!r}")
        if asyncio.get_running_loop().time() > deadline:
            asking.cancel()
            raise AssertionError("the daemon's approval dialog never reached the window")
        request = next((m for m in window.sent if m.get("method") == "confirm_required"), None)
        await asyncio.sleep(0.01)
    params = request["params"]
    answer = await server._handle_confirm(
        {
            "plan_id": params["plan_id"],
            "confirmed": True,
            "input_channel": "dialog",
            "approved_indices": [line["index"] for line in params["actions"]],
        },
        window,
    )
    refusal = await asyncio.wait_for(asking, timeout)
    if refusal is not None:
        raise AssertionError(f"the approval did not take: {refusal!r} (confirm answered {answer!r})")
    return found
