"""Deterministic daemon used for the manual Svelte confirmation smoke test."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.request import urlopen

import websockets

from pilot.real_folders import use_scratch_folders

# Every Heliox folder under a scratch root of its own, before pilot.config loads: never the person's real ones
use_scratch_folders()  # a temporary root, removed when the script ends

from pilot.actions import Action, ActionPlan, ActionType, BrowserParams  # noqa: E402
from pilot.agents.executor import Executor  # noqa: E402
from pilot.config import PilotConfig  # noqa: E402
from pilot.security.audit import AuditLogger  # noqa: E402
from pilot.security.permissions import PermissionChecker  # noqa: E402
from pilot.security.validator import ActionValidator  # noqa: E402
from pilot.server import PilotServer  # noqa: E402
from pilot.system.browser import browser_close, browser_get_page_info, browser_screenshot  # noqa: E402


def _record(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


async def run(
    evidence_path: Path,
    token: str,
    target_url: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8785,
    screenshot_path: Path | None = None,
    neural_token: str | None = None,
    mcp_token: str | None = None,
) -> None:
    config = PilotConfig()
    config.server.auth_token = token
    server = PilotServer(config)
    if neural_token:
        server._neural_auth_token = neural_token
    if mcp_token:
        server._mcp_auth_token = mcp_token
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(evidence_path.with_suffix(".audit.jsonl")),
    )
    command_number = 0
    browser_execute_count = 0
    confirmation_count = 0
    execution_socket_id: int | None = None
    confirmation_socket_id: int | None = None
    stop_event = asyncio.Event()

    async def execute(params, ws):
        nonlocal browser_execute_count, command_number, execution_socket_id
        command_number += 1
        execution_socket_id = id(ws)
        plan_id = f"ui-smoke-{command_number}"
        action = Action(
            action_type=ActionType.BROWSER_NAVIGATE,
            target=target_url,
            parameters=BrowserParams(url=target_url),
        )
        plan = ActionPlan(
            actions=[action],
            explanation="Navigate the controlled browser to the local Heliox smoke page",
            raw_input=str(params.get("input", "")),
        )
        confirmed, approved, required = await server._wait_for_confirmation(
            plan_id,
            plan,
            ws,
            reason="Live UI-to-daemon browser-command smoke test.",
            force_all_actions=True,
        )
        if not confirmed or approved != required:
            _record(
                evidence_path,
                {
                    "browser_execute_count": browser_execute_count,
                    "decision": "denied",
                    "executed": False,
                    "plan_id": plan_id,
                },
            )
            return {"status": "cancelled", "message": f"DENIED {plan_id}: browser command did not run"}

        results = await executor.execute(plan, plan_id=plan_id, user_confirmed=True)
        result = results[0]
        if result.success:
            browser_execute_count += 1
        page_info = json.loads(await browser_get_page_info()) if result.success else None
        if result.success and screenshot_path is not None:
            await browser_screenshot(str(screenshot_path), full_page=True)
        _record(
            evidence_path,
            {
                "browser_execute_count": browser_execute_count,
                "browser_page": page_info,
                "confirmation_count": confirmation_count,
                "decision": "approved",
                "executed": result.success,
                "output": result.output,
                "plan_id": plan_id,
                "same_websocket": execution_socket_id == confirmation_socket_id,
                "screenshot": str(screenshot_path) if screenshot_path is not None else None,
            },
        )
        return {
            "status": "success" if result.success else "error",
            "message": (
                f"APPROVED {plan_id}: browser command executed; page={page_info}"
                if result.success
                else f"APPROVED {plan_id}: browser command failed: {result.error}"
            ),
        }

    async def confirm(params, ws):
        nonlocal confirmation_count, confirmation_socket_id
        confirmation_count += 1
        confirmation_socket_id = id(ws)
        return await server._handle_confirm(params, ws)

    async def smoke_status(_params, _ws):
        return {
            "browser_execute_count": browser_execute_count,
            "confirmation_count": confirmation_count,
            "same_websocket": (
                execution_socket_id is not None
                and confirmation_socket_id is not None
                and execution_socket_id == confirmation_socket_id
            ),
            "target_url": target_url,
        }

    async def smoke_shutdown(_params, _ws):
        asyncio.get_running_loop().call_later(0.05, stop_event.set)
        return {"status": "stopping"}

    server._handlers = {
        "execute": execute,
        "confirm": confirm,
        "smoke_status": smoke_status,
        "smoke_shutdown": smoke_shutdown,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.unlink(missing_ok=True)
    try:
        async with websockets.serve(server._handle_connection, host, port):
            _record(evidence_path, {"event": "ready", "host": host, "port": port, "target_url": target_url})
            await stop_event.wait()
    finally:
        await browser_close()
        _record(
            evidence_path,
            {
                "browser_execute_count": browser_execute_count,
                "browser_closed": True,
                "event": "cleanup",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--token")
    parser.add_argument("--target-url", default="http://127.0.0.1:1420/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8785)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--neural-token")
    parser.add_argument("--mcp-token")
    args = parser.parse_args()
    token = args.token
    if not token:
        with urlopen(  # noqa: S310 - fixed loopback smoke endpoint
            "http://127.0.0.1:1420/api/auth_token",
            timeout=5,
        ) as response:
            token = response.read().decode("utf-8").strip()
    if not token:
        parser.error("the resolved token is empty")
    asyncio.run(
        run(
            args.evidence,
            token,
            args.target_url,
            host=args.host,
            port=args.port,
            screenshot_path=args.screenshot,
            neural_token=args.neural_token,
            mcp_token=args.mcp_token,
        )
    )


if __name__ == "__main__":
    main()
