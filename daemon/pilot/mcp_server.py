"""Local Model Context Protocol bridge for Heliox OS.

The MCP process is intentionally separate from the public documentation MCP.
It speaks MCP over stdio to a trusted local host, then uses a rotated,
least-privilege credential to call a small daemon RPC allow-list. It cannot
approve plans, edit configuration, read API keys, or invoke the daemon's raw
``execute`` method.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import math
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import urlsplit

import websockets
from mcp.server import MCPServer
from mcp_types import ToolAnnotations
from pydantic import Field, StringConstraints

from pilot import __version__
from pilot.config import RUNTIME_DIR, PilotConfig

logger = logging.getLogger("pilot.mcp")

TaskInput = Annotated[str, Field(min_length=1, max_length=20_000), StringConstraints(strip_whitespace=True)]
DecisionQuestion = Annotated[str, Field(min_length=1, max_length=4_000), StringConstraints(strip_whitespace=True)]
DecisionOptions = Annotated[list[str], Field(min_length=2, max_length=64)]
TypedState = dict[str, Any] | list[Any] | str
TypedQuestions = Annotated[dict[str, dict[str, Any]], Field(min_length=1, max_length=64)]
SessionId = Annotated[
    str,
    Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$"),
    StringConstraints(strip_whitespace=True),
]
TaskId = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
    StringConstraints(strip_whitespace=True),
]
RequestId = Annotated[
    str,
    Field(max_length=128, pattern=r"^[A-Za-z0-9._:-]*$"),
    StringConstraints(strip_whitespace=True),
]


def _package_version() -> str:
    """Return the source/package version used by the running MCP process."""
    return __version__


_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
_MODEL_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
_LOCAL_MODEL_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
_SUBMIT = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=True,
)
_CANCEL = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class DaemonRpcError(RuntimeError):
    """A safe, user-visible local daemon connection or RPC failure."""


class RpcClient(Protocol):
    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]: ...


class HelioxDaemonClient:
    """One-request authenticated client for the local Heliox WebSocket RPC."""

    def __init__(
        self,
        *,
        uri: str | None = None,
        token_file: str | Path | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between 0.1 and 120")
        config = PilotConfig.load()
        host = config.server.host
        if host in {"0.0.0.0", "::", "[::]"}:
            host = "127.0.0.1"
        self._uri = uri or f"ws://{host}:{config.server.port}"
        self._validate_loopback_uri(self._uri)
        self._token_file = Path(token_file) if token_file else RUNTIME_DIR / "mcp_auth_token"
        self._timeout_seconds = timeout_seconds

    @staticmethod
    def _validate_loopback_uri(uri: str) -> None:
        parsed = urlsplit(uri)
        if parsed.scheme not in {"ws", "wss"} or parsed.hostname is None or parsed.port is None:
            raise ValueError("daemon URL must be a WebSocket URL with an explicit port")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("daemon URL must not contain credentials, a query, or a fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("daemon URL must target the Heliox WebSocket root")
        hostname = parsed.hostname.casefold()
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = hostname == "localhost"
        if not is_loopback:
            raise ValueError("local Heliox MCP may connect only to a loopback daemon")

    def _read_token(self) -> str:
        try:
            token = self._token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DaemonRpcError("Cannot connect to Heliox OS. Start the desktop app or daemon first.") from exc
        if not token:
            raise DaemonRpcError("The local Heliox MCP credential is empty; restart the daemon.")
        return token

    async def _response_for(
        self,
        websocket: Any,
        request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        for _ in range(100):
            raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("id") != request_id:
                continue
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or "Heliox daemon RPC failed")
                raise DaemonRpcError(message)
            result = payload.get("result")
            if not isinstance(result, dict):
                raise DaemonRpcError("Heliox daemon returned an invalid RPC result.")
            return result
        raise DaemonRpcError("Heliox daemon did not return a matching RPC response.")

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        request_timeout = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        if not math.isfinite(request_timeout) or not 0.1 <= request_timeout <= 120:
            raise ValueError("RPC timeout must be between 0.1 and 120 seconds")
        token = self._read_token()
        auth_id = f"auth-{uuid.uuid4()}"
        request_id = f"mcp-{uuid.uuid4()}"
        try:
            async with websockets.connect(
                self._uri,
                open_timeout=request_timeout,
                close_timeout=2,
                max_size=4 * 1024 * 1024,
            ) as websocket:
                await websocket.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "auth",
                            "params": {"token": token},
                            "id": auth_id,
                        }
                    )
                )
                auth = await self._response_for(websocket, auth_id, request_timeout)
                if auth.get("role") != "mcp_local":
                    raise DaemonRpcError("The daemon rejected the local MCP identity.")

                await websocket.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": method,
                            "params": params or {},
                            "id": request_id,
                        }
                    )
                )
                return await self._response_for(websocket, request_id, request_timeout)
        except DaemonRpcError:
            raise
        except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
            raise DaemonRpcError("Cannot connect to the local Heliox daemon. Start Heliox OS and try again.") from exc


def create_mcp_server(client: RpcClient | None = None) -> MCPServer:
    """Build the local stdio MCP server, optionally with an injected test client."""

    daemon = client or HelioxDaemonClient()
    mcp = MCPServer(
        name="heliox-local",
        title="Heliox OS Local Control",
        description="Approval-gated local desktop automation through Heliox OS.",
        version=_package_version(),
        website_url="https://www.helioxos.dev",
        instructions=(
            "This server controls only the current user's local Heliox daemon. "
            "preview_heliox_task is advisory. submit_heliox_task returns before execution; "
            "the user must review every proposed action in the Heliox UI. Never claim a "
            "submitted task completed until get_heliox_task_status returns a terminal result. "
            "score_heliox_typed_decision is independent research-only guidance and must never "
            "be used to authorize, approve, or execute an action. "
            "score_heliox_typed_questions uses the same local model for bounded batches of "
            "independent typed questions; it is not an action planner. No MCP tool can approve a task."
        ),
    )

    @mcp.tool(
        title="Check Heliox health",
        description="Check whether the local Heliox daemon is reachable and healthy.",
        annotations=_READ_ONLY,
        structured_output=True,
    )
    async def heliox_health() -> dict[str, Any]:
        return await daemon.call("health")

    @mcp.tool(
        title="Get Heliox system status",
        description="Read the local daemon's current high-level system status.",
        annotations=_READ_ONLY,
        structured_output=True,
    )
    async def get_heliox_system_status() -> dict[str, Any]:
        return await daemon.call("system_status")

    @mcp.tool(
        title="List Heliox capabilities",
        description="List the action types and agents exposed by the running Heliox daemon.",
        annotations=_READ_ONLY,
        structured_output=True,
    )
    async def list_heliox_capabilities() -> dict[str, Any]:
        return await daemon.call("capabilities")

    @mcp.tool(
        title="Preview a Heliox task",
        description=(
            "Ask Heliox to produce a side-effect-free advisory plan. The submitted task is "
            "replanned later and every resulting action requires visible user approval."
        ),
        annotations=_MODEL_READ_ONLY,
        structured_output=True,
    )
    async def preview_heliox_task(input: TaskInput, session_id: SessionId = "default") -> dict[str, Any]:
        return await daemon.call(
            "mcp_plan_task",
            {"input": input, "session_id": session_id},
        )

    @mcp.tool(
        title="Score a typed decision (research only)",
        description=(
            "Score a user-supplied yes/no, choice, or ordered option set with the explicitly "
            "configured local research model. This tool is off until enabled in Heliox config, "
            "does not save or upload inputs, and never authorizes, approves, or executes actions."
        ),
        annotations=_LOCAL_MODEL_READ_ONLY,
        structured_output=True,
    )
    async def score_heliox_typed_decision(
        state: dict[str, Any],
        kind: Literal["noul", "choice", "score"],
        question: DecisionQuestion,
        options: DecisionOptions,
    ) -> dict[str, Any]:
        return await daemon.call(
            "mcp_typed_decision",
            {"state": state, "kind": kind, "question": question, "options": options},
            timeout_seconds=120,
        )

    @mcp.tool(
        title="Score typed questions (research only)",
        description=(
            "Score up to 64 independent typed questions against one shared state using the "
            "explicitly configured local research model. Each question has type (noul, choice, "
            "or score), instructions, and type-specific criteria. Batches are bounded to 256 "
            "candidate options, do not save or upload inputs, and never authorize actions."
        ),
        annotations=_LOCAL_MODEL_READ_ONLY,
        structured_output=True,
    )
    async def score_heliox_typed_questions(
        state: TypedState,
        questions: TypedQuestions,
    ) -> dict[str, Any]:
        return await daemon.call(
            "mcp_typed_decision",
            {"state": state, "questions": questions},
            timeout_seconds=120,
        )

    @mcp.tool(
        title="Submit a Heliox task",
        description=(
            "Submit a task to the full Heliox planner, policy, approval, execution, and "
            "verification pipeline. Returns a task id; execution cannot proceed until the "
            "user approves every proposed action in the Heliox UI."
        ),
        annotations=_SUBMIT,
        structured_output=True,
    )
    async def submit_heliox_task(
        input: TaskInput,
        session_id: SessionId = "default",
        request_id: RequestId = "",
    ) -> dict[str, Any]:
        return await daemon.call(
            "mcp_submit_task",
            {"input": input, "session_id": session_id, "request_id": request_id},
        )

    @mcp.tool(
        title="Get a Heliox task result",
        description=(
            "Read approval, execution, cancellation, and verified terminal result state for a "
            "task previously submitted through this local MCP."
        ),
        annotations=_READ_ONLY,
        structured_output=True,
    )
    async def get_heliox_task_status(task_id: TaskId) -> dict[str, Any]:
        return await daemon.call("mcp_task_status", {"task_id": task_id})

    @mcp.tool(
        title="Cancel a Heliox task",
        description=(
            "Request safe cancellation of a task created through this local MCP. This tool "
            "cannot cancel tasks created directly in the Heliox UI."
        ),
        annotations=_CANCEL,
        structured_output=True,
    )
    async def cancel_heliox_task(task_id: TaskId) -> dict[str, Any]:
        return await daemon.call("mcp_cancel_task", {"task_id": task_id})

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Heliox OS MCP server over stdio")
    parser.add_argument("--daemon-url", default=None, help="Override the local daemon WebSocket URL")
    parser.add_argument("--token-file", default=None, help="Override the rotated MCP token file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    server = create_mcp_server(HelioxDaemonClient(uri=args.daemon_url, token_file=args.token_file))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
