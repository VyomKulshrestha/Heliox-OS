"""WebSocket JSON-RPC 2.0 server for the Pilot daemon."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import signal
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.asyncio.server import Server, ServerConnection

from pilot.config import LOG_FILE, STATE_DIR, PilotConfig, ensure_dirs

logger = logging.getLogger("pilot.server")

CONFIRM_TIMEOUT_SECONDS = 300


@dataclass
class JsonRpcRequest:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None

    @classmethod
    def parse(cls, raw: str) -> JsonRpcRequest:
        data = json.loads(raw)
        if data.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC version")
        return cls(
            method=data["method"],
            params=data.get("params", {}),
            id=data.get("id"),
        )


def _success_response(req_id: str | int | None, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": req_id})


def _error_response(req_id: str | int | None, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id})


def _notification(method: str, params: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params})


@dataclass
class PendingConfirmation:
    """Tracks a plan awaiting user confirmation."""

    plan_id: str
    event: asyncio.Event
    confirmed: bool = False


class PilotServer:
    """Main daemon server managing WebSocket connections and agent dispatch."""

    def __init__(self, config: PilotConfig) -> None:
        self.config = config
        self._server: Server | None = None
        self._clients: set[ServerConnection] = set()
        self._handlers: dict[str, Any] = {}
        self._planner: Any = None
        self._executor: Any = None
        self._verifier: Any = None
        self._memory: Any = None
        self._vault: Any = None
        self._running = False
        self._pending_confirms: dict[str, PendingConfirmation] = {}

    async def initialize(self) -> None:
        """Initialize all agent components."""
        from pilot.agents.executor import Executor
        from pilot.agents.planner import Planner
        from pilot.agents.verifier import Verifier
        from pilot.memory.store import MemoryStore
        from pilot.models.router import ModelRouter
        from pilot.security.audit import AuditLogger
        from pilot.security.permissions import PermissionChecker
        from pilot.security.validator import ActionValidator
        from pilot.security.vault import KeyVault

        self._vault = KeyVault(self.config)
        model_router = ModelRouter(self.config, self._vault)
        audit = AuditLogger()
        validator = ActionValidator(self.config)
        permissions = PermissionChecker(self.config)
        self._memory = MemoryStore()
        await self._memory.initialize()

        self._planner = Planner(model_router, self._memory)
        self._executor = Executor(self.config, validator, permissions, audit)
        self._verifier = Verifier(model_router)

        self._handlers = {
            "execute": self._handle_execute,
            "confirm": self._handle_confirm,
            "get_config": self._handle_get_config,
            "update_config": self._handle_update_config,
            "get_history": self._handle_get_history,
            "store_api_key": self._handle_store_api_key,
            "delete_api_key": self._handle_delete_api_key,
            "list_api_keys": self._handle_list_api_keys,
            "list_ollama_models": self._handle_list_ollama_models,
            "health": self._handle_health,
            "ping": self._handle_ping,
            "system_status": self._handle_system_status,
            "capabilities": self._handle_capabilities,
        }

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)
        try:
            async for message in websocket:
                try:
                    request = JsonRpcRequest.parse(str(message))
                    response = await self._dispatch(request, websocket)
                    if response and request.id is not None:
                        await websocket.send(response)
                except json.JSONDecodeError:
                    await websocket.send(_error_response(None, -32700, "Parse error"))
                except ValueError as e:
                    await websocket.send(_error_response(None, -32600, str(e)))
                except Exception as e:
                    logger.exception("Handler error")
                    await websocket.send(_error_response(None, -32603, f"Internal error: {e}"))
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected: %s", remote)

    async def _dispatch(self, request: JsonRpcRequest, ws: ServerConnection) -> str | None:
        handler = self._handlers.get(request.method)
        if handler is None:
            return _error_response(request.id, -32601, f"Method not found: {request.method}")
        result = await handler(request.params, ws)
        return _success_response(request.id, result)

    # -- Core execution pipeline --

    MAX_RETRIES = 2

    async def _handle_execute(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Agentic pipeline: plan -> execute -> verify -> [retry on failure].

        If execution fails, the error is fed back to the planner for re-planning
        up to MAX_RETRIES times. Confirmation gates pause for user approval on
        Tier 2+ actions.
        """
        user_input = params.get("input", "")
        if not user_input.strip():
            return {"status": "error", "message": "Empty input"}

        error_context = ""
        all_results: list = []
        last_verification = None
        last_explanation = ""

        for attempt in range(1 + self.MAX_RETRIES):
            # Phase 1: Planning (or re-planning with error context)
            if attempt == 0:
                await ws.send(_notification("status", {"phase": "planning"}))
            else:
                await ws.send(_notification("status", {"phase": f"re-planning (attempt {attempt + 1})"}))

            plan = await self._planner.plan(user_input, error_context=error_context)
            if plan.error:
                if attempt < self.MAX_RETRIES:
                    error_context = plan.error
                    continue
                return {"status": "error", "message": plan.error}

            last_explanation = plan.explanation
            plan_id = str(uuid.uuid4())[:8]

            await ws.send(
                _notification(
                    "plan_preview",
                    {
                        "plan_id": plan_id,
                        "actions": [a.model_dump() for a in plan.actions],
                        "explanation": plan.explanation,
                    },
                )
            )

            # Phase 2: Confirmation gate
            needs_confirm = any(a.requires_confirmation for a in plan.actions)
            if needs_confirm:
                confirmed = await self._wait_for_confirmation(plan_id, plan, ws)
                if not confirmed:
                    return {
                        "status": "cancelled",
                        "message": "Plan was denied by user.",
                        "explanation": plan.explanation,
                    }

            # Phase 3: Execution
            await ws.send(_notification("status", {"phase": "executing"}))
            results = await self._executor.execute(plan)
            all_results = results

            # Phase 4: Verification
            await ws.send(_notification("status", {"phase": "verifying"}))
            verification = await self._verifier.verify(plan, results)
            last_verification = verification

            if verification.passed:
                asyncio.create_task(self._memory.record(user_input, plan, results))
                return {
                    "status": "success",
                    "results": [r.model_dump() for r in results],
                    "verification": verification.model_dump(),
                    "explanation": plan.explanation,
                }

            # Execution failed — build error context for retry
            failed_details = [d for d in verification.details if "FAILED" in d or "MISMATCH" in d]
            error_msgs = [r.error for r in results if r.error]
            error_context = "\n".join(failed_details + error_msgs)

            if attempt < self.MAX_RETRIES:
                await ws.send(
                    _notification(
                        "status",
                        {
                            "phase": "retrying — previous attempt failed",
                        },
                    )
                )
            else:
                break

        asyncio.create_task(self._memory.record(user_input, plan, all_results))
        return {
            "status": "partial_failure",
            "results": [r.model_dump() for r in all_results],
            "verification": last_verification.model_dump() if last_verification else {},
            "explanation": last_explanation,
        }

    async def _wait_for_confirmation(self, plan_id: str, plan: Any, ws: ServerConnection) -> bool:
        """Send a confirmation request and block until the user responds or timeout."""
        pending = PendingConfirmation(plan_id=plan_id, event=asyncio.Event())
        self._pending_confirms[plan_id] = pending

        await ws.send(
            _notification(
                "confirm_required",
                {
                    "plan_id": plan_id,
                    "actions": [a.model_dump() for a in plan.actions if a.requires_confirmation],
                },
            )
        )

        try:
            await asyncio.wait_for(pending.event.wait(), timeout=CONFIRM_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("Confirmation timed out for plan %s", plan_id)
            return False
        finally:
            self._pending_confirms.pop(plan_id, None)

        return pending.confirmed

    async def _handle_confirm(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Resolve a pending confirmation request from the UI."""
        plan_id = params.get("plan_id", "")
        confirmed = params.get("confirmed", False)

        pending = self._pending_confirms.get(plan_id)
        if pending is None:
            return {"status": "error", "message": f"No pending confirmation for plan_id: {plan_id}"}

        pending.confirmed = bool(confirmed)
        pending.event.set()
        return {"status": "ok", "confirmed": pending.confirmed}

    # -- Config --

    async def _handle_get_config(self, params: dict, ws: ServerConnection) -> dict:
        from dataclasses import asdict

        data = asdict(self.config)
        data.pop("server", None)
        return data

    async def _handle_update_config(self, params: dict, ws: ServerConnection) -> dict:
        section = params.get("section", "")
        values = params.get("values", {})

        if section == "" and "first_run_complete" in values:
            self.config.first_run_complete = values["first_run_complete"]
            self.config.save()
            return {"status": "ok"}

        target = getattr(self.config, section, None)
        if target is None:
            return {"status": "error", "message": f"Unknown config section: {section}"}
        for k, v in values.items():
            if hasattr(target, k):
                setattr(target, k, v)
        self.config.save()

        # Re-init cloud client if cloud provider changed
        if section == "model" and ("cloud_provider" in values or "provider" in values):
            if self.config.model.cloud_provider:
                from pilot.models.cloud import CloudClient

                self._planner._model._cloud = CloudClient(self.config, self._vault)
                logger.info("Cloud client re-initialized for provider: %s", self.config.model.cloud_provider)

        return {"status": "ok"}

    # -- History --

    async def _handle_get_history(self, params: dict, ws: ServerConnection) -> dict:
        limit = params.get("limit", 50)
        offset = params.get("offset", 0)
        entries = await self._memory.get_history(limit=limit, offset=offset)
        return {"entries": entries}

    # -- API key management --

    async def _handle_store_api_key(self, params: dict, ws: ServerConnection) -> dict:
        provider = params.get("provider", "")
        key = params.get("api_key", "") or params.get("key", "")
        if not provider or not key:
            return {"status": "error", "message": "provider and api_key are required"}
        await self._vault.store_key(provider, key)
        # Re-init cloud client with the new provider
        if self.config.model.cloud_provider == provider:
            from pilot.models.cloud import CloudClient

            self._planner._model._cloud = CloudClient(self.config, self._vault)
        return {"status": "ok"}

    async def _handle_delete_api_key(self, params: dict, ws: ServerConnection) -> dict:
        provider = params.get("provider", "")
        if not provider:
            return {"status": "error", "message": "provider is required"}
        await self._vault.delete_key(provider)
        return {"status": "ok"}

    async def _handle_list_api_keys(self, params: dict, ws: ServerConnection) -> dict:
        providers = await self._vault.list_providers()
        return {"providers": providers}

    # -- Ollama model discovery --

    async def _handle_list_ollama_models(self, params: dict, ws: ServerConnection) -> dict:
        from pilot.models.ollama import OllamaClient

        client = OllamaClient(self.config.model.ollama_base_url)
        try:
            models = await client.list_models()
            return {"models": models, "available": True}
        except Exception:
            return {"models": [], "available": False}

    # -- Health --

    async def _handle_health(self, params: dict, ws: ServerConnection) -> dict:
        from pilot.models.router import ModelRouter

        router: ModelRouter = self._planner._model
        backends = await router.check_health()
        return {"backends": backends}

    async def _handle_ping(self, params: dict, ws: ServerConnection) -> dict:
        return {"pong": True, "version": "0.2.0"}

    async def _handle_system_status(self, params: dict, ws: ServerConnection) -> dict:
        """Return current system information."""
        from pilot.system.platform_detect import get_platform_info

        info = get_platform_info()
        return {
            "platform": info,
            "capabilities_count": len(self._executor._dispatch_table),
        }

    async def _handle_capabilities(self, params: dict, ws: ServerConnection) -> dict:
        """Return all available action types."""
        from pilot.actions import ActionType

        return {
            "action_types": [t.value for t in ActionType],
            "count": len(ActionType),
        }

    # -- Broadcast --

    async def broadcast(self, method: str, params: Any) -> None:
        msg = _notification(method, params)
        for client in list(self._clients):
            try:
                await client.send(msg)
            except Exception:
                self._clients.discard(client)

    # -- Lifecycle --

    async def start(self) -> None:
        self._running = True
        await self.initialize()

        host = self.config.server.host
        port = self.config.server.port
        if not self.config.server.auth_token:
            self.config.server.auth_token = secrets.token_urlsafe(32)

        logger.info("Starting Pilot daemon on ws://%s:%d", host, port)
        self._server = await websockets.serve(
            self._handle_connection,
            host,
            port,
        )
        logger.info("Pilot daemon ready")

    async def stop(self) -> None:
        self._running = False
        for pending in self._pending_confirms.values():
            pending.event.set()
        self._pending_confirms.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._memory:
            await self._memory.close()
        logger.info("Pilot daemon stopped")


def _setup_logging() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


def main() -> None:
    """Entry point for the pilot-daemon command."""
    ensure_dirs()
    _setup_logging()
    config = PilotConfig.load()
    server = PilotServer(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run() -> None:
        await server.start()
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _signal_handler)

        await stop_event.wait()
        await server.stop()

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        loop.run_until_complete(server.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
