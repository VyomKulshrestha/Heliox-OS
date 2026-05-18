"""WebSocket JSON-RPC 2.0 server for the Pilot daemon."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import logging
import os
import secrets
import signal
import sys
from pathlib import Path
from typing import Any

import aiosqlite
import websockets
from websockets.server import ServerConnection

# Global Storage and Directory layouts
STATE_DIR = Path.home() / ".heliox" / "state"
LOG_FILE = STATE_DIR / "pilot_daemon.log"
DB_PATH = "plan_history.db"

logger = logging.getLogger("pilot.server")


class PilotConfig:
    """System configuration mapping parser placeholder."""

    class Server:
        host: str = "127.0.0.1"
        port: int = 8000
        auth_token: str = ""

    class Security:
        dry_run: bool = False

    def __init__(self):
        self.server = self.Server()
        self.security = self.Security()

    @classmethod
    def load(cls):
        return cls()


def _notification(method: str, params: Any) -> str:
    """Helper to pack JSON-RPC formatted data payloads."""
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params})


def ensure_dirs() -> None:
    """Pre-configures required base file directories."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


class PilotServer:
    """Unified Pilot Daemon Server orchestrating background subsystems,

    plugin engines, screen vision loops, and asynchronous audit handlers.
    """

    def __init__(self, config: PilotConfig):
        self.config = config
        self._running: bool = False
        self._server = None
        self._clients: set[ServerConnection] = set()

        # Engine subsystem references
        self._plugin_registry = None
        self._subconscious = None
        self._screen_vision = None
        self._orchestrator = None
        self._background = None
        self._reflector = None
        self._memory = None
        self._budget_tracker = None
        self._tribe_engine = None
        self._attention_ui = None
        self._stress_gate = None
        self._intent_predictor = None
        self._voice_listener = None
        self._autonomous = None
        self._proactive = None
        self._planner = None
        self._executor = None
        self._verifier = None

        self._pending_confirms: dict[str, Any] = {}
        self._new_features_announcement: str = "Welcome to Heliox 0.6.0 Pipeline!"

    async def ensure_db(self) -> None:
        """Asynchronously sets up the plan_history compliance table if it doesn't exist."""
        logger.info("Verifying asynchronous plan_history compliance database...")
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS plan_history (
                        plan_id TEXT PRIMARY KEY,
                        action_plan TEXT NOT NULL,
                        critic_verdict TEXT,
                        user_confirmation_decision TEXT,
                        execution_outcome TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await conn.commit()
            logger.info("Compliance audit log table is initialized and verified.")
        except Exception as e:
            logger.error("Failed to execute database schema generation: %s", e)

    async def initialize(self) -> None:
        """Runs pre-start evaluations and component setups."""
        logger.info("Initializing subsystem engines...")
        await self.ensure_db()

    async def _handle_connection(self, ws: ServerConnection, path: str = "") -> None:
        """Manages message distribution over live sockets."""
        self._clients.add(ws)
        try:
            async for _message in ws:
                pass
        except Exception as e:
            logger.debug("Socket context reset: %s", e)
        finally:
            self._clients.discard(ws)

    # ── Orchestrator Configuration Routes ──

    async def _handle_update_config(self, params: dict, ws: ServerConnection) -> dict:
        return {"status": "success", "message": "Configuration map updated."}

    async def _handle_agent_spawn(self, params: dict, ws: ServerConnection) -> dict:
        return {"status": "success", "agent_id": secrets.token_hex(4)}

    async def _handle_voice_event(self, params: dict, ws: ServerConnection) -> dict:
        return {"status": "received"}

    # ── Plugin Interface Infrastructure Handlers ──

    async def _handle_plugin_tools(self, params: dict, ws: ServerConnection) -> dict:
        if self._plugin_registry:
            return {"tools": self._plugin_registry.get_all_tools()}
        return {"error": "Plugin registry not initialized"}

    async def _handle_plugin_toggle(self, params: dict, ws: ServerConnection) -> dict:
        name = params.get("name", "")
        enabled = params.get("enabled", True)
        if not name:
            return {"error": "No plugin name provided"}
        if self._plugin_registry:
            if enabled:
                ok = self._plugin_registry.enable_plugin(name)
            else:
                ok = self._plugin_registry.disable_plugin(name)
            return {"success": ok, "plugin": name, "enabled": enabled}
        return {"error": "Plugin registry not initialized"}

    async def _handle_plugin_market_list(self, params: dict, ws: ServerConnection) -> dict:
        repo_root = Path(__file__).parent.parent.parent
        registry_path = repo_root / "plugins" / "registry.json"

        if not registry_path.exists():
            return {"plugins": [], "error": "Registry not found"}

        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            plugins = data.get("plugins", [])

            installed = set()
            if self._plugin_registry:
                installed = {p.name for p in self._plugin_registry.get_all_plugins()}

            for plugin in plugins:
                plugin["installed"] = plugin.get("name") in installed

            return {"plugins": plugins}
        except Exception as e:
            logger.error("Failed to load plugin registry: %s", e)
            return {"plugins": [], "error": str(e)}

    async def _handle_plugin_install(self, params: dict, ws: ServerConnection) -> dict:
        plugin_name = params.get("plugin_name", "")
        if not plugin_name:
            return {"error": "plugin_name is required"}

        plugin_dir = Path.home() / ".heliox" / "plugins" / plugin_name
        plugin_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = plugin_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({"name": plugin_name, "installed_from_marketplace": True}, indent=2),
            encoding="utf-8",
        )

        if self._plugin_registry:
            count = self._plugin_registry.discover()
            logger.info("Plugin installed: %s (total plugins: %d)", plugin_name, count)

        return {
            "success": True,
            "plugin": plugin_name,
            "path": str(plugin_dir),
        }

    async def _handle_plugin_uninstall(self, params: dict, ws: ServerConnection) -> dict:
        import shutil

        plugin_name = params.get("plugin_name", "")
        if not plugin_name:
            return {"error": "plugin_name is required"}

        plugin_dir = Path.home() / ".heliox" / "plugins" / plugin_name
        if not plugin_dir.exists():
            return {"error": f"Plugin not found: {plugin_name}"}

        try:
            shutil.rmtree(plugin_dir)
            logger.info("Plugin uninstalled: %s", plugin_name)
            return {"success": True, "plugin": plugin_name}
        except Exception as e:
            logger.error("Failed to uninstall plugin %s: %s", plugin_name, e)
            return {"error": str(e)}

    # ── Subconscious Persona Memory Controllers ──

    async def _handle_persona_rules(self, params: dict, ws: ServerConnection) -> dict:
        if self._subconscious:
            context = await self._subconscious.get_persona_context()
            stats = await self._subconscious.get_stats()
            return {"context": context, **stats}
        return {"error": "Subconscious agent not initialized"}

    async def _handle_persona_consolidate(self, params: dict, ws: ServerConnection) -> dict:
        if self._subconscious:
            result = await self._subconscious.consolidate()
            return result
        return {"error": "Subconscious agent not initialized"}

    async def _handle_persona_add_preference(self, params: dict, ws: ServerConnection) -> dict:
        key = params.get("key", "")
        value = params.get("value", "")
        if not key or not value:
            return {"error": "Both key and value required"}
        if self._subconscious:
            await self._subconscious.add_manual_preference(key, value)
            return {"status": "ok", "key": key, "value": value}
        return {"error": "Subconscious agent not initialized"}

    async def _handle_subconscious_stats(self, params: dict, ws: ServerConnection) -> dict:
        if self._subconscious:
            return await self._subconscious.get_stats()
        return {"error": "Subconscious agent not initialized"}

    # ── Screen Vision Core Threads ──

    async def _handle_screen_context(self, params: dict, ws: ServerConnection) -> dict:
        if self._screen_vision:
            return {
                "summary": self._screen_vision.get_context_for_planner(),
                **self._screen_vision.get_context().to_dict(),
            }
        return {"error": "Screen vision not initialized"}

    async def _handle_screen_current_app(self, params: dict, ws: ServerConnection) -> dict:
        if self._screen_vision:
            return {"active_app": self._screen_vision.get_current_app()}
        return {"error": "Screen vision not initialized"}

    async def _handle_screen_vision_stats(self, params: dict, ws: ServerConnection) -> dict:
        if self._screen_vision:
            return self._screen_vision.get_stats()
        return {"error": "Screen vision not initialized"}

    async def _handle_screen_vision_toggle(self, params: dict, ws: ServerConnection) -> dict:
        enabled = params.get("enabled", True)
        if self._screen_vision:
            if enabled:
                interval = params.get("interval_seconds", self.config.screen_vision.capture_interval_seconds)
                describe = params.get("enable_describe", False)
                await self._screen_vision.start(interval, describe)
            else:
                await self._screen_vision.stop()
            return {"status": "ok", "enabled": enabled}
        return {"error": "Screen vision not initialized"}

    # ── Core Broadcast Core Logic ──

    async def broadcast(self, method: str, params: Any) -> None:
        """Broadcast a notification frame out to the connected UI workers."""
        msg = _notification(method, params)
        for client in list(self._clients):
            try:
                await client.send(msg)
            except Exception:
                self._clients.discard(client)

    # ── Daemon Server Lifecycle Loops ──

    async def start(self) -> None:
        self._running = True
        await self.initialize()

        host = self.config.server.host
        port = self.config.server.port
        if not self.config.server.auth_token:
            self.config.server.auth_token = secrets.token_urlsafe(32)

        logger.info("Starting Pilot daemon on ws://%s:%d", host, port)
        self._server = await websockets.serve(self._handle_connection, host, port)
        logger.info("Pilot daemon setup operational")

        if hasattr(self, "_new_features_announcement") and self._new_features_announcement:
            await asyncio.sleep(1)
            await self.broadcast(
                "feature_announcement",
                {"message": self._new_features_announcement, "version": "0.6.0"},
            )

    async def stop(self) -> None:
        """Stop the Pilot daemon server and clean up all resources."""
        self._running = False
        # ── Stop LAN mesh ──
        if self._mesh:
            await self._mesh.stop()
        if self._orchestrator:
            await self._orchestrator.stop_all()
        if self._background:
            self._background.stop_all()
        for pending in self._pending_confirms.values():
            pending.event.set()
        self._pending_confirms.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._reflector:
            await self._reflector.close()
        if self._memory:
            await self._memory.close()
        if self._budget_tracker:
            await self._budget_tracker.close()
        if self._tribe_engine and self._tribe_engine.is_loaded:
            self._tribe_engine.unload_model()
        logger.info("Pilot daemon stopped securely")

    # ── Budget Tracking Allocators ──

    async def _handle_budget_stats(self, params: dict, ws: ServerConnection) -> dict:
        if not self._budget_tracker:
            return {}
        return await self._budget_tracker.get_stats()

    async def _handle_budget_reset(self, params: dict, ws: ServerConnection) -> dict:
        if not self._budget_tracker:
            return {"status": "ok"}
        await self._budget_tracker.reset_current_month()
        return {"status": "ok"}

    # ── Cognitive Infrastructure Processing (TRIBE v2) ──

    async def _handle_cognitive_stats(self, params: dict, ws: ServerConnection) -> dict:
        return {
            "tribe_engine": self._tribe_engine.get_stats() if self._tribe_engine else None,
            "attention_ui": self._attention_ui.get_stats() if self._attention_ui else None,
            "stress_gate": self._stress_gate.get_stats() if self._stress_gate else None,
            "intent_predictor": (
                self._intent_predictor.get_stats() if self._intent_predictor else None
            ),
        }

    async def _handle_cognitive_state(self, params: dict, ws: ServerConnection) -> dict:
        if not self._tribe_engine:
            return {"error": "Cognitive engine not initialized"}
        state = await self._tribe_engine.predict_cognitive_state(
            stimulus_description=params.get("stimulus", ""),
        )
        return state.to_dict()

    async def _handle_attention_toggle(self, params: dict, ws: ServerConnection) -> dict:
        if not self._attention_ui:
            return {"error": "Attention UI not initialized"}
        enabled = self._attention_ui.toggle(params.get("enabled"))
        return {"enabled": enabled}

    async def _handle_stress_gate_toggle(self, params: dict, ws: ServerConnection) -> dict:
        if not self._stress_gate:
            return {"error": "Stress gate not initialized"}
        enabled = self._stress_gate.toggle(params.get("enabled"))
        return {"enabled": enabled}

    async def _handle_intent_predictor_toggle(self, params: dict, ws: ServerConnection) -> dict:
        if not self._intent_predictor:
            return {"error": "Intent predictor not initialized"}
        enabled = self._intent_predictor.toggle(params.get("enabled"))
        return {"enabled": enabled}

    async def _handle_tribe_model_toggle(self, params: dict, ws: ServerConnection) -> dict:
        if not self._tribe_engine:
            return {"error": "TRIBE engine not initialized"}
        action = params.get("action", "status")
        if action == "load":
            success = await self._tribe_engine.load_model()
            return {"loaded": success, "fallback": self._tribe_engine.is_fallback}
        elif action == "unload":
            self._tribe_engine.unload_model()
            return {"loaded": False}
        return {
            "loaded": self._tribe_engine.is_loaded,
            "fallback": self._tribe_engine.is_fallback,
            "available": self._tribe_engine.is_available,
        }

    # ── Voice Engine Streaming (JARVIS Mode) ──

    async def _voice_command_dispatch(self, command_text: str) -> None:
        logger.info("Voice command received: '%s'", command_text)
        await self.broadcast("voice_command", {"command": command_text, "status": "executing"})

        try:
            screen_ctx = ""
            if self._screen_vision:
                try:
                    base_ctx = self._screen_vision.get_context_for_planner()
                    screen_ctx = f"{base_ctx}\nUser language: {language}"
                except Exception:
                    screen_ctx = f"User language: {language}"
            else:
                screen_ctx = f"User language: {language}"

            plan = await self._planner.plan(command_text, screen_context=screen_ctx)
            if plan.error:
                await self.broadcast(
                    "voice_result",
                    {"command": command_text, "status": "error", "message": plan.error},
                )
                from pilot.system.voice import speak

                await speak(f"Sorry, I couldn't process that. {plan.error[:100]}")
                return

            await self.broadcast(
                "plan_preview",
                {
                    "plan_id": "voice",
                    "actions": [a.model_dump() for a in plan.actions],
                    "explanation": plan.explanation,
                    "source": "voice",
                    "language": language,
                },
            )

            results = await self._executor.execute_plan(plan)
            verification = await self._verifier.verify(plan, results)

            output_parts = []
            for r in results:
                if r.output:
                    output_parts.append(r.output[:200])

            result_text = " ".join(output_parts) if output_parts else plan.explanation
            status = "success" if verification.passed else "partial"

            await self.broadcast(
                "voice_result",
                {
                    "command": command_text,
                    "status": status,
                    "result": result_text[:500],
                    "language": language,
                },
            )

            from pilot.system.voice import speak

            spoken = (
                result_text[:300] if len(result_text) < 300 else result_text[:297] + "..."
            )
            await speak(f"Done. {spoken}")

        except Exception as e:
            logger.error("Voice command execution failed: %s", e)
            await self.broadcast(
                "voice_result",
                {"command": command_text, "status": "error", "message": str(e)},
            )

            try:
                from pilot.system.voice import speak

                await speak("Sorry, something went wrong while executing your request.")
            except Exception:
                pass

    async def _voice_status_broadcast(self, status: str, data: dict) -> None:
        await self.broadcast("voice_status", {"status": status, **data})

    async def _handle_voice_listener_start(self, params: dict, ws: ServerConnection) -> dict:
        from pilot.system.voice import ContinuousVoiceListener

        wake_words = params.get("wake_words", ["hey heliox", "heliox", "hey pilot"])

        if self._voice_listener and self._voice_listener.is_running:
            return {
                "status": "already_running",
                "wake_words": self._voice_listener.wake_words,
            }

        self._voice_listener = ContinuousVoiceListener(
            wake_words=wake_words,
            on_command=self._voice_command_dispatch,
            on_status=self._voice_status_broadcast,
        )
        result = await self._voice_listener.start()
        return {"status": "started", "message": result, "wake_words": wake_words}

    async def _handle_voice_listener_stop(self, params: dict, ws: ServerConnection) -> dict:
        if not self._voice_listener or not self._voice_listener.is_running:
            return {"status": "not_running"}

        result = await self._voice_listener.stop()
        return {"status": "stopped", "message": result}

    async def _handle_voice_listener_stats(self, params: dict, ws: ServerConnection) -> dict:
        if not self._voice_listener:
            return {"running": False, "message": "Voice listener not initialized"}
        return self._voice_listener.get_stats()

    # ── Autonomous Background Engine Handlers ──

    async def _handle_autonomous_submit(self, params: dict, ws: ServerConnection) -> dict:
        if not self._autonomous:
            return {"error": "Autonomous executor not initialized"}

        goal = params.get("goal", "")
        if not goal.strip():
            return {"error": "Empty goal"}

        source = params.get("source", "text")
        job = await self._autonomous.submit(goal, source=source)
        return {"status": "submitted", "job": job.to_dict()}

    async def _handle_autonomous_cancel(self, params: dict, ws: ServerConnection) -> dict:
        if not self._autonomous:
            return {"error": "Autonomous executor not initialized"}

        job_id = params.get("job_id", "")
        success = await self._autonomous.cancel(job_id)
        return {"cancelled": success, "job_id": job_id}

    async def _handle_autonomous_jobs(self, params: dict, ws: ServerConnection) -> dict:
        if not self._autonomous:
            return {"jobs": []}
        return {"jobs": self._autonomous.list_jobs()}

    async def _handle_autonomous_job(self, params: dict, ws: ServerConnection) -> dict:
        if not self._autonomous:
            return {"error": "Autonomous executor not initialized"}

        job_id = params.get("job_id", "")
        job = self._autonomous.get_job(job_id)
        if not job:
            return {"error": f"Job not found: {job_id}"}
        return job.to_dict()

    # ── Proactive Optimization Suggestions ──

    async def _handle_proactive_start(self, params: dict, ws: ServerConnection) -> dict:
        if not self._proactive:
            return {"error": "Proactive engine not initialized"}
        result = await self._proactive.start()
        return {"status": "started", "message": result}

    async def _handle_proactive_stop(self, params: dict, ws: ServerConnection) -> dict:
        if not self._proactive:
            return {"error": "Proactive engine not initialized"}
        result = await self._proactive.stop()
        return {"status": "stopped", "message": result}

    async def _handle_proactive_stats(self, params: dict, ws: ServerConnection) -> dict:
        if not self._proactive:
            return {"running": False, "message": "Proactive engine not initialized"}
        return self._proactive.get_stats()

    async def _handle_proactive_accept(self, params: dict, ws: ServerConnection) -> dict:
        if not self._proactive:
            return {"error": "Proactive engine not initialized"}

        suggestion_id = params.get("suggestion_id", "")
        action_command = await self._proactive.accept_suggestion(suggestion_id)
        if not action_command:
            return {"error": f"Suggestion not found: {suggestion_id}"}

        if self._autonomous:
            job = await self._autonomous.submit(action_command, source="proactive")
            return {
                "status": "executing",
                "action": action_command,
                "job": job.to_dict(),
            }
        else:
            screen_ctx = ""
            if self._screen_vision:
                try:
                    screen_ctx = self._screen_vision.get_context_for_planner()
                except Exception:
                    pass
            plan = await self._planner.plan(action_command, screen_context=screen_ctx)
            if plan.error:
                return {"error": plan.error}
            results = await self._executor.execute(plan)
            return {
                "status": "completed",
                "action": action_command,
                "results": [
                    {"success": r.success, "output": r.output[:200]} for r in results
                ],
            }

    async def _handle_proactive_dismiss(self, params: dict, ws: ServerConnection) -> dict:
        if not self._proactive:
            return {"error": "Proactive engine not initialized"}

        suggestion_id = params.get("suggestion_id", "")
        dismissed = await self._proactive.dismiss_suggestion(suggestion_id)
        return {"dismissed": dismissed, "suggestion_id": suggestion_id}

    # ── Plan-Level Asynchronous Audit History Log Handlers ──

    async def _handle_get_plan_history(self, params: dict[str, Any], ws: Any) -> dict:
        """RPC handler to fetch the internal plan-level audit log history using aiosqlite."""
        limit = params.get("limit", 20)
        offset = params.get("offset", 0)

        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(
                    """SELECT plan_id, critic_verdict, user_confirmation_decision, execution_outcome, timestamp 
                    FROM plan_history 
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?""",
                    (limit, offset),
                ) as cursor:
                    rows = await cursor.fetchall()

                history = [dict(row) for row in rows]
                return {"status": "ok", "history": history}
        except Exception as e:
            return {"status": "error", "message": f"Database query failed: {str(e)}"}

    async def _handle_get_plan_detail(self, params: dict[str, Any], ws: Any) -> dict:
        """RPC handler to fetch detailed JSON actions for a single specific plan using aiosqlite."""
        plan_id = params.get("plan_id", "")
        if not plan_id:
            return {"status": "error", "message": "Missing required parameter: plan_id"}

        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(
                    """SELECT plan_id, action_plan, critic_verdict, user_confirmation_decision, execution_outcome, timestamp 
                    FROM plan_history
                    WHERE plan_id = ?""",
                    (plan_id,),
                ) as cursor:
                    row = await cursor.fetchone()

                if row is None:
                    return {
                        "status": "error",
                        "message": f"Plan with ID {plan_id} not found.",
                    }

                result = dict(row)
                result["action_plan"] = json.loads(result["action_plan"])
                return {"status": "ok", "plan_detail": result}
        except Exception as e:
            return {"status": "error", "message": f"Database query failed: {str(e)}"}


# ── System Main Process Initialization ──


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
    """Entry point execution script for the pilot server daemon."""
    ensure_dirs()
    _setup_logging()
    config = PilotConfig.load()
    parser = argparse.ArgumentParser(prog="pilot.server")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without running execution blocks",
    )
    args, _ = parser.parse_known_args()
    if args.export_logs:
        export_logs()
        return
    if args.dry_run:
        config.security.dry_run = True
        logger.info("Dry-run execution mode assigned over CLI flags")
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
    
