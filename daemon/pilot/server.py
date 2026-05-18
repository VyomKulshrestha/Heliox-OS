"""WebSocket JSON-RPC 2.0 server for the Pilot daemon."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import logging
import secrets
import signal
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import websockets
from websockets.asyncio.server import Server, ServerConnection

from pilot.config import DATA_DIR, DB_FILE, LOG_FILE, STATE_DIR, PilotConfig, ensure_dirs
from pilot.export_logs import export_logs

logger = logging.getLogger("pilot.server")

CONFIRM_TIMEOUT_SECONDS = 300


@dataclass
class JsonRpcRequest:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None

    @classmethod
    def parse(cls, raw: str) -> JsonRpcRequest:
        """Parse a raw JSON-RPC request string.

        Args:
            raw: The raw JSON string to parse.

        Returns:
            A JsonRpcRequest instance.

        Raises:
            ValueError: If the JSON-RPC version is not "2.0".
        """
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
    plan: Any = None


class PilotServer:
    """Main daemon server managing WebSocket connections and agent dispatch."""

    def __init__(self, config: PilotConfig) -> None:
        """Initialize the PilotServer with the given configuration.

        Args:
            config: PilotConfig instance containing server and model settings.
        """
        self.config = config
        self._server: Server | None = None
        self._clients: set[ServerConnection] = set()
        self._handlers: dict[str, Any] = {}
        self._planner: Any = None
        self._executor: Any = None
        self._verifier: Any = None
        self._destructive_critic: Any = None
        self._reflector: Any = None
        self._multi_agent: Any = None
        self._background: Any = None
        self._orchestrator: Any = None
        self._fusion: Any = None
        self._reasoning: Any = None
        self._decomposer: Any = None
        self._sandbox: Any = None
        self._prompt_improver: Any = None
        self._plugin_registry: Any = None
        self._subconscious: Any = None
        self._screen_vision: Any = None
        self._memory: Any = None
        self._vault: Any = None
        self._permission_audit: Any = None
        self._checkpoint_store: Any = None
        # Cognitive intelligence (TRIBE v2)
        self._tribe_engine: Any = None
        self._attention_ui: Any = None
        self._stress_gate: Any = None
        self._intent_predictor: Any = None
        self._voice_listener: Any = None
        self._autonomous: Any = None
        self._proactive: Any = None
        self._budget_tracker: Any = None
        self._running = False
        self._pending_confirms: dict[str, PendingConfirmation] = {}
        # ── Cancel Token (Issue #92) ──
        self._cancel_event: asyncio.Event | None = None
        self._rss_agent: Any = None
        # ── LAN Mesh Network ──
        self._mesh: Any = None

    async def initialize(self) -> None:
        """Initialize all agent components.

        This method sets up all subsystems including the memory store,
        planner, executor, verifier, orchestrator, multimodal fusion,
        cognitive intelligence, and autonomous execution features.
        """
        from pilot.agents.background import BackgroundTaskManager
        from pilot.agents.code_agent import CodeAgent
        from pilot.agents.comm_agent import CommunicationAgent
        from pilot.agents.executor import Executor
        from pilot.agents.monitor_agent import MonitorAgent
        from pilot.agents.multi_agent import MultiAgentRouter
        from pilot.agents.orchestrator import AgentOrchestrator
        from pilot.agents.planner import Planner
        from pilot.agents.reflector import Reflector
        from pilot.agents.system_agent import SystemAgent
        from pilot.agents.verifier import Verifier
        from pilot.agents.web_agent import WebAgent
        from pilot.memory.store import MemoryStore
        from pilot.models.router import ModelRouter
        from pilot.security.audit import AuditLogger
        from pilot.security.permission_audit import PermissionEscalationAuditStore
        from pilot.security.permissions import PermissionChecker
        from pilot.security.validator import ActionValidator
        from pilot.security.vault import KeyVault
        from pilot.workflows.checkpoints import WorkflowCheckpointStore

        self._vault = KeyVault(self.config)
        model_router = ModelRouter(self.config, self._vault)
        await model_router.initialize()

        from pilot.models.budget_tracker import BudgetTracker

        self._budget_tracker = BudgetTracker(self.config.model, str(DB_FILE))
        await self._budget_tracker.initialize()
        model_router.set_budget_tracker(self._budget_tracker)

        audit = AuditLogger()
        self._permission_audit = PermissionEscalationAuditStore()
        await self._permission_audit.initialize()
        self._checkpoint_store = WorkflowCheckpointStore()
        await self._checkpoint_store.initialize()
        validator = ActionValidator(self.config)
        permissions = PermissionChecker(self.config)
        self._memory = MemoryStore(checkpoint_interval_seconds=self.config.memory.checkpoint_interval_seconds)
        await self._memory.initialize()

        self._planner = Planner(model_router, self._memory)
        self._executor = Executor(self.config, validator, permissions, audit)
        self._verifier = Verifier(model_router)

        # Destructive Critic Agent — secondary safety reviewer for Tier 4 plans.
        from pilot.agents.destructive_critic import DestructiveCriticAgent

        self._destructive_critic = DestructiveCriticAgent(model_router)

        # Advanced agent components
        self._reflector = Reflector(model_router)
        await self._reflector.initialize()
        self._multi_agent = MultiAgentRouter(model_router)
        self._background = BackgroundTaskManager()
        self._background.set_broadcast(self._broadcast_notification)
        self._background.register_builtin_monitors()

        # Multi-Agent Orchestrator — register all specialist agents
        self._orchestrator = AgentOrchestrator(model_router)
        self._orchestrator.set_broadcast(self._broadcast_notification)
        from pilot.agents.registry import AgentRegistry

        AgentRegistry.discover_agents()
        registered = self._orchestrator.auto_register_all_agents(
            executor=self._executor,
            background_manager=self._background,
            model_router=model_router,
        )
        logger.info("Auto-registered %d agents via dynamic discovery", registered)
        await self._orchestrator.start_all()

        from pilot.agents.rss_agent import RssAgent

        self._rss_agent = RssAgent(model_router, self._memory, self.config, self._background)
        self._orchestrator.register_agent(self._rss_agent)

        # Multimodal Fusion Engine — voice + gesture intent fusion
        from pilot.multimodal.fusion import MultimodalFusionEngine

        self._fusion = MultimodalFusionEngine()
        self._fusion.set_broadcast(self._broadcast_notification)

        # Reasoning Event Emitter — thought visualization telemetry
        from pilot.reasoning.events import ReasoningEmitter

        self._reasoning = ReasoningEmitter()
        self._reasoning.set_broadcast(self._broadcast_notification)

        # Task Decomposition Engine
        from pilot.agents.decomposer import TaskDecomposer

        self._decomposer = TaskDecomposer(model_router)

        # Simulation Sandbox — pre-execution risk analysis
        from pilot.agents.sandbox import SimulationSandbox

        self._sandbox = SimulationSandbox()

        # Self-Improving Prompt System
        from pilot.agents.prompt_improver import PromptImprover

        self._prompt_improver = PromptImprover()
        await self._prompt_improver.initialize(str(DB_FILE))

        # Plugin Ecosystem
        from pilot.plugins import PluginRegistry

        self._plugin_registry = PluginRegistry()
        plugin_count = self._plugin_registry.discover()
        logger.info("Plugins loaded: %d", plugin_count)

        # Subconscious Agent — long-term memory consolidation (lazy start)
        try:
            from pilot.agents.subconscious import SubconsciousAgent

            self._subconscious = SubconsciousAgent(model_router)
            await self._subconscious.initialize(str(DB_FILE))
            logger.info("SubconsciousAgent initialized (idle, use persona_consolidate to trigger)")
        except Exception:
            logger.warning("SubconsciousAgent init failed (non-critical)", exc_info=True)

        # Cognitive Hub — unified TRIBE v2 cognitive features
        try:
            from pilot.changelog import announce_new_features, mark_version_seen
            from pilot.cognitive.hub import CognitiveHub

            self._cognitive_hub = CognitiveHub()
            logger.info("CognitiveHub initialized with TRIBE v2")

            announcement = announce_new_features()
            if announcement:
                logger.info("New features announcement: %s", announcement)
                self._new_features_announcement = announcement
                mark_version_seen()
        except Exception:
            logger.warning("CognitiveHub init failed (non-critical)", exc_info=True)
            self._new_features_announcement = None

        # Screen Vision Agent — continuous screen awareness (AUTO-START for JARVIS mode)
        try:
            from pilot.agents.screen_vision import ScreenVisionAgent

            self._screen_vision = ScreenVisionAgent(model_router)
            interval_seconds = self.config.screen_vision.capture_interval_seconds
            asyncio.create_task(self._screen_vision.start(interval_seconds=interval_seconds, enable_describe=False))
            logger.info("ScreenVisionAgent auto-started (every %.1fs, JARVIS mode)", interval_seconds)
        except Exception:
            logger.warning("ScreenVisionAgent init failed (non-critical)", exc_info=True)

        # ── Cognitive Intelligence (TRIBE v2) ──
        try:
            from pilot.cognitive.attention_scorer import AttentionAwareUI
            from pilot.cognitive.intent_predictor import IntentPredictor
            from pilot.cognitive.stress_gate import StressGate
            from pilot.cognitive.tribe_engine import TribeEngine

            self._tribe_engine = TribeEngine.get_instance()
            self._attention_ui = AttentionAwareUI(self._tribe_engine)
            self._attention_ui.set_broadcast(self._broadcast_notification)
            self._stress_gate = StressGate(self._tribe_engine)
            self._intent_predictor = IntentPredictor(self._tribe_engine)

            if self._executor:
                self._executor._stress_gate = self._stress_gate
            if self._fusion:
                self._fusion._intent_predictor = self._intent_predictor
            if getattr(self, "_screen_vision", None):
                self._screen_vision._tribe_engine = self._tribe_engine

            asyncio.create_task(self._tribe_engine.load_model())
            logger.info(
                "Cognitive intelligence initialized (TRIBE v2 %s)",
                "loading" if self._tribe_engine.is_available else "fallback mode",
            )
        except Exception:
            logger.warning("Cognitive intelligence init failed (non-critical)", exc_info=True)

        self._notification_buffer: list[tuple[str, dict[str, Any]]] = []

        # ── Autonomous Executor (JARVIS fire-and-forget) ──
        try:
            from pilot.agents.autonomous import AutonomousExecutor

            self._autonomous = AutonomousExecutor(
                planner=self._planner,
                executor=self._executor,
                verifier=self._verifier,
                decomposer=self._decomposer,
                screen_vision=self._screen_vision,
            )
            self._autonomous.set_broadcast(self._broadcast_notification)
            logger.info("AutonomousExecutor initialized")
        except Exception:
            logger.warning("AutonomousExecutor init failed (non-critical)", exc_info=True)

        # ── Proactive Suggestion Engine (JARVIS anticipation) ──
        try:
            from pilot.agents.proactive import ProactiveSuggestionEngine

            self._proactive = ProactiveSuggestionEngine(screen_vision=self._screen_vision)
            self._proactive.set_broadcast(self._broadcast_notification)
            asyncio.create_task(self._proactive.start())
            logger.info("ProactiveSuggestionEngine auto-started")
        except Exception:
            logger.warning("ProactiveSuggestionEngine init failed (non-critical)", exc_info=True)

        self._handlers = {
            "execute": self._handle_execute,
            "resume_plan": self._handle_resume_plan,
            "export_session_chat": self._handle_export_session_chat,
            "confirm": self._handle_confirm,
            # ── Cancel Token (Issue #92) ──
            "abort": self._handle_abort,
            "get_config": self._handle_get_config,
            "update_config": self._handle_update_config,
            "get_history": self._handle_get_history,
            "memory_checkpoint": self._handle_memory_checkpoint,
            "store_api_key": self._handle_store_api_key,
            "delete_api_key": self._handle_delete_api_key,
            "list_api_keys": self._handle_list_api_keys,
            "list_ollama_models": self._handle_list_ollama_models,
            "health": self._handle_health,
            "ping": self._handle_ping,
            "system_status": self._handle_system_status,
            "capabilities": self._handle_capabilities,
            "reflection_stats": self._handle_reflection_stats,
            "background_tasks": self._handle_background_tasks,
            "background_start": self._handle_background_start,
            "background_stop": self._handle_background_stop,
            "agent_routing": self._handle_agent_routing,
            "agent_stats": self._handle_agent_stats,
            "agent_capabilities": self._handle_agent_capabilities,
            "agent_spawn": self._handle_agent_spawn,
            "voice_event": self._handle_voice_event,
            "gesture_event": self._handle_gesture_event,
            "multimodal_stats": self._handle_multimodal_stats,
            "reasoning_log": self._handle_reasoning_log,
            "reasoning_stats": self._handle_reasoning_stats,
            "decompose_task": self._handle_decompose_task,
            "simulate_plan": self._handle_simulate_plan,
            "prompt_strategies": self._handle_prompt_strategies,
            "prompt_stats": self._handle_prompt_stats,
            "plugin_list": self._handle_plugin_list,
            "plugin_tools": self._handle_plugin_tools,
            "plugin_toggle": self._handle_plugin_toggle,
            "plugin_market_list": self._handle_plugin_market_list,
            "plugin_install": self._handle_plugin_install,
            "plugin_uninstall": self._handle_plugin_uninstall,
            "persona_rules": self._handle_persona_rules,
            "persona_consolidate": self._handle_persona_consolidate,
            "persona_add_preference": self._handle_persona_add_preference,
            "subconscious_stats": self._handle_subconscious_stats,
            "screen_context": self._handle_screen_context,
            "screen_current_app": self._handle_screen_current_app,
            "screen_vision_stats": self._handle_screen_vision_stats,
            "screen_vision_toggle": self._handle_screen_vision_toggle,
            "cognitive_stats": self._handle_cognitive_stats,
            "cognitive_state": self._handle_cognitive_state,
            "attention_toggle": self._handle_attention_toggle,
            "stress_gate_toggle": self._handle_stress_gate_toggle,
            "intent_predictor_toggle": self._handle_intent_predictor_toggle,
            "tribe_model_toggle": self._handle_tribe_model_toggle,
            "voice_listener_start": self._handle_voice_listener_start,
            "voice_listener_stop": self._handle_voice_listener_stop,
            "voice_listener_stats": self._handle_voice_listener_stats,
            "autonomous_submit": self._handle_autonomous_submit,
            "autonomous_cancel": self._handle_autonomous_cancel,
            "autonomous_jobs": self._handle_autonomous_jobs,
            "autonomous_job": self._handle_autonomous_job,
            "proactive_start": self._handle_proactive_start,
            "proactive_stop": self._handle_proactive_stop,
            "proactive_stats": self._handle_proactive_stats,
            "proactive_accept": self._handle_proactive_accept,
            "proactive_dismiss": self._handle_proactive_dismiss,
            "budget_stats": self._handle_budget_stats,
            "budget_reset": self._handle_budget_reset,
            # ── LAN Mesh Network ──
            "mesh_peers": self._handle_mesh_peers,
            "mesh_status": self._handle_mesh_status,
        }

        # ── LAN Mesh Network (opt-in via config) ──
        if self.config.network.enabled:
            try:
                from pilot.network.mesh import HelioxMesh
                from pilot.system.plugins import get_manager as get_plugin_manager

                self._mesh = HelioxMesh(
                    config=self.config.network,
                    executor=self._executor,
                    plugin_manager=get_plugin_manager(),
                )
                logger.info("HelioxMesh initialised (will start with server)")
            except Exception:
                logger.warning("HelioxMesh init failed (non-critical)", exc_info=True)

    async def _broadcast_notification(self, method: str, params: Any) -> None:
        """Broadcast a notification to all connected clients.

        Args:
            method: The notification method name.
            params: The notification parameters.
        """
        # ── Feature 5: Attention-Optimized Notification Timing ──
        # task_complete always bypasses the attention gate — it is the user-facing
        # completion signal and must never be buffered or suppressed.
        if method == "task_complete":
            pass
        elif getattr(self, "_attention_ui", None) and self._attention_ui.enabled:
            try:
                content = params if isinstance(params, dict) else {"data": params}
                scored = await self._attention_ui.score_event(method, content)

                # Buffer non-critical notifications when user is highly focused
                # Fix: scored.priority is a plain str, not an enum — compare directly.
                if not scored.should_display and scored.priority != "critical":
                    if not hasattr(self, "_notification_buffer"):
                        self._notification_buffer = []
                    self._notification_buffer.append((method, params.copy() if isinstance(params, dict) else params))
                    return

                if scored.attention_score < 0.4 and getattr(self, "_notification_buffer", []):
                    logger.info(
                        f"Flushing {len(self._notification_buffer)} buffered notifications during low cognitive load."
                    )
                    for b_meth, b_params in self._notification_buffer:
                        if isinstance(b_params, dict):
                            b_params.setdefault("_cognitive", {})["should_animate"] = False
                            b_params["_cognitive"]["flushed"] = True
                        msg = _notification(b_meth, b_params)
                        for client in list(self._clients):
                            try:
                                await client.send(msg)
                            except Exception:
                                pass
                    self._notification_buffer.clear()

                if isinstance(params, dict):
                    params["_cognitive"] = {
                        "priority": scored.priority,
                        "attention_score": scored.attention_score,
                        "should_animate": scored.should_animate,
                        "display_duration_ms": scored.display_duration_ms,
                    }
            except Exception as e:
                logger.error("Attention scoring failed: %s", e)

        msg = _notification(method, params)
        for client in list(self._clients):
            try:
                await client.send(msg)
            except Exception:
                pass

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket connection from a client.

        Args:
            websocket: The WebSocket connection to the client.
        """
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
        """Dispatch a JSON-RPC request to the appropriate handler.

        Args:
            request: The parsed JSON-RPC request.
            ws: The WebSocket connection.

        Returns:
            A JSON-RPC response string, or None for notifications.
        """
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
        dry_run = bool(params.get("dry_run", self.config.security.dry_run))

        # ── Cancel Token (Issue #92): fresh event per execution session ──
        self._cancel_event = asyncio.Event()
        cancel_event = self._cancel_event

        import time

        from pilot.reasoning.events import (
            CONFIRMATION_APPROVED,
            CONFIRMATION_DENIED,
            CONFIRMATION_REQUIRED,
            CRITIC_REVIEW_APPROVED,
            CRITIC_REVIEW_BLOCKED,
            CRITIC_REVIEW_STARTED,
            CRITIC_REVIEW_WARNED,
            EXECUTOR_ACTION_COMPLETE,
            EXECUTOR_ACTION_STARTED,
            EXECUTOR_ALL_COMPLETE,
            EXECUTOR_ERROR,
            EXECUTOR_STARTED,
            MEMORY_CONTEXT_LOADED,
            MEMORY_SEARCH_STARTED,
            MEMORY_STORE_COMPLETE,
            MEMORY_STORE_STARTED,
            ORCHESTRATOR_AGENT_DELEGATED,
            ORCHESTRATOR_ROUTING,
            PLANNER_ERROR,
            PLANNER_GENERATED_PLAN,
            PLANNER_LLM_CALL,
            PLANNER_REPLANNING,
            PLANNER_STARTED,
            REFLECTION_COMPLETE,
            REFLECTION_STARTED,
            ROUTING_AGENTS_ASSIGNED,
            ROUTING_ANALYSIS_STARTED,
            VERIFICATION_FAILED,
            VERIFICATION_PASSED,
            VERIFICATION_STARTED,
        )

        _start_time = time.time()
        last_plan_id = ""

        def _sanitize_summary(text: str, limit: int = 160) -> str:
            clean = " ".join(str(text).split())
            if len(clean) <= limit:
                return clean
            return clean[: max(0, limit - 3)] + "..."

        async def _emit_task_complete(status: str, summary: str) -> None:
            try:
                duration_ms = int((time.time() - _start_time) * 1000)
                payload = {
                    "status": status,
                    "summary": _sanitize_summary(summary),
                    "duration_ms": duration_ms,
                    "dry_run": dry_run,
                }
                if last_plan_id:
                    payload["plan_id"] = last_plan_id
                await self._broadcast_notification("task_complete", payload)
            except Exception:
                pass

        emit = self._reasoning
        if emit:
            emit.reset()

        input_phase = ""
        await ws.send(_notification("status", {"phase": "receiving input"}))
        if emit:
            input_phase = await emit.phase_start("user_input", "user_input_received", {"input": user_input})
            await emit.phase_complete(
                "user_input", "user_input_received", {"length": len(user_input)}, parent_id=input_phase
            )

        mem_phase = ""
        await ws.send(_notification("status", {"phase": "recalling memory"}))
        if emit:
            mem_phase = await emit.phase_start("memory_recall", MEMORY_SEARCH_STARTED)

        improvement_ctx = await self._reflector.get_improvement_context(user_input)

        if emit:
            await emit.thought(
                "memory_recall", "Searching long-term memory for relevant context...", parent_id=mem_phase
            )
            await emit.phase_complete(
                "memory_recall", MEMORY_CONTEXT_LOADED, {"has_context": bool(improvement_ctx)}, parent_id=mem_phase
            )

        route_phase = ""
        await ws.send(_notification("status", {"phase": "routing agents"}))
        if emit:
            route_phase = await emit.phase_start("agent_routing", ROUTING_ANALYSIS_STARTED, {"input": user_input})

        routing = self._multi_agent.get_routing_summary(user_input)
        await ws.send(_notification("agent_routing", routing))

        if emit:
            await emit.decision(
                "agent_routing",
                "Route to specialist agents",
                options=[r.value for r in self._orchestrator._agents] if self._orchestrator else [],
                chosen=", ".join(routing.get("assigned_agents", [])),
                parent_id=route_phase,
            )
            await emit.phase_complete("agent_routing", ROUTING_AGENTS_ASSIGNED, routing, parent_id=route_phase)

        error_context = improvement_ctx
        all_results: list = []
        last_verification = None
        last_explanation = ""

        for attempt in range(1 + self.MAX_RETRIES):
            # ── Cancel Token: check before each planning attempt ──
            if cancel_event.is_set():
                logger.info("Execution cancelled before attempt %d", attempt + 1)
                return {"status": "cancelled", "message": "Execution was aborted by user."}

            plan_phase = ""
            if emit:
                event_name = PLANNER_STARTED if attempt == 0 else PLANNER_REPLANNING
                plan_phase = await emit.phase_start("planning", event_name, {"attempt": attempt + 1})
                await emit.thought("planning", "Generating structured action plan via LLM...", parent_id=plan_phase)

            if attempt == 0:
                await ws.send(_notification("status", {"phase": "planning"}))
            else:
                await ws.send(_notification("status", {"phase": f"re-planning (attempt {attempt + 1})"}))

            if emit:
                await emit.data_event("planning", PLANNER_LLM_CALL, {"model": "active"}, parent_id=plan_phase)

            _screen_ctx = ""
            if self._screen_vision:
                try:
                    _screen_ctx = self._screen_vision.get_context_for_planner()
                except Exception:
                    pass

            async def stream_token(token: str) -> None:
                await ws.send(_notification("token_stream", {"token": token}))

            stream_callback = stream_token if attempt == 0 else None

            plan = await self._planner.plan(
                user_input, error_context=error_context, screen_context=_screen_ctx, stream_callback=stream_callback
            )
            if plan.error:
                if emit:
                    await emit.phase_error("planning", PLANNER_ERROR, plan.error, parent_id=plan_phase)
                if attempt < self.MAX_RETRIES:
                    error_context = plan.error
                    continue
                await _emit_task_complete("error", plan.error)
                return {"status": "error", "message": plan.error}

            last_explanation = plan.explanation
            plan_id = str(uuid.uuid4())[:8]
            last_plan_id = plan_id
            if self._checkpoint_store:
                await self._checkpoint_store.start_plan(plan_id, user_input, plan)

            if emit:
                await emit.phase_complete(
                    "planning",
                    PLANNER_GENERATED_PLAN,
                    {
                        "plan_id": plan_id,
                        "action_count": len(plan.actions),
                        "explanation": plan.explanation[:120],
                        "action_types": [a.action_type.value for a in plan.actions],
                    },
                    parent_id=plan_phase,
                )

            await ws.send(
                _notification(
                    "plan_preview",
                    {
                        "plan_id": plan_id,
                        "actions": [a.model_dump() for a in plan.actions],
                        "explanation": plan.explanation,
                        "dry_run": dry_run,
                    },
                )
            )

            from pilot.actions import PermissionTier

            critic_verdict_payload: dict[str, Any] | None = None
            plan_has_tier4 = any(a.permission_tier == PermissionTier.ROOT_CRITICAL for a in plan.actions)
            if plan_has_tier4 and self._destructive_critic and not dry_run:
                critic_phase = ""
                await ws.send(_notification("status", {"phase": "critic review"}))
                if emit:
                    critic_phase = await emit.phase_start(
                        "critic_review",
                        CRITIC_REVIEW_STARTED,
                        {"plan_id": plan_id, "action_count": len(plan.actions)},
                    )
                    await emit.thought(
                        "critic_review",
                        "Tier 4 actions detected — running independent safety review...",
                        parent_id=critic_phase,
                    )

                verdict = await self._destructive_critic.review(user_input, plan)
                critic_verdict_payload = verdict.to_dict()
                await ws.send(_notification("critic_verdict", verdict.to_dict()))

                if verdict.is_blocked:
                    await self._record_permission_escalations(
                        plan_id=plan_id,
                        plan=plan,
                        confirmation_decision="blocked_by_critic",
                        critic_verdict=critic_verdict_payload,
                        results=[],
                        execution_error=verdict.recommendation,
                    )
                    if emit:
                        await emit.phase_error(
                            "critic_review",
                            CRITIC_REVIEW_BLOCKED,
                            verdict.recommendation,
                            parent_id=critic_phase,
                        )
                    return {
                        "status": "blocked_by_critic",
                        "verdict": verdict.to_dict(),
                        "message": (f"Plan blocked by safety critic: {verdict.recommendation}"),
                        "explanation": plan.explanation,
                    }

                if emit:
                    event_name = CRITIC_REVIEW_WARNED if verdict.has_warnings else CRITIC_REVIEW_APPROVED
                    await emit.phase_complete(
                        "critic_review",
                        event_name,
                        verdict.to_dict(),
                        parent_id=critic_phase,
                    )

            needs_confirm = any(a.requires_confirmation for a in plan.actions) and not dry_run
            if needs_confirm:
                confirm_phase = ""
                if emit:
                    confirm_phase = await emit.phase_start("confirmation", CONFIRMATION_REQUIRED, {"plan_id": plan_id})
                    await emit.thought(
                        "confirmation",
                        "Dangerous action detected — awaiting user approval...",
                        parent_id=confirm_phase,
                    )

                confirmed = await self._wait_for_confirmation(plan_id, plan, ws)

                if emit:
                    if confirmed:
                        await emit.phase_complete(
                            "confirmation", CONFIRMATION_APPROVED, {"plan_id": plan_id}, parent_id=confirm_phase
                        )
                    else:
                        await emit.phase_error(
                            "confirmation", CONFIRMATION_DENIED, "User denied the plan", parent_id=confirm_phase
                        )

                if not confirmed:
                    await self._record_permission_escalations(
                        plan_id=plan_id,
                        plan=plan,
                        confirmation_decision="denied",
                        critic_verdict=critic_verdict_payload,
                        results=[],
                        execution_error="Plan was denied by user.",
                    )
                    await _emit_task_complete("cancelled", "Plan was denied by user.")
                    return {
                        "status": "cancelled",
                        "message": "Plan was denied by user.",
                        "explanation": plan.explanation,
                    }
            elif not dry_run:
                if emit:
                    skip_phase = await emit.phase_start("confirmation", "confirmation_skipped")
                    await emit.phase_complete(
                        "confirmation", "confirmation_skipped", {"reason": "No dangerous actions"}, parent_id=skip_phase
                    )

            exec_phase = ""
            if emit:
                exec_phase = await emit.phase_start("execution", EXECUTOR_STARTED, {"action_count": len(plan.actions)})

            await ws.send(_notification("status", {"phase": "executing"}))
            action_idx = 0
            _total_actions = len(plan.actions)

            async def _on_action_start(
                action: Any, _exec_phase: str = exec_phase, _total: int = _total_actions
            ) -> None:
                nonlocal action_idx
                action_payload = action.model_dump()
                if dry_run:
                    action_payload["dry_run"] = True
                await ws.send(_notification("action_start", {"action": action_payload}))
                if emit:
                    action_idx += 1
                    await emit.data_event(
                        "execution",
                        EXECUTOR_ACTION_STARTED,
                        {"action_type": action.action_type.value, "target": action.target, "index": action_idx},
                        parent_id=_exec_phase,
                    )
                    await emit.progress(
                        "execution", action_idx, _total, label=action.action_type.value, parent_id=_exec_phase
                    )

            async def _on_action_complete(result: Any, _exec_phase: str = exec_phase, _plan_id: str = plan_id) -> None:
                result_payload = result.model_dump()
                if dry_run:
                    result_payload["dry_run"] = True
                await ws.send(_notification("action_complete", {"result": result_payload}))
                if self._checkpoint_store and result.success:
                    await self._checkpoint_store.record_result(_plan_id, result)
                if emit:
                    event_name = EXECUTOR_ACTION_COMPLETE if result.success else EXECUTOR_ERROR
                    await emit.data_event(
                        "execution",
                        event_name,
                        {"success": result.success, "error": result.error or ""},
                        parent_id=_exec_phase,
                    )

            if self._orchestrator:
                orch_routing = self._orchestrator.get_routing_summary(plan)
                await ws.send(_notification("orchestrator_routing", orch_routing))
                if emit:
                    await emit.data_event("orchestration", ORCHESTRATOR_ROUTING, orch_routing, parent_id=exec_phase)
                    for agent_info in orch_routing.get("assigned_agents", []):
                        role_name = agent_info["role"] if isinstance(agent_info, dict) else str(agent_info)
                        await emit.thought("orchestration", f"Delegating to {role_name} agent...", parent_id=exec_phase)

                results = await self._orchestrator.execute_plan(
                    user_input,
                    plan,
                    on_action_start=_on_action_start,
                    on_action_complete=_on_action_complete,
                    cancel_event=cancel_event,  # ── Cancel Token (Issue #92) ──
                    plan_id=plan_id,
                )
            else:
                results = await self._executor.execute(
                    plan,
                    on_action_start=_on_action_start,
                    on_action_complete=_on_action_complete,
                    cancel_event=cancel_event,  # ── Cancel Token (Issue #92) ──
                    plan_id=plan_id,
                )
            all_results = results
            if needs_confirm and not dry_run:
                await self._record_permission_escalations(
                    plan_id=plan_id,
                    plan=plan,
                    confirmation_decision="approved",
                    critic_verdict=critic_verdict_payload,
                    results=results,
                )

            # ── Cancel Token: if aborted mid-execution, return immediately ──
            if cancel_event.is_set():
                logger.info("Execution was cancelled mid-plan after %d result(s)", len(results))
                await ws.send(_notification("status", {"phase": "aborted"}))
                if self._checkpoint_store:
                    await self._checkpoint_store.mark_status(plan_id, "cancelled")
                return {
                    "status": "cancelled",
                    "message": "Execution was aborted by user.",
                    "results": [r.model_dump() for r in results],
                }

            if emit:
                successes = sum(1 for r in results if r.success)
                await emit.phase_complete(
                    "execution",
                    EXECUTOR_ALL_COMPLETE,
                    {"total": len(results), "successes": successes, "failures": len(results) - successes},
                    parent_id=exec_phase,
                )

            verify_phase = ""
            if emit:
                verify_phase = await emit.phase_start("verification", VERIFICATION_STARTED)
                await emit.thought(
                    "verification", "Checking execution results against expected outcomes...", parent_id=verify_phase
                )

            await ws.send(_notification("status", {"phase": "verifying"}))
            if dry_run:
                from pilot.actions import VerificationResult

                verification = VerificationResult(
                    passed=True,
                    details=["Dry run completed: no actions were executed."],
                    failed_actions=[],
                    rollback_triggered=False,
                )
            else:
                verification = await self._verifier.verify(plan, results)
            last_verification = verification

            if verification.passed:
                if emit:
                    await emit.phase_complete(
                        "verification",
                        VERIFICATION_PASSED,
                        {"details": verification.details[:3]},
                        parent_id=verify_phase,
                    )

                if emit:
                    refl_phase = await emit.phase_start("reflection", REFLECTION_STARTED)
                    await emit.thought(
                        "reflection", "Analyzing performance and extracting lessons...", parent_id=refl_phase
                    )
                    duration_ms = int((time.time() - _start_time) * 1000)
                    await emit.metric("reflection", "total_duration_ms", duration_ms, unit="ms", parent_id=refl_phase)
                    await emit.phase_complete(
                        "reflection", REFLECTION_COMPLETE, {"retry_count": attempt}, parent_id=refl_phase
                    )

                if emit:
                    mem_store_phase = await emit.phase_start("memory_update", MEMORY_STORE_STARTED)
                    await emit.thought(
                        "memory_update", "Persisting interaction to long-term memory...", parent_id=mem_store_phase
                    )

                asyncio.create_task(self._memory.record(user_input, plan, results))
                if self._checkpoint_store:
                    await self._checkpoint_store.mark_status(plan_id, "complete")
                asyncio.create_task(
                    self._reflector.reflect(
                        user_input,
                        plan,
                        results,
                        verification,
                        retry_count=attempt,
                        duration_ms=int((time.time() - _start_time) * 1000),
                    )
                )

                if emit:
                    await emit.phase_complete(
                        "memory_update", MEMORY_STORE_COMPLETE, {"saved": True}, parent_id=mem_store_phase
                    )

                await _emit_task_complete("success", plan.explanation or "Task completed successfully.")
                return {
                    "status": "success",
                    "dry_run": dry_run,
                    "results": [r.model_dump() for r in results],
                    "verification": verification.model_dump(),
                    "explanation": (
                        f"(dry run) {plan.explanation}"
                        if dry_run and plan.explanation
                        else "(dry run) Dry run completed: no changes were made."
                        if dry_run
                        else plan.explanation
                    ),
                    "agent_routing": self._multi_agent.get_routing_summary(user_input),
                }

            if emit:
                await emit.phase_error(
                    "verification", VERIFICATION_FAILED, "; ".join(verification.details[:3]), parent_id=verify_phase
                )

            failed_details = [d for d in verification.details if "FAILED" in d or "MISMATCH" in d]
            error_msgs = [r.error for r in results if r.error]
            error_context = "\n".join(failed_details + error_msgs)

            if attempt < self.MAX_RETRIES:
                await ws.send(
                    _notification(
                        "status",
                        {"phase": "retrying — previous attempt failed"},
                    )
                )
                if emit:
                    await emit.thought(
                        "planning", f"Retry {attempt + 1}: Re-planning with error context...", parent_id=""
                    )
            else:
                break

        if emit:
            mem_final = await emit.phase_start("memory_update", MEMORY_STORE_STARTED)
            await emit.phase_complete("memory_update", MEMORY_STORE_COMPLETE, {"partial": True}, parent_id=mem_final)

        asyncio.create_task(self._memory.record(user_input, plan, all_results))
        if self._checkpoint_store and last_plan_id:
            await self._checkpoint_store.mark_status(last_plan_id, "failed")
        await _emit_task_complete("partial_failure", last_explanation or "Task completed with errors.")
        return {
            "status": "partial_failure",
            "dry_run": dry_run,
            "results": [r.model_dump() for r in all_results],
            "verification": last_verification.model_dump() if last_verification else {},
            "explanation": (
                f"(dry run) {last_explanation}"
                if dry_run and last_explanation
                else "(dry run) Dry run completed: no changes were made."
                if dry_run
                else last_explanation
            ),
        }

    async def _handle_resume_plan(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Resume a previously checkpointed plan from its last completed action."""
        plan_id = str(params.get("plan_id", "")).strip()
        if not plan_id:
            return {"status": "error", "message": "resume_plan requires plan_id"}
        if not self._checkpoint_store:
            return {"status": "error", "message": "Workflow checkpoint store is not initialized"}

        checkpoint = await self._checkpoint_store.get(plan_id)
        if checkpoint is None:
            return {"status": "error", "message": f"No checkpoint found for plan_id: {plan_id}"}

        completed_count = max(0, min(checkpoint.completed_count, len(checkpoint.plan.actions)))
        remaining_actions = checkpoint.plan.actions[completed_count:]
        await ws.send(
            _notification(
                "status",
                {
                    "phase": "resuming",
                    "plan_id": plan_id,
                    "completed_actions": completed_count,
                    "remaining_actions": len(remaining_actions),
                },
            )
        )

        if not remaining_actions:
            await self._checkpoint_store.mark_status(plan_id, "complete")
            return {
                "status": "success",
                "plan_id": plan_id,
                "resumed": False,
                "message": "Plan already completed.",
                "results": [result.model_dump() for result in checkpoint.results],
            }

        self._cancel_event = asyncio.Event()
        cancel_event = self._cancel_event

        from pilot.actions import ActionPlan

        remaining_plan = ActionPlan(
            actions=remaining_actions,
            explanation=checkpoint.plan.explanation,
            raw_input=checkpoint.plan.raw_input,
        )

        async def _on_action_start(action: Any) -> None:
            await ws.send(_notification("action_start", {"action": action.model_dump(), "resumed": True}))

        async def _on_action_complete(result: Any) -> None:
            await ws.send(_notification("action_complete", {"result": result.model_dump(), "resumed": True}))
            if result.success:
                await self._checkpoint_store.record_result(plan_id, result)

        await self._checkpoint_store.mark_status(plan_id, "resuming")
        results = await self._executor.execute(
            remaining_plan,
            on_action_start=_on_action_start,
            on_action_complete=_on_action_complete,
            cancel_event=cancel_event,
            plan_id=plan_id,
            initial_last_output=checkpoint.last_output,
        )

        updated = await self._checkpoint_store.get(plan_id)
        combined_results = [
            *(updated.results if updated else checkpoint.results),
            *[r for r in results if not r.success],
        ]

        if cancel_event.is_set():
            await self._checkpoint_store.mark_status(plan_id, "cancelled")
            return {
                "status": "cancelled",
                "plan_id": plan_id,
                "resumed": True,
                "completed_actions": updated.completed_count if updated else completed_count,
                "results": [result.model_dump() for result in combined_results],
            }

        failed = any(not result.success for result in results)
        final_status = "failed" if failed else "complete"
        await self._checkpoint_store.mark_status(plan_id, final_status)

        verification_payload: dict[str, Any] = {}
        if not failed and len(combined_results) >= len(checkpoint.plan.actions):
            verification = await self._verifier.verify(checkpoint.plan, combined_results)
            verification_payload = verification.model_dump()
            if not verification.passed:
                final_status = "partial_failure"
                await self._checkpoint_store.mark_status(plan_id, "failed")

        return {
            "status": "partial_failure" if final_status in {"failed", "partial_failure"} else "success",
            "plan_id": plan_id,
            "resumed": True,
            "skipped_actions": completed_count,
            "executed_actions": len(results),
            "results": [result.model_dump() for result in combined_results],
            "verification": verification_payload,
            "explanation": checkpoint.plan.explanation,
        }

    async def _record_permission_escalations(
        self,
        *,
        plan_id: str,
        plan: Any,
        confirmation_decision: str,
        critic_verdict: dict[str, Any] | None,
        results: list[Any],
        execution_error: str = "",
    ) -> None:
        """Persist tamper-evident records for elevated permission decisions."""
        if not self._permission_audit:
            return

        from pilot.actions import PermissionTier

        result_by_action: dict[str, list[Any]] = {}
        for result in results:
            action_key = self._action_signature(result.action)
            result_by_action.setdefault(action_key, []).append(result)

        for index, action in enumerate(plan.actions):
            if action.permission_tier < PermissionTier.SYSTEM_MODIFY:
                continue

            matched_result = None
            matches = result_by_action.get(self._action_signature(action))
            if matches:
                matched_result = matches.pop(0)

            if matched_result is None:
                execution_success = None
                action_error = execution_error
            else:
                execution_success = bool(matched_result.success)
                action_error = matched_result.error or ""

            await self._permission_audit.record_event(
                plan_id=plan_id,
                action_index=index,
                action_type=action.action_type.value,
                target=action.target,
                permission_tier=action.permission_tier.name,
                requires_root=action.requires_root,
                destructive=action.destructive,
                confirmation_decision=confirmation_decision,
                critic_verdict=critic_verdict,
                execution_success=execution_success,
                execution_error=action_error,
            )

    @staticmethod
    def _action_signature(action: Any) -> str:
        return json.dumps(action.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    async def _wait_for_confirmation(self, plan_id: str, plan: Any, ws: ServerConnection) -> bool:
        """Send a confirmation request and block until the user responds or timeout.

        Args:
            plan_id: Unique identifier for the plan requiring confirmation.
            plan: The plan object containing actions to be confirmed.
            ws: The WebSocket connection for sending/receiving messages.

        Returns:
            True if the user approved the plan, False otherwise.
        """
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
        """Resolve a pending confirmation request from the UI.

        Args:
            params: JSON-RPC parameters containing plan_id and confirmed status.
            ws: The WebSocket connection.

        Returns:
            A dict with status and confirmation result.
        """
        plan_id = params.get("plan_id", "")
        confirmed = params.get("confirmed", False)

        pending = self._pending_confirms.get(plan_id)
        if pending is None:
            return {"status": "error", "message": f"No pending confirmation for plan_id: {plan_id}"}

        pending.confirmed = bool(confirmed)
        pending.event.set()
        return {"status": "ok", "confirmed": pending.confirmed}

    async def _handle_abort(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Signal the current execution to stop gracefully (Issue #92).

        Sets the per-session cancel_event so the Orchestrator and Executor
        halt at the next action boundary. Returns immediately — cancellation
        propagates asynchronously.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status indicating whether an active execution was aborted.
        """
        if self._cancel_event and not self._cancel_event.is_set():
            self._cancel_event.set()
            logger.info("Abort signal received — cancel_event set, propagating to agents")
            return {"status": "aborted"}
        return {"status": "no_active_execution"}

    # -- Config --

    async def _handle_get_config(self, params: dict, ws: ServerConnection) -> dict:
        """Get the current server configuration.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict containing the server configuration.
        """
        from dataclasses import asdict

        data = asdict(self.config)
        data.pop("server", None)
        return data

    async def _handle_update_config(self, params: dict, ws: ServerConnection) -> dict:
        """Update server configuration.

        Args:
            params: JSON-RPC parameters with section and values.
            ws: The WebSocket connection.

        Returns:
            A dict with status.
        """
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
                if section == "screen_vision" and k == "capture_interval_seconds":
                    v = float(v)
                setattr(target, k, v)
        self.config.save()

        if section == "screen_vision" and "capture_interval_seconds" in values and self._screen_vision:
            self._screen_vision.set_interval(self.config.screen_vision.capture_interval_seconds)

        if section == "model" and ("cloud_provider" in values or "provider" in values):
            if self.config.model.cloud_provider:
                from pilot.models.cloud import CloudClient

                self._planner._model._cloud = CloudClient(self.config, self._vault)
                logger.info("Cloud client re-initialized for provider: %s", self.config.model.cloud_provider)

        return {"status": "ok"}

    # -- History --

    async def _handle_get_history(self, params: dict, ws: ServerConnection) -> dict:
        """Get conversation history from memory store.

        Args:
            params: JSON-RPC parameters with optional limit and offset.
            ws: The WebSocket connection.

        Returns:
            A dict with entries list containing historical interactions.
        """
        limit = params.get("limit", 50)
        offset = params.get("offset", 0)
        entries = await self._memory.get_history(limit=limit, offset=offset)
        return {"entries": entries}

    async def _handle_memory_checkpoint(self, params: dict, ws: ServerConnection) -> dict:
        """Manually trigger a SQLite WAL checkpoint for the memory store.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with checkpoint status and WAL checkpoint statistics.
        """
        if not self._memory:
            return {"status": "error", "message": "Memory store is not initialized"}
        return await self._memory.checkpoint()

    async def _handle_export_session_chat(self, params: dict, ws: ServerConnection) -> dict:
        """Export current UI session chat messages to JSON or CSV."""
        fmt = str(params.get("format", "json")).lower()
        messages = params.get("messages", [])

        if fmt not in {"json", "csv"}:
            return {"status": "error", "message": "format must be 'json' or 'csv'"}
        if not isinstance(messages, list):
            return {"status": "error", "message": "messages must be a list"}

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"heliox-chat-{ts}.{fmt}"

        downloads_dir = Path.home() / "Downloads"
        export_dir = downloads_dir if downloads_dir.exists() else (DATA_DIR / "exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        out_path = export_dir / filename

        try:
            if fmt == "json":
                out_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                with out_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "timestamp_iso",
                            "timestamp_ms",
                            "msg_type",
                            "text",
                            "plan_id",
                            "plan_explanation",
                            "plan_action_count",
                            "plan_actions",
                            "result_count",
                            "result_success_count",
                            "result_error_count",
                            "result_outputs",
                            "verification_passed",
                            "verification_details",
                        ]
                    )

                    for m in messages:
                        if not isinstance(m, dict):
                            continue

                        raw_ts = m.get("timestamp")
                        iso_ts = ""
                        if isinstance(raw_ts, (int, float)):
                            iso_ts = datetime.fromtimestamp(raw_ts / 1000).isoformat()

                        msg_type = str(m.get("type", ""))
                        text = str(m.get("text", ""))

                        plan = m.get("plan", {})
                        if not isinstance(plan, dict):
                            plan = {}
                        plan_id = str(plan.get("plan_id", ""))
                        plan_explanation = str(plan.get("explanation", ""))
                        plan_actions = plan.get("actions", [])
                        if not isinstance(plan_actions, list):
                            plan_actions = []
                        plan_action_count = len(plan_actions)
                        plan_actions_str = " | ".join(
                            f"{idx + 1}. {str(a.get('action_type', ''))} -> {str(a.get('target', ''))}"
                            for idx, a in enumerate(plan_actions)
                            if isinstance(a, dict)
                        )

                        action_results = m.get("actionResults", [])
                        if not isinstance(action_results, list):
                            action_results = []
                        result_count = len(action_results)
                        result_success_count = sum(
                            1 for r in action_results if isinstance(r, dict) and bool(r.get("success", False))
                        )
                        result_error_count = result_count - result_success_count
                        result_outputs = " | ".join(
                            str(r.get("output") or r.get("error") or "").strip()
                            for r in action_results
                            if isinstance(r, dict) and (r.get("output") or r.get("error"))
                        )

                        verification = m.get("verification", {})
                        if not isinstance(verification, dict):
                            verification = {}
                        verification_passed = (
                            verification.get("passed") if isinstance(verification.get("passed"), bool) else ""
                        )
                        verification_details_raw = verification.get("details", [])
                        if not isinstance(verification_details_raw, list):
                            verification_details_raw = []
                        verification_details = " | ".join(str(d) for d in verification_details_raw)

                        writer.writerow(
                            [
                                iso_ts,
                                raw_ts if isinstance(raw_ts, (int, float)) else "",
                                msg_type,
                                text,
                                plan_id,
                                plan_explanation,
                                plan_action_count,
                                plan_actions_str,
                                result_count,
                                result_success_count,
                                result_error_count,
                                result_outputs,
                                verification_passed,
                                verification_details,
                            ]
                        )
        except Exception as e:
            logger.exception("Failed to export session chat")
            return {"status": "error", "message": f"Export failed: {e}"}

        return {
            "status": "ok",
            "path": str(out_path),
            "count": len(messages),
            "format": fmt,
        }

    # -- API key management --

    async def _handle_store_api_key(self, params: dict, ws: ServerConnection) -> dict:
        """Store an API key for a provider in the vault.

        Args:
            params: JSON-RPC parameters with provider and api_key.
            ws: The WebSocket connection.

        Returns:
            A dict with status.
        """
        provider = params.get("provider", "")
        key = params.get("api_key", "") or params.get("key", "")
        if not provider or not key:
            return {"status": "error", "message": "provider and api_key are required"}
        await self._vault.store_key(provider, key)
        if self.config.model.cloud_provider == provider:
            from pilot.models.cloud import CloudClient

            self._planner._model._cloud = CloudClient(self.config, self._vault)
        return {"status": "ok"}

    async def _handle_delete_api_key(self, params: dict, ws: ServerConnection) -> dict:
        """Delete a stored API key for a provider.

        Args:
            params: JSON-RPC parameters with provider.
            ws: The WebSocket connection.

        Returns:
            A dict with status.
        """
        provider = params.get("provider", "")
        if not provider:
            return {"status": "error", "message": "provider is required"}
        await self._vault.delete_key(provider)
        return {"status": "ok"}

    async def _handle_list_api_keys(self, params: dict, ws: ServerConnection) -> dict:
        """List all providers with stored API keys.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with providers list.
        """
        providers = await self._vault.list_providers()
        return {"providers": providers}

    # -- Ollama model discovery --

    async def _handle_list_ollama_models(self, params: dict, ws: ServerConnection) -> dict:
        """List available Ollama models.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with models list and availability status.
        """
        from pilot.models.ollama import OllamaClient

        client = OllamaClient(self.config.model.ollama_base_url)
        try:
            models = await client.list_models()
            return {"models": models, "available": True}
        except Exception:
            return {"models": [], "available": False}

    # -- Health --

    async def _handle_health(self, params: dict, ws: ServerConnection) -> dict:
        """Check the health of all model backends.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with backends health status.
        """
        from pilot.models.router import ModelRouter

        router: ModelRouter = self._planner._model
        backends = await router.check_health()
        return {"backends": backends}

    async def _handle_ping(self, params: dict, ws: ServerConnection) -> dict:
        """Ping the server to check connectivity.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with pong and version.
        """
        return {"pong": True, "version": "0.7.1"}

    async def _handle_system_status(self, params: dict, ws: ServerConnection) -> dict:
        """Return current system information.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with platform info and capabilities count.
        """
        from pilot.system.platform_detect import get_platform_info

        info = get_platform_info()
        return {
            "platform": info,
            "capabilities_count": len(self._executor._dispatch_table),
        }

    async def _handle_capabilities(self, params: dict, ws: ServerConnection) -> dict:
        """Return all available action types.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with action_types list and count.
        """
        from pilot.actions import ActionType

        return {
            "action_types": [t.value for t in ActionType],
            "count": len(ActionType),
        }

    # -- Advanced Agent Endpoints --

    async def _handle_reflection_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return self-improvement reflection statistics.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with reflection statistics from the reflector agent.
        """
        return await self._reflector.get_stats()

    async def _handle_background_tasks(self, params: dict, ws: ServerConnection) -> dict:
        """List all registered background monitoring tasks.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with list of background tasks.
        """
        return {"tasks": self._background.list_tasks()}

    async def _handle_background_start(self, params: dict, ws: ServerConnection) -> dict:
        """Start a background monitoring task.

        Args:
            params: JSON-RPC parameters with task_id.
            ws: The WebSocket connection.

        Returns:
            A dict with status and task_id.
        """
        task_id = params.get("task_id", "")
        ok = self._background.start(task_id)
        return {"status": "started" if ok else "error", "task_id": task_id}

    async def _handle_background_stop(self, params: dict, ws: ServerConnection) -> dict:
        """Stop a background monitoring task.

        Args:
            params: JSON-RPC parameters with task_id.
            ws: The WebSocket connection.

        Returns:
            A dict with status and task_id.
        """
        task_id = params.get("task_id", "")
        ok = self._background.stop(task_id)
        return {"status": "stopped" if ok else "error", "task_id": task_id}

    async def _handle_agent_routing(self, params: dict, ws: ServerConnection) -> dict:
        """Analyze which specialist agent(s) would handle a given input.

        Args:
            params: JSON-RPC parameters with input query.
            ws: The WebSocket connection.

        Returns:
            A dict with routing summary and optionally orchestrator info.
        """
        query = params.get("input", "")
        result = self._multi_agent.get_routing_summary(query)
        if self._orchestrator:
            result["orchestrator"] = self._orchestrator.get_input_routing_summary(query)
        return result

    async def _handle_agent_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return performance stats for all registered agents.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with agent performance statistics.
        """
        if self._orchestrator:
            return self._orchestrator.get_all_stats()
        return {"error": "Orchestrator not initialized"}

    async def _handle_agent_capabilities(self, params: dict, ws: ServerConnection) -> dict:
        """Return all agent capabilities grouped by specialist.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with all agent capabilities.
        """
        if self._orchestrator:
            return self._orchestrator.get_all_capabilities()
        return {"error": "Orchestrator not initialized"}

    async def _handle_agent_spawn(self, params: dict, ws: ServerConnection) -> dict:
        """Dynamically spawn a new specialist agent.

        Args:
            params: JSON-RPC parameters with role.
            ws: The WebSocket connection.

        Returns:
            A dict with status and optionally agent_id.
        """
        role_str = params.get("role", "")
        from pilot.agents.base_agent import AgentRole

        try:
            role = AgentRole(role_str)
        except ValueError:
            return {"status": "error", "message": f"Unknown role: {role_str}"}

        if self._orchestrator:
            agent = await self._orchestrator.spawn_agent(
                role,
                executor=self._executor,
                background_manager=self._background,
            )
            if agent:
                return {"status": "spawned", "agent_id": agent.agent_id}
        return {"status": "error", "message": "Failed to spawn agent"}

    # -- Multimodal Fusion --

    async def _handle_voice_event(self, params: dict, ws: ServerConnection) -> dict:
        """Receive a voice event from the frontend and feed it to fusion engine.

        Args:
            params: JSON-RPC parameters with transcript, confidence, is_final.
            ws: The WebSocket connection.

        Returns:
            A dict with status and optionally fused intent.
        """
        if not self._fusion:
            return {"status": "error", "message": "Fusion engine not initialized"}

        from pilot.multimodal.fusion import InputEvent, ModalityType

        event = InputEvent(
            modality=ModalityType.VOICE,
            transcript=params.get("transcript", ""),
            voice_confidence=params.get("confidence", 0.8),
            is_final=params.get("is_final", False),
        )
        intent = await self._fusion.on_voice_event(event)
        if intent:
            return {"status": "fused", "intent": intent.to_dict()}
        return {"status": "buffered"}

    async def _handle_gesture_event(self, params: dict, ws: ServerConnection) -> dict:
        """Receive a gesture event from the frontend and feed it to fusion engine.

        Args:
            params: JSON-RPC parameters with gesture, confidence, data.
            ws: The WebSocket connection.

        Returns:
            A dict with status and optionally fused intent.
        """
        if not self._fusion:
            return {"status": "error", "message": "Fusion engine not initialized"}

        from pilot.multimodal.fusion import InputEvent, ModalityType

        event = InputEvent(
            modality=ModalityType.GESTURE,
            gesture_name=params.get("gesture", ""),
            gesture_confidence=params.get("confidence", 0.8),
            gesture_data=params.get("data", {}),
        )
        intent = await self._fusion.on_gesture_event(event)
        if intent:
            return {"status": "fused", "intent": intent.to_dict()}
        return {"status": "buffered"}

    async def _handle_multimodal_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return multimodal fusion engine statistics.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with fusion engine stats or error.
        """
        if self._fusion:
            return self._fusion.get_stats()
        return {"error": "Fusion engine not initialized"}

    # -- Reasoning Visualization --

    async def _handle_reasoning_log(self, params: dict, ws: ServerConnection) -> dict:
        """Return the full reasoning event log for the current session.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with events list or error.
        """
        if self._reasoning:
            return {"events": self._reasoning.get_session_log()}
        return {"error": "Reasoning emitter not initialized"}

    async def _handle_reasoning_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return reasoning emitter statistics.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with reasoning stats or error.
        """
        if self._reasoning:
            return self._reasoning.get_stats()
        return {"error": "Reasoning emitter not initialized"}

    # -- Task Decomposition --

    async def _handle_decompose_task(self, params: dict, ws: ServerConnection) -> dict:
        """Decompose a complex goal into subtasks.

        Args:
            params: JSON-RPC parameters with goal.
            ws: The WebSocket connection.

        Returns:
            A dict with decomposed task structure or error.
        """
        goal = params.get("goal", "")
        if not goal:
            return {"error": "No goal provided"}
        if self._decomposer:
            decomp = await self._decomposer.decompose(goal)
            return decomp.to_dict()
        return {"error": "Decomposer not initialized"}

    # -- Simulation Sandbox --

    async def _handle_simulate_plan(self, params: dict, ws: ServerConnection) -> dict:
        """Simulate a plan and return an impact report without execution.

        Args:
            params: JSON-RPC parameters with optional plan_id.
            ws: The WebSocket connection.

        Returns:
            A dict with impact report or error.
        """
        if not self._sandbox:
            return {"error": "Sandbox not initialized"}

        plan_id = params.get("plan_id", "")
        pending = self._pending_confirms.get(plan_id)
        if pending and pending.plan:
            report = self._sandbox.simulate(pending.plan)
            return report.to_dict()

        return {"error": "No plan found to simulate"}

    # -- Self-Improving Prompt System --

    async def _handle_prompt_strategies(self, params: dict, ws: ServerConnection) -> dict:
        """Get proven prompt strategies for a task.

        Args:
            params: JSON-RPC parameters with query.
            ws: The WebSocket connection.

        Returns:
            A dict with strategies or error.
        """
        query = params.get("query", "")
        if not query:
            return {"strategies": ""}
        if self._prompt_improver:
            strategies = await self._prompt_improver.get_relevant_strategies(query)
            return {"strategies": strategies}
        return {"error": "Prompt improver not initialized"}

    async def _handle_prompt_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return prompt improvement statistics.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with prompt improvement stats or error.
        """
        if self._prompt_improver:
            return await self._prompt_improver.get_stats()
        return {"error": "Prompt improver not initialized"}

    # -- Plugin Ecosystem --

    async def _handle_plugin_list(self, params: dict, ws: ServerConnection) -> dict:
        """List all loaded plugins.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with plugin statistics or error.
        """
        if self._plugin_registry:
            return self._plugin_registry.get_stats()
        return {"error": "Plugin registry not initialized"}

    async def _handle_plugin_tools(self, params: dict, ws: ServerConnection) -> dict:
        """List all available plugin tools.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with tools list or error.
        """
        if self._plugin_registry:
            return {"tools": self._plugin_registry.get_all_tools()}
        return {"error": "Plugin registry not initialized"}

    async def _handle_plugin_toggle(self, params: dict, ws: ServerConnection) -> dict:
        """Enable or disable a plugin.

        Args:
            params: JSON-RPC parameters with name and enabled status.
            ws: The WebSocket connection.

        Returns:
            A dict with success status, plugin name, and enabled state.
        """
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
        """Fetch available plugins from the community manifest.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with plugins list from registry.json.
        """
        import json as json_module

        repo_root = Path(__file__).parent.parent.parent
        registry_path = repo_root / "plugins" / "registry.json"

        if not registry_path.exists():
            return {"plugins": [], "error": "Registry not found"}

        try:
            data = json_module.loads(registry_path.read_text(encoding="utf-8"))
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
        """Install a plugin from the marketplace.

        Args:
            params: JSON-RPC parameters with plugin_name.
            ws: The WebSocket connection.

        Returns:
            A dict with installation status.
        """
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
        """Uninstall a plugin.

        Args:
            params: JSON-RPC parameters with plugin_name.
            ws: The WebSocket connection.

        Returns:
            A dict with uninstallation status.
        """
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

    # ── Subconscious Agent Handlers ──

    async def _handle_persona_rules(self, params: dict, ws: ServerConnection) -> dict:
        """Return all persona rules.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with persona context and statistics.
        """
        if self._subconscious:
            context = await self._subconscious.get_persona_context()
            stats = await self._subconscious.get_stats()
            return {"context": context, **stats}
        return {"error": "Subconscious agent not initialized"}

    async def _handle_persona_consolidate(self, params: dict, ws: ServerConnection) -> dict:
        """Force a consolidation cycle.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with consolidation result or error.
        """
        if self._subconscious:
            result = await self._subconscious.consolidate()
            return result
        return {"error": "Subconscious agent not initialized"}

    async def _handle_persona_add_preference(self, params: dict, ws: ServerConnection) -> dict:
        """Manually add a user preference.

        Args:
            params: JSON-RPC parameters with key and value.
            ws: The WebSocket connection.

        Returns:
            A dict with status, key, and value.
        """
        key = params.get("key", "")
        value = params.get("value", "")
        if not key or not value:
            return {"error": "Both key and value required"}
        if self._subconscious:
            await self._subconscious.add_manual_preference(key, value)
            return {"status": "ok", "key": key, "value": value}
        return {"error": "Subconscious agent not initialized"}

    async def _handle_subconscious_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return subconscious agent stats.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with subconscious agent statistics.
        """
        if self._subconscious:
            return await self._subconscious.get_stats()
        return {"error": "Subconscious agent not initialized"}

    # ── Screen Vision Handlers ──

    async def _handle_screen_context(self, params: dict, ws: ServerConnection) -> dict:
        """Return the current screen context summary.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with screen context summary and details.
        """
        if self._screen_vision:
            return {
                "summary": self._screen_vision.get_context_for_planner(),
                **self._screen_vision.get_context().to_dict(),
            }
        return {"error": "Screen vision not initialized"}

    async def _handle_screen_current_app(self, params: dict, ws: ServerConnection) -> dict:
        """Return the currently active application.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with active_app name.
        """
        if self._screen_vision:
            return {"active_app": self._screen_vision.get_current_app()}
        return {"error": "Screen vision not initialized"}

    async def _handle_screen_vision_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return screen vision statistics.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with screen vision statistics.
        """
        if self._screen_vision:
            return self._screen_vision.get_stats()
        return {"error": "Screen vision not initialized"}

    async def _handle_screen_vision_toggle(self, params: dict, ws: ServerConnection) -> dict:
        """Start or stop screen vision.

        Args:
            params: JSON-RPC parameters with enabled, interval_seconds, enable_describe.
            ws: The WebSocket connection.

        Returns:
            A dict with status and enabled state.
        """
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

    # -- Broadcast --

    async def broadcast(self, method: str, params: Any) -> None:
        """Broadcast a notification to all connected clients.

        Args:
            method: The notification method name.
            params: The notification parameters.
        """
        msg = _notification(method, params)
        for client in list(self._clients):
            try:
                await client.send(msg)
            except Exception:
                self._clients.discard(client)

    # -- Lifecycle --

    async def start(self) -> None:
        """Start the Pilot daemon server.

        Initializes all subsystems, starts the WebSocket server on the
        configured host and port, and announces new features to clients.
        """
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

        # ── Start LAN mesh if enabled ──
        if self._mesh:
            asyncio.create_task(self._mesh.start())

        if hasattr(self, "_new_features_announcement") and self._new_features_announcement:
            await asyncio.sleep(1)
            await self._broadcast_notification(
                "feature_announcement",
                {
                    "message": self._new_features_announcement,
                    "version": "0.6.0",
                },
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
        from pilot.system.pty_session import PtySessionManager

        PtySessionManager.close_all()
        logger.info("Pilot daemon stopped")

    # ── Budget Tracking Handlers ──

    async def _handle_budget_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return current-month token usage and cost summary.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with token usage and cost statistics.
        """
        if not self._budget_tracker:
            return {}
        return await self._budget_tracker.get_stats()

    async def _handle_budget_reset(self, params: dict, ws: ServerConnection) -> dict:
        """Delete all token-usage records for the current month.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status.
        """
        if not self._budget_tracker:
            return {"status": "ok"}
        await self._budget_tracker.reset_current_month()
        return {"status": "ok"}

    # ── LAN Mesh Network Handlers ──

    async def _handle_mesh_peers(self, params: dict, ws: ServerConnection) -> dict:
        """Return a list of currently connected LAN peers.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with ``enabled`` flag and ``peers`` list.
        """
        if not self._mesh:
            return {"enabled": False, "peers": []}

        peers = []
        for pid in self._mesh.peer_ids:
            conn = self._mesh.get_connection(pid)
            caps = conn.peer_capabilities if conn else None
            peers.append(
                {
                    "peer_id": pid,
                    "hostname": caps.hostname if caps else "",
                    "can_execute": caps.can_execute if caps else False,
                    "cpu_load": caps.cpu_load if caps else 0.0,
                    "plugin_count": len(caps.plugin_names) if caps else 0,
                }
            )
        return {"enabled": True, "peers": peers}

    async def _handle_mesh_status(self, params: dict, ws: ServerConnection) -> dict:
        """Return overall mesh status and configuration.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with mesh status, instance ID, and config summary.
        """
        if not self._mesh:
            return {
                "enabled": False,
                "reason": "Set [network] enabled = true in config.toml to activate",
            }
        return {
            "enabled": True,
            "instance_id": self._mesh.instance_id,
            "peer_count": len(self._mesh.peer_ids),
            "skill_sync_enabled": self.config.network.skill_sync_enabled,
            "collab_exec_enabled": self.config.network.collab_exec_enabled,
            "port": self.config.network.port,
        }

    # ── Cognitive Intelligence (TRIBE v2) Handlers ──

    async def _handle_cognitive_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Get stats for all cognitive subsystems.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with stats for tribe_engine, attention_ui, stress_gate, intent_predictor.
        """
        return {
            "tribe_engine": self._tribe_engine.get_stats() if self._tribe_engine else None,
            "attention_ui": self._attention_ui.get_stats() if self._attention_ui else None,
            "stress_gate": self._stress_gate.get_stats() if self._stress_gate else None,
            "intent_predictor": (self._intent_predictor.get_stats() if self._intent_predictor else None),
        }

    async def _handle_cognitive_state(self, params: dict, ws: ServerConnection) -> dict:
        """Get current predicted cognitive state.

        Args:
            params: JSON-RPC parameters with optional stimulus description.
            ws: The WebSocket connection.

        Returns:
            A dict with current cognitive state or error.
        """
        if not self._tribe_engine:
            return {"error": "Cognitive engine not initialized"}
        state = await self._tribe_engine.predict_cognitive_state(
            stimulus_description=params.get("stimulus", ""),
        )
        return state.to_dict()

    async def _handle_attention_toggle(self, params: dict, ws: ServerConnection) -> dict:
        """Toggle attention-aware UI scoring.

        Args:
            params: JSON-RPC parameters with optional enabled flag.
            ws: The WebSocket connection.

        Returns:
            A dict with enabled state or error.
        """
        if not self._attention_ui:
            return {"error": "Attention UI not initialized"}
        enabled = self._attention_ui.toggle(params.get("enabled"))
        return {"enabled": enabled}

    async def _handle_stress_gate_toggle(self, params: dict, ws: ServerConnection) -> dict:
        """Toggle stress-aware task gating.

        Args:
            params: JSON-RPC parameters with optional enabled flag.
            ws: The WebSocket connection.

        Returns:
            A dict with enabled state or error.
        """
        if not self._stress_gate:
            return {"error": "Stress gate not initialized"}
        enabled = self._stress_gate.toggle(params.get("enabled"))
        return {"enabled": enabled}

    async def _handle_intent_predictor_toggle(self, params: dict, ws: ServerConnection) -> dict:
        """Toggle JARVIS mode intent prediction.

        Args:
            params: JSON-RPC parameters with optional enabled flag.
            ws: The WebSocket connection.

        Returns:
            A dict with enabled state or error.
        """
        if not self._intent_predictor:
            return {"error": "Intent predictor not initialized"}
        enabled = self._intent_predictor.toggle(params.get("enabled"))
        return {"enabled": enabled}

    async def _handle_tribe_model_toggle(self, params: dict, ws: ServerConnection) -> dict:
        """Load or unload the TRIBE v2 model.

        Args:
            params: JSON-RPC parameters with action (load/unload/status).
            ws: The WebSocket connection.

        Returns:
            A dict with loaded state, fallback, and availability status.
        """
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

    # ── Voice Listener (JARVIS Mode) Handlers ──

    async def _voice_command_dispatch(self, command_text: str) -> None:
        """Called by ContinuousVoiceListener when a voice command is recognized.

        Runs the full ReAct pipeline and speaks the result back.

        Args:
            command_text: The recognized voice command text.
        """
        logger.info("Voice command received: '%s'", command_text)

        language = getattr(
            self._voice_listener,
            "last_detected_language",
            self.config.voice.language if self.config.voice.language != "auto" else "en",
        )

        await self._broadcast_notification(
            "voice_command",
            {
                "command": command_text,
                "status": "executing",
                "language": language,
            },
        )

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

            # Plan with multilingual context — single call only
            plan = await self._planner.plan(command_text, screen_context=screen_ctx)
            if plan.error:
                await self._broadcast_notification(
                    "voice_result",
                    {
                        "command": command_text,
                        "status": "error",
                        "message": plan.error,
                        "language": language,
                    },
                )
                from pilot.system.voice import speak

                await speak(f"Sorry, I couldn't process that. {plan.error[:100]}")
                return

            await self._broadcast_notification(
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

            await self._broadcast_notification(
                "voice_result",
                {
                    "command": command_text,
                    "status": status,
                    "result": result_text[:500],
                    "language": language,
                },
            )

            from pilot.system.voice import speak

            spoken = result_text[:300] if len(result_text) < 300 else result_text[:297] + "..."
            await speak(spoken)

        except Exception as e:
            logger.error("Voice command execution failed: %s", e)

            await self._broadcast_notification(
                "voice_result",
                {
                    "command": command_text,
                    "status": "error",
                    "message": str(e),
                    "language": language,
                },
            )

            try:
                from pilot.system.voice import speak

                await speak("Sorry, something went wrong while executing your request.")
            except Exception:
                pass

    async def _voice_status_broadcast(self, status: str, data: dict) -> None:
        """Called by ContinuousVoiceListener for status updates.

        Args:
            status: The voice listener status.
            data: Additional status data.
        """
        await self._broadcast_notification("voice_status", {"status": status, **data})

    async def _handle_voice_listener_start(self, params: dict, ws: ServerConnection) -> dict:
        """Start the continuous JARVIS-mode voice listener.

        Args:
            params: JSON-RPC parameters with wake_words.
            ws: The WebSocket connection.

        Returns:
            A dict with status, message, and wake_words.
        """
        from pilot.system.voice import ContinuousVoiceListener

        wake_words = params.get("wake_words", ["hey heliox", "heliox", "hey pilot"])

        if self._voice_listener and self._voice_listener.is_running:
            return {"status": "already_running", "wake_words": self._voice_listener.wake_words}

        self._voice_listener = ContinuousVoiceListener(
            wake_words=wake_words,
            on_command=self._voice_command_dispatch,
            on_status=self._voice_status_broadcast,
        )
        result = await self._voice_listener.start()
        return {"status": "started", "message": result, "wake_words": wake_words}

    async def _handle_voice_listener_stop(self, params: dict, ws: ServerConnection) -> dict:
        """Stop the continuous voice listener.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status and message.
        """
        if not self._voice_listener or not self._voice_listener.is_running:
            return {"status": "not_running"}

        result = await self._voice_listener.stop()
        return {"status": "stopped", "message": result}

    async def _handle_voice_listener_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Get voice listener statistics.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with voice listener statistics.
        """
        if not self._voice_listener:
            return {"running": False, "message": "Voice listener not initialized"}
        return self._voice_listener.get_stats()

    # ── Autonomous Executor Handlers ──

    async def _handle_autonomous_submit(self, params: dict, ws: ServerConnection) -> dict:
        """Submit a task for autonomous background execution.

        Args:
            params: JSON-RPC parameters with goal and source.
            ws: The WebSocket connection.

        Returns:
            A dict with status and job information.
        """
        if not self._autonomous:
            return {"error": "Autonomous executor not initialized"}

        goal = params.get("goal", "")
        if not goal.strip():
            return {"error": "Empty goal"}

        source = params.get("source", "text")
        job = await self._autonomous.submit(goal, source=source)
        return {"status": "submitted", "job": job.to_dict()}

    async def _handle_autonomous_cancel(self, params: dict, ws: ServerConnection) -> dict:
        """Cancel a running autonomous job.

        Args:
            params: JSON-RPC parameters with job_id.
            ws: The WebSocket connection.

        Returns:
            A dict with cancelled status and job_id.
        """
        if not self._autonomous:
            return {"error": "Autonomous executor not initialized"}

        job_id = params.get("job_id", "")
        success = await self._autonomous.cancel(job_id)
        return {"cancelled": success, "job_id": job_id}

    async def _handle_autonomous_jobs(self, params: dict, ws: ServerConnection) -> dict:
        """List all autonomous jobs.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with list of jobs.
        """
        if not self._autonomous:
            return {"jobs": []}
        return {"jobs": self._autonomous.list_jobs()}

    async def _handle_autonomous_job(self, params: dict, ws: ServerConnection) -> dict:
        """Get a specific autonomous job by ID.

        Args:
            params: JSON-RPC parameters with job_id.
            ws: The WebSocket connection.

        Returns:
            A dict with job information or error.
        """
        if not self._autonomous:
            return {"error": "Autonomous executor not initialized"}

        job_id = params.get("job_id", "")
        job = self._autonomous.get_job(job_id)
        if not job:
            return {"error": f"Job not found: {job_id}"}
        return job.to_dict()

    # ── Proactive Suggestions Handlers ──

    async def _handle_proactive_start(self, params: dict, ws: ServerConnection) -> dict:
        """Start the proactive suggestion engine.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status and message.
        """
        if not self._proactive:
            return {"error": "Proactive engine not initialized"}
        result = await self._proactive.start()
        return {"status": "started", "message": result}

    async def _handle_proactive_stop(self, params: dict, ws: ServerConnection) -> dict:
        """Stop the proactive suggestion engine.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status and message.
        """
        if not self._proactive:
            return {"error": "Proactive engine not initialized"}
        result = await self._proactive.stop()
        return {"status": "stopped", "message": result}

    async def _handle_proactive_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Get proactive engine statistics.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with proactive engine statistics.
        """
        if not self._proactive:
            return {"running": False, "message": "Proactive engine not initialized"}
        return self._proactive.get_stats()

    async def _handle_proactive_accept(self, params: dict, ws: ServerConnection) -> dict:
        """Accept a proactive suggestion — execute the suggested action.

        Args:
            params: JSON-RPC parameters with suggestion_id.
            ws: The WebSocket connection.

        Returns:
            A dict with execution status and results.
        """
        if not self._proactive:
            return {"error": "Proactive engine not initialized"}

        suggestion_id = params.get("suggestion_id", "")
        action_command = await self._proactive.accept_suggestion(suggestion_id)
        if not action_command:
            return {"error": f"Suggestion not found: {suggestion_id}"}

        if self._autonomous:
            job = await self._autonomous.submit(action_command, source="proactive")
            return {"status": "executing", "action": action_command, "job": job.to_dict()}
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
                "results": [{"success": r.success, "output": r.output[:200]} for r in results],
            }

    async def _handle_proactive_dismiss(self, params: dict, ws: ServerConnection) -> dict:
        """Dismiss a proactive suggestion.

        Args:
            params: JSON-RPC parameters with suggestion_id.
            ws: The WebSocket connection.

        Returns:
            A dict with dismissed status and suggestion_id.
        """
        if not self._proactive:
            return {"error": "Proactive engine not initialized"}

        suggestion_id = params.get("suggestion_id", "")
        dismissed = await self._proactive.dismiss_suggestion(suggestion_id)
        return {"dismissed": dismissed, "suggestion_id": suggestion_id}


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
    parser = argparse.ArgumentParser(prog="pilot.server")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without executing them")
    parser.add_argument(
        "--export-logs",
        action="store_true",
        help="Package all logs, config.toml, and audit trails into a zip on the Desktop for bug reporting.",
    )
    args, _ = parser.parse_known_args()
    if args.export_logs:
        export_logs()
        return
    if args.dry_run:
        config.security.dry_run = True
        logger.info("Dry-run mode enabled via CLI flag")
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
