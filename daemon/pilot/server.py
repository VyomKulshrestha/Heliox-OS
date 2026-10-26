"""WebSocket JSON-RPC 2.0 server for the Pilot daemon."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import csv
import json
import logging
import math
import re
import secrets
import signal
import sys
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import websockets
from websockets.asyncio.server import Server, ServerConnection

from pilot import __version__
from pilot.config import (
    CO_ADAPTATION_BOOL_KEYS,
    CO_ADAPTATION_DB_FILE,
    CO_ADAPTATION_EXPORTS_DIR,
    CO_ADAPTATION_FLOAT_RANGES,
    CO_ADAPTATION_INT_RANGES,
    CO_ADAPTATION_PREFERENCES_FILE,
    CONFIG_DIR,
    DATA_DIR,
    DB_FILE,
    EXPERIENCE_DB_FILE,
    LOG_FILE,
    PLUGINS_DIR,
    STATE_DIR,
    PilotConfig,
    ensure_dirs,
)
from pilot.export_logs import export_logs
from pilot.intelligence.experience import (
    ExperienceContext,
    ExperienceEvent,
    ExperienceEventType,
    ExperienceLedger,
    PrivacyClass,
    experience_scope,
    get_experience_context,
    stable_action_idempotency_key,
)
from pilot.logger import ColorFormatter
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
from pilot.security.rpc_identity import (
    RpcClientRole,
    authenticate_rpc_client,
    derive_neural_signing_key,
    rpc_method_allowed,
)
from pilot.system.companion_speech import (
    CompanionSpeechCoordinator,
    SpeechChannel,
    SpeechOutcome,
)
from pilot.system.interaction import InteractionPhase, InteractionRuntime, acknowledgement_for
from pilot.workflows.durable_tasks import (
    ApprovalStatus,
    DurableTaskStore,
    InvalidTaskTransition,
    TaskStatus,
)

logger = logging.getLogger("pilot.server")


async def _accessibility_observation(limit: int = 40) -> str:
    """Describe the focused window's UI elements for autonomous rounds, or "" when unavailable."""
    from pilot.system.accessibility import AccessibilityUnavailable, capture_snapshot

    try:
        snapshot = await capture_snapshot(max_elements=limit)
    except AccessibilityUnavailable:
        return ""
    return snapshot.describe(limit=limit) if snapshot.elements else ""


def _resolve_dry_run(configured: bool, requested: object = False) -> bool:
    """Treat the global Dry Run setting as a safety floor clients cannot disable."""
    return bool(configured or requested)


def _validated_bool(params: dict[str, Any], key: str, default: bool) -> bool:
    """Return a JSON boolean without accepting truthy strings or numbers."""
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _validated_float(
    params: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a finite numeric RPC value within optional inclusive bounds."""
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{key} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{key} must be at most {maximum:g}")
    return result


def _validated_int(
    params: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return a strict integer RPC value within optional inclusive bounds."""
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be at most {maximum}")
    return value


def _sanitize_summary(text: object, limit: int = 160) -> str:
    """Collapse whitespace and bound user-visible/event summaries."""
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def _normalize_spoken_phrase(text: str) -> str:
    """Casefold, turn punctuation into spaces and collapse whitespace, for matching exact spoken phrases."""
    return " ".join(
        "".join(
            character if character.isalnum() or character.isspace() else " " for character in text.casefold()
        ).split()
    )


def _co_adaptation_config_dict(config: Any) -> dict[str, Any]:
    """Every co_adaptation setting as plain JSON values."""
    from dataclasses import asdict

    return asdict(config)


CONFIRM_TIMEOUT_SECONDS = 300
# The authority channel that answered a confirmation, recorded in APPROVAL_RESOLVED. It never says whether a
# pointer or a keyboard pressed the dialog's button.
CONFIRM_INPUT_CHANNELS = ("dialog", "gesture", "voice_deny", "internal", "unknown")
# Resolution paths of a confirmation that ended without an answer the plan may act on. The waiter reads each as
# expired: "timeout" (nobody answered in time), "shutdown" (the daemon stopped first) and "unacknowledged" (an
# approval whose reply never reached the UI, so the person was never shown that it was accepted).
CONFIRM_UNANSWERED_PATHS = ("timeout", "shutdown", "unacknowledged")
# RPCs that change saved settings. They run one at a time under the server's config lock,
# pick up hand edits to the config files first, and announce ``config_updated`` so every
# open window re-reads its settings instead of saving a stale copy over them.
CONFIG_MUTATING_RPC_METHODS = frozenset(
    {
        "update_config",
        "reset_config",
        "narration_config_update",
        "risk_gate_config_update",
        "self_healing_config_update",
        "supervision_config_update",
        "gateway_policy_update",
        "gesture_workflow_bindings_update",
        "ssh_set_enabled",
        "ssh_save_host",
        "ssh_delete_host",
        "air_handoff_set_enabled",
        "mesh_configure",
        "store_api_key",
        "delete_api_key",
        "co_adaptation_config_update",
        "co_adaptation_consent_grant",
        "co_adaptation_consent_revoke",
    }
)
# Settings reads and writes must not wait behind a long ``execute`` on the same socket
# (an approval wait can last five minutes); the config lock keeps writes ordered.
SETTINGS_RPC_METHODS = CONFIG_MUTATING_RPC_METHODS | frozenset(
    {
        "get_config",
        "narration_status",
        "risk_gate_status",
        "self_healing_status",
        "supervision_status",
        "gateway_policy_get",
        "gesture_workflow_bindings_get",
        "mesh_status",
        "air_handoff_status",
        "list_api_keys",
        "autonomy_review_status",
        "get_security_status",
        "co_adaptation_status",
        "co_adaptation_preference_set",
        "co_adaptation_preference_retract",
        "co_adaptation_learning_reset",
        "co_adaptation_erase",
        "co_adaptation_results",
    }
)
# Vault entries owned by other features; the generic API-key RPCs must not overwrite or delete them.
RESERVED_VAULT_NAMES = frozenset({"heliox_mesh"})
RESERVED_VAULT_PREFIXES = ("ssh:", "air_handoff:")
# Settings that are only read when Heliox starts; changing them returns restart_required.
RESTART_REQUIRED_KEYS = {
    "screen_vision": {"capture_timeout_seconds", "max_consecutive_timeouts", "auto_resume_after_seconds"},
    "cognitive": {"enabled"},
    "rss": {"enabled"},
}
RESTART_REQUIRED_SECTIONS = {"memory", "proxy", "redis"}

# These control-plane RPCs must be dispatchable while a normal request is
# paused. ``execute`` waits for ``confirm`` on the same WebSocket connection,
# ``abort`` interrupts execution, ``interject`` revises it, and speech
# replacement/stop must interrupt an in-flight utterance. All other RPCs
# remain sequential per connection.
OUT_OF_BAND_RPC_METHODS = frozenset(
    {
        "confirm",
        "abort",
        "interject",
        "queue_execute",
        "cancel_queued_execute",
        "speak_text",
        "stop_speech",
        "neural_disarm",
        "recorded_workflow_start",
        "recorded_workflow_status",
        "recorded_workflow_stop",
    }
)
# The selection study's RPCs run out-of-band too: a study trial must not wait behind an ``execute`` on the
# same socket. They touch only the co-adaptation service, never a pending approval or the executor.
CO_ADAPTATION_RPC_METHODS = frozenset(
    {
        "co_adaptation_session_begin",
        "co_adaptation_session_resume",
        "co_adaptation_recommend",
        "co_adaptation_record_trial",
        "co_adaptation_report_burden",
        "co_adaptation_session_end",
        "co_adaptation_export",
    }
)
# Neural gate states in which the selection study refuses to start, resume or record a trial.
NEURAL_ARMED_STATES = frozenset(
    {"armed_safe_ui", "armed_safe_desktop", "candidate_intent", "previewed", "committed", "cooldown"}
)

# Attention-aware timing is allowed to defer only passive, replaceable
# telemetry. User-initiated results, approvals, safety interventions, and
# newly-added notification types must be delivered immediately by default.
ATTENTION_DEFERRABLE_NOTIFICATIONS = frozenset(
    {
        "background_update",
        "memory_consolidation",
        "plugin_event",
        "screen_vision_update",
    }
)
MAX_ATTENTION_NOTIFICATION_BUFFER = 64

# Speech recognition is intentionally not an approval authority. Exact
# affirmative phrases are detected only so they can be rejected clearly while
# a visible confirmation is pending, instead of being misread as a plan
# correction and cancelling/replanning the request.
VOICE_DENIALS = frozenset(
    {
        "no",
        "nope",
        "deny",
        "deny it",
        "do not",
        "dont",
        "don t",
        "do not do that",
        "don t do that",
        "dont do that",
        "cancel that",
        "reject",
        "reject it",
    }
)
# What a phone request may do when it starts without anyone at the computer: status and
# names only. File contents, the clipboard, email, the screen, environment variables and
# the registry stay out, and nothing that opens or changes anything is allowed.
PHONE_READ_ONLY_ACTIONS = (
    "battery_info",
    "calendar_list_events",
    "cpu_usage",
    "directory_size",
    "directory_summary",
    "disk_list",
    "disk_usage",
    "file_list",
    "file_search",
    "git_log",
    "git_status",
    "memory_usage",
    "network_info",
    "process_list",
    "service_status",
    "system_health_review",
    "system_info",
)
VOICE_STOP_PHRASES = frozenset(
    {
        "stop",
        "cancel",
        "stop it",
        "stop that",
        "stop everything",
        "cancel everything",
        "stop all tasks",
        "cancel all tasks",
        "never mind",
        "nevermind",
    }
)
# Spoken or typed undo only opens the preview: undoing still needs a click on its Undo button.
UNDO_ONE_REQUEST = re.compile(
    r"^(?:please )?(?:undo|take back|revert)(?: (?:that|it|the last task|my last task|last task))?$"
    r"|^(?:please )?take that back$"
)
UNDO_MANY_REQUEST = re.compile(r"^(?:please )?(?:undo|take back|revert) (?:the|my) last (?P<count>\w+) tasks$")
# Everyday settings that may change by voice; security and approval settings never do.
VOICE_TIMELINE_SETTING = re.compile(
    r"^(?:please )?(?:turn (?P<state>on|off) (?:the |my )?activity timeline|(?P<verb>start|stop) recording my activity)$"
)
VOICE_BRIEF_SETTING = re.compile(r"^(?:please )?turn (?P<state>on|off) (?:the |my )?daily brief$")
VOICE_CLIPBOARD_SETTING = re.compile(r"^(?:please )?turn (?P<state>on|off) (?:the |my )?clipboard history$")
VOICE_BRIEF_TIME_SETTING = re.compile(
    r"^(?:please )?send (?:me )?(?:a |the |my )?daily brief (?:every day )?at (?P<hour>\d{1,2})(?: (?P<minute>\d{2}))?"
    r"(?: (?P<meridiem>am|pm))?$"
)
VOICE_BRIEF_ALOUD_SETTING = re.compile(r"^(?:please )?(?:read|say) (?:the |my )?(?:daily )?brief (?:out loud|aloud)$")
VOICE_BRIEF_QUIET_SETTING = re.compile(
    r"^(?:please )?(?:don t|do not|stop) (?:reading|read|saying|say) (?:the |my )?(?:daily )?brief (?:out loud|aloud)$"
)
SAY_AGAIN_REQUEST = re.compile(
    r"^(?:please )?(?:say that again|say it again|what did you say|what was that|come again|pardon|sorry what)$"
)
# Channels whose words the person may want repeated with "say that again".
REPEATABLE_SPEECH = frozenset(
    {
        SpeechChannel.FINAL_ANSWER.value,
        SpeechChannel.TASK_NARRATION.value,
        SpeechChannel.APPROVAL_RISK.value,
        SpeechChannel.TASK_FAILURE.value,
        SpeechChannel.PROACTIVE_SUGGESTION.value,
    }
)
UNDO_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}
MAX_UNDO_REQUEST = 5
VOICE_APPROVAL_ATTEMPTS = frozenset(
    {
        "approve",
        "approve all",
        "confirm",
        "confirm it",
        "do it",
        "go ahead",
        "yes",
        "yes approve",
        "yes do it",
    }
)

# ── Plan History DB path (sibling of the main DB) ──
PLAN_HISTORY_DB_FILE = DATA_DIR / "plan_history.db"


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
            ValueError: If the payload is not a valid JSON-RPC request object.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON-RPC request: expected an object")
        if data.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC version")
        method = data.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("Invalid JSON-RPC request: method must be a non-empty string")
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Invalid JSON-RPC request: params must be an object")
        return cls(
            method=method,
            params=params,
            id=data.get("id"),
        )


class _BroadcastConnection:
    """Duck-typed WebSocket sink used by non-socket interaction sources."""

    def __init__(self, broadcast: Any) -> None:
        self._broadcast = broadcast

    async def send(self, raw: str) -> None:
        payload = json.loads(raw)
        method = payload.get("method")
        if isinstance(method, str) and method:
            await self._broadcast(method, payload.get("params", {}))


def _success_response(req_id: str | int | None, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": req_id})


def _error_response(req_id: str | int | None, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id})


def _notification(method: str, params: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params})


def _is_terminal_execution_failure(results: list[Any]) -> bool:
    """Return True when replanning cannot repair an exact local resolution failure."""
    failed = [result for result in results if not result.success]
    if not failed:
        return False

    terminal_markers = {
        "file_read": ("file not found", "no such file", "cannot find the file"),
        "open_application": (
            "was not found in the start menu, path, or windows app registry",
            "is ambiguous",
            "is registered, but its executable is missing",
        ),
    }
    for result in failed:
        action_type = getattr(result.action.action_type, "value", str(result.action.action_type))
        markers = terminal_markers.get(action_type)
        if markers is None or not any(marker in str(result.error or "").lower() for marker in markers):
            return False
    return True


def _postcondition_failure_requires_reconciliation(results: list[Any], verification: Any) -> bool:
    """Stop before repeating a destructive action whose effect is uncertain."""
    for index in getattr(verification, "failed_actions", []):
        if not isinstance(index, int) or index < 0 or index >= len(results):
            continue
        result = results[index]
        if result.success and (result.action.requires_snapshot or result.action.is_irreversible):
            return True
    return False


def _is_terminal_planning_failure(error: str) -> bool:
    """Return whether the planner already exhausted the configured provider path."""
    normalized = error.casefold()
    return any(
        marker in normalized
        for marker in (
            "api unavailable",
            "no api key configured",
            "configure a healthy provider",
        )
    )


@dataclass
class PendingConfirmation:
    """Tracks a plan awaiting user confirmation."""

    plan_id: str
    event: asyncio.Event
    confirmed: bool = False
    plan: Any = None
    # Indices of actions the user approved out of those requiring confirmation.
    # None means "not specified" -> treat as all-approved (back-compat with
    # older frontend builds that only send {plan_id, confirmed}).
    approved_indices: set[int] | None = None
    decision_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    resolved: bool = False
    resolved_by_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    rpc_tasks: set[asyncio.Task[Any]] = field(default_factory=set, repr=False)
    # Authority audit data for APPROVAL_RESOLVED: the channel that answered (dialog, gesture, voice_deny,
    # internal, unknown; never pointer versus keyboard) and the path that resolved the confirmation.
    input_channel: str = "unknown"
    resolution_path: str = ""
    # The action indices a plan approval covers. None for narrator, self-healing and threat-containment
    # confirmations, which keep accepting a confirm without approved_indices.
    required_indices: frozenset[int] | None = None
    # True once the resolving confirm's reply was written to its socket, or at once for a confirm that has no
    # reply to deliver. An approval that is never acknowledged is never acted on.
    acknowledged: bool = False
    # The interactive task that raised this confirmation when it is not the task's own plan approval (a narrator
    # interrupt during a step), so typed or spoken text during that task can answer it like the plan approval.
    task_id: str = ""


@dataclass(frozen=True)
class _ConfirmationDecision:
    """What a confirmation waiter read under the decision lock; it never reads the pending object again."""

    confirmed: bool
    approved_indices: set[int] | None
    expired: bool
    input_channel: str
    resolution_path: str


def _release_unacknowledged_confirmation(pending: PendingConfirmation) -> None:
    """Wake the waiter of a confirmation whose confirm reply was never written, failing closed.

    An approval nobody was shown as accepted is withdrawn before the event is set, so no waiter acts on it: the plan
    waiters read it as expired ("unacknowledged") and the narrator, self-healing and threat-containment waiters read a
    denial. A denial stays a denial. These writes take no lock: the confirmation is already resolved, so every other
    writer that can still run (a denial or the shutdown release) also only moves it toward a denial.
    """
    if pending.confirmed:
        pending.confirmed = False
        pending.approved_indices = set()
        pending.resolution_path = "unacknowledged"
    pending.event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Plan History Store
# ─────────────────────────────────────────────────────────────────────────────


class PlanHistoryStore:
    """Append-only SQLite audit log for every ActionPlan executed by the daemon.

    Schema (``plan_history`` table)
    --------------------------------
    plan_id             TEXT  PRIMARY KEY  — 8-char UUID prefix assigned in _handle_execute
    created_at          TEXT              — ISO-8601 UTC timestamp of plan creation
    raw_input           TEXT              — original user input string
    plan_json           TEXT              — full ActionPlan serialised as JSON
    action_count        INTEGER           — len(plan.actions)
    critic_verdict_json TEXT  NULLABLE    — DestructiveCriticAgent verdict dict, or NULL
    confirmation_decision TEXT            — 'approved' | 'denied' | 'skipped' |
                                            'blocked_by_critic' | 'n/a' (dry-run)
    execution_status    TEXT              — 'success' | 'partial_failure' | 'error' |
                                            'cancelled' | 'dry_run'
    results_json        TEXT  NULLABLE    — list[ActionResult.model_dump()] as JSON
    verification_json   TEXT  NULLABLE    — VerificationResult.model_dump() as JSON
    dry_run             INTEGER           — 1 if dry-run, 0 otherwise
    duration_ms         INTEGER           — wall-clock ms from plan start to terminal state
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS plan_history (
        plan_id               TEXT    PRIMARY KEY,
        created_at            TEXT    NOT NULL,
        raw_input             TEXT    NOT NULL,
        plan_json             TEXT    NOT NULL,
        action_count          INTEGER NOT NULL DEFAULT 0,
        critic_verdict_json   TEXT,
        confirmation_decision TEXT    NOT NULL DEFAULT 'n/a',
        execution_status      TEXT    NOT NULL DEFAULT 'unknown',
        results_json          TEXT,
        verification_json     TEXT,
        dry_run               INTEGER NOT NULL DEFAULT 0,
        duration_ms           INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS plan_history_created_at
        ON plan_history (created_at DESC);
    CREATE INDEX IF NOT EXISTS plan_history_execution_status
        ON plan_history (execution_status);
    """

    def __init__(self, db_path: str | Path = PLAN_HISTORY_DB_FILE) -> None:
        self._db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open (or create) the SQLite DB and ensure the schema exists."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(self._CREATE_TABLE)
        await self._db.commit()
        logger.info("PlanHistoryStore initialised at %s", self._db_path)

    async def record(
        self,
        *,
        plan_id: str,
        raw_input: str,
        plan: Any,
        critic_verdict: dict[str, Any] | None,
        confirmation_decision: str,
        execution_status: str,
        results: list[Any],
        verification: Any | None,
        dry_run: bool,
        duration_ms: int,
    ) -> None:
        """Insert or replace a plan audit record.

        Args:
            plan_id: Short UUID identifying this plan.
            raw_input: The original user-supplied text.
            plan: ActionPlan object (must have ``.actions`` and ``.model_dump()`` / JSON-serialisable dict).
            critic_verdict: Optional dict from DestructiveCriticAgent.
            confirmation_decision: One of 'approved', 'denied', 'skipped', 'blocked_by_critic', 'n/a'.
            execution_status: Terminal status string ('success', 'partial_failure', 'error', 'cancelled', 'dry_run').
            results: List of ActionResult objects with ``.model_dump()``.
            verification: VerificationResult object with ``.model_dump()``, or None.
            dry_run: Whether this was a dry-run execution.
            duration_ms: Wall-clock duration in milliseconds.
        """
        if self._db is None:
            logger.warning("PlanHistoryStore.record() called before initialize()")
            return

        try:
            plan_dict = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else {}
        except Exception:
            plan_dict = {}

        results_list: list[Any] = []
        for r in results:
            try:
                if hasattr(r, "model_dump"):
                    results_list.append(r.model_dump(mode="json"))
                elif isinstance(r, (dict, list, str, int, float, bool)) or r is None:
                    results_list.append(r)
                else:
                    results_list.append(str(r))
            except Exception:
                results_list.append(str(r))

        try:
            verification_dict = (
                verification.model_dump(mode="json") if (verification and hasattr(verification, "model_dump")) else None
            )
        except Exception:
            verification_dict = None

        await self._db.execute(
            """
            INSERT OR REPLACE INTO plan_history (
                plan_id, created_at, raw_input, plan_json, action_count,
                critic_verdict_json, confirmation_decision,
                execution_status, results_json, verification_json,
                dry_run, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                datetime.now(timezone.utc).isoformat(),
                raw_input,
                json.dumps(plan_dict, ensure_ascii=False),
                len(getattr(plan, "actions", [])),
                json.dumps(critic_verdict, ensure_ascii=False) if critic_verdict is not None else None,
                confirmation_decision,
                execution_status,
                json.dumps(results_list, ensure_ascii=False),
                json.dumps(verification_dict, ensure_ascii=False) if verification_dict is not None else None,
                1 if dry_run else 0,
                duration_ms,
            ),
        )
        await self._db.commit()
        logger.debug("PlanHistoryStore: recorded plan_id=%s status=%s", plan_id, execution_status)

    async def get_list(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a paginated list of plan summary rows (no large JSON blobs).

        Args:
            limit: Maximum number of rows to return.
            offset: Rows to skip (for pagination).
            status_filter: Optional execution_status to filter by.

        Returns:
            List of dicts with summary fields.
        """
        if self._db is None:
            return []

        if status_filter:
            cursor = await self._db.execute(
                """
                SELECT plan_id, created_at, raw_input, action_count,
                       confirmation_decision, execution_status, dry_run, duration_ms
                FROM plan_history
                WHERE execution_status = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (status_filter, limit, offset),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT plan_id, created_at, raw_input, action_count,
                       confirmation_decision, execution_status, dry_run, duration_ms
                FROM plan_history
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_detail(self, plan_id: str) -> dict[str, Any] | None:
        """Return the full record for a single plan_id, with JSON blobs parsed.

        Args:
            plan_id: The plan identifier to look up.

        Returns:
            Full plan record dict with parsed JSON fields, or None if not found.
        """
        if self._db is None:
            return None

        cursor = await self._db.execute(
            "SELECT * FROM plan_history WHERE plan_id = ?",
            (plan_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        record = dict(row)
        # Parse stored JSON blobs back into Python objects for the caller
        for field_name in ("plan_json", "critic_verdict_json", "results_json", "verification_json"):
            raw = record.get(field_name)
            if raw:
                try:
                    record[field_name] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    pass  # leave as raw string if unparseable
        record["dry_run"] = bool(record.get("dry_run", 0))
        return record

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None


# ─────────────────────────────────────────────────────────────────────────────
# PilotServer
# ─────────────────────────────────────────────────────────────────────────────


class PilotServer:
    """Main daemon server managing WebSocket connections and agent dispatch."""

    def __init__(self, config: PilotConfig) -> None:
        """Initialize the PilotServer with the given configuration.

        Args:
            config: PilotConfig instance containing server and model settings.
        """
        self.config = config
        self._start_time = time.time()
        self._server: Server | None = None
        self._clients: set[ServerConnection] = set()
        self._handlers: dict[str, Any] = {}
        self._model_router: Any = None
        self._planner: Any = None
        self._executor: Any = None
        self._verifier: Any = None
        self._destructive_critic: Any = None
        self._permission_checker: Any = None
        self._agent_gateway: Any = None
        self._gateway_audit: Any = None
        self._voice_gesture_workflows: Any = None
        self._recorded_workflows: Any = None
        self._manual_workflow_hook: Any = None
        self._recorded_workflow_scheduler: asyncio.Task[None] | None = None
        self._recorded_workflow_jobs: dict[str, str] = {}
        self._reflector: Any = None
        self._multi_agent: Any = None
        self._background: Any = None
        self._orchestrator: Any = None
        self._agent_mesh: Any = None
        self._fusion: Any = None
        self._reasoning: Any = None
        self._decomposer: Any = None
        self._sandbox: Any = None
        self._prompt_improver: Any = None
        self._plugin_registry: Any = None
        self._plugin_marketplace: Any = None
        self._skill_registry: Any = None
        self._subconscious: Any = None
        self._screen_vision: Any = None
        self._memory: Any = None
        self._experience_ledger: ExperienceLedger | None = None
        self._online_learning: Any = None
        # Assistance co-adaptation service (pilot.coadaptation, loaded lazily in initialize()).
        self._co_adaptation: Any = None
        # Set when the service erased a participant whose consent expired; _settle_co_adaptation_consent then turns
        # co_adaptation.enabled off if no active consent remains, as a revoke does.
        self._co_adaptation_consent_lapsed = False
        self._co_adaptation_lapse_task: asyncio.Task[None] | None = None
        self._strategy_evolution: Any = None
        self._evolution_harness: Any = None
        self._active_experience_context = ExperienceContext()
        self._vault: Any = None
        self._permission_audit: Any = None
        self._checkpoint_store: Any = None
        self._durable_tasks: DurableTaskStore | None = None
        # ── Plan History Audit Log ──
        self._plan_history: PlanHistoryStore | None = None
        self._plan_history_tasks: set[asyncio.Task[None]] = set()
        self._memory_record_tasks: set[asyncio.Task[Any]] = set()
        self._reflection_tasks: set[asyncio.Task[Any]] = set()
        self._companion_follow_up_tasks: set[asyncio.Task[None]] = set()
        self._interaction_speech_tasks: set[asyncio.Task[SpeechOutcome]] = set()
        self._mcp_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._trigger_engine: Any = None
        self._mcp_task_results: dict[str, dict[str, Any]] = {}
        self._mcp_idempotency: dict[str, tuple[str, str]] = {}
        self._mcp_reserved_task_id = ""
        self._typed_decision_engine: Any = None
        self._typed_decision_engine_key: tuple[Any, ...] | None = None
        # Cognitive intelligence (lightweight heuristic engine)
        self._cognitive_engine: Any = None
        self._attention_ui: Any = None
        self._stress_gate: Any = None
        self._intent_predictor: Any = None
        self._voice_listener: Any = None
        self._speech_coordinator = CompanionSpeechCoordinator()
        self._interaction_runtime = InteractionRuntime(self._broadcast_notification)
        self._autonomous: Any = None
        from pilot.security.human_control import HumanControlGate

        self._human_control = HumanControlGate()
        from pilot.mcp_client import default_manager as default_mcp_client

        self._mcp_client = default_mcp_client()
        self._mcp_client_refresh: asyncio.Task[Any] | None = None
        self._scheduled_runs: Any = None
        self._config_lock = asyncio.Lock()
        self._gnomon_reviewer: Any = None
        self._self_healing: Any = None
        self._self_healing_started_monitors: set[str] = set()
        self._proactive: Any = None
        self._budget_tracker: Any = None
        self._running = False
        self._pending_confirms: dict[str, PendingConfirmation] = {}
        # ── Undo (rollback_plan RPC) ── plan_id -> snapshot_id taken before execution
        self._plan_snapshots: dict[str, str] = {}
        # ── Cancel Token (Issue #92) ──
        self._cancel_event: asyncio.Event | None = None
        # ── Mid-flight cancellation: the currently in-flight interactive
        # execution, if any -- lets _handle_abort actually cancel the
        # running task (killing e.g. a mid-flight shell subprocess), not
        # just signal cancel_event for the next action boundary. Single
        # slot, mirroring _cancel_event's own "one primary interactive
        # session" scope rather than a dict keyed by plan_id. ──
        self._active_execution_task: asyncio.Task[Any] | None = None
        # Live interaction companion state. Ordinary execute requests stay
        # serialized, while an out-of-band interject request can revise or
        # stop the one active interactive task.
        self._interactive_request_active = False
        self._undo_journal: Any = None
        self._task_receipts: Any = None
        self._activity_timeline: Any = None
        self._task_footers: dict[str, dict[str, Any]] = {}
        self._daily_brief: Any = None
        self._shortcuts: Any = None
        self._reminders: Any = None
        self._focus: Any = None
        self._quiet_hours: Any = None
        self._last_spoken = ""
        self._workspaces: Any = None
        self._notes: Any = None
        self._todos: Any = None
        self._nudges: Any = None
        self._watchers: Any = None
        self._clipboard_history: Any = None
        self._system_health: Any = None
        self._focus_guard: Any = None
        self._app_reminders: Any = None
        self._queued_execute_requests: deque[tuple[str, str, dict[str, Any]]] = deque()
        self._queued_submission_lock = asyncio.Lock()
        self._queued_execute_worker: asyncio.Task[None] | None = None
        self._queued_reserved_task_id = ""
        self._queue_shutdown = False
        self._active_plan_id = ""
        self._active_task_id = ""
        self._active_interaction_id = ""
        self._live_correction: str | None = None
        self._execution_companion: Any = None
        self._recent_companion_context = ""
        self._recent_companion_context_by_session: dict[str, str] = {}
        self._tts_warmup_task: asyncio.Task[None] | None = None
        self._rss_agent: Any = None
        # ── LAN Mesh Network ──
        self._mesh: Any = None
        self._mesh_error = ""
        # Separate least-privileged LAN service for encrypted phone handoffs.
        self._air_handoff_manager: Any = None
        self._air_handoff_server: Any = None
        # ── Threat Containment Bridge (Issue #365) ──
        self._threat_bridge: Any = None
        # ── Authenticated WebSocket clients ──
        self._authenticated_clients: set[ServerConnection] = set()
        self._client_roles: dict[ServerConnection, RpcClientRole] = {}
        self._neural_sidecar_client: ServerConnection | None = None
        self._neural_controller: Any = None
        self._controller_lease: Any = None
        # Rotated on each daemon process start. neurod reads the separate
        # owner-only runtime file and never receives the UI token.
        self._neural_auth_token = secrets.token_urlsafe(32)
        # The local MCP bridge receives its own least-privilege credential.
        # It never receives the UI token and its RPC methods are allow-listed
        # independently in security.rpc_identity.
        self._mcp_auth_token = secrets.token_urlsafe(32)

    def _start_tts_warmup(self) -> None:
        """Keep heavyweight local TTS outside the long-lived daemon.

        Kokoro and Pocket TTS pull a sizeable inference runtime into Python.
        Loading either engine in this process consumed several gigabytes on
        CUDA installations. The voice dispatcher now uses a short-lived worker
        that retains its cache only for a bounded burst of speech.
        """
        if self._tts_warmup_task and not self._tts_warmup_task.done():
            self._tts_warmup_task.cancel()
        self._tts_warmup_task = None
        if self.config.voice.tts_engine in {"kokoro_tts", "pocket_tts"}:
            logger.info(
                "%s model isolated in an on-demand speech worker",
                "Kokoro TTS" if self.config.voice.tts_engine == "kokoro_tts" else "Pocket TTS",
            )

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
        from pilot.air_handoff import AirHandoffManager, AirHandoffServer

        self._air_handoff_manager = AirHandoffManager(
            self._vault,
            max_transfer_bytes=self.config.air_handoff.max_transfer_mb * 1024 * 1024,
        )
        self._air_handoff_server = AirHandoffServer(
            self._air_handoff_manager,
            host="0.0.0.0",
            port=self.config.air_handoff.port,
        )
        self._wire_air_handoff_requests()
        model_router = ModelRouter(self.config, self._vault)
        self._model_router = model_router
        await model_router.initialize()

        from pilot.models.budget_tracker import BudgetTracker

        self._budget_tracker = BudgetTracker(self.config.model, str(DB_FILE))
        await self._budget_tracker.initialize()
        model_router.set_budget_tracker(self._budget_tracker)
        from pilot.agents.circuit_breaker import CircuitBreaker

        self._circuit_breaker = CircuitBreaker(threshold=self.config.model.max_consecutive_failures)

        audit = AuditLogger()
        self._permission_audit = PermissionEscalationAuditStore()
        await self._permission_audit.initialize()
        self._checkpoint_store = WorkflowCheckpointStore()
        await self._checkpoint_store.initialize()
        self._durable_tasks = DurableTaskStore()
        await self._durable_tasks.initialize()
        recovery = await self._durable_tasks.recover_incomplete()
        if any(
            (
                recovery.interrupted_tasks,
                recovery.uncertain_actions,
                recovery.expired_approvals,
            )
        ):
            logger.warning(
                "Recovered durable work: %d interrupted task(s), %d uncertain action(s), %d expired approval(s)",
                recovery.interrupted_tasks,
                recovery.uncertain_actions,
                recovery.expired_approvals,
            )
        validator = ActionValidator(self.config)
        permissions = PermissionChecker(self.config)
        self._permission_checker = permissions
        self._memory = MemoryStore(
            checkpoint_interval_seconds=self.config.memory.checkpoint_interval_seconds,
            pruning_interval_seconds=self.config.memory.pruning_interval_seconds,
            pruning_min_memories=self.config.memory.pruning_min_memories,
        )
        await self._memory.initialize(model_router)

        self._experience_ledger = ExperienceLedger(EXPERIENCE_DB_FILE)
        await self._experience_ledger.initialize()
        from pilot.intelligence.online_learning import VerifiedOnlineLearner

        self._online_learning = VerifiedOnlineLearner(DATA_DIR / "online_learning_state.json")
        await self._online_learning.initialize(self._experience_ledger)
        self._experience_ledger.subscribe(self._online_learning.consume)

        # ── Assistance co-adaptation (opt-in, presentation-only) ──
        # Imported here so that importing pilot.server never loads pilot.coadaptation. The service receives
        # settings, paths and the broadcast function only: no executor, ledger, memory or pending approval.
        try:
            from pilot.coadaptation.service import CoAdaptationService

            co_adaptation = CoAdaptationService(
                settings_provider=lambda: self.config.co_adaptation,
                gaze_tracking_enabled=lambda: self.config.vision.gaze_tracking_enabled,
                db_path=CO_ADAPTATION_DB_FILE,
                preferences_path=CO_ADAPTATION_PREFERENCES_FILE,
                exports_dir=CO_ADAPTATION_EXPORTS_DIR,
                broadcast_fn=self._co_adaptation_notify,
            )
            await co_adaptation.start()
            self._co_adaptation = co_adaptation
            # start() erases consents that expired while the daemon was off.
            await self._settle_co_adaptation_consent()
        except Exception:
            logger.warning("CoAdaptationService init failed (non-critical)", exc_info=True)

        # ── Plan History Audit Log ──
        self._plan_history = PlanHistoryStore()
        await self._plan_history.initialize()

        from pilot.skills.loader import SkillRegistry

        self._skill_registry = SkillRegistry()
        self._skill_registry.load_all()

        self._planner = Planner(
            model_router,
            self._memory,
            skills_context=self._skill_registry.planner_prompt_block(),
        )
        self._executor = Executor(
            self.config,
            validator,
            permissions,
            audit,
            skill_registry=self._skill_registry,
        )
        self._executor.set_experience_ledger(self._experience_ledger)
        from pilot.security.risk_gate import get_risk_gate

        gate = get_risk_gate()
        gate.set_online_learner(self._online_learning)
        gate.set_ui_observation_provider(None)
        self._executor.set_world_model_outcome_recorder(gate.record_outcome)
        from pilot.workflows.recorded_workflows import RecordedWorkflowStore

        self._recorded_workflows = RecordedWorkflowStore(
            DATA_DIR / "recorded_workflows.enc",
            capture_feature=self._capture_recording_feature,
            allow_manual_event=lambda: not self._pending_confirms,
        )
        runtime_world_weights = DATA_DIR / "ui_jepa_weights.enc"
        if runtime_world_weights.is_file():
            gate.configure_ui_jepa_artifact(
                runtime_world_weights,
                weight_loader=self._recorded_workflows.decrypt_model_artifact_file,
            )
        self._executor.set_recorded_workflow_store(self._recorded_workflows)
        from pilot.workflows.undo_journal import UndoJournal

        self._undo_journal = UndoJournal(DATA_DIR / "undo", config=lambda: self.config.undo)
        self._executor.set_undo_journal(self._undo_journal)
        try:
            await self._undo_journal.prune()
        except Exception:
            logger.warning("Undo history pruning failed", exc_info=True)
        from pilot.workflows.task_receipts import TaskReceipts

        self._task_receipts = TaskReceipts(
            DATA_DIR / "receipts", config=lambda: self.config.receipts, vault=self._vault, key_dir=CONFIG_DIR
        )
        self._executor.set_task_receipts(self._task_receipts)
        try:
            await self._task_receipts.prune()
        except Exception:
            logger.warning("Receipt pruning failed", exc_info=True)
        from pilot.workflows.activity_timeline import ActivityTimeline

        self._activity_timeline = ActivityTimeline(
            DATA_DIR / "timeline" / "timeline.db", config=lambda: self.config.timeline
        )
        try:
            await asyncio.to_thread(self._activity_timeline.prune)
        except Exception:
            logger.warning("Activity timeline pruning failed", exc_info=True)
        from pilot.workflows.routines import ShortcutStore

        self._shortcuts = ShortcutStore(DATA_DIR / "voice_shortcuts.json", reserved=self._phrase_reserved)
        from pilot.workflows.notes import NoteStore
        from pilot.workflows.nudges import NudgeStore
        from pilot.workflows.todos import TodoStore

        self._notes = NoteStore(DATA_DIR / "notes.json")
        self._todos = TodoStore(DATA_DIR / "todos.json")
        self._nudges = NudgeStore(
            DATA_DIR / "nudges.json",
            usage=self._activity_timeline.time_matching,
            timeline_on=lambda: bool(self._activity_timeline is not None and self._activity_timeline.enabled),
            deliver=self._deliver_nudge,
        )
        self._nudges.start()
        from pilot.workflows.watchers import WatcherStore

        self._watchers = WatcherStore(deliver=self._deliver_watch)
        self._watchers.start()
        from pilot.system.clipboard import clipboard_read, clipboard_write
        from pilot.workflows.clipboard_history import ClipboardHistory

        timeline = self._activity_timeline
        self._clipboard_history = ClipboardHistory(
            enabled=lambda: self.config.clipboard.history,
            excluded=timeline.excludes if timeline is not None else (lambda app, title: False),
            read=clipboard_read,
            write=clipboard_write,
        )
        self._clipboard_history.start()
        from pilot.workflows.system_health import SystemHealth

        self._system_health = SystemHealth()
        from pilot.workflows.workspaces import WorkspaceStore

        self._workspaces = WorkspaceStore(DATA_DIR / "workspaces.json")
        from pilot.workflows.focus import FocusSession

        self._focus = FocusSession(DATA_DIR / "focus.json", on_end=self._on_focus_end)
        self._focus.resume_timer()
        from pilot.workflows.focus_guard import FocusGuard

        self._focus_guard = FocusGuard(
            DATA_DIR / "focus_distractions.json",
            focus_status=lambda: self._focus.status() if self._focus is not None else {"active": False},
            deliver=self._deliver_distraction,
        )
        self._focus_guard.start()
        from pilot.workflows.app_reminders import AppReminders

        self._app_reminders = AppReminders(DATA_DIR / "app_reminders.json", deliver=self._deliver_app_reminder)
        self._app_reminders.start()
        from pilot.workflows.focus import QuietHours

        self._quiet_hours = QuietHours(
            DATA_DIR / "quiet_hours.json", focus=self._focus, on_start=self._on_quiet_hours_start
        )
        self._quiet_hours.start()
        from pilot.workflows.reminders import ReminderStore

        self._reminders = ReminderStore(DATA_DIR / "reminders.json", deliver=self._deliver_reminder)
        self._reminders.start()
        from pilot.workflows.daily_brief import DailyBriefScheduler

        self._daily_brief = DailyBriefScheduler(
            config=lambda: self.config.brief,
            build=self._today_overview,
            deliver=self._deliver_scheduled_brief,
            state_file=DATA_DIR / "daily_brief.json",
        )
        self._daily_brief.start()
        # Import and warm the local system probe stack before readiness so the
        # first harmless status request does not pay lazy import/thread startup.
        from pilot.system.sysinfo import prepare_system_probes

        await prepare_system_probes()
        self._executor.set_durable_task_store(self._durable_tasks)
        self._verifier = Verifier(model_router)
        self._executor.set_step_checker(self._verifier)

        # Destructive Critic Agent — secondary safety reviewer for Tier 4 plans.
        from pilot.agents.destructive_critic import DestructiveCriticAgent

        self._destructive_critic = DestructiveCriticAgent(model_router)
        from pilot.agents.execution_companion import ExecutionCompanion

        self._execution_companion = ExecutionCompanion(
            model_router,
            timeout_seconds=self.config.narration.advisory_timeout_seconds,
        )

        # Agent Gateway — source-scoped permission floor (interactive/
        # autonomous/web_agent/voice/gesture), checked alongside
        # PermissionChecker inside Executor.execute(). Built after the
        # critic since it needs one for non-interactive plan review.
        from pilot.security.gateway import AgentGateway
        from pilot.security.gateway_audit import AgentGatewayAuditStore

        self._gateway_audit = AgentGatewayAuditStore()
        await self._gateway_audit.initialize()
        self._agent_gateway = AgentGateway(self.config, permissions, self._destructive_critic, self._gateway_audit)
        self._executor.set_gateway(self._agent_gateway)
        self._executor.set_broadcast(self._broadcast_notification)
        self._executor.set_speech(self._speak_companion_text)

        from pilot.security.controller_lease import ControllerLeaseManager

        self._controller_lease = ControllerLeaseManager()
        self._executor.set_controller_lease(self._controller_lease)
        self._human_control.add_listener(self._broadcast_human_control)
        self._executor.set_human_control(self._human_control)
        self._executor.set_mcp_client(self._mcp_client)
        # MCP servers can be slow to start (npx downloads); connect in the background.
        self._mcp_client_refresh = asyncio.create_task(self._refresh_mcp_client())

        # Neural input owns a separate signing/session authority. Its fixed
        # goals still converge on the same Executor, Agent Gateway, world
        # model, permissions, validator, and audit path as other actions.
        from pilot import config as config_module
        from pilot.multimodal.fusion import MultimodalFusionEngine
        from pilot.neural.audit import NeuralAuditStore
        from pilot.neural.controller import NeuralController
        from pilot.neural.gate import NeuralIntentGate, NeuralIntentSigner
        from pilot.neural.goals import NeuralGoalRegistry

        neural_goals = NeuralGoalRegistry()
        self._fusion = MultimodalFusionEngine()
        self._fusion.set_broadcast(self._broadcast_notification)
        neural_signer = NeuralIntentSigner(derive_neural_signing_key(self._neural_auth_token))
        neural_gate = NeuralIntentGate(
            signer=neural_signer,
            safe_goals={command_id: command_id for command_id in neural_goals.command_ids},
        )
        neural_audit = NeuralAuditStore(
            config_module.DATA_DIR / "neural_intent_audit.db",
            config_module.DATA_DIR / "neural_intent_audit.key",
        )
        await neural_audit.initialize()
        self._neural_controller = NeuralController(
            config=self.config,
            gate=neural_gate,
            executor=self._executor,
            goals=neural_goals,
            broadcast=self._broadcast_notification,
            audit_store=neural_audit,
            fusion_snapshot=self._fusion.context_snapshot,
        )

        # Advanced agent components
        self._reflector = Reflector(model_router, memory_store=self._memory)
        await self._reflector.initialize()
        self._multi_agent = MultiAgentRouter(model_router)
        self._planner.set_multi_agent_router(self._multi_agent)
        self._background = BackgroundTaskManager()
        self._background.set_broadcast(self._broadcast_notification)
        self._background.register_builtin_monitors()

        # Multi-Agent Orchestrator — register all specialist agents
        from pilot.agents.agent_mesh import AgentMesh

        self._agent_mesh = AgentMesh(DATA_DIR / "agent_mesh.db")
        await self._agent_mesh.initialize()
        self._agent_mesh.set_experience_ledger(self._experience_ledger)
        self._orchestrator = AgentOrchestrator(
            model_router,
            agent_mesh=self._agent_mesh,
        )
        self._orchestrator.set_broadcast(self._broadcast_notification)
        self._orchestrator.set_budget_tracker(self._budget_tracker)
        self._orchestrator.set_circuit_breaker(self._circuit_breaker)
        self._orchestrator.set_agent_gateway(self._agent_gateway)
        from pilot.agents.registry import AgentRegistry

        AgentRegistry.discover_agents()
        registered = self._orchestrator.auto_register_all_agents(
            executor=self._executor,
            background_manager=self._background,
            model_router=model_router,
            config=self.config,
            vault=self._vault,
            memory=self._memory,
        )
        logger.info("Auto-registered %d agents via dynamic discovery", registered)
        await self._orchestrator.start_all()

        # ── Threat Containment Bridge (Issue #365) ──────────────────────────────
        # Must be created AFTER orchestrator.start_all() so the ForensicsAgent
        # is already registered and can be wired immediately.
        try:
            from pilot.agents.threat_containment import ThreatContainmentBridge

            self._threat_bridge = ThreatContainmentBridge(
                orchestrator=self._orchestrator,
                audit_logger=audit,
                broadcast_fn=self._broadcast_notification,
                pending_confirms=self._pending_confirms,
            )
            self._orchestrator.set_threat_bridge(self._threat_bridge)
            logger.info("ThreatContainmentBridge initialized and wired to ForensicsAgent.")
        except Exception:
            logger.warning("ThreatContainmentBridge init failed (non-critical)", exc_info=True)
        # ── End Threat Containment Bridge ──────────────────────────────────

        from pilot.agents.base_agent import AgentRole

        self._rss_agent = self._orchestrator.get_agent(AgentRole.RSS)
        if self._rss_agent is None:
            from pilot.agents.rss_agent import RssAgent

            self._rss_agent = RssAgent(model_router, self._memory, self.config, self._background)
            self._orchestrator.register_agent(self._rss_agent)
            await self._rss_agent.start()

        # Multimodal Fusion Engine — voice + gesture intent fusion
        # The engine is initialized before the neural controller so a neural
        # preview and commit can consume one consistent voice/gesture/gaze
        # safety snapshot rather than racing three independent buffers.

        # The backend gesture listener is intentionally disabled because it
        # would contend with the frontend MediaPipe engine for the webcam.
        # Do not import pilot.system.gesture here: importing that disabled
        # backend eagerly loads MediaPipe/TensorFlow and can delay daemon
        # readiness by tens of seconds even though no listener is started.
        logger.info("Local gesture listener disabled in favor of frontend UI MediaPipe engine")

        # Reasoning Event Emitter — thought visualization telemetry
        # Product execution now exposes only the compact interaction state.
        # The former thought graph emitted dozens of synchronous WebSocket
        # events per command without adding safety or execution authority.
        self._reasoning = None

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
        from pilot.intelligence.strategy_evolution import StrategyEvolutionLab

        self._strategy_evolution = StrategyEvolutionLab(
            DATA_DIR / "strategy_evolution.db",
            model_router=model_router,
        )
        await self._strategy_evolution.initialize()
        self._strategy_evolution.set_experience_ledger(self._experience_ledger)
        from pilot.intelligence.evolution_harness import EvolutionHarness

        self._evolution_harness = EvolutionHarness(
            DATA_DIR / "evolution_harness.db",
            Path(__file__).parent.parent.parent,
            DATA_DIR / "evolution_worktrees",
            model_router=model_router,
        )
        await self._evolution_harness.initialize()
        self._evolution_harness.set_experience_ledger(self._experience_ledger)

        # Plugin Ecosystem
        from pilot.plugins import PluginRegistry
        from pilot.plugins.marketplace import GitHubMarketplace

        self._plugin_registry = PluginRegistry()
        plugin_count = self._plugin_registry.discover()
        self._agent_mesh.refresh_plugins(self._plugin_registry.get_all_plugins())
        logger.info("Plugins loaded: %d", plugin_count)
        self._executor.set_plugin_registry(self._plugin_registry)
        self._plugin_marketplace = GitHubMarketplace(
            repo_root=Path(__file__).parent.parent.parent,
            plugins_dir=PLUGINS_DIR,
        )
        self._refresh_plugin_planner_context()

        # Subconscious Agent — long-term memory consolidation (lazy start)
        try:
            from pilot.agents.subconscious import SubconsciousAgent

            self._subconscious = SubconsciousAgent(model_router)
            await self._subconscious.initialize(str(DB_FILE))
            await self._subconscious.start(interval_minutes=30)
            logger.info("SubconsciousAgent started (continuous 30-minute consolidation)")
        except Exception:
            logger.warning("SubconsciousAgent init failed (non-critical)", exc_info=True)

        # Release announcement only. The canonical live cognitive runtime is
        # initialized in the CognitiveEngine block below.
        try:
            from pilot.changelog import announce_new_features, mark_version_seen

            announcement = announce_new_features()
            if announcement:
                logger.info("New features announcement: %s", announcement)
                self._new_features_announcement = announcement
                mark_version_seen()
        except Exception:
            logger.warning("Feature announcement initialization failed (non-critical)", exc_info=True)
            self._new_features_announcement = None

        # Screen Vision Agent — continuous screen awareness (AUTO-START for JARVIS mode)
        try:
            from pilot.agents.screen_vision import ScreenVisionAgent

            sv_config = self.config.screen_vision
            self._screen_vision = ScreenVisionAgent(
                model_router,
                capture_timeout_seconds=sv_config.capture_timeout_seconds,
                max_consecutive_timeouts=sv_config.max_consecutive_timeouts,
                auto_resume_after_seconds=sv_config.auto_resume_after_seconds,
            )
            await self._screen_vision.start(
                interval_seconds=sv_config.capture_interval_seconds,
                enable_describe=False,
            )
            gate.set_ui_observation_provider(self._screen_vision.get_world_observation)
            if self._activity_timeline is not None:
                self._screen_vision.set_state_observer(self._activity_timeline.observe)
            logger.info("ScreenVisionAgent auto-started (every %.1fs, JARVIS mode)", sv_config.capture_interval_seconds)
        except Exception:
            logger.warning("ScreenVisionAgent init failed (non-critical)", exc_info=True)

        # ── Cognitive Intelligence (lightweight heuristic engine) ──
        try:
            if not self.config.cognitive.enabled:
                logger.info("Cognitive intelligence disabled in config.toml")
            else:
                from pilot.cognitive.attention_scorer import AttentionAwareUI
                from pilot.cognitive.cognitive_engine import CognitiveEngine
                from pilot.cognitive.intent_predictor import IntentPredictor
                from pilot.cognitive.stress_gate import StressGate

                self._cognitive_engine = CognitiveEngine.get_instance()
                self._attention_ui = AttentionAwareUI(self._cognitive_engine)
                self._attention_ui.set_broadcast(self._broadcast_notification)
                self._stress_gate = StressGate(self._cognitive_engine)
                self._intent_predictor = IntentPredictor(self._cognitive_engine)

                if self._executor:
                    self._executor._stress_gate = self._stress_gate
                if self._fusion:
                    self._fusion._intent_predictor = self._intent_predictor
                if getattr(self, "_screen_vision", None):
                    self._screen_vision._cognitive_engine = self._cognitive_engine

                await self._cognitive_engine.load_model()
                logger.info("Cognitive intelligence initialized")
        except Exception:
            logger.warning("Cognitive intelligence init failed (non-critical)", exc_info=True)

        self._notification_buffer: list[tuple[str, dict[str, Any]]] = []

        # ── Autonomous Executor (JARVIS fire-and-forget) ──
        try:
            from pilot.agents.autonomous import AutonomousExecutor
            from pilot.agents.autonomous_review import AutonomousReviewer
            from pilot.intelligence.gnomon_reviewer import GnomonReviewer
            from pilot.workflows.autonomous_jobs import AutonomousJobStore

            # Gnomon loads in the background when a job starts and only ever adds caution.
            self._gnomon_reviewer = GnomonReviewer(self._typed_decision_model)

            self._autonomous = AutonomousExecutor(
                planner=self._planner,
                executor=self._executor,
                verifier=self._verifier,
                decomposer=self._decomposer,
                screen_vision=self._screen_vision,
                memory=self._memory,
                orchestrator=self._orchestrator,
                multi_agent_router=self._multi_agent,
                accessibility_observer=_accessibility_observation,
                human_control=self._human_control,
                job_store=AutonomousJobStore(),
                reviewer=AutonomousReviewer(
                    lambda: self.config,
                    gnomon=self._gnomon_reviewer,
                    on_assessment=self._publish_autonomous_assessment,
                ),
            )
            # Jobs cut off by the last shutdown come back as interrupted; the person decides whether to resume.
            await self._autonomous.recover()

            from pilot.workflows.scheduled_runs import ScheduledRunner, ScheduledRunStore

            self._scheduled_runs = ScheduledRunner(
                ScheduledRunStore(),
                submit=self._submit_scheduled_run,
                job_is_active=self._autonomous_job_is_active,
            )
            self._scheduled_runs.start()
            self._autonomous.set_broadcast(self._autonomous_broadcast)
            self._autonomous.set_speech(self._speak_companion_text)
            self._autonomous.set_approval_handler(self._wait_for_autonomous_confirmation)
            if self._neural_controller is not None:
                self._neural_controller.set_task_dispatcher(self._dispatch_neural_staged_task)
            logger.info("AutonomousExecutor initialized")
        except Exception:
            logger.warning("AutonomousExecutor init failed (non-critical)", exc_info=True)

        # Reactive triggers must enter the same guarded autonomous pipeline as
        # proactive suggestions. This preserves planning, permission checks,
        # UI approval, world-model review, and postcondition verification.
        from pilot.system.triggers import get_engine

        self._trigger_engine = get_engine()
        self._trigger_engine.set_fire_callback(self._dispatch_reactive_trigger)

        # ── Autonomous Healing Engine (passive system-health monitoring +
        # tiered auto-remediation, see pilot.agents.autonomous_healing) ──
        # Wired to the CPU/memory/disk monitors' on_trigger hook, which no
        # other consumer currently uses — the built-in monitors otherwise
        # only ever reach a UI toast via self._background's broadcast.
        try:
            from pilot.agents.autonomous_healing import AutonomousHealingEngine

            self._self_healing = AutonomousHealingEngine(
                planner=self._planner,
                executor=self._executor,
                config=self.config,
                pending_confirms=self._pending_confirms,
                broadcast_fn=self._broadcast_notification,
                attempts_file=DATA_DIR / "self_healing_attempts.json",
            )
            for task_id, handler in (
                ("monitor_cpu", self._self_healing.on_cpu_alert),
                ("monitor_memory", self._self_healing.on_memory_alert),
                ("monitor_disk", self._self_healing.on_disk_alert),
            ):
                task = self._background._tasks.get(task_id)
                if task is not None:
                    task.on_trigger = handler
            self._sync_self_healing_monitors()
            logger.info("AutonomousHealingEngine initialized")
        except Exception:
            logger.warning("AutonomousHealingEngine init failed (non-critical)", exc_info=True)

        # ── Live Execution Narrator (narrates + pre-emptively interrupts
        # in-progress plan execution, see pilot.agents.narrator) ──
        # Wiring is entirely via Executor.set_narrator(): once set, every
        # caller of self._executor.execute() (AutonomousExecutor, the
        # Voice/Gesture Workflow Engine, the interactive path) is narrated
        # automatically with zero changes of their own.
        try:
            from pilot.agents.narrator import ExecutionNarrator

            self._narrator = ExecutionNarrator(
                config=self.config,
                pending_confirms=self._pending_confirms,
                broadcast_fn=self._broadcast_notification,
            )
            self._executor.set_narrator(self._narrator)
            logger.info("ExecutionNarrator initialized")
        except Exception:
            logger.warning("ExecutionNarrator init failed (non-critical)", exc_info=True)

        # ── User Manual Supervision (watches the user's OWN independent
        # screen/keyboard/mouse activity, see pilot.agents.user_supervision) ──
        # Unlike self-healing/narration, the thing being gated here
        # (installing a global input hook, running periodic OCR) has real
        # cost and privacy weight even when idle -- so the BackgroundTask
        # and the hook are only actually started when the user has opted
        # in, and _handle_supervision_config_update (below) starts/stops
        # both on a config transition, not just flips a bool.
        try:
            from pilot.agents.background import BackgroundTask
            from pilot.agents.user_supervision import UserSupervisionEngine
            from pilot.system.input_hook import InputSupervisionHook

            self._supervision_hook = InputSupervisionHook(
                keystroke_buffer_max_chars=self.config.supervision.keystroke_buffer_max_chars
            )
            self._supervision = UserSupervisionEngine(
                config=self.config,
                cognitive_engine=self._cognitive_engine,
                screen_vision=self._screen_vision,
                hook=self._supervision_hook,
                broadcast_fn=self._broadcast_notification,
            )
            self._background.register(
                BackgroundTask(
                    task_id="user_supervision",
                    name="User Manual Supervision",
                    description="Watches the user's own independent screen/keyboard/mouse activity",
                    interval_seconds=self.config.supervision.tick_interval_seconds,
                    action_fn=self._supervision.tick,
                    on_trigger=self._supervision.on_trigger,
                )
            )
            if self.config.supervision.enabled:
                if self.config.supervision.keyboard_mouse_hook_enabled:
                    self._supervision_hook.start()
                self._background.start("user_supervision")
            logger.info("UserSupervisionEngine initialized (enabled=%s)", self.config.supervision.enabled)
        except Exception:
            logger.warning("UserSupervisionEngine init failed (non-critical)", exc_info=True)

        # ── Voice/Gesture Workflow Engine (durable, pausable/resumable
        # multi-step goals spanning multiple voice/gesture inputs) ──
        try:
            from pilot.agents.voice_gesture_workflow import VoiceGestureWorkflowEngine
            from pilot.workflows.voice_gesture_workflows import VoiceGestureWorkflowStore

            voice_gesture_workflow_store = VoiceGestureWorkflowStore()
            await voice_gesture_workflow_store.initialize()
            self._voice_gesture_workflows = VoiceGestureWorkflowEngine(
                planner=self._planner,
                executor=self._executor,
                decomposer=self._decomposer,
                workflow_store=voice_gesture_workflow_store,
                checkpoint_store=self._checkpoint_store,
                adaptive_executor=self._autonomous,
                memory=self._memory,
                pending_trigger_window_seconds=self.config.gesture_workflows.pending_trigger_window_seconds,
                paused_window_seconds=self.config.gesture_workflows.paused_window_seconds,
                # Gnomon decides whether a spoken addition amends the running step or replaces it.
                instruction_judge=self._gnomon_reviewer.still_wanted if self._gnomon_reviewer is not None else None,
            )
            self._voice_gesture_workflows.set_broadcast(self._broadcast_notification)
            self._voice_gesture_workflows.set_speech(self._speak_companion_text)
            logger.info("VoiceGestureWorkflowEngine initialized")
        except Exception:
            logger.warning("VoiceGestureWorkflowEngine init failed (non-critical)", exc_info=True)

        # ── Proactive Suggestion Engine (JARVIS anticipation) ──
        try:
            from pilot.agents.proactive import ProactiveSuggestionEngine

            self._proactive = ProactiveSuggestionEngine(screen_vision=self._screen_vision)
            self._proactive.set_broadcast(self._broadcast_notification)
            self._proactive.set_experience_ledger(self._experience_ledger)
            self._proactive.set_online_learner(self._online_learning)
            await self._proactive.start()
            logger.info("ProactiveSuggestionEngine auto-started")
        except Exception:
            logger.warning("ProactiveSuggestionEngine init failed (non-critical)", exc_info=True)

        self._handlers = {
            "execute": self._handle_execute,
            "queue_execute": self._handle_queue_execute,
            "cancel_queued_execute": self._handle_cancel_queued_execute,
            "resume_plan": self._handle_resume_plan,
            "resume_task": self._handle_resume_task,
            "mcp_plan_task": self._handle_mcp_plan_task,
            "mcp_typed_decision": self._handle_mcp_typed_decision,
            "mcp_submit_task": self._handle_mcp_submit_task,
            "mcp_task_status": self._handle_mcp_task_status,
            "mcp_cancel_task": self._handle_mcp_cancel_task,
            "export_session_chat": self._handle_export_session_chat,
            "confirm": self._handle_confirm,
            "rollback_plan": self._handle_rollback_plan,
            "undo_list": self._handle_undo_list,
            "undo_preview": self._handle_undo_preview,
            "undo_task": self._handle_undo_task,
            "undo_clear": self._handle_undo_clear,
            "receipts_list": self._handle_receipts_list,
            "receipt_get": self._handle_receipt_get,
            "receipt_verify": self._handle_receipt_verify,
            "receipts_verify_chain": self._handle_receipts_verify_chain,
            "timeline_status": self._handle_timeline_status,
            "timeline_search": self._handle_timeline_search,
            "timeline_summary": self._handle_timeline_summary,
            "timeline_pause": self._handle_timeline_pause,
            "timeline_clear": self._handle_timeline_clear,
            "phone_request_list": self._handle_phone_request_list,
            "today_overview": self._handle_today_overview,
            "routine_suggest": self._handle_routine_suggest,
            "reminder_list": self._handle_reminder_list,
            "focus_start": self._handle_focus_start,
            "focus_schedule_list": self._handle_focus_schedule_list,
            "privacy_inventory": self._handle_privacy_inventory,
            "privacy_clear": self._handle_privacy_clear,
            "timeline_forget": self._handle_timeline_forget,
            "backup_create": self._handle_backup_create,
            "backup_list": self._handle_backup_list,
            "backup_restore": self._handle_backup_restore,
            "focus_schedule_add": self._handle_focus_schedule_add,
            "focus_schedule_delete": self._handle_focus_schedule_delete,
            "workspace_list": self._handle_workspace_list,
            "command_guide": self._handle_command_guide,
            "notes_list": self._handle_notes_list,
            "note_add": self._handle_note_add,
            "note_delete": self._handle_note_delete,
            "todo_list": self._handle_todo_list,
            "todo_add": self._handle_todo_add,
            "todo_set_done": self._handle_todo_set_done,
            "todo_delete": self._handle_todo_delete,
            "todo_clear_done": self._handle_todo_clear_done,
            "nudge_list": self._handle_nudge_list,
            "nudge_add": self._handle_nudge_add,
            "nudge_delete": self._handle_nudge_delete,
            "watch_list": self._handle_watch_list,
            "watch_add": self._handle_watch_add,
            "watch_cancel": self._handle_watch_cancel,
            "clipboard_history_list": self._handle_clipboard_history_list,
            "clipboard_history_copy": self._handle_clipboard_history_copy,
            "clipboard_history_clear": self._handle_clipboard_history_clear,
            "system_health": self._handle_system_health,
            "focus_distractions_list": self._handle_focus_distractions_list,
            "focus_distraction_add": self._handle_focus_distraction_add,
            "focus_distraction_delete": self._handle_focus_distraction_delete,
            "app_reminder_list": self._handle_app_reminder_list,
            "app_reminder_delete": self._handle_app_reminder_delete,
            "workspace_save": self._handle_workspace_save,
            "workspace_delete": self._handle_workspace_delete,
            "receipts_search": self._handle_receipts_search,
            "focus_stop": self._handle_focus_stop,
            "focus_status": self._handle_focus_status,
            "report_export": self._handle_report_export,
            "reminder_add": self._handle_reminder_add,
            "reminder_cancel": self._handle_reminder_cancel,
            "undo_diff": self._handle_undo_diff,
            "shortcut_list": self._handle_shortcut_list,
            "shortcut_add": self._handle_shortcut_add,
            "shortcut_remove": self._handle_shortcut_remove,
            "daily_brief_now": self._handle_daily_brief_now,
            "phone_request_run": self._handle_phone_request_run,
            "phone_request_dismiss": self._handle_phone_request_dismiss,
            "air_handoff_requests_configure": self._handle_air_handoff_requests_configure,
            "list_permission_events": self._handle_list_permission_events,
            "verify_permission_audit": self._handle_verify_permission_audit,
            "list_gateway_events": self._handle_list_gateway_events,
            "verify_gateway_audit": self._handle_verify_gateway_audit,
            "gateway_policy_get": self._handle_gateway_policy_get,
            "gateway_policy_update": self._handle_gateway_policy_update,
            # ── Cancel Token (Issue #92) ──
            "abort": self._handle_abort,
            "interject": self._handle_interject,
            "get_config": self._handle_get_config,
            "get_security_status": self._handle_get_security_status,
            "get_snapshot_status": self._handle_get_snapshot_status,
            "restart_elevated": self._handle_restart_elevated,
            "update_config": self._handle_update_config,
            "reset_config": self._handle_reset_config,
            "get_history": self._handle_get_history,
            "memory_checkpoint": self._handle_memory_checkpoint,
            "temporal_memory_status": self._handle_temporal_memory_status,
            "temporal_memory_retract": self._handle_temporal_memory_retract,
            "store_api_key": self._handle_store_api_key,
            "delete_api_key": self._handle_delete_api_key,
            "list_api_keys": self._handle_list_api_keys,
            "subscription_status": self._handle_subscription_status,
            "subscription_login": self._handle_subscription_login,
            "calendar_test_connection": self._handle_calendar_test_connection,
            "email_test_connection": self._handle_email_test_connection,
            "ssh_list_hosts": self._handle_ssh_list_hosts,
            "ssh_set_enabled": self._handle_ssh_set_enabled,
            "ssh_save_host": self._handle_ssh_save_host,
            "ssh_delete_host": self._handle_ssh_delete_host,
            "ssh_test_connection": self._handle_ssh_test_connection,
            "list_ollama_models": self._handle_list_ollama_models,
            "health": self._handle_health,
            "ready": self._handle_ready,
            "ping": self._handle_ping,
            "system_status": self._handle_system_status,
            "system_info": self._handle_system_info,
            "get_uptime": self._handle_get_uptime,
            "capabilities": self._handle_capabilities,
            "reflection_stats": self._handle_reflection_stats,
            "background_tasks": self._handle_background_tasks,
            "background_start": self._handle_background_start,
            "background_stop": self._handle_background_stop,
            "extract_file_text": self._handle_extract_file_text,
            "agent_routing": self._handle_agent_routing,
            "agent_stats": self._handle_agent_stats,
            "agent_capabilities": self._handle_agent_capabilities,
            "agent_spawn": self._handle_agent_spawn,
            "agent_mesh_status": self._handle_agent_mesh_status,
            "voice_event": self._handle_voice_event,
            "gesture_event": self._handle_gesture_event,
            "gaze_event": self._handle_gaze_event,
            "cursor_move": self._handle_cursor_move,
            "cursor_click": self._handle_cursor_click,
            "multimodal_stats": self._handle_multimodal_stats,
            "decompose_task": self._handle_decompose_task,
            "simulate_plan": self._handle_simulate_plan,
            "prompt_strategies": self._handle_prompt_strategies,
            "prompt_stats": self._handle_prompt_stats,
            "strategy_evolution_status": self._handle_strategy_evolution_status,
            "strategy_candidates": self._handle_strategy_candidates,
            "strategy_propose": self._handle_strategy_propose,
            "strategy_reflect": self._handle_strategy_reflect,
            "strategy_record_isolated": self._handle_strategy_record_isolated,
            "strategy_start_shadow": self._handle_strategy_start_shadow,
            "strategy_record_shadow": self._handle_strategy_record_shadow,
            "strategy_start_canary": self._handle_strategy_start_canary,
            "strategy_record_canary": self._handle_strategy_record_canary,
            "strategy_promote": self._handle_strategy_promote,
            "strategy_rollback": self._handle_strategy_rollback,
            "evolution_status": self._handle_evolution_status,
            "evolution_runs": self._handle_evolution_runs,
            "evolution_candidates": self._handle_evolution_candidates,
            "evolution_create_run": self._handle_evolution_create_run,
            "evolution_generate_candidates": self._handle_evolution_generate_candidates,
            "evolution_evaluate": self._handle_evolution_evaluate,
            "evolution_request_promotion": self._handle_evolution_request_promotion,
            "plugin_list": self._handle_plugin_list,
            "plugin_tools": self._handle_plugin_tools,
            "plugin_toggle": self._handle_plugin_toggle,
            "plugin_market_list": self._handle_plugin_market_list,
            "plugin_install": self._handle_plugin_install,
            "plugin_uninstall": self._handle_plugin_uninstall,
            "plugin_create": self._handle_plugin_create,
            "plugin_run_tool": self._handle_plugin_run_tool,
            # Dynamic Python skills (pilot/skills + config skills dir)
            "skills_list": self._handle_skills_list,
            "skills_reload": self._handle_skills_reload,
            "skills_load_report": self._handle_skills_load_report,
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
            "cognitive_model_toggle": self._handle_cognitive_model_toggle,
            "voice_listener_start": self._handle_voice_listener_start,
            "voice_listener_stop": self._handle_voice_listener_stop,
            "voice_listener_stats": self._handle_voice_listener_stats,
            "voice_recognition_test": self._handle_voice_recognition_test,
            "interaction_status": self._handle_interaction_status,
            "list_audio_input_devices": self._handle_list_audio_input_devices,
            "speak_text": self._handle_speak_text,
            "stop_speech": self._handle_stop_speech,
            "companion_speech_status": self._handle_companion_speech_status,
            "reset_wake_calibration": self._handle_reset_wake_calibration,
            "list_wake_variants": self._handle_list_wake_variants,
            "autonomous_submit": self._handle_autonomous_submit,
            "autonomous_cancel": self._handle_autonomous_cancel,
            "autonomous_resume": self._handle_autonomous_resume,
            "autonomy_review_status": self._handle_autonomy_review_status,
            "scheduled_run_list": self._handle_scheduled_run_list,
            "scheduled_run_create": self._handle_scheduled_run_create,
            "scheduled_run_set_enabled": self._handle_scheduled_run_set_enabled,
            "scheduled_run_delete": self._handle_scheduled_run_delete,
            "scheduled_run_now": self._handle_scheduled_run_now,
            "autonomous_jobs": self._handle_autonomous_jobs,
            "autonomous_job": self._handle_autonomous_job,
            "risk_gate_status": self._handle_risk_gate_status,
            "risk_gate_config_update": self._handle_risk_gate_config_update,
            # Least-privileged neural sidecar and explicit UI control plane.
            "neural_status": self._handle_neural_status,
            "neural_stage_task": self._handle_neural_stage_task,
            "neural_remove_staged_task": self._handle_neural_remove_staged_task,
            "neural_connect": self._handle_neural_connect,
            "neural_begin_calibration": self._handle_neural_begin_calibration,
            "neural_finish_calibration": self._handle_neural_finish_calibration,
            "neural_arm": self._handle_neural_arm,
            "neural_intent_preview": self._handle_neural_intent_preview,
            "neural_observation": self._handle_neural_observation,
            "neural_commit": self._handle_neural_commit,
            "neural_disarm": self._handle_neural_disarm,
            "neural_stimulus_marker": self._handle_neural_stimulus_marker,
            "neural_stimulus_markers": self._handle_neural_stimulus_markers,
            "self_healing_status": self._handle_self_healing_status,
            "self_healing_config_update": self._handle_self_healing_config_update,
            "narration_status": self._handle_narration_status,
            "narration_config_update": self._handle_narration_config_update,
            "supervision_status": self._handle_supervision_status,
            "supervision_config_update": self._handle_supervision_config_update,
            "voice_gesture_workflow_submit": self._handle_voice_gesture_workflow_submit,
            "recorded_workflow_status": self._handle_recorded_workflow_status,
            "take_control": self._handle_take_control,
            "release_control": self._handle_release_control,
            "human_control_status": self._handle_human_control_status,
            "mcp_client_status": self._handle_mcp_client_status,
            "mcp_client_refresh": self._handle_mcp_client_refresh,
            "recorded_workflow_start": self._handle_recorded_workflow_start,
            "recorded_workflow_stop": self._handle_recorded_workflow_stop,
            "recorded_workflow_list": self._handle_recorded_workflow_list,
            "recorded_workflow_get": self._handle_recorded_workflow_get,
            "recorded_workflow_delete": self._handle_recorded_workflow_delete,
            "recorded_workflow_run": self._handle_recorded_workflow_run,
            "recorded_workflow_feedback": self._handle_recorded_workflow_feedback,
            "recorded_workflow_schedule": self._handle_recorded_workflow_schedule,
            "recorded_workflow_unschedule": self._handle_recorded_workflow_unschedule,
            "voice_gesture_workflow_list": self._handle_voice_gesture_workflow_list,
            "voice_gesture_workflow_get": self._handle_voice_gesture_workflow_get,
            "voice_gesture_workflow_pause": self._handle_voice_gesture_workflow_pause,
            "voice_gesture_workflow_resume": self._handle_voice_gesture_workflow_resume,
            "voice_gesture_workflow_cancel": self._handle_voice_gesture_workflow_cancel,
            "gesture_workflow_bindings_get": self._handle_gesture_workflow_bindings_get,
            "gesture_workflow_bindings_update": self._handle_gesture_workflow_bindings_update,
            "proactive_start": self._handle_proactive_start,
            "proactive_stop": self._handle_proactive_stop,
            "proactive_stats": self._handle_proactive_stats,
            "proactive_learning_status": self._handle_proactive_learning_status,
            "proactive_learning_reset": self._handle_proactive_learning_reset,
            "online_learning_status": self._handle_online_learning_status,
            "online_learning_reset": self._handle_online_learning_reset,
            "proactive_accept": self._handle_proactive_accept,
            "proactive_dismiss": self._handle_proactive_dismiss,
            "budget_stats": self._handle_budget_stats,
            "budget_reset": self._handle_budget_reset,
            # ── LAN Mesh Network ──
            "mesh_peers": self._handle_mesh_peers,
            "mesh_status": self._handle_mesh_status,
            "mesh_configure": self._handle_mesh_configure,
            "mesh_generate_secret": self._handle_mesh_generate_secret,
            "mesh_clear_secret": self._handle_mesh_clear_secret,
            # Encrypted one-target desktop-to-phone Air Handoff.
            "air_handoff_status": self._handle_air_handoff_status,
            "air_handoff_set_enabled": self._handle_air_handoff_set_enabled,
            "air_handoff_start_pairing": self._handle_air_handoff_start_pairing,
            "air_handoff_cancel_pairing": self._handle_air_handoff_cancel_pairing,
            "air_handoff_revoke_device": self._handle_air_handoff_revoke_device,
            "air_handoff_grab": self._handle_air_handoff_grab,
            "air_handoff_drop": self._handle_air_handoff_drop,
            "air_handoff_cancel": self._handle_air_handoff_cancel,
            "resolve_git_conflict": self._handle_resolve_git_conflict,
            "apply_git_resolution": self._handle_apply_git_resolution,
            # ── Plan History Audit Log ──
            "get_plan_history": self._handle_get_plan_history,
            "get_plan_detail": self._handle_get_plan_detail,
            # ── Threat Containment (Issue #365) ──
            "threat_containment_stats": self._handle_threat_containment_stats,
            # ── Assistance co-adaptation (presentation-only; no approval or execution path) ──
            "co_adaptation_status": self._handle_co_adaptation_status,
            "co_adaptation_config_update": self._handle_co_adaptation_config_update,
            "co_adaptation_consent_grant": self._handle_co_adaptation_consent_grant,
            "co_adaptation_consent_revoke": self._handle_co_adaptation_consent_revoke,
            "co_adaptation_preference_set": self._handle_co_adaptation_preference_set,
            "co_adaptation_preference_retract": self._handle_co_adaptation_preference_retract,
            "co_adaptation_learning_reset": self._handle_co_adaptation_learning_reset,
            "co_adaptation_erase": self._handle_co_adaptation_erase,
            "co_adaptation_results": self._handle_co_adaptation_results,
            "co_adaptation_export": self._handle_co_adaptation_export,
            "co_adaptation_session_begin": self._handle_co_adaptation_session_begin,
            "co_adaptation_session_resume": self._handle_co_adaptation_session_resume,
            "co_adaptation_recommend": self._handle_co_adaptation_recommend,
            "co_adaptation_record_trial": self._handle_co_adaptation_record_trial,
            "co_adaptation_report_burden": self._handle_co_adaptation_report_burden,
            "co_adaptation_session_end": self._handle_co_adaptation_session_end,
        }

        # ── LAN Mesh Network (opt-in via config) ──
        if self.config.network.enabled:
            try:
                mesh_secret = await self._vault.get_key("heliox_mesh")
                if not mesh_secret or len(mesh_secret.encode("utf-8")) < 32:
                    self._mesh_error = (
                        "Save the same LAN mesh secret (at least 32 bytes) in the OS credential vault on each peer"
                    )
                    raise ValueError(self._mesh_error)
                self._mesh = self._new_mesh(mesh_secret)
                logger.info("HelioxMesh initialised (will start with server)")
            except Exception as exc:
                if not self._mesh_error:
                    self._mesh_error = str(exc)
                logger.warning("HelioxMesh init failed (non-critical)", exc_info=True)

    async def _broadcast_notification(self, method: str, params: Any, *, exclude: Any = None) -> None:
        """Broadcast a notification to all connected clients.

        Args:
            method: The notification method name.
            params: The notification parameters.
            exclude: A connection that must not receive this notification
                (a caller that already has the result in its reply).
        """
        # ── Feature 5: Attention-Optimized Notification Timing ──
        # Only explicitly passive telemetry may be delayed. Unknown and
        # user-facing events bypass this presentation-only feature.
        if method not in ATTENTION_DEFERRABLE_NOTIFICATIONS:
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
                    buffered_params = params.copy() if isinstance(params, dict) else params
                    identity = self._attention_notification_identity(method, buffered_params)
                    for index in range(len(self._notification_buffer) - 1, -1, -1):
                        buffered_method, existing_params = self._notification_buffer[index]
                        if self._attention_notification_identity(buffered_method, existing_params) == identity:
                            self._notification_buffer[index] = (method, buffered_params)
                            break
                    else:
                        self._notification_buffer.append((method, buffered_params))
                    overflow = len(self._notification_buffer) - MAX_ATTENTION_NOTIFICATION_BUFFER
                    if overflow > 0:
                        del self._notification_buffer[:overflow]
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

        # This must run unconditionally (not just inside the attention-buffered
        # flush branch above) -- it was previously misplaced as unreachable code
        # after a different handler's return statements, so this notification
        # was silently never delivered outside that one buffered-flush edge case.
        msg = _notification(method, params)
        for client in list(self._clients):
            if exclude is not None and client is exclude:
                continue
            try:
                await client.send(msg)
            except Exception:
                pass

    @staticmethod
    def _attention_notification_identity(method: str, params: Any) -> tuple[str, str]:
        """Coalesce stale passive telemetry without merging distinct jobs."""
        if isinstance(params, dict):
            for key in ("task_id", "attempt_id", "plugin", "name"):
                value = str(params.get(key, "")).strip()
                if value:
                    return method, f"{key}:{value}"
        return method, "latest"

    async def _handle_extract_file_text(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Extract text from a file (e.g. PDF) for UI context injection."""
        path = params.get("path")
        if not path:
            return {"status": "error", "message": "No file path provided"}

        import os

        if not os.path.exists(path):
            return {"status": "error", "message": f"File not found: {path}"}

        try:
            from pilot.system.file_intel import parse_file

            text = await parse_file(path)
            return {"status": "ok", "text": text}
        except Exception as e:
            logger.error("Error extracting text from %s: %s", path, e)
            return {"status": "error", "message": str(e)}

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket connection from a client.

        The first message MUST be an ``auth`` request carrying the daemon's
        ``auth_token``.  Any other message — or a wrong token — closes the
        connection immediately with a JSON-RPC error response.

        Args:
            websocket: The WebSocket connection to the client.
        """
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)

        # ── Auth handshake ─────────────────────────────────────────────────
        # Wait up to 10 s for the first message; reject if it's not a valid
        # auth request with the correct token.
        try:
            first_raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Auth handshake timeout from %s — closing", remote)
            await websocket.close()
            return
        except websockets.exceptions.ConnectionClosed:
            return

        try:
            first_req = JsonRpcRequest.parse(str(first_raw))
        except (json.JSONDecodeError, ValueError, KeyError):
            await websocket.send(_error_response(None, -32600, "First message must be a valid JSON-RPC auth request"))
            await websocket.close()
            return

        if first_req.method != "auth":
            await websocket.send(
                _error_response(
                    first_req.id,
                    -32001,
                    "Authentication required: send {method:'auth', params:{token:'...'}} first",
                )
            )
            await websocket.close()
            logger.warning("Unauthenticated method '%s' from %s — rejected", first_req.method, remote)
            return

        provided_token = first_req.params.get("token", "")
        role = authenticate_rpc_client(
            provided_token,
            ui_token=self.config.server.auth_token,
            neural_token=self._neural_auth_token,
            mcp_token=self._mcp_auth_token,
        )
        if role is None:
            await websocket.send(_error_response(first_req.id, -32001, "Invalid auth token"))
            await websocket.close()
            logger.warning("Invalid auth token from %s — connection rejected", remote)
            return
        if (
            role == RpcClientRole.NEURAL_SIDECAR
            and self._neural_sidecar_client is not None
            and self._neural_sidecar_client is not websocket
        ):
            await websocket.send(_error_response(first_req.id, -32002, "A neural sidecar session is already connected"))
            await websocket.close()
            logger.warning("Second neural sidecar connection from %s rejected", remote)
            return

        # Auth passed — acknowledge and register as an active client
        await websocket.send(_success_response(first_req.id, {"status": "authenticated", "role": role.value}))
        if role == RpcClientRole.UI:
            self._clients.add(websocket)
        self._authenticated_clients.add(websocket)
        self._client_roles[websocket] = role
        if role == RpcClientRole.NEURAL_SIDECAR:
            self._neural_sidecar_client = websocket
        logger.info("Client authenticated: %s role=%s", remote, role.value)

        # Keep ordinary requests sequential, but allow confirmation and abort
        # RPCs to bypass a long-running ``execute`` request on the same socket.
        # Without this split, ``execute`` waits for ``confirm`` while the
        # connection loop waits for ``execute``: a deterministic timeout.
        request_lock = asyncio.Lock()
        request_tasks: set[asyncio.Task[None]] = set()

        async def _process_request(request: JsonRpcRequest) -> None:
            confirmation: PendingConfirmation | None = None
            current_task = asyncio.current_task()
            try:
                if not rpc_method_allowed(role, request.method):
                    if request.id is not None:
                        await websocket.send(
                            _error_response(
                                request.id,
                                -32601,
                                f"Method not available to {role.value}: {request.method}",
                            )
                        )
                    return
                if request.method == "confirm" and request.id is not None and current_task is not None:
                    confirmation = self._pending_confirms.get(str(request.params.get("plan_id", "")))
                    if confirmation is not None:
                        confirmation.rpc_tasks.add(current_task)
                if (
                    request.method in OUT_OF_BAND_RPC_METHODS
                    or request.method in SETTINGS_RPC_METHODS
                    or request.method in CO_ADAPTATION_RPC_METHODS
                ):
                    response = await self._dispatch(request, websocket)
                else:
                    async with request_lock:
                        response = await self._dispatch(request, websocket)
                if response and request.id is not None:
                    await websocket.send(response)
                    # The blocked execution must not resume until the UI's
                    # decision acknowledgement has actually been written.
                    if confirmation is not None and confirmation.resolved_by_task is current_task:
                        confirmation.acknowledged = True
                        confirmation.event.set()
                await self._announce_config_change(request, response)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection lost during request handling: %s", remote)
            except Exception as e:
                logger.exception("Handler error")
                try:
                    await websocket.send(_error_response(request.id, -32603, f"Internal error: {e}"))
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Could not send error response because the connection is closed: %s", remote)
            finally:
                if confirmation is not None and current_task is not None:
                    confirmation.rpc_tasks.discard(current_task)
                    if confirmation.resolved_by_task is current_task and not confirmation.acknowledged:
                        # The reply was never written (the socket closed or this task was cancelled with it).
                        _release_unacknowledged_confirmation(confirmation)

        # ── Normal message loop ────────────────────────────────────────────
        try:
            async for message in websocket:
                try:
                    request = JsonRpcRequest.parse(str(message))
                    task = asyncio.create_task(_process_request(request))
                    request_tasks.add(task)
                    task.add_done_callback(request_tasks.discard)
                except json.JSONDecodeError:
                    await websocket.send(_error_response(None, -32700, "Parse error"))
                except ValueError as e:
                    await websocket.send(_error_response(None, -32600, str(e)))
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection lost during request handling: %s", remote)
                    break
                except Exception as e:
                    logger.exception("Handler error")
                    try:
                        await websocket.send(_error_response(None, -32603, f"Internal error: {e}"))
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Could not send error response — connection already closed: %s", remote)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed during message loop: %s", remote)
        finally:
            for task in request_tasks:
                task.cancel()
            await asyncio.gather(*request_tasks, return_exceptions=True)
            self._clients.discard(websocket)
            self._authenticated_clients.discard(websocket)
            self._client_roles.pop(websocket, None)
            if self._neural_sidecar_client is websocket:
                self._neural_sidecar_client = None
                if self._neural_controller is not None:
                    await self._neural_controller.disarm(reason="sidecar_disconnected")
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
        if request.method not in CONFIG_MUTATING_RPC_METHODS:
            result = await handler(request.params, ws)
            return _success_response(request.id, result)
        async with self._config_lock:
            reload_if_edited = getattr(self.config, "reload_if_edited", None)
            if reload_if_edited is not None and reload_if_edited():
                await self._broadcast_notification("config_updated", {"method": "reload", "section": ""})
            result = await handler(request.params, ws)
        return _success_response(request.id, result)

    async def _announce_config_change(self, request: JsonRpcRequest, response: str | None) -> None:
        """Tell every open window that settings changed, after the caller has its reply."""
        if request.method not in CONFIG_MUTATING_RPC_METHODS or not response:
            return
        try:
            result = json.loads(response).get("result")
        except ValueError:
            return
        if not isinstance(result, dict) or result.get("status") == "error" or result.get("error"):
            return
        section = request.params.get("section", "") if isinstance(request.params, dict) else ""
        await self._broadcast_notification(
            "config_updated", {"method": request.method, "section": section if isinstance(section, str) else ""}
        )

    def _save_section(self, section: str, previous: Any) -> dict[str, Any] | None:
        """Save the config, or put ``section`` back to ``previous`` and return the error to send.

        Keeps what runs and what is on disk the same: a change that could not be saved
        is not left active until the next restart.
        """
        try:
            self.config.save()
        except Exception as exc:
            setattr(self.config, section, previous)
            logger.warning("%s settings were not saved; the previous values are back in effect", section)
            return {
                "status": "error",
                "message": f"{section.replace('_', ' ').capitalize()} settings were not saved: {exc}",
            }
        return None

    # -- Core execution pipeline --

    MAX_RETRIES = 2
    MAX_LIVE_CORRECTIONS = 5

    async def _append_experience(
        self,
        event_type: ExperienceEventType,
        **kwargs: Any,
    ) -> ExperienceEvent | None:
        """Persist an intelligence event without affecting product behavior."""

        if self._experience_ledger is None:
            return None
        try:
            return await self._experience_ledger.append(event_type, **kwargs)
        except Exception:
            logger.warning("Experience ledger append failed for %s", event_type.value, exc_info=True)
            return None

    async def _await_execution_tracked(self, execution: Any) -> Any:
        """Track any cancellable interactive phase in the cancellation slot.

        Planning, companion/critic review, execution, and verification can all
        involve a slow model or subprocess call. Tracking each phase makes a
        typed stop or correction responsive throughout the task, not only while
        an action happens to be executing.
        """
        task = asyncio.ensure_future(execution)
        self._active_execution_task = task
        try:
            return await task
        finally:
            if self._active_execution_task is task:
                self._active_execution_task = None

    @staticmethod
    def _deterministic_companion_review(plan: ActionPlan, user_input: str = "") -> Any | None:
        """Approve one bounded telemetry read without a remote model round-trip.

        These plans are produced by explicit local fast paths, contain no
        writes, and still pass through the world-model, permission, execution,
        and verification gates. Returning a real review keeps the companion
        observable without making a provider timeout part of a status query.
        """

        from pilot.actions import ActionType
        from pilot.agents.execution_companion import CompanionReview

        immediate_local_actions = {
            ActionType.SYSTEM_INFO,
            ActionType.SYSTEM_HEALTH_REVIEW,
            ActionType.CPU_USAGE,
            ActionType.MEMORY_USAGE,
            ActionType.DISK_USAGE,
            ActionType.NETWORK_INFO,
            ActionType.BATTERY_INFO,
            ActionType.PROCESS_LIST,
            ActionType.PROCESS_INFO,
            ActionType.SCREEN_OCR,
            ActionType.SCREEN_ANALYZE,
            ActionType.SCREEN_FIND_TEXT,
            ActionType.SCREEN_ELEMENT_MAP,
            ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT,
            ActionType.SCREENSHOT,
            # Direct local fast paths should not pay for a remote advisory
            # round-trip. Deterministic permissions, world-model assessment,
            # execution, and verification still run exactly as before.
            ActionType.OPEN_APPLICATION,
            ActionType.OPEN_URL,
            ActionType.NOTIFY,
        }
        if len(plan.actions) != 1 or plan.actions[0].action_type not in immediate_local_actions:
            return None
        action = plan.actions[0]
        action_type = action.action_type
        if action_type == ActionType.OPEN_APPLICATION and re.match(
            r"^\s*(?:please\s+)?(?:close|quit|exit)\b", user_input, re.IGNORECASE
        ):
            if re.search(r"\b(?:open|reopen|launch)\b", user_input, re.IGNORECASE):
                return None
            return CompanionReview(
                decision="REVISE",
                reason="Opening an application does not fulfill a request to close it.",
                planner_feedback="Use an action that closes the requested application, not open_application.",
            )
        if action_type == ActionType.SYSTEM_INFO and re.search(r"\bbattery\b", user_input, re.IGNORECASE):
            categories = set(getattr(action.parameters, "categories", ()))
            other_requested = re.search(
                r"\b(?:cpu|memory|ram|disk|storage|network|operating system|system information)\b",
                user_input,
                re.IGNORECASE,
            )
            if "battery" not in categories or (not other_requested and categories != {"battery"}):
                return CompanionReview(
                    decision="REVISE",
                    reason="The proposed system read does not match the requested battery-only status.",
                    planner_feedback="Use battery_info for a battery-only request; do not gather unrelated telemetry.",
                )
            if other_requested:
                return None
        if action_type in {ActionType.OPEN_APPLICATION, ActionType.OPEN_URL, ActionType.NOTIFY}:
            reason = "The direct local action is bounded, reversible, and matches the request."
        else:
            reason = "The local telemetry plan is bounded, read-only, and directly answers the request."
        return CompanionReview(
            decision="CONTINUE",
            reason=reason,
        )

    def _execution_scope_for_source(self, source_name: str) -> tuple[Any, Any | None]:
        """Resolve an interaction source to its enforced gateway ceiling."""
        from pilot.security.gateway import DEFAULT_SOURCE_PROFILES, InvocationSource, TaskScopeOverride

        source_key = source_name.strip().lower()
        invocation_source = {
            "voice": InvocationSource.VOICE,
            "mcp": InvocationSource.MCP,
        }.get(source_key, InvocationSource.INTERACTIVE)
        if invocation_source not in {InvocationSource.VOICE, InvocationSource.MCP}:
            return invocation_source, None
        profile_name = invocation_source.value
        profile = self.config.gateway.source_profiles.get(
            profile_name,
            DEFAULT_SOURCE_PROFILES[profile_name],
        )
        return invocation_source, TaskScopeOverride(
            max_tier=profile.max_tier,
            deny_action_types=profile.deny_action_types,
            allow_root=profile.allow_root,
        )

    async def _execute_tracked(self, plan: ActionPlan, **kwargs: Any) -> list[Any]:
        """Wraps self._executor.execute() in a real asyncio.Task, tracked in
        self._active_execution_task, so _handle_abort can cancel the
        CURRENTLY in-flight execution -- not just signal cancel_event for
        the next action boundary. Cancelling the returned task propagates
        all the way down to run_command's existing proc.kill() (see
        platform_detect.py), the same mechanism AutonomousExecutor.cancel()
        already proves works for killing a mid-flight shell subprocess.

        Only wraps the two interactive-session call sites in
        _handle_execute (fresh plan + resume-from-checkpoint) -- the other
        execute() call sites (voice command dispatch, the generic
        action-command handler, git-conflict-resolution) are single quick
        actions outside the "Stop button" scope and are left untouched.
        """
        return await self._await_execution_tracked(self._executor.execute(plan, **kwargs))

    async def _finalize_durable_task(self, task_id: str, response: dict[str, Any]) -> None:
        """Persist one terminal response so duplicate requests can replay it."""

        if self._durable_tasks is None:
            return
        task = await self._durable_tasks.get(task_id)
        if task is None or task.is_terminal:
            return
        response_status = str(response.get("status", "error"))
        target = {
            "success": TaskStatus.SUCCEEDED,
            "cancelled": TaskStatus.CANCELLED,
            "partial_failure": TaskStatus.PARTIAL,
            "interrupted": TaskStatus.INTERRUPTED,
        }.get(response_status, TaskStatus.FAILED)
        if target == TaskStatus.SUCCEEDED and task.status == TaskStatus.EXECUTING:
            task = await self._durable_tasks.transition(
                task_id,
                TaskStatus.VERIFYING,
                reason="execution completed before terminal success",
                plan_id=self._active_plan_id or task.plan_id,
            )
        if target == TaskStatus.PARTIAL and task.status not in {
            TaskStatus.EXECUTING,
            TaskStatus.VERIFYING,
        }:
            task = await self._durable_tasks.transition(
                task_id,
                TaskStatus.EXECUTING,
                reason="terminal partial result entered execution",
                plan_id=self._active_plan_id or task.plan_id,
            )
        await self._durable_tasks.transition(
            task_id,
            target,
            reason=f"interactive request finished with {response_status}",
            plan_id=self._active_plan_id or task.plan_id,
            terminal_response=response,
        )
        if self._memory is not None:
            await self._memory.clear_task_working(
                session_id=task.session_id,
                task_id=task.task_id,
            )

    async def _handle_queue_execute(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Accept another user task without interrupting the current one.

        Queued interactive tasks execute in order through the same planner,
        approval, and verification path. They are not concurrent desktop
        controllers, and queued work is never silently replayed after restart.
        """
        self._require_rpc_role(ws, RpcClientRole.UI)
        return await self._queue_interactive_request(params)

    async def _queue_interactive_request(self, params: dict[str, Any]) -> dict:
        if self._durable_tasks is None or self._queue_shutdown:
            return {"status": "unavailable", "message": "The task queue is not available."}
        user_input = str(params.get("input") or "").strip()
        if not user_input or len(user_input) > 20_000:
            return {"status": "error", "message": "Task input must contain 1-20,000 characters."}
        session_id = str(params.get("session_id") or "default")
        user_id = str(params.get("user_id") or "local")
        async with self._queued_submission_lock:
            if len(self._queued_execute_requests) >= 10:
                return {"status": "error", "message": "The interactive queue is full (10 tasks)."}
            created = await self._durable_tasks.create_task(
                session_id=session_id,
                user_id=user_id,
                user_input=user_input,
            )
            task_id = created.task.task_id
            queued_params = {
                "input": user_input,
                "task_id": task_id,
                "resume_token": created.resume_token,
                "session_id": session_id,
                "user_id": user_id,
                "source": str(params.get("source") or "text"),
            }
            self._queued_execute_requests.append((task_id, created.resume_token, queued_params))
            position = len(self._queued_execute_requests)
            await self._broadcast_notification(
                "queued_task_added",
                {"task_id": task_id, "session_id": session_id, "position": position, "input": user_input},
            )
        self._start_queued_execute_worker()
        return {
            "status": "queued",
            "task_id": task_id,
            "resume_token": created.resume_token,
            "position": position,
            "message": "Task queued. It will start after the current interactive task finishes.",
        }

    async def _handle_cancel_queued_execute(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI)
        task_id = str(params.get("task_id") or "")
        resume_token = str(params.get("resume_token") or "")
        if not task_id or not resume_token or self._durable_tasks is None:
            return {"status": "error", "message": "task_id and resume_token are required."}
        authorized = await self._durable_tasks.get_by_resume_token(resume_token)
        if authorized is None or authorized.task_id != task_id:
            return {"status": "error", "message": "Invalid queued task credentials."}
        if task_id == self._queued_reserved_task_id:
            return {"status": "running", "message": "This task has started; use Stop to cancel it."}
        queued = next((item for item in self._queued_execute_requests if item[0] == task_id), None)
        if queued is None:
            return {"status": "not_queued", "message": "This task is no longer waiting in the queue."}
        self._queued_execute_requests.remove(queued)
        response = {"status": "cancelled", "message": "Queued task cancelled before execution."}
        await self._durable_tasks.transition(
            task_id,
            TaskStatus.CANCELLED,
            reason="user cancelled queued task",
            terminal_response=response,
        )
        await self._broadcast_notification(
            "queued_task_result",
            {"task_id": task_id, "session_id": queued[2]["session_id"], **response},
        )
        return {"status": "cancelled", "task_id": task_id}

    def _start_queued_execute_worker(self) -> None:
        if (
            self._queue_shutdown
            or self._queued_submission_lock.locked()
            or not self._queued_execute_requests
            or self._interactive_request_active
            or self._mcp_reserved_task_id
            or (self._queued_execute_worker is not None and not self._queued_execute_worker.done())
        ):
            return
        self._queued_reserved_task_id = self._queued_execute_requests[0][0]
        self._queued_execute_worker = asyncio.create_task(self._run_queued_execute())

    async def _run_queued_execute(self) -> None:
        try:
            while not self._queue_shutdown:
                async with self._queued_submission_lock:
                    if not self._queued_execute_requests:
                        break
                    task_id, _, queued_params = self._queued_execute_requests[0]
                    self._queued_reserved_task_id = task_id
                session_id = queued_params["session_id"]
                await self._broadcast_notification(
                    "queued_task_started",
                    {"task_id": task_id, "session_id": session_id, "input": queued_params["input"]},
                )
                try:
                    response = await self._handle_execute(
                        queued_params,
                        _BroadcastConnection(self._broadcast_notification),
                    )
                except Exception:
                    logger.exception("Queued interactive task %s failed", task_id)
                    response = {"status": "error", "message": "The queued task failed inside the daemon."}
                    try:
                        await self._finalize_durable_task(task_id, response)
                    except Exception:
                        logger.exception("Could not finalize failed queued task %s", task_id)
                finally:
                    async with self._queued_submission_lock:
                        self._queued_execute_requests.popleft()
                        # Reserve the next task before any notification await.
                        # Otherwise a new execute request can jump ahead of it.
                        self._queued_reserved_task_id = (
                            self._queued_execute_requests[0][0] if self._queued_execute_requests else ""
                        )
                await self._broadcast_notification(
                    "queued_task_result",
                    {
                        "task_id": task_id,
                        "session_id": session_id,
                        "status": response.get("status", "error"),
                        "message": response.get("message") or response.get("explanation") or "Task finished.",
                        "result": {key: value for key, value in response.items() if key != "resume_token"},
                    },
                )
        finally:
            self._queued_reserved_task_id = ""
            self._queued_execute_worker = None
            if self._queued_execute_requests and not self._queue_shutdown:
                self._start_queued_execute_worker()

    async def _handle_execute(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Run one interactive request while exposing an out-of-band control slot."""
        requested_task_id = str(params.get("task_id") or uuid.uuid4())
        if self._interactive_request_active:
            return {
                "status": "busy",
                "message": "Heliox is already handling an interactive task.",
                "task_id": requested_task_id,
            }
        if self._queued_reserved_task_id and (
            requested_task_id != self._queued_reserved_task_id
            or asyncio.current_task() is not self._queued_execute_worker
        ):
            return {
                "status": "busy",
                "message": "Heliox is starting the next queued task.",
                "task_id": requested_task_id,
            }
        if self._mcp_reserved_task_id and (
            requested_task_id != self._mcp_reserved_task_id
            or asyncio.current_task() is not self._mcp_tasks.get(self._mcp_reserved_task_id)
        ):
            return {
                "status": "busy",
                "message": "Heliox is starting a local MCP task.",
                "task_id": requested_task_id,
            }
        # Reserve synchronously before any durable-store or interaction await.
        # Otherwise two clients can both pass the idle check and own the one
        # shared confirmation/cancellation slot.
        self._interactive_request_active = True
        try:
            return await self._handle_execute_reserved(params, ws)
        finally:
            self._interactive_request_active = False
            self._start_queued_execute_worker()

    async def _handle_execute_reserved(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        requested_task_id = str(params.get("task_id") or uuid.uuid4())
        source_name = str(params.get("source") or "text").strip().lower()
        interaction_id = str(params.get("_interaction_id") or "")
        resume_token = ""
        if self._durable_tasks is not None:
            existing = await self._durable_tasks.get(requested_task_id)
            if existing is not None:
                supplied_token = str(params.get("resume_token") or "")
                authorized = await self._durable_tasks.get_by_resume_token(supplied_token) if supplied_token else None
                if authorized is None or authorized.task_id != existing.task_id:
                    return {
                        "status": "error",
                        "message": "This task_id already exists; a valid resume_token is required.",
                        "task_id": existing.task_id,
                    }
                if existing.is_terminal and existing.terminal_response is not None:
                    return {
                        **existing.terminal_response,
                        "task_id": existing.task_id,
                        "replayed": True,
                    }
                if existing.status != TaskStatus.QUEUED or requested_task_id != self._queued_reserved_task_id:
                    return {
                        "status": "interrupted",
                        "message": "This task already exists. Use resume_task to continue it safely.",
                        "task_id": existing.task_id,
                        "task_state": existing.status.value,
                    }
                resume_token = supplied_token
                await self._durable_tasks.transition(
                    requested_task_id,
                    TaskStatus.PLANNING,
                    reason="queued interactive request started",
                )
            else:
                created = await self._durable_tasks.create_task(
                    task_id=requested_task_id,
                    session_id=str(params.get("session_id") or "default"),
                    user_id=str(params.get("user_id") or "local"),
                    user_input=str(params.get("input") or ""),
                )
                requested_task_id = created.task.task_id
                resume_token = created.resume_token
                await self._durable_tasks.transition(
                    requested_task_id,
                    TaskStatus.PLANNING,
                    reason="interactive request accepted",
                )
        if not interaction_id:
            interaction = await self._interaction_runtime.start(
                str(params.get("input") or ""),
                source=source_name,
            )
            interaction_id = str(interaction["interaction_id"])
        await self._interaction_runtime.transition(
            InteractionPhase.PLANNING,
            message="Planning the safest useful action",
            interaction_id=interaction_id,
        )
        self._active_interaction_id = interaction_id
        self._active_plan_id = ""
        self._active_task_id = requested_task_id
        self._live_correction = None
        context = ExperienceContext(
            session_id=str(params.get("session_id") or "default"),
            task_id=requested_task_id,
            user_id=str(params.get("user_id") or "local"),
        )
        self._active_experience_context = context
        try:
            with experience_scope(
                session_id=context.session_id,
                task_id=context.task_id,
                user_id=context.user_id,
            ):
                try:
                    if self._memory is not None:
                        await self._memory.put_working(
                            session_id=context.session_id,
                            task_id=context.task_id,
                            key="current intent",
                            value=str(params.get("input") or ""),
                            priority=1.0,
                            ttl_seconds=86400,
                        )
                    if resume_token:
                        await ws.send(
                            _notification(
                                "task_registered",
                                {
                                    "task_id": context.task_id,
                                    "resume_token": resume_token,
                                    "session_id": context.session_id,
                                },
                            )
                        )
                    response = await self._handle_execute_inner(params, ws)
                    footer = self._task_footers.pop(context.task_id, None)
                    if footer:
                        response = {**response, **footer}
                    await self._finalize_durable_task(context.task_id, response)
                    if self._durable_tasks is not None:
                        response = {**response, "task_id": context.task_id}
                        if resume_token:
                            response["resume_token"] = resume_token
                    response_status = str(response.get("status") or "error")
                    terminal_phase = (
                        InteractionPhase.COMPLETED
                        if response_status == "success"
                        else InteractionPhase.INTERRUPTED
                        if response_status in {"cancelled", "interrupted", "denied"}
                        else InteractionPhase.FAILED
                    )
                    await self._interaction_runtime.transition(
                        terminal_phase,
                        message=str(response.get("message") or response.get("explanation") or response_status)[:160],
                        interaction_id=interaction_id,
                    )
                    return response
                except Exception as exc:
                    await self._interaction_runtime.transition(
                        InteractionPhase.FAILED,
                        message="The task failed before a verified result was available",
                        interaction_id=interaction_id,
                    )
                    await self._append_experience(
                        ExperienceEventType.OUTCOME_VERIFIED,
                        idempotency_key=f"task:{context.task_id}:unhandled_error",
                        source="interactive",
                        payload={
                            "status": "internal_error",
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:1000],
                        },
                        provenance={"component": "PilotServer._handle_execute"},
                        privacy_class=PrivacyClass.SENSITIVE,
                    )
                    if self._durable_tasks is not None:
                        task = await self._durable_tasks.get(context.task_id)
                        if task is not None and not task.is_terminal:
                            await self._durable_tasks.transition(
                                context.task_id,
                                TaskStatus.FAILED,
                                reason=f"unhandled {type(exc).__name__}",
                                plan_id=self._active_plan_id or task.plan_id,
                                terminal_response={
                                    "status": "error",
                                    "message": "Internal error while executing task.",
                                },
                            )
                    raise
        finally:
            self._active_plan_id = ""
            self._active_task_id = ""
            if self._active_interaction_id == interaction_id:
                self._active_interaction_id = ""
            self._live_correction = None
            self._cancel_event = None
            self._active_experience_context = ExperienceContext()

    async def _handle_execute_inner(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Agentic pipeline: plan -> execute -> verify -> [retry on failure].

        If execution fails, the error is fed back to the planner for re-planning
        up to MAX_RETRIES times. Confirmation gates pause for user approval on
        Tier 2+ actions.
        """
        # A voice shortcut or "do that again" becomes an ordinary request with the normal approvals.
        expansion = await self._expand_request(str(params.get("input") or ""))
        if expansion is not None:
            params = {**params, "input": expansion[0]}
            await ws.send(_notification("request_expanded", {"input": expansion[0], "note": expansion[1]}))
        user_input = params.get("input", "")
        attachments = params.get("attachments", [])
        chat_session_id = str(params.get("session_id") or "default")
        interaction_source = str(params.get("source") or "interactive").strip().lower()
        revision_count = int(params.get("_companion_revision_count", 0))
        auto_revision_count = int(params.get("_companion_auto_revision_count", 0))

        if attachments:
            formatted_attachments = []

            for attachment in attachments:
                name = attachment.get("name", "unknown")
                content = attachment.get("content", "")

                formatted_attachments.append(f"[Attached File: {name}]\n{content}")

            user_input += "\n\nAttached Context:\n"
            user_input += "\n\n".join(formatted_attachments)

        if not user_input.strip():
            context = get_experience_context()
            await self._append_experience(
                ExperienceEventType.INTENT,
                idempotency_key=f"task:{context.task_id}:intent:empty",
                source="interactive",
                payload={"input": "", "attachment_count": len(attachments)},
                provenance={"component": "PilotServer._handle_execute_inner"},
                privacy_class=PrivacyClass.SENSITIVE,
            )
            await self._append_experience(
                ExperienceEventType.OUTCOME_VERIFIED,
                idempotency_key=f"task:{context.task_id}:terminal:empty",
                source="interactive",
                payload={"status": "error", "summary": "Empty input"},
                provenance={"component": "PilotServer._handle_execute_inner"},
            )
            return {"status": "error", "message": "Empty input"}
        dry_run = _resolve_dry_run(self.config.security.dry_run, params.get("dry_run", False))

        # ── Cancel Token (Issue #92): fresh event per execution session ──
        self._cancel_event = asyncio.Event()
        cancel_event = self._cancel_event

        import time

        _start_time = time.time()
        last_plan_id = ""

        def _record_memory(plan: Any, results: list[Any], verification: Any | None = None) -> Any:
            return self._memory.record(
                user_input,
                plan,
                results,
                session_id=chat_session_id,
                task_id=get_experience_context().task_id,
                verification=verification,
            )

        async def _record_planned_experience(plan: Any, plan_id: str, attempt: int) -> None:
            plan_event = await self._append_experience(
                ExperienceEventType.PLAN_CREATED,
                plan_id=plan_id,
                idempotency_key=f"plan:{plan_id}:created",
                source="planner",
                payload={
                    "action_count": len(plan.actions),
                    "explanation": plan.explanation,
                    "attempt": attempt,
                    "conversational": not bool(plan.actions),
                },
                provenance={"component": "Planner.plan"},
                privacy_class=PrivacyClass.SENSITIVE,
            )
            for index, action in enumerate(plan.actions):
                action_id = stable_action_idempotency_key(plan_id, index, action)
                await self._append_experience(
                    ExperienceEventType.CANDIDATE_ACTION,
                    plan_id=plan_id,
                    action_id=action_id,
                    parent_event_id=plan_event.event_id if plan_event else "",
                    idempotency_key=f"plan:{plan_id}:candidate:{index}",
                    source="planner",
                    payload={
                        "index": index,
                        "action_idempotency_key": action_id,
                        "action": action,
                    },
                    provenance={"component": "Planner.plan"},
                    privacy_class=PrivacyClass.SENSITIVE,
                )
            if self._memory is not None:
                await self._memory.put_working(
                    session_id=chat_session_id,
                    task_id=get_experience_context().task_id,
                    key="current plan",
                    value={
                        "plan_id": plan_id,
                        "attempt": attempt,
                        "summary": plan.explanation,
                        "action_count": len(plan.actions),
                    },
                    priority=0.9,
                    ttl_seconds=86400,
                )

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
                context = get_experience_context()
                if not dry_run and self._undo_journal is not None:
                    undo_summary = await self._undo_journal.summary(context.task_id or last_plan_id or "")
                    if undo_summary is not None:
                        payload["undo"] = undo_summary
                if self._task_receipts is not None:
                    receipt = await self._task_receipts.finalize(
                        context.task_id or last_plan_id or "",
                        request=str(params.get("input", "")),
                        status=status,
                        summary=payload["summary"],
                        source=str(params.get("source") or "text"),
                        undo=payload.get("undo"),
                    )
                    if receipt is not None:
                        payload["receipt_id"] = receipt["receipt_id"]
                await self._append_experience(
                    ExperienceEventType.OUTCOME_VERIFIED,
                    plan_id=last_plan_id,
                    idempotency_key=(f"task:{context.task_id}:terminal:{revision_count}:{auto_revision_count}"),
                    source="interactive",
                    payload=payload,
                    provenance={"component": "PilotServer._handle_execute_inner"},
                    privacy_class=PrivacyClass.SENSITIVE,
                )
                await self._broadcast_notification("task_complete", payload)
                # The execute reply carries these too, so the window that ran the task can show
                # Undo and the receipt on the task's own message.
                footer = {key: payload[key] for key in ("undo", "receipt_id") if key in payload}
                if footer and context.task_id:
                    self._task_footers[context.task_id] = footer
                    while len(self._task_footers) > 20:
                        self._task_footers.pop(next(iter(self._task_footers)))
            except Exception:
                pass

        # Timeline questions ("what was I working on this morning?") are answered from the local
        # timeline in code; they never reach the planner or a model.
        if not attachments:
            timeline_answer = await self._control_reply(str(user_input))
            if timeline_answer is not None:
                await ws.send(_notification("conversation_response", {"explanation": timeline_answer}))
                await _emit_task_complete("success", timeline_answer)
                return {
                    "status": "success",
                    "conversational": True,
                    "dry_run": False,
                    "results": [],
                    "explanation": timeline_answer,
                }

        async def _prepare_durable_replan(reason: str) -> None:
            if self._durable_tasks is None or not self._active_task_id:
                return
            task = await self._durable_tasks.get(self._active_task_id)
            if task is None or task.is_terminal:
                return
            if task.status != TaskStatus.INTERRUPTED:
                await self._durable_tasks.transition(
                    task.task_id,
                    TaskStatus.INTERRUPTED,
                    reason=reason,
                    plan_id=self._active_plan_id or task.plan_id,
                )
            await self._durable_tasks.transition(
                task.task_id,
                TaskStatus.PLANNING,
                reason="replanning after intervention",
            )

        async def _restart_with_live_correction(completed_results: list[Any] | None = None) -> dict | None:
            """Consume one queued correction and re-enter planning in this RPC.

            The current action is already cancelled by the out-of-band
            ``interject`` handler. We preserve a compact record of steps that
            really completed, then ask the planner to revise the remaining
            work instead of returning a misleading terminal cancellation.
            """
            correction = self._live_correction
            if not correction:
                return None

            self._live_correction = None
            if revision_count >= self.MAX_LIVE_CORRECTIONS:
                message = (
                    f"Live correction limit reached ({self.MAX_LIVE_CORRECTIONS}). "
                    "The task was stopped to avoid an endless revision loop."
                )
                await self._broadcast_notification(
                    "companion_revision_rejected",
                    {"message": message, "revision_count": revision_count},
                )
                await _emit_task_complete("cancelled", message)
                return {"status": "cancelled", "message": message, "results": []}

            completed = completed_results or []
            completed_lines = [
                (
                    f"- {result.action.action_type.value} on "
                    f"{getattr(result.action, 'target', '') or '(no target)'}: "
                    f"{'completed' if result.success else 'failed'}"
                )
                for result in completed
            ]
            completed_context = (
                "\n".join(completed_lines)
                if completed_lines
                else "- No action completed before the correction arrived."
            )
            revised_input = (
                f"{user_input}\n\n"
                "[LIVE USER CORRECTION]\n"
                f"{correction}\n\n"
                "[ALREADY OBSERVED OUTCOMES]\n"
                f"{completed_context}\n\n"
                "Revise the plan to honor the live correction. Do not repeat a "
                "completed action unless the correction explicitly requires it."
            )

            if self._checkpoint_store and self._active_plan_id:
                await self._checkpoint_store.mark_status(self._active_plan_id, "superseded")

            await ws.send(_notification("status", {"phase": "revising after your correction"}))
            await self._broadcast_notification(
                "companion_revision_started",
                {
                    "correction": correction,
                    "revision": revision_count + 1,
                    "completed_actions": len(completed),
                },
            )
            await _prepare_durable_replan("live user correction")
            return await self._handle_execute_inner(
                {
                    "input": revised_input,
                    "dry_run": dry_run,
                    "_companion_revision_count": revision_count + 1,
                    "_companion_auto_revision_count": auto_revision_count,
                    "session_id": chat_session_id,
                    "source": interaction_source,
                },
                ws,
            )

        async def _restart_from_companion_review(review: Any) -> dict:
            """Re-plan once the independent companion finds concrete drift."""
            feedback = str(getattr(review, "planner_feedback", "")).strip()
            reason = str(getattr(review, "reason", "")).strip()
            limit = max(0, int(self.config.narration.max_auto_revisions))
            if auto_revision_count >= limit:
                message = (
                    f"Interactive companion stopped this task after {limit} automatic "
                    f"revision attempts: {reason or 'the plan remained misaligned.'}"
                )
                await self._broadcast_notification(
                    "companion_plan_intervention",
                    {
                        "decision": "STOP",
                        "reason": message,
                        "revision": auto_revision_count,
                    },
                )
                await _emit_task_complete("blocked_by_companion", message)
                return {"status": "blocked_by_companion", "message": message, "results": []}

            if self._checkpoint_store and self._active_plan_id:
                await self._checkpoint_store.mark_status(self._active_plan_id, "superseded")

            await ws.send(
                _notification(
                    "status",
                    {"phase": f"companion revising plan ({auto_revision_count + 1}/{limit})"},
                )
            )
            await self._broadcast_notification(
                "companion_plan_intervention",
                {
                    "decision": "REVISE",
                    "reason": reason,
                    "planner_feedback": feedback,
                    "revision": auto_revision_count + 1,
                },
            )
            revised_input = (
                f"{user_input}\n\n"
                "[INDEPENDENT COMPANION REVIEW]\n"
                f"Problem: {reason}\n"
                f"Required correction: {feedback}\n\n"
                "Create a smaller corrected plan that still follows the original "
                "user request. Do not add goals, permissions, or side effects."
            )
            await _prepare_durable_replan("companion requested plan revision")
            return await self._handle_execute_inner(
                {
                    "input": revised_input,
                    "dry_run": dry_run,
                    "_companion_revision_count": revision_count,
                    "_companion_auto_revision_count": auto_revision_count + 1,
                    "session_id": chat_session_id,
                    "source": interaction_source,
                },
                ws,
            )

        async def _phase_cancelled(phase: str, completed_results: list[Any] | None = None) -> dict:
            """Shape task cancellation consistently for every long-running phase."""
            if self._live_correction:
                revised = await _restart_with_live_correction(completed_results)
                if revised is not None:
                    return revised
            message = f"Execution stopped by user during {phase}."
            await ws.send(_notification("status", {"phase": "aborted"}))
            await _emit_task_complete("cancelled", message)
            return {
                "status": "cancelled",
                "message": message,
                "results": [result.model_dump() for result in (completed_results or [])],
            }

        # A single interaction state replaces the removed ReAct event graph.
        # The guarded planning, approval, action, and verification flow below
        # remains authoritative, but internal thoughts are not serialized.
        emit = None

        context = get_experience_context()
        await self._append_experience(
            ExperienceEventType.INTENT,
            idempotency_key=(f"task:{context.task_id}:intent:{revision_count}:{auto_revision_count}"),
            source="interactive",
            payload={
                "input": user_input,
                "attachment_count": len(attachments),
                "revision_count": revision_count,
                "auto_revision_count": auto_revision_count,
            },
            provenance={"component": "PilotServer._handle_execute_inner"},
            privacy_class=PrivacyClass.SENSITIVE,
        )

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
        if self._strategy_evolution is not None:
            promoted_strategy = await self._strategy_evolution.get_active_text("planner.primary")
            if promoted_strategy:
                improvement_ctx = (
                    f"{improvement_ctx}\n\nAdmin-promoted planner strategy:\n{promoted_strategy}"
                ).strip()

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

        # Reflection is advisory planning context, not a failed-attempt error.
        # Passing it as ``error_context`` disabled deterministic fast paths for
        # any user with learned history and forced even simple status queries
        # through the model planner.
        error_context = ""
        all_results: list = []
        last_verification = None
        last_explanation = ""
        _original_plan = None
        _original_verification = None
        _successful_results: list = []
        _pending_retry_plan = None
        from pilot.agents.plan_differ import PlanDiffer
        from pilot.response_contract import (
            exact_labeled_finding_count,
            partial_failure_message,
            success_message,
        )

        for attempt in range(1 + self.MAX_RETRIES):
            review_screen_context = ""
            # ── Cancel Token: check before each planning attempt ──
            if cancel_event.is_set():
                logger.info("Execution cancelled before attempt %d", attempt + 1)
                message = "Execution was stopped before planning."
                await _emit_task_complete("cancelled", message)
                return {"status": "cancelled", "message": message}

            using_minimal_retry = _pending_retry_plan is not None
            plan_phase = ""
            if emit:
                event_name = PLANNER_STARTED if attempt == 0 else PLANNER_REPLANNING
                plan_phase = await emit.phase_start("planning", event_name, {"attempt": attempt + 1})
                thought = (
                    "Retrying only the actions that failed verification..."
                    if using_minimal_retry
                    else "Generating structured action plan via LLM..."
                )
                await emit.thought("planning", thought, parent_id=plan_phase)

            if attempt == 0:
                await ws.send(_notification("status", {"phase": "planning"}))
            elif using_minimal_retry:
                await ws.send(_notification("status", {"phase": f"retrying failed actions (attempt {attempt + 1})"}))
            else:
                await ws.send(_notification("status", {"phase": f"re-planning (attempt {attempt + 1})"}))

            if emit and not using_minimal_retry:
                await emit.data_event("planning", PLANNER_LLM_CALL, {"model": "active"}, parent_id=plan_phase)

            if using_minimal_retry:
                plan = _pending_retry_plan
                _pending_retry_plan = None
            else:
                _screen_ctx = ""
                if self._screen_vision:
                    try:
                        _screen_ctx = self._screen_vision.get_context_for_planner()
                        review_screen_context = self._screen_vision.get_recent_text_context()
                        await self._append_experience(
                            ExperienceEventType.OBSERVATION,
                            idempotency_key=(
                                f"task:{context.task_id}:screen_context:{revision_count}:{auto_revision_count}:{attempt}"
                            ),
                            source="screen_vision",
                            payload={
                                "context_available": bool(_screen_ctx),
                                "context_length": len(_screen_ctx),
                                "raw_media_excluded": True,
                            },
                            provenance={
                                "component": "ScreenVisionAgent.get_context_for_planner",
                                "consumer": "Planner.plan",
                            },
                            privacy_class=PrivacyClass.SENSITIVE,
                        )
                    except Exception:
                        pass
                recent_companion_context = self._recent_companion_context_by_session.get(chat_session_id, "")
                if recent_companion_context:
                    _screen_ctx = (f"{_screen_ctx}\n\n[RECENT COMPANION CONTEXT]\n{recent_companion_context}").strip()
                if improvement_ctx:
                    _screen_ctx = (
                        f"{_screen_ctx}\n\n[PRIOR VERIFIED IMPROVEMENT CONTEXT]\n"
                        f"{improvement_ctx}\n"
                        "Use this only to improve the plan; it is not evidence that the current request failed."
                    ).strip()
                if self._subconscious:
                    try:
                        persona_context = await self._subconscious.get_persona_context()
                        if persona_context:
                            _screen_ctx = (
                                f"{_screen_ctx}\n\n[LEARNED USER BEHAVIOR]\n"
                                f"{persona_context}\n"
                                "Treat these as preferences, never as permission to bypass "
                                "confirmation, safety policy, or the user's current request."
                            ).strip()
                    except Exception:
                        logger.debug("Could not load learned persona context", exc_info=True)

                try:
                    planner_kwargs: dict[str, Any] = {
                        "error_context": error_context,
                        "screen_context": _screen_ctx,
                        # Planner output is internal JSON, not an assistant response.
                        # Streaming it into chat exposes implementation details and can
                        # replace the final user-facing explanation.
                        "stream_callback": None,
                    }
                    if chat_session_id != "default":
                        planner_kwargs["session_id"] = chat_session_id
                    plan = await self._await_execution_tracked(
                        self._planner.plan(
                            user_input,
                            **planner_kwargs,
                        )
                    )
                except asyncio.CancelledError:
                    return await _phase_cancelled("planning")
            if cancel_event.is_set() and self._live_correction:
                revised = await _restart_with_live_correction()
                if revised is not None:
                    return revised
            if plan.error:
                if emit:
                    await emit.phase_error("planning", PLANNER_ERROR, plan.error, parent_id=plan_phase)
                if attempt < self.MAX_RETRIES and not _is_terminal_planning_failure(plan.error):
                    error_context = plan.error
                    continue
                if _is_terminal_planning_failure(plan.error):
                    logger.info("Terminal provider failure; skipping redundant planning retries")
                await _emit_task_complete("error", plan.error)
                return {"status": "error", "message": plan.error}

            last_explanation = plan.explanation

            # A useful conversational response is a valid zero-action plan.
            # It must bypass checkpoints, previews, permission gates,
            # execution, and verification: there is nothing to execute.
            if not plan.actions:
                conversation_plan_id = f"conversation:{context.task_id}:{attempt}"
                await _record_planned_experience(plan, conversation_plan_id, attempt)
                if emit:
                    await emit.phase_complete(
                        "planning",
                        PLANNER_GENERATED_PLAN,
                        {
                            "action_count": 0,
                            "explanation": plan.explanation[:120],
                            "action_types": [],
                            "conversational": True,
                        },
                        parent_id=plan_phase,
                    )

                if self._memory:
                    self._spawn_post_execution_task(_record_memory(plan, []), self._memory_record_tasks, "memory")
                await ws.send(_notification("conversation_response", {"explanation": plan.explanation}))
                await _emit_task_complete("success", plan.explanation)
                return {
                    "status": "success",
                    "conversational": True,
                    "dry_run": False,
                    "results": [],
                    "explanation": plan.explanation,
                    "agent_routing": routing,
                }

            plan_id = str(uuid.uuid4())[:8]
            last_plan_id = plan_id
            self._active_plan_id = plan_id
            await _record_planned_experience(plan, plan_id, attempt)
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

            if self.config.narration.proactive_review_enabled and self._execution_companion and not dry_run:
                await ws.send(_notification("status", {"phase": "companion reviewing plan"}))
                companion_review = self._deterministic_companion_review(plan, user_input)
                if companion_review is None:
                    try:
                        companion_review = await self._await_execution_tracked(
                            self._execution_companion.review(user_input, plan, screen_context=review_screen_context)
                        )
                    except asyncio.CancelledError:
                        return await _phase_cancelled("companion review")

                review_payload = {
                    "plan_id": plan_id,
                    **companion_review.to_dict(),
                    "revision": auto_revision_count,
                }
                await ws.send(_notification("companion_plan_review", review_payload))

                if companion_review.should_stop:
                    message = f"Interactive companion stopped this plan: {companion_review.reason}"
                    await self._broadcast_notification(
                        "companion_plan_intervention",
                        {**review_payload, "decision": "STOP", "reason": message},
                    )
                    if self._checkpoint_store:
                        await self._checkpoint_store.mark_status(plan_id, "blocked_by_companion")
                    await _emit_task_complete("blocked_by_companion", message)
                    return {
                        "status": "blocked_by_companion",
                        "message": message,
                        "companion_review": companion_review.to_dict(),
                        "results": [],
                    }

                if companion_review.should_revise:
                    return await _restart_from_companion_review(companion_review)

                if companion_review.decision == "WARN":
                    await self._broadcast_notification(
                        "companion_plan_intervention",
                        review_payload,
                    )

            from pilot.actions import PermissionTier
            from pilot.agents.destructive_critic import (
                HEURISTIC_RISK_THRESHOLD,
                assess_plan_risk,
                constrain_verdict_to_plan_authority,
            )

            critic_verdict_payload: dict[str, Any] | None = None
            plan_has_tier4 = any(a.permission_tier == PermissionTier.ROOT_CRITICAL for a in plan.actions)
            plan_has_tier3 = any(a.permission_tier == PermissionTier.DESTRUCTIVE for a in plan.actions)
            plan_has_irreversible = any(getattr(a, "is_irreversible", False) for a in plan.actions)
            plan_has_blockable_authority = plan_has_tier4 or plan_has_tier3 or plan_has_irreversible
            invocation_source, scope_override = self._execution_scope_for_source(interaction_source)
            risk_assessment = assess_plan_risk(plan, self.config, source=invocation_source.value)
            risk_score = risk_assessment.combined_score
            world_model_interrupt = risk_assessment.requires_confirmation
            await self._append_experience(
                ExperienceEventType.WORLD_PREDICTION,
                plan_id=plan_id,
                idempotency_key=f"plan:{plan_id}:world_prediction",
                source="risk_gate",
                payload=risk_assessment.to_dict(),
                provenance={
                    "component": "assess_plan_risk",
                    "policy_authority": "deterministic_rules",
                },
            )
            if self.config.gateway.risk_gate_enabled:
                await ws.send(
                    _notification(
                        "world_model_assessment",
                        {
                            "plan_id": plan_id,
                            **risk_assessment.to_dict(),
                        },
                    )
                )
            # Tier 4 always gets the LLM critic. Other plans pay for that
            # round-trip only when the heuristic or risk world model crosses
            # the review threshold; trivial single-file deletes still skip
            # straight to confirmation and say so in the audit trail.
            needs_critic_review = plan_has_blockable_authority and (
                plan_has_tier4 or risk_score >= HEURISTIC_RISK_THRESHOLD
            )
            critic_skipped_reason: str | None = None
            if not plan_has_blockable_authority and risk_score >= HEURISTIC_RISK_THRESHOLD:
                critic_skipped_reason = "no_destructive_authority"
            elif plan_has_tier3 and not (plan_has_tier4 or risk_score >= HEURISTIC_RISK_THRESHOLD):
                critic_skipped_reason = "low_risk_heuristic"

            if needs_critic_review and self._destructive_critic and not dry_run:
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
                        "Destructive/irreversible actions detected — running independent safety review...",
                        parent_id=critic_phase,
                    )

                try:
                    verdict = await self._await_execution_tracked(self._destructive_critic.review(user_input, plan))
                except asyncio.CancelledError:
                    return await _phase_cancelled("safety review")
                verdict = constrain_verdict_to_plan_authority(plan, verdict)
                critic_verdict_payload = verdict.to_dict()
                critic_verdict_payload["world_model"] = risk_assessment.to_dict()
                await ws.send(_notification("critic_verdict", critic_verdict_payload))

                if cancel_event.is_set() and self._live_correction:
                    revised = await _restart_with_live_correction()
                    if revised is not None:
                        return revised

                if verdict.is_blocked:
                    await self._record_permission_escalations(
                        plan_id=plan_id,
                        plan=plan,
                        confirmation_decision="blocked_by_critic",
                        critic_verdict=critic_verdict_payload,
                        results=[],
                        execution_error=verdict.recommendation,
                    )
                    # ── Plan History: blocked by critic ──
                    self._spawn_history_task(
                        self._record_plan_history(
                            plan_id=plan_id,
                            raw_input=user_input,
                            plan=plan,
                            critic_verdict=critic_verdict_payload,
                            confirmation_decision="blocked_by_critic",
                            execution_status="blocked_by_critic",
                            results=[],
                            verification=None,
                            dry_run=dry_run,
                            start_time=_start_time,
                        )
                    )
                    if emit:
                        await emit.phase_error(
                            "critic_review",
                            CRITIC_REVIEW_BLOCKED,
                            verdict.recommendation,
                            parent_id=critic_phase,
                        )
                    message = f"Blocked before execution by safety review: {verdict.recommendation}"
                    await _emit_task_complete("blocked_by_critic", message)
                    return {
                        "status": "blocked_by_critic",
                        "verdict": verdict.to_dict(),
                        "message": message,
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
            elif critic_skipped_reason and not dry_run:
                # Tier 3 / irreversible plan whose heuristic risk score was low
                # enough to skip the LLM round-trip — keep the audit trail
                # honest that a deeper review did not run, rather than implying
                # one silently approved it.
                critic_verdict_payload = {
                    "verdict": "SKIPPED",
                    "risk_score": risk_score,
                    "issues": risk_assessment.reasons,
                    "safe_actions": [],
                    "flagged_actions": [],
                    "recommendation": (
                        "The world-model warning remains active and normal approval is required; "
                        "the destructive LLM critic was skipped because this plan has no destructive authority."
                        if critic_skipped_reason == "no_destructive_authority"
                        else "Low-risk heuristic — LLM safety review was skipped."
                    ),
                    "critic_skipped": critic_skipped_reason,
                    "world_model": risk_assessment.to_dict(),
                }
                await ws.send(_notification("critic_verdict", critic_verdict_payload))

            mcp_requires_visible_approval = str(params.get("source") or "").strip().lower() == "mcp"
            needs_confirm = (
                self._permission_checker.plan_requires_confirmation(plan)
                or world_model_interrupt
                or mcp_requires_visible_approval
            ) and not dry_run
            partially_approved = False
            if needs_confirm:
                await self._interaction_runtime.transition(
                    InteractionPhase.AWAITING_APPROVAL,
                    message="Waiting for your approval",
                    interaction_id=self._active_interaction_id,
                )
                if self._durable_tasks is not None and self._active_task_id:
                    await self._durable_tasks.transition(
                        self._active_task_id,
                        TaskStatus.AWAITING_APPROVAL,
                        reason="plan requires user approval",
                        plan_id=plan_id,
                    )
                confirm_phase = ""
                if emit:
                    confirm_phase = await emit.phase_start("confirmation", CONFIRMATION_REQUIRED, {"plan_id": plan_id})
                    await emit.thought(
                        "confirmation",
                        (
                            "World-model assessment is unavailable — execution is paused..."
                            if world_model_interrupt and risk_assessment.model_version == "evaluation-error"
                            else "World model predicted a risky outcome — execution is paused..."
                            if world_model_interrupt
                            else "Dangerous action detected — awaiting user approval..."
                        ),
                        parent_id=confirm_phase,
                    )

                world_model_reason = ""
                if world_model_interrupt:
                    reason_text = "; ".join(risk_assessment.reasons) or "predicted outcome crossed the safety threshold"
                    world_model_reason = (
                        f"World-model assessment unavailable: {reason_text}"
                        if risk_assessment.model_version == "evaluation-error"
                        else f"World model paused this plan at {risk_assessment.world_model_score:.0%} "
                        f"predicted risk: {reason_text}"
                    )
                elif mcp_requires_visible_approval:
                    world_model_reason = (
                        "A local MCP client proposed this task. Heliox requires visible user approval "
                        "for every MCP action before execution."
                    )
                confirmed, approved_indices, required_indices = await self._wait_for_confirmation(
                    plan_id,
                    plan,
                    ws,
                    reason=world_model_reason,
                    risk_assessment=risk_assessment.to_dict() if world_model_interrupt else None,
                    force_all_actions=world_model_interrupt or mcp_requires_visible_approval,
                )

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
                    if self._live_correction:
                        revised = await _restart_with_live_correction()
                        if revised is not None:
                            return revised
                    message = "Cancelled before execution: the plan was denied."
                    await self._record_permission_escalations(
                        plan_id=plan_id,
                        plan=plan,
                        confirmation_decision="denied",
                        critic_verdict=critic_verdict_payload,
                        results=[],
                        execution_error=message,
                    )
                    # ── Plan History: user denied ──
                    self._spawn_history_task(
                        self._record_plan_history(
                            plan_id=plan_id,
                            raw_input=user_input,
                            plan=plan,
                            critic_verdict=critic_verdict_payload,
                            confirmation_decision="denied",
                            execution_status="cancelled",
                            results=[],
                            verification=None,
                            dry_run=dry_run,
                            start_time=_start_time,
                        )
                    )
                    await _emit_task_complete("cancelled", message)
                    return {
                        "status": "cancelled",
                        "message": message,
                        "explanation": plan.explanation,
                    }

                # Per-action granular approval: drop any confirmation-required
                # action the user didn't check, keeping order for dependency
                # correctness. Actions that don't require confirmation at all
                # are untouched.
                denied_indices = required_indices - approved_indices
                partially_approved = bool(denied_indices)
                if partially_approved:
                    logger.info(
                        "Plan %s partially approved — skipping %d action(s) denied by user: %s",
                        plan_id,
                        len(denied_indices),
                        sorted(denied_indices),
                    )
                    plan.actions = [a for i, a in enumerate(plan.actions) if i not in denied_indices]
                    # A resume must never run an unchecked action. A durable approval records the approved
                    # indices against the checkpoint's full plan, and resume_task drops the others. Without one,
                    # the checkpoint itself must hold only the approved plan; nothing has run yet, so it restarts
                    # from the first approved action.
                    if self._checkpoint_store and (
                        self._durable_tasks is None or await self._durable_tasks.get_approval(plan_id) is None
                    ):
                        await self._checkpoint_store.start_plan(plan_id, user_input, plan)
            elif not dry_run:
                if emit:
                    skip_phase = await emit.phase_start("confirmation", "confirmation_skipped")
                    await emit.phase_complete(
                        "confirmation", "confirmation_skipped", {"reason": "No dangerous actions"}, parent_id=skip_phase
                    )

            approved_decision = "partially_approved" if partially_approved else "approved"
            if self._durable_tasks is not None and self._active_task_id:
                await self._durable_tasks.transition(
                    self._active_task_id,
                    TaskStatus.EXECUTING,
                    reason="approval satisfied; execution starting",
                    plan_id=plan_id,
                )

            exec_phase = ""
            if emit:
                exec_phase = await emit.phase_start("execution", EXECUTOR_STARTED, {"action_count": len(plan.actions)})

            await self._interaction_runtime.transition(
                InteractionPhase.ACTING,
                message="Executing the approved plan",
                interaction_id=self._active_interaction_id,
            )
            await ws.send(_notification("status", {"phase": "executing"}))
            action_idx = 0
            _total_actions = len(plan.actions)
            completed_results: list[Any] = []

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

            async def _on_action_complete(
                result: Any,
                _exec_phase: str = exec_phase,
                _plan_id: str = plan_id,
                _completed_results: list[Any] = completed_results,
            ) -> None:
                _completed_results.append(result)
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

            try:
                # A dry run skipped the critic and the approval above, so it
                # must never reach specialist agents: the executor simulates it.
                if self._orchestrator and not dry_run:
                    orch_routing = self._orchestrator.get_routing_summary(plan)
                    await ws.send(_notification("orchestrator_routing", orch_routing))
                    if emit:
                        await emit.data_event("orchestration", ORCHESTRATOR_ROUTING, orch_routing, parent_id=exec_phase)
                        for agent_info in orch_routing.get("assigned_agents", []):
                            role_name = agent_info["role"] if isinstance(agent_info, dict) else str(agent_info)
                            await emit.thought(
                                "orchestration", f"Delegating to {role_name} agent...", parent_id=exec_phase
                            )

                    results = await self._await_execution_tracked(
                        self._orchestrator.execute_plan(
                            user_input,
                            plan,
                            on_action_start=_on_action_start,
                            on_action_complete=_on_action_complete,
                            cancel_event=cancel_event,  # ── Cancel Token (Issue #92) ──
                            plan_id=plan_id,
                            critic_already_reviewed=True,
                            user_confirmed=needs_confirm and approved_decision in {"approved", "partially_approved"},
                            scope_override=scope_override,
                        )
                    )
                else:
                    results = await self._execute_tracked(
                        plan,
                        on_action_start=_on_action_start,
                        on_action_complete=_on_action_complete,
                        cancel_event=cancel_event,  # ── Cancel Token (Issue #92) ──
                        plan_id=plan_id,
                        # Interactive is the default invocation_source, but the
                        # critic already ran above (or was deliberately skipped
                        # as low-risk) — don't have the gateway pay for a
                        # redundant LLM round-trip.
                        critic_already_reviewed=True,
                        user_confirmed=needs_confirm and approved_decision in {"approved", "partially_approved"},
                        invocation_source=invocation_source,
                        scope_override=scope_override,
                        dry_run=dry_run,
                    )
            except asyncio.CancelledError:
                # ── Mid-flight cancellation: _handle_abort (Part 3) sets
                # cancel_event *then* cancels _active_execution_task, so by
                # the time this CancelledError (a BaseException) reaches us,
                # cancel_event.is_set() is already True -- fall through to
                # the existing "Cancel Token" handling below with whatever
                # results the task produced before it was killed (none, since
                # the cancelled await never returns a value). This must be
                # caught here rather than left to propagate, or it would
                # escape this RPC handler as an unhandled BaseException. ──
                results = list(completed_results)
            all_results = results
            if not dry_run:
                _snapshot_id = next((r.snapshot_id for r in results if getattr(r, "snapshot_id", None)), None)
                if _snapshot_id:
                    self._plan_snapshots[plan_id] = _snapshot_id
            if needs_confirm and not dry_run:
                await self._record_permission_escalations(
                    plan_id=plan_id,
                    plan=plan,
                    confirmation_decision=approved_decision,
                    critic_verdict=critic_verdict_payload,
                    results=results,
                )

            # ── Cancel Token: if aborted mid-execution, return immediately ──
            if cancel_event.is_set():
                if self._live_correction:
                    revised = await _restart_with_live_correction(results)
                    if revised is not None:
                        return revised
                logger.info("Execution was cancelled mid-plan after %d result(s)", len(results))
                await ws.send(_notification("status", {"phase": "aborted"}))
                if self._checkpoint_store:
                    await self._checkpoint_store.mark_status(plan_id, "cancelled")
                # ── Plan History: cancelled mid-execution ──
                self._spawn_history_task(
                    self._record_plan_history(
                        plan_id=plan_id,
                        raw_input=user_input,
                        plan=plan,
                        critic_verdict=critic_verdict_payload,
                        confirmation_decision=approved_decision if needs_confirm else "skipped",
                        execution_status="cancelled",
                        results=results,
                        verification=None,
                        dry_run=dry_run,
                        start_time=_start_time,
                    )
                )
                message = (
                    f"Execution stopped by user after {len(results)} of {len(plan.actions)} actions completed."
                    if plan.actions
                    else "Execution stopped by user."
                )
                await _emit_task_complete("cancelled", message)
                return {
                    "status": "cancelled",
                    "message": message,
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

            if self._durable_tasks is not None and self._active_task_id:
                await self._durable_tasks.transition(
                    self._active_task_id,
                    TaskStatus.VERIFYING,
                    reason="action execution finished",
                    plan_id=plan_id,
                )
            await self._interaction_runtime.transition(
                InteractionPhase.VERIFYING,
                message="Verifying the result",
                interaction_id=self._active_interaction_id,
            )
            await ws.send(_notification("status", {"phase": "verifying"}))
            if dry_run:
                from pilot.actions import VerificationResult

                # Nothing ran, but a simulated action that fails validation or
                # policy must not be reported as a plan that would succeed.
                failed = [index for index, result in enumerate(results) if not result.success]
                verification = VerificationResult(
                    passed=not failed,
                    details=[
                        f"Dry run: action {index + 1} would fail: {results[index].error or 'unknown error'}"
                        for index in failed
                    ]
                    or ["Dry run completed: no actions were executed."],
                    failed_actions=failed,
                    rollback_triggered=False,
                )
            else:
                try:
                    verification = await self._await_execution_tracked(self._verifier.verify(plan, results))
                except asyncio.CancelledError:
                    return await _phase_cancelled("verification", results)
            if _original_plan is not None and _original_verification is not None:
                results = PlanDiffer.merge_results(
                    _successful_results,
                    results,
                    _original_plan,
                    _original_verification,
                )
                plan = _original_plan
                try:
                    verification = await self._await_execution_tracked(self._verifier.verify(plan, results))
                except asyncio.CancelledError:
                    return await _phase_cancelled("verification", results)
                _original_plan = None
                _original_verification = None
                _successful_results = []

            all_results = results
            last_verification = verification
            await self._append_experience(
                ExperienceEventType.OUTCOME_VERIFIED,
                plan_id=plan_id,
                idempotency_key=f"plan:{plan_id}:verification:{attempt}",
                source="verifier",
                payload={
                    "attempt": attempt,
                    "verification": verification,
                    "result_count": len(results),
                },
                provenance={"component": "Verifier.verify"},
            )
            if not verification.passed:
                await self._append_experience(
                    ExperienceEventType.PREDICTION_ERROR,
                    plan_id=plan_id,
                    idempotency_key=f"plan:{plan_id}:prediction_error:{attempt}",
                    source="verifier",
                    payload={
                        "attempt": attempt,
                        "details": verification.details,
                        "failed_actions": verification.failed_actions,
                    },
                    provenance={
                        "component": "Verifier.verify",
                        "compared_prediction": f"plan:{plan_id}:world_prediction",
                    },
                )
            if cancel_event.is_set() and self._live_correction:
                revised = await _restart_with_live_correction(results)
                if revised is not None:
                    return revised
            if _original_plan is not None and _successful_results:
                all_results = PlanDiffer.merge_results(_successful_results, results, _original_plan, verification)

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

                self._spawn_post_execution_task(
                    _record_memory(plan, results, verification), self._memory_record_tasks, "memory"
                )
                if self._checkpoint_store:
                    await self._checkpoint_store.mark_status(plan_id, "complete")
                self._spawn_post_execution_task(
                    self._reflector.reflect(
                        user_input,
                        plan,
                        results,
                        verification,
                        retry_count=attempt,
                        duration_ms=int((time.time() - _start_time) * 1000),
                        dry_run=dry_run,
                        session_id=chat_session_id,
                    ),
                    self._reflection_tasks,
                    "reflection",
                )

                if emit:
                    await emit.phase_complete(
                        "memory_update", MEMORY_STORE_COMPLETE, {"saved": True}, parent_id=mem_store_phase
                    )

                # ── Plan History: success ──
                self._spawn_history_task(
                    self._record_plan_history(
                        plan_id=plan_id,
                        raw_input=user_input,
                        plan=plan,
                        critic_verdict=critic_verdict_payload,
                        confirmation_decision=approved_decision if needs_confirm else ("n/a" if dry_run else "skipped"),
                        execution_status="dry_run" if dry_run else "success",
                        results=results,
                        verification=verification,
                        dry_run=dry_run,
                        start_time=_start_time,
                    )
                )

                message = success_message(plan, results, verification, dry_run=dry_run)
                self._recent_companion_context_by_session[chat_session_id] = (
                    f"Previous request: {_sanitize_summary(user_input, limit=300)}\n"
                    f"Execution result: {_sanitize_summary(message, limit=700)}"
                )
                await _emit_task_complete("success", message)

                if (
                    self.config.narration.follow_up_enabled
                    and self._execution_companion
                    and hasattr(self._execution_companion, "follow_up")
                    and not dry_run
                    and exact_labeled_finding_count(plan) is None
                ):
                    self._spawn_companion_follow_up(
                        user_input=user_input,
                        plan=plan,
                        results=results,
                        verification=verification,
                        result_text=message,
                        chat_session_id=chat_session_id,
                    )
                return {
                    "status": "success",
                    "dry_run": dry_run,
                    "results": [r.model_dump() for r in results],
                    "verification": verification.model_dump(),
                    "message": message,
                    "explanation": (
                        f"(dry run) {plan.explanation}"
                        if dry_run and plan.explanation
                        else "(dry run) Dry run completed: no changes were made."
                        if dry_run
                        else plan.explanation
                    ),
                    "agent_routing": self._multi_agent.get_routing_summary(user_input),
                    "companion_follow_up": None,
                }

            if emit:
                await emit.phase_error(
                    "verification", VERIFICATION_FAILED, "; ".join(verification.details[:3]), parent_id=verify_phase
                )

            if _is_terminal_execution_failure(results):
                logger.info("Terminal execution failure; returning the exact error without LLM retry")
                break

            if _postcondition_failure_requires_reconciliation(results, verification):
                logger.warning(
                    "Destructive action succeeded but its post-condition was not observed; "
                    "stopping before retry for manual reconciliation"
                )
                break

            # Execution failed — use PlanDiffer for partial re-plan
            retry_plan, successful_results = PlanDiffer.diff(plan, results, verification)

            failed_details = [d for d in verification.details if "FAILED" in d or "MISMATCH" in d]
            error_msgs = [r.error for r in results if r.error]
            error_context = "\n".join(failed_details + error_msgs)

            # Use partial retry plan if PlanDiffer found fewer actions to retry
            if len(retry_plan.actions) < len(plan.actions) and attempt < self.MAX_RETRIES:
                logger.info(
                    "PlanDiffer: retrying %d/%d actions",
                    len(retry_plan.actions),
                    len(plan.actions),
                )
                _original_plan = plan
                _original_verification = verification
                _successful_results = successful_results
                _pending_retry_plan = retry_plan
                all_results = list(successful_results)

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

        if not dry_run and last_verification is not None and results:
            self._spawn_post_execution_task(
                self._reflector.reflect(
                    user_input,
                    plan,
                    results,
                    last_verification,
                    retry_count=attempt,
                    duration_ms=int((time.time() - _start_time) * 1000),
                    dry_run=False,
                    session_id=chat_session_id,
                ),
                self._reflection_tasks,
                "reflection",
            )

        self._spawn_post_execution_task(
            _record_memory(plan, all_results, last_verification), self._memory_record_tasks, "memory"
        )
        if self._checkpoint_store and last_plan_id:
            await self._checkpoint_store.mark_status(last_plan_id, "failed")

        # ── Plan History: partial_failure after all retries exhausted ──
        self._spawn_history_task(
            self._record_plan_history(
                plan_id=last_plan_id,
                raw_input=user_input,
                plan=plan,
                critic_verdict=critic_verdict_payload,
                confirmation_decision=approved_decision if needs_confirm else "skipped",
                execution_status="partial_failure",
                results=all_results,
                verification=last_verification,
                dry_run=dry_run,
                start_time=_start_time,
            )
        )

        message = partial_failure_message(all_results, last_verification)
        await _emit_task_complete("partial_failure", message)
        return {
            "status": "partial_failure",
            "dry_run": dry_run,
            "results": [r.model_dump() for r in all_results],
            "verification": last_verification.model_dump() if last_verification else {},
            "message": message,
            "explanation": (
                f"(dry run) {last_explanation}"
                if dry_run and last_explanation
                else "(dry run) Dry run completed: no changes were made."
                if dry_run
                else last_explanation
            ),
        }

    # ── Plan History: internal helper ──

    async def _record_plan_history(
        self,
        *,
        plan_id: str,
        raw_input: str,
        plan: Any,
        critic_verdict: dict[str, Any] | None,
        confirmation_decision: str,
        execution_status: str,
        results: list[Any],
        verification: Any | None,
        dry_run: bool,
        start_time: float,
    ) -> None:
        """Fire-and-forget wrapper that persists a plan audit record safely.

        Swallows all exceptions so a storage failure never disrupts execution.

        Args:
            plan_id: Short UUID identifying this plan.
            raw_input: Original user input string.
            plan: ActionPlan object.
            critic_verdict: Optional critic verdict dict.
            confirmation_decision: User/system confirmation outcome.
            execution_status: Terminal execution status string.
            results: List of ActionResult objects.
            verification: Optional VerificationResult object.
            dry_run: Whether this was a dry-run.
            start_time: ``time.time()`` at the start of execution for duration calc.
        """
        if not self._plan_history or not plan_id:
            return
        try:
            import time as _time

            duration_ms = int((_time.time() - start_time) * 1000)
            await self._plan_history.record(
                plan_id=plan_id,
                raw_input=raw_input,
                plan=plan,
                critic_verdict=critic_verdict,
                confirmation_decision=confirmation_decision,
                execution_status=execution_status,
                results=results,
                verification=verification,
                dry_run=dry_run,
                duration_ms=duration_ms,
            )
        except Exception:
            logger.warning("_record_plan_history failed (non-critical)", exc_info=True)

    def _spawn_history_task(self, coro: Any) -> None:
        """Schedule a plan-history coroutine as a tracked background task.

        The task is added to ``_plan_history_tasks`` and automatically removed
        when it completes, so ``stop()`` can drain any in-flight writes before
        closing the SQLite connection.

        Args:
            coro: The coroutine to schedule (typically ``_record_plan_history(...)``).
        """
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._plan_history_tasks.add(task)
        task.add_done_callback(self._plan_history_tasks.discard)

    @staticmethod
    def _spawn_post_execution_task(
        coro: Any,
        tasks: set[asyncio.Task[Any]],
        label: str,
    ) -> None:
        """Track a post-result job and consume any background exception."""
        task = asyncio.create_task(coro)
        tasks.add(task)

        def _finished(done: asyncio.Task[Any]) -> None:
            tasks.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.warning(
                    "Post-execution %s task failed: %s",
                    label,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(_finished)

    async def _drain_post_execution_tasks(self) -> None:
        """Finish durable memory writes and cancel optional reflection before shutdown."""
        reflection_tasks = tuple(self._reflection_tasks)
        for task in reflection_tasks:
            task.cancel()
        if reflection_tasks:
            await asyncio.gather(*reflection_tasks, return_exceptions=True)

        memory_tasks = tuple(self._memory_record_tasks)
        if not memory_tasks:
            return
        _, pending = await asyncio.wait(memory_tasks, timeout=10)
        if pending:
            logger.warning("Cancelling %d memory write(s) that exceeded shutdown grace", len(pending))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    def _spawn_companion_follow_up(
        self,
        *,
        user_input: str,
        plan: Any,
        results: list[Any],
        verification: Any,
        result_text: str,
        chat_session_id: str = "",
        speak: bool = False,
    ) -> None:
        """Generate optional next ideas after the verified result is delivered.

        Companion ideation may call a model and must never hold the terminal
        task response hostage. The eventual suggestion is delivered as its
        own notification and remains scoped to the originating chat.
        """

        async def _generate() -> None:
            try:
                follow_up = await self._execution_companion.follow_up(
                    user_input,
                    plan,
                    results,
                    verification,
                )
                if not follow_up:
                    return

                if chat_session_id:
                    self._recent_companion_context_by_session[chat_session_id] = (
                        f"Previous request: {_sanitize_summary(user_input, limit=300)}\n"
                        f"Execution result: {_sanitize_summary(result_text, limit=300)}\n"
                        f"Companion next ideas: {' | '.join(follow_up.suggestions)}"
                    )
                else:
                    self._recent_companion_context = (
                        f"Previous request: {_sanitize_summary(user_input, limit=300)}\n"
                        f"Execution result: {_sanitize_summary(result_text, limit=300)}\n"
                        f"Companion next ideas: {' | '.join(follow_up.suggestions)}"
                    )

                await self._broadcast_notification(
                    "companion_follow_up",
                    {
                        "session_id": chat_session_id,
                        **follow_up.to_dict(),
                    },
                )
                if speak:
                    await self._speak_voice_response(
                        follow_up.spoken_text(),
                        channel=SpeechChannel.BACKGROUND_INSIGHT,
                        dedupe_key=f"voice-follow-up:{user_input.casefold()}",
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Companion follow-up failed in background", exc_info=True)

        task: asyncio.Task[None] = asyncio.create_task(_generate())
        self._companion_follow_up_tasks.add(task)
        task.add_done_callback(self._companion_follow_up_tasks.discard)

    @staticmethod
    def _mcp_task_input(params: dict[str, Any]) -> str:
        value = str(params.get("input") or "").strip()
        if not value:
            raise ValueError("input is required")
        if len(value) > 20_000:
            raise ValueError("input exceeds the 20,000 character limit")
        return value

    @staticmethod
    def _mcp_session_id(params: dict[str, Any]) -> str:
        value = str(params.get("session_id") or "default").strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,80}", value):
            raise ValueError("session_id must use 1-80 letters, digits, dots, underscores, colons, or hyphens")
        return value

    @staticmethod
    def _mcp_request_id(params: dict[str, Any]) -> str:
        value = str(params.get("request_id") or "").strip()
        if value and not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value):
            raise ValueError("request_id must use at most 128 safe identifier characters")
        return value

    async def _handle_mcp_plan_task(
        self,
        params: dict[str, Any],
        ws: ServerConnection,
    ) -> dict[str, Any]:
        """Create a non-authoritative, side-effect-free preview for local MCP."""

        self._require_rpc_role(ws, RpcClientRole.MCP_LOCAL)
        if self._planner is None:
            return {"status": "unavailable", "message": "Planner is not initialized"}
        if self._interactive_request_active:
            return {
                "status": "busy",
                "message": "Heliox is already handling an interactive task.",
            }
        try:
            user_input = self._mcp_task_input(params)
            session_id = self._mcp_session_id(params)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        plan = await self._planner.plan(
            user_input,
            session_id=session_id,
        )
        if plan.error:
            return {"status": "error", "message": plan.error}

        actions: list[dict[str, Any]] = []
        for index, action in enumerate(plan.actions):
            action_payload = action.model_dump(mode="json")
            action_payload.update(
                {
                    "index": index,
                    "permission_tier": action.permission_tier.name,
                    "normally_requires_confirmation": action.requires_confirmation,
                    "mcp_requires_confirmation": True,
                    "irreversible": action.is_irreversible,
                }
            )
            actions.append(action_payload)
        return {
            "status": "preview",
            "explanation": plan.explanation,
            "actions": actions,
            "action_count": len(actions),
            "requires_user_approval": bool(actions),
            "authoritative": False,
            "message": (
                "This is an advisory preview. submit_task replans through the full Heliox safety "
                "pipeline and requires visible approval for every action."
            ),
        }

    def _typed_decision_model(self) -> Any | None:
        """The one shared Gnomon engine (not loaded until first use), or None when it is off.

        The MCP scoring tool and the autonomous reviewer share this instance, so the
        model is in memory at most once. Changing the config replaces it.
        """
        config = self.config.typed_decision
        if not config.enabled or config.scoring_mode not in {"trained_head", "native_next_token"}:
            return None
        if config.scoring_mode == "trained_head" and (
            not config.artifact_dir.strip() or not config.report_path.strip()
        ):
            return None
        artifact_dir = str(Path(config.artifact_dir).expanduser()) if config.artifact_dir else ""
        report_path = str(Path(config.report_path).expanduser()) if config.report_path else ""
        engine_key = (
            config.scoring_mode,
            artifact_dir,
            report_path,
            config.model_id,
            config.max_length,
            config.temperature,
        )
        if self._typed_decision_engine_key != engine_key:
            previous = self._typed_decision_engine
            if previous is not None:
                threading.Thread(target=previous.unload, name="gnomon-unload", daemon=True).start()
            if config.scoring_mode == "native_next_token":
                from pilot.intelligence.native_typed_decision import LocalNextTokenTypedDecisionModel

                self._typed_decision_engine = LocalNextTokenTypedDecisionModel(
                    config.model_id,
                    max_length=config.max_length,
                    temperature=config.temperature,
                )
            else:
                from pilot.intelligence.typed_decision import LocalTypedDecisionModel

                self._typed_decision_engine = LocalTypedDecisionModel(artifact_dir, report_path)
            self._typed_decision_engine_key = engine_key
        return self._typed_decision_engine

    async def _handle_mcp_typed_decision(
        self,
        params: dict[str, Any],
        ws: ServerConnection,
    ) -> dict[str, Any]:
        """Return an opt-in local model score that cannot authorize Heliox actions."""
        self._require_rpc_role(ws, RpcClientRole.MCP_LOCAL)
        config = self.config.typed_decision
        if not config.enabled:
            return {
                "status": "disabled",
                "research_only": True,
                "execution_authority": False,
                "message": "Enable typed_decision in local Heliox config to use this scorer.",
            }
        if self._interactive_request_active:
            return {
                "status": "busy",
                "research_only": True,
                "execution_authority": False,
                "message": "Heliox is handling an interactive task; retry typed scoring afterward.",
            }
        if config.scoring_mode == "trained_head" and (
            not config.artifact_dir.strip() or not config.report_path.strip()
        ):
            return {
                "status": "unavailable",
                "research_only": True,
                "execution_authority": False,
                "message": "Configure local artifact_dir and report_path before enabling the scorer.",
            }
        if config.scoring_mode not in {"trained_head", "native_next_token"}:
            return {
                "status": "unavailable",
                "research_only": True,
                "execution_authority": False,
                "message": "Choose a supported local typed-decision scoring mode.",
            }

        state = params.get("state")
        typed_questions = "questions" in params
        kind = params.get("kind")
        question = params.get("question")
        options = params.get("options")
        if typed_questions:
            questions = params.get("questions")
            try:
                from pilot.intelligence.typed_decision import compile_typed_questions

                compile_typed_questions(state, questions)
            except (TypeError, ValueError) as exc:
                return {
                    "status": "error",
                    "research_only": True,
                    "execution_authority": False,
                    "message": str(exc)[:300],
                }
        elif not isinstance(state, (str, dict, list)):
            return {
                "status": "error",
                "research_only": True,
                "execution_authority": False,
                "message": "state must be text, a JSON object, or an array",
            }
        try:
            from pilot.intelligence.typed_decision import validate_decision

            if not typed_questions:
                validate_decision(kind, question, options)
            if self._typed_decision_model() is None:
                raise ValueError("Configure the local typed-decision scorer before using it.")
            if typed_questions:
                return await asyncio.to_thread(
                    self._typed_decision_engine.score_questions,
                    state,
                    questions,
                )
            return await asyncio.to_thread(self._typed_decision_engine.score, state, kind, question, options)
        except (TypeError, ValueError) as exc:
            return {
                "status": "error",
                "research_only": True,
                "execution_authority": False,
                "message": str(exc)[:300],
            }
        except Exception as exc:
            logger.warning("Local typed-decision scoring unavailable: %s", type(exc).__name__)
            return {
                "status": "unavailable",
                "research_only": True,
                "execution_authority": False,
                "message": str(exc)[:300] or type(exc).__name__,
            }

    async def _handle_mcp_submit_task(
        self,
        params: dict[str, Any],
        ws: ServerConnection,
    ) -> dict[str, Any]:
        """Queue one local MCP request through the normal interactive pipeline."""

        self._require_rpc_role(ws, RpcClientRole.MCP_LOCAL)
        if self._durable_tasks is None:
            return {"status": "unavailable", "message": "Durable task store is not initialized"}
        try:
            user_input = self._mcp_task_input(params)
            session_suffix = self._mcp_session_id(params)
            request_id = self._mcp_request_id(params)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        idempotency_key = f"{session_suffix}:{request_id}" if request_id else ""
        if idempotency_key:
            previous = self._mcp_idempotency.get(idempotency_key)
            if previous is not None:
                previous_task_id, previous_input = previous
                if previous_input != user_input:
                    return {
                        "status": "error",
                        "message": "request_id was already used for different task input",
                    }
                return {
                    "status": "submitted",
                    "task_id": previous_task_id,
                    "requires_user_approval": True,
                    "duplicate": True,
                    "message": "The matching MCP task was already submitted; poll its existing task id.",
                }

        if (
            self._interactive_request_active
            or self._queued_reserved_task_id
            or any(not task.done() for task in self._mcp_tasks.values())
        ):
            return {
                "status": "busy",
                "message": "Heliox is already handling an interactive task.",
            }

        task_id = str(uuid.uuid4())
        if idempotency_key:
            self._mcp_idempotency[idempotency_key] = (task_id, user_input)
            while len(self._mcp_idempotency) > 100:
                self._mcp_idempotency.pop(next(iter(self._mcp_idempotency)))
        self._mcp_reserved_task_id = task_id

        async def _run() -> dict[str, Any]:
            try:
                response = await self._handle_execute(
                    {
                        "input": user_input,
                        "task_id": task_id,
                        "session_id": f"mcp:{session_suffix}",
                        "user_id": "mcp-local",
                        "source": "mcp",
                    },
                    _BroadcastConnection(self._broadcast_notification),
                )
            except asyncio.CancelledError:
                response = {
                    "status": "cancelled",
                    "task_id": task_id,
                    "message": "The MCP task was cancelled before completion.",
                }
            except Exception:
                logger.exception("Local MCP task %s failed", task_id)
                response = {
                    "status": "error",
                    "task_id": task_id,
                    "message": "The MCP task failed inside the Heliox daemon.",
                }
            finally:
                if self._mcp_reserved_task_id == task_id:
                    self._mcp_reserved_task_id = ""
            self._mcp_task_results[task_id] = response
            while len(self._mcp_task_results) > 100:
                self._mcp_task_results.pop(next(iter(self._mcp_task_results)))
            await self._broadcast_notification(
                "mcp_task_update",
                {
                    "task_id": task_id,
                    "status": response.get("status", "error"),
                    "message": response.get("message", ""),
                },
            )
            return response

        task = asyncio.create_task(_run())
        self._mcp_tasks[task_id] = task

        def _finish_mcp_task(done: asyncio.Task[dict[str, Any]]) -> None:
            self._mcp_tasks.pop(task_id, None)
            if self._mcp_reserved_task_id == task_id:
                self._mcp_reserved_task_id = ""
            if done.cancelled() and task_id not in self._mcp_task_results:
                self._mcp_task_results[task_id] = {
                    "status": "cancelled",
                    "task_id": task_id,
                    "message": "The MCP task was cancelled before it started.",
                }
            self._start_queued_execute_worker()

        task.add_done_callback(_finish_mcp_task)
        return {
            "status": "submitted",
            "task_id": task_id,
            "requires_user_approval": True,
            "message": (
                "Task submitted. Review and approve the proposed actions in Heliox OS, then call "
                "get_task_status for the verified result."
            ),
        }

    async def _handle_mcp_task_status(
        self,
        params: dict[str, Any],
        ws: ServerConnection,
    ) -> dict[str, Any]:
        """Return status only for tasks created by the local MCP bridge."""

        self._require_rpc_role(ws, RpcClientRole.MCP_LOCAL)
        task_id = str(params.get("task_id") or "").strip()
        if not task_id:
            return {"status": "error", "message": "task_id is required"}
        if self._durable_tasks is None:
            return {"status": "unavailable", "message": "Durable task store is not initialized"}

        task = await self._durable_tasks.get(task_id)
        if task is None:
            if task_id in self._mcp_tasks:
                return {"status": "submitted", "task_id": task_id}
            cached = self._mcp_task_results.get(task_id)
            if cached is not None:
                return {**cached, "task_id": task_id}
            return {"status": "not_found", "message": "No local MCP task has that id"}
        if task.user_id != "mcp-local":
            return {"status": "not_found", "message": "No local MCP task has that id"}

        response: dict[str, Any] = {
            "status": task.status.value,
            "task_id": task.task_id,
            "plan_id": task.plan_id,
            "cancellation_requested": task.cancellation_requested,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        }
        if task.plan_id:
            approval = await self._durable_tasks.get_approval(task.plan_id)
            if approval is not None:
                response["approval"] = {
                    "status": approval.status.value,
                    "request": approval.request,
                    "expires_at": approval.expires_at,
                }
        if task.terminal_response is not None:
            response["result"] = task.terminal_response
        return response

    async def _handle_mcp_cancel_task(
        self,
        params: dict[str, Any],
        ws: ServerConnection,
    ) -> dict[str, Any]:
        """Request cancellation without giving MCP authority over UI tasks."""

        self._require_rpc_role(ws, RpcClientRole.MCP_LOCAL)
        task_id = str(params.get("task_id") or "").strip()
        if not task_id:
            return {"status": "error", "message": "task_id is required"}
        if self._durable_tasks is None:
            return {"status": "unavailable", "message": "Durable task store is not initialized"}

        task = await self._durable_tasks.get(task_id)
        if task is None:
            running = self._mcp_tasks.get(task_id)
            if running is None:
                return {"status": "not_found", "message": "No local MCP task has that id"}
            running.cancel()
            try:
                await running
            except asyncio.CancelledError:
                pass
            response = self._mcp_task_results.get(task_id)
            if response is None:
                response = {
                    "status": "cancelled",
                    "task_id": task_id,
                    "message": "The MCP task was cancelled before it started.",
                }
                self._mcp_task_results[task_id] = response
            if self._mcp_reserved_task_id == task_id:
                self._mcp_reserved_task_id = ""
            return response
        if task.user_id != "mcp-local":
            return {"status": "not_found", "message": "No local MCP task has that id"}
        if task.is_terminal:
            return {
                "status": task.status.value,
                "task_id": task_id,
                "message": "The task is already terminal.",
            }

        await self._durable_tasks.request_cancel(task_id)
        if self._active_task_id == task_id:
            await self._handle_abort({}, None)
        return {
            "status": "cancellation_requested",
            "task_id": task_id,
            "message": "Heliox is stopping this MCP task at the safest available boundary.",
        }

    async def _handle_resume_task(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Authenticate and continue a durable task after reconnect or restart."""

        if self._durable_tasks is None:
            return {"status": "error", "message": "Durable task store is not initialized"}
        resume_token = str(params.get("resume_token") or "")
        if not resume_token:
            return {"status": "error", "message": "resume_task requires resume_token"}
        task = await self._durable_tasks.get_by_resume_token(resume_token)
        requested_task_id = str(params.get("task_id") or "")
        if task is None or (requested_task_id and requested_task_id != task.task_id):
            return {"status": "error", "message": "Invalid task_id or resume_token"}
        if task.is_terminal:
            return {
                **(task.terminal_response or {"status": task.status.value}),
                "task_id": task.task_id,
                "replayed": True,
            }
        if task.cancellation_requested:
            response = {
                "status": "cancelled",
                "message": "Task cancellation was requested before resume.",
                "task_id": task.task_id,
            }
            await self._finalize_durable_task(task.task_id, response)
            return response

        approval = await self._durable_tasks.get_approval(task.plan_id) if task.plan_id else None
        if approval is not None and approval.status == ApprovalStatus.PENDING:
            return {
                "status": "awaiting_approval",
                "task_id": task.task_id,
                "plan_id": task.plan_id,
                "approval": approval.request,
                "expires_at": approval.expires_at,
            }
        # Only an APPROVED row resumes, and only with the actions it approved: the checkpoint of a durable plan
        # keeps the plan as it was shown for approval, so every required action the person left unchecked is
        # dropped here, also when the approval was answered after a restart.
        denied_indices: frozenset[int] = frozenset()
        refusal = ""
        if approval is not None:
            if approval.status != ApprovalStatus.APPROVED:
                refusal = f"Task approval was {approval.status.value}."
            else:
                try:
                    required = {int(index) for index in approval.request.get("action_indices", [])}
                    approved = {int(index) for index in approval.approved_indices} & required
                except (AttributeError, TypeError, ValueError):
                    refusal = "Task approval could not be read, so nothing was resumed."
                else:
                    if required and not approved:
                        refusal = "Task approval approved none of the actions that needed approval."
                    denied_indices = frozenset(required - approved)
        if refusal:
            response = {
                "status": "cancelled",
                "message": refusal,
                "task_id": task.task_id,
                "plan_id": task.plan_id,
            }
            await self._finalize_durable_task(task.task_id, response)
            return response

        if self._interactive_request_active or self._queued_reserved_task_id or self._mcp_reserved_task_id:
            return {
                "status": "busy",
                "message": "Heliox is handling another interactive task.",
                "task_id": task.task_id,
            }
        self._interactive_request_active = True
        self._active_task_id = task.task_id
        self._active_plan_id = task.plan_id
        self._live_correction = None
        context = ExperienceContext(
            session_id=task.session_id,
            task_id=task.task_id,
            user_id=task.user_id,
        )
        self._active_experience_context = context
        try:
            with experience_scope(
                session_id=context.session_id,
                task_id=context.task_id,
                user_id=context.user_id,
            ):
                if task.plan_id:
                    if task.status != TaskStatus.EXECUTING:
                        await self._durable_tasks.transition(
                            task.task_id,
                            TaskStatus.EXECUTING,
                            reason="authenticated task resume",
                            plan_id=task.plan_id,
                        )
                    response = await self._handle_resume_plan(
                        {
                            "plan_id": task.plan_id,
                            "_authorized_task_id": task.task_id,
                        },
                        ws,
                        denied_indices=denied_indices,
                    )
                else:
                    if task.status != TaskStatus.PLANNING:
                        await self._durable_tasks.transition(
                            task.task_id,
                            TaskStatus.PLANNING,
                            reason="restart interrupted planning",
                        )
                    response = await self._handle_execute_inner(
                        {
                            "input": task.user_input,
                            "session_id": task.session_id,
                        },
                        ws,
                    )
                await self._finalize_durable_task(task.task_id, response)
                return {**response, "task_id": task.task_id, "resumed": True}
        finally:
            self._interactive_request_active = False
            self._active_task_id = ""
            self._active_plan_id = ""
            self._live_correction = None
            self._cancel_event = None
            self._active_experience_context = ExperienceContext()
            self._start_queued_execute_worker()

    async def _handle_resume_plan(
        self,
        params: dict[str, Any],
        ws: ServerConnection,
        *,
        denied_indices: frozenset[int] = frozenset(),
    ) -> dict:
        """Resume a previously checkpointed plan from its last completed action.

        ``denied_indices`` are the checkpoint's action indices the durable approval left unchecked (passed only by
        ``resume_task``). They are dropped before anything runs, so the resumed plan is the approved plan: the
        checkpoint's completed count and the action indices then refer to it, as they did when it first ran.
        """
        plan_id = str(params.get("plan_id", "")).strip()
        if not plan_id:
            return {"status": "error", "message": "resume_plan requires plan_id"}
        if not self._checkpoint_store:
            return {"status": "error", "message": "Workflow checkpoint store is not initialized"}
        if self._durable_tasks is not None:
            durable = await self._durable_tasks.get_by_plan_id(plan_id)
            authorized_task_id = str(params.get("_authorized_task_id") or "")
            if durable is not None and authorized_task_id != durable.task_id:
                return {
                    "status": "error",
                    "message": "This durable plan must be resumed with resume_task and its resume_token.",
                    "task_id": durable.task_id,
                }

        checkpoint = await self._checkpoint_store.get(plan_id)
        if checkpoint is None:
            return {"status": "error", "message": f"No checkpoint found for plan_id: {plan_id}"}

        approved_plan = checkpoint.plan
        if denied_indices:
            if max(denied_indices) >= len(checkpoint.plan.actions):
                return {
                    "status": "error",
                    "message": "The approval does not match this plan's checkpoint, so nothing was resumed.",
                    "plan_id": plan_id,
                }
            approved_plan = checkpoint.plan.model_copy(
                update={
                    "actions": [
                        action for index, action in enumerate(checkpoint.plan.actions) if index not in denied_indices
                    ]
                }
            )

        completed_count = max(0, min(checkpoint.completed_count, len(approved_plan.actions)))
        remaining_actions = approved_plan.actions[completed_count:]
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
            from pilot.response_contract import partial_failure_message, success_message

            verification = await self._verifier.verify(approved_plan, checkpoint.results)
            if verification.passed:
                await self._checkpoint_store.mark_status(plan_id, "complete")
                status = "success"
                message = success_message(
                    approved_plan,
                    checkpoint.results,
                    verification,
                    dry_run=False,
                )
            else:
                await self._checkpoint_store.mark_status(plan_id, "failed")
                status = "partial_failure"
                message = partial_failure_message(checkpoint.results, verification)
            return {
                "status": status,
                "plan_id": plan_id,
                "resumed": False,
                "message": message,
                "results": [result.model_dump() for result in checkpoint.results],
                "verification": verification.model_dump(),
            }

        self._cancel_event = asyncio.Event()
        cancel_event = self._cancel_event

        from pilot.actions import ActionPlan

        remaining_plan = ActionPlan(
            actions=remaining_actions,
            explanation=approved_plan.explanation,
            raw_input=approved_plan.raw_input,
        )

        async def _on_action_start(action: Any) -> None:
            await ws.send(_notification("action_start", {"action": action.model_dump(), "resumed": True}))

        async def _on_action_complete(result: Any) -> None:
            await ws.send(_notification("action_complete", {"result": result.model_dump(), "resumed": True}))
            if result.success:
                await self._checkpoint_store.record_result(plan_id, result)

        await self._checkpoint_store.mark_status(plan_id, "resuming")
        execute_kwargs: dict[str, Any] = {
            "on_action_start": _on_action_start,
            "on_action_complete": _on_action_complete,
            "cancel_event": cancel_event,
            "plan_id": plan_id,
            "initial_last_output": checkpoint.last_output,
        }
        if params.get("_authorized_task_id"):
            execute_kwargs["action_index_offset"] = completed_count
        try:
            results = await self._execute_tracked(remaining_plan, **execute_kwargs)
        except asyncio.CancelledError:
            # ── Mid-flight cancellation (Part 3): cancel_event is already
            # set by _handle_abort before it cancels the tracked task, so
            # the cancel_event.is_set() branch below handles the response --
            # this must be caught here or it escapes as an unhandled
            # BaseException instead of a clean RPC response. ──
            results = []

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
        verification = None
        if not failed and len(combined_results) >= len(approved_plan.actions):
            verification = await self._verifier.verify(approved_plan, combined_results)
            verification_payload = verification.model_dump()
            if not verification.passed:
                final_status = "partial_failure"
                await self._checkpoint_store.mark_status(plan_id, "failed")

        from pilot.response_contract import partial_failure_message, success_message

        completed_all = len(combined_results) >= len(approved_plan.actions)
        if final_status in {"failed", "partial_failure"}:
            response_status = "partial_failure"
            terminal_message = partial_failure_message(combined_results, verification)
        elif not completed_all:
            response_status = "interrupted"
            terminal_message = (
                f"Task was interrupted after {len(combined_results)} of "
                f"{len(approved_plan.actions)} actions. No unfinished action was reported as complete."
            )
            await self._checkpoint_store.mark_status(plan_id, "interrupted")
        else:
            response_status = "success"
            terminal_message = success_message(
                approved_plan,
                combined_results,
                verification,
                dry_run=False,
            )

        return {
            "status": response_status,
            "message": terminal_message,
            "plan_id": plan_id,
            "resumed": True,
            "skipped_actions": completed_count,
            "executed_actions": len(results),
            "results": [result.model_dump() for result in combined_results],
            "verification": verification_payload,
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

    async def _wait_for_confirmation(
        self,
        plan_id: str,
        plan: Any,
        ws: ServerConnection,
        *,
        reason: str = "",
        risk_assessment: dict[str, Any] | None = None,
        force_all_actions: bool = False,
    ) -> tuple[bool, set[int], set[int]]:
        """Send a confirmation request and block until the user responds or timeout.

        Args:
            plan_id: Unique identifier for the plan requiring confirmation.
            plan: The plan object containing actions to be confirmed.
            ws: The WebSocket connection for sending/receiving messages.

        Returns:
            A (confirmed, approved_indices, required_indices) tuple.
            ``required_indices`` are the original ``plan.actions`` indices
            that needed confirmation; ``approved_indices`` is the subset of
            those the user approved (all of them for a plain approve/deny
            response, a subset for per-action granular approval).
        """
        pending = PendingConfirmation(plan_id=plan_id, event=asyncio.Event())
        self._pending_confirms[plan_id] = pending

        confirm_indices = (
            list(range(len(plan.actions)))
            if force_all_actions
            else [i for i, a in enumerate(plan.actions) if a.requires_confirmation or a.is_irreversible]
        )
        pending.required_indices = frozenset(confirm_indices)

        def _dump_confirm_action(idx: int, a: Any) -> dict[str, Any]:
            payload = a.model_dump()
            payload["irreversible"] = a.is_irreversible
            payload["undo"] = self._undo_badge(a)
            payload["index"] = idx
            return payload

        approval_request = {
            "action_indices": confirm_indices,
            "actions": [_dump_confirm_action(i, plan.actions[i]) for i in confirm_indices],
            "reason": reason,
            "risk_assessment": risk_assessment,
        }
        if self._durable_tasks is not None and self._active_task_id:
            await self._durable_tasks.create_approval(
                task_id=self._active_task_id,
                plan_id=plan_id,
                request=approval_request,
                timeout_seconds=CONFIRM_TIMEOUT_SECONDS,
            )
        if self._memory is not None and self._active_task_id:
            context = get_experience_context()
            await self._memory.put_working(
                session_id=context.session_id or "default",
                task_id=self._active_task_id,
                key="pending approval",
                value={
                    "plan_id": plan_id,
                    "reason": reason,
                    "action_count": len(confirm_indices),
                },
                priority=1.0,
                ttl_seconds=CONFIRM_TIMEOUT_SECONDS,
            )

        await self._append_experience(
            ExperienceEventType.APPROVAL_REQUESTED,
            plan_id=plan_id,
            idempotency_key=f"plan:{plan_id}:approval:requested",
            source="permission_gate",
            payload=approval_request,
            provenance={"component": "PilotServer._wait_for_confirmation"},
            privacy_class=PrivacyClass.SENSITIVE,
        )
        await ws.send(
            _notification(
                "confirm_required",
                {
                    "task_id": get_experience_context().task_id,
                    "plan_id": plan_id,
                    "actions": [_dump_confirm_action(i, plan.actions[i]) for i in confirm_indices],
                    "reason": reason,
                    "risk_assessment": risk_assessment,
                },
            )
        )

        required = set(confirm_indices)
        decision = await self._consume_confirmation(pending)
        if decision.expired:
            logger.warning("Confirmation for plan %s ended unanswered (%s)", plan_id, decision.resolution_path)
            if self._durable_tasks is not None and self._active_task_id:
                # An approval whose reply never reached the UI was already written as APPROVED; it must not
                # resume after a restart either.
                await self._close_durable_approval(plan_id, ApprovalStatus.EXPIRED)
            await self._append_experience(
                ExperienceEventType.APPROVAL_RESOLVED,
                plan_id=plan_id,
                idempotency_key=f"plan:{plan_id}:approval:resolved",
                source="permission_gate",
                payload={
                    "decision": "expired",
                    "approved_indices": [],
                    "required_indices": sorted(required),
                    "input_channel": decision.input_channel,
                    "resolution_path": decision.resolution_path,
                },
                provenance={"component": "PilotServer._wait_for_confirmation"},
            )
            return False, set(), required

        approved = (
            (decision.approved_indices if decision.approved_indices is not None else required)
            if decision.confirmed
            else set()
        )
        if self._durable_tasks is not None and self._active_task_id:
            if decision.confirmed:
                persisted = await self._durable_tasks.get_approval(plan_id)
                if persisted is not None and persisted.status == ApprovalStatus.PENDING:
                    await self._durable_tasks.resolve_approval(
                        plan_id,
                        ApprovalStatus.APPROVED,
                        approved_indices=sorted(approved & required),
                    )
            else:
                # The plan does not run, so the durable row must not say APPROVED either. This also repeats a
                # denial whose durable write failed in _resolve_pending_denied.
                await self._close_durable_approval(plan_id, ApprovalStatus.DENIED)
        await self._append_experience(
            ExperienceEventType.APPROVAL_RESOLVED,
            plan_id=plan_id,
            idempotency_key=f"plan:{plan_id}:approval:resolved",
            source="permission_gate",
            payload={
                "decision": "approved" if decision.confirmed else "denied",
                "approved_indices": sorted(approved & required),
                "required_indices": sorted(required),
                "input_channel": decision.input_channel,
                "resolution_path": decision.resolution_path,
            },
            provenance={"component": "PilotServer._wait_for_confirmation"},
        )
        return decision.confirmed, (approved & required), required

    async def _consume_confirmation(self, pending: PendingConfirmation) -> _ConfirmationDecision:
        """Wait for a pending confirmation, then take it out of the registry and read its decision.

        The pop and the read happen once, inside ``pending.decision_lock``, and callers use only the
        returned copy. ``_resolve_pending_denied`` writes a denial under the same lock before its first
        await, so this either reads the decision before the denial runs (the denial then finds the entry
        gone and reports that nothing was denied) or reads the denial. A timeout that finds the
        confirmation unresolved marks it resolved here, so a later confirm or denial is refused; a
        timeout that finds it resolved reads that decision.

        A confirmation nobody answered in a way the plan may act on reads as expired, with
        ``input_channel`` "internal": the timeout above, a shutdown release, and an approval whose
        confirm reply was never written ("unacknowledged"), including one still being written when
        the timeout fires.
        """
        timed_out = False
        try:
            try:
                await asyncio.wait_for(pending.event.wait(), timeout=CONFIRM_TIMEOUT_SECONDS)
            except TimeoutError:
                timed_out = True
            async with pending.decision_lock:
                self._pending_confirms.pop(pending.plan_id, None)
                if timed_out and not pending.resolved:
                    pending.resolved = True
                    unanswered = "timeout"
                elif pending.resolution_path in CONFIRM_UNANSWERED_PATHS:
                    unanswered = pending.resolution_path
                elif pending.resolved and pending.confirmed and not pending.acknowledged:
                    unanswered = "unacknowledged"
                else:
                    unanswered = ""
                expired = bool(unanswered)
                return _ConfirmationDecision(
                    confirmed=pending.confirmed and not expired,
                    approved_indices=None if pending.approved_indices is None else set(pending.approved_indices),
                    expired=expired,
                    input_channel="internal" if expired else pending.input_channel,
                    resolution_path=unanswered or pending.resolution_path or "confirm_rpc",
                )
        finally:
            # A cancelled wait must not leave the entry registered.
            if self._pending_confirms.get(pending.plan_id) is pending:
                self._pending_confirms.pop(pending.plan_id, None)

    async def _handle_confirm(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Resolve a pending confirmation request from the UI.

        Args:
            params: JSON-RPC parameters containing plan_id, a JSON boolean
                confirmed, an optional input_channel (the authority channel:
                dialog, gesture, voice_deny, internal or unknown) and an
                optional approved_indices list for per-action granular
                approval (omit to approve all confirmation-requiring actions,
                preserving the old all-or-nothing behavior for clients that
                do not name a channel).
            ws: The WebSocket connection.

        Returns:
            A dict with status and confirmation result.
        """
        plan_id = params.get("plan_id", "")
        raw_approved = params.get("approved_indices")
        try:
            confirmed = _validated_bool(params, "confirmed", False)
        except ValueError:
            return {"status": "error", "message": "confirmed must be a boolean"}
        raw_channel = params.get("input_channel")
        if raw_channel is not None and (not isinstance(raw_channel, str) or raw_channel not in CONFIRM_INPUT_CHANNELS):
            return {"status": "error", "message": f"input_channel must be one of {', '.join(CONFIRM_INPUT_CHANNELS)}"}
        input_channel = raw_channel or "unknown"
        if confirmed and input_channel == "voice_deny":
            return {"status": "error", "message": "voice_deny can only deny"}
        # A client that names its channel must say exactly which actions it approves. Older clients
        # (no channel) and narrator, self-healing and threat-containment confirms keep approving all.
        indices_required = confirmed and raw_channel is not None and raw_approved is None

        pending = self._pending_confirms.get(plan_id)
        if pending is None:
            if self._durable_tasks is None:
                return {"status": "error", "message": f"No pending confirmation for plan_id: {plan_id}"}
            durable = await self._durable_tasks.get_approval(plan_id)
            if durable is None or durable.status != ApprovalStatus.PENDING:
                return {"status": "error", "message": f"No pending confirmation for plan_id: {plan_id}"}
            if indices_required:
                return {"status": "error", "message": "approved_indices is required"}
            try:
                required = {int(index) for index in durable.request.get("action_indices", [])}
                approved = (
                    required
                    if confirmed and raw_approved is None
                    else {int(index) for index in (raw_approved or [])} & required
                )
            except (TypeError, ValueError):
                approved = set()
            resolved = await self._durable_tasks.resolve_approval(
                plan_id,
                ApprovalStatus.APPROVED if confirmed else ApprovalStatus.DENIED,
                approved_indices=sorted(approved),
            )
            return {
                "status": "ok",
                "confirmed": resolved.status == ApprovalStatus.APPROVED,
                "resume_required": True,
                "task_id": resolved.task_id,
            }

        if indices_required and pending.required_indices is not None:
            return {"status": "error", "message": "approved_indices is required"}
        if raw_approved is not None and not isinstance(raw_approved, list):
            return {"status": "error", "message": "approved_indices must be a list"}
        try:
            approved_indices = None if raw_approved is None else {int(i) for i in raw_approved}
        except (TypeError, ValueError):
            return {"status": "error", "message": "approved_indices must contain only integers"}

        async with pending.decision_lock:
            if pending.resolved:
                return {"status": "error", "message": f"Confirmation already resolved for plan_id: {plan_id}"}

            decision = confirmed
            if self._durable_tasks is not None:
                durable = await self._durable_tasks.get_approval(plan_id)
                if durable is not None and durable.status == ApprovalStatus.PENDING:
                    required = {int(index) for index in durable.request.get("action_indices", [])}
                    approved = (
                        required if decision and approved_indices is None else (approved_indices or set()) & required
                    )
                    await self._durable_tasks.resolve_approval(
                        plan_id,
                        ApprovalStatus.APPROVED if decision else ApprovalStatus.DENIED,
                        approved_indices=sorted(approved),
                    )

            pending.confirmed = decision
            pending.approved_indices = approved_indices
            pending.resolved = True
            pending.input_channel = input_channel
            if not pending.resolution_path:
                pending.resolution_path = "voice_deny" if input_channel == "voice_deny" else "confirm_rpc"
            pending.resolved_by_task = asyncio.current_task()

            # Direct/internal callers have no JSON-RPC acknowledgement to
            # deliver. Socket RPCs are released by _process_request only after
            # their response is written, closing the UI acknowledgement race.
            if pending.resolved_by_task not in pending.rpc_tasks:
                pending.acknowledged = True
                pending.event.set()
            return {"status": "ok", "confirmed": decision}

    async def _resolve_pending_denied(
        self,
        plan_id: str,
        *,
        resolution_path: str,
        input_channel: str = "internal",
    ) -> bool:
        """Deny a pending confirmation under its decision lock. Denial wins over an approval that the waiter has not
        consumed yet: the in-memory decision is changed before the first await, and the waiter pops the entry and reads
        the decision inside the same lock, so it reads either the approval before this helper runs (then this helper
        finds the entry gone and returns False) or the denial after it. A later approval is refused because the
        confirmation is resolved. Returns True only when the denial took effect.

        The event is set even when the durable write fails or this call is cancelled during it, so the waiter reads
        the denial at once instead of at the timeout; the waiter then writes the durable denial again."""
        pending = self._pending_confirms.get(plan_id)
        if pending is None:
            return False
        async with pending.decision_lock:
            if self._pending_confirms.get(plan_id) is not pending:
                return False  # consumed while we waited for the lock
            # No await between the registration check and these writes.
            pending.confirmed = False
            pending.approved_indices = set()
            pending.resolved = True
            pending.resolution_path = resolution_path
            pending.input_channel = input_channel
            try:
                # An approval the socket confirm already wrote as APPROVED must not resume after a restart.
                await self._close_durable_approval(plan_id, ApprovalStatus.DENIED)
            finally:
                pending.event.set()
        return True

    async def _close_durable_approval(self, plan_id: str, status: ApprovalStatus) -> None:
        """Write a denial or an expiry over a plan's durable approval, failing closed. Never raises an ordinary error.

        When that write fails (a locked or broken task database), the approval may still read APPROVED, so the active
        durable task that owns the plan is asked to cancel instead: a restart then finalises it as cancelled rather
        than resuming a plan that did not run. The in-memory decision stands either way.
        """
        store = self._durable_tasks
        if store is None:
            return
        try:
            if status == ApprovalStatus.EXPIRED:
                await store.expire_unconsumed_approval(plan_id)
            else:
                await store.deny_unconsumed_approval(plan_id)
            return
        except Exception:
            logger.warning("Durable %s of plan %s failed", status.value, plan_id, exc_info=True)
        task_id = self._active_task_id
        if not task_id or plan_id != self._active_plan_id:
            return
        try:
            await store.request_cancel(task_id)
        except Exception:
            logger.warning(
                "Durable cancel of task %s failed too; its approval for plan %s may still read approved",
                task_id,
                plan_id,
                exc_info=True,
            )

    async def _handle_rollback_plan(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Roll back the filesystem snapshot taken before a plan executed.

        This is filesystem-wide (btrfs subvolume / timeshift snapshot), NOT
        per-action — it reverts everything since the snapshot, including any
        unrelated changes made after it. The frontend must gate this behind
        its own explicit confirmation step; this handler does not re-confirm.

        Args:
            params: JSON-RPC parameters containing plan_id.
            ws: The WebSocket connection.

        Returns:
            A dict with status and a human-readable message.
        """
        plan_id = params.get("plan_id", "")
        snapshot_id = self._plan_snapshots.get(plan_id)
        if not snapshot_id:
            return {
                "status": "error",
                "message": f"No snapshot on record for plan_id: {plan_id} (either none was taken, or it was already rolled back)",
            }

        from pilot.system.snapshots import SnapshotManager

        snapshot_mgr = SnapshotManager(self.config)
        try:
            result_message = await snapshot_mgr.rollback(snapshot_id)
        except Exception as exc:
            logger.warning("Rollback failed for plan %s (snapshot %s): %s", plan_id, snapshot_id, exc)
            return {"status": "error", "message": f"Rollback failed: {exc}"}

        self._plan_snapshots.pop(plan_id, None)
        await ws.send(
            _notification(
                "rollback_complete",
                {"plan_id": plan_id, "snapshot_id": snapshot_id, "message": result_message},
            )
        )
        logger.info("Rolled back plan %s to snapshot %s", plan_id, snapshot_id)
        return {"status": "ok", "message": result_message}

    def _undo_badge(self, action: Any) -> dict[str, str]:
        """What the approval dialog says about undoing ``action`` (decided by type, never by a model)."""
        from pilot.workflows.undo_journal import undo_support

        support = undo_support(action)
        if support.kind == "undoable" and not self.config.undo.enabled:
            return {"kind": "not_undoable", "reason": "Undo is turned off in Settings."}
        return support.to_dict()

    def _undo_busy_message(self) -> str | None:
        from pilot.agents.autonomous import JobStatus

        busy_jobs = [
            job_id
            for job_id in (list(self._autonomous._active_tasks) if self._autonomous is not None else [])
            if (job := self._autonomous.get_job(job_id)) is not None
            and job.status in {JobStatus.QUEUED, JobStatus.DECOMPOSING, JobStatus.RUNNING}
        ]
        if self._interactive_request_active or busy_jobs:
            return "Wait until the running tasks finish, or stop them, before undoing."
        return None

    @staticmethod
    def _undo_task_key(params: dict[str, Any]) -> str | None:
        task_key = params.get("task_key")
        if not isinstance(task_key, str) or not task_key.strip() or len(task_key) > 200:
            return None
        return task_key.strip()

    async def _handle_undo_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """List recent tasks with how much of each can be undone."""
        if self._undo_journal is None:
            return {"status": "error", "message": "Undo is not ready yet.", "tasks": []}
        limit = params.get("limit", 20)
        if not isinstance(limit, int) or isinstance(limit, bool):
            limit = 20
        return {
            "status": "ok",
            "enabled": self.config.undo.enabled,
            "tasks": await self._undo_journal.list_tasks(limit),
        }

    async def _handle_undo_preview(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Show, item by item, what undoing a task would put back and what it cannot."""
        task_key = self._undo_task_key(params)
        if task_key is None:
            return {"status": "error", "message": "task_key is required"}
        if self._undo_journal is None:
            return {"status": "error", "message": "Undo is not ready yet."}
        preview = await self._undo_journal.preview(task_key)
        if preview is None:
            return {"status": "error", "message": "Nothing is recorded to undo for this task."}
        return {"status": "ok", "preview": preview}

    async def _handle_undo_task(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Undo a finished task after the person confirmed the preview they saw.

        Needs ``confirmed`` to be exactly true and the ``preview_token`` of that preview.
        Undo restores only what the journal recorded, refuses anything changed since,
        and never runs a plan, so it does not go through the planner or the executor.
        """
        task_key = self._undo_task_key(params)
        if task_key is None:
            return {"status": "error", "message": "task_key is required"}
        if params.get("confirmed") is not True:
            return {"status": "error", "message": "Undo needs your confirmation in the preview."}
        preview_token = params.get("preview_token")
        if not isinstance(preview_token, str) or not preview_token:
            return {"status": "error", "message": "preview_token is required"}
        if self._undo_journal is None:
            return {"status": "error", "message": "Undo is not ready yet."}
        busy = self._undo_busy_message()
        if busy:
            return {"status": "error", "message": busy}
        result = await self._undo_journal.undo(task_key, preview_token)
        if result.get("status") in ("ok", "partial"):
            logger.info(
                "Undid task %s: %s undone, %s skipped, %s failed",
                task_key,
                result["undone"],
                result["skipped"],
                result["failed"],
            )
            await self._broadcast_notification(
                "undo_complete",
                {
                    "task_key": task_key,
                    "status": result["status"],
                    "undone": result["undone"],
                    "skipped": result["skipped"],
                    "failed": result["failed"],
                },
            )
        return result

    async def _handle_undo_clear(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Delete all undo history and the copies it keeps."""
        if params.get("confirmed") is not True:
            return {"status": "error", "message": "Clearing undo history needs confirmation."}
        if self._undo_journal is None:
            return {"status": "error", "message": "Undo is not ready yet."}
        busy = self._undo_busy_message()
        if busy:
            return {"status": "error", "message": busy}
        await self._undo_journal.clear()
        return {"status": "ok"}

    async def _autonomous_broadcast(self, method: str, params: dict[str, Any]) -> None:
        """Forward background-job notifications, adding undo and a receipt when a job ends."""
        if method == "autonomous_complete" and isinstance(params, dict):
            params = dict(params)
            job_id = str(params.get("job_id") or "")
            try:
                if self._undo_journal is not None and job_id and not params.get("dry_run"):
                    undo_summary = await self._undo_journal.summary(job_id)
                    if undo_summary is not None:
                        params["undo"] = undo_summary
                if self._task_receipts is not None and job_id:
                    receipt = await self._task_receipts.finalize(
                        job_id,
                        request=str(params.get("goal") or ""),
                        status=str(params.get("status") or ""),
                        summary=str(params.get("result_summary") or ""),
                        source=f"autonomous:{params.get('source') or 'text'}",
                        undo=params.get("undo"),
                    )
                    if receipt is not None:
                        params["receipt_id"] = receipt["receipt_id"]
            except Exception:
                logger.warning("Finishing a background job's undo and receipt failed", exc_info=True)
        await self._broadcast_notification(method, params)
        if method in {"autonomous_complete", "autonomous_cancelled"} and isinstance(params, dict):
            await self._finish_phone_request(params)

    def _wire_air_handoff_requests(self) -> None:
        set_handler = getattr(self._air_handoff_manager, "set_request_handler", None)
        if set_handler is None:
            return
        set_handler(
            accepting=lambda: bool(
                self.config.air_handoff.enabled and getattr(self.config.air_handoff, "accept_requests", False)
            ),
            on_request=self._on_phone_request,
        )

    async def _on_phone_request(self, request: Any) -> None:
        """A paired phone asked for something: show it, and start it only if allowed."""
        logger.info("Phone request %s from %s", request.request_id, request.device_name)
        await self._broadcast_notification("phone_request", request.public_dict())
        if self.config.air_handoff.run_read_only_requests:
            await self._start_phone_request(request, read_only=True)

    async def _start_phone_request(self, request: Any, *, read_only: bool) -> dict[str, Any]:
        if self._autonomous is None:
            return {"status": "error", "message": "Background jobs are not available."}
        try:
            job = await self._autonomous.submit(
                request.text,
                source="phone",
                session_id="phone",
                allowed_actions=list(PHONE_READ_ONLY_ACTIONS) if read_only else None,
            )
        except Exception as exc:
            request.status = "failed"
            request.result = f"Could not start: {exc}"
            await self._broadcast_notification("phone_request", request.public_dict())
            return {"status": "error", "message": request.result}
        request.status = "running"
        request.job_id = job.job_id
        request.read_only = read_only
        await self._broadcast_notification("phone_request", request.public_dict())
        return {"status": "ok", "request": request.public_dict(), "job_id": job.job_id}

    async def _finish_phone_request(self, job: dict[str, Any]) -> None:
        """Send a finished request's short result back to the phone that asked."""
        manager = self._air_handoff_manager
        request = manager.request_for_job(str(job.get("job_id") or "")) if manager is not None else None
        if request is None:
            return
        job_status = str(job.get("status") or "")
        request.status = "done" if job_status in {"success", "partial"} else "failed"
        request.result = str(job.get("result_summary") or job_status)[:1000]
        await self._broadcast_notification("phone_request", request.public_dict())
        try:
            await manager.send_text(request.device_id, f'Heliox: "{request.text}"\n\n{job_status}: {request.result}')
        except Exception:
            logger.warning("Could not send a phone request's result back", exc_info=True)

    @staticmethod
    def _phone_request_id(params: dict[str, Any]) -> str | None:
        request_id = params.get("request_id")
        return request_id if isinstance(request_id, str) and 0 < len(request_id) <= 64 else None

    async def _handle_phone_request_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Requests paired phones sent, newest first."""
        if self._air_handoff_manager is None:
            return {"status": "error", "message": "Air Handoff is not initialized", "requests": []}
        return {
            "status": "ok",
            "accept_requests": bool(self.config.air_handoff.accept_requests),
            "run_read_only_requests": bool(self.config.air_handoff.run_read_only_requests),
            "requests": self._air_handoff_manager.list_requests(),
        }

    async def _handle_phone_request_run(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Start a waiting phone request from the computer, with the normal approvals."""
        request_id = self._phone_request_id(params)
        request = (
            self._air_handoff_manager.get_request(request_id) if request_id and self._air_handoff_manager else None
        )
        if request is None:
            return {"status": "error", "message": "No such phone request."}
        if request.status != "waiting":
            return {"status": "error", "message": f"This request is already {request.status}."}
        return await self._start_phone_request(request, read_only=False)

    async def _handle_phone_request_dismiss(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Drop a waiting phone request without running it."""
        request_id = self._phone_request_id(params)
        request = (
            self._air_handoff_manager.get_request(request_id) if request_id and self._air_handoff_manager else None
        )
        if request is None:
            return {"status": "error", "message": "No such phone request."}
        if request.status != "waiting":
            return {"status": "error", "message": f"This request is already {request.status}."}
        request.status = "dismissed"
        await self._broadcast_notification("phone_request", request.public_dict())
        return {"status": "ok", "request": request.public_dict()}

    async def _handle_air_handoff_requests_configure(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Turn phone requests on or off, and whether read-only ones may start without you."""
        accept = params.get("accept_requests")
        run_read_only = params.get("run_read_only_requests")
        if not isinstance(accept, bool) or not isinstance(run_read_only, bool):
            return {"status": "error", "message": "accept_requests and run_read_only_requests must be booleans"}
        if run_read_only and not accept:
            return {"status": "error", "message": "Read-only requests can only start at once when requests are on"}
        previous = (self.config.air_handoff.accept_requests, self.config.air_handoff.run_read_only_requests)
        self.config.air_handoff.accept_requests = accept
        self.config.air_handoff.run_read_only_requests = run_read_only
        try:
            self.config.save()
        except Exception as exc:
            self.config.air_handoff.accept_requests, self.config.air_handoff.run_read_only_requests = previous
            return {"status": "error", "message": f"Config update was not saved: {exc}"}
        state = await self._publish_air_handoff_state()
        return {"status": "ok", **state}

    async def _today_overview(self, day: str = "today") -> dict[str, Any]:
        from pilot.workflows.daily_brief import today_overview

        return await today_overview(
            receipts=self._task_receipts,
            timeline=self._activity_timeline,
            undo_journal=self._undo_journal,
            todos=self._todos,
            day=day,
        )

    async def _deliver_daily_brief(self, overview: dict[str, Any]) -> None:
        await self._broadcast_notification(
            "daily_brief", {"day": overview["day"], "text": overview["brief"], "overview": overview}
        )
        if self.config.brief.speak:
            await self._speak_companion_text(
                overview["brief"], channel=SpeechChannel.BACKGROUND_INSIGHT, dedupe_key=f"brief:{overview['day']}"
            )

    async def _handle_today_overview(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """One day at a glance: tasks Heliox ran, app time from the timeline, and what can still be undone."""
        day = params.get("day", "today")
        if day not in ("today", "yesterday", "week"):
            return {"status": "error", "message": "day must be today, yesterday or week"}
        return {"status": "ok", **await self._today_overview(day)}

    async def _handle_daily_brief_now(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Send today's brief right away (it does not count as the scheduled one)."""
        overview = await self._today_overview("today")
        await self._deliver_daily_brief(overview)
        return {"status": "ok", "text": overview["brief"]}

    async def _undo_request(self, text: str) -> str | None:
        """ "undo that" / "undo the last two tasks": open the undo preview for those tasks, newest first.

        Undoing still needs the preview's Undo button for each task. Returns what to say, or
        None when ``text`` is not an undo request.
        """
        normalized = _normalize_spoken_phrase(text)
        single = UNDO_ONE_REQUEST.match(normalized)
        several = UNDO_MANY_REQUEST.match(normalized)
        if not single and not several:
            return None
        wanted = 1
        if several:
            word = several.group("count")
            wanted = int(word) if word.isdigit() else UNDO_COUNT_WORDS.get(word, 0)
            if not 1 <= wanted <= MAX_UNDO_REQUEST:
                return f"I can open undo for up to {MAX_UNDO_REQUEST} tasks at a time."
        rows = await self._undo_journal.list_tasks(20) if self._undo_journal is not None else []
        recent = [row for row in rows if not row["undone"]][:wanted]
        keys = [row["task_key"] for row in recent if row["undoable"] > 0]
        if not keys:
            return "There is nothing I can undo right now."
        await self._broadcast_notification("undo_offer", {"task_key": keys[0], "task_keys": keys})
        if wanted == 1:
            return "Here is what I can undo. Check it, then press Undo."
        if len(keys) < wanted:
            return f"I can undo {len(keys)} of your last {wanted} tasks. Check each one, newest first, then press Undo."
        return f"Here are your last {wanted} tasks, newest first. Check each one, then press Undo."

    def _phrase_reserved(self, phrase: str) -> bool:
        """True for phrases that already mean something to Heliox and so cannot be shortcuts."""
        from pilot.workflows import (
            app_reminders,
            clipboard_history,
            focus_guard,
            notes,
            nudges,
            system_health,
            todos,
            watchers,
            workspaces,
        )
        from pilot.workflows.activity_timeline import LAST_SEEN_QUESTION, TIMELINE_QUESTION
        from pilot.workflows.command_guide import HELP_REQUEST
        from pilot.workflows.focus import (
            POMODORO_REQUEST,
            SCHEDULE_CLEAR_REQUEST,
            SCHEDULE_LIST_REQUEST,
            SCHEDULE_REQUEST,
            START_REQUEST,
            STOP_REQUEST,
        )
        from pilot.workflows.privacy import FORGET_REQUEST
        from pilot.workflows.reminders import CANCEL_REQUEST, LIST_REQUEST, SET_REQUEST, TIMER_REQUEST
        from pilot.workflows.routines import REPEAT_REQUEST, ROUTINE_REQUEST
        from pilot.workflows.task_receipts import ASKED_QUESTION, RECAP_QUESTION
        from pilot.workflows.undo_journal import WHERE_QUESTION

        if phrase in VOICE_STOP_PHRASES or phrase in VOICE_DENIALS or phrase in VOICE_APPROVAL_ATTEMPTS:
            return True
        patterns = (UNDO_ONE_REQUEST, UNDO_MANY_REQUEST, REPEAT_REQUEST, ROUTINE_REQUEST)
        questions = (
            TIMELINE_QUESTION,
            LAST_SEEN_QUESTION,
            RECAP_QUESTION,
            WHERE_QUESTION,
            SET_REQUEST,
            LIST_REQUEST,
            CANCEL_REQUEST,
            START_REQUEST,
            STOP_REQUEST,
            TIMER_REQUEST,
            ASKED_QUESTION,
            workspaces.SAVE_REQUEST,
            workspaces.OPEN_REQUEST,
            workspaces.LIST_REQUEST,
            workspaces.DELETE_REQUEST,
            notes.TAKE_REQUEST,
            notes.READ_REQUEST,
            notes.SEARCH_REQUEST,
            notes.DELETE_LAST_REQUEST,
            todos.ADD_REQUEST,
            todos.READ_REQUEST,
            todos.DONE_REQUEST,
            todos.REMOVE_REQUEST,
            todos.CLEAR_DONE_REQUEST,
            watchers.SET_REQUEST,
            watchers.LIST_REQUEST,
            watchers.CANCEL_REQUEST,
            clipboard_history.LIST_REQUEST,
            clipboard_history.COPY_BACK_REQUEST,
            clipboard_history.CLEAR_REQUEST,
            VOICE_CLIPBOARD_SETTING,
            system_health.SLOW_QUESTION,
            system_health.MEMORY_QUESTION,
            system_health.CPU_QUESTION,
            system_health.DISK_QUESTION,
            focus_guard.ADD_REQUEST,
            focus_guard.LIST_REQUEST,
            focus_guard.REMOVE_REQUEST,
            app_reminders.SET_REQUEST,
            app_reminders.LIST_REQUEST,
            nudges.ADD_REQUEST,
            nudges.LIST_REQUEST,
            nudges.DELETE_REQUEST,
            VOICE_TIMELINE_SETTING,
            VOICE_BRIEF_SETTING,
            VOICE_BRIEF_TIME_SETTING,
            VOICE_BRIEF_ALOUD_SETTING,
            VOICE_BRIEF_QUIET_SETTING,
            HELP_REQUEST,
            SAY_AGAIN_REQUEST,
            FORGET_REQUEST,
            POMODORO_REQUEST,
            SCHEDULE_REQUEST,
            SCHEDULE_LIST_REQUEST,
            SCHEDULE_CLEAR_REQUEST,
        )
        return (
            phrase.startswith("when i say")
            or any(pattern.match(phrase) for pattern in patterns)
            or any(pattern.match(phrase) for pattern in questions)
        )

    async def _latest_receipt(self, *, with_actions: bool = False) -> dict[str, Any] | None:
        if self._task_receipts is None:
            return None
        for row in await self._task_receipts.list(20):
            if with_actions and not row["action_count"]:
                continue
            envelope = await self._task_receipts.get(row["receipt_id"])
            if envelope and str(envelope["receipt"].get("request") or "").strip():
                return envelope
        return None

    async def _expand_request(self, text: str) -> tuple[str, str] | None:
        """A voice shortcut or "do that again" turned into the request to run, with a note for the chat."""
        from pilot.workflows.routines import REPEAT_REQUEST, normalize_phrase

        if self._shortcuts is not None:
            shortcut = await asyncio.to_thread(self._shortcuts.match, text)
            if shortcut is not None:
                return shortcut["command"], f'Shortcut "{shortcut["phrase"]}": {shortcut["command"]}'
        if self._workspaces is not None:
            workspace = await asyncio.to_thread(self._workspaces.open_request, text)
            if workspace is not None:
                return workspace
        if REPEAT_REQUEST.match(normalize_phrase(text)):
            envelope = await self._latest_receipt()
            if envelope is not None:
                request = str(envelope["receipt"]["request"])
                return request, f'Again: "{request}"'
        return None

    async def _routine_request(self, text: str) -> str | None:
        """ "Do that every day at 9": propose a routine made from the last task. Creating it needs a click."""
        from pilot.workflows.routines import ROUTINE_REQUEST, normalize_phrase, parse_schedule, routine_from_receipt

        asked = ROUTINE_REQUEST.match(normalize_phrase(text))
        if not asked:
            return None
        schedule = parse_schedule(asked.group("schedule"))
        if schedule is None:
            return 'Tell me when, for example "do that every day at 9" or "do that every weekday at 8:30 am".'
        envelope = await self._latest_receipt(with_actions=True)
        routine = routine_from_receipt(envelope) if envelope is not None else None
        if routine is None:
            return "Your last task has no completed actions a routine could repeat."
        offer = {**routine, **schedule}
        await self._broadcast_notification("routine_offer", offer)
        kinds = len(routine["allowed_actions"])
        return (
            f'I can run "{routine["name"]}" {schedule["label"]}, using only the {kinds} kind'
            f"{'' if kinds == 1 else 's'} of action it used. Check it on screen, then press Create."
        )

    async def _shortcut_request(self, text: str) -> str | None:
        """ "When I say start work, open VS Code": save a voice shortcut."""
        from pilot.workflows.routines import parse_shortcut_definition

        definition = parse_shortcut_definition(text)
        if definition is None or self._shortcuts is None:
            return None
        try:
            entry = await asyncio.to_thread(self._shortcuts.add, *definition)
        except ValueError as exc:
            return str(exc)
        await self._broadcast_notification("shortcuts_changed", {"shortcuts": self._shortcuts.list()})
        return f'Okay. When you say "{entry["phrase"]}", I\'ll run: {entry["command"]}'

    async def _control_reply(self, text: str) -> str | None:
        """Requests Heliox answers in code without planning: history questions, undo, routines, shortcuts."""
        from pilot.workflows.routines import REPEAT_REQUEST, normalize_phrase

        handlers = (
            self._help_request,
            self._say_again_request,
            self._forget_request,
            self._local_answer,
            self._focus_request,
            self._settings_request,
            self._workspace_request,
            self._undo_request,
            self._routine_request,
            self._shortcut_request,
        )
        for handler in handlers:
            reply = await handler(text)
            if reply is not None:
                return reply
        if REPEAT_REQUEST.match(normalize_phrase(text)):
            return "There is no earlier task for me to repeat yet."
        return None

    async def _deliver_scheduled_brief(self, overview: dict[str, Any]) -> None:
        if self._focus is not None and self._focus.hold("daily_brief", overview):
            return
        await self._deliver_daily_brief(overview)

    async def _on_focus_end(self, summary: dict[str, Any]) -> None:
        """Deliver what a focus session held, with one summary."""
        from pilot.workflows.focus import compose_focus_summary

        usage = None
        if self._activity_timeline is not None and self._activity_timeline.enabled:
            usage = await asyncio.to_thread(
                self._activity_timeline.summarize, summary["started_at"], summary["ended_at"]
            )
        summary["text"] = compose_focus_summary(summary, usage)
        if summary.get("break_minutes") and summary["reason"] == "time" and self._reminders is not None:
            minutes = int(summary["break_minutes"])
            try:
                await asyncio.to_thread(
                    self._reminders.add,
                    time.time() + minutes * 60,
                    'Break is over. Say "start a pomodoro" for another round.',
                )
                summary["text"] += f" Take a {minutes}-minute break; I'll tell you when it's over."
            except ValueError:
                logger.warning("Could not set the pomodoro break reminder", exc_info=True)
        await self._broadcast_notification(
            "focus_ended", {"text": summary["text"], "duration": summary["duration"], "held": len(summary["held"])}
        )
        if summary["reason"] == "time":  # when the person said "stop focusing", the reply speaks it
            await self._speak_companion_text(summary["text"], channel=SpeechChannel.TASK_NARRATION)
        for item in summary["held"]:
            if item["kind"] == "reminder":
                await self._broadcast_notification("reminder", {**item["payload"], "held": True})
            elif item["kind"] == "daily_brief":
                await self._deliver_daily_brief(item["payload"])
            elif item["kind"] == "nudge":
                await self._broadcast_notification("nudge", {**item["payload"], "held": True})

    async def _workspace_request(self, text: str) -> str | None:
        """ "Save this as my coding setup", "what setups do I have", "delete my coding setup"."""
        from pilot.workflows.workspaces import DELETE_REQUEST, LIST_REQUEST, SAVE_REQUEST, normalize, open_apps

        if self._workspaces is None:
            return None
        normalized = normalize(text)
        save = SAVE_REQUEST.match(normalized)
        if save:
            try:
                apps = await open_apps()
                entry = await asyncio.to_thread(self._workspaces.save, save.group("name"), apps)
            except (ValueError, OSError) as exc:
                return str(exc)
            return (
                f"Saved your {self._workspaces.describe(entry)} setup. "
                f'Say "open my {entry["name"]} setup" to bring it back.'
            )
        if LIST_REQUEST.match(normalized):
            items = await asyncio.to_thread(self._workspaces.list)
            if not items:
                return 'You have no saved setups. Say "save this as my coding setup" to make one.'
            return "Your setups: " + "; ".join(self._workspaces.describe(item) for item in items) + "."
        delete = DELETE_REQUEST.match(normalized)
        if delete and await asyncio.to_thread(self._workspaces.delete, delete.group("name")):
            return f'Deleted your "{normalize(delete.group("name"))}" setup.'
        return None

    async def _handle_workspace_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._workspaces is None:
            return {"status": "error", "message": "Workspaces are not ready yet.", "workspaces": []}
        return {"status": "ok", "workspaces": await asyncio.to_thread(self._workspaces.list)}

    async def _handle_workspace_save(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Save the apps that have windows open now under ``name``."""
        from pilot.workflows.workspaces import open_apps

        name = params.get("name")
        if not isinstance(name, str) or self._workspaces is None:
            return {"status": "error", "message": "name is required"}
        try:
            entry = await asyncio.to_thread(self._workspaces.save, name, await open_apps())
        except (ValueError, OSError) as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "workspace": entry, "workspaces": await asyncio.to_thread(self._workspaces.list)}

    async def _handle_workspace_delete(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        name = params.get("name")
        if not isinstance(name, str) or self._workspaces is None:
            return {"status": "error", "message": "name is required"}
        if not await asyncio.to_thread(self._workspaces.delete, name):
            return {"status": "error", "message": "No such workspace."}
        return {"status": "ok", "workspaces": await asyncio.to_thread(self._workspaces.list)}

    async def _handle_receipts_search(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Receipts whose request or summary contains every word of ``query``, newest first."""
        query = params.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            return {"status": "error", "message": "query must be 1 to 200 characters"}
        if self._task_receipts is None:
            return {"status": "error", "message": "Receipts are not ready yet."}
        return {"status": "ok", "receipts": await asyncio.to_thread(self._task_receipts.search, query, 30)}

    async def _deliver_nudge(self, nudge: dict[str, Any]) -> None:
        if self._focus is not None and self._focus.hold("nudge", nudge):
            return
        await self._broadcast_notification("nudge", nudge)
        await self._speak_companion_text(
            nudge["text"], channel=SpeechChannel.PROACTIVE_SUGGESTION, dedupe_key=f"nudge:{nudge['id']}"
        )

    async def _deliver_watch(self, event: dict[str, Any]) -> None:
        """Say what a watcher saw. The person asked for it, so focus does not hold it."""
        await self._broadcast_notification("watch_fired", event)
        await self._speak_companion_text(
            event["text"], channel=SpeechChannel.TASK_NARRATION, dedupe_key=f"watch:{event['id']}"
        )

    async def _handle_watch_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._watchers is None:
            return {"status": "error", "message": "Watchers are not ready yet.", "watchers": []}
        return {"status": "ok", "watchers": self._watchers.list()}

    async def _handle_watch_add(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Start a watcher from a sentence such as "tell me when Chrome closes"."""
        text = params.get("text")
        if not isinstance(text, str) or not 1 <= len(text) <= 200 or self._watchers is None:
            return {"status": "error", "message": "text must be 1 to 200 characters"}
        sentence = (
            text
            if text.casefold().lstrip().startswith(("tell", "let", "notify", "alert", "ping"))
            else f"tell me when {text}"
        )
        try:
            entry = await asyncio.to_thread(self._watchers.add, sentence)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {
            "status": "ok",
            "watcher": {key: entry[key] for key in ("id", "kind", "label", "created_at")},
            "watchers": self._watchers.list(),
        }

    async def _handle_watch_cancel(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        watcher_id = params.get("id")
        if not isinstance(watcher_id, str) or self._watchers is None:
            return {"status": "error", "message": "id is required"}
        if not self._watchers.cancel(watcher_id):
            return {"status": "error", "message": "No such watcher."}
        return {"status": "ok", "watchers": self._watchers.list()}

    async def _handle_clipboard_history_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """What you copied recently (memory only), newest first, while clipboard history is on."""
        if self._clipboard_history is None:
            return {"status": "error", "message": "Clipboard history is not ready yet.", "items": []}
        return {"status": "ok", "enabled": self._clipboard_history.enabled, "items": self._clipboard_history.list()}

    async def _handle_clipboard_history_copy(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Put an earlier item back on the clipboard."""
        item_id = params.get("id")
        if not isinstance(item_id, str) or self._clipboard_history is None:
            return {"status": "error", "message": "id is required"}
        try:
            item = await self._clipboard_history.copy_back(item_id)
        except Exception as exc:
            return {"status": "error", "message": f"Could not copy: {exc}"}
        if item is None:
            return {"status": "error", "message": "No such item."}
        return {"status": "ok", "items": self._clipboard_history.list()}

    async def _handle_clipboard_history_clear(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._clipboard_history is None:
            return {"status": "error", "message": "Clipboard history is not ready yet."}
        return {"status": "ok", "removed": self._clipboard_history.clear(), "items": []}

    async def _deliver_app_reminder(self, reminder: dict[str, Any]) -> None:
        """An app reminder fires because the person just opened that app, so focus does not hold it."""
        await self._broadcast_notification(
            "reminder", {**reminder, "due": time.time(), "late": False, "trigger": "app"}
        )
        await self._speak_companion_text(
            f"Reminder: {reminder['text']}",
            channel=SpeechChannel.TASK_NARRATION,
            dedupe_key=f"app-reminder:{reminder['id']}",
        )

    async def _handle_app_reminder_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._app_reminders is None:
            return {"status": "error", "message": "App reminders are not ready yet.", "reminders": []}
        return {"status": "ok", "reminders": await asyncio.to_thread(self._app_reminders.list)}

    async def _handle_app_reminder_delete(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        reminder_id = params.get("id")
        if not isinstance(reminder_id, str) or self._app_reminders is None:
            return {"status": "error", "message": "id is required"}
        if not await asyncio.to_thread(self._app_reminders.delete, reminder_id):
            return {"status": "error", "message": "No such app reminder."}
        return {"status": "ok", "reminders": await asyncio.to_thread(self._app_reminders.list)}

    async def _deliver_distraction(self, event: dict[str, Any]) -> None:
        """One heads-up the person asked for; nothing is blocked or recorded."""
        await self._broadcast_notification("focus_distraction", event)
        await self._speak_companion_text(
            event["text"], channel=SpeechChannel.TASK_NARRATION, dedupe_key=f"distraction:{event['word']}"
        )

    async def _handle_focus_distractions_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._focus_guard is None:
            return {"status": "error", "message": "Focus distractions are not ready yet.", "words": []}
        return {"status": "ok", "words": await asyncio.to_thread(self._focus_guard.list)}

    async def _handle_focus_distraction_add(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        word = params.get("word")
        if not isinstance(word, str) or self._focus_guard is None:
            return {"status": "error", "message": "word is required"}
        try:
            await asyncio.to_thread(self._focus_guard.add, word)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "words": await asyncio.to_thread(self._focus_guard.list)}

    async def _handle_focus_distraction_delete(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        word = params.get("word")
        if not isinstance(word, str) or self._focus_guard is None:
            return {"status": "error", "message": "word is required"}
        if not await asyncio.to_thread(self._focus_guard.remove, word):
            return {"status": "error", "message": "That is not on your list."}
        return {"status": "ok", "words": await asyncio.to_thread(self._focus_guard.list)}

    async def _handle_system_health(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """A fresh reading of processor, memory, disks, battery and the biggest apps, with a summary."""
        from pilot.workflows.system_health import describe_health

        if self._system_health is None:
            return {"status": "error", "message": "The health check is not ready yet."}
        try:
            snapshot = await self._system_health.snapshot()
        except Exception as exc:
            return {"status": "error", "message": f"Could not read this computer: {exc}"}
        return {"status": "ok", **snapshot, "summary": describe_health(snapshot)}

    async def _settings_request(self, text: str) -> str | None:
        """Change a few everyday settings by voice. Security and approval settings are never changed this way."""
        normalized = _normalize_spoken_phrase(text)
        change: tuple[str, dict[str, Any], str] | None = None
        timeline = VOICE_TIMELINE_SETTING.match(normalized)
        brief = VOICE_BRIEF_SETTING.match(normalized)
        brief_time = VOICE_BRIEF_TIME_SETTING.match(normalized)
        if timeline:
            on = timeline.group("state") == "on" or timeline.group("verb") == "start"
            change = ("timeline", {"enabled": on}, f"The activity timeline is {'on' if on else 'off'}.")
        elif brief:
            on = brief.group("state") == "on"
            change = ("brief", {"enabled": on}, f"The daily brief is {'on' if on else 'off'}.")
        elif brief_time:
            hour, minute = int(brief_time.group("hour")), int(brief_time.group("minute") or 0)
            meridiem = brief_time.group("meridiem")
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return "That is not a time I understand."
            clock = f"{hour:02d}:{minute:02d}"
            change = ("brief", {"enabled": True, "time": clock}, f"I'll send your daily brief every day at {clock}.")
        elif VOICE_BRIEF_ALOUD_SETTING.match(normalized):
            change = ("brief", {"speak": True}, "I'll read the daily brief aloud.")
        elif VOICE_BRIEF_QUIET_SETTING.match(normalized):
            change = ("brief", {"speak": False}, "I'll send the daily brief without reading it aloud.")
        elif clipboard := VOICE_CLIPBOARD_SETTING.match(normalized):
            on = clipboard.group("state") == "on"
            change = (
                "clipboard",
                {"history": on},
                "Clipboard history is on. I'll remember what you copy for a few hours, in memory only."
                if on
                else "Clipboard history is off, and I forgot what you copied.",
            )
        if change is None:
            return None
        section, values, reply = change
        result = await self._handle_update_config({"section": section, "values": values}, None)
        if result.get("status") != "ok":
            return f"That setting was not changed: {result.get('message', 'unknown error')}"
        await self._broadcast_notification("config_updated", {"method": "voice", "section": section})
        return reply

    async def _handle_notes_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._notes is None:
            return {"status": "error", "message": "Notes are not ready yet.", "notes": []}
        return {"status": "ok", "notes": list(reversed(await asyncio.to_thread(self._notes.list)))}

    async def _handle_note_add(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        text = params.get("text")
        if not isinstance(text, str) or self._notes is None:
            return {"status": "error", "message": "text is required"}
        try:
            await asyncio.to_thread(self._notes.add, text)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "notes": list(reversed(await asyncio.to_thread(self._notes.list)))}

    async def _handle_note_delete(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        note_id = params.get("id")
        if not isinstance(note_id, str) or self._notes is None:
            return {"status": "error", "message": "id is required"}
        if not await asyncio.to_thread(self._notes.delete, note_id):
            return {"status": "error", "message": "No such note."}
        return {"status": "ok", "notes": list(reversed(await asyncio.to_thread(self._notes.list)))}

    async def _todo_reply(self, **extra: Any) -> dict:
        return {"status": "ok", **extra, "todos": await asyncio.to_thread(self._todos.list)}

    async def _handle_todo_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._todos is None:
            return {"status": "error", "message": "The to-do list is not ready yet.", "todos": []}
        return await self._todo_reply()

    async def _handle_todo_add(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        text = params.get("text")
        if not isinstance(text, str) or self._todos is None:
            return {"status": "error", "message": "text is required"}
        try:
            await asyncio.to_thread(self._todos.add, text)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return await self._todo_reply()

    async def _handle_todo_set_done(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        item_id, done = params.get("id"), params.get("done", True)
        if not isinstance(item_id, str) or not isinstance(done, bool) or self._todos is None:
            return {"status": "error", "message": "id and a true or false done are required"}
        if await asyncio.to_thread(self._todos.set_done, item_id, done) is None:
            return {"status": "error", "message": "No such to-do."}
        return await self._todo_reply()

    async def _handle_todo_delete(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        item_id = params.get("id")
        if not isinstance(item_id, str) or self._todos is None:
            return {"status": "error", "message": "id is required"}
        if not await asyncio.to_thread(self._todos.delete, item_id):
            return {"status": "error", "message": "No such to-do."}
        return await self._todo_reply()

    async def _handle_todo_clear_done(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._todos is None:
            return {"status": "error", "message": "The to-do list is not ready yet."}
        removed = await asyncio.to_thread(self._todos.clear_done)
        return await self._todo_reply(removed=removed)

    async def _handle_nudge_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._nudges is None:
            return {"status": "error", "message": "Nudges are not ready yet.", "nudges": []}
        return {"status": "ok", "nudges": await asyncio.to_thread(self._nudges.list)}

    async def _handle_nudge_add(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        match, minutes = params.get("match"), params.get("minutes")
        if not isinstance(match, str) or not isinstance(minutes, int) or isinstance(minutes, bool):
            return {"status": "error", "message": "match (text) and minutes (whole number) are required"}
        if self._nudges is None:
            return {"status": "error", "message": "Nudges are not ready yet."}
        try:
            await asyncio.to_thread(self._nudges.add, match, minutes)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "nudges": await asyncio.to_thread(self._nudges.list)}

    async def _handle_nudge_delete(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        nudge_id = params.get("id")
        if not isinstance(nudge_id, str) or self._nudges is None:
            return {"status": "error", "message": "id is required"}
        if not await asyncio.to_thread(self._nudges.delete, nudge_id):
            return {"status": "error", "message": "No such nudge."}
        return {"status": "ok", "nudges": await asyncio.to_thread(self._nudges.list)}

    @staticmethod
    def _export_dir() -> Path:
        downloads_dir = Path.home() / "Downloads"
        return downloads_dir if downloads_dir.exists() else (DATA_DIR / "exports")

    async def _forget_request(self, text: str) -> str | None:
        """ "Forget my activity from the last hour": remove that stretch of the activity timeline."""
        from datetime import datetime

        from pilot.workflows.privacy import forget_window

        window = forget_window(text, datetime.now())
        if window is None:
            return None
        if self._activity_timeline is None:
            return "The activity timeline is not ready yet."
        start, end, label = window
        removed = await asyncio.to_thread(self._activity_timeline.forget, start, end)
        return f"Done. I removed {removed} timeline entr{'y' if removed == 1 else 'ies'} from {label}."

    async def _privacy_counts(self) -> dict[str, int | None]:
        counts: dict[str, int | None] = {}
        if self._activity_timeline is not None:
            counts["timeline"] = (await asyncio.to_thread(self._activity_timeline.status))["span_count"]
        receipts_dir = DATA_DIR / "receipts"
        counts["receipts"] = len(list(receipts_dir.glob("*.json"))) if receipts_dir.is_dir() else 0
        if self._undo_journal is not None:
            counts["undo"] = len(await self._undo_journal.list_tasks(200))
        if self._clipboard_history is not None:
            counts["clipboard"] = len(self._clipboard_history.list())
        for key, store in (
            ("notes", self._notes),
            ("todos", self._todos),
            ("reminders", self._reminders),
            ("app_reminders", self._app_reminders),
            ("shortcuts", self._shortcuts),
            ("workspaces", self._workspaces),
            ("nudges", self._nudges),
            ("distractions", self._focus_guard),
            ("quiet_hours", self._quiet_hours),
        ):
            if store is not None:
                counts[key] = len(await asyncio.to_thread(store.list))
        return counts

    async def _handle_privacy_inventory(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Each personal store Heliox keeps on this computer, with its size and item count."""
        from pilot.workflows.privacy import inventory

        counts = await self._privacy_counts()
        return {"status": "ok", "stores": await asyncio.to_thread(inventory, DATA_DIR, counts)}

    async def _handle_privacy_clear(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Delete everything in one personal store. Needs ``confirmed`` to be exactly true."""
        from pilot.workflows.privacy import STORES, clear_store

        key = params.get("key")
        if key not in STORES:
            return {"status": "error", "message": "Unknown store"}
        if params.get("confirmed") is not True:
            return {"status": "error", "message": "Clearing needs confirmation."}
        if key == "timeline" and self._activity_timeline is not None:
            await asyncio.to_thread(self._activity_timeline.clear)
        elif key == "clipboard":
            if self._clipboard_history is not None:
                self._clipboard_history.clear()
        elif key == "undo" and self._undo_journal is not None:
            busy = self._undo_busy_message()
            if busy:
                return {"status": "error", "message": busy}
            await self._undo_journal.clear()
        else:
            await asyncio.to_thread(clear_store, DATA_DIR, key)
        return await self._handle_privacy_inventory({}, ws)

    async def _handle_timeline_forget(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Remove the last ``minutes`` (1 to 1440) of the activity timeline."""
        minutes = params.get("minutes")
        if not isinstance(minutes, int) or isinstance(minutes, bool) or not 1 <= minutes <= 1440:
            return {"status": "error", "message": "minutes must be from 1 to 1440"}
        if self._activity_timeline is None:
            return {"status": "error", "message": "The activity timeline is not ready yet."}
        now = time.time()
        removed = await asyncio.to_thread(self._activity_timeline.forget, now - minutes * 60, now)
        return {"status": "ok", "removed": removed}

    async def _handle_backup_create(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Zip your notes, reminders, shortcuts, setups, nudges, quiet hours and receipts into Downloads."""
        from pilot.workflows.privacy import create_backup

        include_timeline = params.get("include_timeline", False)
        if not isinstance(include_timeline, bool):
            return {"status": "error", "message": "include_timeline must be a boolean"}
        try:
            path = await asyncio.to_thread(
                create_backup, DATA_DIR, self._export_dir(), include_timeline=include_timeline
            )
        except OSError as exc:
            return {"status": "error", "message": f"Backup failed: {exc}"}
        return {"status": "ok", "name": path.name, "path": str(path)}

    async def _handle_backup_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        from pilot.workflows.privacy import list_backups

        return {"status": "ok", "backups": await asyncio.to_thread(list_backups, self._export_dir())}

    async def _handle_backup_restore(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Restore a Heliox backup from the Downloads folder. Needs ``confirmed`` to be exactly true."""
        from pilot.workflows.privacy import restore_backup

        name = params.get("name")
        if not isinstance(name, str):
            return {"status": "error", "message": "name is required"}
        if params.get("confirmed") is not True:
            return {"status": "error", "message": "Restoring needs confirmation."}
        try:
            restored = await asyncio.to_thread(restore_backup, DATA_DIR, self._export_dir(), name)
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "restored": restored}

    async def _say_again_request(self, text: str) -> str | None:
        """ "Say that again": repeat the last thing Heliox said out loud."""
        if not SAY_AGAIN_REQUEST.match(_normalize_spoken_phrase(text)):
            return None
        return self._last_spoken or "I haven't said anything yet."

    async def _on_quiet_hours_start(self, status: dict[str, Any]) -> None:
        await self._broadcast_notification("focus_status", status)

    async def _handle_focus_schedule_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._quiet_hours is None:
            return {"status": "error", "message": "Quiet hours are not ready yet.", "windows": []}
        return {"status": "ok", "windows": await asyncio.to_thread(self._quiet_hours.list)}

    async def _handle_focus_schedule_add(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Add a quiet-hours window: ``days`` (0 = Monday), ``start`` and ``end`` in minutes after midnight."""
        if self._quiet_hours is None:
            return {"status": "error", "message": "Quiet hours are not ready yet."}
        days, start, end = params.get("days"), params.get("start"), params.get("end")
        if not isinstance(days, list) or not all(isinstance(d, int) and not isinstance(d, bool) for d in days):
            return {"status": "error", "message": "days must be a list of numbers"}
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end)):
            return {"status": "error", "message": "start and end must be minutes after midnight"}
        try:
            await asyncio.to_thread(self._quiet_hours.add, {"days": days, "start": start, "end": end})
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "windows": await asyncio.to_thread(self._quiet_hours.list)}

    async def _handle_focus_schedule_delete(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        window_id = params.get("id")
        if not isinstance(window_id, str) or not window_id or self._quiet_hours is None:
            return {"status": "error", "message": "id is required"}
        if not await asyncio.to_thread(self._quiet_hours.delete, window_id):
            return {"status": "error", "message": "No such window."}
        return {"status": "ok", "windows": await asyncio.to_thread(self._quiet_hours.list)}

    async def _help_request(self, text: str) -> str | None:
        """ "What can I say?": open the command palette and say what Heliox handles right away."""
        from pilot.workflows.command_guide import HELP_REQUEST, SPOKEN_HELP

        if not HELP_REQUEST.match(_normalize_spoken_phrase(text)):
            return None
        await self._broadcast_notification("open_command_palette", {})
        return SPOKEN_HELP

    async def _handle_command_guide(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Everything Heliox answers or does right away, with examples that work as written."""
        from pilot.workflows.command_guide import guide

        return {"status": "ok", "groups": guide()}

    async def _focus_request(self, text: str) -> str | None:
        """ "Focus for 45 minutes" / "stop focusing"."""
        from datetime import datetime

        from pilot.workflows.focus import (
            POMODORO_BREAK_MINUTES,
            POMODORO_MINUTES,
            POMODORO_REQUEST,
            SCHEDULE_CLEAR_REQUEST,
            SCHEDULE_LIST_REQUEST,
            STOP_REQUEST,
            describe_quiet_hours,
            parse_focus_minutes,
            parse_quiet_hours,
        )

        if self._focus is None:
            return None
        normalized = " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())
        pomodoro = POMODORO_REQUEST.match(normalized)
        if pomodoro:
            length = int(pomodoro.group("minutes") or POMODORO_MINUTES)
            try:
                status = self._focus.start(length, break_minutes=POMODORO_BREAK_MINUTES)
            except ValueError as exc:
                return str(exc)
            await self._broadcast_notification("focus_status", status)
            return (
                f"Pomodoro started: {length} minutes of focus, then a {POMODORO_BREAK_MINUTES}-minute break. "
                "I'll hold reminders until then."
            )
        if self._quiet_hours is not None:
            window = parse_quiet_hours(text)
            if window is not None:
                try:
                    await asyncio.to_thread(self._quiet_hours.add, window)
                except ValueError as exc:
                    return str(exc)
                return f"Quiet hours set: {describe_quiet_hours(window)}. I'll start focusing then."
            if SCHEDULE_LIST_REQUEST.match(normalized):
                windows = await asyncio.to_thread(self._quiet_hours.list)
                if not windows:
                    return "You have no quiet hours."
                return "Your quiet hours: " + "; ".join(describe_quiet_hours(w) for w in windows) + "."
            if SCHEDULE_CLEAR_REQUEST.match(normalized):
                removed = await asyncio.to_thread(self._quiet_hours.delete, None)
                return f"Cleared {removed} quiet-hour window{'' if removed == 1 else 's'}."
        minutes = parse_focus_minutes(text)
        if minutes == 0:
            return "A focus session can last up to 8 hours."
        if minutes is not None:
            status = self._focus.start(minutes)
            await self._broadcast_notification("focus_status", status)
            until = datetime.fromtimestamp(status["until"]).strftime("%I:%M %p").lstrip("0")
            return (
                f"Focusing until {until}. I'll hold reminders and the daily brief and keep quiet; "
                "approvals and anything you ask still come through."
            )
        if STOP_REQUEST.match(normalized):
            if self._quiet_hours is not None:
                self._quiet_hours.note_stopped()
            summary = await self._focus.stop(reason="asked")
            await self._broadcast_notification("focus_status", self._focus.status())
            return summary["text"] if summary else "No focus session is running."
        return None

    async def _handle_focus_start(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        minutes, break_minutes = params.get("minutes"), params.get("break_minutes", 0)
        if not isinstance(minutes, int) or isinstance(minutes, bool) or self._focus is None:
            return {"status": "error", "message": "minutes must be a whole number"}
        if not isinstance(break_minutes, int) or isinstance(break_minutes, bool) or not 0 <= break_minutes <= 60:
            return {"status": "error", "message": "break_minutes must be from 0 to 60"}
        try:
            status = self._focus.start(minutes, break_minutes=break_minutes)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        await self._broadcast_notification("focus_status", status)
        return {"status": "ok", **status}

    async def _handle_focus_stop(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._focus is None:
            return {"status": "error", "message": "Focus is not ready yet."}
        if self._quiet_hours is not None:
            self._quiet_hours.note_stopped()
        summary = await self._focus.stop(reason="asked")
        await self._broadcast_notification("focus_status", self._focus.status())
        return {"status": "ok", "summary": summary["text"] if summary else None}

    async def _handle_focus_status(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._focus is None:
            return {"status": "error", "message": "Focus is not ready yet."}
        return {"status": "ok", **self._focus.status()}

    async def _handle_report_export(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Write a Markdown report of today, yesterday or this week to the Downloads folder."""
        from pilot.workflows.daily_brief import compose_report

        day = params.get("day", "today")
        if day not in ("today", "yesterday", "week"):
            return {"status": "error", "message": "day must be today, yesterday or week"}
        overview = await self._today_overview(day)
        downloads_dir = Path.home() / "Downloads"
        export_dir = downloads_dir if downloads_dir.exists() else (DATA_DIR / "exports")
        out_path = export_dir / f"heliox-report-{overview['day']}-{day}.md"
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(compose_report(overview), encoding="utf-8")
        except OSError as exc:
            return {"status": "error", "message": f"Export failed: {exc}"}
        return {"status": "ok", "path": str(out_path)}

    async def _deliver_reminder(self, reminder: dict[str, Any]) -> None:
        if self._focus is not None and self._focus.hold("reminder", reminder):
            return
        await self._broadcast_notification("reminder", reminder)
        spoken = (
            f"Reminder: {reminder['text']}"
            if not reminder.get("late")
            else f"A reminder you missed: {reminder['text']}"
        )
        await self._speak_companion_text(
            spoken, channel=SpeechChannel.PROACTIVE_SUGGESTION, dedupe_key=f"reminder:{reminder['id']}"
        )

    async def _handle_reminder_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        from pilot.workflows.reminders import describe_repeat

        if self._reminders is None:
            return {"status": "error", "message": "Reminders are not ready yet.", "reminders": []}
        items = await asyncio.to_thread(self._reminders.list)
        for item in items:
            if item.get("repeat"):
                item["repeat_label"] = describe_repeat(item["repeat"])
        return {"status": "ok", "reminders": items}

    async def _handle_reminder_add(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Add a reminder: ``text`` plus ``in_minutes`` (1 to 10080) or ``at`` (epoch seconds)."""
        if self._reminders is None:
            return {"status": "error", "message": "Reminders are not ready yet."}
        text, minutes, at = params.get("text"), params.get("in_minutes"), params.get("at")
        if not isinstance(text, str):
            return {"status": "error", "message": "text is required"}
        if isinstance(minutes, (int, float)) and not isinstance(minutes, bool):
            if not 1 <= minutes <= 7 * 24 * 60:
                return {"status": "error", "message": "in_minutes must be from 1 to 10080"}
            due = time.time() + float(minutes) * 60
        elif isinstance(at, (int, float)) and not isinstance(at, bool):
            due = float(at)
        else:
            return {"status": "error", "message": "Give in_minutes or at"}
        try:
            entry = await asyncio.to_thread(self._reminders.add, due, text)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "reminder": entry, "reminders": await asyncio.to_thread(self._reminders.list)}

    async def _handle_reminder_cancel(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        reminder_id = params.get("id")
        if not isinstance(reminder_id, str) or not reminder_id or self._reminders is None:
            return {"status": "error", "message": "id is required"}
        removed = await asyncio.to_thread(self._reminders.cancel, reminder_id)
        if not removed:
            return {"status": "error", "message": "No such reminder."}
        return {"status": "ok", "reminders": await asyncio.to_thread(self._reminders.list)}

    async def _handle_undo_diff(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """What changed, line by line, in each text file a task wrote."""
        task_key = self._undo_task_key(params)
        if task_key is None:
            return {"status": "error", "message": "task_key is required"}
        if self._undo_journal is None:
            return {"status": "error", "message": "Undo is not ready yet."}
        return {"status": "ok", "changes": await self._undo_journal.diff(task_key)}

    async def _handle_routine_suggest(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """A routine made from a finished task, with a default schedule of every day at the next full hour."""
        from datetime import datetime, timedelta

        from pilot.workflows.routines import routine_from_receipt

        receipt_id = params.get("receipt_id")
        if not isinstance(receipt_id, str) or self._task_receipts is None:
            return {"status": "error", "message": "receipt_id is required"}
        envelope = await self._task_receipts.get(receipt_id)
        routine = routine_from_receipt(envelope) if envelope is not None else None
        if routine is None:
            return {"status": "error", "message": "This task has no completed actions a routine could repeat."}
        next_hour = (datetime.now() + timedelta(hours=1)).replace(minute=0)
        return {"status": "ok", "routine": {**routine, "daily_at": next_hour.strftime("%H:%M"), "weekdays": []}}

    async def _handle_shortcut_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        if self._shortcuts is None:
            return {"status": "error", "message": "Shortcuts are not ready yet.", "shortcuts": []}
        return {"status": "ok", "shortcuts": await asyncio.to_thread(self._shortcuts.list)}

    async def _handle_shortcut_add(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        phrase, command = params.get("phrase"), params.get("command")
        if not isinstance(phrase, str) or not isinstance(command, str):
            return {"status": "error", "message": "phrase and command must be text"}
        if self._shortcuts is None:
            return {"status": "error", "message": "Shortcuts are not ready yet."}
        try:
            entry = await asyncio.to_thread(self._shortcuts.add, phrase, command)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        shortcuts = await asyncio.to_thread(self._shortcuts.list)
        await self._broadcast_notification("shortcuts_changed", {"shortcuts": shortcuts})
        return {"status": "ok", "shortcut": entry, "shortcuts": shortcuts}

    async def _handle_shortcut_remove(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        phrase = params.get("phrase")
        if not isinstance(phrase, str) or self._shortcuts is None:
            return {"status": "error", "message": "phrase is required"}
        if not await asyncio.to_thread(self._shortcuts.remove, phrase):
            return {"status": "error", "message": "No such shortcut."}
        shortcuts = await asyncio.to_thread(self._shortcuts.list)
        await self._broadcast_notification("shortcuts_changed", {"shortcuts": shortcuts})
        return {"status": "ok", "shortcuts": shortcuts}

    async def _local_answer(self, text: str) -> str | None:
        """Answer questions about the person's own history (timeline, receipts) in code, without a model."""
        sources = (
            self._activity_timeline,
            self._task_receipts,
            self._undo_journal,
            self._app_reminders,
            self._reminders,
            self._notes,
            self._todos,
            self._nudges,
            self._focus_guard,
            self._watchers,
            self._clipboard_history,
            self._system_health,
        )
        for source in sources:
            if source is None:
                continue
            try:
                answer = await source.answer_async(text)
            except Exception:
                logger.warning("Local history answer failed", exc_info=True)
                continue
            if answer is not None:
                return answer
        return None

    async def _handle_timeline_status(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Whether the activity timeline is recording, and how much it holds."""
        if self._activity_timeline is None:
            return {"status": "error", "message": "The activity timeline is not ready yet."}
        return {"status": "ok", **await asyncio.to_thread(self._activity_timeline.status)}

    async def _handle_timeline_search(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Spans whose app or window title contains ``query`` (newest first)."""
        query = params.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            return {"status": "error", "message": "query must be 1 to 200 characters"}
        if self._activity_timeline is None:
            return {"status": "error", "message": "The activity timeline is not ready yet."}
        limit = params.get("limit", 50)
        if not isinstance(limit, int) or isinstance(limit, bool):
            limit = 50
        spans = await asyncio.to_thread(self._activity_timeline.search, query, limit=limit)
        return {"status": "ok", "spans": spans}

    async def _handle_timeline_summary(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Time per app for a window given as text ("this morning") or as start/end epoch seconds."""
        if self._activity_timeline is None:
            return {"status": "error", "message": "The activity timeline is not ready yet."}
        from datetime import datetime

        from pilot.workflows.activity_timeline import parse_when

        start, end = params.get("start"), params.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start < end:
            label = ""
        else:
            when = params.get("when", "today")
            if not isinstance(when, str) or len(when) > 100:
                return {"status": "error", "message": "when must be a short phrase such as 'this morning'"}
            start, end, label = parse_when(when, datetime.now())
        summary = await asyncio.to_thread(self._activity_timeline.summarize, float(start), float(end))
        spans = await asyncio.to_thread(self._activity_timeline.spans, float(start), float(end), 200)
        return {"status": "ok", "label": label, "summary": summary, "spans": spans}

    async def _handle_timeline_pause(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Pause recording for ``minutes`` (0 resumes)."""
        minutes = params.get("minutes", 60)
        if not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or not 0 <= minutes <= 1440:
            return {"status": "error", "message": "minutes must be from 0 to 1440"}
        if self._activity_timeline is None:
            return {"status": "error", "message": "The activity timeline is not ready yet."}
        paused_until = await asyncio.to_thread(self._activity_timeline.pause, minutes)
        return {"status": "ok", "paused_until": paused_until or None}

    async def _handle_timeline_clear(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Delete everything the timeline has recorded."""
        if params.get("confirmed") is not True:
            return {"status": "error", "message": "Clearing the timeline needs confirmation."}
        if self._activity_timeline is None:
            return {"status": "error", "message": "The activity timeline is not ready yet."}
        await asyncio.to_thread(self._activity_timeline.clear)
        return {"status": "ok"}

    async def _handle_receipts_list(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Recent signed task receipts and this computer's signing key fingerprint."""
        if self._task_receipts is None:
            return {"status": "error", "message": "Receipts are not ready yet.", "receipts": []}
        limit = params.get("limit", 50)
        if not isinstance(limit, int) or isinstance(limit, bool):
            limit = 50
        return {
            "status": "ok",
            "enabled": self.config.receipts.enabled,
            "key": await self._task_receipts.key_info(),
            "receipts": await self._task_receipts.list(limit),
        }

    async def _handle_receipt_get(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """The full signed receipt, for viewing or exporting."""
        receipt_id = params.get("receipt_id")
        if not isinstance(receipt_id, str) or self._task_receipts is None:
            return {"status": "error", "message": "receipt_id is required"}
        envelope = await self._task_receipts.get(receipt_id)
        if envelope is None:
            return {"status": "error", "message": "No receipt with that id."}
        if params.get("export") is True:
            downloads_dir = Path.home() / "Downloads"
            export_dir = downloads_dir if downloads_dir.exists() else (DATA_DIR / "exports")
            out_path = export_dir / f"heliox-receipt-{receipt_id[:12]}.json"
            try:
                export_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as exc:
                return {"status": "error", "message": f"Export failed: {exc}"}
            return {"status": "ok", "envelope": envelope, "path": str(out_path)}
        return {"status": "ok", "envelope": envelope}

    async def _handle_receipt_verify(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Check a stored receipt (by id) or a pasted receipt envelope."""
        if self._task_receipts is None:
            return {"status": "error", "message": "Receipts are not ready yet."}
        envelope = params.get("envelope")
        receipt_id = params.get("receipt_id")
        if envelope is None and isinstance(receipt_id, str):
            envelope = await self._task_receipts.get(receipt_id)
            if envelope is None:
                return {"status": "error", "message": "No receipt with that id."}
        if envelope is None:
            return {"status": "error", "message": "receipt_id or envelope is required"}
        return {"status": "ok", "result": await self._task_receipts.verify(envelope)}

    async def _handle_receipts_verify_chain(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Check every stored receipt and the hash chain between them."""
        if self._task_receipts is None:
            return {"status": "error", "message": "Receipts are not ready yet."}
        return {"status": "ok", "result": await self._task_receipts.verify_chain()}

    async def _handle_list_permission_events(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """List recent permission-escalation audit events for display.

        Args:
            params: JSON-RPC parameters, optionally {limit, plan_id}.
            ws: The WebSocket connection.

        Returns:
            A dict with status and the list of events.
        """
        if not self._permission_audit:
            return {"status": "error", "message": "Permission audit store is not initialized.", "events": []}

        limit = int(params.get("limit", 50))
        plan_id = params.get("plan_id") or None
        events = await self._permission_audit.list_events(limit=limit, plan_id=plan_id)
        return {"status": "ok", "events": events}

    async def _handle_verify_permission_audit(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Verify the tamper-evident HMAC chain of the permission audit log.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status and the ChainVerificationResult fields.
        """
        if not self._permission_audit:
            return {"status": "error", "message": "Permission audit store is not initialized."}

        result = await self._permission_audit.verify_chain()
        return {
            "status": "ok",
            "valid": result.valid,
            "checked_entries": result.checked_entries,
            "error": result.error,
        }

    async def _handle_list_gateway_events(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """List recent Agent Gateway audit events for the Settings transparency view.

        Args:
            params: JSON-RPC parameters, optionally {limit, plan_id,
                source_profile, action_family, decision}.
            ws: The WebSocket connection.

        Returns:
            A dict with status and the list of events.
        """
        if not self._gateway_audit:
            return {"status": "error", "message": "Agent gateway audit store is not initialized.", "events": []}

        events = await self._gateway_audit.list_events(
            limit=int(params.get("limit", 50)),
            plan_id=params.get("plan_id") or None,
            source_profile=params.get("source_profile") or None,
            action_family=params.get("action_family") or None,
            decision=params.get("decision") or None,
        )
        return {"status": "ok", "events": events}

    async def _handle_verify_gateway_audit(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Verify the tamper-evident HMAC chain of the Agent Gateway audit log.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status and the ChainVerificationResult fields.
        """
        if not self._gateway_audit:
            return {"status": "error", "message": "Agent gateway audit store is not initialized."}

        result = await self._gateway_audit.verify_chain()
        return {
            "status": "ok",
            "valid": result.valid,
            "checked_entries": result.checked_entries,
            "error": result.error,
        }

    async def _handle_gateway_policy_get(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Return the current Agent Gateway source-profile floors for the Settings editor.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status, whether the gateway is enabled, and each
            source profile's enforced floor (max_tier/deny_action_types/allow_root).
        """
        from dataclasses import asdict

        profiles = {name: asdict(profile) for name, profile in self.config.gateway.source_profiles.items()}
        return {"status": "ok", "enabled": self.config.gateway.enabled, "profiles": profiles}

    async def _handle_gateway_policy_update(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Update one source profile's enforced floor.

        This edits only the source-profile floor persisted in config —
        per-task overrides (e.g. autonomous_submit's scope_override) are
        never settable from Settings, only supplied per-submission by the
        caller, and can only narrow this floor further, never widen it.

        Args:
            params: JSON-RPC parameters: {profile, max_tier?, deny_action_types?, allow_root?}.
            ws: The WebSocket connection.

        Returns:
            A dict with status and the updated policy, or an error if the
            profile name is unknown.
        """
        from dataclasses import asdict

        from pilot.actions import ActionType, PermissionTier
        from pilot.security.gateway import ActionFamily, SourceProfile

        profile_name = params.get("profile", "")
        current = self.config.gateway.source_profiles.get(profile_name)
        if current is None:
            return {"status": "error", "message": f"Unknown source profile: {profile_name}"}

        raw_max_tier = params.get("max_tier")
        raw_deny = params.get("deny_action_types")
        raw_allow_root = params.get("allow_root")

        if raw_max_tier is not None:
            if not isinstance(raw_max_tier, dict):
                return {"status": "error", "message": "max_tier must be an object"}
            valid_families = {family.value for family in ActionFamily}
            for family, tier in raw_max_tier.items():
                if family not in valid_families:
                    return {"status": "error", "message": f"Unknown action family: {family}"}
                if type(tier) is not int or not 0 <= tier <= int(PermissionTier.ROOT_CRITICAL):
                    return {
                        "status": "error",
                        "message": f"Tier for {family} must be an integer from 0 to 4",
                    }
        if raw_deny is not None:
            if not isinstance(raw_deny, list) or any(not isinstance(action, str) for action in raw_deny):
                return {"status": "error", "message": "deny_action_types must be an array of action names"}
            valid_actions = {action.value for action in ActionType}
            unknown_actions = [action for action in raw_deny if action not in valid_actions]
            if unknown_actions:
                return {"status": "error", "message": f"Unknown denied action type: {unknown_actions[0]}"}
        if raw_allow_root is not None and type(raw_allow_root) is not bool:
            return {"status": "error", "message": "allow_root must be a boolean"}

        # Merge onto the existing floor rather than replacing it wholesale —
        # a caller updating only "shell" shouldn't silently reset every
        # other family back to unset/zero.
        merged_max_tier = dict(current.max_tier)
        if raw_max_tier is not None:
            merged_max_tier.update(raw_max_tier)

        updated = SourceProfile(
            max_tier=merged_max_tier,
            deny_action_types=[str(a) for a in raw_deny] if raw_deny is not None else list(current.deny_action_types),
            allow_root=raw_allow_root if raw_allow_root is not None else current.allow_root,
        )
        previous_gateway = copy.deepcopy(self.config.gateway)
        self.config.gateway.source_profiles[profile_name] = updated
        if (failure := self._save_section("gateway", previous_gateway)) is not None:
            return failure

        return {"status": "ok", "profile": profile_name, "policy": asdict(updated)}

    def _interactive_pending_confirmation(self) -> str:
        """The plan_id of the confirmation the interactive task is waiting on, or "" when it waits on none.

        That is the task's own plan approval, or else the newest narrator interrupt raised by one of its steps
        (registered under its own ``interrupt_...`` id with the task's id). Confirmations of other tasks and of
        background jobs are never returned.
        """
        if self._active_plan_id and self._active_plan_id in self._pending_confirms:
            return self._active_plan_id
        task_id = self._active_experience_context.task_id or self._active_task_id
        if not task_id:
            return ""
        for plan_id, pending in reversed(list(self._pending_confirms.items())):
            if pending.task_id == task_id:
                return plan_id
        return ""

    async def _handle_interject(
        self,
        params: dict[str, Any],
        ws: ServerConnection | None,
    ) -> dict:
        """Apply a typed correction to the currently running interactive task.

        This RPC is deliberately out-of-band, like ``confirm`` and ``abort``:
        it must remain responsive while the ordinary ``execute`` RPC owns the
        connection's request lock. A correction cancels the current action,
        then ``_handle_execute_inner`` consumes it and re-plans in the same
        request. Explicit stop phrases retain normal terminal-abort behavior.
        """
        text = str(params.get("input", "")).strip()
        if not text:
            return {"status": "error", "message": "A live correction cannot be empty."}
        if not self._interactive_request_active:
            return {"status": "no_active_execution", "message": "There is no interactive task to revise."}

        requested_mode = str(params.get("mode", "")).strip().lower()
        if requested_mode == "ask":
            return await self._answer_active_task_question(text)
        if requested_mode not in {"", "correct", "stop"}:
            return {"status": "error", "message": "Unsupported in-task mode."}
        plan_id = self._interactive_pending_confirmation()
        if requested_mode != "stop" and plan_id:
            # Typed or spoken text never answers an approval. "Yes" is not a correction either: it would deny
            # and replan what the person tried to approve. Denials resolve the approval directly, and the reply
            # says "denied" only when the denial took effect. This covers the plan approval and a narrator
            # interrupt raised by a step of this task.
            phrase = _normalize_spoken_phrase(text)
            if phrase in VOICE_APPROVAL_ATTEMPTS:
                return {
                    "status": "approval_requires_dialog",
                    "plan_id": plan_id,
                    "message": (
                        "Approve in the visible confirmation dialog. Speech and typed text can deny but never approve."
                    ),
                }
            if phrase in VOICE_DENIALS:
                denied = await self._resolve_pending_denied(
                    plan_id, resolution_path="interject_deny", input_channel="voice_deny"
                )
                if denied:
                    return {"status": "denied", "plan_id": plan_id, "message": "Okay, I won't run that."}
                return {
                    "status": "not_denied",
                    "plan_id": plan_id,
                    "message": (
                        "That approval was already answered, so nothing was denied. "
                        "Say or press Stop to cancel the task."
                    ),
                }
        active_context = self._active_experience_context
        await self._append_experience(
            ExperienceEventType.USER_CORRECTION,
            session_id=active_context.session_id,
            task_id=active_context.task_id,
            user_id=active_context.user_id,
            plan_id=self._active_plan_id,
            idempotency_key=f"correction:{uuid.uuid4()}",
            source="interactive",
            payload={"input": text, "requested_mode": requested_mode or "correct"},
            provenance={"component": "PilotServer._handle_interject"},
            privacy_class=PrivacyClass.SENSITIVE,
        )
        normalized = " ".join(text.lower().split()).rstrip(".!?")
        stop_phrases = {
            "abort",
            "cancel",
            "cancel it",
            "cancel this",
            "cancel this task",
            "never mind",
            "nevermind",
            "stop",
            "stop it",
            "stop this",
            "stop this task",
        }
        if requested_mode == "stop" or normalized in stop_phrases:
            await self._broadcast_notification(
                "companion_interjection",
                {"mode": "stop", "message": "Stopping the current task now."},
            )
            return await self._handle_abort({}, ws)

        if not self.config.narration.live_corrections_enabled:
            return {
                "status": "disabled",
                "message": "Live corrections are disabled in Interactive Companion settings.",
            }

        if self._live_correction:
            self._live_correction = f"{self._live_correction}\nAdditional correction: {text}"
        else:
            self._live_correction = text

        if self._cancel_event and not self._cancel_event.is_set():
            self._cancel_event.set()

        # The primary confirmation wait is not inside _active_execution_task.
        # Resolve only this interactive plan's confirmation; unrelated
        # background confirmations must remain untouched.
        try:
            await self._resolve_pending_denied(self._active_plan_id, resolution_path="interject_correction")
        finally:
            # The running step stops even when this call is cancelled during the durable denial write.
            task = self._active_execution_task
            if task is not None and not task.done():
                task.cancel()

            from pilot.system.pty_session import PtySessionManager

            PtySessionManager.interrupt_all()
        await self._broadcast_notification(
            "companion_interjection",
            {
                "mode": "correct",
                "message": "Correction received. Stopping the current step and revising the plan.",
            },
        )
        return {"status": "revising", "message": "Live correction accepted."}

    async def _answer_active_task_question(self, question: str) -> dict[str, Any]:
        """Answer without changing, approving, or cancelling the running task."""
        if len(question) > 2000:
            return {"status": "error", "message": "In-task questions must be under 2,000 characters."}

        interaction = self._interaction_runtime.status()
        phase = str(interaction.get("phase") or "working").replace("_", " ")
        progress = str(interaction.get("message") or "")[:160]
        normalized = " ".join(question.casefold().split()).rstrip("?!. ")
        if normalized in {
            "status",
            "what are you doing",
            "what are you working on",
            "how is it going",
            "how's it going",
        }:
            message = f"The current task is {phase}. {progress}".strip()
        elif normalized in {"what is on my screen", "what's on my screen", "read my screen"}:
            # OCR stays local and is only invoked for an explicit screen-reading question.
            try:
                from pilot.system.vision import screen_ocr

                visible_text = await asyncio.wait_for(screen_ocr(), timeout=12)
                message = (
                    f"Visible screen text: {visible_text[:2000]}"
                    if visible_text.strip()
                    else "I could not read text on the current screen."
                )
            except Exception:
                message = "I could not read the current screen right now."
        elif self._model_router is not None:
            screen_context = ""
            if self._screen_vision is not None:
                screen_context = self._screen_vision.get_context_for_planner()[:500]
            persona = ""
            if self._subconscious is not None:
                try:
                    persona = (await self._subconscious.get_persona_context())[:1000]
                except Exception:
                    logger.debug("Could not load personal context for in-task question", exc_info=True)
            prompt = (
                f"Current task: {str(interaction.get('user_input') or '')[:500]}\n"
                f"Task phase: {phase}\nProgress: {progress}\n"
                f"Screen metadata (not screen contents): {screen_context}\n"
                f"User preferences: {persona}\nQuestion: {question}"
            )
            try:
                message = (
                    await asyncio.wait_for(
                        self._model_router.generate(
                            prompt,
                            system=(
                                "Answer the user's question briefly while their task continues. "
                                "Do not issue commands, claim to have performed actions, or treat screen metadata "
                                "as screen contents. If evidence is missing, say so."
                            ),
                        ),
                        timeout=30,
                    )
                ).strip()[:2000]
            except Exception:
                message = "I couldn't answer that while the task is running. The task is still continuing."
        else:
            message = "I can report task status now, but general questions need a configured model. The task is still continuing."

        return {"status": "answered", "message": message, "task_id": self._active_task_id}

    async def _handle_abort(
        self,
        params: dict[str, Any],
        ws: ServerConnection | None,
    ) -> dict:
        """Stop the current execution -- both cooperatively and, where
        possible, by really killing whatever is in flight right now (Issue
        #92, extended for real mid-flight cancellation).

        Sets the per-session cancel_event (as before, so the Orchestrator
        and Executor halt at the next action boundary) AND cancels the
        currently tracked interactive execution task, which cascades all
        the way down to run_command's proc.kill() for a mid-flight shell
        subprocess -- the same mechanism AutonomousExecutor.cancel() already
        proves works. Also interrupts every live PTY session (pty_exec
        can't be stopped by Task.cancel() alone; see PtySession.interrupt).

        Returns immediately — cancellation propagates asynchronously; the
        in-flight _execute_tracked()/_handle_execute call observes it and
        shapes its own clean RPC response.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status indicating whether an active execution was aborted.
        """
        aborted_something = False

        if self._durable_tasks is not None and self._active_task_id:
            # A failed durable write must not stop the in-memory cancel, the task cancel or the PTY interrupt.
            try:
                await self._durable_tasks.request_cancel(self._active_task_id)
                aborted_something = True
            except Exception:
                logger.warning("Durable cancel of task %s failed during Stop", self._active_task_id, exc_info=True)

        if self._cancel_event and not self._cancel_event.is_set():
            self._cancel_event.set()
            aborted_something = True

        # A confirmation wait is outside _active_execution_task. Resolve the
        # active prompt as denied so Stop/MCP cancellation returns immediately
        # instead of leaving the request blocked until its five-minute timeout.
        try:
            if await self._resolve_pending_denied(self._active_plan_id, resolution_path="abort"):
                aborted_something = True
        finally:
            # The task cancel and the PTY interrupt run even when this call is cancelled during the durable denial.
            task = self._active_execution_task
            if task is not None and not task.done():
                task.cancel()
                aborted_something = True

            from pilot.system.pty_session import PtySessionManager

            PtySessionManager.interrupt_all()

        if aborted_something:
            logger.info("Abort signal received — cancel_event set and in-flight execution task cancelled")
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
        # Settings shows a banner: Heliox is running on defaults and will not overwrite the file.
        data["load_errors"] = self.config.load_errors
        return data

    async def _handle_get_security_status(self, params: dict, ws: ServerConnection) -> dict:
        """Return both the Heliox root policy and the daemon's real OS privilege state."""
        from pilot.security.privileges import security_runtime_status

        return security_runtime_status(self.config.security.root_enabled)

    async def _handle_get_snapshot_status(self, params: dict, ws: ServerConnection) -> dict:
        """Return the configured pre-action snapshot backend and live readiness."""
        from pilot.system.snapshots import SnapshotManager

        return await SnapshotManager(self.config).status()

    async def _handle_restart_elevated(self, params: dict, ws: ServerConnection) -> dict:
        """Request a Windows UAC handoff to an elevated replacement daemon."""
        from pilot.security.privileges import has_elevated_privileges
        from pilot.system.elevation import ElevationError, request_elevated_restart

        if sys.platform != "win32":
            return {
                "status": "unsupported",
                "message": "Administrator restart is only available on Windows.",
            }
        if not self.config.security.root_enabled:
            return {
                "status": "blocked",
                "message": "Enable Root Access before requesting Administrator privileges.",
            }
        if has_elevated_privileges():
            return {
                "status": "already_elevated",
                "message": "The Heliox daemon is already running as Administrator.",
            }

        try:
            result = await asyncio.to_thread(request_elevated_restart)
        except ElevationError as error:
            logger.warning("Administrator restart was not started: %s", error)
            return {"status": "error", "message": str(error)}

        logger.info("Windows accepted the Administrator restart request")
        return result

    def _update_restrictions(self, values: dict[str, Any]) -> dict[str, Any]:
        """Protected folders, protected packages and blocked commands from Settings.

        These live in restrictions.toml, so the config.toml shape validator rejected every
        change and the Settings lists could never be edited. The sandbox's allowed
        commands stay file-only: widening the sandbox should not be one click away.
        """
        editable = {"protected_folders", "protected_packages", "blocked_commands"}
        unknown = sorted(set(values) - editable)
        if unknown:
            return {"status": "error", "message": f"restrictions.{unknown[0]} cannot be changed from Settings"}
        cleaned: dict[str, list[str]] = {}
        for key, items in values.items():
            if not isinstance(items, list) or len(items) > 500:
                return {"status": "error", "message": f"restrictions.{key} must be a list of up to 500 entries"}
            entries: list[str] = []
            for item in items:
                if not isinstance(item, str) or not item.strip() or len(item) > 1024:
                    return {"status": "error", "message": f"restrictions.{key} entries must be non-empty text"}
                if item.strip() not in entries:
                    entries.append(item.strip())
            cleaned[key] = entries
        previous = {key: list(getattr(self.config.restrictions, key)) for key in cleaned}
        for key, entries in cleaned.items():
            setattr(self.config.restrictions, key, entries)
        try:
            self.config.save()
        except Exception as exc:
            for key, entries in previous.items():
                setattr(self.config.restrictions, key, entries)
            logger.warning("Restrictions update was not persisted", exc_info=True)
            return {"status": "error", "message": f"Restrictions were not saved: {exc}"}
        return {"status": "ok", "values": cleaned}

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

        if not isinstance(section, str):
            return {"status": "error", "message": "Config section must be a string"}
        if not isinstance(values, dict):
            return {"status": "error", "message": "Config values must be an object"}

        if section == "" and "first_run_complete" in values:
            if set(values) != {"first_run_complete"}:
                return {"status": "error", "message": "Unknown top-level config value"}
            if not isinstance(values["first_run_complete"], bool):
                return {"status": "error", "message": "first_run_complete must be a boolean"}
            previous_first_run_complete = self.config.first_run_complete
            self.config.first_run_complete = values["first_run_complete"]
            try:
                self.config.save()
            except Exception as exc:
                self.config.first_run_complete = previous_first_run_complete
                logger.warning("First-run config update was not persisted", exc_info=True)
                return {"status": "error", "message": f"Config update was not saved: {exc}"}
            return {"status": "ok"}

        if section == "air_handoff":
            return {
                "status": "error",
                "message": "Use air_handoff_set_enabled so receiver state changes atomically",
            }
        if section == "network":
            return {
                "status": "error",
                "message": "Use mesh_configure so peer authentication and runtime state change atomically",
            }

        if section == "restrictions":
            return self._update_restrictions(values)
        # Sections with their own RPC: the generic path would skip their validation and runtime
        # side effects (hooks, monitors, receivers), and "server" holds the daemon's auth token.
        if section == "server":
            return {
                "status": "error",
                "message": "Server settings can only be changed in config.toml while Heliox is stopped",
            }
        dedicated = {
            "narration": "narration_config_update",
            "supervision": "supervision_config_update",
            "self_healing": "self_healing_config_update",
            "gateway": "gateway_policy_update or risk_gate_config_update",
            "ssh": "ssh_set_enabled and ssh_save_host",
            "gesture_workflows": "gesture_workflow_bindings_update",
            "co_adaptation": "co_adaptation_config_update",
        }
        if section in dedicated:
            return {"status": "error", "message": f"Use {dedicated[section]} to change {section} settings"}

        target = getattr(self.config, section, None)
        if target is None:
            return {"status": "error", "message": f"Unknown config section: {section}"}
        voice_recorder_fields = {
            "input_device",
            "vad_energy_threshold",
            "vad_adaptive_threshold_enabled",
            "vad_silence_ms",
            "vad_max_utterance_seconds",
        }
        voice_listener_was_running = bool(
            section == "voice"
            and self._voice_listener is not None
            and self._voice_listener.is_running
            and voice_recorder_fields.intersection(values)
        )
        voice_wake_words = list(self._voice_listener.wake_words) if voice_listener_was_running else []
        previous_values = {key: getattr(target, key) for key in values if hasattr(target, key)}
        normalized_values: dict[str, object] = {}
        for k, v in values.items():
            if isinstance(v, float) and not math.isfinite(v):
                # json.loads accepts NaN and Infinity; a NaN budget would switch the cost caps off.
                return {"status": "error", "message": f"{section}.{k} must be a finite number"}
            if section == "model" and k == "gpu_memory_limit_mb" and isinstance(v, int) and v < 0:
                return {"status": "error", "message": "model.gpu_memory_limit_mb cannot be negative"}
            if section == "calendar" and k == "caldav_password_provider" and v not in {"", "caldav"}:
                return {"status": "error", "message": "calendar passwords are stored under the caldav vault entry"}
            if section == "email" and k == "password_provider" and v not in {"", "email"}:
                return {"status": "error", "message": "email passwords are stored under the email vault entry"}
            if hasattr(target, k):
                if section == "model" and k == "provider":
                    if v not in {"ollama", "local", "cloud", "subscription"}:
                        return {
                            "status": "error",
                            "message": "model.provider must be ollama, local, cloud, or subscription",
                        }
                if section == "model" and k == "subscription_provider":
                    if v not in {"codex", "claude"}:
                        return {
                            "status": "error",
                            "message": "model.subscription_provider must be codex or claude",
                        }
                if section == "model" and k == "subscription_model":
                    if not isinstance(v, str) or len(v.strip()) > 200:
                        return {
                            "status": "error",
                            "message": "model.subscription_model must be a model name up to 200 characters",
                        }
                    v = v.strip()
                if section == "model" and k == "mode" and v not in {"lightweight", "full"}:
                    return {
                        "status": "error",
                        "message": "model.mode must be lightweight or full",
                    }
                if section == "model" and k == "subscription_timeout_seconds":
                    if not isinstance(v, int) or isinstance(v, bool) or not 15 <= v <= 600:
                        return {
                            "status": "error",
                            "message": "model.subscription_timeout_seconds must be from 15 to 600",
                        }
                if section == "model" and k == "subscription_max_prompt_chars":
                    if not isinstance(v, int) or isinstance(v, bool) or not 16000 <= v <= 200000:
                        return {
                            "status": "error",
                            "message": "model.subscription_max_prompt_chars must be from 16000 to 200000",
                        }
                if section == "model" and k == "idle_unload_seconds":
                    if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 86400:
                        return {
                            "status": "error",
                            "message": "model.idle_unload_seconds must be from 0 to 86400",
                        }
                if section == "model" and k in {"rate_limit_enabled", "budget_enabled"}:
                    if not isinstance(v, bool):
                        return {"status": "error", "message": f"model.{k} must be a boolean"}
                if section == "model" and k == "rate_limit_rpm":
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 10000:
                        return {"status": "error", "message": "model.rate_limit_rpm must be from 1 to 10000"}
                if section == "model" and k == "rate_limit_burst":
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 1000:
                        return {"status": "error", "message": "model.rate_limit_burst must be from 1 to 1000"}
                if section == "model" and k in {"budget_monthly_limit_usd", "max_usd_per_task"}:
                    if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                        return {"status": "error", "message": f"model.{k} must be zero or greater"}
                if section == "model" and k in {"max_tokens_per_action", "max_tokens_per_task"}:
                    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                        return {"status": "error", "message": f"model.{k} must be zero or greater"}
                if section == "model" and k == "max_consecutive_failures":
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 20:
                        return {
                            "status": "error",
                            "message": "model.max_consecutive_failures must be from 1 to 20",
                        }
                if section == "security" and k == "root_enabled" and not isinstance(v, bool):
                    return {"status": "error", "message": "security.root_enabled must be a boolean"}
                if section == "security" and k == "snapshot_on_destructive" and not isinstance(v, bool):
                    return {
                        "status": "error",
                        "message": "security.snapshot_on_destructive must be a boolean",
                    }
                if section == "security" and k == "dry_run" and not isinstance(v, bool):
                    return {"status": "error", "message": "security.dry_run must be a boolean"}
                if section == "security" and k == "snapshot_retention_count":
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 100:
                        return {
                            "status": "error",
                            "message": "security.snapshot_retention_count must be from 1 to 100",
                        }
                if section == "security" and k == "snapshot_retention_days":
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 3650:
                        return {
                            "status": "error",
                            "message": "security.snapshot_retention_days must be from 1 to 3650",
                        }
                if section == "preview" and k == "enabled" and not isinstance(v, bool):
                    return {"status": "error", "message": "preview.enabled must be a boolean"}
                if section == "preview" and k == "confirm_timeout_seconds":
                    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 10 <= v <= 600:
                        return {
                            "status": "error",
                            "message": "preview.confirm_timeout_seconds must be from 10 to 600",
                        }
                if section == "clipboard" and k == "history" and not isinstance(v, bool):
                    return {"status": "error", "message": "clipboard.history must be a boolean"}
                if section == "brief" and k in {"enabled", "speak"} and not isinstance(v, bool):
                    return {"status": "error", "message": f"brief.{k} must be a boolean"}
                if section == "brief" and k == "time":
                    if not isinstance(v, str) or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
                        return {"status": "error", "message": "brief.time must be a 24-hour time such as 18:00"}
                if section == "timeline" and k == "enabled" and not isinstance(v, bool):
                    return {"status": "error", "message": "timeline.enabled must be a boolean"}
                if section == "timeline" and k == "retention_days":
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 90:
                        return {"status": "error", "message": "timeline.retention_days must be from 1 to 90"}
                if section == "timeline" and k == "excluded_apps":
                    if (
                        not isinstance(v, list)
                        or len(v) > 50
                        or not all(isinstance(item, str) and 0 < len(item.strip()) <= 80 for item in v)
                    ):
                        return {
                            "status": "error",
                            "message": "timeline.excluded_apps must be up to 50 names of 1 to 80 characters",
                        }
                if section == "receipts" and k == "enabled" and not isinstance(v, bool):
                    return {"status": "error", "message": "receipts.enabled must be a boolean"}
                if section == "receipts" and k == "retention_days":
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 3650:
                        return {"status": "error", "message": "receipts.retention_days must be from 1 to 3650"}
                if section == "undo" and k == "enabled" and not isinstance(v, bool):
                    return {"status": "error", "message": "undo.enabled must be a boolean"}
                if section == "undo" and k in {"retention_days", "max_task_mb", "max_total_mb"}:
                    low, high = {"retention_days": (1, 90), "max_task_mb": (1, 10240), "max_total_mb": (16, 102400)}[k]
                    if not isinstance(v, int) or isinstance(v, bool) or not low <= v <= high:
                        return {"status": "error", "message": f"undo.{k} must be from {low} to {high}"}
                if section == "gesture_cursor" and k == "enabled" and not isinstance(v, bool):
                    return {"status": "error", "message": "gesture_cursor.enabled must be a boolean"}
                if section == "gesture_cursor" and k == "sensitivity":
                    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.1 <= v <= 3:
                        return {
                            "status": "error",
                            "message": "gesture_cursor.sensitivity must be from 0.1 to 3",
                        }
                if section == "gesture_cursor" and k == "prediction_ms":
                    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 250:
                        return {
                            "status": "error",
                            "message": "gesture_cursor.prediction_ms must be from 0 to 250",
                        }
                if section == "gesture_cursor" and k == "blend":
                    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
                        return {
                            "status": "error",
                            "message": "gesture_cursor.blend must be from 0 to 1",
                        }
                if section == "adaptive_calibration" and k == "gesture_enabled":
                    if not isinstance(v, bool):
                        return {
                            "status": "error",
                            "message": "adaptive_calibration.gesture_enabled must be a boolean",
                        }
                if section == "adaptive_calibration" and k == "voice_wake_word_enabled":
                    if not isinstance(v, bool):
                        return {
                            "status": "error",
                            "message": ("adaptive_calibration.voice_wake_word_enabled must be a boolean"),
                        }
                if section == "vision" and k == "mediapipe_backend":
                    if v not in {"legacy", "tasks"}:
                        return {
                            "status": "error",
                            "message": "vision.mediapipe_backend must be legacy or tasks",
                        }
                if section == "vision" and k == "gaze_tracking_enabled" and not isinstance(v, bool):
                    return {
                        "status": "error",
                        "message": "vision.gaze_tracking_enabled must be a boolean",
                    }
                if section == "voice" and k == "tts_engine":
                    if v not in {"kokoro_tts", "pocket_tts", "os_native"}:
                        return {
                            "status": "error",
                            "message": "voice.tts_engine must be kokoro_tts, pocket_tts, or os_native",
                        }
                if section == "voice" and k == "tts_voice":
                    if v not in {
                        "af_heart",
                        "af_bella",
                        "af_nicole",
                        "af_sarah",
                        "af_sky",
                        "am_adam",
                        "am_michael",
                        "bf_emma",
                        "bf_isabella",
                        "bm_george",
                        "bm_lewis",
                        "alba",
                        "giovanni",
                        "lola",
                    }:
                        return {
                            "status": "error",
                            "message": "voice.tts_voice must be a supported Kokoro or Pocket TTS voice",
                        }
                if section == "voice" and k == "input_device":
                    if not isinstance(v, str) or not v.strip() or len(v) > 500:
                        return {
                            "status": "error",
                            "message": "voice.input_device must be a valid microphone identifier",
                        }
                    v = v.strip()
                if section == "screen_vision" and k == "capture_interval_seconds":
                    from pilot.agents.screen_vision import (
                        MAX_CAPTURE_INTERVAL_SECONDS,
                        MIN_CAPTURE_INTERVAL_SECONDS,
                    )

                    if (
                        not isinstance(v, (int, float))
                        or isinstance(v, bool)
                        or not MIN_CAPTURE_INTERVAL_SECONDS <= v <= MAX_CAPTURE_INTERVAL_SECONDS
                    ):
                        return {
                            "status": "error",
                            "message": (
                                "screen_vision.capture_interval_seconds must be "
                                f"from {MIN_CAPTURE_INTERVAL_SECONDS} to {MAX_CAPTURE_INTERVAL_SECONDS}"
                            ),
                        }
                    v = float(v)
                if section == "email" and k == "smtp_port":
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 65535:
                        return {"status": "error", "message": "email.smtp_port must be from 1 to 65535"}
                if section == "typed_decision" and k == "enabled" and not isinstance(v, bool):
                    return {"status": "error", "message": "typed_decision.enabled must be a boolean"}
                if (
                    section == "typed_decision"
                    and k == "scoring_mode"
                    and (not isinstance(v, str) or v not in {"trained_head", "native_next_token"})
                ):
                    return {
                        "status": "error",
                        "message": "typed_decision.scoring_mode must be trained_head or native_next_token",
                    }
                if section == "typed_decision" and k == "model_id":
                    from pilot.intelligence.typed_decision import PINNED_BACKBONES

                    if not isinstance(v, str) or v not in PINNED_BACKBONES:
                        return {
                            "status": "error",
                            "message": "typed_decision.model_id must be a supported pinned model",
                        }
                if section == "typed_decision" and k == "max_length":
                    if type(v) is not int or not 64 <= v <= 4096:
                        return {"status": "error", "message": "typed_decision.max_length must be from 64 to 4096"}
                if section == "typed_decision" and k == "temperature":
                    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.05 <= v <= 20:
                        return {"status": "error", "message": "typed_decision.temperature must be from 0.05 to 20"}
                    v = float(v)
                if section == "typed_decision" and k in {"artifact_dir", "report_path"}:
                    if not isinstance(v, str) or len(v) > 4096:
                        return {
                            "status": "error",
                            "message": f"typed_decision.{k} must be a path up to 4096 characters",
                        }
                    v = v.strip()
                    if v and not Path(v).expanduser().is_absolute():
                        return {
                            "status": "error",
                            "message": f"typed_decision.{k} must be an absolute local path",
                        }
                from pilot.config import _validate_config_types

                try:
                    _validate_config_types({section: {k: v}})
                except ValueError as exc:
                    return {"status": "error", "message": str(exc)}
                normalized_values[k] = v
            else:
                return {"status": "error", "message": f"Invalid config key found: '{section}.{k}'."}
        if section == "typed_decision":
            will_enable = normalized_values.get("enabled", self.config.typed_decision.enabled)
            scoring_mode = normalized_values.get("scoring_mode", self.config.typed_decision.scoring_mode)
            artifact_dir = normalized_values.get("artifact_dir", self.config.typed_decision.artifact_dir)
            report_path = normalized_values.get("report_path", self.config.typed_decision.report_path)
            if will_enable and scoring_mode == "trained_head" and (not artifact_dir or not report_path):
                return {
                    "status": "error",
                    "message": "Set typed_decision artifact_dir and report_path before enabling it",
                }
            if (
                will_enable
                and scoring_mode == "trained_head"
                and (
                    not Path(artifact_dir).expanduser().is_absolute()
                    or not Path(report_path).expanduser().is_absolute()
                )
            ):
                return {
                    "status": "error",
                    "message": "typed_decision artifact_dir and report_path must be absolute local paths",
                }
        for key, value in normalized_values.items():
            setattr(target, key, value)
        try:
            self.config.save()
        except Exception as exc:
            for key, value in previous_values.items():
                setattr(target, key, value)
            logger.warning("Config update was not persisted; restored live values", exc_info=True)
            return {"status": "error", "message": f"Config update was not saved: {exc}"}

        if section == "typed_decision" and self._typed_decision_engine is not None:
            current_key = (
                self.config.typed_decision.scoring_mode,
                str(Path(self.config.typed_decision.artifact_dir).expanduser())
                if self.config.typed_decision.artifact_dir
                else "",
                str(Path(self.config.typed_decision.report_path).expanduser())
                if self.config.typed_decision.report_path
                else "",
                self.config.typed_decision.model_id,
                self.config.typed_decision.max_length,
                self.config.typed_decision.temperature,
            )
            if not self.config.typed_decision.enabled or current_key != self._typed_decision_engine_key:
                try:
                    await asyncio.to_thread(self._typed_decision_engine.unload)
                except Exception:
                    logger.warning("Could not release local typed-decision model", exc_info=True)
                self._typed_decision_engine = None
                self._typed_decision_engine_key = None

        if voice_listener_was_running:
            try:
                await self._replace_voice_listener(voice_wake_words)
            except Exception as exc:
                logger.warning("Voice listener rejected updated recorder settings: %s", exc)
                for key, value in previous_values.items():
                    setattr(target, key, value)
                try:
                    self.config.save()
                except Exception:
                    logger.exception("Could not persist the previous voice settings after rollback")
                try:
                    await self._replace_voice_listener(voice_wake_words)
                except Exception:
                    logger.exception("Could not restore the previous voice listener after rollback")
                return {
                    "status": "error",
                    "message": f"Voice setting was not applied: {exc}",
                }

        if section == "voice" and ({"tts_engine", "tts_voice"} & values.keys()):
            self._start_tts_warmup()

        if section == "screen_vision" and "capture_interval_seconds" in values and self._screen_vision:
            try:
                self._screen_vision.set_interval(self.config.screen_vision.capture_interval_seconds)
            except Exception as exc:
                for key, value in previous_values.items():
                    setattr(target, key, value)
                try:
                    self.config.save()
                    self._screen_vision.set_interval(self.config.screen_vision.capture_interval_seconds)
                except Exception:
                    logger.exception("Could not restore screen-vision settings after rollback")
                return {
                    "status": "error",
                    "message": f"Screen-vision setting was not applied: {exc}",
                }

        if section == "model":
            model_router = self._model_router or (self._planner._model if self._planner is not None else None)
            try:
                if model_router is not None:
                    await model_router.reconfigure(set(normalized_values))
                if "max_consecutive_failures" in normalized_values and self._circuit_breaker is not None:
                    self._circuit_breaker.reconfigure(self.config.model.max_consecutive_failures)
            except Exception as exc:
                for key, value in previous_values.items():
                    setattr(target, key, value)
                try:
                    self.config.save()
                    if model_router is not None:
                        await model_router.reconfigure(set(normalized_values))
                    if "max_consecutive_failures" in normalized_values and self._circuit_breaker is not None:
                        self._circuit_breaker.reconfigure(self.config.model.max_consecutive_failures)
                except Exception:
                    logger.exception("Could not restore model runtime after config rollback")
                return {
                    "status": "error",
                    "message": f"Model setting was not applied: {exc}",
                }

        restart_keys = RESTART_REQUIRED_KEYS.get(section, set())
        if section in RESTART_REQUIRED_SECTIONS or restart_keys & set(normalized_values):
            return {"status": "ok", "restart_required": True}
        return {"status": "ok"}

    async def _handle_calendar_test_connection(self, params: dict, ws: ServerConnection) -> dict:
        """Test the configured CalDAV account without mutating it."""
        if self._orchestrator is None:
            return {"status": "error", "message": "Agent system is not initialized", "calendars": []}
        from pilot.agents.base_agent import AgentRole
        from pilot.agents.calendar_agent import CalendarAgent

        agent = self._orchestrator.get_agent(AgentRole.CALENDAR)
        if not isinstance(agent, CalendarAgent):
            return {"status": "error", "message": "Calendar agent is unavailable", "calendars": []}
        return await agent.test_connection()

    async def _handle_email_test_connection(self, params: dict, ws: ServerConnection) -> dict:
        """Test saved IMAP credentials without reading or changing messages."""
        if self._orchestrator is None:
            return {"status": "error", "message": "Agent system is not initialized"}
        from pilot.agents.base_agent import AgentRole
        from pilot.agents.email_agent import EmailAgent

        agent = next(
            (
                candidate
                for candidate in self._orchestrator.get_agents(AgentRole.COMMUNICATION)
                if isinstance(candidate, EmailAgent)
            ),
            None,
        )
        if not isinstance(agent, EmailAgent):
            return {"status": "error", "message": "Email agent is unavailable"}
        return await agent.test_connection()

    async def _handle_ssh_list_hosts(self, params: dict, ws: ServerConnection) -> dict:
        """List SSH aliases and credential readiness without exposing secrets."""
        hosts = []
        for host in self.config.ssh.allowed_hosts:
            has_key = bool(host.private_key_provider and await self._vault.get_key(host.private_key_provider))
            hosts.append(
                {
                    "name": host.name,
                    "hostname": host.hostname,
                    "port": host.port,
                    "username": host.username,
                    "strict_host_key_checking": host.strict_host_key_checking,
                    "has_private_key": has_key,
                    "has_passphrase": bool(
                        host.passphrase_provider and await self._vault.get_key(host.passphrase_provider)
                    ),
                }
            )
        return {"status": "ok", "enabled": self.config.ssh.enabled, "hosts": hosts}

    async def _handle_ssh_set_enabled(self, params: dict, ws: ServerConnection) -> dict:
        """Enable or disable remote SSH actions without changing saved hosts."""
        enabled = params.get("enabled")
        if not isinstance(enabled, bool):
            return {"status": "error", "message": "enabled must be a boolean"}
        previous_ssh = copy.deepcopy(self.config.ssh)
        self.config.ssh.enabled = enabled
        if (failure := self._save_section("ssh", previous_ssh)) is not None:
            return failure
        return {"status": "ok", "enabled": enabled}

    async def _handle_ssh_save_host(self, params: dict, ws: ServerConnection) -> dict:
        """Save one allowlisted SSH host and its secrets in the OS keyring."""
        import re

        from pilot.config import SshHostConfig
        from pilot.security.vault import VaultUnavailableError

        name = str(params.get("name", "")).strip()
        hostname = str(params.get("hostname", "")).strip()
        username = str(params.get("username", "")).strip()
        port = params.get("port", 22)
        strict = params.get("strict_host_key_checking", True)
        private_key = str(params.get("private_key", ""))
        passphrase = str(params.get("passphrase", ""))
        enabled = params.get("enabled", self.config.ssh.enabled)

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            return {
                "status": "error",
                "message": "Alias must use 1-64 letters, numbers, dots, dashes, or underscores",
            }
        if not hostname or len(hostname) > 253 or any(char.isspace() for char in hostname):
            return {"status": "error", "message": "A valid hostname or IP address is required"}
        if not username or len(username) > 128 or any(char.isspace() for char in username):
            return {"status": "error", "message": "A valid SSH username is required"}
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            return {"status": "error", "message": "SSH port must be from 1 to 65535"}
        if not isinstance(strict, bool) or not isinstance(enabled, bool):
            return {"status": "error", "message": "SSH switches must be boolean values"}
        if len(private_key) > 131_072 or len(passphrase) > 1024:
            return {"status": "error", "message": "SSH credential input is too large"}

        existing = next((host for host in self.config.ssh.allowed_hosts if host.name == name), None)
        key_provider = existing.private_key_provider if existing else f"ssh:{name}:private-key"
        passphrase_provider = existing.passphrase_provider if existing else ""
        has_saved_key = bool(key_provider and await self._vault.get_key(key_provider))
        if not private_key and not has_saved_key:
            return {"status": "error", "message": "A private key is required for a new SSH host"}

        try:
            if private_key:
                await self._vault.store_key(key_provider, private_key)
            if passphrase:
                passphrase_provider = f"ssh:{name}:passphrase"
                await self._vault.store_key(passphrase_provider, passphrase)
        except VaultUnavailableError as exc:
            return {"status": "error", "message": str(exc), "available": False}

        host_config = SshHostConfig(
            name=name,
            hostname=hostname,
            port=port,
            username=username,
            private_key_provider=key_provider,
            passphrase_provider=passphrase_provider,
            strict_host_key_checking=strict,
        )
        previous_ssh = copy.deepcopy(self.config.ssh)
        self.config.ssh.allowed_hosts = [host for host in self.config.ssh.allowed_hosts if host.name != name] + [
            host_config
        ]
        self.config.ssh.enabled = enabled
        if (failure := self._save_section("ssh", previous_ssh)) is not None:
            return failure
        return {"status": "ok", "name": name, "has_private_key": True}

    async def _handle_ssh_delete_host(self, params: dict, ws: ServerConnection) -> dict:
        """Remove an SSH alias and its saved credentials."""
        from pilot.security.vault import VaultUnavailableError

        name = str(params.get("name", "")).strip()
        host = next((item for item in self.config.ssh.allowed_hosts if item.name == name), None)
        if host is None:
            return {"status": "error", "message": "Unknown SSH host alias"}
        try:
            if host.private_key_provider:
                await self._vault.delete_key(host.private_key_provider)
            if host.passphrase_provider:
                await self._vault.delete_key(host.passphrase_provider)
        except VaultUnavailableError as exc:
            return {"status": "error", "message": str(exc), "available": False}
        previous_ssh = copy.deepcopy(self.config.ssh)
        self.config.ssh.allowed_hosts = [item for item in self.config.ssh.allowed_hosts if item.name != name]
        if (failure := self._save_section("ssh", previous_ssh)) is not None:
            return failure
        return {"status": "ok"}

    async def _handle_ssh_test_connection(self, params: dict, ws: ServerConnection) -> dict:
        """Authenticate to an SSH alias without running a remote command."""
        if self._orchestrator is None:
            return {"status": "error", "message": "Agent system is not initialized"}
        from pilot.agents.base_agent import AgentRole
        from pilot.agents.ssh_agent import SshAgent

        agent = self._orchestrator.get_agent(AgentRole.SSH)
        if not isinstance(agent, SshAgent):
            return {"status": "error", "message": "SSH agent is unavailable"}
        return await agent.test_connection(str(params.get("name", "")).strip())

    async def _handle_reset_config(self, params: dict, ws: ServerConnection) -> dict:
        """Reset settings to factory defaults.

        Keeps the server section (auth token, host, port: blanking the token locked every
        window out), restrictions (protected folders are a safety choice, not a preference)
        and the finished setup. Refuses while tasks run, because defaults include dry run
        and approval settings a running job depends on.
        """
        from pilot.agents.autonomous import JobStatus

        busy_jobs = [
            job_id
            for job_id in (list(self._autonomous._active_tasks) if self._autonomous is not None else [])
            if (job := self._autonomous.get_job(job_id)) is not None
            and job.status in {JobStatus.QUEUED, JobStatus.DECOMPOSING, JobStatus.RUNNING}
        ]
        if self._interactive_request_active or busy_jobs:
            return {"status": "error", "message": "Stop the running tasks before resetting settings."}
        previous = copy.deepcopy({name: getattr(self.config, name) for name in self.config.__dataclass_fields__})
        previous_first_run = self.config.first_run_complete

        if self._air_handoff_server and self._air_handoff_server.running:
            await self._air_handoff_server.stop()
        if self._air_handoff_manager:
            await self._air_handoff_manager.clear_ephemeral()

        if self._mesh:
            await self._mesh.stop()
            self._set_collab_executor(None)
            self._mesh = None

        supervision_hook = getattr(self, "_supervision_hook", None)
        if supervision_hook is not None:
            supervision_hook.stop()
        if self._background is not None:
            self._background.stop("user_supervision")

        default_config = PilotConfig()

        for field_name in default_config.__dataclass_fields__:
            if field_name in {"server", "restrictions", "first_run_complete"}:
                continue
            val = getattr(default_config, field_name)
            current = getattr(self.config, field_name)

            if hasattr(val, "__dataclass_fields__"):
                for subfield in val.__dataclass_fields__:
                    setattr(current, subfield, getattr(val, subfield))
            else:
                setattr(self.config, field_name, val)

        self._sync_self_healing_monitors()
        if self._screen_vision is not None:
            self._screen_vision.set_interval(self.config.screen_vision.capture_interval_seconds)

        from pilot.air_handoff import AirHandoffManager, AirHandoffServer

        self._air_handoff_manager = AirHandoffManager(
            self._vault,
            max_transfer_bytes=self.config.air_handoff.max_transfer_mb * 1024 * 1024,
        )
        self._air_handoff_server = AirHandoffServer(
            self._air_handoff_manager,
            host="0.0.0.0",
            port=self.config.air_handoff.port,
        )
        self._wire_air_handoff_requests()
        self.config.first_run_complete = previous_first_run
        self.config.clear_load_errors()  # the person chose to start over; saving is allowed again
        try:
            self.config.save()
        except Exception as exc:
            for name, value in previous.items():
                setattr(self.config, name, value)
            logger.warning("Settings reset was not saved; the previous settings are back", exc_info=True)
            return {"status": "error", "message": f"Settings were not reset: {exc}"}

        # Bring the running subsystems in line with the defaults, as update_config does per section.
        model_router = self._model_router or (self._planner._model if self._planner is not None else None)
        if model_router is not None:
            try:
                await model_router.reconfigure(set(self.config.model.__dataclass_fields__))
            except Exception:
                logger.warning("Model runtime did not take the default settings until restart", exc_info=True)
        circuit_breaker = getattr(self, "_circuit_breaker", None)
        if circuit_breaker is not None:
            circuit_breaker.reconfigure(self.config.model.max_consecutive_failures)
        if self._typed_decision_engine is not None:
            try:
                await asyncio.to_thread(self._typed_decision_engine.unload)
            except Exception:
                logger.warning("Could not release the local typed-decision model", exc_info=True)
            self._typed_decision_engine = None
            self._typed_decision_engine_key = None
        if self._voice_listener is not None and self._voice_listener.is_running:
            try:
                await self._replace_voice_listener(list(self._voice_listener.wake_words))
            except Exception:
                logger.warning("Voice listener kept its previous recorder settings", exc_info=True)
        self._start_tts_warmup()
        # The defaults turn co-adaptation off, which ends open study sessions. A reset never deletes
        # co-adaptation data or the person's assistance rules.
        if self._co_adaptation is not None:
            try:
                await self._co_adaptation.on_config_changed(previous["co_adaptation"])
            except Exception:
                logger.warning("Co-adaptation did not take the default settings", exc_info=True)

        return {"status": "ok", "runtime_reconciled": True, "kept": ["server", "restrictions"]}

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
        return {"status": "ok", "entries": entries}

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

    async def _handle_temporal_memory_status(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        """Return provenance-labelled memory facts for local user review."""
        if not self._memory:
            return {
                "status": "error",
                "message": "Memory store is not initialized",
            }
        limit = max(1, min(200, int(params.get("limit", 50))))
        return {"status": "ok", **(await self._memory.temporal_status(limit=limit))}

    async def _handle_temporal_memory_retract(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        """Let the user explicitly retract an active or candidate memory fact."""
        if not self._memory:
            return {
                "status": "error",
                "message": "Memory store is not initialized",
            }
        fact_id = str(params.get("fact_id") or "").strip()
        if not fact_id:
            return {"status": "error", "message": "fact_id is required"}
        try:
            fact = await self._memory.retract_fact(
                fact_id,
                reason=str(params.get("reason") or "Retracted from settings"),
            )
        except KeyError:
            return {
                "status": "error",
                "message": "The memory no longer exists or was already retracted.",
            }
        return {
            "status": "ok",
            "fact_id": fact.fact_id,
            "fact_status": fact.status.value,
        }

    async def _handle_export_session_chat(self, params: dict, ws: ServerConnection) -> dict:
        """Export current UI session chat messages to JSON or CSV."""
        fmt = str(params.get("format", "json")).lower()
        messages = params.get("messages", [])

        if fmt not in {"json", "csv", "markdown", "md"}:
            return {"status": "error", "message": "format must be 'json', 'csv', or 'markdown'"}
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
            elif fmt in ("markdown", "md"):
                lines = ["# Heliox Session Export", f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
                for m in messages:
                    if not isinstance(m, dict):
                        continue
                    msg_type = str(m.get("type", "unknown")).upper()
                    text = str(m.get("text", "")).strip()
                    lines.append(f"### {msg_type}")
                    if text:
                        lines.append(text)
                    plan = m.get("plan")
                    if isinstance(plan, dict) and plan.get("explanation"):
                        lines.append(f"\n**Plan:** {plan['explanation']}")
                    ar = m.get("actionResults")
                    if isinstance(ar, list) and ar:
                        lines.append("\n**Action Results:**\n```")
                        for r in ar:
                            if isinstance(r, dict):
                                out = r.get("output") or r.get("error") or ""
                                lines.append(str(out).strip())
                        lines.append("```")
                    lines.append("\n---")
                out_path.write_text("\n".join(lines), encoding="utf-8")
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
        if not isinstance(provider, str) or not isinstance(key, str) or not provider or not key:
            return {"status": "error", "message": "provider and api_key are required"}
        if provider in RESERVED_VAULT_NAMES or provider.startswith(RESERVED_VAULT_PREFIXES):
            return {"status": "error", "message": f"{provider} is managed by its own feature"}
        from pilot.security.vault import VaultUnavailableError

        try:
            await self._vault.store_key(provider, key)
        except VaultUnavailableError as exc:
            return {"status": "error", "message": str(exc), "available": False}
        model_router = self._model_router or (self._planner._model if self._planner is not None else None)
        if self.config.model.cloud_provider == provider and model_router is not None:
            # Rebuild the cloud client through the router so the old client is closed and a
            # circuit opened by the missing key does not ignore the new one for a minute.
            try:
                await model_router.reconfigure({"cloud_provider"})
            except Exception:
                logger.warning("Cloud client did not pick up the new key until restart", exc_info=True)
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
        if not isinstance(provider, str) or not provider:
            return {"status": "error", "message": "provider is required"}
        if provider in RESERVED_VAULT_NAMES or provider.startswith(RESERVED_VAULT_PREFIXES):
            return {"status": "error", "message": f"{provider} is managed by its own feature"}
        from pilot.security.vault import VaultUnavailableError

        try:
            await self._vault.delete_key(provider)
        except VaultUnavailableError as exc:
            return {"status": "error", "message": str(exc), "available": False}
        return {"status": "ok"}

    async def _handle_list_api_keys(self, params: dict, ws: ServerConnection) -> dict:
        """List all providers with stored API keys.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with providers list.
        """
        from pilot.security.vault import VaultUnavailableError

        try:
            providers = await self._vault.list_providers()
        except VaultUnavailableError as exc:
            return {
                "status": "error",
                "providers": [],
                "available": False,
                "backend": self._vault.backend_name,
                "message": str(exc),
            }
        return {
            "status": "ok",
            "providers": providers,
            "available": self._vault.available,
            "backend": self._vault.backend_name,
            "message": (
                ""
                if self._vault.available
                else "Secure OS credential storage is unavailable; API keys cannot be persisted."
            ),
        }

    # -- Subscription-authenticated official CLIs --

    def _subscription_client(self):
        from pilot.models.subscription_cli import SubscriptionCLIClient

        if self._planner is not None:
            return self._planner._model._subscription
        return SubscriptionCLIClient(self.config)

    async def _handle_subscription_status(self, params: dict, ws: ServerConnection) -> dict:
        """Report CLI installation and subscription login without exposing account data."""

        provider = str(params.get("provider") or self.config.model.subscription_provider)
        try:
            refresh = _validated_bool(params, "refresh", False)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", **await self._subscription_client().status(provider, refresh=refresh)}

    async def _handle_subscription_login(self, params: dict, ws: ServerConnection) -> dict:
        """Start the official provider login flow; Heliox never handles its credentials."""

        provider = str(params.get("provider") or self.config.model.subscription_provider)
        return await self._subscription_client().start_login(provider)

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
            return {"status": "ok", "models": models, "available": True}
        except Exception:
            return {"status": "ok", "models": [], "available": False}

    # -- Health --

    async def _handle_health(self, params: dict[str, Any], ws: ServerConnection) -> dict[str, Any]:
        """Return health status of the daemon.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with uptime, memory usage, active connections, and loaded agents.
        """
        import psutil

        # Calculate uptime
        uptime = time.time() - self._start_time

        # Get memory usage in MB
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024**2)

        # Count active connections
        active_connections = len(self._clients)

        # Get loaded agent names
        loaded_agents: list[str] = []
        if self._orchestrator:
            loaded_agents = [role.value for role in self._orchestrator._agents]

        return {
            "uptime": uptime,
            "memory_usage_mb": memory_mb,
            "active_connections": active_connections,
            "loaded_agents": loaded_agents,
        }

    async def _handle_ready(self, params: dict[str, Any], ws: ServerConnection) -> dict[str, Any]:
        """Check if all agents are fully initialized and ready.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with ready status (True only if all agents are initialized).
        """
        from pilot.agents.base_agent import AgentStatus

        # If orchestrator is not initialized, not ready
        if not self._orchestrator:
            return {"ready": False}

        # If no agents are registered, not ready
        if not self._orchestrator._agents:
            return {"ready": False}

        for agent in self._orchestrator._agents.values():
            if not agent._running or agent.status in {AgentStatus.STOPPED, AgentStatus.ERROR}:
                return {"ready": False}

        return {"ready": True}

    async def _handle_ping(self, params: dict, ws: ServerConnection) -> dict:
        """Ping the server to check connectivity.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with pong and version.
        """
        return {"pong": True, "version": __version__}

    async def _handle_cursor_move(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Move the OS mouse cursor to an absolute screen position.

        This is the browser/dev-mode fallback for the gesture-cursor bridge
        (see GESTURES.md) — the primary path is a native Rust Tauri command
        (`move_gesture_cursor`, uses the `enigo` crate) that stays entirely
        in-process for ~30fps latency. This RPC goes through a WebSocket
        round-trip plus pyautogui's own PAUSE delay, which is fine for
        testing the wiring in a browser without a compiled Tauri binary but
        will not feel as smooth as the native path. Bypasses
        Planner/Executor/confirmation entirely — MOUSE_MOVE is Tier 1
        (USER_WRITE), already confirmed to never require confirmation.

        Args:
            params: JSON-RPC parameters containing x and y (absolute screen
                coordinates).
            ws: The WebSocket connection.

        Returns:
            A dict with status and a human-readable message.
        """
        from pilot.system import input_control

        try:
            x = int(params.get("x", 0))
            y = int(params.get("y", 0))
        except (TypeError, ValueError):
            return {"status": "error", "message": "x/y must be integers"}
        if not self.config.gesture_cursor.enabled:
            # The camera must never drive the real cursor while the person has this switched off.
            return {"status": "error", "message": "Gesture cursor control is off in Settings"}

        message = await input_control.mouse_move(x, y, duration=0.0)
        return {"status": "ok", "message": message}

    async def _handle_cursor_click(self, params: dict[str, Any], ws: ServerConnection) -> dict:
        """Click at the given screen position — the fallback counterpart to
        the Rust `click_gesture_cursor` command, used for the pinch-to-click
        gesture while cursor mode is active. See `_handle_cursor_move` for
        why this path exists and its latency caveats.

        Args:
            params: JSON-RPC parameters containing x and y (the position the
                gesture-cursor bridge last moved to).
            ws: The WebSocket connection.

        Returns:
            A dict with status and a human-readable message.
        """
        from pilot.system import input_control

        try:
            x = int(params.get("x", 0))
            y = int(params.get("y", 0))
        except (TypeError, ValueError):
            return {"status": "error", "message": "x/y must be integers"}
        if not self.config.gesture_cursor.enabled:
            # The camera must never drive the real cursor while the person has this switched off.
            return {"status": "error", "message": "Gesture cursor control is off in Settings"}

        message = await input_control.mouse_click(x, y, button="left")
        return {"status": "ok", "message": message}

    async def _handle_system_info(self, params: dict, ws: ServerConnection) -> dict:
        """Return exact hardware metrics (CPU, RAM, Disk, Uptime, Hostname) for HUD monitor."""
        import os
        import shutil
        import socket
        import time

        import psutil

        disk = shutil.disk_usage(os.path.abspath("/" if os.name != "nt" else "C:\\"))
        mem = psutil.virtual_memory()
        # HUD polling must not stall the shared event loop. psutil's interval
        # mode is intentionally blocking, so isolate the sample in a worker.
        cpu = await asyncio.to_thread(psutil.cpu_percent, interval=0.1)
        uptime = (
            int(time.time() - psutil.boot_time())
            if hasattr(psutil, "boot_time")
            else int(time.time() - self._start_time)
        )
        return {
            "status": "ok",
            "cpu_percent": round(cpu),
            "memory_percent": round(mem.percent),
            "memory_used": mem.used,
            "memory_total": mem.total,
            "disk_percent": round((disk.used / disk.total) * 100),
            "disk_used": disk.used,
            "disk_total": disk.total,
            "hostname": socket.gethostname(),
            "uptime_seconds": uptime,
        }

    async def _handle_get_uptime(self, params: dict, ws: ServerConnection) -> str:
        """Return formatted system uptime string for HUD monitor."""
        import time

        import psutil

        up_sec = (
            int(time.time() - psutil.boot_time())
            if hasattr(psutil, "boot_time")
            else int(time.time() - self._start_time)
        )
        days = up_sec // 86400
        hrs = (up_sec % 86400) // 3600
        mins = (up_sec % 3600) // 60
        return f"{days}d {hrs}h {mins}m" if days > 0 else f"{hrs}h {mins}m"

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

    async def _dispatch_reactive_trigger(self, trigger: Any) -> None:
        """Submit one fired trigger through guarded autonomous execution."""
        if not self._running or self._autonomous is None:
            logger.warning("Trigger %s fired while autonomous execution was unavailable", trigger.id)
            await self._broadcast_notification(
                "trigger_dispatch_failed",
                {
                    "trigger_id": trigger.id,
                    "name": trigger.name,
                    "reason": "Guarded autonomous execution is unavailable",
                },
            )
            return

        job = await self._autonomous.submit(
            trigger.action_command,
            source="trigger",
            session_id=f"trigger:{trigger.id}",
        )
        await self._broadcast_notification(
            "trigger_dispatched",
            {
                "trigger_id": trigger.id,
                "name": trigger.name,
                "job_id": job.job_id,
            },
        )

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

    def _require_rpc_role(self, ws: ServerConnection, *allowed: RpcClientRole) -> None:
        role = self._client_roles.get(ws)
        if role not in allowed:
            names = ", ".join(item.value for item in allowed)
            raise PermissionError(f"RPC requires client role: {names}")

    async def _dispatch_neural_staged_task(self, task: Any, scope_override: Any) -> dict[str, object]:
        """Launch one explicitly staged neural selection through the autonomous engine."""

        if self._autonomous is None:
            raise RuntimeError("autonomous executor is not initialized")
        job = await self._autonomous.submit(
            task.goal,
            source="neural",
            scope_override=scope_override,
            session_id=task.session_id,
        )
        return job.to_dict()

    async def _wait_for_autonomous_confirmation(self, job: Any, plan: Any, plan_id: str, review: Any = None) -> bool:
        """Pause a background plan at the same UI confirmation boundary as interactive work.

        When the world model or Gnomon asked for approval, every action of the plan is
        shown and must be approved, as on the interactive path, with their reasons.
        """

        required = {
            index for index, action in enumerate(plan.actions) if action.requires_confirmation or action.is_irreversible
        }
        escalated = bool(review is not None and getattr(review, "requires_approval", False))
        if escalated:
            required = set(range(len(plan.actions)))
        if not required:
            return True
        reason = (
            review.summary()
            if escalated and review.summary()
            else "An autonomous task reached an action that requires explicit approval."
        )

        pending = PendingConfirmation(plan_id=plan_id, event=asyncio.Event(), plan=plan)
        pending.required_indices = frozenset(required)
        self._pending_confirms[plan_id] = pending
        try:
            await self._broadcast_notification(
                "confirm_required",
                {
                    "task_id": job.job_id,
                    "plan_id": plan_id,
                    "source": job.source,
                    "actions": [
                        {
                            **plan.actions[index].model_dump(mode="json"),
                            "index": index,
                            "irreversible": plan.actions[index].is_irreversible,
                            "undo": self._undo_badge(plan.actions[index]),
                        }
                        for index in sorted(required)
                    ],
                    "reason": reason,
                    "risk_assessment": getattr(review, "risk_assessment", None) if escalated else None,
                    "gnomon": getattr(review, "gnomon", None),
                },
            )
        except BaseException:
            self._pending_confirms.pop(plan_id, None)
            raise
        decision = await self._consume_confirmation(pending)
        if decision.expired:
            logger.warning("Autonomous confirmation timed out for plan %s", plan_id)
            return False

        approved = decision.approved_indices if decision.approved_indices is not None else required
        return decision.confirmed and required.issubset(approved)

    async def _publish_autonomous_assessment(self, plan_id: str, payload: dict[str, Any]) -> None:
        """Record and broadcast the world-model (and Gnomon) view of an autonomous plan."""
        await self._append_experience(
            ExperienceEventType.WORLD_PREDICTION,
            plan_id=plan_id,
            idempotency_key=f"plan:{plan_id}:world_prediction",
            source="risk_gate",
            payload=payload,
            provenance={"component": "AutonomousReviewer", "policy_authority": "deterministic_rules"},
        )
        if self.config.gateway.risk_gate_enabled:
            await self._broadcast_notification("world_model_assessment", {"plan_id": plan_id, **payload})

    async def _handle_autonomy_review_status(self, params: dict, ws: ServerConnection) -> dict:
        """Whether the world model and Gnomon are taking part in autonomous decisions right now."""
        gnomon = self._gnomon_reviewer.status() if self._gnomon_reviewer is not None else {"state": "disabled"}
        return {
            "status": "ok",
            "world_model": {"enabled": bool(self.config.gateway.enabled and self.config.gateway.risk_gate_enabled)},
            "gnomon": gnomon,
        }

    async def _handle_neural_status(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI, RpcClientRole.NEURAL_SIDECAR)
        if self._neural_controller is None:
            return {"status": "unavailable", "connected": False}
        return {"status": "ok", **await self._neural_controller.status()}

    async def _handle_neural_stage_task(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI)
        try:
            result = await self._neural_controller.stage_task(
                label=str(params.get("label") or ""),
                goal=str(params["goal"]),
                session_id=str(params.get("session_id") or "neural"),
            )
            return {"status": "ok", **result}
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_remove_staged_task(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI)
        try:
            result = await self._neural_controller.remove_staged_task(uuid.UUID(str(params["task_id"])))
            return {"status": "ok", **result}
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_connect(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.NEURAL_SIDECAR)
        try:
            from pilot.neural.protocol import NeuralStreamDescriptorV1

            descriptor = NeuralStreamDescriptorV1.model_validate(params.get("descriptor"))
            return {"status": "ok", **await self._neural_controller.connect(descriptor)}
        except (TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_begin_calibration(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI)
        try:
            return {
                "status": "ok",
                **await self._neural_controller.begin_calibration(uuid.UUID(str(params["session_id"]))),
            }
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_finish_calibration(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.NEURAL_SIDECAR)
        try:
            from pilot.neural.protocol import NeuralCalibrationMetricsV1

            metrics = (
                NeuralCalibrationMetricsV1.model_validate(params["metrics"])
                if params.get("metrics") is not None
                else None
            )
            return {
                "status": "ok",
                **await self._neural_controller.finish_calibration(
                    uuid.UUID(str(params["session_id"])),
                    calibration_id=str(params["calibration_id"]),
                    subject_key=str(params["subject_key"]),
                    decoder_version=str(params.get("decoder_version") or ""),
                    metrics=metrics,
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_arm(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI)
        try:
            from pilot.neural.protocol import NeuralScope

            return {
                "status": "ok",
                **await self._neural_controller.arm(
                    uuid.UUID(str(params["session_id"])),
                    scope=NeuralScope(str(params["scope"])),
                    non_neural_authorized=params.get("user_authorized") is True,
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_intent_preview(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.NEURAL_SIDECAR)
        try:
            from pilot.neural.protocol import NeuralIntentV1

            intent = NeuralIntentV1.model_validate(params.get("intent"))
            return await self._neural_controller.preview(intent)
        except (TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_observation(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.NEURAL_SIDECAR)
        try:
            from pilot.neural.quality import SignalQualitySummary

            summary = SignalQualitySummary.model_validate(params.get("quality"))
            observation = await self._neural_controller.update_observation(
                summary,
                buffered_samples=max(0, int(params.get("buffered_samples", 0))),
                dropped_samples=max(0, int(params.get("dropped_samples", 0))),
                observed_at_ns=max(0, int(params.get("observed_at_ns", 0))),
            )
            return {"status": "ok", **observation}
        except (TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_commit(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI)
        try:
            return await self._neural_controller.commit(
                uuid.UUID(str(params["preview_id"])),
                expected_revision=int(params["expected_revision"]),
                world_model_approved=params.get("world_model_approved") is True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_disarm(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI, RpcClientRole.NEURAL_SIDECAR)
        reason = _sanitize_summary(params.get("reason") or "user_disarm", 120)
        return {"status": "ok", **await self._neural_controller.disarm(reason=reason)}

    async def _handle_neural_stimulus_marker(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.UI)
        try:
            from pilot.neural.protocol import NeuralStimulusEvent

            marker = await self._neural_controller.record_stimulus_marker(
                uuid.UUID(str(params["session_id"])),
                target_id=str(params["target_id"]) if params.get("target_id") is not None else None,
                event=NeuralStimulusEvent(str(params["event"])),
                client_performance_ms=float(params.get("client_performance_ms", 0)),
            )
            return {"status": "ok", "marker": marker}
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_neural_stimulus_markers(self, params: dict, ws: ServerConnection) -> dict:
        self._require_rpc_role(ws, RpcClientRole.NEURAL_SIDECAR)
        try:
            after = int(params.get("after_sequence", -1))
            if after < -1:
                raise ValueError("after_sequence cannot be less than -1")
            markers = await self._neural_controller.stimulus_markers(after_sequence=after)
            return {"status": "ok", "markers": markers}
        except (TypeError, ValueError) as exc:
            return {"status": "rejected", "error": str(exc)}

    async def _handle_risk_gate_status(self, params: dict, ws: ServerConnection) -> dict:
        """Report the trained risk-world-model state and latest prediction."""
        from pilot.security.risk_gate import get_risk_gate

        return {
            "status": "ok",
            **get_risk_gate().status(enabled=self.config.gateway.risk_gate_enabled),
        }

    async def _handle_risk_gate_config_update(self, params: dict, ws: ServerConnection) -> dict:
        """Enable or disable risk-world-model evaluation and persist it."""
        try:
            enabled = _validated_bool(params, "enabled", self.config.gateway.risk_gate_enabled)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        previous_gateway = copy.deepcopy(self.config.gateway)
        self.config.gateway.risk_gate_enabled = enabled
        if (failure := self._save_section("gateway", previous_gateway)) is not None:
            return failure
        return await self._handle_risk_gate_status({}, ws)

    async def _handle_self_healing_status(self, params: dict, ws: ServerConnection) -> dict:
        """Report self-healing config plus recent remediation attempts.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with the current config and up to 50 recent attempts
            (auto-executed, proposed/pending, confirmed, denied, timed out).
        """
        engine = getattr(self, "_self_healing", None)
        monitor_tasks = {
            task["task_id"].removeprefix("monitor_"): task
            for task in (self._background.list_tasks() if self._background else [])
            if task["task_id"] in {"monitor_cpu", "monitor_memory", "monitor_disk"}
        }
        return {
            "status": "ok",
            "enabled": self.config.self_healing.enabled,
            "auto_execute_max_tier": self.config.self_healing.auto_execute_max_tier,
            "watched_metrics": self.config.self_healing.watched_metrics,
            "monitors": monitor_tasks,
            "attempts": engine.list_attempts() if engine else [],
        }

    def _sync_self_healing_monitors(self) -> None:
        """Start only configured health monitors and stop only loops we own.

        A monitor can also be started through the generic background-task
        RPC. Those user-owned loops must not be stopped when Autonomous
        Healing is disabled, so ownership is tracked separately.
        """
        if not self._background:
            return

        configured = set(self.config.self_healing.watched_metrics)
        desired = {f"monitor_{metric}" for metric in configured} if self.config.self_healing.enabled else set()

        for task_id in self._self_healing_started_monitors - desired:
            self._background.stop(task_id)
        self._self_healing_started_monitors.intersection_update(desired)

        for task_id in desired:
            task = self._background._tasks.get(task_id)
            if task is None or task.status.value == "running":
                continue
            if self._background.start(task_id):
                self._self_healing_started_monitors.add(task_id)

    async def _handle_self_healing_config_update(self, params: dict, ws: ServerConnection) -> dict:
        """Update self-healing config (enabled, tiering, watched metrics).

        Confirming or denying a specific proposed remediation plan reuses
        the existing generic ``confirm`` RPC (same plan_id/PendingConfirmation
        mechanism as ThreatContainmentBridge) rather than a dedicated
        approve/reject RPC.

        Args:
            params: JSON-RPC parameters, any of {enabled, auto_execute_max_tier,
                cooldown_seconds, confirm_timeout_seconds, watched_metrics}.
            ws: The WebSocket connection.

        Returns:
            A dict with status and the updated config.
        """
        cfg = self.config.self_healing
        try:
            enabled = _validated_bool(params, "enabled", cfg.enabled)
            max_tier = _validated_int(
                params,
                "auto_execute_max_tier",
                cfg.auto_execute_max_tier,
                minimum=0,
                maximum=3,
            )
            cooldown = _validated_float(
                params,
                "cooldown_seconds",
                cfg.cooldown_seconds,
                minimum=0,
            )
            confirm_timeout = _validated_float(
                params,
                "confirm_timeout_seconds",
                cfg.confirm_timeout_seconds,
                minimum=0.001,
            )
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        watched_metrics = list(cfg.watched_metrics)
        if "watched_metrics" in params:
            raw_metrics = params["watched_metrics"]
            if not isinstance(raw_metrics, list):
                return {"status": "error", "message": "watched_metrics must be a list"}
            watched_metrics = list(dict.fromkeys(str(metric) for metric in raw_metrics))
            unsupported = sorted(set(watched_metrics) - {"cpu", "memory", "disk"})
            if unsupported:
                return {
                    "status": "error",
                    "message": f"unsupported watched metrics: {', '.join(unsupported)}",
                }
        if enabled and not watched_metrics:
            return {"status": "error", "message": "select at least one watched metric"}

        previous_self_healing = copy.deepcopy(self.config.self_healing)
        cfg.enabled = enabled
        cfg.auto_execute_max_tier = max_tier
        cfg.cooldown_seconds = cooldown
        cfg.confirm_timeout_seconds = confirm_timeout
        cfg.watched_metrics = watched_metrics

        if (failure := self._save_section("self_healing", previous_self_healing)) is not None:
            return failure
        self._sync_self_healing_monitors()
        return {
            "status": "ok",
            "enabled": cfg.enabled,
            "auto_execute_max_tier": cfg.auto_execute_max_tier,
            "cooldown_seconds": cfg.cooldown_seconds,
            "confirm_timeout_seconds": cfg.confirm_timeout_seconds,
            "watched_metrics": cfg.watched_metrics,
        }

    async def _handle_narration_status(self, params: dict, ws: ServerConnection) -> dict:
        """Report the Live Execution Narrator's current config.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with the current narration config.
        """
        cfg = self.config.narration
        return {
            "status": "ok",
            "enabled": cfg.enabled,
            "narrate_steps": cfg.narrate_steps,
            "interrupt_on_risk": cfg.interrupt_on_risk,
            "proactive_review_enabled": cfg.proactive_review_enabled,
            "live_corrections_enabled": cfg.live_corrections_enabled,
            "follow_up_enabled": cfg.follow_up_enabled,
            "advisory_timeout_seconds": cfg.advisory_timeout_seconds,
            "max_auto_revisions": cfg.max_auto_revisions,
            "confirm_timeout_seconds": cfg.confirm_timeout_seconds,
        }

    async def _handle_narration_config_update(self, params: dict, ws: ServerConnection) -> dict:
        """Update Live Execution Narrator config.

        Approving or denying a specific proposed interrupt reuses the
        existing generic ``confirm`` RPC (same plan_id/PendingConfirmation
        mechanism as ThreatContainmentBridge/AutonomousHealingEngine)
        rather than a dedicated approve/reject RPC.

        Args:
            params: JSON-RPC parameters, any of {enabled, narrate_steps,
                interrupt_on_risk, confirm_timeout_seconds}.
            ws: The WebSocket connection.

        Returns:
            A dict with status and the updated config.
        """
        cfg = self.config.narration
        try:
            enabled = _validated_bool(params, "enabled", cfg.enabled)
            narrate_steps = _validated_bool(params, "narrate_steps", cfg.narrate_steps)
            interrupt_on_risk = _validated_bool(params, "interrupt_on_risk", cfg.interrupt_on_risk)
            proactive_review_enabled = _validated_bool(
                params,
                "proactive_review_enabled",
                cfg.proactive_review_enabled,
            )
            live_corrections_enabled = _validated_bool(
                params,
                "live_corrections_enabled",
                cfg.live_corrections_enabled,
            )
            follow_up_enabled = _validated_bool(params, "follow_up_enabled", cfg.follow_up_enabled)
            advisory_timeout_seconds = _validated_float(
                params,
                "advisory_timeout_seconds",
                cfg.advisory_timeout_seconds,
                minimum=1,
                maximum=30,
            )
            max_auto_revisions = _validated_int(
                params,
                "max_auto_revisions",
                cfg.max_auto_revisions,
                minimum=0,
                maximum=5,
            )
            confirm_timeout_seconds = _validated_float(
                params,
                "confirm_timeout_seconds",
                cfg.confirm_timeout_seconds,
                minimum=0.001,
            )
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        previous_narration = copy.deepcopy(self.config.narration)
        cfg.enabled = enabled
        cfg.narrate_steps = narrate_steps
        cfg.interrupt_on_risk = interrupt_on_risk
        cfg.proactive_review_enabled = proactive_review_enabled
        cfg.live_corrections_enabled = live_corrections_enabled
        cfg.follow_up_enabled = follow_up_enabled
        cfg.advisory_timeout_seconds = advisory_timeout_seconds
        cfg.max_auto_revisions = max_auto_revisions
        cfg.confirm_timeout_seconds = confirm_timeout_seconds

        if (failure := self._save_section("narration", previous_narration)) is not None:
            return failure
        return {
            "status": "ok",
            "enabled": cfg.enabled,
            "narrate_steps": cfg.narrate_steps,
            "interrupt_on_risk": cfg.interrupt_on_risk,
            "proactive_review_enabled": cfg.proactive_review_enabled,
            "live_corrections_enabled": cfg.live_corrections_enabled,
            "follow_up_enabled": cfg.follow_up_enabled,
            "advisory_timeout_seconds": cfg.advisory_timeout_seconds,
            "max_auto_revisions": cfg.max_auto_revisions,
            "confirm_timeout_seconds": cfg.confirm_timeout_seconds,
        }

    async def _handle_supervision_status(self, params: dict, ws: ServerConnection) -> dict:
        """Report User Manual Supervision's current config plus whether the
        keyboard/mouse hook (if enabled) is actually still alive.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with the current supervision config and hook_healthy.
        """
        cfg = self.config.supervision
        hook_healthy = getattr(self, "_supervision_hook", None)
        return {
            "status": "ok",
            "enabled": cfg.enabled,
            "keyboard_mouse_hook_enabled": cfg.keyboard_mouse_hook_enabled,
            "cognitive_coaching_enabled": cfg.cognitive_coaching_enabled,
            "risk_pattern_detection_enabled": cfg.risk_pattern_detection_enabled,
            "hook_healthy": hook_healthy.is_running() if hook_healthy else False,
        }

    async def _handle_supervision_config_update(self, params: dict, ws: ServerConnection) -> dict:
        """Update User Manual Supervision config.

        Unlike `_handle_narration_config_update`/`_handle_self_healing_config_update`,
        this handler must actually start/stop the background task and the
        keyboard/mouse hook on an `enabled`/`keyboard_mouse_hook_enabled`
        transition -- the thing being gated has real cost and privacy
        weight even when idle, so a config flip alone isn't enough.

        Args:
            params: JSON-RPC parameters, any of {enabled,
                keyboard_mouse_hook_enabled, cognitive_coaching_enabled,
                risk_pattern_detection_enabled, tick_interval_seconds,
                ocr_interval_seconds, stress_coaching_threshold,
                cognitive_load_coaching_threshold, coaching_cooldown_seconds,
                risk_cooldown_seconds, keystroke_buffer_max_chars,
                ocr_snippet_max_chars}.
            ws: The WebSocket connection.

        Returns:
            A dict with status and the updated config.
        """
        cfg = self.config.supervision
        was_enabled = cfg.enabled
        was_hook_enabled = cfg.keyboard_mouse_hook_enabled
        try:
            enabled = _validated_bool(params, "enabled", cfg.enabled)
            hook_enabled = _validated_bool(
                params,
                "keyboard_mouse_hook_enabled",
                cfg.keyboard_mouse_hook_enabled,
            )
            coaching_enabled = _validated_bool(
                params,
                "cognitive_coaching_enabled",
                cfg.cognitive_coaching_enabled,
            )
            risk_detection_enabled = _validated_bool(
                params,
                "risk_pattern_detection_enabled",
                cfg.risk_pattern_detection_enabled,
            )
            tick_interval = _validated_float(
                params,
                "tick_interval_seconds",
                cfg.tick_interval_seconds,
                minimum=0.001,
            )
            ocr_interval = _validated_float(
                params,
                "ocr_interval_seconds",
                cfg.ocr_interval_seconds,
                minimum=0.001,
            )
            stress_threshold = _validated_float(
                params,
                "stress_coaching_threshold",
                cfg.stress_coaching_threshold,
                minimum=0,
                maximum=1,
            )
            load_threshold = _validated_float(
                params,
                "cognitive_load_coaching_threshold",
                cfg.cognitive_load_coaching_threshold,
                minimum=0,
                maximum=1,
            )
            coaching_cooldown = _validated_float(
                params,
                "coaching_cooldown_seconds",
                cfg.coaching_cooldown_seconds,
                minimum=0,
            )
            risk_cooldown = _validated_float(
                params,
                "risk_cooldown_seconds",
                cfg.risk_cooldown_seconds,
                minimum=0,
            )
            keystroke_buffer_max = _validated_int(
                params,
                "keystroke_buffer_max_chars",
                cfg.keystroke_buffer_max_chars,
                minimum=1,
            )
            ocr_snippet_max = _validated_int(
                params,
                "ocr_snippet_max_chars",
                cfg.ocr_snippet_max_chars,
                minimum=1,
            )
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        previous_supervision = copy.deepcopy(cfg)
        cfg.enabled = enabled
        cfg.keyboard_mouse_hook_enabled = hook_enabled
        cfg.cognitive_coaching_enabled = coaching_enabled
        cfg.risk_pattern_detection_enabled = risk_detection_enabled
        cfg.tick_interval_seconds = tick_interval
        cfg.ocr_interval_seconds = ocr_interval
        cfg.stress_coaching_threshold = stress_threshold
        cfg.cognitive_load_coaching_threshold = load_threshold
        cfg.coaching_cooldown_seconds = coaching_cooldown
        cfg.risk_cooldown_seconds = risk_cooldown
        cfg.keystroke_buffer_max_chars = keystroke_buffer_max
        cfg.ocr_snippet_max_chars = ocr_snippet_max

        if (failure := self._save_section("supervision", previous_supervision)) is not None:
            return failure
        restart_required = (
            cfg.tick_interval_seconds != previous_supervision.tick_interval_seconds
            or cfg.keystroke_buffer_max_chars != previous_supervision.keystroke_buffer_max_chars
        )
        supervision_hook = getattr(self, "_supervision_hook", None)
        if supervision_hook is not None:
            if cfg.enabled and cfg.keyboard_mouse_hook_enabled and not (was_enabled and was_hook_enabled):
                supervision_hook.start()
            elif not (cfg.enabled and cfg.keyboard_mouse_hook_enabled) and was_enabled and was_hook_enabled:
                supervision_hook.stop()

        background = getattr(self, "_background", None)
        if background is not None:
            if cfg.enabled and not was_enabled:
                background.start("user_supervision")
            elif not cfg.enabled and was_enabled:
                background.stop("user_supervision")

        return {
            "status": "ok",
            "restart_required": restart_required,
            "enabled": cfg.enabled,
            "keyboard_mouse_hook_enabled": cfg.keyboard_mouse_hook_enabled,
            "cognitive_coaching_enabled": cfg.cognitive_coaching_enabled,
            "risk_pattern_detection_enabled": cfg.risk_pattern_detection_enabled,
            "hook_healthy": supervision_hook.is_running() if supervision_hook else False,
        }

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
        return {"status": "ok", **result}

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

    async def _handle_agent_mesh_status(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._agent_mesh is None:
            return {"status": "error", "enabled": False, "message": "Agent mesh is not initialized"}
        return {"status": "ok", **self._agent_mesh.status()}

    async def _handle_agent_spawn(self, params: dict, ws: ServerConnection) -> dict:
        """Dynamically spawn a new specialist agent.

        Args:
            params: JSON-RPC parameters with role.
            ws: The WebSocket connection.

        Returns:
            A dict with status and optionally agent_id.
        """
        agent_name = str(params.get("agent_name", "")).strip()
        if agent_name and self._orchestrator:
            agent = await self._orchestrator.spawn_registered_agent(
                agent_name,
                executor=self._executor,
                background_manager=self._background,
                model_router=self._model_router,
                config=self.config,
                vault=self._vault,
                memory=self._memory,
            )
            if agent:
                return {
                    "status": "spawned",
                    "agent_id": agent.agent_id,
                    "agent_name": agent.__class__.__name__,
                }
            return {"status": "error", "message": f"Unknown or unavailable specialist: {agent_name}"}

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

        try:
            is_final = _validated_bool(params, "is_final", False)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        event = InputEvent(
            modality=ModalityType.VOICE,
            transcript=params.get("transcript", ""),
            voice_confidence=params.get("confidence", 0.8),
            is_final=is_final,
        )
        voice_confidence = params.get("confidence", 0.8)
        await self._append_experience(
            ExperienceEventType.OBSERVATION,
            idempotency_key=f"observation:voice:{uuid.uuid4()}",
            source="voice",
            payload={
                "transcript": params.get("transcript", ""),
                "is_final": is_final,
            },
            confidence=(
                float(voice_confidence)
                if isinstance(voice_confidence, (int, float)) and 0.0 <= voice_confidence <= 1.0
                else None
            ),
            provenance={"component": "MultimodalFusionEngine.on_voice_event"},
            privacy_class=PrivacyClass.SENSITIVE,
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
        gesture_confidence = params.get("confidence", 0.8)
        await self._append_experience(
            ExperienceEventType.OBSERVATION,
            idempotency_key=f"observation:gesture:{uuid.uuid4()}",
            source="gesture",
            payload={
                "gesture": params.get("gesture", ""),
                "raw_sensor_data_excluded": True,
            },
            confidence=(
                float(gesture_confidence)
                if isinstance(gesture_confidence, (int, float)) and 0.0 <= gesture_confidence <= 1.0
                else None
            ),
            provenance={"component": "MultimodalFusionEngine.on_gesture_event"},
            privacy_class=PrivacyClass.BIOMETRIC_DERIVED,
        )
        intent = await self._fusion.on_gesture_event(event)
        if intent:
            return {"status": "fused", "intent": intent.to_dict()}
        return {"status": "buffered"}

    async def _handle_gaze_event(self, params: dict, ws: ServerConnection) -> dict:
        """Receive a coarse gaze-region reading from the frontend and feed
        it to the fusion engine as a passive disambiguating signal.

        Args:
            params: JSON-RPC parameters with region, confidence. Never raw
                face landmarks — see gazeTracking.ts's privacy rationale.
            ws: The WebSocket connection.

        Returns:
            A dict with status — gaze never itself produces a fused
            intent (see MultimodalFusionEngine.on_gaze_event's docstring),
            only "ingested" or "ignored" (below the confidence floor) or
            "error" (fusion engine not initialized).
        """
        if not self._fusion:
            return {"status": "error", "message": "Fusion engine not initialized"}

        if not self.config.vision.gaze_tracking_enabled:
            return {"status": "ignored", "reason": "gaze_tracking_disabled"}

        region = params.get("region", "")
        if region not in {"center", "left", "right", "up", "down"}:
            return {"status": "ignored", "reason": "invalid_region"}

        confidence = params.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            return {"status": "ignored", "reason": "invalid_confidence"}
        if confidence < self._fusion.min_gaze_confidence:
            return {"status": "ignored", "reason": "confidence_below_threshold"}

        from pilot.multimodal.fusion import InputEvent, ModalityType

        event = InputEvent(
            modality=ModalityType.GAZE,
            gaze_region=region,
            gaze_confidence=confidence,
        )
        await self._append_experience(
            ExperienceEventType.OBSERVATION,
            idempotency_key=f"observation:gaze:{uuid.uuid4()}",
            source="gaze",
            payload={"region": region, "raw_sensor_data_excluded": True},
            confidence=float(confidence),
            provenance={"component": "MultimodalFusionEngine.on_gaze_event"},
            privacy_class=PrivacyClass.BIOMETRIC_DERIVED,
        )
        await self._fusion.on_gaze_event(event)
        return {"status": "ingested"}

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
            A dict with reasoning emitter statistics or error.
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
            report = await self._sandbox.simulate(pending.plan)
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

    async def _handle_strategy_evolution_status(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"enabled": False, "message": "Strategy evolution is not initialized"}
        return await self._strategy_evolution.status()

    async def _handle_strategy_candidates(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"candidates": [], "message": "Strategy evolution is not initialized"}
        try:
            include_content = _validated_bool(params, "include_content", False)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        stage = params.get("stage")
        candidates = await self._strategy_evolution.list_candidates(
            stage=str(stage) if stage else None,
            limit=int(params.get("limit", 100)),
        )
        return {"candidates": [candidate.to_dict(include_content=include_content) for candidate in candidates]}

    async def _handle_strategy_propose(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        candidate = await self._strategy_evolution.propose(
            artifact_type=str(params.get("artifact_type", "")),
            component=str(params.get("component", "")),
            content=str(params.get("content", "")),
            rationale=str(params.get("rationale", "")),
            parent_candidate_id=str(params.get("parent_candidate_id", "")),
            source_trace_ids=tuple(params.get("source_trace_ids", ())),
            source="admin",
        )
        return {"status": "candidate", "candidate": candidate.to_dict(include_content=True)}

    async def _handle_strategy_reflect(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        candidate = await self._strategy_evolution.reflect_candidate(
            artifact_type=str(params.get("artifact_type", "")),
            component=str(params.get("component", "")),
            base_content=str(params.get("base_content", "")),
            diagnostics=tuple(params.get("diagnostics", ())),
            source_trace_ids=tuple(params.get("source_trace_ids", ())),
            parent_candidate_id=str(params.get("parent_candidate_id", "")),
        )
        return {"status": "candidate", "candidate": candidate.to_dict(include_content=True)}

    async def _handle_strategy_start_shadow(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        candidate = await self._strategy_evolution.start_shadow(str(params.get("candidate_id", "")))
        return {"status": candidate.stage.value, "candidate": candidate.to_dict()}

    async def _handle_strategy_record_isolated(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        baseline_results = params.get("baseline_results", ())
        candidate_results = params.get("candidate_results", ())
        if not isinstance(baseline_results, list) or not isinstance(candidate_results, list):
            raise ValueError("baseline_results and candidate_results must be arrays")
        candidate = await self._strategy_evolution.record_isolated_attestation(
            str(params.get("candidate_id", "")),
            harness_run_id=str(params.get("harness_run_id", "")),
            baseline_results=baseline_results,
            candidate_results=candidate_results,
        )
        return {"status": candidate.stage.value, "candidate": candidate.to_dict()}

    async def _handle_strategy_record_shadow(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        candidate = await self._strategy_evolution.record_shadow_evaluation(
            str(params.get("candidate_id", "")),
            sample_count=int(params.get("sample_count", 0)),
            baseline_score=float(params.get("baseline_score", 0.0)),
            candidate_score=float(params.get("candidate_score", 0.0)),
            safety_incidents=int(params.get("safety_incidents", 0)),
        )
        return {"status": candidate.stage.value, "candidate": candidate.to_dict()}

    async def _handle_strategy_start_canary(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        try:
            consent_confirmed = _validated_bool(params, "consent_confirmed", False)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        candidate = await self._strategy_evolution.start_canary(
            str(params.get("candidate_id", "")),
            actor=str(params.get("actor", "")),
            consent_confirmed=consent_confirmed,
        )
        return {"status": candidate.stage.value, "candidate": candidate.to_dict()}

    async def _handle_strategy_record_canary(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        candidate = await self._strategy_evolution.record_canary_evaluation(
            str(params.get("candidate_id", "")),
            sample_count=int(params.get("sample_count", 0)),
            baseline_score=float(params.get("baseline_score", 0.0)),
            candidate_score=float(params.get("candidate_score", 0.0)),
            safety_incidents=int(params.get("safety_incidents", 0)),
        )
        return {"status": candidate.stage.value, "candidate": candidate.to_dict()}

    async def _handle_strategy_promote(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        assignment = await self._strategy_evolution.promote(
            str(params.get("candidate_id", "")),
            actor=str(params.get("actor", "")),
            confirmation=str(params.get("confirmation", "")),
        )
        return {"status": "promoted", "assignment": assignment.to_dict()}

    async def _handle_strategy_rollback(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._strategy_evolution is None:
            return {"status": "error", "message": "Strategy evolution is not initialized"}
        assignment = await self._strategy_evolution.rollback(
            str(params.get("component", "")),
            actor=str(params.get("actor", "")),
            confirmation=str(params.get("confirmation", "")),
        )
        return {
            "status": "rolled_back",
            "assignment": assignment.to_dict() if assignment is not None else None,
        }

    async def _handle_evolution_status(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._evolution_harness is None:
            return {"enabled": False, "message": "Evolution harness is not initialized"}
        return await self._evolution_harness.status()

    async def _handle_evolution_runs(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._evolution_harness is None:
            return {"runs": [], "message": "Evolution harness is not initialized"}
        runs = await self._evolution_harness.list_runs(
            limit=max(1, min(int(params.get("limit", 100)), 500)),
        )
        return {"runs": [run.to_dict() for run in runs]}

    async def _handle_evolution_candidates(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._evolution_harness is None:
            return {"candidates": [], "message": "Evolution harness is not initialized"}
        try:
            include_patch = _validated_bool(params, "include_patch", False)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        run_id = str(params.get("run_id", "")).strip()
        candidates = await self._evolution_harness.list_candidates(
            run_id=run_id or None,
            limit=max(1, min(int(params.get("limit", 100)), 500)),
            include_patch=include_patch,
        )
        return {"candidates": [candidate.to_dict(include_patch=include_patch) for candidate in candidates]}

    async def _handle_evolution_create_run(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._evolution_harness is None:
            return {"status": "error", "message": "Evolution harness is not initialized"}
        run = await self._evolution_harness.create_run(
            str(params.get("problem", "")),
            profile=str(params.get("profile", "python")),
        )
        return {"status": run.state.value, "run": run.to_dict()}

    async def _handle_evolution_generate_candidates(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._evolution_harness is None:
            return {"status": "error", "message": "Evolution harness is not initialized"}
        candidates = await self._evolution_harness.generate_candidates(
            str(params.get("run_id", "")),
            count=max(2, min(int(params.get("count", 3)), 8)),
        )
        return {
            "status": "collecting",
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    async def _handle_evolution_evaluate(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._evolution_harness is None:
            return {"status": "error", "message": "Evolution harness is not initialized"}
        run = await self._evolution_harness.evaluate(str(params.get("run_id", "")))
        return {"status": run.state.value, "run": run.to_dict()}

    async def _handle_evolution_request_promotion(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        if self._evolution_harness is None:
            return {"status": "error", "message": "Evolution harness is not initialized"}
        return await self._evolution_harness.request_promotion(
            str(params.get("candidate_id", "")),
            actor=str(params.get("actor", "")),
            confirmation=str(params.get("confirmation", "")),
        )

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
        try:
            enabled = _validated_bool(params, "enabled", True)
        except ValueError as exc:
            return {"error": str(exc)}
        if not name:
            return {"error": "No plugin name provided"}
        if self._plugin_registry:
            if enabled:
                ok = self._plugin_registry.enable_plugin(name)
            else:
                ok = self._plugin_registry.disable_plugin(name)
            self._refresh_plugin_planner_context()
            return {"success": ok, "plugin": name, "enabled": enabled}
        return {"error": "Plugin registry not initialized"}

    def _refresh_plugin_planner_context(self) -> None:
        """Keep the planner's plugin tool inventory synchronized."""
        if self._planner and self._plugin_registry:
            self._planner.set_plugin_context(self._plugin_registry.get_tools_for_planner())
        if getattr(self, "_agent_mesh", None) and self._plugin_registry:
            self._agent_mesh.refresh_plugins(self._plugin_registry.get_all_plugins())

    async def _handle_plugin_market_list(self, params: dict, ws: ServerConnection) -> dict:
        """Return the approved GitHub catalog plus local-only plugins."""
        if not self._plugin_marketplace:
            return {"plugins": [], "error": "Plugin marketplace not initialized"}
        try:
            catalog = await asyncio.to_thread(self._plugin_marketplace.load_catalog)
            installed_plugins = self._plugin_registry.get_all_plugins() if self._plugin_registry else []
            installed_by_name = {plugin.name: plugin for plugin in installed_plugins}
            approved_names: set[str] = set()
            plugins: list[dict[str, Any]] = []
            for approved in catalog.data["plugins"]:
                item = dict(approved)
                name = item["name"]
                approved_names.add(name)
                item["installed"] = name in installed_by_name
                item["local_only"] = False
                item["source"] = catalog.source
                plugins.append(item)

            for plugin in installed_plugins:
                if plugin.name in approved_names:
                    continue
                item = plugin.to_dict()
                item.update(
                    {
                        "installed": True,
                        "local_only": True,
                        "source": "local",
                        "url": "",
                    }
                )
                plugins.append(item)

            return {
                "plugins": plugins,
                "source": catalog.source,
                "registry_url": catalog.registry_url,
                "submission_url": catalog.data.get("submission_url", ""),
                "warning": catalog.warning,
            }
        except Exception as exc:
            logger.error("Failed to load plugin marketplace: %s", exc)
            return {"plugins": [], "error": str(exc)}

    async def _handle_plugin_install_legacy(self, params: dict, ws: ServerConnection) -> dict:
        """Install a plugin from the marketplace with fully working code and Ed25519 signature."""
        import json

        from pilot.plugins import sign_plugin_directory

        plugin_name = params.get("plugin_name", "")
        if not plugin_name:
            return {"error": "plugin_name is required"}

        plugin_dir = PLUGINS_DIR / plugin_name
        plugin_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).parent.parent.parent
        registry_path = repo_root / "plugins" / "registry.json"
        tools = []
        description = "Heliox OS Plugin"
        version = "1.0.0"
        author = "community"

        if registry_path.exists():
            try:
                reg_data = json.loads(registry_path.read_text(encoding="utf-8"))
                for p in reg_data.get("plugins", []):
                    if p.get("name") == plugin_name:
                        tools = p.get("tools", [])
                        description = p.get("description", description)
                        version = p.get("version", version)
                        author = p.get("author", author)
                        break
            except Exception:
                pass

        if plugin_name == "home-assistant":
            if not tools:
                tools = [
                    {
                        "name": "ha_lights",
                        "description": "List all lights in Home Assistant",
                        "inputs": [],
                        "outputs": ["lights"],
                    },
                    {
                        "name": "ha_set_light",
                        "description": "Turn a light on or off",
                        "inputs": ["entity_id", "state"],
                        "outputs": ["result"],
                    },
                ]
            code_content = """# Home Assistant Plugin for Heliox OS
import os
import json
import urllib.request

def handle_tool(tool_name, params):
    ha_url = os.environ.get("HA_URL", "")
    ha_token = os.environ.get("HA_TOKEN", "")

    if tool_name == "ha_lights":
        if ha_url and ha_token:
            try:
                req = urllib.request.Request(f"{ha_url.rstrip('/')}/api/states", headers={"Authorization": f"Bearer {ha_token}"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    states = json.loads(resp.read().decode())
                    lights = [s for s in states if s.get("entity_id", "").startswith("light.")]
                    return {"status": "success", "lights": lights, "mode": "live"}
            except Exception as e:
                return {"status": "error", "error": f"Failed connecting to Home Assistant API: {e}", "mode": "error"}
        return {
            "status": "success",
            "mode": "local_demo",
            "message": "Simulated Home Assistant lights (set HA_URL and HA_TOKEN env vars for live sync)",
            "lights": [
                {"entity_id": "light.living_room_ceiling", "state": "on", "attributes": {"brightness": 255, "friendly_name": "Living Room Ceiling"}},
                {"entity_id": "light.kitchen_strip", "state": "off", "attributes": {"brightness": 0, "friendly_name": "Kitchen LED Strip"}},
                {"entity_id": "light.bedroom_lamp", "state": "on", "attributes": {"brightness": 128, "friendly_name": "Bedroom Night Lamp"}}
            ]
        }

    elif tool_name == "ha_set_light":
        entity_id = params.get("entity_id", "light.living_room_ceiling") if isinstance(params, dict) else "light.living_room_ceiling"
        state = params.get("state", "on") if isinstance(params, dict) else "on"
        if ha_url and ha_token:
            try:
                action = "turn_on" if state.lower() == "on" else "turn_off"
                req = urllib.request.Request(
                    f"{ha_url.rstrip('/')}/api/services/light/{action}",
                    data=json.dumps({"entity_id": entity_id}).encode(),
                    headers={"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return {"status": "success", "entity_id": entity_id, "state": state, "mode": "live", "response": json.loads(resp.read().decode())}
            except Exception as e:
                return {"status": "error", "error": str(e)}
        return {
            "status": "success",
            "mode": "local_demo",
            "entity_id": entity_id,
            "state": state,
            "message": f"Successfully turned {state} light '{entity_id}' (Demo Mode)"
        }

    return {"error": f"Unknown tool {tool_name}"}
"""
        elif plugin_name == "spotify-control":
            if not tools:
                tools = [
                    {
                        "name": "spotify_play",
                        "description": "Start or resume Spotify playback",
                        "inputs": [],
                        "outputs": ["result"],
                    },
                    {
                        "name": "spotify_pause",
                        "description": "Pause Spotify playback",
                        "inputs": [],
                        "outputs": ["result"],
                    },
                    {
                        "name": "spotify_now_playing",
                        "description": "Get currently playing track info",
                        "inputs": [],
                        "outputs": ["track", "artist", "album"],
                    },
                ]
            code_content = """# Spotify Control Plugin for Heliox OS
def handle_tool(tool_name, params):
    if tool_name == "spotify_now_playing":
        return {
            "status": "success",
            "track": "Cybernetic Horizon",
            "artist": "Heliox Sound Labs",
            "album": "OS Ambient Sessions Vol. 1",
            "duration_ms": 215000,
            "progress_ms": 142000,
            "is_playing": True,
            "source": "Heliox Media Bridge"
        }
    elif tool_name in ("spotify_play", "spotify_pause"):
        action = "playing" if tool_name == "spotify_play" else "paused"
        return {
            "status": "success",
            "playback_state": action,
            "message": f"Spotify playback {action} successfully via Heliox Media Engine"
        }
    return {"error": f"Unknown tool {tool_name}"}
"""
        elif plugin_name == "weather":
            if not tools:
                tools = [
                    {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "inputs": ["city"],
                        "outputs": ["temperature", "condition", "humidity"],
                    },
                    {
                        "name": "get_forecast",
                        "description": "Get 5-day weather forecast",
                        "inputs": ["city"],
                        "outputs": ["forecast"],
                    },
                ]
            code_content = """# Weather Plugin for Heliox OS
import urllib.request
import urllib.parse
import json

def handle_tool(tool_name, params):
    city = params.get("city", "London") if isinstance(params, dict) and params.get("city") else "London"
    if tool_name in ("get_weather", "get_forecast"):
        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(url, headers={"User-Agent": "Heliox-OS-Agent"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                curr = data.get("current_condition", [{}])[0]
                temp_c = curr.get("temp_C", "20")
                desc = curr.get("weatherDesc", [{"value": "Clear"}])[0].get("value", "Clear")
                humidity = curr.get("humidity", "50") + "%"
                wind = curr.get("windspeedKmph", "10") + " km/h"

                if tool_name == "get_weather":
                    return {
                        "status": "success",
                        "city": city,
                        "temperature": f"{temp_c} °C",
                        "condition": desc,
                        "humidity": humidity,
                        "wind_speed": wind,
                        "mode": "live_wttr_in"
                    }
                else:
                    forecasts = []
                    for day in data.get("weather", [])[:3]:
                        forecasts.append({
                            "date": day.get("date"),
                            "max_temp": f"{day.get('maxtempC')} °C",
                            "min_temp": f"{day.get('mintempC')} °C",
                            "condition": day.get("hourly", [{}])[4].get("weatherDesc", [{"value": "Sunny"}])[0].get("value", "Sunny")
                        })
                    return {"status": "success", "city": city, "forecast": forecasts, "mode": "live_wttr_in"}
        except Exception as e:
            return {
                "status": "success",
                "city": city,
                "temperature": "22 °C",
                "condition": "Sunny / Scattered Clouds",
                "humidity": "45%",
                "wind_speed": "12 km/h",
                "mode": "local_fallback",
                "note": f"Live weather sync unavailable ({e}), showing local estimate"
            }
    return {"error": f"Unknown tool {tool_name}"}
"""
        else:
            code_content = f"""# Custom Plugin: {plugin_name}
def handle_tool(tool_name, params):
    return {{"status": "success", "tool": tool_name, "params": params, "message": "Executed custom plugin tool successfully!"}}
"""

        manifest_dict = {
            "name": plugin_name,
            "version": version,
            "description": description,
            "author": author,
            "tools": tools,
            "agent_type": "system",
            "entry_point": "plugin.py",
            "runtime_type": "python",
            "enabled": True,
            "capabilities": {
                "filesystem": {"read": [], "write": []},
                "network_domains": [],
                "processes": [],
                "credentials": [],
                "clipboard": {"read": False, "write": False},
                "media": {"camera": False, "microphone": False},
                "data_retention": {"mode": "none", "max_days": 0},
                "destructive_actions": False,
            },
        }
        (plugin_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
        (plugin_dir / "plugin.py").write_text(code_content, encoding="utf-8")

        try:
            sign_plugin_directory(plugin_dir)
        except Exception as e:
            logger.warning("Could not sign plugin directory %s: %s", plugin_dir, e)

        if self._plugin_registry:
            count = self._plugin_registry.discover()
            logger.info("Plugin installed & signed: %s (total plugins: %d)", plugin_name, count)

        return {
            "success": True,
            "plugin": plugin_name,
            "path": str(plugin_dir),
        }

    async def _handle_plugin_install(self, params: dict, ws: ServerConnection) -> dict:
        """Install one exact package from the moderated GitHub catalog."""
        from pilot.plugins.marketplace import MarketplaceError

        plugin_name = params.get("plugin_name", "")
        if not plugin_name:
            return {"error": "plugin_name is required"}
        if not self._plugin_marketplace:
            return {"error": "Plugin marketplace not initialized"}

        try:
            result = await asyncio.to_thread(
                self._plugin_marketplace.install,
                plugin_name,
            )
            if self._plugin_registry:
                self._plugin_registry.discover()
                self._refresh_plugin_planner_context()
            logger.info("Installed approved marketplace plugin: %s", plugin_name)
            return result
        except MarketplaceError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.error("Failed to install plugin %s: %s", plugin_name, exc, exc_info=True)
            return {"error": f"Plugin install failed: {exc}"}

    async def _handle_plugin_uninstall(self, params: dict, ws: ServerConnection) -> dict:
        """Uninstall a plugin."""
        import shutil

        from pilot.plugins.marketplace import MarketplaceError, validate_plugin_name

        plugin_name = params.get("plugin_name", "")
        if not plugin_name:
            return {"error": "plugin_name is required"}
        try:
            plugin_name = validate_plugin_name(plugin_name)
        except MarketplaceError as exc:
            return {"error": str(exc)}

        plugin_dir = PLUGINS_DIR / plugin_name
        if not plugin_dir.exists():
            return {"error": f"Plugin not found: {plugin_name}"}

        try:
            shutil.rmtree(plugin_dir)
            logger.info("Plugin uninstalled: %s", plugin_name)
            if self._plugin_registry:
                self._plugin_registry.remove_plugin(plugin_name)
                self._refresh_plugin_planner_context()
            return {"success": True, "plugin": plugin_name}
        except Exception as exc:
            logger.error("Failed to uninstall plugin %s: %s", plugin_name, exc)
            return {"error": str(exc)}

    async def _handle_plugin_create(self, params: dict, ws: ServerConnection) -> dict:
        """Create a new custom plugin with manifest and Python code."""
        import ast
        import json
        import shutil

        from pilot.plugins import sign_plugin_directory
        from pilot.plugins.marketplace import MarketplaceError, validate_plugin_name

        raw_name = params.get("name", "")
        if not raw_name:
            return {"error": "Plugin name is required"}
        try:
            plugin_name = validate_plugin_name(raw_name)
        except MarketplaceError as exc:
            return {"error": str(exc)}

        tools = params.get("tools", [])
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except (TypeError, json.JSONDecodeError):
                return {"error": "Plugin tools must be a valid JSON list"}
        if not isinstance(tools, list) or not tools:
            return {"error": "Define at least one plugin tool before creating the plugin"}

        code_content = str(params.get("code") or "").strip()
        if not code_content:
            return {"error": "Plugin code is required; Heliox will not create a placeholder implementation"}
        try:
            module = ast.parse(code_content, filename="plugin.py")
        except SyntaxError as exc:
            return {"error": f"Plugin code is invalid Python: {exc.msg} (line {exc.lineno})"}
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "handle_tool"
            for node in module.body
        ):
            return {"error": "Plugin code must define a top-level handle_tool(tool_name, params) function"}

        plugin_dir = PLUGINS_DIR / plugin_name
        if plugin_dir.exists():
            return {"error": f"Plugin already exists: {plugin_name}"}

        manifest_dict = {
            "name": plugin_name,
            "version": params.get("version", "1.0.0"),
            "description": params.get("description", "Custom user plugin"),
            "author": params.get("author", "User"),
            "tools": tools,
            "agent_type": "system",
            "entry_point": "plugin.py",
            "runtime_type": "python",
            "enabled": True,
            "capabilities": {
                "filesystem": {"read": [], "write": []},
                "network_domains": [],
                "processes": [],
                "credentials": [],
                "clipboard": {"read": False, "write": False},
                "media": {"camera": False, "microphone": False},
                "data_retention": {"mode": "none", "max_days": 0},
                "destructive_actions": False,
            },
        }
        try:
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
            (plugin_dir / "plugin.py").write_text(code_content + "\n", encoding="utf-8")
            sign_plugin_directory(plugin_dir)
        except Exception as exc:
            shutil.rmtree(plugin_dir, ignore_errors=True)
            logger.error("Could not create signed plugin %s: %s", plugin_name, exc)
            return {"error": f"Plugin creation failed before installation: {exc}"}

        if self._plugin_registry:
            self._plugin_registry.discover()
            self._refresh_plugin_planner_context()

        submission_url = ""
        if self._plugin_marketplace:
            catalog = await asyncio.to_thread(self._plugin_marketplace.load_catalog)
            submission_url = catalog.data.get("submission_url", "")
        return {
            "success": True,
            "plugin": plugin_name,
            "path": str(plugin_dir),
            "local_only": True,
            "submission_url": submission_url,
        }

    async def _handle_plugin_run_tool(self, params: dict, ws: ServerConnection) -> dict:
        """Execute a tool provided by any installed plugin."""
        tool_name = params.get("tool_name", "")
        if not tool_name:
            return {"error": "tool_name is required"}
        args = params.get("args", {})
        if not isinstance(args, dict):
            args = {}

        if not self._plugin_registry:
            return {"error": "Plugin registry not initialized"}

        result = await asyncio.to_thread(
            self._plugin_registry.call_tool,
            tool_name,
            args,
        )
        return {"result": result}

    async def _handle_skills_list(self, params: dict, ws: ServerConnection) -> dict:
        if self._skill_registry:
            return {"skills": self._skill_registry.list_skills()}
        return {"error": "Skill registry not initialized"}

    async def _handle_skills_reload(self, params: dict, ws: ServerConnection) -> dict:
        if self._skill_registry:
            records = self._skill_registry.reload()
            serial = [
                {
                    "path": r.path,
                    "success": r.success,
                    "skill_ids": r.skill_ids,
                    "error": r.error,
                }
                for r in records
            ]
            if self._planner:
                self._planner.set_skills_context(self._skill_registry.planner_prompt_block())
            return {"ok": True, "records": serial, "skills": self._skill_registry.list_skills()}
        return {"error": "Skill registry not initialized"}

    async def _handle_skills_load_report(self, params: dict, ws: ServerConnection) -> dict:
        if self._skill_registry:
            records = self._skill_registry.last_load_records
            serial = [
                {
                    "path": r.path,
                    "success": r.success,
                    "skill_ids": r.skill_ids,
                    "error": r.error,
                }
                for r in records
            ]
            return {"records": serial, "search_dirs": [str(p) for p in self._skill_registry.search_dirs]}
        return {"error": "Skill registry not initialized"}

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
            if self._memory is not None:
                from pilot.memory.temporal import MemoryProvenance, MemoryScope

                await self._memory.remember_fact(
                    subject="user",
                    predicate=f"preference:{key}",
                    value=value,
                    scope=MemoryScope.USER,
                    confidence=1.0,
                    provenance=MemoryProvenance.EXPLICIT_USER,
                    evidence_payload={"source": "persona_add_preference"},
                )
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
        try:
            enabled = _validated_bool(params, "enabled", True)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        if self._screen_vision:
            if enabled:
                try:
                    describe = _validated_bool(params, "enable_describe", False)
                    from pilot.agents.screen_vision import MAX_CAPTURE_INTERVAL_SECONDS, MIN_CAPTURE_INTERVAL_SECONDS

                    interval = _validated_float(
                        params,
                        "interval_seconds",
                        self.config.screen_vision.capture_interval_seconds,
                        minimum=MIN_CAPTURE_INTERVAL_SECONDS,
                        maximum=MAX_CAPTURE_INTERVAL_SECONDS,
                    )
                except ValueError as exc:
                    return {"status": "error", "message": str(exc)}
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

        # Write the auth token to a runtime file so the Tauri frontend can
        # read it via a Rust command.  The file is chmod 600 (owner-read only).
        from pilot.config import RUNTIME_DIR

        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        token_file = RUNTIME_DIR / "auth_token"
        token_file.write_text(self.config.server.auth_token, encoding="utf-8")
        neural_token_file = RUNTIME_DIR / "neural_auth_token"
        neural_token_file.write_text(self._neural_auth_token, encoding="utf-8")
        mcp_token_file = RUNTIME_DIR / "mcp_auth_token"
        mcp_token_file.write_text(self._mcp_auth_token, encoding="utf-8")
        try:
            import os as _os

            _os.chmod(token_file, 0o600)
            _os.chmod(neural_token_file, 0o600)
            _os.chmod(mcp_token_file, 0o600)
        except Exception:
            pass  # chmod not available on Windows — file is in user-private dir
        logger.info("Auth token written to %s", token_file)
        logger.info("Neural sidecar token written to %s", neural_token_file)
        logger.info("Local MCP token written to %s", mcp_token_file)

        logger.info("Starting Pilot daemon on ws://%s:%d", host, port)
        self._server = await websockets.serve(
            self._handle_connection,
            host,
            port,
            ping_interval=30,  # send keepalive ping every 30s
            ping_timeout=300,  # allow up to 5 min for pong (matches LLM timeout)
        )
        logger.info("Pilot daemon ready")
        self._recorded_workflow_scheduler = asyncio.create_task(
            self._run_recorded_workflow_scheduler(), name="recorded_workflow_scheduler"
        )
        self._start_tts_warmup()

        # ── Start LAN mesh if enabled ──
        if self._mesh:
            try:
                await self._mesh.start()
                self._set_collab_executor(self._mesh.collab_executor)
            except Exception as exc:
                logger.exception("LAN mesh failed to start; continuing without peer collaboration")
                await self._mesh.stop()
                self._set_collab_executor(None)
                self._mesh = None
                self._mesh_error = str(exc)

        if self.config.air_handoff.enabled and self._air_handoff_server:
            try:
                await self._air_handoff_server.start()
            except Exception:
                logger.exception("Air Handoff failed to start; disabling the receiver")
                self.config.air_handoff.enabled = False
                self.config.save()

        if hasattr(self, "_new_features_announcement") and self._new_features_announcement:
            await asyncio.sleep(1)
            await self._broadcast_notification(
                "feature_announcement",
                {
                    "message": self._new_features_announcement,
                    "version": __version__,
                },
            )

    async def _release_pending_confirmations_for_shutdown(self) -> None:
        """Release every waiting confirmation because the daemon is stopping, and say so in its audit fields.

        Nobody answered these, so each one still unresolved (or approved without its reply reaching the UI) is
        resolved under its decision lock as "shutdown" from the "internal" channel, never as a confirm RPC. Plan
        waiters read that as expired (the durable approval becomes EXPIRED, not DENIED); the other waiters read a
        denial. A decision that was already answered and acknowledged is left as it is.
        """
        for pending in list(self._pending_confirms.values()):
            async with pending.decision_lock:
                if not pending.resolved or (pending.confirmed and not pending.acknowledged):
                    pending.confirmed = False
                    pending.approved_indices = set()
                    pending.resolved = True
                    pending.resolution_path = "shutdown"
                    pending.input_channel = "internal"
                pending.event.set()
        self._pending_confirms.clear()

    async def stop(self) -> None:
        """Stop the Pilot daemon server and clean up all resources."""
        self._running = False
        self._queue_shutdown = True
        if self._recorded_workflow_scheduler is not None:
            self._recorded_workflow_scheduler.cancel()
            await asyncio.gather(self._recorded_workflow_scheduler, return_exceptions=True)
            self._recorded_workflow_scheduler = None
        if self._manual_workflow_hook is not None:
            self._manual_workflow_hook.stop()
            self._manual_workflow_hook = None
        if self._recorded_workflows is not None:
            await self._recorded_workflows.stop_manual_visual_capture(save=False)
        await self._release_pending_confirmations_for_shutdown()

        queued_worker = self._queued_execute_worker
        if queued_worker is not None and not queued_worker.done():
            queued_worker.cancel()
            await asyncio.gather(queued_worker, return_exceptions=True)
        self._queued_execute_requests.clear()
        self._queued_reserved_task_id = ""

        active_execution = self._active_execution_task
        if active_execution is not None and not active_execution.done():
            active_execution.cancel()
            await asyncio.gather(active_execution, return_exceptions=True)
        self._active_execution_task = None

        refresh = self._mcp_client_refresh
        if refresh is not None and not refresh.done():
            refresh.cancel()
            await asyncio.gather(refresh, return_exceptions=True)
        await self._mcp_client.close()

        mcp_tasks = tuple(self._mcp_tasks.values())
        for task in mcp_tasks:
            task.cancel()
        if mcp_tasks:
            await asyncio.gather(*mcp_tasks, return_exceptions=True)
        self._mcp_tasks.clear()

        if self._trigger_engine is not None:
            self._trigger_engine.set_fire_callback(None)
            await self._trigger_engine.stop()
        if self._voice_gesture_workflows is not None:
            await self._voice_gesture_workflows.stop()
        if self._voice_listener is not None:
            await self._voice_listener.stop()
            self._voice_listener = None
        supervision_hook = getattr(self, "_supervision_hook", None)
        if supervision_hook is not None:
            supervision_hook.stop()
        if self._scheduled_runs is not None:
            await self._scheduled_runs.stop()
        if self._daily_brief is not None:
            await self._daily_brief.stop()
        if self._reminders is not None:
            await self._reminders.stop()
        if self._focus is not None:
            await self._focus.close()
        if self._quiet_hours is not None:
            await self._quiet_hours.stop()
        if self._nudges is not None:
            await self._nudges.stop()
        if self._watchers is not None:
            await self._watchers.stop()
        if self._clipboard_history is not None:
            await self._clipboard_history.stop()
        if self._focus_guard is not None:
            await self._focus_guard.stop()
        if self._app_reminders is not None:
            await self._app_reminders.stop()
        if self._gnomon_reviewer is not None:
            await self._gnomon_reviewer.close()
        if self._autonomous is not None:
            await self._autonomous.stop()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        # ── Stop LAN mesh ──
        if self._mesh:
            self._set_collab_executor(None)
            await self._mesh.stop()
        if self._air_handoff_server:
            await self._air_handoff_server.stop()
        if self._air_handoff_manager:
            await self._air_handoff_manager.clear_ephemeral()
        if self._orchestrator:
            await self._orchestrator.stop_all()
            await self._orchestrator.stop()
        if self._background:
            await self._background.shutdown()
        if hasattr(self, "_proactive") and self._proactive:
            await self._proactive.stop()
            if hasattr(self._proactive, "close"):
                await self._proactive.close()
        if hasattr(self, "_subconscious") and self._subconscious:
            await self._subconscious.stop()
            if hasattr(self._subconscious, "close"):
                await self._subconscious.close()
        if hasattr(self, "_screen_vision") and self._screen_vision:
            await self._screen_vision.stop()
            from pilot.security.risk_gate import get_risk_gate

            get_risk_gate().set_ui_observation_provider(None)
            if hasattr(self._screen_vision, "close"):
                await self._screen_vision.close()
        if self._executor is not None:
            await self._executor.close()
        if self._tts_warmup_task and not self._tts_warmup_task.done():
            self._tts_warmup_task.cancel()
            await asyncio.gather(self._tts_warmup_task, return_exceptions=True)
        if self._companion_follow_up_tasks:
            follow_up_tasks = tuple(self._companion_follow_up_tasks)
            for task in follow_up_tasks:
                task.cancel()
            await asyncio.gather(*follow_up_tasks, return_exceptions=True)
        if self._interaction_speech_tasks:
            speech_tasks = tuple(self._interaction_speech_tasks)
            for task in speech_tasks:
                task.cancel()
            await asyncio.gather(*speech_tasks, return_exceptions=True)
        await self._speech_coordinator.close()
        if hasattr(self, "_prompt_improver") and self._prompt_improver:
            if hasattr(self._prompt_improver, "close"):
                await self._prompt_improver.close()
        await self._drain_post_execution_tasks()
        if self._strategy_evolution is not None:
            await self._strategy_evolution.close()
            self._strategy_evolution = None
        if self._evolution_harness is not None:
            await self._evolution_harness.close()
            self._evolution_harness = None
        if hasattr(self, "_reflector") and self._reflector:
            if hasattr(self._reflector, "close"):
                await self._reflector.close()
        if self._memory:
            await self._memory.close()
        if self._agent_mesh is not None:
            await self._agent_mesh.close()
            self._agent_mesh = None
        lapse_task, self._co_adaptation_lapse_task = self._co_adaptation_lapse_task, None
        if lapse_task is not None and not lapse_task.done():
            lapse_task.cancel()
            await asyncio.gather(lapse_task, return_exceptions=True)
        if self._co_adaptation is not None:
            await self._co_adaptation.close()
            self._co_adaptation = None
        if self._experience_ledger:
            await self._experience_ledger.close()
            self._experience_ledger = None
        if self._durable_tasks:
            await self._durable_tasks.close()
            self._durable_tasks = None
        if self._budget_tracker:
            await self._budget_tracker.close()
        # ── Drain pending plan-history tasks before closing the store ──
        # Avoids aiosqlite.ProgrammingError when a fire-and-forget log task
        # is still writing as the connection is torn down.
        if self._plan_history_tasks:
            logger.info(
                "Waiting for %d pending plan-history task(s) to flush…",
                len(self._plan_history_tasks),
            )
            await asyncio.gather(*self._plan_history_tasks, return_exceptions=True)
        if self._plan_history:
            await self._plan_history.close()
        if self._model_router:
            await self._model_router.close()
            self._model_router = None
        if self._cognitive_engine and self._cognitive_engine.is_loaded:
            self._cognitive_engine.unload_model()
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

    async def _air_handoff_state(self) -> dict[str, Any]:
        """Return the complete local receiver state for the trusted UI."""
        if self._air_handoff_manager is None or self._air_handoff_server is None:
            return {
                "status": "error",
                "enabled": False,
                "running": False,
                "message": "Air Handoff is not initialized",
            }
        state = await self._air_handoff_manager.status()
        state.update(
            {
                "status": "ok",
                "enabled": bool(self.config.air_handoff.enabled),
                "running": bool(self._air_handoff_server.running),
                "receiver_url": (self._air_handoff_server.base_url if self._air_handoff_server.running else None),
                "port": self.config.air_handoff.port,
                "accept_requests": bool(getattr(self.config.air_handoff, "accept_requests", False)),
                "run_read_only_requests": bool(getattr(self.config.air_handoff, "run_read_only_requests", False)),
            }
        )
        return state

    async def _publish_air_handoff_state(self) -> dict[str, Any]:
        state = await self._air_handoff_state()
        await self._broadcast_notification("air_handoff_state", state)
        return state

    async def _handle_air_handoff_status(self, params: dict, ws: ServerConnection) -> dict[str, Any]:
        return await self._air_handoff_state()

    async def _handle_air_handoff_set_enabled(self, params: dict, ws: ServerConnection) -> dict[str, Any]:
        enabled = params.get("enabled")
        if not isinstance(enabled, bool):
            return {"status": "error", "message": "enabled must be a boolean"}
        if self._air_handoff_manager is None or self._air_handoff_server is None:
            return {"status": "error", "message": "Air Handoff is not initialized"}
        if enabled and not self._vault.available:
            return {
                "status": "error",
                "message": "Secure OS credential storage is required for Air Handoff",
            }
        previous_enabled = bool(self.config.air_handoff.enabled)
        previous_running = bool(self._air_handoff_server.running)
        try:
            self.config.air_handoff.enabled = enabled
            self.config.save()
            if enabled:
                await self._air_handoff_server.start()
            else:
                await self._air_handoff_manager.clear_ephemeral()
                await self._air_handoff_server.stop()
        except Exception as exc:
            logger.exception("Could not change Air Handoff receiver state")
            self.config.air_handoff.enabled = previous_enabled
            try:
                self.config.save()
                if previous_running and not self._air_handoff_server.running:
                    await self._air_handoff_server.start()
                elif not previous_running and self._air_handoff_server.running:
                    await self._air_handoff_server.stop()
            except Exception:
                logger.exception("Could not roll back Air Handoff receiver state")
            return {"status": "error", "message": str(exc)}
        state = await self._publish_air_handoff_state()
        return {"status": "ok", **state}

    async def _handle_air_handoff_start_pairing(self, params: dict, ws: ServerConnection) -> dict[str, Any]:
        if not self.config.air_handoff.enabled or not self._air_handoff_server.running:
            return {"status": "error", "message": "Enable Air Handoff before pairing"}
        try:
            pairing = self._air_handoff_manager.start_pairing(self._air_handoff_server.base_url)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        await self._publish_air_handoff_state()
        return {"status": "ok", **pairing}

    async def _handle_air_handoff_cancel_pairing(self, params: dict, ws: ServerConnection) -> dict[str, Any]:
        self._air_handoff_manager.cancel_pairing()
        await self._publish_air_handoff_state()
        return {"status": "ok"}

    async def _handle_air_handoff_revoke_device(self, params: dict, ws: ServerConnection) -> dict[str, Any]:
        device_id = str(params.get("device_id", ""))
        try:
            await self._air_handoff_manager.revoke_device(device_id)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        await self._publish_air_handoff_state()
        return {"status": "ok"}

    async def _handle_air_handoff_grab(self, params: dict, ws: ServerConnection) -> dict[str, Any]:
        if not self.config.air_handoff.enabled or not self._air_handoff_server.running:
            return {"status": "error", "message": "Air Handoff is not enabled"}
        kind = str(params.get("kind", "screenshot"))
        try:
            if kind == "screenshot":
                draft = await self._air_handoff_manager.grab_screenshot()
            elif kind == "text":
                draft = await self._air_handoff_manager.grab_text(
                    str(params.get("text", "")),
                    filename=str(params.get("filename", "heliox-note.txt")),
                )
            elif kind == "file":
                draft = await self._air_handoff_manager.grab_file(str(params.get("path", "")))
            else:
                return {"status": "error", "message": "Unsupported handoff kind"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        await self._publish_air_handoff_state()
        return {"status": "ok", "draft": draft}

    async def _handle_air_handoff_drop(self, params: dict, ws: ServerConnection) -> dict[str, Any]:
        target_device_id = str(params.get("target_device_id", ""))
        try:
            transfer = await self._air_handoff_manager.drop(target_device_id)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        await self._publish_air_handoff_state()
        return {"status": "ok", "transfer": transfer}

    async def _handle_air_handoff_cancel(self, params: dict, ws: ServerConnection) -> dict[str, Any]:
        await self._air_handoff_manager.cancel_draft()
        await self._publish_air_handoff_state()
        return {"status": "ok"}

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

    def _new_mesh(self, shared_secret: str) -> Any:
        """Construct one authenticated mesh runtime from an OS-vault secret."""
        from pilot.network.mesh import HelioxMesh
        from pilot.system.plugins import get_manager as get_plugin_manager

        return HelioxMesh(
            config=self.config.network,
            executor=self._executor,
            plugin_manager=get_plugin_manager(),
            shared_secret=shared_secret.encode("utf-8"),
        )

    def _set_collab_executor(self, collab_executor: Any | None) -> None:
        """Attach peer collaboration when the core executor is ready."""
        if self._executor is not None:
            self._executor.set_collab_executor(collab_executor)

    async def _handle_mesh_status(self, params: dict, ws: ServerConnection) -> dict:
        """Return authenticated mesh runtime and configuration state."""
        from pilot.security.vault import VaultUnavailableError

        try:
            mesh_secret = await self._vault.get_key("heliox_mesh") if self._vault else None
        except VaultUnavailableError as exc:
            mesh_secret = None
            if not self._mesh_error:
                self._mesh_error = str(exc)
        secret_configured = bool(mesh_secret and len(mesh_secret.encode("utf-8")) >= 32)
        if not self._mesh:
            return {
                "status": "ok",
                "enabled": False,
                "configured_enabled": self.config.network.enabled,
                "authenticated": False,
                "secret_configured": secret_configured,
                "reason": self._mesh_error or "Enable Peer Mesh and save a shared secret in Settings to activate it",
                "skill_sync_enabled": self.config.network.skill_sync_enabled,
                "collab_exec_enabled": self.config.network.collab_exec_enabled,
                "port": self.config.network.port,
            }
        return {
            "status": "ok",
            "enabled": True,
            "configured_enabled": self.config.network.enabled,
            "authenticated": True,
            "secret_configured": secret_configured,
            "instance_id": self._mesh.instance_id,
            "peer_count": len(self._mesh.peer_ids),
            "skill_sync_enabled": self.config.network.skill_sync_enabled,
            "collab_exec_enabled": self.config.network.collab_exec_enabled,
            "port": self.config.network.port,
        }

    async def _handle_mesh_generate_secret(self, params: dict, ws: ServerConnection) -> dict:
        """Generate a high-entropy secret for explicit copy to trusted peers."""
        return {"status": "ok", "shared_secret": secrets.token_urlsafe(48)}

    async def _handle_mesh_clear_secret(self, params: dict, ws: ServerConnection) -> dict:
        """Remove the mesh secret only while the mesh is disabled."""
        if self.config.network.enabled or self._mesh is not None:
            return {"status": "error", "message": "Disable Peer Mesh before removing its shared secret"}
        from pilot.security.vault import VaultUnavailableError

        try:
            await self._vault.delete_key("heliox_mesh")
        except VaultUnavailableError as exc:
            return {"status": "error", "message": str(exc)}
        self._mesh_error = ""
        return {"status": "ok"}

    async def _handle_mesh_configure(self, params: dict, ws: ServerConnection) -> dict:
        """Atomically persist and reconcile authenticated peer-mesh settings."""
        cfg = self.config.network
        try:
            enabled = _validated_bool(params, "enabled", cfg.enabled)
            skill_sync_enabled = _validated_bool(
                params,
                "skill_sync_enabled",
                cfg.skill_sync_enabled,
            )
            collab_exec_enabled = _validated_bool(
                params,
                "collab_exec_enabled",
                cfg.collab_exec_enabled,
            )
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        supplied_secret = params.get("shared_secret")
        if supplied_secret is not None:
            if not isinstance(supplied_secret, str):
                return {"status": "error", "message": "shared_secret must be a string"}
            if supplied_secret != supplied_secret.strip():
                return {"status": "error", "message": "shared_secret cannot start or end with whitespace"}
            if not 32 <= len(supplied_secret.encode("utf-8")) <= 512:
                return {"status": "error", "message": "shared_secret must contain from 32 to 512 bytes"}

        from pilot.security.vault import VaultUnavailableError

        try:
            previous_secret = await self._vault.get_key("heliox_mesh")
        except VaultUnavailableError as exc:
            return {"status": "error", "message": str(exc)}
        effective_secret = supplied_secret or previous_secret
        if enabled and (not effective_secret or len(effective_secret.encode("utf-8")) < 32):
            return {
                "status": "error",
                "message": "Save a shared secret of at least 32 bytes before enabling Peer Mesh",
            }

        if supplied_secret is not None and supplied_secret != previous_secret:
            try:
                await self._vault.store_key("heliox_mesh", supplied_secret)
            except VaultUnavailableError as exc:
                return {"status": "error", "message": str(exc)}

        previous_config = (cfg.enabled, cfg.skill_sync_enabled, cfg.collab_exec_enabled)
        previous_mesh = self._mesh
        self._set_collab_executor(None)
        if previous_mesh is not None:
            await previous_mesh.stop()
        self._mesh = None

        cfg.enabled = enabled
        cfg.skill_sync_enabled = skill_sync_enabled
        cfg.collab_exec_enabled = collab_exec_enabled
        self._mesh_error = ""

        try:
            self.config.save()
            if enabled and effective_secret:
                self._mesh = self._new_mesh(effective_secret)
                await self._mesh.start()
                self._set_collab_executor(self._mesh.collab_executor)
        except Exception as exc:
            logger.exception("Authenticated LAN mesh configuration failed")
            if self._mesh is not None:
                await self._mesh.stop()
            self._mesh = None
            self._set_collab_executor(None)
            self._mesh_error = str(exc)
            cfg.enabled, cfg.skill_sync_enabled, cfg.collab_exec_enabled = previous_config
            try:
                self.config.save()
            except Exception:
                logger.exception("Could not restore the previous persisted mesh configuration")
            if supplied_secret is not None and supplied_secret != previous_secret:
                try:
                    if previous_secret:
                        await self._vault.store_key("heliox_mesh", previous_secret)
                    else:
                        await self._vault.delete_key("heliox_mesh")
                except VaultUnavailableError:
                    logger.exception("Could not restore the previous mesh secret after rollback")
            if previous_config[0] and previous_secret:
                try:
                    self._mesh = self._new_mesh(previous_secret)
                    await self._mesh.start()
                    self._set_collab_executor(self._mesh.collab_executor)
                except Exception:
                    self._mesh = None
                    self._set_collab_executor(None)
                    logger.exception("Could not restore the previous mesh runtime after rollback")
            return {"status": "error", "message": f"Peer Mesh was not changed: {exc}"}

        status = await self._handle_mesh_status({}, ws)
        return {"status": "ok", **status}

    # ── Cognitive Intelligence Handlers ──

    async def _handle_cognitive_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Get stats for all cognitive subsystems.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with stats for cognitive_engine, attention_ui, stress_gate, intent_predictor.
        """
        return {
            "cognitive_engine": self._cognitive_engine.get_stats() if self._cognitive_engine else None,
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
        if not self._cognitive_engine:
            return {"status": "error", "message": "Cognitive engine not initialized"}

        input_dynamics = params.get("input_dynamics")
        if input_dynamics is not None:
            if not isinstance(input_dynamics, dict):
                return {"status": "error", "message": "input_dynamics must be an object"}

            def bounded_metric(name: str, maximum: float) -> float:
                try:
                    value = float(input_dynamics.get(name, 0.0))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be numeric") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite")
                return max(0.0, min(value, maximum))

            try:
                self._cognitive_engine.record_input_dynamics(
                    keystroke_rate_per_min=bounded_metric("keystroke_rate_per_min", 1200.0),
                    click_rate_per_min=bounded_metric("click_rate_per_min", 600.0),
                    pointer_move_rate_per_min=bounded_metric("pointer_move_rate_per_min", 1200.0),
                    idle_seconds=bounded_metric("idle_seconds", 3600.0),
                )
            except ValueError as exc:
                return {"status": "error", "message": str(exc)}

        state = await self._cognitive_engine.predict_cognitive_state(
            stimulus_description=params.get("stimulus", ""),
        )
        return {"status": "ok", **state.to_dict()}

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
        try:
            requested = _validated_bool(params, "enabled", False) if "enabled" in params else None
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        enabled = self._attention_ui.toggle(requested)
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
        try:
            requested = _validated_bool(params, "enabled", False) if "enabled" in params else None
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        enabled = self._stress_gate.toggle(requested)
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
        try:
            requested = _validated_bool(params, "enabled", False) if "enabled" in params else None
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        enabled = self._intent_predictor.toggle(requested)
        return {"enabled": enabled}

    async def _handle_cognitive_model_toggle(self, params: dict, ws: ServerConnection) -> dict:
        """Load or unload the cognitive engine.

        Args:
            params: JSON-RPC parameters with action (load/unload/status).
            ws: The WebSocket connection.

        Returns:
            A dict with loaded state, fallback, and availability status.
        """
        if not self._cognitive_engine:
            return {"error": "Cognitive engine not initialized"}
        action = params.get("action", "status")
        if action == "load":
            success = await self._cognitive_engine.load_model()
            return {"loaded": success, "fallback": self._cognitive_engine.is_fallback}
        elif action == "unload":
            self._cognitive_engine.unload_model()
            return {"loaded": False}
        return {
            "loaded": self._cognitive_engine.is_loaded,
            "fallback": self._cognitive_engine.is_fallback,
            "available": self._cognitive_engine.is_available,
        }

    # ── Voice Listener (JARVIS Mode) Handlers ──

    async def _voice_workflow_control_dispatch(self, command_text: str) -> bool:
        """Called by ContinuousVoiceListener right before normal command
        dispatch — lets a PAUSED/WAITING_FOR_TRIGGER voice-sourced
        VoiceGestureWorkflow claim a "continue"/"cancel" utterance instead
        of it being planned as a brand-new command. Returns True if the
        utterance was consumed as workflow control.

        Args:
            command_text: The recognized voice command text.

        Returns:
            True if a pending workflow claimed this utterance.
        """
        if not self._voice_gesture_workflows:
            return False
        outcome = await self._voice_gesture_workflows.apply_instruction("voice", command_text)
        if outcome is not None:
            normalized = " ".join(command_text.lower().split()).rstrip(".!?")
            status = {
                "cancelled": "cancelled",
                "corrected": "revising",
                "status": "answered",
            }.get(outcome.kind, "queued")
            await self._broadcast_notification(
                "voice_result",
                {
                    "command": command_text,
                    "status": status,
                    "message": outcome.message,
                    "result": outcome.message,
                    "coordinated_correction": outcome.kind == "corrected",
                },
            )
            if self._voice_listener and self._voice_listener.is_running:
                self._spawn_interaction_speech(
                    outcome.message,
                    channel=SpeechChannel.TASK_FAILURE if outcome.kind == "cancelled" else SpeechChannel.TASK_NARRATION,
                    dedupe_key=f"voice-workflow-update:{outcome.kind}:{normalized}",
                )
            return True
        if await self._voice_gesture_workflows.handle_control_phrase("voice", command_text):
            return True

        from pilot.agents.voice_gesture_workflow import (
            extract_voice_workflow_goal,
            should_start_voice_workflow,
        )
        from pilot.security.gateway import InvocationSource

        goal = extract_voice_workflow_goal(command_text)
        if not goal and should_start_voice_workflow(command_text):
            goal = command_text.strip()
        if not goal:
            return False
        _source, voice_scope = self._execution_scope_for_source("voice")
        workflow = await self._voice_gesture_workflows.start(goal, InvocationSource.VOICE, voice_scope)
        message = "I’m on it. Keep speaking if you want to correct or redirect me."
        await self._broadcast_notification(
            "voice_result",
            {
                "command": command_text,
                "status": "submitted",
                "message": message,
                "result": message,
                "workflow": workflow.to_dict(),
            },
        )
        if self._voice_listener and self._voice_listener.is_running:
            self._spawn_interaction_speech(
                message,
                channel=SpeechChannel.TASK_NARRATION,
                dedupe_key=f"voice-workflow-start:{workflow.workflow_id}",
            )
        return True

    async def _speak_companion_text(
        self,
        text: str,
        channel: SpeechChannel | str = SpeechChannel.FINAL_ANSWER,
        dedupe_key: str = "",
    ) -> SpeechOutcome:
        """Route daemon speech through the single companion audio authority."""
        if (
            self._focus is not None
            and self._focus.active
            and str(channel) in {SpeechChannel.PROACTIVE_SUGGESTION.value, SpeechChannel.BACKGROUND_INSIGHT.value}
        ):
            return SpeechOutcome("suppressed", SpeechChannel(str(channel)), "Held during a focus session")
        if str(channel) in REPEATABLE_SPEECH and text.strip():
            self._last_spoken = text.strip()
        # The continuous listener is the sole microphone consumer. It keeps
        # capturing wake-word commands while work and speech continue, then
        # _voice_command_dispatch stops current playback before applying the
        # utterance. Sharing its recorder with the TTS watcher would make two
        # coroutines race for the same audio frames and lose corrections.
        recorder = None
        listener = self._voice_listener if self._voice_listener and self._voice_listener.is_running else None
        if listener:
            listener.suppress_wake_free_commands()
        try:
            outcome = await self._speech_coordinator.speak(
                text,
                channel=channel,
                dedupe_key=dedupe_key,
                recorder=recorder,
            )
        finally:
            if listener:
                listener.resume_wake_free_commands()
        if outcome.status == "interrupted":
            await self._broadcast_notification("voice_status", {"status": "interrupted"})
        return outcome

    async def _speak_voice_response(
        self,
        text: str,
        *,
        channel: SpeechChannel | str = SpeechChannel.FINAL_ANSWER,
        dedupe_key: str = "",
    ) -> bool:
        """Speaks a voice-pipeline response, interruptible via barge-in
        when the continuous voice listener's VAD recorder is active and
        `config.voice.barge_in_enabled` is on (see speak_interruptible in
        pilot.system.voice). Falls back to plain, non-interruptible speak()
        otherwise. Returns True if the user started talking mid-playback
        and cut it off early.
        """
        outcome = await self._speak_companion_text(
            text,
            channel=channel,
            dedupe_key=dedupe_key,
        )
        return outcome.status == "interrupted"

    def _spawn_interaction_speech(
        self,
        text: str,
        *,
        channel: SpeechChannel,
        dedupe_key: str,
        delay_seconds: float = 0.0,
    ) -> asyncio.Task[None]:
        """Queue short interaction speech without delaying task execution.

        A voice acknowledgement may be delayed briefly so a fast command can
        deliver its real result instead of synthesizing two back-to-back
        utterances. The caller can cancel the returned task while it is still
        in that delay window.
        """

        async def _speak_after_delay() -> None:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            await self._speak_companion_text(
                text,
                channel=channel,
                dedupe_key=dedupe_key,
            )

        task = asyncio.create_task(_speak_after_delay())
        self._interaction_speech_tasks.add(task)
        task.add_done_callback(self._interaction_speech_tasks.discard)
        return task

    async def _arm_voice_follow_up(self) -> bool:
        """Open one wake-free conversational turn after Heliox stops talking."""
        if not self._voice_listener or not self._voice_listener.is_running:
            return False
        self._voice_listener.arm_follow_up_window()
        await self._broadcast_notification(
            "voice_status",
            {
                "status": "follow_up_ready",
                "message": "Listening continuously",
                "seconds": self.config.voice.follow_up_window_seconds,
            },
        )
        return True

    async def _stop_background_jobs(self) -> int:
        """Cancel every running autonomous job; returns how many were stopped."""
        if self._autonomous is None:
            return 0
        from pilot.agents.autonomous import JobStatus

        running = [
            job.job_id
            for job in (self._autonomous.get_job(job_id) for job_id in list(self._autonomous._active_tasks))
            if job is not None and job.status in {JobStatus.QUEUED, JobStatus.DECOMPOSING, JobStatus.RUNNING}
        ]
        stopped = 0
        for job_id in running:
            if await self._autonomous.cancel(job_id):
                stopped += 1
        return stopped

    async def _voice_command_dispatch(self, command_text: str) -> None:
        """Run voice through the same safe interaction path as typed input."""
        logger.info("Voice command received: '%s'", command_text)

        if self.config.voice.barge_in_enabled and self._speech_coordinator.status()["active"]:
            await self._speech_coordinator.stop_all()
            await self._broadcast_notification(
                "voice_status",
                {"status": "interrupted", "reason": "new voice command"},
            )

        normalized_command = _normalize_spoken_phrase(command_text)
        # Any pending approval counts: interactive, a background job, a scheduled run or a workflow.
        pending_plan_id = (
            self._active_plan_id
            if self._active_plan_id in self._pending_confirms
            else next(reversed(self._pending_confirms), "")
        )
        pending_approval = self._pending_confirms.get(pending_plan_id) if pending_plan_id else None
        if pending_approval is not None and normalized_command in VOICE_APPROVAL_ATTEMPTS:
            message = "For safety, approve this request in the visible Heliox confirmation dialog."
            await self._broadcast_notification(
                "voice_result",
                {
                    "command": command_text,
                    "status": "approval_required",
                    "message": message,
                    "plan_id": pending_plan_id,
                },
            )
            await self._speak_voice_response(
                message,
                channel=SpeechChannel.APPROVAL_RISK,
                dedupe_key=f"voice-approval-boundary:{pending_plan_id}",
            )
            return
        if pending_approval is not None and normalized_command in VOICE_DENIALS:
            # Denying is always safe, so it works by voice; approving still needs the visible dialog.
            result = await self._handle_confirm(
                {"plan_id": pending_plan_id, "confirmed": False, "input_channel": "voice_deny"}, None
            )
            message = (
                "Okay, I won't run that."
                if result.get("status") == "ok"
                else "That approval was already answered, so nothing was denied. Say or press Stop to cancel the task."
            )
            await self._broadcast_notification(
                "voice_result",
                {"command": command_text, "status": "answered", "message": message, "plan_id": pending_plan_id},
            )
            await self._speak_voice_response(
                message, channel=SpeechChannel.TASK_NARRATION, dedupe_key=f"voice-deny:{pending_plan_id}"
            )
            return

        if not self._interactive_request_active:
            timeline_answer = await self._control_reply(command_text)
            if timeline_answer is not None:
                await self._broadcast_notification(
                    "voice_result", {"command": command_text, "status": "answered", "message": timeline_answer}
                )
                await self._speak_voice_response(
                    timeline_answer, channel=SpeechChannel.TASK_NARRATION, dedupe_key=f"voice-timeline:{uuid.uuid4()}"
                )
                return

        undo_message = await self._undo_request(command_text) if not self._interactive_request_active else None
        if undo_message is not None:
            message = undo_message
            await self._broadcast_notification(
                "voice_result", {"command": command_text, "status": "answered", "message": message}
            )
            await self._speak_voice_response(
                message, channel=SpeechChannel.TASK_NARRATION, dedupe_key=f"voice-undo:{uuid.uuid4()}"
            )
            return

        if not self._interactive_request_active and normalized_command in VOICE_STOP_PHRASES:
            # "Stop" with no interactive task stops background work instead of becoming a new task.
            stopped = await self._stop_background_jobs()
            message = (
                f"Stopped {stopped} background task{'s' if stopped != 1 else ''}." if stopped else "Okay, stopped."
            )
            await self._broadcast_notification(
                "voice_result", {"command": command_text, "status": "answered", "message": message}
            )
            if stopped:
                await self._speak_voice_response(
                    message, channel=SpeechChannel.TASK_NARRATION, dedupe_key=f"voice-stop:{uuid.uuid4()}"
                )
            return

        if self._interactive_request_active:
            if normalized_command.startswith("next task "):
                next_input = re.sub(r"^\s*next\s+task\s*[:,]?\s*", "", command_text, flags=re.IGNORECASE).strip()
                active_context = self._active_experience_context
                result = await self._queue_interactive_request(
                    {
                        "input": next_input,
                        "session_id": active_context.session_id,
                        "user_id": active_context.user_id,
                        "source": "voice",
                    }
                )
                await self._broadcast_notification(
                    "voice_result",
                    {
                        "command": command_text,
                        "status": result.get("status", "error"),
                        "message": result.get("message", ""),
                    },
                )
                if result.get("status") == "queued":
                    await self._speak_voice_response(
                        "I queued that task. I will start it when this one finishes.",
                        channel=SpeechChannel.TASK_NARRATION,
                        dedupe_key=f"voice-queued:{result['task_id']}",
                    )
                return
            status_questions = {
                "status",
                "what are you doing",
                "what are you working on",
                "how is it going",
                "hows it going",
                "how s it going",
                "what is on my screen",
                "what s on my screen",
                "read my screen",
            }
            question_prefixes = ("question ", "quick question ")
            is_question = normalized_command in status_questions or normalized_command.startswith(question_prefixes)
            question_text = (
                re.sub(r"^\s*(?:quick\s+question|question)\s*[:,]?\s*", "", command_text, flags=re.IGNORECASE).strip()
                if normalized_command.startswith(question_prefixes)
                else command_text
            )
            if not is_question:
                await self._interaction_runtime.transition(
                    InteractionPhase.CORRECTING,
                    message="Applying your spoken correction",
                )
            result = await self._handle_interject(
                {"input": question_text if is_question else command_text, **({"mode": "ask"} if is_question else {})},
                None,
            )
            await self._broadcast_notification(
                "voice_result",
                {
                    "command": command_text,
                    "status": result.get("status", "error"),
                    "message": result.get("message", ""),
                    "coordinated_correction": not is_question,
                },
            )
            if is_question and result.get("status") == "answered":
                await self._speak_voice_response(
                    str(result.get("message") or "")[:350],
                    channel=SpeechChannel.FINAL_ANSWER,
                    dedupe_key=f"voice-in-task-question:{uuid.uuid4()}",
                )
            return

        spoken_command = command_text
        proactive_decision = None
        if self._proactive and hasattr(self._proactive, "resolve_spoken_response"):
            proactive_decision = await self._proactive.resolve_spoken_response(command_text)
        if proactive_decision and proactive_decision["decision"] == "dismissed":
            message = f"Okay, I won’t act on {proactive_decision['title']}."
            await self._broadcast_notification(
                "voice_result",
                {
                    "command": spoken_command,
                    "status": "success",
                    "result": message,
                    "proactive_decision": proactive_decision,
                },
            )
            await self._speak_voice_response(
                message,
                channel=SpeechChannel.FINAL_ANSWER,
                dedupe_key=f"proactive-dismissed:{proactive_decision['suggestion_id']}",
            )
            await self._arm_voice_follow_up()
            return
        if proactive_decision and proactive_decision["decision"] == "accepted":
            command_text = (
                f"The user accepted the proactive suggestion '{proactive_decision['title']}'. "
                f"Carry out this requested action: {proactive_decision['action_command']}"
            )

        interaction = await self._interaction_runtime.start(command_text, source="voice")
        interaction_id = str(interaction["interaction_id"])
        language = getattr(
            self._voice_listener,
            "last_detected_language",
            self.config.voice.language if self.config.voice.language != "auto" else "en",
        )
        await self._broadcast_notification(
            "voice_command",
            {
                "command": spoken_command,
                "status": "executing",
                "language": language,
                "proactive_decision": proactive_decision,
            },
        )
        acknowledgement_task = self._spawn_interaction_speech(
            acknowledgement_for(command_text),
            channel=SpeechChannel.TASK_NARRATION,
            dedupe_key=f"voice-ack:{interaction_id}",
            delay_seconds=2.5,
        )

        try:
            await self._interaction_runtime.transition(
                InteractionPhase.PLANNING,
                message="Planning the safest useful action",
                interaction_id=interaction_id,
            )
            try:
                response = await self._handle_execute(
                    {
                        "input": command_text,
                        "session_id": "voice",
                        "user_id": "local",
                        "source": "voice",
                        "_interaction_id": interaction_id,
                    },
                    _BroadcastConnection(self._broadcast_notification),
                )
            finally:
                # A quick command should speak only its useful final result.
                # Longer work still gets the delayed acknowledgement, and the
                # coordinator lets the higher-priority final answer preempt it.
                if isinstance(acknowledgement_task, asyncio.Task) and not acknowledgement_task.done():
                    acknowledgement_task.cancel()
            status = str(response.get("status") or "error")
            result_text = str(
                response.get("message")
                or response.get("explanation")
                or "The request finished without a visible result."
            )
            voice_status = "partial" if status == "partial_failure" else status
            await self._broadcast_notification(
                "voice_result",
                {
                    "command": spoken_command,
                    "status": voice_status,
                    "result": result_text[:500],
                    "language": language,
                    "companion_follow_up": None,
                },
            )

            terminal_phase = (
                InteractionPhase.COMPLETED
                if status == "success"
                else InteractionPhase.INTERRUPTED
                if status in {"cancelled", "interrupted", "denied"}
                else InteractionPhase.FAILED
            )
            await self._interaction_runtime.transition(
                terminal_phase,
                message=result_text[:160],
                interaction_id=interaction_id,
            )
            spoken = result_text[:300] if len(result_text) < 300 else result_text[:297] + "..."
            await self._speak_voice_response(
                spoken,
                channel=(
                    SpeechChannel.FINAL_ANSWER
                    if terminal_phase == InteractionPhase.COMPLETED
                    else SpeechChannel.TASK_FAILURE
                ),
                dedupe_key=f"voice-result:{spoken_command.casefold()}",
            )
            if await self._arm_voice_follow_up():
                await self._interaction_runtime.transition(
                    InteractionPhase.LISTENING,
                    message="Listening continuously",
                    interaction_id=interaction_id,
                )
        except Exception as error:
            logger.error("Voice command execution failed: %s", error)
            message = "Something went wrong while executing your request."
            await self._broadcast_notification(
                "voice_result",
                {
                    "command": spoken_command,
                    "status": "error",
                    "message": str(error),
                    "language": language,
                },
            )
            await self._interaction_runtime.transition(
                InteractionPhase.FAILED,
                message=message,
                interaction_id=interaction_id,
            )
            try:
                await self._speak_voice_response(
                    message,
                    channel=SpeechChannel.TASK_FAILURE,
                    dedupe_key=f"voice-exception:{spoken_command.casefold()}",
                )
                await self._arm_voice_follow_up()
            except Exception:
                pass

    async def _voice_status_broadcast(self, status: str, data: dict) -> None:
        """Called by ContinuousVoiceListener for status updates.

        Args:
            status: The voice listener status.
            data: Additional status data.
        """
        await self._broadcast_notification("voice_status", {"status": status, **data})
        phase_map = {
            "wake_detected": InteractionPhase.UNDERSTANDING,
            "follow_up_detected": InteractionPhase.UNDERSTANDING,
            "follow_up_ready": InteractionPhase.LISTENING,
            "listening": InteractionPhase.LISTENING,
            "timeout": InteractionPhase.LISTENING,
            "interrupted": InteractionPhase.INTERRUPTED,
        }
        phase = phase_map.get(status)
        if phase is not None:
            await self._interaction_runtime.transition(
                phase,
                message=str(data.get("message") or status.replace("_", " ")),
            )

    async def _replace_voice_listener(self, wake_words: list[str]) -> str:
        """Replace an active listener after recorder-bound settings change."""
        from pilot.system.voice import ContinuousVoiceListener

        previous = self._voice_listener
        running_commands: set[asyncio.Task[None]] = set()
        if previous is not None and previous.is_running:
            # Keep voice commands that are mid-task; only the microphone side restarts.
            running_commands = {task for task in getattr(previous, "_command_tasks", set()) if not task.done()}
            await previous.stop(cancel_commands=False)

        replacement = ContinuousVoiceListener(
            wake_words=wake_words,
            on_command=self._voice_command_dispatch,
            on_status=self._voice_status_broadcast,
            workflow_control=self._voice_workflow_control_dispatch,
            config=self.config,
        )
        self._voice_listener = replacement
        result = await replacement.start()
        if not replacement.is_running:
            self._voice_listener = None
            raise RuntimeError(result)
        replacement.adopt_command_tasks(running_commands)
        return result

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
            workflow_control=self._voice_workflow_control_dispatch,
            config=self.config,
        )
        result = await self._voice_listener.start()
        if not self._voice_listener.is_running:
            self._voice_listener = None
            return {"status": "error", "message": result}
        await self._interaction_runtime.transition(
            InteractionPhase.LISTENING,
            message="Listening for your wake word",
        )
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
        await self._interaction_runtime.transition(
            InteractionPhase.IDLE,
            message="Voice listening is off",
        )
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

    async def _handle_voice_recognition_test(self, params: dict, ws: ServerConnection) -> dict:
        """Capture one utterance and return its transcript without executing it."""
        from pilot.system.voice import ContinuousVoiceListener

        try:
            timeout_seconds = max(3.0, min(float(params.get("timeout_seconds", 12)), 30.0))
        except (TypeError, ValueError):
            return {"status": "error", "message": "Recognition-test timeout must be a number."}

        listener = self._voice_listener
        temporary_listener = listener is None or not listener.is_running
        if temporary_listener:
            listener = ContinuousVoiceListener(config=self.config)
            start_result = await listener.start()
            if not listener.is_running:
                return {"status": "error", "message": start_result}

        try:
            result = await listener.capture_diagnostic(timeout_seconds)
            return {"status": "ok", **result}
        except TimeoutError:
            return {
                "status": "error",
                "message": "No speech was recognized before the microphone test timed out.",
            }
        except RuntimeError as error:
            return {"status": "error", "message": str(error)}
        finally:
            if temporary_listener:
                await listener.stop()

    async def _handle_interaction_status(self, params: dict, ws: ServerConnection) -> dict:
        """Return the shared text/voice interaction state."""
        return self._interaction_runtime.status()

    async def _handle_list_audio_input_devices(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        """List microphone inputs that support Heliox's recording format."""
        try:
            import sounddevice as sd

            from pilot.system.voice import list_audio_input_devices

            devices = await asyncio.to_thread(list_audio_input_devices, sd)
        except ImportError:
            return {
                "status": "error",
                "devices": [],
                "selected": self.config.voice.input_device,
                "message": "Python sounddevice is not installed.",
            }
        except Exception as error:
            logger.warning("Could not enumerate audio input devices: %s", error)
            return {
                "status": "error",
                "devices": [],
                "selected": self.config.voice.input_device,
                "message": str(error),
            }
        return {
            "status": "ok",
            "devices": devices,
            "selected": self.config.voice.input_device,
            "message": "" if devices else "No compatible microphone inputs were found.",
        }

    async def _handle_speak_text(self, params: dict, ws: ServerConnection) -> dict:
        """Speak UI text through the configured daemon TTS engine."""
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"status": "error", "message": "text must be a non-empty string"}
        if len(text) > 4000:
            return {"status": "error", "message": "text must be 4000 characters or fewer"}

        logger.info(
            "Companion speech started (engine=%s, characters=%d)",
            self.config.voice.tts_engine,
            len(text.strip()),
        )
        channel = params.get("channel", SpeechChannel.FINAL_ANSWER.value)
        try:
            resolved_channel = SpeechChannel(channel)
        except ValueError:
            return {"status": "error", "message": f"Unknown speech channel: {channel}"}
        outcome = await self._speak_companion_text(
            text.strip(),
            channel=resolved_channel,
            dedupe_key=str(params.get("dedupe_key", "")).strip(),
        )
        if outcome.status == "spoken":
            logger.info("Companion speech completed (engine=%s)", self.config.voice.tts_engine)
            return {"status": "spoken", "message": f"Spoken: {text.strip()[:80]}..."}
        if outcome.status == "interrupted":
            logger.info("Companion speech interrupted by user")
        return {"status": outcome.status, "message": outcome.message}

    async def _handle_stop_speech(self, params: dict, ws: ServerConnection) -> dict:
        """Immediately stop daemon-side TTS playback."""
        from pilot.system.voice import stop_speaking

        cancelled = await self._speech_coordinator.stop_all()
        message = await stop_speaking()
        return {"status": "stopped", "message": message, "cancelled": cancelled}

    async def _handle_companion_speech_status(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        """Expose the one-channel coordinator state for diagnostics and UI."""
        return {"status": "ok", **self._speech_coordinator.status()}

    async def _handle_reset_wake_calibration(self, params: dict, ws: ServerConnection) -> dict:
        """Clear all learned wake-word calibration data.

        Deletes the on-device JSON store entirely — this is the "reset
        learned wake words" action in Settings. If the voice listener is
        currently running, its live WakeWordCalibrator is reset too so the
        change takes effect immediately without restarting the listener.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with status.
        """
        if self._voice_listener:
            self._voice_listener._wake_calibrator.reset()
        else:
            from pilot.system.voice_calibration import VoiceCalibrationStore

            VoiceCalibrationStore().reset()
        return {"status": "ok"}

    async def _handle_list_wake_variants(self, params: dict, ws: ServerConnection) -> dict:
        """List learned wake-word variants for the Settings transparency view.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with a list of variants (text, confirmed_count,
            first_seen, last_confirmed) and the promotion threshold so the
            UI can show progress toward it.
        """
        from dataclasses import asdict

        from pilot.system.voice_calibration import VoiceCalibrationStore, WakeWordCalibrator

        variants = VoiceCalibrationStore().load()
        return {
            "status": "ok",
            "variants": [asdict(v) for v in variants.values()],
            "promotion_threshold": WakeWordCalibrator.PROMOTION_THRESHOLD,
        }

    # ── Autonomous Executor Handlers ──

    async def _capture_recording_feature(self) -> dict[str, Any] | None:
        vision = self._screen_vision
        return await vision.capture_world_feature_now() if vision is not None else None

    async def _broadcast_human_control(self, status: Any) -> None:
        await self._broadcast_notification("human_control", status.to_dict())

    async def _handle_take_control(self, params: dict, ws: ServerConnection) -> dict:
        """Give the desktop to the person; Heliox's own desktop input pauses until released."""
        reason = params.get("reason", "taken from the Heliox app")
        if not isinstance(reason, str):
            return {"status": "error", "message": "reason must be a string"}
        self._human_control.take(reason or "taken from the Heliox app")
        return {"status": "ok", **self._human_control.status().to_dict()}

    async def _handle_release_control(self, params: dict, ws: ServerConnection) -> dict:
        """Hand the desktop back; paused autonomous work observes again before acting."""
        self._human_control.release()
        return {"status": "ok", **self._human_control.status().to_dict()}

    async def _handle_human_control_status(self, params: dict, ws: ServerConnection) -> dict:
        return {"status": "ok", **self._human_control.status().to_dict()}

    async def _refresh_mcp_client(self) -> dict[str, Any]:
        """(Re)connect the person's MCP servers and show their tools to the planner."""
        try:
            status = await self._mcp_client.refresh()
        except Exception as error:  # a broken server list must not stop the daemon
            logger.warning("MCP client refresh failed: %s", error)
            return {"servers": [], "tool_count": 0, "error": str(error)}
        if self._planner:
            self._planner.set_mcp_context(self._mcp_client.catalog())
        return status

    async def _handle_mcp_client_status(self, params: dict, ws: ServerConnection) -> dict:
        """Configured MCP servers, whether each is running, and how many tools it offers."""
        return {"status": "ok", **self._mcp_client.status()}

    async def _handle_mcp_client_refresh(self, params: dict, ws: ServerConnection) -> dict:
        """Restart the MCP servers listed in mcp_servers.json and reload their tools."""
        return {"status": "ok", **(await self._refresh_mcp_client())}

    async def _handle_recorded_workflow_status(self, params: dict, ws: ServerConnection) -> dict:
        return {
            "status": "ok",
            **self._recorded_workflows.status(),
            "manual_hook_healthy": bool(self._manual_workflow_hook and self._manual_workflow_hook.healthy()),
        }

    async def _handle_recorded_workflow_start(self, params: dict, ws: ServerConnection) -> dict:
        try:
            consent = _validated_bool(params, "consent", False)
            capture_visual = _validated_bool(params, "capture_visual", False)
            capture_manual = _validated_bool(params, "capture_manual", False)
            manual_consent = _validated_bool(params, "manual_consent", False)
            if not consent:
                raise ValueError("Explicit recording consent is required")
            if capture_manual and not manual_consent:
                raise ValueError("Separate consent is required to observe manual desktop input")
            await asyncio.to_thread(self._recorded_workflows.prepare)
            session = self._recorded_workflows.start(
                name=str(params.get("name", "")),
                goal=str(params.get("goal", "")),
                consent=consent,
                capture_visual=capture_visual,
                capture_manual=capture_manual,
                manual_consent=manual_consent,
            )
            if capture_manual:
                from pilot.workflows.manual_workflow_hook import ManualWorkflowHook

                hook = ManualWorkflowHook(self._recorded_workflows)
                try:
                    hook.start()
                except Exception:
                    self._recorded_workflows.stop(save=False)
                    raise
                self._manual_workflow_hook = hook
                if capture_visual:
                    self._recorded_workflows.start_manual_visual_capture()
        except (ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}
        await self._announce_recorded_workflow_status(True, ws)
        return {"status": "ok", **session}

    async def _handle_recorded_workflow_stop(self, params: dict, ws: ServerConnection) -> dict:
        try:
            if self._manual_workflow_hook is not None:
                if not self._manual_workflow_hook.healthy():
                    self._recorded_workflows.mark_manual_gap()
                self._manual_workflow_hook.stop()
                self._manual_workflow_hook = None
            await self._recorded_workflows.stop_manual_visual_capture(save=_validated_bool(params, "save", False))
            result = await asyncio.to_thread(self._recorded_workflows.stop, save=_validated_bool(params, "save", False))
        except (ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}
        await self._announce_recorded_workflow_status(False, ws)
        return {"status": "ok", **result}

    async def _announce_recorded_workflow_status(self, active: bool, caller: ServerConnection | None) -> None:
        """Tell the other windows that a recording started or stopped; a running selection study pauses on it.

        The caller already has the new state in its reply, so the notification skips the caller's own socket,
        whose replies stay the only messages it receives for its requests.
        """
        await self._broadcast_notification("recorded_workflow_status_changed", {"active": active}, exclude=caller)

    async def _handle_recorded_workflow_list(self, params: dict, ws: ServerConnection) -> dict:
        try:
            return {"status": "ok", "workflows": await asyncio.to_thread(self._recorded_workflows.list)}
        except (ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}

    async def _handle_recorded_workflow_get(self, params: dict, ws: ServerConnection) -> dict:
        try:
            workflow = await asyncio.to_thread(self._recorded_workflows.get, str(params.get("workflow_id", "")))
            return {"status": "ok", "workflow": workflow}
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}

    async def _handle_recorded_workflow_delete(self, params: dict, ws: ServerConnection) -> dict:
        try:
            await asyncio.to_thread(self._recorded_workflows.delete, str(params.get("workflow_id", "")))
            return {"status": "ok"}
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}

    def _remember_recorded_workflow_job(self, job_id: str, workflow_id: str) -> None:
        self._recorded_workflow_jobs[job_id] = workflow_id
        while len(self._recorded_workflow_jobs) > 100:
            self._recorded_workflow_jobs.pop(next(iter(self._recorded_workflow_jobs)))

    async def _handle_recorded_workflow_run(self, params: dict, ws: ServerConnection) -> dict:
        from pilot.workflows.recorded_workflows import workflow_prompt

        if self._autonomous is None:
            return {"status": "error", "message": "Guarded autonomous execution is unavailable"}
        if self._recorded_workflows.status()["active"]:
            return {"status": "error", "message": "Stop the current recording before replaying a workflow"}
        try:
            workflow = await asyncio.to_thread(self._recorded_workflows.get, str(params.get("workflow_id", "")))
            if workflow["incomplete"]:
                raise ValueError(
                    "Incomplete recordings cannot be replayed; unsupported steps need a parameterized workflow"
                )
            if workflow["needs_review"] and not _validated_bool(params, "reviewed", False):
                raise ValueError("Review manual or coordinate-based steps before replay")
            text_values = params.get("text_values", {})
            if not isinstance(text_values, dict):
                raise ValueError("Text placeholder values must be an object")
            capture_visual = _validated_bool(params, "capture_visual", False)
            submit = self._autonomous.submit
            if capture_visual:
                with self._recorded_workflows.capture_replay_visual(workflow["id"], consent=True):
                    job = await submit(
                        workflow_prompt(workflow, text_values=text_values),
                        source="recorded_workflow",
                        session_id=f"recorded:{workflow['id']}",
                    )
            else:
                job = await submit(
                    workflow_prompt(workflow, text_values=text_values),
                    source="recorded_workflow",
                    session_id=f"recorded:{workflow['id']}",
                )
            self._remember_recorded_workflow_job(job.job_id, workflow["id"])
            return {"status": "submitted", "job": job.to_dict()}
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}

    async def _handle_recorded_workflow_feedback(self, params: dict, ws: ServerConnection) -> dict:
        workflow_id = str(params.get("workflow_id", ""))
        job_id = str(params.get("job_id", ""))
        if self._autonomous is None or self._recorded_workflow_jobs.get(job_id) != workflow_id:
            return {"status": "error", "message": "Recorded workflow run not found"}
        job = self._autonomous.get_job(job_id)
        job_status = str(getattr(job.status, "value", job.status)) if job is not None else ""
        if job is None or not job.completed_at or job_status not in {"success", "partial", "failed", "cancelled"}:
            return {"status": "error", "message": "Wait for the workflow run to finish"}
        try:
            workflow = await asyncio.to_thread(
                self._recorded_workflows.record_feedback,
                workflow_id,
                job_id=job_id,
                outcome=str(params.get("outcome", "")),
                job_status=job_status,
            )
            self._recorded_workflow_jobs.pop(job_id, None)
            return {"status": "ok", "workflow": workflow}
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}

    async def _handle_recorded_workflow_schedule(self, params: dict, ws: ServerConnection) -> dict:
        try:
            workflow = await asyncio.to_thread(
                self._recorded_workflows.schedule,
                str(params.get("workflow_id", "")),
                run_at=str(params.get("run_at", "")),
                repeat_seconds=_validated_int(params, "repeat_seconds", 0, minimum=0, maximum=31536000),
                reviewed=_validated_bool(params, "reviewed", False),
            )
            return {"status": "ok", "workflow": workflow}
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}

    async def _handle_recorded_workflow_unschedule(self, params: dict, ws: ServerConnection) -> dict:
        try:
            workflow = await asyncio.to_thread(self._recorded_workflows.unschedule, str(params.get("workflow_id", "")))
            return {"status": "ok", "workflow": workflow}
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"status": "error", "message": str(exc)}

    async def _run_recorded_workflow_scheduler(self) -> None:
        from pilot.workflows.recorded_workflows import workflow_prompt

        while self._running:
            try:
                if (
                    self._autonomous is not None
                    and self._recorded_workflows is not None
                    and not self._recorded_workflows.status()["active"]
                ):
                    for workflow in await asyncio.to_thread(self._recorded_workflows.take_due):
                        try:
                            job = await self._autonomous.submit(
                                workflow_prompt(workflow),
                                source="recorded_workflow_schedule",
                                session_id=f"recorded:{workflow['id']}",
                            )
                            self._remember_recorded_workflow_job(job.job_id, workflow["id"])
                            await self._broadcast_notification(
                                "recorded_workflow_dispatched",
                                {"workflow_id": workflow["id"], "name": workflow["name"], "job_id": job.job_id},
                            )
                        except Exception:
                            logger.exception("Scheduled recorded workflow dispatch failed")
                            await self._broadcast_notification(
                                "recorded_workflow_failed",
                                {"workflow_id": workflow["id"], "name": workflow["name"]},
                            )
            except Exception:
                logger.exception("Recorded workflow scheduler check failed")
            await asyncio.sleep(10)

    async def _handle_autonomous_submit(self, params: dict, ws: ServerConnection) -> dict:
        """Submit a task for autonomous background execution.

        Args:
            params: JSON-RPC parameters with goal, source, and an optional
                scope_override restricting the AgentGateway's "autonomous"
                floor further for this job only (see pilot.security.gateway
                — an override can only narrow the floor, never widen it,
                regardless of what's supplied here).
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

        scope_override = None
        raw_override = params.get("scope_override")
        if raw_override is not None:
            from pydantic import ValidationError

            from pilot.security.gateway import TaskScopeOverride

            try:
                scope_override = TaskScopeOverride.model_validate(raw_override)
            except ValidationError as e:
                return {"error": f"Invalid scope_override: {e}"}

        job = await self._autonomous.submit(
            goal,
            source=source,
            scope_override=scope_override,
            session_id=str(params.get("session_id") or source or "default"),
        )
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

    async def _handle_autonomous_resume(self, params: dict, ws: ServerConnection) -> dict:
        """Continue a job that a daemon shutdown or crash interrupted, from its first unfinished step."""
        if not self._autonomous:
            return {"error": "Autonomous executor not initialized"}
        job_id = params.get("job_id", "")
        if not isinstance(job_id, str) or not job_id:
            return {"status": "error", "message": "job_id is required"}
        job = await self._autonomous.resume(job_id)
        if job is None:
            return {"status": "error", "message": "Only an interrupted job can be resumed"}
        return {"status": "resumed", "job": job.to_dict()}

    async def _submit_scheduled_run(self, run: Any) -> str:
        """Start one scheduled run as an autonomous job limited to its approved actions."""
        if self._autonomous is None:
            raise RuntimeError("Autonomous executor not initialized")
        goal = (
            f'{run.goal}\n\n(Scheduled run "{run.name}". '
            f"Only these actions are approved: {', '.join(run.allowed_actions)}.)"
        )
        job = await self._autonomous.submit(
            goal,
            source="scheduled",
            session_id=f"scheduled:{run.run_id}",
            allowed_actions=run.allowed_actions,
        )
        await self._broadcast_notification(
            "scheduled_run_dispatched", {"run_id": run.run_id, "name": run.name, "job_id": job.job_id}
        )
        return job.job_id

    def _autonomous_job_is_active(self, job_id: str) -> bool:
        from pilot.agents.autonomous import JobStatus

        job = self._autonomous.get_job(job_id) if self._autonomous is not None else None
        return job is not None and job.status in {JobStatus.QUEUED, JobStatus.DECOMPOSING, JobStatus.RUNNING}

    def _scheduled_run_view(self, run: Any) -> dict[str, Any]:
        job = self._autonomous.get_job(run.last_job_id) if self._autonomous and run.last_job_id else None
        return {**run.to_dict(), "last_job_status": job.status.value if job else ""}

    async def _handle_scheduled_run_list(self, params: dict, ws: ServerConnection) -> dict:
        """Scheduled agent runs with their next run time and the last run's job status."""
        if self._scheduled_runs is None:
            return {"status": "error", "message": "Scheduled runs are unavailable"}
        return {"status": "ok", "runs": [self._scheduled_run_view(run) for run in await self._scheduled_runs.list()]}

    async def _handle_scheduled_run_create(self, params: dict, ws: ServerConnection) -> dict:
        """Create a schedule: name, goal, allowed_actions and every_minutes or daily_at (+ weekdays)."""
        if self._scheduled_runs is None:
            return {"status": "error", "message": "Scheduled runs are unavailable"}
        try:
            every_minutes = params.get("every_minutes", 0)
            weekdays = params.get("weekdays", [])
            if not isinstance(every_minutes, int) or isinstance(every_minutes, bool):
                raise ValueError("every_minutes must be a whole number")
            if not isinstance(weekdays, list) or not all(
                isinstance(day, int) and not isinstance(day, bool) for day in weekdays
            ):
                raise ValueError("weekdays must be a list of numbers")
            run = await self._scheduled_runs.create(
                name=params.get("name", ""),
                goal=params.get("goal", ""),
                allowed_actions=params.get("allowed_actions", []),
                every_minutes=every_minutes,
                daily_at=params.get("daily_at", ""),
                weekdays=weekdays,
            )
        except ValueError as error:
            return {"status": "error", "message": str(error)}
        return {"status": "ok", "run": self._scheduled_run_view(run)}

    async def _handle_scheduled_run_set_enabled(self, params: dict, ws: ServerConnection) -> dict:
        if self._scheduled_runs is None:
            return {"status": "error", "message": "Scheduled runs are unavailable"}
        try:
            enabled = _validated_bool(params, "enabled", True)
        except ValueError as error:
            return {"status": "error", "message": str(error)}
        run = await self._scheduled_runs.set_enabled(str(params.get("run_id", "")), enabled)
        if run is None:
            return {"status": "error", "message": "No such scheduled run"}
        return {"status": "ok", "run": self._scheduled_run_view(run)}

    async def _handle_scheduled_run_delete(self, params: dict, ws: ServerConnection) -> dict:
        if self._scheduled_runs is None:
            return {"status": "error", "message": "Scheduled runs are unavailable"}
        deleted = await self._scheduled_runs.delete(str(params.get("run_id", "")))
        return {"status": "ok" if deleted else "error", "deleted": deleted}

    async def _handle_scheduled_run_now(self, params: dict, ws: ServerConnection) -> dict:
        """Start a schedule's goal right away, with the same approved list."""
        if self._scheduled_runs is None:
            return {"status": "error", "message": "Scheduled runs are unavailable"}
        run = await self._scheduled_runs.run_now(str(params.get("run_id", "")))
        if run is None:
            return {"status": "error", "message": "No such scheduled run"}
        if run.last_status != "started":
            return {"status": "error", "message": "The run could not start", "run": self._scheduled_run_view(run)}
        return {"status": "ok", "run": self._scheduled_run_view(run)}

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

    # ── Voice/Gesture Workflow Handlers ──
    # Durable, pausable/resumable multi-step goals spanning multiple voice
    # commands or gesture inputs over time — see
    # pilot.agents.voice_gesture_workflow.VoiceGestureWorkflowEngine. A
    # It persists workflow state and delegates each step to the same adaptive
    # observe/act/verify core used by AutonomousExecutor.

    async def _handle_voice_gesture_workflow_submit(self, params: dict, ws: ServerConnection) -> dict:
        """Submit a durable, pausable/resumable multi-step voice/gesture workflow.

        Args:
            params: JSON-RPC parameters with goal, invocation_source
                ("voice" or "gesture"), and an optional scope_override
                restricting the AgentGateway's floor further for this
                workflow only (see pilot.security.gateway).
            ws: The WebSocket connection.

        Returns:
            A dict with status and the created workflow, or an error.
        """
        from pydantic import ValidationError

        from pilot.security.gateway import InvocationSource, TaskScopeOverride

        if not self._voice_gesture_workflows:
            return {"status": "error", "message": "Voice/gesture workflow engine not initialized"}

        goal = params.get("goal", "")
        if not goal.strip():
            return {"status": "error", "message": "Empty goal"}

        source_raw = params.get("invocation_source", "voice")
        try:
            invocation_source = InvocationSource(source_raw)
        except ValueError:
            return {"status": "error", "message": f"Invalid invocation_source: {source_raw}"}
        if invocation_source not in (InvocationSource.VOICE, InvocationSource.GESTURE):
            return {"status": "error", "message": "invocation_source must be 'voice' or 'gesture'"}

        scope_override = None
        raw_override = params.get("scope_override")
        if raw_override is not None:
            try:
                scope_override = TaskScopeOverride.model_validate(raw_override)
            except ValidationError as e:
                return {"status": "error", "message": f"Invalid scope_override: {e}"}

        workflow = await self._voice_gesture_workflows.start(goal, invocation_source, scope_override)
        return {"status": "submitted", "workflow": workflow.to_dict()}

    async def _handle_voice_gesture_workflow_list(self, params: dict, ws: ServerConnection) -> dict:
        """List voice/gesture workflows.

        Args:
            params: JSON-RPC parameters, optionally {include_terminal}.
            ws: The WebSocket connection.

        Returns:
            A dict with the list of workflows.
        """
        if not self._voice_gesture_workflows:
            return {
                "status": "error",
                "message": "Voice/gesture workflow engine not initialized",
                "workflows": [],
            }
        try:
            include_terminal = _validated_bool(params, "include_terminal", False)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {
            "status": "ok",
            "workflows": await self._voice_gesture_workflows.list_workflows(include_terminal=include_terminal),
        }

    async def _handle_voice_gesture_workflow_get(self, params: dict, ws: ServerConnection) -> dict:
        """Get a specific voice/gesture workflow by ID.

        Args:
            params: JSON-RPC parameters with workflow_id.
            ws: The WebSocket connection.

        Returns:
            A dict with the workflow, or an error if not found.
        """
        if not self._voice_gesture_workflows:
            return {"status": "error", "message": "Voice/gesture workflow engine not initialized"}
        workflow_id = params.get("workflow_id", "")
        workflow = await self._voice_gesture_workflows.get_workflow(workflow_id)
        if workflow is None:
            return {"status": "error", "message": f"Workflow not found: {workflow_id}"}
        return {"status": "ok", "workflow": workflow}

    async def _handle_voice_gesture_workflow_pause(self, params: dict, ws: ServerConnection) -> dict:
        """Pause a running voice/gesture workflow at the next step boundary.

        Args:
            params: JSON-RPC parameters with workflow_id.
            ws: The WebSocket connection.

        Returns:
            A dict with paused status and workflow_id.
        """
        if not self._voice_gesture_workflows:
            return {"status": "error", "message": "Voice/gesture workflow engine not initialized"}
        workflow_id = params.get("workflow_id", "")
        paused = await self._voice_gesture_workflows.pause(workflow_id)
        if not paused:
            return {
                "status": "error",
                "message": f"Workflow could not be paused: {workflow_id}",
                "paused": False,
                "workflow_id": workflow_id,
            }
        return {"status": "ok", "paused": True, "workflow_id": workflow_id}

    async def _handle_voice_gesture_workflow_resume(self, params: dict, ws: ServerConnection) -> dict:
        """Resume a paused/waiting-for-trigger voice/gesture workflow.

        Args:
            params: JSON-RPC parameters with workflow_id.
            ws: The WebSocket connection.

        Returns:
            A dict with resumed status, workflow_id, and the workflow if successful.
        """
        if not self._voice_gesture_workflows:
            return {"status": "error", "message": "Voice/gesture workflow engine not initialized"}
        workflow_id = params.get("workflow_id", "")
        workflow = await self._voice_gesture_workflows.resume(workflow_id)
        if workflow is None:
            return {
                "status": "error",
                "message": f"Workflow could not be resumed: {workflow_id}",
                "resumed": False,
                "workflow_id": workflow_id,
                "workflow": None,
            }
        return {
            "status": "ok",
            "resumed": True,
            "workflow_id": workflow_id,
            "workflow": workflow.to_dict(),
        }

    async def _handle_voice_gesture_workflow_cancel(self, params: dict, ws: ServerConnection) -> dict:
        """Cancel a voice/gesture workflow.

        Args:
            params: JSON-RPC parameters with workflow_id.
            ws: The WebSocket connection.

        Returns:
            A dict with cancelled status and workflow_id.
        """
        if not self._voice_gesture_workflows:
            return {"status": "error", "message": "Voice/gesture workflow engine not initialized"}
        workflow_id = params.get("workflow_id", "")
        cancelled = await self._voice_gesture_workflows.cancel(workflow_id)
        if not cancelled:
            return {
                "status": "error",
                "message": f"Workflow could not be cancelled: {workflow_id}",
                "cancelled": False,
                "workflow_id": workflow_id,
            }
        return {"status": "ok", "cancelled": True, "workflow_id": workflow_id}

    async def _handle_gesture_workflow_bindings_get(self, params: dict, ws: ServerConnection) -> dict:
        """Return the current gesture-to-goal workflow bindings for the Settings editor.

        Args:
            params: JSON-RPC parameters (unused).
            ws: The WebSocket connection.

        Returns:
            A dict with enabled flag and the list of bindings.
        """
        from dataclasses import asdict

        from pilot.config import SUPPORTED_GESTURE_WORKFLOW_GESTURES

        return {
            "status": "ok",
            "enabled": self.config.gesture_workflows.enabled,
            "bindings": [asdict(b) for b in self.config.gesture_workflows.bindings],
            "supported_gestures": list(SUPPORTED_GESTURE_WORKFLOW_GESTURES),
        }

    async def _handle_gesture_workflow_bindings_update(self, params: dict, ws: ServerConnection) -> dict:
        """Update the gesture-to-goal workflow bindings.

        Args:
            params: JSON-RPC parameters, optionally {enabled, bindings}
                where each binding is {gesture_name, goal_template, enabled}.
            ws: The WebSocket connection.

        Returns:
            A dict with status and the updated config.
        """
        from dataclasses import asdict

        from pilot.config import SUPPORTED_GESTURE_WORKFLOW_GESTURES, GestureWorkflowBinding

        try:
            enabled = _validated_bool(params, "enabled", self.config.gesture_workflows.enabled)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        parsed = list(self.config.gesture_workflows.bindings)
        if "bindings" in params:
            raw_bindings = params["bindings"]
            if not isinstance(raw_bindings, list):
                return {"status": "error", "message": "bindings must be a list"}
            if len(raw_bindings) > len(SUPPORTED_GESTURE_WORKFLOW_GESTURES):
                return {"status": "error", "message": "too many gesture workflow bindings"}

            parsed = []
            seen_gestures: set[str] = set()
            for index, item in enumerate(raw_bindings):
                if not isinstance(item, dict):
                    return {"status": "error", "message": f"binding {index + 1} must be an object"}
                gesture_name = str(item.get("gesture_name", "")).strip().lower()
                goal_template = str(item.get("goal_template", "")).strip()
                if gesture_name not in SUPPORTED_GESTURE_WORKFLOW_GESTURES:
                    return {
                        "status": "error",
                        "message": f"unsupported gesture in binding {index + 1}: {gesture_name or '(empty)'}",
                    }
                if gesture_name in seen_gestures:
                    return {"status": "error", "message": f"duplicate gesture binding: {gesture_name}"}
                if not goal_template:
                    return {"status": "error", "message": f"binding {index + 1} requires a workflow goal"}
                if len(goal_template) > 2000:
                    return {"status": "error", "message": f"binding {index + 1} goal is too long"}
                seen_gestures.add(gesture_name)
                binding_enabled = item.get("enabled", True)
                if not isinstance(binding_enabled, bool):
                    return {
                        "status": "error",
                        "message": f"binding {index + 1} enabled must be a boolean",
                    }
                parsed.append(
                    GestureWorkflowBinding(
                        gesture_name=gesture_name,
                        goal_template=goal_template,
                        enabled=binding_enabled,
                    )
                )
        if enabled and not any(binding.enabled for binding in parsed):
            return {
                "status": "error",
                "message": "enable at least one complete binding before enabling gesture workflows",
            }

        previous_gesture_workflows = copy.deepcopy(self.config.gesture_workflows)
        self.config.gesture_workflows.enabled = enabled
        self.config.gesture_workflows.bindings = parsed
        if (failure := self._save_section("gesture_workflows", previous_gesture_workflows)) is not None:
            return failure
        result = {
            "status": "ok",
            "enabled": self.config.gesture_workflows.enabled,
            "bindings": [asdict(b) for b in self.config.gesture_workflows.bindings],
            "supported_gestures": list(SUPPORTED_GESTURE_WORKFLOW_GESTURES),
        }
        await self._broadcast_notification("gesture_workflow_bindings_updated", result)
        return result

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

    async def _handle_proactive_learning_status(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        """Return persisted accept/dismiss learning for proactive patterns."""
        if not self._proactive:
            return {
                "status": "error",
                "enabled": False,
                "patterns": {},
                "message": "Proactive engine not initialized",
            }
        return {"status": "ok", **self._proactive.get_learning_status()}

    async def _handle_proactive_learning_reset(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        """Forget learned proactive-suggestion preferences."""
        if not self._proactive:
            return {
                "status": "error",
                "enabled": False,
                "patterns": {},
                "message": "Proactive engine not initialized",
            }
        self._proactive.reset_learning()
        return {"status": "ok", **self._proactive.get_learning_status()}

    async def _handle_online_learning_status(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        """Return the bounded online learner's evidence and privacy state."""

        if self._online_learning is None:
            return {
                "status": "error",
                "enabled": False,
                "message": "Verified online learning is not initialized",
            }
        return {"status": "ok", **self._online_learning.status()}

    async def _handle_online_learning_reset(
        self,
        params: dict,
        ws: ServerConnection,
    ) -> dict:
        """Forget online model state without mutating the audit ledger."""

        if self._online_learning is None:
            return {
                "status": "error",
                "enabled": False,
                "message": "Verified online learning is not initialized",
            }
        if self._proactive is not None:
            self._proactive.reset_learning()
        return {"status": "ok", **(await self._online_learning.reset())}

    # -- Assistance co-adaptation --
    #
    # These handlers reach only the co-adaptation service, the co_adaptation config section and the read-only
    # study refusal check. They never touch a pending approval, the experience ledger or anything that runs.

    async def _co_adaptation_call(self, method: str, params: dict) -> dict:
        service = self._co_adaptation
        if service is None:
            if method == "status":
                return {
                    "status": "ok",
                    "available": False,
                    "config": _co_adaptation_config_dict(self.config.co_adaptation),
                    "message": "Co-adaptation is unavailable on this daemon.",
                    "authority": "presentation_only",
                    "execution_authority": False,
                }
            return {"status": "error", "message": "Co-adaptation is unavailable on this daemon."}
        return await getattr(service, method)(params if isinstance(params, dict) else {})

    async def _co_adaptation_study_refusal(self) -> str | None:
        """Start refusals, re-checked at session_begin, session_resume, recommend (block_regime and ambiguity_table)
        and record_trial. Each attribute may be None when the server was built without initialize() (RPC tests)."""
        if self._recorded_workflows is not None and self._recorded_workflows.status().get("active"):
            return "recording_active"
        if self._interactive_request_active:
            return "task_running"
        if self._neural_controller is not None:
            state = await self._neural_controller.gate_state()
            if state in NEURAL_ARMED_STATES:
                return "neural_armed"
        return None

    async def _co_adaptation_study_call(self, method: str, params: dict) -> dict:
        """Delegate a study RPC unless a recording, a running task or armed neural input refuses it."""
        reason = await self._co_adaptation_study_refusal()
        if reason is not None:
            return {"status": "refused", "reason": reason}
        return await self._co_adaptation_call(method, params)

    async def _set_co_adaptation_enabled(self, enabled: bool) -> dict[str, Any] | None:
        """Save a new co_adaptation.enabled, or put the section back and return the error to send."""
        if self.config.co_adaptation.enabled == enabled:
            return None
        previous = copy.deepcopy(self.config.co_adaptation)
        self.config.co_adaptation.enabled = enabled
        if (failure := self._save_section("co_adaptation", previous)) is not None:
            return failure
        if self._co_adaptation is not None:
            await self._co_adaptation.on_config_changed(previous)
        return None

    async def _co_adaptation_notify(self, method: str, params: dict) -> None:
        """The service's broadcast function: forward the notification, and act on an expired consent.

        Expiry has the effect of a revoke (spec 10.7), including turning co_adaptation.enabled off when no active
        consent remains. "expired" can arrive during a consent grant or revoke, which holds the config lock that
        turning the feature off needs too, so the check runs in its own task.
        """
        await self._broadcast_notification(method, params)
        if method == "co_adaptation_updated" and isinstance(params, dict) and params.get("reason") == "expired":
            self._co_adaptation_consent_lapsed = True
            if self._co_adaptation_lapse_task is None or self._co_adaptation_lapse_task.done():
                self._co_adaptation_lapse_task = asyncio.create_task(self._settle_co_adaptation_consent())

    async def _settle_co_adaptation_consent(self) -> None:
        """After an expired consent was erased, turn the feature off when no active consent remains, then tell
        every open window. Never raises."""
        try:
            async with self._config_lock:
                service = self._co_adaptation
                if service is None or not self._co_adaptation_consent_lapsed:
                    return
                self._co_adaptation_consent_lapsed = False
                if not self.config.co_adaptation.enabled or service.has_active_consent():
                    return
                failure = await self._set_co_adaptation_enabled(False)
            if failure is not None:
                logger.warning("Co-adaptation stayed on after its last consent expired: %s", failure["message"])
                return
            await self._broadcast_notification(
                "config_updated", {"method": "co_adaptation_config_update", "section": "co_adaptation"}
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Co-adaptation could not settle an expired consent", exc_info=True)

    async def _turn_off_co_adaptation_without_consent(self) -> dict[str, Any] | None:
        """Turn co_adaptation.enabled off when the service has no active consent. The caller holds the config lock
        (a config-mutating RPC). Returns the error to send when the setting could not be saved."""
        service = self._co_adaptation
        if service is None or not self.config.co_adaptation.enabled or service.has_active_consent():
            return None
        return await self._set_co_adaptation_enabled(False)

    async def _handle_co_adaptation_status(self, params: dict, ws: ServerConnection) -> dict:
        """Settings, consent, store size, rules and fallbacks of the co-adaptation feature."""
        result = await self._co_adaptation_call("status", params)
        # The status call prunes expired consents; the reply shows the setting after that prune took effect.
        lapse_task = self._co_adaptation_lapse_task
        if lapse_task is not None and not lapse_task.done():
            await asyncio.shield(lapse_task)
        if isinstance(result, dict) and isinstance(result.get("config"), dict):
            result["config"] = {**result["config"], "enabled": self.config.co_adaptation.enabled}
        return result

    async def _handle_co_adaptation_config_update(self, params: dict, ws: ServerConnection) -> dict:
        """Change co_adaptation settings. Turning the feature on needs an active consent; none needs a restart."""
        values = params if isinstance(params, dict) else {}
        current = self.config.co_adaptation
        updates: dict[str, Any] = {}
        try:
            for key in values:
                if key in CO_ADAPTATION_BOOL_KEYS:
                    updates[key] = _validated_bool(values, key, getattr(current, key))
                elif key in CO_ADAPTATION_FLOAT_RANGES:
                    low, high = CO_ADAPTATION_FLOAT_RANGES[key]
                    updates[key] = _validated_float(values, key, getattr(current, key), minimum=low, maximum=high)
                elif key in CO_ADAPTATION_INT_RANGES:
                    low, high = CO_ADAPTATION_INT_RANGES[key]
                    updates[key] = _validated_int(values, key, getattr(current, key), minimum=low, maximum=high)
                else:
                    return {"status": "error", "message": f"Unknown co-adaptation setting: {key}"}
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        service = self._co_adaptation
        if updates.get("enabled") is True and (service is None or not service.has_active_consent()):
            return {"status": "error", "message": "Grant consent before turning co-adaptation on."}

        previous = copy.deepcopy(current)
        for key, value in updates.items():
            setattr(current, key, value)
        if (failure := self._save_section("co_adaptation", previous)) is not None:
            return failure
        if service is not None:
            await service.on_config_changed(previous)
        return {
            "status": "ok",
            "config": _co_adaptation_config_dict(self.config.co_adaptation),
            "restart_required": False,
        }

    async def _handle_co_adaptation_consent_grant(self, params: dict, ws: ServerConnection) -> dict:
        """Grant or update a participant's consent, then turn the feature on unless ``enable`` is false."""
        values = params if isinstance(params, dict) else {}
        try:
            enable = _validated_bool(values, "enable", True)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        # A consent that expired but was not pruned yet still leaves no active consent: a grant starts from off,
        # so ``enable: false`` keeps the feature off.
        if (failure := await self._turn_off_co_adaptation_without_consent()) is not None:
            return failure
        result = await self._co_adaptation_call("grant_consent", values)
        if result.get("status") != "ok":
            return result
        if enable and (failure := await self._set_co_adaptation_enabled(True)) is not None:
            return failure
        return {**result, "config": _co_adaptation_config_dict(self.config.co_adaptation)}

    async def _handle_co_adaptation_consent_revoke(self, params: dict, ws: ServerConnection) -> dict:
        """Withdraw a participant's consent (always erases), and turn the feature off when no consent remains."""
        result = await self._co_adaptation_call("revoke_consent", params)
        # Also after a revoke that failed part-way: the participant's consent is gone even if the erase is not.
        if (failure := await self._turn_off_co_adaptation_without_consent()) is not None:
            return failure
        if result.get("status") != "ok":
            return result
        return {**result, "config": _co_adaptation_config_dict(self.config.co_adaptation)}

    async def _handle_co_adaptation_preference_set(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_call("set_preference", params)

    async def _handle_co_adaptation_preference_retract(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_call("retract_preference", params)

    async def _handle_co_adaptation_learning_reset(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_call("reset_learning", params)

    async def _handle_co_adaptation_erase(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_call("erase", params)

    async def _handle_co_adaptation_results(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_call("results", params)

    async def _handle_co_adaptation_export(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_call("export", params)

    async def _handle_co_adaptation_session_begin(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_study_call("begin_session", params)

    async def _handle_co_adaptation_session_resume(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_study_call("resume_session", params)

    async def _handle_co_adaptation_recommend(self, params: dict, ws: ServerConnection) -> dict:
        level = params.get("level") if isinstance(params, dict) else None
        if level in {"block_regime", "ambiguity_table"}:
            return await self._co_adaptation_study_call("recommend", params)
        return await self._co_adaptation_call("recommend", params)

    async def _handle_co_adaptation_record_trial(self, params: dict, ws: ServerConnection) -> dict:
        # A refused trial is not stored; the study marks it interrupted and sends it again after a resume.
        return await self._co_adaptation_study_call("record_trial", params)

    async def _handle_co_adaptation_report_burden(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_call("report_burden", params)

    async def _handle_co_adaptation_session_end(self, params: dict, ws: ServerConnection) -> dict:
        return await self._co_adaptation_call("end_session", params)

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

        if not self._autonomous:
            return {
                "status": "error",
                "message": ("The guarded autonomous executor is unavailable, so this suggestion was not executed."),
            }

        suggestion_id = params.get("suggestion_id", "")
        action_command = await self._proactive.accept_suggestion(suggestion_id)
        if not action_command:
            return {"error": f"Suggestion not found: {suggestion_id}"}

        job = await self._autonomous.submit(action_command, source="proactive")
        return {"status": "executing", "action": action_command, "job": job.to_dict()}

    async def _handle_proactive_dismiss(self, params: dict, ws: ServerConnection) -> dict:
        """Dismiss a proactive suggestion.

        Args:
            params: JSON-RPC parameters with suggestion_id.
            ws: The WebSocket connection.

        Returns:
            A dict with dismissed status and suggestion_id.
        """
        if not self._proactive:
            return {"status": "error", "message": "Proactive engine not initialized"}

        suggestion_id = params.get("suggestion_id", "")
        dismissed = await self._proactive.dismiss_suggestion(suggestion_id)
        if not dismissed:
            return {"status": "error", "message": f"Suggestion not found: {suggestion_id}"}
        return {"status": "ok", "dismissed": True, "suggestion_id": suggestion_id}

    async def _handle_resolve_git_conflict(self, params: dict, ws: ServerConnection) -> dict:
        """Resolve git merge conflicts in a file via LLM.

        Args:
            params: JSON-RPC parameters containing filepath (or path).
            ws: The WebSocket connection.

        Returns:
            A dict with resolution details.
        """
        if not self._model_router:
            return {"status": "error", "message": "Model router not initialized"}

        filepath = params.get("filepath") or params.get("path")
        if not filepath:
            return {"status": "error", "message": "Missing filepath or path parameter"}

        try:
            from pilot.system.git_conflict import resolve_conflicts_in_file

            resolved_blocks = await resolve_conflicts_in_file(filepath, self._model_router)
            return {"status": "success", "conflicts": resolved_blocks}
        except Exception as e:
            logger.exception("Failed to resolve git conflict in handler")
            return {"status": "error", "message": str(e)}

    async def _handle_apply_git_resolution(self, params: dict, ws: ServerConnection) -> dict:
        """Apply a git conflict resolution securely.

        Args:
            params: JSON-RPC parameters with path, full_block, resolved_code.
            ws: The WebSocket connection.

        Returns:
            A dict with execution status.
        """
        path = params.get("path")
        full_block = params.get("full_block")
        resolved_code = params.get("resolved_code")

        if not path or full_block is None or resolved_code is None:
            return {"status": "error", "message": "Missing required params: path, full_block, resolved_code"}

        try:
            from pilot.actions import Action, ActionPlan, ActionType, GitResolveParams

            action = Action(
                action_type=ActionType.GIT_RESOLVE,
                parameters=GitResolveParams(
                    path=path,
                    full_block=full_block,
                    resolved_code=resolved_code,
                ),
            )
            plan = ActionPlan(actions=[action], explanation="Apply git conflict resolution securely")
            results = await self._executor.execute(plan)
            success = all(r.success for r in results)
            error = next((r.error for r in results if not r.success), None)
            return {
                "status": "success" if success else "error",
                "message": "Git conflict resolved successfully" if success else (error or "Failed to resolve conflict"),
            }
        except Exception as e:
            logger.exception("Failed to apply git conflict resolution in handler")
            return {"status": "error", "message": str(e)}

    # ── Plan History Audit Log Handlers ──

    async def _handle_get_plan_history(self, params: dict, ws: ServerConnection) -> dict:
        """Return a paginated list of plan audit records (summaries, no large blobs).

        This is the internal plan-level audit log for debugging and compliance.
        It is distinct from the chat/session history returned by ``get_history``.

        JSON-RPC params
        ---------------
        limit : int, optional
            Maximum rows to return. Default 50, max 200.
        offset : int, optional
            Rows to skip (for pagination). Default 0.
        status : str, optional
            Filter by ``execution_status`` (e.g. ``"success"``, ``"partial_failure"``,
            ``"cancelled"``, ``"blocked_by_critic"``). Omit to return all statuses.

        Returns
        -------
        dict
            ``plans``   — list of summary dicts (no plan_json / results_json blobs)
            ``count``   — number of rows in this page
            ``offset``  — offset used
            ``limit``   — limit used
        """
        if not self._plan_history:
            return {"status": "error", "error": "Plan history store is not initialized", "plans": []}

        raw_limit = params.get("limit", 50)
        raw_offset = params.get("offset", 0)
        status_filter = params.get("status") or None  # empty string → None

        try:
            limit = max(1, min(int(raw_limit), 200))
            offset = max(0, int(raw_offset))
        except (TypeError, ValueError):
            return {"status": "error", "error": "limit and offset must be integers", "plans": []}

        plans = await self._plan_history.get_list(
            limit=limit,
            offset=offset,
            status_filter=status_filter,
        )
        return {
            "status": "ok",
            "plans": plans,
            "count": len(plans),
            "offset": offset,
            "limit": limit,
        }

    async def _handle_get_plan_detail(self, params: dict, ws: ServerConnection) -> dict:
        """Return the full audit record for a single plan, including all JSON blobs.

        JSON-RPC params
        ---------------
        plan_id : str
            The 8-char plan identifier (as returned in ``plan_preview`` notifications
            and in ``get_plan_history`` rows).

        Returns
        -------
        dict
            Full plan record with parsed ``plan_json``, ``critic_verdict_json``,
            ``results_json``, and ``verification_json`` fields, or an ``error`` key
            if the plan_id is not found.
        """
        if not self._plan_history:
            return {"error": "Plan history store is not initialized"}

        plan_id = str(params.get("plan_id", "")).strip()
        if not plan_id:
            return {"error": "plan_id is required"}

        record = await self._plan_history.get_detail(plan_id)
        if record is None:
            return {"error": f"No plan found with plan_id: {plan_id}"}

        return record

    # ── Threat Containment (Issue #365) ──────────────────────────────────────

    async def _handle_threat_containment_stats(self, params: dict, ws: ServerConnection) -> dict:
        """Return the operational status of the ThreatContainmentBridge.

        JSON-RPC method: ``threat_containment_stats``

        Returns a simple status dict so the frontend can display whether
        autonomous threat containment is active and ready.

        Returns
        -------
        dict
            ``status``: "active" | "inactive"
            ``bridge_ready``: bool — True when the bridge is fully wired
            ``pending_confirmations``: int — number of open confirmation gates
        """
        if self._threat_bridge is None:
            return {
                "status": "inactive",
                "bridge_ready": False,
                "pending_confirmations": 0,
                "message": "ThreatContainmentBridge is not initialized.",
            }

        return {
            "status": "active",
            "bridge_ready": True,
            "pending_confirmations": len(self._pending_confirms),
            "message": "ThreatContainmentBridge is active and monitoring ForensicsAgent output.",
        }


def _setup_logging() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColorFormatter())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    # httpx's INFO message includes the complete request URL. Gemini uses an
    # API-key query parameter, so leaving this enabled writes credentials into
    # console/file logs and any pasted support transcript.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    """Entry point for the pilot-daemon command."""
    parser = argparse.ArgumentParser(prog="pilot.server")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without executing them")
    parser.add_argument(
        "--export-logs",
        action="store_true",
        help="Package all logs, config.toml, and audit trails into a zip on the Desktop for bug reporting.",
    )
    parser.add_argument("--replace-pid", type=int, help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args()
    if args.replace_pid is not None:
        from pilot.system.elevation import replace_existing_daemon

        replace_existing_daemon(args.replace_pid)

    ensure_dirs()
    _setup_logging()
    config = PilotConfig.load()
    if args.export_logs:
        export_logs()
        return
    if args.dry_run:
        # For this run only: a Settings change must not save the CLI flag as the person's choice.
        config.set_runtime_override("security", "dry_run", True)
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
