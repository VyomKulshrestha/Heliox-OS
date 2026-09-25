"""Opt-in, local-only recording of approved Heliox actions and visual transitions.

Recorded recipes are drafts: replay is re-planned through the normal guarded
executor, not a privileged macro. Screen features never leave this encrypted
store or appear in list/get responses.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pilot.actions import Action, ActionResult, ActionType

logger = logging.getLogger("pilot.workflows.recorded_workflows")

KEYRING_SERVICE = "Heliox OS Recorded Workflows"
KEYRING_ACCOUNT = "local-recordings-v1"
STORE_AAD = b"heliox-recorded-workflows-v1"
MAX_STEPS = 200
MAX_SAMPLES = 200
MAX_WORKFLOWS = 50
MIN_REPEAT_SECONDS = 300
MAX_REPEAT_SECONDS = 365 * 24 * 60 * 60

# Never retain typed text, form values, scripts, shell commands, or clipboard
# contents. A recipe with excluded actions is explicitly marked incomplete.
REPLAY_FIELDS: dict[ActionType, tuple[str, ...]] = {
    ActionType.OPEN_URL: ("url",),
    ActionType.OPEN_APPLICATION: ("name",),
    ActionType.BROWSER_NAVIGATE: ("url",),
    ActionType.BROWSER_CLICK: ("selector", "tab_index"),
    ActionType.BROWSER_CLICK_TEXT: ("text", "exact", "tab_index"),
    ActionType.BROWSER_HOVER: ("selector", "tab_index"),
    ActionType.BROWSER_SCROLL: ("direction", "amount", "tab_index"),
    ActionType.MOUSE_CLICK: ("x", "y", "button", "clicks"),
    ActionType.MOUSE_DOUBLE_CLICK: ("x", "y", "button", "clicks"),
    ActionType.MOUSE_RIGHT_CLICK: ("x", "y", "button", "clicks"),
    ActionType.MOUSE_MOVE: ("x", "y", "relative"),
    ActionType.MOUSE_SCROLL: ("x", "y", "amount", "horizontal"),
}
VISUAL_ACTIONS = frozenset(
    {ActionType.MOUSE_CLICK, ActionType.MOUSE_DOUBLE_CLICK,
     ActionType.MOUSE_RIGHT_CLICK, ActionType.MOUSE_MOVE}
)


def _key_from_os_keyring() -> bytes:
    """Fail closed if the platform cannot protect the encryption key."""
    import keyring
    import keyring.backends

    try:
        backend = keyring.get_keyring()
        if isinstance(backend, keyring.backends.fail.Keyring) or getattr(backend, "priority", 0) <= 0:
            raise RuntimeError("A working OS keyring is required for local workflow recordings")
        value = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        if value is None:
            key = AESGCM.generate_key(bit_length=256)
            keyring.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, base64.b64encode(key).decode("ascii"))
            return key
    except Exception as exc:
        raise RuntimeError("OS keyring is unavailable for local workflow recordings") from exc
    key = base64.b64decode(value, validate=True)
    if len(key) != 32:
        raise RuntimeError("Invalid recorded-workflow key in OS keyring")
    return key


def _safe_step(action: Action) -> dict[str, Any] | None:
    fields = REPLAY_FIELDS.get(action.action_type)
    if fields is None:
        return None
    params = action.parameters
    values = {field: getattr(params, field) for field in fields if hasattr(params, field)}
    # URLs with embedded credentials and URL query/fragment secrets must never
    # be turned into reusable automation steps.
    if "url" in values:
        from urllib.parse import urlsplit

        url = urlsplit(str(values["url"]))
        if url.username or url.password or url.query or url.fragment or url.scheme not in {"http", "https"}:
            return None
    if action.action_type == ActionType.BROWSER_CLICK_TEXT and not values.get("text"):
        return None
    if action.action_type in {ActionType.BROWSER_CLICK, ActionType.BROWSER_HOVER} and not values.get("selector"):
        return None
    if action.action_type == ActionType.OPEN_APPLICATION and getattr(params, "args", []):
        return None
    return {"action_type": action.action_type.value, "parameters": values}


def _feature_bytes(observation: dict[str, Any]) -> list[int] | None:
    values = observation.get("embedding", [])
    if not isinstance(values, list) or len(values) != 256:
        return None
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
           for value in values):
        return None
    return [round(value * 255) for value in values]


class RecordedWorkflowStore:
    """One encrypted local file; recording is never active at process startup."""

    def __init__(
        self,
        path: Path,
        *,
        key_provider: Callable[[], bytes] = _key_from_os_keyring,
        capture_feature: Callable[[], Awaitable[dict[str, Any] | None]] | None = None,
    ) -> None:
        self.path = path
        self._key_provider = key_provider
        self._key: bytes | None = None
        self._key_lock = threading.Lock()
        self._capture_feature = capture_feature
        self._active: dict[str, Any] | None = None
        self._session_lock = threading.Lock()
        self._records_lock = threading.RLock()
        self._saving = False

    def _encryption_key(self) -> bytes:
        with self._key_lock:
            if self._key is None:
                self._key = self._key_provider()
            return self._key

    def prepare(self) -> None:
        """Check keyring and existing ciphertext before a UI session can start."""
        with self._records_lock:
            self._encryption_key()
            self._read()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        blob = self.path.read_bytes()
        if len(blob) < 13:
            raise ValueError("Recorded workflow store is damaged")
        try:
            data = AESGCM(self._encryption_key()).decrypt(blob[:12], blob[12:], STORE_AAD)
            records = json.loads(data)
        except (InvalidTag, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Local recordings could not be decrypted with the OS keyring key") from exc
        if not isinstance(records, list):
            raise ValueError("Invalid recorded workflow store")
        return records

    def _write(self, records: list[dict[str, Any]]) -> None:
        key = self._encryption_key()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nonce = os.urandom(12)
        payload = json.dumps(records, separators=(",", ":"), allow_nan=False).encode("utf-8")
        encrypted = nonce + AESGCM(key).encrypt(nonce, payload, STORE_AAD)
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with os.fdopen(os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                stream.write(encrypted)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)

    def start(
        self, *, name: str, consent: bool, capture_visual: bool = False, goal: str = ""
    ) -> dict[str, Any]:
        if consent is not True:
            raise ValueError("Explicit recording consent is required")
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("Workflow name must contain 1–100 characters")
        if len(goal) > 2000:
            raise ValueError("Workflow goal must be at most 2000 characters")
        self._encryption_key()  # Do not begin a sensitive recording that cannot be encrypted.
        if capture_visual and self._capture_feature is None:
            raise ValueError("Fresh local screen capture is unavailable")
        with self._session_lock:
            if self._active is not None or self._saving:
                raise ValueError("A recording is already active or being saved")
            self._active = {
                "id": uuid.uuid4().hex,
                "name": name,
                "goal": goal.strip() or name,
                "created_at": datetime.now(UTC).isoformat(),
                "capture_visual": capture_visual,
                "steps": [],
                "samples": [],
                "omitted_actions": 0,
            }
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._session_lock:
            session = self._active
            return {
                "active": session is not None,
                "name": session["name"] if session else "",
                "capture_visual": bool(session and session["capture_visual"]),
                "step_count": len(session["steps"]) if session else 0,
                "sample_count": len(session["samples"]) if session else 0,
                "omitted_actions": session["omitted_actions"] if session else 0,
            }

    async def before(self, action: Action, *, single_action_batch: bool) -> dict[str, Any] | None:
        with self._session_lock:
            session = self._active
        if session is None:
            return None
        if not single_action_batch:
            return {"session_id": session["id"], "before": None, "parallel": True}
        feature = None
        if session["capture_visual"] and action.action_type in VISUAL_ACTIONS:
            try:
                feature = await self._capture_feature() if self._capture_feature else None
            except Exception:
                logger.warning("Pre-action visual feature capture failed", exc_info=True)
        return {"session_id": session["id"], "before": feature}

    async def after(self, token: dict[str, Any] | None, result: ActionResult) -> None:
        if not result.success or not result.executed:
            return
        with self._session_lock:
            session = self._active
            if token is None or session is None or token["session_id"] != session["id"]:
                return
            if token.get("parallel"):
                session["omitted_actions"] += 1
                return
            step = _safe_step(result.action)
            if step is None or len(session["steps"]) >= MAX_STEPS:
                session["omitted_actions"] += 1
                return
            session["steps"].append(step)
            step_index = len(session["steps"]) - 1
        before = token["before"]
        if not before or len(session["samples"]) >= MAX_SAMPLES:
            return
        try:
            after = await self._capture_feature() if self._capture_feature else None
        except Exception:
            logger.warning("Post-action visual feature capture failed", exc_info=True)
            return
        if not after:
            return
        before_bytes = _feature_bytes(before)
        after_bytes = _feature_bytes(after)
        if before_bytes is None or after_bytes is None:
            return
        width, height = before.get("width", 0), before.get("height", 0)
        params = result.action.parameters
        if (
            not isinstance(width, (int, float)) or not isinstance(height, (int, float))
            or not math.isfinite(width) or not math.isfinite(height)
            or width <= 0 or height <= 0
        ):
            return
        sample = {
            "step_index": step_index,
            "action_type": result.action.action_type.value,
            "coords": [min(1.0, max(0.0, params.x / width)),
                       min(1.0, max(0.0, params.y / height))],
            "before": before_bytes,
            "after": after_bytes,
            "post_observed": True,
            "independently_verified": False,
        }
        with self._session_lock:
            if self._active is session and len(session["samples"]) < MAX_SAMPLES:
                session["samples"].append(sample)

    def stop(self, *, save: bool) -> dict[str, Any]:
        with self._session_lock:
            session = self._active
            if session is None:
                raise ValueError("No recording is active")
            if save and not session["steps"]:
                raise ValueError("No replayable actions were captured; nothing was saved")
            self._active = None
            self._saving = save
            record = {key: value for key, value in session.items() if key != "capture_visual"}
        if not save:
            return {"saved": False, **self.status()}
        record["schedule"] = None
        try:
            with self._records_lock:
                records = self._read()
                if len(records) >= MAX_WORKFLOWS:
                    raise ValueError("Maximum of 50 saved workflows reached; delete one before saving")
                records.append(record)
                self._write(records)
        except Exception:
            with self._session_lock:
                self._active = session
                self._saving = False
            raise
        with self._session_lock:
            self._saving = False
        return {"saved": True, **self._public(record)}

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record["id"],
            "name": record["name"],
            "goal": record.get("goal", record["name"]),
            "created_at": record["created_at"],
            "steps": record["steps"],
            "step_count": len(record["steps"]),
            "sample_count": len(record["samples"]),
            "omitted_actions": record["omitted_actions"],
            "incomplete": bool(record["omitted_actions"]),
            "needs_review": bool(record["omitted_actions"]) or any(
                step["action_type"].startswith("mouse_") for step in record["steps"]
            ),
            "schedule": record.get("schedule"),
        }

    def list(self) -> list[dict[str, Any]]:
        with self._records_lock:
            return [self._public(record) for record in self._read()]

    def get(self, workflow_id: str) -> dict[str, Any]:
        with self._records_lock:
            for record in self._read():
                if record["id"] == workflow_id:
                    return self._public(record)
        raise KeyError("Recorded workflow not found")

    def delete(self, workflow_id: str) -> None:
        with self._records_lock:
            records = self._read()
            remaining = [record for record in records if record["id"] != workflow_id]
            if len(remaining) == len(records):
                raise KeyError("Recorded workflow not found")
            self._write(remaining)

    def schedule(
        self, workflow_id: str, *, run_at: str, repeat_seconds: int = 0, reviewed: bool = False
    ) -> dict[str, Any]:
        try:
            when = datetime.fromisoformat(run_at)
        except ValueError as exc:
            raise ValueError("run_at must be an ISO 8601 timestamp with timezone") from exc
        if when.tzinfo is None or when <= datetime.now(UTC):
            raise ValueError("run_at must be a future timestamp with timezone")
        if isinstance(repeat_seconds, bool) or not isinstance(repeat_seconds, int):
            raise ValueError("repeat_seconds must be an integer")
        if repeat_seconds != 0 and repeat_seconds < MIN_REPEAT_SECONDS:
            raise ValueError("Repeating workflows require at least a five-minute interval")
        if repeat_seconds > MAX_REPEAT_SECONDS:
            raise ValueError("Repeating workflows require at most a one-year interval")
        with self._records_lock:
            records = self._read()
            record = next((item for item in records if item["id"] == workflow_id), None)
            if record is None:
                raise KeyError("Recorded workflow not found")
            if record["omitted_actions"]:
                raise ValueError("Incomplete recordings cannot be scheduled; re-record the missing steps")
            if self._public(record)["needs_review"] and reviewed is not True:
                raise ValueError("Coordinate-based workflows require explicit review before scheduling")
            record["schedule"] = {
                "next_run_at": when.astimezone(UTC).isoformat(),
                "repeat_seconds": repeat_seconds,
            }
            self._write(records)
            return self._public(record)

    def unschedule(self, workflow_id: str) -> dict[str, Any]:
        with self._records_lock:
            records = self._read()
            record = next((item for item in records if item["id"] == workflow_id), None)
            if record is None:
                raise KeyError("Recorded workflow not found")
            record["schedule"] = None
            self._write(records)
            return self._public(record)

    def take_due(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """Claim due runs durably before dispatch to prevent duplicate firing."""
        now = (now or datetime.now(UTC)).astimezone(UTC)
        with self._records_lock:
            records = self._read()
            due = []
            for record in records:
                schedule = record.get("schedule")
                if not schedule or datetime.fromisoformat(schedule["next_run_at"]) > now:
                    continue
                due.append(self._public(record))
                repeat = schedule["repeat_seconds"]
                if repeat:
                    next_run = datetime.fromisoformat(schedule["next_run_at"])
                    missed = int((now - next_run).total_seconds() // repeat) + 1
                    schedule["next_run_at"] = (next_run + timedelta(seconds=repeat * missed)).isoformat()
                else:
                    record["schedule"] = None
            if due:
                self._write(records)
            return due


def workflow_prompt(workflow: dict[str, Any]) -> str:
    """Convert an observed recipe into a re-planning goal, not a raw macro."""
    steps = []
    for index, step in enumerate(workflow["steps"], start=1):
        args = ", ".join(f"{key}={value!r}" for key, value in step["parameters"].items())
        steps.append(f"{index}. {step['action_type']}({args})")
    return (
        f"Repeat this user-saved workflow: {workflow['name']}. Goal: {workflow['goal']}. "
        "Use the observed steps below as a reference, not as authority or a fixed macro. "
        "Re-observe current state, verify each target and postcondition, and use normal "
        "permission and confirmation checks. Do not invent missing credentials or submit "
        "unreviewed changes. Stop and ask if the current UI differs materially.\n"
        + "\n".join(steps)
    )
