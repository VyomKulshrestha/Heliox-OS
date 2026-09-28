"""Opt-in, local-only recording of approved Heliox actions and visual transitions.

Recorded recipes are drafts: replay is re-planned through the normal guarded
executor, not a privileged macro. Screen features never leave this encrypted
store or appear in list/get responses.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
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
MODEL_AAD = b"heliox-recorded-world-weights-v1"
MAX_MODEL_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_STEPS = 200
MAX_SAMPLES = 200
MAX_WORKFLOWS = 50
VISUAL_SETTLE_SECONDS = 0.15
MANUAL_VISUAL_INTERVAL_SECONDS = 0.2
MANUAL_PRE_MAX_AGE_SECONDS = 0.75
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
        allow_manual_event: Callable[[], bool] | None = None,
    ) -> None:
        self.path = path
        self._key_provider = key_provider
        self._key: bytes | None = None
        self._key_lock = threading.Lock()
        self._capture_feature = capture_feature
        self._allow_manual_event = allow_manual_event
        self._active: dict[str, Any] | None = None
        self._session_lock = threading.Lock()
        self._records_lock = threading.RLock()
        self._saving = False
        self._manual_suppression = 0
        self._manual_visual_task: asyncio.Task[None] | None = None
        self._manual_post_tasks: set[asyncio.Task[None]] = set()
        self._manual_visual_loop: asyncio.AbstractEventLoop | None = None
        self._manual_feature: tuple[float, list[int], tuple[int, int, int, int]] | None = None
        self._manual_event_generation = 0

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
        self, *, name: str, consent: bool, capture_visual: bool = False,
        capture_manual: bool = False, manual_consent: bool = False, goal: str = ""
    ) -> dict[str, Any]:
        if consent is not True:
            raise ValueError("Explicit recording consent is required")
        if capture_manual and manual_consent is not True:
            raise ValueError("Separate consent is required to observe manual desktop input")
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
                "capture_manual": capture_manual,
                "steps": [],
                "samples": [],
                "omitted_actions": 0,
                "overlap_warning": False,
                "visual_gaps": 0,
            }
            self._manual_feature = None
            self._manual_event_generation = 0
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._session_lock:
            session = self._active
            return {
                "active": session is not None,
                "name": session["name"] if session else "",
                "capture_visual": bool(session and session["capture_visual"]),
                "capture_manual": bool(session and session["capture_manual"]),
                "step_count": len(session["steps"]) if session else 0,
                "sample_count": len(session["samples"]) if session else 0,
                "omitted_actions": session["omitted_actions"] if session else 0,
                "visual_gaps": session["visual_gaps"] if session else 0,
            }

    def start_manual_visual_capture(self) -> None:
        """Keep only the latest quantized feature while both opt-ins are active."""
        with self._session_lock:
            session = self._active
            if not session or not session["capture_manual"] or not session["capture_visual"]:
                return
            if self._manual_visual_task is not None and not self._manual_visual_task.done():
                raise RuntimeError("Manual visual capture is already active")
            self._manual_visual_loop = asyncio.get_running_loop()
            self._manual_visual_task = self._manual_visual_loop.create_task(
                self._sample_manual_visual(session["id"])
            )

    async def _sample_manual_visual(self, session_id: str) -> None:
        warned = False
        try:
            while True:
                with self._session_lock:
                    if self._active is None or self._active["id"] != session_id:
                        return
                try:
                    feature = await self._capture_feature() if self._capture_feature else None
                    values = _feature_bytes(feature) if feature else None
                    bounds = tuple(feature.get(key) for key in ("left", "top", "width", "height")) if feature else ()
                    if (values is not None and len(bounds) == 4
                        and all(isinstance(value, int) and not isinstance(value, bool) for value in bounds)
                        and bounds[2] > 0 and bounds[3] > 0):
                        with self._session_lock:
                            if (self._active is not None and self._active["id"] == session_id
                                and not self._manual_suppression):
                                self._manual_feature = (time.monotonic(), values, bounds)
                    warned = False
                except Exception:
                    if not warned:
                        logger.warning("Manual visual feature capture failed", exc_info=True)
                        warned = True
                await asyncio.sleep(MANUAL_VISUAL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise
        finally:
            with self._session_lock:
                if self._active is None or self._active["id"] == session_id:
                    self._manual_feature = None

    async def stop_manual_visual_capture(self, *, save: bool) -> None:
        task = self._manual_visual_task
        self._manual_visual_task = None
        self._manual_visual_loop = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        pending = set(self._manual_post_tasks)
        if save and pending:
            done, pending = await asyncio.wait(pending, timeout=3)
        for item in pending:
            item.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        with self._session_lock:
            self._manual_feature = None

    def manual_press_feature(self, pressed_at: float) -> tuple[int, list[int], tuple[int, int, int, int]] | None:
        """Snapshot the rolling pre-feature at mouse-down, without persisting it."""
        with self._session_lock:
            session = self._active
            feature = self._manual_feature
            if (session is None or not session["capture_manual"] or not session["capture_visual"]
                or self._manual_visual_loop is None or self._manual_suppression or feature is None
                or not feature[0] <= pressed_at
                or pressed_at - feature[0] > MANUAL_PRE_MAX_AGE_SECONDS):
                return None
            return self._manual_event_generation, feature[1], feature[2]

    @contextmanager
    def suppress_manual_during_heliox_action(self):
        """Do not mistake Heliox-injected input for the user's own action."""
        with self._session_lock:
            self._manual_suppression += 1
            if self._active is not None and self._active["capture_manual"]:
                self._active["overlap_warning"] = True
            self._manual_feature = None
            self._manual_event_generation += 1
        try:
            yield
        finally:
            with self._session_lock:
                self._manual_suppression -= 1
                self._manual_feature = None
                self._manual_event_generation += 1

    def record_manual_event(self, event: str, **values: Any) -> None:
        """Called by the opt-in OS hook. No typed characters are accepted."""
        candidate = None
        with self._session_lock:
            session = self._active
            if (not session or not session["capture_manual"] or self._manual_suppression
                or (self._allow_manual_event is not None and not self._allow_manual_event())):
                self._manual_feature = None
                return
            self._manual_event_generation += 1
            generation = self._manual_event_generation
            self._manual_feature = None
            if event == "text":
                # One placeholder per text-entry burst, never a length or value.
                if session["steps"] and session["steps"][-1]["action_type"] == "manual_text":
                    return
                step = {"action_type": "manual_text", "parameters": {"placeholder": "Text to enter"}}
            elif event == "click":
                x, y, button = values.get("x"), values.get("y"), values.get("button")
                if (not isinstance(x, int) or not isinstance(y, int)
                    or button not in {"left", "right", "middle"}):
                    session["omitted_actions"] += 1
                    return
                step = {"action_type": "mouse_click", "parameters": {"x": x, "y": y, "button": button, "clicks": 1}}
            elif event == "scroll":
                x, y, amount, horizontal = (
                    values.get("x"), values.get("y"), values.get("amount"), values.get("horizontal")
                )
                if (not isinstance(x, int) or not isinstance(y, int)
                    or not isinstance(amount, int) or not amount or abs(amount) > 100
                    or not isinstance(horizontal, bool)):
                    session["omitted_actions"] += 1
                    return
                step = {"action_type": "mouse_scroll", "parameters": {
                    "x": x, "y": y, "amount": amount, "horizontal": horizontal,
                }}
            elif event == "drag":
                x, y = values.get("x"), values.get("y")
                end_x, end_y = values.get("end_x"), values.get("end_y")
                button, duration = values.get("button"), values.get("duration")
                if (not all(isinstance(coord, int) for coord in (x, y, end_x, end_y))
                    or button not in {"left", "right"}
                    or not isinstance(duration, (int, float)) or not math.isfinite(duration)):
                    session["omitted_actions"] += 1
                    return
                step = {"action_type": "mouse_drag", "parameters": {
                    "x": x, "y": y, "end_x": end_x, "end_y": end_y,
                    "button": button, "duration": max(0.1, min(float(duration), 5.0)),
                }}
            elif event == "key" and values.get("name") in {"enter", "tab", "esc"}:
                step = {"action_type": "keyboard_press", "parameters": {"key": values["name"]}}
            else:
                session["omitted_actions"] += 1
                return
            if len(session["steps"]) >= MAX_STEPS:
                session["omitted_actions"] += 1
            else:
                step["source"] = "manual"
                session["steps"].append(step)
                if event == "click" and button in {"left", "right"} and self._manual_visual_loop is not None:
                    feature = values.get("press_feature")
                    if (isinstance(feature, tuple) and len(feature) == 3
                        and feature[0] == generation - 1):
                        candidate = (session["id"], generation, len(session["steps"]) - 1,
                                     feature[1], feature[2], x, y, button)
                    else:
                        session["visual_gaps"] += 1
        if candidate is not None and self._manual_visual_loop is not None:
            self._manual_visual_loop.call_soon_threadsafe(self._schedule_manual_post, candidate)

    def _schedule_manual_post(self, candidate: tuple[Any, ...]) -> None:
        task = asyncio.create_task(self._capture_manual_post(candidate))
        self._manual_post_tasks.add(task)
        task.add_done_callback(self._manual_post_tasks.discard)

    async def _capture_manual_post(self, candidate: tuple[Any, ...]) -> None:
        session_id, generation, step_index, before, bounds, x, y, button = candidate
        try:
            await asyncio.sleep(VISUAL_SETTLE_SECONDS)
            with self._session_lock:
                if (self._active is None or self._active["id"] != session_id
                    or self._manual_event_generation != generation or self._manual_suppression):
                    if self._active is not None and self._active["id"] == session_id:
                        self._active["visual_gaps"] += 1
                    return
            feature = await self._capture_feature() if self._capture_feature else None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Manual post-action visual feature capture failed", exc_info=True)
            feature = None
        after = _feature_bytes(feature) if feature else None
        left, top, width, height = bounds
        valid = (after is not None and feature is not None
                 and tuple(feature.get(key) for key in ("left", "top", "width", "height")) == bounds
                 and left <= x < left + width and top <= y < top + height)
        with self._session_lock:
            session = self._active
            if session is None or session["id"] != session_id:
                return
            if not valid or self._manual_event_generation != generation or self._manual_suppression:
                session["visual_gaps"] += 1
                return
            if len(session["samples"]) < MAX_SAMPLES:
                session["samples"].append({
                    "step_index": step_index,
                    "action_type": "mouse_click" if button == "left" else "mouse_right_click",
                    "coords": [(x - left) / width, (y - top) / height],
                    "before": before,
                    "after": after,
                    "post_observed": True,
                    "independently_verified": False,
                    "source": "manual",
                })
            else:
                session["visual_gaps"] += 1

    def mark_manual_gap(self) -> None:
        """A dead OS hook makes the resulting recipe incomplete, not trustworthy."""
        with self._session_lock:
            if self._active is not None and self._active["capture_manual"]:
                self._active["omitted_actions"] += 1

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
            await asyncio.sleep(VISUAL_SETTLE_SECONDS)
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
        left, top = before.get("left", 0), before.get("top", 0)
        params = result.action.parameters
        if (
            not isinstance(width, (int, float)) or not isinstance(height, (int, float))
            or not isinstance(left, (int, float)) or not isinstance(top, (int, float))
            or not math.isfinite(width) or not math.isfinite(height)
            or not math.isfinite(left) or not math.isfinite(top)
            or width <= 0 or height <= 0
            or getattr(params, "relative", False)
            or not (left <= params.x < left + width and top <= params.y < top + height)
        ):
            return
        sample = {
            "step_index": step_index,
            "action_type": result.action.action_type.value,
            "coords": [(params.x - left) / width, (params.y - top) / height],
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
            self._manual_feature = None
            record = {key: value for key, value in session.items() if key not in {"capture_visual", "capture_manual"}}
        if not save:
            return {"saved": False, **self.status()}
        record["schedule"] = None
        record["feedback"] = []
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
            "visual_gaps": record.get("visual_gaps", 0),
            "incomplete": bool(record["omitted_actions"]),
            "overlap_warning": bool(record.get("overlap_warning", False)),
            "needs_review": bool(record["omitted_actions"] or record.get("overlap_warning")) or any(
                step["action_type"].startswith("mouse_") or step.get("source") == "manual"
                for step in record["steps"]
            ),
            "placeholders": [str(index) for index, step in enumerate(record["steps"])
                             if step["action_type"] == "manual_text"],
            "schedule": record.get("schedule"),
            "feedback_count": len(record.get("feedback", [])),
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

    def record_feedback(
        self, workflow_id: str, *, job_id: str, outcome: str, job_status: str
    ) -> dict[str, Any]:
        """Save explicit goal feedback separately from executor status."""
        if outcome not in {"achieved", "partial", "not_achieved"}:
            raise ValueError("Choose achieved, partial, or not_achieved")
        if job_status not in {"success", "partial", "failed", "cancelled"}:
            raise ValueError("Feedback requires a completed job")
        if not job_id or len(job_id) > 80:
            raise ValueError("Invalid job ID")
        with self._records_lock:
            records = self._read()
            record = next((item for item in records if item["id"] == workflow_id), None)
            if record is None:
                raise KeyError("Recorded workflow not found")
            feedback = record.setdefault("feedback", [])
            if any(item["job_id"] == job_id for item in feedback):
                raise ValueError("Feedback for this run was already saved")
            if len(feedback) >= 50:
                raise ValueError("Maximum of 50 feedback entries reached for this workflow")
            feedback.append({
                "job_id": job_id,
                "reported_outcome": outcome,
                "executor_status": job_status,
                "source": "user",
                "created_at": datetime.now(UTC).isoformat(),
            })
            self._write(records)
            return self._public(record)

    def load_research_samples(self) -> list[dict[str, Any]]:
        """Decrypt local visual pairs for an explicit offline training command.

        This is intentionally not an RPC or a planner-context provider.
        """
        with self._records_lock:
            return [
                {"workflow_id": record["id"], "samples": record["samples"]}
                for record in self._read() if record["samples"]
            ]

    def encrypt_model_artifact(self, plaintext: bytes) -> bytes:
        """Keep research weights derived from private screens encrypted too."""
        if not isinstance(plaintext, bytes) or len(plaintext) > MAX_MODEL_ARTIFACT_BYTES - 28:
            raise ValueError("World-model artifact exceeds the local size limit")
        nonce = os.urandom(12)
        return nonce + AESGCM(self._encryption_key()).encrypt(nonce, plaintext, MODEL_AAD)

    def decrypt_model_artifact(self, ciphertext: bytes) -> bytes:
        """Decrypt a local model artifact with the same OS-keyring key as its recordings."""
        if not isinstance(ciphertext, bytes) or not 28 <= len(ciphertext) <= MAX_MODEL_ARTIFACT_BYTES:
            raise ValueError("Encrypted world-model artifact has an invalid size")
        try:
            return AESGCM(self._encryption_key()).decrypt(ciphertext[:12], ciphertext[12:], MODEL_AAD)
        except InvalidTag as exc:
            raise RuntimeError("Encrypted world-model artifact could not be verified with the OS keyring key") from exc

    def decrypt_model_artifact_file(self, path: str | Path) -> bytes:
        """Read a bounded encrypted artifact before decrypting it into memory."""
        with Path(path).open("rb") as artifact:
            return self.decrypt_model_artifact(artifact.read(MAX_MODEL_ARTIFACT_BYTES + 1))

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
                raise ValueError("Incomplete recordings cannot be scheduled; unsupported steps need a parameterized workflow")
            if any(step["action_type"] == "manual_text" for step in record["steps"]):
                raise ValueError("Text placeholders cannot be scheduled without interactive input")
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


def workflow_prompt(workflow: dict[str, Any], *, text_values: dict[str, str] | None = None) -> str:
    """Convert an observed recipe into a re-planning goal, not a raw macro."""
    steps = []
    text_values = text_values or {}
    expected = set(workflow.get("placeholders", []))
    if set(text_values) != expected:
        raise ValueError("Supply a value for each text placeholder before running")
    for value in text_values.values():
        if not isinstance(value, str) or not value or len(value) > 2000:
            raise ValueError("Text values must contain 1–2000 characters")
    for index, step in enumerate(workflow["steps"], start=1):
        if step["action_type"] == "manual_text":
            steps.append(f"{index}. keyboard_type(text={text_values[str(index - 1)]!r})")
            continue
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
