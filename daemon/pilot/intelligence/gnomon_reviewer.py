"""Gnomon as an escalation-only reviewer for autonomous work.

Gnomon (the typed-decision model) answers the questions it was trained on about
a proposed desktop action. During autonomous jobs, scheduled runs, MCP tasks and
voice workflows Heliox asks it four of them:

* scope: would the plan stay inside what the person asked?
* harm: how bad would the plan be if it were wrong?
* untrusted instruction: is the plan following text found in tool or screen
  output rather than the person's request?
* completion: can the task truthfully be reported as verified complete?

Every answer can only add caution: require approval, or reject a completion
claim so the loop keeps working. Gnomon never approves, never lowers a tier and
never runs anything; when it is not loaded, out of memory or slow, Heliox
proceeds on its deterministic rules, world model and approvals alone.

The model needs a CUDA GPU and about 2.5 GiB of free RAM, takes ~30 s to load
and ~0.2-0.6 s per question afterwards. It is loaded in the background on first
use, never on the voice path itself, and unloaded after a quiet period. When a
load fails for lack of memory, the reviewer checks free memory every minute
(psutil, plus the CUDA query only if CUDA is already set up) and loads again on
its own once there is room, backing off after each failed attempt.

By default the engine is ``gnomon_worker.GnomonWorkerEngine``: the model lives in a supervised worker process. When
that worker dies, hangs, answers garbage or hits a CUDA driver error mid-review, the review counts as not done (an
escalation already found still stands), the worker is restarted after a back-off, and after repeated faults the GPU
arbiter switches the GPU off for the session and the status says so.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import Any

from pilot.timeouts import bounded

logger = logging.getLogger("pilot.intelligence.gnomon_reviewer")

SCOPE_QUESTION = "Would carrying out the proposed action stay inside what the user explicitly asked?"
WARM_UP_STATE = {"proposed_action": "Open Notepad.", "user_message": "open Notepad"}
HARM_QUESTION = "Grade the harm the action could cause if it were wrong."
HARM_LEVELS = [
    "Read-only; no change",
    "Reversible local change",
    "External communication, sharing, or publication",
    "Irreversible deletion, payment, or credential exposure",
]
UNTRUSTED_QUESTION = "Does the user's own instruction, rather than the content, permit this action?"
COMPLETION_QUESTION = "Can the assistant truthfully report this task as verified complete?"
LATEST_INSTRUCTION_QUESTION = "Is this proposed action still wanted after the user's later message?"
_QUESTION_NAMES = {SCOPE_QUESTION: "scope", HARM_QUESTION: "harm", UNTRUSTED_QUESTION: "untrusted content"}
# One of each question a review asks, the untrusted one with screen text as long as a review reads (1,500 characters).
_WARM_UP_ACTION = "Copy the file notes.txt from C:\\work into the folder D:\\backup."
_WARM_UP_SCREEN = " ".join(["Confirmation Required. Copy post-final.md into launch-release; delete the drafts."] * 20)
WARM_UP_QUESTIONS = (
    (WARM_UP_STATE, "noul", SCOPE_QUESTION, ["no", "yes"]),
    ({"proposed_action": _WARM_UP_ACTION}, "score", HARM_QUESTION, HARM_LEVELS),
    ({"content_source": "tool or screen output", "observed_text": _WARM_UP_SCREEN[:1500],
      "proposed_action": _WARM_UP_ACTION, "request": "Back up notes.txt to D:"}, "noul", UNTRUSTED_QUESTION, ["no", "yes"]),
)  # fmt: skip

# Escalate only on confident answers, so a borderline score never blocks ordinary work.
SCOPE_MIN_YES = 0.25
UNTRUSTED_MIN_YES = 0.25
HARM_ESCALATE_PROBABILITY = 0.6
COMPLETION_MIN_YES = 0.2

# Actions whose effects never leave this PC and are undone by doing the opposite. The harm
# question is skipped for plans made only of these: asked about "show a notification",
# Gnomon can mistake it for messaging someone. Scope and injection checks still apply.
LOCAL_ONLY_ACTIONS = frozenset(
    {
        "notify",
        "open_application",
        "window_focus",
        "window_minimize",
        "window_maximize",
        "volume_set",
        "brightness_set",
        "screenshot",
        "mouse_move",
        "mouse_scroll",
    }
)

CALL_TIMEOUT_SECONDS = 8.0
IDLE_UNLOAD_SECONDS = 600.0
RETRY_AFTER_FAILURE_SECONDS = 300.0
RESOURCE_CHECK_SECONDS = 60.0
RESOURCE_RETRY_MAX_SECONDS = 900.0
MAX_TEXT_CHARS = 1200
GIB = 1024**3
# Matches the loader guard in typed_decision.require_available_host_memory, which runs first.
HOST_RAM_FLOOR_BYTES = int(2.5 * GIB)

_REQUIRED_GIB = re.compile(r"at least ([0-9]+(?:\.[0-9]+)?) GiB", re.IGNORECASE)


@dataclass
class ResourceShortage:
    """Why Gnomon could not load when the cause was memory, with the numbers to show the person."""

    resource: str  # "memory" (system RAM) or "graphics memory" (CUDA)
    required_bytes: int | None
    free_bytes: int | None = None


def classify_resource_shortage(error: BaseException) -> ResourceShortage | None:
    """Read a load failure; None unless it says memory was short."""
    message = str(error)
    lowered = message.lower()
    match = _REQUIRED_GIB.search(message)
    required = int(float(match.group(1)) * GIB) if match else None
    if "system ram" in lowered or isinstance(error, MemoryError):
        return ResourceShortage("memory", required)
    if "cuda memory" in lowered or "out of memory" in lowered:
        return ResourceShortage("graphics memory", required)
    return None


# Messages that mean the CUDA context itself is broken. Nothing recovers it inside this process.
_DRIVER_ERROR_MARKERS = (
    "cuda error: unknown error",
    "cuda_error_unknown",
    "sticky error",
    "device-side assert",
    "cuda error: an illegal memory access",
)
DRIVER_ERROR_REASON = "The graphics driver reported an error while loading. Restart Heliox to try again."


def is_driver_error(error: BaseException) -> bool:
    """True when a load failure says the CUDA context is unrecoverable (not a memory shortage)."""
    lowered = str(error).lower()
    return any(marker in lowered for marker in _DRIVER_ERROR_MARKERS)


# Gnomon in its own process (gnomon_worker.GnomonWorkerEngine): what the person reads after the worker faulted.
_FAULT_WORDS = {
    "exit": "Gnomon's process stopped",
    "hang": "Gnomon's process stopped answering",
    "garbage": "Gnomon's process answered garbage",
    "driver": "The graphics driver reported an error in Gnomon's process",
    "start": "Gnomon's process could not start",
}
STOPPED_DURING_REVIEW = "Gnomon's process stopped during this review, so some checks did not run."
GPU_OFF_SUFFIX = " Gnomon needs the graphics card, so it does not review plans until then."


def is_worker_fault(error: BaseException) -> bool:
    return bool(getattr(error, "worker_fault", False))


def _free_host_memory() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _free_gpu_memory() -> int | None:
    """Free CUDA memory, only when torch has already set CUDA up; never creates a context."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_initialized():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def _gb(value: int) -> str:
    return f"{round(value / GIB, 1):g}"


def describe_shortage(shortage: ResourceShortage) -> str:
    """Plain words for the services strip, e.g. 'Needs 2.5 GB of free memory, 1.8 GB free now'."""
    free = f", {_gb(shortage.free_bytes)} GB free now" if shortage.free_bytes is not None else ""
    if shortage.required_bytes is None:
        return f"Ran out of {shortage.resource} while loading{free}"
    return f"Needs {_gb(shortage.required_bytes)} GB of free {shortage.resource}{free}"


@dataclass
class GnomonVerdict:
    """What Gnomon added to a decision; ``escalate`` is the only thing callers act on."""

    escalate: bool = False
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    # What the person should know that is not a reason to ask: scope not reviewed, deletes kept by undo.
    notes: list[str] = field(default_factory=list)
    scope_reviewed: bool = True
    # False when the time budget ran out before every question was asked; what was found still counts.
    complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "Gnomon",
            "escalate": self.escalate,
            "reasons": self.reasons,
            "notes": self.notes,
            "scores": {key: round(value, 3) for key, value in self.scores.items()},
            "scope_reviewed": self.scope_reviewed,
            "complete": self.complete,
            "execution_authority": False,
        }


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_action(action: Any) -> str:
    """One plain-language clause for an action, in the style of Gnomon's training states."""
    kind = action.action_type.value
    params = action.parameters
    target = str(action.target or "").strip()

    def value(name: str) -> str:
        item = getattr(params, name, None)
        return str(item).strip() if item not in (None, "", [], {}) else ""

    if kind == "shell_command":
        args = " ".join(str(arg) for arg in (getattr(params, "args", None) or []))
        return f"Run the command `{(value('command') + ' ' + args).strip()}`"
    if kind in {"shell_script", "ssh_script"}:
        return f"Run this script: {_clip(value('script'), 300)}"
    if kind in {"code_execute", "code_generate_and_run"}:
        return f"Run {value('language') or 'code'}: {_clip(value('code') or value('task_description'), 300)}"
    if kind == "keyboard_type":
        enter = " and press Enter" if getattr(params, "press_enter", False) else ""
        return f"Type '{_clip(value('text'), 200)}'{enter}"
    if kind in {"api_send_email", "email_reply"}:
        return f"Email {value('to') or target}: {_clip(value('subject') + ' ' + (value('message') or value('reply_body')), 200)}"
    if kind in {"api_slack", "api_discord", "api_webhook"}:
        return f"Post to {value('channel') or target}: {_clip(value('message') or value('body'), 200)}"
    if kind == "mcp_call_tool":
        return f"Call the {value('mcp_server')} tool {value('mcp_tool')} with {_clip(str(getattr(params, 'mcp_arguments', {}) or {}), 200)}"
    if kind == "file_delete":
        return f"Delete {value('path') or target}"
    if kind in {"file_move", "file_copy"}:
        source, destination = value("path") or target, value("destination")
        src, dst = PureWindowsPath(source.replace("/", "\\")), PureWindowsPath(destination.replace("/", "\\"))
        if kind == "file_move" and destination and src.parent == dst.parent:
            return f"Rename {src.name} to {dst.name} in the same folder ({src.parent})"
        verb = "Move" if kind == "file_move" else "Copy"
        if destination and src.name == dst.name:
            return f"{verb} the file {src.name} from {src.parent} into the folder {dst.parent}"
        return f"{verb} {source} to {destination}"
    if kind == "file_write":
        return f"Write to {value('path') or target}: {_clip(value('content'), 160)}"
    if kind in {"open_url", "browser_navigate"}:
        return f"Open {value('url') or target}"
    if kind == "open_application":
        return f"Open the {value('name') or target} app on this computer"
    if kind == "notify":
        text = (value("summary") + " " + value("body")).strip() or target
        return f"Show a desktop notification on this computer's screen reading '{_clip(text, 160)}'"
    if kind == "volume_set":
        return f"Set this computer's volume to {value('level') or target}%"
    if kind == "brightness_set":
        return f"Set this screen's brightness to {value('level') or target}%"
    if kind == "clipboard_write":
        # The text matters: a password on the clipboard and an address on the clipboard are not the same plan.
        return f"Put '{_clip(value('text'), 160)}' on the clipboard"
    if kind == "process_kill":
        return f"Close {value('name') or target} by ending its process"
    if kind == "window_close":
        return f"Close the {value('title') or target} window"
    if kind in _WINDOW_VERBS:
        return f"{_WINDOW_VERBS[kind]} the {value('title') or target} window"
    if kind == "package_install":
        return f"Install the {value('name') or target} package"
    if kind == "download_file":
        return f"Download {target} to {value('destination') or 'the Downloads folder'}"
    label = kind.replace("_", " ")
    return f"{label.capitalize()} {target}".strip()


def describe_plan(plan: Any) -> str:
    return _clip("; then ".join(describe_action(action) for action in plan.actions) + ".", 900)


# -- multi-part requests --------------------------------------------------------------------------
#
# Gnomon was trained on one action, or one group of like actions, against the one request it answers. For a
# request with several parts ("put X in a new folder, delete the drafts and zip it") its scope answer is wrong
# too often to show: whole plan against whole request, it flagged 6 of 12 correct held-out plans and all 5
# launch-folder plans of 9 November 2026 (0.1-1% in scope); scored step by step against the matching part of
# the request it still flagged 5 of 12 and missed a near miss the whole-plan question caught. So the scope
# question is asked only for one-part requests, where gate G6 measured it, and a multi-part request is reported
# as "scope not reviewed". Harm and untrusted-instruction checks still run. Numbers: docs/GNOMON_TRAINING.md,
# "Multi-step plans" (10 November 2026).

_REQUEST_VERBS = (
    "add|archive|back up|clear|close|copy|create|delete|download|email|empty|get|install|make|maximize|minimize|"
    "move|mute|open|post|put|quit|remove|rename|save|send|set|sort|tidy|turn|write|zip"
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|:\s+|;\s*")
_PART_BREAK = re.compile(rf",\s*(?:and\s+|then\s+)?|\s+(?:and|then|and then)\s+(?=(?:{_REQUEST_VERBS})\b)", re.I)
_WINDOW_VERBS = {"window_minimize": "Minimize", "window_maximize": "Maximize", "window_focus": "Switch to"}
SCOPE_NOT_REVIEWED = (
    "Scope not reviewed: your request has several parts, and Gnomon judges scope reliably only for one-part requests."
)


def request_parts(request: str) -> list[str]:
    """The parts of a request, split at sentences, colons, semicolons, commas and 'and'/'then' before a
    desktop verb ("... into launch-release, delete the drafts and zip it" has three). [request] for one part."""
    parts: list[str] = []
    for sentence in _SENTENCE_BREAK.split(request):
        for piece in _PART_BREAK.split(sentence):
            piece = piece.strip(" .,")
            if len(piece.split()) >= 2 and piece not in parts:
                parts.append(piece)
    return parts if len(parts) > 1 else [request]


# A live correction is never added to the request Gnomon reads. Measured with v7 on the 10 November launch-folder
# revision: "<request> Correction: Skip the backup." made the untrusted-instruction check flag the revised plan
# (0.5% permitted), and appending any correction cut a correct eval plan from 76% to 4%; the original request alone
# gave 76% and left every planted-instruction plan flagged. A step that only a correction asked for is therefore
# judged against the original request, which can add an approval but never removes one.
REVIEWED_AGAINST_ORIGINAL = (
    "Reviewed against your original request: Gnomon does not read corrections, so a step only your correction "
    "asked for may be flagged."
)


def journal_undoable_delete(action: Any) -> bool:
    """A file delete the undo journal backs up before it runs (the approval dialog shows the same badge).

    Deciding this by rule rather than in the text Gnomon reads: Gnomon's harm family grades every delete as an
    irreversible deletion, so a sentence about the backup would argue with its training. The rule cannot hide a
    real loss: only Heliox's own file_delete qualifies, and only while undo is on; deletes by shell commands,
    package removals, emails and other external actions keep their grade.
    """
    from pilot.workflows.undo_journal import undo_support

    return action.action_type.value == "file_delete" and undo_support(action).kind == "undoable"


class GnomonReviewer:
    """Shares one loaded Gnomon model; reviews return None until it is ready."""

    def __init__(
        self,
        engine_provider: Callable[[], Any | None],
        *,
        call_timeout: float = CALL_TIMEOUT_SECONDS,
        idle_unload_seconds: float = IDLE_UNLOAD_SECONDS,
        resource_check_seconds: float = RESOURCE_CHECK_SECONDS,
    ) -> None:
        self._engine_provider = engine_provider
        self._call_timeout = call_timeout
        self._idle_unload_seconds = idle_unload_seconds
        self._state = "cold"  # cold | loading | ready | unavailable | disabled
        self._last_error = ""
        self._failed_at = float("-inf")  # never failed: 0.0 read as a failure just now, in the minutes after boot
        self._last_used = 0.0
        self._lock = asyncio.Lock()
        self._loader: threading.Thread | None = None
        self._loaded_engine: Any = None
        self._idle_task: asyncio.Task[None] | None = None
        self._resource_check_seconds = resource_check_seconds
        self._shortage: ResourceShortage | None = None
        self._resource_failures = 0  # load attempts in a row that failed for lack of memory
        self._resource_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._driver_error = False  # the CUDA context is broken; only a restart helps
        self._closed = False
        self._question_times: list[tuple[str, float, float]] = []  # (question, waited, answered) in seconds
        # Worker supervision (an engine in its own process): the last fault, a pending restart, the GPU switched
        # off for the session, or a card that is hot or busy right now.
        self._faults_seen = 0
        self._fault_kind = ""
        self._restart_pending = False
        self._restart_handle: asyncio.TimerHandle | None = None
        self._restart_at = 0.0
        self._gpu_off = False
        self._unsupported = False  # not supported on this system (no CUDA): said once, never retried
        self._gpu_wait_reason = ""

    # -- lifecycle -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        loaded = self._loaded_engine
        if self._state == "ready" and loaded is not None and getattr(loaded, "alive", True) is False:
            # The worker died while nobody was asking (killed, or a GPU reset): notice it here, not mid-review.
            note_exit = getattr(loaded, "note_exit", None)
            fault = note_exit() if callable(note_exit) else None
            if fault is not None:
                self._on_worker_fault(fault)
        state = self._state
        current = self._engine_provider()
        if state == "ready" and current is not self._loaded_engine:
            state = "cold"  # Settings replaced or switched off the engine since it loaded
        if state in {"cold", "disabled"}:
            state = "cold" if current is not None else "disabled"
        status: dict[str, Any] = {
            "state": state,
            "last_error": self._last_error,
            "execution_authority": False,
            "reason": "",
            "waiting_for": None,
            "retrying": False,
        }
        shortage = self._shortage
        if state == "unavailable" and shortage is not None:
            if shortage.resource == "memory":
                shortage.free_bytes = _free_host_memory()  # cheap; keeps "free now" current
            retrying = self._resource_task is not None and not self._resource_task.done()
            status["reason"] = describe_shortage(shortage) + (". Tries again on its own." if retrying else ".")
            status["waiting_for"] = {
                "resource": shortage.resource,
                "required_gb": round(shortage.required_bytes / GIB, 1) if shortage.required_bytes else None,
                "free_gb": round(shortage.free_bytes / GIB, 1) if shortage.free_bytes is not None else None,
            }
            status["retrying"] = retrying
        elif state == "unavailable" and self._driver_error:
            status["reason"] = DRIVER_ERROR_REASON
        elif state == "unavailable" and self._gpu_off:
            status["reason"] = self._last_error + ("" if self._unsupported else GPU_OFF_SUFFIX)
        elif state == "unavailable" and self._fault_kind:
            wait = max(0.0, self._restart_at - time.monotonic())
            restart = f" Restarting it in {wait:.0f} s." if self._restart_pending else ""
            status["reason"] = _FAULT_WORDS.get(self._fault_kind, "Gnomon's process failed") + "." + restart
            status["retrying"] = self._restart_pending
        elif state == "unavailable" and self._gpu_wait_reason:
            status["reason"] = self._gpu_wait_reason + (" Tries again on its own." if self._restart_pending else "")
            status["retrying"] = self._restart_pending
        elif state == "unavailable":
            status["reason"] = f"Could not load: {self._last_error}" if self._last_error else "Could not load."
        engine = current if current is not None else loaded
        worker_status = getattr(engine, "worker_status", None)
        worker = worker_status() if callable(worker_status) else None
        status["device"] = "cuda" if state == "ready" else None
        status["process"] = worker["process"] if worker else "daemon"
        status["worker"] = worker
        status["last_fault"] = worker.get("last_fault") if worker else None
        status["gpu_disabled"] = self._gpu_off and not self._unsupported
        status["supported"] = not self._unsupported
        return status

    def warm(self) -> None:
        """Start loading Gnomon in the background if it is enabled and not loaded; never blocks."""
        if self._state in {"loading", "ready"}:
            return
        if self._state == "unavailable":
            if self._driver_error:
                return  # a broken graphics driver context cannot recover inside this process
            if self._gpu_off or self._restart_pending:
                return  # GPU off for the session, or the worker's restart is already scheduled
            if self._resource_task is not None and not self._resource_task.done():
                return  # the memory watch owns retries while memory is short
            if time.monotonic() - self._failed_at < RETRY_AFTER_FAILURE_SECONDS:
                return
        self._start_load()

    def _start_load(self) -> None:
        engine = self._engine_provider()
        if engine is None:
            self._state = "disabled"
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        self._state = "loading"
        self._loader = threading.Thread(target=self._load, args=(engine,), name="gnomon-load", daemon=True)
        self._loader.start()
        self._ensure_idle_watch()

    def _load(self, engine: Any) -> None:
        try:
            engine.load()
        except Exception as error:  # out of memory, no CUDA, missing artifacts
            if is_worker_fault(error) or getattr(error, "gpu_unavailable", False):
                self._load_refused(engine, error)
                return
            shortage = classify_resource_shortage(error)
            if shortage is not None:
                shortage.free_bytes = _free_host_memory() if shortage.resource == "memory" else _free_gpu_memory()
                self._resource_failures += 1
            else:
                self._resource_failures = 0
            self._shortage = shortage
            self._driver_error = shortage is None and is_driver_error(error)
            self._last_error = str(error)[:300] or type(error).__name__
            self._failed_at = time.monotonic()
            self._state = "unavailable"
            logger.info("Gnomon unavailable for autonomous review: %s", self._last_error)
            if shortage is not None:
                self._schedule_resource_watch()
            return
        # The first forward pass sets up CUDA kernels (about 15 s for the 2B model), and the first harm question (four
        # answers scored together) and the first long untrusted-content question are slow again (harm 1.3 s cold
        # against 0.7 s warm on 10 November 2026). Pay all three while loading, not in the person's first review.
        for state, kind, question, options in WARM_UP_QUESTIONS:
            try:
                engine.score(state, kind, question, options)
            except Exception as error:
                if is_worker_fault(error):  # the worker died while warming up: not loaded after all
                    self._load_refused(engine, error)
                    return
                logger.info("Gnomon warm-up call failed: %s", str(error)[:200] or type(error).__name__)
                break
        self._loaded_engine = engine
        self._last_used = time.monotonic()
        self._shortage = None
        self._driver_error = False
        self._resource_failures = 0
        self._last_error = ""
        self._fault_kind = ""
        self._gpu_wait_reason = ""
        self._state = "ready"
        logger.info("Gnomon loaded for autonomous review")

    # -- a worker process that faulted, or a card that is not usable now --------

    def _load_refused(self, engine: Any, error: BaseException) -> None:
        """Loader thread: the worker faulted while loading, or the GPU arbiter said no."""
        self._shortage = None
        self._driver_error = False
        self._last_error = str(error)[:300] or type(error).__name__
        self._failed_at = time.monotonic()
        if is_worker_fault(error):
            self._on_worker_fault(error, engine)
            return
        self._fault_kind = ""
        self._gpu_wait_reason = str(error)
        retry = getattr(error, "retry_seconds", None)
        self._state = "unavailable"
        if retry is None:
            self._gpu_off = True
            self._unsupported = getattr(error, "code", "") == "unsupported"
            logger.info("Gnomon unavailable for this session: %s", self._last_error)
        else:
            logger.info("Gnomon waits for the graphics card: %s", self._last_error)
            self._schedule_restart(float(retry))

    def _on_worker_fault(self, error: BaseException, engine: Any = None) -> None:
        """The worker died, hung, answered garbage or hit a driver error; the engine has already killed it.

        The pending review answers "not reviewed"; the worker is restarted after the engine's back-off unless the
        arbiter has switched the GPU off for this session.
        """
        engine = engine if engine is not None else self._loaded_engine
        take_fault = getattr(engine, "take_fault", None)
        if callable(take_fault):
            take_fault()
        self._faults_seen += 1
        self._loaded_engine = None
        self._state = "unavailable"
        self._fault_kind = str(getattr(error, "kind", "") or "exit")
        self._last_error = str(error)[:300] or type(error).__name__
        self._failed_at = time.monotonic()
        if engine is not None and getattr(engine, "gpu_disabled", False):
            self._gpu_off = True
            self._last_error = str(getattr(engine, "gpu_disabled_reason", "") or self._last_error)
            self._restart_pending = False
            logger.warning("Gnomon stopped for this session: %s", self._last_error)
            return
        restart_delay = getattr(engine, "restart_delay", None)
        delay = float(restart_delay()) if callable(restart_delay) else RETRY_AFTER_FAILURE_SECONDS
        self._schedule_restart(delay)

    def _schedule_restart(self, delay: float) -> None:
        """From any thread: load again after ``delay`` seconds."""
        loop = self._loop
        if loop is None or self._closed:
            return
        self._restart_pending = True
        self._restart_at = time.monotonic() + delay
        try:
            loop.call_soon_threadsafe(self._arm_restart, delay)
        except RuntimeError:  # the loop has shut down
            self._restart_pending = False

    def _arm_restart(self, delay: float) -> None:
        if self._closed:
            return
        if self._restart_handle is not None:
            self._restart_handle.cancel()
        self._restart_handle = asyncio.get_running_loop().call_later(delay, self._restart_now)

    def _restart_now(self) -> None:
        self._restart_handle = None
        self._restart_pending = False
        if self._closed or self._state != "unavailable" or self._gpu_off:
            return
        if self._engine_provider() is None:
            self._state = "disabled"
            return
        logger.info("Starting Gnomon again")
        self._start_load()

    # -- waiting for memory ------------------------------------------------

    def _schedule_resource_watch(self) -> None:
        """Called from the loader thread: start the memory watch on the event loop."""
        loop = self._loop
        if loop is None or self._closed:
            return
        try:
            loop.call_soon_threadsafe(self._start_resource_watch)
        except RuntimeError:  # the loop has shut down
            pass

    def _start_resource_watch(self) -> None:
        if self._closed or self._shortage is None or self._state != "unavailable":
            return
        if self._resource_task is None or self._resource_task.done():
            self._resource_task = asyncio.get_running_loop().create_task(self._resource_watch())

    def _resources_sufficient(self) -> bool:
        """Cheap check before a load attempt; refreshes the free numbers shown in status()."""
        shortage = self._shortage
        if shortage is None:
            return True
        host_free = _free_host_memory()
        if shortage.resource == "memory":
            shortage.free_bytes = host_free
            needed = shortage.required_bytes or HOST_RAM_FLOOR_BYTES
            return host_free is None or host_free >= needed
        if host_free is not None and host_free < HOST_RAM_FLOOR_BYTES:
            return False  # the RAM guard runs before the GPU check; it would fail first
        gpu_free = _free_gpu_memory()
        if gpu_free is not None:
            shortage.free_bytes = gpu_free
            if shortage.required_bytes is not None:
                return gpu_free >= shortage.required_bytes
        return True  # cannot measure cheaply: the backoff below bounds the attempts

    async def _resource_watch(self) -> None:
        """Check free memory every interval; load once there is room. One attempt per back-off window."""
        backoff = min(
            self._resource_check_seconds * 2 ** max(self._resource_failures - 1, 0),
            max(RESOURCE_RETRY_MAX_SECONDS, self._resource_check_seconds),
        )
        next_attempt = time.monotonic() + backoff
        while not self._closed and self._state == "unavailable" and self._shortage is not None:
            await asyncio.sleep(self._resource_check_seconds)
            if self._closed or self._state != "unavailable" or self._shortage is None:
                return
            if self._engine_provider() is None:
                self._state = "disabled"
                return
            if time.monotonic() < next_attempt or not self._resources_sufficient():
                continue
            logger.info("Memory is free again; loading Gnomon")
            self._resource_task = None  # let warm() through and allow the next failure to start a new watch
            self._start_load()
            return

    def _ensure_idle_watch(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = loop.create_task(self._idle_watch())

    async def _idle_watch(self) -> None:
        while True:
            await asyncio.sleep(min(60.0, self._idle_unload_seconds))
            if self._state == "ready" and time.monotonic() - self._last_used > self._idle_unload_seconds:
                await self.unload()

    async def _cancel_resource_watch(self) -> None:
        task, self._resource_task = self._resource_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def unload(self) -> None:
        """Release the model this reviewer loaded (not whichever engine the config points at now)."""
        await self._cancel_resource_watch()
        async with self._lock:
            engine, self._loaded_engine = self._loaded_engine, None
            if engine is not None:
                await asyncio.to_thread(engine.unload)
            if self._state == "ready":
                self._state = "cold"

    async def close(self) -> None:
        self._closed = True
        if self._restart_handle is not None:
            self._restart_handle.cancel()
            self._restart_handle = None
        if self._idle_task is not None:
            self._idle_task.cancel()
            await asyncio.gather(self._idle_task, return_exceptions=True)
        await self.unload()

    # -- scoring ---------------------------------------------------------

    async def _score(self, state: dict[str, Any], kind: str, question: str, options: list[str]) -> dict | None:
        if self._state != "ready":
            self.warm()
            return None
        engine = self._engine_provider()
        if engine is None:
            return None
        if engine is not self._loaded_engine:  # the config changed and the engine was replaced
            self._state = "cold"
            self.warm()
            return None
        queued = time.monotonic()
        async with self._lock:
            started = time.monotonic()
            try:
                result = await bounded(
                    asyncio.to_thread(engine.score, state, kind, question, options), self._call_timeout
                )
            except Exception as error:
                logger.info("Gnomon review skipped: %s", str(error)[:200] or type(error).__name__)
                if is_worker_fault(error) and engine is self._loaded_engine:
                    self._on_worker_fault(error, engine)
                return None
            finally:
                self._last_used = time.monotonic()
                # Kept for the log line of a review that runs out of time: which question took how long.
                self._question_times.append(
                    (_QUESTION_NAMES.get(question, "question"), started - queued, self._last_used - started)
                )
        return result.get("answer") if isinstance(result, dict) and result.get("status") == "ok" else None

    async def _yes(self, state: dict[str, Any], question: str) -> float | None:
        answer = await self._score(state, "noul", question, ["no", "yes"])
        return float(answer["yes_probability"]) if answer else None

    async def review_plan(
        self,
        goal: str,
        plan: Any,
        untrusted_context: str = "",
        *,
        undo_enabled: bool = False,
        time_budget: float | None = None,
    ) -> GnomonVerdict | None:
        """Scope (one-part requests only), harm and (when tool/screen output fed this step) untrusted-instruction
        checks.

        ``undo_enabled``: Heliox's undo journal is on, so a file delete it backs up is not graded as an
        irreversible deletion (``journal_undoable_delete``); the rest of the plan is still graded.
        ``time_budget``: seconds; a question not started by then is skipped and the verdict is marked incomplete.
        Whatever was found before still escalates.
        """
        if self._state != "ready":
            self.warm()
            return None
        deadline = time.monotonic() + time_budget if time_budget is not None else None
        faults_before = self._faults_seen
        del self._question_times[:-50]  # bounded; this review's questions follow index asked_before
        asked_before = len(self._question_times)
        proposed = describe_plan(plan)
        request = _clip(goal, 600)
        verdict = GnomonVerdict()

        def out_of_time() -> bool:
            if deadline is not None and time.monotonic() >= deadline:
                verdict.complete = False
            return not verdict.complete

        if len(request_parts(request)) > 1:
            verdict.scope_reviewed = False
            verdict.notes.append(SCOPE_NOT_REVIEWED)
        elif not out_of_time():
            in_scope = await self._yes({"proposed_action": proposed, "user_message": request}, SCOPE_QUESTION)
            if in_scope is not None:
                verdict.scores["in_scope"] = in_scope
                if in_scope < SCOPE_MIN_YES:
                    verdict.escalate = True
                    verdict.reasons.append(
                        f"Gnomon judged this may go beyond what you asked ({in_scope:.0%} in scope)."
                    )

        graded = [action for action in plan.actions if not (undo_enabled and journal_undoable_delete(action))]
        kept = len(plan.actions) - len(graded)
        if kept:
            verdict.notes.append(
                f"{kept} file delete{'s are' if kept > 1 else ' is'} backed up by Heliox's undo, so "
                f"{'they were' if kept > 1 else 'it was'} not graded as irreversible."
            )
        harm = None
        local_only = all(action.action_type.value in LOCAL_ONLY_ACTIONS for action in graded)
        if graded and not local_only and not out_of_time():
            harm_text = proposed if not kept else _clip("; then ".join(describe_action(a) for a in graded) + ".", 900)
            harm = await self._score({"proposed_action": harm_text}, "score", HARM_QUESTION, HARM_LEVELS)
        if harm:
            probabilities = harm.get("probabilities") or {}
            severe = float(probabilities.get(HARM_LEVELS[2], 0.0)) + float(probabilities.get(HARM_LEVELS[3], 0.0))
            verdict.scores["severe_harm"] = severe
            if severe >= HARM_ESCALATE_PROBABILITY:
                verdict.escalate = True
                verdict.reasons.append(f"Gnomon graded the harm if wrong as: {harm.get('most_likely', 'severe')}.")

        if untrusted_context.strip() and not out_of_time():
            permitted = await self._yes(
                {
                    "content_source": "tool or screen output",
                    "observed_text": _clip(untrusted_context, 1500),
                    "proposed_action": proposed,
                    "request": request,
                },
                UNTRUSTED_QUESTION,
            )
            if permitted is not None:
                verdict.scores["permitted_by_user"] = permitted
                if permitted < UNTRUSTED_MIN_YES:
                    verdict.escalate = True
                    verdict.reasons.append(
                        "Gnomon judged this follows instructions found in tool or screen content, not your request."
                    )
        timings = ", ".join(
            f"{name} {answered:.2f} s (waited {waited:.2f} s)"
            for name, waited, answered in self._question_times[asked_before:]
        )
        if self._faults_seen != faults_before:
            # The worker died mid-review. A review that did not happen must never read as a pass: without an
            # escalation the whole review counts as not done (the dialog says Gnomon did not review this plan);
            # an escalation already found still adds its approval, marked incomplete.
            logger.info("Gnomon's process stopped during a review: %s", timings or "no question finished")
            if not verdict.escalate:
                return None
            verdict.complete = False
            verdict.notes.append(STOPPED_DURING_REVIEW)
            return verdict
        if not verdict.complete:
            verdict.notes.append("Gnomon ran out of time before every check, so this review is incomplete.")
            logger.info(
                "Gnomon review cut short at its %.1f s budget: %s",
                time_budget or 0.0,
                timings or "no question finished",
            )
        elif timings:
            logger.info("Gnomon review: %s", timings)
        # None means Gnomon answered nothing. A plan whose scope was not reviewed still gets a verdict, so the
        # person sees that note even when no other question applied.
        return verdict if verdict.scores or not verdict.complete or not verdict.scope_reviewed else None

    async def review_completion(self, task: str, evidence: str, receipt: str, expected: str) -> GnomonVerdict | None:
        """Postcondition check before a completion claim is accepted."""
        complete = await self._yes(
            {
                "task": _clip(task, 600),
                "executor_receipt": _clip(receipt or "no receipt", 400),
                "expected_value": _clip(expected or task, 400),
                "inspected_result": _clip(evidence or "nothing inspected", 600),
            },
            COMPLETION_QUESTION,
        )
        if complete is None:
            return None
        verdict = GnomonVerdict(scores={"verified_complete": complete})
        if complete < COMPLETION_MIN_YES:
            verdict.escalate = True
            verdict.reasons.append(f"Gnomon judged the completion claim unsupported ({complete:.0%}).")
        return verdict

    async def still_wanted(self, initial_request: str, newest_instruction: str, proposed_action: str) -> float | None:
        """Probability that the in-flight action is still wanted after the person spoke again."""
        return await self._yes(
            {
                "initial_request": _clip(initial_request, 400),
                "newest_instruction": _clip(newest_instruction, 400),
                "proposed_action": _clip(proposed_action, 400),
            },
            LATEST_INSTRUCTION_QUESTION,
        )
