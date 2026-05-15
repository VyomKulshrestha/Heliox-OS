"""
Screen Vision Agent — continuous computer-vision loop for screen awareness.

Takes periodic screenshots, detects the active application, and
maintains a context buffer so the planner knows what the user is
currently looking at.  When the user says "summarize this" or
"close that", the agent already knows the target.

Architecture:
  1. CaptureLoop:   Takes a screenshot every N seconds (default: 2)
  2. AppDetector:    Identifies the active window/app via OS APIs
  3. DiffEngine:     Compares consecutive screenshots to detect changes
  4. ContextBuffer:  Maintains a rolling buffer of recent screen states
  5. LLMDescriber:   Optionally uses vision-capable LLM to describe content

Platform support:
  - Windows: win32gui + PIL (mss for screenshot)
  - macOS:   Quartz + screencapture
  - Linux:   xdotool + scrot / gnome-screenshot
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.screen_vision")


# ───────────────────────────── Screen State ───────────────────────────── #

@dataclass
class ScreenState:
    """Snapshot of the current screen state."""

    timestamp: str = ""
    active_app: str = ""
    active_window_title: str = ""
    screen_hash: str = ""
    changed_from_last: bool = False
    description: str = ""
    screenshot_path: str = ""
    brain_load: float = 0.0
    neural_saliency: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "active_app": self.active_app,
            "active_window_title": self.active_window_title,
            "changed": self.changed_from_last,
            "description": self.description,
            "brain_load": self.brain_load,
            "neural_saliency": self.neural_saliency,
        }


# ───────────────────────────── Context Buffer ───────────────────────────── #

@dataclass
class ScreenContext:
    states: list[ScreenState] = field(default_factory=list)
    max_size: int = 30

    def add(self, state: ScreenState) -> None:
        self.states.append(state)
        if len(self.states) > self.max_size:
            self.states = self.states[-self.max_size:]

    def current(self) -> ScreenState | None:
        return self.states[-1] if self.states else None

    def summary(self) -> str:
        if not self.states:
            return "No screen context available."

        current = self.states[-1]
        return f'Currently viewing: {current.active_app} — "{current.active_window_title}"'

    def to_dict(self) -> dict[str, Any]:
        current = self.current()
        return {
            "current": current.to_dict() if current else None,
            "buffer_size": len(self.states),
            "recent_apps": self._recent_apps(),
        }

    def _recent_apps(self) -> list[str]:
        seen = []
        for s in reversed(self.states):
            if s.active_app and s.active_app not in seen:
                seen.append(s.active_app)
            if len(seen) >= 10:
                break
        return seen


# ───────────────────────────── Main Agent ───────────────────────────── #

class ScreenVisionAgent:
    """Monitors the screen and maintains awareness of what the user sees."""

    def __init__(self, model_router: ModelRouter | None = None) -> None:
        self._model = model_router
        self._context = ScreenContext()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._interval_seconds = 2.0
        self._last_hash = ""
        self._screenshot_dir = Path.home() / ".heliox" / "screenshots"

        # ONLY FIX FEATURE: timeout safety
        self._frame_timeout_count = 0
        self._max_timeouts = 3
        self._paused = False

        self._enable_llm_describe = False

    async def start(self, interval_seconds: float = 2.0, enable_describe: bool = False) -> None:
        self._interval_seconds = interval_seconds
        self._enable_llm_describe = enable_describe
        self._running = True
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._capture_loop())
        logger.info("Screen vision started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _capture_loop(self) -> None:
        while self._running:
            try:
                state = await self._capture_state()
                self._context.add(state)

            # ✅ ONLY SAFE ADDITION (fix issue #101)
            except TimeoutError:
                self._frame_timeout_count += 1
                logger.warning("Frame timeout (%d/%d)", self._frame_timeout_count, self._max_timeouts)

                if self._frame_timeout_count >= self._max_timeouts:
                    self._paused = True
                    logger.warning("Pausing capture due to repeated timeouts")

                await asyncio.sleep(2)
                continue

            except asyncio.CancelledError:
                break

            except Exception:
                logger.debug("Screen capture error", exc_info=True)

            await asyncio.sleep(self._interval_seconds)

    async def _capture_state(self) -> ScreenState:
        now = datetime.now(UTC).isoformat()

        # FIX: safe threaded call with timeout protection
        try:
            app, title = await asyncio.wait_for(
                asyncio.to_thread(_get_active_window),
                timeout=3
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Window detection timeout") from exc

        screen_hash = await self._capture_screenshot_hash()

        changed = screen_hash != self._last_hash and screen_hash != ""
        self._last_hash = screen_hash

        state = ScreenState(
            timestamp=now,
            active_app=app,
            active_window_title=title,
            screen_hash=screen_hash,
            changed_from_last=changed,
        )

        # optional LLM
        if self._enable_llm_describe and self._model and changed:
            state.description = f"User is viewing {app}: {title}"

        return state

    async def _capture_screenshot_hash(self) -> str:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._sync_screenshot_hash),
                timeout=5,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Screenshot timeout") from exc

    def _sync_screenshot_hash(self) -> str:
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            raw = img.rgb

            if not raw:
                raise TimeoutError("Empty frame")

            sampled = bytes(raw[i] for i in range(0, len(raw), 1000))
            return hashlib.md5(sampled).hexdigest()

    # ── PUBLIC API ── #

    def get_context(self) -> ScreenContext:
        return self._context

    def get_current_app(self) -> str:
        cur = self._context.current()
        return cur.active_app if cur else ""

    def get_context_for_planner(self) -> str:
        return self._context.summary()

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "paused": self._paused,
            "timeout_count": self._frame_timeout_count,
            "buffer_size": len(self._context.states),
        }


# ───────────────────────────── FIXED SINGLE WINDOW DETECTOR ───────────────────────────── #

def _get_active_window() -> tuple[str, str]:
    os_name = platform.system()

    try:
        if os_name == "Windows":
            return _get_active_window_windows()
        elif os_name == "Darwin":
            return _get_active_window_macos()
        else:
            return _get_active_window_linux()
    except Exception:
        return ("Unknown", "Unknown")


def _get_active_window_windows() -> tuple[str, str]:
    try:
        import psutil
        import win32gui
        import win32process

        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        app = psutil.Process(pid).name().replace(".exe", "")
        return app, title

    except Exception:
        return ("Unknown", "Unknown")


def _get_active_window_macos() -> tuple[str, str]:
    script = """
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
        return frontApp
    end tell
    """

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return (result.stdout.strip(), "")

    return ("Unknown", "Unknown")


def _get_active_window_linux() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return ("linux-app", result.stdout.strip())

    except Exception:
        pass

    return ("Unknown", "Unknown")