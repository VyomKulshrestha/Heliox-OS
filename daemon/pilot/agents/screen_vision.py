"""
Screen Vision Agent — continuous computer-vision loop for screen awareness.

Takes periodic screenshots, detects the active application, and
maintains a context buffer so the planner knows what the user is
currently looking at.

Architecture:
  1. CaptureLoop:   Takes a screenshot every N seconds (default: 2)
  2. AppDetector:    Identifies the active window/app via OS APIs
  3. DiffEngine:     Compares consecutive screenshots
  4. ContextBuffer:  Maintains rolling screen states
  5. LLMDescriber:   Optional vision LLM descriptions
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.screen_vision")


# ───────────────────────────── Screen State ───────────────────────────── #

@dataclass
class ScreenState:
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
            self.states = self.states[-self.max_size :]

    def current(self) -> ScreenState | None:
        return self.states[-1] if self.states else None

    def summary(self) -> str:
        if not self.states:
            return "No screen context available."

        current = self.states[-1]
        return f'Currently viewing: {current.active_app} — "{current.active_window_title}"'


# ───────────────────────────── Main Agent ───────────────────────────── #

class ScreenVisionAgent:
    def __init__(self, model_router: ModelRouter | None = None) -> None:
        self._model = model_router
        self._context = ScreenContext()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._interval_seconds = 2.0
        self._last_hash = ""
        self._screenshot_dir = Path.home() / ".heliox" / "screenshots"

        self._frame_timeout_count = 0
        self._max_timeouts = 3
        self._paused = False

    async def start(self, interval_seconds: float = 2.0) -> None:
        self._interval_seconds = interval_seconds
        self._running = True
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _capture_loop(self) -> None:
        while self._running:
            try:
                state = await self._capture_state()
                self._context.add(state)

            except Exception:
                logger.debug("capture error", exc_info=True)

            await asyncio.sleep(self._interval_seconds)

    async def _capture_state(self) -> ScreenState:
        now = datetime.now(UTC).isoformat()

        app, title = await asyncio.to_thread(_get_active_window)

        state = ScreenState(
            timestamp=now,
            active_app=app,
            active_window_title=title,
        )

        return state

    def get_context(self) -> ScreenContext:
        return self._context


# ───────────────────────────── ACTIVE WINDOW (FIXED, SINGLE VERSION) ───────────────────────────── #

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
    script = '''
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
        return frontApp
    end tell
    '''

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

        return ("Unknown", "Unknown")

    except Exception:
        return ("Unknown", "Unknown")