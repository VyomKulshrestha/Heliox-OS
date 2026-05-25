"""Screen Vision Agent — continuous computer-vision loop for screen awareness.

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

MIN_CAPTURE_INTERVAL_SECONDS = 0.5
MAX_CAPTURE_INTERVAL_SECONDS = 60.0


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


@dataclass
class ScreenContext:
    """Rolling buffer of recent screen states."""

    states: list[ScreenState] = field(default_factory=list)
    max_size: int = 30

    def add(self, state: ScreenState) -> None:
        self.states.append(state)
        if len(self.states) > self.max_size:
            self.states = list(self.states)[len(self.states) - self.max_size :]

    def current(self) -> ScreenState | None:
        return self.states[-1] if self.states else None

    def summary(self) -> str:
        """Generate a human-readable summary of recent screen context."""
        if not self.states:
            return "No screen context available."

        current = self.states[-1]
        lines = [
            f'Currently viewing: {current.active_app} — "{current.active_window_title}"',
        ]
        if current.description:
            lines.append(f"Content: {current.description}")

        # Recent app history (last 5 unique apps)
        recent_apps: list[str] = []
        for s in reversed(self.states):
            if s.active_app and s.active_app not in recent_apps:
                recent_apps.append(s.active_app)
            if len(recent_apps) >= 5:
                break
        if len(recent_apps) > 1:
            lines.append(f"Recent apps: {', '.join(recent_apps)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        current = self.current()
        return {
            "current": current.to_dict() if current else None,
            "buffer_size": len(self.states),
            "recent_apps": self._recent_apps(),
        }

    def _recent_apps(self) -> list[str]:
        seen: list[str] = []
        for s in reversed(self.states):
            if s.active_app and s.active_app not in seen:
                seen.append(s.active_app)
            if len(seen) >= 10:
                break
        return seen


class ScreenVisionAgent:
    """Monitors the screen and maintains awareness of what the user sees."""

    def __init__(self, model_router: ModelRouter | None = None) -> None:
        self._model = model_router
        self._context = ScreenContext()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._interval_seconds: float = 3.0
        self._last_hash: str = ""
        self._screenshot_dir = Path.home() / ".heliox" / "screenshots"
        self._enable_llm_describe = False  # Disabled by default (expensive)

    def set_interval(self, interval_seconds: float) -> None:
        """Update the capture cadence while keeping it inside safe bounds."""
        self._interval_seconds = max(
            MIN_CAPTURE_INTERVAL_SECONDS,
            min(float(interval_seconds), MAX_CAPTURE_INTERVAL_SECONDS),
        )

    async def start(self, interval_seconds: float = 3.0, enable_describe: bool = False) -> None:
        """Start the screen monitoring loop."""
        self.set_interval(interval_seconds)
        self._enable_llm_describe = enable_describe
        if self._task and not self._task.done():
            await self.stop()
        self._running = True
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._capture_loop())
        logger.info("Screen vision started (every %.1fs, describe=%s)", self._interval_seconds, enable_describe)

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Screen vision stopped")

    async def _capture_loop(self) -> None:
        """Main capture loop."""
        while self._running:
            try:
                state = await self._capture_state()
                self._context.add(state)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Screen capture error", exc_info=True)
            await asyncio.sleep(self._interval_seconds)

    async def _capture_state(self) -> ScreenState:
        """Capture current screen state."""
        now = datetime.now(UTC).isoformat()
        # Run blocking OS calls in a thread to avoid starving the event loop
        app, title = await asyncio.to_thread(_get_active_window)
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

        # Optional: use LLM to describe what's on screen
        if self._enable_llm_describe and changed and self._model:
            state.description = await self._describe_screen(state)

        # ── Feature 1 & 7: Neural Cognitive Load + Saliency Overlay ──
        tribe_engine = getattr(self, "_tribe_engine", None)
        if tribe_engine and tribe_engine.is_loaded:
            stimulus = f"Visual focus on {app}: {title}"
            if state.description:
                stimulus += f" - {state.description}"
            # Fetch load and saliency map
            cog_state = await tribe_engine.predict_cognitive_state(stimulus)
            state.brain_load = cog_state.cognitive_load
            # Generate a 2D uniform saliency heatmap representing ventral stream activation
            # (Mocked to a simple distribution for UI arc reactor integration)
            if hasattr(cog_state, "raw_activations"):
                mean_act = cog_state.raw_activations.get("mean", 0.5)
                state.neural_saliency = [mean_act] * 16

        return state

    async def _capture_screenshot_hash(self) -> str:
        """Take a screenshot and return its hash for change detection."""
        try:
            return await asyncio.to_thread(self._sync_screenshot_hash)
        except ImportError:
            # Fallback: use OS screencapture and hash the file
            return await asyncio.to_thread(self._sync_fallback_screenshot_hash)

    def _sync_screenshot_hash(self) -> str:
        """Synchronous screenshot hash — runs in a thread."""
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            img = sct.grab(monitor)
            raw = img.rgb
            sampled = bytes(raw[i] for i in range(0, len(raw), 1000))
            return hashlib.md5(sampled).hexdigest()

    def _sync_fallback_screenshot_hash(self) -> str:
        """Synchronous fallback screenshot hash — runs in a thread."""
        os_name = platform.system()
        tmp_path = self._screenshot_dir / "_latest.png"
        try:
            if os_name == "Windows":
                ps_cmd = f"""
                Add-Type -AssemblyName System.Windows.Forms
                $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
                $bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
                $bitmap.Save('{tmp_path}')
                """
                subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    capture_output=True,
                    timeout=5,
                )
            elif os_name == "Darwin":
                subprocess.run(
                    ["screencapture", "-x", str(tmp_path)],
                    capture_output=True,
                    timeout=5,
                )
            else:
                subprocess.run(
                    ["scrot", str(tmp_path)],
                    capture_output=True,
                    timeout=5,
                )

            if tmp_path.exists():
                data = tmp_path.read_bytes()
                return hashlib.md5(data[::1000]).hexdigest()
        except Exception:
            logger.debug("Fallback screenshot failed", exc_info=True)
        return ""

    async def _describe_screen(self, state: ScreenState) -> str:
        """Use vision-capable LLM to describe what's on screen."""
        # Simple text-based description from metadata
        return f"User is viewing {state.active_app}: {state.active_window_title}"

    # ── Public API ──

    def get_context(self) -> ScreenContext:
        """Get the current screen context buffer."""
        return self._context

    def get_current_app(self) -> str:
        """Get the name of the currently active application."""
        current = self._context.current()
        return current.active_app if current else ""

    def get_context_for_planner(self) -> str:
        """Return screen context formatted for planner injection."""
        return self._context.summary()

    def get_stats(self) -> dict[str, Any]:
        """Return vision agent statistics."""
        return {
            "running": self._running,
            "interval_seconds": self._interval_seconds,
            "buffer_size": len(self._context.states),
            "llm_describe_enabled": self._enable_llm_describe,
            "current": self._context.current().to_dict() if self._context.current() else None,
            "recent_apps": self._context._recent_apps(),
        }

    async def detect_actionable_elements(
        self,
        description: str = "",
        region: str | None = None,
        max_elements: int = 20,
        action_filter: str = "",
    ) -> str:
        """Detect interactive UI elements via zero-shot VLM inference.

        Calls ``screen_detect_elements()`` from ``vision.py`` and caches
        the result in the latest ``ScreenState`` for downstream use.

        Parameters
        ----------
        description:
            Optional natural-language filter, e.g. ``"the submit button"``.
        region:
            ``"x,y,w,h"`` crop string, or ``None`` for full screen.
        max_elements:
            Maximum number of elements to return.
        action_filter:
            ``"click"``, ``"type"``, or ``""`` (all).

        Returns
        -------
        str
            JSON string — same schema as ``screen_detect_elements()``.
        """
        from pilot.system.vision import screen_detect_elements

        parsed_region = _parse_region(region)
        result = await screen_detect_elements(
            description=description,
            region=parsed_region,
            max_elements=max_elements,
            action_filter=action_filter,
        )

        # Cache in the current ScreenState so the planner can reference it
        current = self._context.current()
        if current is not None:
            current.description = f"[element_detection] {result}"

        return result

    def get_click_target(self, description: str) -> dict[str, Any] | None:
        """Find the best-matching cached element for a natural-language description.

        Searches the most recent element detection result stored in the
        ``ScreenState`` description field.  Uses simple substring matching
        on element labels — no LLM call required.

        Parameters
        ----------
        description:
            Natural-language description, e.g. ``"login button"``.

        Returns
        -------
        dict or None
            The best-matching element dict with ``label``, ``type``,
            ``action``, ``bbox``, and ``confidence``, or ``None`` if no
            match is found.
        """
        import json as _json

        current = self._context.current()
        if current is None or not current.description.startswith("[element_detection]"):
            return None

        raw = current.description[len("[element_detection]") :].strip()
        # The cached value is truncated — we can only search what's stored
        try:
            data = _json.loads(raw)
            elements = data.get("elements", [])
        except Exception:
            return None

        desc_lower = description.lower()
        best: dict[str, Any] | None = None
        best_score = -1

        for el in elements:
            label = el.get("label", "").lower()
            # Score: number of description words found in the label
            score = sum(1 for word in desc_lower.split() if word in label)
            if score > best_score:
                best_score = score
                best = el

        return best if best_score > 0 else None


# ── Platform-Specific Window Detection ──


def _get_active_window() -> tuple[str, str]:
    """Get the active window's application name and title."""
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
    """Windows: Get active window via win32gui or ctypes."""
    try:
        import psutil
        import win32gui
        import win32process

        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        app_name = process.name().replace(".exe", "")
        return (app_name, title)
    except ImportError:
        # Fallback using ctypes
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd) + 1
        buf = ctypes.create_unicode_buffer(length)
        user32.GetWindowTextW(hwnd, buf, length)
        title = buf.value
        # Extract app name from title heuristic
        app = title.rsplit(" - ", 1)[-1] if " - " in title else title
        return (app, title)


def _get_active_window_macos() -> tuple[str, str]:
    """macOS: Get active window via AppleScript."""
    script = """
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
        set frontTitle to ""
        try
            tell process frontApp
                set frontTitle to name of front window
            end tell
        end try
        return frontApp & "|" & frontTitle
    end tell
    """
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split("|", 1)
        return (parts[0], parts[1] if len(parts) > 1 else "")
    return ("Unknown", "Unknown")


def _get_active_window_linux() -> tuple[str, str]:
    """Linux: Get active window via xdotool."""
    try:
        wid = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if wid.returncode != 0:
            return ("Unknown", "Unknown")

        window_id = wid.stdout.strip()

        name_result = subprocess.run(
            ["xdotool", "getwindowname", window_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        title = name_result.stdout.strip() if name_result.returncode == 0 else ""

        # Get PID and process name
        pid_result = subprocess.run(
            ["xdotool", "getwindowpid", window_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        app = "Unknown"
        if pid_result.returncode == 0:
            pid = pid_result.stdout.strip()
            comm = Path(f"/proc/{pid}/comm")
            if comm.exists():
                app = comm.read_text().strip()

        return (app, title)
    except Exception:
        return ("Unknown", "Unknown")


def _parse_region(region: str | None) -> tuple[int, int, int, int] | None:
    """Parse a ``"x,y,w,h"`` region string into a tuple, or return None."""
    if not region:
        return None
    try:
        parts = [int(v.strip()) for v in region.split(",")]
        if len(parts) == 4:
            return (parts[0], parts[1], parts[2], parts[3])
    except (ValueError, AttributeError):
        pass
    return None
