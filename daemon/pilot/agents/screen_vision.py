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
import json
import logging
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class GuiBoundingBox:
    """Pixel-space bounding box for a visual UI target."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    def clipped(self, screen_width: int | None, screen_height: int | None) -> GuiBoundingBox:
        x = max(0, self.x)
        y = max(0, self.y)
        width = max(1, self.width)
        height = max(1, self.height)
        if screen_width is not None:
            x = min(x, max(0, screen_width - 1))
            width = min(width, max(1, screen_width - x))
        if screen_height is not None:
            y = min(y, max(0, screen_height - 1))
            height = min(height, max(1, screen_height - y))
        return GuiBoundingBox(x=x, y=y, width=width, height=height)


@dataclass(frozen=True)
class GuiActionTarget:
    """Actionable visual target returned by the native GUI VLM layer."""

    action: str
    label: str
    bounding_box: GuiBoundingBox
    confidence: float
    text: str = ""
    rationale: str = ""

    @property
    def x(self) -> int:
        return self.bounding_box.center_x

    @property
    def y(self) -> int:
        return self.bounding_box.center_y

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["x"] = self.x
        data["y"] = self.y
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


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

    async def locate_gui_target(
        self,
        instruction: str,
        *,
        screen_width: int | None = None,
        screen_height: int | None = None,
    ) -> GuiActionTarget:
        """Ask a vision model for direct click/type coordinates for a native GUI task."""
        from pilot.system.vision import screen_analyze

        prompt = self._build_vlm_target_prompt(instruction, screen_width, screen_height)
        response = await screen_analyze(prompt)
        return self.parse_gui_target_response(response, screen_width=screen_width, screen_height=screen_height)

    @staticmethod
    def _build_vlm_target_prompt(
        instruction: str,
        screen_width: int | None,
        screen_height: int | None,
    ) -> str:
        bounds = "unknown screen size"
        if screen_width is not None and screen_height is not None:
            bounds = f"{screen_width}x{screen_height} screen"
        return (
            "You are a native GUI targeting model. Inspect the screenshot and return one JSON object only. "
            "Find the single best UI target for the user's instruction and include pixel coordinates in "
            f"the {bounds}. Use this schema: "
            '{"action":"click|type|observe","label":"target name","bbox":{"x":0,"y":0,"width":1,"height":1},'
            '"confidence":0.0,"text":"","rationale":"short reason"}. '
            f"Instruction: {instruction}"
        )

    @staticmethod
    def parse_gui_target_response(
        response: str,
        *,
        screen_width: int | None = None,
        screen_height: int | None = None,
    ) -> GuiActionTarget:
        """Parse and validate a JSON GUI target emitted by a VLM."""
        payload = _extract_json_object(response)
        bbox = payload.get("bbox") or payload.get("bounding_box") or {}
        if not isinstance(bbox, dict):
            raise ValueError("VLM target response must include a bbox object.")

        box = GuiBoundingBox(
            x=_coerce_int(bbox.get("x"), "bbox.x"),
            y=_coerce_int(bbox.get("y"), "bbox.y"),
            width=_coerce_int(bbox.get("width", bbox.get("w")), "bbox.width"),
            height=_coerce_int(bbox.get("height", bbox.get("h")), "bbox.height"),
        ).clipped(screen_width, screen_height)
        confidence = float(payload.get("confidence", 0.0))
        confidence = max(0.0, min(confidence, 1.0))

        return GuiActionTarget(
            action=str(payload.get("action") or "click").strip().lower(),
            label=str(payload.get("label") or payload.get("target") or "visual target").strip(),
            bounding_box=box,
            confidence=confidence,
            text=str(payload.get("text") or ""),
            rationale=str(payload.get("rationale") or ""),
        )

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
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        title = name_result.stdout.strip() if name_result.returncode == 0 else ""

        # Get PID and process name
        pid_result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowpid"],
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


def _extract_json_object(response: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    cleaned = response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    elif "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.index("{") : cleaned.rindex("}") + 1]

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("VLM target response must be a JSON object.")
    return data


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VLM target response has invalid {field_name}.") from exc
