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
        self._interval_seconds: float = 2.0
        self._last_hash: str = ""
        self._screenshot_dir = Path.home() / ".heliox" / "screenshots"
        self._enable_llm_describe = False

        # Timeout handling
        self._frame_timeout_count = 0
        self._max_timeouts = 3
        self._paused = False

    async def start(
        self,
        interval_seconds: float = 2.0,
        enable_describe: bool = False,
    ) -> None:
        """Start the screen monitoring loop."""
        self._interval_seconds = interval_seconds
        self._enable_llm_describe = enable_describe
        self._running = True
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._task = asyncio.create_task(self._capture_loop())

        logger.info(
            "Screen vision started (every %.1fs, describe=%s)",
            interval_seconds,
            enable_describe,
        )

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

            except TimeoutError:
                self._frame_timeout_count += 1

                logger.warning(
                    "Screen frame timeout detected (%d/%d)",
                    self._frame_timeout_count,
                    self._max_timeouts,
                )

                if self._frame_timeout_count >= self._max_timeouts:
                    self._paused = True

                    logger.warning(
                        "Display appears inactive. "
                        "Pausing ScreenVisionAgent..."
                    )

                    await asyncio.sleep(5)

            except Exception:
                logger.debug("Screen capture error", exc_info=True)

            await asyncio.sleep(self._interval_seconds)

    async def _capture_state(self) -> ScreenState:
        """Capture current screen state."""

        now = datetime.now(UTC).isoformat()

        app, title = await asyncio.to_thread(_get_active_window)

        screen_hash = await self._capture_screenshot_hash()

        # Reset timeout state after successful capture
        if self._paused:
            logger.info(
                "Display activity restored. "
                "Resuming ScreenVisionAgent..."
            )
            self._paused = False

        self._frame_timeout_count = 0

        changed = screen_hash != self._last_hash and screen_hash != ""
        self._last_hash = screen_hash

        state = ScreenState(
            timestamp=now,
            active_app=app,
            active_window_title=title,
            screen_hash=screen_hash,
            changed_from_last=changed,
        )

        if self._enable_llm_describe and changed and self._model:
            state.description = await self._describe_screen(state)

        tribe_engine = getattr(self, "_tribe_engine", None)

        if tribe_engine and tribe_engine.is_loaded:
            stimulus = f"Visual focus on {app}: {title}"

            if state.description:
                stimulus += f" - {state.description}"

            cog_state = await tribe_engine.predict_cognitive_state(stimulus)

            state.brain_load = cog_state.cognitive_load

            if hasattr(cog_state, "raw_activations"):
                mean_act = cog_state.raw_activations.get("mean", 0.5)
                state.neural_saliency = [mean_act] * 16

        return state

    async def _capture_screenshot_hash(self) -> str:
        """Take a screenshot and return its hash for change detection."""

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._sync_screenshot_hash),
                timeout=5,
            )

        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                "Screen capture timed out"
            ) from exc

        except ImportError:
            return await asyncio.to_thread(
                self._sync_fallback_screenshot_hash
            )

    def _sync_screenshot_hash(self) -> str:
        """Synchronous screenshot hash — runs in a thread."""

        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[1]

            img = sct.grab(monitor)

            raw = img.rgb

            if not raw:
                raise TimeoutError("No screen frame received")

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

        return (
            f"User is viewing "
            f"{state.active_app}: {state.active_window_title}"
        )

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
            "paused": self._paused,
            "timeout_count": self._frame_timeout_count,
            "current": (
                self._context.current().to_dict()
                if self._context.current()
                else None
            ),
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