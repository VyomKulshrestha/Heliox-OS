"""Proactive Suggestions Engine — JARVIS anticipates your needs.

Watches the ScreenVisionAgent's context buffer and detects patterns
that suggest the user might benefit from AI assistance. When a pattern
is detected, a gentle suggestion is broadcast to the UI.

Examples:
  - User has been on StackOverflow for 5+ minutes → offer to analyze the error
  - User opened a terminal with a Python traceback → offer to debug it
  - User switched to Figma → offer to convert design to code
  - User is in VS Code with a TODO comment → offer to implement it
  - User has been idle for 10+ minutes → offer a productivity check

Architecture:
  [ScreenContext buffer] → [Pattern Matchers] → [Cooldown filter]
                                                      ↓
                                              [Broadcast suggestion]
                                                      ↓
                                              [User accepts/dismisses]
                                                      ↓ (accept)
                                              [Execute via ReAct pipeline]
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from pilot.agents.screen_vision import ScreenContext, ScreenVisionAgent

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.agents.proactive")


@dataclass
class Suggestion:
    """A proactive suggestion to show the user."""

    suggestion_id: str
    title: str
    description: str
    action_command: str  # The command to execute if accepted
    trigger_reason: str
    priority: str = "low"  # low, medium, high
    timestamp: float = field(default_factory=time.time)
    dismissed: bool = False
    accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "title": self.title,
            "description": self.description,
            "action_command": self.action_command,
            "trigger_reason": self.trigger_reason,
            "priority": self.priority,
            "timestamp": self.timestamp,
        }


# ── Pattern Matchers ──────────────────────────────────────────────────

# Each matcher inspects the screen context and returns a Suggestion or None.

_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "stackoverflow_debug",
        "app_keywords": ["chrome", "firefox", "edge", "brave", "msedge"],
        "title_keywords": ["stack overflow", "stackoverflow"],
        "min_dwell_seconds": 120,
        "title": "Need help debugging?",
        "description": "You've been on Stack Overflow for a while. I can analyze the error you're looking at.",
        "action": "Take a screenshot and analyze the error shown on screen. Suggest a fix.",
        "priority": "medium",
        "cooldown_seconds": 300,
    },
    {
        "id": "terminal_error",
        "app_keywords": [
            "terminal",
            "cmd",
            "powershell",
            "windowsterminal",
            "iterm",
            "alacritty",
            "wezterm",
            "hyper",
            "conemu",
        ],
        "title_keywords": ["error", "traceback", "exception", "failed", "fatal"],
        "min_dwell_seconds": 5,
        "title": "I see an error in your terminal",
        "description": "Looks like there's an error. Want me to analyze it and suggest a fix?",
        "action": "Take a screenshot of the terminal, analyze the error, and suggest a fix.",
        "priority": "high",
        "cooldown_seconds": 60,
    },
    {
        "id": "figma_design",
        "app_keywords": ["figma"],
        "title_keywords": [],
        "min_dwell_seconds": 30,
        "title": "Convert this design to code?",
        "description": "I see you're working in Figma. I can take a screenshot and generate HTML/CSS code from the design.",
        "action": "Take a screenshot of the current Figma design and generate responsive HTML/CSS code.",
        "priority": "low",
        "cooldown_seconds": 600,
    },
    {
        "id": "vscode_coding",
        "app_keywords": ["code", "code - insiders", "cursor"],
        "title_keywords": ["todo", "fixme", "hack", "bug"],
        "min_dwell_seconds": 60,
        "title": "Want me to help with that TODO?",
        "description": "I notice a TODO/FIXME in your code. I can take a look and suggest an implementation.",
        "action": "Take a screenshot and analyze the TODO/FIXME comment visible in the editor. Suggest an implementation.",
        "priority": "low",
        "cooldown_seconds": 300,
    },
    {
        "id": "browser_research",
        "app_keywords": ["chrome", "firefox", "edge", "brave", "msedge"],
        "title_keywords": ["google", "search", "how to", "tutorial", "guide", "docs"],
        "min_dwell_seconds": 180,
        "title": "Need a summary?",
        "description": "You've been researching for a while. Want me to summarize what you've found?",
        "action": "Take a screenshot and summarize the content currently visible on screen.",
        "priority": "low",
        "cooldown_seconds": 600,
    },
    {
        "id": "email_compose",
        "app_keywords": ["chrome", "firefox", "edge", "outlook", "thunderbird"],
        "title_keywords": ["compose", "new message", "draft", "gmail", "outlook"],
        "min_dwell_seconds": 120,
        "title": "Need help writing?",
        "description": "Looks like you're composing a message. I can help draft or proofread it.",
        "action": "Take a screenshot and help improve or complete the email/message being composed.",
        "priority": "low",
        "cooldown_seconds": 600,
    },
]


class ProactiveSuggestionEngine:
    """Watches screen context and generates proactive suggestions."""

    def __init__(self, screen_vision: ScreenVisionAgent | None = None) -> None:
        self._screen_vision = screen_vision
        self._broadcast: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._check_interval = 10.0  # Check every 10 seconds
        self._cooldowns: dict[str, float] = {}  # pattern_id → last_triggered_time
        self._pending_suggestions: list[Suggestion] = []
        self._suggestion_history: list[Suggestion] = []
        self._app_dwell_tracker: dict[str, float] = {}  # app_name → first_seen_time
        self._enabled = True

    def set_broadcast(self, fn: Callable[[str, Any], Coroutine[Any, Any, None]]) -> None:
        self._broadcast = fn

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> str:
        """Start the proactive suggestion engine."""
        if self._running:
            return "Proactive engine is already running."

        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Proactive suggestion engine started")
        return "Proactive suggestions enabled. I'll watch for opportunities to help."

    async def stop(self) -> str:
        """Stop the proactive suggestion engine."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Proactive suggestion engine stopped")
        return "Proactive suggestions disabled."

    async def accept_suggestion(self, suggestion_id: str) -> str | None:
        """User accepted a suggestion — return the action command to execute."""
        for s in self._pending_suggestions:
            if s.suggestion_id == suggestion_id:
                s.accepted = True
                self._pending_suggestions.remove(s)
                self._suggestion_history.append(s)
                return s.action_command
        return None

    async def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """User dismissed a suggestion."""
        for s in self._pending_suggestions:
            if s.suggestion_id == suggestion_id:
                s.dismissed = True
                self._pending_suggestions.remove(s)
                self._suggestion_history.append(s)
                return True
        return False

    async def _watch_loop(self) -> None:
        """Main loop — periodically checks screen context for patterns."""
        while self._running:
            try:
                if self._enabled and self._screen_vision:
                    context = self._screen_vision.get_context()
                    current = context.current()

                    if current and current.active_app:
                        await self._check_patterns(current)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Proactive watch error", exc_info=True)

            await asyncio.sleep(self._check_interval)

    async def _check_patterns(self, current: Any) -> None:
        """Check all patterns against the current screen state."""
        now = time.time()
        app_lower = current.active_app.lower()
        title_lower = current.active_window_title.lower()

        # Update dwell tracker
        dwell_key = f"{app_lower}:{title_lower[:50]}"
        if dwell_key not in self._app_dwell_tracker:
            self._app_dwell_tracker[dwell_key] = now
            # Clean old entries
            cutoff = now - 3600
            self._app_dwell_tracker = {k: v for k, v in self._app_dwell_tracker.items() if v > cutoff}

        dwell_seconds = now - self._app_dwell_tracker.get(dwell_key, now)

        for pattern in _PATTERNS:
            # Check cooldown
            last_triggered = self._cooldowns.get(pattern["id"], 0)
            if now - last_triggered < pattern.get("cooldown_seconds", 300):
                continue

            # Check app match
            app_match = any(kw in app_lower for kw in pattern["app_keywords"])
            if not app_match:
                continue

            # Check title keywords (if any required)
            title_keywords = pattern.get("title_keywords", [])
            if title_keywords:
                title_match = any(kw in title_lower for kw in title_keywords)
                if not title_match:
                    continue

            # Check dwell time
            min_dwell = pattern.get("min_dwell_seconds", 0)
            if dwell_seconds < min_dwell:
                continue

            # Pattern matched! Generate suggestion
            suggestion = Suggestion(
                suggestion_id=f"{pattern['id']}_{int(now)}",
                title=pattern["title"],
                description=pattern["description"],
                action_command=pattern["action"],
                trigger_reason=f"Detected {app_lower} with context: {title_lower[:60]}",
                priority=pattern.get("priority", "low"),
            )

            # Mark cooldown
            self._cooldowns[pattern["id"]] = now

            # Add to pending
            self._pending_suggestions.append(suggestion)

            # Broadcast to UI
            if self._broadcast:
                try:
                    await self._broadcast("proactive_suggestion", suggestion.to_dict())
                except Exception:
                    pass

            logger.info(
                "Proactive suggestion: [%s] %s (dwell: %.0fs)",
                pattern["id"],
                suggestion.title,
                dwell_seconds,
            )

            # Only one suggestion per cycle
            break

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        return {
            "running": self._running,
            "enabled": self._enabled,
            "pending_count": len(self._pending_suggestions),
            "history_count": len(self._suggestion_history),
            "pending": [s.to_dict() for s in self._pending_suggestions],
            "check_interval": self._check_interval,
        }
