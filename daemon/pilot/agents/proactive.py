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
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.config import DATA_DIR
from pilot.intelligence.experience import (
    ExperienceEventType,
    ExperienceLedger,
    PrivacyClass,
)

if TYPE_CHECKING:
    from pilot.agents.screen_vision import ScreenContext, ScreenVisionAgent
    from pilot.intelligence.online_learning import VerifiedOnlineLearner

logger = logging.getLogger("pilot.agents.proactive")


@dataclass
class Suggestion:
    """A proactive suggestion to show the user."""

    suggestion_id: str
    title: str
    description: str
    action_command: str  # The command to execute if accepted
    trigger_reason: str
    pattern_id: str = ""
    learned_relevance: float = 0.5
    priority: str = "low"  # low, medium, high
    context_app: str = ""
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
            "pattern_id": self.pattern_id,
            "learned_relevance": round(self.learned_relevance, 3),
            "priority": self.priority,
            "context_app": self.context_app,
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

    def __init__(
        self,
        screen_vision: ScreenVisionAgent | None = None,
        feedback_path: Path | None = None,
    ) -> None:
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
        self._feedback_path = feedback_path or (DATA_DIR / "proactive_feedback.json")
        self._feedback: dict[str, dict[str, float | int]] = self._load_feedback()
        self._experience_ledger: ExperienceLedger | None = None
        self._online_learner: VerifiedOnlineLearner | None = None
        self._last_observation_key = ""
        self._ignore_after_seconds = 15 * 60

    def set_broadcast(self, fn: Callable[[str, Any], Coroutine[Any, Any, None]]) -> None:
        self._broadcast = fn

    def set_experience_ledger(self, ledger: ExperienceLedger) -> None:
        self._experience_ledger = ledger

    def set_online_learner(self, learner: VerifiedOnlineLearner) -> None:
        """Use verified adaptation for ranking only; execution remains guarded."""

        self._online_learner = learner

    async def _append_experience(self, event_type: ExperienceEventType, **kwargs: Any) -> None:
        if self._experience_ledger is None:
            return
        try:
            await self._experience_ledger.append(event_type, **kwargs)
        except Exception:
            logger.debug("Proactive experience append failed", exc_info=True)

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
                self._record_feedback(s.pattern_id, "accepted")
                await self._append_experience(
                    ExperienceEventType.SUGGESTION_FEEDBACK,
                    idempotency_key=f"suggestion:{s.suggestion_id}:feedback",
                    source="proactive",
                    payload={
                        "suggestion_id": s.suggestion_id,
                        "pattern_id": s.pattern_id,
                        "decision": "accepted",
                        "context_app": s.context_app,
                        "priority": s.priority,
                    },
                    confidence=s.learned_relevance,
                    provenance={"component": "ProactiveSuggestionEngine.accept_suggestion"},
                )
                return s.action_command
        return None

    async def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """User dismissed a suggestion."""
        for s in self._pending_suggestions:
            if s.suggestion_id == suggestion_id:
                s.dismissed = True
                self._pending_suggestions.remove(s)
                self._suggestion_history.append(s)
                self._record_feedback(s.pattern_id, "dismissed")
                await self._append_experience(
                    ExperienceEventType.SUGGESTION_FEEDBACK,
                    idempotency_key=f"suggestion:{s.suggestion_id}:feedback",
                    source="proactive",
                    payload={
                        "suggestion_id": s.suggestion_id,
                        "pattern_id": s.pattern_id,
                        "decision": "dismissed",
                        "context_app": s.context_app,
                        "priority": s.priority,
                    },
                    confidence=s.learned_relevance,
                    provenance={"component": "ProactiveSuggestionEngine.dismiss_suggestion"},
                )
                return True
        return False

    async def resolve_spoken_response(self, text: str) -> dict[str, str] | None:
        """Resolve a natural yes/no reply against the one visible suggestion."""
        if len(self._pending_suggestions) != 1:
            return None
        normalized = " ".join(re.sub(r"[^\w\s']", "", text.lower()).split())
        affirmative = {
            "yes",
            "yes do it",
            "do it",
            "do that",
            "go ahead",
            "please do",
            "sounds good",
            "sure",
        }
        negative = {
            "no",
            "no thanks",
            "not now",
            "dismiss it",
            "skip it",
            "don't do it",
            "do not do it",
        }
        suggestion = self._pending_suggestions[0]
        if normalized in affirmative:
            action_command = await self.accept_suggestion(suggestion.suggestion_id)
            if not action_command:
                return None
            return {
                "decision": "accepted",
                "suggestion_id": suggestion.suggestion_id,
                "title": suggestion.title,
                "action_command": action_command,
            }
        if normalized in negative:
            dismissed = await self.dismiss_suggestion(suggestion.suggestion_id)
            if not dismissed:
                return None
            return {
                "decision": "dismissed",
                "suggestion_id": suggestion.suggestion_id,
                "title": suggestion.title,
                "action_command": "",
            }
        return None

    async def _watch_loop(self) -> None:
        """Main loop — periodically checks screen context for patterns."""
        while self._running:
            try:
                if self._enabled:
                    current = self._current_live_screen_state()
                    if current and current.active_app:
                        await self._check_patterns(current)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Proactive watch error", exc_info=True)

            await asyncio.sleep(self._check_interval)

    def _current_live_screen_state(self) -> Any | None:
        """Return context only while periodic screen monitoring is live.

        Screen history intentionally survives a stop so an active, explicit
        task can still explain its last observation. Background suggestions
        must not treat that retained history as a current user activity.
        """
        vision = self._screen_vision
        if vision is None or not vision.is_running or vision.is_paused():
            return None
        return vision.get_context().current()

    async def _check_patterns(self, current: Any) -> None:
        """Check all patterns against the current screen state."""
        now = time.time()
        await self._expire_ignored_suggestions(now)
        # Keep one visible decision at a time. Generating more while the user
        # considers the current card would create hidden pending items and
        # corrupt accept/dismiss learning.
        if self._pending_suggestions:
            return

        app_lower = current.active_app.lower()
        title_lower = current.active_window_title.lower()

        # Update dwell tracker
        dwell_key = f"{app_lower}:{title_lower[:50]}"
        if dwell_key != self._last_observation_key:
            self._last_observation_key = dwell_key
            await self._append_experience(
                ExperienceEventType.OBSERVATION,
                idempotency_key=f"observation:screen-context:{time.time_ns()}",
                source="screen_context",
                payload={
                    "active_app": current.active_app,
                    "window_title": current.active_window_title,
                    "raw_media_excluded": True,
                },
                provenance={"component": "ProactiveSuggestionEngine._check_patterns"},
                privacy_class=PrivacyClass.SENSITIVE,
            )
        if dwell_key not in self._app_dwell_tracker:
            self._app_dwell_tracker[dwell_key] = now
            # Clean old entries
            cutoff = now - 3600
            self._app_dwell_tracker = {k: v for k, v in self._app_dwell_tracker.items() if v > cutoff}

        dwell_seconds = now - self._app_dwell_tracker.get(dwell_key, now)

        for pattern in _PATTERNS:
            learned = self._feedback.get(pattern["id"], {})
            accepted = int(learned.get("accepted", 0))
            dismissed = int(learned.get("dismissed", 0))
            shown = int(learned.get("shown", 0))
            last_feedback = float(learned.get("last_feedback", 0.0))
            learned_relevance = (accepted + 1) / (accepted + dismissed + 2)
            adaptation_state = "candidate"
            if self._online_learner is not None:
                adaptation = self._online_learner.score_suggestion(
                    pattern_id=str(pattern["id"]),
                    app_name=app_lower,
                    priority=str(pattern.get("priority", "low")),
                )
                adaptation_state = adaptation.state
                if adaptation.state == "promoted":
                    learned_relevance = adaptation.probability

            # Repeated rejection suppresses this pattern for a week. A future
            # accepted suggestion lifts suppression automatically.
            if dismissed >= 3 and accepted == 0 and now - last_feedback < 7 * 24 * 60 * 60:
                continue

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
            base_dwell = float(pattern.get("min_dwell_seconds", 0))
            dwell_multiplier = max(0.65, min(1.75, 1.45 - learned_relevance))
            min_dwell = base_dwell * dwell_multiplier
            if dwell_seconds < min_dwell:
                continue

            # Pattern matched! Generate suggestion
            suggestion = Suggestion(
                suggestion_id=f"{pattern['id']}_{int(now)}",
                title=pattern["title"],
                description=pattern["description"],
                action_command=pattern["action"],
                trigger_reason=f"Detected {app_lower} with context: {title_lower[:60]}",
                pattern_id=pattern["id"],
                learned_relevance=learned_relevance,
                priority=(
                    "high"
                    if learned_relevance >= 0.72 and adaptation_state == "promoted"
                    else pattern.get("priority", "low")
                ),
                context_app=app_lower,
            )

            # Mark cooldown
            self._cooldowns[pattern["id"]] = now

            # Add to pending
            self._pending_suggestions.append(suggestion)
            self._record_feedback(pattern["id"], "shown")
            await self._append_experience(
                ExperienceEventType.SUGGESTION_SHOWN,
                idempotency_key=f"suggestion:{suggestion.suggestion_id}:shown",
                source="proactive",
                payload=suggestion.to_dict(),
                confidence=suggestion.learned_relevance,
                provenance={"component": "ProactiveSuggestionEngine._check_patterns"},
                privacy_class=PrivacyClass.SENSITIVE,
            )

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

    async def _expire_ignored_suggestions(self, now: float) -> None:
        """Record an untouched card as missing feedback after a bounded wait."""

        expired = [
            suggestion
            for suggestion in self._pending_suggestions
            if now - suggestion.timestamp >= self._ignore_after_seconds
        ]
        for suggestion in expired:
            self._pending_suggestions.remove(suggestion)
            self._suggestion_history.append(suggestion)
            self._record_feedback(suggestion.pattern_id, "ignored")
            await self._append_experience(
                ExperienceEventType.SUGGESTION_FEEDBACK,
                idempotency_key=f"suggestion:{suggestion.suggestion_id}:feedback",
                source="proactive",
                payload={
                    "suggestion_id": suggestion.suggestion_id,
                    "pattern_id": suggestion.pattern_id,
                    "decision": "ignored",
                    "context_app": suggestion.context_app,
                    "priority": suggestion.priority,
                },
                confidence=suggestion.learned_relevance,
                provenance={"component": "ProactiveSuggestionEngine._expire_ignored_suggestions"},
            )

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        return {
            "running": self._running,
            "enabled": self._enabled,
            "pending_count": len(self._pending_suggestions),
            "history_count": len(self._suggestion_history),
            "pending": [s.to_dict() for s in self._pending_suggestions],
            "check_interval": self._check_interval,
            "learning": self.get_learning_status(),
        }

    def get_learning_status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "patterns": {
                pattern_id: {
                    "shown": int(values.get("shown", 0)),
                    "accepted": int(values.get("accepted", 0)),
                    "dismissed": int(values.get("dismissed", 0)),
                    "ignored": int(values.get("ignored", 0)),
                    "learned_relevance": round(
                        (int(values.get("accepted", 0)) + 1)
                        / (int(values.get("accepted", 0)) + int(values.get("dismissed", 0)) + 2),
                        3,
                    ),
                }
                for pattern_id, values in sorted(self._feedback.items())
            },
        }

    def reset_learning(self) -> None:
        self._feedback = {}
        try:
            self._feedback_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove proactive learning file", exc_info=True)

    def _load_feedback(self) -> dict[str, dict[str, float | int]]:
        try:
            raw = json.loads(self._feedback_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {str(key): value for key, value in raw.items() if isinstance(value, dict)}
        except (OSError, ValueError):
            pass
        return {}

    def _record_feedback(self, pattern_id: str, event: str) -> None:
        if not pattern_id or event not in {"shown", "accepted", "dismissed", "ignored"}:
            return
        values = self._feedback.setdefault(
            pattern_id,
            {
                "shown": 0,
                "accepted": 0,
                "dismissed": 0,
                "ignored": 0,
                "last_feedback": 0.0,
            },
        )
        values[event] = int(values.get(event, 0)) + 1
        if event in {"accepted", "dismissed"}:
            values["last_feedback"] = time.time()
        self._save_feedback()

    def _save_feedback(self) -> None:
        try:
            self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._feedback_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self._feedback, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self._feedback_path)
        except OSError:
            logger.warning("Could not persist proactive learning feedback", exc_info=True)
