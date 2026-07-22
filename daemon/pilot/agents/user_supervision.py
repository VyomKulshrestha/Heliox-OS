"""UserSupervisionEngine — watches the user's OWN independent screen/keyboard/
mouse activity, never anything Heliox itself executes (that's ExecutionNarrator's
job, see agents/narrator.py).

Two independent, advisory-only trigger sources evaluated on one internal tick:

- **Cognitive coaching** — the cognitive engine (`pilot.cognitive.cognitive_engine.CognitiveEngine`)
  is fed a REAL stimulus for the first time (an OCR screen snippet + the
  active window title), instead of the synthetic labels every existing call
  site uses. A stress/cognitive-load threshold crossing triggers a gentle
  check-in.
- **Risk-pattern detection** — the OCR snippet and a transient keystroke
  buffer (see `pilot.system.input_hook.InputSupervisionHook`) are matched
  against `pilot.security.risk_patterns`' small, hardcoded, auditable rule
  table. A match triggers a direct warning.

Both are advisory-only: unlike `ExecutionNarrator`, which gates a Heliox-issued
plan/action *before it runs* via a real blocking `PendingConfirmation`, Heliox
has no way to intercept or block the user's own OS-level input -- it only
observes a copy via the hook. So these trigger methods return `None`, never
`bool`, and nothing here registers a confirmation or waits for a response.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.security.risk_patterns import match_risk_pattern
from pilot.system.vision import screen_ocr

if TYPE_CHECKING:
    from pilot.agents.screen_vision import ScreenVisionAgent
    from pilot.cognitive.cognitive_engine import CognitiveEngine, CognitiveSnapshot
    from pilot.config import PilotConfig, SupervisionConfig
    from pilot.system.input_hook import InputSupervisionHook

logger = logging.getLogger("pilot.agents.user_supervision")


class UserSupervisionEngine:
    """`BackgroundTask.action_fn`/`on_trigger` pair (see agents/background.py)
    for the User Manual Supervision feature. Registered as a single
    background task; the costlier OCR+cognitive sub-work is internally
    rate-limited to its own slower cadence within each tick."""

    def __init__(
        self,
        config: PilotConfig,
        cognitive_engine: CognitiveEngine,
        screen_vision: ScreenVisionAgent,
        hook: InputSupervisionHook,
        broadcast_fn: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._config = config
        self._engine = cognitive_engine
        self._screen_vision = screen_vision
        self._hook = hook
        self._broadcast_fn = broadcast_fn

        self._last_ocr_check = 0.0
        # cooldown key ("risk:<pattern>" or "coaching") -> last-fired time.
        self._cooldowns: dict[str, float] = {}

    def _cfg(self) -> SupervisionConfig:
        return self._config.supervision

    def _in_cooldown(self, kind: str, key: str = "") -> bool:
        """Checks whether `kind`/`key` is still cooling down; if not, marks
        it as fired now so the next call within the window returns True."""
        cfg = self._cfg()
        cooldown_key = f"{kind}:{key}"
        window = cfg.risk_cooldown_seconds if kind == "risk" else cfg.coaching_cooldown_seconds
        now = time.time()
        if now - self._cooldowns.get(cooldown_key, 0.0) < window:
            return True
        self._cooldowns[cooldown_key] = now
        return False

    def _due_for_ocr(self) -> bool:
        cfg = self._cfg()
        now = time.time()
        if now - self._last_ocr_check < cfg.ocr_interval_seconds:
            return False
        self._last_ocr_check = now
        return True

    def _crosses_coaching_threshold(self, snapshot: CognitiveSnapshot) -> bool:
        cfg = self._cfg()
        return (
            snapshot.stress_level > cfg.stress_coaching_threshold
            or snapshot.cognitive_load > cfg.cognitive_load_coaching_threshold
        )

    # ── BackgroundTask.action_fn ──

    async def tick(self) -> dict[str, Any]:
        """Returns {"triggered": bool, "signals": [...]}. Never includes raw
        OCR/keystroke text -- only pattern names and rounded cognitive
        snapshot floats ever appear in the returned signals."""
        cfg = self._cfg()
        signals: list[dict[str, Any]] = []

        if cfg.keyboard_mouse_hook_enabled:
            hook_snap = await asyncio.to_thread(self._hook.snapshot)
            # Real keystroke/click cadence -- an independent, objective signal
            # for CognitiveEngine's cognitive-load estimate, distinct from the
            # risk-pattern matching below (which never touches CognitiveEngine).
            self._engine.record_input_dynamics(hook_snap.keystroke_rate_per_min, hook_snap.click_rate_per_min)
            if hook_snap.matched_pattern and not self._in_cooldown("risk", hook_snap.matched_pattern):
                signals.append({"kind": "risk", "pattern": hook_snap.matched_pattern, "source": "keystroke"})

        if self._due_for_ocr():
            signals.extend(await self._ocr_signals(cfg))

        return {"triggered": bool(signals), "signals": signals}

    async def _ocr_signals(self, cfg: SupervisionConfig) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        try:
            ocr_text = await screen_ocr()
        except Exception:
            logger.debug("screen_ocr failed during a supervision tick", exc_info=True)
            ocr_text = ""
        snippet = ocr_text[: cfg.ocr_snippet_max_chars]

        state = self._screen_vision.get_context().current()
        window_title = state.active_window_title if state else ""

        if cfg.risk_pattern_detection_enabled and snippet:
            pattern = match_risk_pattern(snippet)
            if pattern and not self._in_cooldown("risk", pattern):
                signals.append({"kind": "risk", "pattern": pattern, "source": "ocr"})

        if cfg.cognitive_coaching_enabled:
            stimulus = f"{snippet} — window: {window_title}".strip(" —")
            snap = await self._engine.predict_cognitive_state(
                stimulus_description=stimulus, screen_region="user_supervision"
            )
            if self._crosses_coaching_threshold(snap) and not self._in_cooldown("coaching"):
                signals.append({"kind": "cognitive", "snapshot": snap.to_dict()})

        return signals

    # ── BackgroundTask.on_trigger ──

    async def on_trigger(self, result: dict[str, Any]) -> None:
        for signal in result.get("signals", []):
            if signal["kind"] == "risk":
                await self.on_risk_pattern_detected(signal["pattern"], signal["source"])
            elif signal["kind"] == "cognitive":
                await self.on_cognitive_checkin(signal["snapshot"])

    async def on_risk_pattern_detected(self, pattern_name: str, source: str) -> None:
        """Advisory only, returns None -- nothing to gate. Broadcasts the
        pattern NAME only, never the matched content."""
        if not self._broadcast_fn:
            return
        message = f"Heads up — this looks like it might be: {pattern_name.replace('_', ' ')}."
        await self._broadcast_fn(
            "supervision_risk_warning",
            {"pattern": pattern_name, "source": source, "message": message},
        )
        logger.info("Supervision risk pattern detected: %s (source=%s)", pattern_name, source)

    async def on_cognitive_checkin(self, snapshot: dict[str, Any]) -> None:
        """Advisory only, returns None."""
        if not self._broadcast_fn:
            return
        message = "You've seemed pretty stressed or overloaded for a while — want to take a short break?"
        await self._broadcast_fn(
            "supervision_cognitive_checkin",
            {
                "message": message,
                "attention_score": snapshot.get("attention_score"),
                "stress_level": snapshot.get("stress_level"),
                "cognitive_load": snapshot.get("cognitive_load"),
            },
        )
