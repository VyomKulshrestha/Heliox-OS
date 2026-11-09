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

# One OCR read may take this long before it counts as a failure.
_OCR_TIMEOUT_SECONDS = 30.0
# After repeated failures the retry gap doubles, up to this many seconds.
_OCR_BACKOFF_MAX_SECONDS = 600.0


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
        # Screen reading health. The periodic read only uses the fast engines
        # (native OS OCR, Tesseract); EasyOCR takes ~20 s of CPU per read.
        self._ocr_running = False
        self._ocr_failures = 0
        self._ocr_retry_at = 0.0
        self._ocr_unavailable_reported = False
        self.screen_reading_error: str | None = None

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
        if self._ocr_running or now < self._ocr_retry_at:
            return False
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
        if not cfg.enabled:
            # Opt-in privacy feature: nothing is read, however the task was started.
            return {"triggered": False, "signals": signals}

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
        ocr_text = await self._read_screen()
        if ocr_text is None:
            # No fast engine worked: skip this tick's OCR-based checks.
            return signals
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

    def screen_reading_status(self) -> dict[str, Any]:
        """For the status RPC: whether the periodic screen read is working, and why not."""
        if self.screen_reading_error is None:
            return {"available": True, "reason": "", "retry_in_seconds": None}
        return {
            "available": False,
            "reason": self.screen_reading_error,
            "retry_in_seconds": max(0, round(self._ocr_retry_at - time.time())),
        }

    async def _read_screen(self) -> str | None:
        """OCR with the fast engines only. Returns None when screen reading failed."""
        self._ocr_running = True
        try:
            text = await asyncio.wait_for(screen_ocr(allow_slow_fallback=False), timeout=_OCR_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            await self._ocr_failed(f"a read took longer than {int(_OCR_TIMEOUT_SECONDS)} seconds")
            return None
        except Exception as exc:
            await self._ocr_failed(str(exc) or exc.__class__.__name__)
            return None
        finally:
            self._ocr_running = False
        self._ocr_failures = 0
        self._ocr_retry_at = 0.0
        self._ocr_unavailable_reported = False
        self.screen_reading_error = None
        return text

    async def _ocr_failed(self, reason: str) -> None:
        self._ocr_failures += 1
        delay = min(max(self._cfg().ocr_interval_seconds, 1.0) * (2**self._ocr_failures), _OCR_BACKOFF_MAX_SECONDS)
        self._ocr_retry_at = time.time() + delay
        self.screen_reading_error = reason
        logger.warning("Supervision screen reading failed (%s); next try in %.0f s", reason, delay)
        if self._ocr_unavailable_reported:
            return
        self._ocr_unavailable_reported = True
        if self._broadcast_fn:
            await self._broadcast_fn(
                "supervision_screen_reading_unavailable",
                {
                    "message": (
                        "Supervision can't read your screen right now, so it is skipping screen checks. "
                        f"Reason: {reason}"
                    ),
                    "reason": reason,
                },
            )

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
