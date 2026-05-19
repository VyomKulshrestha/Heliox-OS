"""Multimodal Intent Fusion Engine — combines voice + gesture into unified commands.

This module implements a time-windowed fusion system that correlates voice
transcriptions with hand gesture events to produce rich, context-aware intents.

Architecture:
  Gesture Controller ──┐
                       ├──→ Multimodal Fusion Engine ──→ Intent Parser ──→ Planner
  Voice Controller ────┘

Key design principles:
  1. Time-window correlation (configurable, default 1.5s)
  2. Confidence scoring for each modality
  3. Gesture modifier semantics (gestures modify/confirm voice commands)
  4. Accidental trigger prevention (minimum hold time + confidence threshold)
  5. Priority rules when signals conflict
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Coroutine

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.multimodal.fusion")


# ── Event types ──


class ModalityType(StrEnum):
    VOICE = "voice"
    GESTURE = "gesture"


class GestureModifier(StrEnum):
    """Semantic meaning of gestures when combined with voice."""

    CONFIRM = "confirm"  # thumbs_up, ok, palm_push
    CANCEL = "cancel"  # palm, thumbs_down, palm_pull
    LAUNCH = "launch"  # finger_gun, snap_ready
    TARGET = "target"  # point_up, index_only (directional)
    NAVIGATE = "navigate"  # swipe_left, swipe_right
    ADJUST = "adjust"  # circular_cw, circular_ccw, pinch
    EXECUTE = "execute"  # fist
    SELECT = "select"  # pinch, ok


# Gesture → Modifier mapping
GESTURE_MODIFIERS: dict[str, GestureModifier] = {
    # Confirm gestures
    "thumbs_up": GestureModifier.CONFIRM,
    "ok": GestureModifier.CONFIRM,
    "palm_push": GestureModifier.CONFIRM,
    # Cancel gestures
    "palm": GestureModifier.CANCEL,
    "thumbs_down": GestureModifier.CANCEL,
    "palm_pull": GestureModifier.CANCEL,
    "middle_finger": GestureModifier.CANCEL,
    # Launch gestures
    "finger_gun": GestureModifier.LAUNCH,
    "snap_ready": GestureModifier.LAUNCH,
    # Target/point gestures
    "point_up": GestureModifier.TARGET,
    "index_only": GestureModifier.TARGET,
    # Navigate gestures
    "swipe_left": GestureModifier.NAVIGATE,
    "swipe_right": GestureModifier.NAVIGATE,
    "swipe_up": GestureModifier.NAVIGATE,
    "swipe_down": GestureModifier.NAVIGATE,
    "two_finger_swipe_left": GestureModifier.NAVIGATE,
    "two_finger_swipe_right": GestureModifier.NAVIGATE,
    # Adjust gestures
    "circular_cw": GestureModifier.ADJUST,
    "circular_ccw": GestureModifier.ADJUST,
    "pinch": GestureModifier.ADJUST,
    "three_up": GestureModifier.ADJUST,
    "four_up": GestureModifier.ADJUST,
    # Execute gestures
    "fist": GestureModifier.EXECUTE,
    "devil_horns": GestureModifier.EXECUTE,
    # Select gestures
    "palm_down": GestureModifier.SELECT,
    "palm_up": GestureModifier.SELECT,
}

# Gesture → standalone command (used when no voice is present)
GESTURE_STANDALONE_COMMANDS: dict[str, str] = {
    "thumbs_up": "confirm",
    "thumbs_down": "deny",
    "palm": "cancel",
    "fist": "execute last command",
    "peace": "toggle voice mode",
    "point_up": "scroll up",
    "finger_gun": "take screenshot",
    "ok": "acknowledge",
    "vulcan": "show diagnostics",
    "call_me": "open settings",
    "devil_horns": "play music",
    "snap_ready": "quick launch",
    "rock": "show system info",
    "palm_push": "confirm action",
    "palm_pull": "cancel action",
}


@dataclass
class InputEvent:
    """A single input event from either voice or gesture."""

    modality: ModalityType
    timestamp: float = field(default_factory=time.time)
    # Voice fields
    transcript: str = ""
    voice_confidence: float = 0.0
    is_final: bool = False
    # Gesture fields
    gesture_name: str = ""
    gesture_confidence: float = 0.0
    gesture_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedIntent:
    """The result of fusing voice + gesture into a single command."""

    command: str  # The natural language command to send to planner
    voice_component: str = ""  # Original voice transcript
    gesture_component: str = ""  # Gesture name that modified the intent
    gesture_modifier: str = ""  # Semantic modifier type
    fusion_type: str = "single"  # single, voice_gesture, gesture_only, voice_only
    confidence: float = 0.0  # Combined confidence score
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "voice_component": self.voice_component,
            "gesture_component": self.gesture_component,
            "gesture_modifier": self.gesture_modifier,
            "fusion_type": self.fusion_type,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ── Multimodal Command Templates ──
# Maps (voice_keyword, gesture_modifier) → enhanced command


MULTIMODAL_TEMPLATES: dict[tuple[str, str], str] = {
    # Voice "open" + gesture modifiers
    ("open", "launch"): "Launch and open {target}",
    ("open", "target"): "Open the item I'm pointing at",
    ("open", "confirm"): "Yes, open {target}",
    # Voice "delete" / "close" + gesture modifiers
    ("delete", "target"): "Delete the item I'm pointing at",
    ("delete", "confirm"): "Yes, confirm delete {target}",
    ("close", "target"): "Close the window I'm pointing at",
    ("close", "confirm"): "Yes, close {target}",
    # Voice "run" / "execute" + gesture modifiers
    ("run", "launch"): "Launch and run {target} immediately",
    ("run", "execute"): "Execute {target} now",
    ("execute", "confirm"): "Confirm and execute {target}",
    # Voice "search" / "find" + gesture modifiers
    ("search", "navigate"): "Search for {target} and navigate to results",
    ("find", "target"): "Find what I'm pointing at",
    # Voice "set" / "change" + gesture modifiers
    ("set", "adjust"): "Adjust {target}",
    ("change", "adjust"): "Adjust and change {target}",
    ("volume", "adjust"): "Adjust volume {target}",
    ("brightness", "adjust"): "Adjust brightness {target}",
    # Voice "stop" / "cancel" + gesture modifiers
    ("stop", "cancel"): "Cancel and stop {target}",
    ("cancel", "cancel"): "Confirm cancellation of {target}",
    # Voice "select" + gesture modifiers
    ("select", "target"): "Select the item I'm pointing at",
    ("select", "select"): "Select {target}",
    # Voice "show" + gesture modifiers
    ("show", "target"): "Show details of what I'm pointing at",
    ("show", "navigate"): "Navigate to and show {target}",
    # Voice "move" + gesture modifiers
    ("move", "navigate"): "Move {target} in the indicated direction",
    ("move", "target"): "Move the item I'm pointing at",
    # Voice "send" / "message" + gesture modifiers
    ("send", "confirm"): "Confirm and send {target}",
    ("send", "launch"): "Launch send for {target}",
}


class MultimodalFusionEngine:
    """Fuses voice and gesture inputs into unified intents.

    Uses a time-windowed approach:
      1. Collects input events (voice + gesture) into a rolling buffer
      2. When a voice command finalizes, checks for recent gestures
      3. When a gesture fires without voice, checks for recent voice
      4. Combines signals using confidence scoring and templates
      5. Emits a FusedIntent to the planner
    """

    def __init__(
        self,
        time_window_ms: float = 1500,
        min_gesture_confidence: float = 0.6,
        min_voice_confidence: float = 0.4,
        gesture_hold_frames: int = 3,
    ) -> None:
        self.time_window_s = time_window_ms / 1000.0
        self.min_gesture_confidence = min_gesture_confidence
        self.min_voice_confidence = min_voice_confidence
        self.gesture_hold_frames = gesture_hold_frames

        self._voice_buffer: list[InputEvent] = []
        self._gesture_buffer: list[InputEvent] = []
        self._recent_intents: list[FusedIntent] = []
        self._on_intent: Callable[[FusedIntent], Coroutine] | None = None
        self._broadcast_fn: Callable[..., Coroutine] | None = None
        self._lock = asyncio.Lock()

        # Anti-flicker: track recent gesture to prevent duplicate fires
        self._last_gesture_name = ""
        self._last_gesture_time = 0.0
        self._gesture_cooldown_s = 1.2

    def set_intent_handler(self, handler: Callable[[FusedIntent], Coroutine]) -> None:
        """Set the callback for when a fused intent is resolved."""
        self._on_intent = handler

    def set_broadcast(self, fn: Callable[..., Coroutine]) -> None:
        """Set the WebSocket broadcast function for UI notifications."""
        self._broadcast_fn = fn

    # ── Input ingestion ──

    async def on_voice_event(self, event: InputEvent) -> FusedIntent | None:
        """Process an incoming voice event."""
        async with self._lock:
            self._voice_buffer.append(event)
            self._prune_buffers()

            # Only fuse on final transcriptions
            if not event.is_final or not event.transcript.strip():
                return None

            if event.voice_confidence < self.min_voice_confidence:
                return None

            # Check for recent gestures within the time window
            recent_gesture = self._find_recent_gesture(event.timestamp)

            if recent_gesture:
                intent = self._fuse_voice_gesture(event, recent_gesture)
            else:
                intent = self._voice_only_intent(event)

            intent = await self._apply_cognitive_intent(intent, voice=event, gesture=recent_gesture)
            return await self._emit_intent(intent)

    async def on_gesture_event(self, event: InputEvent) -> FusedIntent | None:
        """Process an incoming gesture event."""
        async with self._lock:
            # Anti-flicker: check cooldown
            now = time.time()
            if (
                event.gesture_name == self._last_gesture_name
                and now - self._last_gesture_time < self._gesture_cooldown_s
            ):
                return None

            if event.gesture_confidence < self.min_gesture_confidence:
                return None

            self._last_gesture_name = event.gesture_name
            self._last_gesture_time = now

            self._gesture_buffer.append(event)
            self._prune_buffers()

            # Check for recent voice within the time window
            recent_voice = self._find_recent_voice(event.timestamp)

            if recent_voice:
                intent = self._fuse_voice_gesture(recent_voice, event)
            else:
                intent = self._gesture_only_intent(event)

            intent = await self._apply_cognitive_intent(intent, voice=recent_voice, gesture=event)
            return await self._emit_intent(intent)

    # ── Fusion logic ──

    def _fuse_voice_gesture(self, voice: InputEvent, gesture: InputEvent) -> FusedIntent:
        """Combine a voice command with a gesture modifier."""
        modifier = GESTURE_MODIFIERS.get(gesture.gesture_name, "")
        modifier_str = modifier.value if isinstance(modifier, GestureModifier) else str(modifier)

        # Try to find a matching template
        command = self._resolve_template(voice.transcript, modifier_str)
        if not command:
            # Fallback: append gesture context to voice command
            command = self._build_enhanced_command(voice.transcript, gesture.gesture_name, modifier_str)

        # Combined confidence: weighted average (voice 60%, gesture 40%)
        combined_confidence = voice.voice_confidence * 0.6 + gesture.gesture_confidence * 0.4

        return FusedIntent(
            command=command,
            voice_component=voice.transcript,
            gesture_component=gesture.gesture_name,
            gesture_modifier=modifier_str,
            fusion_type="voice_gesture",
            confidence=combined_confidence,
            metadata={
                "voice_timestamp": voice.timestamp,
                "gesture_timestamp": gesture.timestamp,
                "time_delta_ms": abs(voice.timestamp - gesture.timestamp) * 1000,
            },
        )

    def _voice_only_intent(self, voice: InputEvent) -> FusedIntent:
        """Create an intent from voice only."""
        return FusedIntent(
            command=voice.transcript,
            voice_component=voice.transcript,
            fusion_type="voice_only",
            confidence=voice.voice_confidence,
        )

    def _gesture_only_intent(self, gesture: InputEvent) -> FusedIntent:
        """Create an intent from gesture only (standalone command)."""
        command = GESTURE_STANDALONE_COMMANDS.get(gesture.gesture_name, gesture.gesture_name)
        return FusedIntent(
            command=command,
            gesture_component=gesture.gesture_name,
            gesture_modifier=GESTURE_MODIFIERS.get(gesture.gesture_name, ""),
            fusion_type="gesture_only",
            confidence=gesture.gesture_confidence,
        )

    def _resolve_template(self, transcript: str, modifier: str) -> str | None:
        """Try to match a multimodal template."""
        words = transcript.lower().split()
        if not words:
            return None

        # Extract the verb (first word) and target (rest)
        verb = words[0]
        target = " ".join(words[1:]) if len(words) > 1 else ""

        # Check exact verb match
        key = (verb, modifier)
        if key in MULTIMODAL_TEMPLATES:
            return MULTIMODAL_TEMPLATES[key].format(target=target or "it")

        # Check if any keyword appears in the transcript
        for (kw, mod), template in MULTIMODAL_TEMPLATES.items():
            if mod == modifier and kw in transcript.lower():
                return template.format(target=target or transcript)

        return None

    def _build_enhanced_command(self, transcript: str, gesture_name: str, modifier: str) -> str:
        """Build an enhanced command when no template matches."""
        modifier_phrases = {
            "confirm": f"Confirm: {transcript}",
            "cancel": f"Cancel: {transcript}",
            "launch": f"Launch: {transcript}",
            "target": f"{transcript} (targeting pointed element)",
            "navigate": f"Navigate: {transcript}",
            "adjust": f"Adjust: {transcript}",
            "execute": f"Execute immediately: {transcript}",
            "select": f"Select: {transcript}",
        }
        return modifier_phrases.get(modifier, transcript)

    # ── Buffer management ──

    def _find_recent_gesture(self, reference_time: float) -> InputEvent | None:
        """Find the most recent gesture within the time window."""
        cutoff = reference_time - self.time_window_s
        candidates = [
            e
            for e in reversed(self._gesture_buffer)
            if e.timestamp >= cutoff and e.gesture_confidence >= self.min_gesture_confidence
        ]
        return candidates[0] if candidates else None

    def _find_recent_voice(self, reference_time: float) -> InputEvent | None:
        """Find the most recent finalized voice event within the time window."""
        cutoff = reference_time - self.time_window_s
        candidates = [
            e
            for e in reversed(self._voice_buffer)
            if e.timestamp >= cutoff and e.is_final and e.voice_confidence >= self.min_voice_confidence
        ]
        return candidates[0] if candidates else None

    def _prune_buffers(self) -> None:
        """Remove events older than 2x the time window."""
        cutoff = time.time() - (self.time_window_s * 2)
        self._voice_buffer = [e for e in self._voice_buffer if e.timestamp > cutoff]
        self._gesture_buffer = [e for e in self._gesture_buffer if e.timestamp > cutoff]

    # ── Intent emission ──

    async def _emit_intent(self, intent: FusedIntent) -> FusedIntent:
        """Emit a resolved intent to the handler and UI."""
        self._recent_intents.append(intent)
        # Keep only last 20 intents
        if len(self._recent_intents) > 20:
            self._recent_intents = self._recent_intents[-20:]

        logger.info(
            "Fused intent: type=%s cmd='%s' conf=%.2f voice='%s' gesture='%s'",
            intent.fusion_type,
            intent.command,
            intent.confidence,
            intent.voice_component,
            intent.gesture_component,
        )

        # Broadcast to UI
        if self._broadcast_fn:
            await self._broadcast_fn("multimodal_intent", intent.to_dict())

        # Call the intent handler
        if self._on_intent:
            await self._on_intent(intent)

        return intent

    async def _apply_cognitive_intent(
        self, intent: FusedIntent, voice: InputEvent | None, gesture: InputEvent | None
    ) -> FusedIntent:
        """Apply TRIBE v2 neural disambiguation to evaluate best candidate intent."""
        predictor = getattr(self, "_intent_predictor", None)
        if not predictor or not predictor.enabled:
            return intent

        v_text = voice.transcript if voice else ""
        v_conf = voice.voice_confidence if voice else 0.0
        g_name = gesture.gesture_name if gesture else ""
        g_conf = gesture.gesture_confidence if gesture else 0.0

        # Build candidate commands list starting with our template-resolved default
        cands = [intent.command]
        if hasattr(predictor, "_generate_candidates"):
            cands.extend(predictor._generate_candidates(v_text, g_name))

        # Deduplicate while preserving order
        cands = list(dict.fromkeys(cands))

        res = await predictor.predict_best_intent(
            voice_transcript=v_text,
            gesture_name=g_name,
            gesture_confidence=g_conf,
            voice_confidence=v_conf,
            candidate_commands=cands,
        )

        # Override intent with predicted best command
        intent.command = res.selected_command
        if res.candidates:
            intent.confidence = max(intent.confidence, res.candidates[0].boosted_confidence)

        if res.disambiguation_used:
            intent.metadata["tribe_disambiguated"] = True
            intent.metadata["tribe_boost"] = res.confidence_boost

        return intent

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        """Return fusion engine statistics."""
        type_counts: dict[str, int] = {}
        for intent in self._recent_intents:
            type_counts[intent.fusion_type] = type_counts.get(intent.fusion_type, 0) + 1

        return {
            "total_intents": len(self._recent_intents),
            "voice_buffer_size": len(self._voice_buffer),
            "gesture_buffer_size": len(self._gesture_buffer),
            "time_window_ms": self.time_window_s * 1000,
            "intent_types": type_counts,
            "last_intent": (self._recent_intents[-1].to_dict() if self._recent_intents else None),
        }
