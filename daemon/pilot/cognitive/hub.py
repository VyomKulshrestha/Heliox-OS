"""Unified Cognitive Hub — wraps all cognitive modules into one interface.

This is the central integration point that combines:
- Adaptive Biometric Learning Loop
- Ambient Intelligence Mode
- Multi-Modal Neural Bridge
- Cognitive Offloading
- Evolving Persona Architecture
- Cross-Device Cognitive Handoff
- Quantum-Ready Cognitive Pipeline

Usage:
    from pilot.cognitive.hub import CognitiveHub

    hub = CognitiveHub()
    state = await hub.analyze("user is working on complex task")
    suggestion = await hub.get_proactive_suggestion()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pilot.cognitive.ambient_intelligence import AmbientIntelligenceEngine, ProactiveSuggestion
from pilot.cognitive.biometric_loop import BiometricLearningLoop
from pilot.cognitive.clipboard_manager import clipboard_buffer
from pilot.cognitive.cognitive_handoff import CognitiveHandoffEngine
from pilot.cognitive.cognitive_offload import CognitiveOffloader
from pilot.cognitive.evolving_persona import EvolvingPersonaEngine
from pilot.cognitive.neural_bridge import NeuralBridge, NeuralWorkspace
from pilot.cognitive.quantum_cognitive import CognitiveOutput, QuantumCognitivePipeline

logger = logging.getLogger("pilot.cognitive.hub")


# ── Data Structures ──


@dataclass
class UnifiedCognitiveState:
    """Unified state from all cognitive sources."""

    # Core metrics (from TRIBE or pipeline)
    attention: float = 0.5
    stress: float = 0.3
    load: float = 0.4
    confidence: float = 0.5

    # Biometric patterns
    optimal_interaction: bool = False
    optimal_hours: list[int] = field(default_factory=list)

    # Multi-modal
    cognitive_state: str = "unknown"
    emotional_state: str = "neutral"
    engagement_level: float = 0.5
    predicted_need: str = ""

    # Offloading
    is_overloaded: bool = False
    active_anchors: int = 0

    # Persona
    communication_style: dict[str, float] = field(default_factory=dict)
    ui_config: dict[str, Any] = field(default_factory=dict)

    # Handoff
    device_name: str = ""
    handoff_suggestion: str = ""

    # Meta
    timestamp: float = field(default_factory=time.time)


@dataclass
class HubConfig:
    """Configuration for the cognitive hub."""

    enable_biometric: bool = True
    enable_ambient: bool = True
    enable_neural_bridge: bool = True
    enable_offload: bool = True
    enable_persona: bool = True
    enable_handoff: bool = True
    enable_quantum: bool = True


# ── Cognitive Hub ──


class CognitiveHub:
    """Unified cognitive hub that wraps all cognitive modules."""

    def __init__(self, user_id: str = "default", device_name: str = "desktop"):
        self._user_id = user_id
        self._device_name = device_name
        self._config = HubConfig()

        # Initialize all modules
        self._init_modules()

        # Stats
        self._total_analyses = 0

    def _init_modules(self) -> None:
        """Initialize all cognitive modules."""
        # Biometric Learning Loop
        if self._config.enable_biometric:
            self._biometric = BiometricLearningLoop(self._user_id)
        else:
            self._biometric = None

        # Ambient Intelligence
        if self._config.enable_ambient:
            self._ambient = AmbientIntelligenceEngine(self._biometric)
        else:
            self._ambient = None

        # Neural Bridge (Multi-Modal)
        if self._config.enable_neural_bridge:
            self._neural_bridge = NeuralBridge()
        else:
            self._neural_bridge = None

        # Cognitive Offloader
        if self._config.enable_offload:
            self._offloader = CognitiveOffloader()
        else:
            self._offloader = None

        # Evolving Persona
        if self._config.enable_persona:
            self._persona = EvolvingPersonaEngine(self._user_id)
        else:
            self._persona = None

        # Cognitive Handoff
        if self._config.enable_handoff:
            self._handoff = CognitiveHandoffEngine(self._device_name)
        else:
            self._handoff = None

        # Quantum Pipeline
        if self._config.enable_quantum:
            self._quantum = QuantumCognitivePipeline()
        else:
            self._quantum = None

    # ── Main Analysis API ──

    async def analyze(
        self,
        stimulus: str = "",
        context: str = "",
    ) -> UnifiedCognitiveState:
        """Run comprehensive cognitive analysis through all modules."""
        self._total_analyses += 1
        now = time.time()

        state = UnifiedCognitiveState()

        # 1. Get core metrics from Quantum Pipeline
        if self._quantum and stimulus:
            output = await self._quantum.predict(stimulus)
            state.attention = output.attention_score
            state.stress = output.stress_level
            state.load = output.cognitive_load
            state.confidence = output.confidence
        else:
            # Fallback defaults
            state.attention = 0.5
            state.stress = 0.3
            state.load = 0.4
            state.confidence = 0.5

        # 2. Update Biometric Loop
        if self._biometric:
            self._biometric.record_cognitive_sample(
                state.attention,
                state.stress,
                state.load,
                context,
            )
            rec = self._biometric.get_interaction_recommendation()
            state.optimal_interaction = rec.recommended
            state.optimal_hours = self._biometric.get_optimal_window(30)

        # 3. Update Ambient Intelligence
        if self._ambient:
            await self._ambient.update_cognitive_state(
                state.attention,
                state.stress,
                state.load,
                context,
            )

        # 4. Update Neural Bridge
        if self._neural_bridge:
            self._neural_bridge.update_from_input()
            workspace = self._neural_bridge.compute_workspace()
            state.cognitive_state = workspace.cognitive_state
            state.emotional_state = workspace.emotional_state
            state.engagement_level = workspace.engagement_level
            state.predicted_need = workspace.predicted_need

        # 5. Update Offloader
        if self._offloader:
            self._offloader.update_load(state.load)
            state.is_overloaded = self._offloader._state.is_overloaded
            state.active_anchors = len(self._offloader.get_relevant_anchors())

        if state.is_overloaded or state.load > 0.80:
            try:
                import pyperclip

                current_clip = pyperclip.paste()
                if current_clip:
                    clipboard_buffer.push_text(current_clip)
                    # Sync active anchors metric to reflect the buffered items
                    state.active_anchors = len(clipboard_buffer.get_history())
            except Exception as clipboard_err:
                logger.warning(f"Could not read system clipboard buffer: {clipboard_err}")

        # 6. Update Persona
        if self._persona:
            self._persona.record_interaction(
                state.attention,
                state.stress,
                state.load,
            )
            state.communication_style = self._persona._avatar.current_style.to_dict()
            state.ui_config = self._persona.get_ui_config()

        # 7. Update Handoff
        if self._handoff:
            self._handoff.register_activity()
            self._handoff.capture_snapshot(
                state.attention,
                state.stress,
                state.load,
                context,
            )
            state.device_name = self._handoff._device_name
            state.handoff_suggestion = (
                self._handoff.get_handoff_suggestion(
                    state.load,
                    state.stress,
                )
                or ""
            )

        state.timestamp = now

        return state

    # ── Proactive Suggestions ──

    async def get_proactive_suggestion(self) -> ProactiveSuggestion | None:
        """Get a proactive suggestion if warranted."""
        if not self._ambient:
            return None

        # The ambient engine handles its own logic
        return None

    # ── Quick Predictions ──

    async def predict_attention(self, stimulus: str) -> float:
        """Quick attention prediction."""
        if self._quantum:
            output = await self._quantum.predict(stimulus)
            return output.attention_score
        return 0.5

    async def predict_stress(self, stimulus: str) -> float:
        """Quick stress prediction."""
        if self._quantum:
            output = await self._quantum.predict(stimulus)
            return output.stress_level
        return 0.3

    async def predict_load(self, stimulus: str) -> float:
        """Quick cognitive load prediction."""
        if self._quantum:
            output = await self._quantum.predict(stimulus)
            return output.cognitive_load
        return 0.4

    # ── User Feedback ──

    async def record_feedback(
        self,
        interaction_type: str,
        user_response: str,
    ) -> None:
        """Record user feedback to refine predictions."""
        if self._biometric:
            self._biometric.record_interaction_feedback(
                interaction_type,
                user_response,
            )

        if self._persona:
            state = await self.analyze()
            self._persona.record_interaction(
                state.attention,
                state.stress,
                state.load,
                user_response,
            )

    # ── Context Offloading ──

    def get_offload_surface(self) -> dict[str, Any]:
        """Get cognitive offload surface augmented with high-stress clipboard anchors."""
        surface = {}

        if self._offloader:
            surface = self._offloader.get_offload_surface()

        if not isinstance(surface, dict):
            surface = {}

        if "anchors" not in surface or not surface["anchors"]:
            surface["anchors"] = clipboard_buffer.get_history()
        else:
            surface["anchors"] = list(surface["anchors"]) + clipboard_buffer.get_history()

        return surface

    def create_anchor(
        self,
        anchor_type: str,
        title: str,
        summary: str,
    ) -> None:
        """Create a memory anchor."""
        if self._offloader:
            self._offloader.create_anchor(anchor_type, title, summary)

    # ── Persona Actions ──

    def get_greeting(self) -> str:
        """Get personalized greeting."""
        if self._persona:
            return self._persona.get_greeting()
        return "Hello"

    def format_message(self, message: str) -> str:
        """Format message based on communication style."""
        if self._persona:
            return self._persona.format_response(message)
        return message

    # ── Handoff Actions ──

    def get_active_devices(self) -> list[dict[str, Any]]:
        """Get list of active devices."""
        if self._handoff:
            return self._handoff.get_active_devices()
        return []

    def initiate_handoff(self, target_device: str) -> bool:
        """Initiate handoff to another device."""
        if self._handoff:
            return self._handoff.initiate_handoff(target_device) is not None
        return False

    # ── Neural Bridge Input ──

    def record_keystroke(self) -> None:
        """Record keystroke for engagement tracking."""
        if self._neural_bridge:
            self._neural_bridge.record_keystroke()

    def record_mouse_move(self, x: float, y: float) -> None:
        """Record mouse movement."""
        if self._neural_bridge:
            self._neural_bridge.record_mouse_move(x, y)

    def record_click(self) -> None:
        """Record mouse click."""
        if self._neural_bridge:
            self._neural_bridge.record_click()

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        stats = {"total_analyses": self._total_analyses}

        if self._biometric:
            stats["biometric"] = self._biometric.get_stats()

        if self._ambient:
            stats["ambient"] = self._ambient.get_stats()

        if self._neural_bridge:
            stats["neural_bridge"] = self._neural_bridge.get_stats()

        if self._offloader:
            stats["offloader"] = self._offloader.get_stats()

        if self._persona:
            stats["persona"] = self._persona.get_stats()

        if self._handoff:
            stats["handoff"] = self._handoff.get_stats()

        if self._quantum:
            stats["quantum"] = self._quantum.get_stats()

        return stats


# ── Singleton ──

_hub_instance: CognitiveHub | None = None


def get_hub(user_id: str = "default", device_name: str = "desktop") -> CognitiveHub:
    """Get or create the singleton cognitive hub."""
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = CognitiveHub(user_id, device_name)
    return _hub_instance
